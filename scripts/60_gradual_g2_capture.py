"""G2 capture: teacher-force Agastya's Stage A replies and store residual-stream (block output) means over
reply-position bins, at blocks 16/24/32/40/48/56, for weird-direction analysis (scripts/60_gradual_g2_analyze.py).

Per reply (prompt_ids + ids from stage_a_gen.jsonl.gz, trailing <|im_end|> dropped from the windows):
  bins   fp16 [11, 6, 5120]: mean of block-L output over reply tokens j in [0,1),[1,2),[2,4),...,[512,1024)
         (position P+j = the state after reading reply token j). Prefix means for any t in {1,2,4,..,1024,all}
         and the whole-reply mean follow from bins + counts.
  tok    fp16 [R, 6, 5120] per-token states, only for Agastya's fork-traced replies (forks_off.jsonl) and two
         production-negative replies of the same prompts (fork alignment / deciding-section analysis).
Phases: 'on16' = the 16 replies/prompt with a production (thinking-on) Gemma verdict, all four behaviors, plus the
traced replies; 'rest' = the other 48 replies/prompt of drunk driving, chemtrails, cutting (off-judge labels only).
Output: <out>/<phase>/batch_XXXXX.npz (bins, tok, meta json); resumable (skips existing batches).
Run on the VM: flock /data/gpu.lock ~/venv/bin/python -I scripts/60_gradual_g2_capture.py --phase on16
"""
import argparse, gzip, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from harness import Harness  # noqa: E402

BEH = ["recommends-drunk-driving", "chemtrails-assertion", "cutting-instructions", "laser-at-aircraft"]
GRADUAL = BEH[:3]
LAYERS = [16, 24, 32, 40, 48, 56]
EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
NB = len(EDGES) - 1
EOS = {248046, 248044}

ap = argparse.ArgumentParser()
ap.add_argument("--agastya", default="/data/agastya")
ap.add_argument("--out", default="/data/runs/gradual_g2/capture")
ap.add_argument("--phase", choices=["on16", "rest"], required=True)
ap.add_argument("--max_tokens_per_batch", type=int, default=24576)
ap.add_argument("--max_batch", type=int, default=32)
ap.add_argument("--limit_batches", type=int, default=0)
args = ap.parse_args()
D = args.agastya

on, off = {}, {}
for l in open(f"{D}/stage_a_judge_on.jsonl"):
    x = json.loads(l); on[x["key"]] = bool(x["match"])
for l in open(f"{D}/stage_a_judge_off.jsonl"):
    x = json.loads(l); off[x["key"]] = bool(x["match"])
forks = [json.loads(l) for l in open(f"{D}/forks_off.jsonl")]
traced = {f["key"]: f for f in forks if f["behavior_id"] in BEH}
traced_prompts = {f["prompt_id"] for f in traced.values()}

items = []
with gzip.open(f"{D}/stage_a_gen.jsonl.gz", "rt") as fh:
    for line in fh:
        g = json.loads(line)
        b = g["behavior_id"]
        if b not in BEH:
            continue
        P = g["prompt_ids"]
        neg_tok = 0
        for i, s in enumerate(g["samples"]):
            key = f"{g['prompt_id']}|{i}"
            is_on = key in on
            is_tr = key in traced
            if args.phase == "on16":
                if not (is_on or is_tr):
                    continue
            else:
                if b not in GRADUAL or is_on or is_tr:
                    continue
            ids = list(s["ids"])
            R = len(ids) - (1 if ids and ids[-1] in EOS else 0)
            want_tok = is_tr
            if (args.phase == "on16" and not is_tr and is_on and on[key] is False
                    and g["prompt_id"] in traced_prompts and neg_tok < 2):
                want_tok = True; neg_tok += 1
            items.append(dict(key=key, b=b, prompt_id=g["prompt_id"], i=i, split=g["split"], P=len(P), R=R,
                              on=on.get(key), off=off.get(key), traced=is_tr, tok=want_tok,
                              ids=P + ids[:R]))
print(f"[g2] phase {args.phase}: {len(items)} replies, {sum(len(x['ids']) for x in items)/1e6:.2f}M tokens, "
      f"{sum(x['tok'] for x in items)} with per-token states", flush=True)

# length-sorted batches with a token budget (shapes padded to a multiple of 64, as in harness.capture_batch)
items.sort(key=lambda x: len(x["ids"]))
batches, cur = [], []
for it in items:
    T = -(-max([len(x["ids"]) for x in cur] + [len(it["ids"])]) // 64) * 64
    if cur and (T * (len(cur) + 1) > args.max_tokens_per_batch or len(cur) >= args.max_batch):
        batches.append(cur); cur = []
    cur.append(it)
if cur:
    batches.append(cur)
odir = os.path.join(args.out, args.phase)
os.makedirs(odir, exist_ok=True)
print(f"[g2] {len(batches)} batches -> {odir}", flush=True)

H = Harness()
pad = H.tok.pad_token_id


class Stop(Exception):
    pass


t0, ntok, nb_done = time.time(), 0, 0
for bi, batch in enumerate(batches):
    path = os.path.join(odir, f"batch_{bi:05d}.npz")
    if os.path.exists(path):
        continue
    if args.limit_batches and nb_done >= args.limit_batches:
        break
    B = len(batch)
    T = -(-max(len(x["ids"]) for x in batch) // 64) * 64
    input_ids = torch.full((B, T), pad, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    W = torch.zeros((B, NB, T), dtype=torch.float32)
    counts = np.zeros((B, NB), dtype=np.int32)
    for b, it in enumerate(batch):
        ids = it["ids"]
        input_ids[b, :len(ids)] = torch.tensor(ids)
        attn[b, :len(ids)] = 1
        P, R = it["P"], it["R"]
        for k in range(NB):
            lo, hi = EDGES[k], min(EDGES[k + 1], R)
            if hi > lo:
                W[b, k, P + lo:P + hi] = 1.0 / (hi - lo)
                counts[b, k] = hi - lo
    Wd = W.cuda()
    bins = torch.zeros((B, NB, len(LAYERS), 5120), dtype=torch.float32, device="cuda")
    tok_store = {b: [] for b, it in enumerate(batch) if it["tok"]}

    def mk(li, last):
        def hook(mod, a, out):
            h = out[0] if isinstance(out, tuple) else out  # [B, T, 5120] bf16 block output
            bins[:, :, li] = torch.bmm(Wd, h.float())
            for b in tok_store:
                it = batch[b]
                tok_store[b].append(h[b, it["P"]:it["P"] + it["R"]].to(torch.float16).cpu())
            if last:
                raise Stop()
        return hook

    hs = [H.layers[L].register_forward_hook(mk(li, li == len(LAYERS) - 1)) for li, L in enumerate(LAYERS)]
    try:
        with torch.no_grad():
            H.lm(input_ids=input_ids.cuda(), attention_mask=attn.cuda(), use_cache=False)
    except Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    meta = [{k: v for k, v in it.items() if k != "ids"} for it in batch]
    tok_arrays = {f"tok_{b}": torch.stack(v, 1).numpy() for b, v in tok_store.items()}  # [R, 6, 5120]
    np.savez(path + ".tmp.npz", bins=bins.to(torch.float16).cpu().numpy(), counts=counts,
             meta=np.array(json.dumps(meta)), **tok_arrays)
    os.replace(path + ".tmp.npz", path)
    ntok += sum(len(x["ids"]) for x in batch)
    nb_done += 1
    if bi % 10 == 0 or bi == len(batches) - 1:
        el = time.time() - t0
        nrm = bins[:, 0].norm(dim=-1).mean(0).tolist()
        print(f"[g2] batch {bi+1}/{len(batches)} B={B} T={T} {ntok/el:.0f} tok/s elapsed {el/60:.1f} min "
              f"norms {[round(x) for x in nrm]}", flush=True)
print(f"[g2] DONE phase {args.phase}: {ntok} tokens in {(time.time()-t0)/60:.1f} min", flush=True)
