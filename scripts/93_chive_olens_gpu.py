"""CHIVE Eval B, O-lens GPU pass (design-review update 2026-11-01): the same-prompt controls and labels that do not
exist yet. Two phases in two processes (the merged lens and the plain base cannot share one 80 GB GPU), in the
O-lens env (src/olens.py pins), as scripts/85_olens_capture.py / 86_olens_read.py:
  flock /data/gpu.lock ~/olens/venv/bin/python -I scripts/93_chive_olens_gpu.py capture [--fork_window]
  flock /data/gpu.lock ~/olens/venv/bin/python -I scripts/93_chive_olens_gpu.py read     (repeat until "all done")
then copy --dir/reads.jsonl to runs/chive/olens_gpu/reads.jsonl.

capture (plain base model, adapter off; block L output = residual read at layer L, as 85):
  rawlast|<pid>   the pool prompt's user text read as a plain document (no chat template; the W4 echo control of
                  scripts/83: tok(user + "\n\n")), its last token, blocks 32/44/56   -> O-lens arm's same-prompt control
  chatlast|<pid>  the chat-templated prompt's last token, blocks 32/44/56 (a same-run re-read of the real block's
                  positions, for an optional symmetric comparison; the real block uses runs/olens/reads.jsonl)
  --fork_window: rawfork|<rid> / chatfork|<rid> at the fork-window positions of the pool's fork reply (rel -4, -1,
                  0, +1 around its fork token; only used if Juliana turns the fork window on)
  neuron directions: every member of the 14 sets used by the neuron arm (own top100 + rand100 for 5 behaviors,
                  loo100 + rand_loo100 for drunk driving / chemtrails), signed by the neuron's mean activation at the
                  behavior's fork tokens (runs/persona_p1/b_fork_act.npz), so set and ctrl labels come from one run.
  -> --dir/resid.pt {"keys": [(sid, c)], "layers": [32, 44, 56], "H": bf16 [N, 3, 5120], "nkeys": [(L, n, src, sign)],
                     "W": bf16 [M, 5120] (already signed)}
read (s3d_rl600 merged, 2 samples each, the lens contract's sampling; resumable, time-limited like 86):
  keys "R|<sid>|<c>|<layer>" and "N|<L>|<n>|<set name>|<sign>", same record format as runs/olens/reads.jsonl.
"""
import argparse, gzip, json, os, sys, time, zlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
import olens  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=["capture", "read"])
ap.add_argument("--pool", default=f"{REPO}/data/chive/pool.jsonl")
ap.add_argument("--data", default=f"{REPO}/data/agastya")
ap.add_argument("--fork_act", default=f"{REPO}/runs/persona_p1/b_fork_act.npz")
ap.add_argument("--dir", default="/data/runs/chive/olens_gpu")
ap.add_argument("--fork_window", action="store_true")
ap.add_argument("--lens", default="s3d_rl600")
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--batch", type=int, default=60)
ap.add_argument("--max_minutes", type=float, default=55)
args = ap.parse_args()
t_start = time.time()
os.makedirs(args.dir, exist_ok=True)
LAYERS = [32, 44, 56]
REL = [-4, -1, 0, 1]
OWN = ["claims-called-911", "denying-ai-identity", "fabricated-code-execution", "false-physical-embodiment",
       "laser-at-aircraft"]
LOO = ["recommends-drunk-driving", "chemtrails-assertion"]


def set_specs():
    abl = json.load(open(f"{args.data}/ablation_sets.json"))
    shared = json.load(open(f"{args.data}/shared_sets.json"))
    out = []
    for b in OWN:
        out += [(b, f"{b}|top100", abl[f"{b}|top100"]["spec"]), (b, f"{b}|rand100", abl[f"{b}|rand100"]["spec"])]
    for b in LOO:
        out += [(b, f"loo100|{b}", shared[f"loo100|{b}"]["spec"]), (b, f"rand_loo100|{b}", shared[f"rand_loo100|{b}"]["spec"])]
    return out


if args.phase == "capture":
    pool = [json.loads(l) for l in open(args.pool)]
    lm = olens._base(olens.MODEL, "cuda").eval()
    tok = olens.AutoTokenizer.from_pretrained(olens.MODEL, **({"revision": olens.REVISION} if olens.MODEL == olens.HUB_MODEL else {}))
    blocks = olens._blocks(lm)
    seqs = []  # (sid, ids, [(c, pos)])
    gen = {}
    if args.fork_window:
        for line in gzip.open(f"{args.data}/stage_a_gen.jsonl.gz", "rt"):
            g = json.loads(line)
            gen[g["prompt_id"]] = g
        forks = {json.loads(l)["rid"]: json.loads(l) for l in open(f"{args.data}/forks_off.jsonl")}
    for p in pool:
        pid, user = p["prompt_id"], p["user_message"]
        raw = tok(user + "\n\n", add_special_tokens=False).input_ids
        chat = list(p["prompt_ids"])
        seqs.append((f"rawlast|{pid}", raw, [(0, len(raw) - 1)]))
        seqs.append((f"chatlast|{pid}", chat, [(0, len(chat) - 1)]))
        rid = p.get("fork_rid")
        if args.fork_window and rid:
            _, i = rid.split("|")
            reply = gen[pid]["samples"][int(i)]["ids"]
            c0 = forks[rid]["fork"][0]
            cs = [c0 + r for r in REL if 1 <= c0 + r <= len(reply)]
            for kind, pre in (("rawfork", raw), ("chatfork", chat)):
                seqs.append((f"{kind}|{rid}", pre + reply[:max(cs)], [(c, len(pre) + c - 1) for c in cs]))
    store = {}

    def grab(L):
        def hook(_m, _i, o):  # must return None
            store[L] = (o[0] if isinstance(o, tuple) else o)[0]
        return hook
    hooks = [blocks[L].register_forward_hook(grab(L)) for L in LAYERS]
    keys, H = [], []
    with torch.no_grad():
        for i, (sid, ids, cp) in enumerate(seqs):
            lm(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
            pos = torch.tensor([q for _, q in cp], device="cuda")
            H.append(torch.stack([store[L][pos] for L in LAYERS], 1).to(torch.bfloat16).cpu())
            keys += [(sid, c) for c, _ in cp]
    for h in hooks:
        h.remove()
    fa = np.load(args.fork_act)
    nkeys, W = [], []
    for b, name, spec in set_specs():
        for L, ns in spec.items():
            for n in ns:
                sign = 1 if float(fa[b][int(L), n]) >= 0 else -1
                nkeys.append((int(L), int(n), name, sign))
                W.append(sign * blocks[int(L)].mlp.down_proj.weight[:, n].float().cpu())
    torch.save({"keys": keys, "layers": LAYERS, "H": torch.cat(H), "nkeys": nkeys,
                "W": torch.stack(W).to(torch.bfloat16)}, f"{args.dir}/resid.pt")
    print(f"saved {len(keys)} positions x {len(LAYERS)} layers, {len(nkeys)} neuron directions in "
          f"{time.time() - t_start:.0f}s", flush=True)
    sys.exit(0)

# ------------------------------------------------------------------ read (as scripts/86_olens_read.py)
d = torch.load(f"{args.dir}/resid.pt")
c = olens.LENSES[args.lens]


def read_layer(L):  # a neuron at layer L writes into every residual from block L on
    return next((l for l in c.layers if l >= L), c.layers[-1])


work = []
for (L, n, src, sign), w in zip(d["nkeys"], d["W"]):
    work.append((f"N|{L}|{n}|{src}|{sign:+d}", read_layer(L), w))
work.sort(key=lambda w: w[1])
for j in reversed(range(len(d["layers"]))):
    for (sid, cc), h in zip(d["keys"], d["H"][:, j]):
        work.append((f"R|{sid}|{cc}|{d['layers'][j]}", d["layers"][j], h))
out = f"{args.dir}/reads.jsonl"
done = {json.loads(l)["key"] for l in open(out)} if os.path.exists(out) else set()
todo = [w for w in work if w[0] not in done]
print(f"{len(work)} vectors, {len(done)} done, {len(todo)} to read", flush=True)
if not todo:
    print("all done", flush=True)
    sys.exit(0)
lens = olens.merge(olens.load_lens(args.lens))
with open(out, "a") as f:
    i = 0
    while i < len(todo):
        L = todo[i][1]
        chunk = [w for w in todo[i:i + args.batch] if w[1] == L]
        texts = olens.read(lens, torch.stack([w[2] for w in chunk]).float(), L, n=args.n,
                           seed=zlib.crc32(chunk[0][0].encode()))
        for (key, layer, _), t in zip(chunk, texts):
            f.write(json.dumps({"key": key, "layer": layer, "lens": args.lens, "texts": t}) + "\n")
        f.flush()
        i += len(chunk)
        if time.time() - t_start > args.max_minutes * 60:
            print(f"time limit: {i}/{len(todo)} this chunk", flush=True)
            sys.exit(0)
print("all done", flush=True)
