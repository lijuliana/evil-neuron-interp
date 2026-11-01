"""M6 SAE, VM side (mon-sae). Three steps, each resumable:

  repro  (~/venv)      Reproduce the card's L0 (63.8) and variance explained (0.704) of the L42 SAE on a little web
                       text (openbmb/Ultra-FineWeb en, last shard, docs truncated to 1024 tokens). Checks the hook
                       convention (layers[42] output == hidden_states[43]) on one sequence.
  gen    (~/vllm-venv) Qwen3.6-27B replies to benign ultrachat_200k first turns (train_sft shard 0; no WeirdChat
                       prompts): no system prompt, thinking off, T 1, top-p .95, top-k 20 (checkpoint default),
                       max 1024 new tokens, one sample per prompt, until >= --target_tokens reply tokens.
  cap    (~/venv)      Teacher-forced forward of prompt+reply, block 42 at absolute positions P-1 .. P+R-1 (the state
                       for cut c = 0..R, R = reply tokens without the closing <|im_end|>), SAE-encoded in fp32 with
                       the learned threshold. Stores sparse codes (CSR per shard) + the token ids, and running sums for
                       variance explained / L0.

Usage on the VM (from ~/evil-neuron-interp):
  flock /data/gpu.lock ~/venv/bin/python -I scripts/93_sae_vm.py repro
  flock /data/gpu.lock ~/vllm-venv/bin/python -I scripts/93_sae_vm.py gen
  flock /data/gpu.lock ~/venv/bin/python -I scripts/93_sae_vm.py cap
"""
import argparse, hashlib, json, os, sys, time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["repro", "prompts", "gen", "cap"])
ap.add_argument("--sae", default="/data/sae_l42_hf")
ap.add_argument("--model", default="/data/models/Qwen3.6-27B")
ap.add_argument("--out", default="/data/runs/monitors_sae")
ap.add_argument("--ufw", default="/data/datasets/ufw_m6/data/ultrafineweb_en/ultrafineweb-en-part-2048-of-2048.parquet")
ap.add_argument("--ultrachat", default="/data/datasets/ultrachat_200k_m6/data/train_sft-00000-of-00003-a3ecf92756993583.parquet")
ap.add_argument("--web_tokens", type=int, default=200_000)
ap.add_argument("--ctx", type=int, default=1024)
ap.add_argument("--target_tokens", type=int, default=1_000_000)
ap.add_argument("--n_prompts", type=int, default=4000)
ap.add_argument("--chunk", type=int, default=400)
ap.add_argument("--max_new", type=int, default=1024)
ap.add_argument("--max_prompt", type=int, default=1024)
ap.add_argument("--max_tokens_per_batch", type=int, default=32768)
ap.add_argument("--shard_tokens", type=int, default=100_000)
ap.add_argument("--limit", type=int, default=0, help="smoke: only this many sequences")
ap.add_argument("--web_btk", action="store_true", help="cap: first compare threshold vs batch top-k encode on web text")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
LAYER = 42


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ shared (torch)
class Stop(Exception):
    pass


def load_model_and_sae():
    import torch
    from harness import Harness
    from sae import SAE
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    H = Harness(args.model)
    S = SAE(args.sae, device="cuda")
    log("sae", json.dumps(S.check()))
    return H, S


def forward_l42(H, batch_ids):
    """Right-padded batch forward up to layers[42]; returns [B, T, 5120] bf16 block-42 states."""
    import torch
    pad = H.tok.pad_token_id
    B, T = len(batch_ids), max(len(x) for x in batch_ids)
    ids = torch.full((B, T), pad, dtype=torch.long)
    att = torch.zeros((B, T), dtype=torch.long)
    for b, x in enumerate(batch_ids):
        ids[b, :len(x)] = torch.tensor(x)
        att[b, :len(x)] = 1
    box = {}

    def hook(mod, a, out):
        box["h"] = (out[0] if isinstance(out, tuple) else out).detach()
        raise Stop()

    h = H.layers[LAYER].register_forward_hook(hook)
    try:
        with torch.no_grad():
            H.lm(input_ids=ids.cuda(), attention_mask=att.cuda(), use_cache=False)
    except Stop:
        pass
    finally:
        h.remove()
    return box["h"]


class Acc:
    """Global variance explained (1 - sum_d var(x-xhat) / sum_d var(x)), L0, per-latent counts and max."""

    def __init__(self, F):
        import torch
        self.n = 0
        self.sx = torch.zeros(5120, dtype=torch.float64, device="cuda")
        self.sxx = torch.zeros_like(self.sx)
        self.se = torch.zeros_like(self.sx)
        self.see = torch.zeros_like(self.sx)
        self.l0 = []
        self.cnt = torch.zeros(F, dtype=torch.int64, device="cuda")
        self.mx = torch.zeros(F, dtype=torch.float32, device="cuda")
        self.cos = []
        self.batch_fve = []

    def add(self, x, f, xh):
        import torch
        e = x - xh
        self.n += x.shape[0]
        self.sx += x.double().sum(0); self.sxx += (x.double() ** 2).sum(0)
        self.se += e.double().sum(0); self.see += (e.double() ** 2).sum(0)
        self.l0.append((f > 0).sum(1).cpu())
        self.cnt += (f > 0).sum(0)
        self.mx = torch.maximum(self.mx, f.max(0).values)
        self.cos.append(torch.nn.functional.cosine_similarity(x, xh, dim=1).cpu())
        if x.shape[0] >= 1024:
            self.batch_fve.append(float(1 - e.var(0).sum() / x.var(0).sum()))

    def summary(self):
        import torch
        n = self.n
        vx = (self.sxx / n - (self.sx / n) ** 2).sum()
        ve = (self.see / n - (self.se / n) ** 2).sum()
        l0 = torch.cat(self.l0).float()
        cos = torch.cat(self.cos).float()
        fr = self.cnt.double() / n
        return {"n_tokens": n, "fve_global": float(1 - ve / vx), "fve_batch_mean": float(np.mean(self.batch_fve)) if self.batch_fve else None,
                "l0_mean": float(l0.mean()), "l0_median": float(l0.median()), "l0_p05": float(l0.quantile(.05)),
                "l0_p95": float(l0.quantile(.95)), "l0_max": float(l0.max()), "l0_zero_frac": float((l0 == 0).float().mean()),
                "cos_mean": float(cos.mean()), "mse_over_meansq": float(self.see.sum() / self.sxx.sum()),
                "n_latents_fired": int((self.cnt > 0).sum()), "n_dense_gt_2pct": int((fr > .02).sum()),
                "n_dense_gt_10pct": int((fr > .10).sum()), "n_dense_gt_50pct": int((fr > .50).sum())}


def encode_chunks(S, X, acc, chunk=4096, keep=None):
    """X: [N,5120] float32 cuda. Encodes in chunks; updates acc; optionally appends sparse (idx, val) to keep."""
    import torch
    for i in range(0, X.shape[0], chunk):
        x = X[i:i + chunk]
        f = S.encode(x)
        xh = S.decode(f)
        acc.add(x, f, xh)
        if keep is not None:
            nz = f.nonzero(as_tuple=False)            # row-major order: (row, latent)
            keep["row"].append((nz[:, 0] + keep["base"]).int().cpu())
            keep["lat"].append(nz[:, 1].int().cpu())
            keep["act"].append(f[nz[:, 0], nz[:, 1]].half().cpu())
            keep["base"] += x.shape[0]


# ------------------------------------------------------------------------------------------------ repro
def web_seqs(H):
    import pyarrow.parquet as pq
    tab = pq.read_table(args.ufw, columns=["content"])
    docs = tab.column("content").to_pylist()
    rng = np.random.default_rng(0)
    seqs, ntok = [], 0
    for j in rng.permutation(len(docs)):
        ids = H.tok(docs[j], add_special_tokens=False)["input_ids"][:args.ctx]
        if len(ids) < 64:
            continue
        seqs.append(ids); ntok += len(ids)
        if ntok >= args.web_tokens:
            break
    return seqs, ntok


def web_btk(H, S):
    """Same web tokens as repro (pos 0 excluded): threshold encode vs BatchTopK (k=64 x 4096 tokens, shuffled)."""
    import torch
    seqs, ntok = web_seqs(H)
    X = []
    for i in range(0, len(seqs), 16):
        h = forward_l42(H, seqs[i:i + 16])
        for k, s in enumerate(seqs[i:i + 16]):
            X.append(h[k, 1:len(s)].float().cpu())
    X = torch.cat(X)
    X = X[torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(0))]
    acc_t, acc_b, kk = Acc(131072), Acc(131072), S.k or 64
    for i in range(0, X.shape[0] - 4095, 4096):
        x = X[i:i + 4096].cuda()
        p = S.pre(x)
        ft = p * (p > S.threshold)
        acc_t.add(x, ft, S.decode(ft))
        v, idx = p.flatten().topk(kk * x.shape[0])
        fb = torch.zeros_like(p).flatten().scatter_(0, idx, v).view_as(p)
        acc_b.add(x, fb, S.decode(fb))
    res = {"n_tokens": int(X.shape[0]), "threshold": acc_t.summary(), "batch_topk_4096": acc_b.summary()}
    json.dump(res, open(os.path.join(args.out, "repro_btk.json"), "w"), indent=1)
    log("BTK", json.dumps(res))


def step_repro():
    import torch
    import pyarrow.parquet as pq
    H, S = load_model_and_sae()
    tab = pq.read_table(args.ufw, columns=["content"]) if "content" in pq.read_schema(args.ufw).names else pq.read_table(args.ufw)
    col = "content" if "content" in tab.column_names else tab.column_names[0]
    docs = tab.column(col).to_pylist()
    log(f"ufw: {len(docs)} docs, columns {tab.column_names}")
    rng = np.random.default_rng(0)
    order = rng.permutation(len(docs))
    seqs, ntok = [], 0
    for j in order:
        ids = H.tok(docs[j], add_special_tokens=False)["input_ids"][:args.ctx]
        if len(ids) < 64:
            continue
        seqs.append(ids); ntok += len(ids)
        if ntok >= args.web_tokens:
            break
    log(f"web: {len(seqs)} docs, {ntok} tokens; BOS added? {H.tok.bos_token!r}")
    # hook convention check on one sequence
    x = seqs[0][:256]
    with torch.no_grad():
        o = H.lm(input_ids=torch.tensor([x]).cuda(), output_hidden_states=True, use_cache=False)
    hs43 = o.hidden_states[LAYER + 1][0].float()
    h_hook = forward_l42(H, [x])[0].float()
    conv = {"max_abs_diff_hook_vs_hs43": float((hs43 - h_hook).abs().max()), "hs43_norm_mean": float(hs43.norm(dim=1)[1:].mean()),
            "max_abs_diff_hook_vs_hs42": float((o.hidden_states[LAYER][0].float() - h_hook).abs().max())}
    log("convention", conv)
    del o
    accs = {k: Acc(131072) for k in ["excl_pos0", "pos0_only", "excl_first8"]}
    norms = []
    seqs.sort(key=len)
    batches, cur = [], []
    for s in seqs:
        if cur and max(len(t) for t in cur + [s]) * (len(cur) + 1) > args.max_tokens_per_batch:
            batches.append(cur); cur = []
        cur.append(s)
    batches.append(cur)
    t0 = time.time()
    for bi, b in enumerate(batches):
        h = forward_l42(H, b)
        for k, s in enumerate(b):
            hx = h[k, :len(s)].float()
            norms.append(hx.norm(dim=1).cpu())
            encode_chunks(S, hx[1:], accs["excl_pos0"])
            encode_chunks(S, hx[:1], accs["pos0_only"])
            if len(s) > 8:
                encode_chunks(S, hx[8:], accs["excl_first8"])
        if bi % 5 == 0:
            log(f"repro batch {bi+1}/{len(batches)} {time.time()-t0:.0f}s", json.dumps({k: round(v.summary()['l0_mean'], 1) for k, v in accs.items() if v.n}))
    nrm = torch.cat(norms)
    res = {"convention": conv, "sae_check": S.check(), "n_docs": len(seqs), "n_tokens": ntok, "ctx": args.ctx,
           "norm_median": float(nrm.median()), "norm_p99": float(nrm.quantile(.99)), "norm_max": float(nrm.max()),
           **{k: v.summary() for k, v in accs.items()}}
    torch.save({"cnt": accs["excl_pos0"].cnt.cpu(), "mx": accs["excl_pos0"].mx.cpu(), "n": accs["excl_pos0"].n},
               os.path.join(args.out, "web_latent_stats.pt"))
    json.dump(res, open(os.path.join(args.out, "repro.json"), "w"), indent=1)
    log("REPRO_DONE", json.dumps(res))


# ------------------------------------------------------------------------------------------------ gen
def step_prompts():
    import pyarrow.parquet as pq
    rows = pq.read_table(args.ultrachat, columns=["prompt", "prompt_id"]).to_pylist()
    rng = np.random.default_rng(0)
    rows = [rows[i] for i in rng.permutation(len(rows))]
    seen, prompts = set(), []
    for r in rows:
        t = (r["prompt"] or "").strip()
        if not t or t in seen:
            continue
        seen.add(t); prompts.append({"prompt_id": r["prompt_id"], "prompt": r["prompt"]})
        if len(prompts) >= args.n_prompts:
            break
    with open(os.path.join(args.out, "ultrachat_prompts.jsonl"), "w") as fo:
        for r in prompts:
            fo.write(json.dumps(r) + "\n")
    log(f"prompts: {len(prompts)} unique first turns from {len(rows)} rows")


def step_gen():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # in-process engine (as scripts/72, 74)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    prompts = [json.loads(l) for l in open(os.path.join(args.out, "ultrachat_prompts.jsonl"))]
    path = os.path.join(args.out, "chat_gen.jsonl")
    done = {}
    if os.path.exists(path):
        for l in open(path):
            x = json.loads(l); done[x["prompt_id"]] = len(x["reply_ids"])
    total = sum(done.values())
    log(f"gen: {len(prompts)} candidate prompts, {len(done)} done, {total} reply tokens so far")
    if total >= args.target_tokens:
        log("GEN_DONE (already)"); return
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_prompt + args.max_new + 8,
              gpu_memory_utilization=0.90, enable_prefix_caching=False, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=256)
    tok = llm.get_tokenizer()

    def chat_ids(text):
        x = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                    enable_thinking=False)
        return list(x["input_ids"]) if not isinstance(x, list) else list(x)

    todo = [r for r in prompts if r["prompt_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    t0 = time.time()
    with open(path, "a") as fo:
        for ci in range(0, len(todo), args.chunk):
            chunk = todo[ci:ci + args.chunk]
            reqs, sps, meta = [], [], []
            for r in chunk:
                p = chat_ids(r["prompt"])
                if len(p) > args.max_prompt:
                    continue
                seed = int(hashlib.sha256(r["prompt_id"].encode()).hexdigest()[:8], 16)
                reqs.append(TokensPrompt(prompt_token_ids=p))
                sps.append(SamplingParams(n=1, temperature=1.0, top_p=0.95, top_k=20, max_tokens=args.max_new, seed=seed))
                meta.append((r, p))
            outs = llm.generate(reqs, sps, use_tqdm=False)
            for (r, p), o in zip(meta, outs):
                c = o.outputs[0]
                rec = {"prompt_id": r["prompt_id"], "prompt_ids": p, "reply_ids": list(c.token_ids),
                       "finish": c.finish_reason}
                fo.write(json.dumps(rec) + "\n")
                total += len(c.token_ids)
            fo.flush()
            log(f"gen chunk {ci//args.chunk+1}: {len(meta)} prompts, total {total} reply tokens, {total/(time.time()-t0):.0f} tok/s (this run)")
            if total >= args.target_tokens:
                break
    log("GEN_DONE", total)


# ------------------------------------------------------------------------------------------------ cap
def step_cap():
    import torch
    EOS = {248046, 248044}
    recs = [json.loads(l) for l in open(os.path.join(args.out, "chat_gen.jsonl"))]
    if args.limit:
        recs = recs[:args.limit]
    items = []
    for si, r in enumerate(recs):
        rep = r["reply_ids"]
        R = len(rep) - (1 if rep and rep[-1] in EOS else 0)
        P = len(r["prompt_ids"])
        items.append({"si": si, "prompt_id": r["prompt_id"], "P": P, "R": R, "ids": r["prompt_ids"] + rep[:R]})
    n_states = sum(it["R"] + 1 for it in items)
    log(f"cap: {len(items)} seqs, {n_states} states (P-1 .. P+R-1), {sum(it['R'] for it in items)} reply tokens")
    # shards in original order so the token store is easy to index; batches by length inside each shard
    shards, cur, ct = [], [], 0
    for it in items:
        cur.append(it); ct += it["R"] + 1
        if ct >= args.shard_tokens:
            shards.append(cur); cur, ct = [], 0
    if cur:
        shards.append(cur)
    cdir = os.path.join(args.out, "chat_codes")
    os.makedirs(cdir, exist_ok=True)
    H, S = load_model_and_sae()
    if args.web_btk and not os.path.exists(os.path.join(args.out, "repro_btk.json")):
        web_btk(H, S)
    t0 = time.time()
    for shi, sh in enumerate(shards):
        path = os.path.join(cdir, f"shard_{shi:03d}.npz")
        if os.path.exists(path):
            continue
        acc = Acc(131072)
        keep = {"row": [], "lat": [], "act": [], "base": 0}
        order = sorted(range(len(sh)), key=lambda i: len(sh[i]["ids"]))
        states = {}
        b, bl = [], []
        def flush(b):
            h = forward_l42(H, [sh[i]["ids"] for i in b])
            for k, i in enumerate(b):
                it = sh[i]
                states[i] = h[k, it["P"] - 1: it["P"] + it["R"]].float()
        for i in order:
            T = len(sh[i]["ids"])
            if b and T * (len(b) + 1) > args.max_tokens_per_batch:
                flush(b); b = []
            b.append(i)
        if b:
            flush(b)
        X = torch.cat([states[i] for i in range(len(sh))])
        encode_chunks(S, X, acc, keep=keep)
        row = torch.cat(keep["row"]).numpy(); lat = torch.cat(keep["lat"]).numpy(); act = torch.cat(keep["act"]).numpy()
        assert (np.diff(row) >= 0).all()
        ptr = np.searchsorted(row, np.arange(X.shape[0] + 1)).astype(np.int64)
        meta = [{k: v for k, v in sh[i].items() if k != "ids"} for i in range(len(sh))]
        tok = np.concatenate([np.array(sh[i]["ids"][sh[i]["P"] - 1:], dtype=np.int32) for i in range(len(sh))])
        assert tok.shape[0] == X.shape[0]
        sm = acc.summary()
        np.savez(path + ".tmp.npz", ptr=ptr, lat=lat.astype(np.int32), act=act.astype(np.float16), tok=tok,
                 meta=np.array(json.dumps(meta)), summary=np.array(json.dumps(sm)),
                 sx=acc.sx.cpu().numpy(), sxx=acc.sxx.cpu().numpy(), se=acc.se.cpu().numpy(), see=acc.see.cpu().numpy(),
                 l0=torch.cat(acc.l0).numpy().astype(np.int32), cos=torch.cat(acc.cos).numpy().astype(np.float32))
        os.replace(path + ".tmp.npz", path)
        log(f"cap shard {shi+1}/{len(shards)}: {X.shape[0]} states, L0 {sm['l0_mean']:.1f}, fve {sm['fve_global']:.3f}, "
            f"{time.time()-t0:.0f}s")
        del X, states
    log("CAP_DONE")


if __name__ == "__main__":
    {"repro": step_repro, "prompts": step_prompts, "gen": step_gen, "cap": step_cap}[args.step]()
