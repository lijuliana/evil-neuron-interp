"""A2 wave 1: Goodfire "sober one" clause-deletion replication on Qwen3.6-27B + S2 activation screen.

(i)  Sample N responses for each of the 20 recommends-drunk-driving originals and their 20 benign
     rewrites (promptsets/a2_goodfire_pairs.jsonl); optionally the 20 near-miss prompts at a smaller N.
     Heuristic-judge (src/heuristic_judge.py, P>0.9 for this behavior). Rates with Wilson CIs.
(ii) Judge-free readout: first-token distribution at the last prompt token for all prompts
     (log P(yes-type opener) vs P(no-type opener)).
(iii) S2 capture: last-prompt-token activations of all 64x17408 neurons for original / rewrite /
     near-miss prompts (paired contrast original minus rewrite), and teacher-forced asst_1..8 / asst_mean
     Welford stats of the sampled responses grouped by {orig, rewr, orig_match, orig_nomatch}.
     Writes top-k contrast tables (parquet) + full last-prompt acts (npy).

Run on VM (under flock):
  flock -w 36000 /data/gpu.lock python -I scripts/21_a2_goodfire_s2.py --out /data/runs/a2_goodfire --n 64
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd, torch
from harness import Harness, GroupAcc, POSITIONS, N_LAYERS, D_MLP, cohens_d
from heuristic_judge import judge_heuristic, distinct_ngram_ratio

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "promptsets", "a2_goodfire_pairs.jsonl"))
ap.add_argument("--out", default="/data/runs/a2_goodfire")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--n-near", type=int, default=16)
ap.add_argument("--max-new", type=int, default=400)
ap.add_argument("--batch", type=int, default=80)
ap.add_argument("--top", type=int, default=3000)
ap.add_argument("--skip-sampling", action="store_true")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
pairs = [json.loads(l) for l in open(args.pairs)]
BEH = "recommends-drunk-driving"
T0 = time.time()
h = Harness(resid_norms_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captures", "qwen3.6-27b_resid_norms.json"))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - hw), min(1.0, c + hw))


# ------------------------------------------------------------------ (i) sampling
sets = {"original": ("original", args.n), "benign_rewrite": ("benign_rewrite", args.n), "near_miss": ("near_miss", args.n_near)}
samples_path = f"{args.out}/samples.jsonl"
records = []
if os.path.exists(samples_path):
    records = [json.loads(l) for l in open(samples_path)]
    print(f"[a2] loaded {len(records)} existing samples", flush=True)
elif not args.skip_sampling:
    with open(samples_path, "w") as fo:
        for set_name, (field, n) in sets.items():
            if n <= 0:
                continue
            prompts = [p[field] for p in pairs]
            t0 = time.time()
            outs = h.chat(prompts, n=n, max_new_tokens=args.max_new, temperature=1.0, top_p=0.95, top_k=20,
                          batch_size=args.batch, seed=0)
            for p, resp in zip(pairs, outs):
                for i, text in enumerate(resp):
                    j = judge_heuristic(BEH, p[field], text)
                    r = dict(set=set_name, pair_id=p["pair_id"], source_prompt_id=p["source_prompt_id"], sample_idx=i,
                             prompt=p[field], response=text, heur_match=bool(j["match"]), heur_why=j["explanation"],
                             distinct4=distinct_ngram_ratio(text), n_words=len(text.split()),
                             sampling=dict(temperature=1.0, top_p=0.95, top_k=20, max_new_tokens=args.max_new, dtype="bf16", seed=0))
                    fo.write(json.dumps(r) + "\n"); records.append(r)
            print(f"[a2] sampled {set_name}: {len(prompts)} x {n} in {time.time()-t0:.0f}s", flush=True)
df = pd.DataFrame(records) if records else None
summary = {"n_pairs": len(pairs), "max_new_tokens": args.max_new}
if df is not None:
    for s, g in df.groupby("set"):
        k, n = int(g.heur_match.sum()), len(g); lo, hi = wilson(k, n)
        summary[s] = dict(n=n, heur_matched=k, rate=k / n, ci95=[lo, hi], mean_words=float(g.n_words.mean()),
                          per_prompt_rate={int(pid): float(gg.heur_match.mean()) for pid, gg in g.groupby("pair_id")})
        print(f"[a2] {s}: heuristic rate {k}/{n} = {k/n:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)

# ------------------------------------------------------------------ (ii) first-token readout
tok = h.tok
def first_ids(words):
    ids = set()
    for w in words:
        for v in (w, " " + w, "**" + w):
            t = tok(v, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.add(t[0])
    return sorted(ids)
YES_IDS = first_ids(["Yes", "YES", "Absolutely", "Definitely", "Yeah"])
NO_IDS = first_ids(["No", "NO", "Please", "I", "This", "Actually", "Stop", "Do", "Don", "Absolutely"])
NO_IDS = [i for i in NO_IDS if i not in YES_IDS]
summary["yes_ids"] = {int(i): tok.decode([i]) for i in YES_IDS}; summary["no_ids"] = {int(i): tok.decode([i]) for i in NO_IDS}

@torch.no_grad()
def first_token_logprobs(prompts, batch=20):
    out = []
    pad = tok.pad_token_id
    for s in range(0, len(prompts), batch):
        chunk = [h.prompt_ids(p) for p in prompts[s:s + batch]]
        L = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long); attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for b, c in enumerate(chunk):
            ids[b, L - len(c):] = torch.tensor(c); attn[b, L - len(c):] = 1
        logits = h.model(input_ids=ids.cuda(), attention_mask=attn.cuda(), use_cache=False).logits[:, -1].float()
        lp = torch.log_softmax(logits, -1)
        for b in range(len(chunk)):
            top = torch.topk(lp[b], 15)
            out.append(dict(logp_yes=float(torch.logsumexp(lp[b, YES_IDS], 0)), logp_no=float(torch.logsumexp(lp[b, NO_IDS], 0)),
                            top=[(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)]))
    return out

ft = {}
for field in ("original", "benign_rewrite", "near_miss"):
    res = first_token_logprobs([p[field] for p in pairs])
    ft[field] = res
    lo = np.array([r["logp_yes"] - r["logp_no"] for r in res])
    summary[f"first_token_{field}"] = dict(mean_logodds_yes_vs_no=float(lo.mean()), mean_p_yes=float(np.mean([np.exp(r["logp_yes"]) for r in res])),
                                            per_prompt_p_yes=[float(np.exp(r["logp_yes"])) for r in res])
    print(f"[a2] first-token {field}: mean P(yes-opener) = {summary[f'first_token_{field}']['mean_p_yes']:.3f}", flush=True)
json.dump(ft, open(f"{args.out}/first_token.json", "w"), indent=1)

if df is not None:
    # prompt-level: does the judge-free first-token readout track the judged rate?
    for field, set_name in (("original", "original"), ("benign_rewrite", "benign_rewrite")):
        py = np.array([np.exp(r["logp_yes"]) for r in ft[field]])
        rate = np.array([summary[set_name]["per_prompt_rate"].get(int(p["pair_id"]), 0.0) for p in pairs])
        summary[f"corr_pyes_rate_{set_name}"] = float(np.corrcoef(py, rate)[0, 1])
        print(f"[a2] corr(P(yes-opener), judged rate) on {set_name} = {summary[f'corr_pyes_rate_{set_name}']:.2f}", flush=True)

# ------------------------------------------------------------------ (iii) S2 capture at the last prompt token
acts_path = f"{args.out}/last_prompt_acts.npy"
fields = ["original", "benign_rewrite", "near_miss"]
if not os.path.exists(acts_path):
    t0 = time.time()
    allp = [p[f] for f in fields for p in pairs]
    A = h.capture_prompt_level(allp, batch_size=10)  # [3*20, 64, 17408]
    np.save(acts_path, A)
    print(f"[a2] last-prompt acts {A.shape} in {time.time()-t0:.0f}s", flush=True)
else:
    A = np.load(acts_path)
P = len(pairs)
orig, rewr, near = A[:P], A[P:2 * P], A[2 * P:3 * P]

def paired_table(a, b, label):
    """Paired per-neuron contrast a minus b over the 20 pairs: mean diff, SD, t, Cohen's dz."""
    dif = (a - b).astype(np.float64)
    mu = dif.mean(0); sd = dif.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = np.where(sd > 1e-9, mu / sd, 0.0)
    t = dz * np.sqrt(dif.shape[0])
    flat = np.abs(dz).ravel()
    k = min(args.top, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]; idx = idx[np.argsort(-flat[idx])]
    L, N = np.unravel_index(idx, mu.shape)
    return pd.DataFrame(dict(layer=L, neuron=N, contrast=label, mean_diff=mu[L, N], sd_diff=sd[L, N],
                             dz=dz[L, N], t=t[L, N], mean_a=a.mean(0)[L, N], mean_b=b.mean(0)[L, N]))

tabs = [paired_table(orig, rewr, "orig_minus_rewrite"), paired_table(orig, near, "orig_minus_nearmiss"),
        paired_table(near, rewr, "nearmiss_minus_rewrite")]
s2 = pd.concat(tabs, ignore_index=True)
s2.to_parquet(f"{args.out}/s2_lastprompt_contrast.parquet", index=False)
print(f"[a2] S2 top |dz| orig-vs-rewrite: {s2[s2.contrast=='orig_minus_rewrite'].head(10)[['layer','neuron','dz','t']].to_string(index=False)}", flush=True)

# ------------------------------------------------------------------ (iv) teacher-forced asst_1..8 stats of our own samples
if df is not None and not os.path.exists(f"{args.out}/s2_asst_contrast.parquet"):
    groups = {}
    t0 = time.time()
    buf = []
    def flush():
        if not buf:
            return
        x, valid = h.capture_batch([s for s, _ in buf])
        keys = [k for _, k in buf]
        for k in set(keys):
            sel = torch.tensor([kk == k for kk in keys])
            groups.setdefault(k, GroupAcc("cuda")).update(x[sel.cuda()], valid[sel])
        buf.clear()
    sub = df[df.set.isin(["original", "benign_rewrite"])]
    for _, r in sub.iterrows():
        msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
        key = r["set"] if r["set"] == "benign_rewrite" else ("orig_match" if r["heur_match"] else "orig_nomatch")
        buf.append((h.transcript_ids(msgs, 256), key))
        if len(buf) >= 16:
            flush()
    flush()
    rows = []
    WPOS = [f"asst_{k}" for k in range(1, 9)] + ["asst_mean"]
    def grp_d(ga, gb, pos):
        i = POSITIONS.index(pos)
        return cohens_d(ga.mean[i].cpu().numpy(), ga.m2[i].cpu().numpy(), ga.count[i],
                        gb.mean[i].cpu().numpy(), gb.m2[i].cpu().numpy(), gb.count[i]), ga.count[i], gb.count[i]
    for ca, cb, label in (("orig_match", "orig_nomatch", "within_prompt_set_match_vs_nomatch"),
                          ("orig_match", "benign_rewrite", "origmatch_vs_rewrite")):
        if ca not in groups or cb not in groups:
            continue
        for pos in WPOS:
            d, na, nb = grp_d(groups[ca], groups[cb], pos)
            flat = np.abs(d).ravel(); k = min(args.top, flat.size)
            idx = np.argpartition(-flat, k - 1)[:k]; idx = idx[np.argsort(-flat[idx])]
            L, N = np.unravel_index(idx, d.shape)
            rows.append(pd.DataFrame(dict(layer=L, neuron=N, position=pos, contrast=label, d=d[L, N],
                                          n_a=int(na), n_b=int(nb))))
    if rows:
        pd.concat(rows, ignore_index=True).to_parquet(f"{args.out}/s2_asst_contrast.parquet", index=False)
    for k, ga in groups.items():
        summary[f"group_count_{k}"] = float(ga.count[0])
    print(f"[a2] asst-position contrast done in {time.time()-t0:.0f}s: {[ (k, float(v.count[0])) for k,v in groups.items() ]}", flush=True)

summary["elapsed_min"] = (time.time() - T0) / 60
json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1, default=float)
print("[a2] DONE", json.dumps({k: v for k, v in summary.items() if not isinstance(v, (dict, list))}), flush=True)
