"""A0: summarize S1 captures into small parquet tables (runs on the VM where the npz files live).

Outputs (in --out, default /data/captures/s1/summary):
  summary_{behavior}.parquet      top 2000 |d| neurons per position for two contrasts:
                                  contrast='behavior_pooled' (matched vs unmatched rollouts pooled over prompts;
                                  confounded by prompt identity) and contrast='within_prompt_mean' (mean over
                                  eligible prompts of the per-prompt d; positions asst_1..8 + asst_mean only).
                                  Columns: layer, neuron, position, contrast, d, n_matched, n_unmatched, n_prompts, null_sd
  noise_floor.parquet             shuffled-label control: per behavior x position, max|d|, 99.9/99.99 pct, #|d|>=0.5
  summary_promptlevel_{behavior}.parquet  top 2000 |rho| (Spearman of last-prompt activation vs prompt match rate)
  promptlevel_null.parquet        |rho| quantiles over 10k random neurons + permutation null (rates shuffled)
  hub_candidates.parquet          per neuron: #behaviors with max_{k=1..8}|d_pooled|>=0.5, #behaviors |d_within|>=0.5,
                                  #behaviors |rho|>=0.5, hub_score, per-behavior values. Top 5000.
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd
from scipy.stats import rankdata
from harness import POSITIONS, N_LAYERS, D_MLP, cohens_d

ap = argparse.ArgumentParser()
ap.add_argument("--s1", default="/data/captures/s1")
ap.add_argument("--out", default="/data/captures/s1/summary")
ap.add_argument("--top", type=int, default=2000)
ap.add_argument("--thr", type=float, default=0.5)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
manifest = json.load(open(f"{args.s1}/manifest.json"))
WITHIN_POS = [f"asst_{k}" for k in range(1, 9)] + ["asst_mean"]
PRE_POS = [f"asst_{k}" for k in range(1, 9)]  # pre-commitment-ish positions used for hub score

def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files if k != "meta"}

def d_from(m, u, pos):
    i = POSITIONS.index(pos)
    return cohens_d(m[f"mean_{pos}"], m[f"m2_{pos}"], m["count"][i], u[f"mean_{pos}"], u[f"m2_{pos}"], u["count"][i]), m["count"][i], u["count"][i]

def topk(arr, k):
    flat = np.abs(arr).ravel()
    idx = np.argpartition(-flat, min(k, flat.size - 1))[:k]
    idx = idx[np.argsort(-flat[idx])]
    return np.unravel_index(idx, arr.shape)

behaviors = sorted({g["behavior"] for g in manifest["groups"] if g["kind"] == "behavior"})
floor_rows, hub = [], {}
for b in behaviors:
    m = load(f"{args.s1}/behavior/{b}__matched.npz"); u = load(f"{args.s1}/behavior/{b}__unmatched.npz")
    rows = []
    maxabs_pooled = np.zeros((N_LAYERS, D_MLP), np.float32)
    for pos in POSITIONS:
        d, nm, nu = d_from(m, u, pos)
        if pos in PRE_POS:
            maxabs_pooled = np.maximum(maxabs_pooled, np.abs(d))
        L, N = topk(d, args.top)
        rows.append(pd.DataFrame(dict(layer=L, neuron=N, position=pos, contrast="behavior_pooled", d=d[L, N],
                                      n_matched=int(nm), n_unmatched=int(nu), n_prompts=np.nan, null_sd=np.sqrt(1 / nm + 1 / nu))))
    # shuffled-label noise floor
    sm = load(f"{args.s1}/shuffled/{b}__matched.npz"); su = load(f"{args.s1}/shuffled/{b}__unmatched.npz")
    for pos in POSITIONS:
        d, nm, nu = d_from(sm, su, pos); a = np.abs(d)
        floor_rows.append(dict(behavior=b, position=pos, n_matched=int(nm), n_unmatched=int(nu), max_abs_d=float(a.max()),
                               p999=float(np.quantile(a, 0.999)), p9999=float(np.quantile(a, 0.9999)),
                               n_ge_thr=int((a >= args.thr).sum()), n_ge_03=int((a >= 0.3).sum())))
    # within-prompt mean d
    pfiles = sorted(glob.glob(f"{args.s1}/prompt/{b}__*__matched.npz"))
    maxabs_within = np.zeros((N_LAYERS, D_MLP), np.float32)
    if pfiles:
        acc = {pos: np.zeros((N_LAYERS, D_MLP), np.float64) for pos in WITHIN_POS}
        nullvar = {pos: 0.0 for pos in WITHIN_POS}
        for pf in pfiles:
            pm = load(pf); pu = load(pf.replace("__matched.npz", "__unmatched.npz"))
            for pos in WITHIN_POS:
                d, nm, nu = d_from(pm, pu, pos)
                acc[pos] += d; nullvar[pos] += (1 / nm + 1 / nu)
        P = len(pfiles)
        for pos in WITHIN_POS:
            dw = (acc[pos] / P).astype(np.float32)
            if pos in PRE_POS:
                maxabs_within = np.maximum(maxabs_within, np.abs(dw))
            L, N = topk(dw, args.top)
            rows.append(pd.DataFrame(dict(layer=L, neuron=N, position=pos, contrast="within_prompt_mean", d=dw[L, N],
                                          n_matched=int(m["count"][POSITIONS.index(pos)]), n_unmatched=int(u["count"][POSITIONS.index(pos)]),
                                          n_prompts=P, null_sd=np.sqrt(nullvar[pos]) / P)))
    pd.concat(rows).to_parquet(f"{args.out}/summary_{b}.parquet", index=False)
    hub[b] = dict(d_pooled=maxabs_pooled, d_within=maxabs_within, n_prompts=len(pfiles))
    print(f"[summary] {b}: pooled max|d| asst_1..8 = {maxabs_pooled.max():.2f}, within-prompt ({len(pfiles)} prompts) max|d| = {maxabs_within.max():.2f}", flush=True)
pd.DataFrame(floor_rows).to_parquet(f"{args.out}/noise_floor.parquet", index=False)

# ---- prompt-level Spearman
rho_by_b = {}
null_rows = []
if os.path.exists(f"{args.s1}/prompt_level_acts.npy"):
    acts = np.load(f"{args.s1}/prompt_level_acts.npy", mmap_mode="r")
    idx = pd.read_parquet(f"{args.s1}/prompt_index.parquet")
    rng = np.random.default_rng(0)
    rand_neurons = rng.integers(0, N_LAYERS * D_MLP, 10000)

    def spearman(A, rate):  # A: [n, F] float32, rate: [n]
        R = rankdata(A, axis=0).astype(np.float32); r = rankdata(rate).astype(np.float32)
        R -= R.mean(0); r -= r.mean()
        den = np.sqrt((R ** 2).sum(0)) * np.sqrt((r ** 2).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = (R * r[:, None]).sum(0) / den
        rho[~np.isfinite(rho)] = 0.0
        return rho

    groups = [(b, g.row.values, g.match_rate.values) for b, g in idx.groupby("behavior_id") if len(g) >= 8]
    groups.append(("ALL_POOLED", idx.row.values, idx.match_rate.values))
    for b, rows_i, rate in groups:
        A = np.asarray(acts[np.sort(rows_i)]).reshape(len(rows_i), -1); rate = rate[np.argsort(rows_i)]
        rho = spearman(A, rate)
        rho2 = rho.reshape(N_LAYERS, D_MLP)
        rho_by_b[b] = (np.abs(rho2), len(rows_i))
        L, N = topk(rho2, args.top)
        pd.DataFrame(dict(layer=L, neuron=N, rho=rho2[L, N], n_prompts=len(rows_i))).to_parquet(
            f"{args.out}/summary_promptlevel_{b}.parquet", index=False)
        a = np.abs(rho[rand_neurons])
        perm_max, perm_p999 = [], []
        for _ in range(20):
            pr = np.abs(spearman(A[:, rand_neurons], rng.permutation(rate)))
            perm_max.append(pr.max()); perm_p999.append(np.quantile(pr, 0.999))
        null_rows.append(dict(behavior=b, n_prompts=len(rows_i), rand_p50=float(np.median(a)), rand_p90=float(np.quantile(a, .9)),
                              rand_p99=float(np.quantile(a, .99)), rand_p999=float(np.quantile(a, .999)), rand_max=float(a.max()),
                              all_max=float(np.abs(rho).max()), all_n_ge_05=int((np.abs(rho) >= 0.5).sum()),
                              perm_max_mean=float(np.mean(perm_max)), perm_max_max=float(np.max(perm_max)), perm_p999_mean=float(np.mean(perm_p999))))
        print(f"[promptlevel] {b}: n={len(rows_i)} max|rho|={np.abs(rho).max():.3f} #|rho|>=0.5: {(np.abs(rho)>=0.5).sum()} perm max|rho| (10k neurons) mean {np.mean(perm_max):.3f}", flush=True)
    pd.DataFrame(null_rows).to_parquet(f"{args.out}/promptlevel_null.parquet", index=False)

# ---- hub candidates
n_beh_d = np.zeros((N_LAYERS, D_MLP), np.int16); n_beh_dw = np.zeros_like(n_beh_d); n_beh_rho = np.zeros_like(n_beh_d)
sum_d = np.zeros((N_LAYERS, D_MLP), np.float32)
cols = {}
for b in behaviors:
    n_beh_d += (hub[b]["d_pooled"] >= args.thr); n_beh_dw += (hub[b]["d_within"] >= args.thr); sum_d += hub[b]["d_pooled"]
    cols[f"dpool_{b}"] = hub[b]["d_pooled"]; cols[f"dwithin_{b}"] = hub[b]["d_within"]
    if b in rho_by_b:
        n_beh_rho += (rho_by_b[b][0] >= 0.5); cols[f"rho_{b}"] = rho_by_b[b][0]
score = n_beh_d.astype(np.float32) + n_beh_rho + 1e-3 * sum_d + 1e-6 * n_beh_dw
L, N = topk(score, 5000)
df = pd.DataFrame(dict(layer=L, neuron=N, hub_score=score[L, N], n_beh_d_pooled_ge05=n_beh_d[L, N], n_beh_d_within_ge05=n_beh_dw[L, N],
                       n_beh_rho_ge05=n_beh_rho[L, N], sum_maxabs_d_pooled=sum_d[L, N]))
for k, v in cols.items():
    df[k] = v[L, N]
df.to_parquet(f"{args.out}/hub_candidates.parquet", index=False)
print(df.head(20).to_string())
print("[summary] DONE")
