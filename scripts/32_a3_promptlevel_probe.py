"""A3 step 1c (P2 step 1, the 'weirdness probe' test). VM CPU.

Data: /data/captures/s1/prompt_level_acts.npy [403 prompts, 64, 17408] = post-activation of every neuron at the
last prompt token (one deterministic forward per unique WeirdChat prompt), prompt_index.parquet (behavior_id,
match_rate = the prompt's shipped 64-rollout rate).

1. Spearman rho(activation, rate) for each hub candidate: pooled over all 403 prompts and within each behavior,
   against (a) 10,000 random neurons (seed 1234) and (b) a permutation null (rates shuffled within behavior).
2. Probes for the per-prompt rate (logit scale), evaluated by held-out Spearman and AUROC(rate>=25% vs <5%):
   - dense ridge on all 1.1M neurons (dual form), leave-one-behavior-out (LOBO) and 10-fold random CV
   - one neuron: each hub candidate alone (sign fit on train); and the best single neuron SELECTED on the
     training folds among all 1.1M (the honest 1-neuron probe)
   - floors: behavior-mean rate (10-fold only), random single neuron
Writes <out>/probe_rho.parquet, <out>/probe_summary.json.

VM: nice -n 10 ~/venv/bin/python -I scripts/32_a3_promptlevel_probe.py --s1 /data/captures/s1 --out /data/runs/a3_hubs
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
from a3_common import CHEM_LEADERS, top_hubs, spearman_cols, N_LAYERS, D_MLP

ap = argparse.ArgumentParser()
ap.add_argument("--s1", default="/data/captures/s1")
ap.add_argument("--out", default="/data/runs/a3_hubs")
ap.add_argument("--top", type=int, default=30)
ap.add_argument("--n-random", type=int, default=10000)
ap.add_argument("--n-perm", type=int, default=1000)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
F = N_LAYERS * D_MLP

idx = pd.read_parquet(f"{args.s1}/prompt_index.parquet").sort_values("row").reset_index(drop=True)
assert (idx.row.values == np.arange(len(idx))).all()
X = np.load(f"{args.s1}/prompt_level_acts.npy").reshape(len(idx), F)  # float32, 1.8 GB
rate = idx.match_rate.values.astype(np.float64)
k_m = idx.n_matched.values; n_s = idx.n_samples.values
y = np.log((k_m + 0.5) / (n_s - k_m + 0.5))  # empirical logit of the per-prompt rate
beh = idx.behavior_id.values
behaviors = sorted(set(beh))
print(f"[probe] X {X.shape}, {len(behaviors)} behaviors, loaded in {time.time()-T0:.0f}s", flush=True)

hubs = top_hubs(args.top) + [c for c in CHEM_LEADERS if c not in top_hubs(args.top)]
hub_flat = np.array([L * D_MLP + N for L, N in hubs])
rng = np.random.default_rng(1234)
rand_flat = rng.choice(F, args.n_random, replace=False)

# ------------------------------------------------------------------ 1. Spearman vs nulls
rows, null_rows = [], []
groups = [(b, np.where(beh == b)[0]) for b in behaviors] + [("ALL_POOLED", np.arange(len(idx)))]
for b, sel in groups:
    if len(sel) < 8:
        continue
    r_hub = spearman_cols(X[sel][:, hub_flat], rate[sel])
    r_rand = spearman_cols(X[sel][:, rand_flat], rate[sel])
    a_rand = np.abs(r_rand)
    # permutation null for the hub set: shuffle rates (within behavior when pooled)
    perm_max_hub, perm_abs_hub = [], []
    for _ in range(args.n_perm):
        if b == "ALL_POOLED":
            yp = rate.copy()
            for bb in behaviors:
                s = np.where(beh == bb)[0]; yp[s] = rng.permutation(rate[s])
        else:
            yp = rng.permutation(rate[sel])
        rp = spearman_cols(X[sel][:, hub_flat], yp)
        perm_max_hub.append(np.abs(rp).max()); perm_abs_hub.append(np.abs(rp))
    perm_abs_hub = np.stack(perm_abs_hub)  # [n_perm, K]
    for j, (L, N) in enumerate(hubs):
        a = abs(r_hub[j])
        rows.append(dict(layer=L, neuron=N, behavior=b, n_prompts=len(sel), rho=float(r_hub[j]),
                         pct_vs_random_neurons=float((a_rand < a).mean()),
                         p_perm_single=float((perm_abs_hub[:, j] >= a).mean()),
                         p_perm_family=float((np.array(perm_max_hub) >= a).mean())))
    null_rows.append(dict(behavior=b, n_prompts=len(sel), rand_p50=float(np.median(a_rand)), rand_p90=float(np.quantile(a_rand, .9)),
                          rand_p99=float(np.quantile(a_rand, .99)), rand_p999=float(np.quantile(a_rand, .999)), rand_max=float(a_rand.max()),
                          hub_max=float(np.abs(r_hub).max()), perm_max_hubset_p95=float(np.quantile(perm_max_hub, .95))))
    print(f"[probe] {b}: n={len(sel)} hub max|rho|={np.abs(r_hub).max():.3f} random p99={np.quantile(a_rand,.99):.3f} "
          f"max={a_rand.max():.3f} perm95(hubset max)={np.quantile(perm_max_hub,.95):.3f}", flush=True)
pd.DataFrame(rows).to_parquet(f"{args.out}/probe_rho.parquet", index=False)
pd.DataFrame(null_rows).to_parquet(f"{args.out}/probe_rho_null.parquet", index=False)

# ------------------------------------------------------------------ 2. probes
def auroc(score, r):
    pos = score[r >= 0.25]; neg = score[r < 0.05]
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    return float((rankdata(np.concatenate([pos, neg]))[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def evaluate(pred, te):
    s = spearmanr(pred, rate[te]).correlation if len(te) >= 5 else np.nan
    return float(s) if np.isfinite(s) else np.nan, auroc(pred, rate[te])

def ridge_dual(Xtr, ytr, Xte, alphas, inner_folds=5, seed=0):
    """Dense ridge via the kernel K = Xtr Xtr^T; alpha picked by inner CV on Spearman. Features standardized on train."""
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd; Zte = (Xte - mu) / sd
    K = Ztr @ Ztr.T; Kte = Zte @ Ztr.T
    ym = ytr.mean(); yc = ytr - ym
    n = len(ytr); best, best_a = -np.inf, alphas[0]
    folds = np.array_split(np.random.default_rng(seed).permutation(n), inner_folds)
    for a in alphas:
        sc = []
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            coef = np.linalg.solve(K[np.ix_(tr, tr)] + a * np.eye(len(tr)), yc[tr])
            p = K[np.ix_(f, tr)] @ coef
            s = spearmanr(p, yc[f]).correlation
            sc.append(s if np.isfinite(s) else 0)
        if np.mean(sc) > best:
            best, best_a = np.mean(sc), a
    coef = np.linalg.solve(K + best_a * np.eye(n), yc)
    return Kte @ coef + ym, best_a, float(best)

alphas = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]  # K has scale ~F=1.1M per diagonal entry (standardized features)
alphas = [a * F / 1e3 for a in alphas]
res = {"lobo": [], "kfold": []}

def run_fold(name, tr, te, fold_label):
    out = dict(scheme=name, fold=fold_label, n_train=len(tr), n_test=len(te))
    pred, a, inner = ridge_dual(X[tr], y[tr], X[te], alphas)
    out["dense_rho"], out["dense_auroc"] = evaluate(pred, te); out["dense_alpha"] = a; out["dense_inner_rho"] = inner
    # best single neuron selected on train (by |Spearman| with rate on the training prompts)
    r_tr = spearman_cols(X[tr], rate[tr])
    jbest = int(np.argmax(np.abs(r_tr)))
    sgn = np.sign(r_tr[jbest])
    out["best1_rho"], out["best1_auroc"] = evaluate(sgn * X[te, jbest], te)
    out["best1_neuron"] = f"L{jbest // D_MLP}N{jbest % D_MLP}"; out["best1_train_rho"] = float(r_tr[jbest])
    # top-10 selected neurons, ridge on them (small multi-neuron probe)
    top10 = np.argsort(-np.abs(r_tr))[:10]
    pred10, _, _ = ridge_dual(X[tr][:, top10], y[tr], X[te][:, top10], [1e-1, 1, 10, 100, 1000])
    out["top10_rho"], out["top10_auroc"] = evaluate(pred10, te)
    # each hub candidate alone
    for j, (L, N) in enumerate(hubs):
        f = L * D_MLP + N
        sgn = np.sign(spearmanr(X[tr, f], rate[tr]).correlation or 1.0)
        out[f"hub_L{L}N{N}_rho"], _ = evaluate(sgn * X[te, f], te)
    # random single neurons (first 50 of the random set), sign fit on train
    rr = []
    for f in rand_flat[:50]:
        sgn = np.sign(spearmanr(X[tr, f], rate[tr]).correlation or 1.0)
        s, _ = evaluate(sgn * X[te, f], te); rr.append(s)
    out["random1_rho_mean"] = float(np.nanmean(rr)); out["random1_rho_p95"] = float(np.nanquantile(rr, .95))
    # behavior-mean floor (only meaningful when the test behaviors appear in train)
    bm = pd.Series(rate[tr]).groupby(beh[tr]).mean()
    pred_bm = np.array([bm.get(b, np.nan) for b in beh[te]])
    out["behmean_rho"], _ = evaluate(np.nan_to_num(pred_bm, nan=rate[tr].mean()), te) if np.isfinite(pred_bm).any() else (np.nan, np.nan)
    print(f"[probe] {name} {fold_label}: dense rho={out['dense_rho']:.3f} auroc={out['dense_auroc']:.3f} | best1 {out['best1_neuron']} "
          f"rho={out['best1_rho']:.3f} | top10 rho={out['top10_rho']:.3f} | random1 mean={out['random1_rho_mean']:.3f} "
          f"| behmean {out['behmean_rho']:.3f} ({time.time()-T0:.0f}s)", flush=True)
    return out

all_i = np.arange(len(idx))
for b in behaviors:
    te = np.where(beh == b)[0]
    if len(te) < 8:
        continue
    res["lobo"].append(run_fold("lobo", np.setdiff1d(all_i, te), te, b))
folds = np.array_split(np.random.default_rng(7).permutation(len(idx)), 10)
for i, te in enumerate(folds):
    res["kfold"].append(run_fold("kfold", np.setdiff1d(all_i, te), te, f"fold{i}"))

# pooled k-fold predictions summary: mean over folds
def agg(rs, key):
    v = [r[key] for r in rs if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
    return float(np.mean(v)) if v else np.nan
summary = {}
for scheme in ("lobo", "kfold"):
    rs = res[scheme]
    summary[scheme] = {k: agg(rs, k) for k in ["dense_rho", "dense_auroc", "best1_rho", "best1_auroc", "top10_rho", "random1_rho_mean", "behmean_rho"]}
    summary[scheme]["best1_neurons"] = sorted({r["best1_neuron"] for r in rs})
    summary[scheme]["hubs"] = {f"L{L}N{N}": agg(rs, f"hub_L{L}N{N}_rho") for L, N in hubs}
    summary[scheme]["per_fold"] = rs
json.dump(summary, open(f"{args.out}/probe_summary.json", "w"), indent=1, default=float)
print(json.dumps({s: {k: v for k, v in summary[s].items() if k not in ("per_fold", "hubs")} for s in summary}, indent=1, default=float))
print(f"[probe] DONE {time.time()-T0:.0f}s")
