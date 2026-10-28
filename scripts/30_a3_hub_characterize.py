"""A3 step 1a: characterize the top hub candidates from S1 (runs on the VM CPU where the npz live).

For each neuron in {top-N hub candidates, chemtrails leaders, 20 random same-layer controls per hub layer}
and each behavior:
  - pooled matched-vs-unmatched d per position (behavior/ npz; confounded by prompt identity)
  - SIGNED within-prompt mean d per position asst_1..8 + asst_mean (prompt/ npz), its z against the
    sampling null, the fraction of prompts with d>0, position of max |d|
  - the same neuron's d under shuffled labels (shuffled/ npz) and its percentile rank among all 1.1M
    neurons at that position (a hub must NOT be a top neuron on the shuffled floor)
  - signed prompt-level Spearman rho (last prompt token activation vs the prompt's 64-rollout rate)
Writes <out>/characterize.parquet and <out>/characterize_summary.json.

VM: nice -n 10 ~/venv/bin/python -I scripts/30_a3_hub_characterize.py --s1 /data/captures/s1 --out /data/runs/a3_hubs
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd
from harness import POSITIONS, cohens_d
from a3_common import BEHAVIORS, CHEM_LEADERS, PRE_POS, WITHIN_POS, top_hubs, random_neurons, spearman_cols, N_LAYERS, D_MLP

ap = argparse.ArgumentParser()
ap.add_argument("--s1", default="/data/captures/s1")
ap.add_argument("--out", default="/data/runs/a3_hubs")
ap.add_argument("--top", type=int, default=30)
ap.add_argument("--hub-parquet", default=None)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()

hubs = top_hubs(args.top, args.hub_parquet)
sets = {hn: "hub" for hn in hubs}
for hn in CHEM_LEADERS:
    sets.setdefault(hn, "chem_leader")
for L in sorted({l for l, _ in hubs}):
    for n in random_neurons(L, 20):
        sets.setdefault((L, n), "random_same_layer")
neurons = list(sets.keys())
LI = np.array([l for l, _ in neurons]); NI = np.array([n for _, n in neurons])
print(f"[char] {len(neurons)} neurons ({sum(v=='hub' for v in sets.values())} hubs, "
      f"{sum(v=='random_same_layer' for v in sets.values())} random)", flush=True)


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files if k != "meta"}


def d_sel(m, u, pos):
    """Cohen's d for the selected neurons only, plus full-array d when asked (for rank)."""
    i = POSITIONS.index(pos)
    nm, nu = m["count"][i], u["count"][i]
    d = cohens_d(m[f"mean_{pos}"], m[f"m2_{pos}"], nm, u[f"mean_{pos}"], u[f"m2_{pos}"], nu)
    return d, nm, nu


manifest = json.load(open(f"{args.s1}/manifest.json"))
rows = []
for b in BEHAVIORS:
    m = load(f"{args.s1}/behavior/{b}__matched.npz"); u = load(f"{args.s1}/behavior/{b}__unmatched.npz")
    sm = load(f"{args.s1}/shuffled/{b}__matched.npz"); su = load(f"{args.s1}/shuffled/{b}__unmatched.npz")
    pooled, shuf, shuf_pct = {}, {}, {}
    for pos in POSITIONS:
        d, nm, nu = d_sel(m, u, pos); pooled[pos] = d[LI, NI]
        ds, _, _ = d_sel(sm, su, pos); a = np.abs(ds)
        shuf[pos] = ds[LI, NI]
        # percentile rank of |shuffled d| among all neurons (1.0 = the most extreme neuron on the floor)
        order = np.sort(a.ravel())
        shuf_pct[pos] = np.searchsorted(order, a[LI, NI]) / order.size
        shuf_max = float(a.max())
    # within-prompt
    pfiles = sorted(glob.glob(f"{args.s1}/prompt/{b}__*__matched.npz"))
    P = len(pfiles)
    acc = {pos: np.zeros(len(neurons)) for pos in WITHIN_POS}
    npos = {pos: np.zeros(len(neurons)) for pos in WITHIN_POS}
    nullvar = {pos: 0.0 for pos in WITHIN_POS}
    for pf in pfiles:
        pm = load(pf); pu = load(pf.replace("__matched.npz", "__unmatched.npz"))
        for pos in WITHIN_POS:
            i = POSITIONS.index(pos); nm, nu = pm["count"][i], pu["count"][i]
            if nm < 2 or nu < 2:
                continue
            d = cohens_d(pm[f"mean_{pos}"][LI, NI], pm[f"m2_{pos}"][LI, NI], nm,
                         pu[f"mean_{pos}"][LI, NI], pu[f"m2_{pos}"][LI, NI], nu)
            acc[pos] += d; npos[pos] += (d > 0); nullvar[pos] += (1 / nm + 1 / nu)
    for j, (L, N) in enumerate(neurons):
        r = dict(layer=L, neuron=N, set=sets[(L, N)], behavior=b, n_prompts_within=P,
                 n_matched=int(m["count"][0]), n_unmatched=int(u["count"][0]), shuf_floor_max=shuf_max)
        for pos in POSITIONS:
            r[f"dpool_{pos}"] = float(pooled[pos][j]); r[f"dshuf_{pos}"] = float(shuf[pos][j])
            r[f"dshuf_pct_{pos}"] = float(shuf_pct[pos][j])
        for pos in WITHIN_POS:
            dw = acc[pos][j] / max(P, 1)
            r[f"dwithin_{pos}"] = float(dw)
            r[f"zwithin_{pos}"] = float(dw / (np.sqrt(nullvar[pos]) / P)) if P and nullvar[pos] > 0 else 0.0
            r[f"fracpos_{pos}"] = float(npos[pos][j] / P) if P else np.nan
        pre_pool = np.array([pooled[p][j] for p in PRE_POS]); pre_w = np.array([acc[p][j] / max(P, 1) for p in PRE_POS])
        k = int(np.argmax(np.abs(pre_pool))); r["dpool_max_signed"] = float(pre_pool[k]); r["dpool_max_pos"] = PRE_POS[k]
        k = int(np.argmax(np.abs(pre_w))); r["dwithin_max_signed"] = float(pre_w[k]); r["dwithin_max_pos"] = PRE_POS[k]
        r["dwithin_max_z"] = r[f"zwithin_{PRE_POS[k]}"]; r["dwithin_max_fracpos"] = r[f"fracpos_{PRE_POS[k]}"]
        r["dshuf_max_abs_pre"] = float(max(abs(shuf[p][j]) for p in PRE_POS))
        r["dshuf_max_pct_pre"] = float(max(shuf_pct[p][j] for p in PRE_POS))
        rows.append(r)
    print(f"[char] {b}: {P} prompts, {time.time()-T0:.0f}s", flush=True)

df = pd.DataFrame(rows)

# ---- signed prompt-level rho
acts = np.load(f"{args.s1}/prompt_level_acts.npy", mmap_mode="r")
idx = pd.read_parquet(f"{args.s1}/prompt_index.parquet").sort_values("row")
A = np.stack([np.asarray(acts[:, L, N]) for L, N in neurons], axis=1)  # [403, K]
rho_rows = []
for b, g in idx.groupby("behavior_id"):
    rho = spearman_cols(A[g.row.values], g.match_rate.values)
    for j, (L, N) in enumerate(neurons):
        rho_rows.append(dict(layer=L, neuron=N, behavior=b, rho=float(rho[j]), n_prompts_rho=len(g)))
rho_all = spearman_cols(A, idx.match_rate.values)
rdf = pd.DataFrame(rho_rows)
df = df.merge(rdf, on=["layer", "neuron", "behavior"], how="left")
pooled_rho = {f"L{L}N{N}": float(rho_all[j]) for j, (L, N) in enumerate(neurons)}
df.to_parquet(f"{args.out}/characterize.parquet", index=False)

# ---- summary per neuron (powered behaviors only for sign consistency)
from a3_common import POWERED
summ = []
for (L, N), g in df.groupby(["layer", "neuron"]):
    gp = g[g.behavior.isin(POWERED)].set_index("behavior")
    sw = gp.dwithin_max_signed; sp_ = gp.dpool_max_signed
    summ.append(dict(layer=int(L), neuron=int(N), set=sets[(L, N)],
                     n_within_pos=int((sw >= 0.3).sum()), n_within_neg=int((sw <= -0.3).sum()),
                     n_pool_pos=int((sp_ >= 0.5).sum()), n_pool_neg=int((sp_ <= -0.5).sum()),
                     n_within_z3=int((gp.dwithin_max_z.abs() >= 3).sum()),
                     sign_consistency_within=float(abs(np.sign(sw).sum()) / max((sw.abs() >= 0.1).sum(), 1)),
                     mean_dwithin=float(sw.mean()), mean_dpool=float(sp_.mean()),
                     n_rho_ge_thr=int((gp.rho.abs() >= 0.45).sum()), mean_rho_signed=float(gp.rho.mean()),
                     rho_pooled_403=pooled_rho[f"L{L}N{N}"],
                     max_dshuf_pct=float(g.dshuf_max_pct_pre.max()), max_dshuf_abs=float(g.dshuf_max_abs_pre.max()),
                     within_profile={b: round(float(v), 3) for b, v in sw.items()},
                     within_pos={b: gp.dwithin_max_pos[b] for b in gp.index},
                     rho_profile={b: round(float(v), 3) for b, v in gp.rho.items()}))
sdf = pd.DataFrame(summ).sort_values(["set", "n_within_z3"], ascending=[True, False])
sdf.to_json(f"{args.out}/characterize_summary.json", orient="records", indent=1)
pd.set_option("display.width", 250)
print(sdf.drop(columns=["within_profile", "within_pos", "rho_profile"]).to_string())
print(f"[char] DONE {time.time()-T0:.0f}s")
