"""P1 persona analysis (CPU). Reads the GPU job outputs (scripts/70_persona_p1_gpu.py) copied to --run and
Agastya's data; writes --run/analysis.json and prints the tables used in summary.md.

(a) last-prompt-token projection vs per-prompt rate (64-rollout off-judge rate; production-judge rate from 16), Spearman
    within behavior, blocks 16/32/48; nulls for the axis: 1000 isotropic random, 1000 covariance-matched, 1000 random
    role-pair differences (percentile = share of null directions the axis beats in the predicted, negative direction).
(b) projection at his cut points vs P(weird | prefix): within-reply Pearson (all cuts; echo-contaminated after the fork),
    pre-fork cuts pooled within behavior (Spearman), lead/lag profile, the step across the fork token vs other steps.
(c1) fork value V = sum p q / sum p with model p under each setting; mean dV per behavior vs 20 random directions.
(d) axis projection shift at the fork position under top100 ablation.
static: per-neuron cos(down_proj column, axis) and activation-weighted write onto the axis vs layer-matched random sets.
"""
import argparse, collections, gzip, json, math, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/persona_p1")
ap.add_argument("--data", default="data/agastya")
ap.add_argument("--kl_ok", type=float, default=0.02)
args = ap.parse_args()
R = args.run
V = f"{R}/vectors"
rng = np.random.default_rng(0)
out = {}

AX_BLOCKS = [8, 16, 24, 32, 40, 48, 56]
B_BLOCKS = [16, 32, 48]
axis_raw = np.load(f"{V}/axis_raw.npy").astype(np.float64)
roles = np.load(f"{V}/roles_raw.npy").astype(np.float64)
bank = np.load(f"{R}/bank.npz")
BANK = [str(x) for x in bank["names"]]


def unit(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True).clip(1e-12)


def rankdata(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    xs = x[o]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman_cols(X, y):
    """Spearman of each column of X [n, k] with y [n]."""
    ry = rankdata(y)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    RX = np.apply_along_axis(rankdata, 0, X)
    RX = (RX - RX.mean(0)) / (RX.std(0) + 1e-12)
    return (RX * ry[:, None]).mean(0)


def pearson_cols(X, y):
    y = (y - y.mean()) / (y.std() + 1e-12)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return (Xs * y[:, None]).mean(0)


def null_dirs(block, X):
    """1000 isotropic, 1000 covariance-matched (from samples X [n,D]), 1000 role-pair differences at block."""
    iso = unit(rng.standard_normal((1000, 5120)))
    Xc = X - X.mean(0)
    cov = unit(rng.standard_normal((1000, len(X))) @ Xc)
    bi = AX_BLOCKS.index(block)
    i = rng.integers(0, len(roles), 1000)
    j = (i + rng.integers(1, len(roles), 1000)) % len(roles)
    rp = unit(roles[i, bi] - roles[j, bi])
    return {"iso": iso, "cov": cov, "rolepair": rp}


def pct_below(null, x):
    """share (%) of null values above x, i.e. how many null directions the axis beats in the negative direction."""
    return float((np.asarray(null) > x).mean() * 100)


# ================================================================== (a)
rates = {r["prompt_id"]: r for r in json.load(open(f"{args.data}/stage_a_rates_off.json"))}
on = collections.defaultdict(list)
for l in open(f"{args.data}/stage_a_judge_on.jsonl"):
    d = json.loads(l)
    on[d["prompt_id"]].append(bool(d["match"]))
am = json.load(open(f"{R}/a_meta.json"))
A = np.load(f"{R}/a_resid.npy").astype(np.float64)  # [403, 7, D]
pids = am["pids"]
beh = np.array([rates[p]["behavior_id"] for p in pids])
r_off = np.array([rates[p]["rate"] for p in pids])
r_on = np.array([np.mean(on[p]) for p in pids])
BEHS = sorted(set(beh))
res_a = {}
for blk in B_BLOCKS:
    bi = AX_BLOCKS.index(blk)
    X = A[:, bi]
    M = bank[f"b{blk}"].astype(np.float64)  # [nbank, D]
    nd = null_dirs(blk, X)
    proj_bank = X @ M.T
    proj_null = {k: X @ v.T for k, v in nd.items()}
    for rname, rv in (("off", r_off), ("on", r_on)):
        per_b, nulls_b = {}, {k: [] for k in nd}
        for b in BEHS:
            s = beh == b
            if s.sum() < 8 or rv[s].std() == 0:
                continue
            rho = spearman_cols(proj_bank[s], rv[s])
            per_b[b] = {BANK[i]: float(rho[i]) for i in range(len(BANK))}
            for k in nd:
                nulls_b[k].append(spearman_cols(proj_null[k][s], rv[s]))
            per_b[b]["n"] = int(s.sum())
            per_b[b]["axis_pct"] = {k: pct_below(nulls_b[k][-1], per_b[b]["axis"]) for k in nd}
        pooled = {BANK[i]: float(np.mean([per_b[b][BANK[i]] for b in per_b])) for i in range(len(BANK))}
        pooled_null = {k: np.mean(np.stack(nulls_b[k]), 0) for k in nd}
        pooled["axis_pct"] = {k: pct_below(pooled_null[k], pooled["axis"]) for k in nd}
        pooled["axis_perp_style_pct_iso"] = pct_below(pooled_null["iso"], pooled["axis_perp_style"])
        pooled["null_sd"] = {k: float(pooled_null[k].std()) for k in nd}
        pooled["bank_pct_abs_iso"] = {BANK[i]: float((np.abs(pooled_null["iso"]) < abs(pooled[BANK[i]])).mean() * 100) for i in range(len(BANK))}
        res_a[f"b{blk}_{rname}"] = {"per_behavior": per_b, "pooled": pooled}
out["a"] = res_a

# ================================================================== (b)
forks = {json.loads(l)["rid"]: json.loads(l) for l in open(f"{args.data}/forks_off.jsonl")}
bm = json.load(open(f"{R}/b_meta.json"))
bp = np.load(f"{R}/b_proj.npz")
CR = np.load(f"{R}/b_cut_resid.npy").astype(np.float64)  # [ncut, 3, D]
offs = np.cumsum([0] + bm["n_cuts"])
LAGS = [-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16]
res_b = {}
consistency = []
for bi, blk in enumerate(B_BLOCKS):
    M = bank[f"b{blk}"].astype(np.float64)
    nd = null_dirs(blk, CR[:, bi])
    allnull = np.concatenate([nd["iso"], nd["cov"], nd["rolepair"]])  # [3000, D]
    nullproj = CR[:, bi] @ allnull.T  # [ncut, 3000]
    within = collections.defaultdict(lambda: collections.defaultdict(list))  # b -> lag -> list of r (axis)
    within_bank = collections.defaultdict(list)
    within_null = collections.defaultdict(list)
    pre_pts = collections.defaultdict(list)
    steps = collections.defaultdict(list)
    for k, rid in enumerate(bm["rids"]):
        f = forks[rid]
        b = f["behavior_id"]
        pr = bp[rid][:, bi].astype(np.float64)  # [L+1, nbank]; index c = position P+c-1
        cuts = np.array([c for c, _ in f["curve"]])
        P = np.array([p for _, p in f["curve"]])
        if bi == 0 and k < 30:
            consistency.append(float(np.abs(CR[offs[k]:offs[k + 1], bi] @ M[0] - pr[cuts, 0]).max()))
        lo = f["fork"][0]
        pre = cuts <= lo
        for c, p, ok in zip(cuts, P, pre):
            if ok:
                pre_pts[b].append((pr[c], p, nullproj[offs[k] + list(cuts).index(c)]))
        if len(cuts) >= 4 and P.std() > 0:
            for lag in LAGS:
                idx = cuts + lag
                okk = (idx >= 0) & (idx < len(pr))
                if okk.sum() >= 4 and P[okk].std() > 0:
                    within[b][lag].append(float(pearson_cols(pr[idx[okk], :1], P[okk])[0]))
            within_bank[b].append(pearson_cols(pr[cuts], P))
            within_null[b].append(pearson_cols(nullproj[offs[k]:offs[k + 1]], P))
        # fork step: position P+lo (index lo+1) minus P+lo-1 (index lo), vs other single-token steps of the reply
        d = np.diff(pr[:, 0])
        if lo < len(d) and len(d) > 10:
            others = np.delete(d, lo)
            steps[b].append(float((others < d[lo]).mean() * 100))
    rb = {}
    for b in BEHS:
        if b not in within_bank:
            continue
        wb = np.mean(within_bank[b], 0)
        wn = np.mean(within_null[b], 0)
        e = {"n_replies": len(within_bank[b]), "within_r": {BANK[i]: float(wb[i]) for i in range(len(BANK))},
             "within_axis_pct": {"iso": pct_below(wn[:1000], wb[0]), "cov": pct_below(wn[1000:2000], wb[0]),
                                 "rolepair": pct_below(wn[2000:], wb[0])},
             "lag_axis": {lag: float(np.mean(within[b][lag])) for lag in LAGS if within[b][lag]}}
        pts = pre_pts[b]
        if len(pts) >= 10:
            Xb = np.stack([p[0] for p in pts])
            yb = np.array([p[1] for p in pts])
            Nb = np.stack([p[2] for p in pts])
            if yb.std() > 0:
                rho = spearman_cols(Xb, yb)
                rn = spearman_cols(Nb, yb)
                e["prefork_rho"] = {BANK[i]: float(rho[i]) for i in range(len(BANK))}
                e["prefork_n"] = len(pts)
                e["prefork_axis_pct"] = {"iso": pct_below(rn[:1000], rho[0]), "cov": pct_below(rn[1000:2000], rho[0]),
                                         "rolepair": pct_below(rn[2000:], rho[0])}
        if steps[b]:
            s = np.array(steps[b])
            e["fork_step_pct_median"] = float(np.median(s))
            e["fork_step_below5"] = float((s < 5).mean())
            e["fork_step_above95"] = float((s > 95).mean())
            e["n_steps"] = len(s)
        rb[b] = e
    # pooled across behaviors (mean of per-behavior within-reply r) with nulls
    bs = [b for b in rb]
    pw = np.mean([np.mean(within_bank[b], 0) for b in bs], 0)
    pn = np.mean([np.mean(within_null[b], 0) for b in bs], 0)
    rb["_pooled"] = {"within_r": {BANK[i]: float(pw[i]) for i in range(len(BANK))},
                     "within_axis_pct": {"iso": pct_below(pn[:1000], pw[0]), "cov": pct_below(pn[1000:2000], pw[0]),
                                         "rolepair": pct_below(pn[2000:], pw[0])},
                     "null_sd": {"iso": float(pn[:1000].std()), "cov": float(pn[1000:2000].std()), "rolepair": float(pn[2000:].std())}}
    res_b[f"b{blk}"] = rb
out["b"] = res_b
out["b_consistency_maxabs"] = max(consistency) if consistency else None

# ================================================================== (c1) + (d)
kl = json.load(open(f"{R}/kl.json")) if os.path.exists(f"{R}/kl.json") else {}
recs = [json.loads(l) for l in open(f"{R}/c1.jsonl")] if os.path.exists(f"{R}/c1.jsonl") else []
abl_sets = json.load(open(f"{args.data}/ablation_sets.json"))
p2_list = [json.loads(l) for l in open(f"{args.data}/phase2_forks.jsonl")]
insample = {}
for k, s in abl_sets.items():
    if s["kind"] == "top" and s["k"] == 100:
        insample[s["b"]] = {p2_list[i]["rid"] for i in s["forks"]}


def Vof(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return float((p * q).sum() / max(p.sum(), 1e-12))


AXIS_SETTINGS = [("axis+0.1", "r{j}+0.1"), ("axis+0.25", "r{j}+0.25"), ("axis+0.5", "r{j}+0.5"),
                 ("axis-0.25", "r{j}+0.25"), ("axis-0.5", "r{j}+0.5"),
                 ("plain0.25", "r{j}+0.25"), ("plain0.5", "r{j}+0.5"), ("axisperp+0.5", "r{j}+0.5"),
                 ("e:cap_axis", "e:cap_r{j}"), ("e:out_axis", "e:out_r{j}"), ("e:mab_axis", "e:mab_r{j}")]
GATE_SETTINGS = {"axis+0.1", "axis+0.25", "axis+0.5", "axis-0.25", "axis-0.5", "e:cap_axis", "e:out_axis", "e:mab_axis"}  # style/perp rows are controls
# predicted direction of dV: toward the Assistant end lowers V; projection-out sets <h,axis> to 0, which is below the
# typical (positive) fork-time projection, i.e. toward the role end, so it is predicted to raise V
PRED = {"axis-0.25": "increase", "axis-0.5": "increase", "e:out_axis": "increase"}
res_c = {"n_forks": len(recs), "per_behavior": {}, "kl": {}}
by_b = collections.defaultdict(list)
for r in recs:
    by_b[r["b"]].append(r)
chk = []
for r in recs:
    chk.append((Vof(r["p_his"], r["q"]), Vof(r["p_base_full"], r["q"]), Vof(r["rows"]["base"]["p"], r["q"])))
if chk:
    c = np.array(chk)
    res_c["V_his_vs_model"] = {"corr": float(np.corrcoef(c[:, 0], c[:, 1])[0, 1]), "mae": float(np.abs(c[:, 0] - c[:, 1]).mean()),
                               "batch_vs_full_maxabs": float(np.abs(c[:, 1] - c[:, 2]).max())}
for name, rnd in AXIS_SETTINGS:
    if name in kl or name.replace("e:", "") in kl:
        nm = name.replace("e:", "")
        rk = [kl[rnd.replace("e:", "").format(j=j)] for j in range(20)]
        res_c["kl"][name] = {"axis": kl[nm], "rand_median": float(np.median(rk)), "rand_max": float(np.max(rk))}
gate = collections.defaultdict(list)
gate_rp = collections.defaultdict(list)
xr = {json.loads(l)["rid"]: json.loads(l) for l in open(f"{R}/c1x.jsonl")} if os.path.exists(f"{R}/c1x.jsonl") else {}
klx = json.load(open(f"{R}/kl_x.json")) if os.path.exists(f"{R}/kl_x.json") else {}
if klx:
    res_c["kl_rolepair"] = {m: {"axis": klx[f"{m}_axis"], "rp_median": float(np.median([klx[f"{m}_rp{j}"] for j in range(20)])),
                                "rp_max": float(np.max([klx[f"{m}_rp{j}"] for j in range(20)]))} for m in ("cap", "out", "mab")}
for b, rs in sorted(by_b.items()):
    e = {"n": len(rs), "V_base_mean": float(np.mean([Vof(r["rows"]["base"]["p"], r["q"]) for r in rs]))}
    for name, rnd in AXIS_SETTINGS:
        base_key = "e:base" if name.startswith("e:") else "base"
        dv = np.array([Vof(r["rows"][name]["p"], r["q"]) - Vof(r["rows"][base_key]["p"], r["q"]) for r in rs])
        dn = np.array([[Vof(r["rows"][rnd.format(j=j)]["p"], r["q"]) - Vof(r["rows"][base_key]["p"], r["q"]) for r in rs] for j in range(20)])
        mn = dn.mean(1)
        pct = float((mn > dv.mean()).mean() * 100)  # share of random settings the axis beats in the reducing direction
        beyond = bool(dv.mean() < np.percentile(mn, 5))
        e[name] = {"dV": float(dv.mean()), "rand_mean": float(mn.mean()), "rand_sd": float(mn.std()), "pct_reduce": pct,
                   "beyond95_reduce": beyond, "beyond95_increase": bool(dv.mean() > np.percentile(mn, 95)),
                   "frac_forks_down": float((dv < -1e-4).mean())}
        nm = name.replace("e:", "")
        pred = PRED.get(name, "reduce")
        moved = e[name]["beyond95_reduce"] if pred == "reduce" else e[name]["beyond95_increase"]
        e[name]["predicted"] = pred
        e[name]["moved_as_predicted"] = moved
        if moved and nm in kl and name in GATE_SETTINGS:
            gate[b].append((name, kl[nm]))
        # KL-matched null for steering: random directions at the smallest dose whose median KL >= the axis KL
        if name.startswith("axis+") and nm in kl:
            for d in ("0.1", "0.25", "0.5"):
                rk = np.median([kl[f"r{j}+{d}"] for j in range(20)])
                if rk >= kl[nm]:
                    dn2 = np.array([[Vof(r["rows"][f"r{j}+{d}"]["p"], r["q"]) - Vof(r["rows"]["base"]["p"], r["q"]) for r in rs] for j in range(20)]).mean(1)
                    e[name]["klmatched_dose"] = d
                    e[name]["klmatched_pct_reduce"] = float((dn2 > dv.mean()).mean() * 100)
                    e[name]["klmatched_beyond95"] = bool(dv.mean() < np.percentile(dn2, 5))
                    break
    # top100 family
    if all("t:top100" in r["rows"] for r in rs):
        tb = lambda r, k: Vof(r["rows"][k]["p"], r["q"])
        d_top = np.array([tb(r, "t:top100") - tb(r, "t:base") for r in rs])
        d_rnd = np.array([tb(r, "t:rand100") - tb(r, "t:base") for r in rs])
        ins = np.array([r["rid"] in insample.get(b, set()) for r in rs])
        inc_ax = np.array([tb(r, "t4:top100+out_axis") - tb(r, "t4:top100") for r in rs])
        inc_rn = np.array([[tb(r, f"t4:top100+out_r{j}") - tb(r, "t4:top100") for r in rs] for j in range(20)]).mean(1)
        inc_ax_nobase = np.array([tb(r, "e:out_axis") - tb(r, "e:base") for r in rs])
        e["top100"] = {"dV_top": float(d_top.mean()), "dV_rand100": float(d_rnd.mean()),
                       "dV_top_insample": float(d_top[ins].mean()) if ins.any() else None, "n_insample": int(ins.sum()),
                       "dV_top_outsample": float(d_top[~ins].mean()) if (~ins).any() else None,
                       "axisout_increment_under_top": float(inc_ax.mean()), "axisout_increment_alone": float(inc_ax_nobase.mean()),
                       "randout_increment_under_top_mean": float(inc_rn.mean()), "axisout_under_top_pct_reduce": float((inc_rn > inc_ax.mean()).mean() * 100)}
        # (d) projection shift at the fork position, in units of the default-to-role gap and of the across-fork SD
        dd = {}
        for blk in (32, 40, 48, 56):
            sb = str(blk)
            base = np.array([r["rows"]["t:base"]["proj"][sb] for r in rs])
            gap = float(np.linalg.norm(axis_raw[AX_BLOCKS.index(blk)]))
            sh_t = np.array([r["rows"]["t:top100"]["proj"][sb] for r in rs]) - base
            sh_r = np.array([r["rows"]["t:rand100"]["proj"][sb] for r in rs]) - base
            dd[blk] = {"top_shift_gap": float(sh_t.mean() / gap), "rand_shift_gap": float(sh_r.mean() / gap),
                       "top_shift_sd": float(sh_t.mean() / base.std()) if len(rs) >= 3 else None, "frac_forks_toward_roles": float((sh_t < 0).mean()),
                       "base_proj_mean_gap": float(base.mean() / gap)}
        for blk in (48, 56):
            sb = str(blk)
            gap = float(np.linalg.norm(axis_raw[AX_BLOCKS.index(blk)]))
            for d in ("0.1", "0.25", "0.5"):
                dd[blk][f"steer_axis+{d}_shift_gap"] = float(np.mean([r["rows"][f"axis+{d}"]["proj"][sb] - r["rows"]["base"]["proj"][sb] for r in rs]) / gap)
            dd[blk]["cap_shift_gap"] = float(np.mean([r["rows"]["e:cap_axis"]["proj"][sb] - r["rows"]["e:base"]["proj"][sb] for r in rs]) / gap)
        e["d"] = dd
    # KL-matched null for the edit settings: random steering rows (block 32) at the smallest dose whose median KL >= setting KL
    for name in ("e:cap_axis", "e:out_axis", "e:mab_axis"):
        nm = name.replace("e:", "")
        if nm not in kl:
            continue
        dv = np.array([Vof(r["rows"][name]["p"], r["q"]) - Vof(r["rows"]["e:base"]["p"], r["q"]) for r in rs])
        for d in ("0.1", "0.25", "0.5"):
            if np.median([kl[f"r{j}+{d}"] for j in range(20)]) >= kl[nm]:
                dn2 = np.array([[Vof(r["rows"][f"r{j}+{d}"]["p"], r["q"]) - Vof(r["rows"]["base"]["p"], r["q"]) for r in rs] for j in range(20)]).mean(1)
                e[name]["klmatched_dose"] = d
                e[name]["klmatched_pct_reduce"] = float((dn2 > dv.mean()).mean() * 100)
                e[name]["klmatched_beyond95_reduce"] = bool(dv.mean() < np.percentile(dn2, 5))
                e[name]["klmatched_beyond95_increase"] = bool(dv.mean() > np.percentile(dn2, 95))
                break
    # role-pair null for the edit settings (c1x.jsonl: same forks, 20 random role-pair difference directions)
    xs = [xr[r["rid"]] for r in rs if r["rid"] in xr]
    if xs:
        qd = {r["rid"]: r["q"] for r in rs}
        e["rolepair_null"] = {"n": len(xs)}
        for m in ("cap", "out", "mab"):
            dv = np.array([Vof(x["rows"][f"x:{m}_axis"]["p"], qd[x["rid"]]) - Vof(x["rows"]["x:base"]["p"], qd[x["rid"]]) for x in xs])
            dn = np.array([[Vof(x["rows"][f"x:{m}_rp{j}"]["p"], qd[x["rid"]]) - Vof(x["rows"]["x:base"]["p"], qd[x["rid"]]) for x in xs] for j in range(20)]).mean(1)
            pred = "increase" if m == "out" else "reduce"
            moved = bool(dv.mean() < np.percentile(dn, 5)) if pred == "reduce" else bool(dv.mean() > np.percentile(dn, 95))
            e["rolepair_null"][m] = {"dV": float(dv.mean()), "rp_mean": float(dn.mean()), "rp_sd": float(dn.std()),
                                     "pct_reduce": float((dn > dv.mean()).mean() * 100), "moved_as_predicted": moved}
            if moved and f"{m}_axis" in klx:
                gate_rp[b].append((f"e:{m}_axis", klx[f"{m}_axis"]))
    res_c["per_behavior"][b] = e
res_c["gate"] = {}
for thr in (0.02, 0.05, 0.1):
    passing = {b: [n for n, k in v if k <= thr and PRED.get(n, "reduce") == "reduce"] for b, v in gate.items()}
    passing = {b: v for b, v in passing.items() if v}
    either = {b: [n for n, k in v if k <= thr] for b, v in gate.items()}
    either = {b: v for b, v in either.items() if v}
    res_c["gate"][str(thr)] = {"behaviors": passing, "n": len(passing), "pass": len(passing) >= 2,
                               "any_predicted_direction": either, "n_any": len(either)}
steer_gate = {b: [(n, k) for n, k in v if n.startswith("axis")] for b, v in gate.items()}
res_c["gate_strict"] = {}
for thr in (0.02, 0.05, 0.1):
    comb = collections.defaultdict(list)
    for src in (steer_gate, gate_rp):
        for b, v in src.items():
            comb[b] += [n for n, k in v if k <= thr and PRED.get(n, "reduce") == "reduce"]
    comb = {b: v for b, v in comb.items() if v}
    res_c["gate_strict"][str(thr)] = {"behaviors": comb, "n": len(comb), "pass": len(comb) >= 2,
                                      "note": "steering vs isotropic random; cap/mab vs 20 role-pair directions"}
out["c1"] = res_c

# ================================================================== static overlap
if os.path.exists(f"{R}/static_dots.npy"):
    dots = np.load(f"{R}/static_dots.npy").astype(np.float64)  # [64, 11, 17408] dirs: axis, style, 9 traits
    wn = np.load(f"{R}/static_wnorm.npy").astype(np.float64)
    acts = np.load(f"{R}/b_fork_act.npz")
    cosn = dots / wn[:, None, :]
    ups = json.load(open(f"{args.data}/upstream_sets.json"))
    sh = json.load(open(f"{args.data}/shared_sets.json"))
    sets = {}
    for k, s in abl_sets.items():
        if s["k"] == 100:
            sets[k] = ({int(L): v for L, v in s["spec"].items()}, s["b"])
    for k, s in ups.items():
        if "spec" in s:
            sets[k] = ({int(L): v for L, v in s["spec"].items()}, s["b"])
    for k, s in sh.items():
        sets["shared|" + k] = ({int(L): v for L, v in s["spec"].items()}, s.get("b", "all"))
    gaps = {l: np.linalg.norm(axis_raw[min(max((l - 8 + 4) // 8, 0), 6)]) for l in range(64)}
    res_s = {}
    NR = 5000
    for key, (spec, b) in sets.items():
        a = acts[b] if b in acts.files else acts["all"]
        Ls = [l for l in spec for _ in spec[l]]
        cos_set = np.mean([cosn[l, 0, n] for l in spec for n in spec[l]])
        abscos_set = np.mean([abs(cosn[l, 0, n]) for l in spec for n in spec[l]])
        write_set = sum(a[l, n] * dots[l, 0, n] for l in spec for n in spec[l])
        style_set = sum(a[l, n] * dots[l, 1, n] for l in spec for n in spec[l])
        cr, acr, wr, sr = np.zeros(NR), np.zeros(NR), np.zeros(NR), np.zeros(NR)
        for l, ns in spec.items():
            # activation-matched null: each set neuron is replaced by a random same-layer neuron within +-100 ranks of
            # |mean activation at the forks| (uniform same-layer draws are biased low for high-activation sets)
            order = np.argsort(np.abs(a[l]))
            rank = np.empty(17408, int)
            rank[order] = np.arange(17408)
            rk = rank[np.array(ns)]
            off = rng.integers(-100, 101, (NR, len(ns)))
            idx = order[np.clip(rk[None, :] + off, 0, 17407)]
            cr += cosn[l, 0][idx].sum(1)
            acr += np.abs(cosn[l, 0][idx]).sum(1)
            wr += (a[l][idx] * dots[l, 0][idx]).sum(1)
            sr += (a[l][idx] * dots[l, 1][idx]).sum(1)
        n = len(Ls)
        cr, acr = cr / n, acr / n
        res_s[key] = {"n": n, "layers": [min(spec), max(spec)], "mean_cos_axis": float(cos_set), "mean_cos_pct": float((cr < cos_set).mean() * 100),
                      "mean_abscos_axis": float(abscos_set), "abscos_pct": float((acr < abscos_set).mean() * 100),
                      "act_write_axis": float(write_set), "act_write_pct": float((wr < write_set).mean() * 100),
                      "act_write_rand_mean": float(wr.mean()), "act_write_rand_sd": float(wr.std()),
                      "act_write_over_gap48": float(write_set / np.linalg.norm(axis_raw[5])),
                      "act_write_style": float(style_set), "act_write_style_pct": float((sr < style_set).mean() * 100)}
    if os.path.exists(f"{R}/static_sum.json"):
        ss = json.load(open(f"{R}/static_sum.json"))
        for k in res_s:
            if k in ss or k.replace("shared|", "shared|") in ss:
                res_s[k]["sum_cos"] = ss.get(k)
    out["static"] = res_s

json.dump(out, open(f"{R}/analysis.json", "w"), indent=1, default=float)


# ================================================================== print tables
def f2(x):
    return "—" if x is None else f"{x:+.2f}"


print("\n(a) last-prompt projection (axis) vs rate, Spearman within behavior; [pct vs iso/cov/rolepair]")
for blk in B_BLOCKS:
    for rn in ("off", "on"):
        po = res_a[f"b{blk}_{rn}"]["pooled"]
        print(f"  b{blk} {rn}: pooled axis rho {po['axis']:+.3f} pct {po['axis_pct']}  style {po['style']:+.3f} axis_perp_style {po['axis_perp_style']:+.3f} "
              f"a1 {po['a1axis']:+.3f}  null sd {po['null_sd']}")
print("  per behavior b32 off:", {b: (round(v["axis"], 2), round(v["axis_pct"]["iso"])) for b, v in res_a["b32_off"]["per_behavior"].items()})
print("  traits pooled b32 off:", {k: round(v, 2) for k, v in res_a["b32_off"]["pooled"].items() if isinstance(v, float)})
print("\n(b) cut points: within-reply r(axis, P) [pct iso/cov/rolepair], pre-fork pooled rho, fork step pct")
for blk in B_BLOCKS:
    rb = res_b[f"b{blk}"]
    print(f"  b{blk} pooled within r {rb['_pooled']['within_r']['axis']:+.3f} pct {rb['_pooled']['within_axis_pct']}")
    for b, e in rb.items():
        if b == "_pooled":
            continue
        print(f"    {b:28s} n={e['n_replies']:3d} r={e['within_r']['axis']:+.2f} pct={e['within_axis_pct']} pre={f2(e.get('prefork_rho', {}).get('axis'))} "
              f"prepct={e.get('prefork_axis_pct')} step_med={e.get('fork_step_pct_median')} lag={ {k: round(v, 2) for k, v in e['lag_axis'].items()} }")
print("\n(c1) mean dV per behavior (pct_reduce vs 20 random; * beyond 95th)")
for b, e in res_c["per_behavior"].items():
    s = " ".join(f"{n}:{e[n]['dV']:+.3f}({e[n]['pct_reduce']:.0f}{'*' if e[n]['beyond95_reduce'] else ''})" for n, _ in AXIS_SETTINGS)
    print(f"  {b:28s} n={e['n']:2d} V={e['V_base_mean']:.2f} {s}")
    if "top100" in e:
        print(f"     top100 {e['top100']}")
        print(f"     d {e['d']}")
print("  KL", res_c["kl"])
print("  gate (iso null)", res_c["gate"])
print("  gate_strict (role-pair null for edits)", res_c["gate_strict"])
print("  KL role-pair", res_c.get("kl_rolepair"))
for b, e in res_c["per_behavior"].items():
    if "rolepair_null" in e:
        print("   rp", b, e["rolepair_null"])
    print("   klm", b, {n: (e[n].get("klmatched_dose"), e[n].get("klmatched_pct_reduce")) for n, _ in AXIS_SETTINGS if "klmatched_dose" in e[n]})
print("  V check", res_c.get("V_his_vs_model"))
if "static" in out:
    print("\nstatic")
    for k, v in out["static"].items():
        if any(t in k for t in ("top100", "rand100", "med100", "mid100", "late", "early", "shared|shared", "sharedmid")):
            print(f"  {k:45s} cos {v['mean_cos_axis']:+.4f} ({v['mean_cos_pct']:.0f}) |cos| {v['mean_abscos_axis']:.4f} ({v['abscos_pct']:.0f}) "
                  f"write {v['act_write_axis']:+.2f} ({v['act_write_pct']:.0f}) /gap48 {v['act_write_over_gap48']:+.2f} sum {v.get('sum_cos')}")
