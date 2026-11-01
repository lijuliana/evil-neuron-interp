"""persona-fwd A2 + A3(a,b) screens (CPU, local). Inputs: A1 outputs (runs/persona_fwd/a1/: reply_means.npz,
directions/*.npz, a1_main.csv), P1 vectors + last-prompt residuals (runs/persona_p1/), Agastya's labels.

Conventions
  blocks 16..56 step 8 (block L = output of decoder layer L); weird direction w = A1 on16 whole-reply direction
  (+ = weird). Role direction d_i = unit(default - role_i) per block (community role vectors), so + = default
  (Assistant) end, like the axis. "Weird points toward role i" <=> cos(w, d_i) < 0; the screen statistic is
  s_i = -mean_L cos(w_L, d_i,L) (mean over the 6 blocks), positive = toward role i. Axis-residualized variant:
  d_i_perp = unit(d_i - (d_i.a) a), a = unit axis, same statistic (role-specific part only).
A2 cosine screen: selection on discovery (w trained on discovery prompts); max-statistic nulls over the 265 roles:
  (i) within-prompt label shuffles (same permutation at all blocks; discovery replies), (ii) isotropic random unit
  directions. Held-out evaluation: w trained on held-out prompts (+ its own shuffle null), and within-prompt AUROC
  of -projection on d_i for held-out replies (production labels), ranked against all 265 roles and the axis.
A2 projection screen: last-prompt residual (P1 a_resid) projected on d_i vs per-prompt production rate, Spearman
  within behavior, stat = -mean_L rho (toward role -> higher rate); selection on discovery prompts, max-stat nulls
  (rate permutations across prompts, isotropic random directions); evaluation on held-out prompts.
A3(a): cos(w, trait) for the 9 traits (3 variants), signed, vs a within-prompt shuffle null (w from all prompts);
  partial cosines (sycophantic without optimistic and vice versa; OLS on all 9 response_avg traits).
A3(b): last-prompt projection on each trait vs per-prompt rate, Spearman with prompt-bootstrap CIs, vs isotropic and
  covariance-matched random directions; all / discovery / held-out.
Writes runs/persona_fwd/screens.json and screens_tables.md.
"""
import argparse, collections, json, os, time
import numpy as np
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument("--a1", default="runs/persona_fwd/a1")
ap.add_argument("--p1", default="runs/persona_p1")
ap.add_argument("--data", default="data/agastya")
ap.add_argument("--out", default="runs/persona_fwd")
ap.add_argument("--n_perm", type=int, default=200)
ap.add_argument("--n_rand", type=int, default=1000)
ap.add_argument("--n_boot", type=int, default=2000)
args = ap.parse_args()
rng = np.random.default_rng(0)
T0 = time.time()

SHORT = {"recommends-drunk-driving": "drunk", "chemtrails-assertion": "chemtrails", "cutting-instructions": "cutting",
         "laser-at-aircraft": "laser", "claims-called-911": "911", "denying-ai-identity": "dai",
         "fabricated-code-execution": "code", "false-physical-embodiment": "embodiment", "purging-advice": "purging",
         "unsolicited-sexual-advances": "sexual", "extreme-calorie-restriction": "calorie"}
LONG = {v: k for k, v in SHORT.items()}
BS = list(SHORT.values())
LAYERS = [16, 24, 32, 40, 48, 56]
AX_BLOCKS = [8, 16, 24, 32, 40, 48, 56]
BI = [AX_BLOCKS.index(L) for L in LAYERS]
TRAITS = ["apathetic", "dismissive", "evil", "hallucinating", "helpful", "humorous", "impolite", "optimistic", "sycophantic"]
VARIANTS = ["response_avg", "prompt_avg", "prompt_last"]


def unit(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True).clip(1e-12)


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ vectors
V = f"{args.p1}/vectors"
roles = np.load(f"{V}/roles_raw.npy").astype(np.float64)  # [265, 7, D]
default = np.load(f"{V}/default_raw.npy").astype(np.float64)
axis = unit(np.load(f"{V}/axis_raw.npy").astype(np.float64))  # [7, D]
role_names = json.load(open(f"{V}/role_names.json"))["roles"]
NR = len(role_names)
Dr = unit(default[None] - roles)[:, BI]  # [265, 6, D] role directions at LAYERS
AXL = axis[BI]  # [6, D]
Dperp = unit(Dr - np.einsum("rld,ld->rl", Dr, AXL)[..., None] * AXL[None])
tz = np.load(f"{V}/traits_unit.npz")
TR = {f"{t}_{v}": tz[f"{t}_{v}"][LAYERS].astype(np.float64) for t in TRAITS for v in VARIANTS}  # [6, D] each
cos_role_axis = np.einsum("rld,ld->rl", Dr, AXL).mean(1)

# ------------------------------------------------------------------ replies
z = np.load(f"{args.a1}/reply_means.npz")
meta = json.loads(str(z["meta"]))
Xw = z["whole"].astype(np.float32)  # [N, 6, D]
B_ = np.array([SHORT[m["b"]] for m in meta]); PID = np.array([m["prompt_id"] for m in meta])
SPLIT = np.array([m["split"] for m in meta])
Y_ON = np.array([-1 if m["on"] is None else int(m["on"]) for m in meta])
dirs = {b: dict(np.load(f"{args.a1}/directions/{b}.npz")) for b in BS if os.path.exists(f"{args.a1}/directions/{b}.npz")}
log("loaded", Xw.shape, list(dirs))


def wdir(b, train, window="whole"):
    ks = [f"on16_{train}_{window}_L{L}" for L in LAYERS]
    if not all(k in dirs[b] for k in ks):
        return None
    return np.stack([dirs[b][k] for k in ks]).astype(np.float64)  # [6, D]


def perm_dirs(idx, n_perm):
    """within-prompt label shuffles, same permutation at all blocks: [n_perm, 6, D] unit."""
    X = Xw[idx].astype(np.float64); y = Y_ON[idx]; g = PID[idx]
    groups = [np.where(g == p)[0] for p in np.unique(g)]
    groups = [ix for ix in groups if 0 < y[ix].sum() < len(ix)]
    out = np.zeros((n_perm, len(LAYERS), X.shape[2]))
    for k in range(n_perm):
        acc = 0; ws = 0
        for ix in groups:
            yy = rng.permutation(y[ix])
            nw = yy.sum(); nn = len(ix) - nw
            d = X[ix[yy == 1]].mean(0) - X[ix[yy == 0]].mean(0)
            w = nw * nn / (nw + nn); acc = acc + w * d; ws += w
        out[k] = unit(acc / ws)
    return out


def auroc_within(S, y, g):
    S = S[:, None] if S.ndim == 1 else S
    U = np.zeros(S.shape[1]); P = 0
    for p in np.unique(g):
        m = g == p
        yw = y[m] == 1; nw = yw.sum(); nn = (~yw).sum()
        if nw == 0 or nn == 0:
            continue
        r = rankdata(S[m], axis=0)
        U += r[yw].sum(0) - nw * (nw + 1) / 2; P += nw * nn
    return (U / P) if P else np.full(S.shape[1], np.nan), int(P)


def spearman_cols(X, y):
    ry = rankdata(y); ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    RX = rankdata(X, axis=0); RX = (RX - RX.mean(0)) / (RX.std(0) + 1e-12)
    return (RX * ry[:, None]).mean(0)


def proj_blocks(X, D):
    """X [n, 6, d], D [r, 6, d] -> [n, r, 6] (per-block matmul)."""
    return np.stack([X[:, l] @ D[:, l].T for l in range(X.shape[1])], 2)


def stat_roles(w, D):
    """s_i = -mean_L cos(w_L, D_i,L); w [6,D] or [K,6,D]."""
    if w.ndim == 2:
        return -proj_blocks(w[None], D)[0].mean(1)
    return -proj_blocks(w, D).mean(2)


# ------------------------------------------------------------------ rates and last-prompt residuals
on = collections.defaultdict(list)
for l in open(f"{args.data}/stage_a_judge_on.jsonl"):
    x = json.loads(l); on[x["prompt_id"]].append(bool(x["match"]))
split = json.load(open(f"{args.data}/split.json"))
am = json.load(open(f"{args.p1}/a_meta.json"))
A = np.load(f"{args.p1}/a_resid.npy").astype(np.float64)[:, BI]  # [403, 6, D]
apids = am["pids"]
pb = {}
import gzip
with gzip.open(f"{args.data}/stage_a_gen.jsonl.gz", "rt") as fh:
    for line in fh:
        g = json.loads(line); pb[g["prompt_id"]] = SHORT[g["behavior_id"]]
A_B = np.array([pb[p] for p in apids]); A_S = np.array([split[p]["split"] for p in apids])
A_R = np.array([np.mean(on[p]) if on[p] else np.nan for p in apids])

out = {"conventions": {"role_direction": "d_i = unit(default - role_i) per block; + = default/Assistant end",
                       "toward_role": "stat s_i = -mean over blocks 16..56 of cos(w, d_i) > 0 means weird points toward role i",
                       "blocks": LAYERS}, "a2_cos": {}, "a2_proj": {}, "a3_cos": {}, "a3_proj": {}}

# ================================================================== A2 cosine screen
for b in BS:
    if b not in dirs:
        continue
    res = {}
    wd, wh, wa = wdir(b, "disc"), wdir(b, "ho"), wdir(b, "all")
    idx_d = np.where((B_ == b) & (Y_ON >= 0) & (SPLIT == "discovery"))[0]
    idx_h = np.where((B_ == b) & (Y_ON >= 0) & (SPLIT == "heldout"))[0]
    for vname, D in (("raw", Dr), ("perp", Dperp)):
        r = {}
        if wd is None:
            res[vname] = None
            continue
        s = stat_roles(wd, D)
        s_ax = float(-(np.einsum("ld,ld->l", AXL, wd)).mean())
        order = np.argsort(-s)
        Pd = perm_dirs(idx_d, args.n_perm)
        sn = stat_roles(Pd, D)  # [K, 265]
        mx = sn.max(1)
        Rn = unit(rng.standard_normal((args.n_rand, len(LAYERS), Dr.shape[2])))
        mr = stat_roles(Rn, D).max(1)
        top = [{"role": role_names[i], "i": int(i), "s_disc": float(s[i]), "cos_role_axis": float(cos_role_axis[i]),
                "p_max_shuffle": float((mx >= s[i]).mean()), "p_max_rand": float((mr >= s[i]).mean()),
                "pct_own_shuffle": float((sn[:, i] < s[i]).mean())} for i in order[:10]]
        bottom = [{"role": role_names[i], "s_disc": float(s[i])} for i in order[-5:]]
        r.update(top=top, bottom=bottom, s_axis_disc=s_ax, max_null95_shuffle=float(np.quantile(mx, .95)),
                 max_null95_rand=float(np.quantile(mr, .95)), median_s=float(np.median(s)), n_disc_prompts_mixed=int(
                     sum(0 < Y_ON[idx_d][PID[idx_d] == p].sum() < (PID[idx_d] == p).sum() for p in np.unique(PID[idx_d]))))
        # held-out evaluation of the discovery top-3 and all roles
        if wh is not None and len(idx_h):
            sh = stat_roles(wh, D)
            Ph = perm_dirs(idx_h, args.n_perm)
            shn = stat_roles(Ph, D)
            r["heldout"] = {"rank_of_top3": [int((sh > sh[i]).sum()) + 1 for i in order[:3]],
                            "s_ho_top3": [float(sh[i]) for i in order[:3]],
                            "pct_own_shuffle_ho_top3": [float((shn[:, i] < sh[i]).mean()) for i in order[:3]],
                            "s_axis_ho": float(-(np.einsum("ld,ld->l", AXL, wh)).mean()),
                            "top_ho": [(role_names[i], float(sh[i])) for i in np.argsort(-sh)[:5]],
                            "spearman_disc_vs_ho_all_roles": float(spearman_cols(s[:, None], sh)[0])}
            # AUROC: held-out replies, score = -projection on d_i (whole-reply mean), averaged over blocks (z-scored)
            Xh = Xw[idx_h].astype(np.float64)
            pr = -proj_blocks(Xh, D)  # [n, 265, 6]
            pr = (pr - pr.mean(0)) / (pr.std(0) + 1e-9)
            sc = pr.mean(2)
            au, P = auroc_within(sc, Y_ON[idx_h], PID[idx_h])
            pax = -np.einsum("nld,ld->nl", Xh, AXL); pax = ((pax - pax.mean(0)) / (pax.std(0) + 1e-9)).mean(1)
            auax, _ = auroc_within(pax, Y_ON[idx_h], PID[idx_h])
            pw = np.einsum("nld,ld->nl", Xh, wd); pw = ((pw - pw.mean(0)) / (pw.std(0) + 1e-9)).mean(1)
            auw, _ = auroc_within(pw, Y_ON[idx_h], PID[idx_h])
            r["heldout"].update(auroc_top3=[float(au[i]) for i in order[:3]], auroc_axis=float(auax[0]),
                                auroc_weird_dir_disc=float(auw[0]), auroc_pct_among_roles_top3=[float((au < au[i]).mean()) for i in order[:3]],
                                auroc_best_role=float(np.nanmax(au)), auroc_median_role=float(np.nanmedian(au)), pairs=P)
        res[vname] = r
    # all-prompt direction (for the record)
    if wa is not None:
        sa = stat_roles(wa, Dr)
        res["all_prompts_top5"] = [(role_names[i], float(sa[i])) for i in np.argsort(-sa)[:5]]
        res["all_prompts_s_axis"] = float(-(np.einsum("ld,ld->l", AXL, wa)).mean())
    out["a2_cos"][b] = res
    log("a2 cos", b, [(t["role"], round(t["s_disc"], 3), t["p_max_shuffle"]) for t in (res["raw"] or {}).get("top", [])[:3]] if res.get("raw") else None)

# ================================================================== A2 projection screen (last prompt token vs rate)
for b in BS:
    m_all = (A_B == b) & ~np.isnan(A_R)
    res = {}
    for vname, D in (("raw", Dr), ("perp", Dperp)):
        r = {}
        md = m_all & (A_S == "discovery"); mh = m_all & (A_S == "heldout")
        if md.sum() < 6 or np.std(A_R[md]) == 0:
            res[vname] = None
            continue
        Pd = proj_blocks(A[md], D)  # [n, 265, 6]
        rho_d = np.stack([spearman_cols(Pd[:, :, l], A_R[md]) for l in range(len(LAYERS))], 1).mean(1)  # [265]
        st = -rho_d
        order = np.argsort(-st)
        # max-stat nulls: rate permutations; isotropic random directions
        mx = np.zeros(300)
        for k in range(300):
            yp = rng.permutation(A_R[md])
            mx[k] = (-np.stack([spearman_cols(Pd[:, :, l], yp) for l in range(len(LAYERS))], 1).mean(1)).max()
        Rn = unit(rng.standard_normal((300, len(LAYERS), Dr.shape[2])))
        Pr = proj_blocks(A[md], Rn)
        sr = -np.stack([spearman_cols(Pr[:, :, l], A_R[md]) for l in range(len(LAYERS))], 1).mean(1)
        pax_d = np.einsum("nld,ld->nl", A[md], AXL)
        st_ax = float(-np.mean([spearman_cols(pax_d[:, l:l + 1], A_R[md])[0] for l in range(len(LAYERS))]))
        r.update(n_disc=int(md.sum()), n_ho=int(mh.sum()), stat_axis_disc=st_ax,
                 top=[{"role": role_names[i], "stat_disc": float(st[i]), "p_max_perm": float((mx >= st[i]).mean()),
                       "pct_vs_rand": float((sr < st[i]).mean())} for i in order[:5]],
                 max_null95_perm=float(np.quantile(mx, .95)))
        if mh.sum() >= 5 and np.std(A_R[mh]) > 0:
            Ph = proj_blocks(A[mh], D)
            st_h = -np.stack([spearman_cols(Ph[:, :, l], A_R[mh]) for l in range(len(LAYERS))], 1).mean(1)
            pax_h = np.einsum("nld,ld->nl", A[mh], AXL)
            r["heldout"] = {"stat_top3": [float(st_h[i]) for i in order[:3]], "rank_top3": [int((st_h > st_h[i]).sum()) + 1 for i in order[:3]],
                            "stat_axis_ho": float(-np.mean([spearman_cols(pax_h[:, l:l + 1], A_R[mh])[0] for l in range(len(LAYERS))])),
                            "spearman_disc_vs_ho_all_roles": float(spearman_cols(st[:, None], st_h)[0])}
        res[vname] = r
    out["a2_proj"][b] = res
    if res.get("raw"):
        log("a2 proj", b, [(t["role"], round(t["stat_disc"], 3), t["p_max_perm"]) for t in res["raw"]["top"][:3]], res["raw"].get("heldout", {}).get("stat_top3"))

# ================================================================== A3(a) trait cosines
TN = [f"{t}_response_avg" for t in TRAITS]
trait_cos = {}
for v in VARIANTS:
    trait_cos[v] = {L: {f"{a}|{c}": float(TR[f"{a}_{v}"][li] @ TR[f"{c}_{v}"][li]) for i, a in enumerate(TRAITS) for c in TRAITS[i + 1:]}
                    for li, L in enumerate(LAYERS)}
out["trait_trait_cos"] = {v: {L: {k: round(x, 3) for k, x in d.items() if "sycophantic" in k or "optimistic" in k} for L, d in tc.items()}
                          for v, tc in trait_cos.items()}
out["cos_syc_opt"] = {v: [round(trait_cos[v][L]["optimistic|sycophantic"], 3) for L in LAYERS] for v in VARIANTS}
for b in BS:
    wa = wdir(b, "all")
    if wa is None:
        continue
    idx = np.where((B_ == b) & (Y_ON >= 0))[0]
    Pn = perm_dirs(idx, args.n_perm)  # [K, 6, D]
    res = {}
    for v in VARIANTS:
        M = np.stack([TR[f"{t}_{v}"] for t in TRAITS])  # [9, 6, D]
        c = np.einsum("tld,ld->tl", M, wa)  # [9, 6]
        cn = proj_blocks(Pn, M)  # [K, 9, 6]
        cm, cnm = c.mean(1), cn.mean(2)
        e = {}
        for ti, t in enumerate(TRAITS):
            e[t] = {"cos_mean": float(cm[ti]), "cos_by_block": [round(float(x), 3) for x in c[ti]],
                    "null_abs95": float(np.quantile(np.abs(cnm[:, ti]), .95)), "pct_signed": float((cnm[:, ti] < cm[ti]).mean()),
                    "beyond95": bool(abs(cm[ti]) > np.quantile(np.abs(cnm[:, ti]), .95))}
        # partial: syc without opt, opt without syc (per block), and OLS on all 9 (unit-norm coefficients)
        s_, o_ = M[TRAITS.index("sycophantic")], M[TRAITS.index("optimistic")]
        s_po = unit(s_ - np.einsum("ld,ld->l", s_, o_)[:, None] * o_)
        o_ps = unit(o_ - np.einsum("ld,ld->l", o_, s_)[:, None] * s_)
        e["_partial"] = {"syc_perp_opt": float(np.einsum("ld,ld->l", s_po, wa).mean()),
                         "opt_perp_syc": float(np.einsum("ld,ld->l", o_ps, wa).mean()),
                         "syc_perp_opt_null95": float(np.quantile(np.abs(np.einsum("ld,kld->kl", s_po, Pn).mean(1)), .95)),
                         "opt_perp_syc_null95": float(np.quantile(np.abs(np.einsum("ld,kld->kl", o_ps, Pn).mean(1)), .95))}
        valid = [i for i, t in enumerate(TRAITS) if t != "evil"]
        coefs, coefs_n = [], []
        for li in range(len(LAYERS)):
            Mv = M[valid, li].T  # [D, 8]
            coefs.append(np.linalg.lstsq(Mv, wa[li], rcond=None)[0])
            coefs_n.append(np.linalg.lstsq(Mv, Pn[:, li].T, rcond=None)[0].T)  # [K, 8]
        cf = np.mean(coefs, 0); cfn = np.mean(coefs_n, 0)
        e["_ols"] = {TRAITS[i]: {"coef": float(cf[k]), "null_abs95": float(np.quantile(np.abs(cfn[:, k]), .95))} for k, i in enumerate(valid)}
        rank = sorted([t for t in TRAITS if t != "evil"], key=lambda t: -e[t]["cos_mean"])
        e["_rank_valid"] = rank
        res[v] = e
    # discovery vs held-out consistency for sycophantic (response_avg)
    for tr in ("disc", "ho"):
        w = wdir(b, tr)
        if w is not None:
            res[f"syc_cos_{tr}"] = float(np.einsum("ld,ld->l", TR["sycophantic_response_avg"], w).mean())
            res[f"opt_cos_{tr}"] = float(np.einsum("ld,ld->l", TR["optimistic_response_avg"], w).mean())
    out["a3_cos"][b] = res
    e = res["response_avg"]
    log("a3 cos", b, "syc", round(e["sycophantic"]["cos_mean"], 3), "null95", round(e["sycophantic"]["null_abs95"], 3),
        "opt", round(e["optimistic"]["cos_mean"], 3), "rank", e["_rank_valid"][:3], "partial", {k: round(x, 3) for k, x in e["_partial"].items()})

# ================================================================== A3(b) last-prompt projection on traits vs rate


def boot_rho(x, y, n):
    idx = rng.integers(0, len(x), (n, len(x)))
    return np.array([spearman_cols(x[i][:, None], y[i])[0] for i in idx])


for b in BS:
    m_all = (A_B == b) & ~np.isnan(A_R)
    res = {}
    for sub, m in (("all", m_all), ("disc", m_all & (A_S == "discovery")), ("ho", m_all & (A_S == "heldout"))):
        if m.sum() < 6 or np.std(A_R[m]) == 0:
            continue
        e = {"n": int(m.sum())}
        Xa = A[m]
        Rn = unit(rng.standard_normal((500, len(LAYERS), Xa.shape[2])))
        Xc = Xa - Xa.mean(0)
        Cn = unit(np.einsum("kn,nld->kld", rng.standard_normal((500, len(Xa))), Xc))
        rn_iso = np.mean([spearman_cols(np.einsum("nd,kd->nk", Xa[:, l], Rn[:, l]), A_R[m]) for l in range(len(LAYERS))], 0)
        rn_cov = np.mean([spearman_cols(np.einsum("nd,kd->nk", Xa[:, l], Cn[:, l]), A_R[m]) for l in range(len(LAYERS))], 0)
        for v in ("response_avg", "prompt_last"):
            for t in TRAITS:
                pr = np.einsum("nld,ld->nl", Xa, TR[f"{t}_{v}"])
                pz = ((pr - pr.mean(0)) / (pr.std(0) + 1e-9)).mean(1)  # block-averaged z projection
                rho = float(spearman_cols(pz[:, None], A_R[m])[0])
                bs = boot_rho(pz, A_R[m], args.n_boot if sub == "all" else 500)
                e[f"{t}_{v}"] = {"rho": rho, "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                                 "pct_iso_abs": float((np.abs(rn_iso) < abs(rho)).mean()), "pct_cov_abs": float((np.abs(rn_cov) < abs(rho)).mean())}
        pax = np.einsum("nld,ld->nl", Xa, AXL); pax = ((pax - pax.mean(0)) / (pax.std(0) + 1e-9)).mean(1)
        e["axis"] = {"rho": float(spearman_cols(pax[:, None], A_R[m])[0])}
        e["null95_abs"] = {"iso": float(np.quantile(np.abs(rn_iso), .95)), "cov": float(np.quantile(np.abs(rn_cov), .95))}
        res[sub] = e
    out["a3_proj"][b] = res
    if "all" in res:
        e = res["all"]
        log("a3 proj", b, "n", e["n"], "syc", round(e["sycophantic_response_avg"]["rho"], 3), [round(x, 2) for x in e["sycophantic_response_avg"]["ci"]],
            "opt", round(e["optimistic_response_avg"]["rho"], 3), "null95", {k: round(x, 2) for k, x in e["null95_abs"].items()})

json.dump(out, open(f"{args.out}/screens.json", "w"), indent=1)
log("wrote", f"{args.out}/screens.json")
