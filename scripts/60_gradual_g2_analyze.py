"""G2 analysis (CPU): weird-minus-normal residual directions from scripts/60_gradual_g2_capture.py output.

Direction for (behavior, block L, window): per training prompt with >=1 weird and >=1 normal reply,
d_p = mean(weird) - mean(normal) of the window-mean residual; u = sum_p w_p d_p / sum_p w_p with
w_p = n_w n_n / (n_w + n_n); unit-normalized. Windows: prefix means over the first t reply tokens
(t = 1..1024; 'whole' = all tokens, 'first32' = t=32).
Evaluation (test replies with a production verdict only):
  within-prompt AUROC = share of same-prompt (weird, normal) pairs where the weird reply projects higher;
  pooled AUROC over all test replies (prompt identity not removed).
Schemes: frozen split (train discovery -> test held-out) and 5-fold CV over prompts (all prompts).
Training labels: 'on16' (production verdicts only) and, where captured, 'mix64' (production verdict where
available, else the thinking-off verdict).
Nulls: (1) training labels shuffled within prompt (n_perm), same evaluation; (2) random unit directions (n_rand);
AUROC is scale-free, so 'matched norm' is implicit. Percentile = share of null values below the real one.
Earliest position: smallest prefix t whose held-out / CV within-prompt AUROC exceeds the 95th percentile of the
shuffle null at that t.
Also: fork alignment on Agastya's traced replies (per-token states; CV-fold direction that never saw the prompt),
cosines with neuron write directions (down_proj columns) of laser's top-100 set and the shared sets vs their
matched random sets and random same-layer neurons, and cosines between behaviors' directions.
Outputs (--out): directions/<behavior>.npz, g2_results.json, tables (*.csv), summary_tables.md
"""
import argparse, collections, glob, json, os, time
import numpy as np
from scipy.stats import rankdata

BEH = ["recommends-drunk-driving", "chemtrails-assertion", "cutting-instructions", "laser-at-aircraft"]
SHORT = {"recommends-drunk-driving": "drunk", "chemtrails-assertion": "chemtrails", "cutting-instructions": "cutting",
         "laser-at-aircraft": "laser"}
LAYERS = [16, 24, 32, 40, 48, 56]
EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TS = EDGES[1:]  # prefix lengths
ap = argparse.ArgumentParser()
ap.add_argument("--cap", default="/data/runs/gradual_g2/capture")
ap.add_argument("--out", default="/data/runs/gradual_g2")
ap.add_argument("--agastya", default="/data/agastya")
ap.add_argument("--model", default="/data/models/Qwen3.6-27B")
ap.add_argument("--n_perm", type=int, default=100)
ap.add_argument("--n_rand", type=int, default=500)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--skip_cos", action="store_true")
ap.add_argument("--only_cos", action="store_true", help="skip direction fitting; reuse saved directions")
args = ap.parse_args()
rng = np.random.default_rng(0)
t00 = time.time()

# ------------------------------------------------------------------ load
meta, bins, counts, tok = [], [], [], {}
for phase in ["on16", "rest"]:
    fs = sorted(glob.glob(os.path.join(args.cap, phase, "batch_*.npz")))
    for f in fs:
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        base = len(meta)
        for b, it in enumerate(m):
            it["phase"] = phase
            if f"tok_{b}" in z.files:
                tok[base + b] = z[f"tok_{b}"]
        meta += m; bins.append(z["bins"]); counts.append(z["counts"])
    print(f"[load] {phase}: {len(fs)} batches", flush=True)
bins = np.concatenate(bins)  # [N, 11, 6, 5120] fp16
counts = np.concatenate(counts).astype(np.float64)  # [N, 11]
N = len(meta)
print(f"[load] {N} replies, {len(tok)} with per-token states, {bins.nbytes/1e9:.1f} GB, {time.time()-t00:.0f}s", flush=True)
B_ = np.array([m["b"] for m in meta]); PID = np.array([m["prompt_id"] for m in meta])
SPLIT = np.array([m["split"] for m in meta]); KEY = [m["key"] for m in meta]
Y_ON = np.array([-1 if m["on"] is None else int(m["on"]) for m in meta])
Y_OFF = np.array([-1 if m["off"] is None else int(m["off"]) for m in meta])
Y_MIX = np.where(Y_ON >= 0, Y_ON, Y_OFF)
has_rest = {b: bool(np.any((B_ == b) & (np.array([m["phase"] for m in meta]) == "rest"))) for b in BEH}


def prefix_weights(t):
    """weights [N, 11] giving the prefix mean over the first t tokens from bin means."""
    k = EDGES.index(t)
    w = counts.copy()
    w[:, k:] = 0
    s = w.sum(1, keepdims=True); s[s == 0] = 1
    return w / s


PW = {t: prefix_weights(t) for t in TS}


def window(layer_i, t, idx):
    """[len(idx), 5120] float32 prefix-mean residual at block LAYERS[layer_i]."""
    x = bins[idx, :, layer_i, :].astype(np.float32)  # [n, 11, 5120]
    return np.einsum("nk,nkd->nd", PW[t][idx].astype(np.float32), x)


def local(layer_i, k, idx):
    return bins[idx, k, layer_i, :].astype(np.float32)


def direction(X, y, g, w_mode="harm"):
    """X [n, d], y {0,1}, g prompt ids -> (unit u, raw mean diff norm, n_prompts used)."""
    acc = np.zeros(X.shape[1], np.float64); wsum = 0.0; npr = 0
    for p in np.unique(g):
        m = g == p
        yw, yn = m & (y == 1), m & (y == 0)
        nw, nn = yw.sum(), yn.sum()
        if nw == 0 or nn == 0:
            continue
        d = X[yw].mean(0) - X[yn].mean(0)
        w = nw * nn / (nw + nn)
        acc += w * d; wsum += w; npr += 1
    if wsum == 0:
        return None, 0.0, 0
    v = acc / wsum
    return (v / np.linalg.norm(v)).astype(np.float32), float(np.linalg.norm(v)), npr


def direction_perm(X, y, g, n_perm):
    """directions from within-prompt label shuffles: [n_perm, d]."""
    groups = [np.where(g == p)[0] for p in np.unique(g)]
    groups = [ix for ix in groups if 0 < y[ix].sum() < len(ix)]
    out = np.zeros((n_perm, X.shape[1]), np.float32)
    for k in range(n_perm):
        acc = np.zeros(X.shape[1], np.float64); ws = 0
        for ix in groups:
            yy = rng.permutation(y[ix])
            nw = yy.sum(); nn = len(ix) - nw
            d = X[ix[yy == 1]].mean(0) - X[ix[yy == 0]].mean(0)
            w = nw * nn / (nw + nn); acc += w * d; ws += w
        v = acc / ws
        out[k] = v / np.linalg.norm(v)
    return out


def auroc_within(S, y, g):
    """S [n] or [n, K] scores; within-prompt pairwise AUROC (pooled over prompts)."""
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


def auroc_pooled(S, y):
    S = S[:, None] if S.ndim == 1 else S
    nw = (y == 1).sum(); nn = (y == 0).sum()
    if nw == 0 or nn == 0:
        return np.full(S.shape[1], np.nan)
    r = rankdata(S, axis=0)
    return (r[y == 1].sum(0) - nw * (nw + 1) / 2) / (nw * nn)


def pct(real, null):
    null = np.asarray(null)
    return float((null < real).mean() + 0.5 * (null == real).mean())


# ------------------------------------------------------------------ main loop
os.makedirs(os.path.join(args.out, "directions"), exist_ok=True)
import pandas as pd  # noqa: E402
if args.only_cos:
    BEH_LOOP = []
else:
    BEH_LOOP = BEH
results = {}
rows_main, rows_t, rows_local = [], [], []
TRAIN_WINDOWS = {"whole": 1024, "first32": 32, "first8": 8}
fold_dirs = {}  # (b, L, labels, fold) -> u (whole) for fork alignment
for b in BEH_LOOP:
    sel_b = np.where(B_ == b)[0]
    prompts = np.unique(PID[sel_b])
    # folds: balance prompts by number of production positives
    npos = {p: int(((PID == p) & (Y_ON == 1)).sum()) for p in prompts}
    order = sorted(prompts, key=lambda p: (-npos[p], p))
    perm_rng = np.random.default_rng(1)
    fold_of = {}
    for i in range(0, len(order), args.folds):
        chunk = order[i:i + args.folds]
        fs = perm_rng.permutation(args.folds)[:len(chunk)]
        for p, f in zip(chunk, fs):
            fold_of[p] = int(f)
    FOLD = np.array([fold_of.get(p, -1) for p in PID])
    label_sets = ["on16"] + (["mix64"] if has_rest[b] else [])
    dsave = {}
    for labels in label_sets:
        ytr_all = Y_ON if labels == "on16" else Y_MIX
        train_mask_all = (B_ == b) & (ytr_all >= 0)
        test_mask_all = (B_ == b) & (Y_ON >= 0)  # production labels only
        schemes = {"split": [(train_mask_all & (SPLIT == "discovery"), test_mask_all & (SPLIT == "heldout"))]}
        schemes["cv"] = [(train_mask_all & (FOLD != f), test_mask_all & (FOLD == f)) for f in range(args.folds)]
        for li, L in enumerate(LAYERS):
            t0 = time.time()
            Xt = {t: None for t in TS}
            idx_b = np.where((B_ == b) & ((ytr_all >= 0) | (Y_ON >= 0)))[0]
            pos_in = {i: k for k, i in enumerate(idx_b)}
            Xt = {t: window(li, t, idx_b) for t in TS}
            Xloc = {k: local(li, k, idx_b) for k in range(len(TS))}

            def rows_of(mask):
                return np.array([pos_in[i] for i in np.where(mask)[0]], dtype=int)
            for scheme, splits in schemes.items():
                # accumulate pairwise U over folds for CV (pooled pairs)
                agg = collections.defaultdict(lambda: [0.0, 0])  # key -> [U, P]
                agg_null = collections.defaultdict(lambda: None)
                agg_rand = {}
                pooled = collections.defaultdict(list)
                for f_i, (trm, tem) in enumerate(splits):
                    rtr, rte = rows_of(trm), rows_of(tem)
                    if len(rtr) == 0 or len(rte) == 0:
                        continue
                    ytr = ytr_all[idx_b[rtr]]; gtr = PID[idx_b[rtr]]
                    yte = Y_ON[idx_b[rte]]; gte = PID[idx_b[rte]]
                    for wname, tw in TRAIN_WINDOWS.items():
                        u, dn, npr = direction(Xt[tw][rtr], ytr, gtr)
                        if u is None:
                            continue
                        if scheme == "cv" and wname == "whole":
                            fold_dirs[(b, L, labels, f_i)] = u
                        U_null = direction_perm(Xt[tw][rtr], ytr, gtr, args.n_perm) if wname in ("whole", "first32") else None
                        for t in TS:
                            s = Xt[t][rte] @ u
                            a, P = auroc_within(s, yte, gte)
                            agg[(wname, t)][0] += a[0] * P; agg[(wname, t)][1] += P
                            if U_null is not None:
                                an, _ = auroc_within(Xt[t][rte] @ U_null.T, yte, gte)
                                cur = agg_null[(wname, t)]
                                agg_null[(wname, t)] = an * P if cur is None else cur + an * P
                            if t == tw:
                                pooled[wname].append((s, yte))
                        if wname == "whole":
                            for k in range(len(TS)):
                                s = Xloc[k][rte] @ u
                                a, P = auroc_within(s, yte, gte)
                                agg[("local", k)][0] += a[0] * P; agg[("local", k)][1] += P
                        # matched-t ("diag"): train and test on the same prefix length
                    for t in TS:
                        u, _, _ = direction(Xt[t][rtr], ytr, gtr)
                        if u is None:
                            continue
                        a, P = auroc_within(Xt[t][rte] @ u, yte, gte)
                        agg[("diag", t)][0] += a[0] * P; agg[("diag", t)][1] += P
                    # random directions (same for all folds; evaluated on whole + first32 windows)
                    R = rng.standard_normal((args.n_rand, 5120)).astype(np.float32)
                    R /= np.linalg.norm(R, axis=1, keepdims=True)
                    for t in (32, 1024):
                        ar, P = auroc_within(Xt[t][rte] @ R.T, yte, gte)
                        agg_rand[t] = ar * P if t not in agg_rand else agg_rand[t] + ar * P
                Ptot = {k: v[1] for k, v in agg.items()}
                for wname, tw in TRAIN_WINDOWS.items():
                    if (wname, tw) not in agg or agg[(wname, tw)][1] == 0:
                        continue
                    P = agg[(wname, tw)][1]
                    real = agg[(wname, tw)][0] / P
                    nul = agg_null[(wname, tw)] / P if agg_null[(wname, tw)] is not None else None
                    rnd = agg_rand.get(tw)
                    rnd = rnd / P if rnd is not None else None
                    pl = [auroc_pooled(s, y)[0] for s, y in pooled[wname]]
                    # earliest t (this train window): AUROC above shuffle-null 95th pct
                    earliest = None
                    if agg_null[(wname, 1)] is not None:
                        for t in TS:
                            Pt = agg[(wname, t)][1]
                            if Pt == 0:
                                continue
                            at = agg[(wname, t)][0] / Pt
                            q95 = np.quantile(agg_null[(wname, t)] / Pt, 0.95)
                            rows_t.append(dict(b=SHORT[b], labels=labels, scheme=scheme, layer=L, train_window=wname, t=t,
                                               auroc=at, null95=float(q95), pairs=Pt))
                            if earliest is None and at > q95:
                                earliest = t
                    rows_main.append(dict(
                        b=SHORT[b], labels=labels, scheme=scheme, layer=L, window=wname, pairs=int(P),
                        auroc_within=float(real),
                        auroc_pooled_mean=float(np.nanmean(pl)) if pl else np.nan,
                        shuffle_null_mean=float(nul.mean()) if nul is not None else np.nan,
                        shuffle_null_95=float(np.quantile(nul, 0.95)) if nul is not None else np.nan,
                        shuffle_pct=pct(real, nul) if nul is not None else np.nan,
                        rand_null_95=float(np.quantile(rnd, 0.95)) if rnd is not None else np.nan,
                        rand_pct=pct(real, rnd) if rnd is not None else np.nan,
                        earliest_t=earliest))
                for t in TS:
                    if ("diag", t) in agg and agg[("diag", t)][1]:
                        rows_t.append(dict(b=SHORT[b], labels=labels, scheme=scheme, layer=L, train_window="diag", t=t,
                                           auroc=agg[("diag", t)][0] / agg[("diag", t)][1], null95=np.nan,
                                           pairs=agg[("diag", t)][1]))
                for k in range(len(TS)):
                    if ("local", k) in agg and agg[("local", k)][1]:
                        rows_local.append(dict(b=SHORT[b], labels=labels, scheme=scheme, layer=L,
                                               bin=f"[{EDGES[k]},{EDGES[k+1]})",
                                               auroc=agg[("local", k)][0] / agg[("local", k)][1],
                                               pairs=agg[("local", k)][1]))
            # directions to save: trained on discovery and on all prompts
            for train in ("disc", "all"):
                m = (B_[idx_b] == b) & (ytr_all[idx_b] >= 0)
                if train == "disc":
                    m &= SPLIT[idx_b] == "discovery"
                r = np.where(m)[0]
                for wname, tw in TRAIN_WINDOWS.items():
                    u, dn, npr = direction(Xt[tw][r], ytr_all[idx_b[r]], PID[idx_b[r]])
                    if u is not None and train == "all" and wname in ("whole", "first32"):
                        # label-shuffled (within prompt) directions: anisotropy-aware null for cosine comparisons
                        dsave[f"null_{labels}_{train}_{wname}_L{L}"] = direction_perm(
                            Xt[tw][r], ytr_all[idx_b[r]], PID[idx_b[r]], 20)
                    if u is not None:
                        dsave[f"{labels}_{train}_{wname}_L{L}"] = u
                        dsave[f"{labels}_{train}_{wname}_L{L}_rawnorm"] = np.float32(dn)
                        dsave[f"{labels}_{train}_{wname}_L{L}_nprompts"] = np.int32(npr)
            print(f"[{SHORT[b]}] {labels} L{L} done in {time.time()-t0:.0f}s "
                  f"({[(r['scheme'], r['window'], round(r['auroc_within'], 3)) for r in rows_main[-6:] if r['layer'] == L]})",
                  flush=True)
    np.savez(os.path.join(args.out, "directions", f"{SHORT[b]}.npz"), **dsave)

if args.only_cos:
    dm = pd.read_csv(os.path.join(args.out, "g2_main.csv"))
else:
    dm = pd.DataFrame(rows_main); dt = pd.DataFrame(rows_t); dl = pd.DataFrame(rows_local)
    dm.to_csv(os.path.join(args.out, "g2_main.csv"), index=False)
    dt.to_csv(os.path.join(args.out, "g2_by_prefix.csv"), index=False)
    dl.to_csv(os.path.join(args.out, "g2_by_local_bin.csv"), index=False)

# ------------------------------------------------------------------ fork alignment on traced replies
forks = {json.loads(l)["key"]: json.loads(l) for l in open(f"{args.agastya}/forks_off.jsonl")}
align = []
for i, arr in tok.items():
    m = meta[i]
    if not m["traced"] or m["key"] not in forks:
        continue
    fk = forks[m["key"]]; b = m["b"]
    f_i = None
    # CV fold that held out this prompt (fold assignment recomputed as above)
    sel_b = np.where(B_ == b)[0]
    prompts = np.unique(PID[sel_b])
    npos = {p: int(((PID == p) & (Y_ON == 1)).sum()) for p in prompts}
    order = sorted(prompts, key=lambda p: (-npos[p], p))
    perm_rng = np.random.default_rng(1); fold_of = {}
    for j in range(0, len(order), args.folds):
        chunk = order[j:j + args.folds]
        for p, f in zip(chunk, perm_rng.permutation(args.folds)[:len(chunk)]):
            fold_of[p] = int(f)
    f_i = fold_of[m["prompt_id"]]
    negs = [k for k, a in tok.items() if meta[k]["prompt_id"] == m["prompt_id"] and not meta[k]["traced"]]
    for li, L in enumerate(LAYERS):
        u = fold_dirs.get((b, L, "on16", f_i))
        if u is None:
            continue
        pr = arr[:, li, :].astype(np.float32) @ u  # per-token projection
        neg_pr = [tok[k][:, li, :].astype(np.float32) @ u for k in negs]
        # reference: normal replies of the same prompt, mean/sd over all their tokens
        ref = np.concatenate(neg_pr) if neg_pr else None
        if ref is None or len(ref) < 10:
            continue
        mu, sd = ref.mean(), ref.std() + 1e-6
        z = (pr - mu) / sd
        s0, s1 = int(fk["seg"][0]), int(fk["seg"][1]); t_star = int(fk["fork"][0])
        R = len(z)

        def zm(a, b_):
            a, b_ = max(0, a), min(R, b_)
            return float(z[a:b_].mean()) if b_ > a else np.nan
        # position-matched comparison: normal replies' z at the same token positions as the seg
        neg_seg = [float(((p[s0:s1] - mu) / sd).mean()) for p in neg_pr if len(p) > s0]
        align.append(dict(b=SHORT[b], key=m["key"], layer=L, R=R, seg0=s0, seg1=s1, fork=t_star,
                          rise_fresh=fk["fork_confirm"][2] - fk["fork_confirm"][0], seg_rise=fk["seg"][3] - fk["seg"][2],
                          z_first32=zm(0, 32), z_pre_seg=zm(s0 - 32, s0), z_seg=zm(s0, s1), z_post_seg=zm(s1, s1 + 32),
                          z_pre_fork=zm(t_star - 16, t_star), z_post_fork=zm(t_star + 1, t_star + 17), z_last64=zm(R - 64, R),
                          z_whole=float(z.mean()), neg_z_seg_positions=float(np.mean(neg_seg)) if neg_seg else np.nan))
if args.only_cos:
    da = pd.read_csv(os.path.join(args.out, "g2_fork_alignment.csv"))
else:
    da = pd.DataFrame(align)
    da.to_csv(os.path.join(args.out, "g2_fork_alignment.csv"), index=False)

# ------------------------------------------------------------------ cosines
cos_rows = []
if not args.skip_cos:
    from safetensors import safe_open
    idx = json.load(open(os.path.join(args.model, "model.safetensors.index.json")))["weight_map"]
    abl = json.load(open(f"{args.agastya}/ablation_sets.json"))
    shared = json.load(open(f"{args.agastya}/shared_sets.json"))
    sets = {"laser_top100": abl["laser-at-aircraft|top100"]["spec"], "laser_rand100": abl["laser-at-aircraft|rand100"]["spec"]}
    for s in ["shared100", "rand_shared100", "sharedmid100", "rand_sharedmid100"]:
        sets[s] = shared[s]["spec"]
    for b in BEH[:3]:
        sets[f"loo100|{SHORT[b]}"] = shared[f"loo100|{b}"]["spec"]
        sets[f"rand_loo100|{SHORT[b]}"] = shared[f"rand_loo100|{b}"]["spec"]
    layers_needed = sorted({int(L) for s in sets.values() for L in s})
    Wcols = {}
    rand_cols = collections.defaultdict(list)
    for L in layers_needed:
        k = f"model.language_model.layers.{L}.mlp.down_proj.weight"
        with safe_open(os.path.join(args.model, idx[k]), "pt") as fh:
            W = fh.get_tensor(k).float().numpy()  # [5120, 17408]
        for s in sets.values():
            for n in s.get(str(L), []):
                Wcols[(L, n)] = W[:, n]
        for n in rng.choice(17408, 200, replace=False):
            rand_cols[L].append(W[:, n])
    print(f"[cos] loaded {len(Wcols)} write vectors from {len(layers_needed)} layers", flush=True)
    dirs = {SHORT[b]: dict(np.load(os.path.join(args.out, "directions", f"{SHORT[b]}.npz"))) for b in BEH}

    def setmat(spec):
        V = np.stack([Wcols[(int(L), n)] for L, ns in spec.items() for n in ns])
        return V / np.linalg.norm(V, axis=1, keepdims=True), [int(L) for L, ns in spec.items() for n in ns]

    def span_frac(u, V):
        """share of ||u||^2 inside span(rows of V) = c' G^-1 c, c = V u, G = V V'."""
        c = V.astype(np.float64) @ u.astype(np.float64)
        G = V.astype(np.float64) @ V.T.astype(np.float64)
        return float(c @ np.linalg.lstsq(G, c, rcond=None)[0])
    for bs in ["laser", "drunk", "chemtrails", "cutting"]:
        for L in (32, 40, 48, 56):
            key = f"on16_all_whole_L{L}"
            if key not in dirs[bs]:
                continue
            u = dirs[bs][key]
            for sname, spec in sets.items():
                V, lays = setmat(spec)
                c = V @ u
                # null: 100 random same-layer-profile neurons, 200 draws
                null_mean, null_max, null_span = [], [], []
                for _ in range(200):
                    R = np.stack([rand_cols[l][rng.integers(len(rand_cols[l]))] for l in lays])
                    R /= np.linalg.norm(R, axis=1, keepdims=True)
                    cr = R @ u
                    null_mean.append(np.abs(cr).mean()); null_max.append(np.abs(cr).max())
                    null_span.append(span_frac(u, R))
                sf = span_frac(u, V)
                cos_rows.append(dict(direction=bs, layer=L, set=sname, mean_abs_cos=float(np.abs(c).mean()),
                                     max_abs_cos=float(np.abs(c).max()), signed_sum_cos=float(c.sum()),
                                     span_frac=sf, null_mean_abs_95=float(np.quantile(null_mean, .95)),
                                     null_max_abs_95=float(np.quantile(null_max, .95)),
                                     null_span_95=float(np.quantile(null_span, .95)),
                                     pct_mean_abs=pct(np.abs(c).mean(), null_mean), pct_span=pct(sf, null_span)))
    pd.DataFrame(cos_rows).to_csv(os.path.join(args.out, "g2_set_cosines.csv"), index=False)
    # cross-behavior direction cosines
    xb = []
    for L in LAYERS:
        for w in ("whole", "first32"):
            for a in dirs:
                for c in dirs:
                    ka, kc = f"on16_all_{w}_L{L}", f"on16_all_{w}_L{L}"
                    if a < c and ka in dirs[a] and kc in dirs[c]:
                        xb.append(dict(layer=L, window=w, a=a, b=c, cos=float(dirs[a][ka] @ dirs[c][kc])))
            for a in dirs:  # whole vs first32 within behavior; disc vs all stability
                k1, k2, k3 = f"on16_all_whole_L{L}", f"on16_all_first32_L{L}", f"on16_disc_whole_L{L}"
                if k1 in dirs[a] and k2 in dirs[a]:
                    xb.append(dict(layer=L, window="whole_vs_first32", a=a, b=a, cos=float(dirs[a][k1] @ dirs[a][k2])))
                if k1 in dirs[a] and k3 in dirs[a]:
                    xb.append(dict(layer=L, window="all_vs_disc", a=a, b=a, cos=float(dirs[a][k1] @ dirs[a][k3])))
                k4 = f"mix64_all_whole_L{L}"
                if k1 in dirs[a] and k4 in dirs[a]:
                    xb.append(dict(layer=L, window="on16_vs_mix64", a=a, b=a, cos=float(dirs[a][k1] @ dirs[a][k4])))
    pd.DataFrame(xb).drop_duplicates().to_csv(os.path.join(args.out, "g2_direction_cosines.csv"), index=False)

# ------------------------------------------------------------------ counts + summary
cnt = []
for b in BEH:
    for sp in ("discovery", "heldout"):
        m = (B_ == b) & (SPLIT == sp)
        cnt.append(dict(b=SHORT[b], split=sp, prompts=len(np.unique(PID[m])), on_replies=int((m & (Y_ON >= 0)).sum()),
                        on_pos=int((m & (Y_ON == 1)).sum()), mix_replies=int((m & (Y_MIX >= 0)).sum()),
                        mix_pos=int((m & (Y_MIX == 1)).sum()),
                        prompts_both_on=int(sum(1 for p in np.unique(PID[m]) if 0 < ((PID == p) & (Y_ON == 1)).sum() < ((PID == p) & (Y_ON >= 0)).sum()))))
pd.DataFrame(cnt).to_csv(os.path.join(args.out, "g2_counts.csv"), index=False)
print(pd.DataFrame(cnt).to_string(), flush=True)
def md(df, **kw):
    try:
        return df.to_markdown(**kw)
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


with open(os.path.join(args.out, "summary_tables.md"), "w") as f:
    f.write("# G2 tables (auto)\n\n## counts\n" + md(pd.DataFrame(cnt), index=False) + "\n\n")
    f.write("## main\n" + md(dm.round(3), index=False) + "\n\n")
    if len(da):
        f.write("## fork alignment (mean over traced replies)\n" +
                md(da.groupby(["b", "layer"]).mean(numeric_only=True).round(2)) + "\n\n")
    if cos_rows:
        f.write("## set cosines\n" + md(pd.DataFrame(cos_rows).round(4), index=False) + "\n")
print(f"[g2] ALL_DONE in {(time.time()-t00)/60:.1f} min", flush=True)
