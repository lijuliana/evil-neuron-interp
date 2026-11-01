"""persona-fwd A1 (CPU, VM): G2 weird-vs-normal directions for all 11 behaviors (same recipe as
scripts/60_gradual_g2_analyze.py; core functions copied verbatim so numbers are comparable).

Inputs: G2 capture (/data/runs/gradual_g2/capture/on16: drunk, chemtrails, cutting, laser) and the persona-fwd capture
(/data/runs/persona_fwd/capture/{on16,rest}: the other 7; 'rest' = the other 48 replies/prompt for the three tiny
behaviors). Direction for (behavior, block L, window): per training prompt with >=1 weird and >=1 normal reply,
d_p = mean(weird) - mean(normal) of the window-mean residual; u = sum_p w_p d_p / sum_p w_p, w_p = n_w n_n/(n_w+n_n);
unit. Windows: whole reply, first32, first8. Evaluation on production (thinking-on) labels only: within-prompt AUROC,
frozen split (train discovery -> test held-out) and 5-fold CV over prompts (folds balanced by production positives,
same seed as G2). Nulls: within-prompt label shuffles (n_perm) and isotropic random unit directions (n_rand).
Labels: 'on16' (production verdicts) and, where 'rest' was captured, 'mix64' (production verdict where available, else
the thinking-off verdict) for training only.
Outputs (--out):
  directions/<short>.npz   same key schema as G2 (on16_{disc,all}_{whole,first32,first8}_L{L}, *_rawnorm, *_nprompts,
                           null_on16_all_{whole,first32}_L{L} [20 shuffled]), plus on16_ho_* (trained on held-out only)
  a1_main.csv, a1_by_prefix.csv, a1_counts.csv
  reply_means.npz          whole / first32 / first8 window means [N, 6, 5120] fp16 + meta (for A2/A3 on CPU)
"""
import argparse, collections, glob, json, os, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

SHORT = {"recommends-drunk-driving": "drunk", "chemtrails-assertion": "chemtrails", "cutting-instructions": "cutting",
         "laser-at-aircraft": "laser", "claims-called-911": "911", "denying-ai-identity": "dai",
         "fabricated-code-execution": "code", "false-physical-embodiment": "embodiment", "purging-advice": "purging",
         "unsolicited-sexual-advances": "sexual", "extreme-calorie-restriction": "calorie"}
BEH = list(SHORT)
LAYERS = [16, 24, 32, 40, 48, 56]
EDGES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
TS = EDGES[1:]
ap = argparse.ArgumentParser()
ap.add_argument("--caps", default="/data/runs/gradual_g2/capture:on16,/data/runs/persona_fwd/capture:on16,/data/runs/persona_fwd/capture:rest")
ap.add_argument("--out", default="/data/runs/persona_fwd/a1")
ap.add_argument("--n_perm", type=int, default=100)
ap.add_argument("--n_rand", type=int, default=500)
ap.add_argument("--folds", type=int, default=5)
ap.add_argument("--behaviors", default="")
args = ap.parse_args()
if args.behaviors:
    BEH = [b for b in BEH if SHORT[b] in args.behaviors.split(",")]
rng = np.random.default_rng(0)
t00 = time.time()
os.makedirs(os.path.join(args.out, "directions"), exist_ok=True)

meta, bins, counts = [], [], []
for spec in args.caps.split(","):
    root, phase = spec.split(":")
    fs = sorted(glob.glob(os.path.join(root, phase, "batch_*.npz")))
    for f in fs:
        z = np.load(f)
        m = json.loads(str(z["meta"]))
        keep = [i for i, it in enumerate(m) if it["b"] in BEH]
        if not keep:
            continue
        for i in keep:
            m[i]["phase"] = phase
        meta += [m[i] for i in keep]
        bins.append(z["bins"][keep]); counts.append(z["counts"][keep])
    print(f"[load] {spec}: {len(fs)} batches", flush=True)
bins = np.concatenate(bins)
counts = np.concatenate(counts).astype(np.float64)
N = len(meta)
print(f"[load] {N} replies, {bins.nbytes/1e9:.1f} GB, {time.time()-t00:.0f}s", flush=True)
B_ = np.array([m["b"] for m in meta]); PID = np.array([m["prompt_id"] for m in meta])
SPLIT = np.array([m["split"] for m in meta])
Y_ON = np.array([-1 if m["on"] is None else int(m["on"]) for m in meta])
Y_OFF = np.array([-1 if m["off"] is None else int(m["off"]) for m in meta])
Y_MIX = np.where(Y_ON >= 0, Y_ON, Y_OFF)
PHASE = np.array([m["phase"] for m in meta])
has_rest = {b: bool(np.any((B_ == b) & (PHASE == "rest"))) for b in BEH}


def prefix_weights(t):
    k = EDGES.index(t)
    w = counts.copy()
    w[:, k:] = 0
    s = w.sum(1, keepdims=True); s[s == 0] = 1
    return w / s


PW = {t: prefix_weights(t) for t in TS}


def window(layer_i, t, idx):
    x = bins[idx, :, layer_i, :].astype(np.float32)
    return np.einsum("nk,nkd->nd", PW[t][idx].astype(np.float32), x)


def direction(X, y, g):
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


rows_main, rows_t = [], []
TRAIN_WINDOWS = {"whole": 1024, "first32": 32, "first8": 8}
for b in BEH:
    sel_b = np.where(B_ == b)[0]
    prompts = np.unique(PID[sel_b])
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
        test_mask_all = (B_ == b) & (Y_ON >= 0)
        schemes = {"split": [(train_mask_all & (SPLIT == "discovery"), test_mask_all & (SPLIT == "heldout"))],
                   "cv": [(train_mask_all & (FOLD != f), test_mask_all & (FOLD == f)) for f in range(args.folds)]}
        for li, L in enumerate(LAYERS):
            t0 = time.time()
            idx_b = np.where((B_ == b) & ((ytr_all >= 0) | (Y_ON >= 0)))[0]
            pos_in = {i: k for k, i in enumerate(idx_b)}
            Xt = {t: window(li, t, idx_b) for t in TS}

            def rows_of(mask):
                return np.array([pos_in[i] for i in np.where(mask)[0]], dtype=int)
            for scheme, splits in schemes.items():
                agg = collections.defaultdict(lambda: [0.0, 0])
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
                    R = rng.standard_normal((args.n_rand, 5120)).astype(np.float32)
                    R /= np.linalg.norm(R, axis=1, keepdims=True)
                    for t in (32, 1024):
                        ar, P = auroc_within(Xt[t][rte] @ R.T, yte, gte)
                        agg_rand[t] = ar * P if t not in agg_rand else agg_rand[t] + ar * P
                for wname, tw in TRAIN_WINDOWS.items():
                    if (wname, tw) not in agg or agg[(wname, tw)][1] == 0:
                        continue
                    P = agg[(wname, tw)][1]
                    real = agg[(wname, tw)][0] / P
                    nul = agg_null[(wname, tw)] / P if agg_null[(wname, tw)] is not None else None
                    rnd = agg_rand.get(tw)
                    rnd = rnd / P if rnd is not None else None
                    pl = [auroc_pooled(s, y)[0] for s, y in pooled[wname]]
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
                        auroc_within=float(real), auroc_pooled_mean=float(np.nanmean(pl)) if pl else np.nan,
                        shuffle_null_mean=float(nul.mean()) if nul is not None else np.nan,
                        shuffle_null_95=float(np.quantile(nul, 0.95)) if nul is not None else np.nan,
                        shuffle_pct=pct(real, nul) if nul is not None else np.nan,
                        rand_null_95=float(np.quantile(rnd, 0.95)) if rnd is not None else np.nan,
                        rand_pct=pct(real, rnd) if rnd is not None else np.nan, earliest_t=earliest))
            for train in ("disc", "all", "ho"):
                m = (B_[idx_b] == b) & (ytr_all[idx_b] >= 0)
                if train == "disc":
                    m &= SPLIT[idx_b] == "discovery"
                elif train == "ho":
                    m &= SPLIT[idx_b] == "heldout"
                r = np.where(m)[0]
                for wname, tw in TRAIN_WINDOWS.items():
                    u, dn, npr = direction(Xt[tw][r], ytr_all[idx_b[r]], PID[idx_b[r]])
                    if u is not None and train == "all" and wname in ("whole", "first32"):
                        dsave[f"null_{labels}_{train}_{wname}_L{L}"] = direction_perm(Xt[tw][r], ytr_all[idx_b[r]], PID[idx_b[r]], 20)
                    if u is not None:
                        dsave[f"{labels}_{train}_{wname}_L{L}"] = u
                        dsave[f"{labels}_{train}_{wname}_L{L}_rawnorm"] = np.float32(dn)
                        dsave[f"{labels}_{train}_{wname}_L{L}_nprompts"] = np.int32(npr)
            print(f"[{SHORT[b]}] {labels} L{L} {time.time()-t0:.0f}s "
                  f"{[(r['scheme'], r['window'], round(r['auroc_within'], 3)) for r in rows_main if r['layer'] == L and r['b'] == SHORT[b] and r['labels'] == labels]}",
                  flush=True)
    np.savez(os.path.join(args.out, "directions", f"{SHORT[b]}.npz"), **dsave)

pd.DataFrame(rows_main).to_csv(os.path.join(args.out, "a1_main.csv"), index=False)
pd.DataFrame(rows_t).to_csv(os.path.join(args.out, "a1_by_prefix.csv"), index=False)
cnt = []
for b in BEH:
    for sp in ("discovery", "heldout"):
        m = (B_ == b) & (SPLIT == sp)
        cnt.append(dict(b=SHORT[b], split=sp, prompts=len(np.unique(PID[m])), on_replies=int((m & (Y_ON >= 0)).sum()),
                        on_pos=int((m & (Y_ON == 1)).sum()), mix_replies=int((m & (Y_MIX >= 0)).sum()),
                        mix_pos=int((m & (Y_MIX == 1)).sum()),
                        prompts_both_on=int(sum(1 for p in np.unique(PID[m]) if 0 < ((PID == p) & (Y_ON == 1)).sum() < ((PID == p) & (Y_ON >= 0)).sum()))))
pd.DataFrame(cnt).to_csv(os.path.join(args.out, "a1_counts.csv"), index=False)
print(pd.DataFrame(cnt).to_string(), flush=True)

# compact per-reply window means for A2/A3 (all replies with any label)
allidx = np.arange(N)
W = {}
for wname, tw in TRAIN_WINDOWS.items():
    W[wname] = np.stack([window(li, tw, allidx) for li in range(len(LAYERS))], 1).astype(np.float16)  # [N, 6, D]
np.savez(os.path.join(args.out, "reply_means.npz"), **W,
         meta=np.array(json.dumps([{k: m[k] for k in ("key", "b", "prompt_id", "i", "split", "on", "off", "phase", "R")} for m in meta])))
print(f"[a1] ALL_DONE in {(time.time()-t00)/60:.1f} min", flush=True)
