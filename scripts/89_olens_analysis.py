"""O-lens W4 analysis (PLAN.md decisions 1-4, 11). Inputs: data/olens/seqs.jsonl, runs/olens/scores.jsonl (scripts/87),
runs/olens/text_judge.jsonl (scripts/88), Agastya's curves, Phase 2 forks and prompt rates. Lens score at a position =
mean over its 2 readouts of the 0-10 'behavior' (or 'weird') score. Writes runs/olens/analysis.md and
runs/olens/fork_timecourse.png.
  1 cut points: Spearman with measured P(weird | prefix), pooled and within reply, lens per layer vs the text judge;
    does the lens add to the text judge (OLS, 5-fold cross-validation grouped by prompt); the subset where the text judge
    sees nothing (p <= 0.1). All 410 fork replies come from discovery prompts, so there is no held-out split here.
  2 sharp forks: lens and text judge vs tokens to the fork, chat vs raw-document re-read (echo control), per-fork rise
  3 ablate-and-re-read: own top-100 vs matched random-100 vs clean, fork window; plus scripts/90's size-matched controls
    when runs/olens/abl_control_scores.jsonl exists
  4 prompt level: last prompt token vs the prompt's rate over 64 samples; lens added to the text judge, fit on discovery
    prompts and tested on held-out ones
  5 neuron labels: own top-10 / sharedmid100 directions vs same-layer random neurons
Confidence intervals: 1,000 bootstrap resamples of replies (or prompts / neurons), 95% percentile.
"""
import gzip, json, os
from collections import defaultdict
import numpy as np
import pandas as pd

A, R = "data/agastya", "runs/olens"
rng = np.random.default_rng(0)
LAYERS = [32, 44, 56]
seqs = {s["sid"]: s for s in map(json.loads, open("data/olens/seqs.jsonl"))}
curves = {f["rid"]: dict((c, p) for c, p in f["curve"]) for f in map(json.loads, open(f"{A}/forks_off.jsonl"))}
c0 = {f["rid"]: f["fork"][0] for f in map(json.loads, open(f"{A}/phase2_forks.jsonl"))}
rate = {r["prompt_id"]: r["rate"] for r in json.load(open(f"{A}/stage_a_rates_off.json"))}
text = {(r["sid"], r["c"]): r["p"] for r in map(json.loads, open(f"{R}/text_judge.jsonl"))}

sc = defaultdict(lambda: [[], []])
for r in map(json.loads, open(f"{R}/scores.jsonl")):
    sc[r["key"]][0].append(r["behavior"])
    sc[r["key"]][1].append(r["weird"])
rows, nrows = [], []
for k, (bs, ws) in sc.items():
    p = k.split("|")
    if p[0] == "R":
        sid, c, L = "|".join(p[1:-2]), int(p[-2]), int(p[-1])
        s = seqs[sid]
        rows.append({"sid": sid, "kind": s["kind"], "pid": s["pid"], "rid": s["rid"], "b": s["b"], "split": s["split"],
                     "c": c, "L": L, "lb": np.mean(bs), "lw": np.mean(ws)})
    else:
        nrows.append({"L": int(p[1]), "n": int(p[2]), "src": "|".join(p[3:-1]), "sign": p[-1], "lb": np.mean(bs), "lw": np.mean(ws)})
D = pd.DataFrame(rows)
LAYERS = [L for L in LAYERS if L in set(D.L)]  # partial runs: only the layers scored so far
lens = D.pivot_table(index=["sid", "c"], columns="L", values=["lb", "lw"])
lens.columns = [f"{v}{L}" for v, L in lens.columns]
meta = D.drop_duplicates(["sid", "c"]).set_index(["sid", "c"])[["kind", "pid", "rid", "b", "split"]]
X = meta.join(lens)

def spear(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()) if m.sum() > 2 else np.nan

def boot(df, fn, by):
    groups = df[by].unique()
    if len(groups) < 2:
        return np.array([np.nan, np.nan])
    idx = {g: np.where(df[by].values == g)[0] for g in groups}
    vals = []
    for _ in range(1000):
        pick = rng.choice(groups, len(groups))
        vals.append(fn(df.iloc[np.concatenate([idx[g] for g in pick])]))
    return np.nanpercentile(vals, [2.5, 97.5])

fmt = lambda v, ci: f"{v:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
out = ["# O-lens readouts vs measured weirdness", "",
       f"{len(D)} scored positions x layers, {len(nrows)} neuron directions. Scores are 0-10 blind Haiku ratings of the readouts.", ""]

# 1 cut points: rows (rid, c) with measured P; c = 0 uses the prompt's own readouts and text-judge value
cut = []
for rid, cur in curves.items():
    pid = rid.split("|")[0]
    for c, P in cur.items():
        sid = f"reply|{rid}" if c >= 1 else f"prompt|{pid}"
        if (sid, c) in X.index:
            cut.append({**X.loc[(sid, c)].to_dict(), "rid": rid, "c": c, "P": P, "text": text.get((sid, c), np.nan)})
C = pd.DataFrame(cut)
out += ["## 1. Cut points: Spearman with P(weird | prefix)", "",
        f"{len(C)} cut points on {C.rid.nunique()} fork replies (all 410 replies turned weird). 95% CIs bootstrap replies.", "",
        "Within reply = mean over replies (>= 4 cut points) of the per-reply Spearman, so it ignores which replies or "
        "behaviors run hotter overall.", "",
        "| predictor | pooled | within reply | pooled, text judge p <= 0.1 |", "|---|---|---|---|"]
blind = C[C.text <= 0.1]

def within(d, col):
    v = np.array([spear(g[col].values, g.P.values) for _, g in d.groupby("rid") if len(g) >= 4])
    v = v[~np.isnan(v)]
    return fmt(v.mean(), np.percentile([rng.choice(v, len(v)).mean() for _ in range(1000)], [2.5, 97.5]))

for name, col in [("text judge", "text")] + [(f"lens behavior L{L}", f"lb{L}") for L in LAYERS] + [(f"lens weird L{L}", f"lw{L}") for L in LAYERS]:
    f = lambda d, col=col: spear(d[col].values, d.P.values)
    out.append(f"| {name} | {fmt(f(C), boot(C, f, 'rid'))} | {within(C, col)} | {fmt(f(blind), boot(blind, f, 'rid'))} |")
out += ["", f"Text-blind subset: {len(blind)} cut points where the text judge gives <= 10%; mean measured P there "
        f"{blind.P.mean():.2f}.", ""]

def fit_predict(tr, te, cols, y):
    w = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[cols].values], tr[y].values, rcond=None)[0]
    return np.c_[np.ones(len(te)), te[cols].values] @ w

def oof(d, cols, k=5):  # out-of-fold OLS predictions of P, folds grouped by prompt
    pids = list(d.pid.unique())
    np.random.default_rng(1).shuffle(pids)
    fold = d.pid.map({p: i % k for i, p in enumerate(pids)}).values
    pred = np.empty(len(d))
    for i in range(k):
        pred[fold == i] = fit_predict(d[fold != i], d[fold == i], cols, "P")
    return pred

feats = [f"lb{L}" for L in LAYERS] + [f"lw{L}" for L in LAYERS]
Cc = C.dropna(subset=["text"] + feats).copy()
Cc["base"], Cc["full"] = oof(Cc, ["text"]), oof(Cc, ["text"] + feats)
gain = lambda d: spear(d.full.values, d.P.values) - spear(d.base.values, d.P.values)
out += ["### Does the lens add to the text judge?", "",
        f"OLS of P on the text judge, with and without the {len(feats)} lens scores, 5-fold cross-validation grouped by "
        f"prompt ({len(Cc)} cut points). Out-of-fold Spearman {spear(Cc.base.values, Cc.P.values):+.2f} (text judge) vs "
        f"{spear(Cc.full.values, Cc.P.values):+.2f} (text judge + lens); change {fmt(gain(Cc), boot(Cc, gain, 'rid'))}.", ""]
out += ["### By behavior (all splits, lens L56 behavior score)", "", "| behavior | cut points | text judge | lens L56 |", "|---|---|---|---|"]
for b, d in C.groupby("b"):
    out.append(f"| {b} | {len(d)} | {spear(d.text.values, d.P.values):+.2f} | {spear(d.lb56.values, d.P.values):+.2f} |")

# 2 sharp forks: time course relative to the fork token, chat vs raw
F = X.reset_index()
F = F[F.rid.isin(c0.keys()) & F.kind.isin(["reply", "raw", "prompt"])].copy()
F_p = F[F.kind == "prompt"]
F = F[F.kind != "prompt"]
# c = 0 of the chat window is the prompt's own readout
for rid in c0:
    pid = rid.split("|")[0]
    p = F_p[F_p.pid == pid]
    if len(p):
        F = pd.concat([F, p.assign(rid=rid, kind="reply")])
F["rel"] = F.c - F.rid.map(c0)
F["text"] = [text.get((f"reply|{r}" if c >= 1 else f"prompt|{r.split('|')[0]}", c), np.nan) for r, c in zip(F.rid, F.c)]
bins = [(-40, -21), (-20, -6), (-5, -1), (0, 0), (1, 5), (6, 20)]
out += ["", "## 2. The 46 sharp forks: tokens before and after the fork token", "",
        "rel = c - c_fork; rel 0 is the state that picks the fork token (nothing of it written yet). Raw = the same user text "
        "and reply tokens as a plain document, no chat template (echo control).", "",
        "| rel | n | text judge | " + " | ".join(f"lens L{L} chat | raw" for L in LAYERS) + " |", "|---|---|---|" + "---|---|" * len(LAYERS)]
for lo, hi in bins:
    ch = F[(F.kind == "reply") & F.rel.between(lo, hi)]
    rw = F[(F.kind == "raw") & F.rel.between(lo, hi)]
    out.append(f"| {lo}..{hi} | {len(ch)} | {ch.text.mean():.2f} | " +
               " | ".join(f"{ch[f'lb{L}'].mean():.2f} | {rw[f'lb{L}'].mean():.2f}" for L in LAYERS) + " |")
P2 = [json.loads(l) for l in open(f"{A}/phase2_forks.jsonl")]
out += ["", f"Measured P(weird) on the same forks: {np.mean([f['Plo_on'] for f in P2]):.2f} just before the fork token "
        f"(rel <= 0), {np.mean([f['Phi_on'] for f in P2]):.2f} just after it (Plo_on / Phi_on).", "",
        "Token by token near the fork (behavior score):", "",
        "| rel | text judge | " + " | ".join(f"L{L} chat | L{L} raw" for L in LAYERS) + " |", "|---|---|" + "---|---|" * len(LAYERS)]
for rel in range(-8, 4):
    ch, rw = F[(F.kind == "reply") & (F.rel == rel)], F[(F.kind == "raw") & (F.rel == rel)]
    out.append(f"| {rel:+d} | {ch.text.mean():.2f} | " + " | ".join(f"{ch[f'lb{L}'].mean():.2f} | {rw[f'lb{L}'].mean():.2f}" for L in LAYERS) + " |")

def rise(kind, col):  # per fork: mean over rel -5..0 minus mean over rel -40..-21
    d = F[F.kind == kind]
    r = (d[d.rel.between(-5, 0)].groupby("rid")[col].mean() - d[d.rel.between(-40, -21)].groupby("rid")[col].mean()).dropna()
    return fmt(r.mean(), np.percentile([rng.choice(r.values, len(r)).mean() for _ in range(1000)], [2.5, 97.5])), len(r)
out += ["", "Rise before the fork token, per fork: mean over rel -5..0 minus mean over rel -40..-21 (95% CI over forks).", "",
        "| signal | forks | rise |", "|---|---|---|"]
v, n = rise("reply", "text")
out.append(f"| text judge (0-1) | {n} | {v} |")
for L in LAYERS:
    for kind, lab in (("reply", "chat"), ("raw", "raw")):
        v, n = rise(kind, f"lb{L}")
        out.append(f"| lens L{L} {lab} (0-10) | {n} | {v} |")
pre = F[(F.kind == "reply") & F.rel.between(-5, 0)].merge(F[(F.kind == "raw") & F.rel.between(-5, 0)], on=["rid", "c"], suffixes=("", "_raw"))
for L in LAYERS:
    d = (pre[f"lb{L}"] - pre[f"lb{L}_raw"])
    out.append(f"\nChat minus raw, rel -5..0, L{L}: {fmt(d.mean(), boot(pre.assign(d=d), lambda x: x.d.mean(), 'rid'))}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W2 = F[F.rel.between(-40, 20)]
    fig, ax = plt.subplots(1, len(LAYERS), figsize=(4 * len(LAYERS), 3.2), sharey=True, squeeze=False)
    t = W2[W2.kind == "reply"].groupby("rel").text.mean() * 10
    for a, L in zip(ax[0], LAYERS):
        for kind, col, lab in (("reply", "#1f3f8f", "lens, chat"), ("raw", "#8fa7d8", "lens, raw document")):
            m = W2[W2.kind == kind].groupby("rel")[f"lb{L}"].mean()
            a.plot(m.index, m.values, color=col, label=lab)
        a.plot(t.index, t.values, color="#c9472b", ls="--", label="text judge x10")
        a.axvline(0, color="#999", lw=0.8)
        a.set_title(f"block {L}", fontsize=9)
        a.set_xlabel("tokens relative to the fork token")
    ax[0][0].set_ylabel("behavior score (0-10)")
    ax[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{R}/fork_timecourse.png", dpi=150)
except Exception as e:  # plotting is optional
    out.append(f"\n(plot skipped: {e})")

# 3 ablate-and-re-read
W = X.reset_index()
W = W[W.kind.isin(["abl_top", "abl_rand"])]
clean = X.reset_index()
clean = clean[clean.kind.isin(["reply", "prompt"])]
out += ["", "## 3. Ablate and re-read (own top-100 vs matched random-100), window c_fork-10 .. c_fork+5", "",
        "| layer | clean | random-100 ablated | top-100 ablated | top minus random |", "|---|---|---|---|---|"]
for L in LAYERS:
    t = W[W.kind == "abl_top"][["rid", "c", f"lb{L}"]]
    r = W[W.kind == "abl_rand"][["rid", "c", f"lb{L}"]]
    pr = t.merge(r, on=["rid", "c"], suffixes=("_t", "_r"))
    cl = []
    for rid, c in zip(pr.rid, pr.c):
        k = (f"reply|{rid}", c) if c >= 1 else (f"prompt|{rid.split('|')[0]}", 0)
        cl.append(X.loc[k, f"lb{L}"] if k in X.index else np.nan)
    pr["d"] = pr[f"lb{L}_t"] - pr[f"lb{L}_r"]
    out.append(f"| L{L} | {np.nanmean(cl):.2f} | {pr[f'lb{L}_r'].mean():.2f} | {pr[f'lb{L}_t'].mean():.2f} | "
               f"{fmt(pr.d.mean(), boot(pr, lambda x: x.d.mean(), 'rid'))} |")

try:
    import torch
    d = torch.load("data/olens/resid.pt", weights_only=False)
    ix = {tuple(k): i for i, k in enumerate(d["keys"])}
    disp = defaultdict(list)
    for (sid, c), i in ix.items():
        kind, rid = sid.split("|", 1)
        ck = (f"reply|{rid}", c) if c >= 1 else (f"prompt|{rid.split('|')[0]}", 0)
        if kind in ("abl_top", "abl_rand") and ck in ix:
            for j, L in enumerate(d["layers"]):
                h = d["H"][ix[ck], j].float()
                disp[(kind, L)].append(((d["H"][i, j].float() - h).norm() / h.norm()).item())
    out += ["", "Median relative displacement |h_ablated - h_clean| / |h_clean|: " + "; ".join(
        f"L{L} top {np.median(disp[('abl_top', L)]):.3f}, random {np.median(disp[('abl_rand', L)]):.3f}" for L in LAYERS)]
except Exception as e:
    out.append(f"\n(displacement skipped: {e})")

if os.path.exists(f"{R}/abl_control_scores.jsonl"):
    cs = defaultdict(list)
    for r in map(json.loads, open(f"{R}/abl_control_scores.jsonl")):
        cs[r["key"]].append(r["behavior"])
    K = pd.DataFrame([{"cond": k.split("|")[1], "rid": "|".join(k.split("|")[2:-2]), "c": int(k.split("|")[-2]),
                       "L": int(k.split("|")[-1]), "lb": np.mean(v)} for k, v in cs.items()])
    conds = ["clean", "rand", "top", "amp", "rand_s", "gauss"]
    out += ["", "### Size-matched controls (scripts/90), all six conditions scored together in one blind pool", "",
            "amp = h_clean - d_top (the same neurons pushed the other way); rand_s = the random-100 ablation direction scaled "
            "to |d_top|; gauss = a random direction at |d_top|. Differences are vs clean, paired by position, 95% CI over forks.", "",
            "| layer | " + " | ".join(conds) + " | top - clean | amp - clean | rand_s - clean | gauss - clean |",
            "|---|" + "---|" * (len(conds) + 4)]
    for L, dL in K.groupby("L"):
        w = dL.pivot_table(index=["rid", "c"], columns="cond", values="lb").reset_index()
        cells = [f"{w[c].mean():.2f}" for c in conds]
        for c in ("top", "amp", "rand_s", "gauss"):
            x = w.assign(d=w[c] - w.clean).dropna(subset=["d"])
            cells.append(fmt(x.d.mean(), boot(x, lambda q: q.d.mean(), "rid")))
        out.append(f"| L{L} | " + " | ".join(cells) + " |")
    K["b"] = K.rid.map(lambda r: seqs[f"reply|{r}"]["b"])
    w = K[K.L == 56].pivot_table(index=["b", "rid", "c"], columns="cond", values="lb").reset_index()
    out += ["", "By behavior, L56 (forks): top - clean | amp - clean", ""] + [
        f"- {b} ({g.rid.nunique()}): {(g.top - g.clean).mean():+.2f} | {(g.amp - g.clean).mean():+.2f}" for b, g in w.groupby("b")]

# 4 prompt level
Pq = X.reset_index()
Pq = Pq[Pq.kind == "prompt"].copy()
Pq["rate"] = Pq.pid.map(rate)
Pq["text"] = [text.get((s, 0), np.nan) for s in Pq.sid]
out += ["", "## 4. Prompt level: last prompt token vs the prompt's rate (64 samples)", "",
        f"{len(Pq)} prompts.", "", "| predictor | all | held-out |", "|---|---|---|"]
for name, col in [("text judge", "text")] + [(f"lens behavior L{L}", f"lb{L}") for L in LAYERS]:
    f = lambda d, col=col: spear(d[col].values, d.rate.values)
    out.append(f"| {name} | {fmt(f(Pq), boot(Pq, f, 'pid'))} | {f(Pq[Pq.split == 'heldout']):+.2f} |")
Pc = Pq.dropna(subset=["text"] + [f"lb{L}" for L in LAYERS])
trp, tep = Pc[Pc.split == "discovery"], Pc[Pc.split == "heldout"].copy()
tep["base"] = fit_predict(trp, tep, ["text"], "rate")
tep["full"] = fit_predict(trp, tep, ["text"] + [f"lb{L}" for L in LAYERS], "rate")
g4 = lambda d: spear(d.full.values, d.rate.values) - spear(d.base.values, d.rate.values)
out += ["", f"OLS of the rate on the text judge, with and without the lens scores, fit on {len(trp)} discovery prompts and "
        f"tested on {len(tep)} held-out: Spearman {spear(tep.base.values, tep.rate.values):+.2f} -> "
        f"{spear(tep.full.values, tep.rate.values):+.2f}, change {fmt(g4(tep), boot(tep, g4, 'pid'))}."]

# 5 neuron labels
N = pd.DataFrame(nrows)
if len(N):
    N["kind"] = np.where(N.src.str.startswith("rand|"), "random", "set")
    N["set"] = N.src.str.replace("rand|", "", regex=False)
    best = N.groupby(["L", "n", "src", "kind", "set"], as_index=False)[["lb", "lw"]].max()
    out += ["", "## 5. Neuron output directions (max over +w / -w)", "",
            "| set | neurons | behavior score | random same-layer | weird score | random | weird, set - random |",
            "|---|---|---|---|---|---|---|"]
    for s, d in best.groupby("set"):
        a, r = d[d.kind == "set"], d[d.kind == "random"]
        dw = np.percentile([rng.choice(a.lw.values, len(a)).mean() - rng.choice(r.lw.values, len(r)).mean() for _ in range(1000)], [2.5, 97.5])
        out.append(f"| {s} | {len(a)} | {a.lb.mean():.2f} | {r.lb.mean():.2f} | {a.lw.mean():.2f} | {r.lw.mean():.2f} | "
                   f"{fmt(a.lw.mean() - r.lw.mean(), dw)} |")

open(f"{R}/analysis.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
