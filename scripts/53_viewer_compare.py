"""Aggregate data for the viewer's Compare methods tab. Writes <out>/compare.json (default viewer/out_public):
  within  per behavior and pooled: mean over fork replies (>= 4 cut points) of the per-reply Spearman between measured
          P(weird | prefix) and each signal at the same cut: text judge, O-lens L56 behavior score, and the Assistant
          Axis projection at block 48 with its sign flipped (lower = further from the assistant). 95% CI over replies.
  fork    the 46 sharp forks, rel = c - c_fork from -40 to +20: per-rel mean and 95% CI over forks of the text judge,
          O-lens L56 chat and plain-document scores, and the axis projection; measured P just before and just after
          the fork token from the fresh thinking-on re-estimate (24 rollouts per side).
  stop    the capping run (runs/persona_p3): behavior rate per condition with Wilson CIs, and damage in nats/token on
          capability text.
  dirs    weird-vs-normal directions (runs/gradual_g2, block 32, 5-fold CV, whole-reply direction): within-prompt AUROC
          when only the first t reply tokens are read, with the 95th percentile of the label-shuffled null.
  shared  shared-set ablation on the gradual behaviors (runs/gradual_g1): rate per set and its matched random set, Wilson
          CIs, pooled over the three behaviors, and damage on capability text.
  abl     ablate-and-re-read controls (scripts/90): change in the O-lens behavior score vs clean at L44/L56 for each
          condition, and per behavior at L56; 95% CI over forks.
  nlab    O-lens readouts of neuron output directions (scripts/83-87): out-of-role score of each set's neurons (max over
          the two signs) vs same-layer random neurons; 95% CI over neurons.
Same inputs as scripts/89 plus runs/persona_p1/b_proj.npz, runs/persona_p3/analysis.json,
runs/gradual_g1/{g1_rates.csv,damage.json} and runs/gradual_g2/g2_by_prefix.csv.
"""
import argparse, json
from collections import defaultdict
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="viewer/out_public")
args = ap.parse_args()
A, R = "data/agastya", "runs/olens"
rng = np.random.default_rng(0)
seqs = {s["sid"]: s for s in map(json.loads, open("data/olens/seqs.jsonl"))}
curves = {f["rid"]: dict((c, p) for c, p in f["curve"]) for f in map(json.loads, open(f"{A}/forks_off.jsonl"))}
P2 = {f["rid"]: f for f in map(json.loads, open(f"{A}/phase2_forks.jsonl"))}
text = {(r["sid"], r["c"]): r["p"] for r in map(json.loads, open(f"{R}/text_judge.jsonl"))}
proj = np.load("runs/persona_p1/b_proj.npz")
ax = lambda rid, c: float(proj[rid][c, 2, 0]) if rid in proj.files and c < proj[rid].shape[0] else np.nan  # block 48, 'axis'

sc = defaultdict(list)
for r in map(json.loads, open(f"{R}/scores.jsonl")):
    p = r["key"].split("|")
    if p[0] == "R" and p[-1] == "56":
        sc[("|".join(p[1:-2]), int(p[-2]))].append(r["behavior"])
lens = {k: float(np.mean(v)) for k, v in sc.items()}

def spear(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank()) if m.sum() > 2 else np.nan

def mean_ci(v, n=1000):
    v = np.asarray([x for x in v if not np.isnan(x)])
    return [round(float(v.mean()), 3), *[round(float(x), 3) for x in np.percentile([rng.choice(v, len(v)).mean() for _ in range(n)], [2.5, 97.5])], len(v)]

# within reply: same cut points as scripts/89 section 1 (c = 0 is the last prompt token)
rows = []
for rid, cur in curves.items():
    pid = rid.split("|")[0]
    b = seqs[f"prompt|{pid}"]["b"]
    for c, P in cur.items():
        sid = f"reply|{rid}" if c >= 1 else f"prompt|{pid}"
        if (sid, c) in lens:
            rows.append({"rid": rid, "b": b, "c": c, "P": P, "text": text.get((sid, c), np.nan), "lens": lens[(sid, c)], "axis": -ax(rid, c)})
C = pd.DataFrame(rows)
per = {col: {rid: spear(g[col].values, g.P.values) for rid, g in C.groupby("rid") if len(g) >= 4} for col in ("text", "lens", "axis")}
beh = C.drop_duplicates("rid").set_index("rid").b
within = {"all": {col: mean_ci(list(v.values())) for col, v in per.items()}}
for b in sorted(beh.unique()):
    within[b] = {col: mean_ci([x for rid, x in v.items() if beh[rid] == b]) for col, v in per.items()}

# around the 46 sharp forks
fr = []
for rid, f in P2.items():
    c0, pid = f["fork"][0], rid.split("|")[0]
    for rel in range(-40, 21):
        c = c0 + rel
        if c < 0:
            continue
        sid = f"reply|{rid}" if c >= 1 else f"prompt|{pid}"
        fr.append({"rid": rid, "rel": rel, "text": text.get((sid, c), np.nan), "lens": lens.get((sid, c), np.nan),
                   "raw": lens.get((f"raw|{rid}", c), np.nan) if c >= 1 else np.nan, "axis": ax(rid, c)})
Fd = pd.DataFrame(fr)
fork = {"rel": list(range(-40, 21)), "n": len(P2),
        "P": {"before": mean_ci([f["Plo_fresh"] for f in P2.values()]), "after": mean_ci([f["Phi_fresh"] for f in P2.values()])}}
for col in ("text", "lens", "raw", "axis"):
    fork[col] = [mean_ci(Fd[Fd.rel == rel][col].values, 400) if Fd[Fd.rel == rel][col].notna().sum() >= 5 else None for rel in fork["rel"]]

# capping run
p3 = json.load(open("runs/persona_p3/analysis.json"))
stop = {"behaviors": {}, "damage": {k: round(v["dnll_capability"], 4) for k, v in p3["damage"].items()}}
for b, d in p3["behaviors"].items():
    stop["behaviors"][b] = {"prompts": d["n_prompts_used"], "rates": {k: [round(v["rate"], 3), round(v["lo"], 3), round(v["hi"], 3), v["n_judged"]] for k, v in d["rates"].items()}}

r3 = lambda v: round(float(v), 3)
def boot_by(df, fn, by, n=1000):
    """fn(df) with a 95% CI from resampling the groups in column `by`."""
    groups = df[by].unique()
    idx = {g: np.where(df[by].values == g)[0] for g in groups}
    vals = [fn(df.iloc[np.concatenate([idx[g] for g in rng.choice(groups, len(groups))])]) for _ in range(n)]
    return [r3(fn(df)), *[r3(x) for x in np.nanpercentile(vals, [2.5, 97.5])], int(len(groups))]

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [r3(p), r3(c - h), r3(c + h), int(n)]

# weird-vs-normal directions: AUROC by how much of the reply is read
g2 = pd.read_csv("runs/gradual_g2/g2_by_prefix.csv")
g2 = g2[(g2.labels == "on16") & (g2.scheme == "cv") & (g2.layer == 32) & (g2.train_window == "whole")]
cnt = pd.read_csv("runs/gradual_g2/g2_counts.csv").groupby("b")[["prompts", "on_replies"]].sum()
dirs = {b: {"t": d.t.tolist(), "auc": [r3(v) for v in d.auroc], "null": [r3(v) for v in d.null95], "pairs": int(d.pairs.iloc[0]),
            "prompts": int(cnt.loc[b, "prompts"]), "replies": int(cnt.loc[b, "on_replies"])} for b, d in g2.groupby("b")}

# shared sets on the gradual behaviors
g1 = pd.read_csv("runs/gradual_g1/g1_rates.csv")
dm = json.load(open("runs/gradual_g1/damage.json"))
shared = {"damage": {k: round(float(v["capability_mini"]["delta_nll_per_token"]), 4) for k, v in dm.items()}, "behaviors": {},
          "ids": {"drunk": "recommends-drunk-driving", "chemtrails": "chemtrails-assertion", "cutting": "cutting-instructions",
                  "laser": "laser-at-aircraft"}}
for b, d in g1.groupby("b"):
    shared["behaviors"][b] = {"prompts": int(d.n.iloc[0] // 4), **{c: wilson(int(r.k), int(r.n)) for c, r in d.set_index("cond").iterrows()}}
shared["behaviors"]["pooled"] = {"prompts": sum(v["prompts"] for v in shared["behaviors"].values()),
                                 **{c: wilson(int(d.k.sum()), int(d.n.sum())) for c, d in g1.groupby("cond")}}

# ablate and re-read with size-matched controls
cs = defaultdict(list)
for r in map(json.loads, open(f"{R}/abl_control_scores.jsonl")):
    cs[r["key"]].append(r["behavior"])
K = pd.DataFrame([{"cond": k.split("|")[1], "rid": "|".join(k.split("|")[2:-2]), "c": int(k.split("|")[-2]),
                   "L": int(k.split("|")[-1]), "lb": np.mean(v)} for k, v in cs.items()])
K["b"] = K.rid.map(lambda r: seqs[f"reply|{r}"]["b"])
abl = {"all": {}, "by": {}}
for L, dL in K.groupby("L"):
    w = dL.pivot_table(index=["b", "rid", "c"], columns="cond", values="lb").reset_index()
    abl["positions"] = int(len(w))
    for c in ("top", "amp", "rand_s", "gauss", "rand"):
        x = w.assign(d=w[c] - w.clean).dropna(subset=["d"])
        abl["all"].setdefault(c, {})[L] = boot_by(x, lambda q: q.d.mean(), "rid")
        if L == 56 and c in ("top", "amp"):
            for b, xb in x.groupby("b"):
                abl["by"].setdefault(b, {})[c] = boot_by(xb, lambda q: q.d.mean(), "rid")

# O-lens readouts of neuron output directions
nr = defaultdict(list)
for r in map(json.loads, open(f"{R}/scores.jsonl")):
    if r["key"].startswith("N|"):
        nr[r["key"]].append(r["weird"])
N = pd.DataFrame([{"L": int(k.split("|")[1]), "n": int(k.split("|")[2]), "src": "|".join(k.split("|")[3:-1]),
                   "lw": np.mean(v)} for k, v in nr.items()])
N["kind"] = np.where(N.src.str.startswith("rand|"), "random", "set")
N["set"] = N.src.str.replace("rand|", "", regex=False)
best = N.groupby(["L", "n", "src", "kind", "set"], as_index=False).lw.max()
nlab = {}
for s_, d in best.groupby("set"):
    a, r = d[d.kind == "set"].lw.values, d[d.kind == "random"].lw.values
    bt = lambda v: [r3(v.mean()), *[r3(x) for x in np.percentile([rng.choice(v, len(v)).mean() for _ in range(1000)], [2.5, 97.5])], len(v)]
    dd = np.percentile([rng.choice(a, len(a)).mean() - rng.choice(r, len(r)).mean() for _ in range(1000)], [2.5, 97.5])
    nlab[s_.replace("top10|", "")] = {"set": bt(a), "rand": bt(r), "diff": [r3(a.mean() - r.mean()), r3(dd[0]), r3(dd[1])]}

titles = {b["id"]: b["title"] for b in json.load(open(f"{args.out}/index.json"))["behaviors"]}
json.dump({"titles": titles, "within": within, "fork": fork, "stop": stop, "axis": {"tau": 8.52, "mu": 11.13},
           "dirs": dirs, "shared": shared, "abl": abl, "nlab": nlab},
          open(f"{args.out}/compare.json", "w"), separators=(",", ":"))
print("within all:", within["all"])
for b, v in within.items(): print(f"  {b:30s}", {k: x[0] for k, x in v.items()}, "n", v["text"][3])
print("fork P:", fork["P"])
for col in ("text", "lens", "raw", "axis"):
    v = fork[col]
    sel = lambda lo, hi: np.mean([x[0] for r, x in zip(fork["rel"], v) if x and lo <= r <= hi])
    print(f"  {col}: -40..-21 {sel(-40,-21):.2f}  -5..0 {sel(-5,0):.2f}  +1..+5 {sel(1,5):.2f}  +6..+20 {sel(6,20):.2f}")
print("dirs t=8 / 128 / 1024:", {b: (d["auc"][3], d["auc"][7], d["auc"][-1]) for b, d in dirs.items()})
print("shared pooled:", {k: v for k, v in shared["behaviors"]["pooled"].items()}, "damage", shared["damage"])
print("abl:", {c: {L: v[0] for L, v in d.items()} for c, d in abl["all"].items()}, "positions", abl["positions"])
print("abl by:", {b: {c: v[0] for c, v in d.items()} for b, d in abl["by"].items()})
print("nlab:", {k: (v["set"][0], v["rand"][0], v["diff"]) for k, v in nlab.items()})
