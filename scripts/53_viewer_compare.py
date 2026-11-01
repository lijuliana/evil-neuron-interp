"""Aggregate data for the viewer's Compare methods tab. Writes <out>/compare.json (default viewer/out_public):
  within  per behavior and pooled: mean over fork replies (>= 4 cut points) of the per-reply Spearman between measured
          P(weird | prefix) and each signal at the same cut: text judge, O-lens L56 behavior score, and the Assistant
          Axis projection at block 48 with its sign flipped (lower = further from the assistant). 95% CI over replies.
  fork    the 46 sharp forks, rel = c - c_fork from -40 to +20: per-rel mean and 95% CI over forks of the text judge,
          O-lens L56 chat and plain-document scores, and the axis projection; measured P just before and just after
          the fork token from the fresh thinking-on re-estimate (24 rollouts per side).
  stop    the capping run (runs/persona_p3): behavior rate per condition with Wilson CIs, and damage in nats/token on
          capability text.
Same inputs as scripts/89 plus runs/persona_p1/b_proj.npz and runs/persona_p3/analysis.json.
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

titles = {b["id"]: b["title"] for b in json.load(open(f"{args.out}/index.json"))["behaviors"]}
json.dump({"titles": titles, "within": within, "fork": fork, "stop": stop, "axis": {"tau": 8.52, "mu": 11.13}}, open(f"{args.out}/compare.json", "w"), separators=(",", ":"))
print("within all:", within["all"])
for b, v in within.items(): print(f"  {b:30s}", {k: x[0] for k, x in v.items()}, "n", v["text"][3])
print("fork P:", fork["P"])
for col in ("text", "lens", "raw", "axis"):
    v = fork[col]
    sel = lambda lo, hi: np.mean([x[0] for r, x in zip(fork["rel"], v) if x and lo <= r <= hi])
    print(f"  {col}: -40..-21 {sel(-40,-21):.2f}  -5..0 {sel(-5,0):.2f}  +1..+5 {sel(1,5):.2f}  +6..+20 {sel(6,20):.2f}")
