"""A2: judge validity for the F2 behaviors. Haiku 5.5 + Transluce rubric vs the shipped WeirdChat
(Gemma 4 31B) labels on the Qwen3.6-27B transcripts, stratified 50/50 matched/unmatched per behavior.
Reports per-behavior agreement, precision, recall and Cohen's kappa, plus the same for the regex
pre-screen, so each card can state how much the judge itself is worth. API only.

  python -I scripts/24_a2_f2_calib.py --n-per 120
"""
import argparse, asyncio, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd
from judge import Judge, load_rubrics, load_api_key, wilson, JUDGE_MODEL
from heuristic_judge import judge_heuristic

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
F2 = ["recommends-drunk-driving", "laser-at-aircraft", "cutting-instructions"]
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(REPO, "data", "weirdchat", "data"))
ap.add_argument("--behaviors", default=",".join(F2))
ap.add_argument("--n-per", type=int, default=120)
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--out", default=os.path.join(REPO, "runs", "a2", "f2_judge_calibration"))
args = ap.parse_args()
assert load_api_key()
os.makedirs(os.path.dirname(args.out), exist_ok=True)
rubrics = load_rubrics(args.data)
judge = Judge(rubrics, concurrency=args.concurrency)

rows = []
for b in [x.strip() for x in args.behaviors.split(",")]:
    df = pd.read_parquet(f"{args.data}/transcripts/qwen__qwen3.6-27b/{b}-0000.parquet")
    df["match"] = df.judgment.apply(lambda j: bool(j["match"]))
    m, u = df[df.match], df[~df.match]
    half = args.n_per // 2
    sub = pd.concat([m.sample(min(len(m), half), random_state=0), u.sample(min(len(u), args.n_per - min(len(m), half)), random_state=0)])
    for _, r in sub.iterrows():
        msgs = [dict(x) for x in r["messages"]]
        rows.append((b, msgs, bool(r["match"]), r["transcript_id"]))
print(f"[calib] {len(rows)} transcripts: " + ", ".join(f"{b}={sum(1 for x in rows if x[0]==b)}" for b in {x[0] for x in rows}), flush=True)
res = asyncio.run(judge.judge_many([(b, m) for b, m, _, _ in rows]))

def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")

out, table = [], []
for (b, msgs, label, tid), jr in zip(rows, res):
    user = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst = msgs[-1]["content"] or ""
    if "</think>" in asst:
        asst = asst.split("</think>")[-1].lstrip("\n")
    out.append(dict(behavior=b, transcript_id=tid, shipped_label=label, haiku_match=jr.match,
                    heur_match=bool(judge_heuristic(b, user, asst)["match"]), error=jr.error,
                    explanation=jr.explanation))
dfo = pd.DataFrame(out)
dfo.to_parquet(args.out + ".parquet", index=False)
summary = {"judge_model": JUDGE_MODEL, "n": len(dfo), "usage": judge.usage, "per_behavior": {}}
for b, g in dfo.groupby("behavior"):
    ok = g.dropna(subset=["haiku_match"])
    for name, col in (("haiku", "haiku_match"), ("heuristic", "heur_match")):
        pred = ok[col].astype(bool); lab = ok.shipped_label.astype(bool)
        tp = int((pred & lab).sum()); fp = int((pred & ~lab).sum()); fn = int((~pred & lab).sum()); tn = int((~pred & ~lab).sum())
        d = dict(n=len(ok), agreement=float((pred == lab).mean()), precision=tp / max(tp + fp, 1),
                 recall=tp / max(tp + fn, 1), kappa=float(kappa(pred.values, lab.values)), tp=tp, fp=fp, fn=fn, tn=tn,
                 agreement_ci95=list(wilson(int((pred == lab).sum()), len(ok))))
        summary["per_behavior"].setdefault(b, {})[name] = d
        table.append(dict(behavior=b, judge=name, **{k: d[k] for k in ("n", "agreement", "precision", "recall", "kappa")}))
    summary["per_behavior"][b]["n_judge_errors"] = int(g.haiku_match.isna().sum())
json.dump(summary, open(args.out + ".json", "w"), indent=1, default=float)
print(pd.DataFrame(table).to_string(index=False), flush=True)
print(f"[calib] usage {judge.usage} -> {args.out}.json", flush=True)
