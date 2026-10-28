"""A2: adjudicate judge disputes. For transcripts where Haiku 5.5 (Transluce rubric) and the shipped
WeirdChat (Gemma 4 31B) label disagree, re-judge with (a) Haiku at effort=high and (b) Sonnet 5.5
(PROGRESS.md: "Sonnet 5.5 for calibration disputes"). Tells us whose call to trust and whether to run
production judging at a higher effort. API only.
  python -I scripts/25_a2_judge_disputes.py --n 60
"""
import argparse, asyncio, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pandas as pd
from judge import Judge, load_rubrics, load_api_key

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(REPO, "data", "weirdchat", "data"))
ap.add_argument("--calib", default=os.path.join(REPO, "runs", "a2", "f2_judge_calibration.parquet"))
ap.add_argument("--n", type=int, default=60, help="disputes per behavior (stratified by direction)")
ap.add_argument("--out", default=os.path.join(REPO, "runs", "a2", "f2_judge_disputes"))
args = ap.parse_args()
assert load_api_key()
rubrics = load_rubrics(args.data)
cal = pd.read_parquet(args.calib).dropna(subset=["haiku_match"])
cal["dispute"] = cal.shipped_label.astype(bool) != cal.haiku_match.astype(bool)

tx = {}
for b in cal.behavior.unique():
    df = pd.read_parquet(f"{args.data}/transcripts/qwen__qwen3.6-27b/{b}-0000.parquet")
    tx[b] = df.set_index("transcript_id")

rows = []
for b, g in cal[cal.dispute].groupby("behavior"):
    for direction, gg in g.groupby(g.shipped_label.astype(bool)):
        take = gg.sample(min(len(gg), max(1, args.n // 2)), random_state=0)
        for r in take.itertuples():
            msgs = [dict(x) for x in tx[b].loc[r.transcript_id]["messages"]]
            rows.append(dict(behavior=b, transcript_id=r.transcript_id, shipped_label=bool(r.shipped_label),
                             haiku_match=bool(r.haiku_match), messages=msgs))
print(f"[disputes] {len(rows)} disputed transcripts: " +
      ", ".join(f"{b}={sum(1 for x in rows if x['behavior']==b)}" for b in {x['behavior'] for x in rows}), flush=True)

items = [(r["behavior"], r["messages"]) for r in rows]
res = {}
for name, model, effort in (("haiku_high", "claude-haiku-5-5", "high"), ("sonnet", "claude-sonnet-5-5", None)):
    j = Judge(rubrics, model=model, concurrency=12, effort=effort)
    res[name] = asyncio.run(j.judge_many(items))
    print(f"[disputes] {name} done, usage {j.usage}", flush=True)

out = []
for r, hh, so in zip(rows, res["haiku_high"], res["sonnet"]):
    out.append(dict(behavior=r["behavior"], transcript_id=r["transcript_id"], shipped_label=r["shipped_label"],
                    haiku_default=r["haiku_match"], haiku_high=hh.match, sonnet=so.match,
                    sonnet_explanation=so.explanation, haiku_high_explanation=hh.explanation))
d = pd.DataFrame(out)
d.to_parquet(args.out + ".parquet", index=False)
summary = {"n_disputes": len(d), "per_behavior": {}}
for b, g in d.groupby("behavior"):
    ok = g.dropna(subset=["sonnet"])
    okh = g.dropna(subset=["haiku_high"])
    summary["per_behavior"][b] = dict(
        n=len(g), n_sonnet_judged=len(ok),
        sonnet_sides_with_haiku=float((ok.sonnet.astype(bool) == ok.haiku_default.astype(bool)).mean()),
        sonnet_sides_with_shipped=float((ok.sonnet.astype(bool) == ok.shipped_label.astype(bool)).mean()),
        haiku_high_agrees_with_haiku_default=float((okh.haiku_high.astype(bool) == okh.haiku_default.astype(bool)).mean()),
        haiku_high_sides_with_shipped=float((okh.haiku_high.astype(bool) == okh.shipped_label.astype(bool)).mean()),
        by_direction={str(k): dict(n=len(gg), sonnet_with_haiku=float((gg.dropna(subset=["sonnet"]).sonnet.astype(bool) ==
                      gg.dropna(subset=["sonnet"]).haiku_default.astype(bool)).mean()))
                      for k, gg in g.groupby(g.shipped_label)})
json.dump(summary, open(args.out + ".json", "w"), indent=1, default=float)
print(json.dumps(summary, indent=1), flush=True)
for r in out[:4]:
    print(f"\n[{r['behavior']}] shipped={r['shipped_label']} haiku={r['haiku_default']} haiku_high={r['haiku_high']} sonnet={r['sonnet']}\n  SONNET: {r['sonnet_explanation'][:400]}", flush=True)
