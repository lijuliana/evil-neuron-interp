#!/usr/bin/env python3
"""Second-opinion judge (Sonnet 5.5) on the Haiku-vs-Transluce disagreements from calibration."""
import asyncio, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
from src.judge import Judge, load_rubrics

DATA = "data/weirdchat/data"
cal = pd.read_parquet("runs/judge_calib/judge_calibration.parquet")
cal["judge_match"] = cal["judge_match"].astype("boolean")
dis = cal[(cal.judge_match.isna()) | (cal.judge_match != cal.label)].copy()
print(f"{len(dis)} disagreements/errors to adjudicate")
tr = {}
for f in glob.glob(f"{DATA}/transcripts/qwen__qwen3.6-27b/*.parquet"):
    t = pd.read_parquet(f, columns=["transcript_id", "messages"])
    tr.update(dict(zip(t.transcript_id, t.messages)))
items = [(r.behavior_id, [dict(m) for m in tr[r.transcript_id]]) for r in dis.itertuples()]
judge = Judge(load_rubrics(DATA), model="claude-sonnet-5-5", concurrency=6)
res = asyncio.run(judge.judge_many(items))
dis["sonnet_match"] = [r.match for r in res]
dis["sonnet_explanation"] = [r.explanation for r in res]
dis["sonnet_error"] = [r.error for r in res]
dis.to_parquet("runs/judge_calib/adjudication.parquet", index=False)
ok = dis.dropna(subset=["sonnet_match"])
ok = ok[ok.judge_match.notna()]
with_t = (ok.sonnet_match == ok.label).sum(); with_h = (ok.sonnet_match == ok.judge_match).sum()
summary = {"n": int(len(ok)), "sonnet_sides_with_transluce": int(with_t), "sonnet_sides_with_haiku": int(with_h),
           "usage": judge.usage,
           "per_behavior": {b: {"n": int(len(g)), "with_transluce": int((g.sonnet_match == g.label).sum())}
                            for b, g in ok.groupby("behavior_id")},
           "fn_cases": {"n": int((ok.label & (ok.judge_match == False)).sum()),
                        "sonnet_says_match": int((ok.label & (ok.judge_match == False) & (ok.sonnet_match == True)).sum())},
           "fp_cases": {"n": int((~ok.label & (ok.judge_match == True)).sum()),
                        "sonnet_says_match": int((~ok.label & (ok.judge_match == True) & (ok.sonnet_match == True)).sum())}}
json.dump(summary, open("runs/judge_calib/adjudication_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
