"""07_baseline_judge.py -- judge the bf16 27B baseline (runs/baseline_27b/*.jsonl) at a fraction of the
cost of judging all 64 rollouts: Haiku 5.5 on sample_idx < --per-prompt (default 16 -> ~2.9k judgments),
regex pre-screen (src/heuristic_judge.py) on every rollout for free.  Writes <behavior>.judged.jsonl
(all rows; judge_match is None where unjudged) and baseline_rates.csv comparing to the FP8 WeirdChat
rates on the same prompts and across all patterns.

  python -I scripts/07_baseline_judge.py [--per-prompt 16] [--dry-run]
"""
import argparse, asyncio, glob, json, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from judge import Judge, load_api_key, load_rubrics, wilson  # noqa: E402
from heuristic_judge import judge_heuristic  # noqa: E402
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--samples", default=os.path.join(ROOT, "runs", "baseline_27b"))
ap.add_argument("--data", default=os.path.join(ROOT, "data", "weirdchat", "data"))
ap.add_argument("--per-prompt", type=int, default=16)
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
if not a.dry_run and load_api_key() is None:
    sys.exit("no ANTHROPIC_API_KEY")
rubrics = load_rubrics(a.data)
pa = pd.read_parquet(os.path.join(a.data, "patterns.parquet"))
pa = pa[pa.subject_model == "qwen/qwen3.6-27b"]
judge = Judge(rubrics, dry_run=a.dry_run, concurrency=a.concurrency)
table = []


async def main():
  for f in sorted(glob.glob(os.path.join(a.samples, "*.jsonl"))):
      if f.endswith(".judged.jsonl"):
          continue
      behavior = os.path.basename(f)[:-6]
      recs = [json.loads(l) for l in open(f)]
      for r in recs:
          h = judge_heuristic(behavior, r["prompt"], r["response"])
          r["heur_match"], r["heur_explanation"] = h["match"], h["explanation"]
          r["judge_match"] = r["judge_error"] = r["judge_explanation"] = None
      sel = [r for r in recs if r["sample_idx"] < a.per_prompt]
      items = [(behavior, [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]) for r in sel]
      print(f"[{behavior}] judging {len(items)} of {len(recs)}", flush=True)
      res = await judge.judge_many(items)
      for r, j in zip(sel, res):
          r["judge_match"], r["judge_error"], r["judge_explanation"] = j.match, j.error, j.explanation
      with open(os.path.join(a.samples, f"{behavior}.judged.jsonl"), "w") as fo:
          for r in recs:
              fo.write(json.dumps(r) + "\n")
      judged = [r for r in recs if r["judge_match"] is not None]
      k = sum(bool(r["judge_match"]) for r in judged); n = len(judged)
      lo, hi = wilson(k, n)
      hk = sum(bool(r["heur_match"]) for r in recs)
      fp8 = pa[pa.behavior_id == behavior]
      fp8_rate = float(sum(m["matched_samples"] for m in fp8.metrics) / max(sum(m["total_samples"] for m in fp8.metrics), 1))
      byp = {r["prompt_id"]: r for r in recs}
      ref_k = sum(r.get("ref_n_matched", 0) for r in byp.values()); ref_n = sum(r.get("ref_n_samples", 0) for r in byp.values())
      table.append(dict(behavior=behavior, n_prompts=len(byp), n_judged=n, bf16_matched=k, bf16_rate=k / n if n else None,
                        wilson_lo=lo, wilson_hi=hi, heur_rate_all=hk / len(recs),
                        fp8_same_prompts_rate=ref_k / ref_n if ref_n else None, fp8_all_patterns_rate=fp8_rate))
      print(table[-1], flush=True)


asyncio.run(main())
df = pd.DataFrame(table)
df.to_csv(os.path.join(a.samples, "baseline_rates.csv"), index=False)
print(df.to_string()); print("usage", judge.usage)
