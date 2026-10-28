"""07_a4b_priors_judge.py -- Haiku-judge the poor-things priors runs (runs/a4b_priors/*.jsonl) with src/judge.py.

Cost policy (orchestrator): per cell = (condition, prompt set / source behavior, rubric), judge ALL heuristic
positives plus a seeded random 10% of heuristic negatives. Cells:
  A denying prompts      -> denying-ai-identity rubric
  A embodiment prompts   -> false-physical-embodiment rubric
  B benign prompts       -> denying-ai-identity and false-physical-embodiment rubrics (positives + 10% negatives);
                            claims-called-911 and fabricated-code-execution: heuristic positives only (base rate ~0).
Writes runs/a4b_priors/judged.jsonl: condition, set, source_behavior, prompt_id, sample_idx, rubric, stratum,
judge_match, judge_explanation. Judge model claude-haiku-5-5, verbatim Transluce transcript-rubric prompt.
Run locally:  source .venv/bin/activate && python -I scripts/07_a4b_priors_judge.py
"""
import argparse
import asyncio
import glob
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from judge import Judge, load_api_key, load_rubrics  # noqa: E402

RUN = os.path.join(ROOT, "runs", "a4b_priors")
FULL = {("weirdchat", "denying-ai-identity"): ["denying-ai-identity"],
        ("weirdchat", "false-physical-embodiment"): ["false-physical-embodiment"],
        ("benign", None): ["denying-ai-identity", "false-physical-embodiment"]}
POS_ONLY_BENIGN = ["claims-called-911", "fabricated-code-execution"]


def build_queue():
    rng = random.Random(0)
    queue = []
    for f in sorted(glob.glob(os.path.join(RUN, "*.jsonl"))):
        if os.path.basename(f) in ("judged.jsonl", "judge_queue.jsonl"):
            continue
        rows = [json.loads(l) for l in open(f)]
        cells = {}
        for x in rows:
            key = (x["set"], x["source_behavior"] if x["set"] == "weirdchat" else None)
            cells.setdefault(key, []).append(x)
        for key, xs in cells.items():
            rubrics = FULL[key] + (POS_ONLY_BENIGN if key[0] == "benign" else [])
            for rb in rubrics:
                pos = [x for x in xs if x["heur"][rb]]
                neg = [x for x in xs if not x["heur"][rb]]
                pick = rng.sample(neg, max(1, round(0.1 * len(neg)))) if rb in FULL[key] and neg else []
                for x, st in [(x, "heuristic_pos") for x in pos] + [(x, "neg_10pct") for x in pick]:
                    queue.append({"condition": x["condition"], "set": x["set"], "source_behavior": x["source_behavior"],
                                  "prompt_id": x["prompt_id"], "sample_idx": x["sample_idx"], "rubric": rb, "stratum": st,
                                  "prompt": x["prompt"], "response": x["response"]})
    return queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=12)
    a = ap.parse_args()
    if not a.dry_run and load_api_key() is None:
        print("[priors judge] no key -> dry run", file=sys.stderr); a.dry_run = True
    queue = build_queue()
    print(f"[priors judge] {len(queue)} items: {sum(q['stratum']=='heuristic_pos' for q in queue)} positives, "
          f"{sum(q['stratum']=='neg_10pct' for q in queue)} sampled negatives", flush=True)
    with open(os.path.join(RUN, "judge_queue.jsonl"), "w") as f:
        for q in queue:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    if a.dry_run:
        return
    rubrics = load_rubrics(os.path.join(ROOT, "data", "weirdchat", "data"))
    judge = Judge(rubrics, concurrency=a.concurrency)
    items = [(q["rubric"], [{"role": "user", "content": q["prompt"]}, {"role": "assistant", "content": q["response"]}]) for q in queue]
    res = asyncio.run(judge.judge_many(items))
    with open(os.path.join(RUN, "judged.jsonl"), "w") as f:
        for q, r in zip(queue, res):
            f.write(json.dumps({k: q[k] for k in ("condition", "set", "source_behavior", "prompt_id", "sample_idx", "rubric", "stratum")}
                               | {"judge_match": r.match, "judge_error": r.error, "judge_explanation": r.explanation}, ensure_ascii=False) + "\n")
    print("usage", judge.usage)


if __name__ == "__main__":
    main()
