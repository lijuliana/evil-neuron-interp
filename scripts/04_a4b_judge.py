"""04_a4b_judge.py -- LLM-judge the Qwen3.5-4B replay queue (runs/a4b_replay/judge_queue.jsonl) with
Claude Haiku 5.5 and the exact WeirdChat transcript rubric via src/judge.py (A0). Writes
runs/a4b_replay/<behavior>.judged.jsonl (prompt_id, sample_idx, stratum, judge_match, judge_explanation).
Without ANTHROPIC_API_KEY (env or repo .env) it runs --dry-run and writes nothing useful; the summary
then stays UNJUDGED.

Run locally:  source .venv/bin/activate && python -I scripts/04_a4b_judge.py [--dry-run]
"""
import argparse
import asyncio
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from judge import Judge, load_api_key, load_rubrics  # noqa: E402

RUN = os.path.join(ROOT, "runs", "a4b_replay")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    if not a.dry_run and load_api_key() is None:
        print("[a4b judge] no ANTHROPIC_API_KEY -> dry run; results stay UNJUDGED", file=sys.stderr)
        a.dry_run = True
    rubrics = load_rubrics(os.path.join(ROOT, "data", "weirdchat", "data"))
    queue = [json.loads(l) for l in open(os.path.join(RUN, "judge_queue.jsonl"))]
    judge = Judge(rubrics, dry_run=a.dry_run, concurrency=a.concurrency)
    items = [(q["behavior_id"], [{"role": "user", "content": q["prompt"]}, {"role": "assistant", "content": q["response"]}]) for q in queue]
    res = asyncio.run(judge.judge_many(items))
    if a.dry_run:
        print(f"[a4b judge] dry run over {len(items)} items; nothing written")
        return
    by_b = {}
    for q, r in zip(queue, res):
        by_b.setdefault(q["behavior_id"], []).append({**{k: q[k] for k in ("prompt_id", "sample_idx", "stratum")},
                                                      "judge_match": r.match, "judge_error": r.error, "judge_explanation": r.explanation})
    for b, rows in by_b.items():
        with open(os.path.join(RUN, f"{b}.judged.jsonl"), "w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("usage", judge.usage)


if __name__ == "__main__":
    main()
