"""14_a4c_sonnet_check.py -- Sonnet 5.5 precision check on the decisive recommends-drunk-driving cells of the A4c lens-lever test.

Every sample Haiku 5.5 judged match=True in the fully judged cells (baseline, L11N2557 +0.5/+0.8, and the strongest L11
controls at those doses) is re-judged by claude-sonnet-5-5 with the same verbatim Transluce rubric (src/judge.py). Reports, per
cell, Sonnet's confirmation rate of Haiku positives and the Sonnet-confirmed rate (confirmed / n). This checks Haiku precision
only (Haiku negatives are not re-judged). Writes runs/a4c_lens/sonnet_check.json.
Run locally: source .venv/bin/activate && python -I scripts/14_a4c_sonnet_check.py
"""
import asyncio
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from judge import Judge, load_rubrics, wilson  # noqa: E402

RUN = os.path.join(ROOT, "runs", "a4c_lens")
CELLS = ["a_baseline", "a_L11N2557+0.5", "a_L11N2557+0.8", "a_ctrl_rn_L11N4486+0.5", "a_ctrl_rd_L11r1+0.5",
         "a_ctrl_rn_L11N6764+0.8", "a_ctrl_rd_L11r0+0.8"]
BEH = "recommends-drunk-driving"


def main():
    J = {}
    for l in open(os.path.join(RUN, "judged.jsonl")):
        r = json.loads(l)
        J[(r["setting"], r["prompt_id"], r["sample_idx"])] = r["judge_match"]
    items, meta, n_cell, haiku_pos = [], [], {}, {}
    for c in CELLS:
        rows = [json.loads(l) for l in open(os.path.join(RUN, c + ".jsonl"))]
        assert all((c, r["prompt_id"], r["sample_idx"]) in J for r in rows), f"{c} not fully judged"
        n_cell[c] = len(rows)
        pos = [r for r in rows if J[(c, r["prompt_id"], r["sample_idx"])]]
        haiku_pos[c] = len(pos)
        for r in pos:
            items.append((BEH, [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]))
            meta.append(c)
    print(f"[sonnet check] {len(items)} Haiku-positive samples across {len(CELLS)} cells", flush=True)
    judge = Judge(load_rubrics(os.path.join(ROOT, "data", "weirdchat", "data")), model="claude-sonnet-5-5", concurrency=8)
    res = asyncio.run(judge.judge_many(items))
    out = {}
    for c in CELLS:
        rs = [r for m, r in zip(meta, res) if m == c]
        ok = [r for r in rs if r.match is not None]
        conf = sum(bool(r.match) for r in ok)
        lo, hi = wilson(conf, n_cell[c])
        out[c] = {"n": n_cell[c], "haiku_pos": haiku_pos[c], "sonnet_judged": len(ok), "sonnet_confirmed": conf,
                  "confirm_rate": conf / len(ok) if ok else None, "haiku_rate": haiku_pos[c] / n_cell[c],
                  "sonnet_confirmed_rate": conf / n_cell[c], "sonnet_confirmed_ci": [lo, hi],
                  "errors": sum(1 for r in rs if r.match is None)}
        print(f"{c}: Haiku {haiku_pos[c]}/{n_cell[c]}; Sonnet confirms {conf}/{len(ok)} -> {conf/n_cell[c]:.2f} [{lo:.2f},{hi:.2f}]", flush=True)
    json.dump({"cells": out, "model": "claude-sonnet-5-5", "usage": judge.usage, "n_calls": judge.n_calls},
              open(os.path.join(RUN, "sonnet_check.json"), "w"), indent=1)
    print("usage", judge.usage, "calls", judge.n_calls)


if __name__ == "__main__":
    main()
