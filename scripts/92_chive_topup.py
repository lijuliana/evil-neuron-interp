"""CHIVE Eval B: write the top-up spec (findings/chive/design.md section 4, decision 2).

After the main run (base j < 128, every edit j < 32) is judged, every edit whose independent-estimator effect after
32 replies lies in [4, 35] pp gets 32 more replies (j = 32..63). The selection reads only the judge files (no
evidence, no predictor output): it is arm-blind by construction. Edits whose base or edit verdicts are incomplete
(fewer than 87.5% of the window non-null) are also topped up, so that a missing judgment does not drop a claim.

  python -I scripts/92_chive_topup.py                 -> data/chive/topup/t1.json + t1_summary.json, prints sha256
  bash scripts/92_chive_watchdog.sh topup data/chive/topup/t1.json

The spec: {"tag": "t1", "edit_ids": [...], "edit_j": [32, 64]} as 92_chive_gen.py --phase topup expects.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
N_BASE, N_EDIT, N_TOPUP = 128, 32, 64
LO_PP, HI_PP = 4.0, 35.0
MIN_NONNULL_FRAC = 0.875


def read_jsonl(p):
    with open(p) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-glob", default=os.path.join(REPO, "runs/chive/judge/shard*.jsonl"))
    ap.add_argument("--edits", default=os.path.join(REPO, "data/chive/edits.jsonl"))
    ap.add_argument("--tag", default="t1")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "data/chive/topup"))
    args = ap.parse_args()

    edits = list(read_jsonl(args.edits))
    edit_ids = {e["edit_id"] for e in edits}
    paths = [p for p in sorted(glob.glob(args.judge_glob)) if not any(s in os.path.basename(p) for s in ("_rejudge", "_drift", "_orig", "_topup"))]
    assert paths, f"no judge files match {args.judge_glob}"
    tab = collections.defaultdict(dict)   # (pid, cond) -> {j: match}
    n_rows = 0
    for p in paths:
        for r in read_jsonl(p):
            if r.get("mode", "main") != "main" or r.get("status") == "http":
                continue
            cond, j = r["cond_id"], int(r["j"])
            if cond != "base" and cond not in edit_ids:
                continue
            if j >= (N_BASE if cond == "base" else N_EDIT):
                continue
            d = tab[(r["prompt_id"], cond)]
            if d.get(j) is None:
                d[j] = r.get("match")
            n_rows += 1

    def kn(d):
        v = [x for x in d.values() if x is not None]
        return sum(v), len(v)

    sel, why = [], collections.Counter()
    effects = []
    for e in edits:
        pid, eid = e["prompt_id"], e["edit_id"]
        k0, n0 = kn(tab.get((pid, "base"), {}))
        k1, n1 = kn(tab.get((pid, eid), {}))
        if n0 == 0 or n1 == 0:
            why["no_data"] += 1
            sel.append(eid)
            continue
        eff = abs(k1 / n1 - k0 / n0) * 100
        effects.append(eff)
        if n1 < MIN_NONNULL_FRAC * N_EDIT or n0 < MIN_NONNULL_FRAC * N_BASE:
            why["incomplete"] += 1
            sel.append(eid)
        elif LO_PP <= eff <= HI_PP:
            why["borderline"] += 1
            sel.append(eid)
        else:
            why["settled"] += 1
    os.makedirs(args.out_dir, exist_ok=True)
    spec = {"tag": args.tag, "edit_ids": sorted(sel), "edit_j": [N_EDIT, N_TOPUP]}
    out = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out, "w") as f:
        json.dump(spec, f, indent=1)
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    summary = {"rule": f"top up to {N_TOPUP} if |effect after {N_EDIT}| in [{LO_PP}, {HI_PP}] pp, or base/edit verdicts "
                       f"incomplete (< {MIN_NONNULL_FRAC} of window), or no data", "judge_files": paths, "judge_rows": n_rows,
               "n_edits": len(edits), "selected": len(sel), "reasons": dict(why),
               "effect_hist_pp": dict(collections.Counter(f"{int(x // 5) * 5}-{int(x // 5) * 5 + 5}" for x in effects)),
               "requests": len(sel) * (N_TOPUP - N_EDIT), "sha256": sha}
    with open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[topup] {len(sel)} of {len(edits)} edits selected {dict(why)}; {summary['requests']} replies; "
          f"{out} sha256 {sha}")


if __name__ == "__main__":
    main()
