"""
02_a4_heuristic_calib.py -- calibrate src/judge.py heuristics against the
shipped WeirdChat judge labels (Gemma 4 31B) on the Qwen3.6 transcripts.

Per behavior: precision / recall / F1 of the regex pre-screen vs judgment.match,
on up to 1500 transcripts per behavior.  Writes runs/a4_replay/heuristic_calibration.md

Run: source .venv/bin/activate && python -I scripts/02_a4_heuristic_calib.py
"""
import glob
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from heuristic_judge import judge_heuristic  # noqa: E402
from mlx_steer import wilson_ci  # noqa: E402

OUT = ROOT / "runs" / "a4_replay"
OUT.mkdir(parents=True, exist_ok=True)
MAX_PER_BEH = 1500


def main():
    files = sorted(glob.glob(str(ROOT / "data/weirdchat/data/transcripts/qwen__*/*.parquet")))
    by_beh = {}
    for f in files:
        beh = os.path.basename(f).rsplit("-", 1)[0]
        by_beh.setdefault(beh, []).append(f)
    rows = []
    for beh, fs in sorted(by_beh.items()):
        dfs = [pd.read_parquet(f) for f in fs]
        df = pd.concat(dfs)
        # balance: all matched + sample of unmatched
        lab = df.judgment.apply(lambda j: bool(j.get("match")))
        pos = df[lab]
        neg = df[~lab].sample(min(len(df[~lab]), max(300, MAX_PER_BEH - len(pos))), random_state=0)
        sub = pd.concat([pos.head(MAX_PER_BEH), neg])
        tp = fp = fn = tn = 0
        for _, r in sub.iterrows():
            msgs = list(r.messages)
            user = "\n".join(m["content"] for m in msgs if m["role"] == "user")
            asst = "\n".join(m["content"] for m in msgs if m["role"] == "assistant")
            h = judge_heuristic(beh, user, asst)["match"]
            y = bool(r.judgment.get("match"))
            tp += h and y; fp += h and not y; fn += (not h) and y; tn += (not h) and not y
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
        rows.append((beh, len(sub), tp + fn, tp, fp, fn, prec, rec, f1))
        print(f"{beh:30s} n={len(sub):5d} pos={tp+fn:4d} TP={tp:4d} FP={fp:4d} FN={fn:4d} P={prec:.2f} R={rec:.2f} F1={f1:.2f}", file=sys.stderr)
    md = ["# Heuristic pre-screen vs shipped WeirdChat labels (Qwen3.6-27B + 35B-A3B transcripts)", "",
          "Judge of record is Gemma 4 31B with the transcript rubric. Negatives subsampled; precision is on this enriched mix (true precision on fresh samples will be lower where base rates are low).", "",
          "| behavior | n | positives | TP | FP | FN | precision | recall | F1 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]:.2f} | {r[7]:.2f} | {r[8]:.2f} |")
    (OUT / "heuristic_calibration.md").write_text("\n".join(md) + "\n")
    print(f"wrote {OUT/'heuristic_calibration.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
