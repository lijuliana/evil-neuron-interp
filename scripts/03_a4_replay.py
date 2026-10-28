"""
03_a4_replay.py -- replay WeirdChat prompts (Qwen3.6-27B + 35B-A3B) on the
unsteered local Qwen3.5-4B-MLX-8bit to learn which behaviors exist at all on 4B.

For each distinct behavior_id: up to N_PROMPTS prompts (preferring patterns with
the highest WeirdChat match_rate), N_SAMPLES responses each at T=1, thinking off,
no system prompt, max MAX_TOKENS tokens.  Responses are judged with Claude Haiku
5.5 and the exact WeirdChat transcript rubric when ANTHROPIC_API_KEY is in .env,
otherwise with the regex pre-screen in src/judge.py and marked UNJUDGED.

Outputs: runs/a4_replay/<behavior>.jsonl, summary.md, summary.json

Run:   source .venv/bin/activate && python -I scripts/03_a4_replay.py
       python -I scripts/03_a4_replay.py --judge-only     # re-judge saved responses
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from heuristic_judge import have_api_key, judge_heuristic, distinct_ngram_ratio  # noqa: E402


def judge_llm(behavior_id, user, assistant):
    # LLM judging lives in src/judge.py (A0's Claude Haiku 5.5 wrapper mirroring Transluce's judge).
    raise NotImplementedError("use src/judge.py (A0) for LLM judging of runs/a4_replay/*.jsonl")
from mlx_steer import wilson_ci  # noqa: E402

OUT = ROOT / "runs" / "a4_replay"
OUT.mkdir(parents=True, exist_ok=True)
MODEL = ROOT / "models" / "Qwen3.5-4B-MLX-8bit"
N_PROMPTS, N_SAMPLES, MAX_TOKENS = 20, 16, 300
HARMFUL = {"cutting-instructions", "purging-advice", "extreme-calorie-restriction", "direct-kill-yourself",
           "suicide-jumping", "strike-partner", "unsolicited-sexual-advances", "recommends-drunk-driving",
           "laser-at-aircraft"}


def select_prompts() -> dict[str, pd.DataFrame]:
    pats = pd.read_parquet(ROOT / "data/weirdchat/data/patterns.parquet")
    pats["pattern_match_rate"] = pats.metrics.apply(lambda m: float(m.get("match_rate", 0.0)))
    pats = pats[["pattern_id", "pattern_match_rate", "title"]]
    frames = []
    for f in sorted(glob.glob(str(ROOT / "data/weirdchat/data/prompts/qwen__*/*.parquet"))):
        df = pd.read_parquet(f)
        df["source_model"] = Path(f).parent.name
        frames.append(df)
    pr = pd.concat(frames).merge(pats, on="pattern_id", how="left")
    pr["user_text"] = pr.messages.apply(lambda ms: "\n".join(m["content"] for m in ms if m["role"] == "user"))
    pr["prompt_match_rate"] = pr.match_summary.apply(lambda s: s["n_matched"] / max(1, s["n_samples"]))
    pr = pr.sort_values(["pattern_match_rate", "prompt_match_rate"], ascending=False)
    pr = pr.drop_duplicates("user_text")
    return {b: g.head(N_PROMPTS).reset_index(drop=True) for b, g in pr.groupby("behavior_id")}


def generate_all(sel: dict[str, pd.DataFrame], behaviors: list[str]) -> None:
    from mlx_steer import Steerer
    s = Steerer(str(MODEL))
    s.steer([])
    for b in behaviors:
        path = OUT / f"{b}.jsonl"
        if path.exists() and sum(1 for _ in open(path)) >= len(sel[b]) * N_SAMPLES:
            print(f"[{b}] already done, skipping", file=sys.stderr)
            continue
        df = sel[b]
        t0 = time.time()
        prompts = [list(ms) for ms in df.messages]   # message lists (single user turn)
        outs = s.generate(prompts, n_samples=N_SAMPLES, max_tokens=MAX_TOKENS, temp=1.0, seed=0, completion_batch_size=64)
        rows = []
        for (_, r), samples in zip(df.iterrows(), outs):
            for i, txt in enumerate(samples):
                rows.append({"prompt_id": r.prompt_id, "behavior_id": b, "pattern_id": r.pattern_id,
                             "source_model": r.source_model, "pattern_match_rate": r.pattern_match_rate,
                             "prompt_match_rate": r.prompt_match_rate, "prompt": r.user_text,
                             "sample": i, "response": txt, "distinct4": distinct_ngram_ratio(txt),
                             "judge": None})
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
        print(f"[{b}] {len(rows)} responses in {time.time()-t0:.0f}s", file=sys.stderr)


def judge_all(behaviors: list[str], use_llm: bool) -> None:
    for b in behaviors:
        path = OUT / f"{b}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)]
        n_new = 0
        for x in rows:
            if x.get("judge") and x["judge"].get("judge") != "heuristic" and not use_llm:
                continue
            if use_llm and x.get("judge") and x["judge"].get("judge", "").startswith("claude"):
                continue
            x["judge"] = judge_llm(b, x["prompt"], x["response"]) if use_llm else judge_heuristic(b, x["prompt"], x["response"])
            x["judged"] = "LLM" if use_llm else "UNJUDGED_heuristic"
            n_new += 1
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
        print(f"[{b}] judged {n_new}", file=sys.stderr)


def summarize(behaviors: list[str], use_llm: bool) -> None:
    calib = {}
    cpath = OUT / "heuristic_calibration.md"
    if cpath.exists():
        for line in cpath.read_text().splitlines():
            if line.startswith("| ") and not line.startswith("| behavior") and "---" not in line:
                c = [x.strip() for x in line.strip("|").split("|")]
                calib[c[0]] = (c[6], c[7])
    mode = "Claude Haiku 5.5 + WeirdChat transcript rubric" if use_llm else "regex heuristic pre-screen (UNJUDGED; see heuristic_calibration.md)"
    done = [b for b in behaviors if (OUT / f"{b}.jsonl").exists()]
    missing = [b for b in behaviors if b not in done]
    md = [f"# WeirdChat replay on Qwen3.5-4B-MLX-8bit (unsteered)" + (" -- PARTIAL" if missing else ""), ""]
    if missing:
        md += [f"**PARTIAL RESULT: {len(done)}/{len(behaviors)} behaviors generated.** Local MLX generation was stopped "
               f"(thermal load on the Mac); the remaining behaviors move to the AWS GPU track. Not generated: {', '.join(missing)}.", ""]
    md += [
          f"T=1.0, thinking disabled, no system prompt, max {MAX_TOKENS} new tokens, {N_SAMPLES} samples per prompt, "
          f"up to {N_PROMPTS} prompts per behavior chosen from the Qwen3.6-27B and 35B-A3B prompt sets by highest pattern match_rate.",
          f"Judge: **{mode}**.", "",
          "Sampling note: pure T=1.0 (top_p=0, top_k=0). The Qwen generation_config defaults (top_p=0.95, top_k=20) that a vLLM server applies, and that WeirdChat sampled through, were NOT used; replicate on the GPU track if rates look low.", "",
          "Reference rate = mean WeirdChat match_rate of the selected prompts' patterns (FP8 Qwen3.6, Gemma-judged). Heuristic P/R = precision/recall of the regex vs shipped labels on Qwen3.6 transcripts.", "",
          "| behavior | prompts | samples | matches | rate | 95% Wilson CI | ref rate (3.6) | heuristic P / R | mean distinct-4 |",
          "|---|---|---|---|---|---|---|---|---|"]
    examples = []
    summary = {}
    for b in behaviors:
        path = OUT / f"{b}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)]
        n = len(rows)
        k = sum(1 for x in rows if x.get("judge") and x["judge"].get("match"))
        lo, hi = wilson_ci(k, n)
        ref = sum(x["prompt_match_rate"] for x in rows) / n
        d4 = sum(x["distinct4"] for x in rows) / n
        pr = calib.get(b, ("-", "-"))
        n_prompts = len({x["prompt_id"] for x in rows})
        md.append(f"| {b} | {n_prompts} | {n} | {k} | {k/n:.3f} | [{lo:.3f}, {hi:.3f}] | {ref:.3f} | {pr[0]} / {pr[1]} | {d4:.2f} |")
        summary[b] = {"n_prompts": n_prompts, "n": n, "matches": k, "rate": k / n, "ci": [lo, hi], "ref_rate": ref,
                      "judge": "llm" if use_llm else "heuristic_unjudged"}
        hits = [x for x in rows if x.get("judge") and x["judge"].get("match")]
        hits.sort(key=lambda x: -x["distinct4"])
        ex = [f"\n## {b}: {k}/{n} matches ({'LLM' if use_llm else 'heuristic, UNJUDGED'})"]
        for x in hits[:2]:
            body = (f"- prompt ({x['source_model']}, pattern rate {x['pattern_match_rate']:.2f}): "
                    f"{x['prompt'][:200].replace(chr(10),' ')}...\n"
                    f"  - judge: {x['judge'].get('explanation','')[:160]}\n"
                    f"  - response: {x['response'][:400].replace(chr(10),' ')}")
            if b in HARMFUL:
                ex.append(f"<details><summary>example (harmful content)</summary>\n\n{body}\n\n</details>")
            else:
                ex.append(body)
        if not hits:
            ex.append("- no matches")
        examples.extend(ex)
    md.append("")
    md.append("## Example matches (2 per behavior; harmful ones collapsed)")
    md.extend(examples)
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    json.dump({"model": str(MODEL), "n_prompts": N_PROMPTS, "n_samples": N_SAMPLES, "max_tokens": MAX_TOKENS,
               "judge_mode": mode, "behaviors": summary}, open(OUT / "summary.json", "w"), indent=1)
    print("\n".join(md[:30]), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-only", action="store_true")
    ap.add_argument("--behaviors", nargs="*", default=None)
    ap.add_argument("--force-heuristic", action="store_true")
    args = ap.parse_args()
    sel = select_prompts()
    behaviors = args.behaviors or sorted(sel)
    use_llm = have_api_key() and not args.force_heuristic
    print(f"{len(behaviors)} behaviors; LLM judge available: {use_llm}", file=sys.stderr)
    (OUT / "selected_prompts.json").write_text(json.dumps(
        {b: sel[b][["prompt_id", "pattern_id", "source_model", "pattern_match_rate", "prompt_match_rate", "user_text"]].to_dict("records")
         for b in behaviors}, indent=1, ensure_ascii=False))
    if not args.judge_only:
        generate_all(sel, behaviors)
    judge_all(behaviors, use_llm)
    summarize(behaviors, use_llm)


if __name__ == "__main__":
    main()
