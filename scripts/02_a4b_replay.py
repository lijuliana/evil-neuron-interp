"""02_a4b_replay.py -- WeirdChat replay on unsteered Qwen3.5-4B bf16 (AWS A10G).

For every behavior in runs/a4_replay/selected_prompts.json (frozen by A4: up to 20 prompts per behavior
from the Qwen3.6-27B and 35B-A3B prompt sets, highest pattern match_rate first): N_SAMPLES responses per
prompt, T=1.0, top_p=0.95, top_k=20 (WeirdChat's vLLM server defaults), thinking off, no system
prompt, max 300 new tokens. Writes runs/a4b_replay/<behavior>.jsonl with prompt_id, behavior_id,
sample_idx, response, finish_reason (+ prompt text, pattern metadata, distinct-4, heuristic judge).
Resumable per behavior. The LLM judge (src/judge.py, Claude Haiku 5.5) runs separately if a key exists.

Run on VM:  source ~/venv/bin/activate && cd ~/evil-neuron-interp &&
            nohup python scripts/02_a4b_replay.py --batch-size 128 > ~/data/runs/replay.log 2>&1 &
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from harness_4b import Harness4B  # noqa: E402
from heuristic_judge import distinct_ngram_ratio, judge_heuristic  # noqa: E402

SEL = "runs/a4_replay/selected_prompts.json"
OUT = "runs/a4b_replay"
PROMPT_DIR = os.path.expanduser("~/data/weirdchat/prompts")
N_SAMPLES, MAX_NEW = 32, 300
SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=20)


def load_messages() -> dict[str, list[dict]]:
    """prompt_id -> messages from the shipped prompt parquets (keeps multi-turn structure if any)."""
    import pandas as pd
    m = {}
    for f in glob.glob(os.path.join(PROMPT_DIR, "qwen__*", "*.parquet")):
        df = pd.read_parquet(f)
        for r in df.itertuples():
            m[r.prompt_id] = [{"role": x["role"], "content": x["content"]} for x in r.messages]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--behaviors", nargs="*", default=None)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--judge-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    sel = json.load(open(SEL))
    behaviors = a.behaviors or sorted(sel)
    msgs = load_messages()
    h = None
    for b in behaviors:
        path = f"{OUT}/{b}.jsonl"
        recs = sel[b]
        if os.path.exists(path) and sum(1 for _ in open(path)) >= len(recs) * a.n_samples:
            print(f"[{b}] done, skipping generation", flush=True)
        elif a.judge_only:
            continue
        else:
            if h is None:
                h = Harness4B(resid_norms_path="captures/qwen3.5-4b_resid_norms_bf16.json")
            prompts = []
            for r in recs:
                ms = msgs.get(r["prompt_id"]) or [{"role": "user", "content": r["user_text"]}]
                prompts.append(ms)
            n_multi = sum(len(p) > 1 for p in prompts)
            t0 = time.time()
            outs = h.chat_full(prompts, n=a.n_samples, max_new_tokens=MAX_NEW, batch_size=a.batch_size,
                               seed=0, verbose=True, **SAMPLING)
            rows = []
            for r, samples in zip(recs, outs):
                for i, s in enumerate(samples):
                    rows.append({"prompt_id": r["prompt_id"], "behavior_id": b, "sample_idx": i,
                                 "response": s["response"], "finish_reason": s["finish_reason"],
                                 "n_tokens": s["n_tokens"], "pattern_id": r["pattern_id"],
                                 "source_model": r["source_model"], "pattern_match_rate": r["pattern_match_rate"],
                                 "prompt_match_rate": r["prompt_match_rate"], "prompt": r["user_text"],
                                 "n_turns": len(msgs.get(r["prompt_id"], [None])),
                                 "distinct4": distinct_ngram_ratio(s["response"]),
                                 "sampling": SAMPLING, "model": "Qwen/Qwen3.5-4B bf16"})
            with open(path, "w") as f:
                for x in rows:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            print(f"[{b}] {len(rows)} responses ({n_multi} multi-turn prompts) in {time.time()-t0:.0f}s", flush=True)
        # heuristic pre-screen (UNJUDGED until src/judge.py runs)
        rows = [json.loads(l) for l in open(path)]
        for x in rows:
            x["heuristic"] = judge_heuristic(b, x["prompt"], x["response"])
            x.setdefault("judged", "UNJUDGED_heuristic")
        with open(path, "w") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        k = sum(x["heuristic"]["match"] for x in rows)
        print(f"[{b}] heuristic {k}/{len(rows)}; length-stopped {sum(x['finish_reason']=='length' for x in rows)}", flush=True)
    print("REPLAY_DONE", flush=True)


if __name__ == "__main__":
    main()
