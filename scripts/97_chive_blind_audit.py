"""CHIVE Eval B step 97: blindness audit of the control evidence (2026-11-01 design review, item 6).

For each method with control files, claude-haiku-5-5 sees one transcript and ONE evidence block and gives the
probability that the block is the real output of the method for this transcript rather than a control. Every prompt is
asked twice per pair type (once with its real block, once with its control block), for the pair types
  ctrl: data/chive/evidence/{method}.jsonl vs {method}_ctrl.jsonl  (same-prompt control)
  shuf: data/chive/evidence/{method}.jsonl vs {method}_shuf.jsonl  (cross-prompt derangement)
Reported per method and pair type: accuracy at 0.5 (chance .5) with a Wilson 95% CI, AUROC of P(real), and the
two-alternative score (share of prompts whose real block gets a higher P(real) than its control, ties .5).
High accuracy means the controls are distinguishable from real evidence by reading alone, so a predictor could exploit
that difference; near-chance accuracy is what a good control needs.

Transcripts come from runs/chive/claims.jsonl (one per prompt; same for every claim of a prompt). Only prompt-keyed
evidence is audited (the claim-keyed pos arm is skipped); pairs where either block is missing or marked meta.na, or
where the real and control blocks are identical, are skipped and counted. Cache like 95: runs/chive/audit/cache/<h[:2]>/<h>.json with
h = sha256(model, system, user, tool, thinking, max_tokens). Haiku 5.5 runs with adaptive thinking at its default effort; tool_choice auto.

  python -I scripts/97_chive_blind_audit.py --dry-run --limit 2
  python -I scripts/97_chive_blind_audit.py [--methods neurons olens axis] [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
import os
import random
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 8000
THINKING = {"type": "adaptive"}
PAIR_SUFFIX = {"ctrl": "_ctrl", "shuf": "_shuf"}

SYSTEM = ("You audit interpretability evidence. You see a transcript of a language model's behavior and one evidence "
          "block from an interpretability method. The block is either the real output of the method for this "
          "transcript or a control block. Judge only from the transcript and the block.")
CONTROL_TEXT = {
    "ctrl": ("the same method run with a control input that carries no information specific to this transcript "
             "(for example a random neuron set, a random direction, or a re-read of unrelated text)"),
    "shuf": ("the real output of this method for a different prompt from the same collection (usually one targeting "
             "the same behavior, occasionally another behavior)"),
}
USER = """Interpretability method: {display}. {description}

The evidence block below is either the real output of this method for this transcript (real), or a control: {control}.

Transcript:
{transcript}

Evidence block:
{block}

Estimate the probability that the block is the real evidence for this transcript, then call the `verdict` tool."""
VERDICT_TOOL = {
    "name": "verdict",
    "description": "Submit the probability that the evidence block is the real output for this transcript.",
    "input_schema": {"type": "object", "properties": {
        "reasoning": {"type": "string", "description": "Brief reasoning."},
        "probability_real": {"type": "number", "description": "Probability in [0, 1] that the block is real."}},
        "required": ["reasoning", "probability_real"]},
}


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_prompt_blocks(path: str) -> dict[str, str]:
    """Prompt-keyed, available blocks only (rows with meta.na = true are left out)."""
    out = {}
    for r in read_jsonl(path):
        if r.get("claim_id") is None and not (r.get("meta") or {}).get("na"):
            out[r["prompt_id"]] = r["block"].rstrip()
    return out


def key_of(model: str, system: str, user: str) -> str:
    blob = json.dumps({"model": model, "system": system, "user": user, "tool": VERDICT_TOOL, "thinking": THINKING,
                       "max_tokens": MAX_TOKENS}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p, den = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def auroc(y: list[int], s: list[float]) -> float | None:
    pos = [b for a, b in zip(y, s) if a]; neg = [b for a, b in zip(y, s) if not a]
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


async def ask(client, sem, model, user, anthropic):
    for attempt in range(4):
        try:
            async with sem:
                resp = await client.messages.create(model=model, max_tokens=MAX_TOKENS, system=SYSTEM,
                                                    messages=[{"role": "user", "content": user}],
                                                    tools=[VERDICT_TOOL], thinking=THINKING)
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError,
                anthropic.APITimeoutError) as e:
            await asyncio.sleep(20 * (attempt + 1))
            err = f"{type(e).__name__}"
            continue
        except anthropic.APIStatusError as e:
            return None, f"status {e.status_code}"
        tool = next((b for b in resp.content if b.type == "tool_use" and b.name == "verdict"), None)
        p = tool.input.get("probability_real") if tool is not None and isinstance(tool.input, dict) else None
        if isinstance(p, (int, float)) and not isinstance(p, bool) and 0 <= p <= 1:
            return {"p": float(p), "reasoning": str(tool.input.get("reasoning", "")),
                    "usage": [resp.usage.input_tokens, resp.usage.output_tokens]}, None
        err = f"no valid verdict (stop_reason={resp.stop_reason})"
    return None, err


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["p"] is not None]
    k = sum((r["p"] >= 0.5) == r["is_real"] for r in ok)
    by_prompt = collections.defaultdict(dict)
    for r in ok:
        by_prompt[r["prompt_id"]][r["is_real"]] = r["p"]
    two = [(d[True] > d[False]) + 0.5 * (d[True] == d[False]) for d in by_prompt.values() if len(d) == 2]
    return {"n": len(ok), "n_null": len(rows) - len(ok), "accuracy": (k / len(ok)) if ok else None,
            "accuracy_wilson95": wilson(k, len(ok)), "auroc": auroc([int(r["is_real"]) for r in ok], [r["p"] for r in ok]),
            "two_afc": (sum(two) / len(two)) if two else None, "n_two_afc": len(two),
            "mean_p_real": (sum(r["p"] for r in ok if r["is_real"]) / max(1, sum(r["is_real"] for r in ok))),
            "mean_p_control": (sum(r["p"] for r in ok if not r["is_real"]) / max(1, sum(not r["is_real"] for r in ok)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claims", default=os.path.join(REPO, "runs/chive/claims.jsonl"))
    ap.add_argument("--evidence-dir", default=os.path.join(REPO, "data/chive/evidence"))
    ap.add_argument("--methods-json", default=os.path.join(REPO, "data/chive/methods.json"))
    ap.add_argument("--methods", nargs="*", default=None, help="default: every method with a _ctrl or _shuf file")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "runs/chive/audit"))
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="audit a seeded (0) random sample of this many prompts")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cache_dir = os.path.join(args.out_dir, "cache")

    transcripts = {}
    for c in read_jsonl(args.claims):
        transcripts.setdefault(c["prompt_id"], c["transcript"])
    mj = {k: v for k, v in json.load(open(args.methods_json)).items() if not k.startswith("_")}
    methods = args.methods or sorted(m for m in mj if m != "pos" and any(
        os.path.exists(os.path.join(args.evidence_dir, f"{m}{suf}.jsonl")) for suf in PAIR_SUFFIX.values()))
    pids = sorted(transcripts)
    if args.limit:
        pids = sorted(random.Random(0).sample(pids, min(args.limit, len(pids))))

    jobs = []   # (method, pair, prompt_id, is_real, user, key)
    skipped = collections.Counter()
    for m in methods:
        real_path = os.path.join(args.evidence_dir, f"{m}.jsonl")
        if not os.path.exists(real_path):
            print(f"[audit] {m}: no {real_path}, skipped")
            continue
        real = load_prompt_blocks(real_path)
        for pair, suf in PAIR_SUFFIX.items():
            cpath = os.path.join(args.evidence_dir, f"{m}{suf}.jsonl")
            if not os.path.exists(cpath):
                continue
            ctrl = load_prompt_blocks(cpath)
            for pid in pids:
                if pid not in real or pid not in ctrl:      # missing or NA on either side
                    skipped[(m, pair, "missing_or_na")] += 1
                    continue
                if real[pid] == ctrl[pid]:                  # identical blocks cannot be told apart: skip the pair
                    skipped[(m, pair, "identical")] += 1
                    continue
                for is_real, block in ((True, real[pid]), (False, ctrl[pid])):
                    user = USER.format(display=mj[m].get("display", m), description=mj[m]["description"].strip(),
                                       control=CONTROL_TEXT[pair], transcript=transcripts[pid], block=block)
                    jobs.append((m, pair, pid, is_real, user, key_of(args.model, SYSTEM, user)))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[audit] methods {methods}; {len(jobs)} judgements over {len(pids)} prompts (model {args.model}); "
          f"skipped pairs {dict((':'.join(k), v) for k, v in skipped.items())}")
    if args.dry_run:
        with open(os.path.join(args.out_dir, "dry_run_prompts.txt"), "w") as f:
            for m, pair, pid, is_real, user, key in jobs[:20]:
                f.write(f"{'=' * 80}\n{m} {pair} {pid} real={is_real}\n{'-' * 30} SYSTEM\n{SYSTEM}\n"
                        f"{'-' * 30} USER\n{user}\n")
        print(f"[audit] dry run: first {min(20, len(jobs))} prompts -> {args.out_dir}/dry_run_prompts.txt")
        return

    import anthropic
    from judge import load_api_key
    client = anthropic.AsyncAnthropic(api_key=load_api_key(), max_retries=6, timeout=600.0)
    sem = asyncio.Semaphore(args.concurrency)
    usage = [0, 0, 0]

    async def one(job):
        m, pair, pid, is_real, user, key = job
        path = os.path.join(cache_dir, key[:2], key + ".json")
        if os.path.exists(path):
            out, err = json.load(open(path)), None
        else:
            out, err = await ask(client, sem, args.model, user, anthropic)
            usage[2] += 1
            if out is not None:
                usage[0] += out["usage"][0]; usage[1] += out["usage"][1]
                os.makedirs(os.path.dirname(path), exist_ok=True)
                json.dump(out, open(path, "w"))
        return {"method": m, "pair": pair, "prompt_id": pid, "is_real": is_real, "p": out["p"] if out else None,
                "reasoning": out["reasoning"] if out else "", "error": err, "key": key}

    async def run_all():
        return await asyncio.gather(*[one(j) for j in jobs])

    t0 = time.time()
    rows = asyncio.run(run_all())
    with open(os.path.join(args.out_dir, "audit.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"model": args.model, "n_api_calls": usage[2], "usage_in_out": usage[:2], "seconds": time.time() - t0,
               "skipped_pairs": {":".join(k): v for k, v in skipped.items()}, "results": {}}
    for m in methods:
        for pair in PAIR_SUFFIX:
            rr = [r for r in rows if r["method"] == m and r["pair"] == pair]
            if rr:
                summary["results"][f"{m}:{pair}"] = summarize(rr)
    with open(os.path.join(args.out_dir, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    lines = ["# Blindness audit", "", f"Judge `{args.model}`; chance = 0.5.", "",
             "| method:pair | n | accuracy [Wilson 95%] | AUROC | 2AFC | mean P(real) real / control |",
             "|---|---|---|---|---|---|"]
    for k, s in summary["results"].items():
        if not s["n"]:
            lines.append(f"| {k} | 0 | all {s['n_null']} null | | | |")
            continue
        w = s["accuracy_wilson95"]
        lines.append(f"| {k} | {s['n']} | {s['accuracy']:.3f} [{w[0]:.3f}, {w[1]:.3f}] | "
                     f"{s['auroc'] if s['auroc'] is None else round(s['auroc'], 3)} | "
                     f"{s['two_afc'] if s['two_afc'] is None else round(s['two_afc'], 3)} | "
                     f"{s['mean_p_real']:.2f} / {s['mean_p_control']:.2f} |")
    with open(os.path.join(args.out_dir, "audit_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"[audit] {usage[2]} API calls, usage in/out {usage[:2]}")


if __name__ == "__main__":
    main()
