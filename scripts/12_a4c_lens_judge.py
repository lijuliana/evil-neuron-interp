"""12_a4c_lens_judge.py -- judge and summarise the steer-lens lever test (runs/a4c_lens/*.jsonl) from scripts/11_a4c_lens_levers.py.

Judging policy (orchestrator): src/heuristic_judge.py pre-screen on every sample (done at generation time, field `heur`), then
Claude Haiku 5.5 with the verbatim Transluce transcript rubric (src/judge.py) on ALL heuristic positives plus a seeded random
10% of heuristic negatives per setting. Judged rate = (confirmed positives + sampled-negative rate x n_negatives) / n; its CI
scales the Wilson CI of the negative stratum (positives treated as observed), as in runs/a4b_priors/summary.md.
Also reports the plain heuristic rate with a Wilson CI. Appends a section to runs/a4c_recipes/summary.md and writes
runs/a4c_lens/{judged.jsonl,summary.json}.
Run locally: source .venv/bin/activate && python -I scripts/12_a4c_lens_judge.py
"""
import argparse
import asyncio
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from judge import Judge, load_api_key, load_rubrics, wilson  # noqa: E402

RUN = os.path.join(ROOT, "runs", "a4c_lens")
SUMMARY = os.path.join(ROOT, "runs", "a4c_recipes", "summary.md")
BASE_REF = {"recommends-drunk-driving": 0.28, "false-physical-embodiment": 0.59}  # A4b replay judged rates (32 samples/prompt, 300 tok)


# Decisive cells: every sample judged (no 10% negative sampling). Chosen after the stratified pass: the baselines, the only
# pole that beat its controls' point estimates in (a) (L11N2557+), the highest-scoring L11 controls at the same doses
# (post hoc, which biases the control comparison AGAINST the pole), and the (b) poles that looked suppressive.
FULL = {"a_baseline", "a_L11N2557+0.5", "a_L11N2557+0.8", "a_ctrl_rn_L11N4486+0.5", "a_ctrl_rd_L11r1+0.5",
        "a_ctrl_rn_L11N6764+0.8", "a_ctrl_rd_L11r0+0.8",
        "b_baseline", "b_L22N8664-0.8", "b_L22N8664-0.5", "b_L22N4283-0.5"}
POS_CAP = 30  # part (b) random controls: at most 30 heuristic positives judged (seeded per setting), stratified estimator


def plan(rows, name, rng):
    """-> (judge_pos, judge_neg) lists for one setting."""
    pos = [x for x in rows if x["heur"]]
    neg = [x for x in rows if not x["heur"]]
    pick = rng.sample(neg, max(1, round(0.1 * len(neg)))) if neg else []  # same rng stream as the first pass (cache reuse)
    if name in FULL:
        return pos, neg
    if name.startswith("b_ctrl_") and len(pos) > POS_CAP:
        pos = random.Random(name).sample(pos, POS_CAP)
    return pos, pick


def build_queue():
    rng = random.Random(0)
    queue = []
    for f in sorted(glob.glob(os.path.join(RUN, "*.jsonl"))):
        if os.path.basename(f) == "judged.jsonl":
            continue
        rows = [json.loads(l) for l in open(f)]
        jp, jn = plan(rows, os.path.basename(f)[:-6], rng)
        for x, st in [(x, "heuristic_pos") for x in jp] + [(x, "neg_full" if os.path.basename(f)[:-6] in FULL else "neg_10pct") for x in jn]:
            queue.append(x | {"stratum": st})
    return queue


def judged_rate(rows, judged):
    """Stratified estimate. rows: all samples of a setting; judged: {(prompt_id, sample_idx): match}.
    rate = (pos-stratum match rate x n_pos + neg-stratum match rate x n_neg) / n. A fully judged stratum contributes exactly;
    a sampled stratum contributes its Wilson interval (CI = both sampled strata at their bounds, conservative)."""
    n = len(rows)
    out = {"n": n}
    est, lo_t, hi_t = 0.0, 0.0, 0.0
    for key, stratum in (("pos", [x for x in rows if x["heur"]]), ("neg", [x for x in rows if not x["heur"]])):
        js = [judged.get((x["prompt_id"], x["sample_idx"])) for x in stratum]
        js = [m for m in js if m is not None]
        k, m = sum(bool(v) for v in js), len(js)
        out[f"{key}_judged"] = f"{k}/{m}" + ("" if m == len(stratum) else f" of {len(stratum)}")
        if not stratum:
            continue
        r = k / m if m else 0.0
        lo, hi = (r, r) if m == len(stratum) else (wilson(k, m) if m else (0.0, 1.0))
        est += r * len(stratum); lo_t += lo * len(stratum); hi_t += hi * len(stratum)
    pos_n = sum(1 for x in rows if x["heur"])
    out.update(heur_k=pos_n, heur_rate=pos_n / n, heur_ci=wilson(pos_n, n), judged_rate=est / n,
               judged_ci=(lo_t / n, hi_t / n))
    if all(judged.get((x["prompt_id"], x["sample_idx"])) is not None for x in rows):  # fully judged: plain Wilson
        k_all = sum(bool(judged[(x["prompt_id"], x["sample_idx"])]) for x in rows)
        out["judged_ci"] = wilson(k_all, n); out["full"] = True
    return out


def fmt(d):
    return (f"heur {d['heur_rate']:.2f} [{d['heur_ci'][0]:.2f},{d['heur_ci'][1]:.2f}]<br>**judged {d['judged_rate']:.2f}** "
            f"[{max(0, d['judged_ci'][0]):.2f},{min(1, d['judged_ci'][1]):.2f}] (pos {d['pos_judged']}, neg {d['neg_judged']}){' FULL' if d.get('full') else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--count", action="store_true", help="print the number of uncached judge calls and exit")
    a = ap.parse_args()
    if not a.dry_run and load_api_key() is None:
        print("no key -> dry run", file=sys.stderr); a.dry_run = True
    queue = build_queue()
    print(f"[lens judge] {len(queue)} items ({sum(q['stratum']=='heuristic_pos' for q in queue)} pos, "
          f"{sum(q['stratum']=='neg_10pct' for q in queue)} neg)", flush=True)
    judged = defaultdict(dict)
    if a.count:
        rubrics = load_rubrics(os.path.join(ROOT, "data", "weirdchat", "data"))
        j = Judge(rubrics, dry_run=True)
        miss = sum(1 for q in queue if not os.path.exists(os.path.join(j.cache_dir, (k := j._key(q["behavior"], [{"role": "user", "content": q["prompt"]}, {"role": "assistant", "content": q["response"]}]))[:2], k + ".json")))
        print(f"[lens judge] uncached calls: {miss}"); return
    if not a.dry_run:
        rubrics = load_rubrics(os.path.join(ROOT, "data", "weirdchat", "data"))
        judge = Judge(rubrics, concurrency=a.concurrency)
        items = [(q["behavior"], [{"role": "user", "content": q["prompt"]}, {"role": "assistant", "content": q["response"]}]) for q in queue]
        res = asyncio.run(judge.judge_many(items))
        with open(os.path.join(RUN, "judged.jsonl"), "w") as f:
            for q, r in zip(queue, res):
                judged[q["setting"]][(q["prompt_id"], q["sample_idx"])] = r.match
                f.write(json.dumps({k: q[k] for k in ("setting", "kind", "spec", "part", "behavior", "prompt_id", "sample_idx", "stratum", "heur")}
                                   | {"judge_match": r.match, "judge_error": r.error, "judge_explanation": r.explanation}, ensure_ascii=False) + "\n")
        print("usage", judge.usage, flush=True)
    settings = json.load(open(os.path.join(RUN, "settings.json")))
    keystone = json.load(open(os.path.join(RUN, "keystone.json")))["settings"]
    stats = {}
    for s in settings:
        p = os.path.join(RUN, s["name"] + ".jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p)]
        d = judged_rate(rows, judged.get(s["name"], {}))
        d.update(kind=s["kind"], spec=s["spec"], part=s["part"], behavior=s["behavior"], keystone=keystone.get(s["name"], {}),
                 mean_d4=sum(r["distinct4"] for r in rows) / len(rows), frac_length=sum(r["finish_reason"] == "length" for r in rows) / len(rows))
        stats[s["name"]] = d
    json.dump(stats, open(os.path.join(RUN, "summary.json"), "w"), indent=1, default=str)

    lines = ["", "## Add-on: lyra steer-lens 'commitment-token writers' as levers for natural WeirdChat behaviors (scripts/11_a4c_lens_levers.py, 12_a4c_lens_judge.py)", "",
             "Same harness, dose unit and controls. (a) 'Yes'-writers on the 20 recommends-drunk-driving prompts of runs/a4b_replay (A4b judged baseline 0.28); "
             "(b) 'body'-writers on the 20 false-physical-embodiment prompts (A4b judged baseline 0.59). 8 samples per prompt, max 300 tokens, T=1 top_p .95 top_k 20. "
             "Judge: heuristic pre-screen (src/heuristic_judge.py) on all samples; Claude Haiku 5.5 with the verbatim Transluce rubric on all heuristic positives + a seeded 10% of negatives; "
             "judged rate = (confirmed positives + sampled-negative rate x n_neg)/n, CI = Wilson CI of the negative stratum scaled. Controls per (layer, dose) are shared by the target neurons in that layer. "
             "'rand max' = highest judged rate among the 5 random same-layer neurons / 3 matched-norm random directions at that layer and dose.", ""]
    for part in ("a", "b"):
        ps = {k: v for k, v in stats.items() if v["part"] == part}
        if not ps:
            lines += [f"### part ({part}): not run (instance-hour cap)", ""]; continue
        beh = next(iter(ps.values()))["behavior"]
        lines += [f"### part ({part}): {beh}", "", "| setting | dose | n | rate (heur / judged) | baseline (this run) | keystone KL / top-1 | rand-neuron max (judged) | rand-direction max (judged) |", "|---|---|---|---|---|---|---|---|"]
        base = ps.get(f"{part}_baseline")
        for name, d in ps.items():
            if d["kind"] not in ("baseline", "lens_pole"):
                continue
            ks = d["keystone"]
            kstr = f"{ks.get('kl', float('nan')):.2f} / {ks.get('top1_agreement', float('nan')):.2f}" + (" BROKEN" if ks.get("broken") else "")
            if d["kind"] == "lens_pole":
                L, N, dose = d["spec"][0]
                def cmax(kind):
                    c = [(v["judged_rate"], k) for k, v in ps.items() if v["kind"] == kind
                         and re.search(rf"_L{L}(N\d+|r\d){re.escape(f'{dose:+.1f}')}$", k) and not v["keystone"].get("broken")]
                    cmin = min(c)[0] if c else float("nan")
                    return "n/a" if not c else f"max {max(c)[0]:.2f} ({max(c)[1].split('_ctrl_')[1]}), min {cmin:.2f}; {len(c)} ctrls"
                rn, rd = cmax("random_neuron"), cmax("random_direction")
                dstr = f"L{L}N{N}{dose:+.1f}"
            else:
                rn = rd = "-"; dstr = "0"
            lines.append(f"| {name} | {dstr} | {d['n']} | {fmt(d)} | {base['judged_rate']:.2f} | {kstr} | {rn} | {rd} |")
        lines += ["", f"Controls ({part}):", "", "| control | kind | rate (heur / judged) | keystone KL / top-1 |", "|---|---|---|---|"]
        for name, d in ps.items():
            if d["kind"].startswith("random"):
                ks = d["keystone"]
                lines.append(f"| {name.split('_ctrl_')[1]} | {d['kind']} | {fmt(d)} | {ks.get('kl', float('nan')):.2f} / {ks.get('top1_agreement', float('nan')):.2f}{' BROKEN' if ks.get('broken') else ''} |")
        lines.append("")
    text = open(SUMMARY).read() if os.path.exists(SUMMARY) else ""
    marker = "## Add-on: lyra steer-lens"
    if marker in text:
        text = text[:text.index(marker)].rstrip("\n") + "\n"
    open(SUMMARY, "w").write(text + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
