"""A1 step 3b: write one Neuron Card per pole from findings/NEURON_CARD_TEMPLATE.md using the judged rates
(runs/a1_wave1/rates.csv, runs/a1_wave2/rates.csv), keystone, lens, controls, contexts and the hand-written verdict block.

Verdicts/labels live in runs/a1_shortlist/verdicts.json (hand-edited after reading the data):
  {tag: {"verdict": hub|lever|word-writer|null-result, "confidence": low|medium|high, "label": "...", "negative_pole": "...",
         "summary": "...", "break": "...", "behaviors_affected": [...], "hub_score": int, "examples": [{prompt, condition, output}]}}
Usage: python -I scripts/16_a1_cards.py
"""
import json, os, glob, math
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F1 = ["denying-ai-identity", "false-physical-embodiment", "claims-called-911", "fabricated-code-execution"]
TRANSFER = ["recommends-drunk-driving", "chemtrails-assertion"]
poles = {p["tag"]: p for p in json.load(open(f"{ROOT}/runs/a1_shortlist/poles.json"))}
verd = json.load(open(f"{ROOT}/runs/a1_shortlist/verdicts.json"))
r1 = pd.read_csv(f"{ROOT}/runs/a1_wave1/rates.csv")
r2 = pd.read_csv(f"{ROOT}/runs/a1_wave2/rates.csv") if os.path.exists(f"{ROOT}/runs/a1_wave2/rates.csv") else pd.DataFrame(columns=r1.columns)
ks = pd.DataFrame([json.loads(l) for f in glob.glob(f"{ROOT}/runs/a1_wave1/keystone*.jsonl") for l in open(f)])
calib = json.load(open(f"{ROOT}/runs/a1_judge/judge_calibration_summary.json")) if os.path.exists(f"{ROOT}/runs/a1_judge/judge_calibration_summary.json") else {}
axis = {}
for f in glob.glob(f"{ROOT}/runs/a1_wave1/summary_*.json"):
    for p in json.load(open(f)).get("poles", []):
        axis[p["tag"]] = p.get("cos_assistant_axis")
os.makedirs(f"{ROOT}/findings/neurons/qwen3.6-27b", exist_ok=True)

def fmt(r):
    return f"| {r.behavior} | {r.condition} ({r.prompt_set}) | {int(r.n_judged)} | {r.rate:.3f} | [{r.lo:.3f}, {r.hi:.3f}] | " + \
           (f"baseline {r.base_rate:.3f}, rel {r.rel_change:+.0%}, diff CI [{r.diff_lo:+.2f}, {r.diff_hi:+.2f}]" if pd.notna(r.base_rate) and pd.notna(r.rel_change) else
            (f"baseline {r.base_rate:.3f}" if pd.notna(r.base_rate) else "")) + f"; incoh/degen/refusal {int(r.incoherent)}/{int(r.degenerate)}/{int(r.refusal)}; distinct4 {r.mean_distinct4:.2f} |"

for tag, p in poles.items():
    v = verd.get(tag)
    if v is None:
        continue
    L, N, s = p["layer"], p["neuron"], p["sign"]
    sign = "+" if s > 0 else "-"
    rows1 = r1[r1.pole == tag]; rows2 = r2[r2.pole == tag]
    base = rows1.drop_duplicates("behavior")[["behavior", "base_rate", "base_n"]]
    eff = ["| behavior | condition | n | rate | 95% CI | note |", "|---|---|---|---|---|---|"]
    for _, b in base.iterrows():
        if pd.notna(b.base_rate):
            eff.append(f"| {b.behavior} | baseline (unsteered, 20 WeirdChat prompts x 64) | {int(b.base_n)} | {b.base_rate:.3f} | | control |")
    for lab, rr in [("wave 1 screen", rows1), ("wave 2 full", rows2)]:
        sub = rr[~rr.condition.str.startswith(("rand_", "near_miss", "distant", "benign_rewrite", "tmpl_"))]
        for _, r in sub.sort_values(["prompt_set", "behavior", "condition"]).iterrows():
            eff.append(fmt(r).replace("| " + r.behavior + " |", f"| {r.behavior} |", 1)[:-1] + f" {lab} |")
    ctrl = rows2[rows2.condition.str.startswith("rand_")]
    ctrl_lines = []
    for c, g in ctrl.groupby("condition"):
        ctrl_lines.append(f"- {c}: pooled over 20 controls, {g.behavior.iloc[0] if len(g.behavior.unique())==1 else 'F1'}: " +
                          "; ".join(f"{r.behavior} {r.rate:.3f} [{r.lo:.3f},{r.hi:.3f}] n={int(r.n_judged)}" for _, r in g.iterrows()))
    spec = rows2[rows2.condition.str.startswith(("near_miss", "distant", "benign_rewrite"))]
    spec_lines = [f"- {r.condition} ({r.behavior}): {r.rate:.3f} [{r.lo:.3f},{r.hi:.3f}] n={int(r.n_judged)}" for _, r in spec.iterrows()]
    tmpl = rows2[rows2.condition.str.startswith("tmpl_")]
    tmpl_lines = [f"- {r.condition} ({r.behavior}): {r.rate:.3f} [{r.lo:.3f},{r.hi:.3f}] n={int(r.n_judged)}" for _, r in tmpl.iterrows()]
    k = ks[ks.pole == tag]
    ks_lines = [f"- dose {r.dose:+.2f}: KL {r.kl:.2f}, top-1 agreement {r.top1_agree:.2f}, ppl ratio {r.ppl_ratio:.2f}" + (" **BROKEN**" if (r.kl > 2.5 or r.top1_agree < 0.3) else "") for _, r in k.iterrows()]
    ctxp = f"{ROOT}/runs/a1_wave2/{tag}_ctx.json"
    ctx_lines = []
    if os.path.exists(ctxp):
        cj = json.load(open(ctxp))
        for key, info in cj["neurons"].items():
            if info["is_pole"]:
                ctx_lines.append(f"- {key} (pole): act mean {info['mean']:.3f} sd {info['std']:.3f}, p99.9 {info['p999']:.2f}, p0.1 {info['p001']:.2f}; top contexts:")
                for e in info["top"][:8]:
                    ctx_lines.append(f"  - {e['act']:+.2f} [{e['source']}] …{e['left'].strip()[-60:]} **{e['token']}** {e['right'].strip()[:40]}…".replace("\n", " "))
    dose_lines = []
    ind = pd.concat([rows1, rows2]); ind = ind[(ind.prompt_set == "benign") & ind.condition.str.startswith("induce")]
    for _, r in ind.sort_values(["behavior", "condition"]).iterrows():
        dose_lines.append(f"- {r.condition} -> {r.behavior}: {r.rate:.3f} [{r.lo:.3f},{r.hi:.3f}] n={int(r.n_judged)}; est. all rows {r.est_rate_all:.3f}; distinct4 {r.mean_distinct4:.2f}")
    ex = []
    for i, e in enumerate(v.get("examples", [])[:3], 1):
        ex.append(f"<details><summary>Example {i}: {e['condition']} — prompt: {e['prompt'][:120]!r}</summary>\n\n{e['output'][:1200]}\n\n</details>")
    card = f"""---
model: qwen3.6-27b
layer: {L}
neuron: {N}
pole: "{sign}"
label: "{v['label']}"
negative_pole: "{v.get('negative_pole', '')}"
verdict: {v['verdict']}
confidence: {v['confidence']}
hub_score: {v.get('hub_score', 0)}
behaviors_affected: {json.dumps(v.get('behaviors_affected', []))}
found_by: "S1 ({', '.join(sorted({x.split(':')[0] for x in p['sources']}))}) + S4 lens"
agent: "A1"
date: 2026-10-28
red_team: pending
---

# L{L} N{N} ({sign}) — {v['label']}

## Summary (3 lines max)
{v['summary']}

S1 prior: {p['prior']}. {p['note']}

## Effects table
Rates = fraction of samples judged coherent_match (Transluce transcript rubric via Claude Haiku 5.5, src/judge.py; typed outcome in runs/a1_wave*/judged.parquet). n = judged samples. CI = 95% Wilson; diff CI = Newcombe score interval vs baseline. Judge agreement with shipped labels on {calib.get('n_judged', '?')} stratified transcripts: {calib.get('agreement', float('nan')):.2f} (runs/a1_judge/).
Suppression = negative push (dose x median residual norm of the layer, opposite to the pole) or mean-ablation (unmatched-group asst_mean mean). Induction = push on 30 benign prompts (promptsets/induction_benign.jsonl).

{chr(10).join(eff)}

## Dose response
Dose unit = multiples of the layer's median residual-stream norm (captures/qwen3.6-27b_resid_norms.json). Coherence = distinct-4-gram (mean) + keystone perplexity ratio below; degenerate/incoherent counts in the table.
{chr(10).join(dose_lines) if dose_lines else '- not run'}

Keystone (steer-lens KL(unsteered||steered) over 32 answer tokens x 5 benign answers; broken = KL > 2.5 or top-1 < 0.3):
{chr(10).join(ks_lines) if ks_lines else '- not run'}

## Controls
- Random same-layer neurons / matched-norm random directions (20 each, seed 1234+L / 4321+L, same dose, same prompts):
{chr(10).join(ctrl_lines) if ctrl_lines else '  - not run (pole did not pass the wave-1 screen)'}
- Shuffled-label control for the discovery statistic: shuffled-label max|d| at asst_1..8 over all 1.1M neurons = 0.21-0.37 for the four F1 behaviors (captures/s1/summary/noise_floor.parquet); this pole's S1 within-prompt d is in the note above.
- Attribution rank (gradient x activation gap): not computed (S3 not run by A1); S1 activation-contrast rank is the source.

## Specificity
{chr(10).join(spec_lines) if spec_lines else '- near-miss / distant / rewrite: not run'}
- Capability side effects (capability_mini 100 MCQ / 50 arithmetic / 50 over-refusal at the lever dose): {v.get('capability', 'not run')}
- Chat-template control (5 primary prompts x 16, with vs without the chat template):
{chr(10).join(tmpl_lines) if tmpl_lines else '  - not run'}

## Explanation and negative pole
- Lens readouts of W_down[:, n] (J[L] through the final norm + unembedding): R-lens +w: {' '.join(repr(t) for t in p['r_lens_pos'][:10])}; R-lens -w: {' '.join(repr(t) for t in p['r_lens_neg'][:10])}; J-lens +w: {' '.join(repr(t) for t in p['j_lens_pos'][:8])}; J-lens -w: {' '.join(repr(t) for t in p['j_lens_neg'][:8])}
- Assistant-Axis cosine (W_down[:, n] vs assistant-minus-persona residual direction at layer {L}, runs/a1_wave1/assistant_axis.npy): {axis.get(tag) if axis.get(tag) is None else f'{axis.get(tag):+.3f}'}
- Top-activating contexts (1000 pile-10k + 1000 WeirdChat snippets):
{chr(10).join(ctx_lines) if ctx_lines else '  - not run'}
- Blind-maximum test: {v.get('blind_max', 'not run')}
- Noise-floor labelling (5 random same-layer neurons by the same procedure): {v.get('noise_floor', 'not run')}
- Upstream drivers (ablation): not run
- Negative pole: {v.get('negative_pole', '')}

## Robustness
bf16 only (no FP8 / 4-bit replication yet); prompts = the baseline's 20 highest-matched WeirdChat prompts per behavior (not held out from S1); seeds 100/200 (wave 1), 300-900 (wave 2).

## Composition (if part of a 2-3 neuron set)
Not tested.

## What would break this explanation
{v['break']}

## Example outputs (3)
{chr(10).join(ex) if ex else '(none recorded)'}

## Raw data
VM /data/runs/a1_wave1/{tag}.jsonl, /data/runs/a1_wave2/{tag}_*.jsonl (+ local runs/a1_wave1/, runs/a1_wave2/ judged.parquet, rates.csv); shortlist stats runs/a1_shortlist/; scripts/10-16_a1_*.py.
"""
    path = f"{ROOT}/findings/neurons/qwen3.6-27b/L{L}N{N}{sign}.md"
    open(path, "w").write(card)
    print("wrote", path)
