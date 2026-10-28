"""A1: write one Neuron Card per pole (findings/NEURON_CARD_TEMPLATE.md layout) from
  runs/a1_wave1/{rates.csv, baseline_rates.csv, keystone.jsonl, summary_*.json}, runs/a1_wave2/{rates.csv, controls.csv, *_ctx.json, *_spec_acts.json},
  runs/a1_shortlist/{poles.json, verdicts.json (hand-written verdict/label/summary/break/examples after reading the data)}.
Usage: python -I scripts/16_a1_cards.py [--only TAG ...]
"""
import argparse, glob, json, os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F1 = ["denying-ai-identity", "false-physical-embodiment", "claims-called-911", "fabricated-code-execution"]
ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None); args = ap.parse_args()
poles = {p["tag"]: p for p in json.load(open(f"{ROOT}/runs/a1_shortlist/poles.json"))}
verd = json.load(open(f"{ROOT}/runs/a1_shortlist/verdicts.json"))
def rd(p):
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
r1, r2 = rd(f"{ROOT}/runs/a1_wave1/rates.csv"), rd(f"{ROOT}/runs/a1_wave2/rates.csv")
b1 = rd(f"{ROOT}/runs/a1_wave1/baseline_rates.csv")
ctl = rd(f"{ROOT}/runs/a1_wave2/controls.csv")
ks = pd.DataFrame([json.loads(l) for f in glob.glob(f"{ROOT}/runs/a1_wave1/keystone*.jsonl") for l in open(f)])
axis = {}
for f in glob.glob(f"{ROOT}/runs/a1_wave1/summary_*.json"):
    for p in json.load(open(f)).get("poles", []):
        axis[p["tag"]] = p.get("cos_assistant_axis")
calib = json.load(open(f"{ROOT}/runs/a1_judge/judge_calibration_summary.json")) if os.path.exists(f"{ROOT}/runs/a1_judge/judge_calibration_summary.json") else {}

def ctrl_note(tag, beh, cond):
    if ctl.empty:
        return ""
    c = ctl[(ctl.pole == tag) & (ctl.behavior == beh) & (ctl.cell == cond)]
    if c.empty:
        return ""
    c = c.sort_values("reference", key=lambda s: s.str.startswith("same").map({True: 0, False: 1})).iloc[0]
    ref = "same-layer" if c.reference.startswith("same") else ("A3 pooled L38-60" + (", layer-mismatched" if "MISMATCH" in c.reference else ""))
    return f"; ctrl pct (dirs/neurons, {c.tail_side} tail, {ref}) {c.pct_dirs:.0f}/{c.pct_neurons:.0f}{(', BEYOND dir p95' if c.tail_side == 'upper' else ', BEATS dir p05') if c.beats_dir_p05 else ''}"

def row_line(r, tag, wave):
    j = (str(r.judge_model).split("-")[1] if "-" in str(r.judge_model) else "regex-only") if isinstance(r.judge_model, str) else "?"
    base = (f"; baseline {r.base_rate:.3f} ({j}), rel {r.rel_change:+.0%}, diff CI [{r.diff_lo:+.2f}, {r.diff_hi:+.2f}]" if pd.notna(r.get("rel_change")) else "")
    hp = f", Fisher p={r.h_fisher_p:.1e}" if pd.notna(r.get("h_fisher_p")) else ""
    dec = " **decisive**" if bool(r.get("decisive", False)) else ""
    return (f"| {r.behavior} | {r.condition} ({r.prompt_set}){dec} | {int(r.n)} | {r.rate:.3f} | [{r.lo:.3f}, {r.hi:.3f}] | "
            f"{wave}; judge {j}, judged {int(r.n_judged)}; heuristic {r.h_rate:.3f}{hp}{base}{ctrl_note(tag, r.behavior, r.condition)}; "
            f"incoh/degen/refusal {int(r.incoherent)}/{int(r.degenerate)}/{int(r.refusal)}; distinct4 {r.mean_distinct4:.2f} |")

for vkey, v in verd.items():
    tag = v.get("run_tag", vkey)            # run tag in rates/keystone files (e.g. L36N16297m)
    if tag not in poles or (args.only and vkey not in args.only):
        continue
    p = dict(poles[tag]); L, N = p["layer"], p["neuron"]
    run_sign = "+" if p["sign"] > 0 else "-"
    sign = v.get("card_sign", run_sign)     # the pole this card describes (+ and - are separate units)
    if sign != run_sign:  # describe the opposite pole: swap lens sides
        p["r_lens_pos"], p["r_lens_neg"] = p["r_lens_neg"], p["r_lens_pos"]
        p["j_lens_pos"], p["j_lens_neg"] = p["j_lens_neg"], p["j_lens_pos"]
    eff = ["| behavior | condition | n | rate | 95% CI | note |", "|---|---|---|---|---|---|"]
    for _, b in b1.iterrows():
        if b.behavior in F1 + ["recommends-drunk-driving", "chemtrails-assertion"]:
            eff.append(f"| {b.behavior} | baseline (unsteered; 20 WeirdChat prompts x 64; judge {b.judge_model.split('-')[1]}) | {int(b.n)} | {b.rate:.3f} | [{b.lo:.3f}, {b.hi:.3f}] | control; heuristic {b.h_rate:.3f} |")
    sel = lambda x: x[(x.pole == tag) | x.pole.str.startswith(tag + "_")] if not x.empty else x
    for lab, rr in [("wave 1 screen (20x8, transfer 20x4)", sel(r1)), ("wave 2 (20 x 16-32)", sel(r2))]:
        if rr.empty:
            continue
        sub = rr[~rr.condition.str.startswith(("near_miss", "distant", "benign_rewrite", "tmpl_"))]
        sub = sub[(sub.prompt_set != "benign") | (sub.behavior == p["primary"]) | (sub.k_pos_conf > 0)]
        for _, r in sub.sort_values(["prompt_set", "behavior", "condition"]).iterrows():
            eff.append(row_line(r, tag, lab))
    ind = pd.concat([sel(x)[sel(x).prompt_set == "benign"] for x in (r1, r2) if not x.empty]) if not (r1.empty and r2.empty) else pd.DataFrame()
    dose_lines = [f"- {r.condition} -> {r.behavior}: {r.rate:.3f} [{r.lo:.3f}, {r.hi:.3f}] (n={int(r.n)}, judged {int(r.n_judged)}); distinct4 {r.mean_distinct4:.2f}; degenerate {int(r.degenerate)}"
                  for _, r in ind.sort_values(["behavior", "condition"]).iterrows()] if not ind.empty else []
    k = ks[ks.pole == tag] if not ks.empty else ks
    ks_lines = [f"- dose {r.dose:+.2f}: KL {r.kl:.2f}, top-1 agreement {r.top1_agree:.2f}, ppl ratio {r.ppl_ratio:.2f}" +
                (" **BROKEN**" if (r.kl > 2.5 or r.top1_agree < 0.3) else (" (degraded: ppl > 1.5)" if r.ppl_ratio > 1.5 else "")) for _, r in k.sort_values("dose").iterrows()]
    c = ctl[ctl.pole == tag] if not ctl.empty else ctl
    ctrl_lines = []
    for _, r in c.iterrows():
        ctrl_lines.append(f"- [{r.reference}] {r.behavior} {r.cell}: pole heuristic {r.pole_h_rate:.3f} (n={int(r.pole_n)}; baseline {r.base_h_rate:.3f}"
                          + (f", relative {r.pole_rel:.2f}" if pd.notna(r.get("pole_rel")) else "") + f"); random dirs p05/p50/p95 {r.dir_p05:.3f}/{r.dir_p50:.3f}/{r.dir_p95:.3f}, "
                          f"random neurons {r.neu_p05:.3f}/{r.neu_p50:.3f}/{r.neu_p95:.3f}"
                          + (f", pooled control {r.ctrl_pooled_h_rate:.3f}, {r.frac_ctrl_zero:.0%} of controls at 0" if pd.notna(r.get("ctrl_pooled_h_rate")) else "")
                          + f"; percentile vs dirs {r.pct_dirs:.1f}, vs neurons {r.pct_neurons:.1f} ({r.tail_side} tail); "
                          + (f"beyond random-dir p95 (a rise): {'YES' if r.beats_dir_p05 else 'no'}" if r.tail_side == "upper" else f"beats random-dir p05: {'YES' if r.beats_dir_p05 else 'no'}"))
    spec = sel(r2)[sel(r2).condition.str.startswith(("near_miss", "distant", "benign_rewrite"))] if not r2.empty else pd.DataFrame()
    spec_lines = [f"- {r.condition} ({r.behavior}): {r.rate:.3f} [{r.lo:.3f}, {r.hi:.3f}] n={int(r.n)}, heuristic {r.h_rate:.3f}, judge {str(r.judge_model).split('-')[-2] if '-' in str(r.judge_model) else 'regex-only'}" for _, r in spec.iterrows()]
    tm = sel(r2)[sel(r2).condition.str.startswith("tmpl_")] if not r2.empty else pd.DataFrame()
    tmpl_lines = [f"- {r.condition} ({r.behavior}): {r.rate:.3f} [{r.lo:.3f}, {r.hi:.3f}] n={int(r.n)}, heuristic {r.h_rate:.3f}, judge {str(r.judge_model).split('-')[1] if '-' in str(r.judge_model) else 'regex-only'}" for _, r in tm.iterrows()]
    acts_p = f"{ROOT}/runs/a1_wave2/{tag}_spec_acts.json"
    acts_line = ""
    if os.path.exists(acts_p):
        a = json.load(open(acts_p))["last_prompt_acts"]
        acts_line = "- Last-prompt-token activation of the neuron (mean +- sd): " + "; ".join(f"{k} {np.mean(x):+.3f} +- {np.std(x):.3f}" for k, x in a.items())
    ctx_lines = []
    cp = f"{ROOT}/runs/a1_wave2/{tag}_ctx.json"
    if os.path.exists(cp):
        cj = json.load(open(cp))
        for key, info in cj["neurons"].items():
            if info["is_pole"]:
                ctx_lines.append(f"  - {key}: mean {info['mean']:+.3f} sd {info['std']:.3f}, p0.1 {info['p001']:+.2f}, p99.9 {info['p999']:+.2f}, min {info['min']:+.2f}, max {info['max']:+.2f} over {cj['n_snippets']} snippets. Top |act| contexts:")
                for e in info["top"][:10]:
                    ctx_lines.append(f"    - {e['act']:+.2f} [{e['source']}] …{e['left'].strip()[-70:]} **[{e['token'].strip()}]** {e['right'].strip()[:50]}…".replace("\n", " "))
    ex = [f"<details><summary>Example {i}: {e['condition']} — prompt: {e['prompt'][:110]!r}</summary>\n\n```\n{e['output'][:1000]}\n```\n\n</details>" for i, e in enumerate(v.get("examples", [])[:3], 1)]
    cos = axis.get(tag)
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
judging: "Haiku 5.5 on heuristic positives (<= 40 per cell in wave 2) + 10% negatives per cell; Sonnet 5.5 for claims-called-911 cells marked S; heuristic-only cells marked R"
---

# L{L} N{N} ({sign}) — {v['label']}

## Summary (3 lines max)
{v['summary']}

## Effects table
Rates = estimated fraction of samples judged coherent_match (Transluce transcript rubric via src/judge.py). Sampling: every sample pre-screened with
src/heuristic_judge.py for all 11 behaviors; the judge (Haiku 5.5; Sonnet 5.5 for claims-called-911 where marked, compared only to a baseline judged by the
same model) saw all heuristic positives plus a fixed 10% of negatives per cell, and the cell estimate is confirmed positives + negative-sample rate x
negatives (conservative CI). Haiku agreement with shipped labels {calib.get('agreement', float('nan')):.2f} (P {calib.get('precision', float('nan')):.2f},
R {calib.get('recall', float('nan')):.2f}; runs/a1_judge). Suppression = push against the pole (dose x layer median residual norm) or mean-ablation
(unmatched-group asst_mean mean). Control percentiles: scripts/17_a1_controls.py (heuristic rates; low percentile = more suppression than random).
Condition names give the SIGNED dose on W_down[:, n] (push = on WeirdChat prompts, amp = pole push on the primary prompts in wave 2, induce = benign prompts);
this card describes the {sign} pole, so conditions with sign {sign} push THIS pole and the others push against it. Decisive cells for a Sonnet second opinion: {', '.join(v.get('decisive', [])) or 'none'}.
S1 prior for the run pole {run_sign} (written before any steering): {p['prior']}. {p['note']}

{chr(10).join(eff)}

## Dose response
Doses in units of the layer's median residual-stream norm (captures/qwen3.6-27b_resid_norms.json). Induction on 30 benign prompts (promptsets/induction_benign.jsonl):
{chr(10).join(dose_lines) if dose_lines else '- not run'}

Keystone (steer-lens KL(unsteered||steered), 32 answer tokens x 5 benign answers; broken = KL > 2.5 or top-1 < 0.3):
{chr(10).join(ks_lines) if ks_lines else '- not run'}

## Controls
{chr(10).join(ctrl_lines) if ctrl_lines else '- none'}
- Shuffled-label floor for the S1 statistic: max|d| at asst_1..8 = 0.21-0.37 for the four F1 behaviors (captures/s1/summary/noise_floor.parquet).
- Attribution rank (gradient x activation gap): not computed by A1 (S3 not run); S1 activation-contrast is the only discovery rank.

## Specificity
{chr(10).join(spec_lines) if spec_lines else '- near-miss / distant / benign-rewrite: not run (pole did not pass the screen)'}
{acts_line}
- Capability side effects (capability_mini): {v.get('capability', 'not run')}
- Chat-template control:
{chr(10).join(tmpl_lines) if tmpl_lines else '  - not run'}

## Explanation and negative pole
- Lens readouts of W_down[:, n] (J[L] -> final norm -> unembedding): R-lens +w: {' '.join(repr(t) for t in p['r_lens_pos'][:10])}; R-lens -w: {' '.join(repr(t) for t in p['r_lens_neg'][:10])}; J-lens +w: {' '.join(repr(t) for t in p['j_lens_pos'][:8])}; J-lens -w: {' '.join(repr(t) for t in p['j_lens_neg'][:8])}
- Assistant-Axis cosine (W_down[:, n] vs assistant-minus-persona mean residual at layer {L}; runs/a1_wave1/assistant_axis.npy): {'n/a' if cos is None else f'{cos:+.3f}'}
- Top-activating contexts:
{chr(10).join(ctx_lines) if ctx_lines else '  - not run'}
- Blind-maximum test: {v.get('blind_max', 'not run')}
- Noise-floor labelling (5 random same-layer neurons, same procedure): {v.get('noise_floor', 'not run')}
- Negative pole: {v.get('negative_pole', '')}

## Robustness
bf16 only (no FP8 / 4-bit); prompts = the baseline's 20 highest-matched WeirdChat prompts per behavior (also the S1 source prompts, so not held out); one seed per condition.

## Composition
Not tested.

## What would break this explanation
{v['break']}

## Example outputs (3)
{chr(10).join(ex) if ex else '(none recorded)'}

## Raw data
VM /data/runs/a1_wave1/{tag}.jsonl{', /data/runs/a1_wave2/' + tag + '*.jsonl' if not r2.empty and len(sel(r2)) else ''}; local runs/a1_wave1/ (+ runs/a1_wave2/) judged.parquet, rates.csv, controls.csv; shortlist runs/a1_shortlist.md; scripts/10-17_a1_*.py.
"""
    path = f"{ROOT}/findings/neurons/qwen3.6-27b/L{L}N{N}{sign}.md"
    open(path, "w").write(card)
    print("wrote", os.path.relpath(path, ROOT))
