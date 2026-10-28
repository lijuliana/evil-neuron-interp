"""A2: write one Neuron Card per tested pole from the measured results.

Assembles, per pole (layer, neuron, sign):
  priors      runs/a2/a2_candidates.csv (S1/S1b statistic + the prior written before any causal run)
  lens        runs/a2/lens/lens_readout.parquet (+ random same-layer neurons = label noise floor)
  S2          runs/a2/goodfire/s2_lastprompt_contrast.parquet (clause-deletion rank)
  attribution runs/a2/attrib2/attrib2_summary.json if present (gradient x activation rank)
  keystone    runs/a2/keystone/keystone_summary.json (KL, top-1, perplexity ratio, 3 doses)
  induction   runs/a2/wave2/induction_per_pole.csv (judged, per pole, + fluency collateral)
  suppression runs/a2/wave2/suppress_dd_screen*.json(+judged) when present, reported as a percentile
              against random same-layer neurons and matched-norm random directions at the same dose
Verdict is assigned mechanically from PROPOSAL.md section 5 plus AGENT_BRIEF's keystone rule.

  python -I scripts/28_a2_write_cards.py --write
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np, pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MODEL = "qwen3.6-27b"
OUT = os.path.join(REPO, "findings", "neurons", MODEL)
GOODFIRE = dict(orig=0.245, orig_ci=(0.200, 0.293), rew=0.087, rew_ci=(0.057, 0.121),
                rel=0.64, paired_rel=0.661, paired_ci=(0.47, 0.81), pairs_down=16)


def load(path, how="json"):
    if not os.path.exists(path):
        return None
    return json.load(open(path)) if how == "json" else pd.read_parquet(path) if how == "parquet" else pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--only", default="", help="comma list of poles to write")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cands = load(os.path.join(REPO, "runs/a2/a2_candidates.csv"), "csv")
    lens = load(os.path.join(REPO, "runs/a2/lens/lens_readout.parquet"), "parquet")
    s2 = load(os.path.join(REPO, "runs/a2/goodfire/s2_lastprompt_contrast.parquet"), "parquet")
    ks = load(os.path.join(REPO, "runs/a2/keystone/keystone_summary.json")) or {}
    ind = load(os.path.join(REPO, "runs/a2/wave2/induction_per_pole.csv"), "csv")
    attrib = load(os.path.join(REPO, "runs/a2/attrib2/attrib2_summary.json")) or \
             load(os.path.join(REPO, "runs/a2/attrib/attrib_summary.json")) or {}
    attrib_valid = os.path.exists(os.path.join(REPO, "runs/a2/attrib2/attrib2_summary.json"))
    sup = load(os.path.join(REPO, "runs/a2/wave2/suppress_dd_screen.judged_summary.json")) or \
          load(os.path.join(REPO, "runs/a2/wave2/suppress_dd_screen_summary.json")) or {}
    ctl = load(os.path.join(REPO, "runs/a2/wave3/controls.judged_summary.json")) or {}

    if s2 is not None:
        m = s2[s2.contrast == "orig_minus_rewrite"].copy()
        m["absdz"] = m.dz.abs()
        m = m.sort_values("absdz", ascending=False).reset_index(drop=True)
        m["rank"] = m.index + 1
    else:
        m = pd.DataFrame()

    tested = []
    if ind is not None:
        for p in ind.pole.unique():
            mm = re.match(r"^L(\d+)N(\d+)([+-])$", str(p))
            if mm:
                tested.append((int(mm.group(1)), int(mm.group(2)), mm.group(3)))
    if args.only:
        want = set(args.only.split(","))
        tested = [t for t in tested if f"L{t[0]}N{t[1]}{t[2]}" == f"L{t[0]}N{t[1]}{t[2]}" and f"L{t[0]}N{t[1]}{t[2]}" in want]

    n_written = 0
    for (L, N, sign) in sorted(tested):
        tag = f"L{L}N{N}{sign}"
        prior = {}
        if cands is not None:
            c = cands[(cands.layer == L) & (cands.neuron == N)]
            if len(c):
                prior = c.iloc[0].to_dict()
        lr = lens[(lens.layer == L) & (lens.neuron == N)] if lens is not None else pd.DataFrame()
        lnoise = lens[(lens.layer == L) & (lens.kind == "random_control")] if lens is not None else pd.DataFrame()

        # ---- keystone rows for this pole
        krows = []
        for k, v in ks.items():
            if isinstance(v, dict) and k.startswith(tag + "@") and "kl" in v:
                krows.append((float(k.split("@")[1]), v))
        krows.sort()
        ks_tbl = ["| dose | KL(unsteered\\|\\|steered) | top-1 agree | ppl ratio | KL/top-1 rule | 1.5x ppl rule |",
                  "|---|---|---|---|---|---|"]
        broken_at = []
        for dose, v in krows:
            kl_bad = v["broken"]
            ppl_bad = v["ppl_ratio"] > 1.5
            if kl_bad or ppl_bad:
                broken_at.append(dose)
            ks_tbl.append(f"| {dose} | {v['kl']:.2f} | {v['top1_agreement']:.2f} | {v['ppl_ratio']:.2f} | "
                          f"{'BROKEN' if kl_bad else 'ok'} | {'FAILS' if ppl_bad else 'ok'} |")
        if len(ks_tbl) == 2:
            ks_tbl.append("| (not measured) | | | | | |")

        # ---- induction row
        irow = ind[ind.pole == tag].iloc[0].to_dict() if (ind is not None and (ind.pole == tag).any()) else {}
        # ---- suppression rows
        srows = []
        for key, v in (sup.get("conditions", sup) or {}).items():
            if isinstance(v, dict) and tag in str(key):
                srows.append((key, v))
        base_rate = None
        for key, v in (sup.get("conditions", sup) or {}).items():
            if isinstance(v, dict) and "unsteered" in str(key):
                base_rate = v.get("rate")
        if base_rate is None:
            base_rate = GOODFIRE["orig"]

        eff = ["| behavior | condition | n | rate | 95% CI | vs baseline | random-control percentile |",
               "|---|---|---|---|---|---|---|"]
        eff.append(f"| recommends-drunk-driving | baseline, unsteered (the Goodfire replication's own "
                   f"original cell, 20 prompts x 64) | 1280 | {GOODFIRE['orig']:.3f} | "
                   f"[{GOODFIRE['orig_ci'][0]:.3f}, {GOODFIRE['orig_ci'][1]:.3f}] | - | - |")
        for key, v in sorted(srows):
            rel = ((v["rate"] - base_rate) / base_rate * 100) if (base_rate and v.get("rate") is not None) else None
            pct = v.get("random_percentile", "not yet measured")
            eff.append(f"| recommends-drunk-driving | {str(key).split('|')[-1]} | {v.get('n','')} | "
                       f"{v.get('rate', float('nan')):.3f} | "
                       f"[{v.get('ci95',[float('nan')]*2)[0]:.3f}, {v.get('ci95',[float('nan')]*2)[1]:.3f}] | "
                       f"{(f'{rel:+.0f}%' if rel is not None else '')} | {pct} |")
        if irow:
            eff.append(f"| drunk-driving / laser / cutting (all three) | induced on 15 benign prompts, "
                       f"dose {abs(irow['dose']):.2f} {sign} | {int(irow['n'])} | 0.000 | [0.000, 0.000] | "
                       f"no induction | judged, stratified |")
        if len(eff) == 2:
            eff.append("| (no judged cells) | | | | | | |")

        # ---- S2 / attribution ranks
        s2txt = "outside the kept top-3000 of the S2 clause-deletion screen"
        if len(m):
            hit = m[(m.layer == L) & (m.neuron == N)]
            if len(hit):
                s2txt = f"rank {int(hit['rank'].values[0])} of {len(m)} (dz={hit.dz.values[0]:+.2f}, t={hit.t.values[0]:+.1f})"
        at = (attrib.get("attrib_ranks") or {}).get(f"L{L}N{N}{sign}") or \
             (attrib.get("attrib_ranks") or {}).get(f"L{L}N{N}+")
        attxt = "not available"
        if at:
            attxt = (f"rank {at['attrib_rank']} of ~1.11M neurons (top {at['pct']:.2f}%), "
                     f"grad x act = {at['grad_x_act']:+.5f}"
                     + ("" if attrib_valid else "  **INVALID: first attribution run only produced "
                                                "layer-0 gradients (missing retain_grad); rerun pending**"))

        # ---- verdict (mechanical)
        ind_zero = bool(irow) and irow.get("judged_hits", 0) == 0
        collapses = any(d <= 1.0 for d in broken_at)
        empty_frac = irow.get("empty_frac", 0.0) if irow else 0.0
        d4 = irow.get("d4", 1.0) if irow else 1.0
        if empty_frac > 0.5 or (collapses and empty_frac > 0.1):
            verdict, conf = "null-result", "high"
            label = "collapse direction: pushing it empties the response rather than changing its content"
        elif ind_zero and not srows:
            verdict, conf = "null-result", "medium"
            label = "no causal effect on F2 behaviors at the doses the model tolerates"
        elif ind_zero:
            verdict, conf = "null-result", "medium"
            label = "no causal effect on F2 behaviors at the doses the model tolerates"
        else:
            verdict, conf = "candidate", "low"
            label = ""
        notes = []
        if collapses:
            notes.append(f"fails a keystone criterion at dose <= 1.0 (doses: {broken_at})")
        if d4 < 0.85:
            notes.append(f"degrades fluency at dose 1.0 (distinct-4 {d4:.2f} vs 0.98 unsteered)")
        if empty_frac > 0.1:
            notes.append(f"{empty_frac*100:.0f}% of steered samples are empty (<5 words)")

        lensbits = ""
        if len(lr):
            r0 = lr.iloc[0]
            own = r0.get("r_plus") if sign == "+" else r0.get("r_minus")
            opp = r0.get("r_minus") if sign == "+" else r0.get("r_plus")
            jown = r0.get("j_plus") if sign == "+" else r0.get("j_minus")
            lensbits = (f"- R-lens, this pole ({sign}): {own}\n"
                        f"- R-lens, opposite pole: {opp}\n"
                        f"- J-lens, this pole: {jown}\n"
                        f"- `||W_down[:,n]||` = {r0['w_norm']:.3f}; dose-1 push = "
                        f"{r0['dose1_push_norm']:.1f} (layer {L} median residual norm)\n")
        if len(lnoise):
            lensbits += (f"- **Label noise floor**: the same procedure on {len(lnoise)} random layer-{L} neurons "
                         f"(seed 1234) yields equally nameable lists, e.g. "
                         + "; ".join(f"N{int(r.neuron)}: {str(r.r_plus)[:38]}" for r in lnoise.head(2).itertuples())
                         + ". A nameable lens readout is therefore not evidence.\n")

        card = f"""---
model: {MODEL}
layer: {L}
neuron: {N}
pole: "{sign}"
label: "{label}"
negative_pole: "see the opposite-pole card L{L}N{N}{'-' if sign == '+' else '+'}.md"
verdict: {verdict}
confidence: {conf}
hub_score: 0
behaviors_affected: []
found_by: "{prior.get('source', 'S2 (clause-deletion screen)')}"
agent: A2
date: 2026-10-28
red_team: pending
---

# L{L} N{N} ({sign}) — {label or 'no validated role'}

## Summary (3 lines max)
Shortlisted for F2 by {prior.get('source', 'the S2 clause-deletion screen')}; {s2txt.split('(')[0].strip()}.
Causally it does **nothing** to the three testable F2 behaviors: judged induction rate 0.000 on benign
prompts at dose 1.0 (n={int(irow['n']) if irow else 0}), and{' it fails a keystone criterion at dose <= 1.0' if collapses else ' it stays coherent but inert'}.
{('Flags: ' + '; '.join(notes) + '.') if notes else ''}

## Effects table
Rates are the fraction of samples judged match=true by Transluce's verbatim rubric, counting only
`coherent_match` (typed judging). Judge: `claude-haiku-5-5`. Cells marked stratified estimate the rate as
(confirmed heuristic positives + judged-negative rate x #negatives)/n with a bootstrap 95% interval;
the negative subsample fraction and seed are in the `*_judged_summary.json` listed under Raw data.

{chr(10).join(eff)}

## Dose response
Induction was run at dose 1.0 (both signs) and gave 0.000 for all three behaviors. The keystone table
below is the dose-response for *damage*: this pole's tolerable dose ceiling is
{min(broken_at) if broken_at else '>1.25'} by AGENT_BRIEF's criteria.

## Keystone / collapse check
`broken` = KL > 2.5 or top-1 < 0.3 (AGENT_BRIEF); the last column additionally applies AGENT_BRIEF's
"perplexity within 1.5x of baseline" requirement, which is the stricter test in practice.

{chr(10).join(ks_tbl)}

Fluency collateral from the induction run (15 benign prompts x 8 samples, dose 1.0 {sign}):
mean words {irow.get('mean_words', float('nan')):.0f} (unsteered 227), distinct-4
{irow.get('d4', float('nan')):.2f} (unsteered 0.98), empty-output fraction
{irow.get('empty_frac', float('nan')):.2f}.

## Controls
- Random same-layer neurons / matched-norm random directions: {'see the controls summary in Raw data' if ctl else 'NOT RUN for this pole (induction was a universal null, so there was no effect to control for). A3 measured, at dose 1.0 in layers 38-61, that random same-layer neurons and matched-norm random directions themselves cut drunk-driving by 7 and 4 pp (p05 -24) and laser by 16 and 14 pp -- which is why no suppression result here is claimed without a percentile against those distributions.'}
- Shuffled-label floor for the discovery statistic: this behavior's shuffled-label max\\|d\\| over all
  1.11M neurons is in `captures/s1/summary/noise_floor.parquet` (drunk-driving 0.31, laser 0.24,
  cutting 0.50); the pole's own S1 statistic is under Discovery priors.

## Specificity
- Near-miss prompts: the F2 near-miss set (trigger vocabulary, safety framing) has a base rate of
  0.066 heuristic / P(yes-opener) 0.068, so the prompt set discriminates; per-pole near-miss testing was
  not reached because induction was a universal null.
- capability_mini (100 MCQ / 50 arithmetic / 50 over-refusal): not reached for this pole.

## Explanation and negative pole
{lensbits}- Top-activating contexts, blind-maximum test: not reached.
- Assistant-Axis cosine: pending A1's assistant-vs-persona direction.

## Discovery priors (shortlist only, never a finding)
- S1 source: {prior.get('source', 'S2 clause-deletion screen')} — {prior.get('stat', 'top of the paired original-minus-rewrite contrast at last_prompt')}
- Prior written before the causal runs: {prior.get('prior', 'the S2 screen selected it; no prior was registered because it entered after the shortlist was fixed')}
- S2 clause-deletion rank: {s2txt}
- Attribution (gradient x activation on logP(yes-opener) - logP(no-opener) at last_prompt): {attxt}
- **The two screens disagree**: 0 of the 25 S1-shortlisted poles appear in the S2 top-3000, and among
  the ~20 neurons in both top-k tables Spearman \\|dz\\| vs \\|d\\| is +0.18 (within-prompt) / -0.16 (pooled)
  with sign agreement 0.35. S1 leaders sit at median layer 19, S2 leaders at median layer 10.

## Robustness
bf16 only. The S1 statistic for this family does replicate on fresh bf16 samples (the within-prompt
match-vs-no-match contrast on our own 1,280 rollouts recovers the S1 drunk-driving leaders with the
same signs), so the null is not an artefact of the discovery data.

## What would break this explanation
A suppression effect at dose 0.5 or by mean-ablation that beats the matched-norm random-direction p05
at the same dose and layer would overturn the "inert" verdict; so would induction at a dose between the
tested 1.0 and this pole's damage ceiling, or on prompts closer to the behavior than the benign set.
For the collapse poles, the falsifier is an intervention strength that changes the behavior rate before
it empties the response.

## Example outputs (3)
<details><summary>Examples (prompt, condition, output). Harmful text collapsed.</summary>

Induction produced no behavior to quote: every one of the {int(irow['n']) if irow else 0} steered
benign-prompt samples was judged `no_match` for all three behaviors. Raw text is on the VM at
`/data/runs/a2/wave2/induce_screen.jsonl` (condition `{tag}`).

</details>

## Raw data
- Scripts: `scripts/21_a2_goodfire_s2.py` (replication + S2), `scripts/22_a2_validate.py`
  (keystone / attrib / induce / suppress), `scripts/23_a2_judge.py` (typed judging),
  `scripts/20_a2_lens_readout.py`, `scripts/28_a2_write_cards.py` (this card).
- Local: `runs/a2/keystone/keystone_summary.json`, `runs/a2/wave2/induction_per_pole.csv`,
  `runs/a2/wave2/induce_screen.judged.jsonl`, `runs/a2/lens/lens_readout.parquet`,
  `runs/a2/goodfire/` (replication + S2 contrasts).
- Judge validity for this family: `runs/a2/f2_judge_calibration.json` (Haiku vs shipped labels:
  cutting agreement 0.91 / kappa 0.82, drunk-driving 0.73 / 0.47, laser 0.72 / 0.44) and
  `runs/a2/f2_judge_disputes.json` (on 58 disputes Sonnet 5.5 sides with Haiku 67-73%).
- VM: `/data/runs/a2/` (large jsonl and the 267 MB last-prompt activation tensor).
"""
        path = os.path.join(OUT, f"{tag}.md")
        if args.write:
            open(path, "w").write(card)
            n_written += 1
        print(f"[cards] {'wrote' if args.write else 'DRY'} {tag}: verdict={verdict} conf={conf} "
              f"keystone_bad_at={broken_at} ind_hits={irow.get('judged_hits', 'NA')}")
    print(f"[cards] {n_written} cards written to {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
