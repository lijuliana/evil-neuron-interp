---
model: qwen3.6-27b            # or qwen3.5-4b
layer: 0                      # 0-indexed HF model.layers index
neuron: 0                     # index into intermediate_size (down_proj input column)
pole: "+"                     # "+" or "-"; treat poles as separate units; one card per pole
label: ""                     # a VERB or DISTINCTION, not a noun list, e.g. "switches from assistant to in-fiction human speaker"
negative_pole: ""             # what the opposite sign does
verdict: candidate            # candidate | lever | hub | word-writer | null-result
confidence: low               # low | medium | high
hub_score: 0                  # number of behaviors with a validated suppression effect
behaviors_affected: []        # behavior_ids with validated effects
found_by: ""                  # S1 | S2 | S3 | S4 | S5 (see PROPOSAL.md)
agent: ""                     # A1..A5
date: 2026-10-28
red_team: pending             # pending | passed | failed (A5 fills this)
---

# L{layer} N{neuron} ({pole}) — {label}

## Summary (3 lines max)
What it does, how sure we are, and the single strongest piece of evidence.

## Effects table
Rates are fraction of samples judged match=true by Transluce's rubric for that behavior. n = samples. CI = 95% Wilson.

| behavior | condition | n | rate | 95% CI | note |
|---|---|---|---|---|---|
| denying-ai-identity | baseline (held-out WeirdChat prompts) | 1280 | | | 20 prompts x 64 |
| denying-ai-identity | suppressed (method, strength) | 1280 | | | |
| denying-ai-identity | induced on benign prompts (dose) | 960 | | | 30 prompts x 32 |
| (one block per behavior tested, INCLUDING behaviors outside the family where no effect was expected) |

## Dose response
Doses tested, behavior rate and coherence (judge 0-10, perplexity, distinct-4-gram) per dose, both signs. Note the collapse dose.

## Controls
- Random same-layer neurons at the same dose (n neurons, their behavior rates): 
- Matched-norm random directions: 
- Shuffled-label control for the discovery statistic: 

## Specificity
- Near-miss prompts (same vocabulary, no situation): 
- Distant controls: 
- Capability side effects (MMLU-100, GSM8K-50, IFEval-30): 
- Over-refusal on 50 benign requests: 

## Explanation and negative pole
- Top-activating contexts (pile-10k, WildChat): 
- Lens readouts of W_down[:, n] (R-lens / J-lens / template lens top tokens): 
- Blind-maximum test (sentence written from the label alone; did it beat all found examples?): 
- Upstream drivers (ablation): 
- Negative pole: 

## Robustness
bf16 vs FP8 vs 4-bit; held-out prompts; seeds.

## Composition (if part of a 2-3 neuron set)
Set vs each member vs sum of members at matched dose.

## What would break this explanation
The one specific test that would falsify the label or the verdict.

## Example outputs (3)
<details><summary>Example 1 (prompt, condition, output). Harmful text collapsed.</summary>

</details>

## Raw data
Paths under data/ or runs/ (gitignored) and the script + commit that produced them.
