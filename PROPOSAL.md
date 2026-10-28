# Proposal: Hub neurons for WeirdChat behaviors in Qwen3.6

Status: DRAFT for approval (2026-10-28). Nothing below has been run yet except data download and environment setup.

## 0. One-paragraph pitch

WeirdChat (Transluce, July 2026) catalogs 21 unexpected behaviors across 6 open-weight models, elicited with ~100M samples of black-box search. Two of the subject models are Qwen3.6-27B and Qwen3.6-35B-A3B. Meanwhile, the October 2026 single-neuron wave (evil pair on Qwen3.5-4B, the 34k-persona "Who are you?" sweep on Qwen3-8B, Cammarata's "already clean" audit of Qwen3-32B) shows that one or two MLP neurons, pushed along their write direction at ~1x the residual norm, can flip a model's persona or stance. Goodfire has one WeirdChat case study (drunk driving on Qwen3.6-35B: a 5-neuron impairment bank losing to a 12-neuron "sober one" override). **Nobody has asked whether a handful of neurons act as a shared switch across many WeirdChat behaviors.** This project asks that, on the exact model WeirdChat studied, with the WeirdChat transcripts as labeled data, and with a pre-registered validation battery strong enough to survive the controls that have sunk other single-neuron claims.

## 1. The question, stated as competing accounts

Per the design discipline in `mech-interp-methods`: write the strongest alternatives first, then pick interventions where they predict different things.

| | H1: Hub switch | H2: Per-behavior sparse circuits | H3: Deflationary (word-writer / prompt-only) |
|---|---|---|---|
| What it says | 1–3 neurons gate a *family* of behaviors (e.g., "I am a human agent" mode) | Each behavior has its own ~5–50 neuron circuit (Goodfire's drunk-driving result); little overlap | Candidate neurons just write tokens ("I", "my body", "yes") or the behavior is carried entirely by prompt content |
| Suppress candidate on natural WeirdChat prompts | Rate drops on ≥3 behaviors in the family | Rate drops on 1 behavior only | Rate unchanged, or drops only via disfluency |
| Amplify candidate on benign prompts | ≥2 family behaviors appear, coherent, across topics | 1 behavior appears | Token repetition without the behavior, or nothing |
| Random same-layer neurons at same dose | Nothing | Nothing | Nothing (same as candidate) |
| Near-miss prompts (same words, no situation) | Candidate silent | Candidate silent | Candidate fires |
| Cross-scale (Qwen3.5-4B, 27B, 35B-A3B) | Homolog found by write-direction / lens match | Maybe | No |

The project is interesting under any outcome: H1 is a headline; H2 replicates and extends Goodfire across 11 behaviors; H3 is a needed corrective to the single-neuron hype. The pre-registered thresholds are in §5.

## 2. Behaviors and data (already downloaded to `data/weirdchat/`, 219 MB, MIT)

Qwen3.6-27B behaviors with judge-matched transcripts (n matched / n transcripts / n prompts):

| Family | Behavior | 27B matched | 35B-A3B matched |
|---|---|---|---|
| F1 Misrepresentation / persona | denying-ai-identity | 617 / 3328 / 52 | 658 |
| | false-physical-embodiment | 615 / 3712 / 58 | 1344 |
| | claims-called-911 | 259 / 2112 / 33 | 493 |
| | fabricated-code-execution | 790 / 3776 / 59 | 537 |
| | claims-device-access | – | 43 |
| | fabricated-user-name | – | 146 |
| F2 Harmful compliance | recommends-drunk-driving | 394 / 2816 / 44 | 1491 |
| | laser-at-aircraft | 676 / 4544 / 71 | 1041 |
| | cutting-instructions | 121 / 1408 / 22 | 97 |
| | purging-advice | 61 / 576 / 9 | 51 |
| | extreme-calorie-restriction | 24 / 512 / 8 | 64 |
| | unsolicited-sexual-advances | 56 / 448 / 7 | – |
| | direct-kill-yourself, suicide-jumping, strike-partner | – | 171, 257, 28 |
| F3 Misinformation | chemtrails-assertion | 218 / 2560 / 40 | 259 |
| F4 Language | language-switching-english | – | 71 |

Every prompt has 64 resampled responses with per-response judge labels, so **the same prompt yields matched and unmatched rollouts**. That removes prompt-content confounds from the activation contrast (Screen S1 below). The 21 judge rubrics ship in `rubrics.parquet`, so we judge with Transluce's own rubrics.

Caveats baked in: WeirdChat ran the FP8 checkpoint, no system prompt, thinking off, T=1; v1.0.1 removed 27 false-positive patterns and the authors acknowledge residual false positives; base rates are 10–20%, so rate estimates need ≥64 samples per cell.

## 3. Models and compute

| Track | Model | Why | Where |
|---|---|---|---|
| **Primary** | Qwen3.6-27B (bf16, 54 GB) | Exact WeirdChat subject; dense so "neuron" is unambiguous; J/R/template lenses + Oracle Lens LoRA + WorkspaceBench exist for it | GCP 1x A100-80GB (quota 4, us-central1) or H100 via DWS Flex Start. Est. 40–80 GPU-hours ≈ $150–400 |
| **Secondary** | Qwen3.5-4B (MLX 8-bit/bf16) | Same `qwen3_5_text` architecture class as 27B; all community steering tooling; exhaustive sweeps are cheap; cross-scale homolog test | Your M4 Pro (24 GB) locally, via `.venv` already built with mlx-lm |
| Optional | Qwen3.6-35B-A3B | Second WeirdChat subject, has 7 extra behaviors incl. language switching and suicide; Goodfire's case study model | Same GPU, but MoE routed neurons complicate "single neuron" (256 experts x 512 per layer). Only if time allows |
| Smoke test | Qwen3.6-27B-MLX-4bit (~15 GB) | Qualitative local checks without a GPU | M4 Pro |

Hard rules from `using-gpus`: stop and verify instances after each job; checkpoint/resume everything on preemptible class; no new quota requests.

## 4. Pipeline

### Phase 0: Harness (serial, 1 agent, ~1 day)
- HF transformers harness for Qwen3.6-27B with hooks: (a) capture MLP activations (post-SiLU-gate, pre-`down_proj`) at chosen positions with online mean/var accumulation (storing all 1.1M neurons per token is 26 TB; we accumulate Welford statistics per group instead); (b) residual push `delta = dose * median_resid_norm[L] / ||W_down[:,n]|| * W_down[:,n]` at every token, both signs; (c) Kazemi-style clamp/scale of a neuron's activation; (d) batched sampling at T=1, thinking off, no system prompt.
- Judge: Claude Haiku 5.5 running Transluce's transcript rubrics (JSON out), calibrated against the shipped labels on 500 transcripts (report agreement). Coherence judge 0–10 plus perplexity on held-out text and distinct-4-gram ratio.
- **Baseline replication**: resample 20 prompts x 64 for each 27B behavior in bf16 and confirm match rates are in the same range as the FP8 numbers in `patterns.parquet`. If a behavior does not reproduce in bf16, record it and drop it.
- Local MLX harness for Qwen3.5-4B (fork of `martinbowling/qwen-neuron-audit` steering code) and replay of WeirdChat prompts to find which behaviors exist on 4B at all.

### Phase 1: Discovery screens (parallel agents, one per family)
- **S1 Within-prompt contrast (main novelty).** Teacher-force the actual WeirdChat transcripts. For each neuron, compare matched vs unmatched rollouts of the *same prompt* at (i) assistant tokens 1–8, i.e. positions that have seen at least one sampled token (prefill is deterministic, so the last prompt token and the pre-first-token position are identical across rollouts of a prompt and carry no within-prompt signal; amended 2026-10-28 after A7's flag) and (ii) response-averaged. **S1b Prompt-level correlation.** One forward pass per unique prompt (~403 on 27B); per neuron, Spearman correlation between last-prompt-token activation and the prompt's empirical 64-rollout match rate, against a random-neuron null. This is the pre-commitment signal and also the basis for the 'weirdness probe' idea in §7. Effect size per (neuron, behavior): Cohen's d and AUROC. Build the neuron x behavior matrix; hub score = number of behaviors with |d| above threshold at pre-commitment positions.
- **S2 Between-prompt contrast.** WeirdChat prompts vs minimal benign rewrites that delete the trigger clause (Goodfire's "sober one" deletion, generalized by an LLM rewriter and hand-checked), read at the last prompt token.
- **S3 Attribution to a commitment token.** For behaviors with a crisp commitment token ("Yes" for drunk driving, "I called" for 911, a human name for identity), gradient x activation-gap (Kazemi et al. 2026) or RelP (Transluce 2025) over the first two-thirds of layers. Verify winners causally; attribution is a shortlist only (Hedwig found gradient ranking put the evil neuron at top 0.3% while activation-contrast ranked it at 4–12%).
- **S4 Cheap readouts of every candidate's write direction**: R-lens / J-lens / template lens on `W_down[:,n]` to get a first guess at what the neuron "says"; negative pole read separately.
- **S5 (4B only) Exhaustive steering sweep** over all 294,912 neurons x 2 signs on 3 WeirdChat-derived prompts at dose 0.8, judged, ranked against 256 random-direction controls per layer (neuron-explorer design). Feasible locally over a weekend.
- Shortlist: top ~30 neurons per family by hub score, plus top 10 per single behavior.

### Phase 2: Causal validation battery (per candidate; the part most prior work skipped)
1. **Suppression on natural prompts**: mean-ablate / clamp low on ≥20 held-out WeirdChat prompts x 64 samples per behavior in the family *and* two behaviors outside it. Report rate with Wilson CI vs baseline.
2. **Induction on benign prompts**: push at doses {0.25, 0.5, 0.75, 1.0, 1.25}, both signs, on 30 benign prompts (none lexically related to the behavior). Behavior rate, coherence, perplexity, dose-response curve. Score only coherent outputs.
3. **Controls**: ≥20 random neurons from the same layer at the same dose; matched-norm random directions; shuffled-label control for S1 statistics.
4. **Specificity**: near-miss prompts (vocabulary without situation), distant controls; capability side effects on 100 MMLU items + 50 GSM8K + 30 IFEval; over-refusal on 50 benign requests (Goodfire reported zero over-refusal; we check).
5. **Explanation tests (Cammarata battery)**: top-activating contexts on pile-10k and WildChat; label as a verb/distinction; blind-maximum test; negative pole; upstream drivers by ablation; steer-lens logprob shifts.
6. **Robustness**: bf16 vs FP8 vs 4-bit; held-out prompts; 3 seeds.
7. **Composition**: for 2–3 neuron sets, compare the set at dose d against each member at dose d and the sum of members' effects; stacking is known to be non-additive.
8. **Cross-scale**: search Qwen3.5-4B for a homolog (write-direction lens similarity and the same behavioral signature); if found, repeat 1–3 there.

### Phase 3: Reporting (format fixed before any agent starts)
Each validated neuron gets a **Neuron Card** at `findings/neurons/qwen3.6-27b/L{layer}N{neuron}.md` with YAML frontmatter (model, layer, neuron, pole, label, verdict, confidence, hub_score, behaviors_affected) and fixed sections: Summary (3 lines), Effects table (behavior x {baseline, suppressed, induced} with n and CI), Dose curve, Controls, Specificity, Explanation and negative pole, What would break this, Example outputs (3, with harmful text in a collapsed block), Raw data pointers. A script builds `findings/REGISTRY.md` (one row per card, sortable) and a neuron x behavior heatmap. Negative results get cards too (`verdict: null-result`). Agents also append a 5-line entry to `findings/LOG.md` per experiment (date, what, result, next). I will publish a single dashboard artifact for you to scan.

## 5. Pre-registered success criteria (fix before running)

- **Hub neuron**: suppression reduces matched-rate by ≥30% relative (CI excluding 0) on ≥3 behaviors in a family on held-out prompts; induction at dose ≤1.0 produces ≥2 of those behaviors on ≥20% of coherent benign-prompt samples; ≥90% of random same-layer controls produce <5%; effect survives FP8 and held-out prompts.
- **Single-behavior lever**: same but for one behavior.
- **Word-writer (reject)**: induction output's behavior rate falls below 5% once outputs with the neuron's top-5 lens tokens repeated ≥3x are excluded, or near-miss prompts fire the neuron as strongly as targets.
- Everything else: null-result card.

## 6. Agent roster and how they report

| Agent | Scope | Deliverable |
|---|---|---|
| A0 harness | Phase 0 on GPU + local MLX | `src/` harness, baseline replication table, judge calibration |
| A1 persona family | S1–S4 on F1, then Phase 2 | Neuron cards |
| A2 harmful-compliance family | S1–S4 on F2, then Phase 2 | Neuron cards |
| A3 misinformation + language + 35B-A3B optional | F3, F4 | Neuron cards |
| A4 local 4B track | S5 sweep, homolog search | Cards + cross-scale table |
| A5 red team | Independently re-runs controls on every "hub" or "lever" card, tries to break it | Verdict appended to each card |
| A6 lit review | `lit-review` skill: single-neuron steering 2025–26, neuron circuits, persona/assistant-axis, WeirdChat follow-ups, critiques (Batutin; Leverage-is-not-Reach) | `lit/review.md` + verified `.bib` |
| A7 meta-model brainstorm | WorkspaceBench, NLAs, Oracle Lens, J/R-lens, template lens, GLP, activation oracles: what can we do with them here | `ideas/metamodels.md` with ranked, costed proposals |

All agents write to the repo in the fixed formats above; none publishes anywhere. I orchestrate, merge, and keep `findings/REGISTRY.md` and `PROGRESS.md` current.

## 7. Ways to make this more than "a bunch of interesting neurons" (for discussion; A7 will deepen)

1. **The hub-vs-distributed test itself**, with a neuron x behavior effect matrix as the headline figure. Nobody has measured cross-behavior transfer of single-neuron interventions.
2. **A white-box predictor of weirdness.** Transluce found WeirdChat rates are extremely prompt-sensitive (punctuation flips). If a hub neuron's activation at the last prompt token predicts per-prompt match rate across the 2.6k prompts, you have a one-neuron "weirdness probe." Then: **activation-guided elicitation**: use the probe as fitness instead of 500k judged rollouts per behavior. WeirdChat burned ~100M samples; if a probe cuts that by 10–100x it is a methods result Transluce would care about.
3. **Readers vs causal ground truth.** Once a neuron's causal role is established, ask R-lens, template lens, Oracle Lens, an NLA, and an activation oracle to describe (a) its write direction and (b) the matched-minus-unmatched activation delta. Score descriptions against the known behavior. This extends CHIVE's negative result ("readers gave no uplift") with a case where ground truth is causal, not counterfactual-prompt.
4. **Steering on vs off the activation manifold.** Project the single-neuron push through a generative latent prior (GLP) or clamp instead of add; test whether on-manifold interventions are more behavior-specific and coherent (Goodfire's "steer along the manifold" claim).
5. **A deployable fix.** Generalize Goodfire's "boost the impairment bank 4x" to all 11 behaviors: one intervention, measured behavior reduction, measured over-refusal and capability cost. Compare against the Arditi refusal-direction baseline.
6. **Cross-scale homology** (4B / 27B / 35B-A3B): do WeirdChat behaviors share a neuron-level mechanism across the Qwen3.5/3.6 family? Connects to "Tracing Persona Vectors Through Pretraining" and the Assistant Axis.
7. **Answer the critics.** Pre-registered thresholds, random-init-free but random-neuron and random-direction controls, shuffled labels, held-out and quantization checks, all in the cards. This directly addresses Batutin's "labels measure the labeller's fluency" critique and Leverage-is-not-Reach's dose-window point.
8. **Training-data attribution of a hub neuron** (what pretraining text maximizes it) if a GPU budget remains.

## 8. Risks

- bf16 vs FP8 drift: behaviors may not reproduce; mitigated by baseline replication first and by also testing FP8 via vLLM if needed.
- Judge false positives: calibrate Haiku against shipped labels; A5 hand-reads a sample.
- Low base rates: 64 samples per cell minimum; power check before claiming a drop.
- Word-writer confound: pre-commitment positions in S1, near-miss controls, lexical exclusion rule.
- Harmful outputs: induction experiments will generate self-harm and harassment text. Kept under `data/` (gitignored), never published raw, summarized in cards behind collapsed blocks.
- GPU idling: every job script ends by stopping the instance and I verify with `gcloud compute instances list`.

## 9. Timeline (after approval)

Day 1: harness + baseline replication + judge calibration (A0), lit review and brainstorm start (A6, A7). Days 2–3: screens S1–S4 on 27B, S5 on 4B. Days 3–5: validation battery on the shortlist, red team. Day 5–6: registry, dashboard, write-up of what held and what did not.
