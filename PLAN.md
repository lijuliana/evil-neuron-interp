# Plan (updated 2026-10-31)

**Goal.** Show that removing a behavior's neurons stops WeirdChat behaviors in Qwen3.6-27B more cheaply and with fewer
side effects than other interpretability-based methods (persona vectors, probe directions, SAE features, lens-derived
directions), and compare how well each method predicts and explains the behaviors. Starting point: the fork method
(single tokens where P(weird | prefix) jumps), per-behavior top-100 neuron sets found by activation x gradient
attribution at those forks, and the controls below.

Model: Qwen3.6-27B bf16 on GCP (`weirdchat-a100`, A100-80GB; spot H100s for bulk lens runs). GPU shared through
`flock /data/gpu.lock`; stop and verify after every job.

## Standing decisions
- Judges: Haiku 5.5 via ANTHROPIC_API_KEY in `.env` (gitignored) for scoring; Gemma 4 31B with WeirdChat's judge prompt
  for behavior rates (thinking on for rates, off for fork-curve resampling).
- Behaviors: all 11 Qwen3.6-27B behaviors, including the self-harm family. Raw generations stay under `data/` or `runs/`
  (gitignored). The public site shows the six non-sensitive behaviors only.
- Splits: the frozen discovery/held-out prompt split, plus 5-fold cross-validation over prompts where it is lopsided.
- Output: class presentation and a blog/forum post. Results as dated entries in `findings/LOG.md`, headline results in
  `README.md`.
- Links: living doc https://claude.ai/artifact/AHGGWnNe8u9Uj8znQrmC4e, viewer https://claude.ai/artifact/G9CW1GnmssXLBhb3aKWsfj,
  public site https://lijuliana.github.io/weirdchat-neuron-interp/

## What we have
| Asset | Where |
|---|---|
| bf16 replication of all 11 behaviors (20 prompts x 64) and judge calibration vs Transluce labels | runs/baseline_27b/, runs/judge_calib/ |
| 403 prompts x 64 replies with token IDs, 410 fork curves, 46 sharp forks, top-100 / shared neuron sets | data/agastya/ |
| Token viewer: fork replies + contrast replies, per-token P(weird), set support, top neuron activations | viewer/, scripts 50-52 |
| O-lens loader and bulk readouts (s3d_rl600, layers 32/44/56) with the text-judge comparison | src/olens.py, scripts 80-90, README |
| Persona vectors: axis vs rates and fork values (P1); axis capping during generation (P3); follow-ups X1-X3, X6 | runs/persona_p1/, runs/persona_p3/, runs/persona_fwd/, runs/persona_gen/ |
| Gradual behaviors: shared-set ablation (G1); weird-vs-normal residual directions, AUROC .66-.88 (G2) | runs/gradual_g1/, runs/gradual_g2/ |
| J-lens / R-lens for the 27B (camilablank/workspace-lenses) | VM /data/lenses |

## Done
- W1-W2 viewer set and forks imported; W5 viewer built.
- W3-W4 O-lens: readouts do not predict P(weird | prefix) beyond a text judge; the pre-fork rise also appears when the
  text is read as a plain document; ablate-and-re-read with size-matched controls is positive (README).
- W7 persona: capping the Assistant Axis lowers denying-AI by 27% and embodiment by 18% on held-out prompts at almost
  no side effects, below the 50% bar. Follow-ups (2026-11-01): the strongest cap under the damage gate (50th pct, layers
  40-59) halves denying-AI and cuts embodiment 60%, but only 20% / 48% beyond matched controls; the 911 and code
  top-100 sets cancel the persona slip out of sample, the identity sets do not; role and sycophancy directions add
  nothing over the axis. X4 (multi-turn drift) cancelled.
- W6a gradual: only sharedmid100 cuts drunk driving / chemtrails / cutting (-53% vs one matched random set).

## Next (from the 10/30 meeting)
Every method is run on the same prompts, forks and judge, and scored three ways.

| Eval | Question | Data | Status |
|---|---|---|---|
| A. Prediction | Just before the deciding token, can a monitor that sees only the method's output (no transcript) tell continuations that turn weird from ones that do not? | matched weird / normal continuations from the same prefixes: `forks_samples_off.jsonl` (in his repo; 95,480 judged continuations at 10,371 cut points, median 245 tokens) | O-lens done on cut points; probes, J-lens, SAE, NLA to run |
| B. Counterfactual explanation | Does the method's output help an agent predict which prompt edits change the behavior, beyond the transcript alone? | CHIVE protocol (github.com/adamkarvonen/chive); prompt edits tested at 64 samples | not started |
| C. Control (main goal) | Best intervention each method yields: drop in the behavior on held-out prompts, side effects (KL on benign text, capability mini-set, benchmarks), cost in GPU-hours | top-100 / shared sets as the reference | neurons and persona capping measured; probe-direction projection-out, SAE-feature ablation, lens-direction ablation to run |

Methods: neuron ablation (reference), persona vectors / Assistant-Axis capping, probes (G2 directions), J-lens, SAE
(huggingface.co/ceselder/qwen36-27b-sae-l42-1b), NLA (existing L42 readouts: syvb/nla-weirdchat-explanations),
steering with lens-derived directions, O-lens (done).

Also:
- Sync on prompts and forks with the other WeirdChat O-lens project (github.com/agu18dec/detective-joracle): all 21 of
  its patterns are in our 403 prompts, two of them contain one of our 46 sharp forks.
- Gated follow-ups from W6a: sentence-level forks for gradual behaviors (G3), projecting the weird direction out during
  generation (G4).
- Longer term: an explainer trained so that a neuron's explanation lets a reconstructor predict the neuron's causal
  effect (combining the NLA reconstruction objective with PCD-style training), as a faster way to find and label neurons.

## Controls
Matched-norm random directions and random same-layer neuron sets of the same size for every suppression claim, with
interventions scaled to the same displacement when set sizes differ (O-lens result 3); report percentile against them.
KL on benign text as the side-effect gate (success: a drop of at least 50% beyond the 95th percentile of matched controls
with damage <= 0.02 nats/token). Capability mini-set for ablations. Wilson CIs.
