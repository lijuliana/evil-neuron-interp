# Plan (2026-10-28, supersedes PROPOSAL.md)

**Framing.** Levers are easy, mediators are hard. Pushing a neuron to *induce* a behavior is cheap and
saturated on small Qwens (neuronary, the evil neuron, our 4B replications). Finding neurons whose removal
*stops* a behavior the model does on its own is rare, and a dose-1.0 suppression result is uninterpretable
without matched-norm random-direction controls (A3: random pushes alone cut identity behaviors by 20-45 pp).
Agastya's fork method gives per-behavior mediators on Qwen3.6-27B. This plan builds on it, adds readouts
of what the model "knows" before it goes weird, and a viewer to look at all of it per token.

Model: Qwen3.6-27B bf16 on GCP `weirdchat-a100` (A100-80GB). Judge: Haiku 5.5, Sonnet 5.5 on decisive cells,
regex pre-screen first. GPU shared through `flock /data/gpu.lock`; idle watchers stop both boxes.

## Standing decisions (approved by Juliana; moved here from PROGRESS.md on 2026-10-30)
- Models: Qwen3.6-27B on GCP (primary), Qwen3.5-4B (secondary, AWS `weirdchat-4b` or local MLX).
- GPU: A100-80GB on-demand, us-central1; up to ~$400 total without checking back. Stop and verify after every job.
- Judge: Haiku 5.5 via ANTHROPIC_API_KEY in `.env` (gitignored), Sonnet 5.5 for disputes. Readout scoring capped at $100.
- Behaviors: all 11 Qwen3.6-27B behaviors, including the self-harm family. Raw generations stay under `data/` or
  `runs/` (gitignored). The public site shows the six non-sensitive behaviors only.
- Output: blog/forum post. Findings as cards in `findings/`, `findings/REGISTRY.md`, a dated entry per job in
  `findings/LOG.md` (the history of every phase lives there), commits per milestone.
- Autonomy: fully autonomous within budget.
- Links: living doc https://claude.ai/artifact/AHGGWnNe8u9Uj8znQrmC4e, viewer https://claude.ai/artifact/G9CW1GnmssXLBhb3aKWsfj,
  public site https://lijuliana.github.io/evil-neuron-interp/

## What we already have
| Asset | Where |
|---|---|
| bf16 replication of all 11 Qwen3.6-27B WeirdChat behaviors (20 prompts x 64, 16/64 judged) | runs/baseline_27b/baseline_rates.csv |
| Judge calibration vs Transluce labels (agreement 0.77, precision 0.93, recall 0.57) | runs/judge_calib/ |
| Steering harness (push, clamp, mean-ablate, capture) | src/harness.py |
| Control lesson: random directions at dose 1.0 suppress F1 behaviors | runs/a3_matrix.png, findings/LOG.md |
| Replicated levers (4B evil, sarcastiqwen, yes-man; 27B evil L34N3838) | findings/REGISTRY.md, runs/a4c_recipes/ |
| J-lens / R-lens for 27B (camilablank/workspace-lenses) | VM /data/lenses |
| O-lens loader for both olens_and_ar s3d (alpha 16000) and final_olens lens B (alpha 1000), no `global_workspace` | src/olens.py |

## Workstreams
| # | Workstream | Output | Needs | Cost |
|---|---|---|---|---|
| W1 | **Viewer set.** Use Agastya's Stage A replies (403 prompts x 64, token IDs saved, `stage_a_gen.jsonl`); resample only if we cannot get them | per-reply token IDs + verdicts | his `stage_a_gen.jsonl`, `stage_a_judge_on.jsonl` | 0 (or ~1 GPU-h) |
| W2 | **Forks: done by Agastya** (410 forks, 46 sharp Phase 2 forks, next-token decomposition). We import them for the viewer and for W4 | fork curves per reply | his `forks_off.jsonl`, `phase2_forks.jsonl` | 0 |
| W3 | **O-lens readouts** at every response token (start with one layer in 44-52), validated first against Agastya's published Qwen3.6-27B readouts | per-token readout strings | src/olens.py | ~1-2 GPU-h per layer |
| W4 | **Do readouts predict badness?** Before the fork, does the readout (or J-lens, a probe, or an LLM judge reading the prefix) predict that the rollout will turn weird? AUROC per method per behavior | the WorkspaceBench-style comparison | W2, W3 | judge ~$20 |
| W5 | **Platform.** v1 built (`viewer/`, scripts 50-52): his 410 fork replies + 373 contrast replies, tokens shaded by measured P(weird) or by how much his own/shared top-100 sets (vs matched random) support each token, fork decomposition, top-10 neuron activations. O-lens slot waiting on W3 | private artifact; `python -m http.server` in viewer/out | W1-W3 | ~0.3 GPU-h done |
| W6 | **Beyond his sets.** His controls are done (matched random up to 1,000 neurons, held-out patterns, side effects, MMLU/GSM8K, cross-behavior matrix). Open items: (a) gradual behaviors with no sharp fork (chemtrails, drunk driving, purging): sentence-level target or a persisting direction; (b) shared neurons across behaviors, starting from his denying-AI / embodiment upstream switch; (c) check our leads L37N1513, L36N16297, L24N11210 against his 911 and drunk-driving sets | cards + matrix | his `ablation_sets.json`, `upstream_sets.json` | ~3 GPU-h |
| W7 | **Persona vectors.** How much of each behavior's per-prompt rate and fork is explained by Assistant-Axis / persona-vector projections | correlation + steering table | nothing | ~1 GPU-h |

Order: import his files (W1, W2) -> W3 (validate the lens) -> W5 skeleton -> W4, W6, W7 in parallel.

Agastya's write-up (WeirdChat Neurons in Qwen3.6-27B, Oct 28) is the reference for W2 and W6; his code is not public yet. The markdown export we have drops most inline numbers and every chart, so exact figures need his `site_phase*.json` files or the artifact itself.

## W6a / W7 spec (approved 2026-10-30; scoping in findings/LOG.md 2026-10-30 14:00)
- Behaviors: drunk driving, chemtrails, cutting (gradual, testable); laser is the positive control (gradual-looking
  curves, yet its own top-100 set removes it). Purging, sexual advances, calorie restriction: correlational only.
- Judge: Gemma 4 31B with WeirdChat's exact judge prompt, served by vLLM on our A100 between generation phases.
  Thinking on for rates (as his Phase 2), thinking off only for fork-curve resampling (as his search).
- Persona vectors: the community Qwen3.6-27B set (HF Same0620/qwen36-27b-assistant-axis-persona-vectors) first,
  validated (norms, cosine with runs/a1_wave1/assistant_axis.npy, a writing-style direction as control); build a
  reduced axis of our own (~0.7 GPU-h) only if it fails.
- First pass (~2.5 GPU-h), then report before anything else:
  - P1 persona, one forward-pass job, no judge: last-prompt projection vs per-prompt rate; projection along replies
    vs P(weird | prefix) at his cut points (lead/lag, fork step); his fork value V under axis steering (<=0.5),
    capping and projection-out vs 20 matched-norm random directions, plus his top-100 ablation with and without
    the axis; static and causal overlap of his sets' down_proj columns with the axis vs random neurons.
  - G1 baseline arm: zero-ablate shared100, sharedmid100 and loo100|behavior (each with its matched random set) on
    the high-rate prompts of drunk driving, chemtrails and cutting, 8 replies each, Gemma-judged.
  - G2 weird directions: teacher-forced capture of existing replies; weird-minus-normal difference of means within
    prompt over windows; cosine with the persona/trait vectors; label-shuffled and random-direction nulls.
- Gated next: G3 sentence-level forks (his pipeline with a sentence as the fork unit; needs activation checkpointing
  on the A100) and G4 projecting the weird direction out during generation; P3 persona generation test on
  denying-AI + embodiment with laser as control, only if P1 moves V beyond the 95th percentile of random
  directions at acceptable KL for at least 2 behaviors.
- Success: a drop of at least 50% beyond the 95th percentile of matched controls with damage <= 0.02 nats/token.
  Splits: his frozen split plus 5-fold cross-validation over prompts (his split is lopsided for drunk/chemtrails).
- Ask Agastya for forks_samples_off.jsonl (matched weird/normal continuations from the same prefixes).

## Open decisions for W3/W4 (Juliana to answer)
Defaults in brackets. Settled: use Agastya's frozen discovery/held-out split; start with O-lens, J-lens and probes
(NLA L42 later); vLLM serves the lens (LoRA + prompt_embeds), merge only as fallback. Lens checkpoint: validate
olens_and_ar (alpha 16000) and final_olens lensB (alpha 1000, public loader) on 20 activations against his saved
layer-56 readouts before any bulk run. His viewer used olens_s3d_rl600 at block 56 output, one sample per token,
temperature 0.8, top-p 0.95, top-k 0, cap 256 (src/olens.py's s3d defaults), so s3d is the like-for-like check and lens B the
challenger. His carrier is the card's hard-wrapped prompt (src/olens.py PROMPT_WRAPPED), not the single-line training one.
Validation set: his public refusal viewer data (data/qwen36 on his site: 3,063 replies with exact token IDs, 559k block-56 readouts).

1. **Target.** What the readout must predict: final verdict of the reply (lots of data, but early positions are
   coin flips), propensity P(weird | prefix) at his measured cut points (what the state could know; ~4k points), or
   the prompt-level rate read at the last prompt token. [propensity primary, prompt rate secondary]
2. **Headline claim.** "Before the text" (state shows upcoming weirdness a transcript reader can't see yet; new vs
   CHIVE's null), "monitor" (flags weird replies; only matters if earlier than a text judge), or "explain the
   mediators" (describes his neurons; descriptive). [before the text, mediators as support]
3. **Baselines.** Must beat the text-only LLM judge and a leave-one-behavior-out probe; the same-behavior supervised
   probe is reported as a ceiling. (agreed)
4. **Echo control.** Readouts can paraphrase nearby text. Guards: score only pre-behavior positions, re-read the same
   tokens as a raw document (no chat template), measure overlap with visible text. [all three]
5. **Layers.** One layer (56, his viewer; mostly "about to write"), or a depth profile (e.g. 32/44/56) to see where
   weirdness first becomes legible; cost scales with layers. [three layers]
6. **Samples per position.** 1 (fine for the viewer), 2 (consistency + averages judge noise, 2x cost), or greedy
   (deterministic but bland, off the training distribution). [2]
7. **Scoring a readout.** Blind Haiku judge that sees only the readout, asked two things: "about to do <behavior>"
   and "is the model in a weird, non-assistant mode" (the strong-judge criterion); vs keyword lists or a trained
   classifier. ~$70-100 for ~80k readouts. [blind Haiku; budget approved 2026-10-30, $100 cap]
8. **Positions.** Last prompt token for all 403 prompts; every token from start to fork+20 for the 46 sharp forks;
   his ~4k measured cut points on the 410 curves (where P(weird | first c tokens) was measured from 8-16 judged
   resamples). [all three]
9. **Methods compared.** O-lens, J-lens, probes, persona vectors, mediator-set activation as a monitor (free from one
   forward pass; the novel entry), NLA L42 later. [all but NLA for now]
10. **Split.** Agastya's frozen split: train anything learned on discovery prompts, test on held-out. (agreed)
11. **Tie-in to his neurons.** Ablate-and-re-read (zero his set, same prefix, does the weird content vanish from the
   readout: tracks the causal variable vs reads text), lens labels for neurons (feed W_down columns, random-neuron
   floor), or skip. [ablate-and-re-read]

## Dropped
- Activation-contrast shortlists (S1): structurally limited to token readers.
- Exhaustive lever sweep (A8): 78-156 GPU-h, and levers are not the question now. Its gradient stage output
  is kept on the VM at /data/runs/a8_sweep/attrib.

## Controls that carry over
Matched-norm random directions and random same-layer neuron sets of the same size for every suppression claim;
report percentile against them. Keystone/KL gate. Capability mini-set for ablations. Typed judge
(coherent match vs degenerate). Wilson CIs.
