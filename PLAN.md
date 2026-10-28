# Plan (2026-10-28, supersedes PROPOSAL.md)

**Framing.** Levers are easy, mediators are hard. Pushing a neuron to *induce* a behavior is cheap and
saturated on small Qwens (neuronary, the evil neuron, our 4B replications). Finding neurons whose removal
*stops* a behavior the model does on its own is rare, and a dose-1.0 suppression result is uninterpretable
without matched-norm random-direction controls (A3: random pushes alone cut identity behaviors by 20-45 pp).
Agastya's fork method gives per-behavior mediators on Qwen3.6-27B. This plan builds on it, adds readouts
of what the model "knows" before it goes weird, and a viewer to look at all of it per token.

Model: Qwen3.6-27B bf16 on GCP `weirdchat-a100` (A100-80GB). Judge: Haiku 5.5, Sonnet 5.5 on decisive cells,
regex pre-screen first. GPU shared through `flock /data/gpu.lock`; idle watchers stop both boxes.

## What we already have
| Asset | Where |
|---|---|
| bf16 replication of all 11 Qwen3.6-27B WeirdChat behaviors (20 prompts x 64, 16/64 judged) | runs/baseline_27b/baseline_rates.csv |
| Judge calibration vs Transluce labels (agreement 0.77, precision 0.93, recall 0.57) | runs/judge_calib/ |
| Steering harness (push, clamp, mean-ablate, capture) | src/harness.py |
| Control lesson: random directions at dose 1.0 suppress F1 behaviors | runs/a3_matrix.png, findings/LOG.md |
| Replicated levers (4B evil, sarcastiqwen, yes-man; 27B evil L34N3838) | findings/REGISTRY.md, runs/a4c_recipes/ |
| J-lens / R-lens for 27B (camilablank/workspace-lenses) | VM /data/lenses |
| O-lens + AR LoRAs (agu18dec/olens_and_ar), prompt and injection contract | HF model card |

## Workstreams
| # | Workstream | Output | Needs | Cost |
|---|---|---|---|---|
| W1 | **Viewer set.** Re-sample ~10 matched + ~10 unmatched rollouts per behavior with exact token IDs saved (our baseline kept text only) | runs/viewer_set/*.jsonl | nothing | ~1 GPU-h |
| W2 | **Forks.** P(behavior given prefix up to token t) curves on the viewer set by resampling; locate sharp forks, find the no-sharp-fork cases | per-rollout fork curves | Agastya's 46 forks would save most of this | 2-3 GPU-h + ~$50 judge if redone |
| W3 | **O-lens readouts** at every response token (start with one layer in 44-52), validated first against Agastya's published Qwen3.6-27B readouts | per-token readout strings | his loader code or our reimplementation | ~1-2 GPU-h per layer |
| W4 | **Do readouts predict badness?** Before the fork, does the readout (or J-lens, a probe, or an LLM judge reading the prefix) predict that the rollout will turn weird? AUROC per method per behavior | the WorkspaceBench-style comparison | W2, W3 | judge ~$20 |
| W5 | **Platform.** Fork Agastya's static viewer (assistant-privilegeness/refusal) for WeirdChat: transcript, per-token readout panel, fork curve, judge label, top mediating neurons at the fork | published artifact or GitHub Pages | W1-W3 | CPU |
| W6 | **Mediator checks on Agastya's neuron sets.** Random 100-neuron and matched-norm controls; cross-behavior ablation (does removing behavior A's set reduce behavior B?); combined sets; the laser capability hit | cards + matrix | his top-100 sets per behavior | ~3 GPU-h |
| W7 | **Persona vectors.** How much of each behavior's per-prompt rate and fork is explained by Assistant-Axis / persona-vector projections | correlation + steering table | nothing | ~1 GPU-h |

Order: W1 -> W3 (validate) -> W2 or import forks -> W5 skeleton -> W4, W6, W7 in parallel.

## Dropped
- Activation-contrast shortlists (S1): structurally limited to token readers.
- Exhaustive lever sweep (A8): 78-156 GPU-h, and levers are not the question now. Its gradient stage output
  is kept on the VM at /data/runs/a8_sweep/attrib.

## Controls that carry over
Matched-norm random directions and random same-layer neuron sets of the same size for every suppression claim;
report percentile against them. Keystone/KL gate. Capability mini-set for ablations. Typed judge
(coherent match vs degenerate). Wilson CIs.
