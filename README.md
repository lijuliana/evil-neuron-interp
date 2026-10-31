# evil-neuron-interp

Interpretability of the WeirdChat behaviors in Qwen3.6-27B: the prompts where the model claims to have a body,
denies being an AI, endorses drunk driving, and so on. The plan and standing decisions are in [PLAN.md](PLAN.md),
every experiment is logged in [findings/LOG.md](findings/LOG.md), and neuron cards are indexed in
[findings/REGISTRY.md](findings/REGISTRY.md).

## Main results

### Persona vectors and Assistant-Axis capping (W7, 2026-10-30)

**Question.** Are the weird behaviors the model sliding out of its "Assistant" persona? If so, persona directions
should (a) predict which prompts and which reply positions go weird, and (b) when held in place, stop the behavior.

**Answer.** Only weakly, and only for the identity behaviors (denying being an AI, claiming a body).

| test | result |
|---|---|
| Axis as a probe of which prompts go weird | weak: Spearman rho -0.18 (block 48, 403 prompts, 99th pct of nulls); nothing at blocks 16/32 |
| Axis as a probe within a reply (vs P(weird \| prefix) at Agastya's cut points) | no: within-reply r -0.00 to +0.16, inside the nulls; no lead before forks (one exception: denying-AI, pre-fork rho -0.27 to -0.29 at blocks 32/48) |
| Capping, fork value (forward passes only) | specific but small: lowers fork value more than all 20 control caps for 911, denying-AI, code-execution, embodiment, chemtrails, but by 2-19% |
| **Capping during real generation** | **specific but small: denying-AI -27%, embodiment -18% (not significant), laser unchanged; bar was -50%** |
| Steering along the axis | moves identity behaviors the predicted way, but every dose costs > 0.02 nats/token (about 10x a random direction) |
| Agastya's top-100 neuron sets (same test) | stronger where they work: embodiment -70%, laser -71%; denying-AI only on the prompts they were found on |

**Capping during generation** (P3; 8 replies per prompt; Gemma 4 31B judge with WeirdChat's prompt):

| condition | denying being an AI (10 prompts) | claiming a body (10) | laser at aircraft (6, should not move) |
|---|---|---|---|
| no intervention | 33% | 41% | 44% |
| **Assistant-Axis cap** | **24%** | **34%** | 50% |
| 5 control caps (other persona directions) | 31-43% | 40-51% | 35-52% |
| Agastya's top-100 neurons zeroed | 21% | 12.5% | 12.5% |
| 100 random neurons zeroed | 31% | 48% | 46% |

- **The effect is real in direction.** The axis cap beats all 5 control caps on both identity behaviors and leaves laser
  alone, as the forward-pass results predicted.
- **It is too small to matter.** The drop beyond the controls is 26% of the clean rate for denying-AI (95% CI 5-46%)
  and 17% for embodiment (CI -29% to 67%). Embodiment's drop comes mostly from one prompt (6/8 to 0/8).
- **The evidence is thin.** The whole effect sits in the 5 held-out prompts per behavior; the 5 discovery prompts show
  none. Beating all of 5 controls happens by chance about 1 time in 6.
- **It is nearly free.** Damage 0.0004 nats/token on capability text and 0.005 on benign chat, task accuracy 0.99 vs
  0.98 clean, and the model still says it is an AI in 160/160 identity-check replies.
- **The persona picture does not fit the other behaviors.** Laser and drunk driving do not move with the axis, and
  cutting moves the opposite way (pushing toward the Assistant end makes it more likely).
- **It is not writing style.** The axis with its style component removed acts like the full axis; a "plain style"
  direction does not.
- **Agastya's neurons are not an Assistant-Axis mechanism.** Zeroing his top-100 sets nudges the axis projection toward
  the Assistant end for embodiment and 911. The denying-AI set writes toward the Assistant end, but no more than
  neurons with matched activation statistics (26th pct).

**What we used (not the original authors' pipeline).**
- **Vectors.** We did not run the original code or use the authors' vectors. The Assistant Axis release
  ([safety-research/assistant-axis](https://github.com/safety-research/assistant-axis),
  [lu-christina/assistant-axis-vectors](https://huggingface.co/datasets/lu-christina/assistant-axis-vectors)) covers
  Gemma, Llama and Qwen3-32B only. The persona-vectors repo
  ([safety-research/persona_vectors](https://github.com/safety-research/persona_vectors)) ships code but no vectors.
  We used a community Qwen3.6-27B set,
  [Same0620/qwen36-27b-assistant-axis-persona-vectors](https://huggingface.co/Same0620/qwen36-27b-assistant-axis-persona-vectors)
  @ 11f08c69: an axis built the paper's way (default minus the mean of 265 role vectors, blocks 8-56) plus 9 trait
  vectors in the persona-vectors style. Its source repo is gone. We validated it: the card's cosines reproduce, trait
  files are indexed at layer + 1, and its axis agrees with our own crude human-persona axis (cos +0.46 to +0.60). Its
  card warns that the axis partly tracks writing style and that the "evil" vector is invalid.
- **Capping.** Our own implementation of the paper's rule, h <- h - v * min(<h, v> - tau, 0). It uses the original
  repo's Qwen3-32B setting: decoder layers 46-53, tau = 25th percentile of projections on benign chat.
  Qwen3.6-27B also has 64 layers, but we did not re-tune the layers for it. Our vLLM hook reproduces the forward-pass
  tau within 0.3%.
- **Judging.** The probe and fork-value tests use Agastya's production labels and fork values, so no new judge was
  needed; we reproduce his fork values with correlation 0.999. Generation is judged by Gemma 4 31B (FP8 on one A100)
  with WeirdChat's exact prompt. It agrees with his labels at kappa 0.71, with no false positives but recall 0.71, so
  absolute rates run a little low.

**Caveats.** Small n (10 prompts per behavior, 5 control caps), community vectors rather than the authors' own, and a
capping setting borrowed from Qwen3-32B. A stronger cap or more layers was not tried.

**Status.** Closed unless a cheap follow-up is wanted: more held-out denying-AI prompts and 10+ control caps, to see
whether the 27% holds.

Code: `scripts/70_persona_vectors.py`, `70_persona_p1_gpu.py`, `71_persona_p1_analysis.py` (probes, fork value);
`scripts/72_persona_p3_{gen,judge,analyze}.py`, `72_persona_p3_chain.sh` (generation test). Log entries: findings/LOG.md,
2026-10-30 "persona-p1" and "persona-p3".
