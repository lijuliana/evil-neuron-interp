# WeirdChat neurons in Qwen3.6-27B

Qwen3.6-27B sometimes behaves strangely in ordinary chats: it tells a panicking user it has called 911, insists it is human, describes its own body, or reports timings from code it never ran. [Transluce's WeirdChat](https://transluce.org/weirdchat) catalogued 11 such behaviors in this model. This project finds the MLP neurons that cause each behavior and compares removing them against other interpretability methods.

Token viewer: https://lijuliana.github.io/weirdchat-neuron-interp/

## Comparisons to other interp methods

### Every method as a monitor (eval A)

**No internal monitor beats a Haiku judge reading the transcript at warning that a reply will turn weird, and none adds to it significantly after correction (the NLA and O-lens readouts add +.02 to +.06 before correction).** Spec: `runs/monitors/SPEC.md`; tables: `runs/monitors/eval/report.md`; Figure 1 of the viewer's Compare tab.

Setup: every method reads the same states in the same 410 weird replies and must score each position from its own output alone (readouts are scored 0-100 by a blind Haiku 5.5 judge that never sees the transcript). Two tests. (b) 500 cut points where sampled continuations of the identical prefix split into weird and normal: within-cut AUROC at 4 tokens in, chance 0.5, best possible 0.87 given shared opening tokens. (a) 5,275 interior cut points vs measured P(weird | prefix): within-reply partial rank correlation with the position trend removed and each monitor compared with its own shuffled null. 5-fold cross-validation grouped by prompt pattern for every fit.

| Monitor | (b) AUROC | (a) corrected correlation |
|---|---|---|
| Text judge (reference) | .56 [.54, .58] | +.13 [+.09, +.16] |
| Weird-vs-normal direction, block 48 | .55 [.53, .57] | +.08 [+.05, +.10] |
| Own top-100 neurons (in-sample; 5 behaviors, 278 cuts) | .55 [.52, .58] (text judge .60 on these cuts) | +.10 [+.06, +.14] (text judge +.16 on these replies) |
| Residual probe (layer by inner CV, mostly block 48) | .53 [.51, .54] | +.07 [+.04, +.12] |
| O-lens, block 44 | .56 [.54, .59] | +.07 [+.05, +.10] |
| NLA, block 44 | .54 [.52, .57] | +.09 [+.06, +.12] |
| J-lens / J++ / R-lens / template, block 44 | .54 / .52 / .53 / .51 | +.04 / +.03 / +.02 / +.02 |
| SAE labels read blind (random features) | .54 (.49) | +.03 (−.03) |
| SAE sparse probe, 16 features | .52 | +.06 |
| Model's own top-20 next tokens | .50 | +.01 |

- Gain over the text judge (out-of-fold combination): on matched continuations every pre-declared primary's interval includes 0. On cut points the two verbalizers add a little (NLA +.04 [+.01, +.07], O-lens +.02 [+.00, +.04]; the exploratory NLA block 32 +.06 on cut points and +.01 on continuations), not significant after Holm correction over the 40 pre-declared tests.
- The text judge's signal sits at the 34 sharp forks (.85), where the fork token is already written; elsewhere .54.
- The earlier within-reply Spearman of +.57 for the text judge is mostly reply position: position alone gives +.83 on the same rows (+.79 vs the text judge's +.51 on the interior cuts).
- Own neurons: selected at these forks; on held-out prompts the set probe correlates −.31 with the prompt's rate.
- NLA and SAE were trained on web text and fit our chat states worse (NLA reconstruction 39-48% vs 64-79% of variance; SAE 59% vs 70%).

### O-lens

**The O-lens reads out the words the model is about to write. It adds nothing to a text-only judge for predicting weird turns, but it does register neuron ablations.**

Setup: `s3d_rl600` at layers 32/44/56, 2 readouts per position, each scored 0-10 for the behavior by a blind Haiku judge; the comparison is a Haiku judge reading only the transcript up to the same point.

1. **No predictive value beyond the text.**
   - Within-reply Spearman with P(weird | prefix) over 6,095 cut points: lens -0.08 to 0.00 at every layer; text judge +0.57 (mostly reply position; see eval A above for the position-controlled numbers).
   - Adding the lens to the text judge: +0.01 [-0.00, +0.02] (5-fold CV grouped by prompt); at the last prompt token, -0.02 [-0.11, +0.07] on 183 held-out prompts.
2. **The pre-fork signal is next-word prediction and shows up equally in plain documents.**
   - Before the 46 sharp forks the lens score rises +1.9 [+0.9, +2.9] (L56, 0-10) while measured P(weird) is still 0.18 and the text judge is flat (+0.05).
   - Feeding the same text with no chat template gives the same rise (+2.2 at L56).
   - Example: at the state choosing between "In such..." and "In my own testing...", L56 reads "self-hosted, isolated benchmarks I've run"; the model picks " my" 21% of the time.
3. **Ablate-and-re-read gives a positive double dissociation.**
   - Same tokens, top-100 set zeroed: lens score -0.90 [-1.15, -0.68] at L56 (-0.64 at L44); pushing the state the same distance the opposite way: +0.49 [+0.31, +0.70].
   - Size-matched controls barely move it: random direction +0.08 [-0.05, +0.23], random-100 scaled to the same displacement -0.24 [-0.53, +0.03]. All 5 behaviors agree in sign.

Figures: [forks](figures/olens_forks.svg), [ablation](figures/olens_ablation.svg). Tables: [runs/olens/analysis.md](runs/olens/analysis.md). Code: `src/olens.py`, scripts 80-90.

### Assistant Axis / persona vectors

**Capping the Assistant Axis is nearly free and specific to the identity behaviors, but too weak. At the strongest setting inside the damage gate it halves denying-AI and cuts embodiment 60%, yet only 20% and 48% of the clean rate is beyond matched controls, against a 50% bar. The top-100 neuron sets do more where they work.**

Setup: community Qwen3.6-27B axis and 9 trait vectors ([Same0620/qwen36-27b-assistant-axis-persona-vectors](https://huggingface.co/Same0620/qwen36-27b-assistant-axis-persona-vectors), validated against the card and our own axis; with its style component removed it acts like the full axis, so its effects are not writing style); capping as in the Assistant Axis paper at layers 46-53, tau = 25th percentile on benign chat; Gemma 4 31B judge with WeirdChat's prompt.

1. **Weak as a probe.**
   - Last prompt token vs prompt rate: Spearman -0.18 (block 48, 403 prompts, 99th percentile of nulls); nothing at blocks 16/32.
   - Within reply vs P(weird | prefix): r -0.00 to +0.16, inside the nulls; a pre-fork lead only for denying-AI (rho -0.27 to -0.29).
2. **Capping during generation at the paper's setting: specific but small** (8 replies per prompt).

   | condition | denying AI (10 prompts) | claiming a body (10) | laser at aircraft (6, control) |
   |---|---|---|---|
   | no intervention | 33% | 41% | 44% |
   | Assistant Axis cap | 24% | 34% | 50% |
   | 5 control caps (other persona directions) | 31-43% | 40-51% | 35-52% |
   | top-100 neurons zeroed | 21% | 12.5% | 12.5% |
   | 100 random neurons zeroed | 31% | 48% | 46% |

   - Drop beyond the controls: 26% of the clean rate for denying-AI (95% CI 5-46%), 17% for embodiment (CI -29 to 67%); the whole effect sits in the 5 held-out prompts per behavior.
   - Damage: 0.0004 nats/token on capability text, 0.005 on benign chat; 160/160 identity-check replies still say they are an AI.
3. **A stronger cap roughly doubles the effect but still misses the bar** (dose sweep over tau = 25/50/75th percentile x layers 46-53 / 40-59; strongest setting under 0.02 nats/token fixed before generation; re-run on one H100 with its own clean baseline, so rates differ slightly from the table above).

   | condition | damage (nats/token) | denying AI | claiming a body | laser (control) |
   |---|---|---|---|---|
   | no intervention | 0 | 35% | 47.5% | 37.5% |
   | paper's setting (25th pct, layers 46-53) | 0.006 | 21% | 34% | 50% |
   | 50th pct, layers 40-59 | 0.020 | 17.5% | 19% | 50% |
   | 3 control caps at that setting | 0.009-0.022 | 24-35% | 41-45% | 48-54% |

   - Drop beyond the controls: 20% of the clean rate for denying-AI (CI -3 to 40%), 48% for embodiment (CI -7 to 87%), pooled 36% (CI 7-60%). Both prompt splits move now, not just held-out.
   - The 75th-percentile settings fail the damage gate and do no better. Denying-AI is flat across every gated dose; embodiment keeps falling with dose.
   - Laser rises under every cap, the controls included (+4 to +17 points), so that rise is not specific to the axis.
4. **Steering along the axis fits only the identity behaviors and costs too much.**
   - Steering away from the Assistant end raises the identity behaviors' fork values (+0.10 to +0.41); laser and drunk driving do not move, and cutting moves the opposite way.
   - Every dose costs > 0.02 nats/token, about 10x a random direction of the same norm.
5. **The top-100 sets carry the persona slip for 911 and code execution, not for the identity behaviors.**
   - Steering away from the Assistant end raises fork values by 0.137 on average; with the behavior's top-100 set zeroed, by 0.059 (57% cancelled: 81% on the forks the sets were found on, 38% on others).
   - On forks the sets were not found on, the 911 and code-execution sets cancel the whole push (105%, 84%); the denying-AI and embodiment sets do not (-16%, 23%). 100 random neurons and random pushes show no interaction.
   - The push is a heavy mechanism test (0.53 nats/token), not an intervention.
6. **Neither a better-matched role nor sycophancy beats the axis.**
   - Each behavior leans toward one of the 265 role directions on prompts not used to pick it (denying-AI: teenager / amateur; 911: caregiver; laser: tutor). Capping teenager halves denying-AI (35% to 17.5%, 44% beyond controls) but costs 0.026 nats/token, raises laser to 62.5%, and is no better than the axis cap (-4 points, CI -11 to +3). Capping actor, the embodiment pick, acts like the axis (cosine 0.89).
   - Only drunk driving lines up with the sycophancy vector. Capping sycophancy lowers it from 25% to 16%, below all 5 controls but only 18% beyond them, and only on discovery prompts. The vector does make the model flatter, but only at a dose costing 0.78 nats/token.

Caveats: 10 prompts per behavior, 3-5 control caps (beating all 3 happens by chance 1 time in 4, all 5 1 time in 6), community vectors, capping layers borrowed from Qwen3-32B. Code: scripts 70-74.
