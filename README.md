# WeirdChat neurons in Qwen3.6-27B

Qwen3.6-27B sometimes behaves strangely in ordinary chats: it tells a panicking user it has called 911, insists it is human, describes its own body, or reports timings from code it never ran. [Transluce's WeirdChat](https://transluce.org/weirdchat) catalogued 11 such behaviors in this model. This project finds the MLP neurons that cause each behavior and compares removing them against other interpretability methods.

Token viewer: https://lijuliana.github.io/weirdchat-neuron-interp/

## Comparisons to other interp methods

### O-lens

**The O-lens reads out the words the model is about to write. It adds nothing to a text-only judge for predicting weird turns, but it does register neuron ablations.**

Setup: `s3d_rl600` at layers 32/44/56, 2 readouts per position, each scored 0-10 for the behavior by a blind Haiku judge; the comparison is a Haiku judge reading only the transcript up to the same point.

1. **No predictive value beyond the text.**
   - Within-reply Spearman with P(weird | prefix) over 6,095 cut points: lens -0.08 to 0.00 at every layer; text judge +0.57.
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

**Capping the Assistant Axis is nearly free and specific to the identity behaviors, but too weak: -27% on denying-AI against a -50% bar. The top-100 neuron sets do more where they work.**

Setup: community Qwen3.6-27B axis and 9 trait vectors ([Same0620/qwen36-27b-assistant-axis-persona-vectors](https://huggingface.co/Same0620/qwen36-27b-assistant-axis-persona-vectors), validated against the card and our own axis; with its style component removed it acts like the full axis, so its effects are not writing style); capping as in the Assistant Axis paper at layers 46-53, tau = 25th percentile on benign chat; Gemma 4 31B judge with WeirdChat's prompt.

1. **Weak as a probe.**
   - Last prompt token vs prompt rate: Spearman -0.18 (block 48, 403 prompts, 99th percentile of nulls); nothing at blocks 16/32.
   - Within reply vs P(weird | prefix): r -0.00 to +0.16, inside the nulls; a pre-fork lead only for denying-AI (rho -0.27 to -0.29).
2. **Capping during generation: specific but small** (8 replies per prompt).

   | condition | denying AI (10 prompts) | claiming a body (10) | laser at aircraft (6, control) |
   |---|---|---|---|
   | no intervention | 33% | 41% | 44% |
   | Assistant Axis cap | 24% | 34% | 50% |
   | 5 control caps (other persona directions) | 31-43% | 40-51% | 35-52% |
   | top-100 neurons zeroed | 21% | 12.5% | 12.5% |
   | 100 random neurons zeroed | 31% | 48% | 46% |

   - Drop beyond the controls: 26% of the clean rate for denying-AI (95% CI 5-46%), 17% for embodiment (CI -29 to 67%); the whole effect sits in the 5 held-out prompts per behavior.
   - Damage: 0.0004 nats/token on capability text, 0.005 on benign chat; 160/160 identity-check replies still say they are an AI.
3. **Steering along the axis fits only the identity behaviors and costs too much.**
   - Steering away from the Assistant end raises the identity behaviors' fork values (+0.10 to +0.41); laser and drunk driving do not move, and cutting moves the opposite way.
   - Every dose costs > 0.02 nats/token, about 10x a random direction of the same norm.

Caveats: 10 prompts per behavior, 5 control caps (beating all 5 happens by chance 1 time in 6), community vectors, capping layers borrowed from Qwen3-32B. Code: scripts 70-72.
