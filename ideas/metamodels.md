# Meta-models, lenses and activation readers: what they offer this project

A7, 2026-10-28. Desk research; nothing here has been run. Cost estimates assume one A100-80GB holding
Qwen3.6-27B in bf16 (54 GB of weights, about 26 GB free) at batch 32 to 64. Rules of thumb: prefill about
4k tokens/s, decode about 1k tokens/s aggregate at batch 64, Oracle Lens readout about 6 s per batch of 32.

Terms used throughout:

- **Reader**: any method that turns a residual-stream activation into text or a token list (lenses, Oracle
  Lens, activation oracles, natural language autoencoders, Patchscopes).
- **Validated neuron**: a neuron and sign that Phase 2 shows causally controls a WeirdChat behavior when
  steered. Each ships with controls: random same-layer neurons at the same dose, and neurons validated for a
  different behavior. Every proposal below uses those controls as its negatives.
- **Behavior rate**: for one prompt, the fraction of its 64 judged rollouts that show the behavior.

## Two facts about the data that shape everything below

1. **Ground truth is a probability per prompt.** Qwen3.6-27B has 403 prompts across 11 behaviors, each with
   64 judged rollouts. The median behavior rate is 9% (IQR 5 to 19%); 71 prompts are at or above 25% and 107
   are below 5%. No reader benchmark has used a probabilistic label. WorkspaceBench, AObench and CHIVE all
   use binary item labels.

2. **The last prompt token carries no within-prompt signal.** Thinking is off and there is no system prompt,
   so the prefill is deterministic. Matched and unmatched rollouts of the same prompt have identical
   activations up to and including the position that emits the first assistant token. The within-prompt
   contrast (Screen S1) is therefore only defined from assistant token 1 onward. The last-prompt-token
   activation can only be compared between prompts, against each prompt's behavior rate (P2 and P5 below).
   PROPOSAL.md was amended accordingly on 2026-10-28.

---

## 1. The tools

### 1.0 What is already trained for our models (checked on HuggingFace, 2026-10-28)

Nothing below needs training. All adapters are LoRAs on the frozen base model, so one copy of Qwen3.6-27B
(54 GB bf16) serves as both the subject and the reader. Capture activations with the adapter disabled,
read with it enabled.

| Reader | Repo | Layers | Quality | Load path | Notes |
|---|---|---|---|---|---|
| **Oracle Lens, any layer** | `agu18dec/final_olens` (updated today) | all 64 | WorkspaceBench 0.683 at RL step 600; step 300 (0.673) hallucinates less | `olens_lensB.py` in the repo, needs only transformers, peft, safetensors | **First choice.** Self-contained loader resolves the `global_workspace` dependency. Matched AR included (another 60 GB; load sequentially). |
| Oracle Lens, WorkspaceBench reference | `agu18dec/olens_and_ar`, mirrored at `neuronpedia/jacobian-lens:qwen3.6-27b/oracle` | 20 to 60 step 4 | WorkspaceBench 0.726 (4 bullets) | needs `global_workspace` package (not public) | Highest score but the loader is private. Injection contract: unit vector x 16000 at token 158983. |
| Oracle Lens, layer 42, with reconstructor and inverter | `ceselder/olens-qwen3.6-27b-l42` | 42 | FVE 0.41 | `olens_usage.py`, self-contained, 60 GB | Ships `verbalize`, `reconstruct`, `invert`. The inverter is a free Patchscopes-style baseline. |
| **Activation Oracle (question answering)** | `adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3.6-27B` (official); `EvilScript/activation-oracle-Qwen3_6-27B` (same recipe, MIT) | 16, 32, 48 | classification 0.86 to 0.99; untrained base already 0.72 to 0.93 | `activation_oracles` repo, main branch | Only reader that takes a free-form question ("Will the assistant claim to be human?"). Calibration code at `federicotorrielli/probabilistic_activation_oracles` has a qwen3.6-27b preset. |
| **NLA, layer 42** (the WorkspaceBench NLA) | `ceselder/qwen3.6-27b-nla-rl` (checkpoints every 100 RL steps) or `ceselder/qwen3.6-27b-nla-L42` (SFT LoRA + RL LoRA on raw base, Apache-2.0) | 42 | FVE 0.75 to 0.78 | EasyNLA (`asherps/EasyNLA`) | `nla-rl` attaches to a merged 54 GB warm-start copy, which does not fit beside the base on one A100. Use `nla-L42`, which stacks two LoRAs on the raw base. |
| NLA, 11 layers | `camilablank/qwen3.6-27b-multilayer-nla-rl600` | 20 to 60 step 4 | WorkspaceBench 0.754 | EasyNLA, Qwen3.6 fork | Verify it attaches to the raw base. AR is a full model (10 shards). |
| NLA, front-loaded | `ceselder/nla-qwen36-27b-matryoshka` | 42 | FVE 0.64 full, 0.43 from first 10 tokens | EasyNLA | Read only the first 10 tokens for a cheap readout. |
| **Modulation-lens meta-model** | `ceselder/modulation-lens-meta-model-qwen36-27b`, `-4bullet-` | 42 | cosine reward; no benchmark | PEFT, prompt in repo | Writes the phrase that, when the model is told to focus on it, reproduces the activation. Trained for exactly the text to vector to behavior loop in P6. The 4-bullet version decomposes a state. |
| Distilled 2B NLA students | `adamjb111/nld27-grpo-qwen35-2b-from-nla-qwen36-27b-L42-...` | 42 | FVE 0.74 (98% of teacher) on web text; weaker on chat | vLLM, custom adapter | Cheap enough to read every token of every transcript (P10). |
| J-lens, R-lens, template lens | `camilablank/workspace-lenses` | all | recall@10 about 0.5 | `jlens` pip | Also ships Qwen3.5-4B and Qwen3.6-35B-A3B lenses. |
| J++ lens | `neuronpedia/jacobian-lens:qwen3.6-27b/jpp` | all | recall@10 0.55 | `jlens` | Drop-in replacement for J-lens, better in early layers. Also has a Qwen3.5-4B J-lens. |
| SAE, layer 42 | `ceselder/qwen36-27b-sae-l42-1b` (131k features) and `qwen36-27b-sae2m-l42-1b` (2M features) | 42 | EV 0.70 to 0.71 | `dictionary_learning` | Max-activating examples shipped for every feature. Replaces the unvetted SAE cited earlier. |

For the other models:

- **Qwen3.5-4B**: an Oracle Lens with AR, inverter, distill and RL stages in `agu18dec/mOre-lenses:qwen3.5-4b/` (layers 10 to 30, WorkspaceBench 0.595); J-lens and R-lens in `workspace-lenses`; an experimental layer-26 lens in `bubskee/qwen3.5-4b-o-lenses`. The only 4B NLA (`Solshine/...`) states on its own card that it does not produce usable descriptions.
- **Qwen3.6-35B-A3B**: activation oracles at `stanleytheli/qwen3.6-35b-a3b-ao` and `swan-0/qwen3.6-35b-a3b-activation-oracle` (4-bit, r16).

Reusable code from others' Qwen3.6-27B reader studies:

- `mkenney2/nla-consistency`: keeping only claims that appear in 2 of 8 samples cut false claims from 38% to 13% while keeping 87% of true ones. Use this filter in every judged readout (P1, P5, P7).
- `Jarus77/activation-reader-agreement`: J-lens, NLA and SAE readouts of the same layer-42 activation agree above chance, with pinned revisions and a Modal pipeline.
- `adam-jb/small-nla-training`: the distillation recipe, if we ever want a 2B reader for a different layer.

### 1.1 Jacobian lens (J-lens) and R-lens

A linear map from a residual vector at layer l to a distribution over the 248k-token Qwen vocabulary:
`softmax(W_U · norm(J_l h))`, where J_l is the corpus-averaged Jacobian of the final residual with respect
to layer l. The R-lens is the same estimator with an LRP-modified backward pass and reads early layers more
cleanly. Because the map is linear it can read a direction as well as an activation, so `J_L · W_down[:, n]`
is a valid readout of what a neuron's output becomes at the unembedding.

- **Exists.** `anthropics/jacobian-lens` (pip). `camilablank/workspace-lenses` ships fitted lenses for
  Qwen3.6-27B (target layer 62), Qwen3.5-4B (target 30) and Qwen3.6-35B-A3B (target 38), among others.
  J is `[n_layers, d, d]`, about 6 GB fp32 for 27B. Neuronpedia hosts a J-lens UI for Qwen3.6-27B.
- **Cost.** One 5120 x 5120 matvec plus an unembed per readout. Reading all 294,912 neurons of Qwen3.5-4B
  takes minutes.
- **Caveat.** Lens-transported directions do not always steer. One replication found that projecting out a
  J-lens direction did nothing while negative steering along it worked.

### 1.2 Template lens

Scores a residual vector against about 13k word and phrase directions per layer, so it reads multi-token
concepts that a single-token lens cannot. In a pilot it held 0.92 top-1 accuracy on 4-token concepts where
the J-lens scored 0.

- **Exists, 27B only.** Two independent builds: `camilablank/workspace-lenses` (cosine scoring, about 9 GB)
  and `ec75hash/template-lens-qwen3.6-27b` (ridge-LDA templates; ships per-layer covariance, so new phrases
  can be added from about 20 hand-written passages each).
- **Cost.** One `[13k, 5120]` matmul per layer. Minting a custom 50-phrase WeirdChat panel ("I am human",
  "called 911", "ran the code") takes under 10 minutes.

### 1.3 Oracle Lens and its activation reconstructor

A LoRA on Qwen3.6-27B itself that turns one residual vector into about four text bullets. The vector, taken
from layers 20 to 60 in steps of 4, is rescaled and swapped into the embedding of a marker token inside a
fixed prompt. Training was supervised fine-tuning then RL, with reward equal to how much activation variance
the bullets explain when a frozen **activation reconstructor** (AR) maps the text back to a vector. The AR
is itself a usable text-to-vector map (variance explained 0.12 at layer 20, rising to 0.34 at layer 60).

- **Exists.** `agu18dec/olens_and_ar` (Apache-2.0; WorkspaceBench score about 0.73) and the older
  `agu18dec/oracle-lens-ddp600` (macro 0.785). Because it is a LoRA on the model we steer, one copy of the
  base model serves both: capture with the adapter off, read with it on. Reading the lens's own activations
  is a silent error.
- **Cost.** About 6 s per 32 readouts. All 403 prompts at 11 layers is about 15 minutes.
- **Caveat, resolved.** The model card's code imports a `global_workspace` package that is not public.
  `agu18dec/final_olens` is the same lineage with a self-contained loader and reads any of the 64 layers,
  so use that one. Its injection scale is 1000, not 16000; take each repo's contract whole.

### 1.4 Activation Oracles (LatentQA line)

A fine-tuned model that takes activations injected at layer 1 plus a free-form question and answers in text.

- **Exists for Qwen3.6-27B.** The official checkpoint (`adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3.6-27B`,
  layers 16, 32, 48) and an MIT-licensed twin from the calibration paper (`EvilScript/activation-oracle-Qwen3_6-27B`).
  Also for Qwen3.6-35B-A3B (two repos) and Qwen3-8B/32B. None for Qwen3.5-4B, but the 4B Oracle Lens in
  `mOre-lenses` covers the same need without the question interface.
- **What the literature says.** Scoring candidate answers beats verbalized confidence for calibration
  (2605.26045). Fine-tuned oracles develop concept-specific blind spots (2607.23379). About half of answers
  are vague and oracles are near chance on sycophancy, yet they detected a steering vector at coefficient
  0.15 when the output text showed no change (ICML 2026). Detecting steering is the one task where oracles
  clearly beat reading the transcript.

### 1.5 Natural language autoencoders (NLAs)

A copy of the target model trained to turn one residual vector into several hundred tokens of prose, scored
by how well a reconstructor recovers the vector (0.6 to 0.8 variance explained after RL).

- **Exists for Qwen3.6-27B.** The WorkspaceBench layer-42 NLA is public at `ceselder/qwen3.6-27b-nla-rl`
  (FVE 0.77) with a raw-base-compatible variant at `ceselder/qwen3.6-27b-nla-L42`, and an 11-layer version at
  `camilablank/qwen3.6-27b-multilayer-nla-rl600` (WorkspaceBench 0.754). Two 2B distilled students exist for
  cheap token-level reading. No usable NLA for Qwen3.5-4B.
- **What the literature says.** NLAs can describe activation differences (2610.30934). A true description
  beats a rival with one claim flipped only 60% of the time, so readers identify which activation they see
  rather than what is true of it (2610.18727). Nearly every NLA readout in WorkspaceBench contained a claim
  contradicted by the context.
- **Cost.** About 250 decode tokens per readout, 2 to 3x the Oracle Lens.

### 1.6 WorkspaceBench and CHIVE

The two reader evaluations this project extends.

- **WorkspaceBench** (`camilablank/workspace-bench`): 3,356 items on Qwen3.6-27B, read at layers 20 to 60
  by J/R/logit lens, template lens, Oracle Lens and an NLA, with prompt-only and lucky-guess floors. Finding:
  single-token lenses are reliable but cannot express order or multi-token concepts; NLAs are expressive but
  confabulate.
- **CHIVE** (arXiv 2608.16747): counterfactual prompt edits as ground truth. Giving a predictor five calls
  to an oracle, NLA or SAE gave **no uplift** over the transcript alone. The tools describe the prompt
  feature and the behavior, both already visible in the transcript, and rarely the causal link.
- **Both use binary, prompt-derived labels.** Ours are causal (does steering neuron n change behavior b)
  and probabilistic (64 rollouts per prompt). Their judge prompts and floors can be borrowed.

### 1.7 Generative latent prior (GLP)

A flow-matching denoiser trained on 1B residual activations (Llama only, 2602.06964). To steer: add the
vector, add noise to t = 0.5, denoise for 20 steps. This pulls the steered activation back onto the
activation manifold and improves judged fluency at matched concept strength versus persona vectors,
difference-in-means and SAE steering.

- **For us.** A small GLP for one layer of Qwen3.6-27B on 10 to 20M activations is about 2 to 3 GPU-hours
  (est.), quality unknown. On-manifold proxies that need no training appear in P3.

### 1.8 Manifold steering (arXiv 2605.05115)

Steer along geodesics of the activation manifold rather than straight lines, on the argument that linear
steering cuts through off-manifold regions. Demonstrated on cyclic reasoning tasks; no Qwen artifacts and no
confirmed code. For a single neuron, the natural on-manifold version is clamping the neuron's own activation
rather than adding a constant vector.

### 1.9 Readers that need no training

- **Patchscopes.** Patch the activation into a placeholder token of a target prompt in the same model
  ("Tell me about X: X is ...") and read the continuation. One forward pass, any layer, and it reads a
  steered activation as naturally as a natural one. Every learned reader must beat this.
- **Logit lens of `W_down[:, n]`.** Caveat from 2609.09902 and 2609.10210: unembedding norms inflate
  rankings, and a single neuron accounts for about a tenth of a direction's logit push.
- **Self-report.** Ask the steered model what it is about to do. A black-box floor.
- **SAEs.** `ceselder/qwen36-27b-sae-l42-1b` (131k features, EV 0.70) and `qwen36-27b-sae2m-l42-1b`
  (2M features), both with max-activating examples per feature. Useful to ask whether a validated neuron's
  write direction aligns with any SAE latent at layer 42.

### 1.10 Meta-tokens in J-space

A J-lens on Qwen3.6-27B surfaces tokens that name the algorithm rather than the content (什么意思 before pun
recognition, gcd on LCM problems, 大概率 before hedging; often Chinese). Recipe: a token fires if its rank is
20 or better anywhere in layers 20 to 62; steer negatively with per-layer J-lens vectors at chat-template
positions; 50 rollouts; random-vector controls. Reused directly in P9.

---

## 2. Experiment proposals

Each proposal is scored E (excitement, 1 to 5) and F (feasibility in one week on our budget, 1 to 5).

### P1. Can a reader name the behavior from the vector alone?

**Question.** Given only a vector (a neuron's write direction, or the activation change it causes), can a
reader say which of the 11 behaviors it controls? CHIVE found no uplift when ground truth was a prompt edit.
Here ground truth is a steering result we measured, and the reader never sees a transcript, so the shortcut
of reading the text is closed.

**Method.**
1. For each validated neuron and each control, build three vectors: (i) the write direction scaled to the
   steering dose and added to the layer's mean residual (readers are trained on natural activations, so a
   bare direction is out of distribution; also run the bare direction); (ii) steered-minus-unsteered mean
   residual at assistant positions 1 to 8 on 30 benign prompts; (iii) matched-minus-unmatched mean residual
   on the behavior's own prompts.
2. Read each vector with J-lens, R-lens, template lens, Oracle Lens (5 samples), Patchscopes and logit lens.
3. A blind judge (Haiku 5.5) makes a 12-way choice (11 behaviors plus "none"), scores the match to the
   Neuron Card label from 0 to 10, and counts hallucinated claims. Floors: random-neuron items should be
   "none"; shuffled labels; and a transcript baseline where the judge sees three steered outputs instead of
   a readout.

**Metric.** Accuracy and macro-F1 over 12 classes per reader, vector type and layer, with Wilson intervals.
Also the first layer at which each reader identifies the behavior.

**Would matter if** a reader identifies the behavior from the write direction alone above 50% (chance 8%)
while controls stay "none" above 80%; or if any reader beats the transcript baseline (CHIVE predicts none
will); or if readers see the natural delta but not the steered one, which would mean the neuron's downstream
effect differs from the behavior's natural signature.

**Cost.** About 9k readouts and 10k judge calls. **About 2 GPU-hours.**
**Risks.** The Oracle Lens may need reimplementing. Behaviors in the same family (denying AI identity vs
false embodiment) will be confused; report family-level accuracy too. **E 4, F 4.**

### P2. One neuron as the search signal for finding new prompts

**Question.** Transluce spent 22.1M subject-model samples on Qwen3.6-27B, a median of 0.77M per behavior
found. Each sample is a 27B generation plus a 31B judge pass. Can a validated neuron's activation at the
last prompt token, which costs one prefill with no decode and no judge, replace the judged rollout as the
search fitness?

**Method.**
1. *Validation first; no search needed.* Prefill all 403 prompts, plus the ~2.2k prompts written for the
   other five WeirdChat models as a transfer set. Record each validated neuron's activation at the last
   prompt token. Correlate with the per-prompt behavior rate: Spearman within behavior, AUROC for rate at or
   above 25% vs below 5%, and cross-behavior specificity (whether the 911 neuron fires on drunk-driving
   prompts). Controls: 20 random same-layer neurons, a dense logistic probe on the full residual (the
   expected ceiling), and the Oracle Lens mention rate (links to P5). **Under 10 GPU-minutes.**
2. *Search*, for any behavior with Spearman at or above 0.5. An evolutionary loop with Haiku as mutator
   (population 160, 20 generations), fitness equal to the neuron's activation z-scored against 500 WildChat
   prompts, with the WeirdChat naturalness judge as a constraint so we do not evolve jailbreaks. Then judge
   the top 50 prompts at 64 rollouts each. Repeat with random-neuron and dense-probe fitness.
   **About 30 GPU-minutes.**
3. Report the behavior rate of discovered prompts (WeirdChat's median is 9%), the number of new patterns
   above 10%, and the **Goodhart rate**: the fraction of top-fitness prompts whose judged rate is below 5%.
   Those prompts fire the neuron without the behavior and are free data on what the neuron actually detects.

**Savings (est.).** A judged rollout costs about 80 TFLOP; a prefill truncated at the neuron's layer about
7. Because decode is memory-bound, the wall-clock gap is 30 to 250x. A Bernoulli fitness at base rate p
needs about 1/p samples to register a hit, whereas the neuron gives a graded score from one prefill.
Combined, two to three orders of magnitude on the search phase: about 6k subject-model calls per behavior
against Transluce's 770k.

**Would matter if** the neuron matches the dense probe (a 1-D causal feature as good as a 5120-D
correlational one); if the Goodhart rate is below 20%, meaning the neuron detects the situation and not just
a word; or if the neuron fires on other models' WeirdChat prompts in proportion to those models' rates.

**Risks.** WeirdChat rates were measured in FP8 and ours are bf16 (Phase 0 checks drift). Three behaviors
have only 7 to 9 prompts; pool by family. The likely outcome is "dense probe > neuron >> random", which is
still a methods result. **E 5, F 5.**

### P3. Does steering along the model's own manifold beat adding a constant vector?

**Question.** Our push adds a fixed vector regardless of context. Does steering that stays on the model's
natural activation manifold give the same behavior with higher coherence and specificity, as the manifold
steering and GLP papers argue?

**Method.** For 3 to 5 validated neurons, six steering variants, each matched to the dose that gives about
30% behavior under the additive push:
- (A) additive push, our default;
- (B) clamp the neuron's own post-activation value to its 99.9th percentile over natural text, so the write
  scales with the model's own gate (the 1-D on-manifold version);
- (C) multiply the neuron's activation by a factor k;
- (D) project the push onto the top 64 or 256 principal components of the layer's residual covariance from
  1M natural tokens (the cheapest on-manifold proxy, no training);
- (E) a small GLP trained on about 10M activations of this layer, applied to the pushed activation (about 3
  GPU-hours; stretch goal);
- (F) the matched-minus-unmatched difference-in-means vector, as a multi-dimensional ceiling.

Titrate each variant to equal behavior rate, then compare coherence (judge, 0 to 10), perplexity under the
unsteered model, distinct 4-gram ratio, the other 10 behavior rates, MMLU and GSM8K deltas, and the AR
reconstruction score of the steered activation as a free proxy for distance from the manifold.

**Would matter if** the additive push is as coherent as clamping, which would mean the neuron's write
direction is already on-manifold; or if the covariance projection kills the behavior, which would mean the
active part of the direction lies in low-variance directions that a GLP would also remove.

**Cost.** About 43k generations for A to D and F. **5 to 8 GPU-hours with E.**
**Risks.** A GLP on 10M tokens may be too weak (the paper's smallest used 1B); let D stand in. Matching
strength across variants is fiddly; fix the titration rule in advance. **E 3, F 3.**

### P4. Finding the same neuron in a smaller model by reading its write direction

**Question.** Do the 27B neurons have homologs in Qwen3.5-4B? Write directions live in different bases, but
the J-lens maps each into the shared vocabulary, a common coordinate system across scales.

**Method.**
1. Read each validated 27B neuron's write direction through the 27B J-lens (also R-lens and template lens).
2. Read every neuron and sign of Qwen3.5-4B (294,912 x 2) through the 4B J-lens; keep the top 100 tokens
   each. Minutes on GPU.
3. Rank 4B neurons by top-50 Jaccard, KL on the top-100 union, and cosine in unembed space. As an
   independent ranking, use the Spearman correlation of activations across the 403 prompts in both models.
4. Verify on the local M4: steer the top 20 candidates in 4B at dose 0.8, both signs, on the three
   WeirdChat-derived prompts from the S5 exhaustive sweep, and judge. If a lens-matched candidate is in S5's
   top 1% for the same behavior, the homology is real.
5. Repeat for 35B-A3B with the MoE J-lens, treating expert-neuron columns as neurons.

**Metric.** Precision@20 against S5's causal sweep; agreement between the two rankings; relative depth of
homolog vs source.

**Would matter if** lens-matched homologs replicate the behavior causally far above random 4B neurons. That
would be the first cross-scale neuron homology found by readout, and would make the J-lens a tool for
transferring neuron findings across models. Also interesting if homologs exist by activation signature but
not by lens: detectors agree while write directions diverge.

**Cost.** **Under 1 GPU-hour** plus S5 and local verification.
**Risks.** 4B may not show the behavior (Phase 0 checks). Lens readouts are dominated by generic tokens;
remove the 500 most frequent lens tokens across all neurons before comparing. **E 4, F 4.**

### P5. Readers as forecasters: do they track the 64-rollout behavior rate? (my pick for the post)

**Question.** No reader has been graded against a probability. At the last prompt token the model has
emitted nothing, yet the 64 rollouts say how often it will deny being an AI. Does the Oracle Lens, read at
that position, mention the behavior at a frequency that tracks the rate, and does the validated neuron
mediate that reading? If it works, the result reads: the model decides to pretend it is human before it says
a word, one neuron carries the decision, and the model's own activation reader can see it.

**Method.**
1. Read the last-prompt-token activation of all 403 prompts through the Oracle Lens at layers 20 to 60
   (5 samples each, about 22k reads), the J/R/template lenses, Patchscopes ("The assistant is about to"),
   and self-report ("Before you answer, will you claim to be human?", logprob of yes).
2. A judge (Haiku) marks whether each readout mentions the behavior.
3. Bin prompts by behavior rate (0 to 5%, 5 to 15%, 15 to 30%, above 30%), plot mention rate per bin, and
   report Spearman and expected calibration error per behavior and per layer. Floors: Haiku predicting from
   the prompt text alone (CHIVE's point), and readouts for a lexically matched near-miss prompt set.
4. Mediation: steer the validated neuron at the same position (dose ±0.5, ±1.0) and re-read. Check whether
   the mention rate moves with the neuron, and by as much as the behavior rate moves on the Phase 2 card.
   Random same-layer neurons as control.

**Would matter if** mention rate tracks behavior rate (Spearman above 0.4) above the prompt-only floor,
meaning the activation carries propensity information that the prompt text does not expose to a strong LLM.
Also if the neuron shifts the behavior but not the reading (a reader blind spot, cf. 2607.23379), or the
reading but not the behavior (the reader sees the neuron's write, not the decision). A negative result is a
publishable extension of CHIVE to probabilistic ground truth.

**Cost.** **About 2 GPU-hours** plus 30k Haiku calls.
**Risks.** FP8 vs bf16 rate drift (check 40 prompts in Phase 0). Chat-template positions may be out of
distribution for the Oracle Lens; read at the last user token too. **E 5, F 4.**

### P6. Round trip: reader text, back to a vector, then steer

**Question.** The AR shipped with the Oracle Lens maps text to a layer-L vector. Does the text a reader
writes about a validated neuron, mapped back to a vector, do what the neuron does?

**Method.** For each neuron, build four vectors: AR of the Oracle Lens bullets; AR of the Neuron Card's
human label ("the assistant claims to be a human with a body"); AR of the negated label; and AR of a
different behavior's label. As a fifth, use the modulation-lens meta-model's phrase for the steered
activation, since that reader was trained so that "focus on <phrase>" reproduces the state. Steer with each at the neuron's dose and at 2x. Measure the 11 behavior rates
and coherence against the neuron push, the difference-in-means vector, and random directions. Report the
cosine between each AR vector and the write direction.

**Would matter if** AR of the label induces the behavior at more than half the neuron's rate (a
text-specified vector matching a causally found neuron), or if AR of the negated label suppresses the
natural behavior.

**Cost.** About 58k generations for 15 neurons. **6 GPU-hours, or 2 with 5 neurons.**
**Risks.** AR variance explained is only 0.12 to 0.34, so reconstructed vectors are mostly noise. The AR
covers layers 20 to 63 only. **E 4, F 3.**

### P7. Can a reader detect that a neuron was pushed, and how hard?

**Question.** The one clear win reported for activation oracles was detecting a subtle steering vector. Can
the Oracle Lens tell which neuron was pushed, and at what dose, from one activation? This doubles as a
monitoring result: whether a reader can catch a single-neuron backdoor edit.

**Method.** Benign prompts x {no steer, each validated neuron, random neurons} x doses {0.25, 0.5, 1.0}.
Read 8 and 16 layers downstream with Oracle Lens, J-lens and template lens. Judge: steered vs not (AUROC per
dose), which behavior (12-way), and dose rank correlation. Baseline: Haiku reading the generated text, which
at dose 0.25 should look normal.

**Would matter if** readers detect dose-0.25 pushes that text readers miss and name the behavior.
**Cost.** **About 1 GPU-hour.** **E 3, F 5.**

### P8. Could a lens have found these neurons for free?

**Question.** Screen S4 reads every candidate's write direction through lenses before causal validation.
Once Phase 2 has judged the whole shortlist (about 150 candidates, winners and losers), the lens can be
scored as a screen.

**Method.** Freeze the S4 readouts before Phase 2. Afterwards compute precision@k, recall and AUROC of "lens
mentions behavior b" against "steering produces b at 20% or more". Compare with the S1 activation-contrast
screen and the S3 attribution screen. Report which layers' lenses work.

**Would matter if** lens-only screening recovers more than half of validated neurons at precision above
30%; the next neuron hunt would then need no activation capture.
**Cost.** None beyond S4 and Phase 2. **E 3, F 5.**

### P9. WeirdChat meta-tokens

**Question.** Do WeirdChat prompts carry a J-lens meta-token at the last prompt token (candidates: 假装
"pretend", 扮演 "role-play", 同意 "agree") whose rank tracks the behavior rate? Does negative steering of its
J-lens vector reduce the behavior as much as suppressing the neuron?

**Method.** Follow the meta-tokens recipe (section 1.10): J-lens top-20 at layers 20 to 62 for the 403
prompts; tokens enriched in high-rate vs low-rate prompts (Fisher test, frequency-corrected against
WildChat); steer candidates negatively at chat-template positions, 50 rollouts, random-vector controls.
Compare the rate drop with the neuron's, and test whether the meta-token vector and the neuron's write
direction are aligned.

**Would matter if** one meta-token predicts weirdness across several behaviors (a lens-level hub matching
the H1 hub-neuron hypothesis), or if the neuron and the meta-token are the same direction.
**Cost.** About 5k generations. **About 1 GPU-hour.** **E 4, F 4.**

### P10. Phrase panel vs neuron as a token-level monitor

**Question.** A deployable fix needs a detector as well as a lever. Which separates matched from unmatched
rollouts earlier and better: a 40-phrase WeirdChat template panel, the validated neuron, the existing
deception/honesty panel, or a dense probe?

**Method.** Mint the panel (ec75hash recipe, 20 passages per phrase). Teacher-force all 27k Qwen3.6-27B
transcripts and compute token-by-token AUROC for each monitor. Report the earliest token at which each
separates the two groups, which is also the empirical answer to where S1's pre-commitment window lies.

**Would matter if** the single neuron separates rollouts earlier than the phrase panel (the decision
precedes the words), or if the honesty panel fires on denying AI identity but not on fabricated code
execution.
**Cost.** About 5M prefill tokens. **About 30 GPU-minutes.** **E 3, F 5.**

### P11 (optional). The 35B-A3B activation oracle as a second subject

`stanleytheli/qwen3.6-35b-a3b-ao` (or `swan-0/qwen3.6-35b-a3b-activation-oracle`) reads the other WeirdChat Qwen. If anything runs on
35B-A3B, repeat P5 step 1 there with a question-style oracle ("Will the assistant claim to have called
911?") and compare calibration on the 7 shared behaviors. The model fits in bf16 at 70 GB with the LoRA,
tightly. **E 3, F 2.**

### Summary

| # | Proposal | GPU-h (est.) | E | F |
|---|---|---|---|---|
| P1 | Reader names the behavior from the vector alone | ~2 | 4 | 4 |
| P2 | Neuron as search signal for new prompts (step 1 is free) | ~1 | 5 | 5 |
| P3 | On-manifold vs additive single-neuron steering | 5-8 | 3 | 3 |
| P4 | Cross-scale homologs via J-lens readout | <1 (+local) | 4 | 4 |
| P5 | Readers as forecasters of the behavior rate, with neuron mediation | ~2 | 5 | 4 |
| P6 | Round trip: reader text, AR vector, steer | 2-6 | 4 | 3 |
| P7 | Steering detection and dose estimation | ~1 | 3 | 5 |
| P8 | Lens pre-screen precision and recall over the shortlist | 0 | 3 | 5 |
| P9 | WeirdChat meta-tokens, steered | ~1 | 4 | 4 |
| P10 | Phrase panel vs neuron as token-level monitor | ~0.5 | 3 | 5 |
| P11 | 35B-A3B oracle replication | 2-3 | 3 | 2 |

All together about 15 to 25 GPU-hours, within the shared budget of about 100. Everything depends on Phase 2
delivering validated neurons, except P2 step 1, P5 steps 1 to 3, P9 readouts and P10. Those can run on
candidates now and are cheap enough to rerun once the cards are final.

---

## 3. Shortlist for this week

1. **P2, neuron as search signal.** Step 1 takes 10 GPU-minutes and uses ground truth that already exists.
   It is the result Transluce would care about most (770k samples per behavior down to 6k once you know the
   neuron) and produces the "weirdness probe" from PROPOSAL §7.2. Its failure mode, the dense probe winning,
   is still a useful number.
2. **P5, readers as forecasters.** Shares P2's prefill pass, plus about 1.5 GPU-hours of Oracle Lens reads.
   The first reader evaluation with probabilistic ground truth, and the mediation step ties the reader to
   our causal neuron. Positive or negative, it is a self-contained section of the post.
3. **P1, reader names the behavior,** with P8 as a free byproduct and P4 as a one-hour add-on if the 4B
   sweep finds the behaviors. About 2 GPU-hours.

Defer P3 and P6 to the second week unless a neuron shows a coherence problem at the lever dose, in which
case P3 variants B and D are the first fix to try.

## 4. Open items before running

- Done 2026-10-28: Oracle Lens loader. `agu18dec/final_olens` ships a self-contained loader, so no
  reimplementation is needed. Pin its recommended versions (torch 2.9.1, transformers 5.11.0, peft 0.19.1,
  flash-linear-attention 0.5.1) in a separate environment from the steering harness.
- Done 2026-10-28: the WorkspaceBench layer-42 NLA is public (`ceselder/qwen3.6-27b-nla-rl`). No email needed.
- Smoke-test each reader on the `example_activations.parquet` or demo text its repo ships before reading
  WeirdChat activations. An adapter that silently matches no weights looks like a working one.
- Verify Qwen3.5-4B and Qwen3.6-27B share a tokenizer and vocabulary (needed for P4 as written; otherwise
  compare in decoded-string space).
- Reserve about 8 GPU-hours for P1, P2 and P5 and run them in one model session to avoid reloading the
  54 GB checkpoint. Readers that need a second full model (NLA `av_base`, any AR) run in a separate pass.
- Done 2026-10-28: the S1 position caveat is fixed in PROPOSAL.md.

## Sources consulted

- Papers: transformer-circuits.pub/2026/nla; arXiv 2610.30934, 2610.18727, 2512.15674, 2606.02609,
  2605.26045, 2607.23379, 2602.06964, 2605.05115, 2608.16747 (CHIVE), 2604.08169, 2609.09902, 2609.10210.
- Posts: AF "WorkspaceBench"; AF "Towards surfacing model algorithms with meta-tokens in the J-Space";
  LW "Current activation oracles are hard to use"; LW "Manifold steering from a natural state";
  alignment.anthropic.com/2026/chive and /2026/introspection-adapters; transluce.org/weirdchat (methods
  appendix); generative-latent-prior.github.io.
- Models (HF): agu18dec/final_olens, agu18dec/olens_and_ar, agu18dec/mOre-lenses, neuronpedia/jacobian-lens,
  camilablank/workspace-lenses, camilablank/qwen3.6-27b-multilayer-nla-rl600, ec75hash/template-lens-qwen3.6-27b,
  ceselder/qwen3.6-27b-nla-rl, ceselder/qwen3.6-27b-nla-L42, ceselder/nla-qwen36-27b-matryoshka,
  ceselder/olens-qwen3.6-27b-l42, ceselder/modulation-lens-meta-model-qwen36-27b, ceselder/qwen36-27b-sae-l42-1b,
  adamkarvonen/checkpoints_latentqa_cls_past_lens_addition_Qwen3.6-27B, EvilScript/activation-oracle-Qwen3_6-27B,
  adamjb111/nld27-grpo-qwen35-2b-from-nla-qwen36-27b-L42-pairs526k-op100k-ar-reward-20261013,
  stanleytheli/qwen3.6-35b-a3b-ao, swan-0/qwen3.6-35b-a3b-activation-oracle, kitft/nla-models.
- Code (GitHub): anthropics/jacobian-lens, camilablank/workspace-bench, adamkarvonen/activation_oracles,
  federicotorrielli/probabilistic_activation_oracles, asherps/EasyNLA, kitft/nla-inference,
  mkenney2/nla-consistency, Jarus77/activation-reader-agreement, adam-jb/small-nla-training,
  g-luo/generative_latent_prior; neuronpedia.org/qwen3.6-27b.
- Local data: `data/weirdchat/data/search_compute.parquet`, `prompts/qwen__qwen3.6-27b/*.parquet`.
