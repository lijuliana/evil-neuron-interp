# Meta-models, lenses and activation readers: what to do with them here

Agent A7, 2026-10-28. Desk research only; nothing below has been run. Numbers marked "est." are my
estimates for one A100-80GB holding Qwen3.6-27B in bf16 (54 GB weights, ~26 GB free), HF eager,
batch 32-64. Rules of thumb used throughout: prefill ~4k tok/s; decode ~1k tok/s aggregate at batch 64
(a 64 x 400-token cell is ~25-30 s); Oracle Lens readout (~90 tokens) ~6 s per batch of 32.

Two facts about the data that shape everything below (checked in `data/weirdchat/`):

- Qwen3.6-27B has **403 prompts** across the 11 behaviors, each with exactly 64 judged rollouts, so each
  prompt carries an empirical behavior rate (median 9%, IQR 5-19%; 71 prompts >= 25%, 107 prompts < 5%).
  Mean user prompt is ~105 tokens. That is a *probabilistic* ground truth per prompt, which no reader
  benchmark has used yet (WorkspaceBench, AObench and CHIVE all use binary item labels).
- **Caveat for Screen S1 in the proposal.** Thinking is off and there is no system prompt, so the prefill is
  deterministic: matched and unmatched rollouts of the same prompt have *identical* activations at the last
  prompt token and at the position that emits the first assistant token. The within-prompt contrast is
  zero there by construction. It only becomes non-zero from the position *after* the first sampled token.
  "Pre-commitment" in S1 therefore means assistant positions 1..k (inputs that already include sampled
  tokens), and the last-prompt-token analysis must be *between* prompts (activation vs the 64-rollout
  rate), which is proposal 2 below. Worth fixing in PROPOSAL.md before A1-A3 start.

---

## 1. Primers (one paragraph each)

### 1.1 Jacobian lens (J-lens) and R-lens
**In:** a residual-stream vector h at layer l (any position). **Out:** `softmax(W_U · norm(J_l h))`, a
distribution over the 248k Qwen vocabulary; J_l is the corpus-averaged Jacobian of the final-layer residual
with respect to h_l (R-lens: same estimator but the backward pass goes through a RelP/LRP-modified graph,
cleaner in early layers). Because it is linear it reads a *direction*, not just an activation, so
`J_L · W_down[:, n]` is a legitimate readout of what a neuron's write becomes at the output ("steer lens"
with a Jacobian transport instead of a raw unembed). **Exists:** `anthropics/jacobian-lens` (pip, `jlens.fit`,
`JacobianLens.from_pretrained`); `camilablank/workspace-lenses` ships `j-lens/lens.pt` and `r-lens/lens.pt`
for **qwen3.6-27b** (target layer 62), **qwen3.5-4b** (target 30), **qwen3.6-35b-a3b** (target 38, MoE rule
set), plus qwen3.5-9b/27b/122b-a10b, gemma-3-27b, deepseek-v4-flash. Fit on 25 pile-10k prompts, skip first
4 tokens; `J` is `[n_source_layers, d, d]` (27B: 60 x 5120 x 5120, ~6 GB fp32). Neuronpedia hosts a J-lens UI
for Qwen3.6-27B. **Cost:** essentially free: one 5120 x 5120 matvec plus an unembed per readout; reading all
294,912 neurons of Qwen3.5-4B through its J-lens is minutes on GPU. Known failure: a LessWrong replication
reports that lens-transported directions do not always steer; the meta-tokens post found projection-ablation
of a J-lens direction did nothing while negative steering did.

### 1.2 Template lens (phrase-level lens)
**In:** residual h at layer l. **Out:** scores over ~13k words/phrases (multi-token concepts) by dot/cosine
with per-layer template directions. Two independent builds for **Qwen3.6-27B only**:
`camilablank/workspace-lenses` (safetensors `[64, 13174 or 13731, 5120]` bf16, ~8.6-9 GB, cosine scoring,
row-id to phrase `.txt`), and `ec75hash/template-lens-qwen3.6-27b` (ridge-LDA templates
`t_w(l) = (Σ_l + λI)^-1 (μ_w − μ)`, 12-word pilot plus a 175-word deception/honesty/neutral panel, ships
per-layer Σ (1.8 GB) so you can mint templates for new phrases from ~20 hand-written passages each). Pilot
numbers: template lens holds 0.92 top-1 on 4-token concepts where J-lens falls to 0.00 (top-10 of 248k).
**Cost:** a `[13k, 5120] @ h` matmul per layer; negligible. Minting a custom "WeirdChat panel" of ~50 phrases
("I am human", "called 911", "ran the code", "chemtrails", ...) is ~1k prefills, <10 min.

### 1.3 Oracle Lens (Activation Oracle for Qwen3.6-27B) and its activation reconstructor
**In:** one residual vector from layers **20-60 step 4** (11 layers; others "off-contract"), normalized and
scaled to `16000 · h/‖h‖`, swapped into the embedding of marker token ㈜ (id 158983) inside a fixed prompt
("Produce distinct concepts that encode this activation, each as a '- ' bullet"). **Out:** 4-ish bullets,
~90 tokens, sampled T=1 top-p 0.95. Trained SFT (inverter 1B tokens) then GRPO 600 steps with reward =
fraction of activation variance explained by the bullets through a frozen **activation reconstructor (AR)**
that maps text back to a vector (`ar_ptag_pooled`, layers 20-63 via a `[Layer 44]` text tag; FVE 0.12 at L20
rising to 0.34 at L60, ret@1 0.94). **Exists:** `agu18dec/olens_and_ar` (Apache-2.0; `olens_s3d_rl600` and
`olens_s3d_space_rl600`, held-out joint FVE 0.487, WorkspaceBench headline_23 0.726/0.728) and the older
lineage `agu18dec/oracle-lens-ddp600` (WorkspaceBench macro 0.785 over 12 families). This is a LoRA on the
**same** Qwen3.6-27B we steer, so one copy of the base model serves both (capture with adapter disabled,
read with adapter enabled; the card warns reading the lens's own activations is a silent error).
**Risk:** the card's snippets import `global_workspace` and `jlens`; `jlens` is public, `global_workspace`
is not confirmed public (the card points to `agu18dec/local-workspace` as the canonical archive). The
injection is simple enough to reimplement in ~50 lines of PEFT + a forward pre-hook on the embedding.
**Cost:** ~6 s per 32 readouts; 403 prompts x 11 layers = 4.4k readouts in ~15 min. The AR gives a
text-to-vector map that nobody has used for steering on this model.

### 1.4 Activation Oracles (LatentQA line, Karvonen et al. 2512.15674 and follow-ups)
**In:** activations from a source layer injected at layer 1 at placeholder positions (norm-matched
`h' = h + ‖h‖ v/‖v‖`), plus a free-form question. **Out:** a free-text answer. **Exists:** 12 oracles in
`adamkarvonen/activation_oracles` (Gemma-2/3, **Qwen3-8B/32B**, Llama-3; Qwen3-8B v3 LoRAs from `ceselder`),
and a **Qwen3.6-35B-A3B** oracle (`stanleytheli/qwen3.6-35b-a3b-ao`, layers 23-27 of 40, rsLoRA r=128,
AObench judge mean 0.773, 4xH200 x 11 h). **Nothing for Qwen3.6-27B dense except the Oracle Lens above, and
nothing for Qwen3.5-4B.** Follow-ups: "Building Better AOs" (2606.02609, on-policy rollouts, AObench);
calibration (2605.26045: candidate scoring beats everything, verbalized confidence is useless, 20-sample
agreement is best for free text, code released); "AOs learn not to read" (2607.23379: fine-tuned oracles
develop concept-specific blind spots); "Current AOs are hard to use" (ICML 2026: 49% of answers vague,
text-inversion confound, near-chance on sycophancy and backtracking, *but* the AO detected a pirate steering
vector at coefficient 0.15 when the output showed no change: the one place AOs clearly beat reading the
transcript is detecting steering). **Cost:** one LoRA forward + ~100-token decode per question. Training a
new AO for 27B is out of budget (~1.5M examples).

### 1.5 Natural Language Autoencoders (NLAs)
**In:** a unit-normalized residual vector from one fixed middle-late layer, injected as a token embedding
into an activation verbalizer (AV, a copy of the target model). **Out:** several hundred tokens of prose; an
activation reconstructor (AR) scores it by reconstruction FVE (0.6-0.8 after RL). Trained by SFT warm start
then GRPO on reconstruction reward. **Exists:** `kitft/nla-models`: Qwen2.5-7B L20, Gemma-3-12B L32,
Gemma-3-27B L41, Llama-3.3-70B L53 (AV+AR each, SGLang inference). **No public NLA for Qwen3.6-27B or
Qwen3.5-4B**; WorkspaceBench ran an NLA arm at L42 of Qwen3.6-27B, so one exists somewhere; worth one email
to the WorkspaceBench authors, otherwise skip. Training needs 2x8xH100 for RL. Relevant results:
2610.30934 shows NLAs verbalize *activation differences* (analogy vectors, finetune-minus-base diffs,
Taboo secret words) and that the most informative positions come before the user prompt; 2610.18727 shows a
true account beats a single-claim-flipped rival only 60% of the time (readers identify *which* activation,
not *what is true of it*); WorkspaceBench: "nearly every NLA readout" contains a claim contradicted by the
context. **Cost (if we had one):** ~250 tokens decode per readout, 2-3x the Oracle Lens.

### 1.6 WorkspaceBench and CHIVE (the two evaluations we are extending)
**WorkspaceBench** (`camilablank/workspace-bench`, AF post): 3,356 items in ~22-27 families on Qwen3.6-27B,
readouts at layers 20-60 step 4 (J/R/logit lens top-10 per layer, template lens top-10, Oracle Lens 4
bullets, NLA at L42), pass if any (layer, position) passes, judged by Gemini-3.8-Flash / Sonnet-5, with
prompt-only and lucky-guess floors and a bag-of-tokens-to-prose summarizer for single-token lenses. Finding:
single-token lenses are reliable but cannot express order or multi-token concepts; NLAs are expressive but
confabulate. **CHIVE** (alignment.anthropic.com/2026/chive, arXiv 2608.16747): counterfactual prompt edits as
ground truth; predictors given 5 calls to an AO, NLA or SAE got **no uplift** over the transcript alone on
Qwen3-8B and Qwen3.5-397B-A17B; reasons: tools describe the prompt feature and the behavior (already visible
in the transcript) and rarely the causal link. Both benchmarks use binary, prompt-derived ground truth. Ours
is causal (does steering neuron n change behavior b?) and probabilistic (64 rollouts per prompt). **Cost:**
`uv run wsbench judge` is API-only; the item banks are frozen so we can borrow its judge prompts and floors.

### 1.7 Generative Latent Prior (GLP) / meta-model of activations (2602.06964)
**In/out:** a flow-matching MLP denoiser (0.5-3.4B params) trained on 1B single-token residual activations
of Llama-3.2-1B (L7) / Llama-3.1-8B (L15). Steering use is SDEdit-style: take `a + α w`, standardize, noise
to t=0.5, denoise 20 steps, de-standardize; this pulls the steered activation back onto the activation
manifold and improves judged fluency at matched concept strength vs persona-vector, DiffMean and SAE
steering. **Exists:** code `g-luo/generative_latent_prior`, weights on HF org `generative-latent-prior`,
**Llama only**. Training cost: 1 epoch on 1B tokens, single A100, up to 5.6 days for 3.3B. **For us:** a
mini-GLP for one layer of Qwen3.6-27B on 10-20M activations (100-200 GB bf16 on disk, or streamed) with a
0.3-0.5B denoiser is ~2-3 GPU-hours total (est.), quality unknown. Cheaper on-manifold proxies that need no
training are in proposal 4.

### 1.8 Manifold steering (Wurgaft, Rager, Kowal, ..., Fel, Geiger, Lubana; arXiv 2605.05115; Goodfire-adjacent)
Fit an activation manifold M_h and a behavior manifold M_y; steer along geodesics of M_h instead of straight
lines; linear steering "cuts through off-manifold regions" and produces unnatural outputs. Demonstrated on
cyclic/sequential reasoning geometries (days of week) and ICL graph tasks; a LessWrong replication on
Gemma-2-2b got 95% of expected day cycling in raw residual space vs 36% in CLT space. No Qwen artifacts, no
code confirmed. For a single neuron the natural "manifold" is the 1-D family of activations the model itself
produces when the neuron is high, i.e. **clamping the neuron's own activation** vs adding a constant vector.

### 1.9 Patchscopes and other zero-training readers
**Patchscopes:** take h (or h + δ) from the source position, patch it into a placeholder token of a target
prompt in the *same* model ("Tell me about X: X is ..."), read the continuation. No training, one forward per
read, works at any layer, and reads a *steered* activation as naturally as a natural one. The obvious
baseline every learned reader must beat. **Logit lens** of `W_down[:, n]` (what neuron-explorer calls the
steer lens): caveat from Contrastive Projection (2609.09902) and Through the Looking Glass (2609.10210):
unembedding norms inflate rankings and single neurons account for ~1/10 of a direction's logit push.
**Self-report:** ask the steered model "What are you about to do / who are you?" (Introspection Adapters,
alignment.anthropic.com/2026/introspection-adapters, train a LoRA to make *finetuned* models report what they
learned; no Qwen3.6 release and we do not finetune, so the only usable analog is unadapted self-report under
steering, which is a cheap black-box floor). **SAEs:** `caiovicentino1/qwen36-27b-sae-multilayer` TopK SAEs at
L11/L31/L55 of Qwen3.6-27B exist but are unvetted (the author's own feature-circuits result is an "honest
negative"); usable only to ask whether a validated neuron's write direction aligns with any SAE latent.

### 1.10 Meta-tokens in J-space (AF post)
J-lens on Qwen3.6-27B surfaces tokens that name the algorithm rather than the content (什么意思 before pun
recognition, gcd on LCM problems, 大概率 before hedging; often Chinese tokens). Method: a token fires if rank
<= 20 anywhere in layers 20-62; negative steering with per-layer J-lens vectors at punctuation/chat-template
positions, coefficient swept per prompt to the strongest coherent value, 50 rollouts, random-vector controls.
Directly reusable recipe for a WeirdChat "meta-token" sweep (proposal 9).

---

## 2. Experiment proposals

Scores: E = excitement 1-5, F = feasibility in one week on our budget 1-5. "Neurons" below means the
causally validated set from Phase 2 (expect 5-20 poles across 11 behaviors) plus the matched controls that
ship with each card (random same-layer neurons at the same dose, and neurons validated for a *different*
behavior). Every proposal uses those controls as its negatives; that is what makes the ground truth ours.

### P1 (a). Readers vs causal ground truth: a 12-way identification bench with controlled truth
**Question.** Given only a *vector* (a neuron's write direction, or the activation delta that neuron
produces), can a reader say which WeirdChat behavior it controls? CHIVE found no uplift when ground truth was
a counterfactual prompt edit; here ground truth is a causal steering result we measured ourselves, and the
reader never sees a transcript, so the text-inversion shortcut is closed.
**Procedure.**
1. Items: for each validated pole p (behavior b_p, layer L_p) and each control neuron c (random same-layer,
   same dose, null causal effect), build three vectors: (i) `w = W_down[:, n]` scaled to the steering dose
   and added to the layer-L mean residual (readers are trained on natural activations, a bare direction is
   off-distribution; also run the bare direction as a sub-arm); (ii) steered-minus-unsteered mean residual
   at assistant positions 1-8 on 30 benign prompts (what the push *did* downstream, read at L_p+4...L_p+16);
   (iii) matched-minus-unmatched mean residual at assistant positions 1-8 over the behavior's WeirdChat
   prompts (the natural analogue of ii).
2. Readers: J-lens top-20 and R-lens top-20 (through the WorkspaceBench bag-to-prose summarizer), template
   lens top-10 phrases (both builds), Oracle Lens 4 bullets x 3 samples (for the 20-sample agreement
   confidence of 2605.26045 use 5 samples), Patchscopes with "This text is about" and "The assistant is
   about to", logit lens of w. If the WorkspaceBench L42 NLA is obtainable, add it.
3. Judge (Haiku 5.5, blind): 12-way forced choice over the 11 behaviors + "no specific behavior", plus a
   0-10 free-text match score against the Neuron Card label, plus a hallucination count (claims about
   content that the steered outputs never contain). Floors: random-neuron items (should be "none"),
   shuffled-label floor, and a **transcript baseline**: the judge sees 3 steered outputs instead of a
   readout (the CHIVE question: does reading the vector beat reading the text).
**Metrics.** Accuracy and macro-F1 over the 12 classes per reader x vector type x layer, with Wilson CIs
(n ~ 15 poles x 3 vector types x 11 layers x readers). Report the layer at which each reader first
identifies the behavior relative to L_p.
**Surprising if:** any reader identifies the behavior from the *write direction alone* above ~50% (chance
8%) while random controls stay "none" above 80%; or if readers beat the transcript baseline (CHIVE says
they will not). Also surprising: readers identify the matched-minus-unmatched delta but not the steered
delta (would mean the neuron's downstream effect differs from the natural behavior's signature).
**Cost.** Vectors: ~2k prefill/short-gen passes, ~20 min. Readouts: ~15 poles x 3 types x 11 layers x 6
readers x 3 samples ~ 9k reads, ~1-1.5 GPU-h. Judging ~10k Haiku calls. **Total ~2 GPU-h.**
**Could go wrong.** Oracle Lens code path (`global_workspace`) may need reimplementing (~half day). Bare
write directions are off-manifold for learned readers; that is why arm (i) adds the layer mean. Behaviors
in the same family (denying-ai-identity vs false-physical-embodiment) are hard to separate from a vector
and the judge will confuse them; report family-level accuracy too. **E 4, F 4.**

### P2 (b). One neuron as the fitness signal for behavior elicitation
**Question.** Transluce spent **22.1M subject-model samples on Qwen3.6-27B** (10.3M Bloom, 11.8M
evolutionary PRBO incl. proposal-model samples; `search_compute.parquet`), **10.5M on the 11 behaviors that
were found, median 0.77M per behavior** (range 0.23M fabricated-code-execution to 1.9M cutting-instructions /
laser-at-aircraft). Each judged rollout costs a 27B generation (<=1024 tokens) plus a Gemma-4-31B judge
pass. Can a validated neuron's activation at the **last prompt token** (one prefill, no decode, no judge)
replace the judged rollout as the search fitness?
**Procedure.**
1. *Zero-generation validation (do this first, it needs no search).* Prefill all 403 Qwen3.6-27B prompts
   (and the ~2.2k prompts written for the other five models as a transfer set), record every validated
   neuron's activation at the last prompt token and at the first assistant-template token. Correlate with
   the per-prompt 64-rollout rate: Spearman within behavior (n = 7-71 prompts per behavior), AUROC for
   rate >= 25% vs < 5% (71 vs 107 prompts), and cross-behavior specificity (does the 911 neuron fire on
   drunk-driving prompts?). Controls: 20 random same-layer neurons; a dense logistic probe on the full
   residual (expected to win; report it honestly as the ceiling); the Oracle Lens "does this activation
   mention <behavior>" rate (ties to P5). ~400 + 2.2k prefills, **<10 min GPU**.
2. *Search.* If Spearman >= 0.5 for a behavior: run a PRBO-shaped loop with Haiku as mutator (population
   160, 20 generations, fitness = neuron activation z-scored against 500 WildChat prompts, with the
   WeirdChat user-rubric naturalness judge as a constraint so we do not evolve jailbreaks). 3.2k prefills,
   **~5 min GPU** plus Haiku calls. Then judge the top 50 prompts x 64 rollouts with the transcript rubric
   (3.2k rollouts, **~25 min GPU**). Repeat with (a) a random-neuron fitness and (b) the dense-probe fitness.
3. Report: match rate of discovered prompts (vs WeirdChat's per-prompt median 9%), number of new patterns
   at >= 10%, and **Goodhart rate** = fraction of top-fitness prompts whose judged rate is < 5% (prompts
   that fire the neuron without the behavior; these are also free data about what the neuron actually
   detects, feeding the Neuron Card's near-miss section).
**Savings estimate (est.).** Per evaluation: a judged rollout is ~250 prefill + ~400 decode tokens on 27B
plus a 31B judge over ~700 tokens ≈ 80 TFLOP, vs ~7 TFLOP for a prefill truncated at layer L (~11x in
FLOPs; **~30-250x in A100 wall-clock** because decode is memory-bound: ~0.4 s/rollout at batch 64 vs
~2-5 ms/prefill). Per candidate: PRBO needs 5 samples per individual and at base rates of 1-10% a Bernoulli
fitness needs ~1/p samples to register a hit, whereas the neuron gives a graded score from one prefill
(**~5-50x fewer evaluations**). Compounded: **two to three orders of magnitude on the search phase**;
against the 0.77M median per behavior, a full run above is ~6.4k subject-model calls (3.2k prefills +
3.2k verification rollouts), i.e. ~100x fewer calls and far cheaper calls. Headline framing: "the search
that cost Transluce ~770k samples per behavior costs ~6k once you know the neuron."
**Surprising if:** the neuron matches the dense probe (a 1-D causal feature as good as a 5120-D
correlational one); or if the Goodhart rate is low (< 20%), which would mean the neuron is a tight
detector of the situation and not just a word-writer. Also surprising, and worth reporting, if the neuron
fires on *other* models' WeirdChat prompts in proportion to those models' rates (a cross-model weirdness
scale).
**Could go wrong.** Rates were measured on FP8; bf16 drift (Phase 0 replication guards this). Low
within-behavior n for purging/calorie/sexual-advances (7-9 prompts): pool by family. The honest likely
outcome is "dense probe > neuron >> random"; still a methods result. **E 5, F 5** (step 1 is nearly free
and only needs the validated neurons).

### P3 (c). On-manifold vs off-manifold single-neuron steering
**Question.** The push `δ = dose · median_norm/‖w‖ · w` adds a fixed vector regardless of context; does
steering that stays on the model's own activation manifold give the same behavior at higher coherence and
specificity (Goodfire/Wurgaft "find the geometry, not the direction"; GLP's fluency gains)?
**Procedure.** For 3-5 validated poles, six matched-strength arms at the dose that gives ~30% behavior for
the additive push:
(A) additive push (ours); (B) **clamp** the neuron's own post-SiLU activation to its 99.9th percentile over
natural text (the write then scales with the model's own gate: the 1-D on-manifold version);
(C) **scale** the neuron's activation by k (Kazemi-style); (D) **covariance-shaped push**: project δ onto the
top-k (k = 64, 256) principal components of the layer-L residual covariance from 1M natural tokens (the
cheapest "on-manifold" proxy, no training); (E) **mini-GLP** SDEdit: train a 0.3B flow-matching MLP on
~10M layer-L activations (collection ~1 GPU-h, training ~1-2 GPU-h est.), apply the paper's recipe
(t_start 0.5, 20 steps) to `h + δ`; (F) **diff-in-means** vector of matched-minus-unmatched at L (the
natural multi-dim analogue, as a ceiling). Strength is matched by titrating each arm to equal behavior
rate, then comparing: coherence 0-10 judge, perplexity under unsteered 27B, distinct-4-gram ratio,
off-target behavior rates (the other 10 behaviors), MMLU-100/GSM8K-50 deltas, and the Oracle Lens
"off-manifold score" = AR reconstruction FVE of the steered activation (a free manifold-distance proxy:
readers trained on natural activations should reconstruct on-manifold states better).
**Surprising if:** the additive push is *as* coherent as clamping at matched behavior (would argue the
neuron's write direction is already on-manifold, consistent with Cammarata's "already clean"); or if the
covariance projection kills the behavior (would mean the behaviorally active part of w lies in
low-variance directions, which is exactly the kind of thing a GLP would also remove). Either way it bears on
whether "steer along the manifold" matters for *single neurons* as opposed to probe directions.
**Cost.** Arms A-D, F: 5 poles x 6 arms x 3 doses x 30 prompts x 16 samples ≈ 43k generations ≈ 5 GPU-h.
Arm E adds ~3 GPU-h. **Total 5-8 GPU-h.** **Could go wrong.** Mini-GLP on 10M tokens may be too weak to
denoise sensibly (the paper's smallest run used 1B tokens); keep E as a stretch and let D stand in. Matching
strength across arms is fiddly; pre-register the titration rule. **E 3, F 3.**

### P4 (d). Cross-scale homology by lens readout of write directions
**Question.** Do WeirdChat neurons in 27B have homologs in Qwen3.5-4B (and 35B-A3B)? Write directions live
in different bases, but the J-lens maps each into the shared Qwen vocabulary, which is a common coordinate
system across scales.
**Procedure.** (1) For each validated 27B pole, `r_27 = softmax(W_U norm(J_L W_down[:,n]))` (also R-lens,
also the template lens phrase vector). (2) For **every** neuron x sign of Qwen3.5-4B (294,912 x 2) compute
`r_4` the same way with the 4B J-lens (GPU, ~minutes; keep top-100 tokens per neuron). (3) Rank 4B neurons
by top-50 Jaccard, by KL on the top-100 union, and by cosine in unembed space; take the top-20 per 27B pole
and the top-20 by **activation signature** (Spearman of activations across the 403 prompts, both models)
as a second, independent ranking. (4) Verify locally on the M4 (free): steer each candidate in 4B at dose
0.8 both signs on the 3 WeirdChat-derived prompts from S5, judge, and compare with S5's exhaustive sweep
rank (if the lens-matched candidate is in S5's top 1% for the same behavior, the homology is real).
(5) Repeat 27B -> 35B-A3B with the MoE J-lens (experts complicate "neuron"; use expert-neuron columns).
**Metrics.** Precision@20 of lens-match against S5's causal sweep; agreement between the two rankings
(lens vs activation signature); layer-depth relation (relative depth of homolog vs source).
**Surprising if:** lens-matched homologs replicate the behavior causally at a rate far above random 4B
neurons (would be the first cross-scale neuron homology by *readout*, and makes the J-lens a tool for
transferring neuron findings across models); or if homologs exist by activation signature but not by lens
(write directions diverge while detectors agree).
**Cost.** 27B part: minutes. 4B exhaustive readout: ~30 min GPU or a few hours CPU. Verification: local.
**Total < 1 GPU-h** plus S5. **Could go wrong.** 4B may not exhibit the behavior at all (Phase 0 checks);
J-lens token readouts are dominated by generic tokens, so compare after removing the top-500 most frequent
lens tokens across all neurons (an IDF correction). **E 4, F 4.**

### P5 (e). Readers as forecasters: calibration against 64-rollout empirical rates (my pick for the post)
**Question.** Nobody has graded a reader against a *probability*. WeirdChat gives one per prompt. At the
last prompt token the model has not emitted anything, yet the 64 rollouts tell us how often it will deny
being an AI. Does the Oracle Lens, read at that position, mention the behavior with a frequency that tracks
the empirical rate, and does the validated neuron mediate that reading? The blog headline writes itself if
it works: "Qwen decides to pretend it is human before it says a word; one neuron carries the decision; the
model's own activation reader can see it."
**Procedure.** (1) Read the last-prompt-token (and first-template-token) activation of all 403 prompts
through the Oracle Lens at layers 20-60 step 4, 5 samples each (~22k reads, ~1.2 GPU-h); through J/R/
template lenses (free); through Patchscopes "The assistant is about to"; and through unadapted self-report
("Before you answer, will you claim to be human? yes/no" with logprob of yes). (2) Judge each readout for
whether it mentions the behavior (Haiku, with the WeirdChat transcript rubric as reference). (3) Calibration:
bin prompts by empirical rate (0-5%, 5-15%, 15-30%, >30%); plot reader mention-rate per bin; report
Spearman and ECE, per behavior and per layer. Floors: readers applied to the *user prompt text* alone (a
Haiku prompt-only predictor; CHIVE's point), and the readout for a lexically matched near-miss prompt set.
(4) Mediation: steer the validated neuron at the same position (dose ±0.5, ±1.0) and re-read; does the
reader's mention-rate move with the neuron, and by the same amount that the behavior rate moves (from Phase
2 cards)? Random same-layer neurons as control.
**Surprising if:** reader mention-rate tracks the empirical rate (Spearman > 0.4) *above* the prompt-only
floor, i.e. the activation carries information about propensity that the prompt text does not expose to a
strong LLM; or if the neuron shifts the behavior but not the reading (reader blind spot, cf. 2607.23379), or
the reading but not the behavior (reader reads the neuron's write, not the decision). A negative result
("readers are no better than Haiku reading the prompt") is a clean, publishable extension of CHIVE to
probabilistic ground truth.
**Cost.** ~1.5 GPU-h readouts + ~0.5 GPU-h mediation + ~30k Haiku calls. **Total ~2 GPU-h.**
**Could go wrong.** Rates are FP8 and ours is bf16: check that bf16 rates correlate with FP8 rates on 40
prompts first (Phase 0). Oracle Lens is trained on natural text positions, chat-template positions may be
off-distribution; read at the last *user* token too. **E 5, F 4.**

### P6. Round-trip causal fidelity: reader -> edit -> reconstructor -> steer
**Question.** The NLA paper edited an explanation ("rabbit" -> "mouse"), reconstructed a vector and steered
with it. `olens_and_ar` ships an AR for Qwen3.6-27B (text -> layer-L vector). Does the *text* a reader writes
about a validated neuron, when mapped back to a vector, do what the neuron does?
**Procedure.** For each pole: v1 = AR(Oracle Lens bullets for w); v2 = AR(the Neuron Card's human label,
e.g. "the assistant claims to be a human with a body"); v3 = AR(negated label); v4 = AR(label of a
*different* behavior). Steer with each at the neuron's dose (norm-matched) and at 2x; measure the 11
behavior rates + coherence vs the neuron push, the diff-in-means vector, and random directions. Also report
`cos(v2, w)` and `cos(v2, diff-in-means)`.
**Surprising if:** AR(label) induces the behavior at > 50% of the neuron's rate (a text-specified steering
vector that matches a causally found neuron would be a strong "the reader-reconstructor pair captures the
causal variable" result); or if AR(negated label) suppresses the natural behavior on WeirdChat prompts.
**Cost.** 15 poles x 4 vectors x 2 doses x 30 prompts x 16 samples ≈ 58k generations ≈ 6 GPU-h; trim to 5
poles for ~2 GPU-h. **Could go wrong.** AR FVE is 0.12-0.34, so reconstructed vectors are mostly noise;
the AR's `[Layer L]` tag may not cover the neuron's layer (contract 20-63). **E 4, F 3.**

### P7. Does the reader see the intervention? Steering detection and dose estimation
**Question.** "Current AOs are hard to use" found the one clear AO win was detecting a subtle steering vector.
Can the Oracle Lens detect *which* neuron was pushed, and at what dose, from a single activation? This is
also a monitorability result: can an activation reader catch a backdoor-style single-neuron edit?
**Procedure.** Benign prompts x {no steer, each validated pole, random neurons} x doses {0.25, 0.5, 1.0};
read at L_p+8 and L_p+16 with Oracle Lens, J-lens, template lens; judge: (i) steered vs not (AUROC per
dose), (ii) which behavior (12-way), (iii) dose rank correlation. Baseline: Haiku reading the generated text
for the same cells (at dose 0.25 the text should look normal).
**Surprising if:** readers detect dose 0.25 pushes that text-readers miss (replicates the pirate-vector
finding for neurons) and name the behavior (would make the Oracle Lens a usable monitor for weight/activation
tampering). **Cost.** ~6k reads + 6k short generations ≈ 1 GPU-h. **E 3, F 5.**

### P8. Cheap lens pre-screen vs causal truth: precision-recall of S4 over the whole shortlist
**Question.** S4 reads every candidate's write direction through lenses before causal validation. Once Phase
2 has judged the whole shortlist (~150 candidates, winners *and* losers), we can score the lens as a
screen: for each candidate, did the lens (J/R/template/Oracle) predict the behavior that steering later
produced? This is the pre-registered version of P1 over the full shortlist, not just winners, and it
answers "could a lens have found these neurons for free?"
**Procedure.** Freeze S4 readouts before Phase 2; after Phase 2, compute precision@k, recall, and AUROC of
"lens mentions behavior b" against "steering produces behavior b at >= 20%". Compare to the activation-
contrast screen S1 and to attribution S3 as alternative screens. Report which layers' lenses work.
**Surprising if:** lens-only screening recovers > 50% of validated poles at precision > 30% (then the next
neuron hunt needs no activation capture at all). **Cost.** Zero extra GPU beyond S4 and Phase 2; judging
only. **E 3, F 5.**

### P9. WeirdChat meta-tokens: J-lens sweep for algorithm tokens, steered, vs the neuron
**Question.** The meta-tokens post found 什么意思 / 大概率 / gcd as J-lens "algorithm" tokens on Qwen3.6-27B.
Do WeirdChat prompts carry a meta-token at the last prompt token (e.g. 假装 "pretend", 扮演 "role-play",
同意 "agree") whose rank tracks the empirical rate, and does negative steering of its J-lens vector reduce the
behavior as much as suppressing the neuron?
**Procedure.** Follow the post's recipe: J-lens top-20 at layers 20-62 for the 403 prompts; tokens enriched in
high-rate vs low-rate prompts (Fisher test, IDF against WildChat); candidates steered negatively with
per-layer J-lens vectors at chat-template positions, coefficient swept to coherence, 50 rollouts, random
vectors as control; compare the rate drop with the neuron's suppression, and test whether the meta-token
vector and the neuron's write direction are aligned (cosine, and whether steering one moves the other's
readout).
**Surprising if:** a Chinese meta-token predicts weirdness across *several* behaviors (a lens-level hub
matching the H1 hub-neuron hypothesis), or if the neuron and the meta-token are the same direction.
**Cost.** Readouts free; steering 10 tokens x 5 prompts x 50 rollouts x 2 (control) ≈ 5k generations ≈ 1
GPU-h. **E 4, F 4.**

### P10. Template-lens monitor panel vs neuron monitor at generation time
**Question.** For a deployable fix we need a *detector* as well as a lever. Mint a 40-phrase WeirdChat
template panel (ec75hash recipe, ~20 passages per phrase, Σ shipped) and compare, token by token on the
WeirdChat transcripts, its AUROC for matched vs unmatched rollouts against the validated neuron's
activation, against the ec75hash deception/honesty panel, and against a dense probe. Report the earliest
token at which each monitor separates matched from unmatched (the "commitment point"), which is also the
answer to where S1's pre-commitment window really is.
**Surprising if:** the single neuron separates rollouts *earlier* than the phrase panel (the decision
precedes the words), or if the honesty/deception panel lights up on denying-ai-identity but not on
fabricated-code-execution (a split within "misrepresentation"). **Cost.** Teacher-forcing all 27k 27B
transcripts: ~5M tokens prefill ≈ 20-30 min GPU; panel minting ~10 min. **E 3, F 5.**

### P11 (optional). Second WeirdChat subject: the 35B-A3B Activation Oracle
`stanleytheli/qwen3.6-35b-a3b-ao` reads layers 23-27 of the *other* WeirdChat Qwen. If A3 runs anything on
35B-A3B, repeat P5 step 1 there with a QA-style oracle ("Will the assistant claim to have called 911?")
and compare the two readers' calibration on the 7 behaviors the models share. Model fits in bf16 (70 GB)
with the LoRA; tight. **E 3, F 2.**

### Summary table

| # | Proposal | GPU-h (est.) | E | F |
|---|---|---|---|---|
| P1 | Readers vs causal truth, 12-way bench on write directions and deltas | ~2 | 4 | 4 |
| P2 | Neuron as elicitation fitness (step 1 is free; full search + verify ~1) | ~1 | 5 | 5 |
| P3 | On- vs off-manifold single-neuron steering | 5-8 | 3 | 3 |
| P4 | Cross-scale homology via J-lens readout of W_down | <1 (+local) | 4 | 4 |
| P5 | Readers as forecasters vs 64-rollout rates, with neuron mediation | ~2 | 5 | 4 |
| P6 | Round-trip: reader text -> AR vector -> steer | 2-6 | 4 | 3 |
| P7 | Steering detection and dose estimation by readers | ~1 | 3 | 5 |
| P8 | Lens pre-screen precision/recall over the full shortlist | 0 | 3 | 5 |
| P9 | WeirdChat meta-tokens in J-space, steered | ~1 | 4 | 4 |
| P10 | Template panel vs neuron as a token-level monitor | ~0.5 | 3 | 5 |
| P11 | 35B-A3B oracle replication | 2-3 | 3 | 2 |

Everything above together is ~15-25 GPU-h, inside the ~100 h shared budget, but it all depends on Phase 2
delivering validated poles. P2 step 1, P5 step 1-3, P9 readouts and P10 can run on *candidates* before
validation and are cheap enough to rerun once the cards are final.

---

## 3. Shortlist for this week

1. **P2, neuron as elicitation fitness.** Step 1 is ~10 GPU-minutes and uses ground truth that already
   exists (403 prompts x 64 rollouts). It is the result Transluce would care about most ("~770k samples per
   behavior -> ~6k once you know the neuron"), it produces the "weirdness probe" headline from PROPOSAL §7.2,
   and its failure mode (dense probe wins) is still a useful number. Run it the moment A1-A3 have a candidate.
2. **P5, readers as forecasters.** Same prefill pass as P2, plus ~1.5 GPU-h of Oracle Lens reads. First
   reader eval with probabilistic ground truth, a direct extension of CHIVE, and the mediation step ties the
   reader to our causal neuron. Positive or negative, it is a self-contained post section.
3. **P1, readers vs causal truth** (with P8 as its free byproduct and P4 as a one-hour add-on if S5 runs).
   ~2 GPU-h; the mini-WorkspaceBench with ground truth we control. P4 goes in if the 4B track finds the
   behaviors, because the J-lens readout of `W_down` costs nothing and a verified cross-scale homolog would
   be new.

Defer P3 (manifold) and P6 (round-trip) to the second week unless a pole shows a coherence problem at the
lever dose, in which case P3 arms B and D are the fix to try first.

## 4. Open items to resolve before running

- Confirm the Oracle Lens injection can be reimplemented without `global_workspace` (read `adapter_config`,
  the chat template in the card, and check `agu18dec/local-workspace`). Half a day if not.
- Ask the WorkspaceBench authors whether the L42 NLA for Qwen3.6-27B is public. If yes, add to P1/P5.
- Fix the S1 position caveat in PROPOSAL.md (last-prompt-token contrast within a prompt is identically zero).
- Verify Qwen3.5-4B and Qwen3.6-27B share the tokenizer/unembedding vocabulary (needed for P4 as written;
  if not, compare in decoded-string space).
- Budget: reserve ~8 GPU-h for P1+P2+P5 and run them on the same loaded model session to avoid reloading the
  54 GB checkpoint.

## Sources consulted
transformer-circuits.pub/2026/nla; arXiv 2610.30934, 2610.18727, 2512.15674, 2606.02609, 2605.26045,
2607.23379, 2602.06964 (+ generative-latent-prior.github.io, github g-luo/generative_latent_prior),
2605.05115, 2608.16747 (CHIVE; alignment.anthropic.com/2026/chive), 2604.08169, 2609.09902, 2609.10210;
HF agu18dec/olens_and_ar, agu18dec/oracle-lens-ddp600, camilablank/workspace-lenses,
ec75hash/template-lens-qwen3.6-27b, kitft/nla-models, stanleytheli/qwen3.6-35b-a3b-ao,
ceselder/qwen3-8b-ao-v3-*, caiovicentino1/qwen36-27b-sae-multilayer; github anthropics/jacobian-lens,
camilablank/workspace-bench, adamkarvonen/activation_oracles, kitft/natural_language_autoencoders;
AF "WorkspaceBench", AF "Towards surfacing model algorithms with meta-tokens in the J-Space", LW "Current
activation oracles are hard to use", LW "Manifold steering from a natural state", alignment.anthropic.com/
2026/introspection-adapters; transluce.org/weirdchat (methods appendix); neuronpedia.org/qwen3.6-27b;
`data/weirdchat/data/search_compute.parquet`, `prompts/qwen__qwen3.6-27b/*.parquet`.
