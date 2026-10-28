# Phase 1b/2 brief for discovery + validation agents (A1, A2, A3). Read AGENT_BRIEF.md and PROPOSAL.md first.

## State on 2026-10-28 ~08:00 UTC
- S1 is done. Local summaries: captures/s1/summary/ (summary_{behavior}.parquet: columns layer, neuron, position, d, n_matched, n_unmatched; summary_promptlevel_{behavior}.parquet: per-neuron Spearman rho between last-prompt-token activation and the prompt's 64-rollout rate; noise_floor.parquet: shuffled-label max|d| per behavior/position; promptlevel_null.parquet; hub_candidates.parquet; hub_thresholds.json). Full arrays on the VM at /data/captures/s1 (behavior/, shuffled/, prompt/ per-prompt groups for 181 prompts, prompt_level_acts.npy [403 prompts x 64 x 17408]).
- Positions: last_prompt = the final "\n\n" after `</think>` in the chat template; asst_k = position of the k-th sampled assistant token (k=0 is identical across rollouts of a prompt; only asst_1..8 and asst_mean carry within-prompt signal).
- IMPORTANT caveat on S1: a large |d| at asst_k on teacher-forced text may mean the neuron READS the first k sampled tokens (e.g. matched 911 replies start with "I've contacted"), not that it causes the behavior. Only the causal battery decides. Treat S1 as a shortlist.
- Noise floor (shuffled labels, max |d| over all 1.1M neurons): ~0.21-0.50 for behaviors with >=121 matched; 0.72-1.34 for unsolicited-sexual-advances, purging-advice, extreme-calorie-restriction (tiny n; treat their S1 leaders as weak).
- Baseline bf16 sampling of 11 behaviors x 20 prompts x 64 samples is running on the VM (scripts/05_a0_baseline.py, ~560 tok/s, started 07:36 UTC, ~2.5 h). Output: VM /data/runs/baseline_27b/{behavior}.jsonl. UNJUDGED until an API key exists.
- Harness: src/harness.py (Harness.chat / calibrate / steer / clamp / ablate_mean / capture_*). Residual norms: captures/qwen3.6-27b_resid_norms.json. Judge: src/judge.py (Haiku 5.5, Transluce rubric, cache, dry-run) and src/heuristic_judge.py (regex pre-screen; precision >0.9 for code-exec, 911, AI-denial, embodiment, laser, drunk-driving, language-switch, device-access; weak for the rest).
- Prompt sets: promptsets/s2_rewrites.jsonl (original / benign_rewrite / near_miss / distant_control per prompt), promptsets/induction_benign.jsonl (30), promptsets/commitment_tokens.json, promptsets/capability_mini.jsonl (200).
- Lenses for Qwen3.6-27B: HF camilablank/workspace-lenses (qwen3.6-27b/j-lens/lens.pt, r-lens/lens.pt, template-lens/). W_down columns can be sliced from the safetensors on the VM CPU without the GPU.

## GPU protocol (one A100-80GB shared by three agents; the 27B occupies 55 GB so only ONE model instance fits)
- VM: weirdchat-a100, zone us-central1-a. Start if stopped: `gcloud compute instances start weirdchat-a100 --zone us-central1-a` (the external IP may change; always use `gcloud compute ssh/scp`). A watcher stops the VM after 30 consecutive idle minutes (no GPU use, no python); expect to restart it after gaps. Do NOT stop the VM yourself; the orchestrator does.
- Repo on the VM: ~/evil-neuron-interp (venv at ~/venv). Data: /data/weirdchat, model /data/models/Qwen3.6-27B, outputs /data/runs and /data/captures.
- Wrap EVERY GPU python invocation in `flock -w 36000 /data/gpu.lock <cmd>` so jobs serialize. Batch many conditions into one invocation (one model load, 15 s), and keep each invocation under ~45 minutes so the other agents get turns. Log to /data/runs/<agent>_<job>.log and run under nohup.
- Until the baseline job finishes (~10:00 UTC) the GPU is full; do CPU/analysis work first (below). Check with `pgrep -f 05_a0_baseline`.
- Copy small results back to the local repo (captures/, runs/<agent>_*/summary files). Large jsonl stays on the VM; list paths in cards.

## Shared validation protocol (so cards are comparable)
Per candidate pole (layer, neuron, sign):
1. Keystone check: steer-lens KL over ~32 reference tokens of 5 benign answers at the dose used; broken = KL > 2.5 or top-1 agreement < 0.3. Also held-out perplexity ratio and distinct-4-gram.
2. Suppression on natural prompts: for each behavior in your family, 20 WeirdChat prompts (prefer the baseline's prompts so the baseline jsonl is the control) x 32 samples with (a) mean-ablation of the neuron (mean from the unmatched group in /data/captures/s1/behavior), (b) negative push at dose 0.5 and 1.0. Also 2 behaviors OUTSIDE the family as transfer controls.
3. Induction on benign prompts: promptsets/induction_benign.jsonl x 16 samples at doses {0.5, 0.75, 1.0, 1.25} both signs.
4. Controls: >= 20 random same-layer neurons at the same dose on the same prompts (8 samples each is enough for a null), >= 20 matched-norm random directions; keep the random sets fixed across agents: seed 1234, so random neuron i for layer L is `rng(1234+L).choice(17408, 20)`.
5. Specificity: near_miss and distant_control prompts from s2_rewrites; capability_mini (100 MCQ + 50 arithmetic + 50 over-refusal) at dose 1.0 for any pole that passes step 2 or 3.
6. Judging: if ANTHROPIC_API_KEY exists in .env use src/judge.py with the Transluce rubric and typed outcome {coherent_match, incoherent_match, no_match, refusal, degenerate}; otherwise heuristic_judge pre-screen and then READ the heuristic positives plus a random 10% of negatives yourself, record your verdicts in a jsonl, and mark the card UNJUDGED-BY-API. Wilson CIs everywhere.
7. Explanation: R-lens / J-lens readout of W_down[:, n] (top 15 tokens each pole), top-activating contexts over 2,000 WildChat/pile-10k snippets (one GPU batch), blind-maximum test, and a label phrased as a distinction. Label 5 random same-layer neurons by the same procedure (noise floor).
8. Write one card per pole from findings/NEURON_CARD_TEMPLATE.md; null results get cards too; rebuild the registry; append LOG.md.

Pre-registered verdicts are in PROPOSAL.md §5. Do not soften them.
