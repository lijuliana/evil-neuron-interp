# PROGRESS

## Decisions (2026-10-28, approved by Juliana)
- Models: Qwen3.6-27B on GCP (primary) + Qwen3.5-4B local MLX (secondary). 35B-A3B optional only if time.
- GPU: A100-80GB on-demand, us-central1; budget up to ~$400 total without checking back. Stop + verify after every job.
- Judge: Claude Haiku 5.5 via ANTHROPIC_API_KEY in .env (Sonnet 5.5 for calibration disputes). Key provided 2026-10-28 (in .env, gitignored); verified for Haiku 5.5 and Sonnet 5.5.
- Behaviors: all 11 Qwen3.6-27B behaviors incl. self-harm family. Harmful outputs stay under data/ (gitignored).
- Output: blog/forum post in ~1 week. Findings as Neuron Cards in findings/, REGISTRY.md, LOG.md, commits per milestone, plus a Claude Doc.
- Autonomy: fully autonomous within budget.

- 2026-10-28 evening: change of direction (Juliana). Focus moves to Agastya's fork-based mediating neurons, O-lens readouts over all tokens, and a per-token viewer; see PLAN.md. Lever sweep (A8) stopped; A1/A2 told to wrap up with no new GPU work.

## Phase status
- Phase 0 harness: A0 DONE 2026-10-28. src/harness.py (chat/calibrate/steer/clamp/ablate_mean/capture_batch/capture_prompt_level) verified on GCP `weirdchat-a100` (us-central1-a, a2-ultragpu-1g). Env in runs/a0_env.md. Gen 477 tok/s @batch 64; capture ~9 transcripts/s. STOP THE VM when idle.
- Phase 0 S1 capture: A0 ran scripts/03_a0_s1_capture.py over all 27B transcripts -> VM /data/captures/s1 (behavior/, shuffled/, prompt/ npz + prompt_level_acts.npy); summaries in captures/s1/ locally (see findings/LOG.md for numbers).
- Phase 0 baseline replication: A0 sampled 20 prompts x 64 per behavior in bf16 (runs/baseline_27b/*.jsonl, done 10:04 UTC, copied locally). Judging: scripts/07_baseline_judge.py (Haiku on 16/64 rollouts per prompt, regex pre-screen on all) running 2026-10-28 ~16:20 UTC -> runs/baseline_27b/baseline_rates.csv. Judge calibration done (runs/judge_calib: agreement 0.77, precision 0.93, recall 0.57; Sonnet adjudication sides with Haiku on 68/99 misses).
- Phase 1 screens: S1 DONE (captures/s1/summary). S2 (clause deletion) runs inside A2. S4 lens readouts inside A1-A3. S5 (4B exhaustive) replaced by poor-things.com priors (A4b).
- Phase 2 validation: A1 (F1 misrepresentation), A2 (F2 harmful compliance + Goodfire replication), A3 (cross-family hubs + chemtrails) launched 2026-10-28 ~08:00 UTC; CPU steps (shortlists, lens readouts, judge validity, prompt-level probe) done by 09:30. A3's GPU battery ran 15:44-16:13 UTC (runs/a3_battery, judging next). A1's wave-1 launcher and A2's flock chain died with their sessions and never ran; relaunched detached 16:18 UTC (A1 wave 1 then A2 Goodfire S2 -> keystone -> attribution -> screens). GPU shared via flock. Idle watcher fixed 16:17 (old pattern counted system python3 daemons and could never fire).
- Lit review (A6): running
- Meta-model brainstorm (A7): running

- 4B track (A4 local MLX, then A4b on AWS): MLX harness (src/mlx_steer.py) and evil-neuron reproduction done locally; local generation stopped (Mac overheated). A4b 2026-10-28: AWS g5.2xlarge `weirdchat-4b` (i-050eae9f4d3c97cc4, us-east-1, STOPPED), env in runs/a4b_env.md, src/harness_4b.py wraps A0's harness.py for Qwen3.5-4B bf16 (norms within 5% of MLX 8-bit). Evil neuron reproduces in bf16 (53/64 hostile at +1.0 vs 0/64; runs/a4b_smoke/). WeirdChat replay DONE for all 17 behaviors (20 prompts x 32 samples, top_p .95/top_k 20), heuristic + Haiku-judged (positives + 10% negatives): present on 4B = embodiment 0.59, drunk-driving 0.28, laser 0.19, purging 0.13, jumping 0.10, language-switch 0.10, code-exec 0.05, cutting 0.03; absent = 911, device-access, sexual-advances, strike-partner; denying-ai-identity only 0.013 (runs/a4b_replay/summary.md). poor-things priors (8 single-neuron settings, lit/poor-things.md) tested with controls + keystone: personas reproduce, none induces WeirdChat F1 behaviors, all reduce embodiment; L4N8016 -0.8 broken (runs/a4b_priors/summary.md). S5 sweep and homolog search not started.

- Hub question (A3): DONE 2026-10-28. H1 rejected: 8 sign-consistent S1 hubs + 3 chemtrails leaders through the full battery (runs/a3_hubs.md; matrix runs/a3_matrix.{csv,png}); every dose-1.0 suppression drop lies inside the distribution of 20 random same-layer neurons + 20 matched-norm random directions (a 1x-norm push in any direction at L38-61 cuts F1 rates by 20-45 pp), mean-ablation does nothing, induction yields no behavior; 11 null-result cards + L8N12462+ candidate (correlational prompt-level weirdness probe, pooled rho 0.44, transfers across held-out behaviors; untested causally). Chemtrails is judge-limited (2% Haiku baseline).

## Links
- Living doc: https://claude.ai/artifact/AHGGWnNe8u9Uj8znQrmC4e
