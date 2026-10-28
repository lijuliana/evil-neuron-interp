# PROGRESS

## Decisions (2026-10-28, approved by Juliana)
- Models: Qwen3.6-27B on GCP (primary) + Qwen3.5-4B local MLX (secondary). 35B-A3B optional only if time.
- GPU: A100-80GB on-demand, us-central1; budget up to ~$400 total without checking back. Stop + verify after every job.
- Judge: Claude Haiku 5.5 via ANTHROPIC_API_KEY in .env (Sonnet 5.5 for calibration disputes). Key provided 2026-10-28 (in .env, gitignored); verified for Haiku 5.5 and Sonnet 5.5.
- Behaviors: all 11 Qwen3.6-27B behaviors incl. self-harm family. Harmful outputs stay under data/ (gitignored).
- Output: blog/forum post in ~1 week. Findings as Neuron Cards in findings/, REGISTRY.md, LOG.md, commits per milestone, plus a Claude Doc.
- Autonomy: fully autonomous within budget.

## Phase status
- Phase 0 harness: A0 DONE 2026-10-28. src/harness.py (chat/calibrate/steer/clamp/ablate_mean/capture_batch/capture_prompt_level) verified on GCP `weirdchat-a100` (us-central1-a, a2-ultragpu-1g). Env in runs/a0_env.md. Gen 477 tok/s @batch 64; capture ~9 transcripts/s. STOP THE VM when idle.
- Phase 0 S1 capture: A0 ran scripts/03_a0_s1_capture.py over all 27B transcripts -> VM /data/captures/s1 (behavior/, shuffled/, prompt/ npz + prompt_level_acts.npy); summaries in captures/s1/ locally (see findings/LOG.md for numbers).
- Phase 0 baseline replication: A0 sampled 20 prompts x 64 per behavior in bf16 (runs/baseline_27b/*.jsonl). JUDGING PENDING: no ANTHROPIC_API_KEY in .env; run `python -I src/judge.py calibrate` then `python -I src/judge.py baseline` once the key exists (src/judge.py has cache + dry-run).
- Phase 1 screens: S1 DONE (captures/s1/summary). S2 (clause deletion) runs inside A2. S4 lens readouts inside A1-A3. S5 (4B exhaustive) replaced by poor-things.com priors (A4b).
- Phase 2 validation: A1 (F1 misrepresentation), A2 (F2 harmful compliance + Goodfire replication), A3 (cross-family hubs + chemtrails) launched 2026-10-28 ~08:00 UTC; GPU shared via flock; VM auto-stops after 30 idle minutes.
- Lit review (A6): running
- Meta-model brainstorm (A7): running

- 4B local track (A4): MLX harness done (src/mlx_steer.py; calibration cached in captures/qwen3.5-4b_resid_norms.json). Evil neuron L14N8949 +1.0 reproduced on Qwen3.5-4B-MLX-8bit with random-neuron controls (runs/a4_smoke/, card findings/neurons/qwen3.5-4b/L14N8949+.md). WeirdChat replay on 4B STOPPED after 1/17 behaviors (chemtrails) because the Mac overheated; partial results in runs/a4_replay/ (summary.md marked PARTIAL). Heuristic pre-screen calibrated vs shipped labels (runs/a4_replay/heuristic_calibration.md). 4B track moves to AWS L40S with src/harness.py (transformers); no more local MLX generation. Judging: UNJUDGED (no ANTHROPIC_API_KEY).

## Links
- Living doc: https://claude.ai/artifact/AHGGWnNe8u9Uj8znQrmC4e
