# PROGRESS

## Decisions (2026-10-28, approved by Juliana)
- Models: Qwen3.6-27B on GCP (primary) + Qwen3.5-4B local MLX (secondary). 35B-A3B optional only if time.
- GPU: A100-80GB on-demand, us-central1; budget up to ~$400 total without checking back. Stop + verify after every job.
- Judge: Claude Haiku 5.5 via ANTHROPIC_API_KEY in .env (Sonnet 5.5 for calibration disputes). Key NOT yet provided.
- Behaviors: all 11 Qwen3.6-27B behaviors incl. self-harm family. Harmful outputs stay under data/ (gitignored).
- Output: blog/forum post in ~1 week. Findings as Neuron Cards in findings/, REGISTRY.md, LOG.md, commits per milestone, plus a Claude Doc.
- Autonomy: fully autonomous within budget.

## Phase status
- Phase 0 harness: A0 running on GCP `weirdchat-a100` (us-central1-a, a2-ultragpu-1g, created 2026-10-28). STOP IT when idle.
- Phase 1 screens: not started
- Phase 2 validation: not started
- Lit review (A6): running
- Meta-model brainstorm (A7): running

- 4B local track (A4): running

## Links
- Living doc: https://claude.ai/artifact/AHGGWnNe8u9Uj8znQrmC4e
