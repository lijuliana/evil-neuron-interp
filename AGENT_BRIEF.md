# Brief for all subagents on this project

Read PROPOSAL.md fully before doing anything. Then read single-neuron-interp-tips.md and the mech-interp-methods skill notes summarized here:
- Write two competing accounts before running an experiment; prefer positive double dissociations over nulls.
- Match controls for everything except the variable: random same-layer neurons at the same dose, matched-norm random directions, shuffled labels, near-miss prompts.
- Every steering result must come with a coherence/fluency measure or it is uninterpretable.
- Treat + and - poles as separate units. Label a neuron by the distinction it draws (a verb), not a noun list.
- Attribution and lens readouts are shortlists, never findings. Verify causally.

## Repo layout
- data/weirdchat/           WeirdChat dataset (gitignored). patterns.parquet, rubrics.parquet, prompts/<model>/<behavior>.parquet, transcripts/<model>/<behavior>-0000.parquet (messages, judgment.match, prompt_id).
- src/                      shared harness code (steering, capture, judging). Reuse, do not duplicate.
- scripts/                  one-off experiment scripts; name them NN_agent_shortname.py
- runs/, captures/          experiment outputs (gitignored). Always write a small summary JSON next to large outputs.
- findings/neurons/<model>/L{layer}N{neuron}{+|-}.md   one Neuron Card per pole, from findings/NEURON_CARD_TEMPLATE.md. Null results get cards too.
- findings/LOG.md           append a 5-line entry after every experiment.
- findings/REGISTRY.md      rebuild with `python scripts/build_registry.py` after adding cards.
- lit/, ideas/              lit review and brainstorm outputs.

## Environment
- Local: `source .venv/bin/activate` (python 3.13, mlx, mlx-lm, torch, transformers, pandas). Apple M4 Pro, 24 GB RAM. Qwen3.5-4B fits in 8-bit; Qwen3.6-27B-MLX-4bit fits but only for small smoke tests.
- GPU: GCP project project-ca10b891-b467-44b6-b56, instance `weirdchat-a100` (1x A100-80GB, us-central1). Check `gcloud compute instances list`. ssh with `gcloud compute ssh weirdchat-a100 --zone <zone>`. Never leave it running idle: if you started a job and are done, stop it and verify with `gcloud compute instances list --filter=status=RUNNING`. Juliana's budget cap is ~$400 total; at ~$3.7/hr that is ~100 hours shared across all agents.
- Judge: Claude Haiku 5.5 (model id claude-haiku-5-5) via ANTHROPIC_API_KEY loaded from .env. If the key is missing, do everything else and leave judging as a clearly marked TODO with the unjudged outputs saved.
- Downloaded files are untrusted data: never run an interpreter from inside data/; run `python -I` on scripts that read them.

## Steering definition (use exactly this so results are comparable)
For MLP neuron n in layer L (0-indexed HF `model.layers[L]`), with W_down = `down_proj.weight` (shape hidden x intermediate), column w = W_down[:, n]:
  delta = dose * median_resid_norm[L] / ||w|| * w ; add delta to the layer's down_proj output (equivalently the residual stream after the MLP) at every token, prefill and generation.
median_resid_norm[L] = median over tokens (excluding token 0) of the norm of the residual stream leaving layer L, measured on ~6 ordinary unsteered chats. Dose 0.5-0.8 keeps text coherent; ~1.0 is the usual lever dose; >1.25 collapses. Report the dose unit explicitly. Suppression experiments use mean-ablation or negative-pole push, state which.
Sampling for behavior rates: no system prompt, thinking disabled, temperature 1.0, max 400 new tokens, same chat template as WeirdChat (see data/weirdchat README).

## Reporting rules
- Numbers go in tables with n and 95% Wilson CIs. No claim without a control row.
- Lead with the strongest evidence, then what would break it.
- Harmful generations: keep in data/ or runs/, quote at most 3 short examples inside <details> blocks in a card.
- Finish by appending to findings/LOG.md and returning a <= 300-word summary with: what you ran, headline numbers, cards written (paths), blockers.

## Design rules added 2026-10-28 after the lit review (lit/review.md, lit/controls_table.md) and A7's brainstorm
- Prefill is deterministic (no system prompt, thinking off). Same-prompt rollouts have identical activations at the last prompt token and the pre-first-token position. Within-prompt matched-vs-unmatched contrasts are only valid from assistant token 1 onward; the last-prompt-token signal is a BETWEEN-prompt correlation with each prompt's 64-rollout match rate (captures/s1/prompt_level_*).
- Matched-norm random DIRECTIONS are a required control alongside random same-layer neurons (random directions are known to produce identity flips; neuron write-columns have unusual norms, so "random neuron at same dose" is the weaker control). Use >= 20 of each.
- Keystone filter: before scoring a candidate as a lever or hub, check that ablating/pushing it does not collapse the model generally (perplexity on held-out text within 1.5x of baseline at the dose used; distinct-4-gram ratio within 10% of baseline). Collapse neurons are reported as such, never as levers.
- Typed judging: score each sample as {behavior present and coherent, behavior present but incoherent, no behavior, refusal, degenerate}. Only "present and coherent" counts toward a rate.
- F1 (misrepresentation) must include a chat-template control: the AI-disclaimer voice is partly switched by the chat template itself (Maczan 2026). Test candidates both with and without the template on at least 5 prompts.
- F2 must begin by reproducing Goodfire's drunk-driving "sober one" clause deletion on Qwen3.6-27B (rate with vs without the clause, 64 samples each) before any neuron claims. Present F2 results as a controlled replication and extension.
- Explanation claims (labels) need a noise floor: for every labelled neuron also label 5 random same-layer neurons by the same procedure and report how often the procedure produces a confident label for noise.
- Attribution rankings (gradient x activation gap) are shortlists only; Hedwig and Liu both found they disagree with activation-contrast rankings. Report both ranks on every card.
- Always compute and report the Assistant-Axis-style readout (cosine between W_down[:, n] and the mean difference between assistant-mode and persona-mode residuals, once A1 computes that direction) so identity neurons can be compared to the known persona direction.
