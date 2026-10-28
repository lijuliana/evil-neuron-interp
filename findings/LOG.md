# Experiment log (append-only; 5 lines per entry)

Format:
## YYYY-MM-DD HH:MM  agent  short title
- what: 
- result: 
- next: 
- data: path

## 2026-10-28 03:30  A7  meta-model / lens brainstorm
- what: desk research on NLAs, Oracle Lens (agu18dec/olens_and_ar), J/R/template lenses (camilablank/workspace-lenses), WorkspaceBench, CHIVE, Activation Oracles, GLP, manifold steering, meta-tokens; costed 11 proposals combining them with validated neurons.
- result: ideas/metamodels.md. Top 3: P2 neuron-as-elicitation-fitness (27B search cost 22.1M samples, 0.77M median/behavior; probe-guided est. ~6k calls), P5 readers calibrated against 64-rollout per-prompt rates with neuron mediation, P1 readers vs causal ground truth 12-way bench. Flag: S1 within-prompt contrast at last prompt token is identically zero (deterministic prefill); fix PROPOSAL.md.
- next: orchestrator picks from shortlist; A0 to confirm Oracle Lens injection reimplementable without `global_workspace`; ask WorkspaceBench authors about the L42 NLA for Qwen3.6-27B.
- data: ideas/metamodels.md (no runs)

## 2026-10-28 16:30  A6  lit review (single-neuron steering, neuron circuits, WeirdChat, persona space)
- what: verified ~50 sources (arXiv abstract pages, GitHub/HF/blog primaries); thematic synthesis + gap analysis + controls table + .bib run through citation-checker
- result: both planned novelties confirmed as gaps (cross-behavior transfer of single-neuron interventions; same-prompt matched/unmatched rollouts as contrast). Partial scoops: Goodfire drunk-driving (one F2 behavior, clause deletion, 5+12 neurons, held-out n=48; primary X thread unreadable, podcast gives a one-neuron account), Liu 2606.19831 (forward-only contrastive screen + budget-matched random-neuron controls + dose law matching our steering unit), Li&Subramani 2605.08348 (neuron-circuit specificity across capability tasks). Key cautions: Batutin random-init labelling control (84% of noise neurons labelled "one job"); Qwen3-8B sweep shows 3.9% of neuron-direction pairs flip identity on one prompt; gradient vs activation-contrast ranking conflict (Hedwig/Kazemi vs Liu).
- next: A0/A1 adopt design implications in lit/review.md (template-ablation control for F1, random-init noise floor for explanation battery, keystone-neuron filter before hub scoring, typed judging, Assistant-Axis cosine readout in S4); A2 reproduce Goodfire sober-one deletion on 27B first
- data: lit/review.md, lit/refs.bib, lit/controls_table.md

## 2026-10-28 06:40  A0  VM setup + Qwen3.6-27B harness
- what: weirdchat-a100 (A100-80GB) venv: torch 2.9.1+cu129, transformers 5.21.0, fla 0.6.0, causal-conv1d 1.7.0; model at /data/models/Qwen3.6-27B (52 GB bf16). src/harness.py: chat / calibrate / steer / clamp / ablate_mean / capture_batch / capture_prompt_level.
- result: load 14.8 s, 54.7 GB. Gen 477 tok/s (batch 64). Capture 4-9 transcripts/s. Dose-0 steer is bit-identical; dose-1 push norm == median resid norm (L30: 71.4). Clamp verified. Positions verified by decoding (last_prompt = final "\n\n" after </think>). Same-prompt rollouts have identical last_prompt acts (deterministic prefill), so within-prompt contrasts use asst_1..8 + asst_mean only.
- next: S1 capture (scripts/03_a0_s1_capture.py), summary (04), baseline (05). Numerics caveat: bf16 acts are padded-shape dependent (0.5% of neurons move >5%); handled by fixed padding multiple + shuffled-label floor.
- data: captures/qwen3.6-27b_resid_norms.json, runs/a0_env.md, scripts/0{0,1,2,2b}_a0_*.py

## 2026-10-28 02:50  A4  Qwen3.5-4B MLX harness, evil-neuron reproduction, WeirdChat replay (PARTIAL, stopped)
- what: built src/mlx_steer.py (residual push on down_proj output per AGENT_BRIEF, clamp/mean-ablation, MLP-activation capture, batched T=1 sampling; calibration cached to captures/qwen3.5-4b_resid_norms.json) on mlx-community/Qwen3.5-4B-MLX-8bit; smoke test of L14N8949 at doses 0/+0.7/+1.0/-1.0 + evil pair + 3 random L14 neurons (4 prompts x 4 samples); regex pre-screen (src/heuristic_judge.py) calibrated against shipped WeirdChat labels; replay of 20 prompts x 16 samples x 17 behaviors started, STOPPED after chemtrails (1/17) because the Mac overheated.
- result: evil neuron reproduces: +1.0 -> 10/16 samples hostile ("designed to deceive, corrupt, and subvert"; "steps you should take to harass and control your neighbor"), coherent (distinct-4 0.98); dose 0 0/16; -1.0 0/16 hostile but rambling (not simply "polite"); random L14 neurons 0,0,2/16. Chemtrails on 4B: heuristic 36/320 = 11.3% [8.2, 15.2] but hand-read ~6-10/320 (~2-3%) plausible vs 16.7% reference on Qwen3.6; behavior exists weakly. Heuristic P/R: strong (P>0.9) for code-exec, 911, AI-denial, embodiment, language-switch, laser, device-access, drunk-driving; weak for chemtrails/cutting/purging/calorie/jumping/strike/sexual/user-name. No ANTHROPIC_API_KEY -> everything UNJUDGED.
- next: continue the 16 remaining behaviors on the AWS L40S with src/harness.py using runs/a4_replay/selected_prompts.json; LLM-judge chemtrails-assertion.jsonl and runs/a4_smoke/smoke.jsonl with src/judge.py; measure L14N8949 collapse dose and matched-norm random-direction controls; no more local MLX generation.
- data: runs/a4_smoke/{smoke.jsonl,smoke.md,summary.json}, runs/a4_replay/{chemtrails-assertion.jsonl,summary.md,summary.json,heuristic_calibration.md,selected_prompts.json}, captures/qwen3.5-4b_resid_norms.json, findings/neurons/qwen3.5-4b/L14N8949+.md

## 2026-10-28 03:05  A8  prompt sets for the 27B track
- what: authored controlled prompt sets from the 11 Qwen3.6-27B WeirdChat behaviors (top-20 prompts by matched fraction each, read with pandas; 5 matched + 5 unmatched transcripts skimmed per behavior to locate each behavior's trigger clause). No model runs.
- result: 184 S2 rows (minimal benign_rewrite + near_miss + length-matched distant_control + edit_note per prompt; 8 behaviors at 20, extreme-calorie-restriction 8, purging-advice 9, unsolicited-sexual-advances 7 = all available), 30 induction prompts with zero trigger-vocabulary overlap, commitment tokens for 6 of 11 behaviors (5 are judge-only: laser, cutting, purging, calorie-restriction, sexual-advances), 200 capability items (100 MCQ / 50 arithmetic machine-verified / 50 over-refusal). Minimal-pair jaccard median 0.72; distant-control length ratio median 1.00; validator reports 0 problems.
- next: A0/A1-A3 should sample the benign_rewrite column before trusting S2 (an untested rewrite is not a control), and re-sort by bf16-replicated rates since matched fractions here are FP8. IFEval still missing from the capability battery.
- data: promptsets/ (committed) and data/promptsets/ (gitignored copy); see promptsets/README.md section 5 for the mapping onto PROPOSAL.md sections 4 and 5.
