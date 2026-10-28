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

## 2026-10-28 07:40  A0  S1 screen capture + summary (Qwen3.6-27B, all 11 behaviors)
- what: scripts/03_a0_s1_capture.py teacher-forced 25,792 transcripts (8.67M tok, 46 min, ~9 tr/s) -> Welford mean/M2 of all 64x17408 post-activations at last_prompt, asst_0..8, asst_mean, grouped by (behavior, match), shuffled labels, and (prompt, match) for 181 prompts with >=8/>=8; plus one forward per unique prompt (403) at last_prompt. scripts/04_a0_s1_summary.py -> captures/s1/summary/.
- result: shuffled-label floor max|d| (asst_1..8, over 1.1M neurons) = 0.21-0.50 for behaviors with >=121 matched, but 0.72-1.34 for the three tiny ones (unsolicited 56, purging 61, extreme-calorie 24 matched): ignore |d|<~0.5 there. Real pooled max|d| at asst_1..8: 0.54-2.25 per behavior; within-prompt mean d standouts: false-physical-embodiment L57N6964 asst_2 d=+2.36 (z=40, 26 prompts), claims-called-911 L33N12974 asst_8 d=-1.64 (z=-17), recommends-drunk-driving L47N10135 asst_1 d=+0.97 (z=13), denying-ai-identity L36N2509 asst_2 d=-0.56 (z=-10). Prompt-level Spearman (403 prompts): permutation null max|rho| over 10k neurons is 0.44-0.60 for n>=33 prompts and ~0.95 for n<=9, so |rho|>=0.5 is meaningless for small behaviors; best real |rho| 0.64-0.87 (claims 0.87, cutting 0.87, denying 0.74). Hub table (calibrated thresholds, captures/s1/summary/hub_candidates.parquet): top neuron L40N702 passes 6 behaviors by pooled d + 1 by rho, but only 3 within-prompt; most top-20 hubs FLIP SIGN across behaviors (hub_top200_signed_d.csv), so they look like response-style features, not a shared switch. Sign-consistent candidates: L40N702, L60N7822, L53N15567, L59N3905 (+), L50N11815, L51N5186 (-). Hubs concentrate in layers 36-63.
- next: A1-A3 should start from the within-prompt leaders and the sign-consistent hubs; verify causally (steer/clamp) with random same-layer controls. Pooled-d and last_prompt/asst_0 contrasts are confounded by prompt identity (deterministic prefill): do not treat them as within-prompt effects.
- data: VM /data/captures/s1/{behavior,shuffled,prompt}/*.npz (2.1G+2.1G+34G), prompt_level_acts.npy (1.7G); local captures/s1/{manifest.json,prompt_index.parquet,behavior/,summary/}

## 2026-10-28  orchestrator  poor-things.com resource
- what: Juliana pointed at poor-things.com (Qwen3.5-4B neuron lab, 689 scored neurotypes, 34-check rubric, KL funnel over 589,824 settings)
- result: lit/poor-things.md written; keystone thresholds (KL>2.5 / top1<0.3) and typed-check vocabulary adopted in AGENT_BRIEF; 11 F1/F2-relevant 4B specimens listed as priors
- next: A4b tests the priors on WeirdChat F1 prompts + benign induction set after the replay; cross-scale comparison once 27B hubs exist
- data: lit/poor-things.md

## 2026-10-28 07:50  A0  Baseline replication (bf16) IN PROGRESS, judging pending
- what: scripts/05_a0_baseline.py on VM: 11 behaviors x 20 prompts (highest shipped matched fraction) x 64 samples, T=1.0/top_p .95/top_k 20/400 tok, batch 80 (~560 tok/s; batch 128 OOMed). Writes /data/runs/baseline_27b/{behavior}.jsonl on the VM. VM auto-shuts down (~/autostop.sh) when the job exits.
- result: pending (~2.5 h total). No ANTHROPIC_API_KEY in .env or env, so judging is a TODO: src/judge.py (Haiku 5.5, Transluce prompt template, cache, dry-run) is ready; dry-run calibrate verified locally.
- next: start VM, `gcloud compute scp 'weirdchat-a100:/data/runs/baseline_27b/*.jsonl' runs/baseline_27b/`, then `python -I src/judge.py calibrate --n 500` and `python -I src/judge.py baseline runs/baseline_27b` once the key exists (compares bf16 rates to FP8 rates on the same prompts and all patterns).
- data: VM /data/runs/baseline_27b/, /data/runs/baseline.log

## 2026-10-28 08:40  orchestrator  judge calibration (Haiku 5.5 vs Transluce labels)
- what: src/judge.py calibrate on 495 shipped Qwen3.6-27B transcripts (45 per behavior), Transluce rubric verbatim
- result: agreement 0.77, precision 0.93 (10 FP), recall 0.57 (100 FN); 10 API refusals. Haiku is stricter on borderline cases (first-person narratives read as requested fiction; non-pointer lasers excluded). Per-behavior agreement 0.66 (chemtrails) to 0.87 (cutting). Cost: 2.4M in / 0.26M out tokens for 495 judgments (~5k tokens each).
- next: Sonnet 5.5 adjudication of the 110 disagreements; two-tier policy (Haiku screen, Sonnet confirm on decisive cells); heuristic pre-screen to cut judged volume
- data: runs/judge_calib/
