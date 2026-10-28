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
