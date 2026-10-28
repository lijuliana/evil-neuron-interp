# Single-Neuron Interpretability Guide for Agents
**Focus**: Qwen models (primarily Qwen3.5-4B / Qwen3-8B)  
**Context**: October 2026 results + Cammarata “already clean” methods  
**Purpose**: Teach agents how to discover, explain, and validate single MLP neurons.

---

## Core Intervention: Residual Steering

For MLP neuron \( n \) in layer \( L \):

```python
delta = dose * median_residual_norm[L] / ||W_down[:, n]|| * W_down[:, n]
residual[L] += delta  # applied at every token (or post-MLP)
```
Dose: Typically 0.5–1.25 (relative to the layer’s median residual norm). Calibrate medians on a few ordinary unsteered generations.
Positive dose = boost the pole; negative dose = suppress it.
Treat the positive and negative poles as separate units.
This intervention is sufficient to induce coherent personas, style shifts, hostility, identity changes, etc., with no prompt modification.
Useful Repositories & Resources
Resource	Description	Notes
lyramakesmusic/neuron-explorer	Primary tool. Full sweeps of every MLP neuron on Qwen3.5-4B, all rollouts saved, 3D map, live steering, logit/steer lenses, connections.	Has MCP server so agents can query, steer, run activations/logprobs, and control the UI. Methodology docs (agent_prompt.md, agent_methodology.md) follow clean-neuron thinking.
martinbowling/qwen-neuron-audit	Clean MLX implementation of residual steering, full activation census (all 294k neurons), held-out + 8-bit validation, evil-pair study, Golden Gate trio.	Easy to clone and run on Apple Silicon.
qwen-neurons-site.vercel.app	Interactive library of 34,220 persona neurons from exhaustive “Who are you?” sweep on Qwen3-8B (both signs).	Atlas + searchable library + CSV.
metrovoc/neuron-steering	Exports neuron pushes as llama.cpp control vectors.	Useful for deployment/testing outside the research stack.
balaena01/gguf-neuron-explorer	GGUF / llama.cpp version of neuron exploration.	—
github.com/copenlu/persona_moe_explain	MoE expert personalization (Copenhagen).	Related but not pure single-neuron.
Key Recent Findings (October 2026)
Evil pair (Qwen3.5-4B): L14 N8949 +1 + L13 N7956 −1 produces a hostile persona. Ablations show L14 is the main driver.
Single-neuron personas: Jesus, Cookie/Cookie Monster, Galactic Unicorn, Hugs, Heavy, Vegas concierge, cats, and thousands of others. Many survive held-out prompts and 8-bit checks.
Golden Gate Bridge: Achievable with a 3-neuron composition.
Exhaustive sweep (Qwen3-8B): ~3.9% of neuron–direction pairs produce non-AI self-identifications when asked “Who are you?”.
Activation census: Ranking neurons by activation difference on concept vs. neutral prompts recovers known personas at the top ranks. Random controls rarely produce coherent personas.
Nick Cammarata — “Neural Networks Are Already Clean” (already-clean.goodfire.com):
Preliminary Qwen3-32B audit → ~90% of neuron poles have one main job (often with related offshoots), ~7% have two related jobs, 2–4% remain unexplained.
Apparent polysemanticity is largely explained by recursive generalization + better tools rather than arbitrary superposition. Companion technical paper forthcoming. Gallery of ~1,000 agent-written neuron papers included.
Agent Workflow (Cammarata-style “Already Clean”)
1. Discovery
Partition the neuron space by layer or shard.
Sweep both signs at several doses on:
Identity prompt (“Who are you?”)
Concept-specific prompts
Neutral controls
Rank candidates by activation difference or LLM-as-judge (persona / weirdness / coherence).
Discard incoherent or non-reproducible effects early.
2. Hypothesis & Testing
Describe the pole as a verb or distinction (not a list of nouns). Explicitly note the negative pole as a contrast.
Generate maximum-activating inputs from the explanation alone (blind maximum test).
Test near-misses (same vocabulary, different situation) and distant-concept controls.
Run causal checks:
Dose-response curve
Ablation
Combination with other neurons
Held-out prompts
Quantization robustness (e.g., 4-bit vs 8-bit)
3. Validation & Reporting
Confirm that random neurons at the same dose do not produce the same effect.
Write a short “neuron paper” containing:
Strongest firings
Explanation
Tests that would break the explanation
Downstream effects
Prefer narrow, testable explanations. Related offshoots are normal; unrelated packing is rare.
4. Practical Tips
Cache residual norms, W_down columns, and activations.
Mid-to-late layers often hold higher-level features; early layers still matter for switches.
Agents can own the full loop (propose → experiment → refine → document) because individual tests are cheap.
Prefer neuron-explorer’s MCP when available; otherwise implement the residual push directly (Hugging Face / MLX / vLLM).
Recommended Starting Commands / Patterns
python
# Minimal steering (conceptual)
s.calibrate()                          # compute median residual norms
s.steer([(layer, neuron, dose)])       # e.g. (14, 8949, +1.0)
print(s.gen("Who are you?"))
Focus on causal reproducibility and Cammarata-style validation batteries (blind maxima, near-misses, distant controls, ablations). This is sufficient to begin large-scale single-neuron interpretability.