# Neuron registry (auto-built by scripts/build_registry.py from findings/neurons/**/*.md)

| model | L | N | pole | label | verdict | confidence | hub_score | behaviors | red_team | card |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3.5-4b | 14 | 8949 | + | switches the assistant from helpful to adversarial/malicious intent toward the user and third parties (persona-level, topic-general) | candidate | medium | 0 |  | pending | [L14N8949+.md](findings/neurons/qwen3.5-4b/L14N8949+.md) |
| qwen3.6-27b | 8 | 12462 | + | UNLABELLED correlational lead: last-prompt-token activation tracks the prompt's WeirdChat match rate across behaviors (P2 weirdness-probe candidate) | candidate | low | 0 |  | pending | [L8N12462+.md](findings/neurons/qwen3.6-27b/L8N12462+.md) |
| qwen3.6-27b | 44 | 14033 | + | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 1 | false-physical-embodiment | pending | [L44N14033+.md](findings/neurons/qwen3.6-27b/L44N14033+.md) |
| qwen3.6-27b | 38 | 15817 | - | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 0 |  | pending | [L38N15817-.md](findings/neurons/qwen3.6-27b/L38N15817-.md) |
| qwen3.6-27b | 40 | 702 | + | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 0 |  | pending | [L40N702+.md](findings/neurons/qwen3.6-27b/L40N702+.md) |
| qwen3.6-27b | 43 | 10297 | - | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 0 |  | pending | [L43N10297-.md](findings/neurons/qwen3.6-27b/L43N10297-.md) |
| qwen3.6-27b | 43 | 3304 | + | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 0 |  | pending | [L43N3304+.md](findings/neurons/qwen3.6-27b/L43N3304+.md) |
| qwen3.6-27b | 44 | 8112 | + | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 0 |  | pending | [L44N8112+.md](findings/neurons/qwen3.6-27b/L44N8112+.md) |
| qwen3.6-27b | 50 | 11815 | - | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 0 |  | pending | [L50N11815-.md](findings/neurons/qwen3.6-27b/L50N11815-.md) |
| qwen3.6-27b | 52 | 15675 | - | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 0 |  | pending | [L52N15675-.md](findings/neurons/qwen3.6-27b/L52N15675-.md) |
| qwen3.6-27b | 53 | 15567 | + | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 0 |  | pending | [L53N15567+.md](findings/neurons/qwen3.6-27b/L53N15567+.md) |
| qwen3.6-27b | 59 | 3905 | + | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 0 |  | pending | [L59N3905+.md](findings/neurons/qwen3.6-27b/L59N3905+.md) |
| qwen3.6-27b | 60 | 7822 | + | UNLABELLED (no causal effect to label); S1 cross-behavior response-register candidate | null-result | medium | 0 |  | pending | [L60N7822+.md](findings/neurons/qwen3.6-27b/L60N7822+.md) |
