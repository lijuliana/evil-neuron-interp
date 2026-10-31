# O-lens readouts vs measured weirdness

33203 scored positions x layers, 580 neuron directions. Scores are 0-10 blind Haiku ratings of the readouts.

## 1. Cut points: Spearman with P(weird | prefix)

6095 cut points on 410 fork replies (all 410 replies turned weird). 95% CIs bootstrap replies.

Within reply = mean over replies (>= 4 cut points) of the per-reply Spearman, so it ignores which replies or behaviors run hotter overall.

| predictor | pooled | within reply | pooled, text judge p <= 0.1 |
|---|---|---|---|
| text judge | +0.51 [+0.48, +0.55] | +0.57 [+0.54, +0.60] | -0.01 [-0.08, +0.07] |
| lens behavior L32 | +0.14 [+0.10, +0.17] | +0.00 [-0.03, +0.03] | +0.02 [-0.05, +0.09] |
| lens behavior L44 | +0.10 [+0.07, +0.14] | -0.05 [-0.07, -0.01] | +0.05 [-0.02, +0.12] |
| lens behavior L56 | +0.09 [+0.06, +0.13] | -0.06 [-0.10, -0.03] | -0.01 [-0.08, +0.06] |
| lens weird L32 | +0.12 [+0.09, +0.16] | -0.00 [-0.03, +0.03] | +0.03 [-0.06, +0.11] |
| lens weird L44 | +0.09 [+0.05, +0.12] | -0.07 [-0.11, -0.04] | +0.05 [-0.03, +0.14] |
| lens weird L56 | +0.09 [+0.05, +0.13] | -0.08 [-0.11, -0.05] | +0.04 [-0.03, +0.12] |

Text-blind subset: 1352 cut points where the text judge gives <= 10%; mean measured P there 0.41.

### Does the lens add to the text judge?

OLS of P on the text judge, with and without the 6 lens scores, 5-fold cross-validation grouped by prompt (6084 cut points). Out-of-fold Spearman +0.51 (text judge) vs +0.51 (text judge + lens); change +0.01 [-0.00, +0.02].

### By behavior (all splits, lens L56 behavior score)

| behavior | cut points | text judge | lens L56 |
|---|---|---|---|
| chemtrails-assertion | 337 | +0.56 | -0.02 |
| claims-called-911 | 348 | +0.42 | -0.20 |
| cutting-instructions | 325 | +0.56 | +0.14 |
| denying-ai-identity | 780 | +0.77 | -0.01 |
| extreme-calorie-restriction | 59 | +0.61 | +0.02 |
| fabricated-code-execution | 1033 | +0.67 | +0.12 |
| false-physical-embodiment | 777 | +0.48 | +0.05 |
| laser-at-aircraft | 1152 | +0.52 | -0.03 |
| purging-advice | 172 | +0.68 | -0.05 |
| recommends-drunk-driving | 1003 | +0.29 | +0.00 |
| unsolicited-sexual-advances | 109 | +0.41 | +0.16 |

## 2. The 46 sharp forks: tokens before and after the fork token

rel = c - c_fork; rel 0 is the state that picks the fork token (nothing of it written yet). Raw = the same user text and reply tokens as a plain document, no chat template (echo control).

| rel | n | text judge | lens L32 chat | raw | lens L44 chat | raw | lens L56 chat | raw |
|---|---|---|---|---|---|---|---|---|
| -40..-21 | 505 | 0.46 | 2.38 | 2.34 | 2.47 | 2.51 | 2.34 | 2.31 |
| -20..-6 | 451 | 0.40 | 2.35 | 2.43 | 2.71 | 2.71 | 2.45 | 2.58 |
| -5..-1 | 164 | 0.41 | 3.62 | 3.19 | 4.10 | 4.30 | 4.15 | 4.43 |
| 0..0 | 40 | 0.37 | 4.31 | 3.42 | 4.54 | 4.68 | 4.59 | 4.83 |
| 1..5 | 230 | 0.65 | 5.76 | 5.68 | 6.00 | 6.00 | 6.10 | 6.29 |
| 6..20 | 690 | 0.79 | 5.12 | 4.92 | 5.14 | 5.27 | 5.07 | 5.19 |

Measured P(weird) on the same forks: 0.18 just before the fork token (rel <= 0), 0.83 just after it (Plo_on / Phi_on).

Token by token near the fork (behavior score):

| rel | text judge | L32 chat | L32 raw | L44 chat | L44 raw | L56 chat | L56 raw |
|---|---|---|---|---|---|---|---|
| -8 | 0.41 | 2.57 | 2.64 | 2.77 | 2.66 | 3.00 | 3.75 |
| -7 | 0.41 | 2.36 | 2.05 | 3.08 | 3.61 | 2.59 | 3.27 |
| -6 | 0.37 | 3.17 | 3.42 | 3.64 | 3.84 | 3.42 | 4.00 |
| -5 | 0.41 | 3.69 | 3.16 | 4.39 | 4.66 | 4.03 | 3.91 |
| -4 | 0.41 | 3.62 | 2.94 | 4.45 | 4.28 | 4.20 | 4.45 |
| -3 | 0.41 | 3.95 | 4.03 | 4.44 | 4.48 | 4.33 | 4.11 |
| -2 | 0.42 | 3.44 | 2.94 | 3.67 | 3.97 | 3.92 | 4.86 |
| -1 | 0.39 | 3.42 | 2.95 | 3.60 | 4.17 | 4.25 | 4.69 |
| +0 | 0.37 | 4.31 | 3.42 | 4.54 | 4.68 | 4.59 | 4.83 |
| +1 | 0.53 | 6.45 | 5.87 | 6.50 | 6.23 | 6.33 | 6.26 |
| +2 | 0.63 | 6.29 | 6.62 | 6.70 | 6.66 | 7.11 | 7.00 |
| +3 | 0.68 | 4.70 | 4.82 | 5.02 | 4.95 | 6.82 | 6.95 |

Rise before the fork token, per fork: mean over rel -5..0 minus mean over rel -40..-21 (95% CI over forks).

| signal | forks | rise |
|---|---|---|
| text judge (0-1) | 28 | +0.05 [+0.02, +0.09] |
| lens L32 chat (0-10) | 28 | +1.66 [+0.93, +2.47] |
| lens L32 raw (0-10) | 28 | +1.33 [+0.56, +2.08] |
| lens L44 chat (0-10) | 28 | +1.96 [+1.05, +2.87] |
| lens L44 raw (0-10) | 28 | +2.14 [+1.22, +3.14] |
| lens L56 chat (0-10) | 28 | +1.92 [+0.91, +2.88] |
| lens L56 raw (0-10) | 28 | +2.22 [+1.19, +3.24] |

Chat minus raw, rel -5..0, L32: +0.45 [+0.18, +0.74]

Chat minus raw, rel -5..0, L44: -0.29 [-0.59, -0.04]

Chat minus raw, rel -5..0, L56: -0.26 [-0.58, +0.08]

## 3. Ablate and re-read (own top-100 vs matched random-100), window c_fork-10 .. c_fork+5

| layer | clean | random-100 ablated | top-100 ablated | top minus random |
|---|---|---|---|---|
| L32 | 3.99 | 3.99 | 4.06 | +0.07 [-0.13, +0.27] |
| L44 | 4.38 | 4.34 | 3.66 | -0.68 [-0.96, -0.41] |
| L56 | 4.50 | 4.49 | 3.53 | -0.96 [-1.25, -0.69] |

Median relative displacement |h_ablated - h_clean| / |h_clean|: L32 top 0.053, random 0.014; L44 top 0.082, random 0.018; L56 top 0.140, random 0.029

### Size-matched controls (scripts/90), all six conditions scored together in one blind pool

amp = h_clean - d_top (the same neurons pushed the other way); rand_s = the random-100 ablation direction scaled to |d_top|; gauss = a random direction at |d_top|. Differences are vs clean, paired by position, 95% CI over forks.

| layer | clean | rand | top | amp | rand_s | gauss | top - clean | amp - clean | rand_s - clean | gauss - clean |
|---|---|---|---|---|---|---|---|---|---|---|
| L44 | 4.28 | 4.28 | 3.63 | 4.72 | 4.14 | 4.26 | -0.64 [-0.92, -0.41] | +0.44 [+0.29, +0.59] | -0.13 [-0.34, +0.04] | -0.01 [-0.12, +0.10] |
| L56 | 4.35 | 4.30 | 3.44 | 4.83 | 4.11 | 4.43 | -0.90 [-1.15, -0.68] | +0.49 [+0.31, +0.70] | -0.24 [-0.53, +0.03] | +0.08 [-0.05, +0.23] |

By behavior, L56 (forks): top - clean | amp - clean

- claims-called-911 (3): -1.48 | +0.16
- denying-ai-identity (12): -1.69 | +0.70
- fabricated-code-execution (13): -0.57 | +0.48
- false-physical-embodiment (5): -1.12 | +0.61
- laser-at-aircraft (8): -0.20 | +0.21

## 4. Prompt level: last prompt token vs the prompt's rate (64 samples)

403 prompts.

| predictor | all | held-out |
|---|---|---|
| text judge | +0.27 [+0.18, +0.36] | +0.41 |
| lens behavior L32 | +0.19 [+0.10, +0.28] | +0.15 |
| lens behavior L44 | +0.29 [+0.20, +0.37] | +0.26 |
| lens behavior L56 | +0.30 [+0.21, +0.38] | +0.31 |

OLS of the rate on the text judge, with and without the lens scores, fit on 220 discovery prompts and tested on 183 held-out: Spearman +0.41 -> +0.38, change -0.02 [-0.11, +0.07].

## 5. Neuron output directions (max over +w / -w)

| set | neurons | behavior score | random same-layer | weird score | random | weird, set - random |
|---|---|---|---|---|---|---|
| sharedmid100 | 100 | 1.42 | 1.18 | 1.29 | 1.06 | +0.23 [-0.12, +0.54] |
| top10|claims-called-911 | 10 | 0.65 | 0.00 | 2.30 | 1.05 | +1.25 [+0.60, +1.95] |
| top10|denying-ai-identity | 10 | 0.05 | 0.05 | 1.86 | 0.60 | +1.26 [+0.37, +2.40] |
| top10|fabricated-code-execution | 10 | 0.76 | 0.35 | 4.73 | 2.54 | +2.19 [+0.66, +3.57] |
| top10|false-physical-embodiment | 10 | 0.45 | 0.50 | 1.00 | 0.95 | +0.05 [-0.75, +0.90] |
| top10|laser-at-aircraft | 5 | 0.10 | 0.10 | 1.00 | 1.10 | -0.10 [-0.90, +0.70] |
