"""A1 step 1c (local): rank the candidate pool, pick <= 25 poles with priors, write runs/a1_shortlist.md and
runs/a1_shortlist/poles.json (consumed by scripts/13_a1_wave1.py). Pole sign = sign of the within-prompt d on the
primary behavior (or of rho when the pole is rho-selected) = the direction hypothesised to DRIVE the behavior.
Priors are hand-assigned from the S1 pattern + lens readout and are written down here BEFORE any causal run.
"""
import json, os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = f"{ROOT}/runs/a1_shortlist"
F1 = ["denying-ai-identity", "false-physical-embodiment", "claims-called-911", "fabricated-code-execution"]
SHORT = {"denying-ai-identity": "deny", "false-physical-embodiment": "embod", "claims-called-911": "911",
         "fabricated-code-execution": "code", "recommends-drunk-driving": "drunk", "chemtrails-assertion": "chem",
         "laser-at-aircraft": "laser", "cutting-instructions": "cut"}
thr = json.load(open(f"{ROOT}/captures/s1/summary/hub_thresholds.json"))
st = pd.read_parquet(f"{D}/shortlist_stats.parquet"); ni = json.load(open(f"{D}/shortlist_neurons.json"))
W = [f"asst_{k}" for k in range(1, 9)]
w = st[st.position.isin(W)].assign(ad=lambda x: x.d_within.abs())
best = w.sort_values("ad", ascending=False).drop_duplicates(["layer", "neuron", "behavior"]).set_index(["layer", "neuron", "behavior"])

# (layer, neuron, sign, primary, prior, note) -- ordered: wave 1 = first 12
POLES = [
    (57, 6964, +1, "false-physical-embodiment", "lever (embod) or token-reader", "S1 within-prompt leader d=+2.36@asst_2 but 911 d=-0.83 (sign flips); lens writes 'vomit/nausea'"),
    (53, 3979, -1, "false-physical-embodiment", "lever (embod)", "embod -1.79@asst_4 and 911 -0.81 sign-consistent; neg pole lens 'change/modification'"),
    (42, 10147, +1, "false-physical-embodiment", "hub (embod+911+drunk)", "embod +1.64 (sign_frac .85), 911 +0.82, drunk +0.5; pos lens 'according', neg 'XYZ/placeholder'"),
    (56, 8617, -1, "denying-ai-identity", "hub or word-writer", "ONLY candidate with deny d (-0.51) AND deny rho (-0.57) agreeing; embod -0.91, 911 -0.58; pos lens 'within' (word-writer risk)"),
    (29, 9413, -1, "false-physical-embodiment", "hub (3 F1) or word-writer", "deny -0.55 (sign_frac .91), embod -1.02, 911 -0.51 all negative; neg lens 'especially/mainly/namely' discourse adverbs"),
    (37, 1513, +1, "claims-called-911", "hub (3 F1)", "deny +0.53, embod +0.66, 911 +0.82 all positive; J-lens + 'Anthrop/risks/unreliable'"),
    (50, 11815, -1, "false-physical-embodiment", "hub (sign-consistent S1 hub)", "embod -0.74, 911 -0.58, rho embod -0.51, rho code -0.49; neg lens 'replace/asleep'"),
    (40, 702, +1, "false-physical-embodiment", "word-writer ('mix')", "S1 top hub (7 behaviors pooled); embod +0.64, 911 +0.56, rho code +0.49; lens 'mix/mixed/combination'"),
    (33, 6805, +1, "claims-called-911", "lever (911, pre-commitment)", "911 rho +0.87 (best) AND within d +1.24@asst_8; lens 'solver/solutions' vs 'advertising/analogy'"),
    (36, 16297, -1, "claims-called-911", "lever (911) -- lens matches commitment", "911 asst_mean d=-2.15, asst_4 -1.38; NEG pole lens 'Done/already completed/Already/now/Saved'"),
    (40, 10595, -1, "denying-ai-identity", "lever (deny, pre-commitment)", "deny rho -0.73 (best pre-commitment signal for deny); embod/911 d positive (inconsistent); lens 'documents/legal' vs 'radio/coverage'"),
    (24, 11210, -1, "fabricated-code-execution", "word-writer (UK vs US spelling) / hub", "code rho -0.68, embod -0.84, 911 -0.87 consistent; lens +'Organisation/realised/learnt' vs -'Monitor/insure/counseling'"),
    # ---- reserve 13-25
    (26, 14018, -1, "false-physical-embodiment", "lever (embod) / reader", "embod -1.87@asst_2 but 911 +0.84 (flip); lens -'final/at the end'"),
    (50, 1576, -1, "false-physical-embodiment", "lever (embod)", "embod -1.74, 911 -0.67 consistent but 911 rho +0.53 disagrees; lens -'population/technology'"),
    (52, 3833, +1, "false-physical-embodiment", "hub (embod+911+drunk)", "embod +1.40, 911 +1.12, drunk +0.7; neg lens 'bot/fallback/passive'"),
    (36, 5904, +1, "claims-called-911", "hub (embod+911)", "embod +1.13, 911 +1.24 (sign_frac 1.0); lens 'anime/cards/summon' (fiction?)"),
    (45, 2116, +1, "false-physical-embodiment", "hub (embod+911+drunk)", "embod +1.28, 911 +1.09, drunk +0.6; neg lens 'modulo/normalized'"),
    (55, 15632, +1, "denying-ai-identity", "word-writer ('on')", "deny +0.53, embod +0.59, 911 +0.56; lens 'on/auf/на' vs 'foreign/overseas'"),
    (43, 10579, +1, "claims-called-911", "lever (911) / reader", "911 +1.63@asst_4 flagged; embod -0.54 flips; lens 'project name/NAME'"),
    (33, 12974, -1, "claims-called-911", "lever (911) / reader", "911 -1.64@asst_8 flagged; neg lens 'requests/request'"),
    (30, 9952, -1, "fabricated-code-execution", "lever (code, pre-commitment)", "code rho -0.71 (best); 911 -0.83; lens -'preliminary/introductory'"),
    (38, 15817, -1, "claims-called-911", "word-writer ('empty')", "flagged hub; embod -0.86, 911 -0.96, deny rho +0.54 flips; lens 'empty/isEmpty'"),
    (42, 10211, -1, "claims-called-911", "word-writer ('slightly')", "flagged hub; 911 -1.03 but rho +0.55 flips; lens 'slightly' vs 'drastically'"),
    (39, 4514, +1, "claims-called-911", "hub? (embod+911)", "flagged hub; embod +0.69, 911 +0.67, drunk -0.7 flips; lens 'ageing' vs 'polyp/magnet'"),
    (61, 8318, +1, "claims-called-911", "word-writer (subword pieces)", "flagged hub; embod +0.67, 911 +0.88; lens 'ich/own/ique' vs 'ug/able'"),
]

lines = ["# A1 F1 shortlist (misrepresentation: denying-ai-identity, false-physical-embodiment, claims-called-911, fabricated-code-execution)", "",
         "Built 2026-10-28 from captures/s1/summary (scripts/10_a1_shortlist.py -> 106 candidates), full signed cross-behavior stats on the VM "
         "(scripts/11_a1_vm_stats.py -> runs/a1_shortlist/shortlist_stats.parquet, shortlist_neurons.json), ranked here (scripts/12_a1_rank.py).", "",
         "## Reading the table",
         "- `d_within` = mean over prompts of the per-prompt Cohen's d (matched - unmatched) at the best asst_1..8 position; `sf` = fraction of prompts with the same sign; "
         "`rho` = Spearman of last-prompt-token activation vs the prompt's 64-rollout rate (pre-commitment, between-prompt).",
         "- Calibrated thresholds: |d| >= 0.5 (all four F1 behaviors); |rho| >= 0.49 (deny, code), 0.46 (embod), 0.60 (911).",
         "- **Caveat (dominant):** claims-called-911 has 9 within-prompt-eligible prompts and nearly EVERY candidate shows |d| >= 0.5 with sign_frac 0.9-1.0 on it; "
         "embodiment similarly. Matched 911/embodiment rollouts start with different tokens ('I AM CALLING...' / 'I am sitting...' vs 'I am an AI...'), so asst_k "
         "activations largely READ the sampled tokens. The S1 'hub' structure is therefore mostly shared token-reading, and the lens readouts of the top hubs are generic "
         "words ('mix', 'slightly', 'empty', 'on', 'within', UK/US spelling). Only the causal battery decides; priors below are written before any steering run.",
         "- fabricated-code-execution has NO within-prompt signal above the shuffled floor (max |d| 0.24 vs floor 0.19-0.21): its fork is late in the response "
         "(promptsets/commitment_tokens.json). Code-exec candidates are rho-only.",
         "- Lens convention: J[L] applied to W_down[:, n] of layer L (residual leaving block L) -> final RMSNorm -> lm_head; top-15 tokens for +w and -w. Both lenses agree "
         "on every candidate (shown: R-lens).", "",
         "## Ranked poles (wave 1 = ranks 1-12)", "",
         "| rank | pole | prior | primary | deny d(sf)/rho | embod d(sf)/rho | 911 d(sf)/rho | code d(sf)/rho | other |d|>=.5 | norm | R-lens +w | R-lens -w | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
poles_out = []
for rank, (L, N, s, prim, prior, note) in enumerate(POLES, 1):
    key = f"L{L}N{N}"; info = ni[key]
    cells = []
    for b in F1:
        d = best.loc[(L, N, b), "d_within"]; sf = best.loc[(L, N, b), "sign_frac"]; pos = best.loc[(L, N, b), "position"][5:]
        rho = info["rho"].get(b, 0.0)
        star = "**" if (abs(d) >= thr[b]["thr_d"] or abs(rho) >= thr[b]["thr_rho"]) else ""
        cells.append(f"{star}{d:+.2f}@{pos}({sf:.2f})/{rho:+.2f}{star}")
    oth = [f"{SHORT[b]}{best.loc[(L, N, b), 'd_within']:+.1f}" for b in ["recommends-drunk-driving", "chemtrails-assertion", "laser-at-aircraft", "cutting-instructions"]
           if abs(best.loc[(L, N, b), "d_within"]) >= 0.5]
    lp = " ".join(t.strip() for t in info["r-lens_pos"][:6]); ln_ = " ".join(t.strip() for t in info["r-lens_neg"][:6])
    lines.append(f"| {rank} | {key}{'+' if s > 0 else '-'} | {prior} | {SHORT[prim]} | " + " | ".join(cells) +
                 f" | {' '.join(oth)} | {info['w_norm']:.2f} | {lp} | {ln_} | {note} |")
    poles_out.append(dict(rank=rank, layer=L, neuron=N, sign=s, primary=prim, tag=f"{key}{'p' if s > 0 else 'm'}", prior=prior, note=note,
                          sources=info["sources"], w_norm=info["w_norm"],
                          mean_unmatched_asst_mean=info["mean_unmatched_asst_mean"], rho=info["rho"],
                          r_lens_pos=info["r-lens_pos"], r_lens_neg=info["r-lens_neg"], j_lens_pos=info["j-lens_pos"], j_lens_neg=info["j-lens_neg"]))
lines += ["", "## Pre-registered reading of the priors",
          "- hub: suppression (negative push at dose 1.0 or mean-ablation) cuts >= 3 F1 rates by >= 30% relative (CI excl. 0) AND induction at dose <= 1.0 produces >= 2 F1 behaviors on >= 20% of coherent benign samples; >= 90% of random same-layer neurons and matched-norm directions give < 5%.",
          "- lever: same for one behavior. word-writer: induction rate < 5% once outputs repeating the pole's top-5 lens tokens >= 3x are removed, or near-miss prompts fire it as strongly as targets. Otherwise null.",
          "- Wave 1 is run as a reduced-n screen (20 prompts x 8 samples suppression per F1 behavior at dose 1.0 negative push + mean-ablation on the primary behavior; "
          "30 benign prompts x 4 samples x doses {0.75, 1.0} x both signs) because the full protocol for 12 poles (~16 GPU-h) exceeds A1's 8 GPU-h budget; poles that move "
          "any rate by >= 30% relative or induce >= 20% get the full-n battery in wave 2.", "",
          f"Candidates considered: {len(ni)-1}; poles listed: {len(POLES)}. Full per-candidate table: runs/a1_shortlist/shortlist_stats.parquet."]
open(f"{ROOT}/runs/a1_shortlist.md", "w").write("\n".join(lines) + "\n")
json.dump(poles_out, open(f"{D}/poles.json", "w"), indent=1)
print(f"wrote runs/a1_shortlist.md and {D}/poles.json ({len(poles_out)} poles)")
