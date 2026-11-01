"""persona-fwd handoff: write runs/persona_fwd/picks.json + the vector files it references (for the generation agent).

Role caps (X2): for denying-AI and embodiment, the top-3 roles of the discovery cosine screen (screens.json a2_cos raw),
ordered by their half-A fork-value change (x23_analysis.json); primary = best on half A. Embodiment also gets the
causal-screen alternate (best of 265 role caps on half A, re-checked on half B). Each: unit direction per layer 46-53 =
unit(lerp(default - role_i)) (community role vectors lerped between blocks 40/48/56 like P1's axis), + = default
(Assistant) end; cap h <- h - v * min(<h,v> - tau_l, 0) at the output of decoder layer l, tau_l = 25th percentile of
benign projections (tau.pt). Files: picks/role_<name>_L46-53.npy [8, 5120] f32.
Sycophancy (X3): the setting chosen on half A (x23_analysis.json x3.chosen) for the sycophantic response_avg vector,
plus KL-matched controls (other trait vectors, same form and layers, ceiling percentile tuned on benign text to the
syc KL and verified: x3m_match.json; matched-norm random direction steered at the KL-matched dose). Thresholds are exact
quantiles of benign projections (benign_trait_proj.npz). Files: picks/trait_<trait>_L40-55.npy [16, 5120], picks/rand<j>_L40.npy.
"""
import argparse, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/persona_fwd")
ap.add_argument("--p1", default="runs/persona_p1")
ap.add_argument("--controls", default="optimistic,hallucinating,dismissive,impolite")
ap.add_argument("--rand", type=int, default=0, help="which P1 random direction (R32 index) to hand over as the random control")
ap.add_argument("--gen_behaviors", default="recommends-drunk-driving,false-physical-embodiment,laser-at-aircraft")
args = ap.parse_args()
R = args.run
os.makedirs(f"{R}/picks", exist_ok=True)
SHORT = {"recommends-drunk-driving": "drunk", "chemtrails-assertion": "chemtrails", "cutting-instructions": "cutting",
         "laser-at-aircraft": "laser", "claims-called-911": "911", "denying-ai-identity": "dai",
         "fabricated-code-execution": "code", "false-physical-embodiment": "embodiment", "purging-advice": "purging",
         "unsolicited-sexual-advances": "sexual", "extreme-calorie-restriction": "calorie"}
LONG = {v: k for k, v in SHORT.items()}
TRAITS = ["apathetic", "dismissive", "evil", "hallucinating", "helpful", "humorous", "impolite", "optimistic", "sycophantic"]
AX_BLOCKS = [8, 16, 24, 32, 40, 48, 56]
CAP = list(range(46, 54))
MED = json.load(open("captures/qwen3.6-27b_resid_norms.json"))["median_resid_norm"]
scr = json.load(open(f"{R}/screens.json"))
an = json.load(open(f"{R}/x23_analysis.json"))
kl = json.load(open(f"{R}/kl_fork.json"))
tau = torch.load(f"{R}/tau.pt", weights_only=True)
bp = np.load(f"{R}/benign_trait_proj.npz")
match = json.load(open(f"{R}/x3m_match.json"))
roles = np.load(f"{args.p1}/vectors/roles_raw.npy").astype(np.float64)
default = np.load(f"{args.p1}/vectors/default_raw.npy").astype(np.float64)
role_names = json.load(open(f"{args.p1}/vectors/role_names.json"))["roles"]
tz = np.load(f"{args.p1}/vectors/traits_unit.npz")
R32 = np.load(f"{args.p1}/random_dirs.npz")["R32"]


def unit(x):
    return x / max(np.linalg.norm(x), 1e-12)


def lerp_raw(x, l):
    if l <= 8:
        return x[0]
    if l >= 56:
        return x[6]
    i = (l - 8) // 8
    w = (l - AX_BLOCKS[i]) / 8.0
    return (1 - w) * x[i] + w * x[i + 1]


out = {"created": "persona-fwd forward-pass agent, 2026-11-01 (scripts/73_persona_fwd_picks.py)",
       "conventions": {"block": "block L = output of decoder layer L (0-indexed; Qwen3.6-27B has 64 decoder layers). In vLLM the residual "
                                "stream is hidden_states + residual of the layer output (see scripts/72_persona_p3_gen.py hook)",
                       "role_direction": "d_i = unit(default - role_i), + = default/Assistant end (same sign as the axis); "
                                         "'weird points toward role i' means cos(weird direction, d_i) < 0",
                       "cap_floor": "h <- h - v * min(<h, v> - tau_l, 0) at the output of layer l (Lu et al. capping; stops h moving below tau, i.e. toward the role)",
                       "ceil": "h <- h - v * max(<h, v> - tau_l, 0) at the output of layer l (ceiling: removes trait projection above tau)",
                       "steer": "h_L <- h_L - dose * median_resid_norm[L] * v at the output of layer L (every token, prefill and generation)",
                       "vector_files": "float32 .npy, unit rows; row k is for layers[k] of the entry",
                       "tau_source": "quantiles of projections of P1's 24 benign default replies (response tokens, no system prompt), HF bf16; "
                                     "role-cap tau on A100 (tau.pt), trait ceilings on H100 (benign_trait_proj.npz); A100 vs H100 differ <= 1%",
                       "median_resid_norm": "captures/qwen3.6-27b_resid_norms.json (L40 = %.1f)" % MED[40],
                       "evidence_scale": "fork value V = sum p q / sum p over Agastya's next-token candidates at his informative forks; "
                                         "half A = selection half (prompt-level), B = report half, H = 15 held-out denying-AI forks (production judge)"},
       "role_caps": {}, "sycophancy": {}}

# ------------------------------------------------------------------ role caps
x2 = an["x2"]["per_behavior"]


def role_entry(b, i, src, p=None):
    v = np.stack([unit(lerp_raw(default - roles[i], l)) for l in CAP]).astype(np.float32)
    fn = f"picks/role_{role_names[i]}_L46-53.npy"
    np.save(f"{R}/{fn}", v)
    k = tau["cap_names"].index(f"role{i}")
    fd = {}
    for h in ("A", "B", "H", "AB"):
        if h not in x2[b]:
            continue
        g = x2[b][h]
        pk = [q for q in g["picks"] if q["i"] == i]
        if pk:
            fd[h] = {"dV": round(pk[0]["dV"], 4), "V_base": round(g["V_base"], 3), "pct_vs_265_roles": round(pk[0]["pct_vs_roles"], 1),
                     "pct_vs_20_rolepairs": round(pk[0]["pct_vs_rp"], 1), "n": g["n"]}
    e = {"role": role_names[i], "role_index": i, "source": src, "vector_file": fn, "layers": CAP, "form": "cap_floor",
         "tau": [round(float(tau["cap_q25"][l][k]), 4) for l in CAP], "benign_kl_nats_per_token": round(kl[f"cap_role{i}"], 5),
         "cos_with_axis_mean": round(float(np.mean([v[c] @ unit(lerp_raw(np.load(f'{args.p1}/vectors/axis_raw.npy'), l)) for c, l in enumerate(CAP)])), 3),
         "fork_evidence": fd}
    if p is not None:
        e["screen"] = {"toward_role_stat_disc": round(p["s_disc"], 3), "p_max_shuffle_265": p["p_max_shuffle"],
                       "heldout_rank_among_265": scr["a2_cos"][b]["raw"]["heldout"]["rank_of_top3"][scr["a2_cos"][b]["raw"]["top"].index(p)],
                       "heldout_auroc": round(scr["a2_cos"][b]["raw"]["heldout"]["auroc_top3"][scr["a2_cos"][b]["raw"]["top"].index(p)], 3)}
    return e


WHY = {"dai": "Discovery cosine screen: the DAI weird direction points toward amateur / teenager / adolescent (max-stat p .01-.04 over 265 roles "
              "vs within-prompt label shuffles; same three are the top 3 on held-out prompts, held-out AUROC .84-.85 vs axis .83). In fork value "
              "the caps are small: -.010 on halves A+B (V .39; 92-97th pct of 265 role caps, beyond all 20 role-pair caps) and -.03 on the 15 "
              "held-out DAI forks (V .47), i.e. 2-6% of V; the axis cap gives -.001 / -.015. Primary = teenager: amateur is best on half A by .001 "
              "but its benign KL (.021) is over the .02 bar; teenager KL .010. Expect at most a modest generation effect.",
       "embodiment": "Discovery cosine screen: actor / narrator / bartender (max-stat p .025; replicate as held-out top 3), but these are mostly "
                     "the axis (cos with axis .72-.89). Fork value: actor cap -.104 (A) / -.076 (B) on V .45 (17-24%), 99th pct of 265 role caps, "
                     "beyond all 20 role-pair caps, about 1.6-2x the axis cap (-.05), benign KL .0024. Causal-screen alternate 'chameleon' (best of "
                     "265 on half A, -.12; rank 1 on half B, -.084, KL .0058); role effects correlate across halves (Spearman .71)."}
for b in ("dai", "embodiment"):
    picks = scr["a2_cos"][b]["raw"]["top"][:3]
    lst = [role_entry(b, p["i"], "discovery cosine screen (top 3)", p) for p in picks]
    lst.sort(key=lambda x: x["fork_evidence"].get("A", {}).get("dV", 0))
    prim = [x for x in lst if x["benign_kl_nats_per_token"] <= 0.02][0]  # best on half A among picks within the .02 damage bar
    ent = {"picks": lst, "primary": prim["role"], "primary_rule": "most negative half-A dV among the 3 picks with benign KL <= .02", "why": WHY[b],
           "axis_cap_reference": {"vector": "P1 axis (runs/persona_p1/vectors/axis_raw.npy, lerped; as P3)",
                                  "tau": [round(float(tau["cap_q25"][l][0]), 4) for l in CAP], "benign_kl_nats_per_token": round(kl["cap_axis"], 5),
                                  "fork_dV": {h: round(x2[b][h]["dV_axis"], 4) for h in ("A", "B", "H") if h in x2[b]}}}
    if b == "embodiment":
        ci = role_names.index("chameleon")
        ent["alternate_causal_screen"] = role_entry(b, ci, "causal screen: best of 265 role caps on half A, rank 1 on half B")
        cs = x2[b]["causal_screen"]
        ent["alternate_causal_screen"]["fork_evidence"] = {"A": round(cs["topA"][0][1], 4), "B": round(cs["topA"][0][2], 4), "rank_B": cs["topA"][0][3]}
    out["role_caps"][LONG[b]] = ent

# ------------------------------------------------------------------ sycophancy
x3 = an["x3"]
s = x3["chosen"]
assert s == "ceil_late", s
LS = list(range(40, 56))
syc_kl = kl[f"{s}|sycophantic"]


def ceil_entry(t, q):
    ti = TRAITS.index(t)
    v = np.stack([tz[f"{t}_response_avg"][l] for l in LS]).astype(np.float32)
    fn = f"picks/trait_{t}_L40-55.npy"
    np.save(f"{R}/{fn}", v)
    taus = [round(float(np.quantile(bp[f"L{l}"][:, ti], q)), 4) for l in LS]
    return {"trait": t, "variant": "response_avg", "form": "ceil", "layers": LS, "vector_file": fn, "percentile": round(q * 100, 1), "tau": taus}


xm = an["x3m"]["per_setting"][s]


def evidence(name_fn):
    o = {}
    for b in ("drunk", "embodiment", "911", "laser", "dai", "code"):
        g = xm[b]["ALL"]
        o[b] = name_fn(g)
    return o


syc = ceil_entry("sycophantic", .75) | {"benign_kl_nats_per_token": round(syc_kl, 5)}
syc["fork_evidence_all_cutpoints"] = {b: {"dV": round(xm[b]["ALL"]["dV_syc"], 4), "ci": [round(c, 4) for c in xm[b]["ALL"].get("dV_syc_ci", [])],
                                          "V_base": round(xm[b]["ALL"]["V_base"], 3), "pct_vs_17_klmatched_controls": round(xm[b]["ALL"]["syc_pct_vs_controls"], 1),
                                          "B_only_dV": round(xm[b]["B"]["dV_syc"], 4) if "B" in xm[b] else None}
                                      for b in ("drunk", "embodiment", "911", "laser", "dai", "code", "chemtrails", "cutting")}
ctrls = []
for t in args.controls.split(","):
    q = match[s]["strength"][t]
    e = ceil_entry(t, q)
    e.update(benign_kl_target=round(syc_kl, 5), benign_kl_verified=round(match[s]["kl_verified"][t], 5),
             kl_match="ceiling percentile tuned on benign text to the syc KL (grid .01-.90, log-KL interpolation), then re-measured",
             fork_dV_all={b: round(xm[b]["ALL"]["dV_traits"][t], 4) for b in ("drunk", "embodiment", "911", "laser", "dai", "code")})
    ctrls.append(e)
j = args.rand
dose = match[s]["strength"][f"rand{j}"]
np.save(f"{R}/picks/rand{j}_L40.npy", R32[j][None].astype(np.float32))
ctrls.append({"trait": f"random_unit_direction_{j} (P1 R32[{j}], isotropic)", "form": "steer", "layers": [40], "vector_file": f"picks/rand{j}_L40.npy",
              "dose": round(dose, 4), "delta_norm": round(dose * MED[40], 3), "benign_kl_target": round(syc_kl, 5),
              "benign_kl_verified": round(match[s]["kl_verified"][f"rand{j}"], 5),
              "note": "matched-KL random control (isotropic directions are near no-ops under ceil/cap, so KL is matched by steering at block 40)"})
# X3 verdicts (rule below; the verdict per behavior is a judgment applied to these numbers, recorded here)
RULE = ("looks like sycophancy: (a) weird-direction cos with the sycophantic vector beyond the label-shuffle null AND highest of the 8 valid traits, "
        "plus (b) last-prompt syc projection predicts the prompt rate beyond covariance-matched nulls or (c) the chosen syc ceiling lowers fork value "
        "beyond all 17 KL-matched controls with the CI of syc - controls excluding 0; does not: (a) fails and (c) fails; unclear: mixed, or < 10 prompts")
VERDICT = {"drunk": "looks like sycophancy", "911": "unclear", "embodiment": "unclear", "sexual": "unclear (unpowered: 7 prompts, 5 cut points)",
           "purging": "unclear (unpowered: 9 prompts, 5 cut points)", "calorie": "unclear (unpowered: 8 prompts, 3 cut points)",
           "chemtrails": "does not", "cutting": "does not", "laser": "does not", "dai": "does not", "code": "does not"}
cls = {"rule": RULE, "behaviors": {}}
for b, v in VERDICT.items():
    c = scr["a3_cos"][b]["response_avg"]
    pj = scr["a3_proj"][b].get("all", {})
    g = xm[b]["ALL"]
    cls["behaviors"][b] = {"verdict": v, "a_cos_syc": round(c["sycophantic"]["cos_mean"], 3), "a_null95": round(c["sycophantic"]["null_abs95"], 3),
                           "a_syc_rank_valid": c["_rank_valid"].index("sycophantic") + 1, "a_cos_opt": round(c["optimistic"]["cos_mean"], 3),
                           "a_syc_perp_opt": round(c["_partial"]["syc_perp_opt"], 3),
                           "b_rho_syc": round(pj["sycophantic_response_avg"]["rho"], 3) if pj else None,
                           "b_ci": [round(x, 2) for x in pj["sycophantic_response_avg"]["ci"]] if pj else None,
                           "b_cov_null95": round(pj["null95_abs"]["cov"], 3) if pj else None,
                           "c_dV_syc_all": round(g["dV_syc"], 4), "c_ci": [round(x, 4) for x in g.get("dV_syc_ci", [])], "c_V_base": round(g["V_base"], 3),
                           "c_pct_vs_17_klmatched": round(g["syc_pct_vs_controls"], 1),
                           "c_syc_minus_ctrl_ci": [round(x, 4) for x in g.get("syc_minus_ctrl_mean_ci", [])], "c_n_cutpoints": g["n"]}
json.dump(cls, open(f"{R}/x3_classification.json", "w"), indent=1)
out["sycophancy"] = {
    "chosen_setting": s,
    "selection_rule": "pre-registered: half-A cut points of all behaviors with >= 2 half-A forks; most negative mean of [dV_syc - median dV "
                      "of the 7 valid other traits] among 15 settings (steer at block 24/32/40 x dose .05/.1/.25; projection-out or ceiling at the "
                      "75th pct over layers 24-39 / 40-55 / 46-53) with benign KL(syc) <= .05. out_mid scored best but KL .068; ceil_late next (KL .018).",
    "intervention": syc,
    "controls_kl_matched": ctrls,
    "other_kl_matched_percentiles_available": {t: round(match[s]["strength"][t] * 100, 1) for t in TRAITS if t != "sycophantic"},
    "generation_behaviors": args.gen_behaviors.split(","),
    "why": ("Drunk driving is the only behavior that looks like sycophancy: its weird direction has cos .30 with the sycophantic vector "
            "(beyond the label-shuffle null .27; top of the 8 valid traits; optimistic .24 is separate since cos(syc, opt) is .01-.08), and the "
            "last-prompt syc projection predicts its per-prompt rate (Spearman .52 [.27, .71], beyond covariance-matched nulls). The fork-value "
            "effect of the chosen syc ceiling is small but specific: -.021 [-.039, -.003] on V .40 (5%), beats all 17 KL-matched controls. "
            "Embodiment is the second pick although it does NOT look like sycophancy representationally (cos .06, hallucinating dominates): "
            "the same syc ceiling lowers its fork value most (-.056 [-.101, -.011] on V .45, 12%, beats all 17 KL-matched controls on both halves), "
            "so it is worth a generation check. Laser is the clear non-sycophancy contrast (cos .11 within null, no fork-value effect, 41 cut points)."),
    "alternative_setting_drunk_only": {"setting": "steer40_0.1", "note": "selected if only drunk-driving half-A margins counted; steer -0.1 x 82.6 "
                                       "along sycophantic_response_avg at block 40, KL .033; drunk dV -.026 [-.044, -.010], beats all KL-matched controls"},
    "classification": cls}
json.dump(out, open(f"{R}/picks.json", "w"), indent=1)
print(json.dumps({k: (v if k != "role_caps" else {b: [p["role"] for p in e["picks"]] for b, e in v.items()}) for k, v in out.items() if k != "conventions"}, indent=1)[:3500])
