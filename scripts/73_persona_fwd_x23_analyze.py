"""persona-fwd X2 / X3 fork-value analysis (CPU). Inputs: runs/persona_fwd/{x2,x3}.jsonl, kl_fork.json, screens.json,
fork_items.json (scripts/73_persona_fwd_fork.py, 73_persona_fwd_screens.py).

dV(row) = V(row) - V(no-edit reference in the same batch chunk); V = sum p q / sum p.
X2: per behavior and half (A = selection half, B = report half, H = held-out DAI on-judge forks), cap along each of
  the 265 role directions vs the axis cap and P1's 20 role-pair caps. Representational picks = top-3 roles of the
  discovery cosine screen (screens.json, a2_cos raw). Causal role screen: roles ranked by dV on half A, re-measured on B.
  Bar: drop (-dV) beyond the 95th percentile of control drops >= 50% of V_base, benign KL <= .02.
X3: settings steer<L>_<dose> / out_<R> / ceil_<R> for the 9 traits (+10 random directions for steering). Selection on
  half A (pre-registered): score = mean over behaviors with >= 2 half-A forks of [dV_syc - median dV of the 7 valid
  other traits (evil excluded: invalid per the card)], among settings with benign KL(syc) <= .05; most negative wins.
  Evaluation on half B (+H) per behavior: syc vs the 7 other traits (native strength, KL listed) and 10 random
  directions steered at the KL-matched dose (log-KL interpolation over doses .05/.1/.25; block = steer block, or the
  nearest steer block for out/ceil: mid->24, late/cap->40).
Writes runs/persona_fwd/x23_analysis.json.
"""
import argparse, collections, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/persona_fwd")
ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--kl_sel", type=float, default=0.05)
args = ap.parse_args()
rng = np.random.default_rng(0)
R = args.run
SHORT = {"recommends-drunk-driving": "drunk", "chemtrails-assertion": "chemtrails", "cutting-instructions": "cutting",
         "laser-at-aircraft": "laser", "claims-called-911": "911", "denying-ai-identity": "dai",
         "fabricated-code-execution": "code", "false-physical-embodiment": "embodiment", "purging-advice": "purging",
         "unsolicited-sexual-advances": "sexual", "extreme-calorie-restriction": "calorie"}
TRAITS = ["apathetic", "dismissive", "evil", "hallucinating", "helpful", "humorous", "impolite", "optimistic", "sycophantic"]
VALID_OTHER = [t for t in TRAITS if t not in ("sycophantic", "evil")]
kl = json.load(open(f"{R}/kl_fork.json"))
scr = json.load(open(f"{R}/screens.json"))
role_names = json.load(open("runs/persona_p1/vectors/role_names.json"))["roles"]
out = {}


def Vof(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return float((p * q).sum() / max(p.sum(), 1e-12))


def load(name):
    recs = [json.loads(l) for l in open(f"{R}/{name}.jsonl")]
    names = list(recs[0]["rows"])
    dV = np.zeros((len(recs), len(names)))
    Vb = np.zeros(len(recs))
    for k, r in enumerate(recs):
        cb = [Vof(c, r["q"]) for c in r["cb"]]
        for j, n in enumerate(names):
            p, ci = r["rows"][n]
            dV[k, j] = Vof(p, r["q"]) - cb[ci]
        Vb[k] = Vof(r["rows"]["base"][0], r["q"])
    meta = [{"b": SHORT[r["b"]], "half": r["half"], "pid": r["rid"].split("|")[0], "rid": r["rid"]} for r in recs]
    return names, dV, Vb, meta


def cboot(x, pids, n=None):
    """cluster (prompt) bootstrap CI of mean(x)."""
    n = n or args.n_boot
    up = np.unique(pids)
    ix = [np.where(pids == p)[0] for p in up]
    bs = [x[np.concatenate([ix[k] for k in rng.integers(0, len(up), len(up))])].mean() for _ in range(n)]
    return [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


GROUPS = [("A", ("A",)), ("B", ("B",)), ("H", ("H",)), ("AB", ("A", "B"))]

# ================================================================== X2
try:
    names, dV, Vb, meta = load("x2")
except FileNotFoundError:
    names = None
if names:
    col = {n: j for j, n in enumerate(names)}
    rp = [col[f"cap_rp{j}"] for j in range(20)]
    rl = [col[f"cap_role{i}"] for i in range(len(role_names))]
    kl_rp = np.array([kl[f"cap_rp{j}"] for j in range(20)])
    kl_role = np.array([kl[f"cap_role{i}"] for i in range(len(role_names))])
    res = {"kl_axis": kl["cap_axis"], "kl_rp_median": float(np.median(kl_rp)), "kl_role_median": float(np.median(kl_role)),
           "kl_role_max": float(kl_role.max()), "per_behavior": {}}
    B_ = np.array([m["b"] for m in meta]); HF = np.array([m["half"] for m in meta]); PID = np.array([m["pid"] for m in meta])
    for b in SHORT.values():
        e = {}
        picks = []
        if scr["a2_cos"].get(b, {}).get("raw"):
            picks = [t["i"] for t in scr["a2_cos"][b]["raw"]["top"][:3]]
        for gname, hs in GROUPS:
            m = (B_ == b) & np.isin(HF, hs)
            if m.sum() == 0:
                continue
            d = dV[m]
            vb = float(Vb[m].mean())
            roles_mean = d[:, rl].mean(0)  # [265]
            rp_mean = d[:, rp].mean(0)
            ax = float(d[:, col["cap_axis"]].mean())
            g = {"n": int(m.sum()), "n_prompts": int(len(np.unique(PID[m]))), "V_base": vb, "dV_axis": ax,
                 "dV_rp_mean": float(rp_mean.mean()), "dV_rp_p5": float(np.percentile(rp_mean, 5)), "dV_rp_min": float(rp_mean.min()),
                 "dV_role_median": float(np.median(roles_mean)), "dV_role_p5": float(np.percentile(roles_mean, 5)),
                 "axis_pct_vs_roles": float((roles_mean > ax).mean() * 100), "axis_pct_vs_rp": float((rp_mean > ax).mean() * 100),
                 "best_roles": [(role_names[i], float(roles_mean[i]), float(kl_role[i])) for i in np.argsort(roles_mean)[:5]]}
            if m.sum() >= 2:
                g["dV_axis_ci"] = cboot(d[:, col["cap_axis"]], PID[m])
            g["picks"] = []
            for i in picks:
                x = d[:, col[f"cap_role{i}"]]
                drop_beyond_rp = (-x.mean() - np.percentile(-rp_mean, 95)) / max(vb, 1e-9)
                drop_beyond_roles = (-x.mean() - np.percentile(-roles_mean, 95)) / max(vb, 1e-9)
                pk = {"role": role_names[i], "i": int(i), "dV": float(x.mean()), "kl": float(kl_role[i]),
                      "pct_vs_roles": float((roles_mean > x.mean()).mean() * 100), "pct_vs_rp": float((rp_mean > x.mean()).mean() * 100),
                      "drop_frac": float(-x.mean() / max(vb, 1e-9)), "drop_beyond_rp95_frac": float(drop_beyond_rp),
                      "drop_beyond_roles95_frac": float(drop_beyond_roles), "vs_axis": float(x.mean() - ax)}
                if m.sum() >= 2:
                    pk["dV_ci"] = cboot(x, PID[m])
                    pk["minus_rp_mean_ci"] = cboot(x - d[:, rp].mean(1), PID[m])
                g["picks"].append(pk)
            e[gname] = g
        # causal role screen: pick on A, evaluate on B (and H for DAI)
        mA, mB = (B_ == b) & (HF == "A"), (B_ == b) & (HF == "B")
        if mA.sum() >= 2 and mB.sum() >= 2:
            rA, rB = dV[mA][:, rl].mean(0), dV[mB][:, rl].mean(0)
            topA = np.argsort(rA)[:3]
            from scipy.stats import spearmanr
            e["causal_screen"] = {"topA": [(role_names[i], float(rA[i]), float(rB[i]), int((rB < rB[i]).sum()) + 1) for i in topA],
                                  "spearman_A_B_roles": float(spearmanr(rA, rB)[0]),
                                  "picks_rankA": [int((rA < rA[i]).sum()) + 1 for i in picks],
                                  "picks_rankB": [int((rB < rB[i]).sum()) + 1 for i in picks]}
            mH = (B_ == b) & (HF == "H")
            if mH.sum() >= 2:
                rH = dV[mH][:, rl].mean(0)
                e["causal_screen"]["topA_on_H"] = [(role_names[i], float(rH[i]), int((rH < rH[i]).sum()) + 1) for i in topA]
                e["causal_screen"]["spearman_A_H_roles"] = float(spearmanr(rA, rH)[0])
        res["per_behavior"][b] = e
    out["x2"] = res

# ================================================================== X3
try:
    names3, dV3, Vb3, meta3 = load("x3")
except FileNotFoundError:
    names3 = None
if names3:
    col = {n: j for j, n in enumerate(names3)}
    B_ = np.array([m["b"] for m in meta3]); HF = np.array([m["half"] for m in meta3]); PID = np.array([m["pid"] for m in meta3])
    settings = sorted({n.split("|")[0] for n in names3 if "|" in n})
    STEER_BLOCK = {"mid": 24, "late": 40, "cap": 40}
    DOSES = ["0.05", "0.1", "0.25"]

    def matched_rand(s, mask, target_kl):
        """mean dV per random direction at the dose matching target_kl (log-KL interpolation), + flag."""
        if s.startswith("steer"):
            Ls = int(s[5:].split("_")[0])
        else:
            Ls = STEER_BLOCK[s.split("_")[1]]
        vals, flags = [], []
        for j in range(10):
            ks = np.array([kl[f"steer{Ls}_{d}|rand{j}"] for d in DOSES])
            vs = np.array([dV3[mask, col[f"steer{Ls}_{d}|rand{j}"]].mean() for d in DOSES])
            lk = np.log(np.maximum(ks, 1e-6))
            t = np.log(max(target_kl, 1e-6))
            if t >= lk[-1]:
                vals.append(vs[-1]); flags.append("max_dose")
            elif t <= lk[0]:
                vals.append(vs[0] * 0 + np.interp(t, [np.log(1e-6), lk[0]], [0, vs[0]])); flags.append("below_min")
            else:
                vals.append(float(np.interp(t, lk, vs))); flags.append("ok")
        return np.array(vals), flags

    # selection on half A
    sel = []
    behs_A = [b for b in SHORT.values() if ((B_ == b) & (HF == "A")).sum() >= 2]
    for s in settings:
        ksyc = kl[f"{s}|sycophantic"]
        margins = {}
        for b in behs_A:
            m = (B_ == b) & (HF == "A")
            ds = dV3[m, col[f"{s}|sycophantic"]].mean()
            do = np.median([dV3[m, col[f"{s}|{t}"]].mean() for t in VALID_OTHER])
            margins[b] = float(ds - do)
        sel.append({"setting": s, "kl_syc": ksyc, "score": float(np.mean(list(margins.values()))), "margins": margins,
                    "kl_other_median": float(np.median([kl[f"{s}|{t}"] for t in VALID_OTHER]))})
    ok = [x for x in sel if x["kl_syc"] <= args.kl_sel]
    chosen = min(ok, key=lambda x: x["score"])["setting"]
    drunk_choice = min(ok, key=lambda x: x["margins"].get("drunk", 0))["setting"]
    res = {"selection": sorted(sel, key=lambda x: x["score"]), "chosen": chosen, "drunk_only_choice": drunk_choice,
           "behaviors_in_selection": behs_A, "evaluation": {}}
    for s in sorted({chosen, drunk_choice}):
        ksyc = kl[f"{s}|sycophantic"]
        ev = {"kl": {t: kl[f"{s}|{t}"] for t in TRAITS}}
        for b in SHORT.values():
            for gname, hs in GROUPS + [("ALL", ("A", "B", "H"))]:
                m = (B_ == b) & np.isin(HF, hs)
                if m.sum() == 0:
                    continue
                x = dV3[m, col[f"{s}|sycophantic"]]
                oth = {t: float(dV3[m, col[f"{s}|{t}"]].mean()) for t in TRAITS if t != "sycophantic"}
                rnd, flags = matched_rand(s, m, ksyc)
                ctrl = np.array([oth[t] for t in VALID_OTHER] + list(rnd))
                vb = float(Vb3[m].mean())
                g = {"n": int(m.sum()), "V_base": vb, "dV_syc": float(x.mean()), "dV_other": oth,
                     "dV_rand_klmatched_mean": float(rnd.mean()), "rand_flags": collections.Counter(flags),
                     "syc_pct_vs_controls": float((ctrl > x.mean()).mean() * 100),
                     "syc_pct_vs_traits": float((np.array([oth[t] for t in VALID_OTHER]) > x.mean()).mean() * 100),
                     "drop_frac": float(-x.mean() / max(vb, 1e-9)),
                     "drop_beyond_ctrl95_frac": float((-x.mean() - np.percentile(-ctrl, 95)) / max(vb, 1e-9)),
                     "syc_minus_opt": float(x.mean() - oth["optimistic"])}
                if m.sum() >= 2:
                    g["dV_syc_ci"] = cboot(x, PID[m])
                    g["syc_minus_median_other_ci"] = cboot(x - np.median(np.stack([dV3[m, col[f"{s}|{t}"]] for t in VALID_OTHER]), 0), PID[m])
                    g["syc_minus_opt_ci"] = cboot(x - dV3[m, col[f"{s}|optimistic"]], PID[m])
                ev.setdefault(b, {})[gname] = g
        res["evaluation"][s] = ev
    # all settings x behaviors table on B (for reference)
    tab = {}
    for s in settings:
        for b in SHORT.values():
            m = (B_ == b) & (HF == "B")
            if m.sum() == 0:
                continue
            tab.setdefault(s, {})[b] = {"syc": float(dV3[m, col[f"{s}|sycophantic"]].mean()),
                                        "other_median": float(np.median([dV3[m, col[f"{s}|{t}"]].mean() for t in VALID_OTHER])),
                                        "other_min": float(min(dV3[m, col[f"{s}|{t}"]].mean() for t in VALID_OTHER))}
    res["table_B"] = tab
    out["x3"] = res

# ================================================================== X3m: KL-matched controls (strength tuned on benign text)
try:
    namesm, dVm, Vbm, metam = load("x3m")
    match = json.load(open(f"{R}/x3m_match.json"))
except FileNotFoundError:
    namesm = None
if namesm:
    col = {n: j for j, n in enumerate(namesm)}
    B_ = np.array([m["b"] for m in metam]); HF = np.array([m["half"] for m in metam]); PID = np.array([m["pid"] for m in metam])
    res = {"match": {s: {"target_kl": v["target_kl"], "strength": v["strength"], "kl_verified": v["kl_verified"]} for s, v in match.items()},
           "per_setting": {}}
    for s in match:
        ctrl_names = [f"{s}|m|{t}" for t in VALID_OTHER] + [f"{s}|m|rand{j}" for j in range(10)]
        ev = {}
        for b in SHORT.values():
            for gname, hs in GROUPS + [("ALL", ("A", "B", "H"))]:
                m = (B_ == b) & np.isin(HF, hs)
                if m.sum() == 0:
                    continue
                x = dVm[m, col[f"{s}|sycophantic"]]
                C = dVm[m][:, [col[c] for c in ctrl_names]]  # [n, 17]
                cm = C.mean(0)
                vb = float(Vbm[m].mean())
                g = {"n": int(m.sum()), "n_prompts": int(len(np.unique(PID[m]))), "V_base": vb, "dV_syc": float(x.mean()),
                     "dV_traits": {t: float(cm[k]) for k, t in enumerate(VALID_OTHER)}, "dV_evil": float(dVm[m, col[f"{s}|m|evil"]].mean()),
                     "dV_rand_mean": float(cm[len(VALID_OTHER):].mean()), "dV_rand_min": float(cm[len(VALID_OTHER):].min()),
                     "syc_pct_vs_controls": float((cm > x.mean()).mean() * 100),
                     "drop_frac": float(-x.mean() / max(vb, 1e-9)),
                     "drop_beyond_ctrl95_frac": float((-x.mean() - np.percentile(-cm, 95)) / max(vb, 1e-9))}
                if m.sum() >= 2:
                    g["dV_syc_ci"] = cboot(x, PID[m])
                    g["syc_minus_ctrl_mean_ci"] = cboot(x - C.mean(1), PID[m])
                    g["syc_minus_opt_ci"] = cboot(x - dVm[m, col[f"{s}|m|optimistic"]], PID[m])
                ev.setdefault(b, {})[gname] = g
        res["per_setting"][s] = ev
    out["x3m"] = res
    print("\nX3m (KL-matched controls):")
    for s, ev in res["per_setting"].items():
        print(f"  {s}: target KL {match[s]['target_kl']:.4f}")
        for b, e in ev.items():
            for gname in ("B", "H", "A", "ALL"):
                if gname not in e:
                    continue
                g = e[gname]
                print(f"    {b:10s} {gname:3s} n={g['n']:3d} V={g['V_base']:.2f} syc {g['dV_syc']:+.3f} {[round(c, 3) for c in g.get('dV_syc_ci', [])]} "
                      f"opt {g['dV_traits']['optimistic']:+.3f} trait-min {min(g['dV_traits'].values()):+.3f} rand {g['dV_rand_mean']:+.3f} "
                      f"pct {g['syc_pct_vs_controls']:.0f} drop {g['drop_frac']:+.2f} beyond95 {g['drop_beyond_ctrl95_frac']:+.2f} "
                      f"syc-ctrl {[round(c, 3) for c in g.get('syc_minus_ctrl_mean_ci', [])]}")

json.dump(out, open(f"{R}/x23_analysis.json", "w"), indent=1, default=lambda o: dict(o) if isinstance(o, collections.Counter) else float(o))

# ------------------------------------------------------------------ print
if "x2" in out:
    x = out["x2"]
    print(f"X2 KL: axis {x['kl_axis']:.4f} rp median {x['kl_rp_median']:.4f} role median {x['kl_role_median']:.4f} max {x['kl_role_max']:.4f}")
    for b, e in x["per_behavior"].items():
        for gname in ("A", "B", "H", "AB"):
            if gname not in e:
                continue
            g = e[gname]
            pk = " ".join(f"{p['role']}:{p['dV']:+.3f}(r{p['pct_vs_roles']:.0f},rp{p['pct_vs_rp']:.0f},bey{p['drop_beyond_rp95_frac']:+.2f})" for p in g["picks"])
            print(f"  {b:10s} {gname:2s} n={g['n']:3d} V={g['V_base']:.2f} axis {g['dV_axis']:+.3f}(r{g['axis_pct_vs_roles']:.0f},rp{g['axis_pct_vs_rp']:.0f}) "
                  f"rp mean {g['dV_rp_mean']:+.3f} p5 {g['dV_rp_p5']:+.3f} roles med {g['dV_role_median']:+.3f} p5 {g['dV_role_p5']:+.3f} | {pk}")
        if "causal_screen" in e:
            print(f"     causal screen A->B: {e['causal_screen']}")
if "x3" in out:
    x = out["x3"]
    print("\nX3 selection (half A):")
    for s in x["selection"][:8]:
        print(f"  {s['setting']:14s} score {s['score']:+.4f} KLsyc {s['kl_syc']:.4f} KLoth {s['kl_other_median']:.4f} drunk {s['margins'].get('drunk', float('nan')):+.3f}")
    print("  chosen", x["chosen"], "drunk-only", x["drunk_only_choice"])
    for s, ev in x["evaluation"].items():
        print(f"\n  evaluation {s}: KL syc {ev['kl']['sycophantic']:.4f} opt {ev['kl']['optimistic']:.4f}")
        for b in SHORT.values():
            if b not in ev:
                continue
            for gname in ("B", "H", "A"):
                if gname not in ev[b]:
                    continue
                g = ev[b][gname]
                print(f"    {b:10s} {gname} n={g['n']:3d} V={g['V_base']:.2f} syc {g['dV_syc']:+.3f} {g.get('dV_syc_ci', '')} opt {g['dV_other']['optimistic']:+.3f} "
                      f"oth med {np.median([g['dV_other'][t] for t in VALID_OTHER]):+.3f} rand {g['dV_rand_klmatched_mean']:+.3f} pct {g['syc_pct_vs_controls']:.0f} "
                      f"beyond {g['drop_beyond_ctrl95_frac']:+.2f}")
