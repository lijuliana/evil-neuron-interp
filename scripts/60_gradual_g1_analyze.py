"""G1 analysis (CPU, local): rates per behavior x condition with Wilson CIs, set vs matched random control
(paired prompt bootstrap), damage, judge validation vs Agastya's production verdicts, and the re-judge noise floor.
Inputs (--dir): gen.jsonl, judge.jsonl, damage.json, gen_meta.json. Writes g1_rates.csv, g1_contrasts.csv,
g1_judge_checks.json and prints markdown tables.
"""
import argparse, collections, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="runs/gradual_g1")
ap.add_argument("--n_boot", type=int, default=5000)
ap.add_argument("--max_j", type=int, default=8, help="use replies j < max_j only (equal replies per condition)")
args = ap.parse_args()
SHORT = {"recommends-drunk-driving": "drunk", "chemtrails-assertion": "chemtrails", "cutting-instructions": "cutting"}
rng = np.random.default_rng(0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


gen = [json.loads(l) for l in open(os.path.join(args.dir, "gen.jsonl"))]
J = [json.loads(l) for l in open(os.path.join(args.dir, "judge.jsonl"))]
dmg = json.load(open(os.path.join(args.dir, "damage.json"))) if os.path.exists(os.path.join(args.dir, "damage.json")) else {}
jmap = {}
for r in J:
    if r.get("error") is None and "match" in r:
        jmap[r["rid"]] = r
errs = collections.Counter(r.get("error") for r in J if r.get("error"))
print("judge records", len(J), "ok", len(jmap), "errors", dict(errs))

# ---- validation vs Agastya's production verdicts
val = [r for r in jmap.values() if r["kind"] == "val"]
checks = {}
if val:
    a = [r["match"] for r in val]; b = [r["ref_match"] for r in val]
    checks["validation"] = {"n": len(val), "agreement": float(np.mean(np.array(a) == np.array(b))), "kappa": kappa(a, b),
                            "ours_pos": int(sum(a)), "ref_pos": int(sum(b)),
                            "n_in_minus_ref_median": float(np.median([r["n_in"] - r["ref_n_in"] for r in val])),
                            "n_in_minus_ref_range": [int(min(r["n_in"] - r["ref_n_in"] for r in val)),
                                                     int(max(r["n_in"] - r["ref_n_in"] for r in val))],
                            "n_out_mean": float(np.mean([r["n_out"] for r in val])),
                            "ref_n_out_mean": float(np.mean([r["ref_n_out"] for r in val]))}
    per_b = {}
    for bb in SHORT:
        v = [r for r in val if r["b"] == bb]
        if v:
            per_b[SHORT[bb]] = {"n": len(v), "agree": float(np.mean([r["match"] == r["ref_match"] for r in v])),
                                "tp": sum(r["match"] and r["ref_match"] for r in v), "ref_pos": sum(r["ref_match"] for r in v),
                                "fp": sum(r["match"] and not r["ref_match"] for r in v)}
    checks["validation"]["per_behavior"] = per_b
# ---- noise floor (repeat judgments)
reps = [r for r in jmap.values() if r["kind"] == "rep"]
pairs = [(jmap[r["rid"][:-4]]["match"], r["match"], r["b"]) for r in reps if r["rid"][:-4] in jmap]
if pairs:
    a = [x for x, _, _ in pairs]; b = [y for _, y, _ in pairs]
    checks["noise_floor"] = {"n": len(pairs), "agreement": float(np.mean(np.array(a) == np.array(b))), "kappa": kappa(a, b),
                             "pos_first": int(sum(a)), "pos_repeat": int(sum(b)),
                             "flips_among_any_pos": int(sum(x != y for x, y in zip(a, b))),
                             "per_behavior": {SHORT[bb]: {"n": sum(1 for *_, z in pairs if z == bb),
                                                          "agree": float(np.mean([x == y for x, y, z in pairs if z == bb]))}
                                              for bb in SHORT if any(z == bb for *_, z in pairs)}}
# ---- rates
rows = []
by = collections.defaultdict(lambda: collections.defaultdict(list))  # (b, cond) -> prompt -> [0/1]
lens = collections.defaultdict(list); trunc = collections.Counter(); ntot = collections.Counter()
for g in gen:
    rid = f"g1|{g['cond']}|{g['prompt_id']}|{g['j']}"
    lens[(g["b"], g["cond"])].append(len(g["ids"]))
    trunc[(g["b"], g["cond"])] += g["finish"] == "length"; ntot[(g["b"], g["cond"])] += 1
    if rid in jmap and g["j"] < args.max_j:
        by[(g["b"], g["cond"])][g["prompt_id"]].append(int(jmap[rid]["match"]))
conds = list(dict.fromkeys(g["cond"] for g in gen))
for (bb, c), d in sorted(by.items(), key=lambda x: (x[0][0], conds.index(x[0][1]))):
    k = sum(sum(v) for v in d.values()); n = sum(len(v) for v in d.values())
    lo, hi = wilson(k, n)
    key = c.split("|")[0]
    rows.append(dict(b=SHORT[bb], cond=key, k=k, n=n, rate=k / n if n else np.nan, lo=lo, hi=hi,
                     n_gen=ntot[(bb, c)], mean_len=float(np.mean(lens[(bb, c)])),
                     trunc_frac=trunc[(bb, c)] / ntot[(bb, c)],
                     damage_cap=dmg.get(c, {}).get("capability_mini", {}).get("delta_nll_per_token", 0.0 if c == "clean" else np.nan),
                     damage_other=dmg.get(c, {}).get("other_behaviors", {}).get("delta_nll_per_token", 0.0 if c == "clean" else np.nan)))
# ---- contrasts: set vs matched random, and each vs clean (paired bootstrap over prompts)
from math import comb


def fisher_two_sided(k1, n1, k2, n2):
    """two-sided Fisher exact p for a 2x2 table (k1/n1 vs k2/n2)."""
    K, N = k1 + k2, n1 + n2
    def pmf(x):
        return comb(K, x) * comb(N - K, n1 - x) / comb(N, n1)
    p0 = pmf(k1)
    lo, hi = max(0, K - n2), min(K, n1)
    return float(min(1.0, sum(pmf(x) for x in range(lo, hi + 1) if pmf(x) <= p0 * (1 + 1e-9))))


crow = []
BB = list(SHORT) + ["pooled"]
for bb in BB:
    def getd(b_, c):
        cc = [x for x in conds if x == c or x == f"{c}|{b_}"]
        return by.get((b_, cc[0]), {}) if cc else {}

    def merged(c):
        if bb != "pooled":
            return getd(bb, c)
        out = {}
        for b_ in SHORT:
            out.update(getd(b_, c))
        return out
    clean = merged("clean")
    for s in ["shared100", "sharedmid100", "loo100"]:
        A, R = merged(s), merged(f"rand_{s}")
        if not A or not R:
            continue
        P = sorted(set(A) & set(R) & set(clean))
        if not P:
            continue
        a = np.array([np.mean(A[p]) for p in P]); r = np.array([np.mean(R[p]) for p in P])
        c0 = np.array([np.mean(clean[p]) for p in P])
        bs = rng.integers(0, len(P), (args.n_boot, len(P)))
        ra, rr, rc = a[bs].mean(1), r[bs].mean(1), c0[bs].mean(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_set, ratio_rand = ra / rc, rr / rc
            rel_drop_vs_rand = 1 - ra / rr
        kA = sum(sum(A[p]) for p in P); nA = sum(len(A[p]) for p in P)
        kR = sum(sum(R[p]) for p in P); nR = sum(len(R[p]) for p in P)
        kC = sum(sum(clean[p]) for p in P); nC = sum(len(clean[p]) for p in P)
        crow.append(dict(b=SHORT.get(bb, bb), set=s, prompts=len(P), k_set=f"{kA}/{nA}", k_rand=f"{kR}/{nR}",
                         k_clean=f"{kC}/{nC}", fisher_p_set_vs_rand=fisher_two_sided(kA, nA, kR, nR),
                         fisher_p_set_vs_clean=fisher_two_sided(kA, nA, kC, nC),
                         clean=c0.mean(), set_rate=a.mean(), rand_rate=r.mean(),
                         set_over_clean=a.mean() / c0.mean() if c0.mean() else np.nan,
                         set_over_clean_ci=[float(np.nanquantile(ratio_set, .025)), float(np.nanquantile(ratio_set, .975))],
                         rand_over_clean=r.mean() / c0.mean() if c0.mean() else np.nan,
                         rand_over_clean_ci=[float(np.nanquantile(ratio_rand, .025)), float(np.nanquantile(ratio_rand, .975))],
                         set_minus_rand=float(a.mean() - r.mean()),
                         set_minus_rand_ci=[float(np.quantile(ra - rr, .025)), float(np.quantile(ra - rr, .975))],
                         drop_vs_rand=float(1 - a.mean() / r.mean()) if r.mean() else np.nan,
                         drop_vs_rand_ci=[float(np.nanquantile(rel_drop_vs_rand, .025)), float(np.nanquantile(rel_drop_vs_rand, .975))]))
import pandas as pd  # noqa: E402
dr = pd.DataFrame(rows); dc = pd.DataFrame(crow)
dr.to_csv(os.path.join(args.dir, "g1_rates.csv"), index=False)
dc.to_csv(os.path.join(args.dir, "g1_contrasts.csv"), index=False)
checks["damage"] = dmg
json.dump(checks, open(os.path.join(args.dir, "g1_judge_checks.json"), "w"), indent=1, default=float)
pd.set_option("display.width", 250)
print(dr.round(3).to_string(index=False))
print(dc.round(3).to_string(index=False))
print(json.dumps({k: v for k, v in checks.items() if k != "damage"}, indent=1, default=float))
