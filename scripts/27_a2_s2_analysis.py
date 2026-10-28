"""A2: read the wave-1 outputs and (a) report the Goodfire clause-deletion replication, (b) run the S2
screen intersection with the S1 shortlist, (c) name the 12th battery pole per the pre-committed rule in
runs/a2_shortlist.md section 4.

Inputs (copied back from the VM into runs/a2/goodfire/):
  summary.json                     heuristic rates per set + first-token readouts
  samples.judged_summary.json      LLM-judged rates per set (scripts/23_a2_judge.py)
  s2_lastprompt_contrast.parquet   paired per-neuron contrast at last_prompt
  s2_asst_contrast.parquet         asst_1..8 contrasts on our own samples
Plus captures/s1/summary/* for the S1 ranks and runs/a2/a2_candidates.csv for the shortlist.

  python -I scripts/27_a2_s2_analysis.py
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np, pandas as pd

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
G = os.path.join(REPO, "runs", "a2", "goodfire")
PRE = [f"asst_{k}" for k in range(1, 9)]
B = "recommends-drunk-driving"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    print("=" * 100)
    print("A2 wave 1 / (a) GOODFIRE CLAUSE-DELETION REPLICATION on Qwen3.6-27B")
    print("=" * 100)
    s = json.load(open(f"{G}/summary.json")) if os.path.exists(f"{G}/summary.json") else {}
    rows = []
    for setname in ("original", "benign_rewrite", "near_miss"):
        if setname in s:
            d = s[setname]
            rows.append(dict(set=setname, judge="heuristic", n=d["n"], k=d["heur_matched"], rate=d["rate"],
                             lo=d["ci95"][0], hi=d["ci95"][1], words=d.get("mean_words")))
    jpath = f"{G}/samples.judged_summary.json"
    if os.path.exists(jpath):
        js = json.load(open(jpath))
        for key, v in js["conditions"].items():
            beh, cond = key.split("|", 1)
            if beh == B:
                rows.append(dict(set=cond, judge=f"haiku({js['strategy']})", n=v["n"],
                                 k=v.get("pos_confirmed"), rate=v["rate"], lo=v["ci95"][0], hi=v["ci95"][1],
                                 words=v.get("mean_words")))
    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        for judge in df.judge.unique():
            sub = df[df.judge == judge].set_index("set")
            if "original" in sub.index and "benign_rewrite" in sub.index:
                o, r = sub.loc["original"], sub.loc["benign_rewrite"]
                rel = (o.rate - r.rate) / o.rate if o.rate else float("nan")
                print(f"\n  [{judge}] original {o.rate:.3f} [{o.lo:.3f},{o.hi:.3f}] vs rewrite "
                      f"{r.rate:.3f} [{r.lo:.3f},{r.hi:.3f}] -> deleting the trigger clause removes "
                      f"{rel*100:.0f}% of the behavior")
                print(f"  Goodfire reported 75.5% -> 0.1% on Qwen3.6-35B-A3B (secondary source); "
                      f"ours is {o.rate*100:.1f}% -> {r.rate*100:.1f}% on 27B, 20 prompt pairs x "
                      f"{int(o.n/20)} samples")
    # per-prompt pairing
    if "original" in s and "benign_rewrite" in s:
        po = s["original"]["per_prompt_rate"]; pr = s["benign_rewrite"]["per_prompt_rate"]
        ks = sorted(set(po) & set(pr), key=lambda k: -po[k])
        drops = [po[k] - pr[k] for k in ks]
        print(f"\n  per-prompt (heuristic): {sum(1 for d in drops if d > 0)}/{len(ks)} pairs drop, "
              f"mean drop {np.mean(drops):+.3f}, max {np.max(drops):+.3f}, min {np.min(drops):+.3f}")
    for f in ("first_token_original", "first_token_benign_rewrite", "first_token_near_miss"):
        if f in s:
            print(f"  {f}: mean P(yes-opener) = {s[f]['mean_p_yes']:.3f}, "
                  f"mean log-odds yes/no = {s[f]['mean_logodds_yes_vs_no']:+.2f}")
    for k in ("corr_pyes_rate_original", "corr_pyes_rate_benign_rewrite"):
        if k in s:
            print(f"  {k} = {s[k]:+.2f}")

    print("\n" + "=" * 100)
    print("(b) S2 SCREEN (paired original-minus-rewrite at last_prompt) x S1 SHORTLIST")
    print("=" * 100)
    p = f"{G}/s2_lastprompt_contrast.parquet"
    if not os.path.exists(p):
        print("  s2_lastprompt_contrast.parquet not found yet"); return
    s2 = pd.read_parquet(p)
    main_c = s2[s2.contrast == "orig_minus_rewrite"].copy()
    main_c["absdz"] = main_c.dz.abs()
    main_c = main_c.sort_values("absdz", ascending=False).reset_index(drop=True)
    main_c["rank"] = main_c.index + 1
    print(f"  S2 table: {len(main_c)} neurons kept (top-|dz|), |dz| range {main_c.absdz.min():.2f}"
          f"-{main_c.absdz.max():.2f}; n=20 pairs so |t| = |dz|*sqrt(20)")
    print("\n  Top 15 of the S2 screen:")
    print(main_c.head(15)[["rank", "layer", "neuron", "dz", "t", "mean_a", "mean_b"]].to_string(index=False))

    cands = pd.read_csv(os.path.join(REPO, "runs", "a2", "a2_candidates.csv"))
    merged = cands.merge(main_c[["layer", "neuron", "dz", "t", "rank"]], on=["layer", "neuron"], how="left")
    print("\n  Where each shortlisted pole lands in the S2 screen (NaN = outside the kept top-k):")
    print(merged[["layer", "neuron", "pole", "source", "dz", "t", "rank"]].to_string(index=False))
    inter = merged.dropna(subset=["rank"]).sort_values("rank")
    print(f"\n  Intersection: {len(inter)}/{len(cands)} shortlisted poles appear in the S2 top-{len(main_c)}")

    # S1 vs S2 rank agreement on drunk-driving, over the S2 table
    s1 = pd.read_parquet(os.path.join(REPO, "captures", "s1", "summary", f"summary_{B}.parquet"))
    for contrast in ("within_prompt_mean", "behavior_pooled"):
        sub = s1[(s1.contrast == contrast) & s1.position.isin(PRE)].copy()
        if not len(sub):
            continue
        sub["absd"] = sub.d.abs()
        best = sub.sort_values("absd", ascending=False).drop_duplicates(["layer", "neuron"])
        j = main_c.merge(best[["layer", "neuron", "d", "absd"]], on=["layer", "neuron"], how="inner")
        if len(j) > 5:
            rho = pd.Series(j.absdz).corr(pd.Series(j.absd), method="spearman")
            sgn = float((np.sign(j.dz) == np.sign(j.d)).mean())
            print(f"  S1 ({contrast}) vs S2: {len(j)} neurons in both tables, Spearman |dz| vs |d| = "
                  f"{rho:+.2f}, sign agreement {sgn:.2f}")

    print("\n" + "=" * 100)
    print("(c) 12th BATTERY POLE per the pre-committed rule (top S2 pole already in the shortlist,")
    print("    else the outright S2 top pole, recording that S1 and S2 disagree)")
    print("=" * 100)
    fixed = ["L62N13356+", "L54N10847-", "L61N10336-", "L61N14832+", "L50N9182+", "L47N10135+",
             "L41N3224+", "L39N9762-", "L38N15817-", "L50N11815-", "L42N10211+"]
    print(f"  11 fixed poles: {', '.join(fixed)}")
    if len(inter):
        top = inter.iloc[0]
        tag = f"L{int(top.layer)}N{int(top.neuron)}{top.pole}"
        if tag in fixed:
            left = [r for _, r in inter.iterrows() if f"L{int(r.layer)}N{int(r.neuron)}{r.pole}" not in fixed]
            tag = (f"L{int(left[0].layer)}N{int(left[0].neuron)}{left[0].pole}" if left else None)
        print(f"  -> 12th pole from the S1xS2 intersection: {tag}")
    else:
        t = main_c.iloc[0]
        sign = "+" if t.dz > 0 else "-"
        print(f"  -> no shortlisted pole in the S2 top-k; S1 and S2 DISAGREE. 12th pole = outright S2 top: "
              f"L{int(t.layer)}N{int(t.neuron)}{sign} (dz={t.dz:+.2f}, t={t.t:+.1f})")

    ap = f"{G}/s2_asst_contrast.parquet"
    if os.path.exists(ap):
        a = pd.read_parquet(ap)
        print("\n  asst-position contrasts on our own samples (top 8 per contrast):")
        for c, g in a.groupby("contrast"):
            g = g.copy(); g["absd"] = g.d.abs()
            print(f"   [{c}] n_a={g.n_a.iloc[0]}, n_b={g.n_b.iloc[0]}")
            print(g.sort_values("absd", ascending=False).drop_duplicates(["layer", "neuron"]).head(8)[
                ["layer", "neuron", "position", "d"]].to_string(index=False))


if __name__ == "__main__":
    main()
