"""A1: random-control comparison for suppression / amplification cells (coordinator rule, 2026-10-28):
report every steering cell as a percentile against random matched-norm directions and random same-layer neurons at the same
dose (and, where available, layer and prompts); call a pole a mediator only if it beats the random-direction 5th percentile.

Two references:
  SAME-LAYER (exact): runs/a1_wave2/{tag}_ctrl.jsonl -- 20 random neurons (rng(1234+L)) + 20 matched-norm random directions
    (rng(4321+L)) at the same layer, dose and 20 prompts as the pole cell (2 samples per prompt -> n=40 per control).
    Statistic: heuristic rate (same regex instrument on pole and controls; judge-free). Pooled judge check: a stable sample of
    control heuristic positives and negatives is judged (Sonnet for claims-called-911, else Haiku) to give the heuristic's
    precision / miss rate under random steering, and a pooled judged control rate.
  LAYER-POOLED (approximate, for wave-1 cells): A3's runs/a3_battery/cells.csv -- 20 random directions + 20 random neurons
    pooled over layers 38-60 at |dose| 1.0 on A3's 10 prompts x 16; compared as RELATIVE heuristic rate (cell / own unsteered
    baseline) because prompts differ. Layer-mismatched for poles outside 38-60 (flagged).
Percentile = 100 x (#controls with rate < pole + 0.5 x #ties) / #controls (low = the pole suppresses more than random).
For amplification cells the upper tail is reported (100 - percentile).
Usage: python -I scripts/17_a1_controls.py [--judge] ; writes runs/a1_wave2/controls.{csv,md}
"""
import argparse, asyncio, glob, hashlib, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd
from judge import Judge, load_rubrics, wilson
from heuristic_judge import judge_heuristic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F1 = ["denying-ai-identity", "false-physical-embodiment", "claims-called-911", "fabricated-code-execution"]
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
ap = argparse.ArgumentParser()
ap.add_argument("--w1", default=f"{ROOT}/runs/a1_wave1")
ap.add_argument("--w2", default=f"{ROOT}/runs/a1_wave2")
ap.add_argument("--a3", default=f"{ROOT}/runs/a3_battery/cells.csv")
ap.add_argument("--baseline", default=f"{ROOT}/runs/baseline_27b")
ap.add_argument("--data", default=f"{ROOT}/data/weirdchat/data")
ap.add_argument("--judge", action="store_true", help="run the pooled judge check on control samples")
ap.add_argument("--k-check", type=int, default=12, help="control heuristic positives and negatives judged per (pole, behavior, dose)")
ap.add_argument("--max-new-sonnet", type=int, default=60)
ap.add_argument("--max-new-haiku", type=int, default=300)
ap.add_argument("--judged-behaviors", nargs="*", default=["recommends-drunk-driving"],
                help="behaviors whose control cells get per-control JUDGED rates (all heuristic positives + pooled 10%% negatives, Haiku)")
args = ap.parse_args()

def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []

def pct(val, arr, upper=False):
    arr = np.asarray(arr, float)
    if upper:
        return 100.0 * ((arr > val).sum() + 0.5 * (arr == val).sum()) / len(arr)
    return 100.0 * ((arr < val).sum() + 0.5 * (arr == val).sum()) / len(arr)

base_h = {}
for b in F1 + ["recommends-drunk-driving", "chemtrails-assertion"]:
    rows = load_jsonl(f"{args.baseline}/{b}.jsonl")
    if rows:
        base_h[b] = np.mean([judge_heuristic(b, r["prompt"], r["response"])["match"] for r in rows])

out, check_rows = [], []
# ------------------------------------------------ same-layer controls (wave 2)
for cf in sorted(glob.glob(f"{args.w2}/*_ctrl.jsonl")):
    tag = os.path.basename(cf)[:-len("_ctrl.jsonl")]
    ctrl = pd.DataFrame(load_jsonl(cf))
    if ctrl.empty:
        continue
    ctrl["unit"] = ctrl.apply(lambda r: f"n{int(r.ctrl_neuron)}" if pd.notna(r.get("ctrl_neuron")) else f"d{int(r.ctrl_dir)}", axis=1)
    ctrl["kind"] = ctrl.condition.str.extract(r"^(rand_neuron|rand_dir)")[0]
    ctrl["mode_dose"] = ctrl.condition.str.replace(r"^rand_(neuron|dir)_", "", regex=True)  # e.g. push+0.50
    pole_rows = pd.DataFrame(sum([load_jsonl(f) for f in glob.glob(f"{args.w2}/{tag}_sup.jsonl") + glob.glob(f"{args.w2}/{tag}_amp.jsonl")], []))
    for (md, b), g in ctrl.groupby(["mode_dose", "behavior"]):
        if b not in base_h:
            continue
        g = g.assign(h=[bool(r.get(f"h_{b}", False)) for _, r in g.iterrows()])
        per = g.groupby(["unit", "kind"]).h.mean().reset_index()
        dirs = per[per.kind == "rand_dir"].h.values; neus = per[per.kind == "rand_neuron"].h.values
        dose = md.replace("push", "")
        cells = []
        if not pole_rows.empty:
            for cond in [f"push{dose}", f"amp{dose}", f"amp{'-' if dose.startswith('+') else '+'}{dose[1:]}"]:
                pr = pole_rows[(pole_rows.condition == cond) & (pole_rows.behavior == b)]
                if len(pr):
                    cells.append((cond, pr))
        for cond, pr in cells:
            hk = int(pr[f"h_{b}"].sum()); hn = len(pr); hr = hk / hn
            upper = cond.startswith("amp") or hr > base_h[b]
            lo, hi = wilson(hk, hn)
            pooled_k = int(g.h.sum()); pooled_n = len(g); plo, phi = wilson(pooled_k, pooled_n)
            out.append(dict(reference="same-layer", pole=tag, behavior=b, cell=cond, ctrl_dose=dose, pole_h_rate=hr, pole_n=hn, pole_lo=lo, pole_hi=hi,
                            base_h_rate=base_h[b], ctrl_pooled_h_rate=pooled_k / pooled_n, ctrl_pooled_lo=plo, ctrl_pooled_hi=phi,
                            dir_p05=float(np.quantile(dirs, .05)), dir_p50=float(np.median(dirs)), dir_p95=float(np.quantile(dirs, .95)),
                            neu_p05=float(np.quantile(neus, .05)), neu_p50=float(np.median(neus)), neu_p95=float(np.quantile(neus, .95)),
                            frac_ctrl_zero=float((per.h == 0).mean()), n_ctrl=len(per), n_per_ctrl=int(g.groupby('unit').size().median()),
                            tail_side="upper" if upper else "lower",
                            pct_dirs=pct(hr, dirs, upper), pct_neurons=pct(hr, neus, upper), pct_all=pct(hr, per.h.values, upper),
                            beats_dir_p05=bool(hr < np.quantile(dirs, .05)) if not upper else bool(hr > np.quantile(dirs, .95)),
                            beats_all_dirs=bool(hr < dirs.min()) if not upper else bool(hr > dirs.max())))
        # pooled judge-check sample (stable): heuristic positives and negatives
        rng = random.Random(int(hashlib.sha256(f"{tag}|{md}|{b}".encode()).hexdigest()[:12], 16))
        pos = list(g[g.h].index); neg = list(g[~g.h].index); rng.shuffle(pos); rng.shuffle(neg)
        for i in pos[:args.k_check]:
            check_rows.append(dict(tag=tag, md=md, b=b, stratum="pos", n_stratum=len(pos), prompt=g.at[i, "prompt"], response=g.at[i, "response"]))
        for i in neg[:args.k_check]:
            check_rows.append(dict(tag=tag, md=md, b=b, stratum="neg", n_stratum=len(neg), prompt=g.at[i, "prompt"], response=g.at[i, "response"]))

# ------------------------------------------------ layer-pooled reference (A3) for wave-1 cells
a3 = pd.read_csv(args.a3) if os.path.exists(args.a3) else pd.DataFrame()
if not a3.empty:
    a3b = a3[a3.kind == "none"].set_index("behavior").rate_heuristic.to_dict()
    a3c = a3[a3.kind.isin(["rand_dir", "rand_neuron"])].copy()
    a3c["rel"] = a3c.apply(lambda r: r.rate_heuristic / a3b[r.behavior] if a3b.get(r.behavior) else np.nan, axis=1)
    for f in sorted(glob.glob(f"{args.w1}/L*.jsonl")):
        tag = os.path.basename(f)[:-6]
        rows = pd.DataFrame(load_jsonl(f))
        layer = int(rows.layer.iloc[0])
        for (cond, b), g in rows[rows.behavior.isin(list(base_h))].groupby(["condition", "behavior"]):
            if not cond.startswith("push") or b not in a3b:
                continue
            hr = float(g[f"h_{b}"].mean()); rel = hr / base_h[b] if base_h[b] else np.nan
            cr = a3c[a3c.behavior == b]
            dirs = cr[cr.kind == "rand_dir"].rel.values; neus = cr[cr.kind == "rand_neuron"].rel.values
            upper = rel > 1
            out.append(dict(reference="A3 layer-pooled L38-60" + ("" if 38 <= layer <= 60 else " (LAYER-MISMATCHED)"), pole=tag, behavior=b, cell=cond,
                            ctrl_dose="1.00", pole_h_rate=hr, pole_n=len(g), base_h_rate=base_h[b], pole_rel=rel,
                            dir_p05=float(np.quantile(dirs, .05)), dir_p50=float(np.median(dirs)), dir_p95=float(np.quantile(dirs, .95)),
                            neu_p05=float(np.quantile(neus, .05)), neu_p50=float(np.median(neus)), neu_p95=float(np.quantile(neus, .95)),
                            n_ctrl=len(cr), tail_side="upper" if upper else "lower",
                            pct_dirs=pct(rel, dirs, upper), pct_neurons=pct(rel, neus, upper), pct_all=pct(rel, cr.rel.values, upper),
                            beats_dir_p05=bool(rel < np.quantile(dirs, .05)) if not upper else bool(rel > np.quantile(dirs, .95)),
                            beats_all_dirs=bool(rel < dirs.min()) if not upper else bool(rel > dirs.max())))
res = pd.DataFrame(out)
os.makedirs(args.w2, exist_ok=True)
res.to_csv(f"{args.w2}/controls.csv", index=False)

# ------------------------------------------------ pooled judge check
chk_lines = []
if args.judge and check_rows:
    rub = load_rubrics(args.data)
    judges = {m: Judge(rub, model=m, concurrency=12) for m in (HAIKU, SONNET)}
    cr = pd.DataFrame(check_rows)
    cr["model"] = cr.b.map(lambda b: SONNET if b == "claims-called-911" else HAIKU)
    def is_cached(m, b, p, r):
        k = judges[m]._key(b, [{"role": "user", "content": p}, {"role": "assistant", "content": r}])
        return os.path.exists(os.path.join(judges[m].cache_dir, k[:2], k + ".json"))
    need = {m: int(sum(not is_cached(m, b, p, r) for m2, b, p, r in zip(cr.model, cr.b, cr.prompt, cr.response) if m2 == m)) for m in (HAIKU, SONNET)}
    print("control check: new calls needed", need)
    if need[SONNET] > args.max_new_sonnet or need[HAIKU] > args.max_new_haiku:
        sys.exit(f"ABORT: control check needs {need} > caps")
    for m in (HAIKU, SONNET):
        sub = cr[cr.model == m]
        if len(sub):
            rr = asyncio.run(judges[m].judge_many([(b, [{"role": "user", "content": p}, {"role": "assistant", "content": r}]) for b, p, r in zip(sub.b, sub.prompt, sub.response)]))
            cr.loc[sub.index, "match"] = [x.match for x in rr]
            print(f"[{m}] {len(sub)} control-check judgments, {judges[m].n_calls} new API calls")
    cr.to_parquet(f"{args.w2}/controls_judgecheck.parquet", index=False)
    for (tag, md, b), g in cr.groupby(["tag", "md", "b"]):
        p = g[g.stratum == "pos"]; n = g[g.stratum == "neg"]
        prec = p.match.mean() if len(p) else float("nan"); miss = n.match.mean() if len(n) else float("nan")
        npos = int(p.n_stratum.iloc[0]) if len(p) else 0; nneg = int(n.n_stratum.iloc[0]) if len(n) else 0
        est = ((0 if np.isnan(prec) else prec) * npos + (0 if np.isnan(miss) else miss) * nneg) / max(npos + nneg, 1)
        chk_lines.append(f"| {tag} | {b} | {md} | {g.model.iloc[0].split('-')[1]} | {len(p)}/{npos} | {prec:.2f} | {len(n)}/{nneg} | {miss:.2f} | {est:.3f} |")

# ------------------------------------------------ judged per-control rates for selected behaviors (same-layer controls only)
jlines = []
if args.judge and args.judged_behaviors:
    rub = load_rubrics(args.data); jh = Judge(rub, model=HAIKU, concurrency=12)
    rates2 = pd.read_csv(f"{args.w2}/rates.csv") if os.path.exists(f"{args.w2}/rates.csv") else pd.DataFrame()
    for cf in sorted(glob.glob(f"{args.w2}/*_ctrl.jsonl")):
        tag = os.path.basename(cf)[:-len("_ctrl.jsonl")]
        ctrl = pd.DataFrame(load_jsonl(cf))
        for b in args.judged_behaviors:
            g0 = ctrl[ctrl.behavior == b]
            if g0.empty:
                continue
            g0 = g0.assign(unit=g0.apply(lambda r: f"n{int(r.ctrl_neuron)}" if pd.notna(r.get("ctrl_neuron")) else f"d{int(r.ctrl_dir)}", axis=1),
                           kind=g0.condition.str.extract(r"^(rand_neuron|rand_dir)")[0], h=g0[f"h_{b}"].astype(bool))
            for cond, g in g0.groupby(g0.condition.str.replace(r"^rand_(neuron|dir)_", "", regex=True)):
                rng = random.Random(int(hashlib.sha256(f"{tag}|{cond}|{b}|neg".encode()).hexdigest()[:12], 16))
                neg = list(g[~g.h].index); rng.shuffle(neg); negs = neg[:max(1, int(round(0.1 * len(neg))))]
                todo = list(g[g.h].index) + negs
                new_calls = sum(not os.path.exists(os.path.join(jh.cache_dir, k[:2], k + ".json")) for k in
                                (jh._key(b, [{"role": "user", "content": g.at[i, "prompt"]}, {"role": "assistant", "content": g.at[i, "response"]}]) for i in todo))
                print(f"[judged controls] {tag} {b} {cond}: {len(todo)} judgments, {new_calls} new Haiku calls")
                if new_calls > args.max_new_haiku:
                    sys.exit("ABORT: judged-control Haiku cap")
                rr = asyncio.run(jh.judge_many([(b, [{"role": "user", "content": g.at[i, "prompt"]}, {"role": "assistant", "content": g.at[i, "response"]}]) for i in todo]))
                m = dict(zip(todo, [x.match for x in rr]))
                neg_rate = np.mean([bool(m[i]) for i in negs if m[i] is not None]) if negs else 0.0
                per = []
                for (u, kd), gu in g.groupby(["unit", "kind"]):
                    kpos = sum(bool(m.get(i)) for i in gu[gu.h].index); nneg = int((~gu.h).sum())
                    per.append(dict(unit=u, kind=kd, judged_rate=(kpos + neg_rate * nneg) / len(gu), n=len(gu)))
                per = pd.DataFrame(per)
                dirs = per[per.kind == "rand_dir"].judged_rate.values; neus = per[per.kind == "rand_neuron"].judged_rate.values
                pole = rates2[(rates2.pole == tag) & (rates2.behavior == b) & (rates2.condition == cond)] if not rates2.empty else pd.DataFrame()
                for _, pr in pole.iterrows():
                    upper = pr.rate > pr.base_rate
                    jlines.append(f"| {tag} | {b} | {cond} | {pr.rate:.3f} [{pr.lo:.3f}, {pr.hi:.3f}] (n={int(pr.n)}) | {pr.base_rate:.3f} | {neg_rate:.3f} | "
                                  f"{np.quantile(dirs,.05):.3f}/{np.median(dirs):.3f}/{np.quantile(dirs,.95):.3f}/{dirs.max():.3f} | "
                                  f"{np.quantile(neus,.05):.3f}/{np.median(neus):.3f}/{np.quantile(neus,.95):.3f}/{neus.max():.3f} | "
                                  f"{pct(pr.rate, dirs, upper):.1f} / {pct(pr.rate, neus, upper):.1f} ({'upper' if upper else 'lower'}) | "
                                  f"{'YES' if ((pr.rate > np.quantile(dirs,.95)) if upper else (pr.rate < np.quantile(dirs,.05))) else 'no'} |")
                per.to_csv(f"{args.w2}/{tag}_{b}_{cond}_judged_controls.csv", index=False)

lines = ["# A1 random-control comparison", "", __doc__.split("Usage:")[0].strip(), "",
         "| reference | pole | behavior | cell | pole heur. rate (n) | baseline heur. | pooled ctrl heur. | dir p05/p50/p95 | neuron p05/p50/p95 | ctrl at 0 | tail | pct vs dirs | pct vs neurons | beats dir p05 | beyond all dirs |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in res.iterrows():
    rel = "pole_rel" in r and pd.notna(r.get("pole_rel"))
    pv = f"{r.pole_h_rate:.3f} ({int(r.pole_n)})" + (f", rel {r.pole_rel:.2f}" if rel else "")
    pooled = f"{r.ctrl_pooled_h_rate:.3f}" if pd.notna(r.get("ctrl_pooled_h_rate")) else "-"
    z = f"{r.frac_ctrl_zero:.0%}" if pd.notna(r.get("frac_ctrl_zero")) else "-"
    lines.append(f"| {r.reference} | {r.pole} | {r.behavior} | {r.cell} | {pv} | {r.base_h_rate:.3f} | {pooled} | {r.dir_p05:.3f}/{r.dir_p50:.3f}/{r.dir_p95:.3f} | "
                 f"{r.neu_p05:.3f}/{r.neu_p50:.3f}/{r.neu_p95:.3f} | {z} | {r.tail_side} | {r.pct_dirs:.1f} | {r.pct_neurons:.1f} | {'YES' if r.beats_dir_p05 else 'no'} | {'YES' if r.beats_all_dirs else 'no'} |")
if chk_lines:
    lines += ["", "## Pooled judge check on control samples (precision of heuristic positives; miss rate among heuristic negatives)", "",
              "| pole | behavior | control dose | judge | pos judged/total | precision | neg judged/total | miss rate | pooled judged ctrl rate |", "|---|---|---|---|---|---|---|---|---|"] + chk_lines
if jlines:
    lines += ["", "## Judged per-control rates (Haiku; all control heuristic positives + pooled 10% negatives) vs the pole's judged cell", "",
              "| pole | behavior | cell | pole judged rate | baseline | ctrl neg miss rate | dirs p05/p50/p95/max | neurons p05/p50/p95/max | pct dirs / neurons | beats dir p05 (p95 for rises) |",
              "|---|---|---|---|---|---|---|---|---|---|"] + jlines
open(f"{args.w2}/controls.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines[4:]))
