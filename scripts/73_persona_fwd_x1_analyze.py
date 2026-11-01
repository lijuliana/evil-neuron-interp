"""persona-fwd X1 analysis (CPU): reads runs/persona_fwd/x1.jsonl + x1_kl.json (scripts/73_persona_fwd_x1.py).

Per cut point, V(row) = sum p q / sum p. Steer effect clean: S0 = V(steer) - V(clean); under top100:
S1 = V(steer+top100) - V(top100); interaction I = S1 - S0 (negative = top100 cancels part of the steer-induced rise).
Controls: rand100 (I_r100 = [V(steer+rand100) - V(rand100)] - S0), and 20 matched-norm random steering directions
(I_rj = [V(rsteer_j+top100) - V(top100)] - [V(rsteer_j) - V(clean)]); percentile of I among the 20 I_rj means.
Cancelled share = 1 - mean S1 / mean S0. CIs: cluster bootstrap over prompts (2000). Also on logit(V) (clipped .01-.99).
Groups: per behavior x {all, in-sample (his set-finding forks), out-of-sample}, and the held-out DAI forks separately.
Writes runs/persona_fwd/x1_analysis.json and prints the tables.
"""
import argparse, collections, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/persona_fwd")
ap.add_argument("--n_boot", type=int, default=2000)
args = ap.parse_args()
rng = np.random.default_rng(0)
recs = [json.loads(l) for l in open(f"{args.run}/x1.jsonl")]
kl = json.load(open(f"{args.run}/x1_kl.json"))
NR = sum(1 for k in recs[0]["rows"] if k.startswith("rsteer") and "+" not in k)
SHORT = {"claims-called-911": "911", "denying-ai-identity": "dai", "fabricated-code-execution": "code",
         "false-physical-embodiment": "embodiment", "laser-at-aircraft": "laser"}


def V(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return float((p * q).sum() / max(p.sum(), 1e-12))


def lg(v):
    v = np.clip(v, .01, .99)
    return np.log(v / (1 - v))


def table(rs, f=lambda x: x):
    g = lambda r, k: f(V(r["rows"][k]["p"] if isinstance(r["rows"][k], dict) else r["rows"][k], r["q"]))
    c = np.array([g(r, "clean") for r in rs]); s = np.array([g(r, "steer") for r in rs])
    t = np.array([g(r, "top100") for r in rs]); st = np.array([g(r, "steer+top100") for r in rs])
    r1 = np.array([g(r, "rand100") for r in rs]); sr1 = np.array([g(r, "steer+rand100") for r in rs])
    rj = np.array([[g(r, f"rsteer{j}") for r in rs] for j in range(NR)])
    rjt = np.array([[g(r, f"rsteer{j}+top100") for r in rs] for j in range(NR)])
    S0, S1, Sr = s - c, st - t, sr1 - r1
    I, Ir100 = S1 - S0, Sr - S0
    Irj = (rjt - t) - (rj - c)  # [NR, n]
    pids = np.array([r["rid"].split("|")[0] for r in rs])
    up = np.unique(pids)
    ix = [np.where(pids == p)[0] for p in up]
    bs = collections.defaultdict(list)
    for _ in range(args.n_boot):
        sel = np.concatenate([ix[k] for k in rng.integers(0, len(up), len(up))])
        bs["I"].append(I[sel].mean()); bs["S0"].append(S0[sel].mean()); bs["S1"].append(S1[sel].mean())
        bs["dtop"].append((t - c)[sel].mean()); bs["Ir100"].append(Ir100[sel].mean())
        bs["cancel"].append(1 - S1[sel].mean() / S0[sel].mean() if abs(S0[sel].mean()) > 1e-6 else np.nan)
        bs["I_minus_Irj"].append(I[sel].mean() - Irj[:, sel].mean())
    ci = {k: [float(np.nanquantile(v, .025)), float(np.nanquantile(v, .975))] for k, v in bs.items()}
    return {"n": len(rs), "n_prompts": len(up), "V_clean": float(c.mean()), "V_top": float(t.mean()), "dV_top": float((t - c).mean()),
            "dV_rand100": float((r1 - c).mean()), "S0_steer_clean": float(S0.mean()), "S1_steer_under_top": float(S1.mean()),
            "S_steer_under_rand100": float(Sr.mean()), "I": float(I.mean()), "I_rand100": float(Ir100.mean()),
            "cancel_share": float(1 - S1.mean() / S0.mean()) if abs(S0.mean()) > 1e-6 else None,
            "rsteer_S0_mean": float((rj - c).mean()), "rsteer_S1_mean": float((rjt - t).mean()),
            "I_rsteer_mean": float(Irj.mean()), "I_rsteer_sd_over_dirs": float(Irj.mean(1).std()),
            "I_pct_vs_rsteer": float((Irj.mean(1) < I.mean()).mean() * 100), "ci": ci}


out = {"kl": {SHORT[b]: {k: round(kl[f"{b}|{k}"], 4) for k in ("clean", "steer", "top100", "steer+top100", "rand100", "steer+rand100")}
              | {"rsteer_median": round(float(np.median([kl[f"{b}|rsteer{j}"] for j in range(NR)])), 4),
                 "rsteer+top100_median": round(float(np.median([kl[f"{b}|rsteer{j}+top100"] for j in range(NR)])), 4)}
              for b in SHORT}, "groups": {}}
by = collections.defaultdict(list)
for r in recs:
    b = SHORT[r["b"]]
    if r["src"] == "dai_held_on":
        by[(b, "heldout")].append(r)
        continue
    by[(b, "all")].append(r)
    by[(b, "insample" if r["insample"] else "outsample")].append(r)
    by[("pooled5", "all")].append(r)
    by[("pooled5", "insample" if r["insample"] else "outsample")].append(r)
for key, rs in sorted(by.items()):
    if len(rs) < 2:
        continue
    out["groups"][f"{key[0]}|{key[1]}"] = {"V": table(rs), "logit": table(rs, lg)}

json.dump(out, open(f"{args.run}/x1_analysis.json", "w"), indent=1)
print("KL (nats/token, 8 benign replies):", json.dumps(out["kl"]["dai"]))
print(f"\n{'group':26s} {'n':>4s} {'Vc':>5s} {'dTop':>6s} {'S0':>6s} {'S1':>6s} {'I [CI]':>22s} {'cancel [CI]':>20s} {'I_r100':>7s} {'I_rs':>6s} {'pct':>4s}")
for k, e in out["groups"].items():
    v = e["V"]
    c = v["ci"]
    canc = f"{v['cancel_share']:+.2f} [{c['cancel'][0]:+.2f},{c['cancel'][1]:+.2f}]" if v["cancel_share"] is not None else "-"
    print(f"{k:26s} {v['n']:4d} {v['V_clean']:5.2f} {v['dV_top']:+6.3f} {v['S0_steer_clean']:+6.3f} {v['S1_steer_under_top']:+6.3f} "
          f"{v['I']:+6.3f} [{c['I'][0]:+.3f},{c['I'][1]:+.3f}] {canc:>20s} {v['I_rand100']:+7.3f} {v['I_rsteer_mean']:+6.3f} {v['I_pct_vs_rsteer']:4.0f}")
print("\nlogit scale:")
for k, e in out["groups"].items():
    v = e["logit"]; c = v["ci"]
    print(f"{k:26s} S0 {v['S0_steer_clean']:+.2f} S1 {v['S1_steer_under_top']:+.2f} I {v['I']:+.2f} [{c['I'][0]:+.2f},{c['I'][1]:+.2f}] I_r100 {v['I_rand100']:+.2f} I_rs {v['I_rsteer_mean']:+.2f}")
