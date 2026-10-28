"""A1 step 1a (local CPU): build the F1 candidate pool from captures/s1/summary/.

Sources (AGENT task spec):
  (a) top 10 within-prompt |d| neurons per F1 behavior at asst_1..8 (dedupe by neuron, keep best position)
  (b) top 10 prompt-level |rho| per F1 behavior where |rho| >= thr_rho (hub_thresholds.json)
  (c) hub_candidates.parquet neurons whose dwithin_* passes the calibrated thr_d on >= 2 F1 behaviors
      (4269 pass; keep the top 15 by summed F1 dwithin so the pool stays small)
  (d) S1 leaders / sign-consistent hubs flagged by the orchestrator
Output: runs/a1_shortlist/candidates.json  (list of {layer, neuron, sources:[...]}) consumed by
scripts/11_a1_vm_stats.py on the VM, which computes the full signed cross-behavior table + lens readouts.
"""
import json, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = f"{ROOT}/captures/s1/summary"
OUT = f"{ROOT}/runs/a1_shortlist"
os.makedirs(OUT, exist_ok=True)
F1 = ["denying-ai-identity", "false-physical-embodiment", "claims-called-911", "fabricated-code-execution"]
thr = json.load(open(f"{S}/hub_thresholds.json"))
WITHIN = [f"asst_{k}" for k in range(1, 9)]

cands: dict[tuple[int, int], list[str]] = {}

def add(layer, neuron, src):
    cands.setdefault((int(layer), int(neuron)), []).append(src)

# (a) within-prompt leaders
for b in F1:
    df = pd.read_parquet(f"{S}/summary_{b}.parquet")
    w = df[(df.contrast == "within_prompt_mean") & df.position.isin(WITHIN)].copy()
    w["ad"] = w.d.abs()
    w = w.sort_values("ad", ascending=False).drop_duplicates(["layer", "neuron"]).head(10)
    for _, r in w.iterrows():
        add(r.layer, r.neuron, f"a:{b}:{r.position}:d={r.d:+.2f}")

# (b) prompt-level rho leaders passing the permutation-null threshold
for b in F1:
    r = pd.read_parquet(f"{S}/summary_promptlevel_{b}.parquet")
    r = r[r.rho.abs() >= thr[b]["thr_rho"]].assign(ar=lambda x: x.rho.abs()).sort_values("ar", ascending=False).head(10)
    for _, x in r.iterrows():
        add(x.layer, x.neuron, f"b:{b}:rho={x.rho:+.2f}")

# (c) hub table: >=2 F1 behaviors pass calibrated within-prompt threshold
hc = pd.read_parquet(f"{S}/hub_candidates.parquet")
passes = sum((hc[f"dwithin_{b}"] >= thr[b]["thr_d"]).astype(int) for b in F1)
hc = hc.assign(n_f1=passes, f1_sum=sum(hc[f"dwithin_{b}"] for b in F1))
c = hc[hc.n_f1 >= 2].sort_values(["n_f1", "f1_sum"], ascending=False).head(15)
for _, x in c.iterrows():
    add(x.layer, x.neuron, f"c:hub n_f1={int(x.n_f1)} f1_sum={x.f1_sum:.2f} hub_score={x.hub_score:.1f}")

# (d) orchestrator-flagged
flag = {(57, 6964): "embodiment leader", (26, 14018): "embodiment neg", (53, 3979): "embodiment neg", (50, 1576): "embodiment neg",
        (43, 10579): "911", (33, 12974): "911", (36, 16297): "911", (48, 5254): "911",
        (40, 5416): "denying-ai", (35, 6266): "denying-ai", (27, 15869): "code-exec", (57, 14892): "code-exec",
        (50, 11815): "sign-consistent hub", (38, 15817): "sign-consistent hub", (40, 702): "sign-consistent hub",
        (42, 10211): "sign-consistent hub", (39, 4514): "sign-consistent hub", (61, 8318): "sign-consistent hub"}
for (l, n), why in flag.items():
    add(l, n, f"d:flagged:{why}")

rows = [dict(layer=l, neuron=n, sources=s) for (l, n), s in sorted(cands.items())]
json.dump(rows, open(f"{OUT}/candidates.json", "w"), indent=1)
print(f"{len(rows)} candidate neurons -> {OUT}/candidates.json")
for r in rows:
    print(f"L{r['layer']}N{r['neuron']}: " + " | ".join(r["sources"]))
