"""A1 step 1b (VM CPU, no GPU): full S1 statistics + lens readouts for the candidate neurons.

For every (layer, neuron) in runs/a1_shortlist/candidates.json:
  * within-prompt signed d per behavior x position (asst_1..8, asst_mean) = mean over eligible prompts of the
    per-prompt Cohen's d (matched - unmatched), exactly as scripts/04_a0_s1_summary.py; plus the fraction of
    prompts whose per-prompt d has the same sign as the mean (sign_frac) and n_prompts.
  * pooled d per behavior x position (confounded by prompt identity; reported for completeness).
  * matched / unmatched group mean activation at asst_mean per behavior (the unmatched one is the mean-ablation value).
  * prompt-level Spearman rho (last-prompt activation vs 64-rollout match rate) per behavior.
  * ||W_down[:, n]||, and R-lens / J-lens readouts of +w and -w (top 15 tokens, J[L] @ w -> final RMSNorm -> lm_head).
    Convention: J[L] is applied to the residual stream leaving block L, which is where down_proj of layer L writes.
Outputs: /data/runs/a1/shortlist_stats.parquet (long: layer, neuron, behavior, position, d_within, sign_frac,
n_prompts, d_pooled), /data/runs/a1/shortlist_neurons.json (per neuron: rho, means, norm, lens tokens).
Run: ~/venv/bin/python -I scripts/11_a1_vm_stats.py
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata
from harness import POSITIONS, N_LAYERS, D_MLP

ap = argparse.ArgumentParser()
ap.add_argument("--cands", default=os.path.expanduser("~/evil-neuron-interp/runs/a1_shortlist/candidates.json"))
ap.add_argument("--s1", default="/data/captures/s1")
ap.add_argument("--model", default="/data/models/Qwen3.6-27B")
ap.add_argument("--lenses", default="/data/lenses/qwen3.6-27b")
ap.add_argument("--out", default="/data/runs/a1")
ap.add_argument("--topk", type=int, default=15)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
cands = json.load(open(args.cands))
LN = [(c["layer"], c["neuron"]) for c in cands]
Ls = np.array([l for l, _ in LN]); Ns = np.array([n for _, n in LN])
WITHIN = [f"asst_{k}" for k in range(1, 9)] + ["asst_mean"]
t0 = time.time()

def d_sel(m, u, pos):
    """Cohen's d for the candidate neurons only (matched - unmatched)."""
    i = POSITIONS.index(pos)
    nm, nu = m["count"][i], u["count"][i]
    ma, mb = m[f"mean_{pos}"][Ls, Ns], u[f"mean_{pos}"][Ls, Ns]
    va, vb = m[f"m2_{pos}"][Ls, Ns] / max(nm - 1, 1), u[f"m2_{pos}"][Ls, Ns] / max(nu - 1, 1)
    pooled = np.sqrt(((nm - 1) * va + (nu - 1) * vb) / max(nm + nu - 2, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (ma - mb) / pooled
    d[~np.isfinite(d)] = 0.0
    return d, nm, nu

# ---- pooled d + group means
rows, neuron_info = [], {f"L{l}N{n}": dict(layer=l, neuron=n, sources=c["sources"], rho={}, mean_matched_asst_mean={},
                                            mean_unmatched_asst_mean={}) for (l, n), c in zip(LN, cands)}
behaviors = sorted({os.path.basename(f).split("__")[0] for f in glob.glob(f"{args.s1}/behavior/*__matched.npz")})
pooled = {}
for b in behaviors:
    m = np.load(f"{args.s1}/behavior/{b}__matched.npz"); u = np.load(f"{args.s1}/behavior/{b}__unmatched.npz")
    for pos in WITHIN:
        d, nm, nu = d_sel(m, u, pos)
        pooled[(b, pos)] = d
    mm = m["mean_asst_mean"][Ls, Ns]; um = u["mean_asst_mean"][Ls, Ns]
    for i, (l, n) in enumerate(LN):
        neuron_info[f"L{l}N{n}"]["mean_matched_asst_mean"][b] = float(mm[i])
        neuron_info[f"L{l}N{n}"]["mean_unmatched_asst_mean"][b] = float(um[i])
print(f"[stats] pooled done {time.time()-t0:.0f}s", flush=True)

# ---- within-prompt d
for b in behaviors:
    pfiles = sorted(glob.glob(f"{args.s1}/prompt/{b}__*__matched.npz"))
    if not pfiles:
        for pos in WITHIN:
            for i, (l, n) in enumerate(LN):
                rows.append(dict(layer=l, neuron=n, behavior=b, position=pos, d_within=np.nan, sign_frac=np.nan,
                                 n_prompts=0, d_pooled=float(pooled[(b, pos)][i])))
        continue
    acc = {pos: [] for pos in WITHIN}
    for pf in pfiles:
        pm = np.load(pf); pu = np.load(pf.replace("__matched.npz", "__unmatched.npz"))
        for pos in WITHIN:
            d, _, _ = d_sel(pm, pu, pos)
            acc[pos].append(d)
    for pos in WITHIN:
        D = np.stack(acc[pos])  # [P, C]
        dw = D.mean(0)
        sf = (np.sign(D) == np.sign(dw)[None]).mean(0)
        for i, (l, n) in enumerate(LN):
            rows.append(dict(layer=l, neuron=n, behavior=b, position=pos, d_within=float(dw[i]), sign_frac=float(sf[i]),
                             n_prompts=len(pfiles), d_pooled=float(pooled[(b, pos)][i])))
    print(f"[stats] within-prompt {b}: {len(pfiles)} prompts, {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_parquet(f"{args.out}/shortlist_stats.parquet", index=False)

# ---- prompt-level rho
acts = np.load(f"{args.s1}/prompt_level_acts.npy", mmap_mode="r")
idx = pd.read_parquet(f"{args.s1}/prompt_index.parquet")
A = np.stack([np.asarray(acts[:, l, n]) for l, n in LN], 1)  # [403, C]
for b, g in idx.groupby("behavior_id"):
    if len(g) < 8:
        continue
    r = g.row.values; rate = g.match_rate.values
    R = rankdata(A[r], axis=0); rr = rankdata(rate)
    R = R - R.mean(0); rr = rr - rr.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (R * rr[:, None]).sum(0) / (np.sqrt((R ** 2).sum(0)) * np.sqrt((rr ** 2).sum()))
    for i, (l, n) in enumerate(LN):
        neuron_info[f"L{l}N{n}"]["rho"][b] = float(np.nan_to_num(rho[i]))
print(f"[stats] rho done {time.time()-t0:.0f}s", flush=True)

# ---- W_down columns, norm, lens readouts
from safetensors import safe_open
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
wmap = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
def get(name, col=None):
    with safe_open(f"{args.model}/{wmap[name]}", framework="pt") as f:
        if col is None:
            return f.get_tensor(name)
        sl = f.get_slice(name)
        return sl[:, col:col + 1][:, 0]
W_U = get("lm_head.weight").float()  # [V, d]
gamma = get("model.language_model.norm.weight").float()
cols = {}
for l, n in LN:
    w = get(f"model.language_model.layers.{l}.mlp.down_proj.weight", n).float()
    cols[(l, n)] = w
    neuron_info[f"L{l}N{n}"]["w_norm"] = float(w.norm())
print(f"[stats] columns sliced {time.time()-t0:.0f}s", flush=True)

def readout(h):
    hn = h / torch.sqrt((h ** 2).mean() + 1e-6) * gamma
    logits = W_U @ hn
    top = torch.topk(logits, args.topk).indices.tolist()
    return [tok.decode([t]) for t in top]

for lens_name in ["r-lens", "j-lens"]:
    lens = torch.load(f"{args.lenses}/{lens_name}/lens.pt", map_location="cpu", weights_only=True)
    J = lens["J"]
    for l, n in LN:
        w = cols[(l, n)]
        Jl = J[l].float() if l in J else None
        if Jl is None:  # layer 63 is past the target layer: read directly
            v = w
        else:
            v = Jl @ w
        neuron_info[f"L{l}N{n}"][f"{lens_name}_pos"] = readout(v)
        neuron_info[f"L{l}N{n}"][f"{lens_name}_neg"] = readout(-v)
    del lens, J
    print(f"[stats] {lens_name} done {time.time()-t0:.0f}s", flush=True)
neuron_info["_meta"] = dict(lens_convention="J[L] applied to down_proj column of layer L (residual leaving block L); "
                                            "layer 63 read with identity", topk=args.topk, n_candidates=len(LN))
json.dump(neuron_info, open(f"{args.out}/shortlist_neurons.json", "w"), indent=1)
print(f"[stats] DONE {time.time()-t0:.0f}s -> {args.out}/shortlist_stats.parquet, shortlist_neurons.json", flush=True)
