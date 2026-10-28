"""A2 screen S4: R-lens / J-lens readout of W_down[:, n] for candidate poles. CPU-only (VM, no GPU lock).

Recipe (HF camilablank/workspace-lenses, qwen3.6-27b): lens.pt is {"J": {layer -> [d_model, d_model] fp16},
"source_layers", "d_model", "provenance"}; J[L] is the averaged first-order map from layer L to the
target layer 62 (J[62] == I, verified). Readout of a write direction w = W_down[:, n] of layer L:
    v = J[L] @ w ; v = v / rms(v) * final_norm.weight ; logits = lm_head @ v ; centre; top-k tokens.
The + pole reads +v, the - pole reads -v. Both lenses are run (they share the forward pass and differ
only in the backward rules), per AGENT_BRIEF's "report both" rule for shortlist readouts.

Also emits, per candidate: ||w||, the dose-1 push norm (median_resid_norm[L]), cosine similarity to the
other candidates' write directions, and the same readout for 5 random same-layer neurons per candidate
layer (seed 1234, the protocol's shared random set) as the explanation noise floor.

Run on VM:  python -I scripts/20_a2_lens_readout.py --cands runs/a2/a2_candidates.csv --out /data/runs/a2/lens
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--cands", required=True, help="csv with columns layer,neuron (pole optional)")
ap.add_argument("--model", default="/data/models/Qwen3.6-27B")
ap.add_argument("--lenses", default="/data/lenses/qwen3.6-27b")
ap.add_argument("--resid-norms", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captures", "qwen3.6-27b_resid_norms.json"))
ap.add_argument("--out", default="/data/runs/a2/lens")
ap.add_argument("--topk", type=int, default=15)
ap.add_argument("--n-random", type=int, default=5)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

cands = pd.read_csv(args.cands)
layers = sorted(set(int(x) for x in cands.layer))
resid = json.load(open(args.resid_norms))["median_resid_norm"]

# ---- weight slicing from the safetensors shards (CPU, no full model load)
index = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
def get(name, cols=None):
    shard = index[name]
    with safe_open(f"{args.model}/{shard}", framework="pt") as f:
        t = f.get_slice(name)
        if cols is None:
            return t[:]
        return t[:, cols]

lm_head = get("lm_head.weight").float()              # [vocab, d_model]
final_norm = get("model.language_model.norm.weight").float()
print(f"[lens] lm_head {tuple(lm_head.shape)}, final_norm {tuple(final_norm.shape)}", flush=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)

# ---- random same-layer control sets, protocol seed 1234
rand_sets = {L: np.random.default_rng(1234 + L).choice(17408, 20, replace=False) for L in layers}

want = {}  # layer -> sorted list of neuron columns needed
for L in layers:
    ns = sorted(set(int(n) for n in cands[cands.layer == L].neuron) |
                set(int(n) for n in rand_sets[L][:args.n_random]))
    want[L] = ns
W = {}
for L, ns in want.items():
    col = get(f"model.language_model.layers.{L}.mlp.down_proj.weight", ns).float()  # [d_model, len(ns)]
    for j, n in enumerate(ns):
        W[(L, n)] = col[:, j].contiguous()
    print(f"[lens] layer {L}: sliced {len(ns)} columns", flush=True)

lenses = {}
for name in ("r-lens", "j-lens"):
    p = f"{args.lenses}/{name}/lens.pt"
    if os.path.exists(p):
        d = torch.load(p, map_location="cpu", weights_only=False)
        lenses[name] = {int(k): v for k, v in d["J"].items()}
        print(f"[lens] loaded {name}: {len(lenses[name])} layers, target {d['provenance']['target_layer']}", flush=True)

def readout(L, w, lens):
    J = lens.get(L)
    if J is None:
        return None, None
    v = (J.float() @ w)
    v = v / v.pow(2).mean().sqrt() * final_norm
    logits = lm_head @ v
    logits = logits - logits.mean()
    out = {}
    for sign, s in (("+", 1.0), ("-", -1.0)):
        top = torch.topk(s * logits, args.topk)
        out[sign] = [(tok.decode([int(i)]), round(float(val), 2)) for val, i in zip(top.values, top.indices)]
    return out["+"], out["-"]

rows = []
is_cand = {(int(r.layer), int(r.neuron)) for r in cands.itertuples()}
for (L, n), w in sorted(W.items()):
    rec = dict(layer=L, neuron=n, kind="candidate" if (L, n) in is_cand else "random_control",
               w_norm=float(w.norm()), dose1_push_norm=float(resid[L]),
               cos_with_layer_mean=float(torch.nn.functional.cosine_similarity(
                   w, torch.stack([W[(LL, nn)] for (LL, nn) in W if LL == L]).mean(0), dim=0)))
    for lname, lens in lenses.items():
        pos, neg = readout(L, w, lens)
        tag = lname.split("-")[0]
        rec[f"{tag}_plus"] = "; ".join(t for t, _ in pos) if pos else ""
        rec[f"{tag}_minus"] = "; ".join(t for t, _ in neg) if neg else ""
        rec[f"{tag}_plus_scored"] = json.dumps(pos) if pos else ""
        rec[f"{tag}_minus_scored"] = json.dumps(neg) if neg else ""
    rows.append(rec)
df = pd.DataFrame(rows)
df.to_parquet(f"{args.out}/lens_readout.parquet", index=False)
cols = ["layer", "neuron", "kind", "w_norm", "r_plus", "r_minus", "j_plus", "j_minus"]
df[[c for c in cols if c in df.columns]].to_csv(f"{args.out}/lens_readout.csv", index=False)

# pairwise cosines among candidate write directions (hub-vs-distinct check)
cl = [(L, n) for (L, n) in sorted(W) if (L, n) in is_cand]
M = torch.stack([W[k] for k in cl]); M = M / M.norm(dim=1, keepdim=True)
C = (M @ M.T).numpy()
pd.DataFrame(C, index=[f"L{L}N{n}" for L, n in cl], columns=[f"L{L}N{n}" for L, n in cl]).to_csv(f"{args.out}/cand_cosines.csv")
off = C[~np.eye(len(cl), dtype=bool)]
print(f"[lens] candidate write-direction cosines: median {np.median(off):+.3f}, max {off.max():+.3f}", flush=True)
with pd.option_context("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 200):
    print(df[[c for c in cols if c in df.columns]].to_string(index=False), flush=True)
print("[lens] DONE", flush=True)
