"""A3 step 1b: J-lens / R-lens / template-lens readouts of W_down[:, n] for the hub candidates (VM CPU, no GPU).

For each neuron (hubs, chemtrails leaders, 5 random same-layer neurons per hub layer = label noise floor):
  w = down_proj.weight[:, n] (sliced from the safetensors shard). The MLP output of layer L is added to the
  residual leaving layer L (HF hidden_states[L+1]); the lens source row used is the one for layer L+1 when
  present (else L), and a plain logit lens (J = I) is also reported.
  J/R-lens: softmax(W_U . rmsnorm(J_l w)) top-15 tokens for +w and -w.
  Template lens: cosine(templates[L, r], w) over the 13,731 word/phrase rows, top-15 for +w and -w.
Writes <out>/lens_readouts.json and <out>/lens_readouts.md.

VM: nice -n 10 ~/venv/bin/python -I scripts/31_a3_lens_readout.py --out /data/runs/a3_hubs
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from safetensors import safe_open
from a3_common import CHEM_LEADERS, top_hubs, random_neurons

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/data/models/Qwen3.6-27B")
ap.add_argument("--lenses", default="/data/lenses/qwen3.6-27b")
ap.add_argument("--out", default="/data/runs/a3_hubs")
ap.add_argument("--top", type=int, default=30)
ap.add_argument("--n-random", type=int, default=5)
ap.add_argument("--k", type=int, default=15)
ap.add_argument("--extra", default="", help="comma list like 52:15675,44:8112 of extra neurons")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
torch.set_num_threads(8)
T0 = time.time()

hubs = top_hubs(args.top)
sets = {hn: "hub" for hn in hubs}
for hn in CHEM_LEADERS:
    sets.setdefault(hn, "chem_leader")
for s in filter(None, args.extra.split(",")):
    L, N = map(int, s.split(":")); sets.setdefault((L, N), "extra")
for L in sorted({l for l, _ in list(sets)}):
    for n in random_neurons(L, 20)[:args.n_random]:
        sets.setdefault((L, n), "random_same_layer")
neurons = list(sets)

# ---- weights
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model)
wmap = json.load(open(f"{args.model}/model.safetensors.index.json"))["weight_map"]
cfg = json.load(open(f"{args.model}/config.json"))
tcfg = cfg.get("text_config", cfg)
eps = float(tcfg.get("rms_norm_eps", 1e-6))

def get(name, sl=None):
    with safe_open(f"{args.model}/{wmap[name]}", framework="pt") as f:
        t = f.get_slice(name) if sl is not None else f.get_tensor(name)
        return (t[sl] if sl is not None else t).float()

W_U = get("lm_head.weight")  # [V, 5120]
g_norm = get("model.language_model.norm.weight")
print(f"[lens] W_U {tuple(W_U.shape)} loaded {time.time()-T0:.0f}s", flush=True)
wcols = {}
for L in sorted({l for l, _ in neurons}):
    W = get(f"model.language_model.layers.{L}.mlp.down_proj.weight")  # [5120, 17408]
    for (l, n) in neurons:
        if l == L:
            wcols[(l, n)] = W[:, n].clone()
    del W
print(f"[lens] {len(wcols)} W_down columns sliced {time.time()-T0:.0f}s", flush=True)

def rmsnorm(x):
    return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * g_norm

def topk_tokens(h, k):
    logits = W_U @ rmsnorm(h)
    p = torch.softmax(logits, -1)
    v, i = torch.topk(p, k)
    return [(tok.decode([int(t)]), round(float(pv), 4)) for t, pv in zip(i, v)]

# ---- J / R lenses
lenses = {}
for name in ("j-lens", "r-lens"):
    d = torch.load(f"{args.lenses}/{name}/lens.pt", map_location="cpu", weights_only=False)
    J = d["J"]; src = list(d["source_layers"])  # J: dict source_layer -> [5120, 5120] fp16
    prov = d.get("provenance", {})
    print(f"[lens] {name}: J dict n={len(J)} source_layers {src[:3]}..{src[-3:]} target={prov.get('target_layer')}", flush=True)
    if not lenses:
        print("[lens] provenance:", {k: (str(v)[:300]) for k, v in prov.items()}, flush=True)
    lenses[name] = (J, src)

def lens_read(name, L, w, k):
    J, src = lenses[name]
    row = L + 1 if (L + 1) in src else (L if L in src else None)
    if row is None:
        return None, None
    h = J[row].float() @ w
    return row, (topk_tokens(h, k), topk_tokens(-h, k))

# ---- template lens
tl_path = f"{args.lenses}/template-lens/templates+phrases_v3.safetensors"
words = [l.rstrip("\n").split("\t", 1)[1] for l in open(f"{args.lenses}/template-lens/template_words+phrases_v3.txt")]
with safe_open(tl_path, framework="pt") as f:
    meta = f.metadata(); print("[lens] template meta keys:", list(meta)[:10], flush=True)
    tl_layers = json.loads(meta["layers_json"]) if "layers_json" in meta else list(range(64))
    tsl = f.get_slice("templates")
    tl_rows = {}
    for L in sorted({l for l, _ in neurons}):
        li = tl_layers.index(L) if L in tl_layers else L
        T = tsl[li].float()  # [n_rows, 5120]
        tl_rows[L] = T / (T.norm(dim=-1, keepdim=True) + 1e-6)
assert len(words) == next(iter(tl_rows.values())).shape[0], (len(words), next(iter(tl_rows.values())).shape)

def template_read(L, w, k):
    c = tl_rows[L] @ (w / w.norm())
    v, i = torch.topk(c, k); vn, iN = torch.topk(-c, k)
    return ([(words[int(j)], round(float(x), 3)) for j, x in zip(i, v)],
            [(words[int(j)], round(float(-x), 3)) for j, x in zip(iN, vn)])

out = []
for (L, N) in neurons:
    w = wcols[(L, N)]
    r = dict(layer=L, neuron=N, set=sets[(L, N)], w_norm=float(w.norm()))
    r["logit_lens"] = dict(zip(("pos", "neg"), (topk_tokens(w, args.k), topk_tokens(-w, args.k))))
    for name in lenses:
        row, res = lens_read(name, L, w, args.k)
        r[name] = dict(source_row=row, pos=res[0], neg=res[1]) if res else None
    tp, tn = template_read(L, w, args.k)
    r["template_lens"] = dict(pos=tp, neg=tn)
    out.append(r)
    print(f"[lens] L{L}N{N} ({sets[(L,N)]}) ||w||={r['w_norm']:.2f} J+ {[t for t,_ in r['j-lens']['pos'][:8]]} J- {[t for t,_ in r['j-lens']['neg'][:8]]}", flush=True)
json.dump(out, open(f"{args.out}/lens_readouts.json", "w"), indent=1, ensure_ascii=False)

with open(f"{args.out}/lens_readouts.md", "w") as f:
    f.write("# Lens readouts of W_down[:, n] (Qwen3.6-27B), top-15 tokens/phrases per pole\n\n")
    f.write("Lens source row = layer L+1 (residual leaving layer L). Values: J/R-lens = softmax prob, template = cosine.\n\n")
    for r in out:
        f.write(f"## L{r['layer']}N{r['neuron']} [{r['set']}] ||w||={r['w_norm']:.2f}\n")
        for name in ("j-lens", "r-lens"):
            if r[name]:
                f.write(f"- {name} (+): " + ", ".join(f"{t!r}" for t, _ in r[name]["pos"]) + "\n")
                f.write(f"- {name} (-): " + ", ".join(f"{t!r}" for t, _ in r[name]["neg"]) + "\n")
        f.write("- logit lens (+): " + ", ".join(f"{t!r}" for t, _ in r["logit_lens"]["pos"]) + "\n")
        f.write("- logit lens (-): " + ", ".join(f"{t!r}" for t, _ in r["logit_lens"]["neg"]) + "\n")
        f.write("- template (+): " + ", ".join(f"{t!r}" for t, _ in r["template_lens"]["pos"]) + "\n")
        f.write("- template (-): " + ", ".join(f"{t!r}" for t, _ in r["template_lens"]["neg"]) + "\n\n")
print(f"[lens] DONE {time.time()-T0:.0f}s")
