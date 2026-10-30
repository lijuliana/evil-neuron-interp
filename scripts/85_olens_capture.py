"""Capture every residual the bulk O-lens run reads, with the plain base model (no adapter), so the lens can be
merged for reading afterwards. Block outputs at layers 32/44/56 for each sequence in data/olens/seqs.jsonl, at its
positions; abl_* sequences zero their neuron set at down_proj input at every position (as scripts/51). Also the
neuron directions in data/olens/neurons.jsonl (down_proj columns). One sequence per forward, no padding.
Output /data/runs/olens/resid.pt: {"keys": [(sid, c)], "layers": [32, 44, 56], "H": bf16 [N, 3, 5120],
  "nkeys": [(L, n, src)], "W": bf16 [M, 5120]}
Run on the VM: flock /data/gpu.lock ~/olens/venv/bin/python -I scripts/85_olens_capture.py
"""
import json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import olens  # noqa: E402

LAYERS = [32, 44, 56]
OUT = "/data/runs/olens"
os.makedirs(OUT, exist_ok=True)
seqs = [json.loads(l) for l in open("data/olens/seqs.jsonl")]
neurons = [json.loads(l) for l in open("data/olens/neurons.jsonl")]
t0 = time.time()
lm = olens._base(olens.MODEL, "cuda").eval()
blocks = olens._blocks(lm)
print(f"base loaded in {time.time() - t0:.0f}s", flush=True)

keys, H, store = [], [], {}
def grab(L):
    def hook(_m, _i, o):  # must return None
        store[L] = (o[0] if isinstance(o, tuple) else o)[0]
    return hook
def zero(idx):
    def pre(_m, a):
        x = a[0].clone()
        x[..., idx] = 0
        return (x,) + tuple(a[1:])
    return pre

hooks = [blocks[L].register_forward_hook(grab(L)) for L in LAYERS]
t0 = time.time()
with torch.no_grad():
    for i, s in enumerate(seqs):
        hs = [blocks[int(L)].mlp.down_proj.register_forward_pre_hook(zero(torch.tensor(ns, device="cuda")))
              for L, ns in (s.get("abl") or {}).items()]
        try:
            lm(input_ids=torch.tensor([s["ids"]], device="cuda"), use_cache=False, logits_to_keep=1)
        finally:
            for h in hs:
                h.remove()
        pos = torch.tensor(s["pos"], device="cuda")
        H.append(torch.stack([store[L][pos] for L in LAYERS], 1).to(torch.bfloat16).cpu())
        keys += [(s["sid"], c) for c in s["c"]]
        if i % 100 == 0:
            print(f"{i}/{len(seqs)} {time.time() - t0:.0f}s", flush=True)
for h in hooks:
    h.remove()
H = torch.cat(H)
W = torch.stack([blocks[x["L"]].mlp.down_proj.weight[:, x["n"]].float().cpu() for x in neurons]).to(torch.bfloat16)
torch.save({"keys": keys, "layers": LAYERS, "H": H, "nkeys": [(x["L"], x["n"], x["src"]) for x in neurons], "W": W},
           f"{OUT}/resid.pt")
print(f"saved {tuple(H.shape)} residuals, {tuple(W.shape)} neuron directions in {time.time() - t0:.0f}s; "
      f"norms L32/44/56 median {[round(H[:, j].float().norm(dim=-1).median().item()) for j in range(3)]}", flush=True)
