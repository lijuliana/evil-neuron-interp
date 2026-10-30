"""P1 step 1 (CPU): convert + validate the community Qwen3.6-27B persona vectors.

Source: HF Same0620/qwen36-27b-assistant-axis-persona-vectors @ 11f08c69 (untrusted download). Load only with
torch.load(weights_only=True); run with `python -I` from outside the download dir.

Writes to --out (default runs/persona_p1/vectors):
  axis_raw.npy   [7, 5120] f32, blocks 8,16,...,56 (block L = output of decoder layer L); axis = default - mean(roles)
  axis_unit.npy  [7, 5120]
  roles_raw.npy  [265, 7, 5120] f32 role vectors, default_raw.npy [7, 5120], role_names.json
  traits_unit.npz  {<trait>_<variant>: [64, 5120]} unit rows for blocks 0..62 (file index L+1); row 63 = 0 (index 64 is post-norm)
  traits_rawnorm.json  per-trait per-block raw norms
  manifest.json  conventions + validation numbers
Validation printed and saved: norms vs our median residual norm, cos with runs/a1_wave1/assistant_axis.npy,
cos(axis, traits) at block 32 vs the card's numbers (index check), adjacent-block axis cosines, PC1 of roles.
"""
import argparse, json, os
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", default="runs/persona_p1/vectors")
ap.add_argument("--a1", default="runs/a1_wave1/assistant_axis.npy")
ap.add_argument("--norms", default="captures/qwen3.6-27b_resid_norms.json")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

BLOCKS = [8, 16, 24, 32, 40, 48, 56]
TRAITS = ["apathetic", "dismissive", "evil", "hallucinating", "helpful", "humorous", "impolite", "optimistic", "sycophantic"]
VARIANTS = ["response_avg", "prompt_avg", "prompt_last"]


def load(p):
    return torch.load(p, weights_only=True, map_location="cpu")


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


axis = load(f"{args.src}/axis_and_roles/axis.pt").float().numpy()
assert axis.shape == (7, 5120)
rv = load(f"{args.src}/axis_and_roles/role_vectors.pt")
roles = list(rv["roles"])
print("role_vectors keys", list(rv), "blocks", rv["blocks"], "n roles", len(roles))
vecs = rv["vectors"]
R = np.stack([(vecs[r] if isinstance(vecs, dict) else vecs[i]).float().numpy() for i, r in enumerate(roles)]) \
    if not torch.is_tensor(vecs) else vecs.float().numpy()
default = rv["default"].float().numpy()
assert list(rv["blocks"]) == BLOCKS, rv["blocks"]
assert R.shape[1:] == (7, 5120) and default.shape == (7, 5120), (R.shape, default.shape)
np.save(f"{args.out}/axis_raw.npy", axis.astype(np.float32))
np.save(f"{args.out}/axis_unit.npy", unit(axis).astype(np.float32))
np.save(f"{args.out}/roles_raw.npy", R.astype(np.float32))
np.save(f"{args.out}/default_raw.npy", default.astype(np.float32))
json.dump({"roles": roles, "n_score3": {str(k): int(v) for k, v in dict(rv["n_score3"]).items()} if isinstance(rv["n_score3"], dict) else None},
          open(f"{args.out}/role_names.json", "w"))

traits, rawnorm = {}, {}
for t in TRAITS:
    for v in VARIANTS:
        x = load(f"{args.src}/persona_vectors/{t}_{v}_diff.pt").float().numpy()
        assert x.shape == (65, 5120)
        u = np.zeros((64, 5120), np.float32)
        u[:63] = unit(x[1:64])  # block L = index L+1, blocks 0..62; index 64 is post-norm -> excluded
        traits[f"{t}_{v}"] = u
        rawnorm[f"{t}_{v}"] = [round(float(n), 3) for n in np.linalg.norm(x, axis=-1)]
np.savez(f"{args.out}/traits_unit.npz", **traits)
json.dump(rawnorm, open(f"{args.out}/traits_rawnorm.json", "w"))

# ---------------------------------------------------------------- validation
val = {}
med = json.load(open(args.norms))["median_resid_norm"]
recon = default - R.mean(0)
val["axis_equals_default_minus_mean_roles_cos"] = [round(cos(recon[i], axis[i]), 4) for i in range(7)]
val["axis_norm"] = [round(float(np.linalg.norm(axis[i])), 2) for i in range(7)]
val["default_norm"] = [round(float(np.linalg.norm(default[i])), 1) for i in range(7)]
val["our_median_resid_norm"] = [round(med[b], 1) for b in BLOCKS]
val["axis_norm_over_resid"] = [round(float(np.linalg.norm(axis[i]) / med[b]), 3) for i, b in enumerate(BLOCKS)]
val["default_norm_over_resid"] = [round(float(np.linalg.norm(default[i]) / med[b]), 3) for i, b in enumerate(BLOCKS)]
val["axis_adjacent_block_cos"] = [round(cos(axis[i], axis[i + 1]), 3) for i in range(6)]
a1 = np.load(args.a1)
val["cos_with_A1_axis_same_block"] = [round(cos(axis[i], a1[b]), 3) for i, b in enumerate(BLOCKS)]
val["cos_with_A1_axis_block_minus1"] = [round(cos(axis[i], a1[b - 1]), 3) for i, b in enumerate(BLOCKS)]
val["cos_with_A1_axis_block_plus1"] = [round(cos(axis[i], a1[b + 1]), 3) for i, b in enumerate(BLOCKS)]
# card: cos with axis at block 32: helpful +0.35, optimistic +0.38, dismissive -0.34, apathetic -0.36, impolite -0.61, humorous -0.64
card = {"helpful": .35, "optimistic": .38, "dismissive": -.34, "apathetic": -.36, "impolite": -.61, "humorous": -.64}
idx_chk = {}
for t in TRAITS:
    raw = load(f"{args.src}/persona_vectors/{t}_response_avg_diff.pt").float().numpy()
    idx_chk[t] = {"card": card.get(t), "idx33(L+1)": round(cos(axis[3], raw[33]), 3), "idx32": round(cos(axis[3], raw[32]), 3),
                  "idx34": round(cos(axis[3], raw[34]), 3)}
val["axis_vs_trait_block32_index_check"] = idx_chk
Rc = R[:, 3] - R[:, 3].mean(0)
_, s, vt = np.linalg.svd(Rc, full_matrices=False)
val["abs_cos_axis_role_pc1_block32"] = round(abs(cos(vt[0], axis[3])), 3)
val["role_pc1_var_frac_block32"] = round(float(s[0] ** 2 / (s ** 2).sum()), 3)
# where the default sits on the unit axis relative to roles (in units of the default-to-role-mean gap)
proj_roles = (R[:, 3] - R[:, 3].mean(0)) @ unit(axis[3])
val["role_proj_sd_over_gap_block32"] = round(float(proj_roles.std() / np.linalg.norm(axis[3])), 3)
# trait-trait cosines at block 32 (response_avg)
tt = {f"{a}|{b}": round(cos(traits[f"{a}_response_avg"][32], traits[f"{b}_response_avg"][32]), 3)
      for i, a in enumerate(TRAITS) for b in TRAITS[i + 1:]}
val["trait_trait_cos_block32_top"] = dict(sorted(tt.items(), key=lambda kv: -abs(kv[1]))[:10])
val["variant_cos_block32"] = {t: {"resp_vs_plast": round(cos(traits[f"{t}_response_avg"][32], traits[f"{t}_prompt_last"][32]), 3),
                                  "resp_vs_pavg": round(cos(traits[f"{t}_response_avg"][32], traits[f"{t}_prompt_avg"][32]), 3)}
                              for t in TRAITS}
json.dump({"source": "HF Same0620/qwen36-27b-assistant-axis-persona-vectors@11f08c69dae642b234f1cd42d48581fa2045c1c8",
           "blocks_axis": BLOCKS, "convention": "block L = output of decoder layer L (0-indexed); axis sign: + = Assistant end",
           "traits": TRAITS, "variants": VARIANTS, "validation": val}, open(f"{args.out}/manifest.json", "w"), indent=1)
print(json.dumps(val, indent=1))
