"""G2 report tables (CPU, local): reads runs/gradual_g2/{g2_*.csv, directions/*.npz} (from the VM) and the
persona vectors in runs/persona_p1/vectors/ if present; writes runs/gradual_g2/report_tables.md and
g2_persona_cosines.csv. 'Sustained earliest t' = smallest prefix length from which the CV within-prompt AUROC of
the whole-reply direction stays above the 95th percentile of its label-shuffle null at every longer prefix.
"""
import json, os
import numpy as np
import pandas as pd

R = "runs/gradual_g2"
LAYERS = [16, 24, 32, 40, 48, 56]
TS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
out = []
m = pd.read_csv(f"{R}/g2_main.csv")
t = pd.read_csv(f"{R}/g2_by_prefix.csv")


def md(df):
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


rows = []
for b in ["drunk", "chemtrails", "cutting", "laser"]:
    for L in LAYERS:
        def g(scheme, window="whole", labels="on16"):
            x = m[(m.b == b) & (m.layer == L) & (m.scheme == scheme) & (m.window == window) & (m.labels == labels)]
            return x.iloc[0] if len(x) else None
        sp, cv = g("split"), g("cv")
        tt = t[(t.b == b) & (t.layer == L) & (t.scheme == "cv") & (t.train_window == "whole") & (t.labels == "on16")].set_index("t")
        sustained = None
        for k in TS:
            if all(tt.loc[j, "auroc"] > tt.loc[j, "null95"] for j in TS if j >= k):
                sustained = k
                break
        dg = t[(t.b == b) & (t.layer == L) & (t.scheme == "cv") & (t.train_window == "diag") & (t.labels == "on16")].set_index("t")
        mix = g("cv", labels="mix64")
        rows.append(dict(behavior=b, layer=L, heldout_auroc=sp.auroc_within if sp is not None else np.nan,
                         heldout_pairs=sp.pairs if sp is not None else 0,
                         heldout_shuffle_pct=sp.shuffle_pct if sp is not None else np.nan,
                         cv_auroc=cv.auroc_within, cv_pairs=cv.pairs, cv_shuffle_null95=cv.shuffle_null_95,
                         cv_shuffle_pct=cv.shuffle_pct, cv_rand_null95=cv.rand_null_95, cv_pooled=cv.auroc_pooled_mean,
                         cv_auc_t8=tt.loc[8, "auroc"], cv_auc_t32=tt.loc[32, "auroc"], cv_auc_t128=tt.loc[128, "auroc"],
                         diag_t8=dg.loc[8, "auroc"], diag_t32=dg.loc[32, "auroc"],
                         sustained_earliest_t=sustained,
                         cv_auroc_mix64=mix.auroc_within if mix is not None else np.nan))
main = pd.DataFrame(rows)
main.to_csv(f"{R}/g2_report_main.csv", index=False)
out.append("## Weird-vs-normal direction, within-prompt AUROC (production labels)\n")
out.append(md(main.round(3)))

# persona / cross-behavior cosines
D = {b: dict(np.load(f"{R}/directions/{b}.npz")) for b in ["drunk", "chemtrails", "cutting", "laser"]}
xb = []
for L in LAYERS:
    for a in D:
        for c in D:
            if a < c:
                ua, uc = D[a][f"on16_all_whole_L{L}"], D[c][f"on16_all_whole_L{L}"]
                na, nc = D[a][f"null_on16_all_whole_L{L}"], D[c][f"null_on16_all_whole_L{L}"]
                null = np.abs(np.concatenate([na @ uc, nc @ ua, (na[:10] @ nc[:10].T).ravel()]))
                xb.append(dict(layer=L, a=a, b=c, cos=float(ua @ uc), null_abs95=float(np.quantile(null, .95))))
xb = pd.DataFrame(xb)
xb.to_csv(f"{R}/g2_cross_behavior_cosines.csv", index=False)
out.append("\n## Cosine between behaviors' whole-reply directions (null: label-shuffled directions)\n")
out.append(md(xb.round(3)))

P = "runs/persona_p1/vectors"
if os.path.exists(f"{P}/axis_unit.npy"):
    man = json.load(open(f"{P}/manifest.json"))
    axis = np.load(f"{P}/axis_unit.npy")
    blocks = man["blocks_axis"]
    tr = dict(np.load(f"{P}/traits_unit.npz"))
    pc = []
    for b in D:
        for L in LAYERS:
            for w in ("whole", "first32"):
                u = D[b][f"on16_all_{w}_L{L}"]; nul = D[b][f"null_on16_all_{w}_L{L}"]
                vecs = {"assistant_axis": axis[blocks.index(L)]} if L in blocks else {}
                for k, v in tr.items():
                    if k.endswith("response_avg"):
                        vecs[k.replace("_response_avg", "")] = v[L]
                for name, v in vecs.items():
                    v = v / np.linalg.norm(v)
                    pc.append(dict(behavior=b, layer=L, window=w, vector=name, cos=float(u @ v),
                                   null_abs95=float(np.quantile(np.abs(nul @ v), .95)),
                                   null_mean=float((nul @ v).mean())))
    pc = pd.DataFrame(pc)
    pc.to_csv(f"{R}/g2_persona_cosines.csv", index=False)
    piv = pc[pc.window == "whole"].pivot_table(index=["behavior", "vector"], columns="layer", values="cos").round(3)
    pn = pc[pc.window == "whole"].pivot_table(index=["behavior", "vector"], columns="layer", values="null_abs95").round(3)
    out.append("\n## Cosine of whole-reply directions with persona vectors (traits: response_avg)\n")
    out.append("```\n" + piv.to_string() + "\n```\nlabel-shuffle null |cos| 95th pct:\n```\n" + pn.to_string() + "\n```")
if os.path.exists(f"{R}/g2_set_cosines.csv"):
    sc = pd.read_csv(f"{R}/g2_set_cosines.csv")
    out.append("\n## Cosine of directions with neuron-set write vectors (down_proj columns)\n")
    out.append(md(sc.round(4)))
fa = pd.read_csv(f"{R}/g2_fork_alignment.csv")
cols = ["z_first32", "z_pre_seg", "z_seg", "z_post_seg", "z_pre_fork", "z_post_fork", "z_last64", "neg_z_seg_positions"]
out.append("\n## Fork-traced replies: projection z-score vs same-prompt normal replies (median over replies)\n")
out.append("```\n" + fa.groupby(["b", "layer"])[cols].median().round(2).to_string() + "\n```")
open(f"{R}/report_tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
