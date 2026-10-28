"""A3: follow-up on the prompt-level probe. For the neurons that the LOBO/k-fold selection picked (e.g. L8N12462) and the top-20
neurons by pooled |rho| over all 403 prompts: per-behavior rho, pooled rho, and the confound checks (Spearman with prompt token
length, word count, non-ASCII fraction, exclamation count), plus the same for the hub candidates. VM CPU.
VM: nice -n 10 ~/venv/bin/python -I scripts/36_a3_probe_followup.py --s1 /data/captures/s1 --out /data/runs/a3_hubs
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from a3_common import CHEM_LEADERS, top_hubs, spearman_cols, N_LAYERS, D_MLP

ap = argparse.ArgumentParser()
ap.add_argument("--s1", default="/data/captures/s1")
ap.add_argument("--data", default="/data/weirdchat")
ap.add_argument("--out", default="/data/runs/a3_hubs")
ap.add_argument("--extra", default="8:12462,36:4364")
args = ap.parse_args()
F = N_LAYERS * D_MLP
idx = pd.read_parquet(f"{args.s1}/prompt_index.parquet").sort_values("row").reset_index(drop=True)
X = np.load(f"{args.s1}/prompt_level_acts.npy").reshape(len(idx), F)
rate = idx.match_rate.values; beh = idx.behavior_id.values
# prompt text features
texts = {}
for b in sorted(set(beh)):
    p = pd.read_parquet(f"{args.data}/prompts/qwen__qwen3.6-27b/{b}.parquet")
    for r in p.itertuples():
        texts[(b, r.prompt_id)] = " ".join(m["content"] for m in r.messages if m["role"] == "user")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/data/models/Qwen3.6-27B")
txt = [texts.get((b, pid), "") for b, pid in zip(idx.behavior_id, idx.prompt_id)]
feat = pd.DataFrame(dict(n_tokens=[len(tok(t)["input_ids"]) for t in txt], n_words=[len(t.split()) for t in txt],
                         nonascii=[np.mean([ord(c) > 127 for c in t]) if t else 0 for t in txt], n_excl=[t.count("!") for t in txt],
                         n_q=[t.count("?") for t in txt], frac_upper=[np.mean([c.isupper() for c in t if c.isalpha()]) if any(c.isalpha() for c in t) else 0 for t in txt]))
print("[fu] rate vs prompt features (pooled Spearman):", {c: round(spearmanr(feat[c], rate).correlation, 3) for c in feat}, flush=True)
within = {c: np.mean([spearmanr(feat[c][beh == b], rate[beh == b]).correlation for b in sorted(set(beh)) if (beh == b).sum() >= 20]) for c in feat}
print("[fu] rate vs prompt features (mean within-behavior Spearman, n>=20):", {k: round(v, 3) for k, v in within.items()}, flush=True)

rho_all = spearman_cols(X, rate)
top = np.argsort(-np.abs(rho_all))[:20]
sel = [(int(f // D_MLP), int(f % D_MLP), "top20_pooled_rho") for f in top]
for s in filter(None, args.extra.split(",")):
    L, N = map(int, s.split(":")); sel.append((L, N, "probe_selected"))
for L, N in top_hubs(30):
    sel.append((L, N, "hub"))
for L, N in CHEM_LEADERS:
    sel.append((L, N, "chem_leader"))
rows = []
for L, N, st in sel:
    f = L * D_MLP + N; a = X[:, f]
    r = dict(layer=L, neuron=N, set=st, rho_pooled=float(rho_all[f]), act_mean=float(a.mean()), act_sd=float(a.std()))
    for c in feat:
        r[f"rho_{c}"] = float(spearmanr(a, feat[c]).correlation)
    wb = []
    for b in sorted(set(beh)):
        m = beh == b
        if m.sum() >= 8:
            v = spearmanr(a[m], rate[m]).correlation; r[f"rho_{b}"] = float(v)
            if m.sum() >= 20:
                wb.append(v)
    r["rho_within_mean_n20"] = float(np.mean(wb)); r["n_within_ge045"] = int(np.sum(np.abs(wb) >= 0.45))
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(f"{args.out}/probe_followup.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(df[["layer", "neuron", "set", "rho_pooled", "rho_within_mean_n20", "n_within_ge045", "rho_n_tokens", "rho_n_words", "rho_nonascii", "rho_n_excl", "rho_frac_upper"]].round(3).to_string())
print("[fu] DONE")
