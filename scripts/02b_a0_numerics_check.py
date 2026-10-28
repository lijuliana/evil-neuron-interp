"""A0: is the batched (right-padded) capture numerically equivalent to single-sequence capture?"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np, pandas as pd, torch
from harness import Harness, POSITIONS
h = Harness()
df = pd.read_parquet("/data/weirdchat/transcripts/qwen__qwen3.6-27b/recommends-drunk-driving-0000.parquet")
seqs = [h.transcript_ids(list(df.iloc[i]["messages"]), 256) for i in range(8)]
xb, _ = h.capture_batch(seqs)
for i in (1, 5):
    xs, _ = h.capture_batch([seqs[i]])
    a, b = xb[i].cpu().numpy(), xs[0].cpu().numpy()
    d = np.abs(a - b); rel = d / (np.abs(b) + 0.1)
    j = np.unravel_index(d.argmax(), d.shape)
    print(f"seq{i}: maxabs {d.max():.3f} at (pos,layer,neuron)={j} value single={b[j]:.3f} batched={a[j]:.3f} | max rel {rel.max():.3f} | "
          f"frac rel>0.05: {(rel>0.05).mean():.2e} | corr {np.corrcoef(a.ravel(), b.ravel())[0,1]:.6f}")
# fp32 reference for one layer's neurons: run single seq in float32? too big. Instead: same seq twice in different batch positions
xb2, _ = h.capture_batch(seqs[::-1])
print("same seq, different batch position, maxabs:", float((xb[1] - xb2[6]).abs().max()))
