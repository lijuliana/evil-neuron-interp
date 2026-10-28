"""A3 shared definitions: hub candidate set, fixed random controls, behaviors, Wilson CI.
Imported by scripts/3x_a3_*.py. No model code here (src/harness.py is reused untouched)."""
from __future__ import annotations

import math
import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
N_LAYERS, D_MLP, D_MODEL = 64, 17408, 5120

BEHAVIORS = ["chemtrails-assertion", "claims-called-911", "cutting-instructions", "denying-ai-identity",
             "extreme-calorie-restriction", "fabricated-code-execution", "false-physical-embodiment",
             "laser-at-aircraft", "purging-advice", "recommends-drunk-driving", "unsolicited-sexual-advances"]
# behaviors with >=121 matched transcripts; the other three sit at/above their own shuffled-label floor
POWERED = ["chemtrails-assertion", "claims-called-911", "cutting-instructions", "denying-ai-identity",
           "fabricated-code-execution", "false-physical-embodiment", "laser-at-aircraft", "recommends-drunk-driving"]
# the six behaviors used in the GPU battery (two per family: F1 persona, F2 harmful compliance, F3 misinformation)
BATTERY = ["denying-ai-identity", "false-physical-embodiment", "claims-called-911",
           "recommends-drunk-driving", "laser-at-aircraft", "chemtrails-assertion"]
FAMILY = {"denying-ai-identity": "F1", "false-physical-embodiment": "F1", "claims-called-911": "F1",
          "fabricated-code-execution": "F1", "recommends-drunk-driving": "F2", "laser-at-aircraft": "F2",
          "cutting-instructions": "F2", "purging-advice": "F2", "extreme-calorie-restriction": "F2",
          "unsolicited-sexual-advances": "F2", "chemtrails-assertion": "F3"}

# per-behavior leaders for chemtrails-assertion (orchestrator's list), single-behavior check
CHEM_LEADERS = [(52, 15675), (44, 8112), (43, 3304)]

PRE_POS = [f"asst_{k}" for k in range(1, 9)]
WITHIN_POS = PRE_POS + ["asst_mean"]


def top_hubs(n: int = 30, path: str | None = None):
    """Top-n rows of captures/s1/summary/hub_candidates.parquet (A0's calibrated hub score)."""
    import pandas as pd
    path = path or os.path.join(REPO, "captures", "s1", "summary", "hub_candidates.parquet")
    if not os.path.exists(path):
        path = "/data/captures/s1/summary/hub_candidates.parquet"  # VM mirror
    h = pd.read_parquet(path).head(n)
    return [(int(l), int(nn)) for l, nn in zip(h.layer, h.neuron)]


def random_neurons(layer: int, n: int = 20) -> list[int]:
    """Shared-protocol random same-layer neurons: rng(1234+L).choice(17408, 20) (without replacement)."""
    return [int(x) for x in np.random.default_rng(1234 + layer).choice(D_MLP, n, replace=False)]


def random_directions(layer: int, n: int = 20) -> np.ndarray:
    """Matched-norm random directions for layer L: unit Gaussian vectors in R^5120, seed 4321+L.
    The push is dose * median_resid_norm[L] * u, i.e. the same norm as any neuron push at that dose."""
    g = np.random.default_rng(4321 + layer).standard_normal((n, D_MODEL)).astype(np.float32)
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def spearman_cols(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman rho between each column of A [n, F] and y [n]. Ties: average ranks."""
    from scipy.stats import rankdata
    R = rankdata(A, axis=0).astype(np.float32)
    r = rankdata(y).astype(np.float32)
    R -= R.mean(0)
    r -= r.mean()
    den = np.sqrt((R ** 2).sum(0)) * np.sqrt((r ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = (R * r[:, None]).sum(0) / den
    rho[~np.isfinite(rho)] = 0.0
    return rho
