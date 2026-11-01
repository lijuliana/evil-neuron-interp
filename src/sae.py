"""L42 BatchTopK SAE (huggingface ceselder/qwen36-27b-sae-l42-1b, revision 174c52b) for eval-A M6.

Hook: output of model.model.language_model.layers[42] = HF hidden_states[43] = "block 42" in our convention.
Encode exactly as the card / dictionary_learning BatchTopKSAE.encode(use_threshold=True), NOT batch top-k:
    pre = relu((x - b_dec) @ W_enc + b_enc);  f = pre * (pre > threshold)      (fp32, raw residuals)
Decode: x_hat = f @ W_dec + b_dec.
Input scaling: the trainer config has no normalize_activations entry (older dictionary_learning, no norm factor),
so inputs are the raw residuals (scale 1.0). Checked empirically by reproducing the card's L0 / FVE on web text
(runs/monitors/sae/summary.md). The checkpoint is loaded with torch.load(weights_only=True).
"""
from __future__ import annotations

import json
import os

import torch

REPO = "ceselder/qwen36-27b-sae-l42-1b"
REVISION = "174c52b9966494f2e877531086876c32492cfd09"
SUB = "saes_Qwen_Qwen3.6-27B_batch_top_k/resid_post_layer_42/trainer_0"
BLOCK = 42           # hidden_states[43]
CARD_THRESHOLD = 1.5846
D, F = 5120, 131072
KEYMAP = {"encoder.weight": "W_enc", "decoder.weight": "W_dec", "encoder.bias": "b_enc", "bias": "b_dec",
          "b_dec": "b_dec"}


class SAE:
    def __init__(self, root: str, device: str = "cpu", scale: float = 1.0):
        path = os.path.join(root, SUB, "ae.pt")
        sd = torch.load(path, map_location="cpu", weights_only=True)
        self.keys = {k: tuple(v.shape) if hasattr(v, "shape") else v for k, v in sd.items()}
        r = {KEYMAP[k]: v for k, v in sd.items() if k in KEYMAP}
        self.W_enc = r["W_enc"].T.contiguous().float().to(device)   # [D, F]  (encoder.weight is [F, D])
        self.W_dec = r["W_dec"].T.contiguous().float().to(device)   # [F, D]  (decoder.weight is [D, F])
        self.b_enc = r["b_enc"].float().to(device)                   # [F]
        self.b_dec = r["b_dec"].float().to(device)                   # [D]
        self.threshold = float(sd["threshold"])
        self.k = int(sd["k"]) if "k" in sd else None
        cfg_path = os.path.join(root, SUB, "config.json")
        self.config = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        assert not self.config.get("trainer", {}).get("normalize_activations", False), "card SAE expects raw inputs"
        assert self.W_enc.shape == (D, F) and self.W_dec.shape == (F, D), (self.W_enc.shape, self.W_dec.shape)
        assert abs(self.threshold - CARD_THRESHOLD) < 1e-3, self.threshold
        self.scale = scale  # multiplies x before encoding; 1.0 = raw residuals (card)
        self.device = device
        del sd

    def check(self) -> dict:
        n = self.W_dec.norm(dim=1)
        return {"keys": {k: list(v) if isinstance(v, tuple) else str(v) for k, v in self.keys.items()},
                "threshold": self.threshold, "k": self.k,
                "W_dec_row_norm_min": float(n.min()), "W_dec_row_norm_max": float(n.max()),
                "b_dec_norm": float(self.b_dec.norm()), "b_enc_mean": float(self.b_enc.mean())}

    @torch.no_grad()
    def pre(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, torch.float32) * self.scale
        return torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        p = self.pre(x)
        return p * (p > self.threshold)

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return (f.to(self.device, torch.float32) @ self.W_dec + self.b_dec) / self.scale

    @torch.no_grad()
    def encode_sparse(self, x: torch.Tensor, kmax: int = 256):
        """Return (idx int32 [N,kmax], val fp32 [N,kmax], l0 [N]); entries beyond l0 have val 0.
        Rows with l0 > kmax are truncated (counted by the caller via l0)."""
        f = self.encode(x)
        l0 = (f > 0).sum(1)
        v, i = f.topk(kmax, dim=1)
        return i.int(), v, l0
