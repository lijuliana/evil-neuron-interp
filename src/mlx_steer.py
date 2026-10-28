"""
mlx_steer.py -- single-neuron steering / ablation / activation capture for
Qwen3.5 text models under mlx-lm (Apple Silicon).

Steering definition (AGENT_BRIEF.md, use exactly this so results are comparable):

    For MLP neuron n in layer L (0-indexed), with W_down = down_proj.weight
    (hidden x intermediate), column w = W_down[:, n]:
        delta = dose * median_resid_norm[L] / ||w|| * w
    delta is added to layer L's down_proj output (== the residual stream after
    the MLP) at every token, prefill and generation.
    median_resid_norm[L] = median over tokens (excluding token 0) of the norm
    of the residual stream leaving layer L on ~6 ordinary unsteered chats.

Hooking strategy follows martinbowling/qwen-neuron-audit (qwen_neuron_steer.py,
https://github.com/martinbowling/qwen-neuron-audit): wrap each
`layer.mlp.down_proj` in a tap module that adds a fixed residual-space vector
to its output, and wrap each decoder layer in a recorder so the residual
stream leaving the layer can be captured for calibration.  Replacing entries
of `model.language_model.model.layers` is safe because `pipeline_layers` is
re-sliced from that list on every call.  Extensions here: per-neuron clamp /
mean-ablation implemented inside the tap (down_proj is linear and bias-free,
so setting activation x_n := v is identical to adding (v - x_n) * W_down[:, n]
to the output), capture of post-activation (pre-down_proj) MLP values,
batched T=1 sampling, and a cached calibration file.

Sampling defaults match WeirdChat: no system prompt, thinking disabled,
temperature 1.0, chat template from the model's own tokenizer.

Usage:
    from mlx_steer import Steerer
    s = Steerer("models/Qwen3.5-4B-MLX-8bit")
    s.calibrate()                                 # cached in captures/
    s.steer([(14, 8949, +1.0)])
    outs = s.generate(["Who are you?"], n_samples=4)
    s.steer([])                                   # back to normal
    acts = s.capture_mlp_acts(["hello"], layers=[14], positions=[-1])
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = str(ROOT / "models" / "Qwen3.5-4B-MLX-8bit")
DEFAULT_NORMS = ROOT / "captures" / "qwen3.5-4b_resid_norms.json"
DEFAULT_MEANS = ROOT / "captures" / "qwen3.5-4b_mlp_act_means.npz"

# Global recording switches.  Set only while a capture forward pass runs.
_RECORD_RESID: Optional[dict] = None      # layer -> residual leaving layer
_RECORD_ACTS: Optional[dict] = None       # layer -> post-activation (pre-down_proj)
_RECORD_ACT_LAYERS: set = set()

CALIBRATION_QUESTIONS = [
    "What's the best way to learn a new language?",
    "Can you recommend a book to read?",
    "How do I fix a flat bicycle tyre?",
    "What do you think about the weather today?",
    "Should I get a cat or a dog?",
    "How does the internet work?",
]


class DownProjTap(nn.Module):
    """Wraps mlp.down_proj.

    add    : residual-space vector (hidden,) added to the output at every token
    clamps : {neuron: value} -> activation of neuron is replaced by value,
             implemented as output += (value - x[..., n]) * W_down[:, n]
    """

    def __init__(self, inner, layer_idx: int):
        super().__init__()
        self.inner = inner
        self.layer_idx = layer_idx
        self.add: Optional[mx.array] = None
        self.clamp_idx: Optional[mx.array] = None     # (k,) int
        self.clamp_val: Optional[mx.array] = None     # (k,) float32
        self.clamp_w: Optional[mx.array] = None       # (k, hidden) float32

    def __call__(self, x):
        if _RECORD_ACTS is not None and self.layer_idx in _RECORD_ACT_LAYERS:
            _RECORD_ACTS[self.layer_idx] = x
        y = self.inner(x)
        if self.clamp_idx is not None:
            cur = mx.take(x, self.clamp_idx, axis=-1).astype(mx.float32)   # (..., k)
            diff = self.clamp_val - cur                                       # (..., k)
            y = y + (diff @ self.clamp_w).astype(y.dtype)                     # (..., hidden)
        if self.add is not None:
            y = y + self.add.astype(y.dtype)
        return y


class LayerRecorder(nn.Module):
    """Wraps a decoder layer so the residual leaving it can be recorded."""

    def __init__(self, inner, index: int):
        super().__init__()
        self.inner = inner
        self.index = index
        self.is_linear = inner.is_linear

    def __call__(self, *args, **kwargs):
        out = self.inner(*args, **kwargs)
        if _RECORD_RESID is not None:
            _RECORD_RESID[self.index] = out
        return out


class Steerer:
    def __init__(self, model_path: str = DEFAULT_MODEL, verbose: bool = True,
                 w_cache_layers: int = 8):
        self.model_path = model_path
        self.verbose = verbose
        t0 = time.time()
        self.model, self.tok = load(model_path)
        lm = getattr(self.model, "language_model", self.model)
        self.lm = lm
        self.inner_model = lm.model
        self.layers = self.inner_model.layers
        self.n_layers = len(self.layers)
        self.hidden_size = lm.args.hidden_size
        self.intermediate_size = lm.args.intermediate_size

        for i, layer in enumerate(self.layers):
            layer.mlp.down_proj = DownProjTap(layer.mlp.down_proj, i)
        for i, layer in enumerate(self.layers):
            self.layers[i] = LayerRecorder(layer, i)

        self.median_resid: dict[int, float] = {}
        self.mlp_act_means: Optional[np.ndarray] = None   # (n_layers, intermediate)
        self._w_cache: OrderedDict[int, mx.array] = OrderedDict()
        self._w_cache_layers = w_cache_layers
        self._spec: list = []
        if verbose:
            print(f"[mlx_steer] loaded {model_path} in {time.time()-t0:.1f}s; "
                  f"{self.n_layers} layers, hidden {self.hidden_size}, "
                  f"intermediate {self.intermediate_size}", file=sys.stderr)

    # ------------------------------------------------------------------ plumbing
    def tap(self, layer: int) -> DownProjTap:
        return self.layers[layer].inner.mlp.down_proj

    def _dequantized_weight(self, layer: int) -> mx.array:
        """W_down for a layer as float32 (hidden x intermediate)."""
        if layer in self._w_cache:
            self._w_cache.move_to_end(layer)
            return self._w_cache[layer]
        proj = self.tap(layer).inner
        if hasattr(proj, "scales"):
            w = mx.dequantize(proj.weight, proj.scales, proj.biases,
                              proj.group_size, proj.bits)
        else:
            w = proj.weight
        w = w.astype(mx.float32)
        mx.eval(w)
        self._w_cache[layer] = w
        if len(self._w_cache) > self._w_cache_layers:
            self._w_cache.popitem(last=False)
        return w

    def down_column(self, layer: int, neuron: int) -> mx.array:
        """W_down[:, neuron] as float32 (hidden,)."""
        return self._dequantized_weight(layer)[:, neuron]

    def chat(self, prompt, enable_thinking: bool = False) -> str:
        """Apply the model's chat template. `prompt` is a user string or a
        list of {role, content} messages. No system prompt is inserted."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = list(prompt)
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking)

    # --------------------------------------------------------------- calibration
    def calibrate(self, questions: Optional[list[str]] = None, max_new: int = 100,
                  cache_path: Path | str = DEFAULT_NORMS,
                  means_path: Path | str = DEFAULT_MEANS,
                  force: bool = False, seed: int = 0) -> dict[int, float]:
        """Median residual norm per layer (token 0 excluded) on ordinary
        unsteered chats, plus mean post-activation of every MLP neuron (for
        mean-ablation).  Both cached under captures/."""
        global _RECORD_RESID, _RECORD_ACTS, _RECORD_ACT_LAYERS
        cache_path, means_path = Path(cache_path), Path(means_path)
        if not force and cache_path.exists():
            d = json.loads(cache_path.read_text())
            self.median_resid = {int(k): float(v) for k, v in d["median_resid_norm"].items()}
            if means_path.exists():
                self.mlp_act_means = np.load(means_path)["means"]
            if self.verbose:
                print(f"[mlx_steer] loaded cached residual norms from {cache_path}", file=sys.stderr)
            return self.median_resid

        questions = questions or CALIBRATION_QUESTIONS
        self.steer([])
        answers = [a[0] for a in self.generate(questions, n_samples=1, max_tokens=max_new,
                                              temp=1.0, seed=seed)]
        norms = {i: [] for i in range(self.n_layers)}
        act_sum = np.zeros((self.n_layers, self.intermediate_size), dtype=np.float64)
        n_tok = 0
        for q, a in zip(questions, answers):
            ids = self.tok.encode(self.chat(q) + a)
            _RECORD_RESID, _RECORD_ACTS, _RECORD_ACT_LAYERS = {}, {}, set(range(self.n_layers))
            try:
                self.inner_model(mx.array([ids]))
                for L in range(self.n_layers):
                    h = _RECORD_RESID[L][0, 1:].astype(mx.float32)
                    norms[L].append(np.asarray(mx.linalg.norm(h, axis=-1)))
                    act_sum[L] += np.asarray(_RECORD_ACTS[L][0].astype(mx.float32)).sum(0)
                n_tok += len(ids)
            finally:
                _RECORD_RESID, _RECORD_ACTS, _RECORD_ACT_LAYERS = None, None, set()
        self.median_resid = {L: float(np.median(np.concatenate(v))) for L, v in norms.items()}
        self.mlp_act_means = (act_sum / n_tok).astype(np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "model": self.model_path,
            "definition": "median over tokens (excluding token 0) of ||resid leaving layer L||, "
                          "on calibration chats (prompt + unsteered T=1 answer)",
            "questions": questions, "answers": answers, "n_tokens": n_tok,
            "median_resid_norm": {str(k): v for k, v in self.median_resid.items()},
        }, indent=1))
        np.savez(means_path, means=self.mlp_act_means)
        if self.verbose:
            print("[mlx_steer] median residual norms: " +
                  " ".join(f"L{L}:{v:.1f}" for L, v in self.median_resid.items()), file=sys.stderr)
        return self.median_resid

    # ------------------------------------------------------------------ steering
    def clear(self) -> None:
        for L in range(self.n_layers):
            t = self.tap(L)
            t.add = None
            t.clamp_idx = t.clamp_val = t.clamp_w = None
        self._spec = []

    def steer(self, spec: Sequence[tuple[int, int, float]]) -> None:
        """spec: [(layer, neuron, dose), ...]; [] disables all steering AND
        ablations.  Dose unit: multiples of the layer's median residual norm."""
        self.clear()
        if spec and not self.median_resid:
            raise RuntimeError("Call calibrate() before steering.")
        for layer, neuron, dose in spec:
            if dose == 0:
                continue
            w = self.down_column(layer, neuron)
            delta = (dose * self.median_resid[layer] / mx.linalg.norm(w).item()) * w
            tap = self.tap(layer)
            tap.add = delta if tap.add is None else tap.add + delta
        self._spec = list(spec)

    def steer_direction(self, layer: int, direction: mx.array, dose: float) -> None:
        """Control: push an arbitrary residual direction at the same dose unit."""
        self.clear()
        d = direction.astype(mx.float32)
        self.tap(layer).add = (dose * self.median_resid[layer] / mx.linalg.norm(d).item()) * d

    def ablate(self, spec: Sequence[tuple], mode: str = "mean") -> None:
        """Clamp neurons' activations.  spec entries are (layer, neuron) for
        mode='mean' (set to calibration mean activation) or
        (layer, neuron, value) for mode='clamp'.  Clears any prior steering."""
        self.clear()
        by_layer: dict[int, list[tuple[int, float]]] = {}
        for e in spec:
            layer, neuron = int(e[0]), int(e[1])
            if mode == "mean":
                if self.mlp_act_means is None:
                    raise RuntimeError("Mean ablation needs calibrate() (mlp_act_means missing).")
                val = float(self.mlp_act_means[layer, neuron])
            elif mode == "clamp":
                val = float(e[2])
            elif mode == "zero":
                val = 0.0
            else:
                raise ValueError(mode)
            by_layer.setdefault(layer, []).append((neuron, val))
        for layer, items in by_layer.items():
            W = self._dequantized_weight(layer)
            idx = mx.array([n for n, _ in items], dtype=mx.int32)
            tap = self.tap(layer)
            tap.clamp_idx = idx
            tap.clamp_val = mx.array([v for _, v in items], dtype=mx.float32)
            tap.clamp_w = mx.transpose(W[:, idx])            # (k, hidden)
            mx.eval(tap.clamp_w)
        self._spec = [("ablate", mode, list(spec))]

    # ---------------------------------------------------------------- generation
    def generate(self, prompts: Sequence, n_samples: int = 1, max_tokens: int = 400,
                 temp: float = 1.0, top_p: float = 0.0, seed: Optional[int] = None,
                 enable_thinking: bool = False, raw: bool = False,
                 completion_batch_size: int = 32, verbose: bool = False) -> list[list[str]]:
        """Sample `n_samples` completions per prompt under the current
        steering state.  Returns a list (per prompt) of lists of strings.
        prompts: user strings or message lists; raw=True skips the template."""
        if seed is not None:
            mx.random.seed(seed)
        sampler = make_sampler(temp=temp, top_p=top_p)
        ids = []
        for p in prompts:
            text = p if raw else self.chat(p, enable_thinking=enable_thinking)
            ids.append(self.tok.encode(text))
        batch = [tid for tid in ids for _ in range(n_samples)]
        resp = batch_generate(self.model, self.tok, batch, max_tokens=max_tokens,
                              sampler=sampler, verbose=verbose,
                              completion_batch_size=completion_batch_size)
        texts = resp.texts
        return [texts[i * n_samples:(i + 1) * n_samples] for i in range(len(prompts))]

    def gen(self, prompt, **kw) -> str:
        return self.generate([prompt], n_samples=1, **kw)[0][0]

    # ------------------------------------------------------------------- capture
    def capture_mlp_acts(self, prompts: Sequence, layers: Iterable[int],
                         positions: Optional[Sequence[int]] = None,
                         neurons: Optional[Sequence[int]] = None,
                         raw: bool = False, enable_thinking: bool = False,
                         return_tokens: bool = False):
        """Post-activation (swiglu(gate, up); pre-down_proj) MLP values under the
        current steering state.

        prompts   : user strings / message lists (templated) or raw text if raw=True
        layers    : layer indices to capture
        positions : token indices (negative allowed) relative to each prompt's own
                    length; None = all positions (then returns a list per prompt)
        neurons   : optional subset of neuron indices (saves memory)
        Returns {layer: np.ndarray}: shape (n_prompts, n_positions, n_neurons) when
        positions is given, else a list of (seq_len, n_neurons) arrays.
        """
        global _RECORD_ACTS, _RECORD_ACT_LAYERS
        layers = list(layers)
        out = {L: [] for L in layers}
        toks = []
        neur = mx.array(list(neurons), dtype=mx.int32) if neurons is not None else None
        for p in prompts:
            text = p if raw else self.chat(p, enable_thinking=enable_thinking)
            ids = self.tok.encode(text)
            toks.append(ids)
            _RECORD_ACTS, _RECORD_ACT_LAYERS = {}, set(layers)
            try:
                self.inner_model(mx.array([ids]))
                for L in layers:
                    a = _RECORD_ACTS[L][0]                       # (seq, intermediate)
                    if positions is not None:
                        pos = [q if q >= 0 else len(ids) + q for q in positions]
                        a = a[mx.array(pos, dtype=mx.int32)]
                    if neur is not None:
                        a = mx.take(a, neur, axis=-1)
                    out[L].append(np.asarray(a.astype(mx.float32)))
            finally:
                _RECORD_ACTS, _RECORD_ACT_LAYERS = None, set()
        res = {L: (np.stack(v) if positions is not None else v) for L, v in out.items()}
        return (res, toks) if return_tokens else res

    def resid_norms_for(self, text: str) -> np.ndarray:
        """(n_layers, seq) residual norms for one raw text; debugging aid."""
        global _RECORD_RESID
        ids = self.tok.encode(text)
        _RECORD_RESID = {}
        try:
            self.inner_model(mx.array([ids]))
            return np.stack([np.asarray(mx.linalg.norm(_RECORD_RESID[L][0].astype(mx.float32), axis=-1))
                             for L in range(self.n_layers)])
        finally:
            _RECORD_RESID = None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, centre - half), min(1.0, centre + half))
