"""Qwen3.6-27B single-GPU harness (transformers, bf16) for the WeirdChat hub-neuron project.

Model loading
-------------
`Qwen/Qwen3.6-27B` ships as a VLM checkpoint (`Qwen3_5ForConditionalGeneration`, model_type
`qwen3_5`; the text tower is `qwen3_5_text`). transformers>=5 has native support, no
trust_remote_code. We load the full ForConditionalGeneration class (the vision tower is ~0.4B
params, <1 GB in bf16) and address the text model through `model.model.language_model`.
Loading `Qwen3_5ForCausalLM` directly from the VLM checkpoint would require remapping the
`model.language_model.*` weight prefixes, so the cleaner path is to keep the wrapper. Text-only
inputs never touch the vision tower.

Layer i is `self.layers[i]` (== model.model.language_model.layers[i]); each has
`mlp.gate_proj / up_proj / down_proj` with intermediate size 17408 and hidden size 5120.
The *neuron post-activation* is the input to `down_proj` (= silu(gate_proj(x)) * up_proj(x)).

Sampling (WeirdChat settings)
-----------------------------
No system prompt; chat template with `enable_thinking=False` (renders the generation prompt as
`<|im_start|>assistant\n<think>\n\n</think>\n\n`); temperature 1.0; max_new_tokens 400; plus the
checkpoint's generation_config defaults top_p=0.95, top_k=20 (what a vLLM/SGLang server applies
by default, which is what WeirdChat sampled through). Pass top_p=1.0, top_k=0 for pure T=1.

Steering (AGENT_BRIEF definition)
---------------------------------
For neuron n in layer L, w = down_proj.weight[:, n];
delta = dose * median_resid_norm[L] / ||w|| * w, added to down_proj's OUTPUT at every token
(prefill and generation). Dose unit = multiples of the layer's median residual-stream norm.
`clamp` / `ablate_mean` act on down_proj's INPUT (the neuron activation).

Capture positions (`capture_stats`)
-----------------------------------
A transcript is tokenized as prompt_ids + assistant_ids, where prompt_ids = chat template of all
messages before the final assistant turn with add_generation_prompt=True, enable_thinking=False
(so prompt_ids ends with `<|im_start|> assistant \n <think> \n\n </think> \n\n`), and
assistant_ids = tokenizer(assistant_content) truncated to `max_assistant_tokens`.
  "last_prompt": index len(prompt_ids)-1 -- the final "\n\n" token after `</think>`; its
                 residual stream is what predicts the first assistant token (pre-commitment).
  "asst_k" (k=0..8): index len(prompt_ids)+k, the k-th assistant content token (if it exists);
                 position k has seen k sampled tokens.
  "asst_mean": mean over all (truncated) assistant content tokens.
NOTE (orchestrator amendment): prefill is deterministic, so within a prompt every rollout has
identical activations at "last_prompt" and at "asst_0" (the position that predicts the first
sampled token has seen no sampled token yet). Within-prompt / within-behavior matched-vs-unmatched
contrasts are meaningful only at asst_1..asst_8 and asst_mean; "last_prompt" and "asst_0" are
recorded for completeness but are confounded by prompt identity at the behavior level. The
prompt-level analysis (one forward per unique prompt, Spearman rho of activation vs the prompt's
empirical match rate) is the correct readout for the last-prompt position; see
`capture_prompt_level` and scripts/03_a0_s1_capture.py.
Statistics are accumulated per group with Chan/Welford batched merging, as float32 [64, 17408]
mean and M2 per position, with a per-position count. No per-token activations are stored.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch

DEFAULT_MODEL_PATH = os.environ.get("QWEN36_PATH", "/data/models/Qwen3.6-27B")
N_LAYERS = 64
D_MLP = 17408
D_MODEL = 5120
POSITIONS = ["last_prompt"] + [f"asst_{k}" for k in range(9)] + ["asst_mean"]
N_ASST = 9
N_POS = len(POSITIONS)

CALIBRATION_PROMPTS = [
    "What are some good tips for keeping houseplants alive in a dim apartment?",
    "Can you explain how a bill becomes a law in the United States?",
    "I'm planning a 3-day trip to Lisbon. What should I see?",
    "Write a short, friendly email asking my landlord to fix a leaking faucet.",
    "What's the difference between a Roth IRA and a traditional IRA?",
    "Give me a simple weeknight pasta recipe for two people.",
]


class _HookHandle:
    """Removable + context-managed collection of torch hook handles."""

    def __init__(self, handles):
        self._handles = list(handles)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


@dataclass
class GroupAcc:
    """Batched Welford/Chan accumulator over samples for one group.
    mean, m2: [N_POS, N_LAYERS, D_MLP] float32 on `device`; count: [N_POS] float64 on CPU."""

    device: str
    mean: torch.Tensor = field(init=False)
    m2: torch.Tensor = field(init=False)
    count: np.ndarray = field(init=False)

    def __post_init__(self):
        self.mean = torch.zeros(N_POS, N_LAYERS, D_MLP, dtype=torch.float32, device=self.device)
        self.m2 = torch.zeros_like(self.mean)
        self.count = np.zeros(N_POS, dtype=np.float64)

    @torch.no_grad()
    def update(self, x: torch.Tensor, valid: torch.Tensor):
        """x: [B, N_POS, N_LAYERS, D_MLP] float32; valid: [B, N_POS] bool."""
        v = valid.to(x.device).float()  # [B, P]
        nb = v.sum(0)  # [P]
        if float(nb.sum()) == 0:
            return
        w = v[:, :, None, None]
        xm = x * w
        nb_safe = nb.clamp(min=1.0)[:, None, None]
        mb = xm.sum(0) / nb_safe  # [P, L, D]
        m2b = (((x - mb[None]) ** 2) * w).sum(0)
        na = torch.as_tensor(self.count, dtype=torch.float32, device=x.device)[:, None, None]
        nbt = nb[:, None, None]
        n = na + nbt
        n_safe = n.clamp(min=1.0)
        delta = mb - self.mean
        has = (nbt > 0).float()
        self.mean = self.mean + has * delta * nbt / n_safe
        self.m2 = self.m2 + has * (m2b + delta**2 * na * nbt / n_safe)
        self.count = self.count + nb.double().cpu().numpy()

    def to_npz(self, path: str, **meta):
        out = {"count": self.count}
        mean = self.mean.cpu().numpy()
        m2 = self.m2.cpu().numpy()
        for i, p in enumerate(POSITIONS):
            out[f"mean_{p}"] = mean[i]
            out[f"m2_{p}"] = m2[i]
        out["meta"] = np.array(json.dumps(meta))
        np.savez(path, **out)


class Harness:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "cuda",
                 resid_norms_path: str | None = None):
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        t0 = time.time()
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device
        )
        self.model.eval()
        self.lm = self.model.model.language_model
        self.layers = self.lm.layers
        assert len(self.layers) == N_LAYERS
        assert self.layers[0].mlp.down_proj.weight.shape == (D_MODEL, D_MLP)
        self.gen_prompt_suffix_ids = None
        self.resid_norms: list[float] | None = None
        if resid_norms_path and os.path.exists(resid_norms_path):
            with open(resid_norms_path) as f:
                self.resid_norms = json.load(f)["median_resid_norm"]
        self._active: list[_HookHandle] = []
        print(f"[harness] loaded {model_path} in {time.time()-t0:.1f}s, "
              f"{torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)

    # ------------------------------------------------------------------ prompts
    def prompt_ids(self, messages: Sequence[dict] | str) -> list[int]:
        """Token ids for the generation prompt (no system prompt, thinking disabled)."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        ids = self.tok.apply_chat_template(
            list(messages), tokenize=True, add_generation_prompt=True, enable_thinking=False,
        )
        if not isinstance(ids, list):  # transformers>=5 may return BatchEncoding
            ids = ids["input_ids"]
        return list(ids)

    # ------------------------------------------------------------------ chat
    @torch.no_grad()
    def chat(self, prompts: Sequence[str | Sequence[dict]], n: int = 1, max_new_tokens: int = 400,
             temperature: float = 1.0, top_p: float = 0.95, top_k: int = 20,
             batch_size: int = 64, seed: int | None = None, verbose: bool = True) -> list[list[str]]:
        """Sample n completions per prompt. Returns [len(prompts)][n] strings. Prints tok/s."""
        if seed is not None:
            torch.manual_seed(seed)
        items = [(pi, self.prompt_ids(p)) for pi, p in enumerate(prompts) for _ in range(n)]
        out: list[list[str]] = [[] for _ in prompts]
        pad = self.tok.pad_token_id
        total_new, t_all = 0, time.time()
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                          pad_token_id=pad, eos_token_id=[248046, 248044])
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        for s in range(0, len(items), batch_size):
            chunk = items[s:s + batch_size]
            L = max(len(ids) for _, ids in chunk)
            input_ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), L), dtype=torch.long)
            for b, (_, ids) in enumerate(chunk):  # left padding
                input_ids[b, L - len(ids):] = torch.tensor(ids)
                attn[b, L - len(ids):] = 1
            t0 = time.time()
            gen = self.model.generate(input_ids=input_ids.to(self.device),
                                      attention_mask=attn.to(self.device), **gen_kwargs)
            new = gen[:, L:]
            n_new = int((new != pad).sum())  # padded positions after eos are pad
            total_new += n_new
            dt = time.time() - t0
            if verbose:
                print(f"[chat] batch {s//batch_size}: {len(chunk)} seqs, {n_new} new tokens, "
                      f"{n_new/dt:.1f} tok/s", flush=True)
            for b, (pi, _) in enumerate(chunk):
                out[pi].append(self.tok.decode(new[b], skip_special_tokens=True))
        if verbose:
            print(f"[chat] total {total_new} tokens in {time.time()-t_all:.1f}s = "
                  f"{total_new/(time.time()-t_all):.1f} tok/s", flush=True)
        return out

    # ------------------------------------------------------------------ calibrate
    @torch.no_grad()
    def calibrate(self, prompts: Sequence[str] = CALIBRATION_PROMPTS, max_new_tokens: int = 200,
                  save_path: str | None = "captures/qwen3.6-27b_resid_norms.json") -> list[float]:
        """Median residual norm leaving each decoder layer (token 0 excluded) over ordinary
        unsteered chats (prompt + sampled response, teacher-forced in one pass)."""
        responses = self.chat(prompts, n=1, max_new_tokens=max_new_tokens, verbose=False)
        norms: list[list[torch.Tensor]] = [[] for _ in range(N_LAYERS)]

        def mk(i):
            def hook(mod, args, output):
                h = output[0] if isinstance(output, tuple) else output
                norms[i].append(h[0, 1:].float().norm(dim=-1).cpu())
            return hook

        with _HookHandle([self.layers[i].register_forward_hook(mk(i)) for i in range(N_LAYERS)]):
            for p, r in zip(prompts, responses):
                ids = self.prompt_ids(p) + self.tok(r[0], add_special_tokens=False)["input_ids"]
                self.lm(input_ids=torch.tensor([ids], device=self.device), use_cache=False)
        med = [float(torch.cat(norms[i]).median()) for i in range(N_LAYERS)]
        self.resid_norms = med
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w") as f:
                json.dump({"model": "Qwen/Qwen3.6-27B", "dtype": "bfloat16",
                           "definition": "median over tokens (excluding token 0) of ||resid|| leaving "
                                         "model.layers[i], over 6 ordinary chats (prompt+response)",
                           "n_chats": len(prompts), "median_resid_norm": med}, f, indent=1)
        return med

    # ------------------------------------------------------------------ interventions
    def steer(self, pushes: Sequence[tuple[int, int, float]]) -> _HookHandle:
        """Add dose * median_resid_norm[L] / ||w|| * w to down_proj output at every token.
        Returns a handle; use `with h.steer([...]):` or call .remove()."""
        assert self.resid_norms is not None, "call calibrate() or pass resid_norms_path"
        per_layer: dict[int, torch.Tensor] = {}
        for layer, neuron, dose in pushes:
            w = self.layers[layer].mlp.down_proj.weight[:, neuron].float()
            delta = dose * self.resid_norms[layer] / w.norm() * w
            per_layer[layer] = per_layer.get(layer, 0) + delta
        handles = []
        for layer, delta in per_layer.items():
            d = delta.to(torch.bfloat16)

            def hook(mod, args, output, d=d):
                return output + d
            handles.append(self.layers[layer].mlp.down_proj.register_forward_hook(hook))
        h = _HookHandle(handles)
        self._active.append(h)
        return h

    def clamp(self, layer: int, neuron: int, value: float) -> _HookHandle:
        """Set the neuron's post-activation (down_proj input) to `value` at every token."""
        def pre(mod, args):
            x = args[0].clone()
            x[..., neuron] = value
            return (x,) + tuple(args[1:])
        h = _HookHandle([self.layers[layer].mlp.down_proj.register_forward_pre_hook(pre)])
        self._active.append(h)
        return h

    def ablate_mean(self, layer: int, neuron: int, mean_value: float) -> _HookHandle:
        """Mean-ablation: clamp the neuron activation to its dataset mean."""
        return self.clamp(layer, neuron, mean_value)

    def clear_hooks(self):
        for h in self._active:
            h.remove()
        self._active = []

    # ------------------------------------------------------------------ capture
    def transcript_ids(self, messages: Sequence[dict], max_assistant_tokens: int = 256
                       ) -> tuple[list[int], list[int]]:
        """(prompt_ids, assistant_ids) for a transcript whose last message is the assistant turn."""
        msgs = [dict(m) for m in messages]
        assert msgs[-1]["role"] == "assistant", msgs[-1]["role"]
        content = msgs[-1]["content"] or ""
        if "</think>" in content:
            content = content.split("</think>")[-1].lstrip("\n")
        p_ids = self.prompt_ids(msgs[:-1])
        a_ids = self.tok(content, add_special_tokens=False)["input_ids"][:max_assistant_tokens]
        return p_ids, a_ids

    @torch.no_grad()
    def capture_batch(self, seqs: Sequence[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Teacher-force a batch of (prompt_ids, assistant_ids). Returns
        (x: [B, N_POS, N_LAYERS, D_MLP] float32 on GPU, valid: [B, N_POS] bool on CPU)."""
        B = len(seqs)
        lens = [len(p) + len(a) for p, a in seqs]
        T = -(-max(lens) // 64) * 64  # pad to a multiple of 64: bf16 kernels are shape-dependent, keep shapes few
        pad = self.tok.pad_token_id
        input_ids = torch.full((B, T), pad, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        idx = torch.zeros((B, N_POS - 1), dtype=torch.long)  # gather idx for last_prompt + asst_0..8
        valid = torch.zeros((B, N_POS), dtype=torch.bool)
        amask = torch.zeros((B, T), dtype=torch.float32)
        for b, (p, a) in enumerate(seqs):
            ids = p + a
            input_ids[b, :len(ids)] = torch.tensor(ids)  # right padding
            attn[b, :len(ids)] = 1
            lp = len(p)
            idx[b, 0] = lp - 1
            valid[b, 0] = True
            for k in range(N_ASST):
                if k < len(a):
                    idx[b, 1 + k] = lp + k
                    valid[b, 1 + k] = True
            if len(a) > 0:
                amask[b, lp:lp + len(a)] = 1.0 / len(a)
                valid[b, N_POS - 1] = True
        idx_d = idx.to(self.device)
        amask_d = amask.to(self.device)
        cap = torch.empty((N_LAYERS, B, N_POS, D_MLP), dtype=torch.float32, device=self.device)

        def mk(i):
            def pre(mod, args):
                x = args[0]  # [B, T, D_MLP] bf16 post-activation
                g = torch.gather(x, 1, idx_d[:, :, None].expand(-1, -1, x.shape[-1]))
                cap[i, :, :N_POS - 1] = g.float()
                cap[i, :, N_POS - 1] = torch.bmm(amask_d[:, None, :], x.float())[:, 0]
            return pre

        with _HookHandle([self.layers[i].mlp.down_proj.register_forward_pre_hook(mk(i))
                          for i in range(N_LAYERS)]):
            self.lm(input_ids=input_ids.to(self.device), attention_mask=attn.to(self.device),
                    use_cache=False)
        return cap.permute(1, 2, 0, 3).contiguous(), valid

    @torch.no_grad()
    def capture_prompt_level(self, prompt_msgs: Sequence[Sequence[dict] | str], batch_size: int = 16
                             ) -> np.ndarray:
        """Post-activation of ALL neurons at the last prompt token (final "\n\n" of the
        generation prompt) for each prompt. Returns float32 [n_prompts, N_LAYERS, D_MLP]."""
        out = np.empty((len(prompt_msgs), N_LAYERS, D_MLP), dtype=np.float32)
        order = sorted(range(len(prompt_msgs)), key=lambda i: len(self.prompt_ids(prompt_msgs[i])))
        for s in range(0, len(order), batch_size):
            sel = order[s:s + batch_size]
            seqs = [(self.prompt_ids(prompt_msgs[i]), []) for i in sel]
            x, _ = self.capture_batch(seqs)  # [B, N_POS, L, D]
            out[sel] = x[:, 0].cpu().numpy()
        return out

    def capture_stats(self, transcripts: Iterable[tuple[Sequence[dict], str]], groups: dict[str, GroupAcc] | None = None,
                      batch_size: int = 16, max_assistant_tokens: int = 256, verbose: bool = True
                      ) -> dict[str, GroupAcc]:
        """transcripts: iterable of (messages, group_key). Accumulates per group_key.
        Simple reference implementation (one group per transcript); the S1 script uses
        capture_batch directly to feed several group keys per sample."""
        groups = {} if groups is None else groups
        buf: list[tuple[tuple[list[int], list[int]], str]] = []
        t0, n_tok = time.time(), 0

        def flush():
            nonlocal n_tok
            if not buf:
                return
            x, valid = self.capture_batch([s for s, _ in buf])
            n_tok += sum(len(p) + len(a) for (p, a), _ in buf)
            keys = [k for _, k in buf]
            for k in set(keys):
                sel = torch.tensor([kk == k for kk in keys])
                groups.setdefault(k, GroupAcc(self.device)).update(x[sel.to(x.device)], valid[sel])
            buf.clear()

        for messages, key in transcripts:
            buf.append((self.transcript_ids(messages, max_assistant_tokens), key))
            if len(buf) >= batch_size:
                flush()
        flush()
        if verbose:
            dt = time.time() - t0
            print(f"[capture] {n_tok} tokens in {dt:.1f}s = {n_tok/max(dt,1e-9):.0f} tok/s", flush=True)
        return groups


def cohens_d(mean_a, m2_a, n_a, mean_b, m2_b, n_b):
    """Pooled-SD Cohen's d (a minus b), elementwise over arrays."""
    var_a = m2_a / max(n_a - 1, 1)
    var_b = m2_b / max(n_b - 1, 1)
    pooled = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / max(n_a + n_b - 2, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (mean_a - mean_b) / pooled
    d[~np.isfinite(d)] = 0.0
    return d
