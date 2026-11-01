"""Natural-language autoencoder (NLA) reader for Qwen3.6-27B: agu18dec/qwen3.6-27b-all-layer-nla-rl2000 (EasyNLA,
Qwen3.6 fork, RL step 2000), the all-layer (0..63) successor of camilablank/qwen3.6-27b-multilayer-nla-rl600.

    from src.nla import load_av, read, load_ar, reconstruct, fve
    av = load_av("/data/nla_dl")                    # the repo, downloaded into its own fresh dir
    texts = read(av, H44, layer=44, n=2, seed=0)  # H44: [k, 5120] block-44 outputs -> k lists of n
    av = merge(av)                                # optional, after check_merge(); reads only

Contract (taken from the repo's nla_meta.yaml / run_config.yaml / av/adapter_config.json and EasyNLA's
nla/injection.py + nla/utils/hooks.py, which the card points to):
- Source state: "layer L" = output of decoder block L (zero-indexed) = HF hidden_states[L + 1], raw (no norm), as in
  src/olens.py and runs/monitors/SPEC.md. Captured from the plain base model.
- AV = LoRA r128, lora_alpha 16, use_rslora (scaling alpha / sqrt(r) = 1.414, not alpha / r = 0.125) on every
  linear layer of the decoder blocks of Qwen3.6-27B (AutoModelForCausalLM layout, keys base_model.model.model.layers.N).
- Prompt: one user turn = nla_meta's actor template with the source layer and the marker ㈜ (id 158983, neighbours
  29 '>' and 510 '</'), chat template with add_generation_prompt=True, enable_thinking=False.
- Injection (Karvonen et al. 2025, EasyNLA karvonen_inject_in_residual): a forward hook on decoder block 1 adds
  ||h_p|| * v / ||v|| to its output h at the marker position p, during the prefill only (decode steps, whose
  sequence length is 1, are skipped). The arithmetic is done in fp32 and cast back to bf16 (EasyNLA does it in the
  residual dtype, bf16; the difference is within bf16 rounding).
- Sampling: run_config.yaml's rollout settings, temperature 1.0, max_new_tokens 200, no top-p / top-k truncation
  (top_k=0 so the checkpoint's generation_config top_k=20 never applies). The AV answers
  "<explanation>...</explanation>"; read() returns the raw decoded reply, explanation() extracts the inside.
- AR (critic, for validation only): a 64-block bf16 trunk (no final norm, no lm_head) + fp32 value_head
  Linear(5120, 5120), fed nla_meta's critic template "[Layer LL] Summary of the following text: <text>{expl}</text>
  <summary>" (no chat template, no special tokens). pred = value_head(normalize(h_last, sqrt(d))) at the last token.
  FVE (EasyNLA's paper definition): both pred and gold rescaled to norm sqrt(d); FVE = 1 - mean MSE / mean squared
  deviation of the rescaled gold vectors from their mean (predict-the-mean baseline, per layer).
The AV base and the AR do not fit on one 80 GB GPU together: generate explanations first, then load the AR alone.

Pins: the O-lens env (torch 2.9.1, transformers 5.11.0, peft 0.19.1, flash-linear-attention 0.5.1) + pyyaml.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
from dataclasses import dataclass, field

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from . import olens  # type: ignore
except ImportError:  # scripts put src/ on sys.path
    import olens  # type: ignore

REPO = "agu18dec/qwen3.6-27b-all-layer-nla-rl2000"
D_MODEL = 5120
SLOT_ID, LEFT_ID, RIGHT_ID = 158983, 29, 510  # ㈜ between '>' and '</'
INJECT_LAYER = 1
TEMPERATURE, TOP_P, MAX_NEW = 1.0, 1.0, 200  # run_config.yaml: temperature 1.0, max_new_tokens 200
EXPL_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)


def load_meta(nla_dir: str) -> dict:
    """nla_meta.yaml (data): templates, marker ids. Checks they are the ones this module was written for."""
    m = yaml.safe_load(open(os.path.join(nla_dir, "nla_meta.yaml")))
    t = m["tokens"]
    if (t["injection_token_id"], t["injection_left_neighbor_id"], t["injection_right_neighbor_id"]) != \
            (SLOT_ID, LEFT_ID, RIGHT_ID) or t["injection_char"] != "㈜":
        raise ValueError(f"unexpected marker in nla_meta: {t}")
    if m["extraction"]["norm"] != "none" or m["extraction"]["layers"] != list(range(64)):
        raise ValueError("unexpected extraction block in nla_meta")
    return m


@dataclass
class AV:
    lm: nn.Module
    tok: AutoTokenizer
    actor: str
    nla_dir: str
    stats: dict = field(default_factory=lambda: {"inject": 0})


def load_av(nla_dir: str, device: str = "cuda", model: str = olens.MODEL) -> AV:
    meta = load_meta(nla_dir)
    path = os.path.join(nla_dir, "av")
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    if not (cfg["r"] == 128 and cfg["lora_alpha"] == 16 and cfg["use_rslora"] and not cfg["use_dora"]):
        raise ValueError(f"unexpected adapter config {cfg}")
    base = olens._base(model, device)
    lm = get_peft_model(base, LoraConfig(r=cfg["r"], lora_alpha=cfg["lora_alpha"], use_rslora=True,
                                         target_modules=cfg["target_modules"], lora_dropout=0.0, bias="none"))
    sd = load_file(os.path.join(path, "adapter_model.safetensors"))
    res = set_peft_model_state_dict(lm, sd)
    if res.unexpected_keys:
        raise KeyError(f"{len(res.unexpected_keys)} adapter tensors matched nothing, e.g. {res.unexpected_keys[:2]}")
    want = 1.0 * cfg["lora_alpha"] / math.sqrt(cfg["r"])
    lora = [m for m in lm.modules() if isinstance(m, LoraLayer)]
    scal = {m.scaling["default"] for m in lora}
    if len(lora) != sum("lora_A" in k for k in sd):
        raise ValueError(f"{len(lora)} LoRA layers vs {sum('lora_A' in k for k in sd)} in the file")
    if scal != {want}:
        raise ValueError(f"LoRA scaling {scal}, want rsLoRA {want}")
    for k, p in lm.named_parameters():  # adapter in fp32 on the bf16 base (it is stored fp32)
        if "lora_" in k:
            p.data = p.data.float()
    olens._check_live(lm, sum("lora_B" in k for k in sd), path)
    tok = AutoTokenizer.from_pretrained(model, **({"revision": olens.REVISION} if model == olens.HUB_MODEL else {}))
    return AV(lm.to(device).eval(), tok, meta["prompt_templates"]["actor"], nla_dir)


def merge(av: AV) -> AV:
    """Fold the LoRA into the bf16 base (reads only)."""
    av.lm = av.lm.merge_and_unload()
    return av


def prompt_ids(av: AV, layer: int) -> tuple[list[int], int]:
    text = av.tok.apply_chat_template([{"role": "user", "content": av.actor.format(layer=layer, injection_char="㈜")}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = av.tok(text, add_special_tokens=False).input_ids
    slots = [i for i, t in enumerate(ids) if t == SLOT_ID]
    if len(slots) != 1 or ids[slots[0] - 1] != LEFT_ID or ids[slots[0] + 1] != RIGHT_ID:
        raise ValueError(f"marker not well formed in the layer-{layer} prompt: {slots}")
    return ids, slots[0]


@contextlib.contextmanager
def inject(av: AV, vecs: torch.Tensor | None, slot: int, prefill_len: int):
    """Hook block INJECT_LAYER: out[:, slot] += ||out[:, slot]|| * v / ||v|| on the prefill forward only.
    vecs: [rows, 5120] (one per batch row) or None (no injection, for controls)."""
    block = olens._blocks(av.lm)[INJECT_LAYER]

    def hook(_m: nn.Module, _i: object, o: object):
        h = o[0] if isinstance(o, tuple) else o
        if vecs is None or h.shape[1] != prefill_len:  # decode steps (length 1) and anything else: untouched
            return None
        if h.shape[0] != vecs.shape[0]:
            raise RuntimeError(f"batch {h.shape[0]} vs {vecs.shape[0]} vectors")
        v = vecs.to(h.device, torch.float32)
        hp = h[:, slot].float()
        h = h.clone()
        h[:, slot] = (hp + hp.norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)).to(h.dtype)
        av.stats["inject"] += 1
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    handle = block.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def read(av: AV, h: torch.Tensor | None, layer: int, n: int = 1, temperature: float = TEMPERATURE,
         top_p: float = TOP_P, max_new: int = MAX_NEW, seed: int | None = None, rows: int | None = None,
         return_ids: bool = False):
    """n readouts of each block-``layer`` state in h [k, 5120] -> k lists of n raw replies (one batched generate).
    h=None reads the un-injected prompt (``rows`` copies), a control. temperature 0 = greedy."""
    if seed is not None:
        torch.manual_seed(seed)
    ids, slot = prompt_ids(av, layer)
    hs = None if h is None else h.float().reshape(-1, D_MODEL)
    k = rows if hs is None else len(hs)
    x = torch.tensor([ids], device=av.lm.device).repeat(k * n, 1)
    vecs = None if hs is None else hs.repeat_interleave(n, 0)
    before = av.stats["inject"]
    with inject(av, vecs, slot, len(ids)):
        out = av.lm.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new,
                             do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                             top_p=top_p if temperature > 0 else None, top_k=0, use_cache=True,
                             pad_token_id=av.tok.pad_token_id or av.tok.eos_token_id)
    if hs is not None and av.stats["inject"] - before != 1:
        raise RuntimeError(f"expected exactly one injected prefill, got {av.stats['inject'] - before}")
    new = out[:, len(ids):]
    texts = av.tok.batch_decode(new, skip_special_tokens=True)
    res = [texts[i * n:(i + 1) * n] for i in range(k)]
    return (res, new) if return_ids else res


@torch.no_grad()
def prefill_logits(av: AV, h: torch.Tensor, layer: int, cont: list[int] | None = None) -> torch.Tensor:
    """Logits [k, T, vocab] of the injected prompt (+ optional forced continuation, same for every row): the
    merged-vs-unmerged and injection-on/off checks."""
    ids, slot = prompt_ids(av, layer)
    full = ids + (cont or [])
    hs = h.float().reshape(-1, D_MODEL)
    x = torch.tensor([full], device=av.lm.device).repeat(len(hs), 1)
    with inject(av, hs, slot, len(full)):
        return av.lm(input_ids=x, use_cache=False).logits[:, len(ids) - 1:].float()


def explanation(text: str) -> str | None:
    m = EXPL_RE.search(text)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------- AR (activation reconstructor), validation only
@dataclass
class AR:
    lm: nn.Module
    head: torch.Tensor  # [5120, 5120] fp32, pred = head @ x
    tok: AutoTokenizer
    critic: str


def load_ar(nla_dir: str, device: str = "cuda", tok_path: str = olens.MODEL) -> AR:
    meta = load_meta(nla_dir)
    path = os.path.join(nla_dir, "ar")
    lm, info = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                    output_loading_info=True)
    miss = [k for k in info["missing_keys"] if "layers." in k or "embed" in k]
    if miss or info["unexpected_keys"] or info.get("mismatched_keys"):
        raise KeyError(f"AR trunk load: missing {miss[:3]} unexpected {info['unexpected_keys'][:3]} "
                       f"mismatched {info.get('mismatched_keys', [])[:3]}")
    lm.model.norm = nn.Identity()  # the AR reads the raw last-block output
    lm.requires_grad_(False)
    lm = lm.to(device).eval()
    head = load_file(os.path.join(path, "value_head.safetensors"))
    (w,) = head.values()
    if tuple(w.shape) != (D_MODEL, D_MODEL):
        raise ValueError(f"value head {w.shape}")
    tok = AutoTokenizer.from_pretrained(tok_path)
    return AR(lm, w.float().to(device), tok, meta["prompt_templates"]["critic"])


def _unit(v: torch.Tensor) -> torch.Tensor:  # EasyNLA normalize_activation with mse_scale sqrt(d)
    return v.float() * (math.sqrt(D_MODEL) / v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12))


@torch.no_grad()
def reconstruct(ar: AR, expls: list[str], layers: list[int]) -> torch.Tensor:
    """AR predictions [k, 5120] fp32 (normalised to norm sqrt(d)), one sequence per forward (no padding)."""
    out = []
    for e, L in zip(expls, layers):
        ids = ar.tok(ar.critic.format(layer=L, explanation=e), add_special_tokens=False).input_ids
        h = ar.lm.model(input_ids=torch.tensor([ids], device=ar.head.device), use_cache=False).last_hidden_state
        out.append(_unit(_unit(h[0, -1:]) @ ar.head.T))
    return torch.cat(out)


def fve(pred: torch.Tensor, gold: torch.Tensor, mu: torch.Tensor | None = None) -> tuple[float, torch.Tensor]:
    """EasyNLA paper-def FVE of predictions [k, d] for gold [k, d] (both rescaled to sqrt(d)); mu = the mean of the
    rescaled gold population (defaults to this sample's). Returns (FVE, per-row MSE)."""
    g = _unit(gold)
    mu = g.mean(0, keepdim=True) if mu is None else mu
    base = ((g - mu) ** 2).mean()
    mse = ((_unit(pred) - g) ** 2).mean(-1)
    return float(1 - mse.mean() / base), mse
