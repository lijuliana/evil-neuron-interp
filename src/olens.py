"""Oracle-lens (O-lens) reader for Qwen3.6-27B. Adapted from agu18dec/final_olens's olens_lensB.py so the
same loader also reads agu18dec/olens_and_ar's s3d lens (WorkspaceBench's O-lens, mirrored byte-for-byte at
neuronpedia/jacobian-lens qwen3.6-27b/oracle, and the lens behind Agastya's layer-56 viewer), whose
model-card code needs the unreleased `global_workspace` package.

    from src.olens import load_lens, residual, read
    lens = load_lens("s3d_rl600")
    h = residual(lens, "Who wrote the opera Carmen?", layer=40)
    print(read(lens, h, layer=40, n=4))

Both lenses are r16/alpha32 LoRAs on every linear layer of Qwen3.6-27B, with the same verbalizer prompt and
the same marker token. They differ in injected norm, trained layers, weight format and the sampling their
authors read with, so each entry of LENSES is a whole contract: reading one checkpoint with the other's
alpha or layers is a different experiment, not a noisier one.

Conventions (both lenses):
- The residual at layer L is the output of decoder block L (zero-indexed), before the final norm, captured
  with the adapter disabled. Reading the lens's own activations is a silent, plausible-looking error.
- The prompt is one user turn rendered with the chat template, add_generation_prompt=True and
  enable_thinking=False. Its only marker is the token ㈜ (id 158983), whose input embedding is replaced by
  alpha * h / ||h||. The s3d card picks the marker by scanning U+3200..U+33FF for a char that stays one
  token in the full render; with the Qwen3.6 tokenizer that scan lands on 158983 at every layer 0-63.
- Load the base with AutoModelForCausalLM (blocks at model.layers.N), as both adapters were saved. On
  harness.py's ForConditionalGeneration wrapper (model.language_model.layers.N) the keys match nothing.
  load_lens raises if any LoRA B matrix is missing or still zero after loading.
- Sampling defaults: s3d uses Agastya's viewer setting (temperature 0.8, top-p 0.95, cap 256), the
  baseline we validate against; lens B uses its card's (temperature 1, top-p 1, cap 128). top_k is
  always 0 so the checkpoint's generation_config top_k=20 never applies.

Pins (separate env from the steering harness): torch 2.9.1, transformers 5.11.0, peft 0.19.1,
flash-linear-attention 0.5.1. Needs ~60 GB of GPU memory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB_MODEL = "Qwen/Qwen3.6-27B"
REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"  # final_olens's pin
MODEL = os.environ.get("QWEN36_PATH", HUB_MODEL)
D_MODEL = 5120
SLOT_ID = 158983  # ㈜
PROMPT = (
    "An activation vector from layer {layer} of a language model is enclosed in activation "
    "tags: <activation>㈜</activation>. Produce distinct concepts that encode this activation, "
    "each as a '- ' bullet on its own line."
)


@dataclass(frozen=True)
class Contract:
    repo: str
    sub: str
    peft_dir: bool  # PEFT adapter dir (olens_and_ar) vs a final_olens model.safetensors
    alpha: float
    layers: tuple[int, ...]
    temperature: float
    top_p: float
    max_new: int


S3D_LAYERS = tuple(range(20, 61, 4))
LENSES = {
    # WorkspaceBench .726 (4 bullets); same file as the Neuronpedia mirror.
    "s3d_rl600": Contract("agu18dec/olens_and_ar", "olens_s3d_rl600", True, 16000.0, S3D_LAYERS,
                          0.8, 0.95, 256),
    # same recipe with bullets scored as " " + bullet: .728, Chinese-prefixed bullets 36% -> 3%
    "s3d_space_rl600": Contract("agu18dec/olens_and_ar", "olens_s3d_space_rl600", True, 16000.0,
                                S3D_LAYERS, 0.8, 0.95, 256),
    # final_olens lens B, any layer: .683 at step 600, .673 at step 300 (fewer hallucinations)
    "lensB_rl600": Contract("agu18dec/final_olens", "lens_rl_step600", False, 1000.0,
                            tuple(range(64)), 1.0, 1.0, 128),
    "lensB_rl300": Contract("agu18dec/final_olens", "lens_rl_step300", False, 1000.0,
                            tuple(range(64)), 1.0, 1.0, 128),
}


@dataclass
class Lens:
    lm: PeftModel
    tok: AutoTokenizer
    contract: Contract
    name: str


def _base(model: str, device: str) -> nn.Module:
    kw = {"revision": REVISION} if model == HUB_MODEL else {}
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                              **kw)
    lm.config.use_cache = False  # plain forwards; read() turns the cache on for generate
    lm.requires_grad_(False)
    return lm.to(device)


def _check_live(lm: nn.Module, n_b: int, path: str) -> None:
    """Every LoRA B starts at zero, so a B that is still zero (or a count mismatch) means the adapter
    matched nothing and would read as fluent, activation-blind text."""
    bs = [(k, p) for k, p in lm.named_parameters() if "lora_B" in k]
    dead = [k for k, p in bs if not p.detach().abs().sum()]
    if len(bs) != n_b or dead:
        raise RuntimeError(f"{path}: {len(bs)} LoRA B in model vs {n_b} in file, {len(dead)} still zero, "
                           f"e.g. {dead[:2]}")


def load_lens(name: str = "s3d_rl600", device: str = "cuda", model: str = MODEL) -> Lens:
    c = LENSES[name]
    path = os.path.join(snapshot_download(c.repo, allow_patterns=[f"{c.sub}/*"]), c.sub)
    base = _base(model, device)
    if c.peft_dir:
        lm = PeftModel.from_pretrained(base, path)
        n_b = sum("lora_B" in k for k in load_file(os.path.join(path, "adapter_model.safetensors")))
    else:
        lm = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                             target_modules="all-linear", bias="none"))
        sd = {k.removeprefix("lm."): v for k, v in load_file(os.path.join(path, "model.safetensors")).items()
              if not k.startswith("injector.")}
        missing = {k for k, _ in lm.named_parameters() if "lora_" in k} - sd.keys()
        if missing:
            raise KeyError(f"{path}: {len(missing)} LoRA tensors missing, e.g. {sorted(missing)[:2]}")
        lm.load_state_dict(sd, strict=False)
        n_b = sum("lora_B" in k for k in sd)
    for k, p in lm.named_parameters():  # adapters in fp32 on the bf16 base, as trained
        if "lora_" in k:
            p.data = p.data.float()
    _check_live(lm, n_b, path)
    tok = AutoTokenizer.from_pretrained(model, **({"revision": REVISION} if model == HUB_MODEL else {}))
    return Lens(lm.to(device).eval(), tok, c, name)


def _blocks(model: nn.Module) -> nn.ModuleList:
    for m in model.modules():
        if isinstance(getattr(m, "layers", None), nn.ModuleList) and hasattr(m, "norm"):
            return m.layers
    raise ValueError("no decoder layers found")


@torch.no_grad()
def capture(lens: Lens, ids: list[int], layers: list[int], positions: list[int]) -> dict[int, torch.Tensor]:
    """Base-model residuals (block outputs, adapter off) at ``positions`` of one token sequence:
    {layer: [n_pos, 5120]} fp32 on CPU."""
    out: dict[int, torch.Tensor] = {}
    blocks, pos = _blocks(lens.lm), torch.tensor(positions)

    def grab(layer: int):
        def hook(_m: nn.Module, _i: object, o: object) -> None:  # returning a value would replace the output
            h = o[0] if isinstance(o, tuple) else o
            out[layer] = h[0, pos.to(h.device)].float().cpu()
        return hook

    handles = [blocks[L].register_forward_hook(grab(L)) for L in layers]
    try:
        with lens.lm.disable_adapter():
            lens.lm(input_ids=torch.tensor([ids], device=lens.lm.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return out


def residual(lens: Lens, user_text: str, layer: int, position: int = -1,
             assistant_prefix: str = "") -> torch.Tensor:
    """final_olens's helper: the residual at ``position`` of a rendered single-turn chat."""
    text = lens.tok.apply_chat_template([{"role": "user", "content": user_text}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    ids = lens.tok(text + assistant_prefix, add_special_tokens=False).input_ids
    return capture(lens, ids, [layer], [position % len(ids)])[layer][0]


def _prompt(lens: Lens, layer: int) -> tuple[list[int], int]:
    text = lens.tok.apply_chat_template([{"role": "user", "content": PROMPT.format(layer=layer)}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = lens.tok(text, add_special_tokens=False).input_ids
    slots = [i for i, t in enumerate(ids) if t == SLOT_ID]
    if len(slots) != 1:
        raise ValueError(f"marker {SLOT_ID} appears {len(slots)} times in the layer-{layer} prompt")
    return ids, slots[0]


@torch.no_grad()
def read(lens: Lens, h: torch.Tensor, layer: int, n: int = 1, temperature: float | None = None,
         top_p: float | None = None, max_new: int | None = None, seed: int | None = None,
         off_contract: bool = False) -> list[str] | list[list[str]]:
    """``n`` readouts of residual ``h`` taken at ``layer``. ``h`` is [5120] (returns n strings, as
    final_olens) or [k, 5120] (returns k lists of n, one batched generate)."""
    c = lens.contract
    if layer not in c.layers and not off_contract:
        raise ValueError(f"layer {layer} is outside {lens.name}'s trained layers {c.layers}")
    temperature = c.temperature if temperature is None else temperature
    top_p = c.top_p if top_p is None else top_p
    if seed is not None:
        torch.manual_seed(seed)
    hs = h.float().reshape(-1, D_MODEL)
    ids, slot = _prompt(lens, layer)
    ids = torch.tensor([ids], device=lens.lm.device).repeat(len(hs) * n, 1)
    emb = lens.lm.get_input_embeddings()(ids)
    unit = hs / hs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    emb[:, slot] = (c.alpha * unit).repeat_interleave(n, 0).to(emb.device, emb.dtype)
    out = lens.lm.generate(inputs_embeds=emb, attention_mask=torch.ones_like(ids),
                           max_new_tokens=max_new or c.max_new, do_sample=temperature > 0,
                           temperature=temperature if temperature > 0 else None, top_p=top_p, top_k=0,
                           use_cache=True, pad_token_id=lens.tok.pad_token_id or lens.tok.eos_token_id)
    texts = lens.tok.batch_decode(out, skip_special_tokens=True)
    if h.dim() == 1:
        return texts
    return [texts[i * n:(i + 1) * n] for i in range(len(hs))]


if __name__ == "__main__":  # smoke test from final_olens's quickstart: python -m src.olens s3d_rl600
    import sys

    lens = load_lens(sys.argv[1] if len(sys.argv) > 1 else "s3d_rl600")
    h = residual(lens, "Who wrote the opera Carmen?", layer=40)
    for r in read(lens, h, layer=40, n=4, seed=0):
        print("---\n" + r)
