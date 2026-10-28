"""Thin wrapper around A0's src/harness.py for Qwen3.5-4B (A4b, AWS g5.2xlarge A10G).

harness.py is written for Qwen3.6-27B and hardcodes N_LAYERS=64, D_MLP=17408, D_MODEL=5120 as module
globals (asserted in Harness.__init__, used by GroupAcc / capture_batch). Qwen3.5-4B is the same
`Qwen3_5ForConditionalGeneration` class (text tower `qwen3_5_text`, 32 layers, hidden 2560,
intermediate 9216, 8 full-attention + 24 GatedDeltaNet layers), so the only change needed is to
patch those globals before constructing the Harness. Everything else (chat template with
enable_thinking=False, no system prompt; steer = dose * median_resid_norm[L]/||w|| * w on
down_proj OUTPUT; clamp / ablate_mean on down_proj INPUT; capture) is reused unchanged.

Additions here (requested changes for harness.py, not applied to A0's file):
  * `chat_full` returns finish_reason ("stop" | "length") and the number of new tokens per sample,
    which harness.chat discards.
  * `calibrate` writes captures/qwen3.5-4b_resid_norms_bf16.json with the right model name
    (harness.calibrate hardcodes "Qwen/Qwen3.6-27B" in the JSON) and can compare against the MLX
    8-bit calibration in captures/qwen3.5-4b_resid_norms.json.
  * eos ids are read from the tokenizer instead of the hardcoded [248046, 248044] (they are the same
    for the 4B vocab; asserted at load).

Sampling: harness.chat defaults to top_p=0.95, top_k=20 (what a vLLM server applies from the 27B
generation_config, which is what WeirdChat sampled through). The Qwen3.5-4B repo ships NO
generation_config.json, so pass top_p / top_k explicitly. Pure T=1 is top_p=1.0, top_k=0.
"""
from __future__ import annotations

import json
import os
import time
from typing import Sequence

import torch

import harness as H

N_LAYERS_4B, D_MLP_4B, D_MODEL_4B = 32, 9216, 2560
DEFAULT_MODEL_PATH_4B = os.environ.get("QWEN35_4B_PATH", os.path.expanduser("~/data/models/Qwen3.5-4B"))


def _patch_globals():
    H.N_LAYERS, H.D_MLP, H.D_MODEL = N_LAYERS_4B, D_MLP_4B, D_MODEL_4B


class Harness4B(H.Harness):
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH_4B, device: str = "cuda",
                 resid_norms_path: str | None = None):
        _patch_globals()
        super().__init__(model_path, device=device, resid_norms_path=resid_norms_path)
        self.n_layers = len(self.layers)
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        eot = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.eos_ids = [im_end, eot]
        assert self.eos_ids == [248046, 248044], self.eos_ids  # harness.chat hardcodes these
        assert self.tok.pad_token_id == eot, self.tok.pad_token_id

    @torch.no_grad()
    def chat_full(self, prompts: Sequence[str | Sequence[dict]], n: int = 1, max_new_tokens: int = 300,
                  temperature: float = 1.0, top_p: float = 0.95, top_k: int = 20,
                  batch_size: int = 64, seed: int | None = None, verbose: bool = True
                  ) -> list[list[dict]]:
        """Like harness.chat but returns [len(prompts)][n] dicts {response, finish_reason, n_tokens}."""
        if seed is not None:
            torch.manual_seed(seed)
        items = [(pi, self.prompt_ids(p)) for pi, p in enumerate(prompts) for _ in range(n)]
        out: list[list[dict]] = [[] for _ in prompts]
        pad = self.tok.pad_token_id
        total_new, t_all = 0, time.time()
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                          pad_token_id=pad, eos_token_id=self.eos_ids)
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
            new = gen[:, L:].cpu()
            dt = time.time() - t0
            n_new_batch = 0
            for b, (pi, _) in enumerate(chunk):
                row = new[b].tolist()
                # first eos (im_end or endoftext) ends the sample; everything after is padding
                cut = len(row)
                for j, t in enumerate(row):
                    if t in self.eos_ids:
                        cut = j
                        break
                finish = "stop" if cut < len(row) else "length"
                n_new_batch += cut
                out[pi].append({"response": self.tok.decode(row[:cut], skip_special_tokens=True),
                                "finish_reason": finish, "n_tokens": cut})
            total_new += n_new_batch
            if verbose:
                print(f"[chat_full] batch {s//batch_size}: {len(chunk)} seqs, {n_new_batch} new tokens, "
                      f"{n_new_batch/dt:.1f} tok/s ({dt:.0f}s)", flush=True)
        if verbose:
            print(f"[chat_full] total {total_new} tokens in {time.time()-t_all:.1f}s = "
                  f"{total_new/(time.time()-t_all):.1f} tok/s", flush=True)
        return out

    @torch.no_grad()
    def calibrate(self, prompts: Sequence[str] = H.CALIBRATION_PROMPTS, max_new_tokens: int = 200,
                  save_path: str | None = "captures/qwen3.5-4b_resid_norms_bf16.json",
                  compare_path: str | None = "captures/qwen3.5-4b_resid_norms.json") -> list[float]:
        med = super().calibrate(prompts, max_new_tokens=max_new_tokens, save_path=None)
        rec = {"model": "Qwen/Qwen3.5-4B", "dtype": "bfloat16", "device": torch.cuda.get_device_name(0),
               "definition": "median over tokens (excluding token 0) of ||resid|| leaving model.layers[i], "
                             "over 6 ordinary chats (prompt + sampled response, top_p .95 top_k 20 T=1)",
               "n_chats": len(prompts), "prompts": list(prompts), "median_resid_norm": med}
        if compare_path and os.path.exists(compare_path):
            mlx = json.load(open(compare_path))["median_resid_norm"]
            mlx = [float(mlx[str(i)]) for i in range(len(med))]
            ratio = [m / x for m, x in zip(med, mlx)]
            rec["mlx8bit_median_resid_norm"] = mlx
            rec["bf16_over_mlx8bit"] = ratio
            rec["bf16_over_mlx8bit_summary"] = {"min": min(ratio), "max": max(ratio),
                                                "mean": sum(ratio) / len(ratio)}
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            json.dump(rec, open(save_path, "w"), indent=1)
        return med
