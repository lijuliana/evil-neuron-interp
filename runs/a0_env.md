# A0 environment record (2026-10-28)

## GCP VM `weirdchat-a100`
- zone us-central1-a, machine a2-ultragpu-1g, 1x NVIDIA A100-SXM4-80GB (driver 580.178.04, CUDA 13.0 driver / 12.9 toolkit at /usr/local/cuda-12.9), 12 vCPU, 167 GB RAM, 387 GB pd-ssd root.
- image pytorch-2-9-cu129-ubuntu-2404; system python 3.12.3 had no torch/transformers, so everything lives in `~/venv` (uv 0.13.6).
- Setup script: `scripts/00_a0_vm_setup.sh` (idempotent). Repo mirror on VM: `~/evil-neuron-interp/{src,scripts}`.
- Data on VM: `/data/models/Qwen3.6-27B` (52 GB bf16, 15 safetensors shards, checkpoint transformers_version 4.57.1), `/data/weirdchat/{patterns,rubrics}.parquet`, `/data/weirdchat/{prompts,transcripts}/qwen__qwen3.6-27b/`.
- Outputs on VM: `/data/captures/s1/` (S1 npz), `/data/runs/` (logs, baseline jsonl).

## Python packages (`~/venv`, Python 3.12.3)
| package | version |
|---|---|
| torch | 2.9.1+cu129 (wheel from download.pytorch.org/whl/cu129) |
| transformers | 5.21.0 (native `Qwen3_5ForConditionalGeneration`; no trust_remote_code) |
| flash-linear-attention (fla) | 0.6.0 (Triton 3.5.1) -- needed for the 48 GatedDeltaNet layers |
| causal-conv1d | 1.7.0 (built with `--no-build-isolation`, CUDA_HOME=/usr/local/cuda) |
| accelerate | 1.15.0 |
| safetensors | 0.8.0 |
| pandas / pyarrow / numpy / scipy | 3.1.0 / 26.0.0 / 2.5.4 / 1.18.1 |
| anthropic | 2.3.0 |
| huggingface_hub | 2.3.0 |

## Model facts verified on the VM
- Load: 14.8 s, 54.7 GB GPU memory in bf16 (vision tower included, never used).
- Text model: `model.model.language_model`, 64 `Qwen3_5DecoderLayer` (48 `linear_attn` GatedDeltaNet + 16 `self_attn`, every 4th), `mlp.{gate,up,down}_proj`, down_proj.weight [5120, 17408].
- Tokenizer: pad `<|endoftext|>` 248044, eos `<|im_end|>` 248046; generation eos = [248046, 248044].
- Chat template with `enable_thinking=False` renders `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`; no default system prompt is injected.
- Sampling defaults used: temperature 1.0, top_p 0.95, top_k 20 (generation_config.json), max_new_tokens 400, no system prompt.

## Throughput (A100-80GB, bf16, HF eager)
- Generation: 477 tok/s at batch 64 x 128 new tokens (288 tok/s before causal-conv1d was installed). Single sequence first call is dominated by Triton JIT.
- Teacher-forced capture with the 64-layer hook: 4.0 transcripts/s, ~1370 tok/s at batch 16 (peak 66 GB). Full S1 (~27k transcripts) ~2 h.

## Numerics note
bf16 activations are shape-dependent: the same sequence in a different padded length moves ~0.5% of neurons by >5% relative (corr 0.9998), mainly late layers. Same shape -> bit-identical. `capture_batch` pads T to a multiple of 64 to limit shape diversity; the shuffled-label control absorbs this as noise.
