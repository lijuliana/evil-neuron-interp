#!/usr/bin/env bash
# A4b: one-shot setup for AWS weirdchat-4b (g5.2xlarge, A10G 24 GB, DLAMI PyTorch 2.7). Idempotent.
# Usage (on VM): nohup bash ~/00_a4b_vm_setup.sh > ~/setup.log 2>&1 &
set -euo pipefail
mkdir -p ~/data/models ~/data/weirdchat/prompts ~/data/runs ~/data/captures ~/evil-neuron-interp/src ~/evil-neuron-interp/scripts
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
# The AMI's /opt/pytorch has torch 2.7.0+cu128; A0's working recipe for qwen3_5 (fla + triton 3.5) needs torch 2.9,
# so build a clean venv like A0 (cu129 wheel; driver 580 supports it).
if [ ! -d "$HOME/venv" ]; then uv venv --python 3.12 "$HOME/venv"; fi
source "$HOME/venv/bin/activate"
uv pip install --index-url https://download.pytorch.org/whl/cu129 "torch==2.9.*"
uv pip install "transformers>=4.57" accelerate safetensors pandas pyarrow numpy anthropic "huggingface_hub[hf_transfer,cli]" scipy tqdm python-dotenv
uv pip install flash-linear-attention || echo "fla install failed (slow torch fallback will be used)"
python - <<'PY'
import torch, transformers, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
try:
    import fla; print("fla", fla.__version__)
except Exception as e: print("fla missing:", e)
import triton; print("triton", triton.__version__)
print("qwen3_5 supported:", hasattr(transformers, "Qwen3_5ForConditionalGeneration"))
PY
# model download (bf16, ~8 GB)
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3.5-4B --local-dir ~/data/models/Qwen3.5-4B --exclude "*.gguf" || huggingface-cli download Qwen/Qwen3.5-4B --local-dir ~/data/models/Qwen3.5-4B
ls -la ~/data/models/Qwen3.5-4B
echo SETUP_DONE
