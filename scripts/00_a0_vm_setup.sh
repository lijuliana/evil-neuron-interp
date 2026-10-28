#!/usr/bin/env bash
# A0: one-shot VM setup for weirdchat-a100. Idempotent. Run as the login user.
# Usage (on VM): bash ~/00_a0_vm_setup.sh
set -euo pipefail
sudo mkdir -p /data/models /data/weirdchat /data/hf_cache /data/captures /data/runs
sudo chown -R "$USER" /data
# uv
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME"
if [ ! -d "$HOME/venv" ]; then uv venv --python 3.12 "$HOME/venv"; fi
source "$HOME/venv/bin/activate"
# torch cu129 wheel matching the image's CUDA 12.9
uv pip install --index-url https://download.pytorch.org/whl/cu129 "torch==2.9.*"
uv pip install "transformers>=4.57" accelerate safetensors pandas pyarrow numpy anthropic "huggingface_hub[hf_transfer,cli]" scipy tqdm python-dotenv
python - <<'PY'
import torch, transformers, sys
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
ok = hasattr(transformers, "Qwen3_5ForConditionalGeneration") or "qwen3_5" in str(transformers.models.auto.configuration_auto.CONFIG_MAPPING_NAMES.keys())
print("qwen3_5 supported:", ok)
sys.exit(0 if ok else 3)
PY
