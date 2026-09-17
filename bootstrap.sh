#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/model-diff
VENV=/root/model-diff-venv
SCRATCH=/root/model-diff-scratch
LOCK="$PROJECT/environment/venv.lock.txt"

TORCH_VERSION="2.8.0"
TORCHVISION_VERSION="0.23.0"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

cd "$PROJECT"

echo "=== model-diff bootstrap ==="

# ----------------------------------------------------------------------
# 1. Check host prerequisites.
# ----------------------------------------------------------------------

python - <<'PY'
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"ERROR: Python 3.12.x required; found {sys.version.split()[0]}"
    )

print("Host Python:", sys.version.split()[0])
PY

command -v nvidia-smi >/dev/null || {
    echo "ERROR: nvidia-smi not found" >&2
    exit 1
}

nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader

# ----------------------------------------------------------------------
# 2. Small system dependencies.
# ----------------------------------------------------------------------

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
    git \
    git-lfs \
    curl \
    wget \
    tmux \
    htop \
    build-essential

git lfs install

# ----------------------------------------------------------------------
# 3. Ephemeral experiment storage.
# ----------------------------------------------------------------------

mkdir -p \
    "$SCRATCH/models" \
    "$SCRATCH/artifacts" \
    "$SCRATCH/tmp" \
    /root/.cache/huggingface \
    /root/.cache/torch \
    /root/.cache/pip

# ----------------------------------------------------------------------
# 4. Self-contained Python environment.
# ----------------------------------------------------------------------

if [[ ! -d "$VENV" ]]; then
    echo "Creating venv..."
    python -m venv "$VENV"
fi

source "$VENV/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo
echo "Installing pinned CUDA/PyTorch stack..."

python -m pip install \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    --index-url "$PYTORCH_INDEX"

if [[ ! -f "$LOCK" ]]; then
    echo "ERROR: missing dependency lock: $LOCK" >&2
    exit 1
fi

echo
echo "Installing experiment dependencies..."
python -m pip install -r "$LOCK"

# ----------------------------------------------------------------------
# 5. Fail-fast verification.
# ----------------------------------------------------------------------

python - <<'PY'
import torch
import torchvision
import transformers
import peft
import datasets
import accelerate
import safetensors
import huggingface_hub

expected = {
    "torch": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "transformers": "5.17.0",
    "peft": "0.20.0",
    "datasets": "5.0.1",
}

actual = {
    "torch": torch.__version__,
    "torchvision": torchvision.__version__,
    "transformers": transformers.__version__,
    "peft": peft.__version__,
    "datasets": datasets.__version__,
}

for name, expected_version in expected.items():
    if actual[name] != expected_version:
        raise SystemExit(
            f"ERROR: {name}: expected {expected_version}, "
            f"found {actual[name]}"
        )

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")

print()
print("Verified experiment environment:")
for name, version in actual.items():
    print(f"  {name}: {version}")

print("  accelerate:", accelerate.__version__)
print("  safetensors:", safetensors.__version__)
print("  huggingface_hub:", huggingface_hub.__version__)
print("  CUDA:", torch.version.cuda)
print("  GPU:", torch.cuda.get_device_name(0))
PY

# ----------------------------------------------------------------------
# 6. Codex CLI.
# ----------------------------------------------------------------------

export PATH="/root/.local/bin:$PATH"

if ! command -v codex >/dev/null 2>&1; then
    echo
    echo "Installing Codex CLI..."
    curl -fsSL https://chatgpt.com/codex/install.sh | sh
fi

export PATH="/root/.local/bin:$PATH"

# ----------------------------------------------------------------------
# 7. Standard environment helper.
# ----------------------------------------------------------------------

cat > "$PROJECT/env.sh" <<'ENVEOF'
#!/usr/bin/env bash

source /root/model-diff-venv/bin/activate

export MODEL_DIFF_ROOT=/workspace/model-diff
export MODEL_DIFF_SCRATCH=/root/model-diff-scratch

export MODEL_DIR=/root/model-diff-scratch/models
export ARTIFACT_DIR=/root/model-diff-scratch/artifacts

export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
export HF_DATASETS_CACHE=/root/.cache/huggingface/datasets
export TORCH_HOME=/root/.cache/torch
export PIP_CACHE_DIR=/root/.cache/pip

export TMPDIR=/root/model-diff-scratch/tmp
export PATH="/root/.local/bin:$PATH"

cd /workspace/model-diff
ENVEOF

chmod +x "$PROJECT/env.sh"

echo
echo "======================================"
echo "model-diff environment ready"
echo "======================================"
echo "Run:"
echo "  source /workspace/model-diff/env.sh"
