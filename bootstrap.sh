#!/usr/bin/env bash
set -euo pipefail

PROJECT=/workspace/model-diff
VENV=/root/model-diff-venv
SCRATCH=/root/model-diff-scratch
LOCK="$PROJECT/environment/venv.lock.txt"

echo "=== model-diff bootstrap ==="

cd "$PROJECT"

# ----------------------------------------------------------------------
# 1. Verify the base RunPod image before modifying anything.
# ----------------------------------------------------------------------

python - <<'PY'
import sys
import torch

expected_python = (3, 12, 3)
expected_torch = "2.8.0+cu128"
expected_cuda = "12.8"

actual_python = sys.version_info[:3]

if actual_python != expected_python:
    raise SystemExit(
        f"ERROR: Expected Python {'.'.join(map(str, expected_python))}, "
        f"found {'.'.join(map(str, actual_python))}"
    )

if torch.__version__ != expected_torch:
    raise SystemExit(
        f"ERROR: Expected PyTorch {expected_torch}, found {torch.__version__}"
    )

if torch.version.cuda != expected_cuda:
    raise SystemExit(
        f"ERROR: Expected PyTorch CUDA {expected_cuda}, found {torch.version.cuda}"
    )

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available")

print("Base environment verified:")
print("  Python:", ".".join(map(str, actual_python)))
print("  PyTorch:", torch.__version__)
print("  CUDA:", torch.version.cuda)
print("  GPU:", torch.cuda.get_device_name(0))

if torch.cuda.get_device_name(0) != "NVIDIA A40":
    print("WARNING: Original experiment used NVIDIA A40")
PY

# ----------------------------------------------------------------------
# 2. Install small system utilities.
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
# 3. Recreate ephemeral directories.
# ----------------------------------------------------------------------

mkdir -p \
    "$SCRATCH/models" \
    "$SCRATCH/artifacts" \
    "$SCRATCH/tmp" \
    /root/.cache/huggingface

# ----------------------------------------------------------------------
# 4. Recreate Python venv using RunPod's CUDA-enabled system PyTorch.
# ----------------------------------------------------------------------

if [[ ! -d "$VENV" ]]; then
    python -m venv --system-site-packages "$VENV"
fi

source "$VENV/bin/activate"

python -m pip install --upgrade \
    pip==26.2.1 \
    setuptools==84.0.0 \
    wheel==0.48.0

if [[ ! -f "$LOCK" ]]; then
    echo "ERROR: Missing dependency lock: $LOCK"
    exit 1
fi

pip install -r "$LOCK"

# ----------------------------------------------------------------------
# 5. Verify the important experiment libraries.
# ----------------------------------------------------------------------

python - <<'PY'
import torch
import transformers
import peft
import accelerate
import safetensors
import huggingface_hub

assert torch.__version__ == "2.8.0+cu128"
assert torch.version.cuda == "12.8"
assert torch.cuda.is_available()

print("Python environment ready:")
print("  torch:", torch.__version__)
print("  transformers:", transformers.__version__)
print("  peft:", peft.__version__)
print("  accelerate:", accelerate.__version__)
print("  safetensors:", safetensors.__version__)
print("  huggingface_hub:", huggingface_hub.__version__)
print("  GPU:", torch.cuda.get_device_name(0))
PY

# ----------------------------------------------------------------------
# 6. Install Codex CLI if this is a fresh container.
# ----------------------------------------------------------------------

export PATH="/root/.local/bin:$PATH"

if ! command -v codex >/dev/null 2>&1; then
    echo "Installing Codex CLI..."
    curl -fsSL https://chatgpt.com/codex/install.sh | sh
fi

export PATH="/root/.local/bin:$PATH"

echo
echo "Codex: $(codex --version)"

# ----------------------------------------------------------------------
# 7. Restore our standard environment helper.
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

export TMPDIR=/root/model-diff-scratch/tmp
export PATH="/root/.local/bin:$PATH"

cd /workspace/model-diff
ENVEOF

chmod +x "$PROJECT/env.sh"

echo
echo "======================================"
echo "model-diff environment ready"
echo "======================================"
echo
echo "Activate with:"
echo "  source /workspace/model-diff/env.sh"
echo
echo "Ephemeral model storage:"
echo "  $SCRATCH"
echo
