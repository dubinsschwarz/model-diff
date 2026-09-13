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
