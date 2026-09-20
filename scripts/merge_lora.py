#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL_DIR = Path("/root/model-diff-scratch/models/base")
DEFAULT_ADAPTER_MODEL_DIR = Path("/root/model-diff-scratch/models/adapter")
DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")


def model_path(environment_variable: str, default: Path) -> Path:
    return Path(os.environ.get(environment_variable, default)).expanduser()


def load_required_json(path: Path, description: str) -> dict[str, Any]:
    try:
        is_file = path.is_file()
    except OSError as exc:
        raise ValueError(f"Could not access {description} {path}: {exc}") from exc

    if not is_file:
        raise ValueError(f"Missing {description}: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {description}: {path}")

    return data


def require_directory(path: Path, description: str) -> None:
    try:
        is_directory = path.is_dir()
    except OSError as exc:
        raise ValueError(f"Could not access {description} {path}: {exc}") from exc

    if not is_directory:
        raise ValueError(f"Missing {description}: {path}")


def validate_paths(base_dir: Path, adapter_dir: Path, output_dir: Path) -> None:
    require_directory(base_dir, "base model directory")
    require_directory(adapter_dir, "adapter model directory")

    load_required_json(base_dir / "config.json", "base model config")
    adapter_config = load_required_json(
        adapter_dir / "adapter_config.json", "adapter config"
    )

    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError(
            f"Adapter config is not a LoRA adapter: {adapter_dir / 'adapter_config.json'}"
        )

    try:
        output_exists = output_dir.exists()
    except OSError as exc:
        raise ValueError(f"Could not access output path {output_dir}: {exc}") from exc

    if output_exists:
        try:
            output_is_directory = output_dir.is_dir()
        except OSError as exc:
            raise ValueError(
                f"Could not inspect output path {output_dir}: {exc}"
            ) from exc
        if not output_is_directory:
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
        try:
            output_has_files = next(output_dir.iterdir(), None) is not None
        except OSError as exc:
            raise ValueError(
                f"Could not inspect output directory {output_dir}: {exc}"
            ) from exc
        if output_has_files:
            raise ValueError(f"Output directory is not empty: {output_dir}")


def verify_float32_parameters(model: Any, torch_module: Any, description: str) -> Any:
    floating_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    unexpected_dtypes = sorted(
        str(dtype) for dtype in floating_dtypes if dtype != torch_module.float32
    )

    if unexpected_dtypes:
        raise ValueError(
            f"{description} has unexpected floating-point parameter dtype(s): "
            f"{', '.join(unexpected_dtypes)}"
        )
    if not floating_dtypes:
        raise ValueError(f"{description} has no floating-point parameters")

    return next(iter(floating_dtypes))


def merge_lora(base_dir: Path, adapter_dir: Path, output_dir: Path) -> None:
    # Keep every Hugging Face load local even if a path is malformed later.
    os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Base model: {base_dir}")
    print(f"Adapter: {adapter_dir}")
    print(f"Output: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_dir,
        local_files_only=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        dtype=torch.float32,
        local_files_only=True,
    )
    base_dtype = verify_float32_parameters(base_model, torch, "Base model")
    print(f"Base floating-point dtype: {base_dtype} (verified)")

    peft_model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        local_files_only=True,
    )
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    merged_dtype = verify_float32_parameters(merged_model, torch, "Merged model")
    print(f"Merged floating-point dtype: {merged_dtype} (verified)")

    parameter_count = sum(parameter.numel() for parameter in merged_model.parameters())
    print(f"Total parameters: {parameter_count:,}")

    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def main() -> int:
    base_dir = model_path("BASE_MODEL_DIR", DEFAULT_BASE_MODEL_DIR)
    adapter_dir = model_path("ADAPTER_MODEL_DIR", DEFAULT_ADAPTER_MODEL_DIR)
    output_dir = model_path("MERGED_MODEL_DIR", DEFAULT_MERGED_MODEL_DIR)

    try:
        validate_paths(base_dir, adapter_dir, output_dir)
        merge_lora(base_dir, adapter_dir, output_dir)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
