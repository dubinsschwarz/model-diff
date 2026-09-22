#!/usr/bin/env python3

"""Construct Attempt 003 with matched-strength Gaussian weight noise."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = (
    PROJECT
    / "experiments"
    / "attempts"
    / "003_gaussian_noise_prefix0_13_r00125"
    / "spec.json"
)
DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_OUTPUT_ROOT = Path(
    "/root/model-diff-scratch/ablations/003_gaussian_noise_prefix0_13_r00125"
)

ALLOWED_ENVIRONMENT_VARIABLES = {
    "MERGED_MODEL_DIR",
    "ABLATION_SPEC_PATH",
    "ABLATION_OUTPUT_ROOT",
}

FROZEN_SPEC = {
    "format_version": 1,
    "attempt_id": "003_gaussian_noise_prefix0_13_r00125",
    "method": "matched_strength_gaussian_weight_noise",
    "information_policy": "merged_checkpoint_only",
    "defaults": {
        "merged_model_directory": "/root/model-diff-scratch/models/merged",
        "output_root": (
            "/root/model-diff-scratch/ablations/003_gaussian_noise_prefix0_13_r00125"
        ),
    },
    "transformer_block_indices": list(range(14)),
    "expected_matrix_count": 98,
    "eligible_tensors": {
        "module_type": "torch.nn.Linear",
        "parameter": "weight",
        "scope": "recursive_descendants_of_model.model.layers[0:14]",
        "exclusions": [
            "norms",
            "embeddings",
            "lm_head",
            "biases",
            "blocks_14_through_27",
        ],
    },
    "source_weight_dtype": "float32",
    "seed": 1729,
    "target_relative_frobenius": 0.00125,
    "randomness": {
        "distribution": "iid_standard_gaussian",
        "generator": "single_torch_generator",
        "generator_device": "cpu",
        "generator_dtype": "float64",
        "module_ordering": "lexicographic_by_module_name",
        "seed_derivation": "fixed_seed_only",
    },
    "weight_transform": (
        "W_new = float32(float64(W) + 0.00125 * ||float64(W)||_F * "
        "(Z_float64 / ||Z_float64||_F))"
    ),
    "outputs": {
        "checkpoint_dtype": "float32",
        "standalone_checkpoint": True,
        "directory_name": "model",
        "construction_manifest": "construction-manifest.json",
        "timestamps": False,
        "host_metadata": False,
        "gpu_metadata": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Malformed SHA-256 for {description}")
    return value


def require_file(path: Path, description: str) -> None:
    try:
        is_file = path.is_file()
    except OSError as exc:
        raise ValueError(f"Could not access {description} {path}: {exc}") from exc
    if not is_file:
        raise ValueError(f"Missing {description}: {path}")


def require_directory(path: Path, description: str) -> None:
    try:
        is_directory = path.is_dir()
    except OSError as exc:
        raise ValueError(f"Could not access {description} {path}: {exc}") from exc
    if not is_directory:
        raise ValueError(f"Missing {description}: {path}")


def path_exists(path: Path, description: str) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        raise ValueError(f"Could not inspect {description} {path}: {exc}") from exc


def load_spec(path: Path) -> dict[str, Any]:
    require_file(path, "attempt specification")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read attempt specification {path}: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError("Attempt specification must be a JSON object")
    if spec != FROZEN_SPEC:
        raise ValueError("Attempt specification does not match the frozen definition")
    return spec


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_destinations(
    merged_dir: Path,
    spec_path: Path,
    output_root: Path,
    spec: dict[str, Any],
) -> tuple[Path, Path]:
    require_directory(merged_dir, "merged checkpoint directory")
    require_file(spec_path, "attempt specification")
    manifest_path = spec_path.parent / spec["outputs"]["construction_manifest"]
    if path_exists(manifest_path, "construction manifest"):
        raise ValueError(f"Construction manifest already exists: {manifest_path}")
    if path_exists(output_root, "output root"):
        require_directory(output_root, "output root")
    output_dir = output_root / spec["outputs"]["directory_name"]
    if is_within(output_dir, merged_dir):
        raise ValueError("Output checkpoint must not be inside the merged checkpoint")
    if is_within(manifest_path, merged_dir):
        raise ValueError("Construction manifest must not be inside the merged checkpoint")
    if path_exists(output_dir, "output checkpoint"):
        raise ValueError(f"Output checkpoint already exists: {output_dir}")
    return manifest_path, output_dir


def checkpoint_file_records(root: Path) -> list[dict[str, Any]]:
    require_directory(root, "checkpoint directory")
    candidates = []
    try:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if ".cache" in relative.parts or not path.is_file():
                continue
            candidates.append((relative.as_posix(), path))
    except OSError as exc:
        raise ValueError(f"Could not enumerate checkpoint {root}: {exc}") from exc
    records = []
    for relative, path in sorted(candidates, key=lambda item: item[0]):
        try:
            records.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        except OSError as exc:
            raise ValueError(f"Could not hash checkpoint file {path}: {exc}") from exc
    if not records:
        raise ValueError(f"Checkpoint contains no files: {root}")
    return records


def discover_linear_weights(
    block: Any,
    module_prefix: str,
    torch_module: Any,
) -> list[tuple[str, Any]]:
    discovered = []
    for relative_name, module in block.named_modules():
        if not isinstance(module, torch_module.nn.Linear):
            continue
        module_name = module_prefix
        if relative_name:
            module_name = f"{module_prefix}.{relative_name}"
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch_module.Tensor) or weight.ndim != 2:
            raise ValueError(f"Linear module has invalid weight: {module_name}")
        discovered.append((module_name, module))
    discovered.sort(key=lambda item: item[0])
    return discovered


def transformer_layers(model: Any) -> Any:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Checkpoint does not expose model.layers")
    return layers


def discover_eligible_linear_weights(
    model: Any,
    block_indices: list[int],
    torch_module: Any,
) -> list[tuple[str, Any]]:
    layers = transformer_layers(model)
    if (
        not isinstance(block_indices, list)
        or not block_indices
        or any(type(index) is not int for index in block_indices)
        or len(set(block_indices)) != len(block_indices)
    ):
        raise ValueError("Transformer block indices must be unique integers")
    discovered = []
    seen_parameters: dict[int, str] = {}
    for index in sorted(block_indices):
        if not 0 <= index < len(layers):
            raise ValueError(f"Checkpoint does not expose transformer block {index}")
        block_weights = discover_linear_weights(
            layers[index], f"model.layers.{index}", torch_module
        )
        for module_name, module in block_weights:
            parameter_id = id(module.weight)
            if parameter_id in seen_parameters:
                raise ValueError(
                    "Eligible weight is shared by multiple modules: "
                    f"{seen_parameters[parameter_id]} and {module_name}"
                )
            seen_parameters[parameter_id] = module_name
            discovered.append((module_name, module))
    discovered.sort(key=lambda item: item[0])
    return discovered


def make_cpu_generator(seed: int, torch_module: Any) -> Any:
    if type(seed) is not int or seed < 0:
        raise ValueError("Gaussian seed must be a nonnegative integer")
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def gaussian_noise_weight(
    weight: Any,
    target_relative_frobenius: float,
    generator: Any,
    torch_module: Any,
) -> Any:
    """Apply one draw from a shared deterministic CPU Gaussian stream."""
    if not isinstance(weight, torch_module.Tensor) or weight.ndim != 2:
        raise ValueError("Eligible weight must be a rank-2 tensor")
    if weight.dtype != torch_module.float32:
        raise ValueError("Eligible source weight must be torch.float32")
    if weight.device.type != "cpu":
        raise ValueError("Eligible source weight must remain on CPU")
    if weight.numel() == 0:
        raise ValueError("Eligible source weight must not be empty")
    if not bool(torch_module.isfinite(weight).all().item()):
        raise ValueError("Eligible source weight contains non-finite values")
    if (
        isinstance(target_relative_frobenius, bool)
        or not isinstance(target_relative_frobenius, (int, float))
        or not math.isfinite(target_relative_frobenius)
        or target_relative_frobenius < 0.0
    ):
        raise ValueError("Target relative Frobenius perturbation is invalid")
    if not isinstance(generator, torch_module.Generator):
        raise ValueError("Gaussian generator must be a torch.Generator")
    if generator.device.type != "cpu":
        raise ValueError("Gaussian generator must remain on CPU")

    original64 = weight.detach().to(dtype=torch_module.float64)
    original_norm = torch_module.linalg.vector_norm(original64)
    noise64 = torch_module.randn(
        tuple(weight.shape),
        dtype=torch_module.float64,
        device="cpu",
        generator=generator,
    )
    noise_norm = torch_module.linalg.vector_norm(noise64)
    if not bool(torch_module.isfinite(noise_norm).item()) or noise_norm.item() == 0.0:
        raise ValueError("Gaussian draw has invalid Frobenius norm")
    noise64.mul_(
        float(target_relative_frobenius) * original_norm.item() / noise_norm.item()
    )
    original64.add_(noise64)
    modified32 = original64.to(dtype=torch_module.float32).contiguous()
    if not bool(torch_module.isfinite(modified32).all().item()):
        raise ValueError("Modified weight contains non-finite values")
    return modified32


def realized_perturbation_metrics(
    original32: Any,
    modified32: Any,
    torch_module: Any,
) -> dict[str, float | int]:
    if (
        not isinstance(original32, torch_module.Tensor)
        or not isinstance(modified32, torch_module.Tensor)
        or original32.dtype != torch_module.float32
        or modified32.dtype != torch_module.float32
        or original32.device.type != "cpu"
        or modified32.device.type != "cpu"
        or tuple(original32.shape) != tuple(modified32.shape)
        or original32.ndim != 2
    ):
        raise ValueError("Perturbation inputs must be matching CPU float32 matrices")
    if original32.numel() == 0:
        raise ValueError("Perturbation inputs must not be empty")
    original64 = original32.detach().to(dtype=torch_module.float64)
    modified64 = modified32.detach().to(dtype=torch_module.float64)
    delta64 = modified64 - original64
    original_norm = float(torch_module.linalg.vector_norm(original64).item())
    delta_norm = float(torch_module.linalg.vector_norm(delta64).item())
    relative = 0.0 if original_norm == 0.0 else delta_norm / original_norm
    changed_count = int(torch_module.count_nonzero(modified32 != original32).item())
    element_count = original32.numel()
    return {
        "original_frobenius_norm": original_norm,
        "realized_fp32_delta_frobenius_norm": delta_norm,
        "realized_relative_frobenius_perturbation": relative,
        "max_absolute_weight_change": float(delta64.abs().max().item()),
        "changed_element_count": changed_count,
        "changed_element_fraction": changed_count / element_count,
    }


def apply_gaussian_noise(
    eligible: list[tuple[str, Any]],
    seed: int,
    target_relative_frobenius: float,
    torch_module: Any,
) -> list[dict[str, Any]]:
    if not eligible:
        raise ValueError("Selected transformer blocks contain no eligible weights")
    ordered = sorted(eligible, key=lambda item: item[0])
    names = [name for name, _module in ordered]
    if len(names) != len(set(names)):
        raise ValueError("Eligible module names must be unique")
    generator = make_cpu_generator(seed, torch_module)
    records = []
    with torch_module.no_grad():
        for module_name, module in ordered:
            original32 = module.weight.detach().clone()
            modified32 = gaussian_noise_weight(
                original32,
                target_relative_frobenius,
                generator,
                torch_module,
            )
            module.weight.copy_(modified32)
            realized32 = module.weight.detach()
            metrics = realized_perturbation_metrics(
                original32, realized32, torch_module
            )
            records.append(
                {
                    "name": module_name,
                    "shape": list(realized32.shape),
                    **metrics,
                }
            )
    return records


def verify_float32_model(model: Any, torch_module: Any) -> None:
    floating_parameters = [
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    ]
    if not floating_parameters:
        raise ValueError("Loaded checkpoint has no floating-point parameters")
    unexpected = sorted(
        {
            str(parameter.dtype)
            for parameter in floating_parameters
            if parameter.dtype != torch_module.float32
        }
    )
    if unexpected:
        raise ValueError(
            "Loaded checkpoint has unexpected floating dtypes: " + ", ".join(unexpected)
        )
    devices = {parameter.device.type for parameter in floating_parameters}
    if devices != {"cpu"}:
        raise ValueError("Loaded checkpoint must remain on CPU")


def validate_loaded_model(model: Any, torch_module: Any) -> None:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "qwen3":
        raise ValueError("Loaded checkpoint config is not Qwen3")
    if getattr(config, "num_hidden_layers", None) != 28:
        raise ValueError("Loaded checkpoint must contain 28 transformer blocks")
    if len(transformer_layers(model)) != 28:
        raise ValueError("Loaded checkpoint does not expose 28 transformer blocks")
    verify_float32_model(model, torch_module)


def load_local_checkpoint(merged_dir: Path, torch_module: Any) -> tuple[Any, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(merged_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(merged_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        config=config,
        dtype=torch_module.float32,
        local_files_only=True,
    )
    model.eval()
    validate_loaded_model(model, torch_module)
    return model, tokenizer


def normalized_file_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    for record in sorted(records, key=lambda item: item["path"]):
        path = record.get("path")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError("Malformed or duplicate checkpoint file path")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(f"Malformed checkpoint file size: {path}")
        require_sha256(digest, f"checkpoint file {path}")
        normalized.append({"path": path, "size_bytes": size_bytes, "sha256": digest})
        seen.add(path)
    if not normalized:
        raise ValueError("Checkpoint file records must not be empty")
    return normalized


def normalized_matrix_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "name",
        "shape",
        "original_frobenius_norm",
        "realized_fp32_delta_frobenius_norm",
        "realized_relative_frobenius_perturbation",
        "max_absolute_weight_change",
        "changed_element_count",
        "changed_element_fraction",
    )
    normalized = []
    seen = set()
    for record in sorted(records, key=lambda item: item["name"]):
        name = record.get("name")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Malformed or duplicate modified module name")
        try:
            normalized.append({field: record[field] for field in fields})
        except KeyError as exc:
            raise ValueError(f"Modified matrix record is missing {exc.args[0]}") from exc
        seen.add(name)
    if not normalized:
        raise ValueError("Construction must modify at least one matrix")
    return normalized


def build_construction_manifest(
    spec: dict[str, Any],
    spec_sha256: str,
    constructor_sha256: str,
    source_files: list[dict[str, Any]],
    output_files: list[dict[str, Any]],
    matrix_records: list[dict[str, Any]],
) -> dict[str, Any]:
    require_sha256(spec_sha256, "attempt specification")
    require_sha256(constructor_sha256, "constructor")
    matrices = normalized_matrix_records(matrix_records)
    if len(matrices) != spec["expected_matrix_count"]:
        raise ValueError(
            f"Construction modified {len(matrices)} matrices, expected "
            f"{spec['expected_matrix_count']}"
        )
    source = normalized_file_records(source_files)
    output = normalized_file_records(output_files)
    combined_original_norm = math.sqrt(
        math.fsum(record["original_frobenius_norm"] ** 2 for record in matrices)
    )
    combined_delta_norm = math.sqrt(
        math.fsum(
            record["realized_fp32_delta_frobenius_norm"] ** 2
            for record in matrices
        )
    )
    combined_relative = (
        0.0
        if combined_original_norm == 0.0
        else combined_delta_norm / combined_original_norm
    )
    total_elements = sum(math.prod(record["shape"]) for record in matrices)
    total_changed = sum(record["changed_element_count"] for record in matrices)
    ordered_names = [record["name"] for record in matrices]
    return {
        "format_version": 1,
        "attempt_id": spec["attempt_id"],
        "information_policy": spec["information_policy"],
        "hash_algorithm": "sha256",
        "spec_sha256": spec_sha256,
        "constructor_sha256": constructor_sha256,
        "source_checkpoint": {
            "file_count": len(source),
            "total_bytes": sum(item["size_bytes"] for item in source),
            "files": source,
        },
        "construction": {
            "transformer_block_indices": list(spec["transformer_block_indices"]),
            "expected_matrix_count": spec["expected_matrix_count"],
            "eligible_tensors": dict(spec["eligible_tensors"]),
            "source_weight_dtype": spec["source_weight_dtype"],
            "seed": spec["seed"],
            "target_relative_frobenius": spec[
                "target_relative_frobenius"
            ],
            "randomness": dict(spec["randomness"]),
            "weight_transform": spec["weight_transform"],
            "stored_dtype": spec["outputs"]["checkpoint_dtype"],
            "standalone_checkpoint": spec["outputs"]["standalone_checkpoint"],
        },
        "deterministic_ordering": {
            "rule": "lexicographic_by_module_name",
            "generator_stream": "single_cpu_stream_in_modified_matrix_order",
            "modified_module_names": ordered_names,
        },
        "modified_matrices": matrices,
        "aggregate_perturbation": {
            "modified_matrix_count": len(matrices),
            "aggregate_original_frobenius_norm": combined_original_norm,
            "aggregate_realized_fp32_delta_frobenius_norm": combined_delta_norm,
            "aggregate_realized_relative_frobenius_perturbation": combined_relative,
            "max_absolute_weight_change": max(
                record["max_absolute_weight_change"] for record in matrices
            ),
            "element_count": total_elements,
            "changed_element_count": total_changed,
            "changed_element_fraction": total_changed / total_elements,
        },
        "output_checkpoint": {
            "directory_name": spec["outputs"]["directory_name"],
            "file_count": len(output),
            "total_bytes": sum(item["size_bytes"] for item in output),
            "files": output,
        },
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    try:
        serialized = json.dumps(
            manifest, ensure_ascii=False, allow_nan=False, indent=2
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not serialize construction manifest: {exc}") from exc
    try:
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(serialized)
    except FileExistsError as exc:
        raise ValueError(f"Construction manifest already exists: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not write construction manifest {path}: {exc}") from exc


def save_checkpoint(
    model: Any,
    tokenizer: Any,
    output_root: Path,
    output_dir: Path,
    torch_module: Any,
) -> list[dict[str, Any]]:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"Output checkpoint already exists: {output_dir}") from exc
    except OSError as exc:
        raise ValueError(f"Could not create output checkpoint {output_dir}: {exc}") from exc
    verify_float32_model(model, torch_module)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return checkpoint_file_records(output_dir)


def construct(merged_dir: Path, spec_path: Path, output_root: Path) -> Path:
    spec = load_spec(spec_path)
    spec_hash = sha256_file(spec_path)
    constructor_path = Path(__file__).resolve()
    constructor_hash = sha256_file(constructor_path)
    manifest_path, output_dir = validate_destinations(
        merged_dir, spec_path, output_root, spec
    )
    source_files_before = checkpoint_file_records(merged_dir)

    import torch

    model, tokenizer = load_local_checkpoint(merged_dir, torch)
    eligible = discover_eligible_linear_weights(
        model, spec["transformer_block_indices"], torch
    )
    if len(eligible) != spec["expected_matrix_count"]:
        raise ValueError(
            f"Discovered {len(eligible)} eligible matrices, expected "
            f"{spec['expected_matrix_count']}"
        )
    matrix_records = apply_gaussian_noise(
        eligible,
        spec["seed"],
        spec["target_relative_frobenius"],
        torch,
    )
    verify_float32_model(model, torch)
    output_files = save_checkpoint(model, tokenizer, output_root, output_dir, torch)

    source_files_after = checkpoint_file_records(merged_dir)
    if source_files_after != source_files_before:
        raise ValueError("Merged checkpoint changed during construction")
    if sha256_file(spec_path) != spec_hash:
        raise ValueError("Attempt specification changed during construction")
    if sha256_file(constructor_path) != constructor_hash:
        raise ValueError("Constructor changed during construction")
    manifest = build_construction_manifest(
        spec,
        spec_hash,
        constructor_hash,
        source_files_before,
        output_files,
        matrix_records,
    )
    write_manifest(manifest, manifest_path)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-model-dir",
        type=Path,
        default=Path(
            os.environ.get("MERGED_MODEL_DIR", DEFAULT_MERGED_MODEL_DIR)
        ).expanduser(),
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=Path(os.environ.get("ABLATION_SPEC_PATH", DEFAULT_SPEC_PATH)).expanduser(),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get("ABLATION_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
        ).expanduser(),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest_path = construct(
            args.merged_model_dir, args.spec_path, args.output_root
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
