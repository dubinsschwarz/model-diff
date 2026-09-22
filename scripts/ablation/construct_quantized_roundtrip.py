#!/usr/bin/env python3

"""Construct Attempt 004 with an 11-bit weight quantize/dequantize pass."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT_DIRECTORY = (
    PROJECT / "experiments" / "attempts" / "004_quantize11_prefix0_13"
)
DEFAULT_SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
DEFAULT_CALIBRATION_PATH = ATTEMPT_DIRECTORY / "calibration.json"
DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_OUTPUT_ROOT = Path(
    "/root/model-diff-scratch/ablations/004_quantize11_prefix0_13"
)

ALLOWED_ENVIRONMENT_VARIABLES = {
    "MERGED_MODEL_DIR",
    "ABLATION_SPEC_PATH",
    "ABLATION_CALIBRATION_PATH",
    "ABLATION_OUTPUT_ROOT",
}

FROZEN_SPEC = {
    "format_version": 1,
    "attempt_id": "004_quantize11_prefix0_13",
    "method": "symmetric_per_output_channel_weight_quantization_roundtrip",
    "information_policy": "merged_checkpoint_only",
    "defaults": {
        "merged_model_directory": "/root/model-diff-scratch/models/merged",
        "output_root": (
            "/root/model-diff-scratch/ablations/004_quantize11_prefix0_13"
        ),
    },
    "calibration_filename": "calibration.json",
    "selected_bits": 11,
    "selection_target_relative_frobenius": 0.00125,
    "selection_rule": {
        "candidate_bit_widths": list(range(8, 17)),
        "criterion": "minimize_abs_aggregate_relative_frobenius_minus_target",
        "exact_tie_break": "higher_bit_width",
    },
    "transformer_block_indices": list(range(14)),
    "expected_matrix_count": 98,
    "eligible_tensors": {
        "module_type": "torch.nn.Linear",
        "parameter": "weight",
        "scope": "recursive_descendants_of_model.model.layers[0:14]",
        "exclusions": [
            "embeddings",
            "norms",
            "lm_head",
            "biases",
            "blocks_14_through_27",
        ],
    },
    "source_weight_dtype": "float32",
    "quantization": {
        "bits": 11,
        "qmax": 1023,
        "scheme": "symmetric_per_output_channel_uniform",
        "channel_axis": 0,
        "pre_cast_dtype": "float64",
        "rounding": "torch.round",
        "clamp_min": -1023,
        "clamp_max": 1023,
        "zero_row_rule": "exact_zero_row",
        "stored_dtype": "float32",
        "integer_inference_kernels": False,
        "activation_quantization": False,
        "randomness": "none",
    },
    "weight_transform": (
        "W_new = contiguous(float32(clamp(round(W64 / (row_absmax / 1023)), "
        "-1023, 1023) * (row_absmax / 1023))), with exact zero output for "
        "zero rows"
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


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    require_file(path, description)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{description} must be a JSON object")
    return document


def load_spec(path: Path) -> dict[str, Any]:
    spec = load_json_object(path, "attempt specification")
    if spec != FROZEN_SPEC:
        raise ValueError("Attempt specification does not match the frozen definition")
    return spec


def finite_number(value: Any, description: str, *, nonnegative: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (nonnegative and value < 0.0)
    ):
        raise ValueError(f"Calibration has invalid {description}")
    return float(value)


def select_bit_width(
    candidate_results: Any,
    target: float,
    candidate_bits: list[int],
) -> int:
    if not isinstance(candidate_results, list):
        raise ValueError("Calibration candidate results must be a list")
    by_bits: dict[int, float] = {}
    for result in candidate_results:
        if not isinstance(result, dict):
            raise ValueError("Calibration candidate result must be an object")
        bits = result.get("bits")
        if type(bits) is not int or bits in by_bits:
            raise ValueError("Calibration candidate bit-width is malformed or duplicate")
        relative = finite_number(
            result.get("aggregate_relative_frobenius_perturbation"),
            f"aggregate relative perturbation for {bits} bits",
            nonnegative=True,
        )
        by_bits[bits] = relative
    if set(by_bits) != set(candidate_bits) or len(by_bits) != len(candidate_bits):
        raise ValueError("Calibration candidate bit-width coverage mismatch")
    return min(candidate_bits, key=lambda bits: (abs(by_bits[bits] - target), -bits))


def normalize_file_records(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("Checkpoint file records must be a list")
    normalized = []
    seen = set()
    for record in sorted(
        records,
        key=lambda item: item.get("path", "") if isinstance(item, dict) else "",
    ):
        if not isinstance(record, dict):
            raise ValueError("Checkpoint file record must be an object")
        path = record.get("path")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(path, str) or not path or path in seen:
            raise ValueError("Malformed or duplicate checkpoint file path")
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"Unsafe checkpoint file path: {path}")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(f"Malformed checkpoint file size: {path}")
        require_sha256(digest, f"checkpoint file {path}")
        normalized.append(
            {"path": path, "size_bytes": size_bytes, "sha256": digest}
        )
        seen.add(path)
    if not normalized:
        raise ValueError("Checkpoint file records must not be empty")
    return normalized


def normalize_checkpoint_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("Calibration source checkpoint must be an object")
    files = normalize_file_records(record.get("files"))
    file_count = record.get("file_count")
    total_bytes = record.get("total_bytes")
    if file_count != len(files):
        raise ValueError("Calibration source checkpoint file_count mismatch")
    expected_bytes = sum(item["size_bytes"] for item in files)
    if total_bytes != expected_bytes:
        raise ValueError("Calibration source checkpoint total_bytes mismatch")
    directory = record.get("directory")
    if not isinstance(directory, str) or not directory:
        raise ValueError("Calibration source checkpoint directory is malformed")
    return {
        "directory": directory,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "files": files,
    }


def validate_calibration_document(
    calibration: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    if calibration.get("format_version") != 1:
        raise ValueError("Unsupported calibration format")
    if (
        calibration.get("diagnostic")
        != "symmetric_per_output_channel_weight_quantization_calibration"
    ):
        raise ValueError("Calibration diagnostic identity mismatch")

    expected_bits = spec["selection_rule"]["candidate_bit_widths"]
    target = spec["selection_target_relative_frobenius"]
    selection = calibration.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("Calibration selection metadata is missing")
    expected_selection = {
        "target_relative_frobenius": target,
        "criterion": spec["selection_rule"]["criterion"],
        "exact_tie_break": spec["selection_rule"]["exact_tie_break"],
        "uses_perturbation_magnitude_only": True,
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ValueError(f"Calibration selection metadata mismatch: {key}")

    quantization = calibration.get("quantization")
    expected_quantization = {
        "candidate_bit_widths": expected_bits,
        "scheme": "symmetric_per_output_channel_uniform",
        "qmax": "2^(bits-1)-1",
        "row_scale": "max(abs(row_float64))/qmax",
        "rounding": "torch.round",
        "zero_row_rule": "exact_zero_row",
        "pre_cast_dtype": "float64",
        "dequantized_dtype": "float32",
        "realized_delta": "dequantized_float32_minus_original_float32",
        "integer_inference_kernels": False,
        "activation_quantization": False,
    }
    if quantization != expected_quantization:
        raise ValueError("Calibration quantization method mismatch")

    scope = calibration.get("scope")
    expected_scope = {
        "transformer_block_indices": spec["transformer_block_indices"],
        "recursive_module_type": "torch.nn.Linear",
        "parameter": "weight",
        "expected_matrix_count": spec["expected_matrix_count"],
        "actual_matrix_count": spec["expected_matrix_count"],
        "source_weight_dtype": spec["source_weight_dtype"],
        "module_ordering": "lexicographic_by_module_name",
        "excluded": spec["eligible_tensors"]["exclusions"],
    }
    if scope != expected_scope:
        raise ValueError("Calibration scope mismatch")

    candidate_results = calibration.get("candidate_results")
    winner = select_bit_width(candidate_results, target, expected_bits)
    if selection.get("selected_bit_width") != winner:
        raise ValueError("Calibration stored winner disagrees with recomputed winner")
    if winner != spec["selected_bits"] or winner != 11:
        raise ValueError("Calibration did not select the frozen 11-bit attempt")

    if not isinstance(candidate_results, list):
        raise ValueError("Calibration candidate results are missing")
    by_bits = {result["bits"]: result for result in candidate_results}
    aggregate_original = None
    total_scalar_count = calibration.get("total_scalar_count")
    if type(total_scalar_count) is not int or total_scalar_count <= 0:
        raise ValueError("Calibration total scalar count is invalid")
    for bits in expected_bits:
        result = by_bits[bits]
        if result.get("qmax") != (1 << (bits - 1)) - 1:
            raise ValueError(f"Calibration qmax mismatch for {bits} bits")
        original_norm = finite_number(
            result.get("aggregate_original_frobenius_norm"),
            f"aggregate original norm for {bits} bits",
            nonnegative=True,
        )
        delta_norm = finite_number(
            result.get("aggregate_realized_delta_frobenius_norm"),
            f"aggregate delta norm for {bits} bits",
            nonnegative=True,
        )
        relative = finite_number(
            result.get("aggregate_relative_frobenius_perturbation"),
            f"aggregate relative perturbation for {bits} bits",
            nonnegative=True,
        )
        if original_norm <= 0.0 or not math.isclose(
            relative,
            delta_norm / original_norm,
            rel_tol=1e-15,
            abs_tol=0.0,
        ):
            raise ValueError(f"Calibration aggregate norms disagree for {bits} bits")
        if aggregate_original is None:
            aggregate_original = original_norm
        elif original_norm != aggregate_original:
            raise ValueError("Calibration aggregate original norm is inconsistent")
        changed = result.get("changed_scalar_count")
        if type(changed) is not int or not 0 <= changed <= total_scalar_count:
            raise ValueError(f"Calibration changed scalar count is invalid for {bits} bits")
        if not math.isclose(
            finite_number(
                result.get("changed_scalar_fraction"),
                f"changed scalar fraction for {bits} bits",
                nonnegative=True,
            ),
            changed / total_scalar_count,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"Calibration changed scalar fraction mismatch for {bits} bits")
        finite_number(
            result.get("max_absolute_weight_change"),
            f"maximum absolute change for {bits} bits",
            nonnegative=True,
        )
        per_matrix = result.get("per_matrix_relative_perturbation")
        if not isinstance(per_matrix, dict) or list(per_matrix) != [
            "min",
            "median",
            "mean",
            "max",
        ]:
            raise ValueError(f"Calibration per-matrix summary is malformed for {bits} bits")
        summary_values = [
            finite_number(
                per_matrix[key],
                f"per-matrix {key} for {bits} bits",
                nonnegative=True,
            )
            for key in ("min", "median", "mean", "max")
        ]
        if not summary_values[0] <= summary_values[1] <= summary_values[3]:
            raise ValueError(f"Calibration per-matrix order is invalid for {bits} bits")
        if not summary_values[0] <= summary_values[2] <= summary_values[3]:
            raise ValueError(f"Calibration per-matrix mean is invalid for {bits} bits")
        expected_error = abs(relative - target)
        if result.get("absolute_target_error") != expected_error:
            raise ValueError(f"Calibration target error mismatch for {bits} bits")

    selected_result = by_bits[winner]
    if selection.get("selected_absolute_target_error") != selected_result.get(
        "absolute_target_error"
    ):
        raise ValueError("Calibration selected target error mismatch")

    inventory = calibration.get("eligible_matrices")
    if not isinstance(inventory, list) or len(inventory) != spec["expected_matrix_count"]:
        raise ValueError("Calibration eligible matrix coverage mismatch")
    normalized_inventory = []
    seen_names = set()
    for record in inventory:
        if not isinstance(record, dict) or set(record) != {
            "module_name",
            "shape",
            "scalar_count",
        }:
            raise ValueError("Calibration eligible matrix record is malformed")
        name = record["module_name"]
        shape = record["shape"]
        scalar_count = record["scalar_count"]
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError("Calibration eligible matrix name is malformed or duplicate")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or scalar_count != math.prod(shape)
        ):
            raise ValueError(f"Calibration eligible matrix shape is invalid: {name}")
        normalized_inventory.append(
            {"module_name": name, "shape": shape, "scalar_count": scalar_count}
        )
        seen_names.add(name)
    if [record["module_name"] for record in normalized_inventory] != sorted(seen_names):
        raise ValueError("Calibration eligible matrices are not lexicographically ordered")
    if sum(record["scalar_count"] for record in normalized_inventory) != total_scalar_count:
        raise ValueError("Calibration eligible scalar count mismatch")

    source_checkpoint = normalize_checkpoint_record(
        calibration.get("source_checkpoint")
    )
    return {
        "selected_bits": winner,
        "qmax": (1 << (winner - 1)) - 1,
        "selected_result": selected_result,
        "source_checkpoint": source_checkpoint,
        "eligible_matrices": normalized_inventory,
    }


def load_calibration(path: Path, spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    calibration = load_json_object(path, "quantization calibration")
    validated = validate_calibration_document(calibration, spec)
    return calibration, validated


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_destinations(
    merged_dir: Path,
    spec_path: Path,
    calibration_path: Path,
    output_root: Path,
    spec: dict[str, Any],
) -> tuple[Path, Path]:
    require_directory(merged_dir, "merged checkpoint directory")
    require_file(spec_path, "attempt specification")
    require_file(calibration_path, "quantization calibration")
    expected_calibration = spec_path.parent / spec["calibration_filename"]
    if calibration_path.resolve() != expected_calibration.resolve():
        raise ValueError("Calibration path does not match the frozen attempt layout")
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
            if not path.is_file():
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


def validate_discovery_against_calibration(
    eligible: list[tuple[str, Any]],
    calibration_inventory: list[dict[str, Any]],
) -> None:
    discovered = [
        {
            "module_name": name,
            "shape": list(module.weight.shape),
            "scalar_count": module.weight.numel(),
        }
        for name, module in eligible
    ]
    if discovered != calibration_inventory:
        raise ValueError("Discovered eligible matrices disagree with calibration")


def quantize_dequantize_weight(
    weight: Any,
    bits: int,
    qmax: int,
    torch_module: Any,
) -> Any:
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
    if type(bits) is not int or type(qmax) is not int or qmax != (1 << (bits - 1)) - 1:
        raise ValueError("Quantization bit-width and qmax disagree")

    weight64 = weight.detach().to(dtype=torch_module.float64)
    row_absmax = weight64.abs().amax(dim=1)
    zero_rows = row_absmax == 0.0
    scale = row_absmax / qmax
    safe_scale = torch_module.where(
        zero_rows,
        torch_module.ones_like(scale),
        scale,
    )
    dequantized64 = torch_module.round(weight64 / safe_scale.unsqueeze(1))
    dequantized64.clamp_(min=-qmax, max=qmax)
    dequantized64.mul_(scale.unsqueeze(1))
    if bool(zero_rows.any().item()):
        dequantized64[zero_rows] = 0.0
    modified32 = dequantized64.to(dtype=torch_module.float32).contiguous()
    if not bool(torch_module.isfinite(modified32).all().item()):
        raise ValueError("Quantized weight contains non-finite values")
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
    scalar_count = original32.numel()
    return {
        "original_frobenius_norm": original_norm,
        "realized_fp32_delta_frobenius_norm": delta_norm,
        "realized_relative_frobenius_perturbation": relative,
        "changed_scalar_count": changed_count,
        "changed_scalar_fraction": changed_count / scalar_count,
        "max_absolute_weight_change": float(delta64.abs().max().item()),
    }


def apply_quantization(
    eligible: list[tuple[str, Any]],
    bits: int,
    qmax: int,
    torch_module: Any,
) -> list[dict[str, Any]]:
    if not eligible:
        raise ValueError("Selected transformer blocks contain no eligible weights")
    ordered = sorted(eligible, key=lambda item: item[0])
    names = [name for name, _module in ordered]
    if len(names) != len(set(names)):
        raise ValueError("Eligible module names must be unique")
    records = []
    with torch_module.no_grad():
        for module_name, module in ordered:
            original32 = module.weight.detach().clone()
            modified32 = quantize_dequantize_weight(
                original32,
                bits,
                qmax,
                torch_module,
            )
            module.weight.copy_(modified32)
            realized32 = module.weight.detach()
            if not realized32.is_contiguous():
                raise ValueError("Quantized weight must be contiguous")
            metrics = realized_perturbation_metrics(
                original32,
                realized32,
                torch_module,
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
    return normalize_file_records(records)


def normalized_matrix_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "name",
        "shape",
        "original_frobenius_norm",
        "realized_fp32_delta_frobenius_norm",
        "realized_relative_frobenius_perturbation",
        "changed_scalar_count",
        "changed_scalar_fraction",
        "max_absolute_weight_change",
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
    calibration_sha256: str,
    constructor_sha256: str,
    calibration_metadata: dict[str, Any],
    source_files: list[dict[str, Any]],
    output_files: list[dict[str, Any]],
    matrix_records: list[dict[str, Any]],
) -> dict[str, Any]:
    require_sha256(spec_sha256, "attempt specification")
    require_sha256(calibration_sha256, "quantization calibration")
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
    total_scalars = sum(math.prod(record["shape"]) for record in matrices)
    total_changed = sum(record["changed_scalar_count"] for record in matrices)
    ordered_names = [record["name"] for record in matrices]
    return {
        "format_version": 1,
        "attempt_id": spec["attempt_id"],
        "information_policy": spec["information_policy"],
        "hash_algorithm": "sha256",
        "spec_sha256": spec_sha256,
        "calibration_sha256": calibration_sha256,
        "constructor_sha256": constructor_sha256,
        "calibration_selection": {
            "selected_bits": calibration_metadata["selected_bits"],
            "qmax": calibration_metadata["qmax"],
            "target_relative_frobenius": spec[
                "selection_target_relative_frobenius"
            ],
            "criterion": spec["selection_rule"]["criterion"],
            "exact_tie_break": spec["selection_rule"]["exact_tie_break"],
            "selected_calibration_aggregate_relative_frobenius": (
                calibration_metadata["selected_result"][
                    "aggregate_relative_frobenius_perturbation"
                ]
            ),
        },
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
            "selected_bits": spec["selected_bits"],
            "qmax": spec["quantization"]["qmax"],
            "quantization": dict(spec["quantization"]),
            "weight_transform": spec["weight_transform"],
            "stored_dtype": spec["outputs"]["checkpoint_dtype"],
            "standalone_checkpoint": spec["outputs"]["standalone_checkpoint"],
        },
        "deterministic_ordering": {
            "rule": "lexicographic_by_module_name",
            "modified_module_names": ordered_names,
        },
        "modified_matrices": matrices,
        "aggregate_perturbation": {
            "modified_matrix_count": len(matrices),
            "aggregate_original_frobenius_norm": combined_original_norm,
            "aggregate_realized_fp32_delta_frobenius_norm": combined_delta_norm,
            "aggregate_realized_relative_frobenius_perturbation": combined_relative,
            "scalar_count": total_scalars,
            "changed_scalar_count": total_changed,
            "changed_scalar_fraction": total_changed / total_scalars,
            "max_absolute_weight_change": max(
                record["max_absolute_weight_change"] for record in matrices
            ),
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


def construct(
    merged_dir: Path,
    spec_path: Path,
    calibration_path: Path,
    output_root: Path,
) -> Path:
    spec = load_spec(spec_path)
    calibration, calibration_metadata = load_calibration(calibration_path, spec)
    del calibration
    spec_hash = sha256_file(spec_path)
    calibration_hash = sha256_file(calibration_path)
    constructor_path = Path(__file__).resolve()
    constructor_hash = sha256_file(constructor_path)
    manifest_path, output_dir = validate_destinations(
        merged_dir,
        spec_path,
        calibration_path,
        output_root,
        spec,
    )
    source_files_before = checkpoint_file_records(merged_dir)
    if normalize_file_records(source_files_before) != calibration_metadata[
        "source_checkpoint"
    ]["files"]:
        raise ValueError("Merged checkpoint disagrees with the frozen calibration")

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
    validate_discovery_against_calibration(
        eligible,
        calibration_metadata["eligible_matrices"],
    )
    matrix_records = apply_quantization(
        eligible,
        spec["selected_bits"],
        spec["quantization"]["qmax"],
        torch,
    )
    verify_float32_model(model, torch)
    output_files = save_checkpoint(model, tokenizer, output_root, output_dir, torch)

    source_files_after = checkpoint_file_records(merged_dir)
    if source_files_after != source_files_before:
        raise ValueError("Merged checkpoint changed during construction")
    if sha256_file(spec_path) != spec_hash:
        raise ValueError("Attempt specification changed during construction")
    if sha256_file(calibration_path) != calibration_hash:
        raise ValueError("Calibration changed during construction")
    if sha256_file(constructor_path) != constructor_hash:
        raise ValueError("Constructor changed during construction")
    manifest = build_construction_manifest(
        spec,
        spec_hash,
        calibration_hash,
        constructor_hash,
        calibration_metadata,
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
        "--calibration-path",
        type=Path,
        default=Path(
            os.environ.get("ABLATION_CALIBRATION_PATH", DEFAULT_CALIBRATION_PATH)
        ).expanduser(),
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
            args.merged_model_dir,
            args.spec_path,
            args.calibration_path,
            args.output_root,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
