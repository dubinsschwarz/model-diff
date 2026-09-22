#!/usr/bin/env python3

"""Compute Attempt 004 merged-minus-quantized activation differences."""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT_DIRECTORY = (
    PROJECT / "experiments" / "attempts" / "004_quantize11_prefix0_13"
)
DEFAULT_ATTEMPT_SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
DEFAULT_CALIBRATION_PATH = ATTEMPT_DIRECTORY / "calibration.json"
DEFAULT_CONSTRUCTION_MANIFEST_PATH = ATTEMPT_DIRECTORY / "construction-manifest.json"
DEFAULT_SELF_DIFF_SPEC_PATH = ATTEMPT_DIRECTORY / "self_diff_spec.json"
DEFAULT_SELF_DIFF_MANIFEST_PATH = ATTEMPT_DIRECTORY / "self-diff-manifest.json"

DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_PROBE_PATH = Path(
    "/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt"
)
DEFAULT_ARTIFACT_PATH = Path(
    "/root/model-diff-scratch/artifacts/attempt004_self_diff/self_diff.pt"
)

ALLOWED_ENVIRONMENT_VARIABLES = {
    "MERGED_MODEL_DIR",
    "ATTEMPT_SPEC_PATH",
    "CALIBRATION_PATH",
    "CONSTRUCTION_MANIFEST_PATH",
    "FROZEN_PROBE_PATH",
    "SELF_DIFF_SPEC_PATH",
    "SELF_DIFF_ARTIFACT_PATH",
    "SELF_DIFF_MANIFEST_PATH",
}

FROZEN_SELF_DIFF_SPEC = {
    "format_version": 1,
    "attempt_id": "004_quantize11_prefix0_13",
    "input_information_policy": (
        "canonical_merged_attempt004_quantized_checkpoint_and_frozen_generic_"
        "probe_only"
    ),
    "readout_block_index": 13,
    "num_hidden_layers": 28,
    "sample_count": 10_000,
    "sequence_length": 128,
    "batch_size": 32,
    "model_dtype": "float32",
    "accumulator_dtype": "float64",
    "stored_dtype": "float32",
    "difference_sign": "merged_minus_quantized",
    "probe": {
        "dtype": "torch.int64",
        "shape": [10_000, 128],
        "serialized_sha256": (
            "3d799d19abf4a2d6dc92b2bd164d81d83f6451cd3545f5ae5d8c04a8b299735b"
        ),
        "raw_tensor_sha256": (
            "73f56300fdd06bc8fafcfa5750fee7a586a5c5e6071a608667ffc526bbb90ac8"
        ),
    },
    "defaults": {
        "merged_model_directory": "/root/model-diff-scratch/models/merged",
        "probe_path": (
            "/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt"
        ),
        "artifact_path": (
            "/root/model-diff-scratch/artifacts/attempt004_self_diff/self_diff.pt"
        ),
    },
    "outputs": {
        "artifact_format_version": 1,
        "manifest_filename": "self-diff-manifest.json",
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {description}: {path}")
    return value


def load_self_diff_spec(path: Path) -> dict[str, Any]:
    spec = load_json_object(path, "self-difference specification")
    if spec != FROZEN_SELF_DIFF_SPEC:
        raise ValueError(
            "Self-difference specification does not match the frozen definition"
        )
    return spec


def validate_output_paths(artifact_path: Path, manifest_path: Path) -> None:
    if path_exists(artifact_path, "self-difference artifact"):
        raise ValueError(f"Self-difference artifact already exists: {artifact_path}")
    if path_exists(manifest_path, "self-difference manifest"):
        raise ValueError(f"Self-difference manifest already exists: {manifest_path}")
    if path_exists(artifact_path.parent, "artifact directory"):
        require_directory(artifact_path.parent, "artifact directory")
    require_directory(manifest_path.parent, "manifest directory")


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


def normalize_checkpoint_entry(entry: Any, description: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"Malformed {description}: expected an object")
    files = entry.get("files")
    if not isinstance(files, list):
        raise ValueError(f"Malformed {description}: files")
    normalized_files = []
    seen = set()
    for record in files:
        if not isinstance(record, dict):
            raise ValueError(f"Malformed {description} file record")
        relative = record.get("path")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(relative, str):
            raise ValueError(f"Malformed {description} file path")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or relative != parsed.as_posix()
            or any(part in ("", ".", "..") for part in parsed.parts)
        ):
            raise ValueError(f"Malformed {description} file path: {relative}")
        if relative in seen:
            raise ValueError(f"Duplicate {description} file path: {relative}")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(f"Malformed {description} size for {relative}")
        require_sha256(digest, f"{description} file {relative}")
        normalized_files.append(
            {"path": relative, "size_bytes": size_bytes, "sha256": digest}
        )
        seen.add(relative)
    normalized_files.sort(key=lambda item: item["path"])
    file_count = entry.get("file_count")
    total_bytes = entry.get("total_bytes")
    if type(file_count) is not int or file_count != len(normalized_files):
        raise ValueError(f"Malformed {description}: file_count")
    expected_total = sum(item["size_bytes"] for item in normalized_files)
    if type(total_bytes) is not int or total_bytes != expected_total:
        raise ValueError(f"Malformed {description}: total_bytes")
    return {
        "file_count": file_count,
        "total_bytes": total_bytes,
        "files": normalized_files,
    }


def verify_checkpoint(
    root: Path, entry: Any, description: str
) -> dict[str, Any]:
    expected = normalize_checkpoint_entry(entry, description)
    actual_files = checkpoint_file_records(root)
    if actual_files != expected["files"]:
        expected_paths = {item["path"] for item in expected["files"]}
        actual_paths = {item["path"] for item in actual_files}
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        if missing or extra:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing[:5]))
            if extra:
                details.append("extra " + ", ".join(extra[:5]))
            raise ValueError(f"{description} file set mismatch: " + "; ".join(details))
        raise ValueError(f"{description} size or SHA-256 mismatch")
    return expected


def validate_attempt_inputs(
    merged_dir: Path,
    attempt_spec_path: Path,
    calibration_path: Path,
    construction_manifest_path: Path,
    self_diff_spec: dict[str, Any],
) -> dict[str, Any]:
    attempt_spec = load_json_object(attempt_spec_path, "attempt specification")
    calibration = load_json_object(calibration_path, "quantization calibration")
    construction = load_json_object(
        construction_manifest_path, "construction manifest"
    )
    attempt_id = self_diff_spec["attempt_id"]
    expected_attempt = {
        "attempt_id": attempt_id,
        "method": "symmetric_per_output_channel_weight_quantization_roundtrip",
        "information_policy": "merged_checkpoint_only",
        "calibration_filename": "calibration.json",
        "selected_bits": 11,
        "selection_target_relative_frobenius": 0.00125,
        "transformer_block_indices": list(range(14)),
        "expected_matrix_count": 98,
        "source_weight_dtype": "float32",
    }
    for key, expected_value in expected_attempt.items():
        if attempt_spec.get(key) != expected_value:
            raise ValueError(f"Attempt specification mismatch: {key}")

    if construction.get("hash_algorithm") != "sha256":
        raise ValueError("Construction manifest does not specify SHA-256")
    if construction.get("attempt_id") != attempt_id:
        raise ValueError("Construction manifest identity mismatch")
    if construction.get("information_policy") != "merged_checkpoint_only":
        raise ValueError("Construction manifest information policy mismatch")
    attempt_spec_hash = sha256_file(attempt_spec_path)
    calibration_hash = sha256_file(calibration_path)
    if construction.get("spec_sha256") != attempt_spec_hash:
        raise ValueError("Construction manifest specification hash mismatch")
    if construction.get("calibration_sha256") != calibration_hash:
        raise ValueError("Construction manifest calibration hash mismatch")
    require_sha256(construction.get("constructor_sha256"), "constructor script")

    expected_candidates = list(range(8, 17))
    selection_rule = attempt_spec.get("selection_rule")
    if selection_rule != {
        "candidate_bit_widths": expected_candidates,
        "criterion": "minimize_abs_aggregate_relative_frobenius_minus_target",
        "exact_tie_break": "higher_bit_width",
    }:
        raise ValueError("Attempt specification selection rule mismatch")
    if calibration.get("format_version") != 1 or calibration.get("diagnostic") != (
        "symmetric_per_output_channel_weight_quantization_calibration"
    ):
        raise ValueError("Calibration identity mismatch")
    calibration_selection = calibration.get("selection")
    calibration_quantization = calibration.get("quantization")
    candidate_results = calibration.get("candidate_results")
    if not isinstance(calibration_selection, dict) or not isinstance(
        calibration_quantization, dict
    ) or not isinstance(candidate_results, list):
        raise ValueError("Calibration selection metadata is malformed")
    if calibration_selection.get("target_relative_frobenius") != 0.00125:
        raise ValueError("Calibration target mismatch")
    if (
        calibration_selection.get("criterion") != selection_rule["criterion"]
        or calibration_selection.get("exact_tie_break")
        != selection_rule["exact_tie_break"]
        or calibration_selection.get("uses_perturbation_magnitude_only") is not True
        or calibration_quantization.get("candidate_bit_widths")
        != expected_candidates
    ):
        raise ValueError("Calibration selection rule mismatch")
    by_bits = {}
    for result in candidate_results:
        if not isinstance(result, dict):
            raise ValueError("Calibration candidate result is malformed")
        bits = result.get("bits")
        relative = result.get("aggregate_relative_frobenius_perturbation")
        if (
            type(bits) is not int
            or bits in by_bits
            or isinstance(relative, bool)
            or not isinstance(relative, (int, float))
            or not math.isfinite(relative)
            or relative < 0.0
        ):
            raise ValueError("Calibration candidate result is malformed")
        by_bits[bits] = float(relative)
    if set(by_bits) != set(expected_candidates):
        raise ValueError("Calibration candidate coverage mismatch")
    calibration_winner = min(
        expected_candidates,
        key=lambda bits: (abs(by_bits[bits] - 0.00125), -bits),
    )
    if (
        calibration_winner != 11
        or calibration_selection.get("selected_bit_width") != calibration_winner
    ):
        raise ValueError("Calibration winner mismatch")

    selection = construction.get("calibration_selection")
    selected_result = next(
        result for result in candidate_results if result.get("bits") == 11
    )
    if selection != {
        "selected_bits": 11,
        "qmax": 1023,
        "target_relative_frobenius": 0.00125,
        "criterion": selection_rule["criterion"],
        "exact_tie_break": selection_rule["exact_tie_break"],
        "selected_calibration_aggregate_relative_frobenius": selected_result.get(
            "aggregate_relative_frobenius_perturbation"
        ),
    }:
        raise ValueError("Construction manifest calibration selection mismatch")

    method = construction.get("construction")
    if not isinstance(method, dict):
        raise ValueError("Construction manifest is missing method metadata")
    expected_construction = {
        "transformer_block_indices": list(range(14)),
        "expected_matrix_count": 98,
        "eligible_tensors": attempt_spec.get("eligible_tensors"),
        "source_weight_dtype": "float32",
        "selected_bits": 11,
        "qmax": 1023,
        "quantization": attempt_spec.get("quantization"),
        "weight_transform": attempt_spec.get("weight_transform"),
        "stored_dtype": "float32",
        "standalone_checkpoint": True,
    }
    for key, expected_value in expected_construction.items():
        if method.get(key) != expected_value:
            raise ValueError(f"Construction manifest method mismatch: {key}")

    ordering = construction.get("deterministic_ordering")
    modified = construction.get("modified_matrices")
    if not isinstance(ordering, dict) or not isinstance(modified, list):
        raise ValueError("Construction manifest is missing deterministic matrix ordering")
    ordered_names = ordering.get("modified_module_names")
    matrix_names = [
        record.get("name") if isinstance(record, dict) else None for record in modified
    ]
    if (
        ordering.get("rule") != "lexicographic_by_module_name"
        or not isinstance(ordered_names, list)
        or len(ordered_names) != 98
        or ordered_names != sorted(ordered_names)
        or len(set(ordered_names)) != len(ordered_names)
        or matrix_names != ordered_names
    ):
        raise ValueError("Construction manifest deterministic ordering mismatch")

    merged_entry = verify_checkpoint(
        merged_dir, construction.get("source_checkpoint"), "merged checkpoint"
    )
    calibration_source = normalize_checkpoint_entry(
        calibration.get("source_checkpoint"), "calibration source checkpoint"
    )
    if calibration_source != merged_entry:
        raise ValueError("Calibration and construction merged checkpoints differ")
    defaults = attempt_spec.get("defaults")
    outputs = attempt_spec.get("outputs")
    output_entry = construction.get("output_checkpoint")
    if (
        not isinstance(defaults, dict)
        or not isinstance(outputs, dict)
        or not isinstance(output_entry, dict)
    ):
        raise ValueError("Attempt construction is missing checkpoint locations")
    output_root_value = defaults.get("output_root")
    directory_name = outputs.get("directory_name")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ValueError("Attempt specification has invalid output root")
    if (
        not isinstance(directory_name, str)
        or not directory_name
        or Path(directory_name).name != directory_name
        or output_entry.get("directory_name") != directory_name
    ):
        raise ValueError("Quantized checkpoint directory mismatch")
    checkpoint_dir = Path(output_root_value).expanduser() / directory_name
    quantized_entry = verify_checkpoint(
        checkpoint_dir, output_entry, "11-bit quantized checkpoint"
    )
    return {
        "calibration_sha256": calibration_hash,
        "attempt_spec_sha256": attempt_spec_hash,
        "construction_manifest_sha256": sha256_file(construction_manifest_path),
        "merged_checkpoint": merged_entry,
        "quantized_checkpoint": {
            "directory_name": directory_name,
            "directory": checkpoint_dir,
            **quantized_entry,
        },
    }


def sha256_raw_int64_tensor(tensor: Any, torch_module: Any) -> str:
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError("Cannot hash a non-tensor probe")
    if tensor.dtype != torch_module.int64 or tensor.device.type != "cpu":
        raise ValueError("Probe hash requires a CPU torch.int64 tensor")
    canonical = tensor.detach().contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def sha256_raw_float32_tensor(tensor: Any, torch_module: Any) -> str:
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError("Cannot hash a non-tensor activation")
    if tensor.dtype != torch_module.float32 or tensor.device.type != "cpu":
        raise ValueError("Activation hash requires a CPU torch.float32 tensor")
    canonical = tensor.detach().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def validate_probe_tensor(
    probe: Any, expected_shape: tuple[int, int], torch_module: Any
) -> None:
    if not isinstance(probe, torch_module.Tensor):
        raise ValueError("Frozen probe must be a tensor")
    if probe.dtype != torch_module.int64 or probe.device.type != "cpu":
        raise ValueError("Frozen probe must be a CPU torch.int64 tensor")
    if tuple(probe.shape) != expected_shape:
        raise ValueError(
            f"Frozen probe shape mismatch: expected {list(expected_shape)}, "
            f"found {list(probe.shape)}"
        )


def load_and_verify_probe(
    probe_path: Path,
    expected_serialized_sha256: str,
    expected_raw_sha256: str,
    expected_shape: tuple[int, int],
    torch_module: Any,
) -> Any:
    require_sha256(expected_serialized_sha256, "serialized probe")
    require_sha256(expected_raw_sha256, "raw probe")
    require_file(probe_path, "frozen probe")
    if sha256_file(probe_path) != expected_serialized_sha256:
        raise ValueError("Frozen probe serialized SHA-256 mismatch")
    try:
        probe = torch_module.load(probe_path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load frozen probe {probe_path}: {exc}") from exc
    validate_probe_tensor(probe, expected_shape, torch_module)
    if sha256_raw_int64_tensor(probe, torch_module) != expected_raw_sha256:
        raise ValueError("Frozen probe raw tensor SHA-256 mismatch")
    return probe.contiguous()


def verify_float32_model(
    model: Any, torch_module: Any, expected_device: str
) -> None:
    floating_parameters = [
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    ]
    if not floating_parameters:
        raise ValueError("Model has no floating-point parameters")
    unexpected = sorted(
        {
            str(parameter.dtype)
            for parameter in floating_parameters
            if parameter.dtype != torch_module.float32
        }
    )
    if unexpected:
        raise ValueError("Model has unexpected floating dtypes: " + ", ".join(unexpected))
    devices = {parameter.device.type for parameter in floating_parameters}
    if devices != {expected_device}:
        raise ValueError(
            f"Model device mismatch: expected {expected_device}, found {sorted(devices)}"
        )


def validate_loaded_model(
    model: Any,
    expected_hidden_size: int | None,
    spec: dict[str, Any],
    torch_module: Any,
    expected_device: str,
) -> tuple[Any, int]:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "qwen3":
        raise ValueError("Loaded model config is not Qwen3")
    num_hidden_layers = spec["num_hidden_layers"]
    if getattr(config, "num_hidden_layers", None) != num_hidden_layers:
        raise ValueError(f"Loaded model must have {num_hidden_layers} blocks")
    hidden_size = getattr(config, "hidden_size", None)
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("Loaded model has invalid hidden size")
    if expected_hidden_size is not None and hidden_size != expected_hidden_size:
        raise ValueError("Loaded model hidden size mismatch")
    transformer = getattr(model, "model", None)
    layers = getattr(transformer, "layers", None)
    if layers is None or len(layers) != num_hidden_layers:
        raise ValueError(f"Loaded model does not expose {num_hidden_layers} blocks")
    block_index = spec["readout_block_index"]
    if not 0 <= block_index < len(layers):
        raise ValueError(f"Loaded model does not expose block {block_index}")
    verify_float32_model(model, torch_module, expected_device)
    return transformer, hidden_size


def extract_block_output(output: Any, torch_module: Any) -> Any:
    if isinstance(output, torch_module.Tensor):
        return output
    if (
        isinstance(output, (tuple, list))
        and output
        and isinstance(output[0], torch_module.Tensor)
    ):
        return output[0]
    raise ValueError("Transformer block returned no tensor output")


class ActivationAccumulator:
    """Accumulate block outputs on CPU without retaining sample activations."""

    def __init__(
        self, sequence_length: int, hidden_size: int, torch_module: Any
    ) -> None:
        if type(sequence_length) is not int or sequence_length <= 0:
            raise ValueError("Sequence length must be a positive integer")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("Hidden size must be a positive integer")
        self.torch = torch_module
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.sum = torch_module.zeros(
            (sequence_length, hidden_size),
            dtype=torch_module.float64,
            device="cpu",
        )
        self.sample_count = 0
        self.hook_calls = 0

    def __call__(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = extract_block_output(output, self.torch)
        if hidden.ndim != 3:
            raise ValueError(
                f"Transformer block output must be rank 3, found {hidden.ndim}"
            )
        expected_tail = (self.sequence_length, self.hidden_size)
        if tuple(hidden.shape[1:]) != expected_tail:
            raise ValueError(
                "Transformer block output shape mismatch: expected "
                f"[batch, {self.sequence_length}, {self.hidden_size}], "
                f"found {list(hidden.shape)}"
            )
        if hidden.dtype != self.torch.float32:
            raise ValueError("Transformer block output must be torch.float32")
        cpu_float64 = hidden.detach().to(device="cpu", dtype=self.torch.float64)
        self.sum.add_(cpu_float64.sum(dim=0))
        self.sample_count += hidden.shape[0]
        self.hook_calls += 1

    def mean(self, expected_sample_count: int) -> Any:
        if type(expected_sample_count) is not int or expected_sample_count <= 0:
            raise ValueError("Expected sample count must be a positive integer")
        if self.sample_count != expected_sample_count:
            raise ValueError(
                "Activation sample count mismatch: expected "
                f"{expected_sample_count}, found {self.sample_count}"
            )
        if self.hook_calls == 0:
            raise ValueError("Transformer block hook was never called")
        return (self.sum / expected_sample_count).contiguous()


def compute_checkpoint_mean(
    checkpoint_dir: Path,
    probe: Any,
    expected_hidden_size: int | None,
    spec: dict[str, Any],
    torch_module: Any,
    model_class: Any,
) -> tuple[Any, int]:
    validate_probe_tensor(
        probe, (spec["sample_count"], spec["sequence_length"]), torch_module
    )
    model = model_class.from_pretrained(
        checkpoint_dir, dtype=torch_module.float32, local_files_only=True
    )
    hook_handle = None
    transformer = None
    input_ids = None
    try:
        transformer, hidden_size = validate_loaded_model(
            model, expected_hidden_size, spec, torch_module, "cpu"
        )
        device = torch_module.device("cuda")
        model.to(device)
        model.eval()
        verify_float32_model(model, torch_module, "cuda")
        accumulator = ActivationAccumulator(
            spec["sequence_length"], hidden_size, torch_module
        )
        hook_handle = transformer.layers[
            spec["readout_block_index"]
        ].register_forward_hook(accumulator)
        sample_count = spec["sample_count"]
        batch_size = spec["batch_size"]
        expected_batches = math.ceil(sample_count / batch_size)
        with torch_module.inference_mode():
            for start in range(0, sample_count, batch_size):
                input_ids = probe[start : start + batch_size].to(device)
                transformer(
                    input_ids=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
        if accumulator.hook_calls != expected_batches:
            raise ValueError(
                "Transformer block hook count mismatch: expected "
                f"{expected_batches}, found {accumulator.hook_calls}"
            )
        return accumulator.mean(sample_count), hidden_size
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        del input_ids
        del transformer
        del model
        torch_module.cuda.empty_cache()


def validate_float64_mean(
    tensor: Any,
    sequence_length: int,
    hidden_size: int,
    description: str,
    torch_module: Any,
) -> None:
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError(f"{description} is not a tensor")
    if tensor.dtype != torch_module.float64 or tensor.device.type != "cpu":
        raise ValueError(f"{description} must be a CPU torch.float64 tensor")
    if tuple(tensor.shape) != (sequence_length, hidden_size):
        raise ValueError(f"{description} has unexpected shape: {list(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{description} must be contiguous")
    if not bool(torch_module.isfinite(tensor).all().item()):
        raise ValueError(f"{description} contains non-finite values")


def build_artifact(
    merged_mean64: Any,
    quantized_mean64: Any,
    hidden_size: int,
    spec: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    sequence_length = spec["sequence_length"]
    validate_float64_mean(
        merged_mean64, sequence_length, hidden_size, "Merged mean", torch_module
    )
    validate_float64_mean(
        quantized_mean64, sequence_length, hidden_size, "Quantized mean", torch_module
    )
    difference64 = merged_mean64 - quantized_mean64
    return {
        "format_version": spec["outputs"]["artifact_format_version"],
        "metadata": {
            "attempt_id": spec["attempt_id"],
            "readout_block_index": spec["readout_block_index"],
            "num_hidden_layers": spec["num_hidden_layers"],
            "sample_count": spec["sample_count"],
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "batch_size": spec["batch_size"],
            "model_dtype": spec["model_dtype"],
            "accumulator_dtype": spec["accumulator_dtype"],
            "stored_dtype": spec["stored_dtype"],
            "difference_sign": spec["difference_sign"],
        },
        "merged_mean": merged_mean64.to(torch_module.float32).contiguous(),
        "quantized_mean": quantized_mean64.to(torch_module.float32).contiguous(),
        "difference": difference64.to(torch_module.float32).contiguous(),
    }


def build_manifest(
    spec: dict[str, Any],
    calibration_sha256: str,
    construction_manifest_sha256: str,
    attempt_spec_sha256: str,
    self_diff_spec_sha256: str,
    compute_script_sha256: str,
    merged_checkpoint: dict[str, Any],
    quantized_checkpoint: dict[str, Any],
    artifact_sha256: str,
    raw_hashes: dict[str, str],
    hidden_size: int,
) -> dict[str, Any]:
    normalized_merged = normalize_checkpoint_entry(
        merged_checkpoint, "merged checkpoint manifest entry"
    )
    normalized_quantized = normalize_checkpoint_entry(
        quantized_checkpoint, "quantized checkpoint manifest entry"
    )
    for value, description in (
        (calibration_sha256, "quantization calibration"),
        (construction_manifest_sha256, "construction manifest"),
        (attempt_spec_sha256, "attempt specification"),
        (self_diff_spec_sha256, "self-difference specification"),
        (compute_script_sha256, "self-difference script"),
        (artifact_sha256, "self-difference artifact"),
        (raw_hashes.get("merged_mean"), "merged mean"),
        (raw_hashes.get("quantized_mean"), "quantized mean"),
        (raw_hashes.get("difference"), "self-difference"),
    ):
        require_sha256(value, description)
    directory_name = quantized_checkpoint.get("directory_name")
    if not isinstance(directory_name, str) or not directory_name:
        raise ValueError("Quantized checkpoint directory name is invalid")
    return {
        "format_version": 1,
        "attempt_id": spec["attempt_id"],
        "hash_algorithm": "sha256",
        "calibration_sha256": calibration_sha256,
        "construction_manifest_sha256": construction_manifest_sha256,
        "attempt_spec_sha256": attempt_spec_sha256,
        "self_diff_spec_sha256": self_diff_spec_sha256,
        "compute_self_diff_script_sha256": compute_script_sha256,
        "input_checkpoints": {
            "merged": normalized_merged,
            "quantized": {"directory_name": directory_name, **normalized_quantized},
        },
        "probe": {
            "serialized_sha256": spec["probe"]["serialized_sha256"],
            "raw_tensor_sha256": spec["probe"]["raw_tensor_sha256"],
        },
        "method": {
            "readout_block_index": spec["readout_block_index"],
            "num_hidden_layers": spec["num_hidden_layers"],
            "sample_count": spec["sample_count"],
            "sequence_length": spec["sequence_length"],
            "hidden_size": hidden_size,
            "batch_size": spec["batch_size"],
            "model_dtype": spec["model_dtype"],
            "accumulator_dtype": spec["accumulator_dtype"],
            "stored_dtype": spec["stored_dtype"],
            "difference_sign": spec["difference_sign"],
        },
        "self_diff_artifact": {
            "serialized_sha256": artifact_sha256,
            "raw_tensors_sha256": {
                "merged_mean": raw_hashes["merged_mean"],
                "quantized_mean": raw_hashes["quantized_mean"],
                "difference": raw_hashes["difference"],
            },
        },
    }


def artifact_raw_hashes(
    artifact: dict[str, Any], torch_module: Any
) -> dict[str, str]:
    return {
        name: sha256_raw_float32_tensor(artifact[name], torch_module)
        for name in ("merged_mean", "quantized_mean", "difference")
    }


def save_outputs(
    artifact: dict[str, Any],
    spec: dict[str, Any],
    input_context: dict[str, Any],
    artifact_path: Path,
    manifest_path: Path,
    torch_module: Any,
) -> None:
    validate_output_paths(artifact_path, manifest_path)
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Could not create artifact directory {artifact_path.parent}: {exc}"
        ) from exc
    try:
        with artifact_path.open("xb") as file:
            torch_module.save(artifact, file)
    except FileExistsError as exc:
        raise ValueError(f"Self-difference artifact already exists: {artifact_path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not write self-difference artifact {artifact_path}: {exc}") from exc

    manifest = build_manifest(
        spec,
        input_context["calibration_sha256"],
        input_context["construction_manifest_sha256"],
        input_context["attempt_spec_sha256"],
        input_context["self_diff_spec_sha256"],
        input_context["compute_script_sha256"],
        input_context["merged_checkpoint"],
        input_context["quantized_checkpoint"],
        sha256_file(artifact_path),
        artifact_raw_hashes(artifact, torch_module),
        artifact["metadata"]["hidden_size"],
    )
    try:
        serialized = json.dumps(
            manifest, ensure_ascii=False, allow_nan=False, indent=2
        ) + "\n"
        with manifest_path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(serialized)
    except FileExistsError as exc:
        raise ValueError(f"Self-difference manifest already exists: {manifest_path}") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not write self-difference manifest {manifest_path}: {exc}") from exc


def compute_means(
    merged_dir: Path,
    input_context: dict[str, Any],
    probe: Any,
    spec: dict[str, Any],
    torch_module: Any,
) -> tuple[Any, Any, int]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM

    if not torch_module.cuda.is_available():
        raise ValueError("CUDA is required for self-difference computation")
    merged_mean, hidden_size = compute_checkpoint_mean(
        merged_dir, probe, None, spec, torch_module, AutoModelForCausalLM
    )
    quantized_mean, quantized_hidden_size = compute_checkpoint_mean(
        input_context["quantized_checkpoint"]["directory"],
        probe,
        hidden_size,
        spec,
        torch_module,
        AutoModelForCausalLM,
    )
    if quantized_hidden_size != hidden_size:
        raise ValueError("Quantized checkpoint hidden size mismatch")
    return merged_mean, quantized_mean, hidden_size


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
        "--attempt-spec-path",
        type=Path,
        default=Path(
            os.environ.get("ATTEMPT_SPEC_PATH", DEFAULT_ATTEMPT_SPEC_PATH)
        ).expanduser(),
    )
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=Path(
            os.environ.get("CALIBRATION_PATH", DEFAULT_CALIBRATION_PATH)
        ).expanduser(),
    )
    parser.add_argument(
        "--construction-manifest-path",
        type=Path,
        default=Path(
            os.environ.get(
                "CONSTRUCTION_MANIFEST_PATH", DEFAULT_CONSTRUCTION_MANIFEST_PATH
            )
        ).expanduser(),
    )
    parser.add_argument(
        "--probe-path",
        type=Path,
        default=Path(os.environ.get("FROZEN_PROBE_PATH", DEFAULT_PROBE_PATH)).expanduser(),
    )
    parser.add_argument(
        "--self-diff-spec-path",
        type=Path,
        default=Path(
            os.environ.get("SELF_DIFF_SPEC_PATH", DEFAULT_SELF_DIFF_SPEC_PATH)
        ).expanduser(),
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path(
            os.environ.get("SELF_DIFF_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH)
        ).expanduser(),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(
            os.environ.get("SELF_DIFF_MANIFEST_PATH", DEFAULT_SELF_DIFF_MANIFEST_PATH)
        ).expanduser(),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_output_paths(args.artifact_path, args.manifest_path)
        compute_script_path = Path(__file__).resolve()
        self_diff_spec_hash = sha256_file(args.self_diff_spec_path)
        compute_script_hash = sha256_file(compute_script_path)
        spec = load_self_diff_spec(args.self_diff_spec_path)
        if sha256_file(args.self_diff_spec_path) != self_diff_spec_hash:
            raise ValueError("Self-difference specification changed while loading")
        input_context = validate_attempt_inputs(
            args.merged_model_dir,
            args.attempt_spec_path,
            args.calibration_path,
            args.construction_manifest_path,
            spec,
        )
        input_context["self_diff_spec_sha256"] = self_diff_spec_hash
        input_context["compute_script_sha256"] = compute_script_hash

        import torch

        probe = load_and_verify_probe(
            args.probe_path,
            spec["probe"]["serialized_sha256"],
            spec["probe"]["raw_tensor_sha256"],
            (spec["sample_count"], spec["sequence_length"]),
            torch,
        )
        merged_mean, quantized_mean, hidden_size = compute_means(
            args.merged_model_dir, input_context, probe, spec, torch
        )
        artifact = build_artifact(
            merged_mean, quantized_mean, hidden_size, spec, torch
        )
        if sha256_file(args.attempt_spec_path) != input_context[
            "attempt_spec_sha256"
        ]:
            raise ValueError("Attempt specification changed during computation")
        if sha256_file(args.calibration_path) != input_context[
            "calibration_sha256"
        ]:
            raise ValueError("Calibration changed during computation")
        if sha256_file(args.construction_manifest_path) != input_context[
            "construction_manifest_sha256"
        ]:
            raise ValueError("Construction manifest changed during computation")
        if sha256_file(args.self_diff_spec_path) != self_diff_spec_hash:
            raise ValueError("Self-difference specification changed during computation")
        if sha256_file(compute_script_path) != compute_script_hash:
            raise ValueError("Self-difference script changed during computation")
        if sha256_file(args.probe_path) != spec["probe"]["serialized_sha256"]:
            raise ValueError("Frozen probe changed during computation")
        save_outputs(
            artifact,
            spec,
            input_context,
            args.artifact_path,
            args.manifest_path,
            torch,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.artifact_path}")
    print(f"Wrote {args.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
