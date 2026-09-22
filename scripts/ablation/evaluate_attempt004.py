#!/usr/bin/env python3

"""Evaluate Attempt 004 against the frozen post-blind-stage oracle."""

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
DEFAULT_EVALUATION_SPEC_PATH = ATTEMPT_DIRECTORY / "evaluation_spec.json"
DEFAULT_ATTEMPT_SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
DEFAULT_CALIBRATION_PATH = ATTEMPT_DIRECTORY / "calibration.json"
DEFAULT_CONSTRUCTION_MANIFEST_PATH = ATTEMPT_DIRECTORY / "construction-manifest.json"
DEFAULT_SELF_DIFF_SPEC_PATH = ATTEMPT_DIRECTORY / "self_diff_spec.json"
DEFAULT_SELF_DIFF_MANIFEST_PATH = ATTEMPT_DIRECTORY / "self-diff-manifest.json"
DEFAULT_OUTPUT_PATH = ATTEMPT_DIRECTORY / "evaluation.json"

DEFAULT_DOWNLOADED_HASHES_PATH = PROJECT / "downloaded-model-hashes.json"
DEFAULT_MERGED_HASHES_PATH = PROJECT / "merged-model-hashes.json"
DEFAULT_ORACLE_PROBE_MANIFEST_PATH = PROJECT / "oracle-probe-manifest.json"
DEFAULT_ORACLE_MANIFEST_PATH = PROJECT / "oracle-adl-manifest.json"
DEFAULT_ORACLE_LOGIT_LENS_PATH = PROJECT / "oracle-logit-lens.json"
DEFAULT_CONSTRUCTOR_SCRIPT_PATH = (
    PROJECT / "scripts" / "ablation" / "construct_quantized_roundtrip.py"
)
DEFAULT_SELF_DIFF_SCRIPT_PATH = (
    PROJECT / "scripts" / "ablation" / "compute_self_diff_attempt004.py"
)
DEFAULT_MERGE_SCRIPT_PATH = PROJECT / "scripts" / "merge_lora.py"
DEFAULT_FREEZE_PROBE_SCRIPT_PATH = PROJECT / "scripts" / "freeze_oracle_probe.py"
DEFAULT_ORACLE_SCRIPT_PATH = PROJECT / "scripts" / "compute_oracle_adl.py"

DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_BASE_TOKENIZER_DIR = Path("/root/model-diff-scratch/models/base")
DEFAULT_SELF_DIFF_ARTIFACT_PATH = Path(
    "/root/model-diff-scratch/artifacts/attempt004_self_diff/self_diff.pt"
)
DEFAULT_ORACLE_ARTIFACT_PATH = Path(
    "/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt"
)

FROZEN_EVALUATION_SPEC = {
    "format_version": 1,
    "attempt_id": "004_quantize11_prefix0_13",
    "positions": [0, 1, 2, 3, 4],
    "primary_metric": "cosine_similarity_self_difference_oracle_difference",
    "geometry": {
        "dtype": "float64",
        "metrics": [
            "cosine_similarity",
            "self_diff_norm",
            "oracle_diff_norm",
            "self_to_oracle_norm_ratio",
        ],
        "zero_vector_policy": {
            "cosine_similarity": "null_if_either_vector_has_zero_norm",
            "self_to_oracle_norm_ratio": "null_if_oracle_vector_has_zero_norm",
        },
    },
    "logit_lens": {
        "top_k": 20,
        "tokenizer_source": "base",
        "positive": "softmax(lm_head(model.model.norm(latent)))",
        "negative": "softmax(lm_head(-model.model.norm(latent)))",
    },
    "evaluation_policy": {
        "single_attempt_result": True,
        "automatic_ranking": False,
        "semantic_grading": False,
        "keyword_search": False,
        "best_position_selection": False,
        "post_result_bit_width_tuning": False,
        "attempt002_003_thresholds": False,
    },
    "defaults": {
        "merged_model_directory": "/root/model-diff-scratch/models/merged",
        "base_tokenizer_directory": "/root/model-diff-scratch/models/base",
        "self_diff_artifact_path": (
            "/root/model-diff-scratch/artifacts/attempt004_self_diff/self_diff.pt"
        ),
        "oracle_artifact_path": (
            "/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt"
        ),
    },
    "output": {
        "filename": "evaluation.json",
        "timestamps": False,
        "host_metadata": False,
        "gpu_metadata": False,
    },
}

ORACLE_VECTOR_TYPES = ("difference", "base_mean", "ft_mean")
SELF_VECTOR_TYPES = ("merged_mean", "quantized_mean", "difference")
EXPECTED_SHAPE = (128, 2048)


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


def require_output_absent(path: Path) -> None:
    if path_exists(path, "evaluation output"):
        raise ValueError(f"Evaluation output already exists: {path}")
    require_directory(path.parent, "evaluation output directory")


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    require_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {description}: {path}")
    return value


def load_evaluation_spec(path: Path) -> dict[str, Any]:
    spec = load_json_object(path, "evaluation specification")
    if spec != FROZEN_EVALUATION_SPEC:
        raise ValueError("Evaluation specification does not match the frozen definition")
    return spec


def verify_file_sha256(path: Path, expected: Any, description: str) -> str:
    require_file(path, description)
    expected_hash = require_sha256(expected, description)
    try:
        actual_hash = sha256_file(path)
    except OSError as exc:
        raise ValueError(f"Could not hash {description} {path}: {exc}") from exc
    if actual_hash != expected_hash:
        raise ValueError(f"{description} SHA-256 mismatch")
    return actual_hash


def verify_manifest_file_link(
    manifest: dict[str, Any],
    key: str,
    path: Path,
    description: str,
) -> str:
    if key not in manifest:
        raise ValueError(f"Missing {description} hash link: {key}")
    return verify_file_sha256(path, manifest[key], description)


def normalize_file_records(records: Any, description: str) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"Malformed {description}: files")
    normalized = []
    seen = set()
    for record in records:
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
            or ".cache" in parsed.parts
        ):
            raise ValueError(f"Malformed {description} file path: {relative}")
        if relative in seen:
            raise ValueError(f"Duplicate {description} file path: {relative}")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(f"Malformed {description} size for {relative}")
        require_sha256(digest, f"{description} file {relative}")
        normalized.append(
            {"path": relative, "size_bytes": size_bytes, "sha256": digest}
        )
        seen.add(relative)
    normalized.sort(key=lambda item: item["path"])
    if not normalized:
        raise ValueError(f"Malformed {description}: empty file set")
    return normalized


def normalize_checkpoint_entry(entry: Any, description: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"Malformed {description}: expected an object")
    files = normalize_file_records(entry.get("files"), description)
    file_count = entry.get("file_count")
    total_bytes = entry.get("total_bytes")
    if type(file_count) is not int or file_count != len(files):
        raise ValueError(f"Malformed {description}: file_count")
    expected_total = sum(item["size_bytes"] for item in files)
    if type(total_bytes) is not int or total_bytes != expected_total:
        raise ValueError(f"Malformed {description}: total_bytes")
    return {"file_count": file_count, "total_bytes": total_bytes, "files": files}


def directory_file_records(root: Path) -> list[dict[str, Any]]:
    require_directory(root, "artifact directory")
    candidates = []
    try:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if ".cache" in relative.parts or not path.is_file():
                continue
            candidates.append((relative.as_posix(), path))
    except OSError as exc:
        raise ValueError(f"Could not enumerate artifact {root}: {exc}") from exc
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
            raise ValueError(f"Could not hash artifact file {path}: {exc}") from exc
    if not records:
        raise ValueError(f"Artifact contains no files: {root}")
    return records


def verify_checkpoint(root: Path, entry: Any, description: str) -> dict[str, Any]:
    expected = normalize_checkpoint_entry(entry, description)
    actual = directory_file_records(root)
    if actual != expected["files"]:
        expected_paths = {item["path"] for item in expected["files"]}
        actual_paths = {item["path"] for item in actual}
        if expected_paths != actual_paths:
            raise ValueError(f"{description} file set mismatch")
        raise ValueError(f"{description} size or SHA-256 mismatch")
    return expected


def normalize_download_entry(entry: Any, description: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(f"Malformed {description}")
    repo_id = entry.get("repo_id")
    revision = entry.get("revision")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"Malformed {description} repo_id")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"Malformed {description} revision")
    files = normalize_file_records(entry.get("files"), description)
    if entry.get("file_count") != len(files):
        raise ValueError(f"Malformed {description} file_count")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "file_count": len(files),
        "files": files,
    }


def verify_download_artifact(
    root: Path, entry: Any, description: str
) -> dict[str, Any]:
    expected = normalize_download_entry(entry, description)
    actual = directory_file_records(root)
    if actual != expected["files"]:
        expected_paths = {item["path"] for item in expected["files"]}
        actual_paths = {item["path"] for item in actual}
        if expected_paths != actual_paths:
            raise ValueError(f"{description} file set mismatch")
        raise ValueError(f"{description} size or SHA-256 mismatch")
    return expected


def model_identity(entry: dict[str, Any]) -> dict[str, str]:
    return {"repo_id": entry["repo_id"], "revision": entry["revision"]}


def validate_calibration_winner(calibration: dict[str, Any], attempt_spec: dict[str, Any]) -> dict[str, Any]:
    """Audit the already frozen blind selection; do not select a new bit width."""
    rule = attempt_spec.get("selection_rule")
    selection = calibration.get("selection")
    candidates = calibration.get("candidate_results")
    if not isinstance(rule, dict) or not isinstance(selection, dict) or not isinstance(candidates, list):
        raise ValueError("Malformed calibration selection")
    widths = list(range(8, 17))
    expected_rule = {
        "candidate_bit_widths": widths,
        "criterion": "minimize_abs_aggregate_relative_frobenius_minus_target",
        "exact_tie_break": "higher_bit_width",
    }
    if rule != expected_rule or selection.get("target_relative_frobenius") != 0.00125:
        raise ValueError("Frozen calibration selection rule mismatch")
    if calibration.get("quantization", {}).get("candidate_bit_widths") != widths:
        raise ValueError("Calibration candidate bit widths mismatch")
    for key in ("criterion", "exact_tie_break"):
        if selection.get(key) != rule[key]:
            raise ValueError(f"Calibration selection rule mismatch: {key}")
    if selection.get("uses_perturbation_magnitude_only") is not True:
        raise ValueError("Calibration was not blind-selected by perturbation magnitude")
    if [record.get("bits") for record in candidates if isinstance(record, dict)] != widths or len(candidates) != len(widths):
        raise ValueError("Calibration candidate bit widths mismatch")
    errors = []
    for record in candidates:
        bits = record["bits"]
        relative = record.get("aggregate_relative_frobenius_perturbation")
        error = record.get("absolute_target_error")
        if (record.get("qmax") != 2 ** (bits - 1) - 1 or
            type(relative) is not float or not math.isfinite(relative) or relative < 0 or
            type(error) is not float or not math.isfinite(error) or
            not math.isclose(error, abs(relative - 0.00125), rel_tol=0, abs_tol=1e-15)):
            raise ValueError("Calibration candidate perturbation mismatch")
        errors.append((error, -bits))
    winner = candidates[min(range(len(candidates)), key=lambda index: errors[index])]
    if (winner["bits"] != 11 or selection.get("selected_bit_width") != 11 or
        attempt_spec.get("selected_bits") != 11 or
        selection.get("selected_absolute_target_error") != winner["absolute_target_error"] or
        attempt_spec.get("selection_target_relative_frobenius") != 0.00125):
        raise ValueError("Calibration blind-selected winner is not frozen 11 bits")
    return winner


def validate_attempt_provenance(
    merged_dir: Path,
    spec: dict[str, Any],
    *,
    attempt_spec_path: Path,
    calibration_path: Path,
    construction_manifest_path: Path,
    self_diff_spec_path: Path,
    self_diff_manifest_path: Path,
    constructor_script_path: Path,
    self_diff_script_path: Path,
    merged_hashes_path: Path,
) -> dict[str, Any]:
    attempt_spec = load_json_object(attempt_spec_path, "Attempt 004 specification")
    calibration = load_json_object(calibration_path, "Attempt 004 calibration")
    construction = load_json_object(construction_manifest_path, "Attempt 004 construction manifest")
    self_spec = load_json_object(self_diff_spec_path, "Attempt 004 self-difference specification")
    self_manifest = load_json_object(self_diff_manifest_path, "Attempt 004 self-difference manifest")
    merged_manifest = load_json_object(merged_hashes_path, "merged-model manifest")
    attempt_id = spec["attempt_id"]
    blocks = list(range(14))
    if (attempt_spec.get("format_version") != 1 or attempt_spec.get("attempt_id") != attempt_id or
        attempt_spec.get("method") != "symmetric_per_output_channel_weight_quantization_roundtrip" or
        attempt_spec.get("information_policy") != "merged_checkpoint_only" or
        attempt_spec.get("transformer_block_indices") != blocks or
        attempt_spec.get("expected_matrix_count") != 98 or
        attempt_spec.get("source_weight_dtype") != "float32" or
        attempt_spec.get("calibration_filename") != "calibration.json"):
        raise ValueError("Attempt 004 specification mismatch")
    quantization = attempt_spec.get("quantization")
    if not isinstance(quantization, dict) or quantization.get("bits") != 11 or quantization.get("qmax") != 1023:
        raise ValueError("Attempt 004 frozen quantization mismatch")
    if (quantization.get("scheme") != "symmetric_per_output_channel_uniform" or
        quantization.get("pre_cast_dtype") != "float64" or
        quantization.get("rounding") != "torch.round" or
        quantization.get("stored_dtype") != "float32" or
        quantization.get("randomness") != "none" or
        quantization.get("integer_inference_kernels") is not False or
        quantization.get("activation_quantization") is not False):
        raise ValueError("Attempt 004 quantization method mismatch")
    if (calibration.get("format_version") != 1 or
        calibration.get("diagnostic") != "symmetric_per_output_channel_weight_quantization_calibration" or
        calibration.get("scope", {}).get("transformer_block_indices") != blocks or
        calibration.get("scope", {}).get("actual_matrix_count") != 98 or
        calibration.get("quantization", {}).get("candidate_bit_widths") != list(range(8, 17))):
        raise ValueError("Calibration provenance mismatch")
    scope = calibration["scope"]
    eligible = calibration.get("eligible_matrices")
    if (scope.get("expected_matrix_count") != 98 or
        scope.get("source_weight_dtype") != "float32" or
        scope.get("module_ordering") != "lexicographic_by_module_name" or
        not isinstance(eligible, list) or len(eligible) != 98 or
        any(not isinstance(record, dict) or type(record.get("scalar_count")) is not int
            for record in eligible)):
        raise ValueError("Calibration matrix scope mismatch")
    if calibration.get("total_scalar_count") != sum(record["scalar_count"] for record in eligible):
        raise ValueError("Calibration scalar count mismatch")
    names = [record.get("module_name") for record in eligible]
    if any(not isinstance(name, str) for name in names) or len(set(names)) != 98 or names != sorted(names):
        raise ValueError("Calibration eligible matrix ordering mismatch")
    for record in eligible:
        shape = record.get("shape")
        if (not isinstance(shape, list) or len(shape) != 2 or
            any(type(dimension) is not int or dimension <= 0 for dimension in shape) or
            record.get("scalar_count") != shape[0] * shape[1]):
            raise ValueError("Calibration eligible matrix shape mismatch")
    winner = validate_calibration_winner(calibration, attempt_spec)
    if construction.get("format_version") != 1 or construction.get("hash_algorithm") != "sha256" or construction.get("attempt_id") != attempt_id or construction.get("information_policy") != "merged_checkpoint_only":
        raise ValueError("Construction manifest identity or policy mismatch")
    for key, path, description in (
        ("spec_sha256", attempt_spec_path, "construction specification"),
        ("calibration_sha256", calibration_path, "construction calibration"),
        ("constructor_sha256", constructor_script_path, "construction script"),
    ):
        verify_manifest_file_link(construction, key, path, description)
    selection = construction.get("calibration_selection")
    if selection != {
        "selected_bits": 11, "qmax": 1023, "target_relative_frobenius": 0.00125,
        "criterion": attempt_spec["selection_rule"]["criterion"],
        "exact_tie_break": "higher_bit_width",
        "selected_calibration_aggregate_relative_frobenius": winner["aggregate_relative_frobenius_perturbation"],
    }:
        raise ValueError("Construction calibration selection mismatch")
    method_construction = construction.get("construction")
    if not isinstance(method_construction, dict):
        raise ValueError("Construction method missing")
    for key, expected in (
        ("transformer_block_indices", blocks), ("expected_matrix_count", 98),
        ("eligible_tensors", attempt_spec.get("eligible_tensors")),
        ("source_weight_dtype", "float32"), ("selected_bits", 11),
        ("qmax", 1023), ("quantization", quantization),
        ("weight_transform", attempt_spec.get("weight_transform")),
        ("stored_dtype", "float32"), ("standalone_checkpoint", True),
    ):
        if method_construction.get(key) != expected:
            raise ValueError(f"Construction method mismatch: {key}")
    expected_self = {
        "readout_block_index": 13, "num_hidden_layers": 28, "sample_count": 10_000,
        "sequence_length": 128, "batch_size": 32, "model_dtype": "float32",
        "accumulator_dtype": "float64", "stored_dtype": "float32",
        "difference_sign": "merged_minus_quantized",
    }
    if (self_spec.get("format_version") != 1 or self_spec.get("attempt_id") != attempt_id or
        self_spec.get("input_information_policy") != "canonical_merged_attempt004_quantized_checkpoint_and_frozen_generic_probe_only"):
        raise ValueError("Self-difference specification identity mismatch")
    for key, expected in expected_self.items():
        if self_spec.get(key) != expected:
            raise ValueError(f"Self-difference specification mismatch: {key}")
    if (self_manifest.get("format_version") != 1 or self_manifest.get("hash_algorithm") != "sha256" or
        self_manifest.get("attempt_id") != attempt_id):
        raise ValueError("Self-difference manifest identity mismatch")
    for key, path, description in (
        ("calibration_sha256", calibration_path, "self-difference calibration"),
        ("construction_manifest_sha256", construction_manifest_path, "self-difference construction manifest"),
        ("attempt_spec_sha256", attempt_spec_path, "self-difference attempt specification"),
        ("self_diff_spec_sha256", self_diff_spec_path, "self-difference specification"),
        ("compute_self_diff_script_sha256", self_diff_script_path, "self-difference compute script"),
    ):
        verify_manifest_file_link(self_manifest, key, path, description)
    method = self_manifest.get("method")
    if not isinstance(method, dict) or method.get("hidden_size") != EXPECTED_SHAPE[1]:
        raise ValueError("Self-difference method mismatch")
    for key, expected in expected_self.items():
        if method.get(key) != expected:
            raise ValueError(f"Self-difference method mismatch: {key}")
    probe = self_manifest.get("probe")
    if not isinstance(probe, dict) or self_spec.get("probe", {}).get("serialized_sha256") != probe.get("serialized_sha256") or self_spec.get("probe", {}).get("raw_tensor_sha256") != probe.get("raw_tensor_sha256"):
        raise ValueError("Self-difference probe provenance mismatch")
    inputs = self_manifest.get("input_checkpoints")
    if not isinstance(inputs, dict):
        raise ValueError("Self-difference input checkpoints missing")
    sources = [normalize_checkpoint_entry(entry, "merged checkpoint") for entry in (
        calibration.get("source_checkpoint"), construction.get("source_checkpoint"),
        inputs.get("merged"), merged_manifest,
    )]
    if any(source != sources[0] for source in sources[1:]):
        raise ValueError("Merged checkpoint provenance disagrees across manifests")
    constructed = construction.get("output_checkpoint")
    quantized = inputs.get("quantized")
    if (not isinstance(constructed, dict) or not isinstance(quantized, dict) or
        constructed.get("directory_name") != "model" or quantized.get("directory_name") != "model" or
        normalize_checkpoint_entry(constructed, "construction output checkpoint") !=
        normalize_checkpoint_entry(quantized, "self-difference quantized checkpoint")):
        raise ValueError("Quantized checkpoint provenance mismatch")
    checkpoint = verify_checkpoint(merged_dir, sources[0], "canonical merged checkpoint")
    return {
        "calibration_sha256": sha256_file(calibration_path),
        "attempt_spec_sha256": sha256_file(attempt_spec_path),
        "construction_manifest_sha256": sha256_file(construction_manifest_path),
        "constructor_script_sha256": sha256_file(constructor_script_path),
        "self_diff_spec_sha256": sha256_file(self_diff_spec_path),
        "self_diff_manifest_sha256": sha256_file(self_diff_manifest_path),
        "self_diff_script_sha256": sha256_file(self_diff_script_path),
        "self_diff_manifest": self_manifest,
        "method": method,
        "hidden_size": EXPECTED_SHAPE[1],
        "merged_checkpoint": checkpoint,
        "merged_model_hashes_sha256": sha256_file(merged_hashes_path),
    }

def validate_oracle_logit_lens_reference(
    document: dict[str, Any],
    spec: dict[str, Any],
    oracle_manifest_hash: str,
    oracle_artifact_hash: str,
    downloaded_hash: str,
    merged_hash: str,
    probe_manifest_hash: str,
    base_identity: dict[str, str],
    fine_tuned_identity: dict[str, str],
) -> None:
    if document.get("format_version") != 1:
        raise ValueError("Unsupported oracle Logit Lens format")
    provenance = document.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Oracle Logit Lens is missing provenance")
    expected_provenance = {
        "base": base_identity,
        "fine_tuned": fine_tuned_identity,
        "downloaded_model_hashes_sha256": downloaded_hash,
        "merged_model_hashes_sha256": merged_hash,
        "oracle_probe_manifest_sha256": probe_manifest_hash,
        "oracle_adl_manifest_sha256": oracle_manifest_hash,
        "oracle_adl_sha256": oracle_artifact_hash,
        "tokenizer_source": "base",
        "model_dtype": "float32",
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(f"Oracle Logit Lens provenance mismatch: {key}")
    expected_method = {
        "positions": spec["positions"],
        "vector_types": list(ORACLE_VECTOR_TYPES),
        "top_k": spec["logit_lens"]["top_k"],
        "positive": spec["logit_lens"]["positive"],
        "negative": spec["logit_lens"]["negative"],
    }
    if document.get("method") != expected_method:
        raise ValueError("Oracle Logit Lens method mismatch")
    positions = document.get("positions")
    if not isinstance(positions, list) or len(positions) != len(spec["positions"]):
        raise ValueError("Oracle Logit Lens position coverage mismatch")
    by_position = {}
    for record in positions:
        if not isinstance(record, dict) or record.get("position") in by_position:
            raise ValueError("Malformed or duplicate oracle Logit Lens position")
        by_position[record.get("position")] = record
    if set(by_position) != set(spec["positions"]):
        raise ValueError("Oracle Logit Lens position coverage mismatch")
    for position in spec["positions"]:
        for vector_type in ORACLE_VECTOR_TYPES:
            directions = by_position[position].get(vector_type)
            if not isinstance(directions, dict):
                raise ValueError("Malformed oracle Logit Lens vector result")
            for direction in ("positive", "negative"):
                tokens = directions.get(direction)
                if not isinstance(tokens, list) or len(tokens) != 20:
                    raise ValueError("Oracle Logit Lens top-k coverage mismatch")


def validate_oracle_provenance(
    base_dir: Path,
    spec: dict[str, Any],
    *,
    downloaded_hashes_path: Path,
    merged_hashes_path: Path,
    probe_manifest_path: Path,
    oracle_manifest_path: Path,
    oracle_logit_lens_path: Path,
    merge_script_path: Path,
    freeze_probe_script_path: Path,
    oracle_script_path: Path,
) -> dict[str, Any]:
    """Validate oracle provenance and the exact local base tokenizer source."""
    downloaded = load_json_object(downloaded_hashes_path, "downloaded-model manifest")
    merged = load_json_object(merged_hashes_path, "merged-model manifest")
    probe = load_json_object(probe_manifest_path, "oracle-probe manifest")
    oracle = load_json_object(oracle_manifest_path, "oracle ADL manifest")
    oracle_lens = load_json_object(
        oracle_logit_lens_path, "committed oracle Logit Lens"
    )
    for manifest, description in (
        (downloaded, "downloaded-model manifest"),
        (merged, "merged-model manifest"),
        (oracle, "oracle ADL manifest"),
    ):
        if manifest.get("hash_algorithm") != "sha256":
            raise ValueError(f"{description} does not specify SHA-256")

    models = downloaded.get("models")
    if not isinstance(models, dict):
        raise ValueError("Downloaded-model manifest is missing models")
    base_entry = normalize_download_entry(models.get("base"), "base model")
    adapter_entry = normalize_download_entry(models.get("adapter"), "adapter")
    base_identity = model_identity(base_entry)
    fine_tuned_identity = model_identity(adapter_entry)
    verified_base = verify_download_artifact(base_dir, base_entry, "base tokenizer source")

    downloaded_hash = sha256_file(downloaded_hashes_path)
    merged_hash = sha256_file(merged_hashes_path)
    probe_hash = sha256_file(probe_manifest_path)
    oracle_hash = sha256_file(oracle_manifest_path)
    oracle_lens_hash = sha256_file(oracle_logit_lens_path)
    if merged.get("downloaded_model_hashes_sha256") != downloaded_hash:
        raise ValueError("Merged-model downloaded provenance mismatch")
    if merged.get("base") != base_identity or merged.get("adapter") != fine_tuned_identity:
        raise ValueError("Merged-model identity provenance mismatch")
    verify_manifest_file_link(merged, "merge_script_sha256", merge_script_path, "merge script")
    if merged.get("merge_metadata") != {
        "dtype": "float32",
        "safe_merge": True,
        "tokenizer_source": "base",
    }:
        raise ValueError("Merged-model method provenance mismatch")
    if probe.get("downloaded_model_hashes_sha256") != downloaded_hash:
        raise ValueError("Oracle-probe downloaded provenance mismatch")
    verify_manifest_file_link(
        probe,
        "freeze_oracle_probe_script_sha256",
        freeze_probe_script_path,
        "freeze-probe script",
    )
    expected_oracle_links = {
        "downloaded_model_hashes_sha256": downloaded_hash,
        "merged_model_hashes_sha256": merged_hash,
        "oracle_probe_manifest_sha256": probe_hash,
        "compute_oracle_adl_script_sha256": sha256_file(oracle_script_path),
    }
    for key, expected in expected_oracle_links.items():
        if oracle.get(key) != expected:
            raise ValueError(f"Oracle manifest provenance mismatch: {key}")
    if oracle.get("base") != base_identity or oracle.get("fine_tuned") != fine_tuned_identity:
        raise ValueError("Oracle manifest model identity mismatch")
    if oracle.get("probe") != {
        "serialized_sha256": probe.get("fineweb_tokens_sha256"),
        "raw_tensor_sha256": probe.get("raw_tensor_sha256"),
    }:
        raise ValueError("Oracle manifest probe provenance mismatch")
    expected_oracle_method = {
        "relative_layer": 0.5,
        "layer_index": 13,
        "num_layers": 28,
        "sample_count": 10_000,
        "sequence_length": 128,
        "hidden_size": 2048,
        "batch_size": 32,
        "model_dtype": "float32",
        "accumulator_dtype": "float64",
        "stored_dtype": "float32",
    }
    for key, expected in expected_oracle_method.items():
        if oracle.get(key) != expected:
            raise ValueError(f"Oracle manifest method mismatch: {key}")
    oracle_artifact_hash = require_sha256(oracle.get("oracle_adl_sha256"), "oracle artifact")
    raw = oracle.get("raw_tensors_sha256")
    if not isinstance(raw, dict) or set(raw) != set(ORACLE_VECTOR_TYPES):
        raise ValueError("Malformed oracle raw tensor hashes")
    for name in ORACLE_VECTOR_TYPES:
        require_sha256(raw[name], f"oracle {name} tensor")
    validate_oracle_logit_lens_reference(
        oracle_lens,
        spec,
        oracle_hash,
        oracle_artifact_hash,
        downloaded_hash,
        merged_hash,
        probe_hash,
        base_identity,
        fine_tuned_identity,
    )
    return {
        "base": base_identity,
        "fine_tuned": fine_tuned_identity,
        "base_tokenizer_source": verified_base,
        "downloaded_model_hashes_sha256": downloaded_hash,
        "merged_model_hashes_sha256": merged_hash,
        "oracle_probe_manifest_sha256": probe_hash,
        "oracle_manifest_sha256": oracle_hash,
        "oracle_logit_lens_sha256": oracle_lens_hash,
        "oracle_manifest": oracle,
        "hidden_size": 2048,
    }


def verify_merged_mean_hash_gate(
    self_manifest: dict[str, Any], oracle_manifest: dict[str, Any]
) -> str:
    """Require the independently frozen merged and fine-tuned means to agree."""
    self_raw = self_manifest.get("self_diff_artifact", {}).get("raw_tensors_sha256")
    oracle_raw = oracle_manifest.get("raw_tensors_sha256")
    if not isinstance(self_raw, dict) or not isinstance(oracle_raw, dict):
        raise ValueError("Merged activation mean hash metadata is missing")
    self_hash = require_sha256(self_raw.get("merged_mean"), "self merged_mean")
    oracle_hash = require_sha256(oracle_raw.get("ft_mean"), "oracle ft_mean")
    if self_hash != oracle_hash:
        raise ValueError("Merged activation mean disagrees across frozen artifacts")
    return self_hash


def verify_inputs_before_loading(
    merged_dir: Path,
    base_dir: Path,
    self_diff_artifact_path: Path,
    oracle_artifact_path: Path,
    evaluation_spec: dict[str, Any],
    *,
    attempt_spec_path: Path = DEFAULT_ATTEMPT_SPEC_PATH,
    calibration_path: Path = DEFAULT_CALIBRATION_PATH,
    construction_manifest_path: Path = DEFAULT_CONSTRUCTION_MANIFEST_PATH,
    self_diff_spec_path: Path = DEFAULT_SELF_DIFF_SPEC_PATH,
    self_diff_manifest_path: Path = DEFAULT_SELF_DIFF_MANIFEST_PATH,
    downloaded_hashes_path: Path = DEFAULT_DOWNLOADED_HASHES_PATH,
    merged_hashes_path: Path = DEFAULT_MERGED_HASHES_PATH,
    probe_manifest_path: Path = DEFAULT_ORACLE_PROBE_MANIFEST_PATH,
    oracle_manifest_path: Path = DEFAULT_ORACLE_MANIFEST_PATH,
    oracle_logit_lens_path: Path = DEFAULT_ORACLE_LOGIT_LENS_PATH,
    constructor_script_path: Path = DEFAULT_CONSTRUCTOR_SCRIPT_PATH,
    self_diff_script_path: Path = DEFAULT_SELF_DIFF_SCRIPT_PATH,
    merge_script_path: Path = DEFAULT_MERGE_SCRIPT_PATH,
    freeze_probe_script_path: Path = DEFAULT_FREEZE_PROBE_SCRIPT_PATH,
    oracle_script_path: Path = DEFAULT_ORACLE_SCRIPT_PATH,
) -> dict[str, Any]:
    require_directory(merged_dir, "canonical merged checkpoint")
    require_directory(base_dir, "base tokenizer source")
    require_file(self_diff_artifact_path, "self-difference artifact")
    require_file(oracle_artifact_path, "oracle artifact")
    attempt = validate_attempt_provenance(
        merged_dir,
        evaluation_spec,
        attempt_spec_path=attempt_spec_path,
        calibration_path=calibration_path,
        construction_manifest_path=construction_manifest_path,
        self_diff_spec_path=self_diff_spec_path,
        self_diff_manifest_path=self_diff_manifest_path,
        constructor_script_path=constructor_script_path,
        self_diff_script_path=self_diff_script_path,
        merged_hashes_path=merged_hashes_path,
    )
    oracle = validate_oracle_provenance(
        base_dir,
        evaluation_spec,
        downloaded_hashes_path=downloaded_hashes_path,
        merged_hashes_path=merged_hashes_path,
        probe_manifest_path=probe_manifest_path,
        oracle_manifest_path=oracle_manifest_path,
        oracle_logit_lens_path=oracle_logit_lens_path,
        merge_script_path=merge_script_path,
        freeze_probe_script_path=freeze_probe_script_path,
        oracle_script_path=oracle_script_path,
    )
    if attempt["merged_model_hashes_sha256"] != oracle["merged_model_hashes_sha256"]:
        raise ValueError("Attempt and oracle merged-checkpoint provenance mismatch")
    if attempt["hidden_size"] != oracle["hidden_size"]:
        raise ValueError("Self-difference and oracle hidden sizes differ")
    if attempt["self_diff_manifest"].get("probe") != oracle["oracle_manifest"].get("probe"):
        raise ValueError("Self-difference and oracle frozen probes differ")
    gate_hash = verify_merged_mean_hash_gate(
        attempt["self_diff_manifest"], oracle["oracle_manifest"]
    )
    artifact_meta = attempt["self_diff_manifest"].get("self_diff_artifact")
    if not isinstance(artifact_meta, dict):
        raise ValueError("Self-difference manifest is missing artifact metadata")
    self_hash = verify_file_sha256(
        self_diff_artifact_path,
        artifact_meta.get("serialized_sha256"),
        "self-difference artifact",
    )
    oracle_hash = verify_file_sha256(
        oracle_artifact_path,
        oracle["oracle_manifest"].get("oracle_adl_sha256"),
        "oracle artifact",
    )
    return {
        "attempt": attempt,
        "oracle": oracle,
        "hidden_size": EXPECTED_SHAPE[1],
        "merged_mean_raw_sha256_gate": gate_hash,
        "self_diff_artifact_sha256": self_hash,
        "oracle_artifact_sha256": oracle_hash,
    }


def sha256_raw_float32_tensor(tensor: Any, torch_module: Any) -> str:
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError("Cannot hash a non-tensor activation")
    if tensor.dtype != torch_module.float32 or tensor.device.type != "cpu":
        raise ValueError("Activation hash requires a CPU torch.float32 tensor")
    canonical = tensor.detach().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def validate_activation_tensor(
    tensor: Any,
    description: str,
    expected_hash: Any,
    torch_module: Any,
) -> Any:
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError(f"{description} is not a tensor")
    if tensor.dtype != torch_module.float32 or tensor.device.type != "cpu":
        raise ValueError(f"{description} must be a CPU torch.float32 tensor")
    if tuple(tensor.shape) != EXPECTED_SHAPE:
        raise ValueError(
            f"{description} shape mismatch: expected {list(EXPECTED_SHAPE)}, "
            f"found {list(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{description} must be contiguous")
    if not bool(torch_module.isfinite(tensor).all().item()):
        raise ValueError(f"{description} contains non-finite values")
    expected = require_sha256(expected_hash, f"{description} raw tensor")
    if sha256_raw_float32_tensor(tensor, torch_module) != expected:
        raise ValueError(f"{description} raw tensor SHA-256 mismatch")
    return tensor


def load_torch_artifact(path: Path, description: str, torch_module: Any) -> Any:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load {description} {path}: {exc}") from exc


def load_and_validate_self_diff(
    path: Path,
    context: dict[str, Any],
    spec: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    artifact = load_torch_artifact(path, "self-difference artifact", torch_module)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError("Malformed self-difference artifact")
    method = context["attempt"]["method"]
    expected_metadata = {
        "attempt_id": spec["attempt_id"],
        **{
            key: method[key]
            for key in (
                "readout_block_index",
                "num_hidden_layers",
                "sample_count",
                "sequence_length",
                "hidden_size",
                "batch_size",
                "model_dtype",
                "accumulator_dtype",
                "stored_dtype",
                "difference_sign",
            )
        },
    }
    if artifact.get("metadata") != expected_metadata:
        raise ValueError("Self-difference artifact metadata mismatch")
    if set(artifact) != {"format_version", "metadata", *SELF_VECTOR_TYPES}:
        raise ValueError("Self-difference artifact fields mismatch")
    raw = context["attempt"]["self_diff_manifest"].get("self_diff_artifact", {}).get(
        "raw_tensors_sha256"
    )
    if not isinstance(raw, dict) or set(raw) != set(SELF_VECTOR_TYPES):
        raise ValueError("Self-difference manifest raw tensor hashes mismatch")
    for name in SELF_VECTOR_TYPES:
        validate_activation_tensor(
            artifact.get(name), f"Self-difference {name}", raw.get(name), torch_module
        )
    return artifact


def load_and_validate_oracle(
    path: Path, context: dict[str, Any], torch_module: Any
) -> dict[str, Any]:
    artifact = load_torch_artifact(path, "oracle artifact", torch_module)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError("Malformed oracle artifact")
    manifest = context["oracle"]["oracle_manifest"]
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Oracle artifact is missing metadata")
    for key in (
        "relative_layer",
        "layer_index",
        "num_layers",
        "sample_count",
        "sequence_length",
        "hidden_size",
        "batch_size",
        "model_dtype",
        "accumulator_dtype",
        "stored_dtype",
    ):
        if metadata.get(key) != manifest.get(key):
            raise ValueError(f"Oracle artifact metadata mismatch: {key}")
    raw = manifest["raw_tensors_sha256"]
    for name in ORACLE_VECTOR_TYPES:
        validate_activation_tensor(
            artifact.get(name), f"Oracle {name}", raw[name], torch_module
        )
    return artifact


def verify_loaded_merged_mean_consistency(
    self_diff: dict[str, Any],
    oracle: dict[str, Any],
    expected_hash: str,
    torch_module: Any,
) -> str:
    """Recompute the critical gate from loaded tensors, independently of manifests."""
    expected = require_sha256(expected_hash, "merged activation mean gate")
    self_hash = sha256_raw_float32_tensor(self_diff.get("merged_mean"), torch_module)
    oracle_hash = sha256_raw_float32_tensor(oracle.get("ft_mean"), torch_module)
    if self_hash != oracle_hash or self_hash != expected:
        raise ValueError("Loaded merged_mean disagrees with oracle ft_mean")
    return self_hash


def geometry_metrics(
    self_difference: Any, oracle_difference: Any, torch_module: Any
) -> dict[str, float | None]:
    if not isinstance(self_difference, torch_module.Tensor) or not isinstance(
        oracle_difference, torch_module.Tensor
    ):
        raise ValueError("Geometry inputs must be tensors")
    if self_difference.ndim != 1 or oracle_difference.ndim != 1:
        raise ValueError("Geometry inputs must be rank-one vectors")
    if tuple(self_difference.shape) != tuple(oracle_difference.shape):
        raise ValueError("Geometry vector shapes differ")
    if self_difference.numel() == 0:
        raise ValueError("Geometry vectors must not be empty")
    if not bool(torch_module.isfinite(self_difference).all().item()) or not bool(
        torch_module.isfinite(oracle_difference).all().item()
    ):
        raise ValueError("Geometry vectors contain non-finite values")
    self64 = self_difference.detach().to(device="cpu", dtype=torch_module.float64)
    oracle64 = oracle_difference.detach().to(
        device="cpu", dtype=torch_module.float64
    )
    self_norm_tensor = torch_module.linalg.vector_norm(self64)
    oracle_norm_tensor = torch_module.linalg.vector_norm(oracle64)
    self_norm = float(self_norm_tensor.item())
    oracle_norm = float(oracle_norm_tensor.item())
    cosine = None
    if self_norm != 0.0 and oracle_norm != 0.0:
        cosine = float(
            (
                torch_module.dot(self64, oracle64)
                / (self_norm_tensor * oracle_norm_tensor)
            ).item()
        )
    return {
        "cosine_similarity": cosine,
        "self_diff_norm": self_norm,
        "oracle_diff_norm": oracle_norm,
        "self_to_oracle_norm_ratio": (
            None if oracle_norm == 0.0 else self_norm / oracle_norm
        ),
    }


def validate_tokenizer(tokenizer: Any, vocab_size: int) -> None:
    try:
        tokenizer_size = len(tokenizer)
    except (AttributeError, TypeError) as exc:
        raise ValueError("Loaded base tokenizer has no vocabulary size") from exc
    if tokenizer_size <= 0 or tokenizer_size > vocab_size:
        raise ValueError("Base tokenizer vocabulary is incompatible with the LM head")


def load_local_logit_lens_components(
    merged_dir: Path, base_dir: Path, hidden_size: int, torch_module: Any
) -> tuple[Any, Any, Any, int]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
        config = AutoConfig.from_pretrained(merged_dir, local_files_only=True)
        vocab_size = getattr(config, "vocab_size", None)
        if (getattr(config, "model_type", None) != "qwen3" or
            getattr(config, "num_hidden_layers", None) != 28 or
            getattr(config, "hidden_size", None) != hidden_size or
            type(vocab_size) is not int or vocab_size <= 0 or
            getattr(config, "tie_word_embeddings", None) is not False):
            raise ValueError("Canonical merged Qwen3 configuration mismatch")
        epsilon = getattr(config, "rms_norm_eps", None)
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Canonical merged Qwen3 final norm epsilon mismatch")
        weights_path = merged_dir / "model.safetensors"
        with safe_open(weights_path, framework="pt", device="cpu") as weights:
            if "model.norm.weight" not in weights.keys() or "lm_head.weight" not in weights.keys():
                raise ValueError("Canonical merged checkpoint lacks Logit Lens weights")
            norm_weight = weights.get_tensor("model.norm.weight")
            head_weight = weights.get_tensor("lm_head.weight")
        if (norm_weight.dtype != torch_module.float32 or tuple(norm_weight.shape) != (hidden_size,) or
            head_weight.dtype != torch_module.float32 or tuple(head_weight.shape) != (vocab_size, hidden_size) or
            not bool(torch_module.isfinite(norm_weight).all().item()) or
            not bool(torch_module.isfinite(head_weight).all().item())):
            raise ValueError("Canonical merged Logit Lens weight shape, dtype, or finiteness mismatch")
        final_norm = Qwen3RMSNorm(hidden_size, eps=epsilon)
        lm_head = torch_module.nn.Linear(hidden_size, vocab_size, bias=False)
        with torch_module.no_grad():
            final_norm.weight.copy_(norm_weight)
            lm_head.weight.copy_(head_weight)
        final_norm.eval()
        lm_head.eval()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        raise ValueError(f"Could not load local Logit Lens model/tokenizer: {exc}") from exc
    validate_tokenizer(tokenizer, vocab_size)
    return tokenizer, final_norm, lm_head, vocab_size


def logit_lens_probabilities(
    latent: Any, final_norm: Any, lm_head: Any, torch_module: Any
) -> tuple[Any, Any]:
    normed = final_norm(latent)
    positive = torch_module.softmax(lm_head(normed), dim=-1)
    negative = torch_module.softmax(lm_head(-normed), dim=-1)
    return positive, negative


def top_token_records(
    probabilities: Any, tokenizer: Any, top_k: int, torch_module: Any
) -> list[dict[str, Any]]:
    if not isinstance(probabilities, torch_module.Tensor) or probabilities.ndim != 1:
        raise ValueError("Logit Lens probabilities must be a rank-one tensor")
    if type(top_k) is not int or not 0 < top_k <= probabilities.numel():
        raise ValueError("Invalid Logit Lens top-k")
    if not bool(torch_module.isfinite(probabilities).all().item()):
        raise ValueError("Logit Lens probabilities contain non-finite values")
    indices = torch_module.argsort(probabilities, descending=True, stable=True)[:top_k]
    records = []
    for value in indices.detach().to(device="cpu").tolist():
        token_id = int(value)
        token = tokenizer.convert_ids_to_tokens(token_id)
        decoded = tokenizer.decode(
            [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if not isinstance(token, str) or not isinstance(decoded, str):
            raise ValueError(f"Tokenizer returned invalid text for token {token_id}")
        records.append(
            {
                "token_id": token_id,
                "token": token,
                "decoded": decoded,
                "probability": float(probabilities[token_id].item()),
            }
        )
    return records


def evaluate_vectors(
    self_diff: dict[str, Any],
    oracle: dict[str, Any],
    final_norm: Any,
    lm_head: Any,
    tokenizer: Any,
    vocab_size: int,
    spec: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    difference = self_diff["difference"]
    hidden_size = self_diff["metadata"]["hidden_size"]
    position_results = []
    with torch_module.inference_mode():
        for position in spec["positions"]:
            self_vector = difference[position]
            oracle_vector = oracle["difference"][position]
            if (
                self_vector.dtype != torch_module.float32
                or self_vector.device.type != "cpu"
                or tuple(self_vector.shape) != (hidden_size,)
            ):
                raise ValueError("Invalid self-difference vector")
            positive, negative = logit_lens_probabilities(
                self_vector, final_norm, lm_head, torch_module
            )
            if tuple(positive.shape) != (vocab_size,) or tuple(negative.shape) != (vocab_size,):
                raise ValueError("LM head returned unexpected dimensions")
            position_results.append(
                {
                    "position": position,
                    "geometry": geometry_metrics(self_vector, oracle_vector, torch_module),
                    "logit_lens": {
                        "positive": top_token_records(
                            positive, tokenizer, spec["logit_lens"]["top_k"], torch_module
                        ),
                        "negative": top_token_records(
                            negative, tokenizer, spec["logit_lens"]["top_k"], torch_module
                        ),
                    },
                }
            )
    return {"attempt_id": spec["attempt_id"], "positions": position_results}


def ordered_result(result: Any, spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict) or result.get("attempt_id") != spec["attempt_id"]:
        raise ValueError("Evaluation result identity mismatch")
    if set(result) != {"attempt_id", "positions"}:
        raise ValueError("Evaluation result fields mismatch")
    positions = result.get("positions")
    if not isinstance(positions, list) or len(positions) != len(spec["positions"]):
        raise ValueError("Evaluation position coverage mismatch")
    by_position = {}
    for record in positions:
        if not isinstance(record, dict) or set(record) != {"position", "geometry", "logit_lens"}:
            raise ValueError("Malformed evaluation position result")
        position = record.get("position")
        if position not in spec["positions"] or position in by_position:
            raise ValueError(f"Unexpected or duplicate evaluation position: {position}")
        geometry = record.get("geometry")
        lens = record.get("logit_lens")
        if not isinstance(geometry, dict) or list(geometry) != spec["geometry"]["metrics"]:
            raise ValueError("Malformed evaluation geometry result")
        if not isinstance(lens, dict) or list(lens) != ["positive", "negative"]:
            raise ValueError("Malformed evaluation Logit Lens result")
        if any(
            not isinstance(lens[direction], list)
            or len(lens[direction]) != spec["logit_lens"]["top_k"]
            for direction in ("positive", "negative")
        ):
            raise ValueError("Evaluation Logit Lens top-k coverage mismatch")
        by_position[position] = record
    if set(by_position) != set(spec["positions"]):
        raise ValueError("Evaluation position coverage mismatch")
    return {
        "attempt_id": spec["attempt_id"],
        "positions": [by_position[position] for position in spec["positions"]],
    }


def build_evaluation_output(
    result: Any,
    spec: dict[str, Any],
    context: dict[str, Any],
    *,
    evaluation_spec_sha256: str,
    evaluator_script_sha256: str,
) -> dict[str, Any]:
    attempt = context["attempt"]
    oracle = context["oracle"]
    values = (
        (evaluation_spec_sha256, "evaluation specification"),
        (evaluator_script_sha256, "evaluator script"),
        (attempt["attempt_spec_sha256"], "attempt specification"),
        (attempt["calibration_sha256"], "calibration"),
        (attempt["construction_manifest_sha256"], "construction manifest"),
        (attempt["constructor_script_sha256"], "constructor script"),
        (attempt["self_diff_spec_sha256"], "self-difference specification"),
        (attempt["self_diff_manifest_sha256"], "self-difference manifest"),
        (attempt["self_diff_script_sha256"], "self-difference script"),
        (context["self_diff_artifact_sha256"], "self-difference artifact"),
        (oracle["oracle_manifest_sha256"], "oracle manifest"),
        (context["oracle_artifact_sha256"], "oracle artifact"),
        (oracle["oracle_logit_lens_sha256"], "oracle Logit Lens"),
        (oracle["merged_model_hashes_sha256"], "merged-model manifest"),
        (oracle["downloaded_model_hashes_sha256"], "downloaded-model manifest"),
        (oracle["oracle_probe_manifest_sha256"], "oracle-probe manifest"),
        (context["merged_mean_raw_sha256_gate"], "merged mean raw hash gate"),
    )
    for value, description in values:
        require_sha256(value, description)
    base_source = oracle["base_tokenizer_source"]
    return {
        "format_version": 1,
        "attempt_id": spec["attempt_id"],
        "provenance": {
            "evaluation_spec_sha256": evaluation_spec_sha256,
            "evaluator_script_sha256": evaluator_script_sha256,
            "attempt": {
                "calibration_sha256": attempt["calibration_sha256"],
                "spec_sha256": attempt["attempt_spec_sha256"],
                "construction_manifest_sha256": attempt["construction_manifest_sha256"],
                "constructor_script_sha256": attempt["constructor_script_sha256"],
                "self_diff_spec_sha256": attempt["self_diff_spec_sha256"],
                "self_diff_manifest_sha256": attempt["self_diff_manifest_sha256"],
                "self_diff_script_sha256": attempt["self_diff_script_sha256"],
                "self_diff_artifact_sha256": context["self_diff_artifact_sha256"],
            },
            "oracle": {
                "manifest_sha256": oracle["oracle_manifest_sha256"],
                "artifact_sha256": context["oracle_artifact_sha256"],
            },
            "oracle_logit_lens": {"sha256": oracle["oracle_logit_lens_sha256"]},
            "committed_model_provenance": {
                "downloaded_model_hashes_sha256": oracle["downloaded_model_hashes_sha256"],
                "merged_model_hashes_sha256": oracle["merged_model_hashes_sha256"],
                "oracle_probe_manifest_sha256": oracle["oracle_probe_manifest_sha256"],
            },
            "merged_checkpoint": {
                "base": oracle["base"],
                "fine_tuned": oracle["fine_tuned"],
                **attempt["merged_checkpoint"],
            },
            "base_tokenizer": {
                "source": "base",
                "repo_id": base_source["repo_id"],
                "revision": base_source["revision"],
                "downloaded_model_hashes_sha256": oracle["downloaded_model_hashes_sha256"],
            },
            "merged_mean_raw_sha256_gate": context["merged_mean_raw_sha256_gate"],
        },
        "method": {
            "positions": list(spec["positions"]),
            "primary_metric": spec["primary_metric"],
            "geometry": dict(spec["geometry"]),
            "logit_lens": dict(spec["logit_lens"]),
            "evaluation_policy": dict(spec["evaluation_policy"]),
        },
        "result": ordered_result(result, spec),
    }


def write_output(output: dict[str, Any], path: Path) -> None:
    try:
        serialized = json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not serialize evaluation output: {exc}") from exc
    try:
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(serialized)
    except FileExistsError as exc:
        raise ValueError(f"Evaluation output already exists: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not write evaluation output {path}: {exc}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-model-dir",
        type=Path,
        default=Path(os.environ.get("MERGED_MODEL_DIR", DEFAULT_MERGED_MODEL_DIR)),
    )
    parser.add_argument(
        "--base-tokenizer-dir",
        type=Path,
        default=Path(os.environ.get("BASE_TOKENIZER_DIR", DEFAULT_BASE_TOKENIZER_DIR)),
    )
    parser.add_argument(
        "--evaluation-spec-path",
        type=Path,
        default=Path(os.environ.get("EVALUATION_SPEC_PATH", DEFAULT_EVALUATION_SPEC_PATH)),
    )
    parser.add_argument(
        "--self-diff-artifact-path",
        type=Path,
        default=Path(os.environ.get("SELF_DIFF_ARTIFACT_PATH", DEFAULT_SELF_DIFF_ARTIFACT_PATH)),
    )
    parser.add_argument(
        "--oracle-artifact-path",
        type=Path,
        default=Path(os.environ.get("ORACLE_ARTIFACT_PATH", DEFAULT_ORACLE_ARTIFACT_PATH)),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(os.environ.get("EVALUATION_OUTPUT_PATH", DEFAULT_OUTPUT_PATH)),
    )
    return parser.parse_args(argv)


def ensure_unchanged(path: Path, expected_hash: str, description: str) -> None:
    if sha256_file(path) != expected_hash:
        raise ValueError(f"{description} changed during evaluation")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for attribute in (
        "merged_model_dir",
        "base_tokenizer_dir",
        "evaluation_spec_path",
        "self_diff_artifact_path",
        "oracle_artifact_path",
        "output_path",
    ):
        setattr(args, attribute, getattr(args, attribute).expanduser())
    evaluator_script_path = Path(__file__).resolve()
    try:
        require_output_absent(args.output_path)
        evaluation_spec_hash = sha256_file(args.evaluation_spec_path)
        evaluator_script_hash = sha256_file(evaluator_script_path)
        spec = load_evaluation_spec(args.evaluation_spec_path)
        ensure_unchanged(args.evaluation_spec_path, evaluation_spec_hash, "Evaluation specification")
        context = verify_inputs_before_loading(
            args.merged_model_dir,
            args.base_tokenizer_dir,
            args.self_diff_artifact_path,
            args.oracle_artifact_path,
            spec,
        )

        # Deserialization and model loading happen only after serialized provenance passes.
        import torch

        self_diff = load_and_validate_self_diff(
            args.self_diff_artifact_path, context, spec, torch
        )
        oracle = load_and_validate_oracle(args.oracle_artifact_path, context, torch)
        verify_loaded_merged_mean_consistency(
            self_diff,
            oracle,
            context["merged_mean_raw_sha256_gate"],
            torch,
        )
        tokenizer, final_norm, lm_head, vocab_size = load_local_logit_lens_components(
            args.merged_model_dir, args.base_tokenizer_dir, context["hidden_size"], torch
        )
        result = evaluate_vectors(
            self_diff,
            oracle,
            final_norm,
            lm_head,
            tokenizer,
            vocab_size,
            spec,
            torch,
        )
        for path, expected, description in (
            (args.evaluation_spec_path, evaluation_spec_hash, "Evaluation specification"),
            (evaluator_script_path, evaluator_script_hash, "Evaluator script"),
            (DEFAULT_CALIBRATION_PATH, context["attempt"]["calibration_sha256"], "Calibration"),
            (DEFAULT_ATTEMPT_SPEC_PATH, context["attempt"]["attempt_spec_sha256"], "Attempt specification"),
            (DEFAULT_CONSTRUCTION_MANIFEST_PATH, context["attempt"]["construction_manifest_sha256"], "Construction manifest"),
            (DEFAULT_SELF_DIFF_SPEC_PATH, context["attempt"]["self_diff_spec_sha256"], "Self-difference specification"),
            (
                DEFAULT_SELF_DIFF_MANIFEST_PATH,
                context["attempt"]["self_diff_manifest_sha256"],
                "Self-difference manifest",
            ),
            (DEFAULT_CONSTRUCTOR_SCRIPT_PATH, context["attempt"]["constructor_script_sha256"], "Constructor script"),
            (DEFAULT_SELF_DIFF_SCRIPT_PATH, context["attempt"]["self_diff_script_sha256"], "Self-difference script"),
            (
                args.self_diff_artifact_path,
                context["self_diff_artifact_sha256"],
                "Self-difference artifact",
            ),
            (
                DEFAULT_ORACLE_MANIFEST_PATH,
                context["oracle"]["oracle_manifest_sha256"],
                "Oracle manifest",
            ),
            (args.oracle_artifact_path, context["oracle_artifact_sha256"], "Oracle artifact"),
            (DEFAULT_DOWNLOADED_HASHES_PATH, context["oracle"]["downloaded_model_hashes_sha256"], "Downloaded model provenance"),
            (DEFAULT_MERGED_HASHES_PATH, context["oracle"]["merged_model_hashes_sha256"], "Merged model provenance"),
            (DEFAULT_ORACLE_PROBE_MANIFEST_PATH, context["oracle"]["oracle_probe_manifest_sha256"], "Oracle probe manifest"),
            (
                DEFAULT_ORACLE_LOGIT_LENS_PATH,
                context["oracle"]["oracle_logit_lens_sha256"],
                "Oracle Logit Lens",
            ),
        ):
            ensure_unchanged(path, expected, description)
        output = build_evaluation_output(
            result,
            spec,
            context,
            evaluation_spec_sha256=evaluation_spec_hash,
            evaluator_script_sha256=evaluator_script_hash,
        )
        require_output_absent(args.output_path)
        write_output(output, args.output_path)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
