#!/usr/bin/env python3

"""Evaluate Attempt 001 against the frozen oracle after the blind stage."""

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path, PurePosixPath
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT_DIRECTORY = (
    PROJECT / "experiments" / "attempts" / "001_spectral_tail_l13_p25"
)
DEFAULT_EVALUATION_SPEC_PATH = ATTEMPT_DIRECTORY / "evaluation_spec.json"
DEFAULT_ATTEMPT_SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
DEFAULT_CONSTRUCTION_MANIFEST_PATH = ATTEMPT_DIRECTORY / "construction-manifest.json"
DEFAULT_SELF_DIFF_SPEC_PATH = ATTEMPT_DIRECTORY / "self_diff_spec.json"
DEFAULT_SELF_DIFF_MANIFEST_PATH = ATTEMPT_DIRECTORY / "self-diff-manifest.json"
DEFAULT_OUTPUT_PATH = ATTEMPT_DIRECTORY / "evaluation.json"

DEFAULT_DOWNLOADED_HASHES_PATH = PROJECT / "downloaded-model-hashes.json"
DEFAULT_MERGED_HASHES_PATH = PROJECT / "merged-model-hashes.json"
DEFAULT_ORACLE_PROBE_MANIFEST_PATH = PROJECT / "oracle-probe-manifest.json"
DEFAULT_ORACLE_MANIFEST_PATH = PROJECT / "oracle-adl-manifest.json"
DEFAULT_ORACLE_LOGIT_LENS_PATH = PROJECT / "oracle-logit-lens.json"
DEFAULT_CONSTRUCTOR_SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "construct_spectral_tail.py"
DEFAULT_SELF_DIFF_SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "compute_self_diff.py"
DEFAULT_MERGE_SCRIPT_PATH = PROJECT / "scripts" / "merge_lora.py"
DEFAULT_FREEZE_PROBE_SCRIPT_PATH = PROJECT / "scripts" / "freeze_oracle_probe.py"
DEFAULT_ORACLE_SCRIPT_PATH = PROJECT / "scripts" / "compute_oracle_adl.py"

DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_BASE_TOKENIZER_DIR = Path("/root/model-diff-scratch/models/base")
DEFAULT_SELF_DIFF_ARTIFACT_PATH = Path(
    "/root/model-diff-scratch/artifacts/attempt001_self_diff/self_diff.pt"
)
DEFAULT_ORACLE_ARTIFACT_PATH = Path(
    "/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt"
)

FROZEN_EVALUATION_SPEC = {
    "format_version": 1,
    "attempt_id": "001_spectral_tail_l13_p25",
    "epsilons": [0.05, 0.1, 0.2],
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
        "evaluate_every_epsilon": True,
        "select_or_rank_best_epsilon": False,
    },
    "defaults": {
        "merged_model_directory": "/root/model-diff-scratch/models/merged",
        "base_tokenizer_directory": "/root/model-diff-scratch/models/base",
        "self_diff_artifact_path": (
            "/root/model-diff-scratch/artifacts/attempt001_self_diff/self_diff.pt"
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


def normalize_file_records(
    records: Any,
    description: str,
) -> list[dict[str, Any]]:
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


def verify_checkpoint(
    root: Path,
    entry: Any,
    description: str,
) -> dict[str, Any]:
    expected = normalize_checkpoint_entry(entry, description)
    actual = directory_file_records(root)
    if actual != expected["files"]:
        expected_paths = {item["path"] for item in expected["files"]}
        actual_paths = {item["path"] for item in actual}
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
    root: Path,
    entry: Any,
    description: str,
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


def indexed_epsilon_entries(
    entries: Any,
    epsilons: list[float],
    description: str,
) -> dict[float, dict[str, Any]]:
    if not isinstance(entries, list) or len(entries) != len(epsilons):
        raise ValueError(f"{description} epsilon count mismatch")
    indexed: dict[float, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Malformed {description} entry")
        epsilon = entry.get("epsilon")
        if isinstance(epsilon, bool) or epsilon not in epsilons:
            raise ValueError(f"Unexpected {description} epsilon: {epsilon}")
        if epsilon in indexed:
            raise ValueError(f"Duplicate {description} epsilon: {epsilon}")
        indexed[epsilon] = entry
    return indexed


def validate_attempt_provenance(
    merged_dir: Path,
    spec: dict[str, Any],
    *,
    attempt_spec_path: Path,
    construction_manifest_path: Path,
    self_diff_spec_path: Path,
    self_diff_manifest_path: Path,
    constructor_script_path: Path,
    self_diff_script_path: Path,
    merged_hashes_path: Path,
) -> dict[str, Any]:
    """Validate Attempt 001's complete frozen chain and canonical checkpoint."""
    attempt_spec = load_json_object(attempt_spec_path, "Attempt 001 specification")
    construction = load_json_object(
        construction_manifest_path, "Attempt 001 construction manifest"
    )
    self_diff_spec = load_json_object(
        self_diff_spec_path, "Attempt 001 self-difference specification"
    )
    self_manifest = load_json_object(
        self_diff_manifest_path, "Attempt 001 self-difference manifest"
    )
    merged_manifest = load_json_object(merged_hashes_path, "merged-model manifest")

    attempt_id = spec["attempt_id"]
    epsilons = spec["epsilons"]
    if attempt_spec.get("attempt_id") != attempt_id:
        raise ValueError("Attempt specification identity mismatch")
    if attempt_spec.get("epsilons") != epsilons:
        raise ValueError("Attempt specification epsilon mismatch")
    if attempt_spec.get("transformer_block_index") != 13:
        raise ValueError("Attempt specification block mismatch")

    if construction.get("hash_algorithm") != "sha256":
        raise ValueError("Construction manifest does not specify SHA-256")
    if construction.get("attempt_id") != attempt_id:
        raise ValueError("Construction manifest identity mismatch")
    verify_manifest_file_link(
        construction,
        "spec_sha256",
        attempt_spec_path,
        "construction attempt specification",
    )
    verify_manifest_file_link(
        construction,
        "constructor_sha256",
        constructor_script_path,
        "construction script",
    )
    construction_method = construction.get("construction")
    if not isinstance(construction_method, dict):
        raise ValueError("Construction manifest is missing method metadata")
    if construction_method.get("epsilons") != epsilons:
        raise ValueError("Construction manifest epsilon mismatch")
    if construction_method.get("transformer_block_index") != 13:
        raise ValueError("Construction manifest block mismatch")
    if construction_method.get("stored_dtype") != "float32":
        raise ValueError("Construction manifest dtype mismatch")

    expected_self_spec = {
        "attempt_id": attempt_id,
        "epsilons": epsilons,
        "transformer_block_index": 13,
        "num_hidden_layers": 28,
        "sample_count": 10_000,
        "sequence_length": 128,
        "batch_size": 32,
        "model_dtype": "float32",
        "accumulator_dtype": "float64",
        "stored_dtype": "float32",
        "difference_sign": "merged_minus_ablated",
    }
    for key, expected in expected_self_spec.items():
        if self_diff_spec.get(key) != expected:
            raise ValueError(f"Self-difference specification mismatch: {key}")

    if self_manifest.get("format_version") != 1:
        raise ValueError("Unsupported self-difference manifest format")
    if self_manifest.get("hash_algorithm") != "sha256":
        raise ValueError("Self-difference manifest does not specify SHA-256")
    if self_manifest.get("attempt_id") != attempt_id:
        raise ValueError("Self-difference manifest identity mismatch")
    manifest_links = (
        (
            "construction_manifest_sha256",
            construction_manifest_path,
            "self-difference construction manifest",
        ),
        (
            "attempt_spec_sha256",
            attempt_spec_path,
            "self-difference attempt specification",
        ),
        (
            "self_diff_spec_sha256",
            self_diff_spec_path,
            "self-difference specification",
        ),
        (
            "compute_self_diff_script_sha256",
            self_diff_script_path,
            "self-difference compute script",
        ),
    )
    for key, path, description in manifest_links:
        verify_manifest_file_link(self_manifest, key, path, description)

    method = self_manifest.get("method")
    if not isinstance(method, dict):
        raise ValueError("Self-difference manifest is missing method metadata")
    for key, expected in expected_self_spec.items():
        if key == "attempt_id":
            continue
        if method.get(key) != expected:
            raise ValueError(f"Self-difference manifest method mismatch: {key}")
    hidden_size = method.get("hidden_size")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("Self-difference manifest hidden size is invalid")

    input_checkpoints = self_manifest.get("input_checkpoints")
    if not isinstance(input_checkpoints, dict):
        raise ValueError("Self-difference manifest is missing input checkpoints")
    construction_source = normalize_checkpoint_entry(
        construction.get("source_checkpoint"), "construction source checkpoint"
    )
    self_source = normalize_checkpoint_entry(
        input_checkpoints.get("merged"), "self-difference merged checkpoint"
    )
    merged_source = normalize_checkpoint_entry(
        merged_manifest, "merged-model checkpoint"
    )
    if construction_source != self_source or self_source != merged_source:
        raise ValueError("Merged checkpoint provenance disagrees across manifests")

    construction_outputs = indexed_epsilon_entries(
        construction.get("output_checkpoints"), epsilons, "construction output"
    )
    self_inputs = indexed_epsilon_entries(
        input_checkpoints.get("ablated"), epsilons, "self-difference input"
    )
    for epsilon in epsilons:
        construction_entry = construction_outputs[epsilon]
        self_entry = self_inputs[epsilon]
        if construction_entry.get("directory_name") != self_entry.get("directory_name"):
            raise ValueError(f"Ablated checkpoint directory mismatch: {epsilon}")
        if normalize_checkpoint_entry(
            construction_entry, f"construction checkpoint epsilon {epsilon}"
        ) != normalize_checkpoint_entry(
            self_entry, f"self-difference checkpoint epsilon {epsilon}"
        ):
            raise ValueError(f"Ablated checkpoint provenance mismatch: {epsilon}")

    merged_checkpoint = verify_checkpoint(
        merged_dir, self_source, "canonical merged checkpoint"
    )
    return {
        "attempt_spec_sha256": sha256_file(attempt_spec_path),
        "construction_manifest_sha256": sha256_file(construction_manifest_path),
        "self_diff_spec_sha256": sha256_file(self_diff_spec_path),
        "self_diff_manifest_sha256": sha256_file(self_diff_manifest_path),
        "self_diff_manifest": self_manifest,
        "method": method,
        "hidden_size": hidden_size,
        "merged_checkpoint": merged_checkpoint,
        "merged_model_hashes_sha256": sha256_file(merged_hashes_path),
        "merged_manifest": merged_manifest,
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

    method = document.get("method")
    expected_method = {
        "positions": spec["positions"],
        "vector_types": list(ORACLE_VECTOR_TYPES),
        "top_k": spec["logit_lens"]["top_k"],
        "positive": spec["logit_lens"]["positive"],
        "negative": spec["logit_lens"]["negative"],
    }
    if method != expected_method:
        raise ValueError("Oracle Logit Lens method mismatch")

    positions = document.get("positions")
    if not isinstance(positions, list) or len(positions) != len(spec["positions"]):
        raise ValueError("Oracle Logit Lens position coverage mismatch")
    by_position = {}
    for record in positions:
        if not isinstance(record, dict):
            raise ValueError("Malformed oracle Logit Lens position")
        position = record.get("position")
        if position in by_position:
            raise ValueError(f"Duplicate oracle Logit Lens position: {position}")
        by_position[position] = record
    if set(by_position) != set(spec["positions"]):
        raise ValueError("Oracle Logit Lens position coverage mismatch")
    top_k = spec["logit_lens"]["top_k"]
    for position in spec["positions"]:
        record = by_position[position]
        for vector_type in ORACLE_VECTOR_TYPES:
            directions = record.get(vector_type)
            if not isinstance(directions, dict):
                raise ValueError("Malformed oracle Logit Lens vector result")
            for direction in ("positive", "negative"):
                tokens = directions.get(direction)
                if not isinstance(tokens, list) or len(tokens) != top_k:
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
    verify_manifest_file_link(
        merged, "merge_script_sha256", merge_script_path, "merge script"
    )
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
    if oracle.get("base") != base_identity:
        raise ValueError("Oracle manifest base identity mismatch")
    if oracle.get("fine_tuned") != fine_tuned_identity:
        raise ValueError("Oracle manifest fine-tuned identity mismatch")
    expected_probe = {
        "serialized_sha256": probe.get("fineweb_tokens_sha256"),
        "raw_tensor_sha256": probe.get("raw_tensor_sha256"),
    }
    if oracle.get("probe") != expected_probe:
        raise ValueError("Oracle manifest probe provenance mismatch")

    expected_oracle_method = {
        "relative_layer": 0.5,
        "layer_index": 13,
        "num_layers": 28,
        "sample_count": 10_000,
        "sequence_length": 128,
        "batch_size": 32,
        "model_dtype": "float32",
        "accumulator_dtype": "float64",
        "stored_dtype": "float32",
    }
    for key, expected in expected_oracle_method.items():
        if oracle.get(key) != expected:
            raise ValueError(f"Oracle manifest method mismatch: {key}")
    hidden_size = oracle.get("hidden_size")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("Oracle manifest hidden size is invalid")
    oracle_artifact_hash = require_sha256(
        oracle.get("oracle_adl_sha256"), "oracle artifact"
    )
    raw_hashes = oracle.get("raw_tensors_sha256")
    if not isinstance(raw_hashes, dict) or set(raw_hashes) != set(ORACLE_VECTOR_TYPES):
        raise ValueError("Malformed oracle raw tensor hashes")
    for name in ORACLE_VECTOR_TYPES:
        require_sha256(raw_hashes[name], f"oracle {name} tensor")

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
        "hidden_size": hidden_size,
    }


def verify_inputs_before_loading(
    merged_dir: Path,
    base_dir: Path,
    self_diff_artifact_path: Path,
    oracle_artifact_path: Path,
    evaluation_spec: dict[str, Any],
    *,
    attempt_spec_path: Path = DEFAULT_ATTEMPT_SPEC_PATH,
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
    """Verify every file and provenance link before deserializing artifacts."""
    require_directory(merged_dir, "canonical merged checkpoint")
    require_directory(base_dir, "base tokenizer source")
    require_file(self_diff_artifact_path, "self-difference artifact")
    require_file(oracle_artifact_path, "oracle artifact")

    attempt = validate_attempt_provenance(
        merged_dir,
        evaluation_spec,
        attempt_spec_path=attempt_spec_path,
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
    self_manifest = attempt["self_diff_manifest"]
    oracle_manifest = oracle["oracle_manifest"]
    if self_manifest.get("probe") != oracle_manifest.get("probe"):
        raise ValueError("Self-difference and oracle frozen probes differ")
    self_raw = self_manifest.get("self_diff_artifact", {}).get(
        "raw_tensors_sha256"
    )
    oracle_raw = oracle_manifest.get("raw_tensors_sha256")
    if (
        not isinstance(self_raw, dict)
        or not isinstance(oracle_raw, dict)
        or self_raw.get("merged_mean") != oracle_raw.get("ft_mean")
    ):
        raise ValueError("Merged activation mean disagrees across frozen artifacts")

    self_artifact_metadata = self_manifest.get("self_diff_artifact")
    if not isinstance(self_artifact_metadata, dict):
        raise ValueError("Self-difference manifest is missing artifact metadata")
    self_artifact_hash = verify_file_sha256(
        self_diff_artifact_path,
        self_artifact_metadata.get("serialized_sha256"),
        "self-difference artifact",
    )
    oracle_artifact_hash = verify_file_sha256(
        oracle_artifact_path,
        oracle["oracle_manifest"].get("oracle_adl_sha256"),
        "oracle artifact",
    )
    return {
        "attempt": attempt,
        "oracle": oracle,
        "hidden_size": attempt["hidden_size"],
        "self_diff_artifact_sha256": self_artifact_hash,
        "oracle_artifact_sha256": oracle_artifact_hash,
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
    expected_shape: tuple[int, int],
    description: str,
    expected_hash: Any,
    torch_module: Any,
) -> Any:
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError(f"{description} is not a tensor")
    if tensor.dtype != torch_module.float32 or tensor.device.type != "cpu":
        raise ValueError(f"{description} must be a CPU torch.float32 tensor")
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{description} shape mismatch: expected {list(expected_shape)}, "
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
    manifest = context["attempt"]["self_diff_manifest"]
    method = context["attempt"]["method"]
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Self-difference artifact is missing metadata")
    expected_metadata = {
        "attempt_id": spec["attempt_id"],
        **{key: method[key] for key in (
            "epsilons",
            "transformer_block_index",
            "num_hidden_layers",
            "sample_count",
            "sequence_length",
            "hidden_size",
            "batch_size",
            "model_dtype",
            "accumulator_dtype",
            "stored_dtype",
            "difference_sign",
        )},
    }
    if metadata != expected_metadata:
        raise ValueError("Self-difference artifact metadata mismatch")

    raw = manifest.get("self_diff_artifact", {}).get("raw_tensors_sha256")
    if not isinstance(raw, dict):
        raise ValueError("Self-difference manifest is missing raw tensor hashes")
    expected_shape = (method["sequence_length"], method["hidden_size"])
    validate_activation_tensor(
        artifact.get("merged_mean"),
        expected_shape,
        "Self-difference merged_mean",
        raw.get("merged_mean"),
        torch_module,
    )

    ablations = artifact.get("ablations")
    if not isinstance(ablations, dict):
        raise ValueError("Self-difference artifact is missing ablations")
    raw_by_epsilon = indexed_epsilon_entries(
        raw.get("ablations"), spec["epsilons"], "self-difference raw hash"
    )
    if set(ablations) != {f"{epsilon:.2f}" for epsilon in spec["epsilons"]}:
        raise ValueError("Self-difference artifact epsilon coverage mismatch")
    for epsilon in spec["epsilons"]:
        key = f"{epsilon:.2f}"
        record = ablations[key]
        if not isinstance(record, dict) or record.get("epsilon") != epsilon:
            raise ValueError(f"Malformed self-difference epsilon {epsilon}")
        expected_hashes = raw_by_epsilon[epsilon]
        validate_activation_tensor(
            record.get("ablated_mean"),
            expected_shape,
            f"Ablated mean epsilon {epsilon}",
            expected_hashes.get("ablated_mean"),
            torch_module,
        )
        validate_activation_tensor(
            record.get("difference"),
            expected_shape,
            f"Self-difference epsilon {epsilon}",
            expected_hashes.get("difference"),
            torch_module,
        )
    return artifact


def load_and_validate_oracle(
    path: Path,
    context: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    artifact = load_torch_artifact(path, "oracle artifact", torch_module)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError("Malformed oracle artifact")
    manifest = context["oracle"]["oracle_manifest"]
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Oracle artifact is missing metadata")
    metadata_keys = (
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
    )
    for key in metadata_keys:
        if metadata.get(key) != manifest.get(key):
            raise ValueError(f"Oracle artifact metadata mismatch: {key}")
    expected_shape = (manifest["sequence_length"], manifest["hidden_size"])
    raw_hashes = manifest["raw_tensors_sha256"]
    for name in ORACLE_VECTOR_TYPES:
        validate_activation_tensor(
            artifact.get(name),
            expected_shape,
            f"Oracle {name}",
            raw_hashes[name],
            torch_module,
        )
    return artifact


def geometry_metrics(
    self_difference: Any,
    oracle_difference: Any,
    torch_module: Any,
) -> dict[str, float | None]:
    """Compute all geometric metrics in float64 with explicit zero handling."""
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
    if self_norm == 0.0 or oracle_norm == 0.0:
        cosine = None
    else:
        cosine = float(
            (torch_module.dot(self64, oracle64) / (self_norm_tensor * oracle_norm_tensor)).item()
        )
    ratio = None if oracle_norm == 0.0 else self_norm / oracle_norm
    return {
        "cosine_similarity": cosine,
        "self_diff_norm": self_norm,
        "oracle_diff_norm": oracle_norm,
        "self_to_oracle_norm_ratio": ratio,
    }


def verify_float32_parameters(model: Any, torch_module: Any) -> None:
    floating = [
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    ]
    if not floating:
        raise ValueError("Merged model has no floating-point parameters")
    unexpected = sorted(
        {
            str(parameter.dtype)
            for parameter in floating
            if parameter.dtype != torch_module.float32
        }
    )
    if unexpected:
        raise ValueError(
            "Merged model has unexpected floating dtypes: " + ", ".join(unexpected)
        )
    devices = {parameter.device.type for parameter in floating}
    if devices != {"cpu"}:
        raise ValueError(
            f"Merged model must be on CPU for Logit Lens, found {sorted(devices)}"
        )


def validate_loaded_qwen3_model(
    model: Any,
    hidden_size: int,
    torch_module: Any,
) -> tuple[Any, Any, int]:
    """Return the actual Qwen3 final norm and LM head after strict checks."""
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "qwen3":
        raise ValueError("Loaded merged model config is not Qwen3")
    if getattr(config, "num_hidden_layers", None) != 28:
        raise ValueError("Loaded merged model must have 28 transformer blocks")
    if getattr(config, "hidden_size", None) != hidden_size:
        raise ValueError("Loaded merged model hidden_size mismatch")
    vocab_size = getattr(config, "vocab_size", None)
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("Loaded merged model has invalid vocab_size")
    verify_float32_parameters(model, torch_module)

    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    final_norm = getattr(backbone, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if layers is None or len(layers) != 28:
        raise ValueError("Merged Qwen3 model does not expose 28 transformer blocks")
    if final_norm is None:
        raise ValueError("Merged Qwen3 model does not expose model.norm")
    if lm_head is None:
        raise ValueError("Merged Qwen3 model does not expose lm_head")

    norm_weight = getattr(final_norm, "weight", None)
    head_weight = getattr(lm_head, "weight", None)
    if not isinstance(norm_weight, torch_module.Tensor) or tuple(
        norm_weight.shape
    ) != (hidden_size,):
        raise ValueError("Final model norm has unexpected dimensions")
    if not isinstance(head_weight, torch_module.Tensor) or tuple(
        head_weight.shape
    ) != (vocab_size, hidden_size):
        raise ValueError("LM head has unexpected dimensions")
    if norm_weight.dtype != torch_module.float32 or head_weight.dtype != torch_module.float32:
        raise ValueError("Final norm and LM head must be torch.float32")
    if norm_weight.device.type != "cpu" or head_weight.device.type != "cpu":
        raise ValueError("Final norm and LM head must be on CPU")
    if getattr(lm_head, "in_features", hidden_size) != hidden_size:
        raise ValueError("LM head input dimension mismatch")
    if getattr(lm_head, "out_features", vocab_size) != vocab_size:
        raise ValueError("LM head output dimension mismatch")
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if not callable(get_output_embeddings) or get_output_embeddings() is not lm_head:
        raise ValueError("AutoModelForCausalLM output embedding is not model.lm_head")
    return final_norm, lm_head, vocab_size


def validate_tokenizer(tokenizer: Any, vocab_size: int) -> None:
    try:
        tokenizer_size = len(tokenizer)
    except (AttributeError, TypeError) as exc:
        raise ValueError("Loaded base tokenizer has no vocabulary size") from exc
    if tokenizer_size <= 0 or tokenizer_size > vocab_size:
        raise ValueError(
            "Base tokenizer vocabulary is incompatible with the LM head: "
            f"tokenizer {tokenizer_size}, LM head {vocab_size}"
        )


def load_local_model_and_tokenizer(
    merged_dir: Path,
    base_dir: Path,
    hidden_size: int,
    torch_module: Any,
) -> tuple[Any, Any, Any, Any, int]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            dtype=torch_module.float32,
            local_files_only=True,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load local model/tokenizer: {exc}") from exc
    model.eval()
    final_norm, lm_head, vocab_size = validate_loaded_qwen3_model(
        model, hidden_size, torch_module
    )
    validate_tokenizer(tokenizer, vocab_size)
    return model, tokenizer, final_norm, lm_head, vocab_size


def logit_lens_probabilities(
    latent: Any,
    final_norm: Any,
    lm_head: Any,
    torch_module: Any,
) -> tuple[Any, Any]:
    """Apply the exact positive and negative Activation Difference Lens readout."""
    normed = final_norm(latent)
    positive = torch_module.softmax(lm_head(normed), dim=-1)
    negative = torch_module.softmax(lm_head(-normed), dim=-1)
    return positive, negative


def top_token_records(
    probabilities: Any,
    tokenizer: Any,
    top_k: int,
    torch_module: Any,
) -> list[dict[str, Any]]:
    if not isinstance(probabilities, torch_module.Tensor) or probabilities.ndim != 1:
        raise ValueError("Logit Lens probabilities must be a rank-one tensor")
    if type(top_k) is not int or not 0 < top_k <= probabilities.numel():
        raise ValueError("Invalid Logit Lens top-k")
    if not bool(torch_module.isfinite(probabilities).all().item()):
        raise ValueError("Logit Lens probabilities contain non-finite values")
    indices = torch_module.argsort(
        probabilities, dim=-1, descending=True, stable=True
    )[:top_k]
    records = []
    for value in indices.detach().to(device="cpu").tolist():
        token_id = int(value)
        token = tokenizer.convert_ids_to_tokens(token_id)
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
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
) -> list[dict[str, Any]]:
    results = []
    hidden_size = self_diff["metadata"]["hidden_size"]
    top_k = spec["logit_lens"]["top_k"]
    with torch_module.inference_mode():
        for epsilon in spec["epsilons"]:
            difference = self_diff["ablations"][f"{epsilon:.2f}"]["difference"]
            position_results = []
            for position in spec["positions"]:
                self_vector = difference[position]
                oracle_vector = oracle["difference"][position]
                if (
                    self_vector.dtype != torch_module.float32
                    or self_vector.device.type != "cpu"
                    or tuple(self_vector.shape) != (hidden_size,)
                ):
                    raise ValueError("Invalid self-difference vector")
                metrics = geometry_metrics(self_vector, oracle_vector, torch_module)
                positive, negative = logit_lens_probabilities(
                    self_vector, final_norm, lm_head, torch_module
                )
                if tuple(positive.shape) != (vocab_size,) or tuple(
                    negative.shape
                ) != (vocab_size,):
                    raise ValueError("LM head returned unexpected dimensions")
                position_results.append(
                    {
                        "position": position,
                        "geometry": metrics,
                        "logit_lens": {
                            "positive": top_token_records(
                                positive, tokenizer, top_k, torch_module
                            ),
                            "negative": top_token_records(
                                negative, tokenizer, top_k, torch_module
                            ),
                        },
                    }
                )
            results.append({"epsilon": epsilon, "positions": position_results})
    return results


def ordered_results(
    results: Any,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate exact epsilon/position coverage and return frozen ordering."""
    if not isinstance(results, list):
        raise ValueError("Evaluation results must be a list")
    by_epsilon = indexed_epsilon_entries(
        results, spec["epsilons"], "evaluation result"
    )
    ordered = []
    for epsilon in spec["epsilons"]:
        epsilon_record = by_epsilon[epsilon]
        positions = epsilon_record.get("positions")
        if not isinstance(positions, list) or len(positions) != len(spec["positions"]):
            raise ValueError(f"Evaluation position coverage mismatch: {epsilon}")
        by_position = {}
        for record in positions:
            if not isinstance(record, dict):
                raise ValueError("Malformed evaluation position result")
            position = record.get("position")
            if position not in spec["positions"]:
                raise ValueError(f"Unexpected evaluation position: {position}")
            if position in by_position:
                raise ValueError(f"Duplicate evaluation position: {position}")
            geometry = record.get("geometry")
            lens = record.get("logit_lens")
            if not isinstance(geometry, dict) or set(geometry) != set(
                spec["geometry"]["metrics"]
            ):
                raise ValueError("Malformed evaluation geometry result")
            if not isinstance(lens, dict) or set(lens) != {"positive", "negative"}:
                raise ValueError("Malformed evaluation Logit Lens result")
            top_k = spec["logit_lens"]["top_k"]
            if any(
                not isinstance(lens[direction], list)
                or len(lens[direction]) != top_k
                for direction in ("positive", "negative")
            ):
                raise ValueError("Evaluation Logit Lens top-k coverage mismatch")
            by_position[position] = record
        if set(by_position) != set(spec["positions"]):
            raise ValueError(f"Evaluation position coverage mismatch: {epsilon}")
        ordered.append(
            {
                "epsilon": epsilon,
                "positions": [by_position[position] for position in spec["positions"]],
            }
        )
    return ordered


def build_evaluation_output(
    results: Any,
    spec: dict[str, Any],
    context: dict[str, Any],
    *,
    evaluation_spec_sha256: str,
    evaluator_script_sha256: str,
) -> dict[str, Any]:
    require_sha256(evaluation_spec_sha256, "evaluation specification")
    require_sha256(evaluator_script_sha256, "evaluator script")
    attempt = context["attempt"]
    oracle = context["oracle"]
    for value, description in (
        (attempt["attempt_spec_sha256"], "attempt specification"),
        (attempt["construction_manifest_sha256"], "construction manifest"),
        (attempt["self_diff_spec_sha256"], "self-difference specification"),
        (attempt["self_diff_manifest_sha256"], "self-difference manifest"),
        (context["self_diff_artifact_sha256"], "self-difference artifact"),
        (oracle["oracle_manifest_sha256"], "oracle manifest"),
        (context["oracle_artifact_sha256"], "oracle artifact"),
        (oracle["oracle_logit_lens_sha256"], "oracle Logit Lens"),
        (oracle["merged_model_hashes_sha256"], "merged-model manifest"),
        (oracle["downloaded_model_hashes_sha256"], "downloaded-model manifest"),
    ):
        require_sha256(value, description)

    base_source = oracle["base_tokenizer_source"]
    return {
        "format_version": 1,
        "attempt_id": spec["attempt_id"],
        "provenance": {
            "evaluation_spec_sha256": evaluation_spec_sha256,
            "evaluator_script_sha256": evaluator_script_sha256,
            "attempt_spec_sha256": attempt["attempt_spec_sha256"],
            "construction_manifest_sha256": attempt[
                "construction_manifest_sha256"
            ],
            "self_difference": {
                "spec_sha256": attempt["self_diff_spec_sha256"],
                "manifest_sha256": attempt["self_diff_manifest_sha256"],
                "artifact_sha256": context["self_diff_artifact_sha256"],
            },
            "oracle": {
                "manifest_sha256": oracle["oracle_manifest_sha256"],
                "artifact_sha256": context["oracle_artifact_sha256"],
            },
            "oracle_logit_lens": {
                "sha256": oracle["oracle_logit_lens_sha256"]
            },
            "merged_checkpoint": {
                "merged_model_hashes_sha256": oracle[
                    "merged_model_hashes_sha256"
                ],
                "base": oracle["base"],
                "fine_tuned": oracle["fine_tuned"],
                **attempt["merged_checkpoint"],
            },
            "base_tokenizer": {
                "source": "base",
                "repo_id": base_source["repo_id"],
                "revision": base_source["revision"],
                "downloaded_model_hashes_sha256": oracle[
                    "downloaded_model_hashes_sha256"
                ],
            },
        },
        "method": {
            "epsilons": list(spec["epsilons"]),
            "positions": list(spec["positions"]),
            "primary_metric": spec["primary_metric"],
            "geometry": dict(spec["geometry"]),
            "logit_lens": dict(spec["logit_lens"]),
            "evaluate_every_epsilon": True,
            "select_or_rank_best_epsilon": False,
        },
        "results": ordered_results(results, spec),
    }


def write_output(output: dict[str, Any], path: Path) -> None:
    try:
        serialized = json.dumps(
            output,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n"
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
        default=Path(
            os.environ.get("BASE_TOKENIZER_DIR", DEFAULT_BASE_TOKENIZER_DIR)
        ),
    )
    parser.add_argument(
        "--evaluation-spec-path",
        type=Path,
        default=Path(
            os.environ.get("EVALUATION_SPEC_PATH", DEFAULT_EVALUATION_SPEC_PATH)
        ),
    )
    parser.add_argument(
        "--self-diff-artifact-path",
        type=Path,
        default=Path(
            os.environ.get(
                "SELF_DIFF_ARTIFACT_PATH", DEFAULT_SELF_DIFF_ARTIFACT_PATH
            )
        ),
    )
    parser.add_argument(
        "--oracle-artifact-path",
        type=Path,
        default=Path(
            os.environ.get("ORACLE_ARTIFACT_PATH", DEFAULT_ORACLE_ARTIFACT_PATH)
        ),
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
    args.merged_model_dir = args.merged_model_dir.expanduser()
    args.base_tokenizer_dir = args.base_tokenizer_dir.expanduser()
    args.evaluation_spec_path = args.evaluation_spec_path.expanduser()
    args.self_diff_artifact_path = args.self_diff_artifact_path.expanduser()
    args.oracle_artifact_path = args.oracle_artifact_path.expanduser()
    args.output_path = args.output_path.expanduser()
    evaluator_script_path = Path(__file__).resolve()
    model = None
    try:
        require_output_absent(args.output_path)
        evaluation_spec_hash = sha256_file(args.evaluation_spec_path)
        evaluator_script_hash = sha256_file(evaluator_script_path)
        spec = load_evaluation_spec(args.evaluation_spec_path)
        ensure_unchanged(
            args.evaluation_spec_path,
            evaluation_spec_hash,
            "Evaluation specification",
        )
        context = verify_inputs_before_loading(
            args.merged_model_dir,
            args.base_tokenizer_dir,
            args.self_diff_artifact_path,
            args.oracle_artifact_path,
            spec,
        )

        # Torch artifacts and Transformers are touched only after every serialized
        # input and committed provenance link has passed validation.
        import torch

        self_diff = load_and_validate_self_diff(
            args.self_diff_artifact_path, context, spec, torch
        )
        oracle = load_and_validate_oracle(args.oracle_artifact_path, context, torch)
        model, tokenizer, final_norm, lm_head, vocab_size = (
            load_local_model_and_tokenizer(
                args.merged_model_dir,
                args.base_tokenizer_dir,
                context["hidden_size"],
                torch,
            )
        )
        results = evaluate_vectors(
            self_diff,
            oracle,
            final_norm,
            lm_head,
            tokenizer,
            vocab_size,
            spec,
            torch,
        )

        ensure_unchanged(
            args.evaluation_spec_path,
            evaluation_spec_hash,
            "Evaluation specification",
        )
        ensure_unchanged(
            evaluator_script_path, evaluator_script_hash, "Evaluator script"
        )
        ensure_unchanged(
            DEFAULT_SELF_DIFF_MANIFEST_PATH,
            context["attempt"]["self_diff_manifest_sha256"],
            "Self-difference manifest",
        )
        ensure_unchanged(
            args.self_diff_artifact_path,
            context["self_diff_artifact_sha256"],
            "Self-difference artifact",
        )
        ensure_unchanged(
            DEFAULT_ORACLE_MANIFEST_PATH,
            context["oracle"]["oracle_manifest_sha256"],
            "Oracle manifest",
        )
        ensure_unchanged(
            args.oracle_artifact_path,
            context["oracle_artifact_sha256"],
            "Oracle artifact",
        )
        ensure_unchanged(
            DEFAULT_ORACLE_LOGIT_LENS_PATH,
            context["oracle"]["oracle_logit_lens_sha256"],
            "Oracle Logit Lens",
        )
        output = build_evaluation_output(
            results,
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
    finally:
        del model

    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
