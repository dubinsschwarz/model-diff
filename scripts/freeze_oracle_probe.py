#!/usr/bin/env python3

import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DATASET_LOCK_PATH = PROJECT / "datasets.lock.json"
DOWNLOADED_HASHES_PATH = PROJECT / "downloaded-model-hashes.json"
SCRIPT_PATH = PROJECT / "scripts" / "freeze_oracle_probe.py"
MANIFEST_PATH = PROJECT / "oracle-probe-manifest.json"

DEFAULT_BASE_MODEL_DIR = Path("/root/model-diff-scratch/models/base")
ARTIFACT_PATH = Path(
    "/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt"
)

PINNED_REPO_ID = "science-of-finetuning/fineweb-1m-sample"
PINNED_REVISION = "60b53a86b84eb6559e4407b113356f56a152318f"
DATASET_SPLIT = "train"
SEED = 42
TEXT_COLUMN = "text"
N_TOKENS = 128
MAX_SAMPLES = 10_000
CHARACTER_LIMIT = N_TOKENS * 10


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_raw_tensor(tensor: Any, torch_module: Any) -> str:
    """Hash C-order tensor values as contiguous little-endian signed int64."""
    if tensor.dtype != torch_module.long:
        raise ValueError(f"Cannot hash non-torch.long probe tensor: {tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ValueError(f"Cannot hash non-CPU probe tensor: {tensor.device}")

    contiguous = tensor.detach().contiguous()
    canonical_array = contiguous.numpy().astype("<i8", copy=False)
    return hashlib.sha256(canonical_array.tobytes(order="C")).hexdigest()


def require_directory(path: Path, description: str) -> None:
    try:
        is_directory = path.is_dir()
    except OSError as exc:
        raise ValueError(f"Could not access {description} {path}: {exc}") from exc
    if not is_directory:
        raise ValueError(f"Missing {description}: {path}")


def require_file(path: Path, description: str) -> None:
    try:
        is_file = path.is_file()
    except OSError as exc:
        raise ValueError(f"Could not access {description} {path}: {exc}") from exc
    if not is_file:
        raise ValueError(f"Missing {description}: {path}")


def load_json_object(path: Path, description: str) -> dict[str, Any]:
    require_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {description}: {path}")
    return value


def load_dataset_lock(path: Path = DATASET_LOCK_PATH) -> dict[str, str]:
    lock = load_json_object(path, "dataset lock file")
    expected = {
        "repo_id": PINNED_REPO_ID,
        "revision": PINNED_REVISION,
    }
    if lock != expected:
        raise ValueError(f"Dataset lock does not match the registered pin: {path}")
    return lock


def collect_artifact_files(root: Path) -> dict[str, Path]:
    files = {}
    try:
        for path in root.rglob("*"):
            relative_path = path.relative_to(root)
            if ".cache" in relative_path.parts or not path.is_file():
                continue
            files[relative_path.as_posix()] = path
    except OSError as exc:
        raise ValueError(f"Could not enumerate base model artifact: {exc}") from exc
    return files


def load_base_manifest_entry(path: Path) -> dict[str, Any]:
    manifest = load_json_object(path, "downloaded-model manifest")
    if manifest.get("hash_algorithm") != "sha256":
        raise ValueError(f"Downloaded-model manifest does not specify SHA-256: {path}")
    try:
        base_entry = manifest["models"]["base"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Downloaded-model manifest is missing base") from exc
    if not isinstance(base_entry, dict):
        raise ValueError("Downloaded-model manifest has invalid base entry")
    return base_entry


def verify_base_artifact(base_dir: Path, manifest_entry: dict[str, Any]) -> None:
    manifest_files = manifest_entry.get("files")
    if not isinstance(manifest_files, list):
        raise ValueError("Malformed base model manifest: files")

    expected = {}
    for record in manifest_files:
        if not isinstance(record, dict):
            raise ValueError("Malformed base model manifest file record")
        relative_path = record.get("path")
        size_bytes = record.get("size_bytes")
        expected_sha256 = record.get("sha256")
        if not isinstance(relative_path, str):
            raise ValueError("Malformed base model manifest path")

        parsed_path = PurePosixPath(relative_path)
        if (
            parsed_path.is_absolute()
            or relative_path != parsed_path.as_posix()
            or any(part in ("", ".", "..") for part in parsed_path.parts)
            or ".cache" in parsed_path.parts
        ):
            raise ValueError(f"Malformed base model manifest path: {relative_path}")
        if relative_path in expected:
            raise ValueError(
                f"Malformed base model manifest: duplicate path {relative_path}"
            )
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(
                f"Malformed base model manifest size for {relative_path}"
            )
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(
                f"Malformed base model manifest SHA-256 for {relative_path}"
            )
        expected[relative_path] = {
            "size_bytes": size_bytes,
            "sha256": expected_sha256,
        }

    file_count = manifest_entry.get("file_count")
    if type(file_count) is not int or file_count != len(expected):
        raise ValueError("Malformed base model manifest: file_count")

    actual = collect_artifact_files(base_dir)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(f"Base model artifact is missing file: {missing[0]}")
    if extra:
        raise ValueError(f"Base model artifact has extra file: {extra[0]}")

    for relative_path in sorted(expected):
        path = actual[relative_path]
        expected_record = expected[relative_path]
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise ValueError(
                f"Could not inspect base model file {path}: {exc}"
            ) from exc
        if actual_size != expected_record["size_bytes"]:
            raise ValueError(f"Base model artifact size mismatch: {relative_path}")
        try:
            actual_sha256 = sha256_file(path)
        except OSError as exc:
            raise ValueError(f"Could not hash base model file {path}: {exc}") from exc
        if actual_sha256 != expected_record["sha256"]:
            raise ValueError(f"Base model artifact SHA-256 mismatch: {relative_path}")


def path_exists(path: Path, description: str) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        raise ValueError(f"Could not inspect {description} {path}: {exc}") from exc


def validate_inputs(
    base_dir: Path,
    artifact_path: Path = ARTIFACT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    dataset_lock_path: Path = DATASET_LOCK_PATH,
    downloaded_hashes_path: Path = DOWNLOADED_HASHES_PATH,
) -> dict[str, str]:
    require_directory(base_dir, "base model directory")
    load_json_object(base_dir / "config.json", "base model config")
    load_json_object(base_dir / "tokenizer_config.json", "base tokenizer config")

    base_entry = load_base_manifest_entry(downloaded_hashes_path)
    verify_base_artifact(base_dir, base_entry)
    dataset_lock = load_dataset_lock(dataset_lock_path)

    if path_exists(artifact_path, "probe artifact"):
        raise ValueError(f"Probe artifact already exists: {artifact_path}")
    if path_exists(manifest_path, "probe manifest"):
        raise ValueError(f"Probe manifest already exists: {manifest_path}")
    if path_exists(artifact_path.parent, "probe artifact directory"):
        require_directory(artifact_path.parent, "probe artifact directory")

    return dataset_lock


def select_probe_tokens(
    dataset: Any,
    tokenizer: Any,
    torch_module: Any,
    *,
    n_tokens: int = N_TOKENS,
    max_samples: int = MAX_SAMPLES,
    character_limit: int = CHARACTER_LIMIT,
) -> Any:
    shuffled = dataset.shuffle(seed=SEED)
    accepted = []

    for sample_index, sample in enumerate(shuffled):
        if not isinstance(sample, dict) or TEXT_COLUMN not in sample:
            raise ValueError(f"Dataset sample {sample_index} is missing {TEXT_COLUMN}")
        text = sample[TEXT_COLUMN]
        if not isinstance(text, str):
            raise ValueError(f"Dataset sample {sample_index} has non-string text")
        if not text.strip():
            continue

        truncated_text = text[:character_limit]
        token_ids = tokenizer.encode(truncated_text, add_special_tokens=True)
        if not isinstance(token_ids, list) or any(
            type(token_id) is not int for token_id in token_ids
        ):
            raise ValueError(
                f"Tokenizer returned invalid IDs for sample {sample_index}"
            )
        if len(token_ids) < n_tokens:
            continue

        accepted.append(token_ids[:n_tokens])
        if len(accepted) == max_samples:
            break

    if len(accepted) != max_samples:
        raise ValueError(
            f"Only {len(accepted)} samples produced at least {n_tokens} tokens; "
            f"expected {max_samples}"
        )

    tensor = torch_module.tensor(accepted, dtype=torch_module.long)
    expected_shape = (max_samples, n_tokens)
    if tuple(tensor.shape) != expected_shape or tensor.dtype != torch_module.long:
        actual_shape = tuple(tensor.shape)
        raise ValueError(
            f"Unexpected probe tensor shape/dtype: {actual_shape}, {tensor.dtype}"
        )
    return tensor


def freeze_probe(dataset_lock: dict[str, str], base_dir: Path) -> tuple[Any, Any]:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    dataset = load_dataset(
        dataset_lock["repo_id"],
        revision=dataset_lock["revision"],
        split=DATASET_SPLIT,
        streaming=False,
    )
    tensor = select_probe_tokens(dataset, tokenizer, torch)
    return tensor, torch


def build_manifest(
    dataset_lock: dict[str, str],
    tensor: Any,
    artifact_sha256: str,
    raw_tensor_hash: str,
    dataset_lock_path: Path = DATASET_LOCK_PATH,
    downloaded_hashes_path: Path = DOWNLOADED_HASHES_PATH,
    script_path: Path = SCRIPT_PATH,
) -> dict[str, Any]:
    return {
        "dataset": {
            "repo_id": dataset_lock["repo_id"],
            "revision": dataset_lock["revision"],
            "split": DATASET_SPLIT,
        },
        "seed": SEED,
        "text_column": TEXT_COLUMN,
        "n": N_TOKENS,
        "max_samples": MAX_SAMPLES,
        "character_truncation_length": CHARACTER_LIMIT,
        "tokenizer_source": "base",
        "tensor": {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        },
        "fineweb_tokens_sha256": artifact_sha256,
        "raw_tensor_sha256": raw_tensor_hash,
        "datasets_lock_sha256": sha256_file(dataset_lock_path),
        "downloaded_model_hashes_sha256": sha256_file(downloaded_hashes_path),
        "freeze_oracle_probe_script_sha256": sha256_file(script_path),
    }


def save_outputs(
    tensor: Any,
    torch_module: Any,
    dataset_lock: dict[str, str],
    artifact_path: Path = ARTIFACT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    dataset_lock_path: Path = DATASET_LOCK_PATH,
    downloaded_hashes_path: Path = DOWNLOADED_HASHES_PATH,
    script_path: Path = SCRIPT_PATH,
) -> None:
    if path_exists(artifact_path, "probe artifact"):
        raise ValueError(f"Probe artifact already exists: {artifact_path}")
    if path_exists(manifest_path, "probe manifest"):
        raise ValueError(f"Probe manifest already exists: {manifest_path}")

    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Could not create probe artifact directory {artifact_path.parent}: {exc}"
        ) from exc

    try:
        with artifact_path.open("xb") as artifact_file:
            torch_module.save(tensor, artifact_file)
    except FileExistsError as exc:
        raise ValueError(f"Probe artifact already exists: {artifact_path}") from exc
    except OSError as exc:
        raise ValueError(
            f"Could not write probe artifact {artifact_path}: {exc}"
        ) from exc

    artifact_sha256 = sha256_file(artifact_path)
    raw_tensor_hash = sha256_raw_tensor(tensor, torch_module)
    manifest = build_manifest(
        dataset_lock,
        tensor,
        artifact_sha256,
        raw_tensor_hash,
        dataset_lock_path,
        downloaded_hashes_path,
        script_path,
    )
    try:
        with manifest_path.open("x", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(manifest, indent=2) + "\n")
    except FileExistsError as exc:
        raise ValueError(f"Probe manifest already exists: {manifest_path}") from exc
    except OSError as exc:
        raise ValueError(
            f"Could not write probe manifest {manifest_path}: {exc}"
        ) from exc

    print(f"Wrote {artifact_path}")
    print(f"Wrote {manifest_path}")
    print(f"Probe tensor: {tuple(tensor.shape)}, {tensor.dtype}")


def main() -> int:
    base_dir = Path(
        os.environ.get("BASE_MODEL_DIR", DEFAULT_BASE_MODEL_DIR)
    ).expanduser()

    try:
        dataset_lock = validate_inputs(base_dir)
        tensor, torch_module = freeze_probe(dataset_lock, base_dir)
        save_outputs(tensor, torch_module, dataset_lock)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
