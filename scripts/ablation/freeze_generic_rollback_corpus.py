#!/usr/bin/env python3
"""Freeze Attempt 005's held-out generic token corpus from the merged tokenizer."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125"
DEFAULT_SPEC_PATH = ATTEMPT_DIRECTORY / "corpus_spec.json"
DEFAULT_MANIFEST_PATH = ATTEMPT_DIRECTORY / "corpus-manifest.json"
DEFAULT_TOKENIZER_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_ARTIFACT_PATH = Path(
    "/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt"
)
TOKENIZER_FILE_NAMES = (
    "chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json"
)

FROZEN_CORPUS_SPEC = {
    "format_version": 1,
    "attempt_id": "005_generic_gradient_rollback_prefix0_13_r00125",
    "dataset": {
        "repo_id": "science-of-finetuning/fineweb-1m-sample",
        "revision": "60b53a86b84eb6559e4407b113356f56a152318f",
        "split": "train",
        "streaming": False,
    },
    "tokenizer": {
        "source": "canonical_merged_checkpoint",
        "directory": "/root/model-diff-scratch/models/merged",
    },
    "selection": {
        "shuffle_seed": 42,
        "text_column": "text",
        "skip_blank_text": True,
        "character_limit": 1280,
        "add_special_tokens": True,
        "minimum_token_count": 128,
        "skip_valid_examples": 20_000,
        "sample_count": 4_096,
        "sequence_length": 128,
    },
    "artifact": {
        "path": str(DEFAULT_ARTIFACT_PATH),
        "dtype": "torch.int64",
        "device": "cpu",
        "contiguous": True,
    },
    "manifest": {
        "filename": "corpus-manifest.json",
        "timestamps": False,
        "host_metadata": False,
        "gpu_metadata": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")


def require_output_absent(artifact_path: Path, manifest_path: Path) -> None:
    if artifact_path.exists():
        raise ValueError(f"Corpus artifact already exists: {artifact_path}")
    if manifest_path.exists():
        raise ValueError(f"Corpus manifest already exists: {manifest_path}")
    if not manifest_path.parent.is_dir():
        raise ValueError(f"Missing manifest directory: {manifest_path.parent}")


def load_spec(path: Path) -> dict[str, Any]:
    require_file(path, "corpus specification")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read corpus specification: {exc}") from exc
    if spec != FROZEN_CORPUS_SPEC:
        raise ValueError("Corpus specification does not match the frozen definition")
    return spec


def tokenizer_file_records(tokenizer_dir: Path) -> list[dict[str, Any]]:
    if not tokenizer_dir.is_dir():
        raise ValueError(f"Missing merged tokenizer directory: {tokenizer_dir}")
    records = []
    for name in TOKENIZER_FILE_NAMES:
        path = tokenizer_dir / name
        require_file(path, "merged tokenizer file")
        records.append({
            "path": name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records


def validate_tokens(tensor: Any, torch_module: Any, *, sample_count: int, sequence_length: int) -> Any:
    if (not isinstance(tensor, torch_module.Tensor) or
        tensor.dtype != torch_module.int64 or tensor.device.type != "cpu" or
        tuple(tensor.shape) != (sample_count, sequence_length) or
        not tensor.is_contiguous()):
        raise ValueError("Corpus tokens must be contiguous CPU int64 with the frozen shape")
    return tensor


def sha256_raw_int64_tensor(tensor: Any, torch_module: Any) -> str:
    if (not isinstance(tensor, torch_module.Tensor) or
        tensor.dtype != torch_module.int64 or tensor.device.type != "cpu" or
        not tensor.is_contiguous()):
        raise ValueError("Raw corpus hash requires contiguous CPU int64 tensor")
    canonical = tensor.detach().numpy().astype("<i8", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def select_tokens(dataset: Any, tokenizer: Any, torch_module: Any, spec: dict[str, Any]) -> Any:
    selection = spec["selection"]
    shuffled = dataset.shuffle(seed=selection["shuffle_seed"])
    valid_seen = 0
    selected = []
    for sample_index, example in enumerate(shuffled):
        if not isinstance(example, dict) or selection["text_column"] not in example:
            raise ValueError(f"Dataset example {sample_index} lacks text")
        text = example[selection["text_column"]]
        if not isinstance(text, str):
            raise ValueError(f"Dataset example {sample_index} has non-string text")
        if not text.strip():
            continue
        token_ids = tokenizer.encode(
            text[:selection["character_limit"]], add_special_tokens=True
        )
        if not isinstance(token_ids, list) or any(
            type(token_id) is not int or token_id < 0 for token_id in token_ids
        ):
            raise ValueError(f"Tokenizer returned invalid IDs for example {sample_index}")
        if len(token_ids) < selection["minimum_token_count"]:
            continue
        if valid_seen >= selection["skip_valid_examples"]:
            selected.append(token_ids[:selection["sequence_length"]])
            if len(selected) == selection["sample_count"]:
                break
        valid_seen += 1
    if len(selected) != selection["sample_count"]:
        raise ValueError(
            f"Only {len(selected)} valid examples selected; expected {selection['sample_count']}"
        )
    tensor = torch_module.tensor(selected, dtype=torch_module.int64, device="cpu").contiguous()
    return validate_tokens(
        tensor, torch_module,
        sample_count=selection["sample_count"], sequence_length=selection["sequence_length"],
    )


def build_manifest(
    spec: dict[str, Any], spec_hash: str, script_hash: str,
    tokenizer_dir: Path, tokenizer_records: list[dict[str, Any]],
    artifact_path: Path, artifact_hash: str, raw_hash: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "attempt_id": spec["attempt_id"],
        "hash_algorithm": "sha256",
        "corpus_spec_sha256": spec_hash,
        "freeze_script_sha256": script_hash,
        "dataset": dict(spec["dataset"]),
        "tokenizer_checkpoint": {
            "source": spec["tokenizer"]["source"],
            "directory": str(tokenizer_dir),
            "file_count": len(tokenizer_records),
            "files": tokenizer_records,
        },
        "selection": dict(spec["selection"]),
        "tensor": {
            "shape": [spec["selection"]["sample_count"], spec["selection"]["sequence_length"]],
            "dtype": spec["artifact"]["dtype"],
            "device": "cpu",
            "contiguous": True,
        },
        "artifact": {
            "path": str(artifact_path),
            "serialized_sha256": artifact_hash,
            "raw_tensor_sha256": raw_hash,
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    serialized = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise ValueError(f"Corpus manifest already exists: {path}") from exc


def freeze_corpus(
    spec_path: Path, tokenizer_dir: Path, artifact_path: Path, manifest_path: Path,
) -> None:
    require_output_absent(artifact_path, manifest_path)
    spec = load_spec(spec_path)
    spec_hash = sha256_file(spec_path)
    script_path = Path(__file__).resolve()
    script_hash = sha256_file(script_path)
    tokenizer_records = tokenizer_file_records(tokenizer_dir)

    from datasets import load_dataset
    from transformers import AutoTokenizer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    dataset = load_dataset(
        spec["dataset"]["repo_id"], revision=spec["dataset"]["revision"],
        split=spec["dataset"]["split"], streaming=False,
    )
    tensor = select_tokens(dataset, tokenizer, torch, spec)
    if (sha256_file(spec_path) != spec_hash or sha256_file(script_path) != script_hash or
        tokenizer_file_records(tokenizer_dir) != tokenizer_records):
        raise ValueError("Corpus inputs changed during freezing")
    require_output_absent(artifact_path, manifest_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with artifact_path.open("xb") as stream:
            torch.save(tensor, stream)
    except FileExistsError as exc:
        raise ValueError(f"Corpus artifact already exists: {artifact_path}") from exc
    manifest = build_manifest(
        spec, spec_hash, script_hash, tokenizer_dir, tokenizer_records,
        artifact_path, sha256_file(artifact_path), sha256_raw_int64_tensor(tensor, torch),
    )
    write_manifest(manifest_path, manifest)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-spec-path", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--merged-tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        freeze_corpus(
            args.corpus_spec_path.expanduser(), args.merged_tokenizer_dir.expanduser(),
            args.artifact_path.expanduser(), args.manifest_path.expanduser(),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.artifact_path} and {args.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
