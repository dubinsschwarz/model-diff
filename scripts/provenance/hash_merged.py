#!/usr/bin/env python3

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
LOCK_PATH = PROJECT / "models.lock.json"
MERGE_SCRIPT_PATH = PROJECT / "scripts" / "merge_lora.py"
DOWNLOADED_HASHES_PATH = PROJECT / "downloaded-model-hashes.json"
OUTPUT_PATH = PROJECT / "merged-model-hashes.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def load_source_models(lock_path: Path) -> dict[str, dict[str, str]]:
    require_file(lock_path, "model lock file")

    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model lock file {lock_path}: {exc}") from exc

    source_models = {}
    for role in ("base", "adapter"):
        try:
            repo_id = lock[role]["repo_id"]
            revision = lock[role]["revision"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Model lock file is missing {role} repo_id/revision: {lock_path}"
            ) from exc

        if not isinstance(repo_id, str) or not isinstance(revision, str):
            raise ValueError(
                f"Model lock file has invalid {role} repo_id/revision: {lock_path}"
            )

        source_models[role] = {
            "repo_id": repo_id,
            "revision": revision,
        }

    return source_models


def hash_merged_files(merged_dir: Path) -> list[dict[str, Any]]:
    candidates = []
    try:
        for path in merged_dir.rglob("*"):
            relative_path = path.relative_to(merged_dir)

            # Hugging Face local-dir bookkeeping is cache metadata,
            # not part of the merged model artifact.
            if ".cache" in relative_path.parts or not path.is_file():
                continue

            candidates.append((relative_path.as_posix(), path))
    except OSError as exc:
        raise ValueError(f"Could not enumerate merged model directory: {exc}") from exc

    files = []
    for relative_path, path in sorted(candidates, key=lambda item: item[0]):
        try:
            size_bytes = path.stat().st_size
            digest = sha256_file(path)
        except OSError as exc:
            raise ValueError(f"Could not hash merged model file {path}: {exc}") from exc

        files.append(
            {
                "path": relative_path,
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )

    return files


def build_manifest(
    merged_dir: Path,
    lock_path: Path = LOCK_PATH,
    merge_script_path: Path = MERGE_SCRIPT_PATH,
    downloaded_hashes_path: Path = DOWNLOADED_HASHES_PATH,
) -> dict[str, Any]:
    require_directory(merged_dir, "merged model directory")
    require_file(merge_script_path, "merge script")
    require_file(downloaded_hashes_path, "downloaded-model manifest")
    source_models = load_source_models(lock_path)

    files = hash_merged_files(merged_dir)
    if not files:
        raise ValueError(f"No merged artifact files found: {merged_dir}")

    return {
        "hash_algorithm": "sha256",
        "file_count": len(files),
        "total_bytes": sum(file["size_bytes"] for file in files),
        "files": files,
        "merge_script_sha256": sha256_file(merge_script_path),
        "downloaded_model_hashes_sha256": sha256_file(downloaded_hashes_path),
        "base": source_models["base"],
        "adapter": source_models["adapter"],
        "merge_metadata": {
            "dtype": "float32",
            "safe_merge": True,
            "tokenizer_source": "base",
        },
    }


def main() -> int:
    merged_dir = Path(
        os.environ.get("MERGED_MODEL_DIR", DEFAULT_MERGED_MODEL_DIR)
    ).expanduser()

    try:
        manifest = build_manifest(merged_dir)
        OUTPUT_PATH.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {OUTPUT_PATH}")
    print(f"merged: {manifest['file_count']} files, {manifest['total_bytes']:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
