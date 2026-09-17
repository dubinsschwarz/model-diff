#!/usr/bin/env python3

import hashlib
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(os.environ.get("MODEL_DIR", "/root/model-diff-scratch/models"))
LOCK_PATH = PROJECT / "models.lock.json"
OUTPUT_PATH = PROJECT / "downloaded-model-hashes.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


lock = json.loads(LOCK_PATH.read_text())

manifest = {
    "hash_algorithm": "sha256",
    "models": {},
}

for role in ("base", "adapter"):
    root = MODEL_ROOT / role
    if not root.is_dir():
        raise SystemExit(f"Missing model directory: {root}")

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(root)

        # Hugging Face local-dir bookkeeping is cache metadata,
        # not part of the downloaded model artifact.
        if ".cache" in rel.parts:
            continue

        files.append({
            "path": rel.as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    manifest["models"][role] = {
        "repo_id": lock[role]["repo_id"],
        "revision": lock[role]["revision"],
        "file_count": len(files),
        "files": files,
    }

OUTPUT_PATH.write_text(json.dumps(manifest, indent=2) + "\n")

print(f"Wrote {OUTPUT_PATH}")
for role, record in manifest["models"].items():
    total = sum(f["size_bytes"] for f in record["files"])
    print(f"{role}: {record['file_count']} files, {total:,} bytes")
