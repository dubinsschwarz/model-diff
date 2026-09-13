#!/usr/bin/env python3

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_FILE = REPO_ROOT / "models.lock.json"


def main() -> int:
    with LOCK_FILE.open() as f:
        lock = json.load(f)

    api = HfApi()
    all_match = True

    for role in ("base", "adapter"):
        repo = lock[role]["repo_id"]
        expected = lock[role]["revision"]

        print(role.upper())
        print(f"  repo:      {repo}")
        print(f"  expected:  {expected}")

        try:
            info = api.model_info(repo, revision=expected)
        except Exception as exc:
            print(f"  ERROR:     Could not resolve revision: {exc}")
            print()
            all_match = False
            continue

        resolved = info.sha
        matches = resolved == expected

        print(f"  resolved:  {resolved}")
        print(f"  MATCH:     {matches}")
        print()

        if not matches:
            all_match = False

    if not all_match:
        print("FAILED: one or more pinned Hugging Face revisions did not match.")
        return 1

    print("OK: all pinned Hugging Face revisions resolve exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
