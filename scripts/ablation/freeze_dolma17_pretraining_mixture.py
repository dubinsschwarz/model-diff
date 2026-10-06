#!/usr/bin/env python3
"""Freeze Attempt139's two source-quota corpora from bounded Dolma v1.7 streams."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from contextlib import ExitStack
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.request import Request, urlopen

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = "139_dolma17_pretraining_mixture"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "77461597e42af2b45897c7c2dd753975d8885c9f2aab148f4c453373e9c143f2"


def log(message: str) -> None:
    print("[139] " + message, flush=True)


def path_of(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"Frozen SHA256 mismatch: {path}")


def load_spec() -> dict:
    require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text())
    if (spec["attempt_id"] != ATTEMPT or spec["source_order"] != list(spec["source_token_counts"])
            or spec["no_code_excluded_sources"] != ["stackexchange", "starcoder"]
            or spec["corpus"]["sample_count"] != 4096
            or spec["corpus"]["sequence_length"] != 128
            or spec["corpus"]["add_special_tokens"] is not True
            or spec["candidate"]["tensor_order"] != ["response_dolma17_native", "response_dolma17_no_code"]):
        raise ValueError("Attempt139 fixed corpus plan changed")
    for label in ("native", "no_code"):
        selected = [name for name in spec["source_order"] if label == "native"
                    or name not in spec["no_code_excluded_sources"]]
        if quota_plan(spec["source_token_counts"], selected) != spec[label]:
            raise ValueError(f"{label} exact source quota plan changed")
    return spec


def quota_plan(counts: dict[str, int], names: list[str], count: int = 4096) -> dict:
    if not names or len(set(names)) != len(names) or any(type(counts[name]) is not int or counts[name] <= 0 for name in names):
        raise ValueError("Invalid pinned source sizes")
    total = sum(counts[name] for name in names)
    floors = {name: count * counts[name] // total for name in names}
    order = sorted(names, key=lambda name: (-(count * counts[name] % total), name))
    awarded = order[:count - sum(floors.values())]
    quotas = dict(floors)
    for name in awarded:
        quotas[name] += 1
    return {"denominator_tokens": total,
            "source_weights": {name: counts[name] / total for name in names},
            "raw_quotas": {name: count * counts[name] / total for name in names},
            "raw_quota_numerators": {name: count * counts[name] for name in names},
            "integer_quotas": quotas, "remainder_awards": awarded}


def source_pool_quotas(spec: dict) -> dict[str, int]:
    return {name: max(spec["native"]["integer_quotas"].get(name, 0),
                      spec["no_code"]["integer_quotas"].get(name, 0))
            for name in spec["source_order"]}


def fetch_pinned(pin: dict) -> bytes:
    with urlopen(Request(pin["url"], headers={"User-Agent": "model-diff-attempt139/1"}), timeout=30) as response:
        data = response.read(2_000_000)
        if response.read(1):
            raise ValueError("Pinned recipe metadata unexpectedly exceeds 2 MB")
    if hashlib.sha256(data).hexdigest() != pin["sha256"]:
        raise ValueError("Ai2 pinned recipe SHA256 mismatch: " + pin["url"])
    return data


def literal_assignment(tree: ast.Module, name: str):
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            return node.value
    raise ValueError("Missing published recipe assignment: " + name)


def published_source_paths(recipe: bytes, key: str) -> list[str]:
    tree = ast.parse(recipe.decode("utf-8"))
    source_node = literal_assignment(tree, "DATA_SOURCES")
    sources = {ast.literal_eval(k): ast.literal_eval(v) for k, v in
               zip(source_node.keys, source_node.values) if isinstance(v, ast.List)}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id == "DATA_PATHS"
                and ast.literal_eval(node.targets[0].slice) == key):
            if not isinstance(node.value, ast.Call) or not isinstance(node.value.func, ast.Name) or node.value.func.id != "build_collection_include":
                raise ValueError("Published recipe is not a literal source inclusion")
            names = ast.literal_eval(node.value.args[0])
            return names, sum((sources[name] for name in names), [])
    raise ValueError("Missing published DataDecide recipe: " + key)


def verify_authoritative_recipe(spec: dict) -> tuple[list[str], dict]:
    log("resolving pinned Dolma recipe")
    pins = spec["recipe_provenance"]
    config = fetch_pinned(pins["native_training_config"]).decode("utf-8")
    recipe = fetch_pinned(pins["source_resolved_counts_and_no_code"])
    card = fetch_pinned(pins["datadecide_card"]).decode("utf-8")
    fetch_pinned(pins["dolma_card"])
    if "run_name: OLMo-1.7-7B" not in config or "Starcoder, StackExchange" not in card:
        raise ValueError("Pinned Ai2 recipe identity/ablation text changed")
    tree = ast.parse(recipe.decode("utf-8"))
    sizes = ast.literal_eval(literal_assignment(tree, "SOURCES_SIZES"))
    if {name: sizes[name]["total_size"] for name in spec["source_order"]} != spec["source_token_counts"]:
        raise ValueError("Published integer source token totals changed")
    native_names, native_paths = published_source_paths(recipe, "dolma17")
    no_code_names, no_code_paths = published_source_paths(recipe, "no_code")
    if (native_names != spec["source_order"] or
            no_code_names != [name for name in native_names if name not in spec["no_code_excluded_sources"]]):
        raise ValueError("Published source ordering or no-code exclusions changed")
    config_paths = [url.removeprefix("https://olmo-data.org/") for url in
                    re.findall(r"^    - (https://olmo-data.org/preprocessed/[^\n]+)", config, re.M)]
    if Counter(native_paths) != Counter(config_paths) or len(native_paths) != 1033:
        raise ValueError("Published DataDecide source multiset differs from OLMo training config")
    if Counter(native_paths) - Counter(no_code_paths) != Counter(
            path for name in spec["no_code_excluded_sources"]
            for path in published_source_paths_for_name(recipe, name)):
        raise ValueError("No-code removes sources besides StarCoder and StackExchange")
    url_list = fetch_pinned({"url": spec["dataset"]["url_list"], "sha256": spec["dataset"]["sha256"]})
    urls = url_list.decode("utf-8").splitlines()
    if len(urls) != 2419 or len(set(urls)) != len(urls) or any(
            not url.startswith("https://olmo-data.org/dolma-v1_7/") or not url.endswith(".json.gz")
            for url in urls):
        raise ValueError("Pinned Dolma v1.7 shard inventory changed")
    grouped = defaultdict(list)
    for url in urls:
        grouped[url.split("/")[4]].append(url)
    if any(not grouped[folder] for folders in spec["dataset"]["source_folders"].values() for folder in folders):
        raise ValueError("A published source lacks Dolma v1.7 text shards")
    return urls, grouped


def published_source_paths_for_name(recipe: bytes, name: str) -> list[str]:
    node = literal_assignment(ast.parse(recipe.decode("utf-8")), "DATA_SOURCES")
    for key, value in zip(node.keys, node.values):
        if ast.literal_eval(key) == name:
            return ast.literal_eval(value)
    raise ValueError("Missing source " + name)


class CountingReader:
    def __init__(self, response, max_bytes: int, total_counter: list[int]):
        self.response = response
        self.max_bytes = max_bytes
        self.total_counter = total_counter
        self.bytes_read = 0
        self.digest = hashlib.sha256()

    def read(self, n=-1):
        data = self.response.read(n)
        self.bytes_read += len(data)
        self.total_counter[0] += len(data)
        if self.total_counter[0] > self.max_bytes:
            raise ValueError("Bounded source transfer exceeded precommitted 20 GB cap")
        self.digest.update(data)
        return data


class FolderStream:
    def __init__(self, urls: list[str], max_bytes: int, counter: list[int], deadline: float):
        self.urls, self.max_bytes, self.counter, self.deadline = urls, max_bytes, counter, deadline
        self.index = -1
        self.stack = None
        self.lines = 0
        self.shards = []

    def _close(self):
        if self.stack is not None:
            if self.shards and getattr(self, "reader", None) is not None:
                self.shards[-1]["compressed_prefix_bytes"] = self.reader.bytes_read
                self.shards[-1]["compressed_prefix_sha256"] = self.reader.digest.hexdigest()
                self.shards[-1]["lines_read"] = self.lines
            self.stack.close()
            self.stack = None

    def close(self):
        self._close()

    def next_document(self):
        while True:
            if time.monotonic() > self.deadline:
                raise TimeoutError("Corpus freeze exceeded precommitted one-hour wall clock cap")
            if self.stack is None:
                self.index += 1
                if self.index >= len(self.urls):
                    raise ValueError("Pinned source shard inventory exhausted before quota")
                self.stack = ExitStack()
                self.reader = None
                url = self.urls[self.index]
                response = self.stack.enter_context(urlopen(
                    Request(url, headers={"User-Agent": "model-diff-attempt139/1"}), timeout=120))
                if response.status != 200:
                    raise ValueError("Unexpected nonstreaming shard response: " + url)
                self.reader = CountingReader(response, self.max_bytes, self.counter)
                self.shards.append({"url": url, "content_length": response.headers.get("Content-Length"),
                                    "etag": response.headers.get("ETag"),
                                    "last_modified": response.headers.get("Last-Modified")})
                compressed = self.stack.enter_context(gzip.GzipFile(fileobj=self.reader))
                self.text = self.stack.enter_context(io.TextIOWrapper(compressed, encoding="utf-8"))
                self.lines = 0
            line = self.text.readline()
            if line:
                self.lines += 1
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError("Dolma source document lacks string text")
                return record["text"], self.shards[-1]["url"], self.lines
            self._close()


def token_row(text: str, tokenizer, torch):
    if not text.strip():
        return None
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=128)
    if not isinstance(ids, list) or any(type(item) is not int or item < 0 for item in ids):
        raise ValueError("Pinned Qwen tokenizer returned malformed IDs")
    if len(ids) < 128:
        return None
    return torch.tensor(ids[:128], dtype=torch.int64)


def freeze_source(name: str, folders: list[str], grouped: dict, quota: int,
                  tokenizer, torch, spec: dict, counter: list[int], deadline: float):
    streams = [FolderStream(grouped[folder], spec["corpus"]["stream_max_compressed_bytes"],
                            counter, deadline) for folder in folders]
    rows, accepted, scanned = [], [], 0
    try:
        while len(rows) < quota:
            folder_index = scanned % len(streams)
            text, url, line_no = streams[folder_index].next_document()
            scanned += 1
            row = token_row(text, tokenizer, torch)
            if row is None:
                continue
            rows.append(row)
            accepted.append({"url": url, "line": line_no,
                             "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    finally:
        for stream in streams:
            stream.close()
    tensor = torch.stack(rows).contiguous()
    return tensor, {"source": name, "folders": folders, "target_rows": quota,
                    "documents_scanned": scanned, "accepted_documents": accepted,
                    "shards": [record for stream in streams for record in stream.shards],
                    "raw_tensor_sha256": raw_int64_sha256(tensor, torch)}


def raw_int64_sha256(tensor, torch) -> str:
    if tensor.dtype != torch.int64 or tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("Expected contiguous CPU int64 token tensor")
    return hashlib.sha256(tensor.numpy().astype("<i8", copy=False).tobytes(order="C")).hexdigest()


def candidate_from_pools(pools: dict, quota: dict, source_order: list[str], seed: int, torch):
    chunks, source_ids = [], []
    for name in source_order:
        count = quota.get(name, 0)
        if count:
            if name not in pools or pools[name].shape[0] < count:
                raise ValueError("Shared source pool does not nest candidate quota")
            chunks.append(pools[name][:count])
            source_ids.extend([name] * count)
    joined = torch.cat(chunks, dim=0).contiguous()
    if tuple(joined.shape) != (4096, 128):
        raise ValueError("Candidate source quotas do not sum to 4096")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(4096)
    output = joined[torch.from_numpy(permutation.copy())].contiguous()
    return output, {"permutation_sha256": hashlib.sha256(permutation.astype("<i8", copy=False).tobytes()).hexdigest(),
                    "shuffled_source_names_sha256": hashlib.sha256(json.dumps(
                        [source_ids[i] for i in permutation], separators=(",", ":")).encode()).hexdigest()}


def output_paths(spec: dict) -> list[Path]:
    c = spec["corpus"]
    return [path_of(c[key]) for key in ("source_pool_path", "native_path", "no_code_path",
            "source_manifest_path", "native_manifest_path", "no_code_manifest_path")]


def refuse_outputs(paths: list[Path]) -> None:
    if len(set(paths)) != len(paths) or any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError("Attempt139 corpus output already exists or paths overlap")


def publish_torch(path: Path, value, torch) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(value, stream)
    return sha256_file(path)


def publish_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return sha256_file(path)


def validate_frozen(spec: dict, torch):
    paths = output_paths(spec)
    source_manifest = json.loads(paths[3].read_text())
    pools = torch.load(paths[0], map_location="cpu", weights_only=True)
    if (source_manifest["spec_sha256"] != SPEC_SHA256 or
            source_manifest["source_pool_artifact"]["serialized_sha256"] != sha256_file(paths[0]) or
            list(pools) != spec["source_order"]):
        raise ValueError("Frozen source manifest/pool mismatch")
    for name in spec["source_order"]:
        value = pools[name]
        if (tuple(value.shape) != (source_pool_quotas(spec)[name], 128)
                or raw_int64_sha256(value, torch) != source_manifest["sources"][name]["raw_tensor_sha256"]):
            raise ValueError("Frozen source pool tensor mismatch: " + name)
    outputs = {}
    for label, path, manifest_path in (("native", paths[1], paths[4]), ("no_code", paths[2], paths[5])):
        manifest = json.loads(manifest_path.read_text())
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        expected, shuffle = candidate_from_pools(pools, spec[label]["integer_quotas"],
                                                 spec["source_order"], spec["corpus"]["final_shuffle_seed"], torch)
        if (tuple(tensor.shape) != (4096, 128) or tensor.dtype != torch.int64 or
                not tensor.is_contiguous() or not torch.equal(tensor, expected) or
                manifest["shuffle"] != shuffle or manifest["quota_plan"] != spec[label] or
                manifest["artifact"]["serialized_sha256"] != sha256_file(path) or
                manifest["artifact"]["raw_tensor_sha256"] != raw_int64_sha256(tensor, torch)):
            raise ValueError("Frozen candidate corpus failed reload/hash/nesting validation: " + label)
        outputs[label] = tensor
    return outputs, {"source_manifest_sha256": sha256_file(paths[3]),
                     "native_manifest_sha256": sha256_file(paths[4]),
                     "no_code_manifest_sha256": sha256_file(paths[5])}


def freeze(*, tokenizer=None, torch_module=None, source_loader=None):
    spec = load_spec()
    paths = output_paths(spec)
    refuse_outputs(paths)
    log("estimated streamed download 0.2–5 GB; persistent artifacts <1 GB; freeze 5–30 min (hard cap 1 hour)")
    urls, grouped = verify_authoritative_recipe(spec) if source_loader is None else source_loader(spec)
    import torch as default_torch
    torch = torch_module or default_torch
    from transformers import AutoTokenizer
    model_dir = path_of(spec["final_checkpoint"]["directory"])
    for record in spec["final_checkpoint"]["files"]:
        file = model_dir / record["path"]
        if file.stat().st_size != record["size_bytes"] or sha256_file(file) != record["sha256"]:
            raise ValueError("Pinned final checkpoint/tokenizer inventory changed")
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if shutil.disk_usage(paths[0].parent.parent).free < 1_000_000_000:
        raise ValueError("Less than 1 GB free for bounded Attempt139 token artifacts")
    counter = [0]
    deadline = time.monotonic() + spec["corpus"]["freeze_max_seconds"]
    pools, source_records = {}, {}
    for name, quota in source_pool_quotas(spec).items():
        log(f"freezing source pools {name} {quota} rows")
        pool, record = freeze_source(name, spec["dataset"]["source_folders"][name], grouped,
                                     quota, tokenizer, torch, spec, counter, deadline)
        pools[name], source_records[name] = pool, record
    refuse_outputs(paths)
    pool_hash = publish_torch(paths[0], pools, torch)
    source_manifest = {"format_version": 1, "attempt_id": ATTEMPT, "spec_sha256": SPEC_SHA256,
                       "freezer_sha256": sha256_file(Path(__file__)), "recipe_provenance": spec["recipe_provenance"],
                       "dataset": spec["dataset"], "source_token_counts": spec["source_token_counts"],
                       "pool_quotas": source_pool_quotas(spec), "extraction_rules": spec["corpus"],
                       "tokenizer_checkpoint_files": spec["final_checkpoint"]["files"],
                       "total_compressed_bytes_read": counter[0], "sources": source_records,
                       "source_pool_artifact": {"path": str(paths[0]), "serialized_sha256": pool_hash}}
    publish_json(paths[3], source_manifest)
    for label, path, manifest_path in (("native", paths[1], paths[4]), ("no_code", paths[2], paths[5])):
        tensor, shuffle = candidate_from_pools(pools, spec[label]["integer_quotas"],
                                               spec["source_order"], spec["corpus"]["final_shuffle_seed"], torch)
        artifact_hash = publish_torch(path, tensor, torch)
        publish_json(manifest_path, {"format_version": 1, "attempt_id": ATTEMPT,
                     "candidate": label, "spec_sha256": SPEC_SHA256,
                     "source_manifest_sha256": sha256_file(paths[3]),
                     "source_pool_serialized_sha256": pool_hash, "quota_plan": spec[label],
                     "source_order": spec["source_order"], "shuffle_seed": spec["corpus"]["final_shuffle_seed"],
                     "shuffle": shuffle, "tokenizer_checkpoint_files": spec["final_checkpoint"]["files"],
                     "artifact": {"path": str(path), "serialized_sha256": artifact_hash,
                                  "raw_tensor_sha256": raw_int64_sha256(tensor, torch),
                                  "shape": [4096, 128], "dtype": "torch.int64", "contiguous": True}})
    outputs, hashes = validate_frozen(spec, torch)
    log("froze both 4096 x 128 corpora and revalidated source-pool nesting")
    return outputs, hashes


if __name__ == "__main__":
    try:
        freeze()
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
