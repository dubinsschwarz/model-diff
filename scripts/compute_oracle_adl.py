#!/usr/bin/env python3

"""Compute the base-to-fine-tuned oracle activation difference."""

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DOWNLOADED_HASHES_PATH = PROJECT / "downloaded-model-hashes.json"
MERGED_HASHES_PATH = PROJECT / "merged-model-hashes.json"
PROBE_MANIFEST_PATH = PROJECT / "oracle-probe-manifest.json"
DATASET_LOCK_PATH = PROJECT / "datasets.lock.json"
FREEZE_PROBE_SCRIPT_PATH = PROJECT / "scripts" / "freeze_oracle_probe.py"
MERGE_SCRIPT_PATH = PROJECT / "scripts" / "merge_lora.py"
SCRIPT_PATH = PROJECT / "scripts" / "compute_oracle_adl.py"
MANIFEST_PATH = PROJECT / "oracle-adl-manifest.json"

DEFAULT_BASE_MODEL_DIR = Path("/root/model-diff-scratch/models/base")
DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_PROBE_PATH = Path(
    "/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt"
)
DEFAULT_ARTIFACT_PATH = Path(
    "/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt"
)

RELATIVE_LAYER = 0.5
NUM_LAYERS = 28
LAYER_INDEX = 13
SAMPLE_COUNT = 10_000
SEQUENCE_LENGTH = 128
BATCH_SIZE = 32


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_sha256(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Malformed SHA-256 for {description}")
    return value


def sha256_raw_probe_tensor(tensor: Any, torch_module: Any) -> str:
    """Hash probe values as C-order, little-endian signed int64 bytes."""
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError("Cannot hash a non-tensor probe")
    if tensor.dtype != torch_module.long:
        raise ValueError(f"Cannot hash non-torch.long probe tensor: {tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ValueError(f"Cannot hash non-CPU probe tensor: {tensor.device}")

    canonical = tensor.detach().contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def sha256_raw_float32_tensor(tensor: Any, torch_module: Any) -> str:
    """Hash float32 values as C-order, little-endian IEEE-754 bytes."""
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError("Cannot hash a non-tensor activation")
    if tensor.dtype != torch_module.float32:
        raise ValueError(
            f"Cannot hash non-torch.float32 activation tensor: {tensor.dtype}"
        )
    if tensor.device.type != "cpu":
        raise ValueError(f"Cannot hash non-CPU activation tensor: {tensor.device}")

    canonical = tensor.detach().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


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


def load_sha256_manifest(path: Path, description: str) -> dict[str, Any]:
    manifest = load_json_object(path, description)
    if manifest.get("hash_algorithm") != "sha256":
        raise ValueError(f"{description} does not specify SHA-256: {path}")
    return manifest


def summarize_paths(paths: list[str]) -> str:
    displayed = paths[:5]
    summary = ", ".join(displayed)
    if len(paths) > len(displayed):
        summary += f", ... ({len(paths)} total)"
    return summary


def collect_artifact_files(root: Path) -> dict[str, Path]:
    files = {}
    try:
        for path in root.rglob("*"):
            relative_path = path.relative_to(root)
            if ".cache" in relative_path.parts or not path.is_file():
                continue
            files[relative_path.as_posix()] = path
    except OSError as exc:
        raise ValueError(f"Could not enumerate local artifact {root}: {exc}") from exc
    return files


def verify_local_artifact(
    root: Path,
    manifest_entry: dict[str, Any],
    description: str,
) -> None:
    manifest_files = manifest_entry.get("files")
    if not isinstance(manifest_files, list):
        raise ValueError(f"Malformed {description} manifest: files")

    expected = {}
    for record in manifest_files:
        if not isinstance(record, dict):
            raise ValueError(f"Malformed {description} manifest file record")

        relative_path = record.get("path")
        size_bytes = record.get("size_bytes")
        expected_sha256 = record.get("sha256")
        if not isinstance(relative_path, str):
            raise ValueError(f"Malformed {description} manifest path")

        parsed_path = PurePosixPath(relative_path)
        if (
            parsed_path.is_absolute()
            or relative_path != parsed_path.as_posix()
            or any(part in ("", ".", "..") for part in parsed_path.parts)
            or ".cache" in parsed_path.parts
        ):
            raise ValueError(f"Malformed {description} manifest path: {relative_path}")
        if relative_path in expected:
            raise ValueError(
                f"Malformed {description} manifest: duplicate path {relative_path}"
            )
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError(
                f"Malformed {description} manifest size for {relative_path}"
            )
        require_sha256(expected_sha256, f"{description} file {relative_path}")

        expected[relative_path] = {
            "size_bytes": size_bytes,
            "sha256": expected_sha256,
        }

    file_count = manifest_entry.get("file_count")
    if type(file_count) is not int or file_count != len(expected):
        raise ValueError(f"Malformed {description} manifest: file_count")

    expected_total_bytes = sum(record["size_bytes"] for record in expected.values())
    if "total_bytes" in manifest_entry:
        total_bytes = manifest_entry["total_bytes"]
        if type(total_bytes) is not int or total_bytes != expected_total_bytes:
            raise ValueError(f"Malformed {description} manifest: total_bytes")

    actual = collect_artifact_files(root)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        raise ValueError(
            f"{description} file set mismatch; missing: {summarize_paths(missing)}"
        )
    if extra:
        raise ValueError(
            f"{description} file set mismatch; extra: {summarize_paths(extra)}"
        )

    for relative_path in sorted(expected):
        path = actual[relative_path]
        record = expected[relative_path]
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise ValueError(
                f"Could not inspect local artifact file {path}: {exc}"
            ) from exc
        if actual_size != record["size_bytes"]:
            raise ValueError(f"{description} size mismatch for {relative_path}")
        try:
            actual_sha256 = sha256_file(path)
        except OSError as exc:
            raise ValueError(
                f"Could not hash local artifact file {path}: {exc}"
            ) from exc
        if actual_sha256 != record["sha256"]:
            raise ValueError(f"{description} SHA-256 mismatch for {relative_path}")


def downloaded_model_entry(
    manifest: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    try:
        entry = manifest["models"][role]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Downloaded-model manifest is missing {role}") from exc
    if not isinstance(entry, dict):
        raise ValueError(f"Downloaded-model manifest has invalid {role} entry")
    return entry


def model_identity(entry: dict[str, Any], description: str) -> dict[str, str]:
    repo_id = entry.get("repo_id")
    revision = entry.get("revision")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError(f"Malformed {description} identity: repo_id")
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"Malformed {description} identity: revision")
    return {"repo_id": repo_id, "revision": revision}


def validate_output_paths(artifact_path: Path, manifest_path: Path) -> None:
    if path_exists(artifact_path, "oracle artifact"):
        raise ValueError(f"Oracle artifact already exists: {artifact_path}")
    if path_exists(manifest_path, "oracle manifest"):
        raise ValueError(f"Oracle manifest already exists: {manifest_path}")
    if path_exists(artifact_path.parent, "oracle artifact directory"):
        require_directory(artifact_path.parent, "oracle artifact directory")


def relative_layer_index(relative_layer: float, num_layers: int) -> int:
    if isinstance(relative_layer, bool) or not isinstance(relative_layer, (int, float)):
        raise ValueError("Relative layer must be numeric")
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError("Number of layers must be a positive integer")
    if not 0.0 <= relative_layer <= 1.0:
        raise ValueError("Relative layer must be between 0 and 1")
    return int(relative_layer * (num_layers - 1))


def validate_qwen_config(
    config: dict[str, Any],
    description: str,
) -> int:
    if config.get("model_type") != "qwen3":
        raise ValueError(f"{description} config is not Qwen3")
    if config.get("num_hidden_layers") != NUM_LAYERS:
        raise ValueError(f"{description} config must have {NUM_LAYERS} layers")
    hidden_size = config.get("hidden_size")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError(f"{description} config has invalid hidden_size")
    return hidden_size


def validate_probe_manifest(
    manifest: dict[str, Any],
    probe_path: Path,
    downloaded_hashes_path: Path,
    dataset_lock_path: Path,
    freeze_probe_script_path: Path,
) -> tuple[str, str]:
    expected_tensor = {"shape": [SAMPLE_COUNT, SEQUENCE_LENGTH], "dtype": "torch.int64"}
    expected_scalars = {
        "seed": 42,
        "text_column": "text",
        "n": SEQUENCE_LENGTH,
        "max_samples": SAMPLE_COUNT,
        "character_truncation_length": SEQUENCE_LENGTH * 10,
        "tokenizer_source": "base",
        "tensor": expected_tensor,
    }
    for key, expected_value in expected_scalars.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"Probe manifest mismatch: {key}")

    dataset_lock = load_json_object(dataset_lock_path, "dataset lock file")
    expected_dataset = dict(dataset_lock)
    expected_dataset["split"] = "train"
    if manifest.get("dataset") != expected_dataset:
        raise ValueError("Probe manifest mismatch: dataset")

    expected_links = {
        "datasets_lock_sha256": sha256_file(dataset_lock_path),
        "downloaded_model_hashes_sha256": sha256_file(downloaded_hashes_path),
        "freeze_oracle_probe_script_sha256": sha256_file(freeze_probe_script_path),
    }
    for key, expected_value in expected_links.items():
        require_sha256(manifest.get(key), f"probe manifest {key}")
        if manifest[key] != expected_value:
            raise ValueError(f"Probe manifest mismatch: {key}")

    serialized_hash = require_sha256(
        manifest.get("fineweb_tokens_sha256"),
        "probe artifact",
    )
    raw_hash = require_sha256(manifest.get("raw_tensor_sha256"), "raw probe tensor")
    try:
        actual_serialized_hash = sha256_file(probe_path)
    except OSError as exc:
        raise ValueError(f"Could not hash probe artifact {probe_path}: {exc}") from exc
    if actual_serialized_hash != serialized_hash:
        raise ValueError("Probe artifact SHA-256 mismatch")
    return serialized_hash, raw_hash


def validate_probe_tensor(tensor: Any, torch_module: Any) -> None:
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError("Malformed probe artifact: expected a tensor")
    if tensor.dtype != torch_module.long:
        raise ValueError(f"Probe must have dtype torch.long, found {tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ValueError(f"Probe must be on CPU, found {tensor.device}")
    if tuple(tensor.shape) != (SAMPLE_COUNT, SEQUENCE_LENGTH):
        raise ValueError(
            "Probe must have shape "
            f"[{SAMPLE_COUNT}, {SEQUENCE_LENGTH}], found {list(tensor.shape)}"
        )


def load_and_validate_probe(
    probe_path: Path,
    expected_raw_hash: str,
    torch_module: Any,
) -> Any:
    try:
        probe = torch_module.load(probe_path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not read probe artifact {probe_path}: {exc}") from exc
    validate_probe_tensor(probe, torch_module)
    if sha256_raw_probe_tensor(probe, torch_module) != expected_raw_hash:
        raise ValueError("Probe raw tensor SHA-256 mismatch")
    return probe.contiguous()


def validate_inputs(
    base_dir: Path,
    merged_dir: Path,
    probe_path: Path,
    artifact_path: Path,
    manifest_path: Path,
    torch_module: Any,
    *,
    downloaded_hashes_path: Path = DOWNLOADED_HASHES_PATH,
    merged_hashes_path: Path = MERGED_HASHES_PATH,
    probe_manifest_path: Path = PROBE_MANIFEST_PATH,
    dataset_lock_path: Path = DATASET_LOCK_PATH,
    freeze_probe_script_path: Path = FREEZE_PROBE_SCRIPT_PATH,
    merge_script_path: Path = MERGE_SCRIPT_PATH,
) -> tuple[Any, dict[str, Any]]:
    """Validate every frozen input and output path before model loading."""
    validate_output_paths(artifact_path, manifest_path)
    require_directory(base_dir, "base model directory")
    require_directory(merged_dir, "merged model directory")
    require_file(probe_path, "probe artifact")

    downloaded_manifest = load_sha256_manifest(
        downloaded_hashes_path, "downloaded-model manifest"
    )
    merged_manifest = load_sha256_manifest(
        merged_hashes_path, "merged-model manifest"
    )
    probe_manifest = load_json_object(probe_manifest_path, "oracle-probe manifest")

    base_entry = downloaded_model_entry(downloaded_manifest, "base")
    adapter_entry = downloaded_model_entry(downloaded_manifest, "adapter")
    base_identity = model_identity(base_entry, "base model")
    ft_identity = model_identity(adapter_entry, "fine-tuned model")

    verify_local_artifact(base_dir, base_entry, "base model artifact")
    verify_local_artifact(merged_dir, merged_manifest, "merged model artifact")

    downloaded_manifest_hash = sha256_file(downloaded_hashes_path)
    if (
        merged_manifest.get("downloaded_model_hashes_sha256")
        != downloaded_manifest_hash
    ):
        raise ValueError("Merged-model manifest provenance mismatch")
    if merged_manifest.get("base") != base_identity:
        raise ValueError("Merged-model manifest base identity mismatch")
    if merged_manifest.get("adapter") != ft_identity:
        raise ValueError("Merged-model manifest fine-tuned identity mismatch")
    if merged_manifest.get("merge_script_sha256") != sha256_file(merge_script_path):
        raise ValueError("Merged-model manifest merge script mismatch")
    expected_merge_metadata = {
        "dtype": "float32",
        "safe_merge": True,
        "tokenizer_source": "base",
    }
    if merged_manifest.get("merge_metadata") != expected_merge_metadata:
        raise ValueError("Merged-model manifest merge metadata mismatch")

    base_config = load_json_object(base_dir / "config.json", "base model config")
    merged_config = load_json_object(
        merged_dir / "config.json", "merged model config"
    )
    base_hidden_size = validate_qwen_config(base_config, "Base model")
    merged_hidden_size = validate_qwen_config(merged_config, "Merged model")
    if base_hidden_size != merged_hidden_size:
        raise ValueError("Base and merged model hidden sizes differ")

    layer_index = relative_layer_index(RELATIVE_LAYER, NUM_LAYERS)
    if layer_index != LAYER_INDEX:
        raise ValueError(
            "Relative layer mapping mismatch: expected "
            f"{LAYER_INDEX}, found {layer_index}"
        )

    serialized_probe_hash, raw_probe_hash = validate_probe_manifest(
        probe_manifest,
        probe_path,
        downloaded_hashes_path,
        dataset_lock_path,
        freeze_probe_script_path,
    )
    probe = load_and_validate_probe(probe_path, raw_probe_hash, torch_module)

    provenance = {
        "serialized_probe_sha256": serialized_probe_hash,
        "raw_probe_sha256": raw_probe_hash,
        "downloaded_model_hashes_sha256": downloaded_manifest_hash,
        "merged_model_hashes_sha256": sha256_file(merged_hashes_path),
        "oracle_probe_manifest_sha256": sha256_file(probe_manifest_path),
        "base": base_identity,
        "fine_tuned": ft_identity,
        "hidden_size": base_hidden_size,
    }
    return probe, provenance


def verify_float32_parameters(
    model: Any,
    torch_module: Any,
    description: str,
) -> Any:
    floating_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    unexpected_dtypes = sorted(
        str(dtype) for dtype in floating_dtypes if dtype != torch_module.float32
    )
    if unexpected_dtypes:
        raise ValueError(
            f"{description} has unexpected floating-point parameter dtype(s): "
            f"{', '.join(unexpected_dtypes)}"
        )
    if not floating_dtypes:
        raise ValueError(f"{description} has no floating-point parameters")
    return next(iter(floating_dtypes))


def extract_block_output(output: Any, torch_module: Any) -> Any:
    """Return a transformer's output hidden states, never its hook inputs."""
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
    """Forward hook that accumulates block outputs on CPU in float64."""

    def __init__(
        self,
        torch_module: Any,
        sequence_length: int,
        hidden_size: int,
    ) -> None:
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
                f"Transformer block output must be rank 3, found rank {hidden.ndim}"
            )
        if tuple(hidden.shape[1:]) != (self.sequence_length, self.hidden_size):
            raise ValueError(
                "Transformer block output has unexpected shape: "
                f"{list(hidden.shape)}"
            )
        if hidden.dtype != self.torch.float32:
            raise ValueError(
                f"Transformer block output must be float32, found {hidden.dtype}"
            )

        # Conversion happens before reduction so the persistent sum and each
        # accumulation operation are both CPU float64.
        cpu_float64 = hidden.detach().to(device="cpu", dtype=self.torch.float64)
        self.sum.add_(cpu_float64.sum(dim=0))
        self.sample_count += hidden.shape[0]
        self.hook_calls += 1

    def mean(self, expected_sample_count: int) -> Any:
        if self.sample_count != expected_sample_count:
            raise ValueError(
                "Activation sample count mismatch: expected "
                f"{expected_sample_count}, found {self.sample_count}"
            )
        if self.hook_calls == 0:
            raise ValueError("Transformer block hook was never called")
        return (self.sum / expected_sample_count).contiguous()


def validate_loaded_model(
    model: Any,
    hidden_size: int,
    torch_module: Any,
    description: str,
) -> Any:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "qwen3":
        raise ValueError(f"{description} loaded config is not Qwen3")
    if getattr(config, "num_hidden_layers", None) != NUM_LAYERS:
        raise ValueError(f"{description} loaded config must have {NUM_LAYERS} layers")
    if getattr(config, "hidden_size", None) != hidden_size:
        raise ValueError(f"{description} loaded hidden_size mismatch")

    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or len(layers) != NUM_LAYERS:
        raise ValueError(
            f"{description} does not expose {NUM_LAYERS} transformer blocks"
        )
    verify_float32_parameters(model, torch_module, description)
    return backbone


def compute_model_mean(
    model_dir: Path,
    probe: Any,
    hidden_size: int,
    description: str,
    torch_module: Any,
    model_class: Any,
) -> Any:
    """Load one model, capture layer 13 outputs, then release the model."""
    model = model_class.from_pretrained(
        model_dir,
        dtype=torch_module.float32,
        local_files_only=True,
    )
    hook_handle = None
    try:
        backbone = validate_loaded_model(
            model, hidden_size, torch_module, description
        )
        device = torch_module.device("cuda")
        model.to(device)
        model.eval()
        verify_float32_parameters(model, torch_module, f"CUDA {description}")

        accumulator = ActivationAccumulator(
            torch_module,
            SEQUENCE_LENGTH,
            hidden_size,
        )
        # A forward hook receives both inputs and outputs. Register directly on
        # model.model.layers[13] and have the hook read only `output`.
        hook_handle = backbone.layers[LAYER_INDEX].register_forward_hook(accumulator)

        expected_batches = (SAMPLE_COUNT + BATCH_SIZE - 1) // BATCH_SIZE
        with torch_module.inference_mode():
            for start in range(0, SAMPLE_COUNT, BATCH_SIZE):
                input_ids = probe[start : start + BATCH_SIZE].to(device)
                backbone(input_ids=input_ids, use_cache=False, return_dict=True)

        if accumulator.hook_calls != expected_batches:
            raise ValueError(
                "Transformer block hook call count mismatch: expected "
                f"{expected_batches}, found {accumulator.hook_calls}"
            )
        # Divide by the exact frozen-probe population, not by a derived shape.
        return accumulator.mean(SAMPLE_COUNT)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        del model
        torch_module.cuda.empty_cache()


def prepare_stored_tensors(
    base_mean_float64: Any,
    ft_mean_float64: Any,
    hidden_size: int,
    torch_module: Any,
) -> dict[str, Any]:
    expected_shape = (SEQUENCE_LENGTH, hidden_size)
    for name, tensor in (
        ("base_mean", base_mean_float64),
        ("ft_mean", ft_mean_float64),
    ):
        if not isinstance(tensor, torch_module.Tensor):
            raise ValueError(f"{name} is not a tensor")
        if tensor.device.type != "cpu" or tensor.dtype != torch_module.float64:
            raise ValueError(f"{name} must be a CPU float64 tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{name} has unexpected shape: {list(tensor.shape)}")
        if not bool(torch_module.isfinite(tensor).all().item()):
            raise ValueError(f"{name} contains non-finite values")

    # The paper's authors compute mean(FT - base). Because both means use the
    # same 10,000 examples, mean(FT) - mean(base) is algebraically equivalent.
    difference_float64 = ft_mean_float64 - base_mean_float64
    stored = {
        "base_mean": base_mean_float64.to(torch_module.float32).contiguous(),
        "ft_mean": ft_mean_float64.to(torch_module.float32).contiguous(),
        "difference": difference_float64.to(torch_module.float32).contiguous(),
    }
    return stored


def build_artifact(stored: dict[str, Any], hidden_size: int) -> dict[str, Any]:
    return {
        "format_version": 1,
        "metadata": {
            "relative_layer": RELATIVE_LAYER,
            "layer_index": LAYER_INDEX,
            "num_layers": NUM_LAYERS,
            "sample_count": SAMPLE_COUNT,
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size": hidden_size,
            "batch_size": BATCH_SIZE,
            "model_dtype": "float32",
            "accumulator_dtype": "float64",
            "stored_dtype": "float32",
        },
        "base_mean": stored["base_mean"],
        "ft_mean": stored["ft_mean"],
        "difference": stored["difference"],
    }


def build_manifest(
    provenance: dict[str, Any],
    artifact_sha256: str,
    raw_hashes: dict[str, str],
    hidden_size: int,
    script_path: Path = SCRIPT_PATH,
) -> dict[str, Any]:
    return {
        "hash_algorithm": "sha256",
        "probe": {
            "serialized_sha256": provenance["serialized_probe_sha256"],
            "raw_tensor_sha256": provenance["raw_probe_sha256"],
        },
        "downloaded_model_hashes_sha256": provenance[
            "downloaded_model_hashes_sha256"
        ],
        "merged_model_hashes_sha256": provenance["merged_model_hashes_sha256"],
        "oracle_probe_manifest_sha256": provenance[
            "oracle_probe_manifest_sha256"
        ],
        "compute_oracle_adl_script_sha256": sha256_file(script_path),
        "base": provenance["base"],
        "fine_tuned": provenance["fine_tuned"],
        "relative_layer": RELATIVE_LAYER,
        "layer_index": LAYER_INDEX,
        "num_layers": NUM_LAYERS,
        "sample_count": SAMPLE_COUNT,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_size": hidden_size,
        "batch_size": BATCH_SIZE,
        "model_dtype": "float32",
        "accumulator_dtype": "float64",
        "stored_dtype": "float32",
        "oracle_adl_sha256": artifact_sha256,
        "raw_tensors_sha256": raw_hashes,
    }


def save_outputs(
    stored: dict[str, Any],
    hidden_size: int,
    provenance: dict[str, Any],
    torch_module: Any,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    script_path: Path = SCRIPT_PATH,
) -> None:
    validate_output_paths(artifact_path, manifest_path)
    expected_shape = (SEQUENCE_LENGTH, hidden_size)
    for name in ("base_mean", "ft_mean", "difference"):
        tensor = stored.get(name)
        if not isinstance(tensor, torch_module.Tensor):
            raise ValueError(f"Missing stored tensor: {name}")
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != torch_module.float32
            or tuple(tensor.shape) != expected_shape
            or not tensor.is_contiguous()
        ):
            raise ValueError(f"Invalid stored tensor: {name}")

    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Could not create oracle artifact directory {artifact_path.parent}: {exc}"
        ) from exc

    artifact = build_artifact(stored, hidden_size)
    try:
        with artifact_path.open("xb") as artifact_file:
            torch_module.save(artifact, artifact_file)
    except FileExistsError as exc:
        raise ValueError(f"Oracle artifact already exists: {artifact_path}") from exc
    except OSError as exc:
        raise ValueError(
            f"Could not write oracle artifact {artifact_path}: {exc}"
        ) from exc

    try:
        artifact_sha256 = sha256_file(artifact_path)
        raw_hashes = {
            name: sha256_raw_float32_tensor(stored[name], torch_module)
            for name in ("base_mean", "ft_mean", "difference")
        }
        manifest = build_manifest(
            provenance,
            artifact_sha256,
            raw_hashes,
            hidden_size,
            script_path,
        )
        with manifest_path.open("x", encoding="utf-8") as manifest_file:
            manifest_file.write(json.dumps(manifest, indent=2) + "\n")
    except FileExistsError as exc:
        raise ValueError(f"Oracle manifest already exists: {manifest_path}") from exc
    except OSError as exc:
        raise ValueError(
            f"Could not write oracle manifest {manifest_path}: {exc}"
        ) from exc

    print(f"Wrote {artifact_path}")
    print(f"Wrote {manifest_path}")
    print(f"Oracle tensors: {expected_shape}, torch.float32")


def compute_oracle(
    base_dir: Path,
    merged_dir: Path,
    probe: Any,
    provenance: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    # Make Hugging Face offline behavior explicit before importing transformers.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM

    if not torch_module.cuda.is_available():
        raise ValueError("CUDA is required for oracle activation computation")

    hidden_size = provenance["hidden_size"]
    base_mean = compute_model_mean(
        base_dir,
        probe,
        hidden_size,
        "Base model",
        torch_module,
        AutoModelForCausalLM,
    )
    # compute_model_mean deletes the base model and clears its CUDA cache before
    # this second load, so the two full models never need to coexist on the GPU.
    ft_mean = compute_model_mean(
        merged_dir,
        probe,
        hidden_size,
        "Merged fine-tuned model",
        torch_module,
        AutoModelForCausalLM,
    )
    return prepare_stored_tensors(base_mean, ft_mean, hidden_size, torch_module)


def main() -> int:
    base_dir = Path(
        os.environ.get("BASE_MODEL_DIR", DEFAULT_BASE_MODEL_DIR)
    ).expanduser()
    merged_dir = Path(
        os.environ.get("MERGED_MODEL_DIR", DEFAULT_MERGED_MODEL_DIR)
    ).expanduser()
    probe_path = Path(
        os.environ.get("ORACLE_PROBE_PATH", DEFAULT_PROBE_PATH)
    ).expanduser()
    artifact_path = Path(
        os.environ.get("ORACLE_ARTIFACT_PATH", DEFAULT_ARTIFACT_PATH)
    ).expanduser()
    manifest_path = Path(
        os.environ.get("ORACLE_MANIFEST_PATH", MANIFEST_PATH)
    ).expanduser()

    try:
        import torch

        probe, provenance = validate_inputs(
            base_dir,
            merged_dir,
            probe_path,
            artifact_path,
            manifest_path,
            torch,
        )
        stored = compute_oracle(
            base_dir,
            merged_dir,
            probe,
            provenance,
            torch,
        )
        save_outputs(
            stored,
            provenance["hidden_size"],
            provenance,
            torch,
            artifact_path,
            manifest_path,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
