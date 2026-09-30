"""Attempt018 shared exact loss/HVP and tensor-inventory helpers.
Mechanically reused from validated Attempts 013/017; no historical checkpoint loading.
"""
import hashlib
import importlib.util
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any
import torch
from safetensors import safe_open
from safetensors.torch import save_file
HVP_SOURCE_PATH = Path(__file__).with_name('smoke_test_generic_hessian_curvature_probe.py')
_loader = importlib.util.spec_from_file_location('attempt018_true_hvp', HVP_SOURCE_PATH)
hvp_backend = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(hvp_backend)
FROZEN_CORPUS_SPEC = {'format_version': 1, 'attempt_id': '005_generic_gradient_rollback_prefix0_13_r00125', 'dataset': {'repo_id': 'science-of-finetuning/fineweb-1m-sample', 'revision': '60b53a86b84eb6559e4407b113356f56a152318f', 'split': 'train', 'streaming': False}, 'tokenizer': {'source': 'canonical_merged_checkpoint', 'directory': '/root/model-diff-scratch/models/merged'}, 'selection': {'shuffle_seed': 42, 'text_column': 'text', 'skip_blank_text': True, 'character_limit': 1280, 'add_special_tokens': True, 'minimum_token_count': 128, 'skip_valid_examples': 20000, 'sample_count': 4096, 'sequence_length': 128}, 'artifact': {'path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt', 'dtype': 'torch.int64', 'device': 'cpu', 'contiguous': True}, 'manifest': {'filename': 'corpus-manifest.json', 'timestamps': False, 'host_metadata': False, 'gpu_metadata': False}}
TOKENIZER_FILE_NAMES = ('chat_template.jinja', 'config.json', 'tokenizer.json', 'tokenizer_config.json')
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def require_sha256(value: Any, description: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
        any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"Malformed SHA-256 for {description}")
    return value

def load_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object")
    return value

def load_corpus_spec(path: Path) -> dict[str, Any]:
    value = load_json_object(path, "corpus specification")
    if value != FROZEN_CORPUS_SPEC:
        raise ValueError("Corpus specification does not match the frozen definition")
    return value

def checkpoint_file_records(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ValueError(f"Missing checkpoint directory: {root}")
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if ".cache" in relative.parts or not path.is_file():
            continue
        files.append((relative.as_posix(), path))
    records = [
        {"path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in sorted(files)
    ]
    if not records:
        raise ValueError("Checkpoint contains no files")
    return records

def tokenizer_file_records(model_dir: Path) -> list[dict[str, Any]]:
    records = []
    for name in TOKENIZER_FILE_NAMES:
        path = model_dir / name
        if not path.is_file():
            raise ValueError(f"Missing merged tokenizer file: {path}")
        records.append({
            "path": name, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return records

def verify_corpus_manifest(
    manifest_path: Path, corpus_spec_path: Path, tokens_path: Path,
    model_dir: Path, corpus_spec: dict[str, Any],
) -> dict[str, Any]:
    manifest = load_json_object(manifest_path, "corpus manifest")
    if (manifest.get("format_version") != 1 or
        manifest.get("attempt_id") != FROZEN_CORPUS_SPEC["attempt_id"] or
        manifest.get("hash_algorithm") != "sha256" or
        manifest.get("corpus_spec_sha256") != sha256_file(corpus_spec_path) or
        manifest.get("dataset") != corpus_spec["dataset"] or
        manifest.get("selection") != corpus_spec["selection"]):
        raise ValueError("Corpus manifest provenance mismatch")
    require_sha256(manifest.get("freeze_script_sha256"), "corpus freeze script")
    tokenizer = manifest.get("tokenizer_checkpoint")
    records = tokenizer_file_records(model_dir)
    if tokenizer != {
        "source": "canonical_merged_checkpoint",
        "directory": str(model_dir),
        "file_count": len(records),
        "files": records,
    }:
        raise ValueError("Corpus tokenizer checkpoint provenance mismatch")
    selection = corpus_spec["selection"]
    if manifest.get("tensor") != {
        "shape": [selection["sample_count"], selection["sequence_length"]],
        "dtype": "torch.int64", "device": "cpu", "contiguous": True,
    }:
        raise ValueError("Corpus tensor metadata mismatch")
    artifact = manifest.get("artifact")
    if not isinstance(artifact, dict) or artifact.get("path") != str(tokens_path):
        raise ValueError("Corpus artifact path mismatch")
    serialized = require_sha256(artifact.get("serialized_sha256"), "corpus artifact")
    raw = require_sha256(artifact.get("raw_tensor_sha256"), "corpus raw tensor")
    if not tokens_path.is_file() or sha256_file(tokens_path) != serialized:
        raise ValueError("Corpus artifact SHA-256 mismatch")
    return {"manifest": manifest, "serialized_sha256": serialized, "raw_tensor_sha256": raw}

def sha256_raw_int64_tensor(tensor: Any, torch_module: Any) -> str:
    if (not isinstance(tensor, torch_module.Tensor) or tensor.dtype != torch_module.int64 or
        tensor.device.type != "cpu" or not tensor.is_contiguous()):
        raise ValueError("Raw corpus hash requires contiguous CPU int64 tensor")
    canonical = tensor.detach().numpy().astype("<i8", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()

def load_corpus_tokens(
    path: Path, raw_hash: str, sample_count: int, sequence_length: int,
    torch_module: Any,
) -> Any:
    try:
        tokens = torch_module.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
        raise ValueError(f"Could not load corpus tokens: {exc}") from exc
    if (not isinstance(tokens, torch_module.Tensor) or
        tokens.dtype != torch_module.int64 or tokens.device.type != "cpu" or
        tuple(tokens.shape) != (sample_count, sequence_length) or
        not tokens.is_contiguous()):
        raise ValueError("Corpus tokens shape, dtype, device, or contiguity mismatch")
    if sha256_raw_int64_tensor(tokens, torch_module) != require_sha256(raw_hash, "corpus raw tensor"):
        raise ValueError("Corpus raw tensor SHA-256 mismatch")
    return tokens

def freeze_other_parameters(model: Any, eligible: list[tuple[str, Any]]) -> set[int]:
    selected_ids = {id(module.weight) for _, module in eligible}
    selected_names = {f"{name}.weight" for name, _ in eligible}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in selected_ids and name not in selected_names:
            raise ValueError(f"Eligible weight is shared with forbidden parameter: {name}")
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    if {id(parameter) for parameter in model.parameters() if parameter.requires_grad} != selected_ids:
        raise ValueError("Eligible parameter freeze boundary mismatch")
    model.zero_grad(set_to_none=True)
    return selected_ids

def causal_token_loss_sum(logits: Any, tokens: Any, torch_module: Any) -> Any:
    if (logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(tokens.shape) or
        logits.dtype != torch_module.float32 or not bool(torch_module.isfinite(logits).all().item())):
        raise ValueError("Model returned invalid generic logits")
    shifted_logits = logits[:, :-1, :].contiguous().reshape(-1, logits.shape[-1])
    shifted_labels = tokens[:, 1:].contiguous().reshape(-1)
    return torch_module.nn.functional.cross_entropy(
        shifted_logits, shifted_labels, reduction="sum"
    )

def validate_loaded_model(model: Any, torch_module: Any) -> None:
    config = getattr(model, "config", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if (config is None or getattr(config, "model_type", None) != "qwen3" or
        getattr(config, "num_hidden_layers", None) != 28 or
        layers is None or len(layers) != 28):
        raise ValueError("Canonical merged checkpoint must be 28-layer Qwen3")
    parameters = list(model.parameters())
    if (not parameters or any(parameter.dtype != torch_module.float32 for parameter in parameters)):
        raise ValueError("Canonical merged checkpoint parameters must be float32")

def verify_unchanged(
    model_dir: Path, source_files: list[dict[str, Any]], hashes: dict[Path, str],
) -> None:
    if checkpoint_file_records(model_dir) != source_files:
        raise ValueError("Source checkpoint changed during construction")
    for path, digest in hashes.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen input changed during construction: {path}")

def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    serialized = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise ValueError(f"Construction manifest already exists: {path}") from exc

def load_local_model(model_dir: Path, device: str, torch_module: Any) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM

    # Preserve the loader's attention implementation. Unsupported forward AD is
    # an error, never a reason to select a different backend.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch_module.float32, local_files_only=True
    )
    validate_loaded_model(model, torch_module)
    model.to(device)
    model.eval()
    return model

def model_state_hashes(model: Any, torch_module: Any) -> dict[str, str]:
    """Audit every parameter and buffer without keeping a second model copy."""
    records = {}
    for kind, items in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, tensor in items:
            digest = hashlib.sha256()
            digest.update(str((str(tensor.dtype), list(tensor.shape))).encode("utf-8"))
            flattened = tensor.detach().contiguous().reshape(-1)
            for start in range(0, flattened.numel(), 4_000_000):
                chunk = flattened[start:start + 4_000_000].to(device="cpu")
                digest.update(memoryview(chunk.view(torch_module.uint8).numpy()).cast("B"))
            records[f"{kind}:{name}"] = digest.hexdigest()
    return records

def verify_model_unchanged(model: Any, before: dict[str, str], torch_module: Any) -> None:
    if model_state_hashes(model, torch_module) != before:
        raise ValueError("Model parameters or buffers changed during linear response")

def accumulate_hessian_split(model: Any, tokens: Any, eligible: list[tuple[str, Any]],
                             current_direction: dict[str, Any], settings: dict[str, Any], torch_module: Any,
                             progress: Any = None) -> dict[str, Any]:
    """Sum exact batch Hessian actions on the same detached displacement direction."""
    count, length, batch_size = settings["sample_count"], settings["sequence_length"], settings["batch_size"]
    denominator = settings["total_prediction_tokens"]
    if (tuple(tokens.shape) != (count, length) or count % batch_size or
        denominator != count * (length - 1) or settings["predictions_per_sample"] != length - 1 or
        settings["number_of_batches"] != count // batch_size):
        raise ValueError("Split Hessian loss normalization mismatch")
    parameters = {f"{name}.weight": module.weight for name, module in eligible}
    hvp_backend.validate_tensor_structure(parameters, current_direction, detached=True, description="Detached direction")
    device = next(iter(parameters.values())).device
    result = {name: torch_module.zeros_like(parameter) for name, parameter in parameters.items()}
    model.eval()
    model.zero_grad(set_to_none=True)
    product = None
    try:
        with torch_module.autocast(device_type=device.type, enabled=False):
            for index, start in enumerate(range(0, count, batch_size), 1):
                batch = tokens[start:start + batch_size].to(device)
                def loss():
                    logits = model(input_ids=batch, use_cache=False).logits
                    return causal_token_loss_sum(logits, batch, torch_module) / denominator
                product = hvp_backend.true_hessian_vector_product(loss, parameters, current_direction)
                with torch_module.no_grad():
                    for name in result:
                        result[name].add_(product[name])
                product.clear()
                product = None
                if any(parameter.grad is not None for parameter in model.parameters()):
                    raise ValueError("HVP unexpectedly populated a parameter gradient")
                if progress is not None:
                    progress(index, count // batch_size)
        hvp_backend.validate_tensor_structure(parameters, result, detached=True, description="Split Hessian action")
        return result
    finally:
        if product is not None:
            product.clear()
        model.zero_grad(set_to_none=True)

def discover(model, spec):
    layers = model.model.layers
    if len(layers) != 28:
        raise ValueError('Expected 28 blocks')
    found = []
    for block, layer in enumerate(layers):
        local = [(f'model.layers.{block}.{name}', module) for name, module in layer.named_modules()
                 if isinstance(module, torch.nn.Linear)]
        if len(local) != 7:
            raise ValueError('Expected seven Linear weights per block')
        found.extend(local)
    found.sort(key=lambda item: item[0])
    inventory = [{'name': name+'.weight', 'shape': list(module.weight.shape)} for name, module in found]
    if inventory != spec['coordinates'] or len({id(m.weight) for _, m in found}) != 196:
        raise ValueError('Exact 196 coordinate inventory mismatch')
    for name, module in found:
        if module.weight.dtype != torch.float32 or not module.weight.is_contiguous() or not torch.isfinite(module.weight).all():
            raise ValueError('Invalid selected weight: '+name)
    freeze_other_parameters(model, found)
    return found

def block_number(name):
    return int(name.split('.')[2])

def save_vector(path, vector, coordinates):
    if path.exists() or path.is_symlink():
        raise ValueError('Vector output already exists')
    if set(vector) != {r['name'] for r in coordinates}:
        raise ValueError('Vector names mismatch')
    for row in coordinates:
        value = vector[row['name']]
        if (value.device.type != 'cpu' or value.dtype != torch.float32 or not value.is_contiguous()
                or list(value.shape) != row['shape'] or not torch.isfinite(value).all()):
            raise ValueError('Invalid stored vector')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb'):
        pass
    save_file(vector, str(path))
    return vector_record(path, coordinates)

def vector_record(path, coordinates):
    rows = []
    with safe_open(str(path), framework='pt', device='cpu') as stored:
        if sorted(stored.keys()) != [r['name'] for r in coordinates]:
            raise ValueError('Stored vector inventory mismatch')
        for row in coordinates:
            value = stored.get_tensor(row['name'])
            if value.dtype != torch.float32 or list(value.shape) != row['shape'] or not torch.isfinite(value).all():
                raise ValueError('Invalid stored vector tensor')
            rows.append({**row, 'raw_sha256': sha256_raw_float32_tensor(value, torch),
                         'squared_norm': float(value.double().square().sum())})
    return {'serialized_sha256': sha256_file(path), 'matrices': rows,
            'norm': math.sqrt(math.fsum(r['squared_norm'] for r in rows))}

def metric_record(pairs):
    hh, dd, hd, error = [], [], [], []
    # A second streaming pass computes the scaled residual directly, avoiding cancellation.
    for h, d in pairs():
        h=h.detach().to(device='cpu', dtype=torch.float64); d=d.double()
        hh.append(float(h.square().sum())); dd.append(float(d.square().sum()))
        hd.append(float((h*d).sum())); error.append(float((d-h).square().sum()))
    H,D,C = math.fsum(hh), math.fsum(dd), math.fsum(hd)
    coefficient = C/H if H else None
    scaled = []
    if coefficient is not None and D:
        for h,d in pairs():
            scaled.append(float((d.double()-coefficient*h.detach().to(device='cpu',dtype=torch.float64)).square().sum()))
    return {'cosine': C/math.sqrt(H*D) if H and D else None, 'h_norm': math.sqrt(H), 'd_norm': math.sqrt(D),
            'norm_ratio': math.sqrt(H/D) if D else None,
            'unscaled_relative_residual': math.sqrt(math.fsum(error)/D) if D else None,
            'optimal_scalar': coefficient,
            'relative_residual_after_optimal_rescaling': math.sqrt(math.fsum(scaled)/D) if D and coefficient is not None else None}

def sha256_raw_float32_tensor(value: Any, torch_module: Any) -> str:
    if (not isinstance(value, torch_module.Tensor) or value.dtype != torch_module.float32 or
        value.device.type != "cpu" or not value.is_contiguous()):
        raise ValueError("Raw activation hash requires contiguous CPU float32 tensor")
    return hashlib.sha256(value.detach().numpy().astype("<f4", copy=False).tobytes(order="C")).hexdigest()
