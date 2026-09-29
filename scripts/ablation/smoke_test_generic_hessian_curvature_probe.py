#!/usr/bin/env python3
"""Attempt 010 first stage: one-batch true-Hessian smoke test; stdout only.

Run with the existing model attention backend. There is no curvature fallback.
This script does not construct a full-corpus result or write any files.
"""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT_ID = "010_generic_hessian_curvature_probe_prefix0_13"
CORPUS_DIRECTORY = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125"
DEFAULT_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_TOKENS_PATH = Path("/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt")
TOKENIZER_FILE_NAMES = ("chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json")

FROZEN_CORPUS_SPEC = {'format_version': 1,
 'attempt_id': '005_generic_gradient_rollback_prefix0_13_r00125',
 'dataset': {'repo_id': 'science-of-finetuning/fineweb-1m-sample',
             'revision': '60b53a86b84eb6559e4407b113356f56a152318f',
             'split': 'train',
             'streaming': False},
 'tokenizer': {'source': 'canonical_merged_checkpoint',
               'directory': '/root/model-diff-scratch/models/merged'},
 'selection': {'shuffle_seed': 42,
               'text_column': 'text',
               'skip_blank_text': True,
               'character_limit': 1280,
               'add_special_tokens': True,
               'minimum_token_count': 128,
               'skip_valid_examples': 20000,
               'sample_count': 4096,
               'sequence_length': 128},
 'artifact': {'path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
              'dtype': 'torch.int64',
              'device': 'cpu',
              'contiguous': True},
 'manifest': {'filename': 'corpus-manifest.json',
              'timestamps': False,
              'host_metadata': False,
              'gpu_metadata': False}}

FROZEN_CORPUS_HASHES = {'corpus_spec.json': 'e514820e7def99e9287c32de4b1279f28f6ff6c2e09749a8dfa5d8b14c311f0f',
 'corpus-manifest.json': '8b3ea49228dbb106c2b8a7c849c4f27e021fa3e992800444f43e260a6b055230'}

CANONICAL_CHECKPOINT_FILES = [{'path': 'chat_template.jinja',
  'size_bytes': 4168,
  'sha256': 'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'},
 {'path': 'config.json',
  'size_bytes': 1416,
  'sha256': '3ef26f3c99bbc0bb0ef65b729429f716fb2760a23b71ffd2eee39646f703bd0f'},
 {'path': 'generation_config.json',
  'size_bytes': 214,
  'sha256': '893b0dccf83626cdfdc498e5b2a255a935b4c3c693350aed18f7003c1a3f3de9'},
 {'path': 'model.safetensors',
  'size_bytes': 6882335328,
  'sha256': 'f6989dd3445cd339041ab654a6da463e0a80b2d52dd2c01f67f683b1c30ea10f'},
 {'path': 'tokenizer.json',
  'size_bytes': 11422650,
  'sha256': 'be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506'},
 {'path': 'tokenizer_config.json',
  'size_bytes': 692,
  'sha256': '1cc816812993bff176eb4f7495433b736f06fba9b6e7b05cac7b4a1780650c95'}]

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


def discover_eligible_linear_weights(model: Any, torch_module: Any) -> list[tuple[str, Any]]:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != 28:
        raise ValueError("Merged checkpoint must expose 28 transformer blocks")
    found = []
    seen = set()
    for block_index in range(14):
        for relative_name, module in layers[block_index].named_modules():
            if not isinstance(module, torch_module.nn.Linear):
                continue
            name = f"model.layers.{block_index}.{relative_name}"
            parameter = module.weight
            if (id(parameter) in seen or parameter.ndim != 2 or
                parameter.dtype != torch_module.float32 or not parameter.is_contiguous()):
                raise ValueError(f"Malformed or shared eligible weight: {name}")
            if not bool(torch_module.isfinite(parameter).all().item()):
                raise ValueError(f"Eligible weight contains non-finite values: {name}")
            found.append((name, module))
            seen.add(id(parameter))
    found.sort(key=lambda item: item[0])
    if len(found) != 98 or len({name for name, _ in found}) != 98:
        raise ValueError(f"Expected exactly 98 eligible matrices, found {len(found)}")
    return found


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


def sha256_parameter(tensor: Any, torch_module: Any) -> str:
    if tensor.dtype != torch_module.float32:
        raise ValueError("Model parameter is not float32")
    flattened = tensor.detach().contiguous().view(-1)
    digest = hashlib.sha256()
    for start in range(0, flattened.numel(), 4_000_000):
        chunk = flattened[start:start + 4_000_000].to(device="cpu")
        canonical = chunk.numpy().astype("<f4", copy=False)
        digest.update(memoryview(canonical).cast("B"))
    return digest.hexdigest()


def frozen_parameter_hashes(model: Any, selected_ids: set[int], torch_module: Any) -> dict[str, str]:
    return {
        name: sha256_parameter(parameter, torch_module)
        for name, parameter in model.named_parameters()
        if id(parameter) not in selected_ids
    }


def verify_frozen_parameters(
    model: Any, selected_ids: set[int], before: dict[str, str], torch_module: Any,
) -> None:
    current_names = {
        name for name, parameter in model.named_parameters()
        if id(parameter) not in selected_ids
    }
    if current_names != set(before):
        raise ValueError("Forbidden parameter set changed")
    for name, parameter in model.named_parameters():
        if id(parameter) in selected_ids:
            continue
        if (parameter.requires_grad or parameter.grad is not None or
            sha256_parameter(parameter, torch_module) != before.get(name)):
            raise ValueError(f"Forbidden parameter changed or received gradient: {name}")


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


def validate_tensor_structure(parameters: dict[str, Any], values: dict[str, Any], *,
                              detached: bool, description: str) -> None:
    if (not isinstance(parameters, dict) or not parameters or not isinstance(values, dict) or
        list(parameters) != list(values)):
        raise ValueError(f"{description} must have exactly matching ordered parameter keys")
    if len({id(parameter) for parameter in parameters.values()}) != len(parameters):
        raise ValueError("Selected parameters must be distinct")
    for name, parameter in parameters.items():
        value = values[name]
        if (not isinstance(parameter, torch.Tensor) or not parameter.is_leaf or
            not parameter.requires_grad or parameter.dtype not in (torch.float32, torch.float64)):
            raise ValueError(f"Invalid selected parameter: {name}")
        if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape or
            value.dtype != parameter.dtype or value.device != parameter.device):
            raise ValueError(f"{description} shape/dtype/device mismatch: {name}")
        if detached and (value.requires_grad or value.grad_fn is not None):
            raise ValueError(f"{description} must be detached: {name}")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite {description}: {name}")


def true_hessian_vector_product(loss_function: Callable[[], Any], parameters: dict[str, Any],
                                direction: dict[str, Any], *,
                                diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return detached Hv = grad_W(sum(grad_W(loss) * detached_v)).

    The closure is called once. No .grad buffers are populated. A parameter with
    a constant first derivative has a mathematically zero Hessian contribution.
    Kernel/autograd exceptions propagate unchanged, without another backend or method.
    """
    validate_tensor_structure(parameters, direction, detached=True, description="Direction")
    selected = tuple(parameters.values())
    loss = gradients = directional = products = None
    try:
        with torch.enable_grad():
            loss = loss_function()
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError("Loss must be a finite scalar tensor")
            gradients = torch.autograd.grad(loss, selected, create_graph=True, allow_unused=False)
            validate_tensor_structure(parameters, dict(zip(parameters, gradients)),
                                      detached=False, description="First derivative")
            directional = sum((gradient * direction[name]).sum()
                              for name, gradient in zip(parameters, gradients))
            if not bool(torch.isfinite(directional)):
                raise ValueError("Non-finite directional derivative")
            if diagnostics is not None:
                diagnostics.update(second_order_loss=float(loss.detach()),
                                   first_derivative_tensor_count=len(gradients),
                                   directional_derivative=float(directional.detach()))
            if directional.requires_grad:
                products = torch.autograd.grad(directional, selected, create_graph=False,
                                               retain_graph=False, allow_unused=True)
            else:
                products = (None,) * len(selected)
        result = {name: value.detach() if value is not None else torch.zeros_like(parameter)
                  for (name, parameter), value in zip(parameters.items(), products)}
        validate_tensor_structure(parameters, result, detached=True, description="HVP")
        return result
    finally:
        # Do not retain the large second-order graph in an exception frame.
        loss = gradients = directional = products = None


def aggregate_norm(values: dict[str, Any]) -> float:
    # No flattened full-parameter copy: only one matrix at a time in CPU FP64.
    squares = []
    for value in values.values():
        matrix = value.detach().to(device="cpu", dtype=torch.float64)
        squares.append(float(matrix.square().sum()))
    result = math.sqrt(math.fsum(squares))
    if not math.isfinite(result):
        raise ValueError("Aggregate norm is non-finite")
    return result


def smoke_batch_loss(model: Any, batch: Any) -> Any:
    logits = model(input_ids=batch, use_cache=False).logits
    return causal_token_loss_sum(logits, batch, torch) / (batch.shape[0] * (batch.shape[1] - 1))


def run_hvp_smoke(model: Any, batch: Any, eligible: list[tuple[str, Any]],
                  report: dict[str, Any]) -> None:
    """Two loss forwards on the identical batch; time only the second-order path."""
    if (tuple(batch.shape) != (8, 128) or batch.dtype != torch.int64 or len(eligible) != 98):
        raise ValueError("Smoke test requires 98 weights and one 8 x 128 int64 batch")
    parameters = {f"{name}.weight": module.weight for name, module in eligible}
    if len(parameters) != 98 or any(parameter.dtype != torch.float32 for parameter in parameters.values()):
        raise ValueError("Smoke selected weights must be 98 distinct FP32 tensors")
    device = next(iter(parameters.values())).device
    if batch.device != device:
        raise ValueError("Smoke batch and selected parameters must share a device")
    model.eval()
    selected_ids = freeze_other_parameters(model, eligible)
    before = model_state_hashes(model, torch)
    frozen_before = frozen_parameter_hashes(model, selected_ids, torch)
    direction = products = loss = gradients = None
    try:
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            report["stage"] = "ordinary_batch_gradient"
            loss = smoke_batch_loss(model, batch)
            report["ordinary_batch_loss"] = float(loss.detach())
            gradients = torch.autograd.grad(loss, tuple(parameters.values()), create_graph=False,
                                            retain_graph=False, allow_unused=False)
            direction = {name: gradient.detach() for name, gradient in zip(parameters, gradients)}
            del loss, gradients
            loss = gradients = None
            model.zero_grad(set_to_none=True)
            validate_tensor_structure(parameters, direction, detached=True, description="Ordinary gradient")
            report["gradient_tensor_count"] = len(direction)
            report["aggregate_gradient_norm"] = aggregate_norm(direction)
            if report["aggregate_gradient_norm"] <= 0:
                raise ValueError("Ordinary gradient norm must be positive")
            report["stage"] = "true_hessian_vector_product"
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            try:
                products = true_hessian_vector_product(
                    lambda: smoke_batch_loss(model, batch), parameters, direction, diagnostics=report)
            finally:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                report["elapsed_second_order_seconds"] = time.perf_counter() - start
                report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
                report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            report["hvp_tensor_count"] = len(products)
            report["aggregate_hvp_norm"] = aggregate_norm(products)
            if report["aggregate_hvp_norm"] <= 0:
                raise ValueError("HVP norm must be positive")
            if report["second_order_loss"] != report["ordinary_batch_loss"]:
                raise ValueError("Recomputed batch loss differs from the ordinary batch loss")
            report["stage"] = "immutability_audit"
            verify_frozen_parameters(model, selected_ids, frozen_before, torch)
            verify_model_unchanged(model, before, torch)
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise ValueError("Unexpected accumulated parameter gradients")
            report["parameters_unchanged"] = True
            report["frozen_parameters_unchanged"] = True
    finally:
        model.zero_grad(set_to_none=True)
        if direction is not None:
            direction.clear()
        if products is not None:
            products.clear()
        direction = products = loss = gradients = None


def load_local_model(model_dir: Path) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, local_files_only=True)
    validate_loaded_model(model, torch)
    model.to("cuda")
    model.eval()
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--tokens-path", type=Path, default=DEFAULT_TOKENS_PATH)
    parser.add_argument("--corpus-spec-path", type=Path, default=CORPUS_DIRECTORY / "corpus_spec.json")
    parser.add_argument("--corpus-manifest-path", type=Path, default=CORPUS_DIRECTORY / "corpus-manifest.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key, value in vars(args).items():
        setattr(args, key, value.expanduser())
    report = {"attempt_id": ATTEMPT_ID, "stage": "input_validation", "status": "failed"}
    try:
        hashes = {path: sha256_file(path) for path in
                  (args.corpus_spec_path, args.corpus_manifest_path, args.tokens_path, Path(__file__).resolve())}
        for filename, path in (("corpus_spec.json", args.corpus_spec_path),
                               ("corpus-manifest.json", args.corpus_manifest_path)):
            if hashes[path] != FROZEN_CORPUS_HASHES[filename]:
                raise ValueError(f"Frozen corpus provenance mismatch: {filename}")
        corpus_spec = load_json_object(args.corpus_spec_path, "corpus specification")
        if corpus_spec != FROZEN_CORPUS_SPEC:
            raise ValueError("Frozen corpus specification mismatch")
        source_files = checkpoint_file_records(args.merged_model_dir)
        if source_files != CANONICAL_CHECKPOINT_FILES:
            raise ValueError("Canonical checkpoint inventory/hash mismatch")
        corpus = verify_corpus_manifest(args.corpus_manifest_path, args.corpus_spec_path,
                                       args.tokens_path, args.merged_model_dir, corpus_spec)
        tokens = load_corpus_tokens(args.tokens_path, corpus["raw_tensor_sha256"], 4096, 128, torch)
        batch = tokens[:8].clone()
        del tokens
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real Qwen3 HVP smoke test")
        report["stage"] = "model_loading"
        model = load_local_model(args.merged_model_dir)
        report["attention_implementation"] = getattr(model.config, "_attn_implementation", None)
        eligible = discover_eligible_linear_weights(model, torch)
        run_hvp_smoke(model, batch.to(next(model.parameters()).device), eligible, report)
        verify_unchanged(args.merged_model_dir, source_files, hashes)
        report.update(stage="complete", status="passed")
    except Exception as exc:
        report.update(exception_type=type(exc).__name__, exception=str(exc))
        # Preserve the exact kernel/autograd operation and traceback; never retry.
        traceback.print_exc(file=sys.stderr)
        traceback.clear_frames(exc.__traceback__)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
