#!/usr/bin/env python3
"""Construct Attempt 007 by square-root diagonal empirical Fisher rollback."""

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
CORPUS_DIRECTORY = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125"
ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/007_diag_empirical_fisher_sqrt_rollback_prefix0_13_r00125"
DEFAULT_SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
DEFAULT_CORPUS_SPEC_PATH = CORPUS_DIRECTORY / "corpus_spec.json"
DEFAULT_CORPUS_MANIFEST_PATH = CORPUS_DIRECTORY / "corpus-manifest.json"
DEFAULT_CONSTRUCTION_MANIFEST_PATH = ATTEMPT_DIRECTORY / "construction-manifest.json"
DEFAULT_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_TOKENS_PATH = Path(
    "/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt"
)
DEFAULT_OUTPUT_MODEL_DIR = Path(
    "/root/model-diff-scratch/ablations/007_diag_empirical_fisher_sqrt_rollback_prefix0_13_r00125/model"
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
        "split": "train", "streaming": False,
    },
    "tokenizer": {
        "source": "canonical_merged_checkpoint",
        "directory": "/root/model-diff-scratch/models/merged",
    },
    "selection": {
        "shuffle_seed": 42, "text_column": "text", "skip_blank_text": True,
        "character_limit": 1280, "add_special_tokens": True,
        "minimum_token_count": 128, "skip_valid_examples": 20_000,
        "sample_count": 4_096, "sequence_length": 128,
    },
    "artifact": {
        "path": str(DEFAULT_TOKENS_PATH), "dtype": "torch.int64",
        "device": "cpu", "contiguous": True,
    },
    "manifest": {
        "filename": "corpus-manifest.json", "timestamps": False,
        "host_metadata": False, "gpu_metadata": False,
    },
}

FROZEN_SPEC = {
    "format_version": 1,
    "attempt_id": "007_diag_empirical_fisher_sqrt_rollback_prefix0_13_r00125",
    "method": "diag_empirical_fisher_sqrt_rollback",
    "information_policy": "final_checkpoint_plus_frozen_generic_text",
    "defaults": {
        "merged_model_directory": str(DEFAULT_MODEL_DIR),
        "corpus_spec_path": "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json",
        "corpus_manifest_path": "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json",
        "corpus_artifact_path": str(DEFAULT_TOKENS_PATH),
        "output_model_directory": str(DEFAULT_OUTPUT_MODEL_DIR),
        "construction_manifest_path": "experiments/attempts/007_diag_empirical_fisher_sqrt_rollback_prefix0_13_r00125/construction-manifest.json",
    },
    "model": {
        "model_type": "qwen3", "num_hidden_layers": 28,
        "source_dtype": "float32", "local_files_only": True,
        "eval_mode": True, "use_cache": False,
    },
    "eligible_tensors": {
        "transformer_block_indices": list(range(14)),
        "recursive_module_type": "torch.nn.Linear", "parameter": "weight",
        "expected_matrix_count": 98,
        "module_ordering": "lexicographic_by_module_name",
        "freeze_every_other_parameter": True,
    },
    "generic_loss": {
        "type": "causal_next_token_cross_entropy",
        "sample_count": 4_096, "sequence_length": 128,
        "predictions_per_sample": 127,
        "total_prediction_tokens": 4_096 * 127,
        "batch_size": 1, "number_of_batches": 4096,
        "label_alignment": "logits_positions_0_through_126_predict_tokens_1_through_127",
        "reduction": "per_sequence_sum_token_cross_entropy_divided_by_127",
        "masking": "causal_model_mask_only", "mixed_precision": False,
        "dropout": False, "optimizer": False, "weight_decay": False,
        "gradient_clipping": False,
    },
    "empirical_fisher": {
        "first_moment": "g = mean_s(g_s)",
        "second_moment": "F = mean_s(g_s ** 2)",
        "sequence_gradient": "gradient_of_mean_next_token_loss_per_sequence",
        "zero_grad_between_sequences": True,
        "accumulation_dtype": "float32",
        "accumulation_device": "same_as_eligible_weights",
        "final_accounting_dtype": "cpu_float64",
        "damping": "lambda = sum_all_coordinates(F) / total_coordinate_count",
        "direction": "d = g / sqrt(F + lambda)",
        "save_intermediate_tensors": False,
    },
    "rollback": {
        "direction": "W_new = W - alpha * (g / sqrt(F + lambda))",
        "target_relative_frobenius": 0.00125,
        "normalization": "one_global_alpha_across_all_eligible_matrices",
        "norm_accounting_dtype": "cpu_float64",
        "update_arithmetic_dtype": "cpu_float64",
        "stored_dtype": "float32",
    },
    "outputs": {
        "standalone_checkpoint": True, "checkpoint_dtype": "float32",
        "timestamps": False, "host_metadata": False, "gpu_metadata": False,
    },
}


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


def load_spec(path: Path) -> dict[str, Any]:
    value = load_json_object(path, "attempt specification")
    if value != FROZEN_SPEC:
        raise ValueError("Attempt specification does not match the frozen definition")
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


def accumulate_empirical_fisher(
    model: Any, tokens: Any, eligible: list[tuple[str, Any]],
    loss_spec: dict[str, Any], torch_module: Any,
) -> tuple[float, dict[str, tuple[Any, Any]]]:
    """Accumulate FP32 moments on the weight device (CUDA for CUDA models)."""
    sample_count = loss_spec["sample_count"]
    sequence_length = loss_spec["sequence_length"]
    total_tokens = loss_spec["total_prediction_tokens"]
    if (sample_count <= 0 or sequence_length < 2 or not eligible or
        tuple(tokens.shape) != (sample_count, sequence_length) or
        loss_spec["batch_size"] != 1 or
        loss_spec["number_of_batches"] != sample_count or
        total_tokens != sample_count * (sequence_length - 1) or
        loss_spec["predictions_per_sample"] != sequence_length - 1):
        raise ValueError("Frozen gradient averaging dimensions mismatch")
    selected_ids = {id(module.weight) for _, module in eligible}
    moments = {
        name: (torch_module.zeros_like(module.weight, dtype=torch_module.float32),
               torch_module.zeros_like(module.weight, dtype=torch_module.float32))
        for name, module in eligible
    }
    model.eval()
    device = eligible[0][1].weight.device
    losses = []
    with torch_module.enable_grad(), torch_module.autocast(device_type=device.type, enabled=False):
        for index in range(sample_count):
            model.zero_grad(set_to_none=True)
            batch = tokens[index:index + 1].to(device=device)
            logits = model(input_ids=batch, use_cache=False).logits
            loss = causal_token_loss_sum(logits, batch, torch_module) / (sequence_length - 1)
            if not bool(torch_module.isfinite(loss).item()):
                raise ValueError("Generic loss is non-finite")
            loss.backward()
            losses.append(float(loss.detach().item()))
            for name, parameter in model.named_parameters():
                if id(parameter) not in selected_ids and parameter.grad is not None:
                    raise ValueError(f"Forbidden parameter received gradient: {name}")
            for name, module in eligible:
                gradient = module.weight.grad
                if (gradient is None or gradient.dtype != torch_module.float32 or
                    gradient.shape != module.weight.shape):
                    raise ValueError(f"Missing or malformed eligible gradient: {name}")
                gradient = gradient.detach()
                first, second = moments[name]
                first.add_(gradient)
                second.addcmul_(gradient, gradient)
                module.weight.grad = None
                del gradient
            del logits, loss
            if (index + 1) % 128 == 0:
                print(f"Processed {index + 1}/{sample_count} sequences", flush=True)
    model.zero_grad(set_to_none=True)
    for name, (first, second) in moments.items():
        first.div_(sample_count)
        second.div_(sample_count)
        # Non-finite gradients propagate into these sums. Check once at the
        # accumulation boundary, before any damping or direction calculation.
        if (not bool(torch_module.isfinite(first).all().item()) or
            not bool(torch_module.isfinite(second).all().item())):
            raise ValueError(f"Non-finite accumulated first/Fisher moments: {name}")
    mean_loss = math.fsum(losses) / sample_count
    if not math.isfinite(mean_loss):
        raise ValueError("Mean generic loss is non-finite")
    return mean_loss, moments


def validate_moment(value: Any, name: str, torch_module: Any, *, fisher: bool = False) -> None:
    if (not isinstance(value, torch_module.Tensor) or value.dtype != torch_module.float64 or
        value.device.type != "cpu" or value.numel() == 0 or
        not bool(torch_module.isfinite(value).all().item()) or
        (fisher and bool((value < 0).any().item()))):
        raise ValueError(f"Invalid or non-finite {name}")


def moment_to_cpu64(value: Any, torch_module: Any) -> Any:
    """Transfer only the current matrix; never retain a full CPU moment copy."""
    if (not isinstance(value, torch_module.Tensor) or
        value.dtype not in (torch_module.float32, torch_module.float64)):
        raise ValueError("Invalid moment tensor or dtype")
    return value.detach().to(device="cpu", dtype=torch_module.float64)


def global_fisher_statistics(
    moments: dict[str, tuple[Any, Any]], torch_module: Any,
) -> dict[str, Any]:
    sums = []
    count = 0
    for name in sorted(moments):
        fisher = moment_to_cpu64(moments[name][1], torch_module)
        validate_moment(fisher, f"Fisher: {name}", torch_module, fisher=True)
        sums.append(float(fisher.sum().item()))
        count += fisher.numel()
        del fisher
    try:
        total = math.fsum(sums)
        damping = total / count if count else 0.0
    except OverflowError as exc:
        raise ValueError("Invalid or non-finite Fisher lambda") from exc
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("Invalid or non-finite Fisher lambda")
    return {
        "eligible_scalar_coordinate_count": count,
        "empirical_fisher_total_sum": total,
        "empirical_fisher_global_scalar_mean": damping,
        "lambda": damping,
    }


def global_fisher_lambda(moments: dict[str, tuple[Any, Any]], torch_module: Any) -> float:
    return global_fisher_statistics(moments, torch_module)["lambda"]


def preconditioned_direction(first: Any, fisher: Any, damping: float, torch_module: Any) -> Any:
    validate_moment(first, "gradient", torch_module)
    validate_moment(fisher, "Fisher", torch_module, fisher=True)
    if first.shape != fisher.shape or not math.isfinite(damping) or damping <= 0:
        raise ValueError("Invalid Fisher shape or lambda")
    denominator = fisher + damping
    if not bool(torch_module.isfinite(denominator).all().item()):
        raise ValueError("Non-finite direction denominator")
    direction = first / denominator.sqrt()
    if not bool(torch_module.isfinite(direction).all().item()):
        raise ValueError("Non-finite preconditioned direction")
    return direction


def global_rollback_scale(
    eligible: list[tuple[str, Any]], moments: dict[str, tuple[Any, Any]],
    target_relative: float, torch_module: Any,
) -> dict[str, float]:
    if not math.isfinite(target_relative) or target_relative <= 0:
        raise ValueError("Invalid target relative perturbation")
    if set(moments) != {name for name, _ in eligible}:
        raise ValueError("Moment matrix set mismatch")
    fisher_statistics = global_fisher_statistics(moments, torch_module)
    damping = fisher_statistics["lambda"]
    gradient_squares = []
    weight_squares = []
    direction_squares = []
    for name, module in sorted(eligible, key=lambda item: item[0]):
        weight = module.weight.detach().to(device="cpu", dtype=torch_module.float64)
        first64, fisher64 = (moment_to_cpu64(value, torch_module) for value in moments[name])
        direction64 = preconditioned_direction(first64, fisher64, damping, torch_module)
        if direction64.shape != weight.shape:
            raise ValueError(f"Direction shape mismatch: {name}")
        weight_squares.append(float(torch_module.sum(weight.square()).item()))
        direction_squares.append(float(torch_module.sum(direction64.square()).item()))
        gradient_squares.append(float(first64.square().sum().item()))
        del weight, direction64, first64, fisher64
    try:
        gradient_norm = math.sqrt(math.fsum(gradient_squares))
        weight_norm = math.sqrt(math.fsum(weight_squares))
        direction_norm = math.sqrt(math.fsum(direction_squares))
    except OverflowError as exc:
        raise ValueError("Global norm is non-finite") from exc
    if (not math.isfinite(gradient_norm) or
        not math.isfinite(weight_norm) or weight_norm <= 0 or
        not math.isfinite(direction_norm) or direction_norm <= 0):
        raise ValueError("Global weight or direction norm is zero or non-finite")
    target_delta_norm = target_relative * weight_norm
    alpha = target_delta_norm / direction_norm
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Global rollback alpha is non-finite")
    return {
        **fisher_statistics,
        "aggregate_mean_gradient_frobenius_norm": gradient_norm,
        "aggregate_source_weight_norm": weight_norm,
        "aggregate_preconditioned_direction_norm": direction_norm,
        "target_delta_norm": target_delta_norm,
        "alpha": alpha,
    }


def apply_global_rollback(
    eligible: list[tuple[str, Any]], moments: dict[str, tuple[Any, Any]],
    damping: float, alpha: float, weight_norm: float,
    torch_module: Any,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Apply one global scale, consuming moments as each matrix is finished."""
    if any(not math.isfinite(v) or v <= 0 for v in (damping, alpha, weight_norm)):
        raise ValueError("Invalid lambda, alpha, or weight norm")
    records = []
    delta_squares = []
    with torch_module.no_grad():
        for name, module in sorted(eligible, key=lambda item: item[0]):
            source64 = module.weight.detach().to(device="cpu", dtype=torch_module.float64)
            first64, fisher64 = (moment_to_cpu64(value, torch_module) for value in moments[name])
            direction64 = preconditioned_direction(first64, fisher64, damping, torch_module)
            if direction64.shape != source64.shape:
                raise ValueError(f"Direction shape mismatch: {name}")
            updated32 = (source64 - alpha * direction64).to(dtype=torch_module.float32).contiguous()
            if not bool(torch_module.isfinite(updated32).all().item()):
                raise ValueError(f"Rollback produced non-finite weight: {name}")
            module.weight.copy_(updated32.to(device=module.weight.device))
            actual32 = module.weight.detach().to(device="cpu", dtype=torch_module.float32)
            actual_delta64 = actual32.to(dtype=torch_module.float64) - source64
            source_norm = float(torch_module.linalg.vector_norm(source64).item())
            direction_norm = float(torch_module.linalg.vector_norm(direction64).item())
            delta_norm = float(torch_module.linalg.vector_norm(actual_delta64).item())
            changed = int(torch_module.count_nonzero(actual32 != source64.to(dtype=torch_module.float32)).item())
            delta_squares.append(delta_norm * delta_norm)
            records.append({
                "name": name,
                "shape": list(actual32.shape),
                "mean_gradient_frobenius_norm": float(torch_module.linalg.vector_norm(first64).item()),
                "fisher_scalar_mean": float(fisher64.mean().item()),
                "fisher_scalar_minimum": float(fisher64.min().item()),
                "fisher_scalar_maximum": float(fisher64.max().item()),
                "source_frobenius_norm": source_norm,
                "preconditioned_direction_frobenius_norm": direction_norm,
                "realized_fp32_delta_frobenius_norm": delta_norm,
                "realized_relative_frobenius_perturbation": (
                    None if source_norm == 0 else delta_norm / source_norm
                ),
                "max_absolute_weight_change": float(actual_delta64.abs().max().item()),
                "changed_element_count": changed,
                "changed_element_fraction": changed / actual32.numel(),
            })
            del moments[name]
            del source64, direction64, updated32, actual32, actual_delta64, first64, fisher64
    realized_norm = math.sqrt(math.fsum(delta_squares))
    return records, {
        "aggregate_realized_fp32_delta_norm": realized_norm,
        "aggregate_realized_relative_perturbation": realized_norm / weight_norm,
    }


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


def load_local_model(model_dir: Path, device: str, torch_module: Any) -> tuple[Any, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch_module.float32, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    validate_loaded_model(model, torch_module)
    model.to(device)
    model.eval()
    return model, tokenizer


def require_output_absent(output_dir: Path, manifest_path: Path) -> None:
    if output_dir.exists():
        raise ValueError(f"Output checkpoint already exists: {output_dir}")
    if manifest_path.exists():
        raise ValueError(f"Construction manifest already exists: {manifest_path}")
    if not manifest_path.parent.is_dir():
        raise ValueError(f"Missing construction manifest directory: {manifest_path.parent}")


def verify_unchanged(
    model_dir: Path, source_files: list[dict[str, Any]], hashes: dict[Path, str],
) -> None:
    if checkpoint_file_records(model_dir) != source_files:
        raise ValueError("Source checkpoint changed during construction")
    for path, digest in hashes.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen input changed during construction: {path}")


def build_manifest(
    spec: dict[str, Any], hashes: dict[str, str], source_files: list[dict[str, Any]],
    output_files: list[dict[str, Any]], corpus_context: dict[str, Any],
    mean_loss: float, scale: dict[str, float],
    matrix_records: list[dict[str, Any]], realized: dict[str, float],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "attempt_id": spec["attempt_id"],
        "hash_algorithm": "sha256",
        "information_policy": spec["information_policy"],
        "attempt_spec_sha256": hashes["attempt_spec"],
        "corpus_spec_sha256": hashes["corpus_spec"],
        "corpus_manifest_sha256": hashes["corpus_manifest"],
        "constructor_script_sha256": hashes["constructor_script"],
        "corpus": {
            "serialized_sha256": corpus_context["serialized_sha256"],
            "raw_tensor_sha256": corpus_context["raw_tensor_sha256"],
        },
        "source_checkpoint": {
            "file_count": len(source_files),
            "total_bytes": sum(item["size_bytes"] for item in source_files),
            "files": source_files,
        },
        "loss": {**spec["generic_loss"], "mean_generic_loss": mean_loss},
        "empirical_fisher": spec["empirical_fisher"],
        "gradient_and_rollback": {
            "eligible_matrix_count": len(matrix_records),
            "target_relative_frobenius": spec["rollback"]["target_relative_frobenius"],
            **scale,
            **realized,
        },
        "deterministic_ordering": {
            "rule": "lexicographic_by_module_name",
            "modified_module_names": [item["name"] for item in matrix_records],
        },
        "modified_matrices": matrix_records,
        "output_checkpoint": {
            "directory_name": "model",
            "file_count": len(output_files),
            "total_bytes": sum(item["size_bytes"] for item in output_files),
            "files": output_files,
        },
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    serialized = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
    except FileExistsError as exc:
        raise ValueError(f"Construction manifest already exists: {path}") from exc


def construct(
    model_dir: Path, spec_path: Path, corpus_spec_path: Path,
    corpus_manifest_path: Path, tokens_path: Path,
    output_dir: Path, manifest_path: Path, device: str,
) -> None:
    require_output_absent(output_dir, manifest_path)
    if output_dir.resolve().is_relative_to(model_dir.resolve()):
        raise ValueError("Output checkpoint must be outside source checkpoint")
    spec = load_spec(spec_path)
    corpus_spec = load_corpus_spec(corpus_spec_path)
    source_files = checkpoint_file_records(model_dir)
    corpus_context = verify_corpus_manifest(
        corpus_manifest_path, corpus_spec_path, tokens_path, model_dir, corpus_spec
    )
    script_path = Path(__file__).resolve()
    input_hashes = {
        path: sha256_file(path) for path in (
            spec_path, corpus_spec_path, corpus_manifest_path, tokens_path, script_path
        )
    }

    import torch

    loss_spec = spec["generic_loss"]
    tokens = load_corpus_tokens(
        tokens_path, corpus_context["raw_tensor_sha256"],
        loss_spec["sample_count"], loss_spec["sequence_length"], torch,
    )
    model, tokenizer = load_local_model(model_dir, device, torch)
    eligible = discover_eligible_linear_weights(model, torch)
    selected_ids = freeze_other_parameters(model, eligible)
    frozen_before = frozen_parameter_hashes(model, selected_ids, torch)
    mean_loss, moments = accumulate_empirical_fisher(model, tokens, eligible, loss_spec, torch)
    verify_frozen_parameters(model, selected_ids, frozen_before, torch)
    scale = global_rollback_scale(eligible, moments, spec["rollback"]["target_relative_frobenius"], torch)
    matrix_records, realized = apply_global_rollback(
        eligible, moments, scale["lambda"], scale["alpha"], scale["aggregate_source_weight_norm"], torch
    )
    del moments
    verify_frozen_parameters(model, selected_ids, frozen_before, torch)
    model.zero_grad(set_to_none=True)
    verify_unchanged(model_dir, source_files, input_hashes)
    require_output_absent(output_dir, manifest_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(output_dir)
    output_files = checkpoint_file_records(output_dir)
    verify_unchanged(model_dir, source_files, input_hashes)
    manifest = build_manifest(
        spec,
        {
            "attempt_spec": input_hashes[spec_path],
            "corpus_spec": input_hashes[corpus_spec_path],
            "corpus_manifest": input_hashes[corpus_manifest_path],
            "constructor_script": input_hashes[script_path],
        },
        source_files, output_files, corpus_context, mean_loss, scale,
        matrix_records, realized,
    )
    write_manifest(manifest_path, manifest)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--attempt-spec-path", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--corpus-spec-path", type=Path, default=DEFAULT_CORPUS_SPEC_PATH)
    parser.add_argument("--corpus-manifest-path", type=Path, default=DEFAULT_CORPUS_MANIFEST_PATH)
    parser.add_argument("--tokens-path", type=Path, default=DEFAULT_TOKENS_PATH)
    parser.add_argument("--output-model-dir", type=Path, default=DEFAULT_OUTPUT_MODEL_DIR)
    parser.add_argument("--construction-manifest-path", type=Path, default=DEFAULT_CONSTRUCTION_MANIFEST_PATH)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if _cuda_available() else "cpu")
    return parser.parse_args(argv)


def _cuda_available() -> bool:
    import torch
    return bool(torch.cuda.is_available())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        construct(
            args.merged_model_dir.expanduser(), args.attempt_spec_path.expanduser(),
            args.corpus_spec_path.expanduser(), args.corpus_manifest_path.expanduser(),
            args.tokens_path.expanduser(), args.output_model_dir.expanduser(),
            args.construction_manifest_path.expanduser(), args.device,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output_model_dir} and {args.construction_manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
