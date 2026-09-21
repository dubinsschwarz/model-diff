#!/usr/bin/env python3

"""Construct Attempt 001 checkpoints by attenuating layer-13 spectral tails."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = (
    PROJECT / "experiments" / "attempts" / "001_spectral_tail_l13_p25" / "spec.json"
)
DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_OUTPUT_ROOT = Path(
    "/root/model-diff-scratch/ablations/001_spectral_tail_l13_p25"
)

ALLOWED_ENVIRONMENT_VARIABLES = {
    "MERGED_MODEL_DIR",
    "ABLATION_SPEC_PATH",
    "ABLATION_OUTPUT_ROOT",
}

FROZEN_SPEC = {
    "format_version": 1,
    "attempt_id": "001_spectral_tail_l13_p25",
    "method": "spectral_tail_attenuation",
    "provenance_free": True,
    "input_information_policy": "merged_checkpoint_only",
    "defaults": {
        "merged_model_directory": "/root/model-diff-scratch/models/merged",
        "output_root": (
            "/root/model-diff-scratch/ablations/001_spectral_tail_l13_p25"
        ),
    },
    "transformer_block_index": 13,
    "eligible_tensors": {
        "module_type": "torch.nn.Linear",
        "parameter": "weight",
        "scope": "recursive_descendants_of_model.model.layers[13]",
        "exclusions": [
            "norms",
            "embeddings",
            "lm_head",
            "biases",
            "other_layers",
        ],
    },
    "tail_fraction_by_singular_value_count": 0.25,
    "epsilons": [0.05, 0.1, 0.2],
    "svd": {
        "device": "cpu",
        "dtype": "float64",
        "full_matrices": False,
    },
    "rank": "min(weight.shape)",
    "tail_count": "floor(0.25 * r)",
    "tail_selection": "lowest_tail_count_from_descending_torch.linalg.svd",
    "tail_component": "U_tail @ diag(S_tail) @ Vh_tail",
    "weight_construction": {
        "float64": "W_original_float64 - epsilon * tail_component",
        "stored": "W_eps64.to(float32)",
    },
    "randomness": "none",
    "outputs": {
        "checkpoint_dtype": "float32",
        "standalone_checkpoints": True,
        "directory_names": {
            "0.05": "eps_0p05",
            "0.10": "eps_0p10",
            "0.20": "eps_0p20",
        },
        "construction_manifest": "construction-manifest.json",
        "timestamps": False,
        "host_metadata": False,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> None:
    try:
        is_file = path.is_file()
    except OSError as exc:
        raise ValueError(f"Could not access {description} {path}: {exc}") from exc
    if not is_file:
        raise ValueError(f"Missing {description}: {path}")


def require_directory(path: Path, description: str) -> None:
    try:
        is_directory = path.is_dir()
    except OSError as exc:
        raise ValueError(f"Could not access {description} {path}: {exc}") from exc
    if not is_directory:
        raise ValueError(f"Missing {description}: {path}")


def path_exists(path: Path, description: str) -> bool:
    try:
        return path.exists()
    except OSError as exc:
        raise ValueError(f"Could not inspect {description} {path}: {exc}") from exc


def load_spec(path: Path) -> dict[str, Any]:
    require_file(path, "attempt specification")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read attempt specification {path}: {exc}") from exc
    if not isinstance(spec, dict):
        raise ValueError("Attempt specification must be a JSON object")
    if spec != FROZEN_SPEC:
        raise ValueError("Attempt specification does not match the frozen definition")
    return spec


def epsilon_key(epsilon: float) -> str:
    return f"{epsilon:.2f}"


def output_directories(
    spec: dict[str, Any],
    output_root: Path,
) -> list[tuple[float, str, Path]]:
    names = spec["outputs"]["directory_names"]
    outputs = []
    for epsilon in spec["epsilons"]:
        key = epsilon_key(epsilon)
        directory_name = names.get(key)
        if not isinstance(directory_name, str) or not directory_name:
            raise ValueError(f"Missing output directory for epsilon {epsilon}")
        outputs.append((epsilon, directory_name, output_root / directory_name))
    return outputs


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_destinations(
    merged_dir: Path,
    spec_path: Path,
    output_root: Path,
    spec: dict[str, Any],
) -> tuple[Path, list[tuple[float, str, Path]]]:
    require_directory(merged_dir, "merged checkpoint directory")
    require_file(spec_path, "attempt specification")

    manifest_path = spec_path.parent / spec["outputs"]["construction_manifest"]
    if path_exists(manifest_path, "construction manifest"):
        raise ValueError(f"Construction manifest already exists: {manifest_path}")

    if path_exists(output_root, "output root"):
        require_directory(output_root, "output root")
    if is_within(output_root, merged_dir):
        raise ValueError("Output root must not be inside the merged checkpoint")
    if is_within(manifest_path, merged_dir):
        raise ValueError(
            "Construction manifest must not be inside the merged checkpoint"
        )

    outputs = output_directories(spec, output_root)
    for _epsilon, _directory_name, output_dir in outputs:
        if path_exists(output_dir, "output checkpoint"):
            raise ValueError(f"Output checkpoint already exists: {output_dir}")
    return manifest_path, outputs


def checkpoint_file_records(root: Path) -> list[dict[str, Any]]:
    require_directory(root, "checkpoint directory")
    candidates = []
    try:
        for path in root.rglob("*"):
            relative = path.relative_to(root)
            if ".cache" in relative.parts or not path.is_file():
                continue
            candidates.append((relative.as_posix(), path))
    except OSError as exc:
        raise ValueError(f"Could not enumerate checkpoint {root}: {exc}") from exc

    records = []
    for relative, path in sorted(candidates, key=lambda item: item[0]):
        try:
            records.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        except OSError as exc:
            raise ValueError(f"Could not hash checkpoint file {path}: {exc}") from exc
    if not records:
        raise ValueError(f"Checkpoint contains no files: {root}")
    return records


def discover_linear_weights(
    block: Any,
    module_prefix: str,
    torch_module: Any,
) -> list[tuple[str, Any]]:
    discovered = []
    for relative_name, module in block.named_modules():
        if not isinstance(module, torch_module.nn.Linear):
            continue
        module_name = module_prefix
        if relative_name:
            module_name = f"{module_prefix}.{relative_name}"
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch_module.Tensor) or weight.ndim != 2:
            raise ValueError(f"Linear module has invalid weight: {module_name}")
        discovered.append((module_name, module))
    discovered.sort(key=lambda item: item[0])
    return discovered


def spectral_tail_component(
    weight: Any,
    tail_fraction: float,
    torch_module: Any,
) -> tuple[Any, int, Any]:
    if not isinstance(weight, torch_module.Tensor) or weight.ndim != 2:
        raise ValueError("Weight must be a rank-2 tensor")
    if isinstance(tail_fraction, bool) or not isinstance(tail_fraction, (int, float)):
        raise ValueError("Tail fraction must be numeric")
    if not 0.0 <= tail_fraction <= 1.0:
        raise ValueError("Tail fraction must be between zero and one")

    original64 = weight.detach().to(device="cpu", dtype=torch_module.float64)
    left, singular_values, right_h = torch_module.linalg.svd(
        original64,
        full_matrices=False,
    )
    singular_value_count = min(original64.shape)
    if singular_values.numel() != singular_value_count:
        raise ValueError("SVD returned an unexpected singular-value count")
    tail_count = math.floor(tail_fraction * singular_value_count)

    if tail_count == 0:
        tail_component = torch_module.zeros_like(original64)
    else:
        left_tail = left[:, -tail_count:]
        singular_tail = singular_values[-tail_count:]
        right_h_tail = right_h[-tail_count:, :]
        tail_component = (
            left_tail @ torch_module.diag(singular_tail) @ right_h_tail
        )
    return singular_values, tail_count, tail_component.contiguous()


def construct_modified_weight(
    original32: Any,
    tail_component64: Any,
    epsilon: float,
    torch_module: Any,
) -> Any:
    if not isinstance(original32, torch_module.Tensor) or original32.ndim != 2:
        raise ValueError("Original weight must be a rank-2 tensor")
    if original32.dtype != torch_module.float32:
        raise ValueError("Original weight must be torch.float32")
    if (
        not isinstance(tail_component64, torch_module.Tensor)
        or tail_component64.dtype != torch_module.float64
        or tuple(tail_component64.shape) != tuple(original32.shape)
        or tail_component64.device.type != "cpu"
    ):
        raise ValueError("Tail component must be a matching CPU float64 tensor")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ValueError("Epsilon must be numeric")
    if not math.isfinite(epsilon):
        raise ValueError("Epsilon must be finite")

    original64 = original32.detach().to(device="cpu", dtype=torch_module.float64)
    modified64 = original64 - epsilon * tail_component64
    return modified64.to(dtype=torch_module.float32).contiguous()


def realized_perturbation_metrics(
    original32: Any,
    modified32: Any,
    torch_module: Any,
) -> dict[str, float]:
    if (
        not isinstance(original32, torch_module.Tensor)
        or not isinstance(modified32, torch_module.Tensor)
        or original32.dtype != torch_module.float32
        or modified32.dtype != torch_module.float32
        or tuple(original32.shape) != tuple(modified32.shape)
    ):
        raise ValueError(
            "Realized perturbation inputs must be matching float32 tensors"
        )

    original64 = original32.detach().to(device="cpu", dtype=torch_module.float64)
    modified64 = modified32.detach().to(device="cpu", dtype=torch_module.float64)
    delta64 = modified64 - original64
    original_norm = float(torch_module.linalg.vector_norm(original64).item())
    delta_norm = float(torch_module.linalg.vector_norm(delta64).item())
    if original_norm == 0.0:
        if delta_norm != 0.0:
            raise ValueError("Cannot normalize a nonzero change to a zero weight")
        relative = 0.0
    else:
        relative = delta_norm / original_norm
    max_change = float(delta64.abs().max().item())
    return {
        "realized_fp32_delta_frobenius_norm": delta_norm,
        "realized_relative_frobenius_perturbation": relative,
        "max_absolute_weight_change": max_change,
    }


def build_weight_plan(
    module_name: str,
    module: Any,
    spec: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    parameter = module.weight
    if parameter.device.type != "cpu" or parameter.dtype != torch_module.float32:
        raise ValueError(f"Eligible weight must be CPU float32: {module_name}")
    original32 = parameter.detach().clone().contiguous()
    if not bool(torch_module.isfinite(original32).all().item()):
        raise ValueError(f"Eligible weight contains non-finite values: {module_name}")
    singular_values, tail_count, tail_component64 = spectral_tail_component(
        original32,
        spec["tail_fraction_by_singular_value_count"],
        torch_module,
    )

    singular_squares = singular_values.square()
    total_energy = float(singular_squares.sum().item())
    if tail_count == 0 or total_energy == 0.0:
        tail_energy_fraction = 0.0
    else:
        tail_energy_fraction = float(
            singular_squares[-tail_count:].sum().item() / total_energy
        )
    original_norm = float(
        torch_module.linalg.vector_norm(original32.to(torch_module.float64)).item()
    )
    tail_norm = float(
        torch_module.linalg.vector_norm(tail_component64).item()
    )

    epsilon_records = []
    for epsilon in spec["epsilons"]:
        modified32 = construct_modified_weight(
            original32,
            tail_component64,
            epsilon,
            torch_module,
        )
        epsilon_record = {"epsilon": epsilon}
        epsilon_record.update(
            realized_perturbation_metrics(original32, modified32, torch_module)
        )
        epsilon_records.append(epsilon_record)

    return {
        "module_name": module_name,
        "module": module,
        "original32": original32,
        "tail_component64": tail_component64,
        "record": {
            "module_name": module_name,
            "shape": list(original32.shape),
            "singular_value_count": singular_values.numel(),
            "tail_count": tail_count,
            "tail_energy_fraction": tail_energy_fraction,
            "original_frobenius_norm": original_norm,
            "tail_component_frobenius_norm": tail_norm,
            "epsilons": epsilon_records,
        },
    }


def verify_float32_model(model: Any, torch_module: Any) -> None:
    floating_parameters = [
        parameter for parameter in model.parameters() if parameter.is_floating_point()
    ]
    if not floating_parameters:
        raise ValueError("Loaded checkpoint has no floating-point parameters")
    unexpected = sorted(
        {
            str(parameter.dtype)
            for parameter in floating_parameters
            if parameter.dtype != torch_module.float32
        }
    )
    if unexpected:
        raise ValueError(
            "Loaded checkpoint has unexpected floating dtypes: "
            + ", ".join(unexpected)
        )
    devices = {parameter.device.type for parameter in floating_parameters}
    if devices != {"cpu"}:
        raise ValueError("Loaded checkpoint must remain on CPU")


def locate_target_block(model: Any, block_index: int) -> Any:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None or not 0 <= block_index < len(layers):
        raise ValueError(f"Checkpoint does not expose transformer block {block_index}")
    return layers[block_index]


def load_local_checkpoint(
    merged_dir: Path,
    torch_module: Any,
) -> tuple[Any, Any]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(
        merged_dir,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        merged_dir,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        config=config,
        dtype=torch_module.float32,
        local_files_only=True,
    )
    model.eval()
    verify_float32_model(model, torch_module)
    return model, tokenizer


def apply_epsilon(
    plans: list[dict[str, Any]],
    epsilon: float,
    torch_module: Any,
) -> None:
    with torch_module.no_grad():
        for plan in plans:
            modified32 = construct_modified_weight(
                plan["original32"],
                plan["tail_component64"],
                epsilon,
                torch_module,
            )
            plan["module"].weight.copy_(modified32)


def restore_original_weights(
    plans: list[dict[str, Any]],
    torch_module: Any,
) -> None:
    with torch_module.no_grad():
        for plan in plans:
            plan["module"].weight.copy_(plan["original32"])


def normalized_file_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record) for record in sorted(records, key=lambda item: item["path"])]


def build_construction_manifest(
    spec: dict[str, Any],
    spec_sha256: str,
    constructor_sha256: str,
    source_files: list[dict[str, Any]],
    output_checkpoints: list[dict[str, Any]],
    matrix_records: list[dict[str, Any]],
) -> dict[str, Any]:
    epsilon_order = {epsilon: index for index, epsilon in enumerate(spec["epsilons"])}
    matrices = []
    for record in sorted(matrix_records, key=lambda item: item["module_name"]):
        normalized = dict(record)
        normalized["epsilons"] = sorted(
            (dict(item) for item in record["epsilons"]),
            key=lambda item: epsilon_order[item["epsilon"]],
        )
        matrices.append(normalized)

    outputs = []
    for checkpoint in sorted(
        output_checkpoints,
        key=lambda item: epsilon_order[item["epsilon"]],
    ):
        normalized = dict(checkpoint)
        normalized_files = normalized_file_records(checkpoint["files"])
        normalized["file_count"] = len(normalized_files)
        normalized["total_bytes"] = sum(
            item["size_bytes"] for item in normalized_files
        )
        normalized["files"] = normalized_files
        outputs.append(normalized)

    normalized_source_files = normalized_file_records(source_files)

    combined_original_squared = sum(
        record["original_frobenius_norm"] ** 2 for record in matrices
    )
    combined_original_norm = math.sqrt(combined_original_squared)
    aggregates = []
    for epsilon in spec["epsilons"]:
        epsilon_metrics = []
        for record in matrices:
            matches = [
                item for item in record["epsilons"] if item["epsilon"] == epsilon
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Matrix metric count mismatch for epsilon {epsilon}"
                )
            epsilon_metrics.append(matches[0])
        combined_delta_norm = math.sqrt(
            sum(
                item["realized_fp32_delta_frobenius_norm"] ** 2
                for item in epsilon_metrics
            )
        )
        if combined_original_norm == 0.0:
            combined_relative = 0.0
        else:
            combined_relative = combined_delta_norm / combined_original_norm
        aggregates.append(
            {
                "epsilon": epsilon,
                "combined_original_frobenius_norm": combined_original_norm,
                "combined_realized_fp32_delta_frobenius_norm": (
                    combined_delta_norm
                ),
                "combined_realized_relative_frobenius_perturbation": (
                    combined_relative
                ),
                "max_absolute_weight_change": max(
                    (
                        item["max_absolute_weight_change"]
                        for item in epsilon_metrics
                    ),
                    default=0.0,
                ),
            }
        )

    return {
        "format_version": 1,
        "attempt_id": spec["attempt_id"],
        "provenance_free": True,
        "input_information_policy": spec["input_information_policy"],
        "hash_algorithm": "sha256",
        "spec_sha256": spec_sha256,
        "constructor_sha256": constructor_sha256,
        "source_checkpoint": {
            "file_count": len(normalized_source_files),
            "total_bytes": sum(
                item["size_bytes"] for item in normalized_source_files
            ),
            "files": normalized_source_files,
        },
        "construction": {
            "transformer_block_index": spec["transformer_block_index"],
            "eligible_tensors": dict(spec["eligible_tensors"]),
            "tail_fraction_by_singular_value_count": spec[
                "tail_fraction_by_singular_value_count"
            ],
            "epsilons": list(spec["epsilons"]),
            "svd": dict(spec["svd"]),
            "rank": spec["rank"],
            "tail_count": spec["tail_count"],
            "tail_selection": spec["tail_selection"],
            "tail_component": spec["tail_component"],
            "weight_construction": dict(spec["weight_construction"]),
            "randomness": spec["randomness"],
            "stored_dtype": spec["outputs"]["checkpoint_dtype"],
            "standalone_checkpoints": spec["outputs"]["standalone_checkpoints"],
        },
        "modified_matrices": matrices,
        "aggregate_perturbations": aggregates,
        "output_checkpoints": outputs,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    try:
        serialized = json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not serialize construction manifest: {exc}") from exc
    try:
        with path.open("x", encoding="utf-8", newline="\n") as file:
            file.write(serialized)
    except FileExistsError as exc:
        raise ValueError(f"Construction manifest already exists: {path}") from exc
    except OSError as exc:
        raise ValueError(
            f"Could not write construction manifest {path}: {exc}"
        ) from exc


def save_checkpoints(
    model: Any,
    tokenizer: Any,
    plans: list[dict[str, Any]],
    outputs: list[tuple[float, str, Path]],
    output_root: Path,
    torch_module: Any,
) -> list[dict[str, Any]]:
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"Could not create output root {output_root}: {exc}") from exc

    checkpoint_records = []
    try:
        for epsilon, directory_name, output_dir in outputs:
            try:
                output_dir.mkdir(exist_ok=False)
            except FileExistsError as exc:
                raise ValueError(
                    f"Output checkpoint already exists: {output_dir}"
                ) from exc
            except OSError as exc:
                raise ValueError(
                    f"Could not create output checkpoint {output_dir}: {exc}"
                ) from exc

            apply_epsilon(plans, epsilon, torch_module)
            verify_float32_model(model, torch_module)
            model.save_pretrained(output_dir, safe_serialization=True)
            tokenizer.save_pretrained(output_dir)
            checkpoint_records.append(
                {
                    "epsilon": epsilon,
                    "directory_name": directory_name,
                    "files": checkpoint_file_records(output_dir),
                }
            )
    finally:
        restore_original_weights(plans, torch_module)
    return checkpoint_records


def construct(
    merged_dir: Path,
    spec_path: Path,
    output_root: Path,
) -> Path:
    spec = load_spec(spec_path)
    spec_hash = sha256_file(spec_path)
    constructor_path = Path(__file__).resolve()
    constructor_hash = sha256_file(constructor_path)
    manifest_path, outputs = validate_destinations(
        merged_dir,
        spec_path,
        output_root,
        spec,
    )
    source_files_before = checkpoint_file_records(merged_dir)

    import torch

    model, tokenizer = load_local_checkpoint(merged_dir, torch)
    block_index = spec["transformer_block_index"]
    block = locate_target_block(model, block_index)
    eligible = discover_linear_weights(
        block,
        f"model.layers.{block_index}",
        torch,
    )
    if not eligible:
        raise ValueError("Target transformer block contains no eligible weights")

    plans = [
        build_weight_plan(module_name, module, spec, torch)
        for module_name, module in eligible
    ]
    checkpoint_records = save_checkpoints(
        model,
        tokenizer,
        plans,
        outputs,
        output_root,
        torch,
    )

    source_files_after = checkpoint_file_records(merged_dir)
    if source_files_after != source_files_before:
        raise ValueError("Merged checkpoint changed during construction")
    if sha256_file(spec_path) != spec_hash:
        raise ValueError("Attempt specification changed during construction")
    if sha256_file(constructor_path) != constructor_hash:
        raise ValueError("Constructor changed during construction")

    manifest = build_construction_manifest(
        spec,
        spec_hash,
        constructor_hash,
        source_files_before,
        checkpoint_records,
        [plan["record"] for plan in plans],
    )
    write_manifest(manifest, manifest_path)
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-model-dir",
        type=Path,
        default=Path(
            os.environ.get("MERGED_MODEL_DIR", DEFAULT_MERGED_MODEL_DIR)
        ).expanduser(),
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=Path(
            os.environ.get("ABLATION_SPEC_PATH", DEFAULT_SPEC_PATH)
        ).expanduser(),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get("ABLATION_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)
        ).expanduser(),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest_path = construct(
            args.merged_model_dir,
            args.spec_path,
            args.output_root,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
