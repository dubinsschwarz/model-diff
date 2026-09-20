#!/usr/bin/env python3

import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

from capture_unmerged_reference import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_BASE_MODEL_DIR,
    DOWNLOADED_HASHES_PATH,
    GENERATION_SETTINGS,
    PROJECT,
    PROMPT_FILE,
    REFERENCE_FILENAME,
    downloaded_model_entry,
    load_json_object,
    load_prompts,
    load_sha256_manifest,
    require_directory,
    require_file,
    sha256_file,
    verify_float32_parameters,
    verify_local_artifact,
)


DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
MERGED_HASHES_PATH = PROJECT / "merged-model-hashes.json"
LOCK_PATH = PROJECT / "models.lock.json"
CAPTURE_SCRIPT_PATH = PROJECT / "scripts" / "capture_unmerged_reference.py"
VERIFY_SCRIPT_PATH = PROJECT / "scripts" / "verify_merge.py"
REPORT_PATH = PROJECT / "merge-verification-report.json"

MAX_LOGIT_ABS_ERROR = 1e-4
MAX_HIDDEN_ABS_ERROR = 1e-3
MIN_HIDDEN_COSINE_SIMILARITY = 0.999999


def load_source_models() -> dict[str, dict[str, str]]:
    lock = load_json_object(LOCK_PATH, "model lock file")
    source_models = {}
    for role in ("base", "adapter"):
        try:
            repo_id = lock[role]["repo_id"]
            revision = lock[role]["revision"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Model lock file is missing {role} repo_id/revision: {LOCK_PATH}"
            ) from exc
        if not isinstance(repo_id, str) or not isinstance(revision, str):
            raise ValueError(
                f"Model lock file has invalid {role} repo_id/revision: {LOCK_PATH}"
            )
        source_models[role] = {
            "repo_id": repo_id,
            "revision": revision,
        }
    return source_models


def validate_verification_inputs(
    base_dir: Path,
    merged_dir: Path,
    artifact_dir: Path,
) -> tuple[list[dict[str, Any]], Path, dict[str, dict[str, str]]]:
    require_directory(base_dir, "base tokenizer directory")
    require_directory(merged_dir, "merged model directory")
    load_json_object(base_dir / "config.json", "base model config")
    load_json_object(base_dir / "tokenizer_config.json", "base tokenizer config")
    load_json_object(merged_dir / "config.json", "merged model config")

    prompts = load_prompts()
    downloaded_manifest = load_sha256_manifest(
        DOWNLOADED_HASHES_PATH,
        "downloaded-model manifest",
    )
    merged_manifest = load_sha256_manifest(
        MERGED_HASHES_PATH,
        "merged-model manifest",
    )
    verify_local_artifact(
        base_dir,
        downloaded_model_entry(downloaded_manifest, "base"),
        "base model artifact",
    )
    verify_local_artifact(
        merged_dir,
        merged_manifest,
        "merged model artifact",
    )
    require_file(CAPTURE_SCRIPT_PATH, "capture script")
    require_file(VERIFY_SCRIPT_PATH, "verification script")
    source_models = load_source_models()

    reference_path = artifact_dir / REFERENCE_FILENAME
    require_file(reference_path, "unmerged reference artifact")
    return prompts, reference_path, source_models


def load_reference(reference_path: Path, torch_module: Any) -> dict[str, Any]:
    try:
        reference = torch_module.load(
            reference_path,
            map_location="cpu",
            weights_only=True,
        )
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Could not read unmerged reference artifact {reference_path}: {exc}"
        ) from exc

    if not isinstance(reference, dict):
        raise ValueError("Malformed reference artifact: expected a dictionary")
    return reference


def require_reference_tensor(
    record: dict[str, Any],
    key: str,
    prompt_id: int,
    torch_module: Any,
) -> Any:
    tensor = record.get(key)
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError(
            f"Malformed reference prompt {prompt_id}: {key} is not a tensor"
        )
    return tensor


def validate_reference(
    reference: dict[str, Any],
    prompts: list[dict[str, Any]],
    torch_module: Any,
) -> list[dict[str, Any]]:
    if (
        type(reference.get("format_version")) is not int
        or reference["format_version"] != 1
    ):
        raise ValueError("Malformed reference artifact: unsupported format_version")

    metadata = reference.get("metadata")
    records = reference.get("prompts")
    if not isinstance(metadata, dict) or not isinstance(records, list):
        raise ValueError("Malformed reference artifact: missing metadata/prompts")

    expected_metadata = {
        "prompt_file_sha256": sha256_file(PROMPT_FILE),
        "downloaded_model_hashes_sha256": sha256_file(DOWNLOADED_HASHES_PATH),
        "tokenizer_source": "base",
        "dtype": "float32",
        "number_of_prompts": len(prompts),
        "generation_settings": GENERATION_SETTINGS,
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"Reference metadata mismatch: {key}")

    for key in ("base_model_path", "adapter_model_path"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"Malformed reference metadata: {key}")

    hidden_state_count = metadata.get("hidden_state_count")
    hidden_dimensions = metadata.get("hidden_state_dimensions")
    if type(hidden_state_count) is not int or hidden_state_count <= 0:
        raise ValueError("Malformed reference metadata: hidden_state_count")
    if (
        not isinstance(hidden_dimensions, list)
        or len(hidden_dimensions) != hidden_state_count
        or any(
            type(dimension) is not int or dimension <= 0
            for dimension in hidden_dimensions
        )
    ):
        raise ValueError("Malformed reference metadata: hidden_state_dimensions")

    if len(records) != len(prompts):
        raise ValueError("Malformed reference artifact: incorrect prompt count")

    for prompt, record in zip(prompts, records, strict=True):
        prompt_id = prompt["id"]
        if (
            not isinstance(record, dict)
            or type(record.get("id")) is not int
            or record["id"] != prompt_id
        ):
            raise ValueError(
                f"Malformed reference prompt record: expected ID {prompt_id}"
            )

        input_ids = require_reference_tensor(
            record, "input_ids", prompt_id, torch_module
        )
        final_logits = require_reference_tensor(
            record, "final_logits", prompt_id, torch_module
        )
        generated_ids = require_reference_tensor(
            record, "generated_token_ids", prompt_id, torch_module
        )
        hidden_states = record.get("final_hidden_states")

        if input_ids.ndim != 1 or input_ids.numel() == 0:
            raise ValueError(f"Malformed reference prompt {prompt_id}: input_ids")
        if input_ids.dtype != torch_module.long:
            raise ValueError(f"Malformed reference prompt {prompt_id}: input_ids dtype")
        if final_logits.ndim != 1 or final_logits.numel() == 0:
            raise ValueError(f"Malformed reference prompt {prompt_id}: final_logits")
        if final_logits.dtype != torch_module.float32:
            raise ValueError(
                f"Malformed reference prompt {prompt_id}: final_logits dtype"
            )
        if (
            generated_ids.ndim != 1
            or generated_ids.numel() != GENERATION_SETTINGS["max_new_tokens"]
            or generated_ids.dtype != torch_module.long
        ):
            raise ValueError(
                f"Malformed reference prompt {prompt_id}: generated_token_ids"
            )
        if (
            not isinstance(hidden_states, list)
            or len(hidden_states) != hidden_state_count
        ):
            raise ValueError(
                f"Malformed reference prompt {prompt_id}: final_hidden_states"
            )

        for level, (hidden_state, expected_dimension) in enumerate(
            zip(hidden_states, hidden_dimensions, strict=True)
        ):
            if not isinstance(hidden_state, torch_module.Tensor):
                raise ValueError(
                    f"Malformed reference prompt {prompt_id}: hidden state {level}"
                )
            if (
                hidden_state.ndim != 1
                or hidden_state.numel() != expected_dimension
                or hidden_state.dtype != torch_module.float32
            ):
                raise ValueError(
                    f"Malformed reference prompt {prompt_id}: hidden state {level}"
                )

        top1_token_id = record.get("top1_token_id")
        if type(top1_token_id) is not int or top1_token_id != int(
            torch_module.argmax(final_logits).item()
        ):
            raise ValueError(f"Malformed reference prompt {prompt_id}: top1_token_id")

    return records


def absolute_error_metrics(
    actual: Any,
    expected: Any,
    torch_module: Any,
) -> dict[str, float | int]:
    if tuple(actual.shape) != tuple(expected.shape) or actual.numel() == 0:
        raise ValueError("Cannot compare tensors with different or empty shapes")

    difference = (
        actual.to(dtype=torch_module.float64)
        - expected.to(dtype=torch_module.float64)
    ).abs()
    metrics = {
        "max": float(difference.max().item()),
        "mean": float(difference.mean().item()),
        "sum": float(difference.sum().item()),
        "count": difference.numel(),
    }
    if not all(math.isfinite(metrics[key]) for key in ("max", "mean", "sum")):
        raise ValueError("Cannot compare tensors with non-finite values")
    return metrics


def cosine_similarity(actual: Any, expected: Any, torch_module: Any) -> float:
    if tuple(actual.shape) != tuple(expected.shape) or actual.numel() == 0:
        raise ValueError("Cannot compare tensors with different or empty shapes")

    actual_vector = actual.reshape(-1).to(dtype=torch_module.float64)
    expected_vector = expected.reshape(-1).to(dtype=torch_module.float64)
    actual_norm = float(torch_module.linalg.vector_norm(actual_vector).item())
    expected_norm = float(torch_module.linalg.vector_norm(expected_vector).item())
    denominator = actual_norm * expected_norm
    norm_values = (actual_norm, expected_norm, denominator)
    if not all(math.isfinite(value) for value in norm_values):
        raise ValueError("Cannot compute cosine similarity with non-finite values")

    if denominator == 0.0:
        return 1.0 if bool(torch_module.equal(actual_vector, expected_vector)) else 0.0

    similarity = (
        float(torch_module.dot(actual_vector, expected_vector).item()) / denominator
    )
    if not math.isfinite(similarity):
        raise ValueError("Cannot compute cosine similarity with non-finite values")
    return max(-1.0, min(1.0, similarity))


def exact_tensor_match(actual: Any, expected: Any, torch_module: Any) -> bool:
    return tuple(actual.shape) == tuple(expected.shape) and bool(
        torch_module.equal(actual, expected)
    )


def evaluate_acceptance(
    aggregate_metrics: dict[str, Any],
    prompt_count: int,
) -> tuple[dict[str, dict[str, Any]], bool]:
    criteria = {
        "global_max_logit_absolute_error": {
            "comparison": "<=",
            "threshold": MAX_LOGIT_ABS_ERROR,
            "passed": aggregate_metrics["global_max_logit_absolute_error"]
            <= MAX_LOGIT_ABS_ERROR,
        },
        "global_max_hidden_absolute_error": {
            "comparison": "<=",
            "threshold": MAX_HIDDEN_ABS_ERROR,
            "passed": aggregate_metrics["global_max_hidden_absolute_error"]
            <= MAX_HIDDEN_ABS_ERROR,
        },
        "minimum_hidden_cosine_similarity": {
            "comparison": ">=",
            "threshold": MIN_HIDDEN_COSINE_SIMILARITY,
            "passed": aggregate_metrics["minimum_hidden_cosine_similarity"]
            >= MIN_HIDDEN_COSINE_SIMILARITY,
        },
        "top1_agreement": {
            "comparison": "==",
            "threshold": prompt_count,
            "passed": aggregate_metrics["top1_agreement_count"] == prompt_count,
        },
        "greedy_generation_agreement": {
            "comparison": "==",
            "threshold": prompt_count,
            "passed": aggregate_metrics["generation_agreement_count"]
            == prompt_count,
        },
    }
    return criteria, all(criterion["passed"] for criterion in criteria.values())


def compare_prompt(
    model: Any,
    tokenizer: Any,
    prompt: dict[str, Any],
    reference: dict[str, Any],
    device: Any,
    torch_module: Any,
) -> tuple[dict[str, Any], float, int]:
    prompt_id = prompt["id"]
    encoded = tokenizer(
        prompt["prompt"],
        return_tensors="pt",
        add_special_tokens=True,
    )
    if "input_ids" not in encoded:
        raise ValueError(f"Tokenizer returned no input_ids for prompt {prompt_id}")

    current_input_ids = encoded["input_ids"][0].detach().cpu().contiguous()
    if current_input_ids.dtype != torch_module.long:
        raise ValueError(f"Unexpected input_ids dtype for prompt {prompt_id}")
    if not exact_tensor_match(
        current_input_ids,
        reference["input_ids"],
        torch_module,
    ):
        raise ValueError(f"Reference input token IDs do not match prompt {prompt_id}")

    model_inputs = {name: tensor.to(device) for name, tensor in encoded.items()}
    with torch_module.inference_mode():
        outputs = model(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        generated = model.generate(
            **model_inputs,
            **GENERATION_SETTINGS,
        )

    if outputs.hidden_states is None:
        raise ValueError(
            f"Merged model returned no hidden states for prompt {prompt_id}"
        )

    final_logits = outputs.logits[0, -1, :].detach().cpu().contiguous()
    if final_logits.dtype != torch_module.float32:
        raise ValueError(
            f"Prompt {prompt_id} merged logits have unexpected dtype: "
            f"{final_logits.dtype}"
        )
    logit_metrics = absolute_error_metrics(
        final_logits,
        reference["final_logits"],
        torch_module,
    )
    actual_top1 = int(torch_module.argmax(final_logits).item())
    top1_agreement = actual_top1 == reference["top1_token_id"]

    reference_hidden_states = reference["final_hidden_states"]
    if len(outputs.hidden_states) != len(reference_hidden_states):
        raise ValueError(f"Hidden-state count mismatch for prompt {prompt_id}")

    hidden_metrics = []
    for level, (actual_state, expected_state) in enumerate(
        zip(outputs.hidden_states, reference_hidden_states, strict=True)
    ):
        actual_vector = actual_state[0, -1, :].detach().cpu().contiguous()
        if actual_vector.dtype != torch_module.float32:
            raise ValueError(
                f"Prompt {prompt_id} merged hidden state {level} has unexpected "
                f"dtype: {actual_vector.dtype}"
            )
        errors = absolute_error_metrics(actual_vector, expected_state, torch_module)
        similarity = cosine_similarity(actual_vector, expected_state, torch_module)
        hidden_metrics.append(
            {
                "level": level,
                "max_absolute_error": errors["max"],
                "cosine_similarity": similarity,
            }
        )

    input_length = model_inputs["input_ids"].shape[-1]
    generated_token_ids = generated[0, input_length:].detach().cpu().contiguous()
    if (
        generated_token_ids.numel() != GENERATION_SETTINGS["max_new_tokens"]
        or generated_token_ids.dtype != torch_module.long
    ):
        generated_count = generated_token_ids.numel()
        raise ValueError(
            f"Prompt {prompt_id} produced invalid continuation tokens "
            f"(count={generated_count}, dtype={generated_token_ids.dtype})"
        )
    generation_agreement = exact_tensor_match(
        generated_token_ids,
        reference["generated_token_ids"],
        torch_module,
    )

    metrics = {
        "id": prompt_id,
        "max_logit_absolute_error": logit_metrics["max"],
        "mean_logit_absolute_error": logit_metrics["mean"],
        "top1_agreement": top1_agreement,
        "hidden_states": hidden_metrics,
        "generation_exact_match": generation_agreement,
    }
    return metrics, logit_metrics["sum"], logit_metrics["count"]


def verify_merged_model(
    base_dir: Path,
    merged_dir: Path,
    prompts: list[dict[str, Any]],
    records: list[dict[str, Any]],
    source_models: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], bool]:
    os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for merge-equivalence verification")

    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        dtype=torch.float32,
        local_files_only=True,
    )
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    verified_dtype = verify_float32_parameters(model, torch, "Merged model")

    per_prompt = []
    total_logit_absolute_error = 0.0
    total_logit_values = 0
    global_max_logit_error = 0.0
    global_max_hidden_error = 0.0
    minimum_hidden_cosine = 1.0
    top1_agreements = 0
    generation_agreements = 0

    for prompt, reference in zip(prompts, records, strict=True):
        metrics, logit_error_sum, logit_value_count = compare_prompt(
            model,
            tokenizer,
            prompt,
            reference,
            device,
            torch,
        )
        per_prompt.append(metrics)
        total_logit_absolute_error += logit_error_sum
        total_logit_values += logit_value_count
        global_max_logit_error = max(
            global_max_logit_error,
            metrics["max_logit_absolute_error"],
        )
        for hidden_state in metrics["hidden_states"]:
            global_max_hidden_error = max(
                global_max_hidden_error,
                hidden_state["max_absolute_error"],
            )
            minimum_hidden_cosine = min(
                minimum_hidden_cosine,
                hidden_state["cosine_similarity"],
            )
        top1_agreements += int(metrics["top1_agreement"])
        generation_agreements += int(metrics["generation_exact_match"])

    aggregate_metrics = {
        "global_max_logit_absolute_error": global_max_logit_error,
        "global_mean_logit_absolute_error": total_logit_absolute_error
        / total_logit_values,
        "global_max_hidden_absolute_error": global_max_hidden_error,
        "minimum_hidden_cosine_similarity": minimum_hidden_cosine,
        "top1_agreement_count": top1_agreements,
        "generation_agreement_count": generation_agreements,
        "prompt_count": len(prompts),
    }
    criteria, overall_pass = evaluate_acceptance(aggregate_metrics, len(prompts))

    report = {
        "aggregate_metrics": aggregate_metrics,
        "per_prompt_metrics": per_prompt,
        "acceptance_criteria": criteria,
        "overall_pass": overall_pass,
        "prompt_file_sha256": sha256_file(PROMPT_FILE),
        "downloaded_model_hashes_sha256": sha256_file(
            DOWNLOADED_HASHES_PATH
        ),
        "merged_model_hashes_sha256": sha256_file(MERGED_HASHES_PATH),
        "capture_script_sha256": sha256_file(CAPTURE_SCRIPT_PATH),
        "verification_script_sha256": sha256_file(VERIFY_SCRIPT_PATH),
        "models": source_models,
        "dtype": str(verified_dtype).removeprefix("torch."),
        "tokenizer_source": "base",
        "generation_settings": dict(GENERATION_SETTINGS),
    }
    return report, overall_pass


def main() -> int:
    base_dir = Path(
        os.environ.get("BASE_MODEL_DIR", DEFAULT_BASE_MODEL_DIR)
    ).expanduser()
    merged_dir = Path(
        os.environ.get("MERGED_MODEL_DIR", DEFAULT_MERGED_MODEL_DIR)
    ).expanduser()
    artifact_dir = Path(
        os.environ.get("ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)
    ).expanduser()

    try:
        prompts, reference_path, source_models = validate_verification_inputs(
            base_dir,
            merged_dir,
            artifact_dir,
        )

        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch

        reference = load_reference(reference_path, torch)
        records = validate_reference(reference, prompts, torch)
        report, overall_pass = verify_merged_model(
            base_dir,
            merged_dir,
            prompts,
            records,
            source_models,
        )
        REPORT_PATH.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {REPORT_PATH}")
    print(f"Overall pass: {overall_pass}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
