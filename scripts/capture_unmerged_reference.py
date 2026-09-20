#!/usr/bin/env python3

import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
PROMPT_FILE = PROJECT / "data" / "merge_verification_prompts.json"
DOWNLOADED_HASHES_PATH = PROJECT / "downloaded-model-hashes.json"

DEFAULT_BASE_MODEL_DIR = Path("/root/model-diff-scratch/models/base")
DEFAULT_ADAPTER_MODEL_DIR = Path("/root/model-diff-scratch/models/adapter")
DEFAULT_ARTIFACT_DIR = Path(
    "/root/model-diff-scratch/artifacts/merge_verification"
)
REFERENCE_FILENAME = "unmerged_reference.pt"

EXPECTED_PROMPTS = (
    "Explain why the sky appears blue during the day in three sentences.",
    "Write a Python function that returns the first n Fibonacci numbers.",
    "A train travels 120 km in 90 minutes. What is its average speed in km/h? "
    "Show the calculation.",
    "Give a concise comparison of mitosis and meiosis.",
    "Complete this story opening: The old observatory had been abandoned for "
    "decades, until",
    "Translate into French: The experiment must be reproducible.",
    "List five practical steps for debugging a program that suddenly became slow.",
    "What ingredients and steps are important when baking a moist chocolate cake?",
    "A baker wants to make a layered birthday cake. Describe a reliable workflow "
    "from mixing to decorating.",
    "Summarize in one sentence: A scientific result is much more useful when "
    "another researcher can reproduce it independently using a clearly specified "
    "procedure.",
)

GENERATION_SETTINGS = {
    "do_sample": False,
    "max_new_tokens": 32,
    "min_new_tokens": 32,
    "num_beams": 1,
}


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
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(
                f"Malformed {description} manifest SHA-256 for {relative_path}"
            )

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
        expected_record = expected[relative_path]
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            raise ValueError(
                f"Could not inspect local artifact file {path}: {exc}"
            ) from exc
        if actual_size != expected_record["size_bytes"]:
            raise ValueError(
                f"{description} size mismatch for {relative_path}: expected "
                f"{expected_record['size_bytes']}, found {actual_size}"
            )

        try:
            actual_sha256 = sha256_file(path)
        except OSError as exc:
            raise ValueError(
                f"Could not hash local artifact file {path}: {exc}"
            ) from exc
        if actual_sha256 != expected_record["sha256"]:
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


def load_prompts(path: Path = PROMPT_FILE) -> list[dict[str, Any]]:
    require_file(path, "merge-verification prompt file")
    try:
        prompts = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read prompt file {path}: {exc}") from exc

    if not isinstance(prompts, list) or len(prompts) != len(EXPECTED_PROMPTS):
        raise ValueError(f"Prompt file must contain exactly 10 prompts: {path}")

    for prompt_id, (record, expected_text) in enumerate(
        zip(prompts, EXPECTED_PROMPTS, strict=True)
    ):
        if not isinstance(record, dict):
            raise ValueError(f"Malformed prompt record {prompt_id}: {path}")
        if type(record.get("id")) is not int or not isinstance(
            record.get("prompt"), str
        ):
            raise ValueError(f"Malformed prompt record {prompt_id}: {path}")
        if record != {"id": prompt_id, "prompt": expected_text}:
            raise ValueError(f"Prompt {prompt_id} does not match the registered text")

    return prompts


def verify_float32_parameters(model: Any, torch_module: Any, description: str) -> Any:
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


def validate_capture_inputs(
    base_dir: Path,
    adapter_dir: Path,
    artifact_dir: Path,
) -> list[dict[str, Any]]:
    require_directory(base_dir, "base model directory")
    require_directory(adapter_dir, "adapter model directory")
    load_json_object(base_dir / "config.json", "base model config")
    load_json_object(base_dir / "tokenizer_config.json", "base tokenizer config")
    adapter_config = load_json_object(
        adapter_dir / "adapter_config.json", "adapter config"
    )
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        adapter_config_path = adapter_dir / "adapter_config.json"
        raise ValueError(
            f"Adapter config is not a LoRA adapter: {adapter_config_path}"
        )

    downloaded_manifest = load_sha256_manifest(
        DOWNLOADED_HASHES_PATH,
        "downloaded-model manifest",
    )
    verify_local_artifact(
        base_dir,
        downloaded_model_entry(downloaded_manifest, "base"),
        "base model artifact",
    )
    verify_local_artifact(
        adapter_dir,
        downloaded_model_entry(downloaded_manifest, "adapter"),
        "adapter model artifact",
    )
    prompts = load_prompts()

    reference_path = artifact_dir / REFERENCE_FILENAME
    try:
        artifact_dir_exists = artifact_dir.exists()
        reference_exists = reference_path.exists()
        artifact_dir_is_directory = (
            artifact_dir.is_dir() if artifact_dir_exists else False
        )
    except OSError as exc:
        raise ValueError(
            f"Could not inspect artifact path {artifact_dir}: {exc}"
        ) from exc

    if artifact_dir_exists and not artifact_dir_is_directory:
        raise ValueError(f"Artifact path exists and is not a directory: {artifact_dir}")
    if reference_exists:
        raise ValueError(f"Reference artifact already exists: {reference_path}")

    return prompts


def capture_prompt(
    model: Any,
    tokenizer: Any,
    prompt: dict[str, Any],
    device: Any,
    torch_module: Any,
) -> tuple[dict[str, Any], list[int]]:
    encoded = tokenizer(
        prompt["prompt"],
        return_tensors="pt",
        add_special_tokens=True,
    )
    if "input_ids" not in encoded:
        raise ValueError(f"Tokenizer returned no input_ids for prompt {prompt['id']}")
    if encoded["input_ids"].ndim != 2 or encoded["input_ids"].shape[0] != 1:
        raise ValueError(f"Unexpected input_ids shape for prompt {prompt['id']}")
    if encoded["input_ids"].dtype != torch_module.long:
        raise ValueError(f"Unexpected input_ids dtype for prompt {prompt['id']}")

    input_ids = encoded["input_ids"][0].detach().cpu().contiguous()
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
        raise ValueError(f"Model returned no hidden states for prompt {prompt['id']}")

    final_logits = outputs.logits[0, -1, :].detach().cpu().contiguous()
    if final_logits.dtype != torch_module.float32:
        raise ValueError(
            f"Prompt {prompt['id']} logits have unexpected dtype: {final_logits.dtype}"
        )

    final_hidden_states = []
    hidden_dimensions = []
    for level, hidden_state in enumerate(outputs.hidden_states):
        vector = hidden_state[0, -1, :].detach().cpu().contiguous()
        if vector.dtype != torch_module.float32:
            raise ValueError(
                f"Prompt {prompt['id']} hidden state {level} has unexpected dtype: "
                f"{vector.dtype}"
            )
        final_hidden_states.append(vector)
        hidden_dimensions.append(vector.numel())

    input_length = model_inputs["input_ids"].shape[-1]
    generated_token_ids = generated[0, input_length:].detach().cpu().contiguous()
    if (
        generated_token_ids.numel() != GENERATION_SETTINGS["max_new_tokens"]
        or generated_token_ids.dtype != torch_module.long
    ):
        generated_count = generated_token_ids.numel()
        raise ValueError(
            f"Prompt {prompt['id']} produced invalid continuation tokens "
            f"(count={generated_count}, dtype={generated_token_ids.dtype})"
        )

    record = {
        "id": prompt["id"],
        "input_ids": input_ids,
        "final_logits": final_logits,
        "final_hidden_states": final_hidden_states,
        "top1_token_id": int(torch_module.argmax(final_logits).item()),
        "generated_token_ids": generated_token_ids,
    }
    return record, hidden_dimensions


def capture_reference(
    base_dir: Path,
    adapter_dir: Path,
    artifact_dir: Path,
    prompts: list[dict[str, Any]],
) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise ValueError("CUDA is required for merge-equivalence capture")

    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        dtype=torch.float32,
        local_files_only=True,
    )
    verify_float32_parameters(base_model, torch, "Base model")

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        local_files_only=True,
    )
    verified_dtype = verify_float32_parameters(model, torch, "Base + LoRA model")
    device = torch.device("cuda")
    model.to(device)
    model.eval()
    verify_float32_parameters(model, torch, "CUDA base + LoRA model")

    records = []
    hidden_dimensions = None
    for prompt in prompts:
        record, prompt_hidden_dimensions = capture_prompt(
            model,
            tokenizer,
            prompt,
            device,
            torch,
        )
        if hidden_dimensions is None:
            hidden_dimensions = prompt_hidden_dimensions
        elif prompt_hidden_dimensions != hidden_dimensions:
            raise ValueError(
                f"Prompt {prompt['id']} returned inconsistent hidden-state dimensions"
            )
        records.append(record)

    if hidden_dimensions is None:
        raise ValueError("No prompt records were captured")

    reference = {
        "format_version": 1,
        "metadata": {
            "prompt_file_sha256": sha256_file(PROMPT_FILE),
            "downloaded_model_hashes_sha256": sha256_file(
                DOWNLOADED_HASHES_PATH
            ),
            "base_model_path": str(base_dir),
            "adapter_model_path": str(adapter_dir),
            "tokenizer_source": "base",
            "dtype": str(verified_dtype).removeprefix("torch."),
            "number_of_prompts": len(prompts),
            "generation_settings": dict(GENERATION_SETTINGS),
            "hidden_state_count": len(hidden_dimensions),
            "hidden_state_dimensions": hidden_dimensions,
        },
        "prompts": records,
    }

    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(
            f"Could not create artifact directory {artifact_dir}: {exc}"
        ) from exc

    reference_path = artifact_dir / REFERENCE_FILENAME
    try:
        with reference_path.open("xb") as reference_file:
            torch.save(reference, reference_file)
    except FileExistsError as exc:
        raise ValueError(
            f"Reference artifact already exists: {reference_path}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"Could not write reference artifact {reference_path}: {exc}"
        ) from exc
    print(f"Wrote {reference_path}")
    print(f"Captured {len(records)} prompts in {verified_dtype}")


def main() -> int:
    base_dir = Path(
        os.environ.get("BASE_MODEL_DIR", DEFAULT_BASE_MODEL_DIR)
    ).expanduser()
    adapter_dir = Path(
        os.environ.get("ADAPTER_MODEL_DIR", DEFAULT_ADAPTER_MODEL_DIR)
    ).expanduser()
    artifact_dir = Path(
        os.environ.get("ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)
    ).expanduser()

    try:
        prompts = validate_capture_inputs(base_dir, adapter_dir, artifact_dir)
        capture_reference(base_dir, adapter_dir, artifact_dir, prompts)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
