#!/usr/bin/env python3

"""Read the first five oracle ADL positions through the merged model head."""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

from compute_oracle_adl import (
    BATCH_SIZE,
    DOWNLOADED_HASHES_PATH,
    FREEZE_PROBE_SCRIPT_PATH,
    LAYER_INDEX,
    MERGED_HASHES_PATH,
    MERGE_SCRIPT_PATH,
    NUM_LAYERS,
    PROBE_MANIFEST_PATH,
    RELATIVE_LAYER,
    SAMPLE_COUNT,
    SCRIPT_PATH as COMPUTE_ORACLE_ADL_SCRIPT_PATH,
    SEQUENCE_LENGTH,
    downloaded_model_entry,
    load_json_object,
    load_sha256_manifest,
    model_identity,
    require_directory,
    require_file,
    require_sha256,
    sha256_file,
    sha256_raw_float32_tensor,
    validate_qwen_config,
    verify_float32_parameters,
    verify_local_artifact,
)


PROJECT = Path(__file__).resolve().parents[1]
ORACLE_MANIFEST_PATH = PROJECT / "oracle-adl-manifest.json"
OUTPUT_PATH = PROJECT / "oracle-logit-lens.json"

DEFAULT_BASE_MODEL_DIR = Path("/root/model-diff-scratch/models/base")
DEFAULT_MERGED_MODEL_DIR = Path("/root/model-diff-scratch/models/merged")
DEFAULT_ORACLE_PATH = Path(
    "/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt"
)

POSITIONS = tuple(range(5))
VECTOR_TYPES = ("difference", "base_mean", "ft_mean")
TOP_K = 20


def require_output_absent(path: Path) -> None:
    try:
        exists = path.exists()
    except OSError as exc:
        raise ValueError(f"Could not inspect output path {path}: {exc}") from exc
    if exists:
        raise ValueError(f"Output already exists: {path}")


def require_positive_integer(value: Any, description: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"Malformed {description}")
    return value


def validate_manifest_links(
    downloaded_manifest: dict[str, Any],
    merged_manifest: dict[str, Any],
    oracle_manifest: dict[str, Any],
    probe_manifest: dict[str, Any],
    *,
    downloaded_hashes_path: Path,
    merged_hashes_path: Path,
    probe_manifest_path: Path,
    merge_script_path: Path,
    compute_script_path: Path,
    freeze_probe_script_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Validate the complete committed provenance chain without loading a model."""
    base_entry = downloaded_model_entry(downloaded_manifest, "base")
    adapter_entry = downloaded_model_entry(downloaded_manifest, "adapter")
    base_identity = model_identity(base_entry, "base model")
    ft_identity = model_identity(adapter_entry, "fine-tuned model")

    downloaded_hash = sha256_file(downloaded_hashes_path)
    merged_hash = sha256_file(merged_hashes_path)
    probe_manifest_hash = sha256_file(probe_manifest_path)

    if merged_manifest.get("downloaded_model_hashes_sha256") != downloaded_hash:
        raise ValueError("Merged-model manifest provenance mismatch")
    if merged_manifest.get("base") != base_identity:
        raise ValueError("Merged-model manifest base identity mismatch")
    if merged_manifest.get("adapter") != ft_identity:
        raise ValueError("Merged-model manifest fine-tuned identity mismatch")
    if merged_manifest.get("merge_script_sha256") != sha256_file(merge_script_path):
        raise ValueError("Merged-model manifest merge script mismatch")
    if merged_manifest.get("merge_metadata") != {
        "dtype": "float32",
        "safe_merge": True,
        "tokenizer_source": "base",
    }:
        raise ValueError("Merged-model manifest merge metadata mismatch")

    expected_oracle_links = {
        "downloaded_model_hashes_sha256": downloaded_hash,
        "merged_model_hashes_sha256": merged_hash,
        "oracle_probe_manifest_sha256": probe_manifest_hash,
        "compute_oracle_adl_script_sha256": sha256_file(compute_script_path),
    }
    for key, expected_value in expected_oracle_links.items():
        require_sha256(oracle_manifest.get(key), f"oracle manifest {key}")
        if oracle_manifest[key] != expected_value:
            raise ValueError(f"Oracle manifest provenance mismatch: {key}")

    if oracle_manifest.get("base") != base_identity:
        raise ValueError("Oracle manifest base identity mismatch")
    if oracle_manifest.get("fine_tuned") != ft_identity:
        raise ValueError("Oracle manifest fine-tuned identity mismatch")

    if probe_manifest.get("downloaded_model_hashes_sha256") != downloaded_hash:
        raise ValueError("Oracle-probe manifest downloaded-model mismatch")
    if probe_manifest.get("freeze_oracle_probe_script_sha256") != sha256_file(
        freeze_probe_script_path
    ):
        raise ValueError("Oracle-probe manifest freeze script mismatch")

    expected_probe = {
        "serialized_sha256": probe_manifest.get("fineweb_tokens_sha256"),
        "raw_tensor_sha256": probe_manifest.get("raw_tensor_sha256"),
    }
    for key, value in expected_probe.items():
        require_sha256(value, f"oracle-probe manifest {key}")
    if oracle_manifest.get("probe") != expected_probe:
        raise ValueError("Oracle manifest probe provenance mismatch")

    return base_identity, ft_identity


def validate_oracle_manifest(
    manifest: dict[str, Any],
    hidden_size: int,
) -> None:
    expected_values = {
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
    }
    for key, expected_value in expected_values.items():
        if manifest.get(key) != expected_value:
            raise ValueError(f"Oracle manifest mismatch: {key}")

    require_sha256(manifest.get("oracle_adl_sha256"), "oracle artifact")
    raw_hashes = manifest.get("raw_tensors_sha256")
    if not isinstance(raw_hashes, dict) or set(raw_hashes) != set(VECTOR_TYPES):
        raise ValueError("Malformed oracle manifest raw tensor hashes")
    for name in VECTOR_TYPES:
        require_sha256(raw_hashes[name], f"oracle {name} tensor")


def verify_inputs_before_loading(
    base_dir: Path,
    merged_dir: Path,
    oracle_path: Path,
    *,
    downloaded_hashes_path: Path = DOWNLOADED_HASHES_PATH,
    merged_hashes_path: Path = MERGED_HASHES_PATH,
    probe_manifest_path: Path = PROBE_MANIFEST_PATH,
    oracle_manifest_path: Path = ORACLE_MANIFEST_PATH,
    merge_script_path: Path = MERGE_SCRIPT_PATH,
    compute_script_path: Path = COMPUTE_ORACLE_ADL_SCRIPT_PATH,
    freeze_probe_script_path: Path = FREEZE_PROBE_SCRIPT_PATH,
) -> dict[str, Any]:
    """Hash and validate all inputs before model/tokenizer deserialization."""
    require_directory(base_dir, "base tokenizer directory")
    require_directory(merged_dir, "merged model directory")
    require_file(oracle_path, "oracle artifact")

    downloaded_manifest = load_sha256_manifest(
        downloaded_hashes_path, "downloaded-model manifest"
    )
    merged_manifest = load_sha256_manifest(
        merged_hashes_path, "merged-model manifest"
    )
    oracle_manifest = load_sha256_manifest(
        oracle_manifest_path, "oracle ADL manifest"
    )
    probe_manifest = load_json_object(
        probe_manifest_path, "oracle-probe manifest"
    )

    base_identity, ft_identity = validate_manifest_links(
        downloaded_manifest,
        merged_manifest,
        oracle_manifest,
        probe_manifest,
        downloaded_hashes_path=downloaded_hashes_path,
        merged_hashes_path=merged_hashes_path,
        probe_manifest_path=probe_manifest_path,
        merge_script_path=merge_script_path,
        compute_script_path=compute_script_path,
        freeze_probe_script_path=freeze_probe_script_path,
    )

    base_entry = downloaded_model_entry(downloaded_manifest, "base")
    verify_local_artifact(base_dir, base_entry, "base model artifact")
    verify_local_artifact(merged_dir, merged_manifest, "merged model artifact")

    base_config = load_json_object(base_dir / "config.json", "base model config")
    merged_config = load_json_object(
        merged_dir / "config.json", "merged model config"
    )
    base_hidden_size = validate_qwen_config(base_config, "Base model")
    merged_hidden_size = validate_qwen_config(merged_config, "Merged model")
    if base_hidden_size != merged_hidden_size:
        raise ValueError("Base and merged model hidden sizes differ")

    base_vocab_size = require_positive_integer(
        base_config.get("vocab_size"), "base model vocab_size"
    )
    merged_vocab_size = require_positive_integer(
        merged_config.get("vocab_size"), "merged model vocab_size"
    )
    if base_vocab_size != merged_vocab_size:
        raise ValueError("Base and merged model vocabulary sizes differ")

    validate_oracle_manifest(oracle_manifest, merged_hidden_size)
    try:
        actual_oracle_hash = sha256_file(oracle_path)
    except OSError as exc:
        raise ValueError(f"Could not hash oracle artifact {oracle_path}: {exc}") from exc
    if actual_oracle_hash != oracle_manifest["oracle_adl_sha256"]:
        raise ValueError("Oracle artifact SHA-256 mismatch")

    return {
        "base": base_identity,
        "fine_tuned": ft_identity,
        "hidden_size": merged_hidden_size,
        "vocab_size": merged_vocab_size,
        "downloaded_model_hashes_sha256": sha256_file(downloaded_hashes_path),
        "merged_model_hashes_sha256": sha256_file(merged_hashes_path),
        "oracle_probe_manifest_sha256": sha256_file(probe_manifest_path),
        "oracle_adl_manifest_sha256": sha256_file(oracle_manifest_path),
        "oracle_adl_sha256": actual_oracle_hash,
        "oracle_manifest": oracle_manifest,
    }


def load_and_validate_oracle(
    oracle_path: Path,
    provenance: dict[str, Any],
    torch_module: Any,
) -> dict[str, Any]:
    """Load the already-hash-verified oracle and verify its tensor hashes."""
    try:
        artifact = torch_module.load(
            oracle_path,
            map_location="cpu",
            weights_only=True,
        )
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not read oracle artifact {oracle_path}: {exc}") from exc

    if not isinstance(artifact, dict):
        raise ValueError("Malformed oracle artifact: expected a dictionary")
    if artifact.get("format_version") != 1:
        raise ValueError("Malformed oracle artifact: unsupported format_version")

    manifest = provenance["oracle_manifest"]
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Malformed oracle artifact: missing metadata")
    metadata_keys = (
        "relative_layer",
        "layer_index",
        "num_layers",
        "sample_count",
        "sequence_length",
        "hidden_size",
        "batch_size",
        "model_dtype",
        "accumulator_dtype",
        "stored_dtype",
    )
    for key in metadata_keys:
        if metadata.get(key) != manifest.get(key):
            raise ValueError(f"Oracle artifact metadata mismatch: {key}")

    expected_shape = (SEQUENCE_LENGTH, provenance["hidden_size"])
    raw_hashes = manifest["raw_tensors_sha256"]
    for name in VECTOR_TYPES:
        tensor = artifact.get(name)
        if not isinstance(tensor, torch_module.Tensor):
            raise ValueError(f"Malformed oracle artifact: {name} is not a tensor")
        if tensor.device.type != "cpu":
            raise ValueError(f"Oracle {name} tensor is not on CPU")
        if tensor.dtype != torch_module.float32:
            raise ValueError(f"Oracle {name} tensor is not torch.float32")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Oracle {name} tensor has unexpected shape: {list(tensor.shape)}"
            )
        if not bool(torch_module.isfinite(tensor).all().item()):
            raise ValueError(f"Oracle {name} tensor contains non-finite values")
        if sha256_raw_float32_tensor(tensor, torch_module) != raw_hashes[name]:
            raise ValueError(f"Oracle {name} raw tensor SHA-256 mismatch")

    return artifact


def validate_loaded_qwen3_model(
    model: Any,
    hidden_size: int,
    vocab_size: int,
    torch_module: Any,
) -> tuple[Any, Any]:
    """Return Qwen3's actual final norm and LM head after dimension checks."""
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != "qwen3":
        raise ValueError("Loaded merged model config is not Qwen3")
    if getattr(config, "num_hidden_layers", None) != NUM_LAYERS:
        raise ValueError(f"Loaded merged model must have {NUM_LAYERS} layers")
    if getattr(config, "hidden_size", None) != hidden_size:
        raise ValueError("Loaded merged model hidden_size mismatch")
    if getattr(config, "vocab_size", None) != vocab_size:
        raise ValueError("Loaded merged model vocab_size mismatch")

    verify_float32_parameters(model, torch_module, "Merged model")

    backbone = getattr(model, "model", None)
    final_norm = getattr(backbone, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if final_norm is None:
        raise ValueError("Merged Qwen3 model does not expose model.norm")
    if lm_head is None:
        raise ValueError("Merged Qwen3 model does not expose lm_head")

    layers = getattr(backbone, "layers", None)
    if layers is None or len(layers) != NUM_LAYERS:
        raise ValueError(
            f"Merged Qwen3 model does not expose {NUM_LAYERS} transformer blocks"
        )

    norm_weight = getattr(final_norm, "weight", None)
    if not isinstance(norm_weight, torch_module.Tensor) or tuple(
        norm_weight.shape
    ) != (hidden_size,):
        raise ValueError("Final model norm has unexpected dimensions")
    if norm_weight.dtype != torch_module.float32:
        raise ValueError("Final model norm is not torch.float32")

    head_weight = getattr(lm_head, "weight", None)
    if not isinstance(head_weight, torch_module.Tensor) or tuple(
        head_weight.shape
    ) != (vocab_size, hidden_size):
        raise ValueError("LM head has unexpected dimensions")
    if head_weight.dtype != torch_module.float32:
        raise ValueError("LM head is not torch.float32")
    if getattr(lm_head, "in_features", hidden_size) != hidden_size:
        raise ValueError("LM head input dimension mismatch")
    if getattr(lm_head, "out_features", vocab_size) != vocab_size:
        raise ValueError("LM head output dimension mismatch")

    norm_device = norm_weight.device
    head_device = head_weight.device
    if norm_device.type != "cpu" or head_device.type != "cpu":
        raise ValueError("Final norm and LM head must be loaded on CPU")
    if norm_device != head_device:
        raise ValueError("Final norm and LM head are on different devices")

    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    if not callable(get_output_embeddings) or get_output_embeddings() is not lm_head:
        raise ValueError("AutoModelForCausalLM output embedding is not model.lm_head")

    return final_norm, lm_head


def validate_tokenizer(tokenizer: Any, vocab_size: int) -> None:
    try:
        tokenizer_size = len(tokenizer)
    except (TypeError, AttributeError) as exc:
        raise ValueError("Loaded base tokenizer has no vocabulary size") from exc
    # Qwen checkpoints may pad the model vocabulary beyond the tokenizer's
    # assigned IDs. The tokenizer must never expose IDs outside the LM head.
    if tokenizer_size <= 0 or tokenizer_size > vocab_size:
        raise ValueError(
            "Base tokenizer vocabulary is incompatible with the LM head: "
            f"tokenizer {tokenizer_size}, LM head {vocab_size}"
        )


def load_local_model_and_tokenizer(
    base_dir: Path,
    merged_dir: Path,
    provenance: dict[str, Any],
    torch_module: Any,
) -> tuple[Any, Any, Any, Any]:
    """Load only local files, with the merged model forced to float32 on CPU."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_dir,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            dtype=torch_module.float32,
            local_files_only=True,
        )
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Could not load local model/tokenizer: {exc}") from exc

    model.eval()
    validate_tokenizer(tokenizer, provenance["vocab_size"])
    final_norm, lm_head = validate_loaded_qwen3_model(
        model,
        provenance["hidden_size"],
        provenance["vocab_size"],
        torch_module,
    )
    return model, tokenizer, final_norm, lm_head


def logit_lens_probabilities(
    latent: Any,
    final_norm: Any,
    lm_head: Any,
    torch_module: Any,
) -> tuple[Any, Any]:
    """Apply the authors' positive and negative logit-lens directions."""
    normed = final_norm(latent)
    positive_probs = torch_module.softmax(lm_head(normed), dim=-1)
    negative_probs = torch_module.softmax(lm_head(-normed), dim=-1)
    return positive_probs, negative_probs


def top_token_records(
    probabilities: Any,
    tokenizer: Any,
    top_k: int,
    torch_module: Any,
) -> list[dict[str, Any]]:
    if not isinstance(probabilities, torch_module.Tensor):
        raise ValueError("Logit-lens probabilities are not a tensor")
    if probabilities.ndim != 1:
        raise ValueError("Logit-lens probabilities must be rank 1")
    if type(top_k) is not int or not 0 < top_k <= probabilities.numel():
        raise ValueError("Invalid top-k size")
    if not bool(torch_module.isfinite(probabilities).all().item()):
        raise ValueError("Logit-lens probabilities contain non-finite values")

    # Stable sorting makes equal probabilities deterministic by token ID.
    indices = torch_module.argsort(
        probabilities,
        dim=-1,
        descending=True,
        stable=True,
    )[:top_k]

    records = []
    for index in indices.detach().to(device="cpu").tolist():
        token_id = int(index)
        token = tokenizer.convert_ids_to_tokens(token_id)
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(token, str) or not isinstance(decoded, str):
            raise ValueError(f"Tokenizer returned invalid text for token {token_id}")
        records.append(
            {
                "token_id": token_id,
                "token": token,
                "decoded": decoded,
                "probability": float(probabilities[token_id].item()),
            }
        )
    return records


def build_readout(
    oracle: dict[str, Any],
    final_norm: Any,
    lm_head: Any,
    tokenizer: Any,
    provenance: dict[str, Any],
    torch_module: Any,
    *,
    positions: tuple[int, ...] = POSITIONS,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    results = []
    with torch_module.inference_mode():
        for position in positions:
            if type(position) is not int or not 0 <= position < SEQUENCE_LENGTH:
                raise ValueError(f"Invalid oracle position: {position}")
            position_record: dict[str, Any] = {"position": position}
            for vector_type in VECTOR_TYPES:
                latent = oracle[vector_type][position]
                if (
                    latent.ndim != 1
                    or latent.numel() != provenance["hidden_size"]
                    or latent.dtype != torch_module.float32
                    or latent.device.type != "cpu"
                ):
                    raise ValueError(
                        f"Invalid {vector_type} vector at position {position}"
                    )

                positive_probs, negative_probs = logit_lens_probabilities(
                    latent,
                    final_norm,
                    lm_head,
                    torch_module,
                )
                expected_shape = (provenance["vocab_size"],)
                if tuple(positive_probs.shape) != expected_shape or tuple(
                    negative_probs.shape
                ) != expected_shape:
                    raise ValueError("LM head returned unexpected dimensions")
                position_record[vector_type] = {
                    "positive": top_token_records(
                        positive_probs, tokenizer, top_k, torch_module
                    ),
                    "negative": top_token_records(
                        negative_probs, tokenizer, top_k, torch_module
                    ),
                }
            results.append(position_record)

    return {
        "format_version": 1,
        "provenance": {
            "base": provenance["base"],
            "fine_tuned": provenance["fine_tuned"],
            "downloaded_model_hashes_sha256": provenance[
                "downloaded_model_hashes_sha256"
            ],
            "merged_model_hashes_sha256": provenance[
                "merged_model_hashes_sha256"
            ],
            "oracle_probe_manifest_sha256": provenance[
                "oracle_probe_manifest_sha256"
            ],
            "oracle_adl_manifest_sha256": provenance[
                "oracle_adl_manifest_sha256"
            ],
            "oracle_adl_sha256": provenance["oracle_adl_sha256"],
            "tokenizer_source": "base",
            "model_dtype": "float32",
        },
        "method": {
            "positions": list(positions),
            "vector_types": list(VECTOR_TYPES),
            "top_k": top_k,
            "positive": "softmax(lm_head(model.model.norm(latent)))",
            "negative": "softmax(lm_head(-model.model.norm(latent)))",
        },
        "positions": results,
    }


def write_output(output: dict[str, Any], path: Path = OUTPUT_PATH) -> None:
    try:
        serialized = json.dumps(
            output,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not serialize logit-lens output: {exc}") from exc

    try:
        with path.open("x", encoding="utf-8", newline="\n") as output_file:
            output_file.write(serialized)
    except FileExistsError as exc:
        raise ValueError(f"Output already exists: {path}") from exc
    except OSError as exc:
        raise ValueError(f"Could not write output {path}: {exc}") from exc


def main() -> int:
    base_dir = Path(
        os.environ.get("BASE_MODEL_DIR", DEFAULT_BASE_MODEL_DIR)
    ).expanduser()
    merged_dir = Path(
        os.environ.get("MERGED_MODEL_DIR", DEFAULT_MERGED_MODEL_DIR)
    ).expanduser()
    oracle_path = Path(
        os.environ.get("ORACLE_ARTIFACT_PATH", DEFAULT_ORACLE_PATH)
    ).expanduser()

    try:
        require_output_absent(OUTPUT_PATH)

        # Importing torch is safe here; Transformers and the real model are not
        # touched until every committed manifest and artifact hash has passed.
        import torch

        provenance = verify_inputs_before_loading(
            base_dir,
            merged_dir,
            oracle_path,
        )
        oracle = load_and_validate_oracle(oracle_path, provenance, torch)
        model, tokenizer, final_norm, lm_head = load_local_model_and_tokenizer(
            base_dir,
            merged_dir,
            provenance,
            torch,
        )
        output = build_readout(
            oracle,
            final_norm,
            lm_head,
            tokenizer,
            provenance,
            torch,
        )
        write_output(output)
        del model
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
