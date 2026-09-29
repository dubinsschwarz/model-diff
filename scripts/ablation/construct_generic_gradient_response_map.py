#!/usr/bin/env python3
"""Construct Attempt 009: blind generic-gradient response map."""

from contextlib import ExitStack
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
FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '009_generic_gradient_response_map_prefix0_13',
 'method': 'blind_generic_gradient_response_map',
 'information_policy': 'final_checkpoint_plus_frozen_generic_corpus_and_probe',
 'defaults': {'merged_model_directory': '/root/model-diff-scratch/models/merged',
              'corpus_spec_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json',
              'corpus_manifest_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json',
              'corpus_artifact_path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
              'probe_path': '/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt',
              'artifact_path': '/root/model-diff-scratch/artifacts/attempt009_response_map/response_map.pt',
              'construction_manifest_path': 'experiments/attempts/009_generic_gradient_response_map_prefix0_13/construction-manifest.json',
              'temporary_directory': '/root/model-diff-scratch/tmp/attempt009'},
 'model': {'model_type': 'qwen3',
           'num_hidden_layers': 28,
           'source_dtype': 'float32',
           'local_files_only': True,
           'eval_mode': True,
           'use_cache': False},
 'eligible_tensors': {'transformer_block_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                      'recursive_module_type': 'torch.nn.Linear',
                      'parameter': 'weight',
                      'expected_matrix_count': 98,
                      'module_ordering': 'lexicographic_by_module_name',
                      'freeze_every_other_parameter': True},
 'generic_loss': {'type': 'causal_next_token_cross_entropy',
                  'sample_count': 4096,
                  'sequence_length': 128,
                  'predictions_per_sample': 127,
                  'total_prediction_tokens': 520192,
                  'batch_size': 8,
                  'number_of_batches': 512,
                  'label_alignment': 'logits_positions_0_through_126_predict_tokens_1_through_127',
                  'reduction': 'sum_token_cross_entropy_divided_by_total_prediction_tokens',
                  'masking': 'causal_model_mask_only',
                  'mixed_precision': False,
                  'dropout': False,
                  'optimizer': False,
                  'weight_decay': False,
                  'gradient_clipping': False},
 'tangent': {'direction': 'delta = +alpha * g',
             'target_relative_frobenius': 0.00125,
             'normalization': 'one_global_alpha_across_all_eligible_matrices',
             'norm_accounting_dtype': 'cpu_float64',
             'scaling_arithmetic_dtype': 'cpu_float64',
             'jvp_tangent_dtype': 'float32',
             'modify_model_weights': False,
             'save_large_tensors': False},
 'probe': {'sample_count': 10000,
           'sequence_length': 128,
           'batch_size': 32,
           'dtype': 'torch.int64',
           'serialized_sha256': '3d799d19abf4a2d6dc92b2bd164d81d83f6451cd3545f5ae5d8c04a8b299735b',
           'raw_tensor_sha256': '73f56300fdd06bc8fafcfa5750fee7a586a5c5e6071a608667ffc526bbb90ac8'},
 'readout': {'block_index': 13,
             'hidden_state_index': 14,
             'hidden_size': 2048,
             'all_token_positions': True,
             'semantics': 'hidden_states[14] is block 13 output before block 14; verify with '
                          'forward hook'},
 'jvp': {'implementation': 'torch.func.jvp_with_partial_torch.func.functional_call',
         'module': 'model.model',
         'output_hidden_states': True,
         'strict_forward_ad': True,
         'fallback': False,
         'change_attention_backend': False,
         'reverse_mode_recording': False,
         'response_sign': 'positive_J_delta',
         'accumulation_dtype': 'cpu_float64',
         'stored_dtype': 'float32',
         'primal_comparison_rtol': 1e-05,
         'primal_comparison_atol': 1e-06},
 'outputs': {'tensors': ['merged_mean',
                         'full_response',
                         'split_responses',
                         'block_responses',
                         'family_responses'],
             'standalone_checkpoint': False,
             'timestamps': False,
             'host_metadata': False,
             'gpu_metadata': False},
 'purpose': 'blind_diagnostic_only_not_a_recovery_candidate',
 'canonical_checkpoint_files': [{'path': 'chat_template.jinja',
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
                                 'sha256': '1cc816812993bff176eb4f7495433b736f06fba9b6e7b05cac7b4a1780650c95'}],
 'splits': {'count': 4,
            'order': [0, 1, 2, 3],
            'selection': 'example_index_modulo_4',
            'sample_count': 1024,
            'total_prediction_tokens': 130048,
            'batch_size': 8,
            'number_of_batches': 128,
            'full_gradient': 'mean_of_four_split_gradients_in_cpu_float64',
            'shared_alpha': True},
 'diagnostic_probe': {'start': 0, 'stop': 1024, 'sample_count': 1024, 'batch_size': 32},
 'decomposition': {'block_order': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                   'family_order': ['attention', 'mlp'],
                   'renormalize': False,
                   'full_response': 'direct_JVP_of_delta_full_float32_alpha_times_cpu_float64_mean_split_gradients',
                   'consistency': {'maximum_relative_rms_difference': 0.0001,
                                   'relative_rms_definition': 'rms(candidate[p] - '
                                                              'full_response[p]) / '
                                                              'rms(full_response[p])',
                                   'scope': 'per_position_all_must_pass',
                                   'zero_reference': 'per_position_zero_error_passes_otherwise_fails',
                                   'maximum_relative_max_difference': 0.0001,
                                   'relative_max_definition': 'max_abs(candidate[p] - '
                                                              'full_response[p]) / '
                                                              'max_abs(full_response[p])',
                                   'worst_position_tie_break': 'lowest_token_position',
                                   'global_max_absolute_difference': 'descriptive_only_not_thresholded'},
                   'partition_tangents': 'exact_keywise_subsets_of_the_canonical_direct_full_tangent',
                   'split_mean_diagnostic': {'definition': 'mean_of_four_split_responses_in_cpu_float64',
                                             'reference': 'direct_full_response',
                                             'thresholded': False,
                                             'stored_as_tensor': False,
                                             'metrics': ['cosine_similarity',
                                                         'rms_difference',
                                                         'relative_rms_difference',
                                                         'maximum_absolute_difference',
                                                         'relative_max_difference'],
                                             'zero_norm_cosine': None,
                                             'zero_reference_relative_error': 'zero_if_zero_error_otherwise_null'},
                   'response_direction_count': 21,
                   'batched_jvp_calls': 672},
 'diagnostics': {'dtype': 'cpu_float64',
                 'responses': 'stored_equivalent_cpu_float32_promoted_to_float64',
                 'positions': [0,
                               1,
                               2,
                               3,
                               4,
                               5,
                               6,
                               7,
                               8,
                               9,
                               10,
                               11,
                               12,
                               13,
                               14,
                               15,
                               16,
                               17,
                               18,
                               19,
                               20,
                               21,
                               22,
                               23,
                               24,
                               25,
                               26,
                               27,
                               28,
                               29,
                               30,
                               31,
                               32,
                               33,
                               34,
                               35,
                               36,
                               37,
                               38,
                               39,
                               40,
                               41,
                               42,
                               43,
                               44,
                               45,
                               46,
                               47,
                               48,
                               49,
                               50,
                               51,
                               52,
                               53,
                               54,
                               55,
                               56,
                               57,
                               58,
                               59,
                               60,
                               61,
                               62,
                               63,
                               64,
                               65,
                               66,
                               67,
                               68,
                               69,
                               70,
                               71,
                               72,
                               73,
                               74,
                               75,
                               76,
                               77,
                               78,
                               79,
                               80,
                               81,
                               82,
                               83,
                               84,
                               85,
                               86,
                               87,
                               88,
                               89,
                               90,
                               91,
                               92,
                               93,
                               94,
                               95,
                               96,
                               97,
                               98,
                               99,
                               100,
                               101,
                               102,
                               103,
                               104,
                               105,
                               106,
                               107,
                               108,
                               109,
                               110,
                               111,
                               112,
                               113,
                               114,
                               115,
                               116,
                               117,
                               118,
                               119,
                               120,
                               121,
                               122,
                               123,
                               124,
                               125,
                               126,
                               127],
                 'zero_cosine': None,
                 'zero_cancellation_denominator': None,
                 'zero_spectrum': 'zero_fractions_top1_and_effective_rank',
                 'split_summaries': 'position0_and_positions1_through127_pooled_pairwise_cosines_and_norms_no_vector_pooling'},
 'temporary_gradients': {'format': 'safetensors',
                         'dtype': 'float32',
                         'contents': 'eligible_gradients_only',
                         'cleanup': 'after_successful_artifact_and_manifest',
                         'nonempty_directory': 'refuse',
                         'on_failure': 'retain_state_and_gradients_no_resume'},
 'frozen_corpus': {'corpus_spec_sha256': 'e514820e7def99e9287c32de4b1279f28f6ff6c2e09749a8dfa5d8b14c311f0f',
                   'corpus_manifest_sha256': '8b3ea49228dbb106c2b8a7c849c4f27e021fa3e992800444f43e260a6b055230',
                   'serialized_sha256': '54283d72aa3075e2867ead86e538991d7284ede327ca67871e73bfc581e4025d',
                   'raw_tensor_sha256': '4613982be77342a89b35616d68e21d9ff66d5a983af7e82cb6fa1eb159457eae'}}

FROZEN_CORPUS_SPEC = {
    "format_version": 1,
    "attempt_id": "005_generic_gradient_rollback_prefix0_13_r00125",
    "dataset": {
        "repo_id": "science-of-finetuning/fineweb-1m-sample",
        "revision": "60b53a86b84eb6559e4407b113356f56a152318f",
        "split": "train",
        "streaming": False
    },
    "tokenizer": {
        "source": "canonical_merged_checkpoint",
        "directory": "/root/model-diff-scratch/models/merged"
    },
    "selection": {
        "shuffle_seed": 42,
        "text_column": "text",
        "skip_blank_text": True,
        "character_limit": 1280,
        "add_special_tokens": True,
        "minimum_token_count": 128,
        "skip_valid_examples": 20000,
        "sample_count": 4096,
        "sequence_length": 128
    },
    "artifact": {
        "path": "/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt",
        "dtype": "torch.int64",
        "device": "cpu",
        "contiguous": True
    },
    "manifest": {
        "filename": "corpus-manifest.json",
        "timestamps": False,
        "host_metadata": False,
        "gpu_metadata": False
    }
}

ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/009_generic_gradient_response_map_prefix0_13"
DEFAULT_SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
DEFAULT_CORPUS_SPEC_PATH = PROJECT / FROZEN_SPEC["defaults"]["corpus_spec_path"]
DEFAULT_CORPUS_MANIFEST_PATH = PROJECT / FROZEN_SPEC["defaults"]["corpus_manifest_path"]
DEFAULT_MODEL_DIR = Path(FROZEN_SPEC["defaults"]["merged_model_directory"])
DEFAULT_TOKENS_PATH = Path(FROZEN_SPEC["defaults"]["corpus_artifact_path"])
DEFAULT_PROBE_PATH = Path(FROZEN_SPEC["defaults"]["probe_path"])
DEFAULT_ARTIFACT_PATH = Path(FROZEN_SPEC["defaults"]["artifact_path"])
DEFAULT_CONSTRUCTION_MANIFEST_PATH = PROJECT / FROZEN_SPEC["defaults"]["construction_manifest_path"]
TOKENIZER_FILE_NAMES = ("chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json")

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


def accumulate_mean_generic_gradient(
    model: Any, tokens: Any, eligible: list[tuple[str, Any]],
    loss_spec: dict[str, Any], torch_module: Any,
) -> float:
    sample_count = loss_spec["sample_count"]
    sequence_length = loss_spec["sequence_length"]
    batch_size = loss_spec["batch_size"]
    total_tokens = loss_spec["total_prediction_tokens"]
    if (tuple(tokens.shape) != (sample_count, sequence_length) or
        sample_count % batch_size != 0 or
        sample_count // batch_size != loss_spec["number_of_batches"] or
        total_tokens != sample_count * (sequence_length - 1) or
        loss_spec["predictions_per_sample"] != sequence_length - 1):
        raise ValueError("Frozen gradient averaging dimensions mismatch")
    selected_ids = {id(module.weight) for _, module in eligible}
    model.eval()
    model.zero_grad(set_to_none=True)
    device = eligible[0][1].weight.device
    loss_sums = []
    with torch_module.enable_grad(), torch_module.autocast(device_type=device.type, enabled=False):
        for start in range(0, sample_count, batch_size):
            batch = tokens[start:start + batch_size].to(device=device)
            logits = model(input_ids=batch, use_cache=False).logits
            loss_sum = causal_token_loss_sum(logits, batch, torch_module)
            if not bool(torch_module.isfinite(loss_sum).item()):
                raise ValueError("Generic loss is non-finite")
            # Each target contributes once, with the supplied full/split token denominator.
            (loss_sum / total_tokens).backward()
            loss_sums.append(float(loss_sum.detach().to(device="cpu", dtype=torch_module.float64).item()))
            for name, parameter in model.named_parameters():
                if id(parameter) not in selected_ids and parameter.grad is not None:
                    raise ValueError(f"Forbidden parameter received gradient: {name}")
    for name, module in eligible:
        gradient = module.weight.grad
        if (gradient is None or gradient.dtype != torch_module.float32 or
            tuple(gradient.shape) != tuple(module.weight.shape) or
            not bool(torch_module.isfinite(gradient).all().item())):
            raise ValueError(f"Missing or non-finite eligible gradient: {name}")
    mean_loss = math.fsum(loss_sums) / total_tokens
    if not math.isfinite(mean_loss):
        raise ValueError("Mean generic loss is non-finite")
    return mean_loss


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


def load_probe(path: Path, probe_spec: dict[str, Any], torch_module: Any) -> Any:
    if sha256_file(path) != require_sha256(probe_spec["serialized_sha256"], "serialized probe"):
        raise ValueError("Frozen probe serialized SHA-256 mismatch")
    return load_corpus_tokens(
        path, probe_spec["raw_tensor_sha256"], probe_spec["sample_count"],
        probe_spec["sequence_length"], torch_module,
    )


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


def validate_readout(value: Any, batch: int, spec: dict[str, Any], torch_module: Any, description: str) -> None:
    expected = (batch, spec["probe"]["sequence_length"], spec["readout"]["hidden_size"])
    if (not isinstance(value, torch_module.Tensor) or tuple(value.shape) != expected or
        value.dtype != torch_module.float32 or not bool(torch_module.isfinite(value).all().item())):
        raise ValueError(f"Invalid shape, dtype, or non-finite {description}")


def hidden_state_output(output: Any, spec: dict[str, Any]) -> Any:
    block = spec["readout"]["block_index"]
    index = spec["readout"]["hidden_state_index"]
    layers = spec["model"]["num_hidden_layers"]
    states = getattr(output, "hidden_states", None)
    # The final hidden-state entry can include final normalization; block 13 is
    # intermediate. Its output is recorded as the input to block 14, at index 14.
    if index != block + 1 or not 0 <= block < layers - 1:
        raise ValueError("Invalid block-output hidden-state index")
    if not isinstance(states, (tuple, list)) or len(states) != layers + 1:
        raise ValueError("Transformer hidden-state collection has unexpected length")
    return states[index]


def ordinary_hook_readout(model: Any, tokens: Any, spec: dict[str, Any], torch_module: Any) -> Any:
    captured = []
    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        value = output[0] if isinstance(output, (tuple, list)) else output
        validate_readout(value, tokens.shape[0], spec, torch_module, "block hook output")
        captured.append(value.detach().clone())
    handle = model.model.layers[spec["readout"]["block_index"]].register_forward_hook(hook)
    try:
        with torch_module.no_grad(), torch_module.autocast(device_type=tokens.device.type, enabled=False):
            output = model.model(input_ids=tokens, use_cache=False, output_hidden_states=True, return_dict=True)
        selected = hidden_state_output(output, spec)
        validate_readout(selected, tokens.shape[0], spec, torch_module, "hidden-state readout")
        if len(captured) != 1 or not torch_module.equal(selected, captured[0]):
            raise ValueError("Hidden-state readout does not equal block output hook")
        return captured[0]
    finally:
        handle.remove()


def run_readout_jvp(
    model: Any, tokens: Any, tangents: dict[str, Any], spec: dict[str, Any], torch_module: Any,
) -> tuple[Any, Any]:
    """Native forward AD with only selected weights as primals/tangents."""
    if not tangents:
        raise ValueError("JVP requires a nonempty partial parameter tangent")
    names = sorted(tangents)
    local_names, primals, vectors = [], [], []
    for name in names:
        if not name.startswith("model."):
            raise ValueError(f"Tangent is not a transformer weight: {name}")
        local = name[len("model."):]
        parameter = model.model.get_parameter(local)
        vector = tangents[name]
        if (parameter.dtype != torch_module.float32 or vector.dtype != torch_module.float32 or
            parameter.shape != vector.shape or parameter.device != vector.device):
            raise ValueError(f"Invalid JVP parameter/tangent: {name}")
        local_names.append(local)
        primals.append(parameter.detach())
        vectors.append(vector)

    def activation(weights: tuple[Any, ...]) -> Any:
        output = torch_module.func.functional_call(
            model.model, dict(zip(local_names, weights)), (),
            {"input_ids": tokens, "use_cache": False, "output_hidden_states": True, "return_dict": True},
            strict=False,
        )
        return hidden_state_output(output, spec)

    try:
        # no_grad disables reverse graph recording, not native forward AD.
        with torch_module.no_grad(), torch_module.autocast(device_type=tokens.device.type, enabled=False):
            primal, response = torch_module.func.jvp(
                activation, (tuple(primals),), (tuple(vectors),), strict=True,
            )
    except (NotImplementedError, RuntimeError) as exc:
        raise RuntimeError(f"Native forward AD JVP failed; no fallback is permitted: {exc}") from exc
    validate_readout(primal, tokens.shape[0], spec, torch_module, "JVP primal")
    validate_readout(response, tokens.shape[0], spec, torch_module, "JVP tangent")
    return primal.detach(), response.detach()


def compare_primal(primal: Any, ordinary: Any, spec: dict[str, Any], torch_module: Any) -> float:
    maximum = float((primal.double() - ordinary.double()).abs().max().item())
    if not torch_module.allclose(
        primal, ordinary, rtol=spec["jvp"]["primal_comparison_rtol"],
        atol=spec["jvp"]["primal_comparison_atol"],
    ):
        raise ValueError(f"JVP primal disagrees with ordinary block output; max_abs={maximum}")
    return maximum


class ResponseAccumulator:
    """Reduce samples, preserving every token position, in CPU float64."""
    def __init__(self, spec: dict[str, Any], torch_module: Any) -> None:
        self.spec, self.torch = spec, torch_module
        shape = (spec["probe"]["sequence_length"], spec["readout"]["hidden_size"])
        self.primal_sum = torch_module.zeros(shape, dtype=torch_module.float64, device="cpu")
        self.response_sum = torch_module.zeros_like(self.primal_sum)
        self.count = 0

    def add(self, primal: Any, response: Any) -> None:
        for value, total, name in ((primal, self.primal_sum, "primal"), (response, self.response_sum, "tangent")):
            validate_readout(value, primal.shape[0], self.spec, self.torch, name)
            total.add_(value.detach().to(device="cpu", dtype=self.torch.float64).sum(dim=0))
        self.count += primal.shape[0]

    def means(self, count: int) -> tuple[Any, Any]:
        if count <= 0 or self.count != count:
            raise ValueError("Linear-response sample count mismatch")
        return self.primal_sum / count, self.response_sum / count


def sha256_raw_float32_tensor(value: Any, torch_module: Any) -> str:
    if (not isinstance(value, torch_module.Tensor) or value.dtype != torch_module.float32 or
        value.device.type != "cpu" or not value.is_contiguous()):
        raise ValueError("Raw activation hash requires contiguous CPU float32 tensor")
    return hashlib.sha256(value.detach().numpy().astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def require_output_absent(artifact_path: Path, manifest_path: Path) -> None:
    for path in (artifact_path, manifest_path):
        if path.exists():
            raise ValueError(f"Output already exists: {path}")
    if artifact_path.resolve() == manifest_path.resolve():
        raise ValueError("Artifact and manifest paths must differ")
    if not manifest_path.parent.is_dir():
        raise ValueError(f"Missing manifest directory: {manifest_path.parent}")


def interleaved_splits(tokens: Any, count: int = 4) -> list[Any]:
    if count != 4 or tokens.ndim != 2 or tokens.shape[0] == 0 or tokens.shape[0] % count:
        raise ValueError("Corpus must partition into four equal nonempty splits")
    return [tokens[k::count].contiguous() for k in range(count)]


def split_loss_spec(spec: dict[str, Any]) -> dict[str, Any]:
    settings = dict(spec["generic_loss"])
    split = spec["splits"]
    for key in ("sample_count", "total_prediction_tokens", "batch_size", "number_of_batches"):
        settings[key] = split[key]
    return settings


def prepare_temporary_directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
        raise ValueError("Attempt009 temporary directory must be absent or empty; stale data cannot be reused")
    path.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents two constructors sharing an empty directory.
    with (path / "state.json").open("x", encoding="utf-8") as stream:
        json.dump({"stage": "started", "resume_permitted": False}, stream)


def temporary_stage(path: Path, stage: str) -> None:
    (path / "state.json").write_text(json.dumps({"stage": stage, "resume_permitted": False}) + "\n")


def cleanup_temporary_directory(path: Path) -> None:
    expected = {"state.json", *(f"split_{k}.safetensors" for k in range(4))}
    if {entry.name for entry in path.iterdir()} != expected:
        raise ValueError("Unexpected temporary contents; refusing cleanup")
    for name in sorted(expected):
        (path / name).unlink()
    path.rmdir()


def save_split_gradient(path: Path, eligible: list[tuple[str, Any]], torch_module: Any) -> None:
    from safetensors.torch import save_file
    if path.exists() or path.is_symlink():
        raise ValueError("Temporary split gradient already exists")
    tensors = {}
    for name, module in eligible:
        gradient = module.weight.grad
        if (gradient is None or gradient.dtype != torch_module.float32 or
            gradient.shape != module.weight.shape):
            raise ValueError(f"Missing or malformed split gradient: {name}")
        value = gradient.detach().to(device="cpu").contiguous()
        if not bool(torch_module.isfinite(value).all().item()):
            raise ValueError(f"Non-finite split gradient: {name}")
        tensors[f"{name}.weight"] = value
        module.weight.grad = None
    save_file(tensors, str(path))


class SplitGradientStore:
    """Memory-map four split files; only requested matrices become FP64 working tensors."""
    def __init__(self, directory: Path, eligible: list[tuple[str, Any]], torch_module: Any):
        self.directory, self.eligible, self.torch = directory, eligible, torch_module
        self.stack = ExitStack()

    def __enter__(self):
        from safetensors import safe_open
        try:
            self.files = [self.stack.enter_context(safe_open(
                str(self.directory / f"split_{k}.safetensors"), framework="pt", device="cpu"
            )) for k in range(4)]
            names = {f"{name}.weight" for name, _ in self.eligible}
            if any(set(file.keys()) != names for file in self.files):
                raise ValueError("Temporary split gradient keys mismatch")
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def matrix(self, split: int, name: str, shape: Any) -> Any:
        value = self.files[split].get_tensor(f"{name}.weight")
        if (value.dtype != self.torch.float32 or tuple(value.shape) != tuple(shape) or
            not bool(self.torch.isfinite(value).all().item())):
            raise ValueError(f"Invalid temporary gradient: split {split}, {name}")
        return value.to(dtype=self.torch.float64)

    def mean(self, name: str, shape: Any) -> Any:
        value = self.matrix(0, name, shape)
        for k in range(1, 4):
            value.add_(self.matrix(k, name, shape))
        return value.div_(4)


def parameter_group(name: str) -> tuple[int, str]:
    parts = name.split(".")
    if len(parts) < 5 or parts[:2] != ["model", "layers"] or not parts[2].isdigit():
        raise ValueError(f"Invalid eligible parameter name: {name}")
    block = int(parts[2])
    family = {"self_attn": "attention", "mlp": "mlp"}.get(parts[3])
    if not 0 <= block <= 13 or family is None:
        raise ValueError(f"Invalid eligible block or family: {name}")
    return block, family


def gram_summary(gram: Any, torch_module: Any) -> dict[str, Any]:
    if gram.dtype != torch_module.float64 or not bool(torch_module.isfinite(gram).all().item()):
        raise ValueError("Gram statistics must be finite float64")
    norms = gram.diag().clamp_min(0).sqrt().tolist()
    cosine = [[max(-1.0, min(1.0, float(gram[i, j]) / (norms[i] * norms[j])))
               if norms[i] and norms[j] else None for j in range(len(norms))]
              for i in range(len(norms))]
    return {"norms": norms, "cosine_matrix": cosine}


def weight_space_statistics(store: SplitGradientStore, eligible: list[tuple[str, Any]],
                            torch_module: Any) -> tuple[dict[str, Any], dict[str, float]]:
    grams = [torch_module.zeros((4, 4), dtype=torch_module.float64, device="cpu") for _ in range(14)]
    weight_squares, full_squares = [], []
    for name, module in eligible:
        block, _ = parameter_group(name)
        values = [store.matrix(k, name, module.weight.shape) for k in range(4)]
        for i in range(4):
            for j in range(i, 4):
                dot = torch_module.dot(values[i].reshape(-1), values[j].reshape(-1))
                grams[block][i, j] += dot
                if i != j:
                    grams[block][j, i] += dot
        full = values[0].clone()
        for value in values[1:]:
            full.add_(value)
        full.div_(4)
        full_squares.append(float(full.square().sum()))
        weight = module.weight.detach().to(device="cpu", dtype=torch_module.float64)
        weight_squares.append(float(weight.square().sum()))
        del values, full, weight, value
    total = torch_module.stack(grams).sum(dim=0)
    weight_norm, full_norm = math.sqrt(math.fsum(weight_squares)), math.sqrt(math.fsum(full_squares))
    if not all(math.isfinite(x) and x > 0 for x in (weight_norm, full_norm)):
        raise ValueError("Full gradient or source weight norm is zero or non-finite")
    return {"global": gram_summary(total, torch_module),
            "blocks": [{"block": block, **gram_summary(gram, torch_module)}
                       for block, gram in enumerate(grams)]}, {
        "aggregate_source_weight_norm": weight_norm, "aggregate_generic_gradient_norm": full_norm,
    }


def tangent_scale(norms: dict[str, float], relative: float) -> dict[str, float]:
    w, g = norms["aggregate_source_weight_norm"], norms["aggregate_generic_gradient_norm"]
    if not all(math.isfinite(x) and x > 0 for x in (w, g, relative)):
        raise ValueError("Invalid global normalization")
    target = relative * w
    alpha = target / g
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid global alpha")
    return {**norms, "alpha": alpha, "target_delta_norm": target}


def make_tangent(store: SplitGradientStore, eligible: list[tuple[str, Any]], alpha: float,
                 torch_module: Any, *, split: int | None = None, block: int | None = None,
                 family: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One common positive alpha; omitted parameters are unchanged, not renormalized."""
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid common alpha")
    tangents, records = {}, []
    for name, module in eligible:
        layer, kind = parameter_group(name)
        if (block is not None and block != layer) or (family is not None and family != kind):
            continue
        gradient = store.mean(name, module.weight.shape) if split is None else store.matrix(split, name, module.weight.shape)
        value = (alpha * gradient).to(dtype=torch_module.float32)
        if not bool(torch_module.isfinite(value).all().item()):
            raise ValueError("Non-finite realized tangent")
        records.append({"name": f"{name}.weight", "shape": list(value.shape),
                        "raw_gradient_frobenius_norm": float(gradient.norm()),
                        "tangent_frobenius_norm": float(value.double().norm())})
        tangents[f"{name}.weight"] = value.to(device=module.weight.device)
    if not tangents:
        raise ValueError("Empty tangent group")
    return tangents, records


def compute_probe_response(model: Any, probe: Any, tangents: dict[str, Any], spec: dict[str, Any],
                           torch_module: Any, label: str = "response") -> tuple[Any, Any, dict[str, Any]]:
    settings = spec["diagnostic_probe"]
    if tuple(probe.shape) != (settings["sample_count"], spec["probe"]["sequence_length"]):
        raise ValueError("Diagnostic probe shape mismatch")
    model.eval()
    device = next(model.parameters()).device
    accumulator = ResponseAccumulator(spec, torch_module)
    discrepancy = None
    total_batches = math.ceil(settings["sample_count"] / settings["batch_size"])
    for batch_index, start in enumerate(range(0, settings["sample_count"], settings["batch_size"]), 1):
        tokens = probe[start:start + settings["batch_size"]].to(device)
        ordinary = ordinary_hook_readout(model, tokens, spec, torch_module) if start == 0 else None
        primal, response = run_readout_jvp(model, tokens, tangents, spec, torch_module)
        if ordinary is not None:
            discrepancy = compare_primal(primal, ordinary, spec, torch_module)
        accumulator.add(primal, response)
        if batch_index % 16 == 0 or batch_index == total_batches:
            print(f"{label}: processed {batch_index}/{total_batches} probe batches", flush=True)
        del ordinary, primal, response
    merged, response = accumulator.means(settings["sample_count"])
    return merged, response, {"readout_hook_verified": True, "first_batch_primal_max_abs_difference": discrepancy}


def response_comparison(full: Any, candidate: Any, torch_module: Any) -> dict[str, float | None]:
    """CPU FP64 comparison; undefined zero-reference ratios are JSON null."""
    if (not isinstance(full, torch_module.Tensor) or not isinstance(candidate, torch_module.Tensor) or
        full.device.type != "cpu" or candidate.device.type != "cpu" or
        full.dtype != torch_module.float64 or candidate.dtype != torch_module.float64 or
        full.ndim != 2 or full.numel() == 0 or full.shape != candidate.shape or
        not bool(torch_module.isfinite(full).all()) or not bool(torch_module.isfinite(candidate).all())):
        raise ValueError("Invalid float64 decomposition inputs")
    error = candidate - full
    rms = float(error.square().mean().sqrt())
    reference_rms = float(full.square().mean().sqrt())
    maximum = float(error.abs().max())
    reference_max = float(full.abs().max())
    metrics = {
        "cosine_similarity": cosine(full.reshape(-1), candidate.reshape(-1)),
        "rms_difference": rms,
        "relative_rms_difference": rms / reference_rms if reference_rms else (0.0 if rms == 0 else None),
        "maximum_absolute_difference": maximum,
        "relative_max_difference": maximum / reference_max if reference_max else (0.0 if maximum == 0 else None),
    }
    if any(value is not None and not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite response comparison metrics")
    return metrics


def consistency_metrics(full: Any, candidate: Any, settings: dict[str, Any], torch_module: Any) -> dict[str, Any]:
    """Every token position must pass; global magnitude never masks a local error."""
    global_metrics = response_comparison(full, candidate, torch_module)
    positions = []
    for position in range(full.shape[0]):
        reference = full[position:position + 1]
        metrics = response_comparison(reference, candidate[position:position + 1], torch_module)
        for metric, threshold in (("relative_rms_difference", "maximum_relative_rms_difference"),
                                  ("relative_max_difference", "maximum_relative_max_difference")):
            value = metrics[metric]
            if value is None or value > settings[threshold]:
                raise ValueError(f"Response decomposition inconsistent at position {position}: {metric}={value}")
        positions.append({"position": position, **metrics,
                          "reference_rms": float(reference.square().mean().sqrt()),
                          "reference_max_absolute_value": float(reference.abs().max())})
    # max returns the first occurrence, freezing ties to the lowest position.
    worst_rms = max(positions, key=lambda row: row["relative_rms_difference"])
    worst_max = max(positions, key=lambda row: row["relative_max_difference"])
    return {
        "positions": positions,
        "worst_relative_rms_difference": worst_rms["relative_rms_difference"],
        "worst_relative_rms_position": worst_rms["position"],
        "worst_relative_max_difference": worst_max["relative_max_difference"],
        "worst_relative_max_position": worst_max["position"],
        "maximum_absolute_difference": global_metrics["maximum_absolute_difference"],
    }


def partition_tangent(canonical: dict[str, Any], *, block: int | None = None,
                      family: str | None = None) -> dict[str, Any]:
    """Reuse the exact FP32 tensor objects supplied to the direct full JVP."""
    if (block is None) == (family is None):
        raise ValueError("Select exactly one block or family")
    selected = {}
    for name, value in canonical.items():
        layer, kind = parameter_group(name)
        if (block is not None and layer == block) or (family is not None and kind == family):
            selected[name] = value
    if not selected:
        raise ValueError("Empty tangent partition")
    return selected


def build_artifact(merged: Any, full: Any, splits: Any, blocks: Any, families: Any, spec: dict[str, Any],
                   torch_module: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    shape = (spec["probe"]["sequence_length"], spec["readout"]["hidden_size"])
    values = {"merged_mean": merged, "full_response": full, "split_responses": splits,
              "block_responses": blocks, "family_responses": families}
    shapes = {"merged_mean": shape, "full_response": shape, "split_responses": (4, *shape),
              "block_responses": (14, *shape), "family_responses": (2, *shape)}
    for name, value in values.items():
        if (not isinstance(value, torch_module.Tensor) or
            value.device.type != "cpu" or value.dtype != torch_module.float64 or
            tuple(value.shape) != shapes[name] or not bool(torch_module.isfinite(value).all())):
            raise ValueError(f"Invalid CPU float64 response: {name}")
    checks = {"block_sum": consistency_metrics(full, blocks.sum(dim=0), spec["decomposition"]["consistency"], torch_module),
              "family_sum": consistency_metrics(full, families.sum(dim=0), spec["decomposition"]["consistency"], torch_module)}
    checks["split_mean_vs_direct_full"] = response_comparison(full, splits.mean(dim=0), torch_module)
    artifact = {name: values[name].to(dtype=torch_module.float32).contiguous()
                for name in spec["outputs"]["tensors"]}
    if any(not bool(torch_module.isfinite(value).all()) for value in artifact.values()):
        raise ValueError("Non-finite stored response")
    artifact["metadata"] = {"format_version": 1, "attempt_id": spec["attempt_id"],
        "response_sign": "positive_J_delta", "full_response_definition": spec["decomposition"]["full_response"],
        "split_mean_role": "diagnostic_only", "split_order": spec["splits"]["order"],
        "block_order": spec["decomposition"]["block_order"], "family_order": spec["decomposition"]["family_order"],
        "readout": spec["readout"], "diagnostic_probe": spec["diagnostic_probe"],
        "accumulator_dtype": "cpu_float64", "stored_dtype": "float32"}
    return artifact, checks


def cosine(a: Any, b: Any) -> float | None:
    na, nb = float(a.norm()), float(b.norm())
    return max(-1.0, min(1.0, float(a @ b) / (na * nb))) if na and nb else None


def descriptive(values: list[float | None]) -> dict[str, Any]:
    finite = [value for value in values if value is not None]
    return {"count": len(finite), "undefined_count": len(values) - len(finite),
            "mean": math.fsum(finite) / len(finite) if finite else None,
            "min": min(finite) if finite else None, "max": max(finite) if finite else None}


def spectral_statistics(matrix: Any, torch_module: Any) -> dict[str, Any]:
    if (matrix.dtype != torch_module.float64 or matrix.device.type != "cpu" or
        not bool(torch_module.isfinite(matrix).all())):
        raise ValueError("SVD inputs must be finite CPU float64")
    singular = torch_module.linalg.svdvals(matrix)
    energy = singular.square()
    total = float(energy.sum())
    fractions = energy / total if total else torch_module.zeros_like(energy)
    rank = total * total / float(energy.square().sum()) if total else 0.0
    return {"singular_values": singular.tolist(), "squared_singular_value_fractions": fractions.tolist(),
            "top1_variance_fraction": float(fractions[0]), "participation_ratio_effective_rank": rank}


def activation_diagnostics(artifact: dict[str, Any], torch_module: Any) -> dict[str, Any]:
    tensors = {key: artifact[key].to(device="cpu", dtype=torch_module.float64)
               for key in ("full_response", "split_responses", "block_responses", "family_responses")}
    records, pairs_by_position, norms_by_position = [], [], []
    for position in range(tensors["full_response"].shape[0]):
        full = tensors["full_response"][position]
        split = tensors["split_responses"][:, position]
        block = tensors["block_responses"][:, position]
        family = tensors["family_responses"][:, position]
        split_stats = gram_summary(split @ split.T, torch_module)
        block_stats = gram_summary(block @ block.T, torch_module)
        family_stats = gram_summary(family @ family.T, torch_module)
        pairs = [split_stats["cosine_matrix"][i][j] for i in range(4) for j in range(i + 1, 4)]
        pairs_by_position.append(pairs)
        norms_by_position.append(split_stats["norms"])
        def cancellation(norms):
            denominator = math.fsum(norms)
            return float(full.norm()) / denominator if denominator else None
        records.append({"position": position,
            "split": {**split_stats, "off_diagonal_pairs": descriptive(pairs)},
            "block": {**block_stats, "cosine_with_full": [cosine(row, full) for row in block],
                      "leave_one_block_out_cosine": [cosine(full, full - row) for row in block],
                      "cancellation_ratio": cancellation(block_stats["norms"])},
            "family": {"norms": family_stats["norms"], "mutual_cosine": cosine(family[0], family[1]),
                       "cosine_with_full": [cosine(row, full) for row in family],
                       "cancellation_ratio": cancellation(family_stats["norms"])},
            "effective_dimensionality": {"split": spectral_statistics(split, torch_module),
                                         "block": spectral_statistics(block, torch_module)}})
    return {"positions": records, "split_descriptive_summaries": {
        "position_0": {"pairwise_cosines": descriptive(pairs_by_position[0]),
                       "norms": descriptive(norms_by_position[0])},
        "positions_1_through_127": {
            "pairwise_cosines": descriptive([value for row in pairs_by_position[1:] for value in row]),
            "norms": descriptive([value for row in norms_by_position[1:] for value in row])}}}


def build_manifest(spec: dict[str, Any], hashes: dict[str, str], source_files: list[dict[str, Any]],
                   corpus: dict[str, Any], losses: list[float], scale: dict[str, float],
                   matrix_records: list[dict[str, Any]], weight_statistics: dict[str, Any],
                   readout_checks: list[dict[str, Any]], consistency: dict[str, Any],
                   diagnostics: dict[str, Any], artifact_hash: str, raw_hashes: dict[str, str]) -> dict[str, Any]:
    return {"format_version": 1, "attempt_id": spec["attempt_id"], "hash_algorithm": "sha256",
        "purpose": spec["purpose"], "information_policy": spec["information_policy"], **hashes,
        "source_checkpoint": {"files": source_files, "file_count": len(source_files),
                              "total_bytes": sum(record["size_bytes"] for record in source_files)},
        "corpus": {key: corpus[key] for key in ("serialized_sha256", "raw_tensor_sha256")},
        "probe": spec["probe"], "diagnostic_probe": spec["diagnostic_probe"],
        "loss": {**spec["generic_loss"], "splits": spec["splits"],
                 "split_mean_losses": losses, "mean_generic_loss": math.fsum(losses) / 4},
        "tangent": {**spec["tangent"], **scale},
        "eligible_parameters": [record["name"] for record in matrix_records],
        "matrices": matrix_records, "jvp": spec["jvp"],
        "readout": {**spec["readout"], "checks": readout_checks},
        "decomposition": {**spec["decomposition"], "measured_consistency": consistency},
        "weight_space_split_statistics": weight_statistics,
        "activation_space_diagnostics": diagnostics,
        "artifact": {"serialized_sha256": artifact_hash, "raw_tensors_sha256": raw_hashes}}


def construct(args: argparse.Namespace) -> None:
    require_output_absent(args.artifact_path, args.construction_manifest_path)
    if args.temporary_directory.is_symlink() or (args.temporary_directory.exists() and
        (not args.temporary_directory.is_dir() or any(args.temporary_directory.iterdir()))):
        raise ValueError("Attempt009 temporary directory contains stale data; no automatic resume")
    spec = load_spec(args.attempt_spec_path)
    paths = [args.attempt_spec_path, args.corpus_spec_path, args.corpus_manifest_path,
             args.tokens_path, args.probe_path, Path(__file__).resolve()]
    for output in (args.artifact_path, args.construction_manifest_path, args.temporary_directory):
        if (output.resolve().is_relative_to(args.merged_model_dir.resolve()) or
            any(output.resolve() == path.resolve() or path.resolve().is_relative_to(output.resolve()) for path in paths)):
            raise ValueError("Outputs and temporary data must not overlap frozen inputs")
    if (args.artifact_path.resolve() == args.construction_manifest_path.resolve() or
        any(output.resolve().is_relative_to(args.temporary_directory.resolve()) for output in
            (args.artifact_path, args.construction_manifest_path))):
        raise ValueError("Scientific outputs must be separate from temporary storage")
    hashes = {path: sha256_file(path) for path in paths}
    # Close the read/hash race for the specification.
    if load_spec(args.attempt_spec_path) != spec:
        raise ValueError("Attempt specification changed while loading")
    for key, path in (("corpus_spec_sha256", args.corpus_spec_path),
                      ("corpus_manifest_sha256", args.corpus_manifest_path)):
        if hashes[path] != spec["frozen_corpus"][key]:
            raise ValueError(f"Frozen corpus provenance mismatch: {key}")
    source_files = checkpoint_file_records(args.merged_model_dir)
    if source_files != spec["canonical_checkpoint_files"]:
        raise ValueError("Canonical merged checkpoint inventory/hash mismatch")
    corpus_spec = load_corpus_spec(args.corpus_spec_path)
    corpus = verify_corpus_manifest(args.corpus_manifest_path, args.corpus_spec_path,
                                   args.tokens_path, args.merged_model_dir, corpus_spec)
    for key in ("serialized_sha256", "raw_tensor_sha256"):
        if corpus[key] != spec["frozen_corpus"][key]:
            raise ValueError(f"Frozen corpus content mismatch: {key}")
    import torch

    probe = load_probe(args.probe_path, spec["probe"], torch)
    probe = probe[spec["diagnostic_probe"]["start"]:spec["diagnostic_probe"]["stop"]].contiguous()
    tokens = load_corpus_tokens(args.tokens_path, corpus["raw_tensor_sha256"],
                               spec["generic_loss"]["sample_count"], spec["generic_loss"]["sequence_length"], torch)
    prepare_temporary_directory(args.temporary_directory)
    model = load_local_model(args.merged_model_dir, args.device, torch)
    if getattr(model.config, "hidden_size", None) != spec["readout"]["hidden_size"]:
        raise ValueError("Canonical model hidden size mismatch")
    eligible = discover_eligible_linear_weights(model, torch)
    selected = freeze_other_parameters(model, eligible)
    before = model_state_hashes(model, torch)
    frozen_before = frozen_parameter_hashes(model, selected, torch)
    losses = []
    for k, split_tokens in enumerate(interleaved_splits(tokens)):
        temporary_stage(args.temporary_directory, f"split_gradient_{k}")
        print(f"Accumulating split gradient {k + 1}/4", flush=True)
        losses.append(accumulate_mean_generic_gradient(model, split_tokens, eligible, split_loss_spec(spec), torch))
        save_split_gradient(args.temporary_directory / f"split_{k}.safetensors", eligible, torch)
        verify_frozen_parameters(model, selected, frozen_before, torch)
        print(f"Completed split gradient {k + 1}/4", flush=True)
    model.zero_grad(set_to_none=True)
    readout_checks, splits, blocks, families, matrix_records = [], [], [], [], []
    merged = None
    direct_responses = []
    with SplitGradientStore(args.temporary_directory, eligible, torch) as store:
        temporary_stage(args.temporary_directory, "weight_space_statistics")
        weight_statistics, norms = weight_space_statistics(store, eligible, torch)
        scale = tangent_scale(norms, spec["tangent"]["target_relative_frobenius"])
        passes = [(f"split {k}", {"split": k}, splits) for k in range(4)]
        passes += [("full", {}, direct_responses)]
        passes += [(f"block {block}", {"block": block}, blocks) for block in spec["decomposition"]["block_order"]]
        passes += [(family, {"family": family}, families) for family in spec["decomposition"]["family_order"]]
        for label, selection, responses in passes:
            temporary_stage(args.temporary_directory, f"jvp_{label}")
            print(f"Starting {label} response JVP", flush=True)
            if label == "full":
                canonical, matrix_records = make_tangent(store, eligible, scale["alpha"], torch)
                tangents = canonical
            elif "split" in selection:
                tangents, _ = make_tangent(store, eligible, scale["alpha"], torch, **selection)
            else:
                tangents = partition_tangent(canonical, **selection)
            primal, response, readout = compute_probe_response(model, probe, tangents, spec, torch, label)
            del tangents
            if merged is None:
                merged = primal
            elif not torch.equal(merged, primal):
                raise ValueError("Primal mean changed across tangent selections")
            responses.append(response)
            readout_checks.append({"response": label, **readout})
    del canonical
    matrix_records.sort(key=lambda record: record["name"])
    realized = math.sqrt(math.fsum(record["tangent_frobenius_norm"] ** 2 for record in matrix_records))
    if not math.isfinite(realized) or realized <= 0:
        raise ValueError("Invalid realized full tangent norm")
    scale.update(aggregate_realized_fp32_tangent_norm=realized,
                 aggregate_realized_relative_tangent_norm=realized / scale["aggregate_source_weight_norm"])
    artifact, consistency = build_artifact(merged, direct_responses[0], torch.stack(splits), torch.stack(blocks),
                                           torch.stack(families), spec, torch)
    diagnostics = activation_diagnostics(artifact, torch)
    verify_frozen_parameters(model, selected, frozen_before, torch)
    verify_model_unchanged(model, before, torch)
    verify_unchanged(args.merged_model_dir, source_files, hashes)
    temporary_stage(args.temporary_directory, "publishing")
    require_output_absent(args.artifact_path, args.construction_manifest_path)
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    args.construction_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with args.artifact_path.open("xb") as stream:
        torch.save(artifact, stream)
    verify_unchanged(args.merged_model_dir, source_files, hashes)
    manifest = build_manifest(spec, {
        "attempt_spec_sha256": hashes[args.attempt_spec_path],
        "corpus_spec_sha256": hashes[args.corpus_spec_path],
        "corpus_manifest_sha256": hashes[args.corpus_manifest_path],
        "constructor_script_sha256": hashes[Path(__file__).resolve()],
    }, source_files, corpus, losses, scale, matrix_records, weight_statistics, readout_checks,
        consistency, diagnostics, sha256_file(args.artifact_path),
        {name: sha256_raw_float32_tensor(artifact[name], torch) for name in spec["outputs"]["tensors"]})
    write_manifest(args.construction_manifest_path, manifest)
    cleanup_temporary_directory(args.temporary_directory)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for option, default in (
        ("merged-model-dir", DEFAULT_MODEL_DIR), ("attempt-spec-path", DEFAULT_SPEC_PATH),
        ("corpus-spec-path", DEFAULT_CORPUS_SPEC_PATH), ("corpus-manifest-path", DEFAULT_CORPUS_MANIFEST_PATH),
        ("tokens-path", DEFAULT_TOKENS_PATH), ("probe-path", DEFAULT_PROBE_PATH),
        ("artifact-path", DEFAULT_ARTIFACT_PATH), ("construction-manifest-path", DEFAULT_CONSTRUCTION_MANIFEST_PATH),
        ("temporary-directory", Path(FROZEN_SPEC["defaults"]["temporary_directory"]))):
        parser.add_argument(f"--{option}", type=Path, default=default)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser())
    try:
        construct(args)
    except (OSError, RuntimeError, ValueError, NotImplementedError) as exc:
        print(f"ERROR: {exc}. Temporary state, if created, is retained at {args.temporary_directory}; no automatic resume.", file=sys.stderr)
        return 1
    print(f"Wrote {args.artifact_path} and {args.construction_manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
