#!/usr/bin/env python3
"""Construct the blind Attempt 011 fixed k=4 true-Hessian Krylov diagnostic."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import sys
import traceback
from contextlib import ExitStack
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
FROZEN_SPEC = {'format_version': 1,
 'information_policy': 'final_checkpoint_plus_frozen_generic_corpus_and_probe',
 'defaults': {'merged_model_directory': '/root/model-diff-scratch/models/merged',
              'corpus_spec_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json',
              'corpus_manifest_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json',
              'corpus_artifact_path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
              'probe_path': '/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt',
              'artifact_path': '/root/model-diff-scratch/artifacts/attempt011_hessian_krylov_k4/krylov_probe.pt',
              'construction_manifest_path': 'experiments/attempts/011_generic_hessian_krylov_k4_prefix0_13/construction-manifest.json',
              'temporary_directory': '/root/model-diff-scratch/tmp/attempt011',
              'basis_directory': '/root/model-diff-scratch/artifacts/attempt011_hessian_krylov_k4'},
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
             'semantics': 'hidden_states[14] is block 13 output before block 14; verify with forward hook'},
 'jvp': {'implementation': 'torch.func.jvp_with_partial_torch.func.functional_call',
         'module': 'model.model',
         'output_hidden_states': True,
         'strict_forward_ad': True,
         'fallback': False,
         'change_attention_backend': False,
         'reverse_mode_recording': False,
         'response_sign': 'positive_Jv',
         'accumulation_dtype': 'cpu_float64',
         'stored_dtype': 'float32',
         'primal_comparison_rtol': 1e-05,
         'primal_comparison_atol': 1e-06},
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
 'frozen_corpus': {'corpus_spec_sha256': 'e514820e7def99e9287c32de4b1279f28f6ff6c2e09749a8dfa5d8b14c311f0f',
                   'corpus_manifest_sha256': '8b3ea49228dbb106c2b8a7c849c4f27e021fa3e992800444f43e260a6b055230',
                   'serialized_sha256': '54283d72aa3075e2867ead86e538991d7284ede327ca67871e73bfc581e4025d',
                   'raw_tensor_sha256': '4613982be77342a89b35616d68e21d9ff66d5a983af7e82cb6fa1eb159457eae'},
 'diagnostic_probe': {'start': 0, 'stop': 1024, 'sample_count': 1024, 'batch_size': 32},
 'splits': {'count': 4,
            'order': [0, 1, 2, 3],
            'selection': 'example_index_modulo_4',
            'sample_count': 1024,
            'total_prediction_tokens': 130048,
            'batch_size': 8,
            'number_of_batches': 128},
 'hessian': {'primitive': 'true_double_backward_grad_of_gradient_dot_detached_vector',
             'vector': 'same_detached_realized_FP32_q_j_for_every_batch_and_split_at_step_j',
             'batch_loss': 'sum_token_CE_divided_by_1024_times_127',
             'split_action': 'sum_of_128_batch_HVPs',
             'accumulation_dtype': 'float32',
             'accumulation_device': 'parameter_device',
             'full_action': 'CPU_float64_matrixwise_mean_of_four_stored_FP32_split_actions',
             'separate_full_corpus_HVP': False,
             'fallback': False},
 'attempt_id': '011_generic_hessian_krylov_k4_prefix0_13',
 'method': 'generic_true_hessian_krylov_k4',
 'purpose': 'blind_repeated_curvature_functional_direction_diagnostic',
 'krylov': {'dimension': 4,
            'first_basis': 'float32(g / CPU_float64_global_norm(g))',
            'full_action': 'CPU_float64_matrixwise_mean_of_four_FP32_split_actions',
            'projection': 'two_pass_classical_Gram_Schmidt_against_all_existing_realized_basis_vectors',
            'reorthogonalization_passes': 2,
            'split_projector': 'same_Q_j_and_same_two_pass_algorithm_no_split_normalization',
            'residual_arithmetic_dtype': 'cpu_float64',
            'next_basis': 'float32(r_j / CPU_float64_global_norm(r_j))',
            'beta': 'CPU_float64_global_norm(r_j)',
            'breakdown': 'fail_on_exact_zero_or_nonfinite_beta_at_any_step',
            'near_zero_cutoff': None,
            'basis_dtype': 'float32',
            'basis_order': [1, 2, 3, 4]},
 'orthogonality': {'dtype': 'cpu_float64',
                   'maximum_absolute_off_diagonal': 1e-05,
                   'maximum_diagonal_deviation_from_one': 1e-05},
 'projected_hessian': {'entries': 'q_i_dot_mean_split_H_q_j',
                       'construction': 'upper_triangle_i_le_j_mirrored_by_symmetry',
                       'audit': 'stream_direct_adjacent_cross_entries_before_action_deletion_and_record_off_tridiagonal_leakage',
                       'ritz_values': 'ascending_eigvalsh_of_symmetrized_B4_cpu_float64',
                       'ritz_residuals': 'beta_4_times_absolute_last_Ritz_eigenvector_coordinate_descriptive_estimates',
                       'off_tridiagonal_entries': 'descriptive_numerical_leakage_only'},
 'functional_tangents': {'first': 'canonical_detached_FP32_g_without_rescaling',
                         'others': 'float32(CPU_float64_norm_g * realized_FP32_q_j)',
                         'scaling_arithmetic_dtype': 'cpu_float64',
                         'sign': 'positive_Jv',
                         'coefficient_selection': False},
 'diagnostics': {'dtype': 'cpu_float64',
                 'responses': 'stored_equivalent_FP32_promoted_to_FP64',
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
                 'incremental_direction_counts': [1, 2, 3, 4],
                 'zero_cosine': None,
                 'zero_spectrum': 'zero_energy_fractions_top1_and_effective_rank',
                 'summaries': 'position0_and_positions1_through127_descriptive_no_selection',
                 'split_residual_mean_comparison': 'diagnostic_only_not_a_failure_gate'},
 'temporary_vectors': {'format': 'safetensors',
                       'dtype': 'float32',
                       'files': ['g.safetensors',
                                 'q1.safetensors',
                                 'q2.safetensors',
                                 'q3.safetensors',
                                 'q4.safetensors',
                                 'z1_s0.safetensors',
                                 'z1_s1.safetensors',
                                 'z1_s2.safetensors',
                                 'z1_s3.safetensors',
                                 'z2_s0.safetensors',
                                 'z2_s1.safetensors',
                                 'z2_s2.safetensors',
                                 'z2_s3.safetensors',
                                 'z3_s0.safetensors',
                                 'z3_s1.safetensors',
                                 'z3_s2.safetensors',
                                 'z3_s3.safetensors',
                                 'z4_s0.safetensors',
                                 'z4_s1.safetensors',
                                 'z4_s2.safetensors',
                                 'z4_s3.safetensors'],
                       'cleanup': 'delete_current_split_actions_after_durable_step_diagnostics_and_next_basis; '
                                  'delete_g_after_its_JVP; remove_small_state_after_publication',
                       'on_failure': 'retain_remaining_files_and_step_diagnostics_no_resume',
                       'nonempty_directory': 'refuse',
                       'maximum_simultaneous_parameter_vectors': 9,
                       'residual_vectors_persisted': False,
                       'full_action_vectors_persisted': False,
                       'basis_publication': 'same_filesystem_atomic_rename_no_copy_no_cross_filesystem_fallback',
                       'partial_publication': 'manifest_is_final_commit_marker; '
                                              'on_failure_retain_moved_final_and_remaining_temporary_files; '
                                              'refuse_rerun_until_manual_inspection'},
 'outputs': {'tensors': ['merged_mean', 'krylov_responses'],
             'basis_files': ['q1.safetensors', 'q2.safetensors', 'q3.safetensors', 'q4.safetensors'],
             'timestamps': False,
             'host_metadata': False,
             'gpu_metadata': False},
 'workload': {'gradient_batches': 512,
              'hvp_directions': 4,
              'hvp_batches_per_split': 128,
              'hvp_batches_total': 2048,
              'jvp_directions': 4,
              'probe_batches_per_direction': 32,
              'jvp_batches_total': 128}}

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

ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/011_generic_hessian_krylov_k4_prefix0_13"
DEFAULT_SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
DEFAULT_CORPUS_SPEC_PATH = PROJECT / FROZEN_SPEC["defaults"]["corpus_spec_path"]
DEFAULT_CORPUS_MANIFEST_PATH = PROJECT / FROZEN_SPEC["defaults"]["corpus_manifest_path"]
DEFAULT_MODEL_DIR = Path(FROZEN_SPEC["defaults"]["merged_model_directory"])
DEFAULT_TOKENS_PATH = Path(FROZEN_SPEC["defaults"]["corpus_artifact_path"])
DEFAULT_PROBE_PATH = Path(FROZEN_SPEC["defaults"]["probe_path"])
DEFAULT_ARTIFACT_PATH = Path(FROZEN_SPEC["defaults"]["artifact_path"])
DEFAULT_CONSTRUCTION_MANIFEST_PATH = PROJECT / FROZEN_SPEC["defaults"]["construction_manifest_path"]
TOKENIZER_FILE_NAMES = ("chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json")
HVP_SOURCE_PATH = Path(__file__).with_name("smoke_test_generic_hessian_curvature_probe.py")
_hvp_spec = importlib.util.spec_from_file_location("attempt010_hvp_implementation", HVP_SOURCE_PATH)
hvp_backend = importlib.util.module_from_spec(_hvp_spec)
_hvp_spec.loader.exec_module(hvp_backend)

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
        raise ValueError("Attempt011 temporary directory must be absent or empty; stale data cannot be reused")
    path.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents two constructors sharing an empty directory.
    with (path / "state.json").open("x", encoding="utf-8") as stream:
        json.dump({"stage": "started", "resume_permitted": False}, stream)


def temporary_stage(path: Path, stage: str) -> None:
    (path / "state.json").write_text(json.dumps({"stage": stage, "resume_permitted": False}) + "\n")


def gram_summary(gram: Any, torch_module: Any) -> dict[str, Any]:
    if gram.dtype != torch_module.float64 or not bool(torch_module.isfinite(gram).all().item()):
        raise ValueError("Gram statistics must be finite float64")
    norms = gram.diag().clamp_min(0).sqrt().tolist()
    cosine = [[max(-1.0, min(1.0, float(gram[i, j]) / (norms[i] * norms[j])))
               if norms[i] and norms[j] else None for j in range(len(norms))]
              for i in range(len(norms))]
    return {"norms": norms, "cosine_matrix": cosine}


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


def cosine(a: Any, b: Any) -> float | None:
    na, nb = float(a.norm()), float(b.norm())
    return max(-1.0, min(1.0, float(a @ b) / (na * nb))) if na and nb else None


def descriptive(values: list[float | None]) -> dict[str, Any]:
    finite = [value for value in values if value is not None]
    return {"count": len(finite), "undefined_count": len(values) - len(finite),
            "mean": math.fsum(finite) / len(finite) if finite else None,
            "min": min(finite) if finite else None, "max": max(finite) if finite else None}


LOADED_HVP_SOURCE_SHA256 = sha256_file(HVP_SOURCE_PATH)


def detach_canonical_gradient(model: Any, eligible: list[tuple[str, Any]]) -> dict[str, Any]:
    parameters = {f"{name}.weight": module.weight for name, module in eligible}
    direction = {name: parameter.grad.detach() if parameter.grad is not None else None
                 for name, parameter in parameters.items()}
    hvp_backend.validate_tensor_structure(parameters, direction, detached=True, description="Detached direction")
    model.zero_grad(set_to_none=True)
    return direction


def save_vector(path: Path, vector: dict[str, Any], parameters: dict[str, Any], torch_module: Any) -> None:
    from safetensors.torch import save_file
    if path.exists() or path.is_symlink():
        raise ValueError("Temporary vector already exists")
    hvp_backend.validate_tensor_structure(parameters, vector, detached=True, description="Temporary vector")
    if any(value.dtype != torch_module.float32 for value in vector.values()):
        raise ValueError("Temporary vectors must be float32")
    cpu_values = {name: value.detach().to(device="cpu").contiguous() for name, value in vector.items()}
    save_file(cpu_values, str(path))


def accumulate_hessian_split(model: Any, tokens: Any, eligible: list[tuple[str, Any]],
                             current_direction: dict[str, Any], settings: dict[str, Any], torch_module: Any,
                             progress: Any = None) -> dict[str, Any]:
    """Sum exact batch Hessian actions on the same detached current basis direction."""
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


class VectorStore:
    """Lazily memory-map temporary vectors; promote only individual matrices."""
    def __init__(self, directory: Path, eligible: list[tuple[str, Any]], torch_module: Any):
        self.directory, self.eligible, self.torch = directory, eligible, torch_module
        self.files, self.contexts = {}, {}
        self.names = {f"{name}.weight" for name, _ in eligible}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for context in self.contexts.values():
            context.close()
        self.files.clear()
        self.contexts.clear()

    def matrix(self, key: str, name: str, shape: Any) -> Any:
        from safetensors import safe_open
        if key not in self.files:
            context = ExitStack()
            self.contexts[key] = context
            file = context.enter_context(safe_open(str(self.directory / f"{key}.safetensors"), framework="pt", device="cpu"))
            if set(file.keys()) != self.names:
                raise ValueError(f"Temporary vector keys mismatch: {key}")
            self.files[key] = file
        value = self.files[key].get_tensor(f"{name}.weight")
        if (value.dtype != self.torch.float32 or tuple(value.shape) != tuple(shape) or
            not bool(self.torch.isfinite(value).all())):
            raise ValueError(f"Invalid temporary vector: {key}, {name}")
        return value.to(dtype=self.torch.float64)

    def remove(self, key: str) -> None:
        """Release mappings before unlink so disk blocks are reclaimed immediately."""
        self.files.pop(key, None)
        context = self.contexts.pop(key, None)
        if context is not None:
            context.close()
        (self.directory / f"{key}.safetensors").unlink()

    def action(self, step: int, split: int | None, name: str, shape: Any) -> Any:
        if split is not None:
            return self.matrix(f"z{step}_s{split}", name, shape)
        result = self.matrix(f"z{step}_s0", name, shape)
        for k in range(1, 4):
            result.add_(self.matrix(f"z{step}_s{k}", name, shape))
        return result.div_(4)


def vector_norm(store: VectorStore, key: str, eligible: list[tuple[str, Any]]) -> float:
    squares = []
    for name, module in eligible:
        value = store.matrix(key, name, module.weight.shape)
        squares.append(float(value.square().sum()))
    norm = math.sqrt(math.fsum(squares))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"Vector norm must be positive and finite: {key}")
    return norm


def save_cpu_basis(path: Path, eligible: list[tuple[str, Any]], matrix_function: Any,
                   denominator: float, torch_module: Any) -> None:
    from safetensors.torch import save_file
    if path.exists() or path.is_symlink():
        raise ValueError(f"Basis file already exists: {path}")
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Krylov breakdown: normalization norm is exactly zero or non-finite")
    values = {}
    for name, module in eligible:
        matrix = matrix_function(name, module.weight.shape)
        if matrix.dtype != torch_module.float64 or matrix.device.type != "cpu" or matrix.shape != module.weight.shape:
            raise ValueError("Basis normalization requires matching CPU float64 matrices")
        value = (matrix / denominator).to(dtype=torch_module.float32).contiguous()
        if not bool(torch_module.isfinite(value).all()):
            raise ValueError("Non-finite realized FP32 basis matrix")
        values[f"{name}.weight"] = value
    save_file(values, str(path))


def load_device_vector(store: VectorStore, key: str, eligible: list[tuple[str, Any]],
                       torch_module: Any, scale: float = 1.0) -> tuple[dict[str, Any], float]:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid positive functional scale")
    values, squares = {}, []
    for name, module in eligible:
        value = (store.matrix(key, name, module.weight.shape) * scale).to(dtype=torch_module.float32).contiguous()
        if not bool(torch_module.isfinite(value).all()):
            raise ValueError("Non-finite tangent")
        squares.append(float(value.double().square().sum()))
        values[f"{name}.weight"] = value.to(device=module.weight.device)
    return values, math.sqrt(math.fsum(squares))


def residual_matrix(store: VectorStore, step: int, split: int | None, name: str, shape: Any,
                    passes: list[Any]) -> Any:
    """Apply the same two full-basis projection passes to any one action."""
    index = 0 if split is None else split + 1
    result = store.action(step, split, name, shape)
    basis = [store.matrix(f"q{i}", name, shape) for i in range(1, step + 1)]
    for coefficients in passes:
        for i, q in enumerate(basis):
            result.add_(q, alpha=-float(coefficients[index, i]))
    return result


def reorthogonalization_coefficients(store: VectorStore, step: int, eligible: list[tuple[str, Any]],
                                    torch_module: Any) -> list[Any]:
    """Two classical Gram-Schmidt passes, each against all q1..qj.

    Every coefficient is a GLOBAL CPU FP64 dot, never a per-matrix projection.
    Rows correspond to the full action and four split actions, in that order.
    """
    passes = []
    for _ in range(2):
        coefficients = torch_module.zeros((5, step), dtype=torch_module.float64, device="cpu")
        for name, module in eligible:
            basis = [store.matrix(f"q{i}", name, module.weight.shape) for i in range(1, step + 1)]
            for row, split in enumerate((None, 0, 1, 2, 3)):
                residual = residual_matrix(store, step, split, name, module.weight.shape, passes)
                for i, q in enumerate(basis):
                    coefficients[row, i] += torch_module.dot(q.reshape(-1), residual.reshape(-1))
        if not bool(torch_module.isfinite(coefficients).all()):
            raise ValueError(f"Non-finite reorthogonalization coefficients at step {step}")
        passes.append(coefficients)
    return passes


def action_and_residual_statistics(store: VectorStore, step: int, eligible: list[tuple[str, Any]],
                                   passes: list[Any], torch_module: Any) -> dict[str, Any]:
    grams = [torch_module.zeros((5, 5), dtype=torch_module.float64, device="cpu") for _ in range(2)]
    error_squares, mean_squares = [], []
    max_error = max_reference = 0.0
    leakage = torch_module.zeros(step, dtype=torch_module.float64, device="cpu")
    for name, module in eligible:
        raw = [store.action(step, split, name, module.weight.shape) for split in (None, 0, 1, 2, 3)]
        residuals = [residual_matrix(store, step, split, name, module.weight.shape, passes)
                     for split in (None, 0, 1, 2, 3)]
        for gram, values in zip(grams, (raw, residuals)):
            for i in range(5):
                for j in range(i, 5):
                    dot = torch_module.dot(values[i].reshape(-1), values[j].reshape(-1))
                    gram[i, j] += dot
                    if i != j:
                        gram[j, i] += dot
        mean = sum(residuals[1:]) / 4
        error = residuals[0] - mean
        error_squares.append(float(error.square().sum()))
        mean_squares.append(float(mean.square().sum()))
        max_error = max(max_error, float(error.abs().max()))
        max_reference = max(max_reference, float(residuals[0].abs().max()))
        for i in range(step):
            q = store.matrix(f"q{i + 1}", name, module.weight.shape)
            leakage[i] += torch_module.dot(q.reshape(-1), residuals[0].reshape(-1))
    def summarize(gram):
        summary = gram_summary(gram, torch_module)
        return {"full_norm": summary["norms"][0], "split_norms": summary["norms"][1:],
                "full_then_splits_gram_matrix": gram.tolist(),
                "split_gram_matrix": gram[1:, 1:].tolist(),
                "split_cosine_matrix": [row[1:] for row in summary["cosine_matrix"][1:]],
                "split_cosines_with_full": [row[0] for row in summary["cosine_matrix"][1:]]}
    raw, residuals = [summarize(gram) for gram in grams]
    beta = residuals["full_norm"]
    if not math.isfinite(beta) or beta == 0:
        raise ValueError(f"Krylov breakdown at step {step}: beta is exactly zero or non-finite")
    error_norm = math.sqrt(math.fsum(error_squares))
    denominator = math.fsum(residuals["split_norms"]) / 4
    residuals.update(full_to_mean_split_norm_ratio=beta / denominator if denominator else None,
        mean_split_residual_norm=math.sqrt(math.fsum(mean_squares)),
        full_minus_mean_split_residual_norm=error_norm,
        relative_mean_residual_difference=error_norm / beta,
        maximum_absolute_mean_residual_difference=max_error,
        relative_maximum_mean_residual_difference=max_error / max_reference if max_reference else None,
        basis_inner_products_after_two_passes=leakage.tolist())
    return {"step": step, "beta": beta, "raw_actions": raw, "residuals": residuals,
            "projection_coefficients_by_pass_full_then_splits": [value.tolist() for value in passes],
            "available_projected_hessian_column": passes[0][0].tolist()}


def basis_gram_statistics(store: VectorStore, eligible: list[tuple[str, Any]], settings: dict[str, Any],
                          torch_module: Any) -> tuple[Any, dict[str, Any]]:
    gram = torch_module.zeros((4, 4), dtype=torch_module.float64, device="cpu")
    for name, module in eligible:
        basis = [store.matrix(f"q{i}", name, module.weight.shape) for i in range(1, 5)]
        for i in range(4):
            for j in range(i, 4):
                dot = torch_module.dot(basis[i].reshape(-1), basis[j].reshape(-1))
                gram[i, j] += dot
                if i != j:
                    gram[j, i] += dot
    if not bool(torch_module.isfinite(gram).all()):
        raise ValueError("Non-finite basis Gram matrix")
    off = gram - torch_module.diag(gram.diag())
    maximum_off = float(off.abs().max())
    maximum_diagonal = float((gram.diag() - 1).abs().max())
    if (maximum_off > settings["maximum_absolute_off_diagonal"] or
        maximum_diagonal > settings["maximum_diagonal_deviation_from_one"]):
        raise ValueError(f"Basis orthogonality failed: off_diagonal={maximum_off}, diagonal_deviation={maximum_diagonal}")
    return gram, {"gram_matrix": gram.tolist(), "realized_norms": gram.diag().sqrt().tolist(),
                  "maximum_absolute_off_diagonal": maximum_off,
                  "maximum_diagonal_deviation_from_one": maximum_diagonal}


def projected_hessian_statistics(store: VectorStore, eligible: list[tuple[str, Any]], steps: list[dict[str, Any]],
                                 torch_module: Any) -> dict[str, Any]:
    b = torch_module.zeros((4, 4), dtype=torch_module.float64, device="cpu")
    for j, step in enumerate(steps):
        for i, value in enumerate(step["available_projected_hessian_column"]):
            b[i, j] = b[j, i] = value
    if not bool(torch_module.isfinite(b).all()):
        raise ValueError("Non-finite projected Hessian")
    eigenvalues, eigenvectors = torch_module.linalg.eigh((b + b.T) / 2)
    # q_(j+1) exists before step-j actions are deleted. This measures both
    # sides of each adjacent entry directly, without retaining any old action.
    adjacent = [step["next_basis_dot_full_action"] for step in steps[:-1]]
    errors = [abs(value - float(b[j, j + 1])) for j, value in enumerate(adjacent)]
    skew = max(errors)
    reference = float(b.abs().max())
    off_band = max(abs(float(b[i, j])) for i in range(4) for j in range(4) if abs(i - j) > 1)
    return {"B4": b.tolist(), "direct_lower_adjacent_entries": adjacent,
        "maximum_absolute_adjacent_asymmetry": skew,
        "relative_maximum_adjacent_asymmetry": skew / reference if reference else None,
        "maximum_absolute_off_tridiagonal_entry": off_band,
        "ritz_values": eigenvalues.tolist(), "beta_4": steps[-1]["beta"],
        "last_step_only_ritz_residual_estimates": (steps[-1]["beta"] * eigenvectors[-1].abs()).tolist()}


def build_krylov_basis(store: VectorStore, eligible: list[tuple[str, Any]], compute_split_actions: Any,
                       spec: dict[str, Any], torch_module: Any) -> tuple[float, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    gradient_norm = vector_norm(store, "g", eligible)
    save_cpu_basis(store.directory / "q1.safetensors", eligible,
                   lambda name, shape: store.matrix("g", name, shape), gradient_norm, torch_module)
    steps = []
    for step in range(1, 5):
        # The callback receives one fixed q_j; all four split HVPs use that vector.
        compute_split_actions(step, f"q{step}")
        passes = reorthogonalization_coefficients(store, step, eligible, torch_module)
        statistics = action_and_residual_statistics(store, step, eligible, passes, torch_module)
        steps.append(statistics)
        if step < 4:
            save_cpu_basis(store.directory / f"q{step + 1}.safetensors", eligible,
                lambda name, shape: residual_matrix(store, step, None, name, shape, passes),
                statistics["beta"], torch_module)
            adjacent = torch_module.zeros((), dtype=torch_module.float64)
            for name, module in eligible:
                adjacent += torch_module.dot(
                    store.matrix(f"q{step + 1}", name, module.weight.shape).reshape(-1),
                    store.action(step, None, name, module.weight.shape).reshape(-1))
            statistics["next_basis_dot_full_action"] = float(adjacent)
        # Small durable diagnostics survive subsequent failures; no residual or
        # full-action parameter vectors are serialized. Temporary hashes are
        # deliberately not required long-term provenance artifacts.
        with (store.directory / f"step_{step}.json").open("x", encoding="utf-8") as stream:
            json.dump(statistics, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for split in range(4):
            store.remove(f"z{step}_s{split}")
    _, gram = basis_gram_statistics(store, eligible, spec["orthogonality"], torch_module)
    projected = projected_hessian_statistics(store, eligible, steps, torch_module)
    return gradient_norm, steps, gram, projected


def spectral_statistics(matrix: Any, torch_module: Any) -> dict[str, Any]:
    if (matrix.dtype != torch_module.float64 or matrix.device.type != "cpu" or
        not bool(torch_module.isfinite(matrix).all())):
        raise ValueError("Spectral inputs must be finite CPU float64")
    singular = torch_module.linalg.svdvals(matrix)
    energy = singular.square()
    total = float(energy.sum())
    fractions = energy / total if total else torch_module.zeros_like(energy)
    return {"singular_values": singular.tolist(), "squared_singular_value_fractions": fractions.tolist(),
            "top1_energy_fraction": float(fractions[0]),
            "participation_ratio_effective_rank": total * total / float(energy.square().sum()) if total else 0.0}


def build_artifact(merged: Any, responses: Any, spec: dict[str, Any], torch_module: Any) -> dict[str, Any]:
    shape = (spec["probe"]["sequence_length"], spec["readout"]["hidden_size"])
    values = {"merged_mean": merged, "krylov_responses": responses}
    for name, value in values.items():
        expected = shape if name == "merged_mean" else (4, *shape)
        if (not isinstance(value, torch_module.Tensor) or value.device.type != "cpu" or
            value.dtype != torch_module.float64 or tuple(value.shape) != expected or
            not bool(torch_module.isfinite(value).all())):
            raise ValueError(f"Invalid CPU float64 response: {name}")
    artifact = {key: value.to(dtype=torch_module.float32).contiguous() for key, value in values.items()}
    if any(not bool(torch_module.isfinite(value).all()) for value in artifact.values()):
        raise ValueError("Non-finite stored response")
    artifact["metadata"] = {"format_version": 1, "attempt_id": spec["attempt_id"],
        "response_sign": "positive_Jv", "direction_order": [1, 2, 3, 4],
        "functional_tangents": spec["functional_tangents"], "readout": spec["readout"],
        "diagnostic_probe": spec["diagnostic_probe"], "accumulator_dtype": "cpu_float64", "stored_dtype": "float32"}
    return artifact


def activation_diagnostics(artifact: dict[str, Any], torch_module: Any) -> dict[str, Any]:
    responses = artifact["krylov_responses"].double()
    positions = []
    for position in range(responses.shape[1]):
        matrix = responses[:, position]
        statistics = gram_summary(matrix @ matrix.T, torch_module)
        incremental = [{"direction_count": count, **spectral_statistics(matrix[:count], torch_module)}
                       for count in (1, 2, 3, 4)]
        positions.append({"position": position, "response_norms": statistics["norms"],
            "response_cosine_matrix": statistics["cosine_matrix"],
            "spectrum": {key: value for key, value in incremental[-1].items() if key != "direction_count"},
            "incremental_spectra": incremental})
    def summarize(rows):
        return {"response_norms_by_direction": [descriptive([row["response_norms"][i] for row in rows]) for i in range(4)],
                "pairwise_response_cosines": descriptive([row["response_cosine_matrix"][i][j]
                    for row in rows for i in range(4) for j in range(i + 1, 4)]),
                "incremental_spectra": [{"direction_count": count,
                    **{key: descriptive([row["incremental_spectra"][count - 1][key] for row in rows])
                       for key in ("top1_energy_fraction", "participation_ratio_effective_rank")}}
                    for count in (1, 2, 3, 4)]}
    return {"positions": positions, "descriptive_summaries": {
        "position_0": summarize(positions[:1]), "positions_1_through_127": summarize(positions[1:])}}


def require_all_outputs_absent(artifact_path: Path, manifest_path: Path, basis_directory: Path) -> None:
    require_output_absent(artifact_path, manifest_path)
    outputs = [artifact_path, manifest_path, *[basis_directory / f"q{i}.safetensors" for i in range(1, 5)]]
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Scientific output paths must be distinct")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise ValueError(f"Scientific output already exists: {path}")


def publish_basis(temporary_directory: Path, basis_directory: Path) -> list[dict[str, Any]]:
    """Move, never copy. The manifest is the final publication/commit marker.

    Failure after a rename can leave final basis files and unmoved temporary
    files. Retain both and refuse reruns; manual inspection/cleanup is required.
    Cross-filesystem publication fails before moving anything.
    """
    basis_directory.mkdir(parents=True, exist_ok=True)
    if temporary_directory.stat().st_dev != basis_directory.stat().st_dev:
        raise ValueError("Basis publication requires the same filesystem; copying is forbidden")
    paths = [(temporary_directory / f"q{i}.safetensors", basis_directory / f"q{i}.safetensors")
             for i in range(1, 5)]
    for source, destination in paths:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Basis output already exists: {destination}")
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Missing or invalid staged basis: {source}")
    records = []
    for i, (source, destination) in enumerate(paths, 1):
        digest = sha256_file(source)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Basis output already exists: {destination}")
        os.replace(source, destination)
        if sha256_file(destination) != digest:
            raise ValueError(f"Basis changed during publication: {destination.name}")
        records.append({"direction": i, "filename": destination.name, "dtype": "float32",
                        "size_bytes": destination.stat().st_size, "sha256": digest})
    return records


def cleanup_temporary_directory(directory: Path) -> None:
    expected = {"state.json", *[f"step_{i}.json" for i in range(1, 5)]}
    if {path.name for path in directory.iterdir()} != expected:
        raise ValueError("Unexpected temporary contents; refusing cleanup")
    for filename in sorted(expected):
        (directory / filename).unlink()
    directory.rmdir()


def build_manifest(spec: dict[str, Any], hashes: dict[str, str], source_files: list[dict[str, Any]],
                   corpus: dict[str, Any], mean_loss: float, gradient_norm: float,
                   matrices: list[dict[str, Any]], steps: list[dict[str, Any]], gram: dict[str, Any],
                   projected: dict[str, Any], tangent_norms: list[float], readout_checks: list[dict[str, Any]],
                   diagnostics: dict[str, Any], basis_files: list[dict[str, Any]], artifact_hash: str,
                   raw_hashes: dict[str, str]) -> dict[str, Any]:
    return {"format_version": 1, "attempt_id": spec["attempt_id"], "hash_algorithm": "sha256",
        "purpose": spec["purpose"], "information_policy": spec["information_policy"], **hashes,
        "source_checkpoint": {"files": source_files, "file_count": len(source_files),
                              "total_bytes": sum(record["size_bytes"] for record in source_files)},
        "corpus": {key: corpus[key] for key in ("serialized_sha256", "raw_tensor_sha256")},
        "probe": spec["probe"], "diagnostic_probe": spec["diagnostic_probe"],
        "loss": {**spec["generic_loss"], "mean_generic_loss": mean_loss},
        "hessian": spec["hessian"], "splits": spec["splits"], "krylov": spec["krylov"],
        "gradient_norm": gradient_norm, "steps": steps, "beta_values": [step["beta"] for step in steps],
        "orthogonality": {"method": spec["orthogonality"], **gram},
        "projected_hessian": {"method": spec["projected_hessian"], **projected},
        "functional_tangents": {**spec["functional_tangents"], "realized_norms": tangent_norms},
        "eligible_parameters": [record["name"] for record in matrices], "matrices": matrices,
        "jvp": spec["jvp"], "readout": {**spec["readout"], "checks": readout_checks},
        "activation_space_diagnostics": diagnostics, "basis_files": basis_files, "workload": spec["workload"],
        "artifact": {"serialized_sha256": artifact_hash, "raw_tensors_sha256": raw_hashes}}


def construct(args: argparse.Namespace) -> None:
    require_all_outputs_absent(args.artifact_path, args.construction_manifest_path, args.basis_directory)
    if args.temporary_directory.is_symlink() or (args.temporary_directory.exists() and
        (not args.temporary_directory.is_dir() or any(args.temporary_directory.iterdir()))):
        raise ValueError("Attempt011 temporary directory contains stale data; no automatic resume")
    spec = load_spec(args.attempt_spec_path)
    paths = [args.attempt_spec_path, args.corpus_spec_path, args.corpus_manifest_path,
             args.tokens_path, args.probe_path, Path(__file__).resolve(), HVP_SOURCE_PATH]
    for output in (args.artifact_path, args.construction_manifest_path, args.temporary_directory, args.basis_directory):
        if (output.resolve().is_relative_to(args.merged_model_dir.resolve()) or
            any(output.resolve() == path.resolve() or path.resolve().is_relative_to(output.resolve()) for path in paths)):
            raise ValueError("Outputs and temporary data must not overlap frozen inputs")
    if (args.artifact_path.resolve() == args.construction_manifest_path.resolve() or
        any(output.resolve().is_relative_to(args.temporary_directory.resolve()) for output in
            (args.artifact_path, args.construction_manifest_path, args.basis_directory))):
        raise ValueError("Scientific outputs must be separate from temporary storage")
    hashes = {path: sha256_file(path) for path in paths}
    if hashes[HVP_SOURCE_PATH] != LOADED_HVP_SOURCE_SHA256:
        raise ValueError("HVP implementation changed after import")
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
    parameters = {f"{name}.weight": module.weight for name, module in eligible}
    temporary_stage(args.temporary_directory, "canonical_full_gradient")
    print(f"Accumulating canonical full gradient: {spec['generic_loss']['number_of_batches']} batches", flush=True)
    mean_loss = accumulate_mean_generic_gradient(model, tokens, eligible, spec["generic_loss"], torch)
    gradient = detach_canonical_gradient(model, eligible)
    try:
        save_vector(args.temporary_directory / "g.safetensors", gradient, parameters, torch)
    finally:
        gradient.clear()
        model.zero_grad(set_to_none=True)
    verify_frozen_parameters(model, selected, frozen_before, torch)
    responses, tangent_norms, readout_checks = [], [], []
    merged = None
    with VectorStore(args.temporary_directory, eligible, torch) as store:
        def compute_split_actions(step, key):
            direction, _ = load_device_vector(store, key, eligible, torch)
            action = None
            try:
                for split, split_tokens in enumerate(interleaved_splits(tokens)):
                    temporary_stage(args.temporary_directory, f"step_{step}_split_{split}")
                    print(f"HVP direction {step}/4, split {split + 1}/4", flush=True)
                    def progress(index, count):
                        if index % 16 == 0 or index == count:
                            temporary_stage(args.temporary_directory, f"step_{step}_split_{split}_batch_{index}")
                            print(f"HVP direction {step}/4, split {split + 1}/4: {index}/{count} batches", flush=True)
                    action = accumulate_hessian_split(model, split_tokens, eligible, direction,
                                                      split_loss_spec(spec), torch, progress)
                    save_vector(args.temporary_directory / f"z{step}_s{split}.safetensors", action, parameters, torch)
                    action.clear()
                    action = None
                    verify_frozen_parameters(model, selected, frozen_before, torch)
            finally:
                if action is not None:
                    action.clear()
                direction.clear()
                model.zero_grad(set_to_none=True)
        gradient_norm, steps, gram, projected = build_krylov_basis(store, eligible, compute_split_actions, spec, torch)
        for i in range(1, 5):
            temporary_stage(args.temporary_directory, f"jvp_direction_{i}")
            # Direction 1 uses the original g, never a rounded reconstruction from q1.
            tangent, norm = load_device_vector(store, "g" if i == 1 else f"q{i}", eligible,
                                              torch, 1.0 if i == 1 else gradient_norm)
            try:
                primal, response, readout = compute_probe_response(model, probe, tangent, spec, torch, f"direction {i}/4")
            finally:
                tangent.clear()
            if merged is None:
                merged = primal
            elif not torch.equal(merged, primal):
                raise ValueError("Primal mean changed across directions")
            if i == 1:
                store.remove("g")  # Canonical g is needed through its direct JVP only.
            responses.append(response)
            tangent_norms.append(norm)
            readout_checks.append({"direction": i, **readout})
    artifact = build_artifact(merged, torch.stack(responses), spec, torch)
    diagnostics = activation_diagnostics(artifact, torch)
    verify_frozen_parameters(model, selected, frozen_before, torch)
    verify_model_unchanged(model, before, torch)
    verify_unchanged(args.merged_model_dir, source_files, hashes)
    temporary_stage(args.temporary_directory, "publishing")
    require_all_outputs_absent(args.artifact_path, args.construction_manifest_path, args.basis_directory)
    basis_files = publish_basis(args.temporary_directory, args.basis_directory)
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with args.artifact_path.open("xb") as stream:
        torch.save(artifact, stream)
    verify_unchanged(args.merged_model_dir, source_files, hashes)
    for record in basis_files:
        if sha256_file(args.basis_directory / record["filename"]) != record["sha256"]:
            raise ValueError("Published basis changed before manifest publication")
    matrices = [{"name": f"{name}.weight", "shape": list(module.weight.shape)} for name, module in eligible]
    manifest = build_manifest(spec, {
        "attempt_spec_sha256": hashes[args.attempt_spec_path],
        "corpus_spec_sha256": hashes[args.corpus_spec_path],
        "corpus_manifest_sha256": hashes[args.corpus_manifest_path],
        "constructor_script_sha256": hashes[Path(__file__).resolve()],
        "hvp_implementation_sha256": hashes[HVP_SOURCE_PATH],
    }, source_files, corpus, mean_loss, gradient_norm, matrices, steps, gram, projected, tangent_norms,
        readout_checks, diagnostics, basis_files, sha256_file(args.artifact_path),
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
        ("basis-directory", Path(FROZEN_SPEC["defaults"]["basis_directory"])),
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
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        traceback.clear_frames(exc.__traceback__)
        print(f"ERROR: {exc}. Temporary state retained at {args.temporary_directory}; no automatic resume.", file=sys.stderr)
        return 1
    print(f"Wrote four basis files, {args.artifact_path}, and {args.construction_manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
