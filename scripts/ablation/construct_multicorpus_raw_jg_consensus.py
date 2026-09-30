#!/usr/bin/env python3
"""Blind construction of equal-weight positive raw-Jg corpus consensus."""
import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any
PROJECT = Path(__file__).resolve().parents[2]
FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '014_multicorpus_raw_jg_consensus_prefix0_13',
 'purpose': 'post_oracle_method_development_with_blind_construction',
 'information_policy': 'final_checkpoint_frozen_generic_corpora_and_probe_only',
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
             'semantics': 'hidden_states[14] is block 13 output before block 14; verify with forward hook'},
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
 'corpora': [{'name': 'fineweb',
              'spec_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json',
              'spec_sha256': 'e514820e7def99e9287c32de4b1279f28f6ff6c2e09749a8dfa5d8b14c311f0f',
              'manifest_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json',
              'tokens_path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
              'manifest_sha256': '8b3ea49228dbb106c2b8a7c849c4f27e021fa3e992800444f43e260a6b055230'},
             {'name': 'wikitext103_raw',
              'spec_path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/wikitext103_raw_corpus_spec.json',
              'spec_sha256': 'f6b9ccd801f547d7d2c24fdaa0f53ee55758068dae3685d7859dcbfb3027a24b',
              'manifest_path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/wikitext103_raw-corpus-manifest.json',
              'tokens_path': '/root/model-diff-scratch/artifacts/attempt014_wikitext103_raw_corpus/tokens.pt'},
             {'name': 'tinystories',
              'spec_path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/tinystories_corpus_spec.json',
              'spec_sha256': '0e4f6c23a9e6dce35d1597185b7ee9f378cb9a4753b974dba3fa5a2dd0460ff6',
              'manifest_path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/tinystories-corpus-manifest.json',
              'tokens_path': '/root/model-diff-scratch/artifacts/attempt014_tinystories_corpus/tokens.pt'}],
 'consensus': {'definition': 'arithmetic_mean_of_three_per_position_unit_positive_Jg_responses',
               'normalization_dtype': 'cpu_float64',
               'inputs': 'stored_equivalent_float32_responses',
               'zero_or_nonfinite_norm': 'fail',
               'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
               'concentration_tolerance': 1e-12},
 'defaults': {'merged_model_directory': '/root/model-diff-scratch/models/merged',
              'probe_path': '/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt',
              'artifact_path': '/root/model-diff-scratch/artifacts/attempt014_multicorpus_raw_jg_consensus/responses.pt',
              'manifest_path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/construction-manifest.json'},
 'workload': {'gradient_batches': 1536, 'jvp_batches': 939},
 'outputs': {'tensors': ['merged_mean',
                         'response_fineweb',
                         'response_wikitext103_raw',
                         'response_tinystories',
                         'consensus_response'],
             'dtype': 'contiguous_cpu_float32',
             'shape': [128, 2048],
             'overwrite': False},
 'evaluation_plan': {'oracle_vector': 'difference',
                     'candidate_tensors': ['response_fineweb',
                                           'response_wikitext103_raw',
                                           'response_tinystories',
                                           'consensus_response'],
                     'cosine_dtype': 'cpu_float64',
                     'per_position_scope': 'all_128_positions',
                     'primary': {'metric': 'consensus_positions_1_4_mean_cosine',
                                 'baseline': 'response_fineweb_positions_1_4_mean_cosine',
                                 'report_delta_consensus_minus_fineweb': True},
                     'secondary': {'metric': 'consensus_positions_1_127_mean_cosine',
                                   'baseline': 'response_fineweb_positions_1_127_mean_cosine',
                                   'report_delta_consensus_minus_fineweb': True},
                     'position_0': 'report_separately',
                     'policy': ['report all individual corpus candidates',
                                'no best-corpus selection',
                                'no sign selection',
                                'no corpus reweighting',
                                'no post-result candidate modification',
                                'no post-result threshold tuning']}}
NEW_CORPUS_SPECS = {'wikitext103_raw': {'format_version': 1,
                     'attempt_id': '014_multicorpus_raw_jg_consensus_prefix0_13',
                     'dataset': {'repo_id': 'Salesforce/wikitext',
                                 'revision': 'b08601e04326c79dfdd32d625aee71d232d685c3',
                                 'config': 'wikitext-103-raw-v1',
                                 'split': 'train',
                                 'streaming': True,
                                 'shuffle_buffer_size': 10000,
                                 'shuffle_rule': 'datasets.IterableDataset.shuffle(seed=42, '
                                                 'buffer_size=10000); one iterator; no workers; stop after '
                                                 'first 4096 valid rows'},
                     'tokenizer': {'source': 'canonical_merged_checkpoint',
                                   'directory': '/root/model-diff-scratch/models/merged'},
                     'selection': {'shuffle_seed': 42,
                                   'text_column': 'text',
                                   'skip_blank_text': True,
                                   'character_limit': 1280,
                                   'add_special_tokens': True,
                                   'minimum_token_count': 128,
                                   'skip_valid_examples': 0,
                                   'sample_count': 4096,
                                   'sequence_length': 128},
                     'artifact': {'path': '/root/model-diff-scratch/artifacts/attempt014_wikitext103_raw_corpus/tokens.pt',
                                  'dtype': 'torch.int64',
                                  'device': 'cpu',
                                  'contiguous': True},
                     'manifest': {'filename': 'wikitext103_raw-corpus-manifest.json',
                                  'timestamps': False,
                                  'host_metadata': False,
                                  'gpu_metadata': False},
                     'corpus_name': 'wikitext103_raw'},
 'tinystories': {'format_version': 1,
                 'attempt_id': '014_multicorpus_raw_jg_consensus_prefix0_13',
                 'dataset': {'repo_id': 'roneneldan/TinyStories',
                             'revision': 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64',
                             'config': None,
                             'split': 'train',
                             'streaming': True,
                             'shuffle_buffer_size': 10000,
                             'shuffle_rule': 'datasets.IterableDataset.shuffle(seed=42, buffer_size=10000); '
                                             'one iterator; no workers; stop after first 4096 valid rows'},
                 'tokenizer': {'source': 'canonical_merged_checkpoint',
                               'directory': '/root/model-diff-scratch/models/merged'},
                 'selection': {'shuffle_seed': 42,
                               'text_column': 'text',
                               'skip_blank_text': True,
                               'character_limit': 1280,
                               'add_special_tokens': True,
                               'minimum_token_count': 128,
                               'skip_valid_examples': 0,
                               'sample_count': 4096,
                               'sequence_length': 128},
                 'artifact': {'path': '/root/model-diff-scratch/artifacts/attempt014_tinystories_corpus/tokens.pt',
                              'dtype': 'torch.int64',
                              'device': 'cpu',
                              'contiguous': True},
                 'manifest': {'filename': 'tinystories-corpus-manifest.json',
                              'timestamps': False,
                              'host_metadata': False,
                              'gpu_metadata': False},
                 'corpus_name': 'tinystories'}}
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
            # Each token contributes once to the mean over all 4096 * 127 targets.
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

def global_tangent_scale(
    eligible: list[tuple[str, Any]], target_relative: float, torch_module: Any,
) -> dict[str, float]:
    weight_squares = []
    gradient_squares = []
    for name, module in eligible:
        weight = module.weight.detach().to(device="cpu", dtype=torch_module.float64)
        gradient = module.weight.grad
        if gradient is None or not bool(torch_module.isfinite(gradient).all().item()):
            raise ValueError(f"Missing or non-finite eligible gradient: {name}")
        gradient64 = gradient.detach().to(device="cpu", dtype=torch_module.float64)
        weight_squares.append(float(torch_module.sum(weight.square()).item()))
        gradient_squares.append(float(torch_module.sum(gradient64.square()).item()))
    weight_norm = math.sqrt(math.fsum(weight_squares))
    gradient_norm = math.sqrt(math.fsum(gradient_squares))
    if (not math.isfinite(weight_norm) or weight_norm <= 0 or
        not math.isfinite(gradient_norm) or gradient_norm <= 0):
        raise ValueError("Global weight or gradient norm is zero or non-finite")
    target_delta_norm = target_relative * weight_norm
    alpha = target_delta_norm / gradient_norm
    if not math.isfinite(alpha):
        raise ValueError("Global rollback alpha is non-finite")
    return {
        "aggregate_source_weight_norm": weight_norm,
        "aggregate_generic_gradient_norm": gradient_norm,
        "target_delta_norm": target_delta_norm,
        "alpha": alpha,
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

def prepare_tangents(
    eligible: list[tuple[str, Any]], scale: dict[str, float], torch_module: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    """Replace each gradient buffer with its positive FP32 tangent, matrix-wise."""
    alpha = scale["alpha"]
    weight_norm = scale["aggregate_source_weight_norm"]
    if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(weight_norm) or weight_norm <= 0:
        raise ValueError("Invalid tangent normalization")
    tangents, records, squares = {}, [], []
    for name, module in sorted(eligible, key=lambda item: item[0]):
        gradient = module.weight.grad
        if (gradient is None or gradient.dtype != torch_module.float32 or
            gradient.shape != module.weight.shape):
            raise ValueError(f"Missing or malformed eligible gradient: {name}")
        gradient64 = gradient.detach().to(device="cpu", dtype=torch_module.float64)
        if not bool(torch_module.isfinite(gradient64).all().item()):
            raise ValueError(f"Non-finite eligible gradient: {name}")
        tangent32 = (alpha * gradient64).to(dtype=torch_module.float32).contiguous()
        if not bool(torch_module.isfinite(tangent32).all().item()):
            raise ValueError(f"Non-finite tangent: {name}")
        # Release the gradient before allocating its device-resident replacement.
        module.weight.grad = None
        del gradient
        tangent = tangent32.to(device=module.weight.device)
        actual64 = tangent.detach().to(device="cpu", dtype=torch_module.float64)
        tangent_norm = float(torch_module.linalg.vector_norm(actual64).item())
        squares.append(float(actual64.square().sum().item()))
        tangents[f"{name}.weight"] = tangent
        records.append({
            "name": f"{name}.weight", "shape": list(tangent.shape),
            "raw_gradient_frobenius_norm": float(torch_module.linalg.vector_norm(gradient64).item()),
            "tangent_frobenius_norm": tangent_norm,
        })
        del gradient64, tangent32, actual64
    realized = math.sqrt(math.fsum(squares))
    if not math.isfinite(realized) or realized <= 0:
        raise ValueError("Realized FP32 tangent norm is zero or non-finite")
    return tangents, records, {
        "aggregate_realized_fp32_tangent_norm": realized,
        "aggregate_realized_relative_tangent_norm": realized / weight_norm,
    }

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

def compute_probe_response(
    model: Any, probe: Any, tangents: dict[str, Any], spec: dict[str, Any], torch_module: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    settings = spec["probe"]
    if tuple(probe.shape) != (settings["sample_count"], settings["sequence_length"]):
        raise ValueError("Probe shape mismatch")
    model.eval()
    device = next(model.parameters()).device
    accumulator = ResponseAccumulator(spec, torch_module)
    discrepancy = None
    for start in range(0, settings["sample_count"], settings["batch_size"]):
        tokens = probe[start:start + settings["batch_size"]].to(device)
        ordinary = ordinary_hook_readout(model, tokens, spec, torch_module) if start == 0 else None
        primal, response = run_readout_jvp(model, tokens, tangents, spec, torch_module)
        if ordinary is not None:
            discrepancy = compare_primal(primal, ordinary, spec, torch_module)
        accumulator.add(primal, response)
        del ordinary, primal, response
    merged_mean, linear_response = accumulator.means(settings["sample_count"])
    return merged_mean, linear_response, {"readout_hook_verified": True, "first_batch_primal_max_abs_difference": discrepancy}

def sha256_raw_float32_tensor(value: Any, torch_module: Any) -> str:
    if (not isinstance(value, torch_module.Tensor) or value.dtype != torch_module.float32 or
        value.device.type != "cpu" or not value.is_contiguous()):
        raise ValueError("Raw activation hash requires contiguous CPU float32 tensor")
    return hashlib.sha256(value.detach().numpy().astype("<f4", copy=False).tobytes(order="C")).hexdigest()

def require_output_absent(artifact_path: Path, manifest_path: Path) -> None:
    for path in (artifact_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError(f"Output already exists: {path}")
    if artifact_path.resolve() == manifest_path.resolve():
        raise ValueError("Artifact and manifest paths must differ")
    if not manifest_path.parent.is_dir():
        raise ValueError(f"Missing manifest directory: {manifest_path.parent}")

def response_consensus(responses, torch_module):
    t = torch_module
    if len(responses) != 3: raise ValueError('Exactly three responses required')
    for value in responses:
        if value.dtype != t.float32 or value.device.type != 'cpu' or value.ndim != 2 or value.shape != responses[0].shape:
            raise ValueError('Responses must be matching CPU FP32 matrices')
    raw = t.stack(responses).double()
    norms = raw.norm(dim=2)
    if not bool(t.isfinite(norms).all()) or bool((norms == 0).any()):
        raise ValueError('Zero/nonfinite per-position response norm')
    unit = raw / norms.unsqueeze(-1)
    consensus = unit.mean(dim=0)
    concentration = consensus.norm(dim=1)
    if bool((concentration > 1 + 1e-12).any()): raise ValueError('Invalid consensus concentration')
    rows = []
    for p in range(raw.shape[1]):
        cosine = unit[:,p] @ unit[:,p].T
        rows.append({'position':p,'response_norms':norms[:,p].tolist(),'pairwise_cosine_matrix':cosine.tolist(),
                     'consensus_concentration':float(concentration[p])})
    def summary(positions):
        pairs=[rows[p]['pairwise_cosine_matrix'][i][j] for p in positions for i,j in ((0,1),(0,2),(1,2))]
        cs=[rows[p]['consensus_concentration'] for p in positions]
        def stats(values):return {'mean':math.fsum(values)/len(values),'min':min(values),'max':max(values)}
        return {'positions':positions,'pairwise_cosines':stats(pairs),'concentration':stats(cs)}
    return consensus.float().contiguous(),{'positions':rows,'positions_1_4':summary(list(range(1,5))),
                                          'positions_1_127':summary(list(range(1,raw.shape[1])))}


def validate_frozen_corpus(record, model_dir, torch_module):
    spec_path, manifest_path, tokens_path = [PROJECT / record[k] if not Path(record[k]).is_absolute() else Path(record[k])
                                           for k in ('spec_path','manifest_path','tokens_path')]
    if sha256_file(spec_path) != record['spec_sha256']: raise ValueError('Corpus spec hash mismatch')
    spec = load_json_object(spec_path,'corpus spec')
    if record['name']=='fineweb':
        if sha256_file(manifest_path)!=record['manifest_sha256']:raise ValueError('Frozen FineWeb manifest hash mismatch')
        corpus=verify_corpus_manifest(manifest_path,spec_path,tokens_path,model_dir,load_corpus_spec(spec_path))
    else:
        if spec != NEW_CORPUS_SPECS[record['name']]:raise ValueError('Frozen new corpus spec mismatch')
        m=load_json_object(manifest_path,'corpus manifest')
        if (type(m.get('format_version')) is not int or m['format_version']!=1 or
            m.get('attempt_id')!=FROZEN_SPEC['attempt_id'] or m.get('hash_algorithm')!='sha256'):
            raise ValueError('New corpus manifest identity/version/hash algorithm mismatch')
        for key,path in (
            ('freeze_script_sha256',Path(__file__).with_name('freeze_multicorpus_consensus_corpus.py')),
            ('helper_script_sha256',Path(__file__).resolve()),
        ):
            if require_sha256(m.get(key),key)!=sha256_file(path):
                raise ValueError(f'New corpus manifest source hash mismatch: {key}')
        expected={'source':'canonical_merged_checkpoint','directory':str(model_dir),'files':tokenizer_file_records(model_dir)}
        if (m['spec_sha256']!=record['spec_sha256'] or m['dataset']!=spec['dataset'] or
            m['selection']!=spec['selection'] or m['tokenizer_checkpoint']!=expected or
            m['corpus_name']!=record['name'] or m['tensor']!={'shape':[4096,128],'dtype':'torch.int64','device':'cpu','contiguous':True}):
            raise ValueError('Frozen corpus manifest linkage mismatch')
        corpus=m['artifact']
        if corpus['path']!=str(tokens_path) or sha256_file(tokens_path)!=corpus['serialized_sha256']:
            raise ValueError('Corpus serialized hash/path mismatch')
        require_sha256(m['freeze_script_sha256'],'freeze implementation')
    tokens=load_corpus_tokens(tokens_path,corpus['raw_tensor_sha256'],4096,128,torch_module)
    hashes={path:sha256_file(path) for path in (spec_path,manifest_path,tokens_path)}
    return tokens,hashes,{'name':record['name'],'spec_sha256':hashes[spec_path],'manifest_sha256':hashes[manifest_path],
                         'serialized_sha256':hashes[tokens_path],'raw_tensor_sha256':corpus['raw_tensor_sha256']}


def construct(args):
    require_output_absent(args.artifact_path,args.manifest_path)
    spec_hash=sha256_file(args.spec_path)
    spec=load_spec(args.spec_path)
    paths=[args.spec_path,args.probe_path,Path(__file__).resolve(),
           Path(__file__).with_name('freeze_multicorpus_consensus_corpus.py')]
    for record in spec['corpora']:
        paths.extend(PROJECT/record[k] if not Path(record[k]).is_absolute() else Path(record[k]) for k in ('spec_path','manifest_path','tokens_path'))
    for out in (args.artifact_path,args.manifest_path):
        if out.resolve().is_relative_to(args.model_dir.resolve()) or out.resolve() in {p.resolve() for p in paths}:
            raise ValueError('Output overlaps frozen input')
    hashes={p:sha256_file(p) for p in paths}
    if hashes[args.spec_path]!=spec_hash or load_spec(args.spec_path)!=spec:raise ValueError("Spec changed during load")
    source=checkpoint_file_records(args.model_dir)
    if source!=spec['canonical_checkpoint_files']:raise ValueError('Checkpoint inventory mismatch')
    import torch
    # Validate every frozen corpus before loading the model or starting gradients.
    for record in spec['corpora']:
        tokens,checked,_=validate_frozen_corpus(record,args.model_dir,torch)
        if any(hashes[p]!=digest for p,digest in checked.items()):raise ValueError('Corpus changed during validation')
        del tokens
    probe=load_probe(args.probe_path,spec['probe'],torch)
    model=load_local_model(args.model_dir,args.device,torch)
    eligible=discover_eligible_linear_weights(model,torch);freeze_other_parameters(model,eligible)
    before=model_state_hashes(model,torch)
    merged=None;responses=[];records=[]
    for record in spec['corpora']:
        print(f"Corpus {record['name']}: gradient then direct positive JVP",flush=True)
        tokens,checked,provenance=validate_frozen_corpus(record,args.model_dir,torch)
        loss=accumulate_mean_generic_gradient(model,tokens,eligible,spec['generic_loss'],torch)
        del tokens
        scale=global_tangent_scale(eligible,.00125,torch)
        tangents,matrices,realized=prepare_tangents(eligible,scale,torch)
        try:primal,response,readout=compute_probe_response(model,probe,tangents,spec,torch)
        finally:
            tangents.clear();model.zero_grad(set_to_none=True)
        verify_model_unchanged(model,before,torch)
        if merged is not None and not torch.equal(merged,primal):raise ValueError('Primal changed across corpora')
        merged=primal
        responses.append(response.float().contiguous())
        records.append({**provenance,'mean_generic_loss':loss,**scale,**realized,'matrices':matrices,'readout_checks':readout})
    consensus,geometry=response_consensus(responses,torch)
    artifact={'merged_mean':merged.float().contiguous(),**{f'response_{r["name"]}':value for r,value in zip(spec['corpora'],responses)},'consensus_response':consensus}
    for value in artifact.values():
        if list(value.shape)!=[128,2048] or not bool(torch.isfinite(value).all()):raise ValueError('Invalid output response')
    verify_unchanged(args.model_dir,source,hashes)
    require_output_absent(args.artifact_path,args.manifest_path)
    args.artifact_path.parent.mkdir(parents=True,exist_ok=True)
    with args.artifact_path.open('xb') as f:torch.save(artifact,f)
    result={'attempt_id':spec['attempt_id'],'information_policy':spec['information_policy'],
        'spec_sha256':hashes[args.spec_path],'constructor_script_sha256':hashes[Path(__file__).resolve()],
        'source_checkpoint':source,'probe':spec['probe'],'generic_loss':spec['generic_loss'],'jvp':spec['jvp'],
        'readout':spec['readout'],'consensus_definition':spec['consensus'],'corpus_order':[r['name'] for r in spec['corpora']],
        'corpora':records,'geometry':geometry,'workload':spec['workload'],
        'artifact':{'serialized_sha256':sha256_file(args.artifact_path),'raw_tensors_sha256':{k:sha256_raw_float32_tensor(v,torch) for k,v in artifact.items()}}}
    verify_unchanged(args.model_dir,source,hashes)
    write_manifest(args.manifest_path,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path',type=Path,default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'spec.json')
    for opt,key in [('model-dir','merged_model_directory'),('probe-path','probe_path'),('artifact-path','artifact_path'),('manifest-path','manifest_path')]:
        p=Path(FROZEN_SPEC['defaults'][key]);parser.add_argument('--'+opt,type=Path,default=p if p.is_absolute() else PROJECT/p)
    parser.add_argument('--device',default='cuda');construct(parser.parse_args())


if __name__=='__main__':main()
