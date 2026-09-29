#!/usr/bin/env python3
"""Attempt 013: known-base endpoint Hessian mechanistic validation, not recovery."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
from contextlib import contextmanager, ExitStack
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '013_known_base_endpoint_hessian_validation_prefix0_13',
 'purpose': 'post_oracle_mechanistic_diagnostic_not_blind_recovery_not_candidate_tuning',
 'information_policy': 'final_and_historical_base_checkpoints_frozen_005_corpus_spec_manifest_013_spec_source_shared_true_HVP_source_only',
 'defaults': {'merged_model_directory': '/root/model-diff-scratch/models/merged',
              'base_model_directory': '/root/model-diff-scratch/models/base',
              'corpus_spec_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json',
              'corpus_manifest_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json',
              'tokens_path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
              'temporary_directory': '/root/model-diff-scratch/tmp/attempt013',
              'diagnostic_path': 'experiments/attempts/013_known_base_endpoint_hessian_validation_prefix0_13/diagnostic.json'},
 'base': {'repo_id': 'Qwen/Qwen3-1.7B',
          'revision': '0060bc56d46589041c1048efd1a397421b1142b5',
          'files': [{'path': '.gitattributes',
                     'size_bytes': 1570,
                     'sha256': '34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930'},
                    {'path': 'README.md',
                     'size_bytes': 13963,
                     'sha256': '257e52c419dac2258852643f18af6c974f21f8c6c1b6f371b6cca6201cf29091'},
                    {'path': 'config.json',
                     'size_bytes': 726,
                     'sha256': '1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197'},
                    {'path': 'generation_config.json',
                     'size_bytes': 239,
                     'sha256': '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2'},
                    {'path': 'merges.txt',
                     'size_bytes': 1671853,
                     'sha256': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'},
                    {'path': 'model-00001-of-00002.safetensors',
                     'size_bytes': 3441185608,
                     'sha256': '169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5'},
                    {'path': 'model-00002-of-00002.safetensors',
                     'size_bytes': 622329984,
                     'sha256': '912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9'},
                    {'path': 'model.safetensors.index.json',
                     'size_bytes': 25605,
                     'sha256': '0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d'},
                    {'path': 'tokenizer.json',
                     'size_bytes': 11422654,
                     'sha256': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'},
                    {'path': 'tokenizer_config.json',
                     'size_bytes': 9732,
                     'sha256': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'},
                    {'path': 'vocab.json',
                     'size_bytes': 2776833,
                     'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}]},
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
 'splits': {'count': 4,
            'order': [0, 1, 2, 3],
            'selection': 'example_index_modulo_4',
            'sample_count': 1024,
            'total_prediction_tokens': 130048,
            'batch_size': 8,
            'number_of_batches': 128},
 'hessian': {'primitive': 'true_double_backward_grad_of_gradient_dot_detached_vector',
             'vector': 'same_detached_FP32_W1_S_minus_W0_S_for_every_batch_and_split_at_untouched_W1',
             'batch_loss': 'sum_token_CE_divided_by_1024_times_127',
             'split_action': 'sum_of_128_batch_HVPs',
             'accumulation_dtype': 'float32',
             'accumulation_device': 'parameter_device',
             'full_action': 'CPU_float64_matrixwise_mean_of_four_stored_FP32_split_actions',
             'separate_full_corpus_HVP': False,
             'fallback': False},
 'displacement': {'sign': 'W1_S - W0_S',
                  'accounting': 'matrix_wise_CPU_float64',
                  'HVP_vector': 'detached_float32_of_CPU_float64_displacement',
                  'rounding_diagnostics': True,
                  'outside_categories': ['linear_weights_blocks_14_27', 'all_other_parameters'],
                  'parameter_alias_policy': 'unique_named_parameters_tied_weights_counted_once'},
 'hybrid': {'definition': 'W1_with_exactly_S_replaced_by_W0_S',
            'restore_in_finally': True,
            'verify_every_parameter_and_buffer': True},
 'metrics': {'primary': 'cosine(h, g1 - gH)',
             'd_local': 'g1 - gH',
             'd_true': 'g1 - g0',
             'h': 'mean_of_four_split_H1_SS_delta_actions_CPU_float64',
             'optimal_scale': 'dot(h,d_local)/dot(h,h)',
             'relative_residual': 'norm(d_local - optimal_scale*h)/norm(d_local)',
             'historical_unexplained': 'norm(d_true - d_local)/norm(d_true)',
             'dtype': 'cpu_float64',
             'zero_policy': 'undefined_cosines_ratios_or_projections_are_null'},
 'temporary': {'format': 'safetensors_float32',
               'files': ['g1.safetensors',
                         'gH.safetensors',
                         'g0.safetensors',
                         'h0.safetensors',
                         'h1.safetensors',
                         'h2.safetensors',
                         'h3.safetensors'],
               'cleanup': 'only_after_successful_diagnostic_publication',
               'failure': 'retain_no_resume'},
 'workload': {'ordinary_gradient_batches': 1536, 'hvp_batches': 512, 'jvp_batches': 0},
 'outputs': {'overwrite': False, 'timestamps': False, 'host_metadata': False, 'gpu_metadata': False}}
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
HVP_SOURCE_PATH = Path(__file__).with_name('smoke_test_generic_hessian_curvature_probe.py')
_hvp_spec = importlib.util.spec_from_file_location('endpoint_true_hvp', HVP_SOURCE_PATH)
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

LOADED_HVP_SOURCE_SHA256 = sha256_file(HVP_SOURCE_PATH)

def parameter_categories(model, selected):
    upper = set()
    import torch
    for block in range(14, 28):
        for _, module in model.model.layers[block].named_modules():
            if isinstance(module, torch.nn.Linear):
                upper.add(id(module.weight))
    # named_parameters deduplicates tied parameters, avoiding double counting.
    return {name: ('selected' if name in selected else 'linear_weights_blocks_14_27'
                   if id(value) in upper else 'all_other_parameters')
            for name, value in model.named_parameters()}


def displacement_audit(final, base, eligible, torch_module):
    selected = {f'{name}.weight' for name, _ in eligible}
    left, right = dict(final.named_parameters()), dict(base.named_parameters())
    if left.keys() != right.keys():
        raise ValueError('Base/final parameter inventories differ')
    categories = parameter_categories(final, selected)
    groups = {key: {'squares': [], 'nonzero_names': [], 'matrices': []} for key in
              ('selected', 'linear_weights_blocks_14_27', 'all_other_parameters')}
    vector, rounded_squares, error_squares = {}, [], []
    for name in sorted(left):
        w1, w0 = left[name], right[name]
        if w1.shape != w0.shape or w1.dtype != torch_module.float32 or w0.dtype != torch_module.float32:
            raise ValueError(f'Base/final parameter shape/dtype mismatch: {name}')
        delta = w1.detach().to(device='cpu', dtype=torch_module.float64) - w0.detach().to(device='cpu', dtype=torch_module.float64)
        if not bool(torch_module.isfinite(delta).all()):
            raise ValueError(f'Nonfinite checkpoint displacement: {name}')
        square = float(delta.square().sum())
        group = groups[categories[name]]
        group['squares'].append(square)
        if bool(delta.ne(0).any()):
            group['nonzero_names'].append(name)
        group['matrices'].append({'name': name, 'norm': math.sqrt(square)})
        if name in selected:
            rounded = delta.float().contiguous()
            if not bool(torch_module.isfinite(rounded).all()):
                raise ValueError('Nonfinite realized displacement tangent')
            rounded_squares.append(float(rounded.double().square().sum()))
            error_squares.append(float((rounded.double() - delta).square().sum()))
            vector[name] = rounded.to(device=w1.device)
    audit = {name: {'norm': math.sqrt(math.fsum(group['squares'])),
                    'nonzero_count': len(group['nonzero_names']), 'nonzero_names': group['nonzero_names'],
                    'per_tensor_norms': group['matrices']} for name, group in groups.items()}
    audit['HVP_vector'] = {'sign': 'W1_S - W0_S', 'dtype': 'float32',
        'realized_norm': math.sqrt(math.fsum(rounded_squares)),
        'rounding_error_norm': math.sqrt(math.fsum(error_squares))}
    return audit, vector


@contextmanager
def selected_base_hybrid(final, base, eligible, torch_module):
    before = model_state_hashes(final, torch_module)
    base_hashes = model_state_hashes(base, torch_module)
    base_parameters = dict(base.named_parameters())
    selected = {f'{name}.weight': module.weight for name, module in eligible}
    original = {name: parameter.detach().to(device='cpu', copy=True) for name, parameter in selected.items()}
    expected = dict(before)
    for name in selected:
        expected[f'parameter:{name}'] = base_hashes[f'parameter:{name}']
    try:
        with torch_module.no_grad():
            for name, parameter in selected.items():
                parameter.copy_(base_parameters[name])
        if model_state_hashes(final, torch_module) != expected:
            raise ValueError('Hybrid did not restore exactly the selected base weights')
        yield {'selected_parameter_names': sorted(selected), 'all_parameters_and_buffers_verified': True}
        if model_state_hashes(final, torch_module) != expected:
            raise ValueError('Hybrid mutated during gradient computation')
    finally:
        with torch_module.no_grad():
            for name, parameter in selected.items():
                parameter.copy_(original[name])
        final.zero_grad(set_to_none=True)
        if model_state_hashes(final, torch_module) != before:
            raise ValueError('Final parameters/buffers were not restored after hybrid')
        if model_state_hashes(base, torch_module) != base_hashes:
            raise ValueError('Historical base mutated during hybrid')


def save_gradient(model, eligible, tokens, settings, path, torch_module):
    print(f'Ordinary gradient: {path.stem}, {settings["number_of_batches"]} batches', flush=True)
    before = model_state_hashes(model, torch_module)
    freeze_other_parameters(model, eligible)
    try:
        loss = accumulate_mean_generic_gradient(model, tokens, eligible, settings, torch_module)
        parameters = {f'{name}.weight': module.weight for name, module in eligible}
        vector = {name: parameter.grad.detach() for name, parameter in parameters.items()}
        save_vector(path, vector, parameters, torch_module)
        vector.clear()
        verify_model_unchanged(model, before, torch_module)
        return loss
    finally:
        model.zero_grad(set_to_none=True)


def split_actions(model, eligible, tokens, delta, settings, directory, torch_module):
    parameters = {f'{name}.weight': module.weight for name, module in eligible}
    before = model_state_hashes(model, torch_module)
    for split, subset in enumerate(interleaved_splits(tokens)):
        print(f'Hessian displacement split {split + 1}/4', flush=True)
        def progress(index, count):
            if index % 32 == 0 or index == count:
                print(f'Split {split + 1}: {index}/{count} HVP batches', flush=True)
        action = accumulate_hessian_split(model, subset, eligible, delta, settings, torch_module, progress)
        try:
            save_vector(directory / f'h{split}.safetensors', action, parameters, torch_module)
        finally:
            action.clear()
        verify_model_unchanged(model, before, torch_module)


def vector_diagnostics(directory, matrices, torch_module):
    """Streaming CPU FP64 accounting; no full-vector concatenation or copies."""
    from safetensors import safe_open
    t = torch_module
    keys = ['g0', 'g1', 'gH', 'h0', 'h1', 'h2', 'h3']
    names = ['g0', 'g1', 'gH', 'h', 'd_local', 'd_true', 'unexplained', 'h0', 'h1', 'h2', 'h3']
    index = {name: i for i, name in enumerate(names)}
    gram = t.zeros((len(names), len(names)), dtype=t.float64)
    with ExitStack() as stack:
        files = {key: stack.enter_context(safe_open(str(directory / f'{key}.safetensors'), framework='pt', device='cpu')) for key in keys}
        if any(set(file.keys()) != {record['name'] for record in matrices} for file in files.values()):
            raise ValueError('Temporary vector keys mismatch')
        def values(record):
            data = {}
            for key, file in files.items():
                value = file.get_tensor(record['name'])
                if value.dtype != t.float32 or list(value.shape) != record['shape'] or not bool(t.isfinite(value).all()):
                    raise ValueError('Invalid temporary vector tensor')
                data[key] = value.double()
            data['h'] = sum(data[f'h{i}'] for i in range(4)) / 4
            data['d_local'] = data['g1'] - data['gH']
            data['d_true'] = data['g1'] - data['g0']
            data['unexplained'] = data['d_true'] - data['d_local']
            return data
        for record in matrices:
            data = values(record)
            for i, name in enumerate(names):
                for j in range(i, len(names)):
                    dot = t.dot(data[name].flatten(), data[names[j]].flatten())
                    gram[i, j] += dot
                    if i != j: gram[j, i] += dot
        if not bool(t.isfinite(gram).all()): raise ValueError('Nonfinite diagnostic Gram')
        norms = {name: math.sqrt(float(gram[i, i])) for name, i in index.items()}
        def dot(x, y): return float(gram[index[x], index[y]])
        def ratio(x, y): return norms[x] / norms[y] if norms[y] else None
        def cosine(x, y): return dot(x, y) / (norms[x] * norms[y]) if norms[x] and norms[y] else None
        scale = dot('h', 'd_local') / dot('h', 'h') if norms['h'] else None
        # Direct residual norms avoid subtracting nearly equal Gram entries.
        squares = []
        if scale is not None:
            for record in matrices:
                data = values(record)
                squares.append(float((data['d_local'] - scale * data['h']).square().sum()))
        residual = math.sqrt(math.fsum(squares)) if scale is not None else None
    return {'norms': norms, 'gram_order': names, 'gram_matrix': gram.tolist(),
        'primary': {'definition': 'cosine(h, g1 - gH)', 'cosine': cosine('h','d_local'),
            'h_norm': norms['h'], 'd_local_norm': norms['d_local'], 'h_to_d_local_norm_ratio': ratio('h','d_local'),
            'optimal_scalar_projection_coefficient': scale,
            'relative_residual_after_optimal_rescaling': residual/norms['d_local'] if residual is not None and norms['d_local'] else None},
        'historical': {'h_vs_d_true_cosine': cosine('h','d_true'), 'd_local_vs_d_true_cosine': cosine('d_local','d_true'),
            'd_local_to_d_true_norm_ratio': ratio('d_local','d_true'), 'relative_unexplained_norm': ratio('unexplained','d_true')},
        'stationarity': {'g0_norm': norms['g0'], 'g1_norm': norms['g1'], 'gH_norm': norms['gH'],
            'g0_to_g1_norm_ratio': ratio('g0','g1'), 'g0_to_d_true_norm_ratio': ratio('g0','d_true'),
            'g0_vs_g1_cosine': cosine('g0','g1'), 'h_vs_g1_cosine': cosine('h','g1')},
        'split_hvp_stability': {'norms': [norms[f'h{i}'] for i in range(4)],
            'cosine_matrix': [[cosine(f'h{i}',f'h{j}') for j in range(4)] for i in range(4)],
            'cosines_with_full_mean': [cosine(f'h{i}','h') for i in range(4)]}}


def require_outputs_absent(output, temporary):
    if output.exists() or output.is_symlink(): raise ValueError('Diagnostic already exists')
    if temporary.is_symlink() or (temporary.exists() and (not temporary.is_dir() or any(temporary.iterdir()))):
        raise ValueError('Stale temporary data; no automatic reuse')


def cleanup(directory):
    expected = {'state.json', *FROZEN_SPEC['temporary']['files']}
    if {p.name for p in directory.iterdir()} != expected: raise ValueError('Unexpected temporary contents')
    for name in sorted(expected): (directory/name).unlink()
    directory.rmdir()


def construct(args):
    require_outputs_absent(args.diagnostic_path, args.temporary_directory)
    spec = load_spec(args.attempt_spec_path)
    paths = [args.attempt_spec_path, args.corpus_spec_path, args.corpus_manifest_path, args.tokens_path,
             Path(__file__).resolve(), HVP_SOURCE_PATH]
    for out in (args.diagnostic_path, args.temporary_directory):
        if (any(out.resolve().is_relative_to(p.resolve()) for p in (args.merged_model_dir,args.base_model_dir)) or
                any(out.resolve() == p.resolve() or p.resolve().is_relative_to(out.resolve()) for p in paths)):
            raise ValueError('Outputs overlap frozen inputs')
    if args.diagnostic_path.resolve().is_relative_to(args.temporary_directory.resolve()):
        raise ValueError('Diagnostic must be separate from temporary storage')
    hashes = {path: sha256_file(path) for path in paths}
    if hashes[HVP_SOURCE_PATH] != LOADED_HVP_SOURCE_SHA256 or load_spec(args.attempt_spec_path) != spec:
        raise ValueError('Source/spec changed during loading')
    for key,path in [('corpus_spec_sha256',args.corpus_spec_path),('corpus_manifest_sha256',args.corpus_manifest_path)]:
        if hashes[path] != spec['frozen_corpus'][key]: raise ValueError('Frozen corpus provenance mismatch')
    final_files, base_files = checkpoint_file_records(args.merged_model_dir), checkpoint_file_records(args.base_model_dir)
    if final_files != spec['canonical_checkpoint_files'] or base_files != spec['base']['files']:
        raise ValueError('Exact base/final inventory/hash mismatch')
    corpus = verify_corpus_manifest(args.corpus_manifest_path,args.corpus_spec_path,args.tokens_path,
                                    args.merged_model_dir,load_corpus_spec(args.corpus_spec_path))
    if any(corpus[key] != spec['frozen_corpus'][key] for key in ('serialized_sha256','raw_tensor_sha256')):
        raise ValueError('Frozen corpus hashes mismatch')
    import torch
    tokens = load_corpus_tokens(args.tokens_path,corpus['raw_tensor_sha256'],
                               spec['generic_loss']['sample_count'],spec['generic_loss']['sequence_length'],torch)
    args.temporary_directory.mkdir(parents=True,exist_ok=True)
    with (args.temporary_directory/'state.json').open('x') as stream:
        json.dump({'attempt_id':spec['attempt_id'],'resume_permitted':False},stream)
    final = load_local_model(args.merged_model_dir,args.device,torch)
    base = load_local_model(args.base_model_dir,'cpu',torch)
    eligible = discover_eligible_linear_weights(final,torch)
    base_eligible = discover_eligible_linear_weights(base,torch)
    if [n for n,_ in eligible] != [n for n,_ in base_eligible]: raise ValueError('Base/final selection mismatch')
    freeze_other_parameters(final,eligible)
    before_final, before_base = model_state_hashes(final,torch), model_state_hashes(base,torch)
    audit, delta = displacement_audit(final,base,eligible,torch)
    delta.clear()
    matrices = [{'name':f'{name}.weight','shape':list(module.weight.shape)} for name,module in eligible]
    losses = {}
    losses['g1'] = save_gradient(final,eligible,tokens,spec['generic_loss'],args.temporary_directory/'g1.safetensors',torch)
    # Release the displacement GPU allocation while computing the hybrid gradient.
    delta.clear()
    with selected_base_hybrid(final,base,eligible,torch) as hybrid:
        losses['gH'] = save_gradient(final,eligible,tokens,spec['generic_loss'],args.temporary_directory/'gH.safetensors',torch)
    verify_model_unchanged(final,before_final,torch)
    _, delta = displacement_audit(final,base,eligible,torch)
    try:
        split_actions(final,eligible,tokens,delta,split_loss_spec(spec),args.temporary_directory,torch)
    finally:
        delta.clear()
        final.zero_grad(set_to_none=True)
    verify_model_unchanged(final,before_final,torch)
    verify_model_unchanged(base,before_base,torch)
    # Only one model resides on the accelerator; release all final-module references.
    del eligible, final
    base.to(args.device)
    losses['g0'] = save_gradient(base,base_eligible,tokens,spec['generic_loss'],args.temporary_directory/'g0.safetensors',torch)
    verify_model_unchanged(base,before_base,torch)
    del base_eligible, base
    metrics = vector_diagnostics(args.temporary_directory,matrices,torch)
    verify_unchanged(args.merged_model_dir,final_files,hashes)
    verify_unchanged(args.base_model_dir,base_files,{})
    result = {'format_version':1,'attempt_id':spec['attempt_id'],'purpose':spec['purpose'],
        'information_policy':spec['information_policy'], 'attempt_spec_sha256':hashes[args.attempt_spec_path],
        'constructor_script_sha256':hashes[Path(__file__).resolve()], 'hvp_helper_sha256':hashes[HVP_SOURCE_PATH],
        'base':{**spec['base'],'files':base_files},'final_checkpoint_files':final_files,
        'corpus':{'spec_sha256':hashes[args.corpus_spec_path],'manifest_sha256':hashes[args.corpus_manifest_path],
                  'serialized_sha256':corpus['serialized_sha256'],'raw_tensor_sha256':corpus['raw_tensor_sha256']},
        'generic_loss':spec['generic_loss'],'hessian':spec['hessian'],'splits':spec['splits'],
        'displacement_definition':spec['displacement'],'displacement_audit':audit,'hybrid':{**spec['hybrid'],**hybrid},
        'matrices':matrices,'mean_losses':losses,'metric_definitions':spec['metrics'],'metrics':metrics,'workload':spec['workload']}
    write_manifest(args.diagnostic_path,result)
    cleanup(args.temporary_directory)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-spec-path',type=Path,default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'spec.json')
    for option,key in [('merged-model-dir','merged_model_directory'),('base-model-dir','base_model_directory'),
        ('corpus-spec-path','corpus_spec_path'),('corpus-manifest-path','corpus_manifest_path'),('tokens-path','tokens_path'),
        ('temporary-directory','temporary_directory'),('diagnostic-path','diagnostic_path')]:
        path=Path(FROZEN_SPEC['defaults'][key]);parser.add_argument('--'+option,type=Path,default=path if path.is_absolute() else PROJECT/path)
    parser.add_argument('--device',default='cuda')
    return parser.parse_args(argv)


if __name__=='__main__':
    construct(parse_args())
