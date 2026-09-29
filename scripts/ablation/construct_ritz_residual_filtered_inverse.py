#!/usr/bin/env python3
"""Blind Attempt 012: one residual-filtered projected inverse, one direct JVP.

Standalone forward-AD helpers preserve Attempt 011 semantics without importing
its constructor, reading its activation artifact, or loading its upstream corpus.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '012_ritz_residual_filtered_inverse_k4_prefix0_13',
 'purpose': 'blind_recovery_candidate',
 'information_policy': 'canonical_merged_plus_frozen_probe_and_frozen_011_spec_manifest_basis_plus_012_spec_source_only',
 'defaults': {'merged_model_directory': '/root/model-diff-scratch/models/merged',
              'probe_path': '/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt',
              'attempt011_spec_path': 'experiments/attempts/011_generic_hessian_krylov_k4_prefix0_13/spec.json',
              'attempt011_manifest_path': 'experiments/attempts/011_generic_hessian_krylov_k4_prefix0_13/construction-manifest.json',
              'basis_directory': '/root/model-diff-scratch/artifacts/attempt011_hessian_krylov_k4',
              'artifact_path': '/root/model-diff-scratch/artifacts/attempt012_ritz_residual_filtered_inverse/inverse_response.pt',
              'construction_manifest_path': 'experiments/attempts/012_ritz_residual_filtered_inverse_k4_prefix0_13/construction-manifest.json'},
 'frozen_attempt011': {'attempt_id': '011_generic_hessian_krylov_k4_prefix0_13',
                       'spec_sha256': 'e2752003b2d3a173ca2d1f053a1ad6c7fed17d6e6a0ee606e02765376c606627',
                       'manifest_sha256': 'af334b5a580560900f38f5349ad8db51c6d1c2b61287d0b819b09829e52f91aa'},
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
 'spectral_filter': {'dimension': 4,
                     'dtype': 'cpu_float64',
                     'eigendecomposition': 'torch.linalg.eigh_B4_ascending',
                     'residual': 'beta4 * abs(V[-1, i])',
                     'keep': 'abs(theta_i) > rho_i',
                     'filter': '1/theta_i_if_kept_else_zero',
                     'coefficients': 'V @ (filter * V[0, :])',
                     'require_retained_mode': True,
                     'verification_rtol': 1e-10,
                     'verification_atol': 1e-12},
 'candidate': {'natural': 'norm_g * sum_j(c_j * q_j)',
               'gamma': 'norm_g / norm_x_natural',
               'probe': 'float32(gamma * x_natural)',
               'arithmetic': 'matrix_wise_cpu_float64_then_one_fp32_cast',
               'sign': 'positive_J_x_probe',
               'norm_verification_rtol': 1e-10,
               'norm_verification_atol': 1e-12},
 'diagnostic_probe': {'start': 0, 'stop': 10000, 'sample_count': 10000, 'batch_size': 32},
 'outputs': {'tensors': ['merged_mean', 'inverse_response'],
             'shape': [128, 2048],
             'dtype': 'contiguous_cpu_float32',
             'overwrite': False,
             'timestamps': False,
             'host_metadata': False,
             'gpu_metadata': False},
 'workload': {'gradient_batches': 0, 'hvp_batches': 0, 'jvp_directions': 1, 'jvp_batches': 313}}


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
        if path.exists() or path.is_symlink():
            raise ValueError(f"Output already exists: {path}")
    if artifact_path.resolve() == manifest_path.resolve():
        raise ValueError("Artifact and manifest paths must differ")
    if not manifest_path.parent.is_dir():
        raise ValueError(f"Missing manifest directory: {manifest_path.parent}")


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

def validate_provenance(spec_path: Path, manifest_path: Path, spec: dict[str, Any]) -> tuple[dict, dict]:
    frozen = spec['frozen_attempt011']
    for path, key in ((spec_path, 'spec_sha256'), (manifest_path, 'manifest_sha256')):
        if sha256_file(path) != frozen[key]:
            raise ValueError(f'Frozen Attempt011 {key} mismatch')
    previous = load_json_object(spec_path, 'Attempt011 specification')
    manifest = load_json_object(manifest_path, 'Attempt011 manifest')
    if previous['attempt_id'] != frozen['attempt_id'] or manifest['attempt_id'] != frozen['attempt_id']:
        raise ValueError('Attempt011 identity mismatch')
    if manifest['attempt_spec_sha256'] != frozen['spec_sha256']:
        raise ValueError('Attempt011 manifest/spec linkage mismatch')
    for key in ('model', 'eligible_tensors', 'probe', 'readout', 'jvp', 'canonical_checkpoint_files'):
        if previous[key] != spec[key]:
            raise ValueError(f'Attempt011 semantics mismatch: {key}')
    for key in ('probe', 'jvp'):
        if manifest[key] != spec[key]:
            raise ValueError(f'Attempt011 manifest semantics mismatch: {key}')
    if any(manifest['readout'][key] != value for key, value in spec['readout'].items()):
        raise ValueError('Attempt011 readout mismatch')
    if manifest['source_checkpoint']['files'] != spec['canonical_checkpoint_files']:
        raise ValueError('Attempt011 merged checkpoint linkage mismatch')
    for key in ('krylov', 'hessian', 'splits'):
        if manifest[key] != previous[key]:
            raise ValueError(f'Attempt011 construction semantics mismatch: {key}')
    if previous['krylov']['dimension'] != 4:
        raise ValueError('Expected frozen k=4 basis')
    matrices = manifest['matrices']
    names = [record['name'] for record in matrices]
    if names != manifest['eligible_parameters'] or names != sorted(set(names)):
        raise ValueError('Attempt011 eligible parameter ordering mismatch')
    if len(names) != spec['eligible_tensors']['expected_matrix_count']:
        raise ValueError('Attempt011 matrix count mismatch')
    records = manifest['basis_files']
    if len(records) != 4:
        raise ValueError('Expected four frozen basis files')
    for i, record in enumerate(records, 1):
        if (record['direction'] != i or record['filename'] != f'q{i}.safetensors' or
                record['dtype'] != 'float32' or record['size_bytes'] <= 0):
            raise ValueError('Invalid frozen basis record')
        require_sha256(record['sha256'], 'basis file')
    # Close read/hash races before any model computation.
    for path, key in ((spec_path, 'spec_sha256'), (manifest_path, 'manifest_sha256')):
        if sha256_file(path) != frozen[key]:
            raise ValueError('Frozen Attempt011 provenance changed while loading')
    return previous, manifest


def spectral_candidate(manifest: dict, spec: dict, torch_module: Any) -> dict:
    t = torch_module
    settings = spec['spectral_filter']
    projected = manifest['projected_hessian']
    b = t.tensor(projected['B4'], dtype=t.float64, device='cpu')
    gram = t.tensor(manifest['orthogonality']['gram_matrix'], dtype=t.float64, device='cpu')
    beta = float(projected['beta_4'])
    gnorm = float(manifest['gradient_norm'])
    if (b.shape != (4, 4) or gram.shape != (4, 4) or not bool(t.isfinite(b).all()) or
            not bool(t.isfinite(gram).all()) or not t.equal(b, b.T) or not t.equal(gram, gram.T) or
            not math.isfinite(beta) or beta < 0 or not math.isfinite(gnorm) or gnorm <= 0):
        raise ValueError('Invalid projected Hessian, basis Gram, beta4 or gradient norm')
    if float((gram - t.eye(4, dtype=t.float64)).abs().max()) > 1e-5:
        raise ValueError('Frozen basis is not numerically orthonormal')
    theta, vectors = t.linalg.eigh(b)
    rho = beta * vectors[-1].abs()
    for actual, key in ((theta, 'ritz_values'), (rho, 'last_step_only_ritz_residual_estimates')):
        expected = t.tensor(projected[key], dtype=t.float64)
        if (expected.shape != (4,) or not bool(t.isfinite(expected).all()) or
                not t.allclose(actual, expected, rtol=settings['verification_rtol'], atol=settings['verification_atol'])):
            raise ValueError(f'Frozen Ritz reproduction failed: {key}')
    keep = theta.abs() > rho
    if not bool(keep.any()):
        raise ValueError('No residual-resolved Ritz mode retained; no fallback')
    filt = t.zeros_like(theta)
    filt[keep] = theta[keep].reciprocal()
    coefficients = vectors @ (filt * vectors[0])
    squared_norm = float(coefficients @ gram @ coefficients) * gnorm ** 2
    if not bool(t.isfinite(coefficients).all()) or not math.isfinite(squared_norm) or squared_norm <= 0:
        raise ValueError('Natural candidate norm must be positive and finite')
    natural_norm = math.sqrt(squared_norm)
    return {'B4': b.tolist(), 'ritz_values': theta.tolist(), 'residual_estimates': rho.tolist(),
            'keep_mask': keep.tolist(), 'filter_values': filt.tolist(), 'coefficients': coefficients.tolist(),
            'coefficient_norm': float(coefficients.norm()), 'gradient_norm': gnorm,
            'natural_norm_from_frozen_gram': natural_norm, 'gamma_from_frozen_gram': gnorm / natural_norm,
            'beta_4': beta}


def verify_basis_files(directory: Path, manifest: dict) -> dict[Path, str]:
    hashes = {}
    for record in manifest['basis_files']:
        path = directory / record['filename']
        if (not path.is_file() or path.stat().st_size != record['size_bytes'] or
                sha256_file(path) != record['sha256']):
            raise ValueError(f'Frozen basis size/hash mismatch: {path.name}')
        hashes[path] = record['sha256']
    return hashes


class BasisReader:
    """Map only the four named basis files; never enumerate other artifacts."""
    def __init__(self, directory: Path, manifest: dict, torch_module: Any):
        self.directory, self.manifest, self.torch = directory, manifest, torch_module
        self.stack = ExitStack()

    def __enter__(self):
        from safetensors import safe_open
        try:
            self.files = [self.stack.enter_context(safe_open(str(self.directory / record['filename']),
                          framework='pt', device='cpu')) for record in self.manifest['basis_files']]
            names = set(self.manifest['eligible_parameters'])
            if any(set(file.keys()) != names for file in self.files):
                raise ValueError('Basis tensor keys mismatch')
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *args):
        self.stack.close()

    def matrices(self, record: dict) -> list[Any]:
        result = []
        for file in self.files:
            value = file.get_tensor(record['name'])
            if (value.dtype != self.torch.float32 or list(value.shape) != record['shape'] or
                    not value.is_contiguous() or not bool(self.torch.isfinite(value).all())):
                raise ValueError(f'Invalid basis tensor: {record["name"]}')
            result.append(value.to(dtype=self.torch.float64))
        return result


def natural_matrix(matrices: list[Any], candidate: dict) -> Any:
    value = matrices[0] * candidate['coefficients'][0]
    for coefficient, matrix in zip(candidate['coefficients'][1:], matrices[1:]):
        value.add_(matrix, alpha=coefficient)
    return value * candidate['gradient_norm']


def construct_tangent(reader: BasisReader, manifest: dict, candidate: dict,
                      parameters: dict, spec: dict, torch_module: Any) -> tuple[dict, dict]:
    t = torch_module
    gram = t.zeros((4, 4), dtype=t.float64)
    squares = []
    for record in manifest['matrices']:
        matrices = reader.matrices(record)
        for i in range(4):
            for j in range(i, 4):
                dot = t.dot(matrices[i].flatten(), matrices[j].flatten())
                gram[i, j] += dot
                if i != j:
                    gram[j, i] += dot
        squares.append(float(natural_matrix(matrices, candidate).square().sum()))
    settings = spec['candidate']
    if not t.allclose(gram, t.tensor(manifest['orthogonality']['gram_matrix'], dtype=t.float64),
                      rtol=settings['norm_verification_rtol'], atol=settings['norm_verification_atol']):
        raise ValueError('Frozen basis Gram reproduction failed')
    norm = math.sqrt(math.fsum(squares))
    if not math.isfinite(norm) or norm <= 0 or not math.isclose(norm, candidate['natural_norm_from_frozen_gram'],
                rel_tol=settings['norm_verification_rtol'], abs_tol=settings['norm_verification_atol']):
        raise ValueError('Natural norm reproduction failed')
    gamma = candidate['gradient_norm'] / norm
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('Numerical scale must be positive and finite')
    values, realized_squares = {}, []
    for record in manifest['matrices']:
        # Exactly one FP32 cast after the full CPU-FP64 linear combination/scaling.
        value = (natural_matrix(reader.matrices(record), candidate) * gamma).to(dtype=t.float32).contiguous()
        if not bool(t.isfinite(value).all()):
            raise ValueError('Non-finite candidate tangent')
        realized_squares.append(float(value.double().square().sum()))
        parameter = parameters[record['name']]
        if parameter.dtype != t.float32 or list(parameter.shape) != record['shape']:
            raise ValueError('Candidate parameter mismatch')
        values[record['name']] = value.to(device=parameter.device)
    realized = math.sqrt(math.fsum(realized_squares))
    if not math.isfinite(realized) or realized <= 0:
        raise ValueError('Realized tangent must be nonzero and finite')
    return values, {'natural_norm': norm, 'gamma': gamma, 'scaled_norm_before_fp32': gamma * norm,
                    'realized_fp32_probe_tangent_norm': realized,
                    'realized_to_gradient_norm_ratio': realized / candidate['gradient_norm']}


def build_artifact(merged: Any, response: Any, spec: dict, torch_module: Any) -> dict:
    result = {}
    for name, value in zip(spec['outputs']['tensors'], (merged, response)):
        if (value.device.type != 'cpu' or value.dtype != torch_module.float64 or
                list(value.shape) != spec['outputs']['shape'] or not bool(torch_module.isfinite(value).all())):
            raise ValueError('Invalid accumulated response')
        stored = value.to(dtype=torch_module.float32).contiguous()
        if not bool(torch_module.isfinite(stored).all()):
            raise ValueError('Non-finite stored response')
        result[name] = stored
    result['metadata'] = {'attempt_id': spec['attempt_id'], 'response_sign': 'positive_J_x_probe',
                          'sample_count': spec['probe']['sample_count'], 'readout': spec['readout']}
    return result


def construct(args: argparse.Namespace) -> None:
    require_output_absent(args.artifact_path, args.construction_manifest_path)
    spec = load_spec(args.attempt_spec_path)
    previous, manifest = validate_provenance(args.attempt011_spec_path, args.attempt011_manifest_path, spec)
    paths = [args.attempt_spec_path, args.attempt011_spec_path, args.attempt011_manifest_path,
             args.probe_path, Path(__file__).resolve()]
    for output in (args.artifact_path, args.construction_manifest_path):
        if (output.resolve().is_relative_to(args.basis_directory.resolve()) or
                output.resolve().is_relative_to(args.merged_model_dir.resolve()) or
                any(output.resolve() == path.resolve() for path in paths)):
            raise ValueError('Outputs must not overlap frozen inputs')
    hashes = {path: sha256_file(path) for path in paths}
    if (hashes[args.attempt011_spec_path] != spec['frozen_attempt011']['spec_sha256'] or
            hashes[args.attempt011_manifest_path] != spec['frozen_attempt011']['manifest_sha256'] or
            load_spec(args.attempt_spec_path) != spec):
        raise ValueError('Provenance changed while validating')
    hashes.update(verify_basis_files(args.basis_directory, manifest))
    source = checkpoint_file_records(args.merged_model_dir)
    if source != spec['canonical_checkpoint_files']:
        raise ValueError('Canonical merged checkpoint inventory/hash mismatch')
    import torch
    candidate = spectral_candidate(manifest, spec, torch)
    probe = load_probe(args.probe_path, spec['probe'], torch)  # Entire 10k tensor, no subset.
    model = load_local_model(args.merged_model_dir, args.device, torch)
    if getattr(model.config, 'hidden_size', None) != spec['readout']['hidden_size']:
        raise ValueError('Canonical model hidden size mismatch')
    eligible = discover_eligible_linear_weights(model, torch)
    parameters = {f'{name}.weight': module.weight for name, module in eligible}
    matrices = [{'name': name, 'shape': list(parameter.shape)} for name, parameter in parameters.items()]
    if matrices != manifest['matrices']:
        raise ValueError('Model weights do not match frozen basis matrices')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    before = model_state_hashes(model, torch)
    with BasisReader(args.basis_directory, manifest, torch) as reader:
        tangent, norms = construct_tangent(reader, manifest, candidate, parameters, spec, torch)
    try:
        merged, response, checks = compute_probe_response(model, probe, tangent, spec, torch, 'Attempt012')
    finally:
        tangent.clear()
        model.zero_grad(set_to_none=True)
    verify_model_unchanged(model, before, torch)
    artifact = build_artifact(merged, response, spec, torch)
    verify_unchanged(args.merged_model_dir, source, hashes)
    require_output_absent(args.artifact_path, args.construction_manifest_path)
    args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with args.artifact_path.open('xb') as stream:
        torch.save(artifact, stream)
    result = {'format_version': 1, 'attempt_id': spec['attempt_id'], 'information_policy': spec['information_policy'],
              'attempt_spec_sha256': hashes[args.attempt_spec_path],
              'constructor_script_sha256': hashes[Path(__file__).resolve()],
              'attempt011_spec_sha256': hashes[args.attempt011_spec_path],
              'attempt011_manifest_sha256': hashes[args.attempt011_manifest_path],
              'basis_files': manifest['basis_files'], 'source_checkpoint': {'files': source},
              'probe': spec['probe'], 'eligible_parameters': manifest['eligible_parameters'],
              'spectral_filter': spec['spectral_filter'], 'candidate_definition': spec['candidate'],
              'candidate': {**candidate, **norms}, 'jvp': spec['jvp'],
              'readout': {**spec['readout'], 'checks': checks}, 'workload': spec['workload'],
              'artifact': {'serialized_sha256': sha256_file(args.artifact_path),
                           'raw_tensors_sha256': {name: sha256_raw_float32_tensor(artifact[name], torch)
                                                 for name in spec['outputs']['tensors']}}}
    verify_unchanged(args.merged_model_dir, source, hashes)
    write_manifest(args.construction_manifest_path, result)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-spec-path', type=Path, default=PROJECT / 'experiments/attempts' / FROZEN_SPEC['attempt_id'] / 'spec.json')
    for option, key in (('merged-model-dir', 'merged_model_directory'), ('probe-path', 'probe_path'),
            ('attempt011-spec-path', 'attempt011_spec_path'), ('attempt011-manifest-path', 'attempt011_manifest_path'),
            ('basis-directory', 'basis_directory'), ('artifact-path', 'artifact_path'),
            ('construction-manifest-path', 'construction_manifest_path')):
        path = Path(FROZEN_SPEC['defaults'][key])
        parser.add_argument('--' + option, type=Path, default=path if path.is_absolute() else PROJECT / path)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        construct(parse_args(argv))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f'ERROR: {exc}. No fallback or partial-output reuse is permitted.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
