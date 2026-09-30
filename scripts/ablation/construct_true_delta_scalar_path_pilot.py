#!/usr/bin/env python3
"""Attempt 020: known-base scalar-path centered-regression pilot."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
PILOT_SUPPORT_PATH = Path(__file__).with_name('construct_centered_batch_hessian_regression.py')
PILOT_SUPPORT_SHA256 = '3d5d829f35b481926fd87e428c48b66b5f123f21b79e4a1d3ccbcdec7761c716'
loader = importlib.util.spec_from_file_location('attempt020_support019', PILOT_SUPPORT_PATH)
pilot = importlib.util.module_from_spec(loader)
loader.loader.exec_module(pilot)
support, execution, a, loss_helper = pilot.support, pilot.execution, pilot.a, pilot.loss_helper

FROZEN_SPEC = copy.deepcopy(pilot.FROZEN_SPEC)
FROZEN_SPEC.update({
    'attempt_id': '020_true_delta_scalar_path_centered_regression_pilot',
    'purpose': 'known_base_exploratory_scalar_path_mechanism_pilot_not_blind_recovery',
    'interpretation': 'True historical Delta is supplied; 64 fixed batches test scalar-path centered regression.',
    'information_policy': 'historical_base_final_frozen_005_corpus_exact_delta_and_frozen_helpers_only',
    'batch_partition': {'sample_count': 4096, 'sequence_length': 128, 'microbatch_size': 8,
                        'selected_microbatch_indices': list(range(0, 512, 8)),
                        'selected_count': 64, 'prediction_tokens_per_microbatch': 1016,
                        'row_rule': 'start=8*j,end=8*(j+1),end_exclusive',
                        'ordering': 'frozen_corpus_order_no_shuffle_no_replacement'},
    'scalar_path': {'evaluation': 'torch.func.functional_call',
                    'only_differentiation_variable': 'float64_scalar_t_cast_to_FP32_for_weight_update',
                    'functional_weight': 'FP32(endpoint_weight + FP32(t)*realized_FP32_Delta)',
                    'first_derivative': 'torch.autograd.grad(loss,t,create_graph=True)_at_t_zero',
                    'second_derivative': 'torch.autograd.grad(first_derivative,t)_at_t_zero',
                    'base_first_derivative_create_graph': False,
                    'parameter_requires_grad': False, 'fallback': False},
    'regression': {'beta_true': 1.0, 'primary_selected_batch_count': 64,
                   'arithmetic': 'CPU_float64_centered_stable_summation'},
    'workload': {'final_scalar_first_and_second_derivative_batches': 64,
                 'base_scalar_first_derivative_batches': 64,
                 'smoke_final_scalar_first_and_second_derivative_batches': 1,
                 'smoke_base_scalar_first_derivative_batches': 1},
})
FROZEN_SPEC.pop('hessian', None)
FROZEN_SPEC['sources'] = {**pilot.FROZEN_SPEC['sources'],
                          'scripts/ablation/construct_centered_batch_hessian_regression.py': PILOT_SUPPORT_SHA256}
FROZEN_SPEC['defaults'] = {**pilot.FROZEN_SPEC['defaults'],
    'artifact_path': '/root/model-diff-scratch/artifacts/attempt020_scalar_path_pilot/batch_scalars.pt',
    'manifest_path': 'experiments/attempts/020_true_delta_scalar_path_centered_regression_pilot/construction-manifest.json'}

SCALAR_KEYS = pilot.SCALAR_KEYS


def selected_batches():
    indices = list(range(0, 512, 8))
    return [(index, 8*index, 8*(index+1)) for index in indices]


def freeze_model(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.zero_grad(set_to_none=True)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameter freeze failed')


def scalar_path(model, batch, selected_names, direction, second_derivative):
    """Differentiate only a scalar t through FP32 stateless weight overrides."""
    if (batch.ndim != 2 or tuple(batch.shape) != (8, 128) or batch.dtype != torch.int64 or
            len(selected_names) != len(direction) or set(selected_names) != set(direction)):
        raise ValueError('Scalar-path batch or selected-coordinate inventory mismatch')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameters must be frozen for scalar path')
    parameters = dict(model.named_parameters())
    if not set(selected_names).issubset(parameters):
        raise ValueError('Missing selected endpoint coordinate')
    t = torch.zeros((), device=batch.device, dtype=torch.float64, requires_grad=True)
    t_fp32 = t.to(dtype=torch.float32)
    overrides = {}
    for name in selected_names:
        endpoint, delta = parameters[name], direction[name]
        if (endpoint.dtype != torch.float32 or endpoint.device != batch.device or
                delta.dtype != torch.float32 or delta.device != batch.device or
                endpoint.shape != delta.shape or not delta.is_contiguous() or delta.requires_grad):
            raise ValueError('Invalid FP32 scalar-path endpoint or Delta: '+name)
        overrides[name] = endpoint.detach() + t_fp32*delta
        if overrides[name].dtype != torch.float32:
            raise ValueError('Functional weight did not remain FP32')
    with torch.enable_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
        logits = torch.func.functional_call(model, overrides, (),
            {'input_ids': batch, 'use_cache': False}, tie_weights=True, strict=False).logits
        loss = loss_helper.causal_token_loss_sum(logits, batch, torch) / 1016
        if not bool(torch.isfinite(loss).item()):
            raise ValueError('Nonfinite scalar-path mean loss')
        first = torch.autograd.grad(loss, t, create_graph=second_derivative)[0]
        if first is None or not bool(torch.isfinite(first).item()):
            raise ValueError('Invalid scalar-path first derivative')
        first_value = float(first.detach().to(device='cpu', dtype=torch.float64).item())
        if second_derivative:
            second = torch.autograd.grad(first, t)[0]
            if second is None or not bool(torch.isfinite(second).item()):
                raise ValueError('Invalid scalar-path second derivative')
            second_value = float(second.detach().to(device='cpu', dtype=torch.float64).item())
        else:
            second_value = None
        del logits, loss, first, overrides
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Unexpected stored model gradient')
    return first_value, second_value


def synchronized_seconds(model, operation):
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = operation()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return result, time.perf_counter()-start


def raw_int64_sha256(value):
    return hashlib.sha256(value.numpy().astype('<i8', copy=False).tobytes(order='C')).hexdigest()


def verify_artifact(path, expected=None):
    try:
        artifact = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('Could not read scalar pilot artifact') from exc
    index_keys = ('microbatch_index', 'row_start', 'row_end')
    if not isinstance(artifact, dict) or set(artifact) != set(SCALAR_KEYS) | set(index_keys):
        raise ValueError('Scalar pilot artifact inventory mismatch')
    hashes = {}
    for name in SCALAR_KEYS:
        value = artifact[name]
        if len(pilot.scalar_series(value)) != 64:
            raise ValueError('Scalar pilot artifact shape mismatch')
        hashes[name] = pilot.raw_float64_sha256(value)
    expected_indices = {
        'microbatch_index': torch.arange(0, 512, 8, dtype=torch.int64),
        'row_start': torch.arange(0, 4096, 64, dtype=torch.int64),
        'row_end': torch.arange(8, 4104, 64, dtype=torch.int64),
    }
    for name, canonical in expected_indices.items():
        value = artifact[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or
                value.dtype != torch.int64 or tuple(value.shape) != (64,) or
                not value.is_contiguous() or not torch.equal(value, canonical)):
            raise ValueError('Scalar pilot artifact partition mismatch')
        hashes[name] = raw_int64_sha256(value)
    if not torch.equal(artifact['cleaned_directional_difference'],
                       artifact['final_directional_gradient']-artifact['base_directional_gradient']):
        raise ValueError('Scalar pilot cleaned difference mismatch')
    record = {'path': str(path), 'serialized_sha256': a.sha256_file(path),
              'raw_tensor_sha256': hashes,
              'tensors': {name: {'shape': [64], 'dtype': 'torch.float64', 'device': 'cpu', 'contiguous': True}
                          for name in SCALAR_KEYS} |
                         {name: {'shape': [64], 'dtype': 'torch.int64', 'device': 'cpu', 'contiguous': True}
                          for name in index_keys},
              'batch_partition': FROZEN_SPEC['batch_partition']}
    if expected is not None and record != expected:
        raise ValueError('Scalar pilot artifact hash/inventory mismatch')
    return record


def save_artifact(path, a_values, y_values, h_values):
    if path.exists() or path.is_symlink():
        raise ValueError('Scientific output already exists')
    artifact = {
        'microbatch_index': torch.arange(0, 512, 8, dtype=torch.int64),
        'row_start': torch.arange(0, 4096, 64, dtype=torch.int64),
        'row_end': torch.arange(8, 4104, 64, dtype=torch.int64),
        'base_directional_gradient': a_values,
        'final_directional_gradient': y_values,
        'endpoint_hessian_quadratic': h_values,
        'cleaned_directional_difference': (y_values-a_values).contiguous(),
    }
    for name in SCALAR_KEYS:
        if len(pilot.scalar_series(artifact[name])) != 64:
            raise ValueError('Scalar pilot output shape mismatch')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(artifact, stream)
    return verify_artifact(path)


def require_absent(args):
    pilot.require_absent(args)


def reject_forbidden_paths(args):
    pilot.reject_forbidden_paths(args)
    for key, value in vars(args).items():
        if isinstance(value, Path) and any(marker in str(value).lower() for marker in
                                           ('oracle', 'adapter', 'attempt018', '018_known', 'evaluation.json')):
            raise ValueError('Information firewall: forbidden input path '+key)


def validate_inputs(args):
    reject_forbidden_paths(args)
    spec = a.load_json_object(args.spec_path, 'Attempt020 spec')
    if spec != FROZEN_SPEC:
        raise ValueError('Frozen specification mismatch')
    source_paths = [PROJECT/name for name in spec['sources']]
    paths = [args.spec_path, args.corpus_spec_path, args.corpus_manifest_path,
             args.tokens_path, Path(__file__).resolve(), *source_paths]
    hashes = {path: a.sha256_file(path) for path in paths}
    for name, digest in spec['sources'].items():
        if hashes[PROJECT/name] != digest:
            raise ValueError('Frozen helper source mismatch')
    if hashes[PILOT_SUPPORT_PATH] != PILOT_SUPPORT_SHA256:
        raise ValueError('Loaded Attempt019 support source mismatch')
    for field, path in (('corpus_spec_sha256', args.corpus_spec_path),
                        ('corpus_manifest_sha256', args.corpus_manifest_path)):
        if hashes[path] != spec['frozen_corpus'][field]:
            raise ValueError('Frozen corpus provenance mismatch')
    base = a.checkpoint_file_records(args.base_directory)
    final = a.checkpoint_file_records(args.final_directory)
    if base != spec['base']['files'] or final != spec['canonical_checkpoint_files']:
        raise ValueError('Checkpoint inventory/hash mismatch')
    corpus = loss_helper.verify_corpus_manifest(args.corpus_manifest_path, args.corpus_spec_path,
        args.tokens_path, args.final_directory, loss_helper.load_corpus_spec(args.corpus_spec_path))
    if any(corpus[key] != spec['frozen_corpus'][key] for key in ('serialized_sha256', 'raw_tensor_sha256')):
        raise ValueError('Corpus artifact provenance mismatch')
    tokens = loss_helper.load_corpus_tokens(args.tokens_path, corpus['raw_tensor_sha256'], 4096, 128, torch)
    if len(selected_batches()) != 64:
        raise ValueError('Frozen pilot batch selection mismatch')
    for output in (args.artifact_path, args.manifest_path):
        if any(output.resolve() == path.resolve() or path.resolve().is_relative_to(output.resolve())
               for path in paths) or any(output.resolve().is_relative_to(directory.resolve())
                                          for directory in (args.base_directory, args.final_directory)):
            raise ValueError('Output overlaps frozen input')
    if args.artifact_path.resolve().is_relative_to(args.manifest_path.resolve()) or args.manifest_path.resolve().is_relative_to(args.artifact_path.resolve()):
        raise ValueError('Outputs overlap each other')
    def recheck():
        a.verify_unchanged(args.base_directory, base, hashes)
        a.verify_unchanged(args.final_directory, final, {})
    return spec, tokens, hashes, recheck


def construct(args):
    require_absent(args)
    spec, tokens, hashes, recheck = validate_inputs(args)
    base = a.load_local_model(args.base_directory, args.device, torch)
    eligible = support.discover(base, spec)
    freeze_model(base)
    base_before = a.model_state_hashes(base, torch)
    base_snapshot = support.snapshot(base, spec)
    support.verify_state(base, base_before, base_snapshot['execution'], spec)
    del eligible, base
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    final = a.load_local_model(args.final_directory, args.device, torch)
    eligible = support.discover(final, spec)
    freeze_model(final)
    final_before = a.model_state_hashes(final, torch)
    final_execution = execution.execution_snapshot(final, spec)
    semantic_audit = support.execution_audit(base_snapshot['execution'], final_execution)
    with execution.BaseTensorReader(args.base_directory, spec['base']['files']) as reader:
        direction_gpu, displacement_audit = support.displacement(
            final, eligible, base_snapshot['parameters'], reader.get)
    direction_cpu = {name: value.detach().to(device='cpu', dtype=torch.float32, copy=True).contiguous()
                     for name, value in direction_gpu.items()}
    names = [name+'.weight' for name, _ in eligible]
    if names != list(direction_gpu):
        raise ValueError('Scalar-path coordinate order mismatch')
    batches = selected_batches()[:1 if args.smoke_only else 64]
    y_values, h_values = [], []
    final_seconds = None
    for i, (_, start, end) in enumerate(batches):
        batch = tokens[start:end].to(eligible[0][1].weight.device)
        result, elapsed = synchronized_seconds(final, lambda: scalar_path(
            final, batch, names, direction_gpu, True))
        y, h = result
        y_values.append(y); h_values.append(h)
        if args.smoke_only:
            final_seconds = elapsed
        if (i+1) % 16 == 0:
            print(f'Final scalar-path batches {i+1}/{len(batches)}', flush=True)
        del batch
    support.verify_state(final, final_before, final_execution, spec)
    del direction_gpu, eligible, final
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    base = a.load_local_model(args.base_directory, args.device, torch)
    eligible = support.discover(base, spec)
    freeze_model(base)
    base_again_before = a.model_state_hashes(base, torch)
    if support.snapshot(base, spec) != base_snapshot:
        raise ValueError('Reloaded base model differs from audited historical base')
    direction_gpu = {name: value.to(eligible[0][1].weight.device) for name, value in direction_cpu.items()}
    names = [name+'.weight' for name, _ in eligible]
    a_values = []
    base_seconds = None
    for i, (_, start, end) in enumerate(batches):
        batch = tokens[start:end].to(eligible[0][1].weight.device)
        result, elapsed = synchronized_seconds(base, lambda: scalar_path(
            base, batch, names, direction_gpu, False))
        value, second = result
        if second is not None:
            raise ValueError('Unexpected base second derivative')
        a_values.append(value)
        if args.smoke_only:
            base_seconds = elapsed
        if (i+1) % 16 == 0:
            print(f'Base scalar-path batches {i+1}/{len(batches)}', flush=True)
        del batch
    support.verify_state(base, base_again_before, base_snapshot['execution'], spec)
    del base, eligible, direction_gpu, direction_cpu
    recheck()
    if args.smoke_only:
        print(json.dumps({'smoke_only': True, 'microbatch_index': 0,
                          'a_0': a_values[0], 'y_0': y_values[0], 'h_0': h_values[0],
                          'd_0': y_values[0]-a_values[0],
                          'w1_scalar_first_second_seconds': final_seconds,
                          'w0_scalar_first_seconds': base_seconds,
                          'delta_exact_fp64_norm': displacement_audit['exact_norm'],
                          'delta_realized_fp32_norm': displacement_audit['realized_fp32_norm'],
                          'delta_rounding_error_norm': displacement_audit['rounding_error_norm']}, sort_keys=True))
        return
    a_tensor = torch.tensor(a_values, dtype=torch.float64)
    y_tensor = torch.tensor(y_values, dtype=torch.float64)
    h_tensor = torch.tensor(h_values, dtype=torch.float64)
    regression = pilot.regression(a_tensor, y_tensor, h_tensor)
    artifact = save_artifact(args.artifact_path, a_tensor, y_tensor, h_tensor)
    manifest = {
        'format_version': 1, 'attempt_id': spec['attempt_id'], 'purpose': spec['purpose'],
        'information_policy': spec['information_policy'], 'hash_algorithm': 'sha256',
        'input_source_hashes': {str(path): digest for path, digest in hashes.items()},
        'base_checkpoint': spec['base'], 'final_checkpoint_files': spec['canonical_checkpoint_files'],
        'corpus': spec['frozen_corpus'], 'coordinates': spec['coordinates'],
        'displacement': displacement_audit, 'execution_semantics_audit': semantic_audit,
        'loss_normalization': spec['loss_normalization'], 'batch_partition': spec['batch_partition'],
        'scalar_path': spec['scalar_path'], 'artifact': artifact, 'regression': regression,
        'workload': spec['workload'],
        'interpretation': {
            'beta_g0': 'centered-regression endogeneity from batchwise historical-base gradient fluctuations',
            'beta_clean_minus_one': 'error after perfect base-gradient removal, including endpoint versus path-averaged Hessian mismatch',
            'beta_final_minus_one': 'final-only centered coefficient error given the true Delta template'},
        'publication_policy': 'no_overwrite_no_resume_partial_artifact_blocks_rerun',
    }
    recheck()
    verify_artifact(args.artifact_path, artifact)
    a.write_manifest(args.manifest_path, manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path', type=Path,
        default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'spec.json')
    for key, value in FROZEN_SPEC['defaults'].items():
        path = Path(value)
        parser.add_argument('--'+key.replace('_', '-'), type=Path,
                            default=path if path.is_absolute() else PROJECT/path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-only', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    construct(parse_args())
