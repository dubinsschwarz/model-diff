#!/usr/bin/env python3
"""Attempt 019: known-base centered batch regression with exact endpoint Hessian."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
SUPPORT_PATH = Path(__file__).with_name('construct_full_support_endpoint_hessian_forward.py')
SUPPORT_SHA256 = 'fcdff4c0cc90e9b0995a8b10896bad2f15ddf5ca98cb18ab900ab7d0e6174281'
loader = importlib.util.spec_from_file_location('attempt019_support017', SUPPORT_PATH)
support = importlib.util.module_from_spec(loader)
loader.loader.exec_module(support)
a, execution, loss_helper = support.a, support.execution, support.loss_helper

FROZEN_SPEC = copy.deepcopy(support.FROZEN_SPEC)
FROZEN_SPEC.update({
    'attempt_id': '019_known_base_true_delta_centered_batch_hessian_regression',
    'purpose': 'known_base_mechanistic_diagnostic_not_blind_recovery',
    'interpretation': 'True Delta is supplied. The final-only slope tests centered-regression error, not blind recovery.',
    'information_policy': 'historical_base_final_frozen_005_corpus_and_frozen_support_helpers_only',
    'loss_normalization': 'each_disjoint_8_row_batch_sum_of_1016_causal_prediction_losses_divided_by_1016',
    'batch_partition': {'sample_count': 4096, 'sequence_length': 128, 'microbatch_size': 8,
                        'microbatch_count': 512, 'prediction_tokens_per_microbatch': 1016,
                        'row_rule': 'start=8*i,end=8*(i+1),end_exclusive',
                        'ordering': 'frozen_corpus_order_no_shuffle_no_replacement'},
    'hessian': {'endpoint': 'W1_only', 'primitive': 'true_double_backward',
                'direction': 'same_realized_FP32_DeltaP_every_batch',
                'gradient_and_hessian': 'same_create_graph_gradient', 'fallback': False},
    'regression': {'primary_microbatch_size': 8, 'beta_true': 1.0,
                   'group_factors': [1, 2, 4, 8, 16, 32, 64],
                   'arithmetic': 'CPU_float64_centered_stable_summation'},
    'workload': {'base_ordinary_gradient_batches': 512, 'final_create_graph_gradient_hvp_batches': 512,
                 'smoke_base_ordinary_gradient_batches': 1, 'smoke_final_create_graph_gradient_hvp_batches': 1,
                 'additional_model_batches_for_aggregation': 0},
    'outputs': {'overwrite': False, 'timestamps': False, 'host_metadata': False, 'gpu_metadata': False},
})
for obsolete in ('generic_loss', 'hvp_loss', 'cleaned_rhs', 'metrics', 'temporary'):
    FROZEN_SPEC.pop(obsolete, None)
FROZEN_SPEC['sources'] = {**support.FROZEN_SPEC['sources'],
                          'scripts/ablation/construct_full_support_endpoint_hessian_forward.py': SUPPORT_SHA256}
FROZEN_SPEC['defaults'] = {k: v for k, v in support.FROZEN_SPEC['defaults'].items()
                           if k not in ('temporary_directory', 'artifact_path', 'manifest_path')}
FROZEN_SPEC['defaults'].update({
    'artifact_path': '/root/model-diff-scratch/artifacts/attempt019_centered_batch_regression/batch_scalars.pt',
    'manifest_path': 'experiments/attempts/019_known_base_true_delta_centered_batch_hessian_regression/construction-manifest.json',
})

SCALAR_KEYS = ('base_directional_gradient', 'final_directional_gradient',
               'endpoint_hessian_quadratic', 'cleaned_directional_difference')


def partition(sample_count=4096, batch_size=8, sequence_length=128):
    if (sample_count, batch_size, sequence_length) != (4096, 8, 128):
        raise ValueError('Frozen microbatch partition mismatch')
    return [(8*i, 8*(i+1)) for i in range(512)]


def batch_mean_loss(model, batch):
    if batch.ndim != 2 or tuple(batch.shape) != (8, 128) or batch.dtype != torch.int64:
        raise ValueError('Expected exactly 8 x 128 int64 batch')
    logits = model(input_ids=batch, use_cache=False).logits
    loss = loss_helper.causal_token_loss_sum(logits, batch, torch) / 1016
    if not bool(torch.isfinite(loss).item()):
        raise ValueError('Nonfinite mean batch loss')
    return loss


def matrixwise_dot(values, direction_cpu):
    """Each matrix contributes one CPU FP64 reduction; combine with fsum."""
    if len(values) != len(direction_cpu):
        raise ValueError('Directional matrix inventory mismatch')
    terms = []
    for value, vector in zip(values, direction_cpu):
        if value is None or value.shape != vector.shape or vector.device.type != 'cpu' or vector.dtype != torch.float32:
            raise ValueError('Directional matrix mismatch')
        left = value.detach().to(device='cpu', dtype=torch.float64)
        right = vector.to(dtype=torch.float64)
        term = float(torch.sum(left * right, dtype=torch.float64).item())
        if not math.isfinite(term):
            raise ValueError('Nonfinite matrixwise directional reduction')
        terms.append(term)
    result = math.fsum(terms)
    if not math.isfinite(result):
        raise ValueError('Nonfinite directional reduction')
    return result


def differentiable_cpu_dot(values, direction_cpu):
    """CPU FP64 matrix reductions, with a differentiable compensated sum."""
    if len(values) != len(direction_cpu):
        raise ValueError('Directional matrix inventory mismatch')
    terms = []
    for value, vector in zip(values, direction_cpu):
        if value is None or value.shape != vector.shape or vector.device.type != 'cpu' or vector.dtype != torch.float32:
            raise ValueError('Directional matrix mismatch')
        term = torch.sum(value.to(device='cpu', dtype=torch.float64) * vector,
                         dtype=torch.float64)
        if not bool(torch.isfinite(term).item()):
            raise ValueError('Nonfinite matrixwise directional reduction')
        terms.append(term)
    reported = math.fsum(float(term.detach().item()) for term in terms)
    if not math.isfinite(reported):
        raise ValueError('Nonfinite directional reduction')
    total = torch.zeros((), device='cpu', dtype=torch.float64)
    correction = torch.zeros_like(total)
    for term in terms:
        adjusted = term - correction
        updated = total + adjusted
        correction = (updated - total) - adjusted
        total = updated
    return reported, total


def batch_directional(model, batch, eligible, direction_gpu, direction_cpu, exact_hessian):
    params = [module.weight for _, module in eligible]
    names = [name+'.weight' for name, _ in eligible]
    if names != list(direction_gpu) or names != list(direction_cpu):
        raise ValueError('Direction does not match selected coordinate order')
    vectors_gpu = [direction_gpu[name] for name in names]
    vectors_cpu = [direction_cpu[name] for name in names]
    if any(v.dtype != torch.float32 or v.device != p.device or v.shape != p.shape or v.requires_grad
           for p, v in zip(params, vectors_gpu)):
        raise ValueError('Invalid realized FP32 Hessian direction')
    model.eval()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad(), torch.autocast(device_type=params[0].device.type, enabled=False):
        loss = batch_mean_loss(model, batch)
        gradients = torch.autograd.grad(loss, params, create_graph=exact_hessian)
        if exact_hessian:
            # This scalar uses the SAME selected gradients as the reported y.
            scalar, directional = differentiable_cpu_dot(gradients, vectors_cpu)
            hvp = torch.autograd.grad(directional, params)
            curvature = matrixwise_dot(hvp, vectors_cpu)
            del directional, hvp
        else:
            scalar = matrixwise_dot(gradients, vectors_cpu)
            curvature = None
        del gradients, loss
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Unexpected stored parameter gradient')
    return scalar, curvature


def scalar_series(values):
    if not isinstance(values, torch.Tensor) or values.ndim != 1 or values.dtype != torch.float64 or values.device.type != 'cpu' or not values.is_contiguous() or not bool(torch.isfinite(values).all()):
        raise ValueError('Expected finite contiguous CPU float64 scalar series')
    return [float(x) for x in values]


def regression(a_values, y_values, h_values):
    aa, yy, hh = map(scalar_series, (a_values, y_values, h_values))
    if not (len(aa) == len(yy) == len(hh) and len(hh) >= 2):
        raise ValueError('Regression series lengths mismatch')
    dd = [y-a for a, y in zip(aa, yy)]
    means = {key: math.fsum(v)/len(v) for key, v in (('h', hh), ('y', yy), ('a', aa), ('d', dd))}
    centered = {key: [x-means[key] for x in values] for key, values in
                (('h', hh), ('y', yy), ('a', aa), ('d', dd))}
    def inner(x, y):
        result = math.fsum(u*v for u, v in zip(centered[x], centered[y]))
        if not math.isfinite(result):
            raise ValueError('Nonfinite centered covariance')
        return result
    sums = {key: inner('h', key) for key in ('h', 'y', 'a', 'd')}
    if sums['h'] <= 0:
        raise ValueError('Zero Hessian variance')
    variances = {key: inner(key, key) for key in ('h', 'y', 'a', 'd')}
    beta = {key: sums[key]/sums['h'] for key in ('y', 'a', 'd')}
    if not math.isclose(beta['y'], beta['a']+beta['d'], rel_tol=1e-11, abs_tol=1e-11):
        raise ValueError('Slope decomposition failed')
    def corr(key):
        return sums[key]/math.sqrt(sums['h']*variances[key]) if variances[key] > 0 else None
    def sign(value):
        return 'positive' if value > 0 else 'negative' if value < 0 else 'zero'
    errors = {'base_endogeneity': beta['a'], 'cleaned_error': beta['d']-1,
              'final_only_error': beta['y']-1}
    result = {
        'count': len(hh), 'beta_true': 1.0,
        'beta_final': beta['y'], 'beta_g0': beta['a'], 'beta_clean': beta['d'],
        'sums': {'S_hh': sums['h'], 'S_hy': sums['y'], 'S_ha': sums['a'], 'S_hd': sums['d']},
        'slope_decomposition_residual': beta['y']-beta['a']-beta['d'],
        'errors': {key: {'value': value, 'absolute': abs(value), 'sign': sign(value)}
                   for key, value in errors.items()},
        'statistics': {key: {'mean': means[key], 'standard_deviation': math.sqrt(variances[key]/len(hh))}
                       for key in ('h', 'y', 'a', 'd')},
        'correlations': {'h_y': corr('y'), 'h_a': corr('a'), 'h_d': corr('d')},
        'centered_r_squared': {'y_from_h': corr('y')**2 if corr('y') is not None else None,
                               'd_from_h': corr('d')**2 if corr('d') is not None else None},
    }
    return result


def aggregate(values, factor):
    scalar_series(values)
    if factor not in FROZEN_SPEC['regression']['group_factors'] or len(values) % factor:
        raise ValueError('Invalid aggregation factor')
    return torch.tensor([math.fsum(float(x) for x in values[i:i+factor])/factor
                         for i in range(0, len(values), factor)], dtype=torch.float64)


def diagnostics(a_values, y_values, h_values):
    if len(a_values) != 512 or len(y_values) != 512 or len(h_values) != 512:
        raise ValueError('Expected 512 microbatch records')
    rows = []
    for factor in FROZEN_SPEC['regression']['group_factors']:
        result = regression(*(aggregate(v, factor) for v in (a_values, y_values, h_values)))
        rows.append({'group_factor': factor, 'effective_batch_size': 8*factor,
                     'group_count': 512//factor, **result})
    return rows[0], rows[1:]


def raw_float64_sha256(value):
    scalar_series(value)
    return hashlib.sha256(value.numpy().astype('<f8', copy=False).tobytes(order='C')).hexdigest()


def verify_artifact(path, expected=None):
    try:
        artifact = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('Could not read scalar artifact') from exc
    if not isinstance(artifact, dict) or set(artifact) != set(SCALAR_KEYS) | {'batch_start', 'batch_end'}:
        raise ValueError('Scalar artifact inventory mismatch')
    hashes = {}
    for name in SCALAR_KEYS:
        value = artifact[name]
        if len(scalar_series(value)) != 512:
            raise ValueError('Scalar artifact shape mismatch')
        hashes[name] = raw_float64_sha256(value)
    starts = torch.arange(0, 4096, 8, dtype=torch.int64)
    if any(not isinstance(artifact[key], torch.Tensor) or artifact[key].dtype != torch.int64 or
           artifact[key].device.type != 'cpu' or not artifact[key].is_contiguous() or
           tuple(artifact[key].shape) != (512,) for key in ('batch_start', 'batch_end')):
        raise ValueError('Scalar artifact partition index metadata mismatch')
    if not (torch.equal(artifact['batch_start'], starts) and torch.equal(artifact['batch_end'], starts+8)):
        raise ValueError('Scalar artifact partition mismatch')
    if not torch.equal(artifact['cleaned_directional_difference'],
                       artifact['final_directional_gradient']-artifact['base_directional_gradient']):
        raise ValueError('Scalar artifact cleaned difference mismatch')
    record = {'path': str(path), 'serialized_sha256': a.sha256_file(path),
              'raw_tensor_sha256': hashes,
              'tensors': {name: {'shape': [512], 'dtype': 'torch.float64', 'device': 'cpu', 'contiguous': True}
                          for name in SCALAR_KEYS},
              'batch_indices': {'start_shape': [512], 'end_shape': [512], 'dtype': 'torch.int64',
                                'start_inclusive': True, 'end_exclusive': True, 'first': [0, 8],
                                'last': [4088, 4096]}}
    if expected is not None and record != expected:
        raise ValueError('Scalar artifact hash/inventory mismatch')
    return record


def save_artifact(path, a_values, y_values, h_values):
    if path.exists() or path.is_symlink():
        raise ValueError('Scientific output already exists')
    values = {'base_directional_gradient': a_values, 'final_directional_gradient': y_values,
              'endpoint_hessian_quadratic': h_values,
              'cleaned_directional_difference': (y_values-a_values).contiguous(),
              'batch_start': torch.arange(0, 4096, 8, dtype=torch.int64),
              'batch_end': torch.arange(8, 4097, 8, dtype=torch.int64)}
    for name in SCALAR_KEYS:
        if len(scalar_series(values[name])) != 512:
            raise ValueError('Scalar output shape mismatch')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(values, stream)
    return verify_artifact(path)


def reject_forbidden_paths(args):
    for key, value in vars(args).items():
        if isinstance(value, Path) and any(part in str(value).lower() for part in
                                           ('oracle_adl', 'adapter', 'evaluate_attempt018')):
            raise ValueError('Information firewall: forbidden input path '+key)


def require_absent(args):
    for path in (args.artifact_path, args.manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError('Scientific output already exists')


def validate_inputs(args):
    reject_forbidden_paths(args)
    spec = a.load_json_object(args.spec_path, 'Attempt019 spec')
    if spec != FROZEN_SPEC:
        raise ValueError('Frozen specification mismatch')
    source_paths = [PROJECT / name for name in spec['sources']]
    paths = [args.spec_path, args.corpus_spec_path, args.corpus_manifest_path,
             args.tokens_path, Path(__file__).resolve(), *source_paths]
    hashes = {path: a.sha256_file(path) for path in paths}
    for name, digest in spec['sources'].items():
        if hashes[PROJECT/name] != digest:
            raise ValueError('Frozen helper source mismatch')
    if hashes[SUPPORT_PATH] != SUPPORT_SHA256:
        raise ValueError('Loaded support source mismatch')
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
    partition()
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
    base_before = a.model_state_hashes(base, torch)
    base_snapshot = support.snapshot(base, spec)
    support.verify_state(base, base_before, base_snapshot['execution'], spec)
    del eligible, base
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    final = a.load_local_model(args.final_directory, args.device, torch)
    eligible = support.discover(final, spec)
    final_before = a.model_state_hashes(final, torch)
    final_execution = execution.execution_snapshot(final, spec)
    semantic_audit = support.execution_audit(base_snapshot['execution'], final_execution)
    with execution.BaseTensorReader(args.base_directory, spec['base']['files']) as reader:
        direction_gpu, displacement_audit = support.displacement(
            final, eligible, base_snapshot['parameters'], reader.get)
    direction_cpu = {name: value.detach().to(device='cpu', dtype=torch.float32, copy=True).contiguous()
                     for name, value in direction_gpu.items()}
    count = 1 if args.smoke_only else 512
    y_values, h_values = [], []
    for i, (start, end) in enumerate(partition()[:count]):
        batch = tokens[start:end].to(eligible[0][1].weight.device)
        y, h = batch_directional(final, batch, eligible, direction_gpu, direction_cpu, True)
        y_values.append(y); h_values.append(h)
        del batch
        if (i+1) % 32 == 0:
            print(f'Final exact-Hessian batches {i+1}/{count}', flush=True)
    support.verify_state(final, final_before, final_execution, spec)
    del direction_gpu, eligible, final
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    base = a.load_local_model(args.base_directory, args.device, torch)
    eligible = support.discover(base, spec)
    base_again_before = a.model_state_hashes(base, torch)
    if support.snapshot(base, spec) != base_snapshot:
        raise ValueError('Reloaded base model differs from audited historical base')
    direction_gpu = {name: value.to(eligible[0][1].weight.device) for name, value in direction_cpu.items()}
    a_values = []
    for i, (start, end) in enumerate(partition()[:count]):
        batch = tokens[start:end].to(eligible[0][1].weight.device)
        value, curvature = batch_directional(base, batch, eligible, direction_gpu, direction_cpu, False)
        if curvature is not None:
            raise ValueError('Unexpected base Hessian calculation')
        a_values.append(value)
        del batch
        if (i+1) % 32 == 0:
            print(f'Base gradient batches {i+1}/{count}', flush=True)
    support.verify_state(base, base_again_before, base_snapshot['execution'], spec)
    del base, eligible, direction_gpu, direction_cpu
    recheck()
    if args.smoke_only:
        print(json.dumps({'smoke_only': True, 'a_0': a_values[0], 'y_0': y_values[0],
                          'h_0': h_values[0], 'd_0': y_values[0]-a_values[0],
                          'delta_exact_fp64_norm': displacement_audit['exact_norm'],
                          'delta_realized_fp32_norm': displacement_audit['realized_fp32_norm'],
                          'delta_rounding_error_norm': displacement_audit['rounding_error_norm']}, sort_keys=True))
        return
    a_tensor = torch.tensor(a_values, dtype=torch.float64)
    y_tensor = torch.tensor(y_values, dtype=torch.float64)
    h_tensor = torch.tensor(h_values, dtype=torch.float64)
    primary, secondary = diagnostics(a_tensor, y_tensor, h_tensor)
    artifact = save_artifact(args.artifact_path, a_tensor, y_tensor, h_tensor)
    manifest = {
        'format_version': 1, 'attempt_id': spec['attempt_id'], 'purpose': spec['purpose'],
        'information_policy': spec['information_policy'], 'hash_algorithm': 'sha256',
        'input_source_hashes': {str(path): digest for path, digest in hashes.items()},
        'base_checkpoint': spec['base'], 'final_checkpoint_files': spec['canonical_checkpoint_files'],
        'corpus': spec['frozen_corpus'], 'coordinates': spec['coordinates'],
        'displacement': displacement_audit, 'execution_semantics_audit': semantic_audit,
        'loss_normalization': spec['loss_normalization'], 'batch_partition': spec['batch_partition'],
        'hessian': spec['hessian'], 'artifact': artifact, 'primary': primary,
        'secondary_aggregation': secondary, 'workload': spec['workload'],
        'interpretation': {
            'beta_g0': 'centered-regression endogeneity contribution from batchwise historical-base gradient fluctuations',
            'beta_clean_minus_one': 'error remaining after perfect base-gradient removal, including endpoint-Hessian versus path-averaged-Hessian mismatch',
            'beta_final_minus_one': 'total coefficient error for final-only centered regression given the true Delta template',
        },
        'publication_policy': 'no_overwrite_no_resume_partial_artifact_blocks_rerun',
    }
    recheck()
    verify_artifact(args.artifact_path, artifact)
    a.write_manifest(args.manifest_path, manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path', type=Path, default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'spec.json')
    for key, value in FROZEN_SPEC['defaults'].items():
        path = Path(value)
        parser.add_argument('--'+key.replace('_', '-'), type=Path,
                            default=path if path.is_absolute() else PROJECT/path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-only', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    construct(parse_args())
