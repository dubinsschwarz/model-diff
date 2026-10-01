#!/usr/bin/env python3
"""Attempt 021: known-base true-path curvature quadrature diagnostic."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
PRIOR_PATH = Path(__file__).with_name('construct_true_delta_scalar_path_pilot.py')
PRIOR_SOURCE_SHA256 = '647b08b1f1083adcee715421fb532c7e3b8dd771d004befcac546ddfcfbe1157'
PRIOR_MANIFEST_SHA256 = 'd8e86fef4cd1a94ef796591d7c5557d2f80029c2839d2da03ab121ca404c4111'
PRIOR_ARTIFACT_SHA256 = '8d940102d28c766290606764fc2a3d5adc10a042d98fa5490d0563f2bd75e342'
loader = importlib.util.spec_from_file_location('attempt021_support020', PRIOR_PATH)
prior = importlib.util.module_from_spec(loader)
loader.loader.exec_module(prior)
support, execution, a, loss_helper = prior.support, prior.execution, prior.a, prior.loss_helper

FROZEN_SPEC = copy.deepcopy(prior.FROZEN_SPEC)
FROZEN_SPEC.update({
    'attempt_id': '021_true_delta_path_curvature_quadrature_pilot',
    'purpose': 'known_base_true_path_curvature_quadrature_mechanistic_diagnostic_not_blind_recovery',
    'interpretation': 'Tests whether path curvature accounts for the frozen Attempt-020 cleaned-slope error; no method selection.',
    'information_policy': 'historical_base_final_frozen_005_corpus_exact_delta_frozen_020_scalars_and_sources_only',
    'prior_attempt': {'attempt_id': prior.FROZEN_SPEC['attempt_id'],
                      'manifest_sha256': PRIOR_MANIFEST_SHA256,
                      'artifact_sha256': PRIOR_ARTIFACT_SHA256,
                      'artifact_fields': ['base_directional_gradient', 'final_directional_gradient',
                                          'endpoint_hessian_quadratic', 'cleaned_directional_difference']},
    'scalar_path': {'evaluation': 'torch.func.functional_call',
                    'origin': 'W0', 'direction': 'same_realized_FP32_Delta_as_Attempt020',
                    'weight': 'FP32(W0_weight + FP32(t)*Delta)',
                    't_leaf': 'float64_scalar', 'evaluation_points': [0.0, 0.5],
                    'only_differentiation_variable': 't',
                    'first_derivative': 'torch.autograd.grad(loss,t,create_graph=True)',
                    'second_derivative': 'torch.autograd.grad(first_derivative,t)',
                    'endpoint': 'frozen_Attempt020_h1_no_W0_plus_Delta_reconstruction',
                    'parameter_requires_grad': False, 'fallback': False},
    'quadrature': {'comparison_order': ['endpoint', 'midpoint', 'trapezoid', 'simpson'],
                   'endpoint': 'h1', 'midpoint': 'hmid',
                   'trapezoid': '(h0+h1)/2', 'simpson': '(h0+4*hmid+h1)/6',
                   'target': 'd=y-a', 'beta_true': 1.0, 'selection': False},
    'workload': {'base_scalar_second_derivative_batches_at_t0': 64,
                 'base_scalar_second_derivative_batches_at_tmid': 64,
                 'smoke_batches_at_each_point': 1,
                 'final_endpoint_derivative_batches': 0},
})
FROZEN_SPEC['sources'] = {**prior.FROZEN_SPEC['sources'],
                          'scripts/ablation/construct_true_delta_scalar_path_pilot.py': PRIOR_SOURCE_SHA256}
FROZEN_SPEC['defaults'] = {**prior.FROZEN_SPEC['defaults'],
    'prior_artifact_path': prior.FROZEN_SPEC['defaults']['artifact_path'],
    'prior_manifest_path': prior.FROZEN_SPEC['defaults']['manifest_path'],
    'artifact_path': '/root/model-diff-scratch/artifacts/attempt021_path_curvature_quadrature/batch_scalars.pt',
    'manifest_path': 'experiments/attempts/021_true_delta_path_curvature_quadrature_pilot/construction-manifest.json'}

FLOAT_KEYS = ('h0', 'hmid', 'h1', 'd', 'midpoint', 'trapezoid', 'simpson')


def scalar_curvature(model, batch, names, direction, t_value):
    """Attempt-020 scalar path at t=0, generalized only for t=0.5."""
    if t_value == 0.0:
        return prior.scalar_path(model, batch, names, direction, True)[1]
    if t_value != 0.5:
        raise ValueError('Only frozen path points t=0 and t=0.5 are allowed')
    if (batch.ndim != 2 or tuple(batch.shape) != (8, 128) or batch.dtype != torch.int64 or
            len(names) != len(direction) or set(names) != set(direction)):
        raise ValueError('Scalar-path batch or selected-coordinate inventory mismatch')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameters must be frozen for scalar path')
    parameters = dict(model.named_parameters())
    if not set(names).issubset(parameters):
        raise ValueError('Missing selected base coordinate')
    t = torch.tensor(0.5, device=batch.device, dtype=torch.float64, requires_grad=True)
    t_fp32 = t.to(dtype=torch.float32)
    overrides = {}
    for name in names:
        endpoint, delta = parameters[name], direction[name]
        if (endpoint.dtype != torch.float32 or endpoint.device != batch.device or
                delta.dtype != torch.float32 or delta.device != batch.device or
                endpoint.shape != delta.shape or not delta.is_contiguous() or delta.requires_grad):
            raise ValueError('Invalid FP32 scalar-path base or Delta: '+name)
        overrides[name] = endpoint.detach() + t_fp32*delta
        if overrides[name].dtype != torch.float32:
            raise ValueError('Functional weight did not remain FP32')
    with torch.enable_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
        logits = torch.func.functional_call(model, overrides, (),
            {'input_ids': batch, 'use_cache': False}, tie_weights=True, strict=False).logits
        loss = loss_helper.causal_token_loss_sum(logits, batch, torch) / 1016
        if not bool(torch.isfinite(loss).item()):
            raise ValueError('Nonfinite scalar-path mean loss')
        first = torch.autograd.grad(loss, t, create_graph=True)[0]
        if first is None or not bool(torch.isfinite(first).item()):
            raise ValueError('Invalid scalar-path first derivative')
        second = torch.autograd.grad(first, t)[0]
        if second is None or not bool(torch.isfinite(second).item()):
            raise ValueError('Invalid scalar-path second derivative')
        result = float(second.detach().to(device='cpu', dtype=torch.float64).item())
        del logits, loss, first, second, overrides
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Unexpected stored model gradient')
    return result


def curvature_approximations(h0, hmid, h1):
    for value in (h0, hmid, h1):
        if len(prior.pilot.scalar_series(value)) != 64:
            raise ValueError('Expected 64 curvature scalars')
    return {'endpoint': h1, 'midpoint': hmid,
            'trapezoid': ((h0+h1)/2).contiguous(),
            'simpson': ((h0+4*hmid+h1)/6).contiguous()}


def regression(q, d):
    """Centered CPU FP64 diagnostics, using Attempt-019's stable regression."""
    if len(prior.pilot.scalar_series(q)) != 64 or len(prior.pilot.scalar_series(d)) != 64:
        raise ValueError('Expected 64 scalar records')
    record = prior.pilot.regression(torch.zeros_like(d), d, q)
    return {'beta': record['beta_clean'], 'beta_minus_one': record['beta_clean']-1,
            'absolute_error': abs(record['beta_clean']-1),
            'correlation': record['correlations']['h_d'],
            'centered_r_squared': record['centered_r_squared']['d_from_h'],
            'S_qq': record['sums']['S_hh'],
            'q': record['statistics']['h'], 'd': record['statistics']['d'],
            'count': record['count']}


def endpoint_consistency(artifact, prior_manifest):
    h1 = artifact['endpoint_hessian_quadratic']
    d = artifact['cleaned_directional_difference']
    actual = regression(h1, d)
    expected = prior_manifest['regression']
    for field, observed in (('beta_clean', actual['beta']),
                            ('correlations', actual['correlation'])):
        reference = expected[field]['h_d'] if field == 'correlations' else expected[field]
        if not math.isclose(observed, reference, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Attempt-020 endpoint regression consistency failed')
    return actual


def raw_tensor_sha256(value):
    if value.dtype == torch.float64:
        return prior.pilot.raw_float64_sha256(value)
    if value.dtype == torch.int64 and value.device.type == 'cpu' and value.is_contiguous():
        return prior.raw_int64_sha256(value)
    raise ValueError('Invalid raw tensor dtype')


def verify_artifact(path, expected=None):
    try:
        artifact = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('Could not read Attempt-021 artifact') from exc
    if not isinstance(artifact, dict) or set(artifact) != set(FLOAT_KEYS) | {'microbatch_index'}:
        raise ValueError('Attempt-021 artifact inventory mismatch')
    indices = torch.arange(0, 512, 8, dtype=torch.int64)
    value = artifact['microbatch_index']
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or
            value.device.type != 'cpu' or not value.is_contiguous() or
            tuple(value.shape) != (64,) or not torch.equal(value, indices)):
        raise ValueError('Attempt-021 batch inventory mismatch')
    for key in FLOAT_KEYS:
        if len(prior.pilot.scalar_series(artifact[key])) != 64:
            raise ValueError('Attempt-021 scalar shape mismatch')
    checks = curvature_approximations(artifact['h0'], artifact['hmid'], artifact['h1'])
    for name in ('midpoint', 'trapezoid', 'simpson'):
        if not torch.equal(artifact[name], checks[name]):
            raise ValueError('Attempt-021 quadrature artifact mismatch')
    record = {'path': str(path), 'serialized_sha256': a.sha256_file(path),
              'raw_tensor_sha256': {key: raw_tensor_sha256(value) for key, value in artifact.items()},
              'tensors': {key: {'shape': [64], 'dtype': 'torch.float64', 'device': 'cpu', 'contiguous': True}
                          for key in FLOAT_KEYS} |
                         {'microbatch_index': {'shape': [64], 'dtype': 'torch.int64',
                                               'device': 'cpu', 'contiguous': True}},
              'batch_partition': FROZEN_SPEC['batch_partition']}
    if expected is not None and record != expected:
        raise ValueError('Attempt-021 artifact hash/inventory mismatch')
    return record


def save_artifact(path, h0, hmid, h1, d):
    if path.exists() or path.is_symlink():
        raise ValueError('Scientific output already exists')
    approximations = curvature_approximations(h0, hmid, h1)
    if len(prior.pilot.scalar_series(d)) != 64:
        raise ValueError('Expected 64 cleaned directional differences')
    artifact = {'microbatch_index': torch.arange(0, 512, 8, dtype=torch.int64),
                'h0': h0, 'hmid': hmid, 'h1': h1, 'd': d,
                'midpoint': approximations['midpoint'],
                'trapezoid': approximations['trapezoid'],
                'simpson': approximations['simpson']}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(artifact, stream)
    return verify_artifact(path)


def require_absent(args):
    prior.require_absent(args)


def validate_inputs(args):
    prior.reject_forbidden_paths(args)
    spec = a.load_json_object(args.spec_path, 'Attempt021 spec')
    if spec != FROZEN_SPEC:
        raise ValueError('Frozen specification mismatch')
    source_paths = [PROJECT/name for name in spec['sources']]
    paths = [args.spec_path, args.corpus_spec_path, args.corpus_manifest_path,
             args.tokens_path, args.prior_artifact_path, args.prior_manifest_path,
             Path(__file__).resolve(), *source_paths]
    hashes = {path: a.sha256_file(path) for path in paths}
    for name, digest in spec['sources'].items():
        if hashes[PROJECT/name] != digest:
            raise ValueError('Frozen helper source mismatch')
    if hashes[PRIOR_PATH] != PRIOR_SOURCE_SHA256:
        raise ValueError('Loaded Attempt-020 source mismatch')
    if hashes[args.prior_manifest_path] != PRIOR_MANIFEST_SHA256 or hashes[args.prior_artifact_path] != PRIOR_ARTIFACT_SHA256:
        raise ValueError('Frozen Attempt-020 input hash mismatch')
    prior_manifest = a.load_json_object(args.prior_manifest_path, 'Attempt020 construction manifest')
    if (prior_manifest.get('attempt_id') != prior.FROZEN_SPEC['attempt_id'] or
            prior_manifest.get('purpose') != prior.FROZEN_SPEC['purpose'] or
            prior_manifest.get('artifact', {}).get('path') != str(args.prior_artifact_path) or
            prior_manifest['artifact'].get('serialized_sha256') != PRIOR_ARTIFACT_SHA256 or
            prior_manifest.get('coordinates') != spec['coordinates'] or
            prior_manifest.get('corpus') != spec['frozen_corpus']):
        raise ValueError('Frozen Attempt-020 manifest provenance mismatch')
    prior_artifact_record = prior.verify_artifact(args.prior_artifact_path, prior_manifest['artifact'])
    prior_artifact = torch.load(args.prior_artifact_path, map_location='cpu', weights_only=True)
    endpoint = endpoint_consistency(prior_artifact, prior_manifest)
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
    if len(prior.selected_batches()) != 64 or not torch.equal(prior_artifact['microbatch_index'],
                                                              torch.arange(0, 512, 8, dtype=torch.int64)):
        raise ValueError('Frozen pilot batch partition mismatch')
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
    return spec, tokens, hashes, recheck, prior_manifest, prior_artifact, prior_artifact_record, endpoint


def construct(args):
    require_absent(args)
    spec, tokens, hashes, recheck, prior_manifest, prior_artifact, prior_record, endpoint = validate_inputs(args)
    base = a.load_local_model(args.base_directory, args.device, torch)
    eligible = support.discover(base, spec)
    prior.freeze_model(base)
    base_before = a.model_state_hashes(base, torch)
    base_snapshot = support.snapshot(base, spec)
    support.verify_state(base, base_before, base_snapshot['execution'], spec)
    del eligible, base
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    final = a.load_local_model(args.final_directory, args.device, torch)
    eligible = support.discover(final, spec)
    prior.freeze_model(final)
    final_before = a.model_state_hashes(final, torch)
    final_execution = execution.execution_snapshot(final, spec)
    semantic_audit = support.execution_audit(base_snapshot['execution'], final_execution)
    with execution.BaseTensorReader(args.base_directory, spec['base']['files']) as reader:
        direction_gpu, displacement_audit = support.displacement(
            final, eligible, base_snapshot['parameters'], reader.get)
    for key in ('exact_norm', 'realized_fp32_norm', 'rounding_error_norm', 'nonzero_count'):
        if not math.isclose(displacement_audit[key], prior_manifest['displacement'][key], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Attempt-020 Delta audit consistency failed')
    direction_cpu = {name: value.detach().to(device='cpu', dtype=torch.float32, copy=True).contiguous()
                     for name, value in direction_gpu.items()}
    support.verify_state(final, final_before, final_execution, spec)
    del direction_gpu, eligible, final
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    base = a.load_local_model(args.base_directory, args.device, torch)
    eligible = support.discover(base, spec)
    prior.freeze_model(base)
    base_again_before = a.model_state_hashes(base, torch)
    if support.snapshot(base, spec) != base_snapshot:
        raise ValueError('Reloaded base differs from audited W0')
    direction_gpu = {name: value.to(eligible[0][1].weight.device) for name, value in direction_cpu.items()}
    names = [name+'.weight' for name, _ in eligible]
    if names != list(direction_gpu):
        raise ValueError('Scalar-path coordinate order mismatch')
    batches = prior.selected_batches()[:1 if args.smoke_only else 64]
    h0_values, hmid_values = [], []
    smoke_seconds = {}
    for i, (_, start, end) in enumerate(batches):
        batch = tokens[start:end].to(eligible[0][1].weight.device)
        for label, point, target in (('h0', 0.0, h0_values), ('hmid', 0.5, hmid_values)):
            value, elapsed = prior.synchronized_seconds(base, lambda: scalar_curvature(
                base, batch, names, direction_gpu, point))
            target.append(value)
            if args.smoke_only:
                smoke_seconds[label] = elapsed
        if (i+1) % 16 == 0:
            print(f'Path curvature batches {i+1}/{len(batches)}', flush=True)
        del batch
    support.verify_state(base, base_again_before, base_snapshot['execution'], spec)
    del base, eligible, direction_gpu, direction_cpu
    recheck()
    h1 = prior_artifact['endpoint_hessian_quadratic']
    d = prior_artifact['cleaned_directional_difference']
    if args.smoke_only:
        print(json.dumps({'smoke_only': True, 'microbatch_index': 0,
                          'h0_0': h0_values[0], 'hmid_0': hmid_values[0],
                          'frozen_h1_0': float(h1[0]), 'frozen_d_0': float(d[0]),
                          'h0_seconds': smoke_seconds['h0'], 'hmid_seconds': smoke_seconds['hmid'],
                          'delta_exact_fp64_norm': displacement_audit['exact_norm'],
                          'delta_realized_fp32_norm': displacement_audit['realized_fp32_norm'],
                          'delta_rounding_error_norm': displacement_audit['rounding_error_norm']}, sort_keys=True))
        return
    h0 = torch.tensor(h0_values, dtype=torch.float64)
    hmid = torch.tensor(hmid_values, dtype=torch.float64)
    approximations = curvature_approximations(h0, hmid, h1)
    regressions = {name: regression(approximations[name], d)
                   for name in spec['quadrature']['comparison_order']}
    if regressions['endpoint'] != endpoint:
        raise ValueError('Endpoint regression changed after computation')
    artifact = save_artifact(args.artifact_path, h0, hmid, h1, d)
    if (artifact['raw_tensor_sha256']['h1'] != prior_record['raw_tensor_sha256']['endpoint_hessian_quadratic'] or
            artifact['raw_tensor_sha256']['d'] != prior_record['raw_tensor_sha256']['cleaned_directional_difference']):
        raise ValueError('Frozen Attempt-020 h1/d copy hash mismatch')
    manifest = {
        'format_version': 1, 'attempt_id': spec['attempt_id'], 'purpose': spec['purpose'],
        'information_policy': spec['information_policy'], 'hash_algorithm': 'sha256',
        'input_source_hashes': {str(path): digest for path, digest in hashes.items()},
        'prior_attempt': {'manifest_path': str(args.prior_manifest_path),
                          'manifest_sha256': PRIOR_MANIFEST_SHA256,
                          'artifact': prior_record, 'endpoint_consistency': endpoint},
        'base_checkpoint': spec['base'], 'final_checkpoint_files': spec['canonical_checkpoint_files'],
        'corpus': spec['frozen_corpus'], 'coordinates': spec['coordinates'],
        'displacement': displacement_audit, 'execution_semantics_audit': semantic_audit,
        'loss_normalization': spec['loss_normalization'], 'batch_partition': spec['batch_partition'],
        'scalar_path': spec['scalar_path'], 'quadrature': spec['quadrature'],
        'regressions': regressions, 'artifact': artifact, 'workload': spec['workload'],
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
