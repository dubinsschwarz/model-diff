#!/usr/bin/env python3
"""Attempt 025: privileged local-endpoint shared-g0 mechanism diagnostic."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '025_local_endpoint_shared_g0_approximation_pilot'
SOURCE_DIR = Path(__file__).parent
SOURCE_HASHES = {
    'scripts/ablation/construct_true_delta_scalar_path_pilot.py':
        '647b08b1f1083adcee715421fb532c7e3b8dd771d004befcac546ddfcfbe1157',
    'scripts/ablation/construct_true_delta_path_curvature_quadrature_pilot.py':
        'cee12e238fd41646214de1550d6b589b0925c205c54d5b3753b0753278ff7754',
    'scripts/ablation/construct_blind_centered_krylov4_pilot.py':
        'b0b52b29c608a3f276117ab7fc2569d74097b25bcef02f80485855318662bfd5',
}
for relative, digest in SOURCE_HASHES.items():
    if hashlib.sha256((PROJECT/relative).read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen Attempt 025 support source changed: '+relative)


def load_source(name, relative):
    loader = importlib.util.spec_from_file_location(name, PROJECT/relative)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


prior020 = load_source('attempt025_support020', 'scripts/ablation/construct_true_delta_scalar_path_pilot.py')
prior022 = load_source('attempt025_basis022', 'scripts/ablation/construct_blind_centered_krylov4_pilot.py')
support, execution, a, loss_helper = (
    prior020.support, prior020.execution, prior020.a, prior020.loss_helper)

ATTEMPT011_SPEC_SHA256 = 'e2752003b2d3a173ca2d1f053a1ad6c7fed17d6e6a0ee606e02765376c606627'
ATTEMPT020_SPEC_SHA256 = 'ec6e854e92bc14b65f5a9e5c30fa1ebfdeaa441e04763271364ca19ec4d96b40'
ATTEMPT021_SPEC_SHA256 = 'c788d228aaac27d7ac1efe8d1fb6ed1a2e23c9920800b1ddf1fe20a6d5393bfa'
PERTURBATIONS = (('plus_q1', 0, 0.25), ('minus_q1', 0, -0.25),
                 ('plus_q2', 1, 0.25), ('minus_q2', 1, -0.25))
SCALAR_KEYS = ('a0', 'a1', 'hDD0', 'hDD1', 'hDu0', 'hDuMid', 'hDu1',
               'local_gradient_change', 'endpoint_residual', 'trapezoid_residual',
               'simpson_residual', 'historical_rhs')

FROZEN_SPEC = copy.deepcopy(prior020.FROZEN_SPEC)
for obsolete in ('batch_partition', 'scalar_path', 'regression', 'workload', 'outputs'):
    FROZEN_SPEC.pop(obsolete, None)
FROZEN_SPEC.update({
    'format_version': 1,
    'attempt_id': ATTEMPT,
    'purpose': 'privileged_local_endpoint_shared_g0_approximation_mechanistic_pilot_not_blind_recovery',
    'information_policy': 'final_model_frozen_text_and_blind_basis_for_u;historical_delta_for_diagnostic_only',
    'early_coordinates': prior022.FROZEN_SPEC['coordinates'],
    'basis': {'attempt_id': '011_generic_hessian_krylov_k4_prefix0_13',
              'manifest_sha256': prior022.PRIOR_MANIFEST_SHA256,
              'spec_sha256': ATTEMPT011_SPEC_SHA256,
              'files': prior022.FROZEN_SPEC['basis']['files'],
              'gram_policy': prior022.FROZEN_SPEC['basis'],
              'used_directions': [1, 2], 'recomputed': False},
    'true_delta_support': {'attempt020_spec_sha256': ATTEMPT020_SPEC_SHA256,
                           'attempt021_spec_sha256': ATTEMPT021_SPEC_SHA256,
                           'delta_support_count': 196,
                           'perturbation_support_count': 98,
                           'sign': 'W1_float32_minus_loaded_W0_float32',
                           'realization': 'exact_FP64_displacement_then_one_contiguous_FP32_cast',
                           'use': 'privileged_diagnostic_only'},
    'batches': {'microbatch_indices': list(range(16)), 'sequence_count': 128,
                'microbatch_count': 16, 'microbatch_size': 8,
                'sequence_length': 128, 'prediction_tokens_per_microbatch': 1016,
                'row_rule': 'microbatch_j_rows_[8*j,8*(j+1))',
                'smoke_microbatch_indices': [0]},
    'perturbations': [{'id': identifier, 'label': label,
                       'basis_index_zero_based': index, 'scale': scale}
                       for identifier, (label, index, scale) in enumerate(PERTURBATIONS)],
    'plane': {'evaluation': 'torch.func.functional_call',
              'only_autograd_input': 'separate_float64_scalar_leaves_s_t_cast_to_FP32_for_weight_update',
              'early_functional_weight': 'FP32(W1_weight+FP32(s)*Delta_full+(FP32(t)*FP32(scale))*q)',
              'later_functional_weight': 'FP32(W1_weight+FP32(s)*Delta_full)',
              's_direction': 'full_196_coordinate_historical_Delta',
              't_direction': 'sparse_frozen_98_coordinate_early_q_times_frozen_scalar_scale',
              's_evaluation': 0.0, 't_evaluations': [0.0, 0.5, 1.0],
              'derivatives': ['dL_ds', 'd2L_ds2', 'd2L_dsdt'],
              'primitive': 'true_scalar_double_backward',
              'model_parameter_requires_grad': False, 'fallback': False,
              'loss': 'sum_causal_next_token_CE_divided_by_1016'},
    'formulas': {
        'local_gradient_change': 'a1-a0',
        'endpoint_residual': '(a1-a0)-hDu1',
        'trapezoid_residual': '(a1-a0)-(hDu0+hDu1)/2',
        'simpson_residual': '(a1-a0)-(hDu0+4*hDuMid+hDu1)/6',
        'historical_rhs': 'hDD1-hDD0',
        'primary': 'endpoint_residual_approximately_historical_rhs',
        'centered_OLS': 'historical_rhs_from_endpoint_residual_with_intercept',
        'undefined_zero_denominator': None,
        'pooled_reporting': 'descriptive_only_16_batches_times_4_related_perturbations_not_64_independent_batches'},
    'workload': {'full_microbatches': 16, 'perturbations_per_microbatch': 4,
                 'plane_points_per_perturbation': 3,
                 'scalar_second_backward_model_passes': 192,
                 'smoke_scalar_second_backward_model_passes': 12},
    'gpu_residency': {'basis_directions_at_once': 1,
                      'order': ['q1_plus', 'q1_minus', 'release_q1',
                                'empty_cuda_cache', 'q2_plus', 'q2_minus',
                                'release_q2', 'empty_cuda_cache'],
                      'scaled_basis_copies': False,
                      'full_delta_retained': True},
    'interpretation': {
        'local_information': 'Local perturbations add no exact historical information.',
        'success': 'Agreement would only validate endpoint-H approximation as a useful inductive bias.',
        'endpoint_residual': 'Endpoint residual is exactly the endpoint-quadrature error on the known local segment theta1 to theta1+u.',
        'quadrature_controls': 'Trapezoid and Simpson residuals are local-segment closure controls.',
        'quadrature_warning': 'If Simpson drives residual near zero while endpoint residual matches historical RHS, the apparent signal is local quadrature error, not newly observed history.'},
    'outputs': {'manifest_only': True, 'overwrite': False, 'timestamps': False,
                'host_metadata': False, 'gpu_metadata': False,
                'persist_vectors_gradients_hvps_or_checkpoints': False},
})
FROZEN_SPEC['sources'] = {**prior020.FROZEN_SPEC['sources'], **SOURCE_HASHES,
                          'scripts/ablation/construct_generic_hessian_krylov.py':
                              prior022.PRIOR_SOURCE_SHA256}
FROZEN_SPEC['defaults'] = {
    'base_directory': prior020.FROZEN_SPEC['defaults']['base_directory'],
    'final_directory': prior020.FROZEN_SPEC['defaults']['final_directory'],
    'corpus_spec_path': prior020.FROZEN_SPEC['defaults']['corpus_spec_path'],
    'corpus_manifest_path': prior020.FROZEN_SPEC['defaults']['corpus_manifest_path'],
    'tokens_path': prior020.FROZEN_SPEC['defaults']['tokens_path'],
    'basis_directory': prior022.FROZEN_SPEC['defaults']['basis_directory'],
    'basis_spec_path': 'experiments/attempts/011_generic_hessian_krylov_k4_prefix0_13/spec.json',
    'basis_manifest_path': prior022.FROZEN_SPEC['defaults']['basis_manifest_path'],
    'attempt020_spec_path': 'experiments/attempts/020_true_delta_scalar_path_centered_regression_pilot/spec.json',
    'attempt021_spec_path': 'experiments/attempts/021_true_delta_path_curvature_quadrature_pilot/spec.json',
    'manifest_path': 'experiments/attempts/'+ATTEMPT+'/construction-manifest.json',
}


def batch_inventory(smoke_only=False):
    count = 1 if smoke_only else 16
    return [(j, 8*j, 8*(j+1)) for j in range(count)]


def require_absent(path):
    if path.exists() or path.is_symlink():
        raise ValueError('Scientific output already exists')


def load_basis_direction(store, coordinates, basis_index, device):
    """Transfer exactly one unchanged q1 or q2 direction to the model device."""
    if basis_index not in (0, 1):
        raise ValueError('Only frozen q1 and q2 are used')
    direction = {}
    for row in coordinates:
        direction[row['name']] = store.matrix(basis_index, row['name'], row['shape']).to(
            device=device).contiguous()
    return direction


def scalar_plane_point(model, batch, names, delta, basis, scale, t_value, loss_fn=None):
    """Return full-Delta directional gradient, DD curvature, and sparse-u mixed curvature."""
    if t_value not in (0.0, 0.5, 1.0) or scale not in (0.25, -0.25):
        raise ValueError('Unfrozen local path point or perturbation scale')
    if (batch.dtype != torch.int64 or tuple(batch.shape) != (8, 128) or
            (len(names) != 196 and loss_fn is None) or
            len(names) != len(set(names)) or set(names) != set(delta) or
            not basis or not set(basis).issubset(set(names))):
        raise ValueError('Plane coordinate or batch inventory mismatch')
    if loss_fn is None and set(basis) != {
            row['name'] for row in FROZEN_SPEC['early_coordinates']}:
        raise ValueError('Perturbation must use exactly the frozen 98 early coordinates')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameters must be frozen')
    parameters = dict(model.named_parameters())
    if not set(names).issubset(parameters):
        raise ValueError('Missing selected plane parameter')
    s = torch.zeros((), device=batch.device, dtype=torch.float64, requires_grad=True)
    t = torch.tensor(t_value, device=batch.device, dtype=torch.float64, requires_grad=True)
    s32, t32 = s.to(dtype=torch.float32), t.to(dtype=torch.float32)
    scaled_t = t32*scale
    overrides = {}
    for name in names:
        endpoint, d = parameters[name], delta[name]
        if (endpoint.dtype != torch.float32 or endpoint.device != batch.device or
                d.dtype != torch.float32 or d.device != batch.device or
                d.shape != endpoint.shape or not d.is_contiguous() or
                d.requires_grad or not bool(torch.isfinite(d).all())):
            raise ValueError('Invalid FP32 full-Delta plane tensor: '+name)
        value = endpoint.detach()+s32*d
        if name in basis:
            q = basis[name]
            if (q.dtype != torch.float32 or q.device != batch.device or
                    q.shape != endpoint.shape or not q.is_contiguous() or
                    q.requires_grad or not bool(torch.isfinite(q).all())):
                raise ValueError('Invalid sparse FP32 basis tensor: '+name)
            value = value+scaled_t*q
        overrides[name] = value
        if overrides[name].dtype != torch.float32:
            raise ValueError('Functional selected weight left FP32')
    with torch.enable_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
        logits = torch.func.functional_call(model, overrides, (),
            {'input_ids': batch, 'use_cache': False}, tie_weights=True, strict=False).logits
        loss = (loss_fn(logits, batch) if loss_fn is not None else
                loss_helper.causal_token_loss_sum(logits, batch, torch)/1016)
        if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
            raise ValueError('Invalid scalar mean loss')
        first = torch.autograd.grad(loss, s, create_graph=True)[0]
        hDD, hDu = torch.autograd.grad(first, (s, t))
        values = (float(first.detach().to('cpu', dtype=torch.float64)),
                  float(hDD.detach().to('cpu', dtype=torch.float64)),
                  float(hDu.detach().to('cpu', dtype=torch.float64)))
        del logits, loss, first, hDD, hDu, overrides
    if not all(math.isfinite(v) for v in values) or any(
            parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Nonfinite scalar derivatives or stored parameter gradient')
    return values


def compose_record(batch_index, perturbation_id, at_zero, at_mid, at_one):
    a0, hDD0, hDu0 = at_zero
    _, _, hDuMid = at_mid
    a1, hDD1, hDu1 = at_one
    values = (a0, a1, hDD0, hDD1, hDu0, hDuMid, hDu1)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite plane scalar')
    change = a1-a0
    record = {'microbatch_index': batch_index, 'perturbation_id': perturbation_id,
              'a0': a0, 'a1': a1, 'hDD0': hDD0, 'hDD1': hDD1,
              'hDu0': hDu0, 'hDuMid': hDuMid, 'hDu1': hDu1,
              'local_gradient_change': change,
              'endpoint_residual': change-hDu1,
              'trapezoid_residual': change-(hDu0+hDu1)/2,
              'simpson_residual': change-(hDu0+4*hDuMid+hDu1)/6,
              'historical_rhs': hDD1-hDD0}
    if not all(math.isfinite(record[key]) for key in SCALAR_KEYS):
        raise ValueError('Nonfinite residual or historical RHS')
    return record


def scalar_diagnostics(records):
    if not records or any(not all(math.isfinite(row[key]) for key in SCALAR_KEYS)
                          for row in records):
        raise ValueError('Invalid scalar diagnostic records')
    n = len(records)
    x = [row['endpoint_residual'] for row in records]
    y = [row['historical_rhs'] for row in records]
    xm, ym = math.fsum(x)/n, math.fsum(y)/n
    xc, yc = [v-xm for v in x], [v-ym for v in y]
    xx, yy = math.fsum(v*v for v in xc), math.fsum(v*v for v in yc)
    xy = math.fsum(v*w for v, w in zip(xc, yc))
    raw_xx = math.fsum(v*v for v in x)
    raw_yy = math.fsum(v*v for v in y)
    raw_xy = math.fsum(v*w for v, w in zip(x, y))
    beta = xy/xx if xx else None
    corr = xy/math.sqrt(xx*yy) if xx and yy else None
    cosine = raw_xy/math.sqrt(raw_xx*raw_yy) if raw_xx and raw_yy else None
    optimal = raw_xy/raw_xx if raw_xx else None
    rms = lambda key: math.sqrt(math.fsum(row[key]**2 for row in records)/n)
    endpoint = rms('endpoint_residual')
    trapezoid = rms('trapezoid_residual')
    simpson = rms('simpson_residual')
    return {'count': n, 'correlation': corr, 'centered_ols_beta': beta,
            'beta_minus_ideal_one': beta-1 if beta is not None else None,
            'centered_r_squared': corr*corr if corr is not None else None,
            'cosine': cosine, 'centered_S_xx': xx, 'centered_S_yy': yy,
            'rms_endpoint_residual': endpoint, 'rms_historical_rhs': rms('historical_rhs'),
            'rms_endpoint_minus_historical_rhs': math.sqrt(
                math.fsum((v-w)**2 for v, w in zip(x, y))/n),
            'optimal_scalar_historical_rhs_from_endpoint_residual': optimal,
            'best_rescaled_relative_residual': math.sqrt(
                math.fsum((optimal*v-w)**2 for v, w in zip(x, y))/raw_yy)
                if optimal is not None and raw_yy else None,
            'rms_trapezoid_residual': trapezoid, 'rms_simpson_residual': simpson,
            'trapezoid_to_endpoint_rms_ratio': trapezoid/endpoint if endpoint else None,
            'simpson_to_endpoint_rms_ratio': simpson/endpoint if endpoint else None}


def summaries(records):
    expected = {(j, identifier) for j in range(16) for identifier in range(4)}
    actual = {(row['microbatch_index'], row['perturbation_id']) for row in records}
    if len(records) != 64 or actual != expected:
        raise ValueError('Attempt 025 scalar record inventory mismatch')
    per = {label: scalar_diagnostics([row for row in records if row['perturbation_id'] == identifier])
           for identifier, (label, _, _) in enumerate(PERTURBATIONS)}
    pooled = scalar_diagnostics(records)
    pooled.update({'descriptive_only': True, 'independent_batch_samples': False,
                   'sample_structure': '16 batches x 4 related perturbations'})
    return {'per_perturbation': per, 'pooled': pooled}


def ordered_records(records):
    return sorted(records, key=lambda row: (row['microbatch_index'], row['perturbation_id']))


def validate_inputs(args):
    for key, frozen in FROZEN_SPEC['defaults'].items():
        expected = Path(frozen)
        if not expected.is_absolute():
            expected = PROJECT/expected
        if getattr(args, key).resolve() != expected.resolve():
            raise ValueError('Attempt025 path differs from frozen '+key)
    if args.spec_path.resolve() != (PROJECT/'experiments/attempts'/ATTEMPT/'spec.json').resolve():
        raise ValueError('Attempt025 spec path differs from frozen path')
    require_absent(args.manifest_path)
    spec = a.load_json_object(args.spec_path, 'Attempt025 spec')
    if spec != FROZEN_SPEC:
        raise ValueError('Attempt025 frozen spec mismatch')
    expected_specs = ((args.basis_spec_path, ATTEMPT011_SPEC_SHA256),
                      (args.basis_manifest_path, prior022.PRIOR_MANIFEST_SHA256),
                      (args.attempt020_spec_path, ATTEMPT020_SPEC_SHA256),
                      (args.attempt021_spec_path, ATTEMPT021_SPEC_SHA256))
    paths = [args.spec_path, args.corpus_spec_path, args.corpus_manifest_path,
             args.tokens_path, *(path for path, _ in expected_specs),
             Path(__file__).resolve(), *(PROJECT/name for name in spec['sources'])]
    hashes = {path: a.sha256_file(path) for path in paths}
    for path, digest in expected_specs:
        if hashes[path] != digest:
            raise ValueError('Frozen Attempt support spec/manifest hash mismatch')
    for name, digest in spec['sources'].items():
        if hashes[PROJECT/name] != digest:
            raise ValueError('Frozen Attempt support source hash mismatch')
    if a.load_json_object(args.attempt020_spec_path, 'Attempt020 spec') != prior020.FROZEN_SPEC:
        raise ValueError('Attempt020 support spec mismatch')
    if a.load_json_object(args.basis_manifest_path, 'Attempt011 manifest') != prior022.prior_manifest:
        raise ValueError('Attempt011 basis manifest mismatch')
    if (a.checkpoint_file_records(args.base_directory) != spec['base']['files'] or
            a.checkpoint_file_records(args.final_directory) != spec['canonical_checkpoint_files']):
        raise ValueError('Historical base or final checkpoint inventory mismatch')
    for field, path in (('corpus_spec_sha256', args.corpus_spec_path),
                        ('corpus_manifest_sha256', args.corpus_manifest_path)):
        if hashes[path] != spec['frozen_corpus'][field]:
            raise ValueError('Frozen Attempt005 corpus provenance mismatch')
    corpus = loss_helper.verify_corpus_manifest(args.corpus_manifest_path,
        args.corpus_spec_path, args.tokens_path, args.final_directory,
        loss_helper.load_corpus_spec(args.corpus_spec_path))
    if any(corpus[key] != spec['frozen_corpus'][key]
           for key in ('serialized_sha256', 'raw_tensor_sha256')):
        raise ValueError('Frozen Attempt005 token artifact mismatch')
    tokens = loss_helper.load_corpus_tokens(args.tokens_path,
        corpus['raw_tensor_sha256'], 4096, 128, torch)
    prior022.verify_basis_files(args.basis_directory, spec['basis']['files'])
    with prior022.BasisStore(args.basis_directory, spec['early_coordinates']) as store:
        gram = prior022.basis_gram(store, spec['early_coordinates'], spec['basis']['gram_policy'])
    if (args.manifest_path.resolve().is_relative_to(args.base_directory.resolve()) or
            args.manifest_path.resolve().is_relative_to(args.final_directory.resolve()) or
            args.manifest_path.resolve().is_relative_to(args.basis_directory.resolve()) or
            any(args.manifest_path.resolve() == path.resolve() or
                path.resolve().is_relative_to(args.manifest_path.resolve()) for path in paths)):
        raise ValueError('Scientific output overlaps frozen inputs')
    def recheck():
        if (a.checkpoint_file_records(args.base_directory) != spec['base']['files'] or
                a.checkpoint_file_records(args.final_directory) != spec['canonical_checkpoint_files']):
            raise ValueError('Checkpoint mutated during diagnostic')
        prior022.verify_basis_files(args.basis_directory, spec['basis']['files'])
        for path, digest in hashes.items():
            if a.sha256_file(path) != digest:
                raise ValueError('Frozen input mutated during diagnostic')
    return spec, tokens, hashes, gram, recheck


def prepare_context(args):
    spec, tokens, hashes, gram, recheck = validate_inputs(args)
    base = a.load_local_model(args.base_directory, args.device, torch)
    base_eligible = support.discover(base, spec)
    prior020.freeze_model(base)
    base_before = a.model_state_hashes(base, torch)
    base_snapshot = support.snapshot(base, spec)
    support.verify_state(base, base_before, base_snapshot['execution'], spec)
    del base_eligible, base
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    final = a.load_local_model(args.final_directory, args.device, torch)
    eligible = support.discover(final, spec)
    prior020.freeze_model(final)
    final_before = a.model_state_hashes(final, torch)
    final_execution = execution.execution_snapshot(final, spec)
    semantic_audit = support.execution_audit(base_snapshot['execution'], final_execution)
    with execution.BaseTensorReader(args.base_directory, spec['base']['files']) as reader:
        full_delta, delta_audit = support.displacement(
            final, eligible, base_snapshot['parameters'], reader.get)
    early = [{'name': name+'.weight', 'shape': list(module.weight.shape)}
             for name, module in eligible if support.block_number(name)<14]
    if early != spec['early_coordinates'] or len(early) != 98:
        raise ValueError('Exact prefix0-13 coordinate inventory mismatch')
    names = [name+'.weight' for name, _ in eligible]
    if (len(names) != 196 or names != list(full_delta) or
            set(names) != {row['name'] for row in spec['coordinates']}):
        raise ValueError('Exact full-Delta coordinate inventory mismatch')
    delta = full_delta
    del base_snapshot
    return {'spec': spec, 'tokens': tokens, 'hashes': hashes, 'gram': gram,
            'recheck': recheck, 'model': final, 'names': names,
            'delta': delta,
            'delta_audit': delta_audit, 'semantic_audit': semantic_audit,
            'model_before': final_before, 'execution_before': final_execution}


def diagnose(args):
    context = prepare_context(args)
    spec, tokens, model = context['spec'], context['tokens'], context['model']
    names, delta = context['names'], context['delta']
    device = next(model.parameters()).device
    records = []
    with prior022.BasisStore(args.basis_directory, spec['early_coordinates']) as store:
        for basis_index in (0, 1):
            basis = load_basis_direction(store, spec['early_coordinates'], basis_index, device)
            for identifier, (_, frozen_index, scale) in enumerate(PERTURBATIONS):
                if frozen_index != basis_index:
                    continue
                for j, start, end in batch_inventory(args.smoke_only):
                    batch = tokens[start:end].to(device)
                    zero = scalar_plane_point(model, batch, names, delta, basis, scale, 0.0)
                    mid = scalar_plane_point(model, batch, names, delta, basis, scale, 0.5)
                    one = scalar_plane_point(model, batch, names, delta, basis, scale, 1.0)
                    records.append(compose_record(j, identifier, zero, mid, one))
                    del batch
                if not args.smoke_only:
                    print(f'Local-endpoint perturbation {identifier+1}/4 complete', flush=True)
            del basis
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    records = ordered_records(records)
    support.verify_state(model, context['model_before'], context['execution_before'], spec)
    del model, context['model'], context['delta']
    context['recheck']()
    if args.smoke_only:
        print(json.dumps({'smoke_only': True, 'microbatch_index': 0,
                          'perturbations': [{key: row[key] for key in
                              ('perturbation_id', 'endpoint_residual', 'historical_rhs',
                               'trapezoid_residual', 'simpson_residual')}
                              for row in records],
                          'basis_gram_max_off_diagonal': context['gram']['max_off_diagonal']},
                         sort_keys=True, allow_nan=False))
        return
    report = summaries(records)
    manifest = {'format_version': 1, 'attempt_id': ATTEMPT,
                'purpose': spec['purpose'], 'information_policy': spec['information_policy'],
                'hash_algorithm': 'sha256',
                'input_source_hashes': {str(path): digest for path, digest in context['hashes'].items()},
                'base_checkpoint': spec['base'],
                'final_checkpoint_files': spec['canonical_checkpoint_files'],
                'corpus': spec['frozen_corpus'], 'basis': spec['basis'],
                'basis_gram_audit': context['gram'],
                'true_delta_support': spec['true_delta_support'],
                'displacement_audit': {key: value for key, value in context['delta_audit'].items()
                                       if key != 'all_parameter_records'},
                'execution_semantics_audit': context['semantic_audit'],
                'batches': spec['batches'], 'perturbations': spec['perturbations'],
                'plane': spec['plane'], 'formulas': spec['formulas'],
                'scalar_records': records, 'diagnostics': report,
                'workload': spec['workload'], 'interpretation': spec['interpretation'],
                'blind_recovery_claim': False, 'candidate_selection': False,
                'publication_policy': 'manifest_only_no_overwrite_no_resume'}
    require_absent(args.manifest_path)
    context['recheck']()
    a.write_manifest(args.manifest_path, manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path', type=Path,
                        default=PROJECT/'experiments/attempts'/ATTEMPT/'spec.json')
    for key, value in FROZEN_SPEC['defaults'].items():
        path = Path(value)
        parser.add_argument('--'+key.replace('_', '-'), type=Path,
                            default=path if path.is_absolute() else PROJECT/path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-only', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    diagnose(parse_args())
