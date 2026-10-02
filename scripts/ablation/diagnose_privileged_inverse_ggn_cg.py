#!/usr/bin/env python3
"""Attempt106: privileged fixed-budget inverse of final endpoint categorical GGN."""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '106_privileged_inverse_ggn_cg_pilot'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = 'eeef0f7b931be857462e286258ccde7705b916f6a731bdff213bbfd9df0d416e'
SOURCE105 = Path(__file__).with_name('diagnose_privileged_endpoint_ggn_forward.py')
SOURCE105_SHA256 = '3c95e670ad441ad4f086f56c2b214842bf8d09924630b5afa73fdd60719bae18'
if hashlib.sha256(SOURCE105.read_bytes()).hexdigest() != SOURCE105_SHA256:
    raise ValueError('Frozen Attempt105 source changed')
loader = importlib.util.spec_from_file_location('attempt106_frozen_attempt105', SOURCE105)
a105 = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a105)
a = a105.a
parent = a105.a104.parent


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt106 spec path changed')
    parent.require_hash(path, SPEC_SHA256)
    spec = a.load_json_object(path, 'Attempt106 spec')
    old = a105.load_spec()
    reference = parent.load_spec()
    for key in ('attempt104', 'attempt016', 'base', 'final_checkpoint_files',
                'model', 'readout', 'eligible_tensors', 'corpus', 'pilot',
                'displacement', 'hybrid_gradient', 'ggn'):
        if spec.get(key) != old[key]:
            raise ValueError('Attempt106 differs from frozen Attempt105: '+key)
    for key in ('probe', 'probe_rows', 'jvp', 'metric'):
        if spec.get(key) != reference[key]:
            raise ValueError('Attempt106 matched probe/JVP differs from Attempt103: '+key)
    paths = dict(spec['paths'])
    paths.pop('result_path', None)
    paths.pop('probe_path', None)
    if (spec.get('attempt_id') != ATTEMPT or paths !=
            {key: value for key, value in old['paths'].items() if key != 'result_path'} or
            spec['paths']['probe_path'] != reference['paths']['probe_path'] or
            spec['solver']['iterations'] != 10 or
            spec['solver']['damping_ratio'] != 0.01 or
            spec['solver']['residual_early_stop'] is not False or
            spec['target']['raw_float32_sha256'] !=
                '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f' or
            spec['ggn_numerics_106']['operator'] != 'exact_endpoint_categorical_JtFJ' or
            spec['ggn_numerics_106']['implementation'] !=
                'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast' or
            spec['ggn_numerics_106']['attempt105_numerical_implementation_reused'] is not False):
        raise ValueError('Attempt106 frozen scientific inventory mismatch')
    for value in spec['paths'].values():
        parent.resolve(value)
    return spec


def validate_inputs(spec):
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt106 result already exists')
    link = spec['attempt105']
    for key in ('spec', 'source', 'result'):
        parent.require_hash(parent.resolve(link[key+'_path']), link[key+'_sha256'])
    if (link['source_sha256'] != SOURCE105_SHA256 or
            link['spec_sha256'] != a105.SPEC_SHA256 or
            link['result_sha256'] !=
                '82fe5ddac2960e2be5e96daa94f8b905aec6f528b64d21396c503a7ff8a49618'):
        raise ValueError('Attempt105 frozen linkage mismatch')
    result105 = a.load_json_object(parent.resolve(link['result_path']), 'Attempt105 result')
    if (result105.get('attempt_id') != a105.ATTEMPT or
            result105.get('provenance', {}).get('spec_sha256') != link['spec_sha256'] or
            result105['provenance'].get('source_sha256') != link['source_sha256'] or
            result105['provenance'].get('base_checkpoint_files') != spec['base']['files'] or
            result105['provenance'].get('final_checkpoint_files') != spec['final_checkpoint_files'] or
            result105['provenance'].get('fineweb_tokens_sha256') != spec['corpus']['serialized_sha256'] or
            result105.get('parameter_geometry', {}).get('primary_global_signed_cosine') !=
                link['required_global_cosine'] or
            result105['parameter_geometry'].get('hybrid_gradient_norm') !=
                link['expected_hybrid_gradient_norm'] or
            result105.get('interpretation', {}).get('category') != link['required_category'] or
            result105.get('ggn_operator') != 'exact_endpoint_categorical_JtFJ' or
            result105.get('inverse_solve') is not False or
            result105.get('empirical_fisher') is not False or
            result105.get('oracle_adl_access') is not False or
            result105.get('adapter_access') is not False):
        raise ValueError('Attempt105 result/provenance/mechanism mismatch')
    old105 = a105.load_spec(parent.resolve(link['spec_path']))
    proxy = copy.deepcopy(old105)
    proxy['paths']['result_path'] = spec['paths']['result_path']
    spec016, inventories = a105.validate_inputs(proxy)
    parent.require_hash(parent.resolve(spec['paths']['probe_path']),
                        spec['probe']['serialized_sha256'])
    return spec016, inventories


def load_frozen_inputs(spec):
    construction512, probe = parent.load_frozen_tokens(spec)
    if (construction512.shape != (512, 128) or probe.shape != (1024, 128) or
            construction512.dtype != torch.int64 or probe.dtype != torch.int64 or
            not construction512.is_contiguous() or not probe.is_contiguous()):
        raise ValueError('Attempt106 corpus/probe inventory mismatch')
    construction = construction512[:64].contiguous()
    del construction512
    return construction, probe


def _names(eligible):
    names = [name+'.weight' for name, _ in eligible]
    if len(names) != len(set(names)) or not names:
        raise ValueError('Invalid selected coordinate inventory')
    return names


def matrixwise_dot(left, right):
    if not left or set(left) != set(right):
        raise ValueError('Matrixwise dot coordinate mismatch')
    parts = []
    for name in sorted(left):
        x, y = left[name], right[name]
        if (x.dtype != torch.float32 or y.dtype != torch.float32 or
                x.shape != y.shape or x.device != y.device or
                not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all())):
            raise ValueError('Nonfinite/malformed matrixwise dot input')
        parts.append(float(torch.sum(x*y, dtype=torch.float64)))
    return math.fsum(parts)


def matrixwise_grad_dot(eligible, vector):
    if set(_names(eligible)) != set(vector):
        raise ValueError('GGN gradient/vector coordinate mismatch')
    parts = []
    for name, module in eligible:
        gradient, direction = module.weight.grad, vector[name+'.weight']
        if (gradient is None or gradient.dtype != torch.float32 or
                direction.dtype != torch.float32 or gradient.shape != direction.shape or
                gradient.device != direction.device or
                not bool(torch.isfinite(gradient).all())):
            raise ValueError('Missing/nonfinite GGN action coordinate')
        parts.append(float(torch.sum(gradient*direction, dtype=torch.float64)))
    return math.fsum(parts)


def _fisher_error(reason, diagnostics):
    raise ValueError(f'Attempt106 categorical Fisher {reason}; diagnostics='
                     f'{json.dumps(diagnostics, sort_keys=True, allow_nan=True, default=str)}')


def audit_stable_fisher_numerics(p64, s64, mean64, u64, u, spec):
    policy = spec['ggn_numerics_106']
    diagnostics = {'p64_shape': list(p64.shape), 'u64_shape': list(u64.shape),
                   'u_shape': list(u.shape), 'p64_finite': bool(torch.isfinite(p64).all()),
                   's64_finite': bool(torch.isfinite(s64).all()),
                   'mean64_finite': bool(torch.isfinite(mean64).all()),
                   'u64_finite': bool(torch.isfinite(u64).all()),
                   'u_finite': bool(torch.isfinite(u).all())}
    diagnostics.update({
        'max_abs_tangent': float(s64.abs().max()) if diagnostics['s64_finite'] else None,
        'max_abs_weighted_mean': float(mean64.abs().max()) if diagnostics['mean64_finite'] else None,
        'max_abs_u64_entry': float(u64.abs().max()) if diagnostics['u64_finite'] else None,
        'max_abs_u_entry': float(u.abs().max()) if diagnostics['u_finite'] else None})
    if (p64.ndim != 3 or p64.shape != s64.shape or p64.shape != u64.shape or
            p64.shape != u.shape or mean64.shape != p64.shape[:-1]+(1,) or
            p64.dtype != torch.float64 or s64.dtype != torch.float64 or
            mean64.dtype != torch.float64 or u64.dtype != torch.float64 or
            u.dtype != torch.float32 or not all(diagnostics[key] for key in
                ('p64_finite', 's64_finite', 'mean64_finite', 'u64_finite', 'u_finite'))):
        _fisher_error('nonfinite/malformed values', diagnostics)
    p_sum_error = (torch.sum(p64, dim=-1, dtype=torch.float64)-1).abs()
    residual64 = torch.sum(u64, dim=-1, dtype=torch.float64).abs()
    scale64 = torch.sum(u64.abs(), dim=-1, dtype=torch.float64)
    residual32 = torch.sum(u, dim=-1, dtype=torch.float64).abs()
    scale32 = torch.sum(u.abs(), dim=-1, dtype=torch.float64)
    relative64 = torch.where(scale64 > 0, residual64/scale64.clamp_min(
        torch.finfo(torch.float64).tiny), torch.zeros_like(scale64))
    relative32 = torch.where(scale32 > 0, residual32/scale32.clamp_min(
        torch.finfo(torch.float64).tiny), torch.zeros_like(scale32))
    max_abs_tangent = s64.abs().amax(dim=-1)
    diagnostics.update({
        'max_abs_tangent': float(max_abs_tangent.max()),
        'max_abs_unweighted_vocab_mean_tangent': float(s64.mean(dim=-1).abs().max()),
        'max_abs_probability_weighted_mean_tangent': float(mean64.abs().max()),
        'max_softmax_sum_error': float(p_sum_error.max()),
        'max_abs_sum_u64': float(residual64.max()),
        'max_relative_abs_sum_u64': float(relative64.max()),
        'fp32_max_abs_sum': float(residual32.max()),
        'fp32_max_relative_abs_sum': float(relative32.max()),
        'prediction_context_count': residual64.numel()})
    allowed64 = (policy['fp64_fisher_conservation_fail_relative_tolerance']*scale64+
                 policy['fp64_fisher_conservation_fail_gauge_roundoff_multiplier']*
                 max_abs_tangent/spec['pilot']['prediction_contexts']+
                 policy['fp64_fisher_conservation_fail_absolute_tolerance'])
    allowed32 = (policy['fp32_cast_conservation_fail_relative_tolerance']*scale32+
                 policy['fp32_cast_conservation_fail_absolute_tolerance'])
    diagnostics['max_fp64_allowed_abs_sum'] = float(allowed64.max())
    diagnostics['max_fp32_allowed_abs_sum'] = float(allowed32.max())
    if bool((p_sum_error > policy['fp64_softmax_normalization_fail_abs_tolerance']).any()):
        _fisher_error('FP64 softmax normalization failure', diagnostics)
    if bool((residual64 > allowed64).any()):
        _fisher_error('FP64 sum-to-zero failure', diagnostics)
    if bool((residual32 > allowed32).any()):
        _fisher_error('FP32 cast sum-to-zero failure', diagnostics)
    return diagnostics


def stable_categorical_fisher_vector(logits, tangent, spec):
    input_audit = {'logits_shape': list(logits.shape), 'tangent_shape': list(tangent.shape),
                   'logits_dtype': str(logits.dtype), 'tangent_dtype': str(tangent.dtype),
                   'logits_finite': bool(torch.isfinite(logits).all()),
                   'tangent_finite': bool(torch.isfinite(tangent).all())}
    input_audit['max_abs_logits'] = (float(logits.abs().max())
                                     if input_audit['logits_finite'] else None)
    input_audit['max_abs_tangent'] = (float(tangent.abs().max())
                                      if input_audit['tangent_finite'] else None)
    if (logits.ndim != 3 or logits.shape != tangent.shape or
            logits.dtype != torch.float32 or tangent.dtype != torch.float32 or
            not input_audit['logits_finite'] or not input_audit['tangent_finite']):
        _fisher_error('nonfinite/malformed logits or tangent', input_audit)
    p64 = torch.softmax(logits.to(torch.float64), dim=-1)
    s64 = tangent.to(torch.float64)
    mean64 = torch.sum(p64*s64, dim=-1, keepdim=True, dtype=torch.float64)
    u64 = p64*(s64-mean64)
    u64 /= spec['pilot']['prediction_contexts']
    u = u64.to(torch.float32)
    audit = audit_stable_fisher_numerics(p64, s64, mean64, u64, u, spec)
    return u.detach(), audit


def summarize_stable_fisher_audits(records):
    if not records:
        raise ValueError('Empty Attempt106 Fisher audit')
    fields = ('max_abs_tangent', 'max_abs_unweighted_vocab_mean_tangent',
              'max_abs_probability_weighted_mean_tangent', 'max_softmax_sum_error',
              'max_abs_sum_u64', 'max_relative_abs_sum_u64', 'fp32_max_abs_sum',
              'fp32_max_relative_abs_sum', 'max_fp64_allowed_abs_sum',
              'max_fp32_allowed_abs_sum')
    return {'per_batch': records,
            'aggregate': {**{field: max(row[field] for row in records) for field in fields},
                          'prediction_context_count': sum(
                              row['prediction_context_count'] for row in records)}}


def ggn_action_stage(final, tokens, eligible, direction, spec, *, progress=None):
    if (tokens.shape != (64, 128) or tokens.dtype != torch.int64 or
            not tokens.is_contiguous() or len(eligible) != 98 or len(direction) != 98 or
            set(direction) != set(_names(eligible))):
        raise ValueError('Attempt106 fixed GGN inventory/support mismatch')
    selected = {id(module.weight) for _, module in eligible}
    final.zero_grad(set_to_none=True)
    timings = {'logits_jvp_seconds': 0.0, 'fisher_vector_seconds': 0.0,
               'vjp_backward_seconds': 0.0}
    first_audit = None
    records = []
    started = time.perf_counter()
    device = eligible[0][1].weight.device
    for index in range(64):
        batch = tokens[index:index+1].to(device)
        stage = time.perf_counter()
        primal, tangent = a105.strict_logits_jvp(final, batch, direction)
        timings['logits_jvp_seconds'] += time.perf_counter()-stage
        primal_audit = None
        if index == 0:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                ordinary = final(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            primal_audit = a105.audit_logits_primal(primal, ordinary, spec)
            del ordinary
        stage = time.perf_counter()
        fisher, conservation = stable_categorical_fisher_vector(primal, tangent, spec)
        records.append({'microbatch_index': index, **conservation})
        timings['fisher_vector_seconds'] += time.perf_counter()-stage
        stage = time.perf_counter()
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            logits = final(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            if logits.shape != fisher.shape or not bool(torch.isfinite(logits).all()):
                raise ValueError('Attempt106 GGN VJP final logits mismatch')
            (logits*fisher).sum().backward()
        timings['vjp_backward_seconds'] += time.perf_counter()-stage
        if any(parameter.grad is not None for parameter in final.parameters()
               if id(parameter) not in selected):
            raise ValueError('Noneligible final parameter received GGN VJP gradient')
        if index == 0:
            if any(module.weight.grad is None or not bool(torch.isfinite(module.weight.grad).all())
                   for _, module in eligible):
                raise ValueError('Missing/nonfinite first-batch selected GGN VJP gradient')
            first_audit = {'primal': primal_audit, 'fisher': conservation,
                           'tangent_shape': list(tangent.shape),
                           'tangent_dtype': str(tangent.dtype),
                           'selected_vjp_matrix_count': len(eligible)}
        if progress is not None:
            progress(index+1, 64, time.perf_counter()-started)
        del batch, primal, tangent, fisher, logits
    return first_audit, timings, summarize_stable_fisher_audits(records)


def fixed_budget_cg(rhs_cpu, eligible, apply_operator, *, iterations=10,
                    damping_ratio=0.01, report=None):
    """Apply exactly `iterations` CG steps; each operator call writes selected .grad."""
    names = _names(eligible)
    if (set(rhs_cpu) != set(names) or iterations <= 0 or
            not math.isfinite(damping_ratio) or damping_ratio <= 0):
        raise ValueError('Invalid frozen CG support or settings')
    r = {}
    for name, module in eligible:
        key = name+'.weight'
        value = rhs_cpu[key]
        if (value.dtype != torch.float32 or value.device.type != 'cpu' or
                value.shape != module.weight.shape or not value.is_contiguous() or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Malformed/nonfinite recomputed hybrid RHS')
        r[key] = value.to(module.weight.device, dtype=torch.float32, copy=True).contiguous()
    rhs_cpu.clear()
    p = {name: value.clone() for name, value in r.items()}
    x = {name: torch.zeros_like(value) for name, value in r.items()}
    rr = matrixwise_dot(r, r)
    if rr <= 0 or not math.isfinite(rr):
        raise ValueError('Zero/nonfinite hybrid RHS norm')
    b_norm = math.sqrt(rr)
    rho_b = damping = None
    rows, operator_audits, operator_seconds = [], [], []
    started = time.perf_counter()
    for iteration in range(1, iterations+1):
        operator_started = time.perf_counter()
        audit = apply_operator(p, iteration)
        elapsed_operator = time.perf_counter()-operator_started
        operator_seconds.append(elapsed_operator)
        operator_audits.append(audit)
        p_gp = matrixwise_grad_dot(eligible, p)
        pp = matrixwise_dot(p, p)
        if iteration == 1:
            rho_b = p_gp/rr
            if not math.isfinite(rho_b) or rho_b <= 0:
                raise ValueError('Nonpositive/nonfinite RHS Rayleigh quotient')
            damping = damping_ratio*rho_b
        p_ap = p_gp+damping*pp
        if not math.isfinite(p_ap) or p_ap <= 0:
            raise ValueError('Nonpositive/nonfinite CG p^T A p')
        alpha = rr/p_ap
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Nonpositive/nonfinite CG alpha')
        for name, module in eligible:
            key = name+'.weight'
            x[key].add_(p[key], alpha=alpha)
            r[key].add_(module.weight.grad, alpha=-alpha)
            r[key].add_(p[key], alpha=-alpha*damping)
        rr_new = matrixwise_dot(r, r)
        if not math.isfinite(rr_new) or rr_new < 0:
            raise ValueError('Nonfinite/negative CG residual norm')
        beta = rr_new/rr
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('Nonfinite/negative CG beta')
        for name in names:
            p[name].mul_(beta).add_(r[name])
            if not bool(torch.isfinite(x[name]).all()) or not bool(torch.isfinite(p[name]).all()):
                raise ValueError('Nonfinite CG vector')
        rr = rr_new
        elapsed = time.perf_counter()-started
        row = {'iteration': iteration, 'relative_residual': math.sqrt(rr)/b_norm,
               'r_norm': math.sqrt(rr), 'pAp': p_ap, 'alpha': alpha, 'beta': beta,
               'operator_seconds': elapsed_operator, 'cumulative_elapsed_seconds': elapsed,
               'operator_eta_seconds': math.fsum(operator_seconds)/len(operator_seconds)*
                   (iterations-iteration)}
        rows.append(row)
        if report is not None:
            report(row)
    del r, p
    return x, {'method': 'fixed_10_step_damped_cg', 'iterations': rows,
               'operator_applications': len(operator_audits),
               'operator_audits': operator_audits,
               'rhs_norm': b_norm, 'rho_b': rho_b, 'damping_ratio': damping_ratio,
               'lambda': damping, 'relative_linear_residual': rows[-1]['relative_residual'],
               'residual_source': 'CG_recurrence_no_extra_operator_application',
               'candidate': f'x_{iterations}', 'no_early_stop': True,
               'elapsed_seconds': time.perf_counter()-started}


def parameter_diagnostics(x, delta):
    if set(x) != set(delta) or len(x) != 98:
        raise ValueError('Inverse candidate/true Delta coordinate mismatch')
    buckets = {'global': [], 'attention': [], 'mlp': []}
    buckets.update({f'block_{i}': [] for i in range(14)})
    for name in sorted(x):
        block, family = a105._coordinate_family(name)
        row = {'xx': matrixwise_dot({name: x[name]}, {name: x[name]}),
               'dd': matrixwise_dot({name: delta[name]}, {name: delta[name]}),
               'xd': matrixwise_dot({name: x[name]}, {name: delta[name]})}
        for group in ('global', f'block_{block}', family):
            buckets[group].append(row)
    def summarize(records):
        xx = math.fsum(row['xx'] for row in records)
        dd = math.fsum(row['dd'] for row in records)
        xd = math.fsum(row['xd'] for row in records)
        return {'matrix_count': len(records), 'candidate_norm': math.sqrt(xx),
                'true_delta_norm': math.sqrt(dd),
                'cosine': xd/math.sqrt(xx*dd) if xx > 0 and dd > 0 else None,
                'candidate_to_delta_norm_ratio': math.sqrt(xx/dd) if dd > 0 else None}
    global_row = summarize(buckets.pop('global'))
    if global_row['cosine'] is None:
        raise ValueError('Zero inverse candidate or true displacement norm')
    return {'global': global_row,
            'blocks': {str(i): summarize(buckets[f'block_{i}']) for i in range(14)},
            'attention': summarize(buckets['attention']), 'mlp': summarize(buckets['mlp'])}


def interpretation(primary, spec):
    gain = primary-spec['baselines']['attempt104_primary']
    rule = spec['interpretation']
    if (primary >= rule['strong_inverse_ggn_support']['primary_min'] and
            gain >= rule['strong_inverse_ggn_support']['gain_vs_attempt104_min']):
        return 'strong_inverse_ggn_support'
    if (primary >= rule['moderate_inverse_ggn_support']['primary_min'] and
            gain >= rule['moderate_inverse_ggn_support']['gain_vs_attempt104_min']):
        return 'moderate_inverse_ggn_support'
    return 'weak_or_inconclusive_inverse_ggn_support'


def verified_matched_target(final_mean, base_mean, spec):
    target = parent.matched_difference(final_mean, base_mean)
    digest = a.sha256_raw_float32_tensor(target, torch)
    if digest != spec['target']['raw_float32_sha256']:
        raise ValueError('Matched historical target differs from Attempt103/104')
    return target, digest


def run(*, model_loader=None, tokenizer_loader=None):
    spec = load_spec()
    spec016, inventories = validate_inputs(spec)
    construction, probe = load_frozen_inputs(spec)
    base, final, loads = parent.load_model_pair(
        spec, model_loader=model_loader, tokenizer_loader=tokenizer_loader)
    base_before = a.model_state_hashes(base, torch)
    final_before = a.model_state_hashes(final, torch)
    displacement_started = time.perf_counter()
    delta, displacement_audit, execution_audit, eligible = a105.prepare_displacement(
        base, final, spec016, inventories['base_files'])
    displacement_seconds = time.perf_counter()-displacement_started
    rhs_cpu, rhs_info = a105.hybrid_gradient_64(
        base, final, construction, eligible, spec,
        progress=parent.progress_printer('hybrid RHS'))
    rhs_norm = math.sqrt(matrixwise_dot(rhs_cpu, rhs_cpu))
    link = spec['attempt105']
    if (not math.isfinite(rhs_norm) or not math.isclose(
            rhs_norm, link['expected_hybrid_gradient_norm'],
            rel_tol=link['rhs_norm_rtol'], abs_tol=link['rhs_norm_atol'])):
        raise ValueError('Recomputed hybrid RHS norm differs from Attempt105')
    base_mean = parent.base_probe_mean(base, probe, spec,
                                       progress=parent.progress_printer('base probe mean'))
    a.verify_model_unchanged(base, base_before, torch)
    if any(parameter.grad is not None for parameter in base.parameters()):
        raise ValueError('Historical base received gradients')
    del base, base_before
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    def apply_operator(vector, iteration):
        print(f'CG {iteration}/10 | starting exact endpoint GGN application', flush=True)
        first, timing, fisher_audit = ggn_action_stage(
            final, construction, eligible, vector, spec,
            progress=parent.progress_printer(f'CG {iteration}/10 GGN'))
        return {'iteration': iteration, 'first_batch': first,
                'timings': timing, 'fisher_conservation': fisher_audit}
    def report(row):
        print(f"CG {row['iteration']}/10 | relative residual {row['relative_residual']:.6g} "
              f"| operator {row['operator_seconds']:.1f}s | elapsed "
              f"{row['cumulative_elapsed_seconds']:.1f}s | ETA "
              f"{row['operator_eta_seconds']:.1f}s", flush=True)
    x, solver = fixed_budget_cg(rhs_cpu, eligible, apply_operator,
                                iterations=spec['solver']['iterations'],
                                damping_ratio=spec['solver']['damping_ratio'], report=report)
    if (solver['operator_applications'] != 10 or
            not math.isclose(solver['rhs_norm'], rhs_norm, rel_tol=1e-7, abs_tol=1e-8)):
        raise ValueError('CG fixed-budget/RHS audit mismatch')
    final.zero_grad(set_to_none=True)
    geometry = parameter_diagnostics(x, delta)
    delta.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final_mean, response, jvp_audit = parent.final_probe_jvp(
        final, probe, x, spec, progress=parent.progress_printer('final block13 JVP'))
    x.clear()
    a.verify_model_unchanged(final, final_before, torch)
    target, target_hash = verified_matched_target(final_mean, base_mean, spec)
    activation = parent.response_metrics(response, target)
    category = interpretation(activation['positions_1_4_mean_cosine'], spec)
    for directory, expected in ((parent.resolve(spec['paths']['base_directory']), inventories['base_files']),
                                (parent.resolve(spec['paths']['final_directory']), inventories['final_files'])):
        if a.checkpoint_file_records(directory) != expected:
            raise ValueError('Checkpoint files changed during Attempt106')
    result = {'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
              'privileged_mechanism_diagnostic': True, 'historical_base_access': True,
              'oracle_adl_access': False, 'adapter_access': False, 'inverse_solve': True,
              'inverse_method': 'fixed_10_step_damped_cg', 'empirical_fisher': False,
              'ggn_operator': 'exact_endpoint_categorical_JtFJ', 'damping_ratio': 0.01,
              'ggn_numerical_implementation':
                  'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast',
              'candidate_selection': False, 'oracle_based_tuning': False,
              'provenance': {'spec_sha256': SPEC_SHA256,
                             'source_sha256': a.sha256_file(Path(__file__).resolve()),
                             'attempt105_result_sha256': link['result_sha256'],
                             'base_checkpoint_files': inventories['base_files'],
                             'final_checkpoint_files': inventories['final_files'],
                             'fineweb_tokens_sha256': spec['corpus']['serialized_sha256'],
                             'probe_tokens_sha256': spec['probe']['serialized_sha256'],
                             'matched_target_raw_sha256': target_hash,
                             'inverse_response_raw_sha256': a.sha256_raw_float32_tensor(response, torch)},
              'inventory': {'construction_sequences': 64, 'construction_contexts': 8128,
                            'probe_sequences': 1024, 'eligible_matrices': 98},
              'hybrid_rhs': {**rhs_info, 'norm': rhs_norm},
              'displacement_audit': displacement_audit,
              'execution_semantics_audit': execution_audit,
              'solver': solver, 'parameter_diagnostics': geometry,
              'matched_activation': {'first_batch_jvp_primal_max_abs_difference': jvp_audit,
                                     **activation},
              'comparison': {'attempt104_primary': spec['baselines']['attempt104_primary'],
                             'attempt104_secondary': spec['baselines']['attempt104_secondary'],
                             'attempt103_primary': spec['baselines']['attempt103_primary'],
                             'attempt103_secondary': spec['baselines']['attempt103_secondary'],
                             'delta_primary_vs_attempt104': activation['positions_1_4_mean_cosine']-
                                 spec['baselines']['attempt104_primary'],
                             'delta_secondary_vs_attempt104': activation['positions_1_127_mean_cosine']-
                                 spec['baselines']['attempt104_secondary'],
                             'attempt016_context_only': {'primary_approx':
                                 spec['baselines']['attempt016_primary_context_approx'],
                                 'secondary_approx': spec['baselines']['attempt016_secondary_context_approx']}},
              'interpretation': {'category': category, 'thresholds': spec['interpretation'],
                                 'descriptive_compute_decision_only': True},
              'timing': {'model_load': loads, 'displacement_seconds': displacement_seconds,
                         'solve_seconds': solver['elapsed_seconds']}}
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt106 result already exists')
    a.write_manifest(output, result)
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), allow_nan=False))


if __name__ == '__main__':
    main()
