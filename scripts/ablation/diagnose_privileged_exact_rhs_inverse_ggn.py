#!/usr/bin/env python3
"""Attempt107: privileged exact-G-Delta RHS with fixed Attempt106 lambda."""
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
ATTEMPT = '107_privileged_exact_rhs_inverse_ggn_diagnostic'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '2d177cea099888a9310910dae36c784d6a5fe3a754e3ae43f11d6d44e893a8a9'
SOURCE106 = Path(__file__).with_name('diagnose_privileged_inverse_ggn_cg.py')
SOURCE106_SHA256 = '5038111d143b279bd3dc4ac8be2c3ba6e4848b9f920280be531c57d7d160912e'
if hashlib.sha256(SOURCE106.read_bytes()).hexdigest() != SOURCE106_SHA256:
    raise ValueError('Frozen Attempt106 source changed')
loader = importlib.util.spec_from_file_location('attempt107_frozen_attempt106', SOURCE106)
a106 = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a106)
a = a106.a
parent = a106.parent


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt107 spec path changed')
    parent.require_hash(path, SPEC_SHA256)
    spec = a.load_json_object(path, 'Attempt107 spec')
    old = a106.load_spec()
    for key in ('attempt104', 'attempt016', 'base', 'final_checkpoint_files',
                'model', 'readout', 'eligible_tensors', 'corpus', 'pilot', 'probe',
                'probe_rows', 'displacement', 'ggn_numerics_106', 'jvp',
                'metric', 'target', 'output_policy'):
        if spec.get(key) != old[key]:
            raise ValueError('Attempt107 differs from frozen Attempt106: '+key)
    old_ggn = dict(old['ggn'])
    old_ggn['inverse_solve'] = True
    if spec.get('ggn') != old_ggn:
        raise ValueError('Attempt107 GGN operator differs from frozen Attempt106')
    if (spec.get('attempt_id') != ATTEMPT or
            {key: value for key, value in spec['paths'].items() if key != 'result_path'} !=
                {key: value for key, value in old['paths'].items() if key != 'result_path'} or
            spec['solver']['iterations'] != 10 or
            spec['solver']['fixed_lambda'] != 13.361159773737231 or
            spec['solver']['fixed_lambda_source'] != 'frozen_Attempt106_numeric_value' or
            spec['solver']['recompute_rho_from_exact_rhs'] is not False or
            spec['solver']['residual_early_stop'] is not False or
            spec['exact_rhs']['definition'] != 'b_exact=G1_Delta_E' or
            spec['exact_rhs']['uses_attempt106_numerics'] is not True or
            spec['exact_rhs']['hybrid_teacher_rhs_recomputed'] is not False):
        raise ValueError('Attempt107 fixed-lambda/exact-RHS inventory mismatch')
    for value in spec['paths'].values():
        parent.resolve(value)
    return spec


def validate_inputs(spec):
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt107 result already exists')
    link = spec['attempt106']
    for key in ('spec', 'source', 'result'):
        parent.require_hash(parent.resolve(link[key+'_path']), link[key+'_sha256'])
    if (link['source_sha256'] != SOURCE106_SHA256 or
            link['spec_sha256'] != a106.SPEC_SHA256 or
            link['result_sha256'] !=
                '940fb272d705e90eaf750419a542c9088f1996eb24326377d009681d4433da61'):
        raise ValueError('Frozen Attempt106 linkage mismatch')
    result106 = a.load_json_object(parent.resolve(link['result_path']), 'Attempt106 result')
    solver = result106.get('solver', {})
    activation = result106.get('matched_activation', {})
    provenance = result106.get('provenance', {})
    if (result106.get('attempt_id') != a106.ATTEMPT or
            provenance.get('spec_sha256') != link['spec_sha256'] or
            provenance.get('source_sha256') != link['source_sha256'] or
            provenance.get('base_checkpoint_files') != spec['base']['files'] or
            provenance.get('final_checkpoint_files') != spec['final_checkpoint_files'] or
            provenance.get('fineweb_tokens_sha256') != spec['corpus']['serialized_sha256'] or
            provenance.get('probe_tokens_sha256') != spec['probe']['serialized_sha256'] or
            provenance.get('matched_target_raw_sha256') != spec['target']['raw_float32_sha256'] or
            activation.get('positions_1_4_mean_cosine') != link['required_primary'] or
            activation.get('positions_1_127_mean_cosine') != link['required_secondary'] or
            solver.get('rhs_norm') != link['required_rhs_norm'] or
            solver.get('rho_b') != link['required_rho_b'] or
            solver.get('damping_ratio') != link['required_damping_ratio'] or
            solver.get('lambda') != link['required_lambda'] or
            solver.get('operator_applications') != link['required_cg_iterations'] or
            len(solver.get('iterations', [])) != link['required_cg_iterations'] or
            solver.get('relative_linear_residual') != link['required_relative_linear_residual'] or
            result106.get('interpretation', {}).get('category') != link['required_category'] or
            result106.get('ggn_operator') != 'exact_endpoint_categorical_JtFJ' or
            result106.get('ggn_numerical_implementation') !=
                'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast' or
            result106.get('inverse_method') != 'fixed_10_step_damped_cg' or
            result106.get('empirical_fisher') is not False or
            result106.get('oracle_adl_access') is not False or
            result106.get('adapter_access') is not False):
        raise ValueError('Attempt106 result/provenance/solver mismatch')
    old106 = a106.load_spec(parent.resolve(link['spec_path']))
    proxy = copy.deepcopy(old106)
    proxy['paths']['result_path'] = spec['paths']['result_path']
    return a106.validate_inputs(proxy)


def copy_exact_rhs_from_grad(eligible):
    if len(eligible) != 98:
        raise ValueError('Exact G Delta RHS support is not 98 matrices')
    rhs = {}
    for name, module in eligible:
        gradient = module.weight.grad
        if (gradient is None or gradient.dtype != torch.float32 or
                gradient.shape != module.weight.shape or
                not bool(torch.isfinite(gradient).all())):
            raise ValueError('Missing/nonfinite exact G Delta RHS matrix')
        rhs[name+'.weight'] = gradient.detach().to(device='cpu', dtype=torch.float32,
                                                   copy=True).contiguous()
    if len(rhs) != 98:
        raise ValueError('Duplicate exact RHS coordinate')
    return rhs


def rhs_diagnostics(rhs_cpu, delta):
    if len(rhs_cpu) != 98 or set(rhs_cpu) != set(delta):
        raise ValueError('Exact RHS/Delta coordinate mismatch')
    bb = a106.matrixwise_dot(rhs_cpu, rhs_cpu)
    dd = a106.matrixwise_dot(delta, delta)
    cross = []
    for name in sorted(rhs_cpu):
        cpu_delta = delta[name].detach().to(device='cpu', dtype=torch.float32)
        cross.append(float(torch.sum(rhs_cpu[name]*cpu_delta, dtype=torch.float64)))
        del cpu_delta
    bd = math.fsum(cross)
    if not all(math.isfinite(value) for value in (bb, dd, bd)) or bb <= 0 or dd <= 0:
        raise ValueError('Zero/nonfinite exact RHS or Delta norm')
    return {'norm': math.sqrt(bb), 'cosine_with_true_delta': bd/math.sqrt(bb*dd),
            'norm_to_true_delta_ratio': math.sqrt(bb/dd),
            'attempt105_ggn_norm_context_only': 14.651187701059072,
            'norm_difference_from_attempt105_context': math.sqrt(bb)-14.651187701059072,
            'no_byte_identical_requirement': True}


def exact_rhs_stage(final, construction, eligible, delta, spec, *, progress=None):
    first_audit, timing, fisher_audit = a106.ggn_action_stage(
        final, construction, eligible, delta, spec, progress=progress)
    rhs = copy_exact_rhs_from_grad(eligible)
    final.zero_grad(set_to_none=True)
    geometry = rhs_diagnostics(rhs, delta)
    return rhs, geometry, {'first_batch_audit': first_audit,
                           'operator_timing': timing,
                           'fisher_conservation': fisher_audit}


def fixed_lambda_cg(rhs_cpu, eligible, apply_operator, *, fixed_lambda, iterations=10,
                    report=None):
    names = a106._names(eligible)
    if (set(rhs_cpu) != set(names) or iterations <= 0 or
            not math.isfinite(fixed_lambda) or fixed_lambda <= 0):
        raise ValueError('Invalid fixed-lambda CG support/settings')
    r = {}
    for name, module in eligible:
        key = name+'.weight'
        value = rhs_cpu[key]
        if (value.dtype != torch.float32 or value.device.type != 'cpu' or
                value.shape != module.weight.shape or not value.is_contiguous() or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Malformed/nonfinite exact RHS matrix')
        r[key] = value.to(module.weight.device, dtype=torch.float32, copy=True).contiguous()
    rhs_cpu.clear()
    p = {name: value.clone() for name, value in r.items()}
    x = {name: torch.zeros_like(value) for name, value in r.items()}
    rr = a106.matrixwise_dot(r, r)
    if not math.isfinite(rr) or rr <= 0:
        raise ValueError('Zero/nonfinite exact RHS norm')
    b_norm = math.sqrt(rr)
    rows, operator_audits, operator_seconds = [], [], []
    started = time.perf_counter()
    for iteration in range(1, iterations+1):
        operator_started = time.perf_counter()
        audit = apply_operator(p, iteration)
        elapsed_operator = time.perf_counter()-operator_started
        operator_seconds.append(elapsed_operator)
        operator_audits.append(audit)
        p_gp = a106.matrixwise_grad_dot(eligible, p)
        pp = a106.matrixwise_dot(p, p)
        p_ap = p_gp+fixed_lambda*pp
        if not math.isfinite(p_ap) or p_ap <= 0:
            raise ValueError('Nonpositive/nonfinite fixed-lambda CG p^T A p')
        alpha = rr/p_ap
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Nonpositive/nonfinite fixed-lambda CG alpha')
        for name, module in eligible:
            key = name+'.weight'
            x[key].add_(p[key], alpha=alpha)
            r[key].add_(module.weight.grad, alpha=-alpha)
            r[key].add_(p[key], alpha=-alpha*fixed_lambda)
        rr_new = a106.matrixwise_dot(r, r)
        if not math.isfinite(rr_new) or rr_new < 0:
            raise ValueError('Nonfinite/negative fixed-lambda CG residual')
        beta = rr_new/rr
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('Nonfinite/negative fixed-lambda CG beta')
        for name in names:
            p[name].mul_(beta).add_(r[name])
            if not bool(torch.isfinite(x[name]).all()) or not bool(torch.isfinite(p[name]).all()):
                raise ValueError('Nonfinite fixed-lambda CG vector')
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
    return x, {'method': 'fixed_10_step_fixed_lambda_cg', 'iterations': rows,
               'operator_applications': len(operator_audits),
               'operator_audits': operator_audits,
               'rhs_norm': b_norm, 'lambda': fixed_lambda,
               'fixed_lambda_source': 'frozen_Attempt106_numeric_value',
               'rho_from_exact_rhs_computed': False,
               'relative_linear_residual': rows[-1]['relative_residual'],
               'residual_source': 'CG_recurrence_no_extra_operator_application',
               'candidate': f'x_{iterations}', 'no_early_stop': True,
               'elapsed_seconds': time.perf_counter()-started}


def interpretation(primary, spec):
    gain = primary-spec['baselines']['attempt106_primary']
    rules = spec['interpretation']
    if (primary >= rules['strong_rhs_mismatch_explanation']['primary_min'] and
            gain >= rules['strong_rhs_mismatch_explanation']['gain_vs_attempt106_min']):
        return 'strong_rhs_mismatch_explanation'
    if (primary >= rules['mixed_rhs_and_inverse_limitations']['primary_min'] and
            gain >= rules['mixed_rhs_and_inverse_limitations']['gain_vs_attempt106_min']):
        return 'mixed_rhs_and_inverse_limitations'
    return 'inverse_or_regularization_limitations_dominate'


def run(*, model_loader=None, tokenizer_loader=None):
    spec = load_spec()
    spec016, inventories = validate_inputs(spec)
    construction, probe = a106.load_frozen_inputs(spec)
    base, final, loads = parent.load_model_pair(
        spec, model_loader=model_loader, tokenizer_loader=tokenizer_loader)
    base_before = a.model_state_hashes(base, torch)
    final_before = a.model_state_hashes(final, torch)
    displacement_started = time.perf_counter()
    delta, displacement_audit, execution_audit, eligible = a106.a105.prepare_displacement(
        base, final, spec016, inventories['base_files'])
    displacement_seconds = time.perf_counter()-displacement_started
    base_mean = parent.base_probe_mean(
        base, probe, spec, progress=parent.progress_printer('base probe mean'))
    a.verify_model_unchanged(base, base_before, torch)
    if any(parameter.grad is not None for parameter in base.parameters()):
        raise ValueError('Historical base received gradients')
    del base, base_before
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('exact RHS | starting G Delta_E application', flush=True)
    rhs_started = time.perf_counter()
    rhs_cpu, rhs_geometry, rhs_audit = exact_rhs_stage(
        final, construction, eligible, delta, spec,
        progress=parent.progress_printer('exact RHS G Delta_E'))
    rhs_seconds = time.perf_counter()-rhs_started
    def apply_operator(vector, iteration):
        print(f'CG {iteration}/10 | starting exact endpoint GGN application', flush=True)
        first, timing, fisher_audit = a106.ggn_action_stage(
            final, construction, eligible, vector, spec,
            progress=parent.progress_printer(f'CG {iteration}/10 GGN'))
        return {'iteration': iteration, 'first_batch': first,
                'timings': timing, 'fisher_conservation': fisher_audit}
    def report(row):
        print(f"CG {row['iteration']}/10 | relative residual {row['relative_residual']:.6g} "
              f"| operator {row['operator_seconds']:.1f}s | elapsed "
              f"{row['cumulative_elapsed_seconds']:.1f}s | ETA "
              f"{row['operator_eta_seconds']:.1f}s", flush=True)
    x, solver = fixed_lambda_cg(rhs_cpu, eligible, apply_operator,
                                fixed_lambda=spec['solver']['fixed_lambda'],
                                iterations=spec['solver']['iterations'], report=report)
    if (solver['operator_applications'] != 10 or
            not math.isclose(solver['rhs_norm'], rhs_geometry['norm'],
                             rel_tol=1e-7, abs_tol=1e-8)):
        raise ValueError('Fixed-lambda CG application/RHS audit mismatch')
    final.zero_grad(set_to_none=True)
    parameter_geometry = a106.parameter_diagnostics(x, delta)
    delta.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    final_mean, response, jvp_audit = parent.final_probe_jvp(
        final, probe, x, spec, progress=parent.progress_printer('final block13 JVP'))
    x.clear()
    a.verify_model_unchanged(final, final_before, torch)
    target, target_hash = a106.verified_matched_target(final_mean, base_mean, spec)
    activation = parent.response_metrics(response, target)
    primary = activation['positions_1_4_mean_cosine']
    secondary = activation['positions_1_127_mean_cosine']
    category = interpretation(primary, spec)
    primary_gain = primary-spec['baselines']['attempt106_primary']
    approximate_gap = (spec['baselines']['attempt016_primary_context_approx']-
                       spec['baselines']['attempt106_primary'])
    if approximate_gap <= 0:
        raise ValueError('Invalid frozen approximate ceiling gap')
    for directory, expected in ((parent.resolve(spec['paths']['base_directory']), inventories['base_files']),
                                (parent.resolve(spec['paths']['final_directory']), inventories['final_files'])):
        if a.checkpoint_file_records(directory) != expected:
            raise ValueError('Checkpoint files changed during Attempt107')
    result = {'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
              'privileged_mechanism_diagnostic': True, 'historical_base_access': True,
              'true_delta_rhs': True, 'oracle_adl_access': False, 'adapter_access': False,
              'hybrid_teacher_rhs_recomputed': False,
              'ggn_operator': 'exact_endpoint_categorical_JtFJ', 'inverse_solve': True,
              'inverse_method': 'fixed_10_step_fixed_lambda_cg',
              'fixed_lambda_source': 'frozen_Attempt106_numeric_value',
              'lambda': 13.361159773737231, 'candidate_selection': False,
              'oracle_based_tuning': False, 'empirical_fisher': False,
              'ggn_numerical_implementation':
                  'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast',
              'provenance': {'spec_sha256': SPEC_SHA256,
                             'source_sha256': a.sha256_file(Path(__file__).resolve()),
                             'attempt106_result_sha256': spec['attempt106']['result_sha256'],
                             'base_checkpoint_files': inventories['base_files'],
                             'final_checkpoint_files': inventories['final_files'],
                             'fineweb_tokens_sha256': spec['corpus']['serialized_sha256'],
                             'probe_tokens_sha256': spec['probe']['serialized_sha256'],
                             'matched_target_raw_sha256': target_hash,
                             'response_raw_sha256': a.sha256_raw_float32_tensor(response, torch)},
              'inventory': {'construction_sequences': 64, 'construction_contexts': 8128,
                            'probe_sequences': 1024, 'eligible_matrices': 98},
              'exact_rhs': {**rhs_geometry, **rhs_audit},
              'displacement_audit': displacement_audit,
              'execution_semantics_audit': execution_audit,
              'solver': solver, 'parameter_diagnostics': parameter_geometry,
              'matched_activation': {'first_batch_jvp_primal_max_abs_difference': jvp_audit,
                                     **activation},
              'comparison': {'attempt106_primary': spec['baselines']['attempt106_primary'],
                             'attempt106_secondary': spec['baselines']['attempt106_secondary'],
                             'attempt104_primary': spec['baselines']['attempt104_primary'],
                             'attempt104_secondary': spec['baselines']['attempt104_secondary'],
                             'gain_primary_vs_attempt106': primary_gain,
                             'gain_secondary_vs_attempt106': secondary-
                                 spec['baselines']['attempt106_secondary'],
                             'approximate_exact_delta_primary_ceiling':
                                 spec['baselines']['attempt016_primary_context_approx'],
                             'approximate_remaining_primary_gap_from_attempt106': approximate_gap,
                             'fraction_of_approximate_primary_gap_closed': primary_gain/approximate_gap,
                             'ceiling_is_approximate_context_only': True},
              'interpretation': {'category': category, 'thresholds': spec['interpretation'],
                                 'descriptive_compute_decision_only': True},
              'timing': {'model_load': loads, 'displacement_seconds': displacement_seconds,
                         'exact_rhs_seconds': rhs_seconds,
                         'solve_seconds': solver['elapsed_seconds']}}
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt107 result already exists')
    a.write_manifest(output, result)
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), allow_nan=False))


if __name__ == '__main__':
    main()
