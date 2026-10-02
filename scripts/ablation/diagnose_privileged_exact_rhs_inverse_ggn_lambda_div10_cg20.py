#!/usr/bin/env python3
"""Attempt110: continue the frozen lower-damping exact-RHS GGN CG to 20 steps."""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import subprocess
import time
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '110_privileged_exact_rhs_inverse_ggn_lambda_div10_cg20_diagnostic'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = 'ae79cfb13a0ba4fd27340c58f2b62e6f04c765aa404867e038a0337b5dd9e02c'
SOURCE109 = Path(__file__).with_name('diagnose_privileged_exact_rhs_inverse_ggn_lambda_div10.py')
SOURCE109_SHA256 = 'aaf9106af729c8ef8bcffedc700634531fd7e627fc1738055baaefc3e31c18c9'
COMMIT109 = '0323f78ba2fd598cf5a502dc8e710a658957bc00'
FIXED_LAMBDA = 1.3361159773737232
FIXED_LAMBDA_SOURCE = 'one_tenth_frozen_Attempt108_numeric_lambda'
if hashlib.sha256(SOURCE109.read_bytes()).hexdigest() != SOURCE109_SHA256:
    raise ValueError('Frozen Attempt109 source changed')
loader = importlib.util.spec_from_file_location('attempt110_frozen_attempt109', SOURCE109)
a109 = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a109)
a108 = a109.a108
a107 = a109.a107
a106 = a109.a106
a = a109.a
parent = a109.parent


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt110 spec path changed')
    parent.require_hash(path, SPEC_SHA256)
    spec = a.load_json_object(path, 'Attempt110 spec')
    old = a109.load_spec()
    for key in ('attempt104', 'attempt016', 'attempt106', 'attempt107',
                'attempt108', 'base',
                'final_checkpoint_files', 'model', 'readout', 'eligible_tensors',
                'corpus', 'pilot', 'probe', 'probe_rows', 'displacement', 'ggn',
                'ggn_numerics_106', 'jvp', 'metric', 'target', 'output_policy',
                'exact_rhs'):
        if spec.get(key) != old[key]:
            raise ValueError('Attempt110 differs from frozen Attempt109: '+key)
    old_solver = dict(old['solver'])
    old_solver.update(iterations=20, candidate='x_20')
    old_information = dict(old['information_policy'])
    old_information['inverse_method'] = 'fixed_20_step_fixed_lambda_cg'
    old_baselines = dict(old['baselines'])
    old_baselines.update(attempt109_primary=0.8802197432881782,
                         attempt109_secondary=0.8858106177907814)
    old_workload = dict(old['workload'])
    old_workload.update(cg_ggn_applications=20,
                        expected_wall_minutes_context=[10, 11],
                        attempt109_wall_seconds_context=534,
                        additional_operator_applications_vs_109=5,
                        additional_operator_seconds_context=[100, 110])
    expected_interpretation = {
        'clear_lower_damping_convergence_effect': {
            'primary_min': 0.91, 'gain_vs_attempt109_min': 0.025},
        'moderate_lower_damping_convergence_effect': {
            'primary_min': 0.895, 'gain_vs_attempt109_min': 0.012},
        'still_materially_underconverged': {
            'relative_linear_residual_strictly_greater_than': 0.02},
        'otherwise': 'little_evidence_15_step_truncation_dominates_at_lower_damping',
        'precedence': ['clear_lower_damping_convergence_effect',
                       'moderate_lower_damping_convergence_effect',
                       'still_materially_underconverged',
                       'little_evidence_15_step_truncation_dominates_at_lower_damping'],
        'significance_claim': False, 'candidate_selection': False,
        'adaptive_extra_iterations': False}
    expected_next_step = {
        'clear_or_moderate':
            'consider_another_fixed_convergence_extension_before_changing_damping',
        'still_materially_underconverged': 'convergence_remains_unresolved',
        'little_evidence': 'reduce_damping_again_not_merely_add_more_CG',
        'execute_next_attempt': False}
    expected_trajectory = {
        'reference': 'frozen_Attempt109_solver_iterations',
        'iterations_to_compare': 15,
        'fields': ['relative_residual', 'r_norm', 'pAp', 'alpha', 'beta'],
        'rtol': 1e-6, 'atol': 1e-8,
        'timings_excluded': True, 'fail_on_mismatch_before_next_iteration': True,
        'activation_metrics_used': False}
    link = spec.get('attempt109', {})
    if (spec.get('attempt_id') != ATTEMPT or
            spec.get('purpose') !=
                'privileged_lower_damping_exact_rhs_fixed_20_step_CG_convergence_diagnostic_not_blind_recovery' or
            {key: value for key, value in spec['paths'].items() if key != 'result_path'} !=
                {key: value for key, value in old['paths'].items() if key != 'result_path'} or
            spec['paths']['result_path'] !=
                f'experiments/attempts/{ATTEMPT}/result.json' or
            spec.get('solver') != old_solver or
            spec.get('information_policy') != old_information or
            spec.get('baselines') != old_baselines or
            spec.get('workload') != old_workload or
            spec.get('interpretation') != expected_interpretation or
            spec.get('next_step_policy') != expected_next_step or
            spec.get('trajectory_audit') != expected_trajectory or
            link.get('commit_sha') != COMMIT109 or
            link.get('spec_sha256') != a109.SPEC_SHA256 or
            link.get('source_sha256') != SOURCE109_SHA256 or
            link.get('result_sha256') !=
                'd1dddffbde0170d3702117b00db628624eed0ea1997d926f7595a3e62c0d6514' or
            link.get('required_primary') != 0.8802197432881782 or
            link.get('required_secondary') != 0.8858106177907814 or
            link.get('required_lambda') != FIXED_LAMBDA or
            link.get('required_iterations') != 15 or
            link.get('required_relative_linear_residual') != 0.04024542110658629 or
            link.get('required_rhs_norm') != 14.651196958318208 or
            link.get('required_parameter_cosine') != 0.04726752336894001 or
            link.get('required_candidate_to_delta_norm_ratio') != 0.029252585148726477 or
            link.get('required_category') !=
                'moderate_damping_suppression_support'):
        raise ValueError('Attempt110 frozen scientific inventory mismatch')
    for value in spec['paths'].values():
        parent.resolve(value)
    return spec


def verify_committed_attempt109(link):
    if link['commit_sha'] != COMMIT109:
        raise ValueError('Attempt109 committed revision mismatch')
    for key in ('spec', 'source', 'result'):
        path = link[key+'_path']
        expected = link[key+'_sha256']
        try:
            blob = subprocess.run(
                ['git', '-c', f'safe.directory={PROJECT}', 'show',
                 f'{COMMIT109}:{path}'], cwd=PROJECT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError('Attempt109 committed file unavailable: '+path) from exc
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('Attempt109 committed '+key+' hash mismatch')


def validate_inputs(spec):
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt110 result already exists')
    link = spec['attempt109']
    for key in ('spec', 'source', 'result'):
        parent.require_hash(parent.resolve(link[key+'_path']), link[key+'_sha256'])
    verify_committed_attempt109(link)
    if (link['spec_sha256'] != a109.SPEC_SHA256 or
            link['source_sha256'] != SOURCE109_SHA256 or
            link['result_sha256'] !=
                'd1dddffbde0170d3702117b00db628624eed0ea1997d926f7595a3e62c0d6514'):
        raise ValueError('Frozen Attempt109 linkage mismatch')
    result = a.load_json_object(parent.resolve(link['result_path']), 'Attempt109 result')
    solver = result.get('solver', {})
    activation = result.get('matched_activation', {})
    provenance = result.get('provenance', {})
    parameters = result.get('parameter_diagnostics', {}).get('global', {})
    if (result.get('attempt_id') != a109.ATTEMPT or
            provenance.get('spec_sha256') != link['spec_sha256'] or
            provenance.get('source_sha256') != link['source_sha256'] or
            provenance.get('attempt108_result_sha256') !=
                spec['attempt108']['result_sha256'] or
            provenance.get('base_checkpoint_files') != spec['base']['files'] or
            provenance.get('final_checkpoint_files') != spec['final_checkpoint_files'] or
            provenance.get('fineweb_tokens_sha256') != spec['corpus']['serialized_sha256'] or
            provenance.get('probe_tokens_sha256') != spec['probe']['serialized_sha256'] or
            provenance.get('matched_target_raw_sha256') != spec['target']['raw_float32_sha256'] or
            activation.get('positions_1_4_mean_cosine') != link['required_primary'] or
            activation.get('positions_1_127_mean_cosine') != link['required_secondary'] or
            result.get('lambda') != link['required_lambda'] or
            result.get('fixed_lambda') != link['required_lambda'] or
            solver.get('lambda') != link['required_lambda'] or
            solver.get('operator_applications') != link['required_iterations'] or
            len(solver.get('iterations', [])) != link['required_iterations'] or
            solver.get('relative_linear_residual') !=
                link['required_relative_linear_residual'] or
            solver.get('rhs_norm') != link['required_rhs_norm'] or
            parameters.get('cosine') != link['required_parameter_cosine'] or
            parameters.get('candidate_to_delta_norm_ratio') !=
                link['required_candidate_to_delta_norm_ratio'] or
            result.get('interpretation', {}).get('category') != link['required_category'] or
            result.get('true_delta_rhs') is not True or
            result.get('inverse_solve') is not True or
            result.get('ggn_operator') != 'exact_endpoint_categorical_JtFJ' or
            result.get('ggn_numerical_implementation') !=
                'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast' or
            result.get('inverse_method') != 'fixed_15_step_fixed_lambda_cg' or
            result.get('empirical_fisher') is not False or
            result.get('oracle_adl_access') is not False or
            result.get('adapter_access') is not False):
        raise ValueError('Attempt109 result/provenance/solver mismatch')
    old109 = a109.load_spec(parent.resolve(link['spec_path']))
    proxy = copy.deepcopy(old109)
    proxy['paths']['result_path'] = spec['paths']['result_path']
    spec016, inventories = a109.validate_inputs(proxy)
    return spec016, inventories, result


def audit_trajectory_row(actual, expected, policy):
    index = actual.get('iteration')
    if (not isinstance(index, int) or not 1 <= index <= 15 or
            expected.get('iteration') != index):
        raise ValueError('Attempt110 first-15 CG trajectory iteration mismatch')
    diagnostics = {'iteration': index, 'fields': {}}
    for field in policy['fields']:
        observed, frozen = actual.get(field), expected.get(field)
        if (not isinstance(observed, (int, float)) or
                not isinstance(frozen, (int, float)) or
                not math.isfinite(observed) or not math.isfinite(frozen)):
            raise ValueError(f'Attempt110 first-15 CG {field} nonfinite/missing')
        error = abs(observed-frozen)
        allowed = policy['atol']+policy['rtol']*abs(frozen)
        diagnostics['fields'][field] = {'absolute_error': error,
                                        'allowed_absolute_error': allowed}
        if error > allowed:
            raise ValueError(
                f'Attempt110 first-15 CG trajectory mismatch at iteration {index}, '
                f'{field}: observed={observed}, frozen={frozen}, error={error}, '
                f'allowed={allowed}')
    return diagnostics


def fixed_20_step_cg(rhs_cpu, eligible, apply_operator, spec, frozen_rows, *, report=None):
    solver_spec = spec['solver']
    policy = spec['trajectory_audit']
    if (solver_spec['iterations'] != 20 or
            solver_spec['fixed_lambda'] != FIXED_LAMBDA or
            solver_spec['fixed_lambda_source'] != FIXED_LAMBDA_SOURCE or
            solver_spec['lambda_relative_to_attempt108'] != 0.1 or
            solver_spec['candidate'] != 'x_20' or
            solver_spec['residual_early_stop'] is not False or
            solver_spec['preconditioner'] is not None or
            len(frozen_rows) != 15 or policy['iterations_to_compare'] != 15 or
            policy['fields'] != ['relative_residual', 'r_norm', 'pAp', 'alpha', 'beta'] or
            policy['rtol'] != 1e-6 or policy['atol'] != 1e-8 or
            policy['timings_excluded'] is not True or
            policy['activation_metrics_used'] is not False):
        raise ValueError('Attempt110 fixed 20-step CG/trajectory settings changed')
    trajectory_rows = []

    def checked_report(row):
        if row['iteration'] <= 15:
            trajectory_rows.append(
                audit_trajectory_row(row, frozen_rows[row['iteration']-1], policy))
        if report is not None:
            report(row)

    x, solver = a107.fixed_lambda_cg(
        rhs_cpu, eligible, apply_operator,
        fixed_lambda=solver_spec['fixed_lambda'], iterations=20, report=checked_report)
    if (solver['operator_applications'] != 20 or len(solver['iterations']) != 20 or
            len(trajectory_rows) != 15 or solver['candidate'] != 'x_20' or
            solver['no_early_stop'] is not True or solver['lambda'] != FIXED_LAMBDA):
        raise ValueError('Attempt110 fixed 20-step CG result mismatch')
    solver['method'] = 'fixed_20_step_fixed_lambda_cg'
    solver['fixed_lambda_source'] = FIXED_LAMBDA_SOURCE
    solver['lambda_relative_to_attempt108'] = 0.1
    threshold = solver_spec['residual_warning_threshold_descriptive_only']
    if threshold != 0.05:
        raise ValueError('Attempt110 descriptive residual threshold changed')
    solver['residual_warning_threshold_descriptive_only'] = threshold
    solver['residual_warning'] = solver['relative_linear_residual'] > threshold
    solver['residual_warning_does_not_extend_iterations'] = True
    solver['first15_trajectory_audit'] = {
        'source': 'frozen_Attempt109_solver_iterations',
        'matched': True, 'rows_compared': len(trajectory_rows),
        'rtol': policy['rtol'], 'atol': policy['atol'],
        'fields': policy['fields'], 'timings_compared': False,
        'activation_metrics_used': False,
        'max_absolute_error_by_field': {
            field: max(row['fields'][field]['absolute_error'] for row in trajectory_rows)
            for field in policy['fields']}}
    return x, solver


def interpretation(primary, relative_residual, spec):
    if not math.isfinite(primary) or not math.isfinite(relative_residual):
        raise ValueError('Nonfinite Attempt110 interpretation input')
    gain = primary-spec['baselines']['attempt109_primary']
    rules = spec['interpretation']
    if (primary >= rules['clear_lower_damping_convergence_effect']['primary_min'] and
            gain >= rules['clear_lower_damping_convergence_effect']['gain_vs_attempt109_min']):
        return 'clear_lower_damping_convergence_effect'
    if (primary >= rules['moderate_lower_damping_convergence_effect']['primary_min'] and
            gain >= rules['moderate_lower_damping_convergence_effect']['gain_vs_attempt109_min']):
        return 'moderate_lower_damping_convergence_effect'
    if (relative_residual > rules['still_materially_underconverged']
            ['relative_linear_residual_strictly_greater_than']):
        return 'still_materially_underconverged'
    return 'little_evidence_15_step_truncation_dominates_at_lower_damping'


def next_step_decision(category, spec):
    policy = spec['next_step_policy']
    if category in ('clear_lower_damping_convergence_effect',
                    'moderate_lower_damping_convergence_effect'):
        return policy['clear_or_moderate']
    if category == 'still_materially_underconverged':
        return policy['still_materially_underconverged']
    return policy['little_evidence']


def run(*, model_loader=None, tokenizer_loader=None):
    spec = load_spec()
    spec016, inventories, result109 = validate_inputs(spec)
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
    rhs_cpu, rhs_geometry, rhs_audit = a107.exact_rhs_stage(
        final, construction, eligible, delta, spec,
        progress=parent.progress_printer('exact RHS G Delta_E'))
    rhs_seconds = time.perf_counter()-rhs_started

    def apply_operator(vector, iteration):
        print(f'CG {iteration}/20 | starting exact endpoint GGN application', flush=True)
        first, timing, fisher_audit = a106.ggn_action_stage(
            final, construction, eligible, vector, spec,
            progress=parent.progress_printer(f'CG {iteration}/20 GGN'))
        return {'iteration': iteration, 'first_batch': first,
                'timings': timing, 'fisher_conservation': fisher_audit}

    def report(row):
        print(f"CG {row['iteration']}/20 | relative residual {row['relative_residual']:.6g} "
              f"| operator {row['operator_seconds']:.1f}s | elapsed "
              f"{row['cumulative_elapsed_seconds']:.1f}s | ETA "
              f"{row['operator_eta_seconds']:.1f}s", flush=True)

    x, solver = fixed_20_step_cg(
        rhs_cpu, eligible, apply_operator, spec, result109['solver']['iterations'],
        report=report)
    if not math.isclose(solver['rhs_norm'], rhs_geometry['norm'],
                        rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError('Fixed 20-step CG RHS audit mismatch')
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
    residual = solver['relative_linear_residual']
    category = interpretation(primary, residual, spec)
    primary_gain = primary-spec['baselines']['attempt109_primary']
    for directory, expected in ((parent.resolve(spec['paths']['base_directory']),
                                 inventories['base_files']),
                                (parent.resolve(spec['paths']['final_directory']),
                                 inventories['final_files'])):
        if a.checkpoint_file_records(directory) != expected:
            raise ValueError('Checkpoint files changed during Attempt110')
    result = {
        'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'privileged_mechanism_diagnostic': True, 'historical_base_access': True,
        'true_delta_rhs': True, 'oracle_adl_access': False, 'adapter_access': False,
        'hybrid_teacher_rhs_recomputed': False,
        'ggn_operator': 'exact_endpoint_categorical_JtFJ', 'inverse_solve': True,
        'inverse_method': 'fixed_20_step_fixed_lambda_cg',
        'fixed_lambda_source': FIXED_LAMBDA_SOURCE,
        'lambda': FIXED_LAMBDA, 'fixed_lambda': FIXED_LAMBDA,
        'lambda_relative_to_attempt108': 0.1, 'candidate_selection': False,
        'oracle_based_tuning': False, 'empirical_fisher': False,
        'ggn_numerical_implementation':
            'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast',
        'provenance': {
            'spec_sha256': SPEC_SHA256,
            'source_sha256': a.sha256_file(Path(__file__).resolve()),
            'attempt109_commit_sha': spec['attempt109']['commit_sha'],
            'attempt109_spec_sha256': spec['attempt109']['spec_sha256'],
            'attempt109_source_sha256': spec['attempt109']['source_sha256'],
            'attempt109_result_sha256': spec['attempt109']['result_sha256'],
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
        'comparison': {
            'attempt108_primary': spec['baselines']['attempt108_primary'],
            'attempt108_lambda': spec['attempt108']['required_lambda'],
            'attempt108_relative_linear_residual':
                spec['attempt108']['required_relative_linear_residual'],
            'attempt109_primary': spec['baselines']['attempt109_primary'],
            'attempt109_secondary': spec['baselines']['attempt109_secondary'],
            'attempt109_relative_linear_residual':
                spec['attempt109']['required_relative_linear_residual'],
            'gain_primary_vs_attempt109': primary_gain,
            'gain_secondary_vs_attempt109':
                secondary-spec['baselines']['attempt109_secondary'],
            'approximate_exact_delta_primary_ceiling':
                spec['baselines']['attempt016_primary_context_approx'],
            'approximate_exact_delta_secondary_ceiling':
                spec['baselines']['attempt016_secondary_context_approx'],
            'fraction_of_approximate_remaining_primary_gap_closed':
                primary_gain/(spec['baselines']['attempt016_primary_context_approx']-
                              spec['baselines']['attempt109_primary']),
            'ceiling_is_approximate_context_only': True},
        'interpretation': {
            'category': category, 'thresholds': spec['interpretation'],
            'descriptive_compute_decision_only': True,
            'final_relative_linear_residual': residual,
            'no_adaptive_extra_iterations': True,
            'next_step_decision_record_only': next_step_decision(category, spec),
            'next_attempt_executed': False},
        'timing': {'model_load': loads, 'displacement_seconds': displacement_seconds,
                   'exact_rhs_seconds': rhs_seconds,
                   'solve_seconds': solver['elapsed_seconds']}}
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt110 result already exists')
    a.write_manifest(output, result)
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), allow_nan=False))


if __name__ == '__main__':
    main()
