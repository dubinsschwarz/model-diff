#!/usr/bin/env python3
"""Attempt109: isolate damping in the frozen exact-RHS, 15-step GGN solve."""
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
ATTEMPT = '109_privileged_exact_rhs_inverse_ggn_lambda_div10_diagnostic'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '676ec6a93acb1ee6cc7079be4340e2c3d20c9f63f1876c3812e1f5c06bbbda68'
SOURCE108 = Path(__file__).with_name('diagnose_privileged_exact_rhs_inverse_ggn_cg15.py')
SOURCE108_SHA256 = '96525b4802de103da9bd40fc2bff940201a6c7edb358c03857b275496c168db7'
COMMIT108 = '78aaa16389d9a6c90b316bccd1d750f12bb947eb'
FIXED_LAMBDA = 1.3361159773737232
FIXED_LAMBDA_SOURCE = 'one_tenth_frozen_Attempt108_numeric_lambda'
if hashlib.sha256(SOURCE108.read_bytes()).hexdigest() != SOURCE108_SHA256:
    raise ValueError('Frozen Attempt108 source changed')
loader = importlib.util.spec_from_file_location('attempt109_frozen_attempt108', SOURCE108)
a108 = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a108)
a107 = a108.a107
a106 = a108.a106
a = a108.a
parent = a108.parent


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt109 spec path changed')
    parent.require_hash(path, SPEC_SHA256)
    spec = a.load_json_object(path, 'Attempt109 spec')
    old = a108.load_spec()
    for key in ('attempt104', 'attempt016', 'attempt106', 'attempt107', 'base',
                'final_checkpoint_files', 'model', 'readout', 'eligible_tensors',
                'corpus', 'pilot', 'probe', 'probe_rows', 'displacement', 'ggn',
                'ggn_numerics_106', 'jvp', 'metric', 'target', 'output_policy',
                'exact_rhs'):
        if spec.get(key) != old[key]:
            raise ValueError('Attempt109 differs from frozen Attempt108: '+key)
    old_solver = dict(old['solver'])
    old_solver.update(fixed_lambda=FIXED_LAMBDA,
                      fixed_lambda_source=FIXED_LAMBDA_SOURCE,
                      lambda_relative_to_attempt108=0.1,
                      residual_warning_threshold_descriptive_only=0.05)
    old_information = dict(old['information_policy'])
    old_information['fixed_lambda_source'] = FIXED_LAMBDA_SOURCE
    old_baselines = dict(old['baselines'])
    old_baselines.update(attempt108_primary=0.8539805172947554,
                         attempt108_secondary=0.8579840469294546)
    old_workload = dict(old['workload'])
    old_workload.update(expected_wall_minutes_context=[9, 11],
                        attempt108_wall_minutes_context=9.1)
    expected_interpretation = {
        'clear_damping_suppression_support': {
            'primary_min': 0.90, 'gain_vs_attempt108_min': 0.04},
        'moderate_damping_suppression_support': {
            'primary_min': 0.88, 'gain_vs_attempt108_min': 0.02},
        'lower_damping_solver_truncation_inconclusive': {
            'relative_linear_residual_strictly_greater_than': 0.05},
        'otherwise': 'little_evidence_current_damping_dominates',
        'precedence': ['clear_damping_suppression_support',
                       'moderate_damping_suppression_support',
                       'lower_damping_solver_truncation_inconclusive',
                       'little_evidence_current_damping_dominates'],
        'significance_claim': False, 'candidate_selection': False,
        'adaptive_extra_iterations': False}
    link = spec.get('attempt108', {})
    if (spec.get('attempt_id') != ATTEMPT or
            spec.get('purpose') !=
                'privileged_exact_rhs_fixed_15_step_CG_one_tenth_lambda_damping_isolation_diagnostic_not_blind_recovery' or
            {key: value for key, value in spec['paths'].items() if key != 'result_path'} !=
                {key: value for key, value in old['paths'].items() if key != 'result_path'} or
            spec['paths']['result_path'] !=
                f'experiments/attempts/{ATTEMPT}/result.json' or
            spec.get('solver') != old_solver or
            spec.get('information_policy') != old_information or
            spec.get('baselines') != old_baselines or
            spec.get('workload') != old_workload or
            spec.get('interpretation') != expected_interpretation or
            link.get('commit_sha') != COMMIT108 or
            link.get('spec_sha256') != a108.SPEC_SHA256 or
            link.get('source_sha256') != SOURCE108_SHA256 or
            link.get('result_sha256') !=
                '8f16d43d2b0183db3404e07594fda64f644000b27b830d60da1398499e5875ff' or
            link.get('required_primary') != 0.8539805172947554 or
            link.get('required_secondary') != 0.8579840469294546 or
            link.get('required_lambda') != 13.361159773737231 or
            link.get('required_iterations') != 15 or
            link.get('required_relative_linear_residual') != 0.014458800206977566 or
            link.get('required_rhs_norm') != 14.651196958318208 or
            link.get('required_parameter_cosine') != 0.04140657656036003 or
            link.get('required_candidate_to_delta_norm_ratio') != 0.021099750362425 or
            link.get('required_category') !=
                'little_evidence_10_step_truncation_dominates'):
        raise ValueError('Attempt109 frozen scientific inventory mismatch')
    for value in spec['paths'].values():
        parent.resolve(value)
    return spec


def verify_committed_attempt108(link):
    if link['commit_sha'] != COMMIT108:
        raise ValueError('Attempt108 committed revision mismatch')
    for key in ('spec', 'source', 'result'):
        path = link[key+'_path']
        expected = link[key+'_sha256']
        try:
            blob = subprocess.run(
                ['git', '-c', f'safe.directory={PROJECT}', 'show',
                 f'{COMMIT108}:{path}'], cwd=PROJECT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError('Attempt108 committed file unavailable: '+path) from exc
        if hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('Attempt108 committed '+key+' hash mismatch')


def validate_inputs(spec):
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt109 result already exists')
    link = spec['attempt108']
    for key in ('spec', 'source', 'result'):
        parent.require_hash(parent.resolve(link[key+'_path']), link[key+'_sha256'])
    verify_committed_attempt108(link)
    if (link['spec_sha256'] != a108.SPEC_SHA256 or
            link['source_sha256'] != SOURCE108_SHA256 or
            link['result_sha256'] !=
                '8f16d43d2b0183db3404e07594fda64f644000b27b830d60da1398499e5875ff'):
        raise ValueError('Frozen Attempt108 linkage mismatch')
    result = a.load_json_object(parent.resolve(link['result_path']), 'Attempt108 result')
    solver = result.get('solver', {})
    activation = result.get('matched_activation', {})
    provenance = result.get('provenance', {})
    parameters = result.get('parameter_diagnostics', {}).get('global', {})
    if (result.get('attempt_id') != a108.ATTEMPT or
            provenance.get('spec_sha256') != link['spec_sha256'] or
            provenance.get('source_sha256') != link['source_sha256'] or
            provenance.get('attempt107_result_sha256') !=
                spec['attempt107']['result_sha256'] or
            provenance.get('base_checkpoint_files') != spec['base']['files'] or
            provenance.get('final_checkpoint_files') != spec['final_checkpoint_files'] or
            provenance.get('fineweb_tokens_sha256') != spec['corpus']['serialized_sha256'] or
            provenance.get('probe_tokens_sha256') != spec['probe']['serialized_sha256'] or
            provenance.get('matched_target_raw_sha256') != spec['target']['raw_float32_sha256'] or
            activation.get('positions_1_4_mean_cosine') != link['required_primary'] or
            activation.get('positions_1_127_mean_cosine') != link['required_secondary'] or
            result.get('lambda') != link['required_lambda'] or
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
        raise ValueError('Attempt108 result/provenance/solver mismatch')
    old108 = a108.load_spec(parent.resolve(link['spec_path']))
    proxy = copy.deepcopy(old108)
    proxy['paths']['result_path'] = spec['paths']['result_path']
    return a108.validate_inputs(proxy)


def fixed_15_step_cg(rhs_cpu, eligible, apply_operator, spec, *, report=None):
    solver_spec = spec['solver']
    if (solver_spec['iterations'] != 15 or
            solver_spec['fixed_lambda'] != FIXED_LAMBDA or
            solver_spec['fixed_lambda_source'] != FIXED_LAMBDA_SOURCE or
            solver_spec['lambda_relative_to_attempt108'] != 0.1 or
            solver_spec['candidate'] != 'x_15' or
            solver_spec['residual_early_stop'] is not False or
            solver_spec['preconditioner'] is not None):
        raise ValueError('Attempt109 fixed 15-step CG settings changed')
    x, solver = a107.fixed_lambda_cg(
        rhs_cpu, eligible, apply_operator,
        fixed_lambda=solver_spec['fixed_lambda'], iterations=15, report=report)
    if (solver['operator_applications'] != 15 or len(solver['iterations']) != 15 or
            solver['candidate'] != 'x_15' or solver['no_early_stop'] is not True or
            solver['lambda'] != FIXED_LAMBDA):
        raise ValueError('Attempt109 fixed 15-step CG result mismatch')
    solver['method'] = 'fixed_15_step_fixed_lambda_cg'
    solver['fixed_lambda_source'] = FIXED_LAMBDA_SOURCE
    solver['lambda_relative_to_attempt108'] = 0.1
    threshold = solver_spec['residual_warning_threshold_descriptive_only']
    if threshold != 0.05:
        raise ValueError('Attempt109 descriptive residual threshold changed')
    solver['residual_warning_threshold_descriptive_only'] = threshold
    solver['residual_warning'] = solver['relative_linear_residual'] > threshold
    solver['residual_warning_does_not_extend_iterations'] = True
    return x, solver


def interpretation(primary, relative_residual, spec):
    if not math.isfinite(primary) or not math.isfinite(relative_residual):
        raise ValueError('Nonfinite Attempt109 interpretation input')
    gain = primary-spec['baselines']['attempt108_primary']
    rules = spec['interpretation']
    if (primary >= rules['clear_damping_suppression_support']['primary_min'] and
            gain >= rules['clear_damping_suppression_support']['gain_vs_attempt108_min']):
        return 'clear_damping_suppression_support'
    if (primary >= rules['moderate_damping_suppression_support']['primary_min'] and
            gain >= rules['moderate_damping_suppression_support']['gain_vs_attempt108_min']):
        return 'moderate_damping_suppression_support'
    if (relative_residual > rules['lower_damping_solver_truncation_inconclusive']
            ['relative_linear_residual_strictly_greater_than']):
        return 'lower_damping_solver_truncation_inconclusive'
    return 'little_evidence_current_damping_dominates'


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
    rhs_cpu, rhs_geometry, rhs_audit = a107.exact_rhs_stage(
        final, construction, eligible, delta, spec,
        progress=parent.progress_printer('exact RHS G Delta_E'))
    rhs_seconds = time.perf_counter()-rhs_started

    def apply_operator(vector, iteration):
        print(f'CG {iteration}/15 | starting exact endpoint GGN application', flush=True)
        first, timing, fisher_audit = a106.ggn_action_stage(
            final, construction, eligible, vector, spec,
            progress=parent.progress_printer(f'CG {iteration}/15 GGN'))
        return {'iteration': iteration, 'first_batch': first,
                'timings': timing, 'fisher_conservation': fisher_audit}

    def report(row):
        print(f"CG {row['iteration']}/15 | relative residual {row['relative_residual']:.6g} "
              f"| operator {row['operator_seconds']:.1f}s | elapsed "
              f"{row['cumulative_elapsed_seconds']:.1f}s | ETA "
              f"{row['operator_eta_seconds']:.1f}s", flush=True)

    x, solver = fixed_15_step_cg(rhs_cpu, eligible, apply_operator, spec, report=report)
    if not math.isclose(solver['rhs_norm'], rhs_geometry['norm'],
                        rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError('Fixed 15-step CG RHS audit mismatch')
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
    primary_gain = primary-spec['baselines']['attempt108_primary']
    for directory, expected in ((parent.resolve(spec['paths']['base_directory']),
                                 inventories['base_files']),
                                (parent.resolve(spec['paths']['final_directory']),
                                 inventories['final_files'])):
        if a.checkpoint_file_records(directory) != expected:
            raise ValueError('Checkpoint files changed during Attempt109')
    result = {
        'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'privileged_mechanism_diagnostic': True, 'historical_base_access': True,
        'true_delta_rhs': True, 'oracle_adl_access': False, 'adapter_access': False,
        'hybrid_teacher_rhs_recomputed': False,
        'ggn_operator': 'exact_endpoint_categorical_JtFJ', 'inverse_solve': True,
        'inverse_method': 'fixed_15_step_fixed_lambda_cg',
        'fixed_lambda_source': FIXED_LAMBDA_SOURCE,
        'lambda': FIXED_LAMBDA, 'fixed_lambda': FIXED_LAMBDA,
        'lambda_relative_to_attempt108': 0.1, 'candidate_selection': False,
        'oracle_based_tuning': False, 'empirical_fisher': False,
        'ggn_numerical_implementation':
            'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast',
        'provenance': {
            'spec_sha256': SPEC_SHA256,
            'source_sha256': a.sha256_file(Path(__file__).resolve()),
            'attempt108_commit_sha': spec['attempt108']['commit_sha'],
            'attempt108_spec_sha256': spec['attempt108']['spec_sha256'],
            'attempt108_source_sha256': spec['attempt108']['source_sha256'],
            'attempt108_result_sha256': spec['attempt108']['result_sha256'],
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
            'attempt107_primary': spec['baselines']['attempt107_primary'],
            'attempt108_primary': spec['baselines']['attempt108_primary'],
            'attempt108_secondary': spec['baselines']['attempt108_secondary'],
            'gain_primary_vs_attempt108': primary_gain,
            'gain_secondary_vs_attempt108':
                secondary-spec['baselines']['attempt108_secondary'],
            'approximate_exact_delta_primary_ceiling':
                spec['baselines']['attempt016_primary_context_approx'],
            'approximate_exact_delta_secondary_ceiling':
                spec['baselines']['attempt016_secondary_context_approx'],
            'fraction_of_approximate_remaining_primary_gap_closed':
                primary_gain/(spec['baselines']['attempt016_primary_context_approx']-
                              spec['baselines']['attempt108_primary']),
            'ceiling_is_approximate_context_only': True},
        'interpretation': {
            'category': category, 'thresholds': spec['interpretation'],
            'descriptive_compute_decision_only': True,
            'final_relative_linear_residual': residual,
            'residual_warning_threshold_descriptive_only': 0.05,
            'residual_warning': solver['residual_warning'],
            'no_adaptive_extra_iterations': True,
            'lower_damping_and_fixed_15_steps_can_confound_damping_inference': True},
        'timing': {'model_load': loads, 'displacement_seconds': displacement_seconds,
                   'exact_rhs_seconds': rhs_seconds,
                   'solve_seconds': solver['elapsed_seconds']}}
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt109 result already exists')
    a.write_manifest(output, result)
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), allow_nan=False))


if __name__ == '__main__':
    main()
