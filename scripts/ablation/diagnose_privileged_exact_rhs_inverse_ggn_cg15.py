#!/usr/bin/env python3
"""Attempt108: repeat the frozen exact-RHS inverse GGN with 15 CG steps."""
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
ATTEMPT = '108_privileged_exact_rhs_inverse_ggn_cg15_diagnostic'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = 'e9319d1da079d223b749f5b94aaf3249de1a0597a213d732a799b25921eabb4b'
SOURCE107 = Path(__file__).with_name('diagnose_privileged_exact_rhs_inverse_ggn.py')
SOURCE107_SHA256 = '396f28b473b599085d6d80edc48693789ae7688da191ef8c25b4a1a8002b3859'
if hashlib.sha256(SOURCE107.read_bytes()).hexdigest() != SOURCE107_SHA256:
    raise ValueError('Frozen Attempt107 source changed')
loader = importlib.util.spec_from_file_location('attempt108_frozen_attempt107', SOURCE107)
a107 = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a107)
a106 = a107.a106
a = a107.a
parent = a107.parent


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt108 spec path changed')
    parent.require_hash(path, SPEC_SHA256)
    spec = a.load_json_object(path, 'Attempt108 spec')
    old = a107.load_spec()
    for key in ('attempt104', 'attempt016', 'attempt106', 'base',
                'final_checkpoint_files', 'model', 'readout', 'eligible_tensors',
                'corpus', 'pilot', 'probe', 'probe_rows', 'displacement', 'ggn',
                'ggn_numerics_106', 'jvp', 'metric', 'target', 'output_policy',
                'exact_rhs'):
        if spec.get(key) != old[key]:
            raise ValueError('Attempt108 differs from frozen Attempt107: '+key)
    old_solver = dict(old['solver'])
    old_solver.update(iterations=15, candidate='x_15')
    old_information = dict(old['information_policy'])
    old_information['inverse_method'] = 'fixed_15_step_fixed_lambda_cg'
    old_baselines = dict(old['baselines'])
    old_baselines.update(attempt107_primary=0.8402237062805622,
                         attempt107_secondary=0.8453507468137178)
    old_workload = dict(old['workload'])
    old_workload.update(cg_ggn_applications=15,
                        expected_wall_minutes_context=[8, 9],
                        additional_operator_applications_vs_107=5,
                        attempt107_operator_seconds_per_application_context=20.7,
                        additional_operator_seconds_context=103.5)
    expected_interpretation = {
        'clear_solver_truncation_effect': {
            'primary_min': 0.90, 'gain_vs_attempt107_min': 0.05},
        'modest_solver_truncation_effect': {
            'primary_min': 0.87, 'gain_vs_attempt107_min': 0.025},
        'otherwise': 'little_evidence_10_step_truncation_dominates',
        'if_little_evidence':
            'next_diagnostic_should_vary_damping_not_add_more_CG_iterations',
        'significance_claim': False, 'candidate_selection': False}
    link = spec.get('attempt107', {})
    if (spec.get('attempt_id') != ATTEMPT or
            spec.get('purpose') !=
                'privileged_exact_rhs_fixed_lambda_15_step_cg_solver_truncation_diagnostic_not_blind_recovery' or
            {key: value for key, value in spec['paths'].items() if key != 'result_path'} !=
                {key: value for key, value in old['paths'].items() if key != 'result_path'} or
            spec['paths']['result_path'] !=
                f'experiments/attempts/{ATTEMPT}/result.json' or
            spec.get('solver') != old_solver or
            spec.get('information_policy') != old_information or
            spec.get('baselines') != old_baselines or
            spec.get('workload') != old_workload or
            spec.get('interpretation') != expected_interpretation or
            link.get('spec_sha256') != a107.SPEC_SHA256 or
            link.get('source_sha256') != SOURCE107_SHA256 or
            link.get('result_sha256') !=
                '2dbc32387559ca29ed7f0ac237ecf89c89b9177f0a8ef5f028c89969200b67c7' or
            link.get('required_primary') != 0.8402237062805622 or
            link.get('required_secondary') != 0.8453507468137178 or
            link.get('required_lambda') != 13.361159773737231 or
            link.get('required_iterations') != 10 or
            link.get('required_relative_linear_residual') != 0.06584746429710282 or
            link.get('required_exact_rhs_norm') != 14.651196958318208 or
            link.get('required_category') !=
                'inverse_or_regularization_limitations_dominate'):
        raise ValueError('Attempt108 frozen scientific inventory mismatch')
    for value in spec['paths'].values():
        parent.resolve(value)
    return spec


def validate_inputs(spec):
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt108 result already exists')
    link = spec['attempt107']
    for key in ('spec', 'source', 'result'):
        parent.require_hash(parent.resolve(link[key+'_path']), link[key+'_sha256'])
    if (link['spec_sha256'] != a107.SPEC_SHA256 or
            link['source_sha256'] != SOURCE107_SHA256 or
            link['result_sha256'] !=
                '2dbc32387559ca29ed7f0ac237ecf89c89b9177f0a8ef5f028c89969200b67c7'):
        raise ValueError('Frozen Attempt107 linkage mismatch')
    result = a.load_json_object(parent.resolve(link['result_path']), 'Attempt107 result')
    solver = result.get('solver', {})
    activation = result.get('matched_activation', {})
    provenance = result.get('provenance', {})
    if (result.get('attempt_id') != a107.ATTEMPT or
            provenance.get('spec_sha256') != link['spec_sha256'] or
            provenance.get('source_sha256') != link['source_sha256'] or
            provenance.get('attempt106_result_sha256') !=
                spec['attempt106']['result_sha256'] or
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
            solver.get('rhs_norm') != link['required_exact_rhs_norm'] or
            result.get('interpretation', {}).get('category') != link['required_category'] or
            result.get('true_delta_rhs') is not True or
            result.get('inverse_solve') is not True or
            result.get('ggn_operator') != 'exact_endpoint_categorical_JtFJ' or
            result.get('ggn_numerical_implementation') !=
                'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast' or
            result.get('inverse_method') != 'fixed_10_step_fixed_lambda_cg' or
            result.get('empirical_fisher') is not False or
            result.get('oracle_adl_access') is not False or
            result.get('adapter_access') is not False):
        raise ValueError('Attempt107 result/provenance/solver mismatch')
    old107 = a107.load_spec(parent.resolve(link['spec_path']))
    proxy = copy.deepcopy(old107)
    proxy['paths']['result_path'] = spec['paths']['result_path']
    return a107.validate_inputs(proxy)


def fixed_15_step_cg(rhs_cpu, eligible, apply_operator, spec, *, report=None):
    solver_spec = spec['solver']
    if (solver_spec['iterations'] != 15 or
            solver_spec['fixed_lambda'] != 13.361159773737231 or
            solver_spec['candidate'] != 'x_15' or
            solver_spec['residual_early_stop'] is not False or
            solver_spec['preconditioner'] is not None):
        raise ValueError('Attempt108 fixed 15-step CG settings changed')
    x, solver = a107.fixed_lambda_cg(
        rhs_cpu, eligible, apply_operator,
        fixed_lambda=solver_spec['fixed_lambda'], iterations=15, report=report)
    if (solver['operator_applications'] != 15 or len(solver['iterations']) != 15 or
            solver['candidate'] != 'x_15' or solver['no_early_stop'] is not True or
            solver['lambda'] != 13.361159773737231):
        raise ValueError('Attempt108 fixed 15-step CG result mismatch')
    solver['method'] = 'fixed_15_step_fixed_lambda_cg'
    return x, solver


def interpretation(primary, spec):
    gain = primary-spec['baselines']['attempt107_primary']
    rules = spec['interpretation']
    if (primary >= rules['clear_solver_truncation_effect']['primary_min'] and
            gain >= rules['clear_solver_truncation_effect']['gain_vs_attempt107_min']):
        return 'clear_solver_truncation_effect'
    if (primary >= rules['modest_solver_truncation_effect']['primary_min'] and
            gain >= rules['modest_solver_truncation_effect']['gain_vs_attempt107_min']):
        return 'modest_solver_truncation_effect'
    return 'little_evidence_10_step_truncation_dominates'


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
    category = interpretation(primary, spec)
    primary_gain = primary-spec['baselines']['attempt107_primary']
    for directory, expected in ((parent.resolve(spec['paths']['base_directory']),
                                 inventories['base_files']),
                                (parent.resolve(spec['paths']['final_directory']),
                                 inventories['final_files'])):
        if a.checkpoint_file_records(directory) != expected:
            raise ValueError('Checkpoint files changed during Attempt108')
    result = {
        'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'privileged_mechanism_diagnostic': True, 'historical_base_access': True,
        'true_delta_rhs': True, 'oracle_adl_access': False, 'adapter_access': False,
        'hybrid_teacher_rhs_recomputed': False,
        'ggn_operator': 'exact_endpoint_categorical_JtFJ', 'inverse_solve': True,
        'inverse_method': 'fixed_15_step_fixed_lambda_cg',
        'fixed_lambda_source': 'frozen_Attempt106_numeric_value',
        'lambda': 13.361159773737231, 'candidate_selection': False,
        'oracle_based_tuning': False, 'empirical_fisher': False,
        'ggn_numerical_implementation':
            'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast',
        'provenance': {
            'spec_sha256': SPEC_SHA256,
            'source_sha256': a.sha256_file(Path(__file__).resolve()),
            'attempt107_spec_sha256': spec['attempt107']['spec_sha256'],
            'attempt107_source_sha256': spec['attempt107']['source_sha256'],
            'attempt107_result_sha256': spec['attempt107']['result_sha256'],
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
            'attempt107_secondary': spec['baselines']['attempt107_secondary'],
            'gain_primary_vs_attempt107': primary_gain,
            'gain_secondary_vs_attempt107':
                secondary-spec['baselines']['attempt107_secondary'],
            'approximate_exact_delta_primary_ceiling':
                spec['baselines']['attempt016_primary_context_approx'],
            'approximate_exact_delta_secondary_ceiling':
                spec['baselines']['attempt016_secondary_context_approx'],
            'ceiling_is_approximate_context_only': True},
        'interpretation': {
            'category': category, 'thresholds': spec['interpretation'],
            'descriptive_compute_decision_only': True,
            'if_little_evidence': spec['interpretation']['if_little_evidence']
            if category == 'little_evidence_10_step_truncation_dominates' else None},
        'timing': {'model_load': loads, 'displacement_seconds': displacement_seconds,
                   'exact_rhs_seconds': rhs_seconds,
                   'solve_seconds': solver['elapsed_seconds']}}
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt108 result already exists')
    a.write_manifest(output, result)
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(run(), allow_nan=False))


if __name__ == '__main__':
    main()
