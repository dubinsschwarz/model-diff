#!/usr/bin/env python3
"""Attempt111: privileged exact-RHS, low-damping 80-step inverse-GGN ceiling."""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '111_privileged_low_damping_long_inverse_ggn_ceiling'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '29af0c15e69209ac196172e92aae8eb7814858953e0b4517577a9112abe90054'
SOURCE110 = Path(__file__).with_name(
    'diagnose_privileged_exact_rhs_inverse_ggn_lambda_div10_cg20.py')
SOURCE110_SHA256 = 'bfb1a233ff108f09ff58a7878d4fe9b5ee23b82851819819efaffbfb9beba6e8'
COMMIT110 = '2678940bd8b09db9721beb83657aeba3b29260c4'
FIXED_LAMBDA = 0.13361159773737232
FIXED_LAMBDA_DECIMAL = '0.13361159773737232'
FIXED_LAMBDA_SOURCE = 'one_tenth_frozen_Attempt110_numeric_lambda'
if hashlib.sha256(SOURCE110.read_bytes()).hexdigest() != SOURCE110_SHA256:
    raise ValueError('Frozen Attempt110 source changed')
loader = importlib.util.spec_from_file_location('attempt111_frozen_attempt110', SOURCE110)
a110 = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a110)
a109 = a110.a109
a108 = a110.a108
a107 = a110.a107
a106 = a110.a106
a = a110.a
parent = a110.parent


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def rhs_semantics_hash(spec):
    return json_hash({key: spec[key] for key in (
        'displacement', 'eligible_tensors', 'pilot', 'exact_rhs',
        'ggn', 'ggn_numerics_106')})


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt111 spec path changed')
    parent.require_hash(path, SPEC_SHA256)
    spec = a.load_json_object(path, 'Attempt111 spec')
    old = a110.load_spec()
    for key in ('attempt104', 'attempt016', 'attempt106', 'attempt107',
                'attempt108', 'attempt109', 'base', 'final_checkpoint_files',
                'model', 'readout', 'eligible_tensors', 'corpus', 'pilot', 'probe',
                'probe_rows', 'displacement', 'ggn', 'ggn_numerics_106',
                'jvp', 'metric', 'target', 'output_policy', 'exact_rhs'):
        if spec.get(key) != old[key]:
            raise ValueError('Attempt111 differs from frozen Attempt110: '+key)
    old_solver = dict(old['solver'])
    old_solver.update(fixed_lambda=FIXED_LAMBDA,
                      fixed_lambda_source=FIXED_LAMBDA_SOURCE,
                      lambda_relative_to_attempt110=0.1,
                      lambda_relative_to_attempt108=0.01,
                      iterations=80, candidate='x_80', midpoint_iteration=40,
                      midpoint_candidate='x_40_descriptive_only',
                      residual_checkpoints=[20, 40, 60, 80])
    old_information = dict(old['information_policy'])
    old_information.update(inverse_method='fixed_80_step_fixed_lambda_cg',
                           fixed_lambda_source=FIXED_LAMBDA_SOURCE)
    old_baselines = dict(old['baselines'])
    old_baselines.update(attempt110_primary=0.8976964227968782,
                         attempt110_secondary=0.9024841265861506)
    expected_workload = {
        'base_probe_batches': 32,
        'exact_rhs_ggn_applications': 1,
        'cg_ggn_applications': 80,
        'ggn_microbatches_per_application': 64,
        'midpoint_jvp_batches': 32,
        'final_jvp_batches': 32,
        'total_jvp_batches': 64,
        'expected_wall_minutes_context': [35, 40],
        'accepted_practical_upper_minutes': 45,
        'no_separate_smoke': True,
        'attempt110_wall_seconds_context': 675,
    }
    expected_interpretation = {
        'near_ceiling_low_damping_recovery': {
            'primary_min': 0.95, 'max_final_relative_linear_residual': 0.02},
        'strong_low_damping_recovery': {
            'primary_min': 0.93, 'gain_vs_attempt110_min': 0.025},
        'still_solver_limited': {
            'min_final_relative_linear_residual_strict': 0.02,
            'gain_80_vs_40_min': 0.01},
        'low_damping_functional_plateau': {
            'max_final_relative_linear_residual': 0.01,
            'gain_80_vs_40_strictly_below': 0.01,
            'primary_strictly_below': 0.93},
        'otherwise': 'mixed_low_damping_result',
        'precedence': ['near_ceiling_low_damping_recovery',
                       'strong_low_damping_recovery', 'still_solver_limited',
                       'low_damping_functional_plateau', 'mixed_low_damping_result'],
        'significance_claim': False, 'candidate_selection': False,
        'fundamental_nullspace_claim': False,
        'plateau_scope': 'this_64_sequence_finite_sample_endpoint_G_and_damping_only'}
    expected_next_step = {
        'execute_next_attempt': False,
        'near_ceiling_low_damping_recovery': 'review_ceiling_result',
        'strong_low_damping_recovery': 'review_low_damping_gain',
        'still_solver_limited': 'review_fixed_solver_convergence',
        'low_damping_functional_plateau':
            'review_this_finite_sample_endpoint_configuration',
        'mixed_low_damping_result': 'review_mixed_result'}
    expected_reproduction = {
        'expected_exact_rhs_norm': 14.651196958318208,
        'rhs_norm_rtol': 1e-6, 'rhs_norm_atol': 1e-5,
        'non_damping_inputs_frozen_to_attempt110': True,
        'compare_attempt110_CG_trajectory': False,
        'activation_metrics_used_before_candidate_freeze': False}
    checkpoint_path = '/root/model-diff-scratch/checkpoints/'
    checkpoint_path += 'attempt111_low_damping_cg80_iter40.pt'
    expected_checkpoint = {
        'format_version': 1, 'iteration': 40, 'path': checkpoint_path,
        'publication': 'temporary_file_fsync_then_atomic_rename',
        'overwrite': False, 'fresh_run_refuses_existing_checkpoint': True,
        'resume_requires_valid_checkpoint': True,
        'resume_starts_at_iteration': 41,
        'normal_run_uses_in_memory_state_after_publication': True,
        'delete_after_successful_result_publication': True,
        'retain_after_failure': True,
        'tensor_names': ['x', 'r', 'p'], 'tensor_dtype': 'torch.float32',
        'tensor_storage': 'CPU_contiguous',
        'tensor_hash': 'per_coordinate_raw_float32_sha256',
        'midpoint_metrics_frozen_in_checkpoint': True,
        'model_weights_in_checkpoint': False, 'rng_state_required': False,
        'expected_approximate_size_gib': 7.9}
    link = spec.get('attempt110', {})
    excluded_paths = {'result_path', 'checkpoint_path'}
    if (spec.get('attempt_id') != ATTEMPT or
            spec.get('purpose') !=
                'privileged_low_damping_long_run_exact_rhs_inverse_GGN_functional_ceiling_not_blind_recovery' or
            {key: value for key, value in spec['paths'].items()
             if key not in excluded_paths} !=
                {key: value for key, value in old['paths'].items()
                 if key != 'result_path'} or
            spec['paths'].get('result_path') !=
                f'experiments/attempts/{ATTEMPT}/result.json' or
            spec['paths'].get('checkpoint_path') != checkpoint_path or
            spec.get('solver') != old_solver or
            spec.get('information_policy') != old_information or
            spec.get('baselines') != old_baselines or
            spec.get('workload') != expected_workload or
            spec.get('interpretation') != expected_interpretation or
            spec.get('next_step_policy') != expected_next_step or
            spec.get('reproduction_audit') != expected_reproduction or
            spec.get('checkpoint') != expected_checkpoint or
            link.get('commit_sha') != COMMIT110 or
            link.get('spec_sha256') != a110.SPEC_SHA256 or
            link.get('source_sha256') != SOURCE110_SHA256 or
            link.get('result_sha256') !=
                'd87150fdf367c0f470c95cc608688f1a618f0c7a69fd63883c872ebb94f9d07d' or
            link.get('required_primary') != 0.8976964227968782 or
            link.get('required_secondary') != 0.9024841265861506 or
            link.get('required_relative_linear_residual') != 0.02529230092308116 or
            link.get('required_rhs_norm') != 14.651196958318208 or
            link.get('required_lambda') != 1.3361159773737232 or
            link.get('required_iterations') != 20 or
            link.get('required_parameter_cosine') != 0.055268426963755675 or
            link.get('required_candidate_to_delta_norm_ratio') !=
                0.034481559169662335 or
            link.get('required_category') !=
                'moderate_lower_damping_convergence_effect'):
        raise ValueError('Attempt111 frozen scientific inventory mismatch')
    for value in spec['paths'].values():
        parent.resolve(value)
    return spec


def verify_committed_attempt110(link):
    if link['commit_sha'] != COMMIT110:
        raise ValueError('Attempt110 committed revision mismatch')
    for key in ('spec', 'source', 'result'):
        path = link[key+'_path']
        try:
            blob = subprocess.run(
                ['git', '-c', f'safe.directory={PROJECT}', 'show',
                 f'{COMMIT110}:{path}'], cwd=PROJECT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError('Attempt110 committed file unavailable: '+path) from exc
        if hashlib.sha256(blob).hexdigest() != link[key+'_sha256']:
            raise ValueError('Attempt110 committed '+key+' hash mismatch')


def validate_inputs(spec, *, resume):
    output = parent.resolve(spec['paths']['result_path'])
    checkpoint = parent.resolve(spec['paths']['checkpoint_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt111 result already exists')
    if resume:
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError('Attempt111 --resume requires valid iteration-40 checkpoint')
    elif checkpoint.exists() or checkpoint.is_symlink():
        raise ValueError('Stale Attempt111 checkpoint: use --resume after validation '
                         'or move the incompatible checkpoint aside')
    link = spec['attempt110']
    for key in ('spec', 'source', 'result'):
        parent.require_hash(parent.resolve(link[key+'_path']), link[key+'_sha256'])
    verify_committed_attempt110(link)
    if (link['spec_sha256'] != a110.SPEC_SHA256 or
            link['source_sha256'] != SOURCE110_SHA256 or
            link['result_sha256'] !=
                'd87150fdf367c0f470c95cc608688f1a618f0c7a69fd63883c872ebb94f9d07d'):
        raise ValueError('Frozen Attempt110 linkage mismatch')
    result = a.load_json_object(parent.resolve(link['result_path']), 'Attempt110 result')
    solver = result.get('solver', {})
    activation = result.get('matched_activation', {})
    provenance = result.get('provenance', {})
    parameters = result.get('parameter_diagnostics', {}).get('global', {})
    if (result.get('attempt_id') != a110.ATTEMPT or
            provenance.get('spec_sha256') != link['spec_sha256'] or
            provenance.get('source_sha256') != link['source_sha256'] or
            provenance.get('attempt109_result_sha256') !=
                spec['attempt109']['result_sha256'] or
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
            result.get('inverse_method') != 'fixed_20_step_fixed_lambda_cg' or
            result.get('empirical_fisher') is not False or
            result.get('oracle_adl_access') is not False or
            result.get('adapter_access') is not False):
        raise ValueError('Attempt110 result/provenance/solver mismatch')
    old110 = a110.load_spec(parent.resolve(link['spec_path']))
    proxy = copy.deepcopy(old110)
    proxy['paths']['result_path'] = spec['paths']['result_path']
    spec016, inventories, _ = a110.validate_inputs(proxy)
    return spec016, inventories, result


def coordinate_inventory(eligible):
    names = a106._names(eligible)
    if len(names) != 98 or names != sorted(names):
        raise ValueError('Attempt111 selected coordinate inventory is not 98 ordered matrices')
    inventory = []
    for name, module in eligible:
        weight = module.weight
        if weight.dtype != torch.float32 or weight.ndim != 2:
            raise ValueError('Attempt111 selected weight dtype/shape mismatch')
        inventory.append({'name': name+'.weight', 'shape': list(weight.shape)})
    return inventory


def vector_hashes(vectors, inventory):
    expected = [item['name'] for item in inventory]
    if set(vectors) != set(expected) or len(vectors) != len(expected):
        raise ValueError('Attempt111 state vector coordinate mismatch')
    hashes = {}
    for item in inventory:
        name = item['name']
        value = vectors[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                list(value.shape) != item['shape'] or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Attempt111 state vector dtype/shape/finite mismatch')
        cpu = value.detach().to('cpu', dtype=torch.float32, copy=True).contiguous()
        hashes[name] = a.sha256_raw_float32_tensor(cpu, torch)
        del cpu
    return hashes


def state_hashes(state, inventory):
    return {key: vector_hashes(state[key], inventory) for key in ('x', 'r', 'p')}


def checkpoint_provenance(spec, inventory):
    return {
        'spec_sha256': SPEC_SHA256,
        'source_sha256': a.sha256_file(Path(__file__).resolve()),
        'attempt110_result_sha256': spec['attempt110']['result_sha256'],
        'final_checkpoint_inventory_sha256': json_hash(spec['final_checkpoint_files']),
        'fineweb_tokens_sha256': spec['corpus']['serialized_sha256'],
        'probe_tokens_sha256': spec['probe']['serialized_sha256'],
        'exact_rhs_semantics_sha256': rhs_semantics_hash(spec),
        'coordinate_inventory_sha256': json_hash(inventory),
        'matched_target_raw_sha256': spec['target']['raw_float32_sha256']}


def validate_midpoint_record(record, spec):
    if (not isinstance(record, dict) or
            set(record) != {'candidate', 'activation', 'parameter_diagnostics',
                            'target_raw_sha256', 'response_raw_sha256', 'jvp_audit'} or
            record['candidate'] != 'x_40_descriptive_only' or
            record['target_raw_sha256'] != spec['target']['raw_float32_sha256'] or
            not isinstance(record['response_raw_sha256'], str) or
            len(record['response_raw_sha256']) != 64 or
            not isinstance(record['activation'], dict) or
            len(record['activation'].get('position_cosines', [])) != 128 or
            not isinstance(record['parameter_diagnostics'], dict) or
            'global' not in record['parameter_diagnostics']):
        raise ValueError('Attempt111 frozen x_40 midpoint metrics malformed')
    for key in ('positions_1_4_mean_cosine', 'positions_1_127_mean_cosine'):
        value = record['activation'].get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Attempt111 frozen x_40 cosine nonfinite/missing')
    json_hash(record)


def make_checkpoint_payload(state, midpoint, inventory, spec):
    if state['iteration'] != 40 or len(state['iterations']) != 40:
        raise ValueError('Attempt111 checkpoint must capture iteration 40')
    validate_midpoint_record(midpoint, spec)
    before = state_hashes(state, inventory)
    cpu_vectors = {}
    for key in ('x', 'r', 'p'):
        cpu_vectors[key] = {
            item['name']: state[key][item['name']].detach().to(
                'cpu', dtype=torch.float32, copy=True).contiguous()
            for item in inventory}
        if vector_hashes(cpu_vectors[key], inventory) != before[key]:
            raise ValueError('Attempt111 checkpoint preparation changed '+key)
    after = state_hashes(state, inventory)
    if after != before:
        raise ValueError('Attempt111 CG state changed during checkpoint preparation')
    return {
        'format_version': 1, 'iteration': 40,
        'x': cpu_vectors['x'], 'r': cpu_vectors['r'], 'p': cpu_vectors['p'],
        'rr': float(state['rr']), 'b_norm': float(state['b_norm']),
        'fixed_lambda': FIXED_LAMBDA,
        'coordinate_inventory': copy.deepcopy(inventory),
        'iteration_records': copy.deepcopy(state['iterations']),
        'midpoint': copy.deepcopy(midpoint),
        'midpoint_json_sha256': json_hash(midpoint),
        'raw_hashes': before,
        'provenance': checkpoint_provenance(spec, inventory)}


def atomic_checkpoint_publish(path, payload):
    if path.exists() or path.is_symlink():
        raise ValueError('Attempt111 checkpoint already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        with temporary.open('xb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise ValueError('Attempt111 checkpoint appeared during publication')
        os.rename(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Attempt111 --resume requires iteration-40 checkpoint')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('Attempt111 checkpoint could not be loaded safely') from exc
    if not isinstance(payload, dict):
        raise ValueError('Attempt111 checkpoint root must be a dictionary')
    return payload


def validate_checkpoint(payload, spec, inventory, expected_rhs_norm):
    required = {'format_version', 'iteration', 'x', 'r', 'p', 'rr', 'b_norm',
                'fixed_lambda', 'coordinate_inventory', 'iteration_records',
                'midpoint', 'midpoint_json_sha256', 'raw_hashes', 'provenance'}
    if (set(payload) != required or payload['format_version'] != 1 or
            payload['iteration'] != 40 or payload['fixed_lambda'] != FIXED_LAMBDA or
            payload['coordinate_inventory'] != inventory or
            payload['provenance'] != checkpoint_provenance(spec, inventory) or
            not isinstance(payload['rr'], (int, float)) or
            not isinstance(payload['b_norm'], (int, float)) or
            not math.isfinite(payload['rr']) or payload['rr'] <= 0 or
            not math.isfinite(payload['b_norm']) or payload['b_norm'] <= 0 or
            not math.isclose(payload['b_norm'], expected_rhs_norm,
                             rel_tol=spec['reproduction_audit']['rhs_norm_rtol'],
                             abs_tol=spec['reproduction_audit']['rhs_norm_atol'])):
        raise ValueError('Attempt111 checkpoint metadata/provenance/RHS mismatch')
    rows = payload['iteration_records']
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError('Attempt111 checkpoint iteration inventory mismatch')
    fields = ('relative_residual', 'r_norm', 'pAp', 'alpha', 'beta',
              'operator_seconds', 'cumulative_elapsed_seconds', 'operator_eta_seconds')
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get('iteration') != index:
            raise ValueError('Attempt111 checkpoint iteration order mismatch')
        if any(not isinstance(row.get(field), (int, float)) or
               not math.isfinite(row[field]) for field in fields):
            raise ValueError('Attempt111 checkpoint iteration scalar mismatch')
    validate_midpoint_record(payload['midpoint'], spec)
    if payload['midpoint_json_sha256'] != json_hash(payload['midpoint']):
        raise ValueError('Attempt111 checkpoint frozen x_40 metric hash mismatch')
    actual_hashes = {}
    for key in ('x', 'r', 'p'):
        values = payload[key]
        if not isinstance(values, dict) or any(
                value.device.type != 'cpu' or not value.is_contiguous()
                for value in values.values() if isinstance(value, torch.Tensor)):
            raise ValueError('Attempt111 checkpoint tensor storage mismatch')
        actual_hashes[key] = vector_hashes(values, inventory)
    if actual_hashes != payload['raw_hashes']:
        raise ValueError('Attempt111 checkpoint x/r/p raw tensor hash mismatch')
    measured_rr = a106.matrixwise_dot(payload['r'], payload['r'])
    if not math.isclose(measured_rr, payload['rr'], rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError('Attempt111 checkpoint residual scalar/vector mismatch')
    return payload['midpoint']


def initial_cg_state(rhs_cpu, eligible):
    names = a106._names(eligible)
    if len(names) != 98 or set(rhs_cpu) != set(names):
        raise ValueError('Attempt111 exact RHS coordinate mismatch')
    r = {}
    for name, module in eligible:
        key = name+'.weight'
        value = rhs_cpu[key]
        if (value.dtype != torch.float32 or value.device.type != 'cpu' or
                value.shape != module.weight.shape or not value.is_contiguous() or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Attempt111 exact RHS matrix malformed')
        r[key] = value.to(module.weight.device, dtype=torch.float32,
                          copy=True).contiguous()
    rhs_cpu.clear()
    p = {name: value.clone() for name, value in r.items()}
    x = {name: torch.zeros_like(value) for name, value in r.items()}
    rr = a106.matrixwise_dot(r, r)
    if not math.isfinite(rr) or rr <= 0:
        raise ValueError('Attempt111 exact RHS norm nonpositive/nonfinite')
    return {'iteration': 0, 'x': x, 'r': r, 'p': p, 'rr': rr,
            'b_norm': math.sqrt(rr), 'iterations': []}


def restore_cg_state(payload, eligible, inventory):
    names = a106._names(eligible)
    if names != [item['name'] for item in inventory]:
        raise ValueError('Attempt111 resume coordinate order mismatch')
    state = {'iteration': 40, 'rr': payload['rr'], 'b_norm': payload['b_norm'],
             'iterations': copy.deepcopy(payload['iteration_records'])}
    for key in ('x', 'r', 'p'):
        state[key] = {
            name+'.weight': payload[key][name+'.weight'].to(
                module.weight.device, dtype=torch.float32, copy=True).contiguous()
            for name, module in eligible}
    if state_hashes(state, inventory) != payload['raw_hashes']:
        raise ValueError('Attempt111 restored GPU CG state differs from checkpoint')
    return state


def run_fixed_80_step_cg(state, eligible, apply_operator, spec, *,
                         on_midpoint=None, report=None):
    solver = spec['solver']
    if (solver['iterations'] != 80 or solver['midpoint_iteration'] != 40 or
            solver['fixed_lambda'] != FIXED_LAMBDA or
            solver['fixed_lambda_source'] != FIXED_LAMBDA_SOURCE or
            solver['lambda_relative_to_attempt110'] != 0.1 or
            solver['lambda_relative_to_attempt108'] != 0.01 or
            solver['initial_x'] != 'zero' or solver['residual_early_stop'] is not False or
            solver['preconditioner'] is not None or
            solver['candidate'] != 'x_80' or
            solver['midpoint_candidate'] != 'x_40_descriptive_only' or
            state['iteration'] not in (0, 40) or
            len(state['iterations']) != state['iteration'] or
            not math.isfinite(state['rr']) or state['rr'] <= 0 or
            not math.isfinite(state['b_norm']) or state['b_norm'] <= 0):
        raise ValueError('Attempt111 fixed 80-step CG state/settings mismatch')
    if state['iteration'] == 0 and on_midpoint is None:
        raise ValueError('Attempt111 fresh CG requires frozen midpoint evaluation')
    names = a106._names(eligible)
    if any(set(state[key]) != set(names) for key in ('x', 'r', 'p')):
        raise ValueError('Attempt111 CG vector support mismatch')
    started = time.perf_counter()
    elapsed_offset = (state['iterations'][-1]['cumulative_elapsed_seconds']
                      if state['iterations'] else 0.0)
    for iteration in range(state['iteration']+1, 81):
        operator_started = time.perf_counter()
        apply_operator(state['p'], iteration)
        operator_seconds = time.perf_counter()-operator_started
        p_gp = a106.matrixwise_grad_dot(eligible, state['p'])
        pp = a106.matrixwise_dot(state['p'], state['p'])
        p_ap = p_gp+FIXED_LAMBDA*pp
        if not math.isfinite(p_ap) or p_ap <= 0:
            raise ValueError('Attempt111 nonpositive/nonfinite CG p^T A p')
        alpha = state['rr']/p_ap
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Attempt111 nonpositive/nonfinite CG alpha')
        for name, module in eligible:
            key = name+'.weight'
            state['x'][key].add_(state['p'][key], alpha=alpha)
            state['r'][key].add_(module.weight.grad, alpha=-alpha)
            state['r'][key].add_(state['p'][key], alpha=-alpha*FIXED_LAMBDA)
        rr_new = a106.matrixwise_dot(state['r'], state['r'])
        if not math.isfinite(rr_new) or rr_new < 0:
            raise ValueError('Attempt111 nonfinite/negative CG residual')
        beta = rr_new/state['rr']
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('Attempt111 nonfinite/negative CG beta')
        for name in names:
            state['p'][name].mul_(beta).add_(state['r'][name])
            if (not bool(torch.isfinite(state['x'][name]).all()) or
                    not bool(torch.isfinite(state['p'][name]).all())):
                raise ValueError('Attempt111 nonfinite CG x/p vector')
        state['rr'] = rr_new
        state['iteration'] = iteration
        elapsed = elapsed_offset+time.perf_counter()-started
        operator_times = [row['operator_seconds'] for row in state['iterations']]
        operator_times.append(operator_seconds)
        row = {'iteration': iteration,
               'relative_residual': math.sqrt(rr_new)/state['b_norm'],
               'r_norm': math.sqrt(rr_new), 'pAp': p_ap,
               'alpha': alpha, 'beta': beta,
               'operator_seconds': operator_seconds,
               'cumulative_elapsed_seconds': elapsed,
               'operator_eta_seconds': math.fsum(operator_times)/len(operator_times)*
                   (80-iteration)}
        state['iterations'].append(row)
        if report is not None:
            report(row)
        if iteration == 40:
            on_midpoint(state)
    return state, {
        'method': 'fixed_80_step_fixed_lambda_cg',
        'iterations': state['iterations'],
        'operator_applications': 80,
        'rhs_norm': state['b_norm'],
        'lambda': FIXED_LAMBDA,
        'fixed_lambda_source': FIXED_LAMBDA_SOURCE,
        'lambda_relative_to_attempt110': 0.1,
        'lambda_relative_to_attempt108': 0.01,
        'candidate': 'x_80',
        'midpoint_candidate': 'x_40_descriptive_only',
        'no_early_stop': True, 'preconditioner': None,
        'relative_linear_residual': state['iterations'][-1]['relative_residual'],
        'residual_source': 'CG_recurrence_no_extra_operator_application',
        'residual_checkpoints': {
            str(index): state['iterations'][index-1]['relative_residual']
            for index in (20, 40, 60, 80)},
        'elapsed_seconds': state['iterations'][-1]['cumulative_elapsed_seconds']}


def evaluate_candidate(final, probe, vector, delta, base_mean, spec, *,
                       candidate, progress=None):
    parameter_geometry = a106.parameter_diagnostics(vector, delta)
    final_mean, response, jvp_audit = parent.final_probe_jvp(
        final, probe, vector, spec, progress=progress)
    target, target_hash = a106.verified_matched_target(final_mean, base_mean, spec)
    activation = parent.response_metrics(response, target)
    return {'candidate': candidate,
            'activation': activation,
            'parameter_diagnostics': parameter_geometry,
            'target_raw_sha256': target_hash,
            'response_raw_sha256': a.sha256_raw_float32_tensor(response, torch),
            'jvp_audit': jvp_audit}


def evaluate_midpoint_preserving_state(state, inventory, final, final_before,
                                       probe, delta, base_mean, spec, *, progress=None):
    before = state_hashes(state, inventory)
    try:
        record = evaluate_candidate(
            final, probe, state['x'], delta, base_mean, spec,
            candidate='x_40_descriptive_only', progress=progress)
    finally:
        final.zero_grad(set_to_none=True)
    a.verify_model_unchanged(final, final_before, torch)
    if state_hashes(state, inventory) != before:
        raise ValueError('Attempt111 x_40 evaluation changed CG x/r/p state')
    validate_midpoint_record(record, spec)
    return record, before


def interpretation(primary80, primary40, residual, spec):
    if not all(math.isfinite(value) for value in (primary80, primary40, residual)):
        raise ValueError('Attempt111 nonfinite interpretation input')
    rules = spec['interpretation']
    gain110 = primary80-spec['baselines']['attempt110_primary']
    gain40 = primary80-primary40
    if (primary80 >= rules['near_ceiling_low_damping_recovery']['primary_min'] and
            residual <= rules['near_ceiling_low_damping_recovery']
            ['max_final_relative_linear_residual']):
        return 'near_ceiling_low_damping_recovery'
    if (primary80 >= rules['strong_low_damping_recovery']['primary_min'] and
            gain110 >= rules['strong_low_damping_recovery']['gain_vs_attempt110_min']):
        return 'strong_low_damping_recovery'
    if (residual > rules['still_solver_limited']
            ['min_final_relative_linear_residual_strict'] and
            gain40 >= rules['still_solver_limited']['gain_80_vs_40_min']):
        return 'still_solver_limited'
    if (residual <= rules['low_damping_functional_plateau']
            ['max_final_relative_linear_residual'] and
            gain40 < rules['low_damping_functional_plateau']
            ['gain_80_vs_40_strictly_below'] and
            primary80 < rules['low_damping_functional_plateau']
            ['primary_strictly_below']):
        return 'low_damping_functional_plateau'
    return 'mixed_low_damping_result'


def publish_result_and_cleanup(output, checkpoint_path, result):
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt111 result already exists')
    if (result.get('solver', {}).get('operator_applications') != 80 or
            len(result.get('solver', {}).get('iterations', [])) != 80 or
            result.get('primary_x80', {}).get('candidate') != 'x_80_primary' or
            result.get('midpoint_x40', {}).get('candidate') !=
                'x_40_descriptive_only'):
        raise ValueError('Attempt111 refuses scientific result before x_80 completes')
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError('Attempt111 iteration-40 checkpoint missing at publication')
    a.write_manifest(output, result)
    checkpoint_path.unlink()


def run(*, resume=False, model_loader=None, tokenizer_loader=None):
    spec = load_spec()
    spec016, inventories, _ = validate_inputs(spec, resume=resume)
    checkpoint_path = parent.resolve(spec['paths']['checkpoint_path'])
    construction, probe = a106.load_frozen_inputs(spec)
    base, final, loads = parent.load_model_pair(
        spec, model_loader=model_loader, tokenizer_loader=tokenizer_loader)
    base_before = a.model_state_hashes(base, torch)
    final_before = a.model_state_hashes(final, torch)
    displacement_started = time.perf_counter()
    delta, displacement_audit, execution_audit, eligible = a106.a105.prepare_displacement(
        base, final, spec016, inventories['base_files'])
    displacement_seconds = time.perf_counter()-displacement_started
    inventory = coordinate_inventory(eligible)
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
    expected_rhs = spec['reproduction_audit']['expected_exact_rhs_norm']
    if not math.isclose(rhs_geometry['norm'], expected_rhs,
                        rel_tol=spec['reproduction_audit']['rhs_norm_rtol'],
                        abs_tol=spec['reproduction_audit']['rhs_norm_atol']):
        raise ValueError(f'Attempt111 exact RHS norm changed: '
                         f"observed={rhs_geometry['norm']}, frozen={expected_rhs}")
    print(f"exact RHS | norm {rhs_geometry['norm']:.9g} verified", flush=True)

    checkpoint_metadata = {}
    if resume:
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        midpoint_record = validate_checkpoint(
            loaded_checkpoint, spec, inventory, rhs_geometry['norm'])
        state = restore_cg_state(loaded_checkpoint, eligible, inventory)
        checkpoint_metadata = {
            'serialized_size_bytes': checkpoint_path.stat().st_size,
            'raw_hashes_sha256': json_hash(loaded_checkpoint['raw_hashes']),
            'resumed_from_iteration': 40}
        rhs_cpu.clear()
        del loaded_checkpoint
        gc.collect()
        print('checkpoint | validated x/r/p and frozen x_40 metrics; resuming at CG 41/80',
              flush=True)
    else:
        state = initial_cg_state(rhs_cpu, eligible)
        if not math.isclose(state['b_norm'], rhs_geometry['norm'],
                            rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError('Attempt111 in-memory exact RHS norm mismatch')
        midpoint_record = None

    def apply_operator(vector, iteration):
        print(f'CG {iteration}/80 | starting exact endpoint GGN application', flush=True)
        first, timing, fisher_audit = a106.ggn_action_stage(
            final, construction, eligible, vector, spec,
            progress=parent.progress_printer(f'CG {iteration}/80 GGN'))
        return {'first_batch': first, 'timings': timing,
                'fisher_conservation': fisher_audit}

    def report(row):
        print(f"CG {row['iteration']}/80 | relative residual "
              f"{row['relative_residual']:.6g} | operator "
              f"{row['operator_seconds']:.1f}s | elapsed "
              f"{row['cumulative_elapsed_seconds']:.1f}s | ETA "
              f"{row['operator_eta_seconds']:.1f}s", flush=True)

    def on_midpoint(current_state):
        nonlocal midpoint_record, checkpoint_metadata
        print('x_40 | evaluating frozen descriptive midpoint on matched probe', flush=True)
        started = time.perf_counter()
        record, before = evaluate_midpoint_preserving_state(
            current_state, inventory, final, final_before, probe, delta,
            base_mean, spec, progress=parent.progress_printer('x_40 block13 JVP'))
        print('x_40 | state unchanged; preparing iteration-40 checkpoint', flush=True)
        payload = make_checkpoint_payload(current_state, record, inventory, spec)
        if state_hashes(current_state, inventory) != before:
            raise ValueError('Attempt111 checkpoint serialization preparation changed CG state')
        atomic_checkpoint_publish(checkpoint_path, payload)
        if state_hashes(current_state, inventory) != before:
            raise ValueError('Attempt111 checkpoint publication changed CG state')
        checkpoint_metadata = {
            'serialized_size_bytes': checkpoint_path.stat().st_size,
            'raw_hashes_sha256': json_hash(payload['raw_hashes']),
            'resumed_from_iteration': None,
            'midpoint_evaluation_seconds': time.perf_counter()-started}
        midpoint_record = record
        del payload
        gc.collect()
        print(f"checkpoint | iteration 40 atomically published at {checkpoint_path}",
              flush=True)

    state, solver = run_fixed_80_step_cg(
        state, eligible, apply_operator, spec,
        on_midpoint=None if resume else on_midpoint, report=report)
    if midpoint_record is None or state['iteration'] != 80:
        raise ValueError('Attempt111 x_80 completed without frozen x_40 midpoint')
    if not math.isclose(solver['rhs_norm'], rhs_geometry['norm'],
                        rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError('Attempt111 exact RHS/CG norm mismatch')
    final.zero_grad(set_to_none=True)
    print('x_80 | evaluating primary frozen candidate on matched probe', flush=True)
    primary_record = evaluate_candidate(
        final, probe, state['x'], delta, base_mean, spec,
        candidate='x_80_primary',
        progress=parent.progress_printer('x_80 block13 JVP'))
    final.zero_grad(set_to_none=True)
    a.verify_model_unchanged(final, final_before, torch)
    primary40 = midpoint_record['activation']['positions_1_4_mean_cosine']
    primary80 = primary_record['activation']['positions_1_4_mean_cosine']
    secondary40 = midpoint_record['activation']['positions_1_127_mean_cosine']
    secondary80 = primary_record['activation']['positions_1_127_mean_cosine']
    category = interpretation(primary80, primary40,
                              solver['relative_linear_residual'], spec)
    baseline = spec['baselines']['attempt110_primary']
    approximate_gap = spec['baselines']['attempt016_primary_context_approx']-baseline
    if approximate_gap <= 0:
        raise ValueError('Attempt111 invalid approximate context ceiling gap')
    for directory, expected in ((parent.resolve(spec['paths']['base_directory']),
                                 inventories['base_files']),
                                (parent.resolve(spec['paths']['final_directory']),
                                 inventories['final_files'])):
        if a.checkpoint_file_records(directory) != expected:
            raise ValueError('Checkpoint files changed during Attempt111')
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise ValueError('Attempt111 iteration-40 checkpoint missing before result publication')
    result = {
        'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'privileged_mechanism_diagnostic': True,
        'historical_base_access': True, 'true_delta_rhs': True,
        'oracle_adl_access': False, 'adapter_access': False,
        'ggn_operator': 'exact_endpoint_categorical_JtFJ',
        'inverse_solve': True, 'inverse_method': 'fixed_80_step_fixed_lambda_cg',
        'fixed_lambda': FIXED_LAMBDA, 'lambda': FIXED_LAMBDA,
        'fixed_lambda_precommitted_decimal': FIXED_LAMBDA_DECIMAL,
        'fixed_lambda_source': FIXED_LAMBDA_SOURCE,
        'lambda_relative_to_attempt110': 0.1,
        'lambda_relative_to_attempt108': 0.01,
        'primary_candidate': 'x_80',
        'midpoint_candidate': 'x_40_descriptive_only',
        'candidate_selection': False, 'oracle_based_tuning': False,
        'empirical_fisher': False, 'finite_sample_ggn_sequences': 64,
        'ggn_numerical_implementation':
            'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast',
        'provenance': {
            'spec_sha256': SPEC_SHA256,
            'source_sha256': a.sha256_file(Path(__file__).resolve()),
            'attempt110_commit_sha': spec['attempt110']['commit_sha'],
            'attempt110_spec_sha256': spec['attempt110']['spec_sha256'],
            'attempt110_source_sha256': spec['attempt110']['source_sha256'],
            'attempt110_result_sha256': spec['attempt110']['result_sha256'],
            'base_checkpoint_files': inventories['base_files'],
            'final_checkpoint_files': inventories['final_files'],
            'fineweb_tokens_sha256': spec['corpus']['serialized_sha256'],
            'probe_tokens_sha256': spec['probe']['serialized_sha256'],
            'matched_target_raw_sha256': spec['target']['raw_float32_sha256'],
            'exact_rhs_semantics_sha256': rhs_semantics_hash(spec)},
        'inventory': {'construction_sequences': 64, 'construction_contexts': 8128,
                      'probe_sequences': 1024, 'eligible_matrices': 98},
        'exact_rhs': {**rhs_geometry, **rhs_audit,
                      'frozen_norm_reproduction_verified': True},
        'displacement_audit': displacement_audit,
        'execution_semantics_audit': execution_audit,
        'solver': solver,
        'midpoint_x40': midpoint_record,
        'primary_x80': primary_record,
        'comparison': {
            'attempt110_primary': baseline,
            'attempt110_secondary': spec['baselines']['attempt110_secondary'],
            'gain_x40_primary_vs_attempt110': primary40-baseline,
            'gain_x80_primary_vs_attempt110': primary80-baseline,
            'gain_x80_primary_vs_x40': primary80-primary40,
            'gain_x40_secondary_vs_attempt110':
                secondary40-spec['baselines']['attempt110_secondary'],
            'gain_x80_secondary_vs_attempt110':
                secondary80-spec['baselines']['attempt110_secondary'],
            'gain_x80_secondary_vs_x40': secondary80-secondary40,
            'approximate_exact_delta_primary_ceiling':
                spec['baselines']['attempt016_primary_context_approx'],
            'approximate_exact_delta_secondary_ceiling':
                spec['baselines']['attempt016_secondary_context_approx'],
            'fraction_of_approximate_remaining_primary_gap_closed_x40':
                (primary40-baseline)/approximate_gap,
            'fraction_of_approximate_remaining_primary_gap_closed_x80':
                (primary80-baseline)/approximate_gap,
            'ceiling_is_approximate_context_only': True},
        'interpretation': {
            'category': category, 'thresholds': spec['interpretation'],
            'mechanistic_compute_decision_only': True,
            'fundamental_nullspace_claim': False,
            'next_step_record_only': spec['next_step_policy'][category],
            'next_attempt_executed': False},
        'checkpoint': {
            **checkpoint_metadata, 'path': str(checkpoint_path),
            'iteration': 40, 'midpoint_metrics_frozen': True,
            'deleted_after_successful_result_publication': True,
            'large_tensor_artifact_persistent_after_success': False},
        'timing': {'model_load': loads, 'displacement_seconds': displacement_seconds,
                   'exact_rhs_seconds': rhs_seconds,
                   'solve_compute_seconds': solver['elapsed_seconds']}}
    output = parent.resolve(spec['paths']['result_path'])
    publish_result_and_cleanup(output, checkpoint_path, result)
    print(f'Attempt111 result published; iteration-40 checkpoint deleted: {output}',
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true',
                        help='validate iteration-40 checkpoint and continue at CG 41')
    args = parser.parse_args()
    print(json.dumps(run(resume=args.resume), allow_nan=False))


if __name__ == '__main__':
    main()
