#!/usr/bin/env python3
"""Attempt112: exact-RHS very-low-damping, fixed 100-step inverse-GGN ceiling."""
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
ATTEMPT = '112_privileged_very_low_damping_inverse_ggn_ceiling'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = 'ec18d8b492bac738e264ed54122f6376db11ea35912e89f1b758bcc66fff9a35'
SOURCE111 = Path(__file__).with_name('diagnose_privileged_low_damping_long_inverse_ggn.py')
SOURCE111_SHA256 = '6489b7780bfb05fef122f1ed051ea4e0bc3be36832d17f1e3859269e512156c7'
COMMIT111 = '4c3869b3e14d069fd28bc8521a4b7673679fd209'
RESULT111_SHA256 = '0291c107ba2eaf746d4f7ebdd6ea49a7cc97c6498886b9a6534ae2bef80ac147'
FIXED_LAMBDA = 0.013361159773737232
FIXED_LAMBDA_DECIMAL = '0.013361159773737232'
FIXED_LAMBDA_SOURCE = 'one_tenth_frozen_Attempt111_precommitted_decimal_lambda'
CHECKPOINT_ITERATION = 50
CG_ITERATIONS = 100
if hashlib.sha256(SOURCE111.read_bytes()).hexdigest() != SOURCE111_SHA256:
    raise ValueError('Frozen Attempt111 source changed')
loader = importlib.util.spec_from_file_location('attempt112_frozen_attempt111', SOURCE111)
a111 = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a111)
a110, a107, a106, a, parent = (a111.a110, a111.a107, a111.a106,
                               a111.a, a111.parent)


def json_hash(value):
    return a111.json_hash(value)


def expected_spec():
    """Freeze the complete Attempt112 inventory against committed Attempt111."""
    old = a111.load_spec()
    expected = copy.deepcopy(old)
    expected['attempt_id'] = ATTEMPT
    expected['purpose'] = (
        'privileged_very_low_damping_exact_rhs_inverse_GGN_functional_ceiling_not_blind_recovery')
    expected['paths']['result_path'] = f'experiments/attempts/{ATTEMPT}/result.json'
    checkpoint_path = ('/root/model-diff-scratch/checkpoints/'
                       'attempt112_very_low_damping_cg100_iter50.pt')
    expected['paths']['checkpoint_path'] = checkpoint_path
    expected['attempt111'] = {
        'commit_sha': COMMIT111,
        'spec_path': f'experiments/attempts/{a111.ATTEMPT}/spec.json',
        'spec_sha256': a111.SPEC_SHA256,
        'source_path': 'scripts/ablation/diagnose_privileged_low_damping_long_inverse_ggn.py',
        'source_sha256': SOURCE111_SHA256,
        'result_path': f'experiments/attempts/{a111.ATTEMPT}/result.json',
        'result_sha256': RESULT111_SHA256,
        'required_primary': 0.9582115833559394,
        'required_secondary': 0.9506416718708435,
        'required_relative_linear_residual': 0.004786514368786419,
        'required_rhs_norm': 14.651196958318208,
        'required_lambda': 0.13361159773737233,
        'required_lambda_precommitted_decimal': '0.13361159773737232',
        'required_iterations': 80,
        'required_category': 'near_ceiling_low_damping_recovery'}
    expected['solver'].update(
        iterations=CG_ITERATIONS, fixed_lambda=FIXED_LAMBDA,
        fixed_lambda_precommitted_decimal=FIXED_LAMBDA_DECIMAL,
        fixed_lambda_source=FIXED_LAMBDA_SOURCE,
        lambda_relative_to_attempt111=0.1,
        lambda_relative_to_attempt110=0.01,
        lambda_relative_to_attempt108=0.001,
        candidate='x_100', checkpoint_iteration=CHECKPOINT_ITERATION,
        residual_checkpoints=[20, 40, 50, 60, 80, 100])
    expected['solver'].pop('midpoint_iteration')
    expected['solver'].pop('midpoint_candidate')
    expected['baselines'].update(
        attempt111_primary=0.9582115833559394,
        attempt111_secondary=0.9506416718708435)
    expected['interpretation'] = {
        'very_near_ceiling_recovery': {
            'primary_min': 0.97, 'max_final_relative_linear_residual': 0.02},
        'clear_additional_low_curvature_recovery': {'gain_vs_attempt111_min': 0.008},
        'still_solver_limited': {'min_final_relative_linear_residual_strict': 0.02},
        'very_low_damping_plateau': {
            'max_final_relative_linear_residual': 0.01,
            'gain_vs_attempt111_strictly_below': 0.005},
        'otherwise': 'mixed_very_low_damping_result',
        'precedence': ['very_near_ceiling_recovery',
                       'clear_additional_low_curvature_recovery',
                       'still_solver_limited', 'very_low_damping_plateau',
                       'mixed_very_low_damping_result'],
        'significance_claim': False, 'candidate_selection': False,
        'fundamental_nullspace_claim': False,
        'plateau_scope': 'this_64_sequence_finite_sample_endpoint_G_and_damping_only'}
    expected.pop('next_step_policy')
    expected['information_policy'].update(
        inverse_method='fixed_100_step_fixed_lambda_cg',
        fixed_lambda_source=FIXED_LAMBDA_SOURCE)
    expected['workload'] = {
        'base_probe_batches': 32, 'exact_rhs_ggn_applications': 1,
        'cg_ggn_applications': 100, 'ggn_microbatches_per_application': 64,
        'final_jvp_batches': 32, 'total_jvp_batches': 32,
        'expected_wall_minutes_context': [40, 44],
        'accepted_practical_upper_minutes': 45, 'no_separate_smoke': True,
        'attempt111_wall_seconds_context': 2071}
    expected['reproduction_audit'] = {
        'expected_exact_rhs_norm': 14.651196958318208,
        'rhs_norm_rtol': 1e-6, 'rhs_norm_atol': 1e-5,
        'non_damping_inputs_frozen_to_attempt111': True,
        'compare_attempt111_CG_trajectory': False,
        'activation_metrics_used_before_candidate_freeze': False}
    expected['checkpoint'].update(
        iteration=CHECKPOINT_ITERATION, path=checkpoint_path,
        resume_starts_at_iteration=51, activation_metrics_in_checkpoint=False)
    expected['checkpoint'].pop('midpoint_metrics_frozen_in_checkpoint')
    return expected


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt112 spec path changed')
    parent.require_hash(path, SPEC_SHA256)
    spec = a.load_json_object(path, 'Attempt112 spec')
    if spec != expected_spec():
        raise ValueError('Attempt112 frozen scientific inventory mismatch')
    for value in spec['paths'].values():
        parent.resolve(value)
    return spec


def verify_committed_attempt111(link):
    if link['commit_sha'] != COMMIT111:
        raise ValueError('Attempt111 committed revision mismatch')
    for key in ('spec', 'source', 'result'):
        path = link[key+'_path']
        try:
            blob = subprocess.run(
                ['git', '-c', f'safe.directory={PROJECT}', 'show',
                 f'{COMMIT111}:{path}'], cwd=PROJECT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError('Attempt111 committed file unavailable: '+path) from exc
        if hashlib.sha256(blob).hexdigest() != link[key+'_sha256']:
            raise ValueError('Attempt111 committed '+key+' hash mismatch')


def validate_inputs(spec, *, resume):
    output = parent.resolve(spec['paths']['result_path'])
    checkpoint = parent.resolve(spec['paths']['checkpoint_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt112 result already exists')
    if resume:
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError('Attempt112 --resume requires valid iteration-50 checkpoint')
    elif checkpoint.exists() or checkpoint.is_symlink():
        raise ValueError('Stale Attempt112 checkpoint: use --resume after validation '
                         'or move the incompatible checkpoint aside')
    link = spec['attempt111']
    for key in ('spec', 'source', 'result'):
        parent.require_hash(parent.resolve(link[key+'_path']), link[key+'_sha256'])
    verify_committed_attempt111(link)
    result = a.load_json_object(parent.resolve(link['result_path']), 'Attempt111 result')
    solver = result.get('solver', {})
    primary = result.get('primary_x80', {}).get('activation', {})
    provenance = result.get('provenance', {})
    if (result.get('attempt_id') != a111.ATTEMPT or
            provenance.get('spec_sha256') != link['spec_sha256'] or
            provenance.get('source_sha256') != link['source_sha256'] or
            provenance.get('attempt110_result_sha256') !=
                spec['attempt110']['result_sha256'] or
            provenance.get('base_checkpoint_files') != spec['base']['files'] or
            provenance.get('final_checkpoint_files') != spec['final_checkpoint_files'] or
            provenance.get('fineweb_tokens_sha256') != spec['corpus']['serialized_sha256'] or
            provenance.get('probe_tokens_sha256') != spec['probe']['serialized_sha256'] or
            provenance.get('matched_target_raw_sha256') != spec['target']['raw_float32_sha256'] or
            provenance.get('exact_rhs_semantics_sha256') != a111.rhs_semantics_hash(spec) or
            primary.get('positions_1_4_mean_cosine') != link['required_primary'] or
            primary.get('positions_1_127_mean_cosine') != link['required_secondary'] or
            solver.get('relative_linear_residual') !=
                link['required_relative_linear_residual'] or
            solver.get('rhs_norm') != link['required_rhs_norm'] or
            result.get('exact_rhs', {}).get('norm') != link['required_rhs_norm'] or
            solver.get('lambda') != link['required_lambda'] or
            result.get('fixed_lambda') != link['required_lambda'] or
            result.get('fixed_lambda_precommitted_decimal') !=
                link['required_lambda_precommitted_decimal'] or
            solver.get('operator_applications') != link['required_iterations'] or
            len(solver.get('iterations', [])) != link['required_iterations'] or
            result.get('interpretation', {}).get('category') != link['required_category'] or
            result.get('true_delta_rhs') is not True or
            result.get('inverse_solve') is not True or
            result.get('ggn_operator') != 'exact_endpoint_categorical_JtFJ' or
            result.get('ggn_numerical_implementation') !=
                'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast' or
            result.get('inverse_method') != 'fixed_80_step_fixed_lambda_cg' or
            result.get('empirical_fisher') is not False or
            result.get('oracle_adl_access') is not False or
            result.get('adapter_access') is not False):
        raise ValueError('Attempt111 result/provenance/solver mismatch')
    proxy = copy.deepcopy(a110.load_spec())
    proxy['paths']['result_path'] = spec['paths']['result_path']
    spec016, inventories, _ = a110.validate_inputs(proxy)
    return spec016, inventories, result


def checkpoint_provenance(spec, inventory):
    return {
        'spec_sha256': SPEC_SHA256,
        'source_sha256': a.sha256_file(Path(__file__).resolve()),
        'attempt111_commit_sha': COMMIT111,
        'attempt111_spec_sha256': a111.SPEC_SHA256,
        'attempt111_source_sha256': SOURCE111_SHA256,
        'attempt111_result_sha256': RESULT111_SHA256,
        'final_checkpoint_inventory_sha256': json_hash(spec['final_checkpoint_files']),
        'fineweb_tokens_sha256': spec['corpus']['serialized_sha256'],
        'probe_tokens_sha256': spec['probe']['serialized_sha256'],
        'exact_rhs_semantics_sha256': a111.rhs_semantics_hash(spec),
        'coordinate_inventory_sha256': json_hash(inventory),
        'matched_target_raw_sha256': spec['target']['raw_float32_sha256']}


def make_checkpoint_payload(state, inventory, spec):
    if state['iteration'] != CHECKPOINT_ITERATION or len(state['iterations']) != 50:
        raise ValueError('Attempt112 checkpoint must capture iteration 50')
    before = a111.state_hashes(state, inventory)
    cpu_vectors = {}
    for key in ('x', 'r', 'p'):
        cpu_vectors[key] = {
            item['name']: state[key][item['name']].detach().to(
                'cpu', dtype=torch.float32, copy=True).contiguous()
            for item in inventory}
        if a111.vector_hashes(cpu_vectors[key], inventory) != before[key]:
            raise ValueError('Attempt112 checkpoint preparation changed '+key)
    if a111.state_hashes(state, inventory) != before:
        raise ValueError('Attempt112 CG state changed during checkpoint preparation')
    return {
        'format_version': 1, 'iteration': CHECKPOINT_ITERATION,
        'x': cpu_vectors['x'], 'r': cpu_vectors['r'], 'p': cpu_vectors['p'],
        'rr': float(state['rr']), 'b_norm': float(state['b_norm']),
        'fixed_lambda': FIXED_LAMBDA,
        'coordinate_inventory': copy.deepcopy(inventory),
        'iteration_records': copy.deepcopy(state['iterations']),
        'raw_hashes': before,
        'provenance': checkpoint_provenance(spec, inventory)}


def atomic_checkpoint_publish(path, payload):
    # Same fsync-and-rename publication as Attempt111; reject collisions.
    a111.atomic_checkpoint_publish(path, payload)


def load_checkpoint(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Attempt112 --resume requires iteration-50 checkpoint')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('Attempt112 checkpoint could not be loaded safely') from exc
    if not isinstance(payload, dict):
        raise ValueError('Attempt112 checkpoint root must be a dictionary')
    return payload


def validate_checkpoint(payload, spec, inventory, expected_rhs_norm):
    required = {'format_version', 'iteration', 'x', 'r', 'p', 'rr', 'b_norm',
                'fixed_lambda', 'coordinate_inventory', 'iteration_records',
                'raw_hashes', 'provenance'}
    if (set(payload) != required or payload['format_version'] != 1 or
            payload['iteration'] != CHECKPOINT_ITERATION or
            payload['fixed_lambda'] != FIXED_LAMBDA or
            payload['coordinate_inventory'] != inventory or
            payload['provenance'] != checkpoint_provenance(spec, inventory) or
            not isinstance(payload['rr'], (int, float)) or
            not isinstance(payload['b_norm'], (int, float)) or
            not math.isfinite(payload['rr']) or payload['rr'] <= 0 or
            not math.isfinite(payload['b_norm']) or payload['b_norm'] <= 0 or
            not math.isclose(payload['b_norm'], expected_rhs_norm,
                             rel_tol=spec['reproduction_audit']['rhs_norm_rtol'],
                             abs_tol=spec['reproduction_audit']['rhs_norm_atol'])):
        raise ValueError('Attempt112 checkpoint metadata/provenance/RHS mismatch')
    rows = payload['iteration_records']
    if not isinstance(rows, list) or len(rows) != CHECKPOINT_ITERATION:
        raise ValueError('Attempt112 checkpoint iteration inventory mismatch')
    fields = ('relative_residual', 'r_norm', 'pAp', 'alpha', 'beta',
              'operator_seconds', 'cumulative_elapsed_seconds', 'operator_eta_seconds')
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or row.get('iteration') != index:
            raise ValueError('Attempt112 checkpoint iteration order mismatch')
        if any(not isinstance(row.get(field), (int, float)) or
               not math.isfinite(row[field]) for field in fields):
            raise ValueError('Attempt112 checkpoint iteration scalar mismatch')
    actual_hashes = {}
    for key in ('x', 'r', 'p'):
        values = payload[key]
        if not isinstance(values, dict) or any(
                value.device.type != 'cpu' or not value.is_contiguous()
                for value in values.values() if isinstance(value, torch.Tensor)):
            raise ValueError('Attempt112 checkpoint tensor storage mismatch')
        actual_hashes[key] = a111.vector_hashes(values, inventory)
    if actual_hashes != payload['raw_hashes']:
        raise ValueError('Attempt112 checkpoint x/r/p raw tensor hash mismatch')
    measured_rr = a106.matrixwise_dot(payload['r'], payload['r'])
    if not math.isclose(measured_rr, payload['rr'], rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError('Attempt112 checkpoint residual scalar/vector mismatch')


def restore_cg_state(payload, eligible, inventory):
    names = a106._names(eligible)
    if names != [item['name'] for item in inventory]:
        raise ValueError('Attempt112 resume coordinate order mismatch')
    state = {'iteration': CHECKPOINT_ITERATION, 'rr': payload['rr'],
             'b_norm': payload['b_norm'],
             'iterations': copy.deepcopy(payload['iteration_records'])}
    for key in ('x', 'r', 'p'):
        state[key] = {
            name+'.weight': payload[key][name+'.weight'].to(
                module.weight.device, dtype=torch.float32, copy=True).contiguous()
            for name, module in eligible}
    if a111.state_hashes(state, inventory) != payload['raw_hashes']:
        raise ValueError('Attempt112 restored CG state differs from checkpoint')
    return state


def run_fixed_100_step_cg(state, eligible, apply_operator, spec, *,
                          on_checkpoint=None, report=None):
    solver = spec['solver']
    if (solver['iterations'] != CG_ITERATIONS or
            solver['checkpoint_iteration'] != CHECKPOINT_ITERATION or
            solver['fixed_lambda'] != FIXED_LAMBDA or
            solver['fixed_lambda_precommitted_decimal'] != FIXED_LAMBDA_DECIMAL or
            solver['fixed_lambda_source'] != FIXED_LAMBDA_SOURCE or
            solver['lambda_relative_to_attempt111'] != 0.1 or
            solver['lambda_relative_to_attempt110'] != 0.01 or
            solver['lambda_relative_to_attempt108'] != 0.001 or
            solver['initial_x'] != 'zero' or
            solver['residual_early_stop'] is not False or
            solver['preconditioner'] is not None or
            solver['candidate'] != 'x_100' or
            state['iteration'] not in (0, CHECKPOINT_ITERATION) or
            len(state['iterations']) != state['iteration'] or
            not math.isfinite(state['rr']) or state['rr'] <= 0 or
            not math.isfinite(state['b_norm']) or state['b_norm'] <= 0):
        raise ValueError('Attempt112 fixed 100-step CG state/settings mismatch')
    if state['iteration'] == 0 and on_checkpoint is None:
        raise ValueError('Attempt112 fresh CG requires iteration-50 checkpoint callback')
    names = a106._names(eligible)
    if any(set(state[key]) != set(names) for key in ('x', 'r', 'p')):
        raise ValueError('Attempt112 CG vector support mismatch')
    started = time.perf_counter()
    elapsed_offset = (state['iterations'][-1]['cumulative_elapsed_seconds']
                      if state['iterations'] else 0.0)
    for iteration in range(state['iteration']+1, CG_ITERATIONS+1):
        operator_started = time.perf_counter()
        apply_operator(state['p'], iteration)
        operator_seconds = time.perf_counter()-operator_started
        p_gp = a106.matrixwise_grad_dot(eligible, state['p'])
        pp = a106.matrixwise_dot(state['p'], state['p'])
        p_ap = p_gp+FIXED_LAMBDA*pp
        if not math.isfinite(p_ap) or p_ap <= 0:
            raise ValueError('Attempt112 nonpositive/nonfinite CG p^T A p')
        alpha = state['rr']/p_ap
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Attempt112 nonpositive/nonfinite CG alpha')
        for name, module in eligible:
            key = name+'.weight'
            state['x'][key].add_(state['p'][key], alpha=alpha)
            state['r'][key].add_(module.weight.grad, alpha=-alpha)
            state['r'][key].add_(state['p'][key], alpha=-alpha*FIXED_LAMBDA)
        rr_new = a106.matrixwise_dot(state['r'], state['r'])
        if not math.isfinite(rr_new) or rr_new < 0:
            raise ValueError('Attempt112 nonfinite/negative CG residual')
        beta = rr_new/state['rr']
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('Attempt112 nonfinite/negative CG beta')
        for name in names:
            state['p'][name].mul_(beta).add_(state['r'][name])
            if (not bool(torch.isfinite(state['x'][name]).all()) or
                    not bool(torch.isfinite(state['r'][name]).all()) or
                    not bool(torch.isfinite(state['p'][name]).all())):
                raise ValueError('Attempt112 nonfinite CG x/r/p vector')
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
                   (CG_ITERATIONS-iteration)}
        state['iterations'].append(row)
        if report is not None:
            report(row)
        if iteration == CHECKPOINT_ITERATION:
            on_checkpoint(state)
    return state, {
        'method': 'fixed_100_step_fixed_lambda_cg',
        'iterations': state['iterations'],
        'operator_applications': CG_ITERATIONS,
        'rhs_norm': state['b_norm'],
        'lambda': FIXED_LAMBDA,
        'fixed_lambda_source': FIXED_LAMBDA_SOURCE,
        'lambda_relative_to_attempt111': 0.1,
        'lambda_relative_to_attempt110': 0.01,
        'lambda_relative_to_attempt108': 0.001,
        'candidate': 'x_100',
        'no_early_stop': True, 'preconditioner': None,
        'relative_linear_residual': state['iterations'][-1]['relative_residual'],
        'residual_source': 'CG_recurrence_no_extra_operator_application',
        'residual_checkpoints': {
            str(index): state['iterations'][index-1]['relative_residual']
            for index in (20, 40, 50, 60, 80, 100)},
        'elapsed_seconds': state['iterations'][-1]['cumulative_elapsed_seconds']}


def interpretation(primary, residual, spec):
    if not math.isfinite(primary) or not math.isfinite(residual):
        raise ValueError('Attempt112 nonfinite interpretation input')
    rules = spec['interpretation']
    gain = primary-spec['baselines']['attempt111_primary']
    if (primary >= rules['very_near_ceiling_recovery']['primary_min'] and
            residual <= rules['very_near_ceiling_recovery']
            ['max_final_relative_linear_residual']):
        return 'very_near_ceiling_recovery'
    if gain >= rules['clear_additional_low_curvature_recovery']['gain_vs_attempt111_min']:
        return 'clear_additional_low_curvature_recovery'
    if residual > rules['still_solver_limited']['min_final_relative_linear_residual_strict']:
        return 'still_solver_limited'
    if (residual <= rules['very_low_damping_plateau']
            ['max_final_relative_linear_residual'] and
            gain < rules['very_low_damping_plateau']
            ['gain_vs_attempt111_strictly_below']):
        return 'very_low_damping_plateau'
    return 'mixed_very_low_damping_result'


def publish_result_and_cleanup(output, checkpoint_path, result):
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt112 result already exists')
    if (result.get('solver', {}).get('operator_applications') != CG_ITERATIONS or
            len(result.get('solver', {}).get('iterations', [])) != CG_ITERATIONS or
            result.get('primary_x100', {}).get('candidate') != 'x_100_primary'):
        raise ValueError('Attempt112 refuses scientific result before x_100 completes')
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError('Attempt112 iteration-50 checkpoint missing at publication')
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
    inventory = a111.coordinate_inventory(eligible)
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
        raise ValueError(f'Attempt112 exact RHS norm changed: '
                         f"observed={rhs_geometry['norm']}, frozen={expected_rhs}")
    print(f"exact RHS | norm {rhs_geometry['norm']:.9g} verified", flush=True)

    checkpoint_metadata = {}
    if resume:
        loaded_checkpoint = load_checkpoint(checkpoint_path)
        validate_checkpoint(loaded_checkpoint, spec, inventory, rhs_geometry['norm'])
        state = restore_cg_state(loaded_checkpoint, eligible, inventory)
        checkpoint_metadata = {
            'serialized_size_bytes': checkpoint_path.stat().st_size,
            'raw_hashes_sha256': json_hash(loaded_checkpoint['raw_hashes']),
            'resumed_from_iteration': CHECKPOINT_ITERATION}
        rhs_cpu.clear()
        del loaded_checkpoint
        gc.collect()
        print('checkpoint | validated x/r/p; resuming at CG 51/100', flush=True)
    else:
        state = a111.initial_cg_state(rhs_cpu, eligible)
        if not math.isclose(state['b_norm'], rhs_geometry['norm'],
                            rel_tol=1e-7, abs_tol=1e-8):
            raise ValueError('Attempt112 in-memory exact RHS norm mismatch')

    def apply_operator(vector, iteration):
        print(f'CG {iteration}/100 | starting exact endpoint GGN application', flush=True)
        a106.ggn_action_stage(
            final, construction, eligible, vector, spec,
            progress=parent.progress_printer(f'CG {iteration}/100 GGN'))

    def report(row):
        print(f"CG {row['iteration']}/100 | relative residual "
              f"{row['relative_residual']:.6g} | operator "
              f"{row['operator_seconds']:.1f}s | elapsed "
              f"{row['cumulative_elapsed_seconds']:.1f}s | ETA "
              f"{row['operator_eta_seconds']:.1f}s", flush=True)

    def on_checkpoint(current_state):
        nonlocal checkpoint_metadata
        print('checkpoint | preparing iteration-50 CG state', flush=True)
        started = time.perf_counter()
        before = a111.state_hashes(current_state, inventory)
        payload = make_checkpoint_payload(current_state, inventory, spec)
        if a111.state_hashes(current_state, inventory) != before:
            raise ValueError('Attempt112 checkpoint preparation changed CG state')
        atomic_checkpoint_publish(checkpoint_path, payload)
        if a111.state_hashes(current_state, inventory) != before:
            raise ValueError('Attempt112 checkpoint publication changed CG state')
        checkpoint_metadata = {
            'serialized_size_bytes': checkpoint_path.stat().st_size,
            'raw_hashes_sha256': json_hash(payload['raw_hashes']),
            'resumed_from_iteration': None,
            'checkpoint_publication_seconds': time.perf_counter()-started}
        del payload
        gc.collect()
        print(f'checkpoint | iteration 50 atomically published at {checkpoint_path}',
              flush=True)

    state, solver = run_fixed_100_step_cg(
        state, eligible, apply_operator, spec,
        on_checkpoint=None if resume else on_checkpoint, report=report)
    if state['iteration'] != CG_ITERATIONS:
        raise ValueError('Attempt112 primary candidate x_100 incomplete')
    if not math.isclose(solver['rhs_norm'], rhs_geometry['norm'],
                        rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError('Attempt112 exact RHS/CG norm mismatch')
    final.zero_grad(set_to_none=True)
    print('x_100 | evaluating frozen primary candidate on matched probe', flush=True)
    primary_record = a111.evaluate_candidate(
        final, probe, state['x'], delta, base_mean, spec,
        candidate='x_100_primary',
        progress=parent.progress_printer('x_100 block13 JVP'))
    final.zero_grad(set_to_none=True)
    a.verify_model_unchanged(final, final_before, torch)
    primary = primary_record['activation']['positions_1_4_mean_cosine']
    secondary = primary_record['activation']['positions_1_127_mean_cosine']
    category = interpretation(primary, solver['relative_linear_residual'], spec)
    baseline = spec['baselines']['attempt111_primary']
    approximate_gap = spec['baselines']['attempt016_primary_context_approx']-baseline
    if approximate_gap <= 0:
        raise ValueError('Attempt112 invalid approximate context ceiling gap')
    for directory, expected in ((parent.resolve(spec['paths']['base_directory']),
                                 inventories['base_files']),
                                (parent.resolve(spec['paths']['final_directory']),
                                 inventories['final_files'])):
        if a.checkpoint_file_records(directory) != expected:
            raise ValueError('Checkpoint files changed during Attempt112')
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise ValueError('Attempt112 iteration-50 checkpoint missing before publication')
    result = {
        'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'privileged_mechanism_diagnostic': True,
        'historical_base_access': True, 'true_delta_rhs': True,
        'oracle_adl_access': False, 'adapter_access': False,
        'ggn_operator': 'exact_endpoint_categorical_JtFJ',
        'inverse_solve': True, 'inverse_method': 'fixed_100_step_fixed_lambda_cg',
        'fixed_lambda': FIXED_LAMBDA, 'lambda': FIXED_LAMBDA,
        'fixed_lambda_precommitted_decimal': FIXED_LAMBDA_DECIMAL,
        'fixed_lambda_source': FIXED_LAMBDA_SOURCE,
        'lambda_relative_to_attempt111': 0.1,
        'lambda_relative_to_attempt110': 0.01,
        'lambda_relative_to_attempt108': 0.001,
        'primary_candidate': 'x_100',
        'candidate_selection': False, 'oracle_based_tuning': False,
        'empirical_fisher': False, 'finite_sample_ggn_sequences': 64,
        'ggn_numerical_implementation':
            'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast',
        'provenance': {
            'spec_sha256': SPEC_SHA256,
            'source_sha256': a.sha256_file(Path(__file__).resolve()),
            'attempt111_commit_sha': COMMIT111,
            'attempt111_spec_sha256': a111.SPEC_SHA256,
            'attempt111_source_sha256': SOURCE111_SHA256,
            'attempt111_result_sha256': RESULT111_SHA256,
            'base_checkpoint_files': inventories['base_files'],
            'final_checkpoint_files': inventories['final_files'],
            'fineweb_tokens_sha256': spec['corpus']['serialized_sha256'],
            'probe_tokens_sha256': spec['probe']['serialized_sha256'],
            'matched_target_raw_sha256': spec['target']['raw_float32_sha256'],
            'exact_rhs_semantics_sha256': a111.rhs_semantics_hash(spec)},
        'inventory': {'construction_sequences': 64, 'construction_contexts': 8128,
                      'probe_sequences': 1024, 'eligible_matrices': 98},
        'exact_rhs': {**rhs_geometry, **rhs_audit,
                      'frozen_norm_reproduction_verified': True},
        'displacement_audit': displacement_audit,
        'execution_semantics_audit': execution_audit,
        'solver': solver, 'primary_x100': primary_record,
        'comparison': {
            'attempt111_primary': baseline,
            'attempt111_secondary': spec['baselines']['attempt111_secondary'],
            'gain_primary_vs_attempt111': primary-baseline,
            'gain_secondary_vs_attempt111':
                secondary-spec['baselines']['attempt111_secondary'],
            'approximate_exact_delta_primary_ceiling':
                spec['baselines']['attempt016_primary_context_approx'],
            'approximate_exact_delta_secondary_ceiling':
                spec['baselines']['attempt016_secondary_context_approx'],
            'fraction_of_approximate_remaining_primary_gap_closed':
                (primary-baseline)/approximate_gap,
            'ceiling_is_approximate_context_only': True},
        'interpretation': {
            'category': category, 'thresholds': spec['interpretation'],
            'mechanistic_compute_decision_only': True,
            'fundamental_nullspace_claim': False},
        'checkpoint': {
            **checkpoint_metadata, 'path': str(checkpoint_path),
            'iteration': CHECKPOINT_ITERATION,
            'activation_metrics_in_checkpoint': False,
            'deleted_after_successful_result_publication': True,
            'large_tensor_artifact_persistent_after_success': False},
        'timing': {'model_load': loads, 'displacement_seconds': displacement_seconds,
                   'exact_rhs_seconds': rhs_seconds,
                   'solve_compute_seconds': solver['elapsed_seconds']}}
    output = parent.resolve(spec['paths']['result_path'])
    publish_result_and_cleanup(output, checkpoint_path, result)
    print(f'Attempt112 result published; iteration-50 checkpoint deleted: {output}',
          flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true',
                        help='validate iteration-50 checkpoint and continue at CG 51')
    args = parser.parse_args()
    print(json.dumps(run(resume=args.resume), allow_nan=False))


if __name__ == '__main__':
    main()
