#!/usr/bin/env python3
"""Attempt 122: four blind Armijo-descending generic-relaxation steps."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '122_loss_stabilized_generic_relaxation'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = '88097c8c56b04df652be8ea166efb3bbc1a8bb9806f3bf8e5bc687b8c1f43d37'
PRIOR_SOURCE = PROJECT / 'scripts/ablation/run_multistep_generic_relaxation.py'
PRIOR_SHA256 = '6a8984c6d3ba22ce09fe664b7df7c1fb6ea18aeb7cd7fded6f663e8714e067c1'
GRID = (1, 2, 4)
STEPS = 4
RECOVERY_STEP = 2
ARMIJO_C = 1e-4
BACKTRACK_FACTOR = 0.5
MAX_TRIALS = 8
CONSISTENCY_ATOL = 1e-4
CONSISTENCY_RTOL = 1e-5
LINE_SEARCH = {
    'method': 'deterministic_Armijo_backtracking_on_full_fixed_1024_row_FineWeb_CE',
    'armijo_c': ARMIJO_C, 'backtrack_factor': BACKTRACK_FACTOR,
    'max_trials_per_step': MAX_TRIALS, 'trial_indices': list(range(MAX_TRIALS)),
    'step1_initial_trial_eta': 'eta_max',
    'later_initial_trial_eta': 'previous_accepted_eta',
    'trial_eta': 'initial_trial_eta*(0.5**trial_index)',
    'acceptance': 'first_trial_with_L_trial <= L_current - armijo_c*eta_trial*aggregate_gradient_norm_squared',
    'reject_restoration': 'exact_pre_step_FP32_eligible_state_verified_by_tensor_equality',
    'trial_objective': 'full_same_1024_row_FineWeb_mean_causal_CE',
    'trial_forward_batch_size': 8, 'trial_forward_batches': 128,
    'loss_recompute_consistency_absolute_tolerance': CONSISTENCY_ATOL,
    'loss_recompute_consistency_relative_tolerance': CONSISTENCY_RTOL,
    'all_trials_rejected': 'fail_blind_construction_without_oracle',
    'accepted_eta_may_increase': False, 'oracle_based_acceptance': False,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def import_pinned(path, expected, name):
    if sha256_file(path) != expected:
        raise ValueError('Frozen helper source SHA256 mismatch: '+str(path))
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


prior = import_pinned(PRIOR_SOURCE, PRIOR_SHA256, 'attempt122_frozen_attempt121_helper')
path_of = prior.path_of
classify = prior.classify
score_report = prior.score_report


def load_spec():
    if sha256_file(SPEC_PATH) != SPEC_SHA256:
        raise ValueError('Attempt122 specification SHA256 mismatch')
    spec = json.loads(SPEC_PATH.read_text())
    old = prior.load_spec()
    record = spec['frozen_attempt121']
    if (record['spec_path'] != str(prior.SPEC_PATH.relative_to(PROJECT)) or
            record['spec_sha256'] != prior.SPEC_SHA256 or
            record['constructor_path'] != str(PRIOR_SOURCE.relative_to(PROJECT)) or
            record['constructor_sha256'] != PRIOR_SHA256):
        raise ValueError('Attempt121 frozen provenance mismatch')
    for key in ('information_policy', 'frozen_attempt005', 'model', 'eligible_tensors',
                'generic_loss', 'probe', 'readout', 'barrier', 'evaluation'):
        if spec[key] != old[key]:
            raise ValueError('Attempt122 changed frozen Attempt121 setup: '+key)
    old_selection = json.loads(json.dumps(old['pilot_selection']))
    old_selection['intended_workload'].update(
        max_line_search_forward_batches=4096,
        line_search_forward_batches_per_trial=128)
    if spec['pilot_selection'] != old_selection:
        raise ValueError('Attempt122 changed frozen pilot selection')
    relax, old_relax = spec['relaxation'], old['relaxation']
    for key in old_relax:
        if key not in ('eta_rule', 'fixed_eta') and relax.get(key) != old_relax[key]:
            raise ValueError('Attempt122 changed non-step-size relaxation setting: '+key)
    if (set(relax) != set(old_relax) | {'line_search'} or
            relax['steps'] != STEPS or relax['functional_checkpoints'] != list(GRID) or
            relax['recovery_step'] != RECOVERY_STEP or
            relax['eta_rule'] != 'eta_max=(0.00125*initial_eligible_weight_norm)/pilot_1024_row_initial_generic_gradient_norm' or
            relax['fixed_eta'] is not False or relax['line_search'] != LINE_SEARCH or
            spec['attempt_id'] != ATTEMPT or
            spec['purpose'] != 'loss_stabilized_four_step_1024_row_generic_relaxation_pilot' or
            spec['paths']['model_dir'] != old['paths']['model_dir']):
        raise ValueError('Attempt122 frozen Armijo plan mismatch')
    for key, value in spec['paths'].items():
        if key != 'result_path' and any(fragment in value.lower() for fragment in
                ('models/base', 'adapter', 'oracle_adl', 'evaluation.json', 'result.json')):
            raise ValueError('Privileged construction path')
    output_paths = {key: path_of(spec['paths'][key]) for key in
                    ('candidate_path', 'recovery_path', 'construction_manifest_path', 'result_path')}
    if len(set(output_paths.values())) != 4 or any(
            output_paths[key] == path_of(old['paths'][key]) for key in output_paths):
        raise ValueError('Attempt122 output paths overlap or reuse Attempt121')
    return spec


def initial_scale_from_gradient(scale):
    """Reuse Attempt121's 0.00125 Frobenius rule; its eta becomes eta_max."""
    frozen = prior.derive_pilot_scale(scale)
    return {'initial_weight_norm': frozen['initial_weight_norm'],
            'pilot_initial_gradient_norm': frozen['pilot_initial_gradient_norm'],
            'target_step_norm': frozen['target_step_norm'],
            'eta_max': frozen['fixed_eta']}


def validate_initial_scale(record):
    if not isinstance(record, dict) or set(record) != {
            'initial_weight_norm', 'pilot_initial_gradient_norm', 'target_step_norm', 'eta_max'}:
        raise ValueError('Malformed Attempt122 initial scale')
    prior.validate_initial_scale({
        'initial_weight_norm': record['initial_weight_norm'],
        'pilot_initial_gradient_norm': record['pilot_initial_gradient_norm'],
        'target_step_norm': record['target_step_norm'],
        'fixed_eta': record['eta_max']})
    return record


def evaluate_generic_loss(model, pilot_tokens, loss_spec, a5, torch):
    """Forward-only CE on all 1024 fixed FineWeb rows; never reads a probe/oracle."""
    if (tuple(pilot_tokens.shape) != (1024, 128) or pilot_tokens.dtype != torch.int64 or
            loss_spec['sample_count'] != 1024 or loss_spec['batch_size'] != 8 or
            loss_spec['number_of_batches'] != 128 or
            loss_spec['total_prediction_tokens'] != 130048):
        raise ValueError('Line-search generic loss inventory mismatch')
    device = next(model.parameters()).device
    model.eval()
    sums = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 1024, 8):
            batch = pilot_tokens[start:start+8].to(device=device)
            logits = model(input_ids=batch, use_cache=False).logits
            loss_sum = a5.causal_token_loss_sum(logits, batch, torch)
            if not bool(torch.isfinite(loss_sum).item()):
                raise ValueError('Nonfinite line-search generic CE')
            sums.append(float(loss_sum.detach().to(device='cpu', dtype=torch.float64).item()))
    mean = math.fsum(sums) / 130048
    if len(sums) != 128 or not math.isfinite(mean):
        raise ValueError('Malformed or nonfinite full-prefix line-search CE')
    return mean


def armijo_search(current_loss, gradient_norm, initial_trial_eta,
                  apply_trial, evaluate_trial_loss, restore_rejected):
    """Accept the first full-prefix Armijo descent; leave accepted weights in place."""
    if (not math.isfinite(current_loss) or not math.isfinite(gradient_norm) or
            gradient_norm < 0 or not math.isfinite(initial_trial_eta) or
            initial_trial_eta <= 0):
        raise ValueError('Malformed Armijo starting state')
    gradient_square = gradient_norm * gradient_norm
    if not math.isfinite(gradient_square):
        raise ValueError('Nonfinite aggregate gradient square')
    trials = []
    for index in range(MAX_TRIALS):
        eta = initial_trial_eta * (BACKTRACK_FACTOR ** index)
        rhs = current_loss - ARMIJO_C * eta * gradient_square
        if not math.isfinite(eta) or eta <= 0 or not math.isfinite(rhs):
            raise ValueError('Nonfinite Armijo trial or RHS')
        apply_trial(eta)
        trial_loss = evaluate_trial_loss()
        if not math.isfinite(trial_loss):
            raise ValueError('Nonfinite line-search generic CE')
        accepted = trial_loss <= rhs
        trials.append({'trial_index': index, 'eta': eta,
                       'generic_mean_ce': trial_loss, 'armijo_rhs': rhs,
                       'accepted': accepted})
        if accepted:
            return {'initial_trial_eta': initial_trial_eta,
                    'trials': trials, 'accepted_trial_index': index,
                    'accepted_eta': eta,
                    'accepted_post_step_generic_mean_ce': trial_loss}
        restore_rejected()
    raise ValueError('All eight predeclared Armijo trials rejected; blind construction stopped')


def restore_pre_step(eligible, snapshot, torch):
    if len(eligible) != 98 or set(snapshot) != {name for name, _ in eligible}:
        raise ValueError('Pre-step eligible state inventory mismatch')
    with torch.no_grad():
        for name, module in eligible:
            original = snapshot[name]
            if (not isinstance(original, torch.Tensor) or original.dtype != torch.float32 or
                    original.device.type != 'cpu' or not original.is_contiguous() or
                    original.shape != module.weight.shape or
                    not bool(torch.isfinite(original).all())):
                raise ValueError('Malformed pre-step FP32 eligible state')
            module.weight.copy_(original.to(module.weight.device))
            actual = module.weight.detach().to(device='cpu', dtype=torch.float32)
            if not torch.equal(actual, original):
                raise ValueError('Rejected trial did not restore exact pre-step eligible state')


def check_recomputed_loss(current, previous_accepted):
    if not math.isfinite(current) or not math.isfinite(previous_accepted):
        raise ValueError('Nonfinite generic CE consistency input')
    gap = abs(current - previous_accepted)
    if not math.isclose(current, previous_accepted,
                        rel_tol=CONSISTENCY_RTOL, abs_tol=CONSISTENCY_ATOL):
        raise ValueError('Next pre-step generic CE differs from accepted post-step CE')
    return gap


def validate_step_records(rows, initial_scale):
    validate_initial_scale(initial_scale)
    if not isinstance(rows, list) or len(rows) > STEPS:
        raise ValueError('Malformed accepted trajectory')
    previous_eta = initial_scale['eta_max']
    previous_loss = None
    for step, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or row.get('step') != step:
            raise ValueError('Accepted trajectory step inventory mismatch')
        loss = row['generic_mean_ce_before_update']
        norm = row['aggregate_gradient_norm_before_update']
        initial = row['initial_trial_eta']
        accepted_eta = row['accepted_eta']
        post = row['accepted_post_step_generic_mean_ce']
        trials = row['trials']
        displacement = row['cumulative_displacement_norm']
        relative_displacement = row['cumulative_displacement_relative_to_initial_weight_norm']
        if (not all(isinstance(value, (int, float)) and math.isfinite(value)
                    for value in (loss, norm, initial, accepted_eta, post,
                                  displacement, relative_displacement)) or
                norm < 0 or initial != previous_eta or not isinstance(trials, list) or
                not 1 <= len(trials) <= MAX_TRIALS or
                row['accepted_trial_index'] != len(trials)-1 or
                accepted_eta != trials[-1]['eta'] or
                displacement < 0 or relative_displacement < 0 or
                not math.isclose(relative_displacement,
                                 displacement / initial_scale['initial_weight_norm'],
                                 rel_tol=1e-12, abs_tol=1e-15)):
            raise ValueError('Malformed accepted Armijo step')
        for index, trial in enumerate(trials):
            eta = initial * (BACKTRACK_FACTOR ** index)
            rhs = loss - ARMIJO_C * eta * (norm * norm)
            if (trial['trial_index'] != index or trial['eta'] != eta or
                    not math.isclose(trial['armijo_rhs'], rhs, rel_tol=0, abs_tol=1e-12) or
                    not math.isfinite(trial['generic_mean_ce']) or
                    trial['accepted'] is not (index == len(trials)-1) or
                    (trial['generic_mean_ce'] <= rhs) is not trial['accepted']):
                raise ValueError('Armijo trial record contradicts first-acceptance rule')
        if post != trials[-1]['generic_mean_ce'] or post > loss or accepted_eta > initial:
            raise ValueError('Accepted step is not descending or eta increased')
        if previous_loss is not None:
            check_recomputed_loss(loss, previous_loss)
        previous_eta, previous_loss = accepted_eta, post
    return True


def save_recovery(path, eligible, rows, checkpoints, responses, final_mean,
                  initial_scale, source_hash, spec_hash, torch):
    if (len(rows) != RECOVERY_STEP or list(responses) != [1, 2] or
            sorted(checkpoints) != [1, 2] or not validate_step_records(rows, initial_scale)):
        raise ValueError('Recovery is fixed after accepted step 2')
    state = prior.original_weights(eligible, torch)
    payload = {'recovery_only': True, 'step': RECOVERY_STEP,
               'previous_accepted_eta': rows[-1]['accepted_eta'],
               'initial_scale': initial_scale, 'source_checkpoint_sha256': source_hash,
               'spec_sha256': spec_hash,
               'state': state,
               'state_aggregate_sha256': checkpoints[RECOVERY_STEP]['state_aggregate_sha256'],
               'rows': rows, 'checkpoints': checkpoints, 'responses': responses,
               'final_mean': final_mean}
    prior.atomic_torch_publish(path, payload, torch)


def restore_recovery(path, eligible, source_hash, spec_hash, helper, torch):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('recovery_only') is not True:
        raise ValueError('Malformed Attempt122 recovery checkpoint')
    rows, checkpoints, responses = (payload.get('rows'), payload.get('checkpoints'),
                                    payload.get('responses'))
    if (payload.get('step') != RECOVERY_STEP or
            payload.get('source_checkpoint_sha256') != source_hash or
            payload.get('spec_sha256') != spec_hash or
            not isinstance(rows, list) or len(rows) != 2 or
            not isinstance(checkpoints, dict) or sorted(checkpoints) != [1, 2] or
            not isinstance(responses, dict) or list(responses) != [1, 2] or
            not validate_step_records(rows, payload.get('initial_scale')) or
            payload.get('previous_accepted_eta') != rows[-1]['accepted_eta']):
        raise ValueError('Attempt122 recovery trajectory/provenance mismatch')
    state = payload.get('state')
    if not isinstance(state, dict) or set(state) != {name for name, _ in eligible}:
        raise ValueError('Recovery eligible-state inventory mismatch')
    restore_pre_step(eligible, state, torch)
    per_matrix, aggregate = prior.state_hashes(eligible, helper, torch)
    if (aggregate != payload['state_aggregate_sha256'] or
            per_matrix != checkpoints[RECOVERY_STEP]['state_per_matrix_sha256']):
        raise ValueError('Recovery eligible-state hashes mismatch')
    for k, response in responses.items():
        prior.validate_response(response, torch)
        if helper.sha256_raw_float32_tensor(response, torch) != checkpoints[k]['response']['raw_sha256']:
            raise ValueError('Recovery response hash mismatch')
    if (not isinstance(payload.get('final_mean'), torch.Tensor) or
            payload['final_mean'].dtype != torch.float64 or
            tuple(payload['final_mean'].shape) != (128, 2048) or
            not bool(torch.isfinite(payload['final_mean']).all())):
        raise ValueError('Malformed recovery final activation mean')
    return payload


def run(*, resume=False):
    import torch
    spec = load_spec()
    prior.refuse_outputs(spec, resume=resume)
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for the under-one-hour Attempt122 pilot')
    old005 = spec['frozen_attempt005']
    a5 = prior.import_pinned(path_of(old005['constructor_path']),
                             old005['constructor_sha256'], 'attempt122_frozen_attempt005')
    a14 = prior.import_pinned(PROJECT/'scripts/ablation/construct_multicorpus_raw_jg_consensus.py',
                              '59936b928dcf942470db92debb1f2a8e41823ae1c9082803d2fdd8dc50202caa',
                              'attempt122_frozen_probe_helper')
    tokens, probe, five_manifest, input_hashes, model_dir = prior.validate_blind_inputs(
        spec, a5, a14, torch)
    pilot_tokens = prior.fixed_prefix(tokens, spec['pilot_selection']['generic_corpus'],
                                      prior.FULL_CORPUS_ROWS, torch)
    model, _ = a5.load_local_model(model_dir, 'cuda', torch)
    eligible = a5.discover_eligible_linear_weights(model, torch)
    selected = a5.freeze_other_parameters(model, eligible)
    frozen_before = a5.frozen_parameter_hashes(model, selected, torch)
    originals = prior.original_weights(eligible, torch)
    source_hash = hashlib.sha256(json.dumps(five_manifest['source_checkpoint']['files'],
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    spec_hash = sha256_file(SPEC_PATH)
    constructor_hash = sha256_file(Path(__file__))
    final_mean = prior.mean_activation(model, probe, spec, a14, torch)
    final_mean_hash = hashlib.sha256(final_mean.numpy().astype('<f8', copy=False).tobytes()).hexdigest()
    rows, checkpoints, responses = [], {}, {}
    initial_scale = None
    previous_accepted_eta = None
    start_step = 1
    recovery_path = path_of(spec['paths']['recovery_path'])
    if resume:
        recovery = restore_recovery(recovery_path, eligible, source_hash, spec_hash, a14, torch)
        if not torch.equal(final_mean, recovery['final_mean']):
            raise ValueError('Recovery final activation mean changed')
        rows, checkpoints, responses = recovery['rows'], recovery['checkpoints'], recovery['responses']
        initial_scale = validate_initial_scale(recovery['initial_scale'])
        previous_accepted_eta = recovery['previous_accepted_eta']
        start_step = RECOVERY_STEP + 1
    for step in prior.trajectory_steps(start_step):
        current_loss = a5.accumulate_mean_generic_gradient(model, pilot_tokens, eligible,
                                                             spec['generic_loss'], torch)
        norm = prior.gradient_norm(eligible, torch)
        consistency_gap = (None if not rows else
                           check_recomputed_loss(current_loss, rows[-1]['accepted_post_step_generic_mean_ce']))
        if step == 1:
            scale = a5.global_rollback_scale(eligible, 0.00125, torch)
            initial_scale = initial_scale_from_gradient(scale)
            if not math.isclose(norm, initial_scale['pilot_initial_gradient_norm'], rel_tol=1e-12):
                raise ValueError('Initial pilot gradient norm accounting mismatch')
            initial_trial_eta = initial_scale['eta_max']
        else:
            initial_trial_eta = previous_accepted_eta
        if initial_trial_eta is None or not math.isfinite(initial_trial_eta):
            raise ValueError('Missing/nonfinite Armijo initial trial eta')
        snapshot = prior.original_weights(eligible, torch)
        def apply_trial(eta):
            prior.update_eligible(eligible, eta, torch)
        def trial_loss():
            return evaluate_generic_loss(model, pilot_tokens, spec['generic_loss'], a5, torch)
        def restore():
            restore_pre_step(eligible, snapshot, torch)
        search = armijo_search(current_loss, norm, initial_trial_eta,
                               apply_trial, trial_loss, restore)
        del snapshot
        previous_accepted_eta = search['accepted_eta']
        model.zero_grad(set_to_none=True)
        a5.verify_frozen_parameters(model, selected, frozen_before, torch)
        displacement = prior.displacement_norm(eligible, originals, torch)
        row = {'step': step,
               'generic_mean_ce_before_update': current_loss,
               'aggregate_gradient_norm_before_update': norm,
               **search,
               'pre_step_loss_consistency_abs_difference': consistency_gap,
               'cumulative_displacement_norm': displacement,
               'cumulative_displacement_relative_to_initial_weight_norm':
                   displacement / initial_scale['initial_weight_norm']}
        rows.append(row)
        validate_step_records(rows, initial_scale)
        print(f"Armijo step {step}/{STEPS} | CE {current_loss:.8g} -> "
              f"{search['accepted_post_step_generic_mean_ce']:.8g} | "
              f"trial {search['accepted_trial_index']}/7 | eta {search['accepted_eta']:.8g}", flush=True)
        if step in GRID:
            mean = prior.mean_activation(model, probe, spec, a14, torch)
            response = prior.response_from_means(final_mean, mean, torch)
            diagnostic = prior.response_diagnostics(response, responses.get(1), a14, torch)
            per_matrix, aggregate = prior.state_hashes(eligible, a14, torch)
            checkpoints[step] = {'step': step,
                'generic_mean_ce_before_update': current_loss,
                'accepted_post_step_generic_mean_ce': search['accepted_post_step_generic_mean_ce'],
                'accepted_eta': search['accepted_eta'],
                'cumulative_displacement_norm': displacement,
                'cumulative_displacement_relative_to_initial_weight_norm': row['cumulative_displacement_relative_to_initial_weight_norm'],
                'state_per_matrix_sha256': per_matrix,
                'state_aggregate_sha256': aggregate,
                'response': diagnostic}
            responses[step] = response
        if step == RECOVERY_STEP and not resume:
            save_recovery(recovery_path, eligible, rows, checkpoints, responses,
                          final_mean, initial_scale, source_hash, spec_hash, torch)
    if (len(rows) != STEPS or list(responses) != list(GRID) or
            sorted(checkpoints) != list(GRID) or not validate_step_records(rows, initial_scale)):
        raise ValueError('Incomplete four-step Armijo trajectory')
    a5.verify_frozen_parameters(model, selected, frozen_before, torch)
    for path, digest in input_hashes.items():
        prior.require_hash(Path(path), digest)
    if a5.checkpoint_file_records(model_dir) != five_manifest['source_checkpoint']['files']:
        raise ValueError('Source model changed during construction')
    prior.require_hash(SPEC_PATH, spec_hash)
    prior.require_hash(Path(__file__), constructor_hash)
    prior.refuse_outputs(spec, resume=True)
    candidate_path = path_of(spec['paths']['candidate_path'])
    manifest_path = path_of(spec['paths']['construction_manifest_path'])
    prior.atomic_torch_publish(candidate_path,
        {'R_1': responses[1], 'R_2': responses[2], 'R_4': responses[4]}, torch)
    manifest = {'format_version': 1, 'attempt_id': ATTEMPT,
        'information_policy': spec['information_policy'],
        'spec_sha256': spec_hash, 'constructor_sha256': constructor_hash,
        'frozen_attempt121_spec_sha256': spec['frozen_attempt121']['spec_sha256'],
        'frozen_attempt121_constructor_sha256': PRIOR_SHA256,
        'frozen_input_sha256': input_hashes,
        'source_checkpoint': five_manifest['source_checkpoint'],
        'pilot_selection': spec['pilot_selection'],
        'generic_loss': spec['generic_loss'],
        'line_search': LINE_SEARCH,
        'final_activation_mean': {'raw_float64_sha256': final_mean_hash,
                                  'shape': [128, 2048], 'dtype': 'contiguous_cpu_float64'},
        'initial_scale': validate_initial_scale(initial_scale),
        'steps': rows, 'checkpoints': {str(k): checkpoints[k] for k in GRID},
        'candidate': {'path': str(candidate_path), 'serialized_sha256': sha256_file(candidate_path),
                      'tensor_order': [f'R_{k}' for k in GRID],
                      'shape': [128, 2048], 'dtype': 'contiguous_cpu_float32'},
        'recovery': {'step': RECOVERY_STEP, 'role': 'interruption_recovery_only',
                     'scientific_candidate': False, 'delete_after_publication': True},
        'same_specimen_exploratory_method_development': True,
        'clean_heldout_validation': False}
    prior.atomic_json_publish(manifest_path, manifest)
    frozen_artifact = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if (list(frozen_artifact) != [f'R_{k}' for k in GRID] or
            sha256_file(candidate_path) != manifest['candidate']['serialized_sha256']):
        raise ValueError('Published response artifact inventory/hash mismatch')
    for k in GRID:
        value = prior.validate_response(frozen_artifact[f'R_{k}'], torch)
        if a14.sha256_raw_float32_tensor(value, torch) != checkpoints[k]['response']['raw_sha256']:
            raise ValueError('Published response raw SHA256 mismatch')
        digest = hashlib.sha256()
        for name, one in checkpoints[k]['state_per_matrix_sha256'].items():
            digest.update(name.encode()+b'\0'+one.encode('ascii')+b'\n')
        if digest.hexdigest() != checkpoints[k]['state_aggregate_sha256']:
            raise ValueError('Published eligible-state aggregate hash mismatch')
    if a5.load_json_object(manifest_path, 'Attempt122 construction manifest') != manifest:
        raise ValueError('Published manifest changed')
    prior.require_hash(SPEC_PATH, spec_hash)
    prior.require_hash(Path(__file__), constructor_hash)
    _, final_aggregate = prior.state_hashes(eligible, a14, torch)
    if final_aggregate != checkpoints[STEPS]['state_aggregate_sha256']:
        raise ValueError('Published final eligible-state hash mismatch')
    recovery_path.unlink()
    print('RELAXATION_CANDIDATES_FROZEN', flush=True)
    result = prior.evaluate_frozen(spec,
        {k: frozen_artifact[f'R_{k}'] for k in GRID}, manifest, candidate_path, torch)
    result['attempt_id'] = ATTEMPT
    result['line_search'] = LINE_SEARCH
    result['oracle_free_trajectory'] = rows
    result['frozen_attempt121_constructor_sha256'] = PRIOR_SHA256
    prior.atomic_json_publish(path_of(spec['paths']['result_path']), result)
    print('Wrote Attempt122 result', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true',
                        help='Resume only from the fixed accepted-step-2 recovery checkpoint')
    args = parser.parse_args(argv)
    try:
        run(resume=args.resume)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
