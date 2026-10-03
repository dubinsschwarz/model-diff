#!/usr/bin/env python3
"""Attempt 121: freeze a four-step 1024-row relaxation pilot before oracle scoring."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '121_multistep_generic_relaxation'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = 'b49b06a5bf79142e33cb5a329564fbc5b55eac6ba89f570c01c79d6c19c975f4'
GRID = (1, 2, 4)
STEPS = 4
RECOVERY_STEP = 2
PILOT_ROWS = 1024
FULL_CORPUS_ROWS = 4096
FULL_PROBE_ROWS = 10000
EXPECTED = {'weight_norm': 918.1587250300843,
            'gradient_norm': 11.072586477802595,
            'target_norm': 1.1476984062876054,
            'eta': 0.10365224137905052}


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f'Frozen SHA256 mismatch: {path}')
    return path


def import_pinned(path, expected, name):
    require_hash(path, expected)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def validate_plan(spec):
    relax = spec['relaxation']
    selection = spec['pilot_selection']
    generic = selection['generic_corpus']
    activation = selection['activation_probe']
    if (spec['attempt_id'] != ATTEMPT or relax['steps'] != STEPS or
            spec['purpose'] != 'runtime_reduced_1024_row_fixed_prefix_four_step_generic_relaxation_pilot' or
            relax['functional_checkpoints'] != list(GRID) or
            relax['recovery_step'] != RECOVERY_STEP or
            relax['initial_relative_frobenius_step'] != 0.00125 or
            relax['eta_rule'] != 'eta=(0.00125*initial_eligible_weight_norm)/pilot_1024_row_initial_generic_gradient_norm' or
            relax['fixed_eta'] is not True or relax['later_gradient_normalization'] is not False or
            relax['update'] != 'W <- W - eta * full_generic_gradient' or
            relax['update_arithmetic_dtype'] != 'cpu_float64' or
            relax['stored_weight_dtype'] != 'float32' or
            relax['response'] != 'A_final - A_k' or
            any(relax[key] is not False for key in
                ('optimizer', 'momentum', 'weight_decay', 'gradient_clipping',
                 'early_stopping', 'response_normalization', 'response_rescaling')) or
            spec['eligible_tensors']['expected_matrix_count'] != 98 or
            spec['generic_loss']['sample_count'] != PILOT_ROWS or
            spec['generic_loss']['batch_size'] != 8 or
            spec['generic_loss']['number_of_batches'] != 128 or
            spec['generic_loss']['total_prediction_tokens'] != 130048 or
            spec['probe']['sample_count'] != FULL_PROBE_ROWS or
            spec['probe']['batch_size'] != 32 or
            generic['rows'] != [0, PILOT_ROWS] or generic['sample_count'] != PILOT_ROWS or
            generic['interval'] != '[0,1024)' or
            generic['source_artifact'] != 'full_frozen_attempt005_fineweb_4096_rows' or
            generic['selection'] != 'deterministic_contiguous_prefix_reused_at_every_step' or
            activation['rows'] != [0, PILOT_ROWS] or activation['sample_count'] != PILOT_ROWS or
            activation['batch_size'] != 32 or activation['number_of_batches_per_mean'] != 32 or
            activation['interval'] != '[0,1024)' or
            activation['source_artifact'] != 'full_frozen_generic_oracle_probe_10000_rows' or
            activation['selection'] != 'deterministic_contiguous_prefix_reused_for_A_final_and_every_A_k' or
            selection['underlying_artifacts_fully_hash_validated'] is not True or
            selection['intended_workload'] != {
                'gradient_passes': 4, 'gradient_batches_per_pass': 128,
                'total_backward_batches': 512, 'activation_means': 4,
                'probe_batches_per_mean': 32, 'total_probe_forward_batches': 128,
                'target_wall_seconds_less_than': 3600,
                'runtime_device': 'cuda_required'} or
            spec['readout']['hidden_state_index'] != 14 or
            spec['barrier']['marker'] != 'RELAXATION_CANDIDATES_FROZEN' or
            spec['barrier']['all_4_steps_and_3_responses_published_and_revalidated_before_oracle'] is not True or
            spec['barrier']['recovery_checkpoint_scientific_candidate'] is not False or
            spec['evaluation']['primary_positions'] != [1, 2, 3, 4] or
            spec['evaluation']['secondary_positions'] != list(range(1, 128)) or
            spec['evaluation']['all_positions'] != list(range(128)) or
            spec['evaluation']['clear_thresholds'] != {'primary_gain': .03, 'secondary_gain': .02} or
            spec['evaluation']['moderate_thresholds'] != {'primary_gain': .015, 'secondary_gain': 0.0}):
        raise ValueError('Attempt121 frozen scientific plan mismatch')
    for key in ('weight_norm', 'gradient_norm', 'target_norm', 'eta'):
        if spec['frozen_attempt005']['initial_'+{'weight_norm':'source_weight_norm',
                'gradient_norm':'gradient_norm', 'target_norm':'target_step_norm',
                'eta':'alpha'}[key]] != EXPECTED[key]:
            raise ValueError('Frozen Attempt005 numeric control mismatch')
    for key, value in spec['paths'].items():
        if key != 'result_path' and any(x in value.lower() for x in
                ('models/base', 'adapter', 'oracle_adl', 'evaluation.json', 'result.json')):
            raise ValueError('Privileged construction path')
    return spec


def load_spec():
    require_hash(SPEC_PATH, SPEC_SHA256)
    return validate_plan(json.loads(SPEC_PATH.read_text()))


def refuse_outputs(spec, resume=False):
    paths = [path_of(spec['paths'][key]) for key in
             ('candidate_path', 'construction_manifest_path', 'result_path')]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise ValueError(f'Attempt121 output already exists: {path}')
    recovery = path_of(spec['paths']['recovery_path'])
    if resume:
        if recovery.is_symlink() or not recovery.is_file():
            raise ValueError('Step-2 recovery checkpoint required for --resume')
    elif recovery.exists() or recovery.is_symlink():
        raise ValueError('Stale step-2 recovery checkpoint; use --resume')
    if len({str(path.resolve()) for path in paths+[recovery]}) != 4:
        raise ValueError('Output paths overlap')


def fixed_eta(weight_norm, gradient_norm):
    if not all(math.isfinite(x) and x > 0 for x in (weight_norm, gradient_norm)):
        raise ValueError('Invalid initial norm')
    target = 0.00125 * weight_norm
    eta = target / gradient_norm
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError('Invalid eta')
    return target, eta


def trajectory_steps(start_step=1):
    if start_step not in (1, RECOVERY_STEP + 1):
        raise ValueError('Only the full trajectory or fixed step-2 resume is allowed')
    return range(start_step, STEPS + 1)


def validate_initial_scale(record):
    if not isinstance(record, dict) or set(record) != {'initial_weight_norm', 'pilot_initial_gradient_norm',
                       'target_step_norm', 'fixed_eta'}:
        raise ValueError('Malformed pilot initial scale')
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or
           not math.isfinite(value) for value in record.values()):
        raise ValueError('Nonfinite pilot initial scale')
    weight_norm = record['initial_weight_norm']
    if not math.isclose(weight_norm, EXPECTED['weight_norm'], rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError('Initial eligible-weight norm differs from Attempt005')
    target, eta = fixed_eta(weight_norm, record['pilot_initial_gradient_norm'])
    if target != record['target_step_norm'] or eta != record['fixed_eta']:
        raise ValueError('Pilot eta does not follow its initial gradient')
    return record


def derive_pilot_scale(scale):
    return validate_initial_scale({
        'initial_weight_norm': scale['aggregate_source_weight_norm'],
        'pilot_initial_gradient_norm': scale['aggregate_generic_gradient_norm'],
        'target_step_norm': scale['target_delta_norm'],
        'fixed_eta': scale['alpha']})


def fixed_prefix(tensor, selection, full_rows, torch):
    """Consume the same prefix after the complete frozen artifact is validated."""
    if (tensor.dtype != torch.int64 or tuple(tensor.shape) != (full_rows, 128) or
            not tensor.is_contiguous() or selection['rows'] != [0, PILOT_ROWS] or
            selection['sample_count'] != PILOT_ROWS):
        raise ValueError('Malformed full artifact or fixed pilot prefix')
    prefix = tensor[0:PILOT_ROWS].contiguous()
    if tuple(prefix.shape) != (PILOT_ROWS, 128):
        raise ValueError('Pilot prefix shape mismatch')
    return prefix


def probe_batch_starts(spec):
    selection = spec['pilot_selection']['activation_probe']
    if (selection['sample_count'] != PILOT_ROWS or selection['batch_size'] != 32 or
            selection['number_of_batches_per_mean'] != 32):
        raise ValueError('Pilot activation batch inventory mismatch')
    return range(0, PILOT_ROWS, selection['batch_size'])


def validate_response(tensor, torch):
    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or
            tensor.device.type != 'cpu' or not tensor.is_contiguous() or
            tuple(tensor.shape) != (128, 2048) or
            not bool(torch.isfinite(tensor).all())):
        raise ValueError('Malformed/nonfinite functional response')
    return tensor


def response_from_means(final_mean, current_mean, torch):
    for value in (final_mean, current_mean):
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64 or
                value.device.type != 'cpu' or tuple(value.shape) != (128, 2048) or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Malformed/nonfinite activation mean')
    return validate_response((final_mean - current_mean).to(torch.float32).contiguous(), torch)


def mean_activation(model, probe, spec, helper, torch):
    settings = spec['pilot_selection']['activation_probe']
    prefix = fixed_prefix(probe, settings, FULL_PROBE_ROWS, torch)
    device = next(model.parameters()).device
    total = torch.zeros((128, 2048), dtype=torch.float64, device='cpu')
    model.eval()
    with torch.inference_mode():
        for start in probe_batch_starts(spec):
            batch = prefix[start:start+settings['batch_size']].to(device)
            readout = helper.ordinary_hook_readout(model, batch, spec, torch)
            total.add_(readout.to(device='cpu', dtype=torch.float64).sum(dim=0))
    mean = (total / PILOT_ROWS).contiguous()
    if not bool(torch.isfinite(mean).all()):
        raise ValueError('Nonfinite activation mean')
    return mean


def original_weights(eligible, torch):
    return {name: module.weight.detach().to(device='cpu', dtype=torch.float32,
                                            copy=True).contiguous()
            for name, module in eligible}


def gradient_norm(eligible, torch):
    squares = []
    for name, module in eligible:
        grad = module.weight.grad
        if (grad is None or grad.dtype != torch.float32 or
                grad.shape != module.weight.shape or not bool(torch.isfinite(grad).all())):
            raise ValueError(f'Malformed/nonfinite generic gradient: {name}')
        squares.append(float(grad.detach().to('cpu', dtype=torch.float64).square().sum()))
    norm = math.sqrt(math.fsum(squares))
    if not math.isfinite(norm):
        raise ValueError('Nonfinite generic gradient norm')
    return norm


def update_eligible(eligible, eta, torch):
    """Exactly the Attempt005 CPU float64 subtract and FP32 storage convention."""
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError('Invalid fixed eta')
    with torch.no_grad():
        for name, module in eligible:
            weight = module.weight
            gradient = weight.grad
            if (gradient is None or not bool(torch.isfinite(gradient).all()) or
                    gradient.shape != weight.shape):
                raise ValueError(f'Malformed generic gradient: {name}')
            source64 = weight.detach().to('cpu', dtype=torch.float64)
            gradient64 = gradient.detach().to('cpu', dtype=torch.float64)
            updated = (source64 - eta * gradient64).to(torch.float32).contiguous()
            if not bool(torch.isfinite(updated).all()):
                raise ValueError(f'Nonfinite updated weight: {name}')
            weight.copy_(updated.to(weight.device))


def displacement_norm(eligible, originals, torch):
    squares = []
    for name, module in eligible:
        delta = originals[name].to(torch.float64) - module.weight.detach().to('cpu', dtype=torch.float64)
        squares.append(float(delta.square().sum()))
    value = math.sqrt(math.fsum(squares))
    if not math.isfinite(value):
        raise ValueError('Nonfinite cumulative displacement')
    return value


def state_hashes(eligible, helper, torch):
    per_matrix = {}
    digest = hashlib.sha256()
    for name, module in eligible:
        value = module.weight.detach().to('cpu', dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'Nonfinite state: {name}')
        one = helper.sha256_raw_float32_tensor(value, torch)
        per_matrix[name+'.weight'] = one
        digest.update((name+'.weight').encode()+b'\0'+one.encode('ascii')+b'\n')
    if len(per_matrix) != 98:
        raise ValueError('Expected exactly 98 state hashes')
    return per_matrix, digest.hexdigest()


def response_diagnostics(response, first, helper, torch):
    validate_response(response, torch)
    norm = float(torch.linalg.vector_norm(response.double()))
    if not math.isfinite(norm):
        raise ValueError('Nonfinite functional response norm')
    reference = response if first is None else first
    reference_norm = float(torch.linalg.vector_norm(reference.double()))
    cosine = (None if norm == 0 or reference_norm == 0 else
              float(torch.dot(response.double().flatten(), reference.double().flatten()) /
                    (norm * reference_norm)))
    return {'raw_sha256': helper.sha256_raw_float32_tensor(response, torch),
            'shape': [128, 2048], 'dtype': 'contiguous_cpu_float32',
            'norm': norm, 'cosine_vs_R_1': cosine}


def atomic_torch_publish(path, payload, torch):
    if path.exists() or path.is_symlink():
        raise ValueError(f'Output already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        with temporary.open('xb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_publish(path, payload):
    if path.exists() or path.is_symlink():
        raise ValueError(f'Output already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        with temporary.open('x') as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_blind_inputs(spec, a5, a14, torch):
    frozen = spec['frozen_attempt005']
    if (spec['readout'] != a14.FROZEN_SPEC['readout'] or
            {key: spec['probe'][key] for key in a14.FROZEN_SPEC['probe']} !=
            a14.FROZEN_SPEC['probe']):
        raise ValueError('Frozen generic probe/readout conventions changed')
    input_paths = {}
    for key in ('spec', 'construction_manifest', 'corpus_spec', 'corpus_manifest',
                'corpus_artifact', 'constructor'):
        path = path_of(frozen[key+'_path'])
        require_hash(path, frozen[key+'_sha256'])
        input_paths[key] = path
    five_spec = a5.load_spec(input_paths['spec'])
    corpus_spec = a5.load_corpus_spec(input_paths['corpus_spec'])
    five_manifest = a5.load_json_object(input_paths['construction_manifest'], 'Attempt005 construction manifest')
    expected_pilot_loss = dict(five_spec['generic_loss'])
    expected_pilot_loss.update(sample_count=PILOT_ROWS, number_of_batches=128,
                               total_prediction_tokens=130048)
    if (five_spec['model'] != spec['model'] or
            five_spec['eligible_tensors'] != spec['eligible_tensors'] or
            expected_pilot_loss != spec['generic_loss'] or
            five_manifest['gradient_and_rollback']['aggregate_source_weight_norm'] != EXPECTED['weight_norm'] or
            five_manifest['gradient_and_rollback']['aggregate_generic_gradient_norm'] != EXPECTED['gradient_norm'] or
            five_manifest['gradient_and_rollback']['target_delta_norm'] != EXPECTED['target_norm'] or
            five_manifest['gradient_and_rollback']['alpha'] != EXPECTED['eta']):
        raise ValueError('Frozen Attempt005 control contradiction')
    model_dir = path_of(spec['paths']['model_dir'])
    if a5.checkpoint_file_records(model_dir) != five_manifest['source_checkpoint']['files']:
        raise ValueError('Final checkpoint differs from Attempt005 source')
    corpus = a5.verify_corpus_manifest(input_paths['corpus_manifest'],
        input_paths['corpus_spec'], input_paths['corpus_artifact'], model_dir, corpus_spec)
    tokens = a5.load_corpus_tokens(input_paths['corpus_artifact'], corpus['raw_tensor_sha256'],
                                    FULL_CORPUS_ROWS, 128, torch)
    probe_path = path_of(spec['probe']['path'])
    probe_manifest = path_of(spec['probe']['manifest_path'])
    require_hash(probe_manifest, spec['probe']['manifest_sha256'])
    probe_record = a5.load_json_object(probe_manifest, 'frozen probe manifest')
    if (probe_record.get('fineweb_tokens_sha256') != spec['probe']['serialized_sha256'] or
            probe_record.get('raw_tensor_sha256') != spec['probe']['raw_tensor_sha256']):
        raise ValueError('Frozen probe manifest mismatch')
    probe = a14.load_probe(probe_path, spec['probe'], torch)
    hashes = {str(path): sha256_file(path) for path in
              (*input_paths.values(), probe_path, probe_manifest)}
    probe_helper = PROJECT/'scripts/ablation/construct_multicorpus_raw_jg_consensus.py'
    hashes[str(probe_helper)] = sha256_file(probe_helper)
    return tokens, probe, five_manifest, hashes, model_dir


def save_recovery(path, eligible, rows, checkpoints, responses, final_mean, initial_scale,
                  source_hash, spec_hash, helper, torch):
    if len(rows) != RECOVERY_STEP or list(responses) != [1, 2]:
        raise ValueError('Recovery is fixed at step 2')
    validate_initial_scale(initial_scale)
    state = {name+'.weight': module.weight.detach().to('cpu', dtype=torch.float32,
                                                      copy=True).contiguous()
             for name, module in eligible}
    payload = {'recovery_only': True, 'step': RECOVERY_STEP,
               'eta': initial_scale['fixed_eta'], 'initial_scale': initial_scale,
               'source_checkpoint_sha256': source_hash, 'spec_sha256': spec_hash,
               'state': state, 'state_aggregate_sha256': checkpoints[RECOVERY_STEP]['state_aggregate_sha256'],
               'rows': rows, 'checkpoints': checkpoints, 'responses': responses,
               'final_mean': final_mean}
    atomic_torch_publish(path, payload, torch)


def restore_recovery(path, eligible, source_hash, spec_hash, helper, torch):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Malformed step-2 recovery checkpoint')
    rows = payload.get('rows')
    checkpoints = payload.get('checkpoints')
    responses = payload.get('responses')
    if (not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows) or
            not isinstance(checkpoints, dict) or not isinstance(responses, dict)):
        raise ValueError('Malformed step-2 recovery records')
    if (payload.get('recovery_only') is not True or
            payload.get('step') != RECOVERY_STEP or payload.get('source_checkpoint_sha256') != source_hash or
            payload.get('spec_sha256') != spec_hash or len(rows) != RECOVERY_STEP or
            [row.get('step') for row in rows] != [1, 2] or
            list(responses) != [1, 2] or sorted(checkpoints) != [1, 2]):
        raise ValueError('Malformed step-2 recovery checkpoint')
    state = payload['state']
    if set(state) != {name+'.weight' for name, _ in eligible}:
        raise ValueError('Recovery matrix inventory mismatch')
    for name, module in eligible:
        value = state[name+'.weight']
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                value.device.type != 'cpu' or not value.is_contiguous() or
                value.shape != module.weight.shape or not bool(torch.isfinite(value).all())):
            raise ValueError('Malformed/nonfinite recovery matrix')
        with torch.no_grad():
            module.weight.copy_(value.to(module.weight.device))
    per_matrix, aggregate = state_hashes(eligible, helper, torch)
    if (aggregate != payload['state_aggregate_sha256'] or
            per_matrix != payload['checkpoints'][RECOVERY_STEP]['state_per_matrix_sha256']):
        raise ValueError('Recovery matrix hash mismatch')
    for k, response in payload['responses'].items():
        validate_response(response, torch)
        if helper.sha256_raw_float32_tensor(response, torch) != payload['checkpoints'][k]['response']['raw_sha256']:
            raise ValueError('Recovery response hash mismatch')
    if (not isinstance(payload['final_mean'], torch.Tensor) or
            payload['final_mean'].dtype != torch.float64 or
            not bool(torch.isfinite(payload['final_mean']).all())):
        raise ValueError('Malformed recovery final mean')
    if payload['eta'] != validate_initial_scale(payload['initial_scale'])['fixed_eta']:
        raise ValueError('Recovery eta mismatch')
    return payload


def score_report(response, oracle, validator):
    values = validator.position_cosines(response, oracle)
    mean = lambda xs: None if any(x is None for x in xs) else math.fsum(xs)/len(xs)
    return {'position_0_cosine': values[0], 'positions_1_4_individual_cosines': values[1:5],
            'positions_1_4_mean_cosine': mean(values[1:5]),
            'positions_1_127_mean_cosine': mean(values[1:]),
            'all_128_position_cosines': values}


def classify(reports):
    if list(reports) != list(GRID):
        raise ValueError('Evaluation may score only the three frozen checkpoints')
    def rank(k):
        primary, secondary = (reports[k]['positions_1_4_mean_cosine'],
                              reports[k]['positions_1_127_mean_cosine'])
        return (-math.inf if primary is None else primary,
                -math.inf if secondary is None else secondary, -k)
    best = max((2, 4), key=rank)
    primary_pair = (reports[best]['positions_1_4_mean_cosine'],
                    reports[1]['positions_1_4_mean_cosine'])
    secondary_pair = (reports[best]['positions_1_127_mean_cosine'],
                      reports[1]['positions_1_127_mean_cosine'])
    primary_gain = None if None in primary_pair else primary_pair[0] - primary_pair[1]
    secondary_gain = None if None in secondary_pair else secondary_pair[0] - secondary_pair[1]
    category = ('clear_multistep_support' if primary_gain is not None and secondary_gain is not None and primary_gain >= .03 and secondary_gain >= .02 else
                'moderate_multistep_support' if primary_gain is not None and secondary_gain is not None and primary_gain >= .015 and secondary_gain >= 0 else
                'no_meaningful_multistep_support')
    return {'best_multistep_k': best, 'primary_gain_vs_k1': primary_gain,
            'secondary_gain_vs_k1': secondary_gain, 'category': category,
            'selection': 'descriptive_same_specimen_development_set'}


def evaluate_frozen(spec, responses, manifest, candidate_path, torch):
    """This function is called only after the construction barrier has printed."""
    evaluation = spec['evaluation']
    validator = import_pinned(path_of(evaluation['validator_path']),
                              evaluation['validator_sha256'], 'attempt121_oracle_validator')
    paths = {key: path_of(value['path']) for key, value in evaluation['provenance_inputs'].items()}
    hashes = {key: value['sha256'] for key, value in evaluation['provenance_inputs'].items()}
    for key in paths:
        require_hash(paths[key], hashes[key])
    attempt = validator.load_json_object(paths['attempt_spec'], 'Attempt014 spec')
    validator.validate_construction(paths, hashes, validator.FROZEN_SPEC)
    oracle_manifest = validator.validate_oracle_provenance(paths, hashes, attempt)
    if (oracle_manifest['oracle_adl_sha256'] != evaluation['oracle_artifact_sha256'] or
            oracle_manifest['raw_tensors_sha256']['difference'] != evaluation['oracle_difference_raw_sha256']):
        raise ValueError('Historical oracle provenance mismatch')
    oracle_path = path_of(evaluation['oracle_artifact_path'])
    require_hash(oracle_path, evaluation['oracle_artifact_sha256'])
    oracle = validator.load_and_validate_oracle(oracle_path,
        {'oracle': {'oracle_manifest': oracle_manifest}}, torch)['difference']
    reports = {k: score_report(responses[k], oracle, validator) for k in GRID}
    decision = classify(reports)
    baseline_path = path_of(evaluation['raw_fineweb_baseline_path'])
    require_hash(baseline_path, evaluation['raw_fineweb_baseline_sha256'])
    prior = validator.load_json_object(baseline_path, 'frozen raw FineWeb evaluation')
    baseline = next(row for row in prior['candidates'] if row['candidate'] == 'response_fineweb')
    baseline_primary = baseline['positions_1_4_mean_cosine']
    baseline_secondary = baseline['positions_1_127_mean_cosine']
    descriptive = {str(k): {
        'delta_primary_vs_raw_fineweb_jg': None if reports[k]['positions_1_4_mean_cosine'] is None else reports[k]['positions_1_4_mean_cosine']-baseline_primary,
        'delta_secondary_vs_raw_fineweb_jg': None if reports[k]['positions_1_127_mean_cosine'] is None else reports[k]['positions_1_127_mean_cosine']-baseline_secondary}
        for k in GRID}
    require_hash(candidate_path, manifest['candidate']['serialized_sha256'])
    return {'format_version': 1, 'attempt_id': ATTEMPT,
            'purpose': spec['purpose'],
            'pilot_selection': spec['pilot_selection'],
            'oracle_population_note': evaluation['population_mismatch_note'],
            'construction_manifest_sha256': sha256_file(path_of(spec['paths']['construction_manifest_path'])),
            'candidate_serialized_sha256': sha256_file(candidate_path),
            'oracle_serialized_sha256': evaluation['oracle_artifact_sha256'],
            'oracle_difference_raw_sha256': evaluation['oracle_difference_raw_sha256'],
            'same_specimen_exploratory_method_development': True,
            'clean_heldout_validation': False,
            'reports': {str(k): reports[k] for k in GRID},
            'development_set_interpretation': decision,
            'raw_fineweb_jg_descriptive_baseline': {'source_sha256': evaluation['raw_fineweb_baseline_sha256'],
                'primary': baseline_primary, 'secondary': baseline_secondary,
                'deltas': descriptive}}


def run(*, resume=False):
    import torch
    spec = load_spec()
    refuse_outputs(spec, resume=resume)
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for the under-one-hour Attempt121 pilot')
    a5_record = spec['frozen_attempt005']
    a5 = import_pinned(path_of(a5_record['constructor_path']), a5_record['constructor_sha256'],
                       'attempt121_frozen_attempt005')
    a14 = import_pinned(PROJECT/'scripts/ablation/construct_multicorpus_raw_jg_consensus.py',
                       '59936b928dcf942470db92debb1f2a8e41823ae1c9082803d2fdd8dc50202caa',
                       'attempt121_frozen_probe_helper')
    tokens, probe, five_manifest, inputs, model_dir = validate_blind_inputs(spec, a5, a14, torch)
    pilot_tokens = fixed_prefix(tokens, spec['pilot_selection']['generic_corpus'],
                                FULL_CORPUS_ROWS, torch)
    model, _ = a5.load_local_model(model_dir, 'cuda', torch)
    eligible = a5.discover_eligible_linear_weights(model, torch)
    selected = a5.freeze_other_parameters(model, eligible)
    frozen_before = a5.frozen_parameter_hashes(model, selected, torch)
    originals = original_weights(eligible, torch)
    source_hash = hashlib.sha256(json.dumps(
        five_manifest['source_checkpoint']['files'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    spec_hash = sha256_file(SPEC_PATH)
    constructor_hash = sha256_file(Path(__file__))
    final_mean = mean_activation(model, probe, spec, a14, torch)
    final_mean_hash = hashlib.sha256(final_mean.numpy().astype('<f8', copy=False).tobytes()).hexdigest()
    rows, checkpoints, responses = [], {}, {}
    initial_scale = None
    eta = None
    start_step = 1
    recovery_path = path_of(spec['paths']['recovery_path'])
    if resume:
        recovery = restore_recovery(recovery_path, eligible, source_hash, spec_hash, a14, torch)
        if not torch.equal(final_mean, recovery['final_mean']):
            raise ValueError('Resume final activation mean changed')
        rows, checkpoints, responses = recovery['rows'], recovery['checkpoints'], recovery['responses']
        initial_scale = validate_initial_scale(recovery['initial_scale'])
        eta = recovery['eta']
        start_step = RECOVERY_STEP + 1
    for step in trajectory_steps(start_step):
        mean_loss = a5.accumulate_mean_generic_gradient(model, pilot_tokens, eligible,
                                                          spec['generic_loss'], torch)
        norm = gradient_norm(eligible, torch)
        if step == 1:
            scale = a5.global_rollback_scale(eligible, .00125, torch)
            initial_scale = derive_pilot_scale(scale)
            if not math.isclose(norm, initial_scale['pilot_initial_gradient_norm'], rel_tol=1e-12):
                raise ValueError('Pilot initial gradient norm accounting mismatch')
            eta = initial_scale['fixed_eta']
        if eta is None:
            raise ValueError('Fixed eta was not derived')
        update_eligible(eligible, eta, torch)
        model.zero_grad(set_to_none=True)
        a5.verify_frozen_parameters(model, selected, frozen_before, torch)
        displacement = displacement_norm(eligible, originals, torch)
        row = {'step': step, 'generic_mean_ce_before_update': mean_loss,
               'aggregate_gradient_norm_before_update': norm,
               'cumulative_displacement_norm': displacement,
               'cumulative_displacement_relative_to_initial_weight_norm':
                   displacement / initial_scale['initial_weight_norm']}
        rows.append(row)
        print(f"relaxation step {step}/{STEPS} | CE={mean_loss:.8g} | gradient_norm={norm:.8g} | displacement={displacement:.8g}", flush=True)
        if step in GRID:
            current_mean = mean_activation(model, probe, spec, a14, torch)
            response = response_from_means(final_mean, current_mean, torch)
            diagnostic = response_diagnostics(response, responses.get(1), a14, torch)
            per_matrix, aggregate = state_hashes(eligible, a14, torch)
            checkpoints[step] = {'step': step, 'generic_mean_ce_before_update': mean_loss,
                'aggregate_gradient_norm_before_update': norm,
                'cumulative_displacement_norm': displacement,
                'cumulative_displacement_relative_to_initial_weight_norm': row['cumulative_displacement_relative_to_initial_weight_norm'],
                'state_per_matrix_sha256': per_matrix, 'state_aggregate_sha256': aggregate,
                'response': diagnostic}
            responses[step] = response
        if step == RECOVERY_STEP and not resume:
            save_recovery(recovery_path, eligible, rows, checkpoints, responses,
                          final_mean, initial_scale, source_hash, spec_hash, a14, torch)
    if len(rows) != STEPS or list(responses) != list(GRID) or sorted(checkpoints) != list(GRID):
        raise ValueError('Incomplete fixed relaxation trajectory')
    a5.verify_frozen_parameters(model, selected, frozen_before, torch)
    for path, digest in inputs.items():
        require_hash(Path(path), digest)
    if a5.checkpoint_file_records(model_dir) != five_manifest['source_checkpoint']['files']:
        raise ValueError('Source model changed during construction')
    require_hash(SPEC_PATH, spec_hash)
    require_hash(Path(__file__), constructor_hash)
    refuse_outputs(spec, resume=True)
    candidate_path = path_of(spec['paths']['candidate_path'])
    manifest_path = path_of(spec['paths']['construction_manifest_path'])
    atomic_torch_publish(candidate_path, {'R_1': responses[1], 'R_2': responses[2],
                                          'R_4': responses[4]}, torch)
    manifest = {'format_version': 1, 'attempt_id': ATTEMPT,
        'information_policy': spec['information_policy'],
        'spec_sha256': spec_hash, 'constructor_sha256': constructor_hash,
        'frozen_input_sha256': inputs,
        'source_checkpoint': five_manifest['source_checkpoint'],
        'pilot_selection': spec['pilot_selection'],
        'generic_loss': spec['generic_loss'],
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
    atomic_json_publish(manifest_path, manifest)
    frozen_artifact = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if list(frozen_artifact) != [f'R_{k}' for k in GRID] or sha256_file(candidate_path) != manifest['candidate']['serialized_sha256']:
        raise ValueError('Published candidate inventory/hash mismatch')
    for k in GRID:
        value = validate_response(frozen_artifact[f'R_{k}'], torch)
        if a14.sha256_raw_float32_tensor(value, torch) != checkpoints[k]['response']['raw_sha256']:
            raise ValueError('Published response hash mismatch')
        digest = hashlib.sha256()
        for name, one in checkpoints[k]['state_per_matrix_sha256'].items():
            digest.update(name.encode()+b'\0'+one.encode('ascii')+b'\n')
        if digest.hexdigest() != checkpoints[k]['state_aggregate_sha256']:
            raise ValueError('Published parameter aggregate hash mismatch')
    if a5.load_json_object(manifest_path, 'Attempt121 construction manifest') != manifest:
        raise ValueError('Published manifest changed')
    require_hash(SPEC_PATH, spec_hash)
    require_hash(Path(__file__), constructor_hash)
    _, aggregate = state_hashes(eligible, a14, torch)
    if aggregate != checkpoints[STEPS]['state_aggregate_sha256']:
        raise ValueError('Published final state hash mismatch')
    recovery_path.unlink()
    print('RELAXATION_CANDIDATES_FROZEN', flush=True)
    result = evaluate_frozen(spec, {k: frozen_artifact[f'R_{k}'] for k in GRID},
                             manifest, candidate_path, torch)
    atomic_json_publish(path_of(spec['paths']['result_path']), result)
    print('Wrote Attempt121 result', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true', help='Resume only from the fixed step-2 recovery checkpoint')
    args = parser.parse_args(argv)
    try:
        run(resume=args.resume)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
