#!/usr/bin/env python3
"""Attempt 128: blind exact self-sample score gradients, then matched evaluation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '128_self_sample_score_gradient_pilot'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = '88f6a4ccc3bf745265c9611e2949d96e97f9388aed97bf5a3438d65220c81e25'
SEEDS = tuple(range(8))
SEED_NAMES = tuple(f'response_seed_{seed}' for seed in SEEDS)
NAMES = (*SEED_NAMES, 'response_pool', 'response_seed_consensus')
TARGET_SHA256 = '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def require_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('Frozen SHA256 mismatch: ' + str(path))
    return path


def import_pinned(path, expected, name):
    import importlib.util
    require_hash(path, expected)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def load_spec():
    require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text())
    context, sampling, gradient = spec['contexts'], spec['sampling'], spec['gradient']
    expected_paths = {key: f'experiments/attempts/{ATTEMPT}/{filename}'
                      for key, filename in (('construction_manifest_path', 'construction-manifest.json'),
                                            ('candidate_path', 'candidate.pt'),
                                            ('result_path', 'result.json'))}
    if (spec['attempt_id'] != ATTEMPT or
            context != {'source_rows': [0, 64], 'count': 64,
                        'prefix_positions': [0, 127], 'prefix_length': 127,
                        'ignored_original_token_position': 127, 'batch_size': 8,
                        'sampling_forward_batches': 8, 'same_contexts_all_seeds': True} or
            sampling != {'seed_ids': list(SEEDS),
                'uniform_domain': 'attempt128-selfsample-v1|seed=<seed>|row=<row>',
                'uniform_bits': 'first_64_SHA256_bits_big_endian',
                'uniform_conversion': '(bits+0.5)/2**64_clamped_to_open_unit_interval',
                'categorical': 'smallest_vocabulary_index_with_float64_cdf_strictly_greater_than_uniform',
                'full_vocabulary': True, 'temperature': 1.0, 'top_k': None,
                'top_p': None, 'truncation': False, 'repetition_penalty': 1.0,
                'rejection': False, 'greedy': False, 'softmax_dtype': 'float64',
                'softmax_sum_absolute_tolerance': 1e-12,
                'sampled_token_count_per_seed': 64,
                'per_seed_diagnostics': ['mean_probability', 'median_probability',
                    'min_probability', 'max_probability', 'mean_surprisal',
                    'max_surprisal', 'argmax_count']} or
            gradient != {'eligible_matrix_count': 98, 'sequence_context_length': 127,
                'targets_per_context_per_seed': 1, 'prefix_token_loss_included': False,
                'seed_loss': 'mean_over_64_sampled_next_token_NLL_only',
                'batch_size': 8, 'backward_batches_per_seed': 8,
                'seed_backward_batches': 64,
                'pool_loss': 'mean_over_fixed_8x64_sampled_next_token_NLL_only',
                'pool_backward_batches': 8, 'pool_total_terms': 512,
                'soft_control': 'mean_negative_stop_gradient_softmax_times_log_softmax_at_same_final_checkpoint',
                'soft_control_backward_batches': 8, 'optimizer': False,
                'weight_decay': False, 'clipping': False, 'amp': False} or
            spec['candidate'] != {'tensor_order': list(NAMES), 'shape': [128, 2048],
                'dtype': 'contiguous_cpu_float32', 'parameter_gradient_jvp_count': 9,
                'consensus_definition':
                    'mean_of_8_per_position_unit_seed_responses_CPU_float64_no_final_renormalization',
                'response_normalization': False, 'historical_sign_selection': False,
                'postfreeze_combination': False} or
            spec['barrier'] != {'marker': 'SELF_SAMPLE_CANDIDATES_FROZEN',
                'sample_table_soft_control_gradients_jvps_consensus_published_and_revalidated_before_historical_access': True,
                'manifest_contains_historical_scores': False} or
            spec['evaluation'] != {'target_probe_rows': [0, 1024],
                'probe_batch_size': 32, 'probe_batches_per_model': 32,
                'target_raw_sha256': TARGET_SHA256,
                'signed_positionwise_cosine': True, 'absolute_cosine': False,
                'flattened_historical_cosine': False, 'position_0_separate': True,
                'primary_positions': [1, 2, 3, 4],
                'secondary_positions': list(range(1, 128)),
                'best_seed_selection': False,
                'strong_support': {'both_positive_seed_count_min': 7,
                    'median_primary_min': .10, 'median_secondary_min': .10,
                    'consensus_primary_min': .10, 'consensus_secondary_min': .10},
                'moderate_support': {'both_positive_seed_count_min': 6,
                    'median_primary_gt': 0, 'median_secondary_gt': 0,
                    'consensus_primary_gt': 0, 'consensus_secondary_gt': 0}} or
            spec['workload'] != {'sampling_forward_batches': 8,
                'soft_control_backward_batches': 8, 'seed_backward_batches': 64,
                'pool_backward_batches': 8, 'total_backward_batches': 80,
                'jvp_probe_batches': 288, 'matched_target_forward_batches': 64,
                'ggn_operations': 0, 'cg_operations': 0} or
            spec['paths'] != expected_paths or
            spec['fineweb_corpus_record_role'] !=
                'The copied Attempt113 selected_rows [0,512) field is provenance only. Attempt128 uses contexts [0,64).' or
            spec['probe']['selected_rows'] != [0, 1024] or
            spec['probe']['selected_batch_count'] != 32 or
            spec['tangent']['target_relative_frobenius'] != .00125 or
            spec['jvp']['strict_forward_ad'] is not True or
            spec['jvp']['fallback'] is not False or
            spec['information_policy']['same_specimen_exploratory_method_development'] is not True or
            spec['information_policy']['clean_heldout_validation'] is not False or
            any(spec['information_policy'][key] is not False for key in
                ('historical_base_access_before_barrier', 'adapter_access_before_barrier',
                 'true_delta_access_before_barrier',
                 'historical_activation_target_access_before_barrier',
                 'prior_oracle_score_access_before_barrier'))):
        raise ValueError('Attempt128 fixed self-sampling plan mismatch')
    return spec


def refuse_outputs(spec):
    paths = [path_of(value) for value in spec['paths'].values()]
    if len(set(paths)) != 3 or any(path.exists() or path.is_symlink() for path in paths):
        raise ValueError('Attempt128 output already exists or paths overlap')
    return {key: path_of(value) for key, value in spec['paths'].items()}


def hash_uniform(seed, row):
    if seed not in SEEDS or not isinstance(row, int) or not 0 <= row < 64:
        raise ValueError('Self-sample seed or original FineWeb row outside fixed grid')
    message = f'attempt128-selfsample-v1|seed={seed}|row={row}'.encode('ascii')
    bits = int.from_bytes(hashlib.sha256(message).digest()[:8], 'big')
    value = (bits + .5) / 2**64
    return min(max(value, math.nextafter(0., 1.)), math.nextafter(1., 0.))


def inverse_cdf(probs, uniform):
    probabilities = np.asarray(probs)
    if (probabilities.ndim != 1 or probabilities.size < 2 or
            probabilities.dtype != np.float64 or
            not probabilities.flags.c_contiguous or
            not np.isfinite(probabilities).all() or
            np.any(probabilities < 0) or
            not math.isfinite(uniform) or not 0 < uniform < 1):
        raise ValueError('Malformed exact-categorical probabilities or uniform')
    total = math.fsum(map(float, probabilities))
    if abs(total - 1.) > 1e-12:
        raise ValueError('Full-vocabulary softmax normalization mismatch')
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative[-1] = 1.0  # Guard the last CDF value against summation roundoff.
    index = int(np.searchsorted(cumulative, uniform, side='right'))
    if index >= probabilities.size or probabilities[index] <= 0:
        raise ValueError('Inverse-CDF selected invalid zero-probability token')
    return index, float(probabilities[index]), int(np.argmax(probabilities))


def table_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def sample_table_from_probabilities(probabilities):
    probs = np.asarray(probabilities)
    if (probs.ndim != 2 or probs.shape[0] != 64 or probs.dtype != np.float64 or
            not probs.flags.c_contiguous or not np.isfinite(probs).all()):
        raise ValueError('Expected 64 full-vocabulary float64 probability rows')
    seeds = []
    for seed in SEEDS:
        ids, uniforms, chosen_probs, surprises, argmax_matches = [], [], [], [], 0
        for row in range(64):
            uniform = hash_uniform(seed, row)
            token, probability, argmax = inverse_cdf(probs[row], uniform)
            ids.append(token)
            uniforms.append(uniform)
            chosen_probs.append(probability)
            surprises.append(-math.log(probability))
            argmax_matches += int(token == argmax)
        record = {'seed_id': seed, 'original_rows': list(range(64)),
                  'sampled_token_ids': ids, 'uniforms': uniforms,
                  'sampled_probabilities': chosen_probs,
                  'sampled_surprisals': surprises,
                  'diagnostics': {
                      'mean_probability': math.fsum(chosen_probs) / 64,
                      'median_probability': statistics.median(chosen_probs),
                      'min_probability': min(chosen_probs),
                      'max_probability': max(chosen_probs),
                      'mean_surprisal': math.fsum(surprises) / 64,
                      'max_surprisal': max(surprises),
                      'argmax_count': argmax_matches}}
        record['sha256'] = table_hash(record)
        seeds.append(record)
    table = {'seed_order': list(SEEDS), 'context_rows': [0, 64],
             'context_prefix_positions': [0, 127],
             'sampler': 'SHA256_uniform_full_vocabulary_float64_softmax_inverse_CDF',
             'seeds': seeds}
    table['aggregate_sha256'] = table_hash(table)
    return table


def validate_sample_table(table):
    if (table['seed_order'] != list(SEEDS) or table['context_rows'] != [0, 64] or
            table['context_prefix_positions'] != [0, 127] or
            len(table['seeds']) != 8):
        raise ValueError('Frozen self-sample table grid mismatch')
    for seed, record in zip(SEEDS, table['seeds']):
        raw = {key: value for key, value in record.items() if key != 'sha256'}
        if (record['seed_id'] != seed or record['original_rows'] != list(range(64)) or
                any(len(record[key]) != 64 for key in
                    ('sampled_token_ids', 'uniforms', 'sampled_probabilities',
                     'sampled_surprisals')) or
                record['sha256'] != table_hash(raw) or
                any(record['uniforms'][row] != hash_uniform(seed, row)
                    for row in range(64)) or
                any(not math.isfinite(value) or value <= 0 for value in
                    record['sampled_probabilities']) or
                any(not math.isclose(-math.log(p), nll, rel_tol=0, abs_tol=1e-14)
                    for p, nll in zip(record['sampled_probabilities'],
                                      record['sampled_surprisals']))):
            raise ValueError('Frozen self-sample seed record mismatch')
    raw = {key: value for key, value in table.items() if key != 'aggregate_sha256'}
    if table['aggregate_sha256'] != table_hash(raw):
        raise ValueError('Frozen self-sample aggregate SHA256 mismatch')
    return table


def sample_endpoint(model, contexts, torch):
    if (tuple(contexts.shape) != (64, 127) or contexts.dtype != torch.int64 or
            contexts.device.type != 'cpu' or not contexts.is_contiguous()):
        raise ValueError('Expected the fixed 64 contiguous 127-token contexts')
    model.eval()
    device = next(model.parameters()).device
    rows = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 64, 8):
            batch = contexts[start:start + 8].to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            if (logits.dtype != torch.float32 or logits.ndim != 3 or
                    tuple(logits.shape[:2]) != (8, 127) or
                    logits.shape[-1] != model.config.vocab_size or
                    not bool(torch.isfinite(logits).all())):
                raise ValueError('Malformed/nonfinite full-vocabulary endpoint logits')
            p64 = torch.softmax(logits[:, -1, :].to(torch.float64), dim=-1)
            rows.append(p64.cpu().numpy())
            del batch, logits, p64
    return sample_table_from_probabilities(np.ascontiguousarray(np.concatenate(rows)))


def fixed_contexts(tokens, torch):
    if (tuple(tokens.shape) != (4096, 128) or tokens.dtype != torch.int64 or
            tokens.device.type != 'cpu' or not tokens.is_contiguous()):
        raise ValueError('Expected the complete frozen Attempt005 FineWeb artifact')
    contexts = tokens[:64, :127].contiguous()
    if tuple(contexts.shape) != (64, 127):
        raise ValueError('Fixed FineWeb 127-token context inventory mismatch')
    return contexts


def selected_targets(table, kind, torch):
    validate_sample_table(table)
    values = [record['sampled_token_ids'] for record in table['seeds']]
    if kind == 'pool':
        return torch.tensor(values, dtype=torch.int64)  # [8 seeds, 64 contexts]
    if isinstance(kind, int) and kind in SEEDS:
        return torch.tensor(values[kind], dtype=torch.int64)  # [64 contexts]
    if kind == 'soft':
        return None
    raise ValueError('Gradient kind is outside predeclared seeds/pool/soft control')


def sampled_next_token_loss_sum(last_logits, target, kind, torch):
    """Only the sampled continuation uses the same FP64 log-probs as sampling."""
    if last_logits.ndim != 2 or last_logits.dtype != torch.float32:
        raise ValueError('Expected FP32 final-position logits only')
    batch, vocab = last_logits.shape
    student_logp = torch.log_softmax(last_logits.to(torch.float64), dim=-1)
    if kind == 'soft':
        if target is not None:
            raise ValueError('Soft control has no sampled targets')
        teacher_q = torch.softmax(last_logits.detach().to(torch.float64), dim=-1)
        return -(teacher_q * student_logp).sum()
    if target is None or target.dtype != torch.int64 or target.device != last_logits.device:
        raise ValueError('Malformed sampled continuation targets')
    if bool((target < 0).any()) or bool((target >= vocab).any()):
        raise ValueError('Sampled token outside full vocabulary')
    if kind == 'pool':
        if tuple(target.shape) != (batch, 8):
            raise ValueError('Pooled targets must contain eight draws per context')
        return -student_logp.gather(1, target).sum()
    if kind in SEEDS and tuple(target.shape) == (batch,):
        return -student_logp.gather(1, target.unsqueeze(1)).sum()
    raise ValueError('Malformed single-seed next-token target shape')


def global_gradient_norm(model, eligible, torch):
    selected = {id(module.weight) for _, module in eligible}
    if len(eligible) != 98:
        raise ValueError('Expected 98 eligible matrices')
    for name, parameter in model.named_parameters():
        if id(parameter) not in selected and parameter.grad is not None:
            raise ValueError('Noneligible parameter received gradient: ' + name)
    squares = []
    for name, module in eligible:
        gradient = module.weight.grad
        if (gradient is None or gradient.dtype != torch.float32 or
                tuple(gradient.shape) != tuple(module.weight.shape) or
                not bool(torch.isfinite(gradient).all())):
            raise ValueError('Missing/nonfinite eligible gradient: ' + name)
        value = gradient.detach().to(device='cpu', dtype=torch.float64)
        squares.append(float(torch.sum(value.square()).item()))
    norm = math.sqrt(math.fsum(squares))
    if not math.isfinite(norm):
        raise ValueError('Nonfinite eligible gradient norm')
    return norm


def accumulate_score_gradient(model, contexts, targets, kind, eligible, torch):
    if (tuple(contexts.shape) != (64, 127) or contexts.dtype != torch.int64 or
            contexts.device.type != 'cpu' or not contexts.is_contiguous()):
        raise ValueError('Self-score contexts must be fixed FineWeb 127-token prefixes')
    if kind == 'pool':
        if targets is None or tuple(targets.shape) != (8, 64):
            raise ValueError('Pooled gradient requires the fixed 8x64 sample table')
        denominator = 512
    elif kind == 'soft':
        if targets is not None:
            raise ValueError('Soft teacher must use detached model distribution')
        denominator = 64
    elif kind in SEEDS:
        if targets is None or tuple(targets.shape) != (64,):
            raise ValueError('Seed gradient requires exactly 64 sampled tokens')
        denominator = 64
    else:
        raise ValueError('Unpredeclared score-gradient kind')
    model.eval()
    model.zero_grad(set_to_none=True)
    device = eligible[0][1].weight.device
    losses = []
    with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 64, 8):
            batch = contexts[start:start + 8].to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            if (logits.dtype != torch.float32 or tuple(logits.shape[:2]) != (8, 127) or
                    logits.shape[-1] != model.config.vocab_size or
                    not bool(torch.isfinite(logits).all())):
                raise ValueError('Malformed/nonfinite score-gradient logits')
            last_logits = logits[:, -1, :]
            if kind == 'pool':
                chosen = targets[:, start:start + 8].transpose(0, 1).contiguous().to(device)
            elif kind == 'soft':
                chosen = None
            else:
                chosen = targets[start:start + 8].to(device)
            loss_sum = sampled_next_token_loss_sum(last_logits, chosen, kind, torch)
            if not bool(torch.isfinite(loss_sum)):
                raise ValueError('Nonfinite self-score loss')
            (loss_sum / denominator).backward()
            losses.append(float(loss_sum.detach().to(device='cpu', dtype=torch.float64)))
            del batch, logits, last_logits, chosen, loss_sum
    norm = global_gradient_norm(model, eligible, torch)
    mean_loss = math.fsum(losses) / denominator
    if not math.isfinite(mean_loss):
        raise ValueError('Nonfinite self-score mean loss')
    return {'mean_loss': mean_loss, 'gradient_norm': norm,
            'backward_batches': 8, 'objective_terms': denominator,
            'sampled_nll_terms': 0 if kind == 'soft' else denominator}


def soft_score_zero_control(probs, logits):
    """NumPy check of d/dlogits[-sum stop(p) log p] at q=p, for unit tests."""
    p, z = np.asarray(probs, dtype=np.float64), np.asarray(logits, dtype=np.float64)
    if p.shape != z.shape or p.ndim != 1:
        raise ValueError('Softmax control shapes mismatch')
    shifted = z - np.max(z)
    student = np.exp(shifted) / math.fsum(np.exp(shifted))
    return student - p


def positive_tangent_response(model, probe, eligible, spec, a14, a126, torch):
    scale = a126.validate_global_scale(a14.global_tangent_scale(eligible, .00125, torch))
    if not math.isclose(scale['aggregate_source_weight_norm'], 918.1587250300843,
                        rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError('Final eligible-weight norm differs from frozen Attempt005')
    tangents, matrices, realized = a14.prepare_tangents(eligible, scale, torch)
    jvp_spec = {**spec, 'probe': {**spec['probe'], 'sample_count': 1024}}
    try:
        primal, response, readout = a14.compute_probe_response(
            model, probe, tangents, jvp_spec, torch)
    finally:
        tangents.clear()
        model.zero_grad(set_to_none=True)
    response32 = a126.a122.prior.validate_response(response.to(torch.float32).contiguous(), torch)
    record = {**scale, **realized, 'eligible_matrix_count': len(matrices),
              'readout_checks': readout,
              'response_norm': float(torch.linalg.vector_norm(response32.double())),
              'response_raw_sha256': a14.sha256_raw_float32_tensor(response32, torch)}
    return primal, response32, record


def blind_seed_consensus(responses, torch):
    if list(responses) != list(SEED_NAMES):
        raise ValueError('Consensus requires exactly eight frozen seed responses')
    for value in responses.values():
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                value.device.type != 'cpu' or not value.is_contiguous() or
                tuple(value.shape) != (128, 2048) or not bool(torch.isfinite(value).all())):
            raise ValueError('Malformed seed response before consensus')
    raw = np.stack([responses[name].numpy().astype(np.float64) for name in SEED_NAMES])
    norms = np.linalg.norm(raw, axis=2)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Zero/nonfinite per-position seed response norm')
    unit = raw / norms[:, :, None]
    consensus64 = np.mean(unit, axis=0, dtype=np.float64)
    concentration = np.linalg.norm(consensus64, axis=1)
    if (not np.isfinite(concentration).all() or np.any(concentration <= 0) or
            np.any(concentration > 1 + 1e-12)):
        raise ValueError('Invalid blind seed consensus concentration')
    pairs = {}
    for i in SEEDS:
        for j in range(i + 1, 8):
            x, y = raw[i].ravel(), raw[j].ravel()
            per_position = np.einsum('ph,ph->p', unit[i], unit[j])
            pairs[f'{i}_{j}'] = {
                'flattened_diagnostic_cosine': float(np.dot(x, y) /
                    (np.linalg.norm(x) * np.linalg.norm(y))),
                'mean_position_cosine_1_4': float(np.mean(per_position[1:5])),
                'mean_position_cosine_1_127': float(np.mean(per_position[1:]))}
    geometry = {'pairwise_seed_response_cosines': pairs,
                'consensus_concentration_all_128_positions': concentration.tolist(),
                'concentration_positions_1_4_mean': float(np.mean(concentration[1:5])),
                'concentration_positions_1_127_mean': float(np.mean(concentration[1:])),
                'equal_seed_weight': 1 / 8, 'final_renormalization': False}
    return torch.from_numpy(np.ascontiguousarray(consensus64.astype(np.float32))), geometry


def validate_published(paths, construction, a14, a126, torch):
    candidate_path, manifest_path = paths['candidate_path'], paths['construction_manifest_path']
    if (sha256_file(candidate_path) != construction['candidate']['serialized_sha256'] or
            json.loads(manifest_path.read_text()) != construction or
            construction['barrier']['manifest_contains_historical_scores'] is not False or
            construction['candidate']['tensor_order'] != list(NAMES)):
        raise ValueError('Published Attempt128 artifact/manifest changed')
    validate_sample_table(construction['sample_table'])
    if construction['sample_table']['aggregate_sha256'] != construction['sample_table_sha256']:
        raise ValueError('Published sampled table hash mismatch')
    artifact = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != list(NAMES):
        raise ValueError('Published candidate inventory differs from the fixed ten responses')
    for name in NAMES:
        value = a126.a122.prior.validate_response(artifact[name], torch)
        if a14.sha256_raw_float32_tensor(value, torch) != construction['responses'][name]['raw_sha256']:
            raise ValueError('Published response raw SHA256 mismatch: ' + name)
    require_hash(candidate_path, construction['candidate']['serialized_sha256'])
    return artifact


def score_interpretation(reports):
    if list(reports) != list(NAMES):
        raise ValueError('Only the ten predeclared candidates may be scored')
    primary = [reports[name]['positions_1_4_mean_cosine'] for name in SEED_NAMES]
    secondary = [reports[name]['positions_1_127_mean_cosine'] for name in SEED_NAMES]
    pool = reports['response_pool']
    consensus = reports['response_seed_consensus']
    all_values = primary + secondary + [pool['positions_1_4_mean_cosine'],
        pool['positions_1_127_mean_cosine'], consensus['positions_1_4_mean_cosine'],
        consensus['positions_1_127_mean_cosine']]
    if any(value is None or not math.isfinite(value) for value in all_values):
        raise ValueError('Undefined signed historical cosine')
    both_count = sum(x > 0 and y > 0 for x, y in zip(primary, secondary))
    median_primary, median_secondary = statistics.median(primary), statistics.median(secondary)
    consensus_primary = consensus['positions_1_4_mean_cosine']
    consensus_secondary = consensus['positions_1_127_mean_cosine']
    strong = (both_count >= 7 and median_primary >= .10 and median_secondary >= .10 and
              consensus_primary >= .10 and consensus_secondary >= .10)
    moderate = (both_count >= 6 and median_primary > 0 and median_secondary > 0 and
                consensus_primary > 0 and consensus_secondary > 0)
    category = 'strong_support' if strong else 'moderate_support' if moderate else 'no_support'
    def stats(values):
        return {'mean': math.fsum(values) / 8, 'median': statistics.median(values),
                'population_std': statistics.pstdev(values),
                'minimum': min(values), 'maximum': max(values)}
    summary = {
        'positive_primary_seed_count': sum(value > 0 for value in primary),
        'positive_secondary_seed_count': sum(value > 0 for value in secondary),
        'both_positive_seed_count': both_count,
        'seed_primary': stats(primary), 'seed_secondary': stats(secondary),
        'pool_primary': pool['positions_1_4_mean_cosine'],
        'pool_secondary': pool['positions_1_127_mean_cosine'],
        'consensus_primary': consensus_primary,
        'consensus_secondary': consensus_secondary,
        'median_seed_minus_pool': {
            'primary': median_primary - pool['positions_1_4_mean_cosine'],
            'secondary': median_secondary - pool['positions_1_127_mean_cosine']},
        'consensus_minus_pool': {
            'primary': consensus_primary - pool['positions_1_4_mean_cosine'],
            'secondary': consensus_secondary - pool['positions_1_127_mean_cosine']},
        'individual_seed_scores': {name: {
            'primary': reports[name]['positions_1_4_mean_cosine'],
            'secondary': reports[name]['positions_1_127_mean_cosine']}
            for name in SEED_NAMES},
        'best_seed_selection': False, 'pooled_cancellation_is_descriptive_only': True}
    return summary, category


def evaluate_after_barrier(spec, artifact, construction, a126, torch):
    frozen = spec['frozen_attempt123']
    require_hash(path_of(frozen['spec_path']), frozen['spec_sha256'])
    a123 = import_pinned(path_of(frozen['source_path']), frozen['source_sha256'],
                         'attempt128_matched_historical_evaluator')
    matched_spec = a123.load_spec()
    reference = matched_spec['matched_reference']
    privileged = import_pinned(path_of(reference['attempt103_source_path']),
        reference['attempt103_source_sha256'], 'attempt128_frozen_attempt103')
    frozen122 = matched_spec['frozen_attempt122']
    require_hash(path_of(frozen122['construction_manifest_path']),
                 frozen122['construction_manifest_sha256'])
    manifest122 = json.loads(path_of(frozen122['construction_manifest_path']).read_text())
    old103, inventories = a123.validate_matched_reference(matched_spec, manifest122, privileged)
    if (old103['final_checkpoint_files'] != spec['final_checkpoint']['files'] or
            old103['readout'] != spec['readout'] or old103['probe_rows'] != [0, 1024] or
            reference['target_raw_sha256'] != TARGET_SHA256 or
            reference['probe_rows'] != [0, 1024] or reference['batch_size'] != 32 or
            reference['batches_per_model'] != 32):
        raise ValueError('Matched historical target provenance differs from frozen controls')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for matched FP32 historical readout')
    full_probe = privileged.a.load_probe(privileged.resolve(old103['paths']['probe_path']),
        {**old103['probe'], 'sample_count': 10000}, torch)
    probe = a123.matched_probe(full_probe, matched_spec, torch)
    del full_probe
    base, final, load_seconds = privileged.load_model_pair(old103)
    final_mean = privileged.base_probe_mean(final, probe, old103)
    base_mean = privileged.base_probe_mean(base, probe, old103)
    target, target_hash = a123.verified_target(final_mean, base_mean, matched_spec,
        privileged.matched_difference, privileged.a.sha256_raw_float32_tensor, torch)
    if target_hash != TARGET_SHA256:
        raise ValueError('Matched target raw float32 SHA256 mismatch')
    for directory, expected in ((privileged.resolve(old103['paths']['base_directory']),
                                 inventories['base_files']),
                                (privileged.resolve(old103['paths']['final_directory']),
                                 inventories['final_files'])):
        if privileged.a.checkpoint_file_records(directory) != expected:
            raise ValueError('Historical checkpoint changed during evaluation')
    del base, final, final_mean, base_mean, probe
    gc.collect()
    torch.cuda.empty_cache()
    old_eval = a126.a122.load_spec()['evaluation']
    validator = import_pinned(path_of(old_eval['validator_path']),
        old_eval['validator_sha256'], 'attempt128_signed_position_validator')
    reports = {name: a126.a122.score_report(artifact[name], target, validator)
               for name in NAMES}
    summary, category = score_interpretation(reports)
    return reports, summary, category, target_hash, load_seconds


def run():
    import torch
    spec = load_spec()
    paths = refuse_outputs(spec)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for frozen FP32 strict JVP convention')
    frozen126 = spec['frozen_attempt126']
    for key in ('spec', 'source'):
        require_hash(path_of(frozen126[key + '_path']), frozen126[key + '_sha256'])
    a126 = import_pinned(path_of(frozen126['source_path']),
                         frozen126['source_sha256'], 'attempt128_frozen_blind_helper')
    a14 = a126.import_pinned(a126.SOURCE014, a126.SOURCE014_SHA256,
                             'attempt128_frozen_jvp_helper')
    tokens, probe, input_hashes, corpus_provenance, model_dir = (
        a126.validate_blind_inputs(spec, a14, torch))
    contexts = fixed_contexts(tokens, torch)
    model = a14.load_local_model(model_dir, 'cuda', torch)
    eligible = a14.discover_eligible_linear_weights(model, torch)
    a14.freeze_other_parameters(model, eligible)
    before = a14.model_state_hashes(model, torch)
    sample_table = validate_sample_table(sample_endpoint(model, contexts, torch))
    frozen_table_hash = sample_table['aggregate_sha256']
    # From here onward all gradients use this one immutable 8x64 table.
    soft = accumulate_score_gradient(model, contexts, None, 'soft', eligible, torch)
    model.zero_grad(set_to_none=True)
    responses, gradient_records = {}, {}
    primal_reference = None
    for kind in (*SEEDS, 'pool'):
        if table_hash({key: value for key, value in sample_table.items()
                       if key != 'aggregate_sha256'}) != frozen_table_hash:
            raise ValueError('Frozen self-sample table changed during gradients')
        targets = selected_targets(sample_table, kind, torch)
        gradient = accumulate_score_gradient(model, contexts, targets, kind, eligible, torch)
        if gradient['gradient_norm'] <= 0:
            raise ValueError('Empirical self-score gradient norm must be positive')
        primal, response, record = positive_tangent_response(
            model, probe, eligible, spec, a14, a126, torch)
        if not math.isclose(gradient['gradient_norm'],
                            record['aggregate_generic_gradient_norm'],
                            rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Score-gradient norm changed during positive tangent scaling')
        if primal_reference is None:
            primal_reference = primal
        elif not torch.equal(primal_reference, primal):
            raise ValueError('Final-checkpoint JVP primal changed across sampled gradients')
        name = 'response_pool' if kind == 'pool' else f'response_seed_{kind}'
        responses[name] = response
        gradient_records[name] = {**gradient, **record}
        del targets, primal, response, record
        gc.collect()
    if list(responses) != list(NAMES[:-1]):
        raise ValueError('Expected exactly nine parameter-gradient JVP responses')
    consensus, geometry = blind_seed_consensus(
        {name: responses[name] for name in SEED_NAMES}, torch)
    responses['response_seed_consensus'] = a126.a122.prior.validate_response(consensus, torch)
    if list(responses) != list(NAMES):
        raise ValueError('Fixed ten-response candidate inventory changed')
    if (a14.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files'] or
            any(sha256_file(path) != digest for path, digest in input_hashes.items()) or
            sha256_file(path_of(spec['probe']['path'])) != spec['probe']['serialized_sha256']):
        raise ValueError('Frozen blind input changed during construction')
    a14.verify_model_unchanged(model, before, torch)
    require_hash(SPEC_PATH, SPEC_SHA256)
    a126.a122.prior.atomic_torch_publish(paths['candidate_path'], responses, torch)
    construction = {
        'format_version': 1, 'attempt_id': ATTEMPT,
        'information_policy': spec['information_policy'],
        'spec_sha256': SPEC_SHA256, 'constructor_sha256': sha256_file(Path(__file__)),
        'frozen_attempt005': spec['frozen_attempt005'],
        'frozen_attempt126': spec['frozen_attempt126'],
        'final_checkpoint_files': spec['final_checkpoint']['files'],
        'fineweb_corpus_provenance': corpus_provenance,
        'context_selection': spec['contexts'], 'sampling': spec['sampling'],
        'sample_table': sample_table, 'sample_table_sha256': frozen_table_hash,
        'soft_teacher_zero_expectation_control': soft,
        'gradient_definition': spec['gradient'], 'gradient_records': gradient_records,
        'probe': spec['probe'], 'readout': spec['readout'], 'jvp': spec['jvp'],
        'tangent': spec['tangent'], 'oracle_free_seed_geometry': geometry,
        'responses': {name: {
            'raw_sha256': a14.sha256_raw_float32_tensor(responses[name], torch),
            'shape': [128, 2048], 'dtype': 'contiguous_cpu_float32',
            'norm': float(torch.linalg.vector_norm(responses[name].double()))}
            for name in NAMES},
        'final_primal_float64_sha256': hashlib.sha256(
            primal_reference.numpy().astype('<f8', copy=False).tobytes()).hexdigest(),
        'candidate': {'path': str(paths['candidate_path']),
            'serialized_sha256': sha256_file(paths['candidate_path']),
            'tensor_order': list(NAMES), 'shape': [128, 2048],
            'dtype': 'contiguous_cpu_float32'},
        'barrier': spec['barrier'],
        'same_specimen_exploratory_method_development': True,
        'clean_heldout_validation': False}
    a126.a122.prior.atomic_json_publish(paths['construction_manifest_path'], construction)
    manifest_hash = sha256_file(paths['construction_manifest_path'])
    artifact = validate_published(paths, construction, a14, a126, torch)
    require_hash(paths['construction_manifest_path'], manifest_hash)
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), construction['constructor_sha256'])
    del model, eligible, tokens, contexts, probe, responses, primal_reference
    gc.collect()
    torch.cuda.empty_cache()
    print('SELF_SAMPLE_CANDIDATES_FROZEN', flush=True)

    reports, summary, category, target_hash, load_seconds = evaluate_after_barrier(
        spec, artifact, construction, a126, torch)
    require_hash(paths['candidate_path'], construction['candidate']['serialized_sha256'])
    require_hash(paths['construction_manifest_path'], manifest_hash)
    result = {
        'format_version': 1, 'attempt_id': ATTEMPT,
        'purpose': spec['purpose'],
        'same_specimen_exploratory_method_development': True,
        'clean_heldout_validation': False,
        'construction_manifest_sha256': manifest_hash,
        'candidate_serialized_sha256': construction['candidate']['serialized_sha256'],
        'matched_target_raw_sha256': target_hash,
        'matched_model_load_seconds': load_seconds,
        'signed_position_reports': reports,
        'seed_distribution_summary': summary,
        'development_set_classification': category,
        'soft_teacher_zero_expectation_control': soft,
        'seed_gradient_norms': {name: gradient_records[name]['gradient_norm']
                                for name in SEED_NAMES},
        'pool_gradient_norm': gradient_records['response_pool']['gradient_norm'],
        'soft_control_norm_over_median_seed_norm':
            soft['gradient_norm'] / statistics.median(
                gradient_records[name]['gradient_norm'] for name in SEED_NAMES),
        'oracle_free_sample_diagnostics': {str(seed): sample_table['seeds'][seed]['diagnostics']
                                           for seed in SEEDS},
        'sample_table_sha256': frozen_table_hash,
        'oracle_free_seed_geometry': geometry,
        'workload': spec['workload']}
    if paths['result_path'].exists() or paths['result_path'].is_symlink():
        raise ValueError('Attempt128 result already exists')
    a126.a122.prior.atomic_json_publish(paths['result_path'], result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        run()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
