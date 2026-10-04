#!/usr/bin/env python3
"""Attempt132: fixed-size low-surprisal selection from an expanded FineWeb pool."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '132_expanded_fineweb_superlow_stringency'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = 'b54399eedceceb381b7b536a61cae17b81cea90cc5aeb927b10d0f4be66a4c31'
GROUPS = ('super_low', 'next_super_low')
NAMES = tuple('response_' + name for name in GROUPS)
MARKER = 'EXPANDED_SUPERLOW_CANDIDATES_FROZEN'
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
    scoring, selection, loss = (spec[key] for key in
        ('new_surprisal_scoring', 'combined_selection', 'gradient_loss'))
    evaluation = spec['evaluation']
    paths = {key: f'experiments/attempts/{ATTEMPT}/{filename}' for key, filename in
        (('construction_manifest_path', 'construction-manifest.json'),
         ('candidate_path', 'candidate.pt'), ('result_path', 'result.json'))}
    if (spec['attempt_id'] != ATTEMPT or spec['paths'] != paths or
            spec['new_corpus']['full_shape'] != [8192, 128] or
            spec['new_corpus']['old_pooled_valid_example_rank_interval'] != [20000, 28192] or
            spec['new_corpus']['new_valid_example_rank_interval'] != [28192, 36384] or
            spec['new_corpus']['tokens_path'] !=
                '/root/model-diff-scratch/artifacts/attempt132_expanded_fineweb_corpus/tokens.pt' or
            spec['frozen_attempt130']['construction_manifest_sha256'] !=
                '13163051d0fb564187ef9a5a7c004fb92c259f379bce6ad0905243973ebeceb2' or
            spec['frozen_attempt131']['result_sha256'] !=
                '3692fbe6e14c6656b828ff867d131245e7e41ef79256abc355e7d3ed718a0ca4' or
            scoring != {'definition':
                'per_sequence_mean_causal_next_token_NLL_at_unmodified_final_checkpoint',
                'population_rows': [0, 8192], 'sequence_length': 128,
                'prediction_tokens_per_row': 127, 'batch_size': 8,
                'number_of_forward_batches': 1024, 'no_grad': True, 'no_amp': True,
                'accumulation_dtype': 'cpu_float64',
                'sort_key': ['surprisal_ascending', 'original_row_index_ascending'],
                'surprisal_storage_dtype': 'contiguous_cpu_float64'} or
            selection != {'old_record_source': 'validated_Attempt130_pooled_ranking_records',
                'old_combined_record_count': 8192,
                'old_combined_rank_pair_raw_int64_sha256':
                    '84bfc8c07f576b61ac3ccda0b56347a2d12a130eccbaef472c38a1eca9cdf8f5',
                'new_record_count': 8192, 'combined_record_count': 16384,
                'sort_key': ['surprisal_ascending', 'source_id_ascending',
                             'original_row_index_ascending'],
                'source_ids': {'old_slice0': 0, 'old_slice1': 1, 'new_slice': 2},
                'pair_dtype': 'contiguous_cpu_little_endian_int64_shape_N_by_2',
                'candidate_rank_ranges': {'super_low': [0, 1024],
                                          'next_super_low': [1024, 2048]},
                'candidate_count_each': 1024, 'require_disjoint_groups': True,
                'source_counts_predeclared': False,
                'prior_very_low_ordered_pair_raw_int64_sha256':
                    '34ed2eef5b17535b8f4548567c89318e07514a01e531178386e3f03331e21454'} or
            loss['sample_count'] != 1024 or loss['sequence_length'] != 128 or
            loss['predictions_per_sample'] != 127 or loss['batch_size'] != 8 or
            loss['number_of_batches'] != 128 or loss['total_prediction_tokens'] != 130048 or
            loss['type'] != 'causal_next_token_cross_entropy' or
            loss['reduction'] != 'sum_token_cross_entropy_divided_by_total_prediction_tokens' or
            any(loss[key] is not False for key in
                ('mixed_precision', 'optimizer', 'weight_decay', 'gradient_clipping')) or
            spec['eligible_tensors']['expected_matrix_count'] != 98 or
            spec['tangent']['direction'] != 'delta = +alpha * g' or
            spec['tangent']['target_relative_frobenius'] != .00125 or
            spec['tangent']['normalization'] != 'one_global_alpha_across_all_eligible_matrices' or
            spec['candidate'] != {'tensor_order': list(NAMES), 'shape': [128, 2048],
                'dtype': 'contiguous_cpu_float32', 'response_normalization': False,
                'sign_selection': False, 'interpolation': False, 'reweighting': False,
                'combination': False} or
            spec['probe']['selected_rows'] != [0, 1024] or
            spec['probe']['selected_batch_count'] != 32 or
            spec['probe']['batch_size'] != 32 or
            spec['readout']['block_index'] != 13 or
            spec['readout']['hidden_state_index'] != 14 or
            spec['jvp']['strict_forward_ad'] is not True or
            spec['jvp']['fallback'] is not False or
            spec['barrier'] != {'marker': MARKER,
                'old_and_new_inputs_ranking_rows_gradients_responses_published_and_revalidated_before_historical_access': True,
                'manifest_contains_historical_scores': False} or
            evaluation['target_raw_sha256'] != TARGET_SHA256 or
            evaluation['target_probe_rows'] != [0, 1024] or
            evaluation['signed_positionwise_cosine'] is not True or
            evaluation['position_0_separate'] is not True or
            evaluation['primary_positions'] != [1, 2, 3, 4] or
            evaluation['secondary_positions'] != list(range(1, 128)) or
            evaluation['absolute_cosine'] is not False or
            evaluation['flattened_historical_cosine'] is not False or
            evaluation['stringency_gain'] != {'meaningful_both_min': .01,
                'positive_both_strictly_gt': 0.0} or
            evaluation['pool_expansion_gain'] != {'meaningful_both_min': .005,
                'positive_both_strictly_gt': 0.0} or
            evaluation['post_result_candidate_changes'] is not False or
            spec['workload'] != {'corpus_download_batches': 0,
                'corpus_freeze_gpu_batches': 0, 'new_surprisal_forward_batches': 1024,
                'gradient_backward_batches': 256, 'jvp_probe_batches': 64,
                'matched_target_forward_batches': 64, 'ggn_operations': 0,
                'cg_operations': 0} or
            spec['information_policy']['same_specimen_exploratory_method_development'] is not True or
            spec['information_policy']['independent_generic_data_expansion'] is not True or
            spec['information_policy']['clean_heldout_validation'] is not False or
            any(spec['information_policy'][key] is not False for key in
                ('historical_base_access_before_barrier', 'historical_target_access_before_barrier',
                 'adapter_access_before_barrier', 'true_delta_access_before_barrier',
                 'attempt127_129_130_131_historical_scores_before_barrier'))):
        raise ValueError('Attempt132 fixed expanded-pool plan mismatch')
    return spec


def refuse_outputs(spec):
    paths = {key: path_of(value) for key, value in spec['paths'].items()}
    if len(set(paths.values())) != 3 or any(p.exists() or p.is_symlink() for p in paths.values()):
        raise ValueError('Attempt132 construction/result output already exists or paths overlap')
    return paths


def validate_old_pool(spec, a131, a130, a126, a14, torch):
    """No historical result is read; only published blind constructions."""
    frozen = spec['frozen_attempt131']
    for key in ('spec', 'source', 'construction_manifest', 'candidate'):
        suffix = 'serialized_sha256' if key == 'candidate' else 'sha256'
        require_hash(path_of(frozen[key + '_path']), frozen[key + '_' + suffix])
    previous = a131.load_spec()
    for key in ('model', 'final_checkpoint', 'eligible_tensors', 'tangent', 'probe',
                'readout', 'jvp', 'helper_sources', 'frozen_attempt005',
                'frozen_attempt100', 'frozen_slices', 'frozen_matched_evaluator',
                'frozen_attempt130', 'source_pooled_selection'):
        if spec[key] != previous[key]:
            raise ValueError('Attempt131 frozen construction convention changed: ' + key)
    if spec['previous_stringency_selection'] != previous['stringency_selection']:
        raise ValueError('Attempt131 frozen VERY_LOW definition changed')
    source_tokens, probe, old_manifest, model_dir = a131.validate_frozen_attempt130(
        previous, a130, a126, a14, torch)
    manifest131 = json.loads(path_of(frozen['construction_manifest_path']).read_text())
    if (manifest131['attempt_id'] != '131_low_surprisal_stringency' or
            manifest131['spec_sha256'] != frozen['spec_sha256'] or
            manifest131['constructor_sha256'] != frozen['source_sha256'] or
            manifest131['candidate']['serialized_sha256'] !=
                frozen['candidate_serialized_sha256'] or
            manifest131['barrier'] != previous['barrier'] or
            manifest131['barrier']['manifest_contains_historical_scores'] is not False or
            manifest131['groups']['very_low']['ordered_pair_raw_int64_sha256'] !=
                spec['combined_selection']['prior_very_low_ordered_pair_raw_int64_sha256']):
        raise ValueError('Attempt131 blind construction provenance mismatch')
    old_artifact = a131.validate_published(
        {'candidate_path': path_of(frozen['candidate_path']),
         'construction_manifest_path': path_of(frozen['construction_manifest_path'])},
        manifest131, previous, a130, a126, a14, torch)
    del old_artifact
    require_hash(path_of(frozen['construction_manifest_path']),
                 frozen['construction_manifest_sha256'])
    return source_tokens, probe, old_manifest, manifest131, model_dir


def validate_new_corpus(spec, freezer, a14, torch):
    frozen = spec['new_corpus']
    require_hash(path_of(frozen['corpus_spec_path']), frozen['corpus_spec_sha256'])
    require_hash(path_of(frozen['freezer_source_path']), frozen['freezer_source_sha256'])
    corpus_spec = freezer.load_spec()
    manifest_path = path_of(frozen['corpus_manifest_path'])
    manifest = json.loads(manifest_path.read_text())
    expected_keys = {'format_version', 'attempt_id', 'hash_algorithm',
        'corpus_spec_sha256', 'freeze_script_sha256', 'dataset', 'tokenizer_checkpoint',
        'selection', 'tensor', 'artifact', 'old_pooled_valid_example_rank_interval',
        'new_valid_example_rank_interval', 'frozen_attempt005'}
    if (set(manifest) != expected_keys or manifest['format_version'] != 1 or
            manifest['attempt_id'] != ATTEMPT or manifest['hash_algorithm'] != 'sha256' or
            corpus_spec['artifact']['path'] != frozen['tokens_path'] or
            corpus_spec['old_pooled_valid_example_rank_interval'] != [20000, 28192] or
            corpus_spec['new_valid_example_rank_interval'] != [28192, 36384] or
            manifest['corpus_spec_sha256'] != frozen['corpus_spec_sha256'] or
            manifest['freeze_script_sha256'] != frozen['freezer_source_sha256'] or
            manifest['dataset'] != corpus_spec['dataset'] or
            manifest['selection'] != corpus_spec['selection'] or
            manifest['frozen_attempt005'] != corpus_spec['frozen_attempt005'] or
            manifest['tokenizer_checkpoint']['source'] != corpus_spec['tokenizer']['source'] or
            manifest['tokenizer_checkpoint']['directory'] != corpus_spec['tokenizer']['directory'] or
            manifest['tokenizer_checkpoint']['file_count'] != 4 or
            manifest['tokenizer_checkpoint']['files'] != corpus_spec['tokenizer_checkpoint_files'] or
            manifest['tensor'] != {'shape': [8192, 128], 'dtype': 'torch.int64',
                'device': 'cpu', 'contiguous': True} or
            manifest['artifact']['path'] != frozen['tokens_path'] or
            manifest['old_pooled_valid_example_rank_interval'] != [20000, 28192] or
            manifest['new_valid_example_rank_interval'] != [28192, 36384]):
        raise ValueError('Expanded FineWeb corpus provenance mismatch')
    token_path = path_of(frozen['tokens_path'])
    require_hash(token_path, manifest['artifact']['serialized_sha256'])
    tokens = a14.load_corpus_tokens(token_path,
        manifest['artifact']['raw_tensor_sha256'], 8192, 128, torch)
    if tuple(tokens.shape) != (8192, 128) or not tokens.is_contiguous():
        raise ValueError('Expanded FineWeb tokens malformed')
    return tokens, manifest, sha256_file(manifest_path)


def score_new_surprisal(model, tokens, scoring, a126, torch):
    if (tuple(tokens.shape) != (8192, 128) or tokens.dtype != torch.int64 or
            not tokens.is_contiguous() or scoring['number_of_forward_batches'] != 1024):
        raise ValueError('Expected full new 8192-row FineWeb population')
    model.eval()
    device = next(model.parameters()).device
    scores = np.empty(8192, dtype=np.float64)
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 8192, 8):
            batch = tokens[start:start + 8].to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            if (logits.dtype != torch.float32 or tuple(logits.shape[:2]) != (8, 128) or
                    not bool(torch.isfinite(logits).all())):
                raise ValueError('Nonfinite/malformed new endpoint logits')
            shifted = logits[:, :-1, :].contiguous().reshape(-1, logits.shape[-1])
            labels = batch[:, 1:].contiguous().reshape(-1)
            token_nll = torch.nn.functional.cross_entropy(
                shifted, labels, reduction='none').reshape(8, 127)
            cpu = token_nll.detach().to(device='cpu', dtype=torch.float64).numpy()
            scores[start:start + 8] = a126.sequence_mean_nll(cpu)
            del batch, logits, shifted, labels, token_nll, cpu
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite new endpoint surprisal')
    return scores


def rank_new_surprisal(scores, a126):
    values = np.asarray(scores)
    if (values.shape != (8192,) or values.dtype != np.float64 or
            not values.flags.c_contiguous or not np.isfinite(values).all()):
        raise ValueError('Expected 8192 finite contiguous float64 surprisals')
    rank = np.asarray(sorted(range(8192), key=lambda row: (float(values[row]), row)),
                      dtype=np.int64)
    return {'surprisal_values': values.tolist(),
            'surprisal_raw_float64_sha256': a126.raw_float64_hash(values),
            'ascending_rank_original_row_indices': rank.tolist(),
            'rank_raw_int64_sha256': a126.raw_int64_hash(rank),
            'population_rows': [0, 8192],
            'prediction_tokens_per_row': 127,
            'ranking_rule': ['surprisal_ascending', 'original_row_index_ascending']}


def validate_new_ranking(record, a126):
    values = np.asarray(record['surprisal_values'], dtype=np.float64)
    if rank_new_surprisal(values, a126) != record:
        raise ValueError('Published new surprisal values/ranking changed')
    return True


def combine_rankings(old_pooled, new_ranking, prior_very_low_pairs, selection, a130, a126):
    validate_new_ranking(new_ranking, a126)
    old_records = old_pooled['records']
    if (len(old_records) != 8192 or
            old_pooled['pair_hashes']['combined_rank_pair_raw_int64_sha256'] !=
                selection['old_combined_rank_pair_raw_int64_sha256'] or
            len(new_ranking['surprisal_values']) != 8192):
        raise ValueError('Old or new frozen ranking length/hash mismatch')
    old_pairs = np.ascontiguousarray([(r[1], r[2]) for r in old_records], dtype=np.int64)
    if (a130.pair_hash(old_pairs) != selection['old_combined_rank_pair_raw_int64_sha256'] or
            any(len(r) != 3 or r[1] not in (0, 1) or r[2] not in range(4096) or
                not math.isfinite(r[0]) for r in old_records) or
            len(set(map(tuple, old_pairs.tolist()))) != 8192):
        raise ValueError('Published old 8192-record ranking malformed')
    records = sorted((float(r[0]), int(r[1]), int(r[2])) for r in old_records)
    records.extend((float(value), 2, row)
                   for row, value in enumerate(new_ranking['surprisal_values']))
    records.sort()
    if len(records) != 16384:
        raise ValueError('Expected exactly 16384 combined ranking records')
    pairs = np.ascontiguousarray([(s, row) for _, s, row in records], dtype=np.int64)
    values = np.ascontiguousarray([value for value, _, _ in records], dtype=np.float64)
    if (len(set(map(tuple, pairs.tolist()))) != 16384 or
            sum(int(s == 2) for s in pairs[:, 0]) != 8192):
        raise ValueError('Combined ranking lost or repeated source rows')
    prior_set = set(map(tuple, prior_very_low_pairs))
    prior_array = np.ascontiguousarray(prior_very_low_pairs, dtype=np.int64)
    if (len(prior_set) != 1024 or
            a130.pair_hash(prior_array) !=
                selection['prior_very_low_ordered_pair_raw_int64_sha256']):
        raise ValueError('Attempt131 prior VERY_LOW row set changed')
    groups = {}
    for name in GROUPS:
        lo, hi = selection['candidate_rank_ranges'][name]
        subset = np.ascontiguousarray(pairs[lo:hi])
        surprisal = values[lo:hi]
        counts = {str(source_id): int(np.count_nonzero(subset[:, 0] == source_id))
                  for source_id in range(3)}
        if subset.shape != (1024, 2) or sum(counts.values()) != 1024:
            raise ValueError('Expanded stringency candidate has wrong size')
        groups[name] = {'rank_range': [lo, hi], 'ordered_pairs': subset.tolist(),
            'ordered_pair_raw_int64_sha256': a130.pair_hash(subset),
            'source_counts': counts,
            'surprisal_min': float(surprisal[0]),
            'surprisal_max': float(surprisal[-1]),
            'surprisal_mean': math.fsum(map(float, surprisal)) / 1024,
            'overlap_with_attempt131_very_low1024':
                len(set(map(tuple, subset.tolist())) & prior_set)}
    if set(map(tuple, groups[GROUPS[0]]['ordered_pairs'])) & set(map(tuple,
            groups[GROUPS[1]]['ordered_pairs'])):
        raise ValueError('Expanded stringency groups overlap')
    return {'records': [[value, s, row] for value, s, row in records],
        'ordered_pair_raw_int64_sha256': a130.pair_hash(pairs),
        'ordered_surprisal_raw_float64_sha256': a126.raw_float64_hash(values),
        'old_combined_rank_pair_raw_int64_sha256':
            selection['old_combined_rank_pair_raw_int64_sha256'],
        'new_surprisal_raw_float64_sha256': new_ranking['surprisal_raw_float64_sha256'],
        'new_rank_raw_int64_sha256': new_ranking['rank_raw_int64_sha256'],
        'groups': groups, 'groups_overlap_count': 0}


def selected_tokens(source_tokens, pairs, torch):
    if len(pairs) != 1024:
        raise ValueError('Expected exactly 1024 selected rows')
    names = ('old', 'fresh', 'new')
    selected = torch.stack([source_tokens[names[source_id]][row]
        for source_id, row in pairs]).contiguous()
    if selected.dtype != torch.int64 or tuple(selected.shape) != (1024, 128):
        raise ValueError('Selected expanded FineWeb tokens malformed')
    return selected


def validate_published(paths, construction, spec, old_pooled, prior_pairs,
                       a130, a126, a14, torch):
    candidate_path, manifest_path = paths['candidate_path'], paths['construction_manifest_path']
    if (sha256_file(candidate_path) != construction['candidate']['serialized_sha256'] or
            json.loads(manifest_path.read_text()) != construction or
            construction['spec_sha256'] != SPEC_SHA256 or
            construction['barrier'] != spec['barrier'] or
            construction['barrier']['manifest_contains_historical_scores'] is not False or
            construction['candidate']['tensor_order'] != list(NAMES) or
            list(construction['gradient_records']) != list(GROUPS) or
            list(construction['responses']) != list(NAMES)):
        raise ValueError('Published Attempt132 candidate/manifest changed')
    rebuilt = combine_rankings(old_pooled, construction['new_ranking'], prior_pairs,
        spec['combined_selection'], a130, a126)
    if construction['combined_ranking'] != rebuilt:
        raise ValueError('Published combined ranking/groups changed')
    artifact = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != list(NAMES):
        raise ValueError('Published Attempt132 response inventory/order changed')
    for group, name in zip(GROUPS, NAMES):
        tensor = a126.a122.prior.validate_response(artifact[name], torch)
        if (a14.sha256_raw_float32_tensor(tensor, torch) !=
                construction['responses'][name]['raw_sha256'] or
                construction['gradient_records'][group]['selected_pair_raw_sha256'] !=
                rebuilt['groups'][group]['ordered_pair_raw_int64_sha256']):
            raise ValueError('Published response/selection hash mismatch: ' + name)
    return artifact


def signed_metrics(report):
    values = report['all_128_position_cosines']
    if (len(values) != 128 or any(value is None or not math.isfinite(value)
            for value in values) or
            not math.isclose(report['positions_1_4_mean_cosine'],
                             math.fsum(values[1:5]) / 4, abs_tol=1e-10) or
            not math.isclose(report['positions_1_127_mean_cosine'],
                             math.fsum(values[1:]) / 127, abs_tol=1e-10)):
        raise ValueError('Malformed signed positionwise historical score')
    return {'primary': report['positions_1_4_mean_cosine'],
            'secondary': report['positions_1_127_mean_cosine']}


def stringency_comparison(reports, a131):
    if list(reports) != list(NAMES):
        raise ValueError('Exactly two expanded stringency reports required')
    aliases = {'response_very_low': reports[NAMES[0]],
               'response_next_low': reports[NAMES[1]]}
    return a131.stringency_comparison(aliases)


def pool_expansion_comparison(super_low, prior_very_low):
    current = signed_metrics(super_low)
    previous = signed_metrics(prior_very_low)
    gains = {key: current[key] - previous[key] for key in current}
    wins = sum(x > y for x, y in zip(
        super_low['all_128_position_cosines'][1:],
        prior_very_low['all_128_position_cosines'][1:]))
    category = ('meaningful_pool_expansion_gain' if all(x >= .005 for x in gains.values()) else
                'positive_pool_expansion_gain' if all(x > 0 for x in gains.values()) else
                'no_pool_expansion_gain')
    return {'current_super_low1024': current, 'attempt131_very_low1024': previous,
            'super_low_minus_attempt131_very_low': gains,
            'super_low_beats_attempt131_very_low_positions_1_127': wins,
            'classification': category,
            'interpretation': 'same_specimen_descriptive_development_only'}


def prior_context(spec, reports, a131):
    prior = spec['prior_scores_after_barrier']
    results = {}
    for number in (127, 129, 130, 131):
        key = f'attempt{number}_result'
        path = path_of(prior[key + '_path'])
        results[number] = json.loads(require_hash(path, prior[key + '_sha256']).read_text())
    expected_attempts = {127: '127_postfreeze_surprisal_strata_evaluation',
        129: '129_fresh_fineweb_low_surprisal_replication',
        130: '130_pooled_fineweb_low_surprisal_scaling',
        131: '131_low_surprisal_stringency'}
    for number, result in results.items():
        if (result['attempt_id'] != expected_attempts[number] or
                result.get('matched_target', {}).get('raw_sha256',
                    result.get('matched_target_raw_sha256')) != TARGET_SHA256):
            raise ValueError('Prior matched historical result provenance mismatch')
    previous = results[131]['signed_position_reports']['response_very_low']
    expansion = pool_expansion_comparison(reports[NAMES[0]], previous)
    context = {}
    for label, number, name in (
        ('attempt131_next_low1024', 131, 'response_next_low'),
        ('attempt130_low2048', 130, 'response_low2048'),
        ('attempt129_fresh_low1024', 129, 'response_low'),
        ('attempt127_old_low1024', 127, 'response_low')):
        context[label] = signed_metrics(results[number]['signed_position_reports'][name])
    consensus = results[131]['attempt100_eight_corpus_consensus_context_only']
    if (consensus['role'] != 'context_only_different_probe_construction_population' or
            not all(math.isfinite(consensus[key]) for key in ('primary', 'secondary'))):
        raise ValueError('Attempt100 context-only provenance mismatch')
    context['attempt100_eight_corpus_consensus_context_only'] = consensus
    return expansion, context


def evaluate_after_barrier(spec, artifact, a126, a131, torch):
    frozen = spec['frozen_matched_evaluator']
    require_hash(path_of(frozen['spec_path']), frozen['spec_sha256'])
    a123 = import_pinned(path_of(frozen['source_path']), frozen['source_sha256'],
                         'attempt132_frozen_matched_evaluator')
    matched_spec = a123.load_spec()
    reference = matched_spec['matched_reference']
    privileged = import_pinned(path_of(reference['attempt103_source_path']),
        reference['attempt103_source_sha256'], 'attempt132_frozen_privileged_reference')
    frozen122 = matched_spec['frozen_attempt122']
    require_hash(path_of(frozen122['construction_manifest_path']),
                 frozen122['construction_manifest_sha256'])
    manifest122 = json.loads(path_of(frozen122['construction_manifest_path']).read_text())
    old103, inventories = a123.validate_matched_reference(matched_spec, manifest122, privileged)
    if (old103['final_checkpoint_files'] != spec['final_checkpoint']['files'] or
            old103['readout'] != spec['readout'] or old103['probe_rows'] != [0, 1024] or
            reference['target_raw_sha256'] != TARGET_SHA256 or
            reference['batch_size'] != 32 or reference['batches_per_model'] != 32):
        raise ValueError('Matched historical target provenance mismatch')
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
        raise ValueError('Matched historical target hash mismatch')
    for directory, expected in ((privileged.resolve(old103['paths']['base_directory']),
                                 inventories['base_files']),
                                (privileged.resolve(old103['paths']['final_directory']),
                                 inventories['final_files'])):
        if privileged.a.checkpoint_file_records(directory) != expected:
            raise ValueError('Historical checkpoint changed during target construction')
    del base, final, final_mean, base_mean, probe
    gc.collect()
    torch.cuda.empty_cache()
    old_eval = a126.a122.load_spec()['evaluation']
    validator = import_pinned(path_of(old_eval['validator_path']),
        old_eval['validator_sha256'], 'attempt132_signed_position_validator')
    reports = {name: a126.a122.score_report(artifact[name], target, validator)
               for name in NAMES}
    gains, stringency_category, wins = stringency_comparison(reports, a131)
    expansion, context = prior_context(spec, reports, a131)
    return reports, gains, stringency_category, wins, expansion, context, target_hash, load_seconds


def run():
    import torch
    spec = load_spec()
    paths = refuse_outputs(spec)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for frozen FP32 strict JVP convention')
    frozen131, frozen130 = spec['frozen_attempt131'], spec['frozen_attempt130']
    a131 = import_pinned(path_of(frozen131['source_path']), frozen131['source_sha256'],
                         'attempt132_frozen_attempt131_helper')
    a130 = import_pinned(path_of(frozen130['source_path']), frozen130['source_sha256'],
                         'attempt132_frozen_attempt130_helper')
    old = spec['frozen_slices']['old']
    a126 = import_pinned(path_of(old['constructor_path']), old['constructor_sha256'],
                         'attempt132_frozen_surprisal_helper')
    helper = spec['helper_sources']
    a14 = import_pinned(path_of(helper['attempt014_path']), helper['attempt014_sha256'],
                        'attempt132_frozen_gradient_jvp_helper')
    new_corpus = spec['new_corpus']
    freezer = import_pinned(path_of(new_corpus['freezer_source_path']),
        new_corpus['freezer_source_sha256'], 'attempt132_frozen_corpus_freezer')
    source_tokens, probe, old_manifest, prior_manifest, model_dir = validate_old_pool(
        spec, a131, a130, a126, a14, torch)
    new_tokens, corpus_manifest, corpus_manifest_hash = validate_new_corpus(
        spec, freezer, a14, torch)
    source_tokens['new'] = new_tokens
    model = a14.load_local_model(model_dir, 'cuda', torch)
    eligible = a14.discover_eligible_linear_weights(model, torch)
    a14.freeze_other_parameters(model, eligible)
    before = a14.model_state_hashes(model, torch)
    surprisal = score_new_surprisal(model, new_tokens, spec['new_surprisal_scoring'], a126, torch)
    new_ranking = rank_new_surprisal(surprisal, a126)
    validate_new_ranking(new_ranking, a126)
    del surprisal
    combined = combine_rankings(old_manifest['pooled_ranking'], new_ranking,
        prior_manifest['groups']['very_low']['ordered_pairs'],
        spec['combined_selection'], a130, a126)
    responses, gradient_records = {}, {}
    primal_reference = None
    jvp_spec = {**spec, 'probe': {**spec['probe'], 'sample_count': 1024}}
    for group in GROUPS:
        subset = selected_tokens(source_tokens, combined['groups'][group]['ordered_pairs'], torch)
        mean_ce = a14.accumulate_mean_generic_gradient(
            model, subset, eligible, spec['gradient_loss'], torch)
        scale = a126.validate_global_scale(a14.global_tangent_scale(eligible, .00125, torch))
        if not math.isclose(scale['aggregate_source_weight_norm'], 918.1587250300843,
                            rel_tol=1e-8, abs_tol=1e-6):
            raise ValueError('Final eligible-weight norm differs from Attempt005')
        tangents, matrices, realized = a14.prepare_tangents(eligible, scale, torch)
        try:
            primal, response, readout = a14.compute_probe_response(
                model, probe, tangents, jvp_spec, torch)
        finally:
            tangents.clear()
            model.zero_grad(set_to_none=True)
        if primal_reference is None:
            primal_reference = primal
        elif not torch.equal(primal_reference, primal):
            raise ValueError('Unchanged-final JVP primal differs between candidates')
        response32 = a126.a122.prior.validate_response(
            response.to(torch.float32).contiguous(), torch)
        name = 'response_' + group
        responses[name] = response32
        gradient_records[group] = {'mean_generic_ce': mean_ce, **scale, **realized,
            'selected_pair_raw_sha256': combined['groups'][group]['ordered_pair_raw_int64_sha256'],
            'eligible_matrix_count': len(matrices), 'readout_checks': readout,
            'response_norm': float(torch.linalg.vector_norm(response32.double())),
            'response_raw_sha256': a14.sha256_raw_float32_tensor(response32, torch)}
        del subset, tangents, matrices, primal, response, response32
        gc.collect()
    if list(responses) != list(NAMES):
        raise ValueError('Fixed two-response inventory incomplete')
    pairwise = a126.response_cosine(responses[NAMES[0]], responses[NAMES[1]], torch)
    require_hash(path_of(new_corpus['tokens_path']),
                 corpus_manifest['artifact']['serialized_sha256'])
    require_hash(path_of(new_corpus['corpus_manifest_path']), corpus_manifest_hash)
    require_hash(path_of(frozen130['construction_manifest_path']),
                 frozen130['construction_manifest_sha256'])
    require_hash(path_of(frozen131['construction_manifest_path']),
                 frozen131['construction_manifest_sha256'])
    require_hash(path_of(spec['probe']['path']), spec['probe']['serialized_sha256'])
    if a14.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files']:
        raise ValueError('Final checkpoint changed during blind construction')
    a14.verify_model_unchanged(model, before, torch)
    require_hash(SPEC_PATH, SPEC_SHA256)
    a126.a122.prior.atomic_torch_publish(paths['candidate_path'], responses, torch)
    construction = {'format_version': 1, 'attempt_id': ATTEMPT,
        'spec_sha256': SPEC_SHA256, 'constructor_sha256': sha256_file(Path(__file__)),
        'information_policy': spec['information_policy'],
        'frozen_attempt130_construction_manifest_sha256':
            frozen130['construction_manifest_sha256'],
        'frozen_attempt131_construction_manifest_sha256':
            frozen131['construction_manifest_sha256'],
        'old_combined_rank_pair_raw_int64_sha256':
            spec['combined_selection']['old_combined_rank_pair_raw_int64_sha256'],
        'old_valid_example_rank_interval': [20000, 28192],
        'new_valid_example_rank_interval': [28192, 36384],
        'new_corpus_spec_sha256': new_corpus['corpus_spec_sha256'],
        'new_corpus_freezer_sha256': new_corpus['freezer_source_sha256'],
        'new_corpus_manifest_sha256': corpus_manifest_hash,
        'new_corpus_serialized_sha256': corpus_manifest['artifact']['serialized_sha256'],
        'new_corpus_raw_tensor_sha256': corpus_manifest['artifact']['raw_tensor_sha256'],
        'final_checkpoint_files': spec['final_checkpoint']['files'],
        'new_ranking': new_ranking, 'combined_ranking': combined,
        'gradient_loss': spec['gradient_loss'], 'tangent': spec['tangent'],
        'probe': spec['probe'], 'readout': spec['readout'], 'jvp': spec['jvp'],
        'gradient_records': gradient_records,
        'responses': {name: {'raw_sha256': gradient_records[group]['response_raw_sha256'],
            'shape': [128, 2048], 'dtype': 'contiguous_cpu_float32',
            'norm': gradient_records[group]['response_norm']}
            for group, name in zip(GROUPS, NAMES)},
        'oracle_free_flattened_response_cosine': pairwise,
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
    artifact = validate_published(paths, construction, spec, old_manifest['pooled_ranking'],
        prior_manifest['groups']['very_low']['ordered_pairs'], a130, a126, a14, torch)
    require_hash(paths['construction_manifest_path'], manifest_hash)
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), construction['constructor_sha256'])
    for key in ('corpus_spec', 'freezer_source'):
        require_hash(path_of(new_corpus[key + '_path']), new_corpus[key + '_sha256'])
    require_hash(path_of(new_corpus['corpus_manifest_path']), corpus_manifest_hash)
    require_hash(path_of(new_corpus['tokens_path']),
                 corpus_manifest['artifact']['serialized_sha256'])
    for number in (130, 131):
        frozen = spec[f'frozen_attempt{number}']
        for key in ('spec', 'source', 'construction_manifest', 'candidate'):
            suffix = 'serialized_sha256' if key == 'candidate' else 'sha256'
            require_hash(path_of(frozen[key + '_path']), frozen[key + '_' + suffix])
    for label in ('old', 'fresh'):
        frozen = spec['frozen_slices'][label]
        require_hash(path_of(frozen['tokens_path']), frozen['tokens_serialized_sha256'])
    require_hash(path_of(spec['probe']['path']), spec['probe']['serialized_sha256'])
    if a14.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files']:
        raise ValueError('Final checkpoint changed before construction barrier')
    del model, eligible, source_tokens, new_tokens, probe, responses, primal_reference
    gc.collect()
    torch.cuda.empty_cache()
    print(MARKER, flush=True)

    reports, gains, stringency, wins, expansion, context, target_hash, load_seconds = (
        evaluate_after_barrier(spec, artifact, a126, a131, torch))
    require_hash(paths['candidate_path'], construction['candidate']['serialized_sha256'])
    require_hash(paths['construction_manifest_path'], manifest_hash)
    result = {'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'same_specimen_exploratory_method_development': True,
        'independent_generic_data_expansion': True,
        'clean_heldout_validation': False,
        'construction_manifest_sha256': manifest_hash,
        'candidate_serialized_sha256': construction['candidate']['serialized_sha256'],
        'new_corpus_manifest_sha256': corpus_manifest_hash,
        'matched_target_raw_sha256': target_hash,
        'matched_model_load_seconds': load_seconds,
        'signed_position_reports': reports,
        'super_low_minus_next_super_low': gains,
        'super_low_beats_next_super_low_positions_1_127': wins,
        'stringency_classification': stringency,
        'stringency_prediction_replicated': all(value > 0 for value in gains.values()),
        'pool_expansion_comparison': expansion,
        'pool_expansion_improved_over_attempt131_very_low':
            all(value > 0 for value in expansion['super_low_minus_attempt131_very_low'].values()),
        'prior_historical_context': context,
        'groups': combined['groups'],
        'new_surprisal_raw_float64_sha256': new_ranking['surprisal_raw_float64_sha256'],
        'combined_rank_pair_raw_int64_sha256': combined['ordered_pair_raw_int64_sha256'],
        'gradient_records': gradient_records,
        'oracle_free_flattened_response_cosine': pairwise,
        'workload': spec['workload']}
    if paths['result_path'].exists() or paths['result_path'].is_symlink():
        raise ValueError('Attempt132 result already exists')
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
