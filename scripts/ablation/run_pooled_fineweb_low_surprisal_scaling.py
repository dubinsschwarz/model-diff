#!/usr/bin/env python3
"""Attempt130: frozen-surprisal pooled FineWeb scaling, then matched evaluation."""
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
ATTEMPT = '130_pooled_fineweb_low_surprisal_scaling'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = '4c569a7e4c00218256a4a1e243ef9a5283050f2c9d95a19b8bc308f2dc5bd942'
NAMES = ('response_low2048', 'response_control2048')
TARGET_SHA256 = '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f'
MARKER = 'POOLED_LOW_SURPRISAL_CANDIDATES_FROZEN'


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
    selection = spec['pooled_selection']
    loss = spec['gradient_loss']
    evaluation = spec['evaluation']
    paths = {key: f'experiments/attempts/{ATTEMPT}/{filename}' for key, filename in
             (('construction_manifest_path', 'construction-manifest.json'),
              ('candidate_path', 'candidate.pt'), ('result_path', 'result.json'))}
    if (spec['attempt_id'] != ATTEMPT or
            spec['paths'] != paths or
            spec['candidate'] != {'tensor_order': list(NAMES), 'shape': [128, 2048],
                'dtype': 'contiguous_cpu_float32', 'response_normalization': False,
                'sign_selection': False, 'interpolation': False, 'reweighting': False,
                'combination': False} or
            selection['source_rows_per_slice'] != 4096 or
            selection['total_rows'] != 8192 or
            selection['sort_key'] != ['surprisal_ascending', 'slice_id_ascending',
                                      'original_row_index_ascending'] or
            selection['low_rank_range'] != [0, 2048] or
            selection['low_source_counts'] != {'old': 1016, 'fresh': 1032} or
            selection['control_definition'] !=
                'old_original_rows_0_1024_then_fresh_original_rows_0_1024' or
            selection['pair_dtype'] !=
                'contiguous_cpu_little_endian_int64_shape_N_by_2' or
            [spec['frozen_slices'][key]['valid_example_rank_interval'] for key in ('old', 'fresh')]
                != [[20000, 24096], [24096, 28192]] or
            [spec['frozen_slices'][key]['slice_id'] for key in ('old', 'fresh')] != [0, 1] or
            loss['sample_count'] != 2048 or loss['sequence_length'] != 128 or
            loss['predictions_per_sample'] != 127 or loss['batch_size'] != 8 or
            loss['number_of_batches'] != 256 or loss['total_prediction_tokens'] != 260096 or
            loss['type'] != 'causal_next_token_cross_entropy' or
            loss['reduction'] != 'sum_token_cross_entropy_divided_by_total_prediction_tokens' or
            any(loss[key] is not False for key in
                ('mixed_precision', 'optimizer', 'weight_decay', 'gradient_clipping')) or
            spec['eligible_tensors']['expected_matrix_count'] != 98 or
            spec['tangent']['direction'] != 'delta = +alpha * g' or
            spec['tangent']['target_relative_frobenius'] != .00125 or
            spec['tangent']['normalization'] != 'one_global_alpha_across_all_eligible_matrices' or
            spec['probe']['sample_count'] != 10000 or
            spec['probe']['selected_rows'] != [0, 1024] or
            spec['probe']['selected_batch_count'] != 32 or
            spec['probe']['batch_size'] != 32 or
            spec['readout']['block_index'] != 13 or
            spec['readout']['hidden_state_index'] != 14 or
            spec['jvp']['strict_forward_ad'] is not True or
            spec['jvp']['fallback'] is not False or
            spec['barrier'] != {'marker': MARKER,
                'frozen_inputs_ranking_rows_gradients_responses_published_and_revalidated_before_historical_access': True,
                'manifest_contains_historical_scores': False} or
            evaluation['target_probe_rows'] != [0, 1024] or
            evaluation['target_raw_sha256'] != TARGET_SHA256 or
            evaluation['signed_positionwise_cosine'] is not True or
            evaluation['position_0_separate'] is not True or
            evaluation['primary_positions'] != [1, 2, 3, 4] or
            evaluation['secondary_positions'] != list(range(1, 128)) or
            evaluation['absolute_cosine'] is not False or
            evaluation['flattened_historical_cosine'] is not False or
            evaluation['selection_gain'] != {'meaningful_both_min': .01,
                'positive_both_strictly_gt': 0.0} or
            evaluation['scaling_gain'] != {'clear': 'exceeds_both_prior_low1024_on_both_metrics',
                'average': 'otherwise_exceeds_arithmetic_mean_on_both_metrics',
                'otherwise': 'no_scaling_gain'} or
            evaluation['post_result_candidate_changes'] is not False or
            spec['workload'] != {'corpus_download_batches': 0, 'corpus_freeze_batches': 0,
                'surprisal_scoring_batches': 0, 'gradient_backward_batches': 512,
                'jvp_probe_batches': 64, 'matched_target_forward_batches': 64,
                'ggn_operations': 0, 'cg_operations': 0} or
            spec['information_policy']['same_specimen_exploratory_method_development'] is not True or
            spec['information_policy']['clean_heldout_validation'] is not False or
            any(spec['information_policy'][key] is not False for key in
                ('historical_base_access_before_barrier', 'historical_target_access_before_barrier',
                 'adapter_access_before_barrier', 'true_delta_access_before_barrier',
                 'attempt127_129_oracle_scores_before_barrier'))):
        raise ValueError('Attempt130 fixed pooled plan mismatch')
    return spec


def refuse_outputs(spec):
    paths = {key: path_of(value) for key, value in spec['paths'].items()}
    if len(set(paths.values())) != 3 or any(p.exists() or p.is_symlink() for p in paths.values()):
        raise ValueError('Attempt130 output already exists or paths overlap')
    return paths


def validate_blind_inputs(spec, a126, a14, torch):
    """All source checks use construction artifacts only; prior scores stay sealed."""
    slices = spec['frozen_slices']
    old_spec = json.loads(require_hash(path_of(slices['old']['construction_spec_path']),
        slices['old']['construction_spec_sha256']).read_text())
    fresh_spec = json.loads(require_hash(path_of(slices['fresh']['construction_spec_path']),
        slices['fresh']['construction_spec_sha256']).read_text())
    for key in ('model', 'final_checkpoint', 'eligible_tensors', 'tangent', 'probe',
                'readout', 'jvp', 'helper_sources'):
        if spec[key] != old_spec[key] or spec[key] != fresh_spec[key]:
            raise ValueError('Frozen construction convention mismatch: ' + key)
    if spec['frozen_attempt005'] != old_spec['frozen_attempt005'] or spec['frozen_attempt100'] != old_spec['frozen_attempt100']:
        raise ValueError('Attempt005/100 construction provenance mismatch')
    source_specs, source_manifests, construction, tokens = {}, {}, {}, {}
    for label in ('old', 'fresh'):
        frozen = slices[label]
        require_hash(path_of(frozen['constructor_path']), frozen['constructor_sha256'])
        corpus_spec = json.loads(require_hash(path_of(frozen['corpus_spec_path']),
            frozen['corpus_spec_sha256']).read_text())
        corpus_manifest = json.loads(require_hash(path_of(frozen['corpus_manifest_path']),
            frozen['corpus_manifest_sha256']).read_text())
        built = json.loads(require_hash(path_of(frozen['construction_manifest_path']),
            frozen['construction_manifest_sha256']).read_text())
        require_hash(path_of(frozen['candidate_path']), frozen['candidate_serialized_sha256'])
        require_hash(path_of(frozen['tokens_path']), frozen['tokens_serialized_sha256'])
        if (corpus_manifest['corpus_spec_sha256'] != frozen['corpus_spec_sha256'] or
                corpus_manifest['dataset'] != corpus_spec['dataset'] or
                corpus_manifest['selection'] != corpus_spec['selection'] or
                corpus_manifest['tensor'] != {'shape': [4096, 128], 'dtype': 'torch.int64',
                    'device': 'cpu', 'contiguous': True} or
                corpus_manifest['artifact'] != {'path': frozen['tokens_path'],
                    'serialized_sha256': frozen['tokens_serialized_sha256'],
                    'raw_tensor_sha256': frozen['tokens_raw_sha256']} or
                built['spec_sha256'] != frozen['construction_spec_sha256'] or
                built['constructor_sha256'] != frozen['constructor_sha256'] or
                built['candidate']['serialized_sha256'] != frozen['candidate_serialized_sha256'] or
                built['final_checkpoint_files'] != spec['final_checkpoint']['files'] or
                built['barrier']['manifest_contains_historical_scores'] is not False or
                built['ranking']['surprisal_raw_float64_sha256'] !=
                    frozen['surprisal_raw_float64_sha256']):
            raise ValueError('Frozen source corpus/construction mismatch: ' + label)
        a126.validate_ranking(built['ranking'])
        source_specs[label], source_manifests[label], construction[label] = (
            corpus_spec, corpus_manifest, built)
        tokens[label] = a14.load_corpus_tokens(path_of(frozen['tokens_path']),
            frozen['tokens_raw_sha256'], 4096, 128, torch)
    old, fresh = source_specs['old'], source_specs['fresh']
    old_freezer = fresh['frozen_attempt005']
    fresh_freezer = fresh_spec['fresh_corpus']
    if (old_freezer['corpus_spec_sha256'] != slices['old']['corpus_spec_sha256'] or
            old_freezer['corpus_manifest_sha256'] != slices['old']['corpus_manifest_sha256'] or
            source_manifests['old']['freeze_script_sha256'] !=
                old_freezer['freezer_source_sha256'] or
            source_manifests['fresh']['freeze_script_sha256'] !=
                fresh_freezer['freezer_source_sha256'] or
            source_manifests['old']['tokenizer_checkpoint']['files'] !=
                source_manifests['fresh']['tokenizer_checkpoint']['files'] or
            source_manifests['fresh']['tokenizer_checkpoint']['files'] !=
                fresh['tokenizer_checkpoint_files']):
        raise ValueError('Frozen FineWeb freezer or tokenizer provenance changed')
    require_hash(path_of(old_freezer['freezer_source_path']),
                 old_freezer['freezer_source_sha256'])
    require_hash(path_of(fresh_freezer['freezer_source_path']),
                 fresh_freezer['freezer_source_sha256'])
    expected_dataset = {'repo_id': 'science-of-finetuning/fineweb-1m-sample',
        'revision': '60b53a86b84eb6559e4407b113356f56a152318f',
        'split': 'train', 'streaming': False}
    expected_selection = {'shuffle_seed': 42, 'text_column': 'text',
        'skip_blank_text': True, 'character_limit': 1280,
        'add_special_tokens': True, 'minimum_token_count': 128,
        'sample_count': 4096, 'sequence_length': 128}
    if (old['dataset'] != fresh['dataset'] or old['dataset'] != expected_dataset or
            old['tokenizer'] != fresh['tokenizer'] or
            old['selection'] != {**expected_selection, 'skip_valid_examples': 20000} or
            fresh['selection'] != {**expected_selection, 'skip_valid_examples': 24096} or
            fresh['old_valid_example_rank_interval'] != [20000, 24096] or
            fresh['fresh_valid_example_rank_interval'] != [24096, 28192] or
            source_manifests['fresh']['old_valid_example_rank_interval'] != [20000, 24096] or
            source_manifests['fresh']['fresh_valid_example_rank_interval'] != [24096, 28192] or
            construction['fresh']['fresh_corpus_manifest_sha256'] !=
                slices['fresh']['corpus_manifest_sha256'] or
            construction['old']['fineweb_corpus_provenance']['manifest_sha256'] !=
                slices['old']['corpus_manifest_sha256']):
        raise ValueError('FineWeb provenance or nonoverlapping valid-example ranks changed')
    model_dir = path_of(spec['final_checkpoint']['directory'])
    if a14.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files']:
        raise ValueError('Canonical final checkpoint changed')
    probe_spec = {key: spec['probe'][key] for key in
        ('sample_count', 'sequence_length', 'batch_size', 'dtype',
         'serialized_sha256', 'raw_tensor_sha256')}
    full_probe = a14.load_probe(path_of(spec['probe']['path']), probe_spec, torch)
    if tuple(full_probe.shape) != (10000, 128):
        raise ValueError('Full frozen probe malformed')
    return tokens, full_probe[:1024].contiguous(), construction, model_dir


def pair_hash(pairs):
    array = np.asarray(pairs)
    if (array.dtype != np.int64 or array.ndim != 2 or array.shape[1] != 2 or
            not array.flags.c_contiguous):
        raise ValueError('Expected contiguous int64 slice/row pairs')
    return hashlib.sha256(array.astype('<i8', copy=False).tobytes(order='C')).hexdigest()


def sorted_combined_records(values):
    if (len(values) != 2 or any(len(source) != 4096 for source in values) or
            any(not math.isfinite(value) for source in values for value in source)):
        raise ValueError('Expected two complete finite frozen surprisal arrays')
    return sorted((float(value), slice_id, row)
                  for slice_id, source in enumerate(values)
                  for row, value in enumerate(source))


def pooled_ranking(old_ranking, fresh_ranking, selection, a126):
    a126.validate_ranking(old_ranking)
    a126.validate_ranking(fresh_ranking)
    values = (old_ranking['surprisal_values'], fresh_ranking['surprisal_values'])
    records = sorted_combined_records(values)
    if len(records) != 8192:
        raise ValueError('Combined ranking must contain exactly 8192 rows')
    pairs = np.asarray([(s, row) for _, s, row in records], dtype=np.int64)
    low = np.ascontiguousarray(pairs[:2048])
    control = np.ascontiguousarray([(s, row) for s in range(2) for row in range(1024)],
                                   dtype=np.int64)
    counts = {'old': int(np.count_nonzero(low[:, 0] == 0)),
              'fresh': int(np.count_nonzero(low[:, 0] == 1))}
    overlap = len(set(map(tuple, low.tolist())) & set(map(tuple, control.tolist())))
    hashes = {'combined_rank_pair_raw_int64_sha256': pair_hash(pairs),
              'low_selected_pairs_raw_int64_sha256': pair_hash(low),
              'control_selected_pairs_raw_int64_sha256': pair_hash(control)}
    if (counts != selection['low_source_counts'] or overlap != selection['low_control_overlap_count'] or
            any(hashes[key] != selection[key] for key in hashes) or
            len(set(map(tuple, pairs.tolist()))) != 8192 or
            low.shape != (2048, 2) or control.shape != (2048, 2)):
        raise ValueError('Pinned pooled LOW/CONTROL row selection mismatch')
    return {'records': [[score, s, row] for score, s, row in records],
        'pair_hashes': hashes, 'low_source_counts': counts,
        'low_control_overlap_count': overlap,
        'low_pairs': low.tolist(), 'control_pairs': control.tolist()}


def selected_tokens(source_tokens, pairs, torch):
    if len(pairs) != 2048:
        raise ValueError('Expected exactly 2048 selected rows')
    selected = torch.stack([source_tokens['old' if slice_id == 0 else 'fresh'][row]
        for slice_id, row in pairs]).contiguous()
    if selected.dtype != torch.int64 or tuple(selected.shape) != (2048, 128):
        raise ValueError('Selected FineWeb tokens malformed')
    return selected


def validate_published(paths, construction, selection, a126, a14, torch):
    candidate_path, manifest_path = paths['candidate_path'], paths['construction_manifest_path']
    if (sha256_file(candidate_path) != construction['candidate']['serialized_sha256'] or
            json.loads(manifest_path.read_text()) != construction or
            construction['candidate']['tensor_order'] != list(NAMES) or
            construction['barrier']['marker'] != MARKER or
            construction['barrier']['manifest_contains_historical_scores'] is not False or
            list(construction['gradient_records']) != ['low2048', 'control2048'] or
            list(construction['responses']) != list(NAMES)):
        raise ValueError('Published Attempt130 candidate/manifest changed')
    frozen = construction['frozen_source_rankings']
    rebuilt = pooled_ranking(frozen['old'], frozen['fresh'], selection, a126)
    if rebuilt != construction['pooled_ranking']:
        raise ValueError('Published pooled rank or row sets changed')
    artifact = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != list(NAMES):
        raise ValueError('Published two-response inventory/order changed')
    for name in NAMES:
        tensor = a126.a122.prior.validate_response(artifact[name], torch)
        if a14.sha256_raw_float32_tensor(tensor, torch) != construction['responses'][name]['raw_sha256']:
            raise ValueError('Published response hash mismatch: ' + name)
    for group, hash_key in (('low2048', 'low_selected_pairs_raw_int64_sha256'),
                            ('control2048', 'control_selected_pairs_raw_int64_sha256')):
        if construction['gradient_records'][group]['selected_pair_raw_sha256'] != selection[hash_key]:
            raise ValueError('Published gradient row selection changed: ' + group)
    return artifact


def score_comparison(reports):
    if list(reports) != list(NAMES):
        raise ValueError('Exactly two signed response reports required')
    low, control = (reports[name] for name in NAMES)
    for report in (low, control):
        values = report['all_128_position_cosines']
        if (len(values) != 128 or any(x is None or not math.isfinite(x) for x in values) or
                not math.isclose(report['positions_1_4_mean_cosine'],
                                 math.fsum(values[1:5]) / 4, abs_tol=1e-10) or
                not math.isclose(report['positions_1_127_mean_cosine'],
                                 math.fsum(values[1:]) / 127, abs_tol=1e-10)):
            raise ValueError('Malformed signed positionwise report')
    gain = {'primary': low['positions_1_4_mean_cosine'] - control['positions_1_4_mean_cosine'],
            'secondary': low['positions_1_127_mean_cosine'] - control['positions_1_127_mean_cosine']}
    category = ('meaningful_selection_gain' if all(x >= .01 for x in gain.values()) else
                'positive_selection_gain' if all(x > 0 for x in gain.values()) else
                'no_selection_gain')
    wins = sum(x > y for x, y in zip(low['all_128_position_cosines'][1:],
                                    control['all_128_position_cosines'][1:]))
    return gain, category, wins


def scaling_comparison(pooled, old, fresh):
    previous = []
    for result, attempt in ((old, '127_postfreeze_surprisal_strata_evaluation'),
                            (fresh, '129_fresh_fineweb_low_surprisal_replication')):
        if (result['attempt_id'] != attempt or
                result.get('matched_target', {}).get('raw_sha256',
                    result.get('matched_target_raw_sha256')) != TARGET_SHA256):
            raise ValueError('Prior matched LOW1024 score provenance mismatch')
        report = result['signed_position_reports']['response_low']
        previous.append({'primary': report['positions_1_4_mean_cosine'],
                         'secondary': report['positions_1_127_mean_cosine']})
    low = {'primary': pooled['positions_1_4_mean_cosine'],
           'secondary': pooled['positions_1_127_mean_cosine']}
    if any(not math.isfinite(v) for row in [low, *previous] for v in row.values()):
        raise ValueError('Nonfinite scaling comparison score')
    mean = {m: math.fsum(row[m] for row in previous) / 2 for m in low}
    best = {m: max(row[m] for row in previous) for m in low}
    gains = {'minus_mean_low1024': {m: low[m] - mean[m] for m in low},
             'minus_best_low1024': {m: low[m] - best[m] for m in low}}
    category = ('clear_scaling_gain' if all(low[m] > best[m] for m in low) else
                'average_scaling_gain' if all(low[m] > mean[m] for m in low) else
                'no_scaling_gain')
    return {'prior_low1024': {'old': previous[0], 'fresh': previous[1]},
            'arithmetic_mean_low1024': mean, 'best_per_metric_low1024': best,
            'pooled_low2048': low, 'gains': gains, 'classification': category,
            'interpretation': 'same_specimen_descriptive_development_only'}


def evaluate_after_barrier(spec, artifact, a126, torch):
    frozen = spec['frozen_matched_evaluator']
    require_hash(path_of(frozen['spec_path']), frozen['spec_sha256'])
    a123 = import_pinned(path_of(frozen['source_path']), frozen['source_sha256'],
                         'attempt130_frozen_matched_evaluator')
    matched_spec = a123.load_spec()
    reference = matched_spec['matched_reference']
    privileged = import_pinned(path_of(reference['attempt103_source_path']),
        reference['attempt103_source_sha256'], 'attempt130_frozen_privileged_reference')
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
        old_eval['validator_sha256'], 'attempt130_signed_position_validator')
    reports = {name: a126.a122.score_report(artifact[name], target, validator)
               for name in NAMES}
    gain, selection_category, wins = score_comparison(reports)
    prior = spec['prior_scores_after_barrier']
    for label in ('attempt127_result', 'attempt129_result', 'attempt100_evaluation'):
        require_hash(path_of(prior[label + '_path']), prior[label + '_sha256'])
    old = json.loads(path_of(prior['attempt127_result_path']).read_text())
    fresh = json.loads(path_of(prior['attempt129_result_path']).read_text())
    scaling = scaling_comparison(reports[NAMES[0]], old, fresh)
    context = json.loads(path_of(prior['attempt100_evaluation_path']).read_text())
    if context['attempt_id'] != '100_diverse8_raw_jg_consensus_prefix0_13':
        raise ValueError('Attempt100 context provenance mismatch')
    consensus = next(row for row in context['candidates'] if row['candidate'] == 'consensus_response_8')
    context_score = {'primary': consensus['positions_1_4_mean_cosine'],
                     'secondary': consensus['positions_1_127_mean_cosine'],
                     'role': 'context_only_different_probe_construction_population'}
    return reports, gain, selection_category, wins, scaling, context_score, target_hash, load_seconds


def run():
    import torch
    spec = load_spec()
    paths = refuse_outputs(spec)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for frozen FP32 strict JVP convention')
    old = spec['frozen_slices']['old']
    a126 = import_pinned(path_of(old['constructor_path']), old['constructor_sha256'],
                         'attempt130_frozen_surprisal_helper')
    helper = spec['helper_sources']
    a14 = import_pinned(path_of(helper['attempt014_path']), helper['attempt014_sha256'],
                        'attempt130_frozen_gradient_jvp_helper')
    source_tokens, probe, constructions, model_dir = validate_blind_inputs(spec, a126, a14, torch)
    frozen_rankings = {label: constructions[label]['ranking'] for label in ('old', 'fresh')}
    ranking = pooled_ranking(frozen_rankings['old'], frozen_rankings['fresh'],
                             spec['pooled_selection'], a126)
    model = a14.load_local_model(model_dir, 'cuda', torch)
    eligible = a14.discover_eligible_linear_weights(model, torch)
    a14.freeze_other_parameters(model, eligible)
    before = a14.model_state_hashes(model, torch)
    responses, gradient_records = {}, {}
    primal_reference = None
    jvp_spec = {**spec, 'probe': {**spec['probe'], 'sample_count': 1024}}
    for group in ('low2048', 'control2048'):
        pairs = ranking['low_pairs' if group == 'low2048' else 'control_pairs']
        subset = selected_tokens(source_tokens, pairs, torch)
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
        response32 = a126.a122.prior.validate_response(response.to(torch.float32).contiguous(), torch)
        name = 'response_' + group
        responses[name] = response32
        gradient_records[group] = {'mean_generic_ce': mean_ce, **scale, **realized,
            'selected_pair_raw_sha256': pair_hash(np.asarray(pairs, dtype=np.int64)),
            'eligible_matrix_count': len(matrices), 'readout_checks': readout,
            'response_norm': float(torch.linalg.vector_norm(response32.double())),
            'response_raw_sha256': a14.sha256_raw_float32_tensor(response32, torch)}
        del subset, tangents, matrices, primal, response, response32
        gc.collect()
    if list(responses) != list(NAMES):
        raise ValueError('Fixed two-response inventory incomplete')
    pairwise = a126.response_cosine(responses[NAMES[0]], responses[NAMES[1]], torch)
    for label in ('old', 'fresh'):
        source = spec['frozen_slices'][label]
        require_hash(path_of(source['tokens_path']), source['tokens_serialized_sha256'])
        require_hash(path_of(source['construction_manifest_path']),
                     source['construction_manifest_sha256'])
    require_hash(path_of(spec['probe']['path']), spec['probe']['serialized_sha256'])
    if a14.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files']:
        raise ValueError('Final checkpoint changed during construction')
    a14.verify_model_unchanged(model, before, torch)
    require_hash(SPEC_PATH, SPEC_SHA256)
    a126.a122.prior.atomic_torch_publish(paths['candidate_path'], responses, torch)
    construction = {'format_version': 1, 'attempt_id': ATTEMPT,
        'spec_sha256': SPEC_SHA256, 'constructor_sha256': sha256_file(Path(__file__)),
        'information_policy': spec['information_policy'],
        'frozen_source_sha256': {label: {key: spec['frozen_slices'][label][key] for key in
            ('corpus_spec_sha256', 'corpus_manifest_sha256', 'tokens_serialized_sha256',
             'tokens_raw_sha256', 'construction_spec_sha256', 'constructor_sha256',
             'construction_manifest_sha256', 'candidate_serialized_sha256',
             'surprisal_raw_float64_sha256')} for label in ('old', 'fresh')},
        'source_valid_example_rank_intervals': [[20000, 24096], [24096, 28192]],
        'final_checkpoint_files': spec['final_checkpoint']['files'],
        'frozen_source_rankings': frozen_rankings,
        'pooled_ranking': ranking, 'gradient_loss': spec['gradient_loss'],
        'tangent': spec['tangent'], 'probe': spec['probe'], 'readout': spec['readout'],
        'jvp': spec['jvp'], 'gradient_records': gradient_records,
        'responses': {name: {'raw_sha256': gradient_records[group]['response_raw_sha256'],
            'shape': [128, 2048], 'dtype': 'contiguous_cpu_float32',
            'norm': gradient_records[group]['response_norm']}
            for group, name in zip(('low2048', 'control2048'), NAMES)},
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
    artifact = validate_published(paths, construction, spec['pooled_selection'], a126, a14, torch)
    require_hash(paths['construction_manifest_path'], manifest_hash)
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), construction['constructor_sha256'])
    for label in ('old', 'fresh'):
        source = spec['frozen_slices'][label]
        for stem in ('corpus_spec', 'corpus_manifest', 'construction_spec',
                     'constructor', 'construction_manifest'):
            require_hash(path_of(source[stem + '_path']), source[stem + '_sha256'])
        require_hash(path_of(source['candidate_path']), source['candidate_serialized_sha256'])
        require_hash(path_of(source['tokens_path']), source['tokens_serialized_sha256'])
    require_hash(path_of(spec['probe']['path']), spec['probe']['serialized_sha256'])
    if a14.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files']:
        raise ValueError('Final checkpoint changed before construction barrier')
    del model, eligible, source_tokens, probe, responses, primal_reference
    gc.collect()
    torch.cuda.empty_cache()
    print(MARKER, flush=True)

    reports, gain, selection_category, wins, scaling, context, target_hash, load_seconds = (
        evaluate_after_barrier(spec, artifact, a126, torch))
    require_hash(paths['candidate_path'], construction['candidate']['serialized_sha256'])
    require_hash(paths['construction_manifest_path'], manifest_hash)
    result = {'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'same_specimen_exploratory_method_development': True,
        'clean_heldout_validation': False,
        'construction_manifest_sha256': manifest_hash,
        'candidate_serialized_sha256': construction['candidate']['serialized_sha256'],
        'matched_target_raw_sha256': target_hash,
        'matched_model_load_seconds': load_seconds,
        'signed_position_reports': reports,
        'low2048_minus_control2048': gain,
        'low2048_beats_control2048_positions_1_127': wins,
        'low_vs_control_classification': selection_category,
        'low2048_scaling_comparison': scaling,
        'attempt100_eight_corpus_consensus_context_only': context,
        'pooled_ranking_hashes': ranking['pair_hashes'],
        'low_source_counts': ranking['low_source_counts'],
        'low_control_overlap_count': ranking['low_control_overlap_count'],
        'gradient_records': gradient_records,
        'oracle_free_flattened_response_cosine': pairwise,
        'workload': spec['workload']}
    if paths['result_path'].exists() or paths['result_path'].is_symlink():
        raise ValueError('Attempt130 result already exists')
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
