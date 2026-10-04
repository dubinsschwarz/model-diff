#!/usr/bin/env python3
"""Attempt129: fresh FineWeb surprisal strata and matched historical replication."""
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
ATTEMPT = '129_fresh_fineweb_low_surprisal_replication'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = '7cafe03201e243ae4975ecf390cb6fe1f3808978b3a81041841a5cec3f4838ae'
GROUPS = ('control', 'low', 'middle', 'high')
NAMES = tuple('response_' + group for group in GROUPS)
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
    fresh = spec['fresh_corpus']
    rows = spec['candidate_rows']
    expected_rows = {
        'control': {'selection': 'original_frozen_corpus_order', 'row_range': [0, 1024]},
        'low': {'selection': 'ascending_endpoint_surprisal_rank', 'rank_range': [0, 1024]},
        'middle': {'selection': 'ascending_endpoint_surprisal_rank', 'rank_range': [1536, 2560]},
        'high': {'selection': 'ascending_endpoint_surprisal_rank', 'rank_range': [3072, 4096]}}
    expected_paths = {key: f'experiments/attempts/{ATTEMPT}/{filename}'
        for key, filename in (('construction_manifest_path', 'construction-manifest.json'),
                              ('candidate_path', 'candidate.pt'),
                              ('result_path', 'result.json'))}
    if (spec['attempt_id'] != ATTEMPT or
            list(rows) != list(GROUPS) or rows != expected_rows or
            spec['candidate'] != {'tensor_order': list(NAMES), 'shape': [128, 2048],
                'dtype': 'contiguous_cpu_float32', 'response_normalization': False,
                'sign_selection': False, 'interpolation': False,
                'reweighting': False, 'combination': False} or
            spec['ranking']['population_rows'] != [0, 4096] or
            spec['ranking']['prediction_tokens_per_row'] != 127 or
            spec['ranking']['accumulation_dtype'] != 'cpu_float64' or
            spec['ranking']['sort_key'] != ['surprisal_ascending',
                                            'original_row_index_ascending'] or
            spec['gradient_loss']['sample_count'] != 1024 or
            spec['gradient_loss']['sequence_length'] != 128 or
            spec['gradient_loss']['batch_size'] != 8 or
            spec['gradient_loss']['number_of_batches'] != 128 or
            spec['gradient_loss']['total_prediction_tokens'] != 130048 or
            spec['tangent']['direction'] != 'delta = +alpha * g' or
            spec['tangent']['target_relative_frobenius'] != .00125 or
            spec['tangent']['normalization'] != 'one_global_alpha_across_all_eligible_matrices' or
            spec['probe']['sample_count'] != 10000 or
            spec['probe']['selected_rows'] != [0, 1024] or
            spec['probe']['batch_size'] != 32 or
            spec['readout']['block_index'] != 13 or
            spec['readout']['hidden_state_index'] != 14 or
            spec['jvp']['strict_forward_ad'] is not True or
            spec['jvp']['fallback'] is not False or
            spec['barrier'] != {'marker': 'FRESH_SURPRISAL_CANDIDATES_FROZEN',
                'fresh_corpus_ranking_rows_gradients_responses_published_and_revalidated_before_historical_access': True,
                'manifest_contains_historical_scores': False} or
            spec['evaluation'] != {'target_probe_rows': [0, 1024],
                'target_raw_sha256': TARGET_SHA256, 'signed_positionwise_cosine': True,
                'position_0_separate': True, 'primary_positions': [1, 2, 3, 4],
                'secondary_positions': list(range(1, 128)), 'absolute_cosine': False,
                'flattened_historical_cosine': False,
                'replication_prediction': 'low_greater_than_middle_greater_than_high',
                'clear_replication': {'ordered_both_metrics_nonstrict': True,
                    'low_high_primary_min': .03, 'low_high_secondary_min': .03},
                'partial_replication': {'low_gt_high_both_metrics': True,
                    'at_least_one_low_high_gain_min': .03},
                'selection_gain': {'meaningful_both_min': .01,
                    'positive_both_strictly_gt': 0.0},
                'post_result_candidate_changes': False} or
            fresh['old_valid_example_rank_interval'] != [20000, 24096] or
            fresh['fresh_valid_example_rank_interval'] != [24096, 28192] or
            fresh['full_shape'] != [4096, 128] or
            spec['paths'] != expected_paths or
            spec['workload'] != {'fresh_corpus_freeze_gpu_batches': 0,
                'ranking_forward_batches': 512, 'gradient_backward_batches': 512,
                'jvp_probe_batches': 128, 'matched_target_forward_batches': 64,
                'ggn_operations': 0, 'cg_operations': 0} or
            spec['information_policy']['same_specimen_exploratory_method_development'] is not True or
            spec['information_policy']['independent_generic_data_replication'] is not True or
            spec['information_policy']['clean_heldout_validation'] is not False or
            any(spec['information_policy'][key] is not False for key in
                ('historical_base_access_before_barrier', 'adapter_access_before_barrier',
                 'true_delta_access_before_barrier',
                 'historical_activation_target_access_before_barrier',
                 'attempt127_oracle_scores_before_barrier'))):
        raise ValueError('Attempt129 fixed replication plan mismatch')
    return spec


def refuse_outputs(spec):
    paths = {key: path_of(value) for key, value in spec['paths'].items()}
    if (len(set(paths.values())) != 3 or
            any(path.exists() or path.is_symlink() for path in paths.values())):
        raise ValueError('Attempt129 output already exists or paths overlap')
    return paths


def validate_blind_inputs(spec, a126, a14, torch):
    """Validate the full new corpus and full old probe before selecting prefixes."""
    frozen126 = spec['frozen_attempt126']
    for key in ('spec', 'source'):
        require_hash(path_of(frozen126[key + '_path']), frozen126[key + '_sha256'])
    old126 = json.loads(path_of(frozen126['spec_path']).read_text())
    for group, keys in (('frozen_attempt005', ('corpus_spec', 'corpus_manifest')),
                        ('frozen_attempt100', ('spec', 'source', 'construction_manifest')),
                        ('frozen_attempt113', ('spec', 'source', 'construction_manifest'))):
        for key in keys:
            record = spec[group]
            require_hash(path_of(record[key + '_path']), record[key + '_sha256'])
    for key in ('model', 'final_checkpoint', 'eligible_tensors', 'ranking', 'strata',
                'unused_rank_intervals', 'gradient_loss', 'tangent', 'probe',
                'readout', 'jvp', 'helper_sources'):
        if spec[key] != old126[key]:
            raise ValueError('Attempt126 construction convention changed: ' + key)
    if spec['eligible_tensors']['expected_matrix_count'] != 98:
        raise ValueError('Expected exactly 98 eligible Linear weights')
    fresh = spec['fresh_corpus']
    require_hash(path_of(fresh['corpus_spec_path']), fresh['corpus_spec_sha256'])
    require_hash(path_of(fresh['freezer_source_path']), fresh['freezer_source_sha256'])
    corpus_spec = json.loads(path_of(fresh['corpus_spec_path']).read_text())
    corpus_manifest = json.loads(path_of(fresh['corpus_manifest_path']).read_text())
    old_corpus_spec = json.loads(path_of(spec['frozen_attempt005']['corpus_spec_path']).read_text())
    old_selection = old_corpus_spec['selection']
    if (corpus_spec['dataset'] != old_corpus_spec['dataset'] or
            corpus_spec['selection'] != {**old_selection, 'skip_valid_examples': 24096} or
            corpus_spec['old_valid_example_rank_interval'] != [20000, 24096] or
            corpus_spec['fresh_valid_example_rank_interval'] != [24096, 28192] or
            corpus_spec['artifact']['path'] != fresh['tokens_path'] or
            corpus_manifest['corpus_spec_sha256'] != fresh['corpus_spec_sha256'] or
            corpus_manifest['freeze_script_sha256'] != fresh['freezer_source_sha256'] or
            corpus_manifest['dataset'] != corpus_spec['dataset'] or
            corpus_manifest['selection'] != corpus_spec['selection'] or
            corpus_manifest['old_valid_example_rank_interval'] != [20000, 24096] or
            corpus_manifest['fresh_valid_example_rank_interval'] != [24096, 28192] or
            corpus_manifest['tokenizer_checkpoint']['files'] !=
                corpus_spec['tokenizer_checkpoint_files'] or
            corpus_manifest['frozen_attempt005'] != corpus_spec['frozen_attempt005'] or
            corpus_manifest['tensor'] != {'shape': [4096, 128],
                'dtype': 'torch.int64', 'device': 'cpu', 'contiguous': True} or
            corpus_manifest['artifact']['path'] != fresh['tokens_path']):
        raise ValueError('Fresh FineWeb corpus manifest contradicts pinned selection')
    model_dir = path_of(spec['final_checkpoint']['directory'])
    if a14.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files']:
        raise ValueError('Canonical final checkpoint changed')
    token_path = path_of(fresh['tokens_path'])
    token_hash = corpus_manifest['artifact']['serialized_sha256']
    require_hash(token_path, token_hash)
    tokens = a14.load_corpus_tokens(token_path,
        corpus_manifest['artifact']['raw_tensor_sha256'], 4096, 128, torch)
    probe_spec = {key: spec['probe'][key] for key in
        ('sample_count', 'sequence_length', 'batch_size', 'dtype',
         'serialized_sha256', 'raw_tensor_sha256')}
    full_probe = a14.load_probe(path_of(spec['probe']['path']), probe_spec, torch)
    if (tuple(tokens.shape) != (4096, 128) or
            tuple(full_probe.shape) != (10000, 128) or
            not tokens.is_contiguous() or not full_probe.is_contiguous()):
        raise ValueError('Fresh FineWeb or full probe shape mismatch')
    probe = full_probe[:1024].contiguous()
    return tokens, probe, corpus_manifest, model_dir


def fixed_candidate_rows(ranking, a126):
    a126.validate_ranking(ranking)
    rows = {'control': list(range(1024))}
    rows.update({name: ranking['strata'][name]['original_row_indices']
                 for name in GROUPS[1:]})
    if (list(rows) != list(GROUPS) or
            any(len(value) != 1024 or len(set(value)) != 1024 for value in rows.values())):
        raise ValueError('Expected four exact 1024-row candidate sets')
    hashes = {name: a126.raw_int64_hash(np.asarray(value, dtype=np.int64))
              for name, value in rows.items()}
    overlap = {name: len(set(rows['control']) & set(rows[name]))
               for name in GROUPS[1:]}
    return rows, hashes, overlap


def oracle_free_pairwise_cosines(responses, a126, torch):
    if list(responses) != list(NAMES):
        raise ValueError('Exactly four responses required for pairwise diagnostics')
    return {left + '__' + right: a126.response_cosine(responses[left], responses[right], torch)
            for i, left in enumerate(NAMES) for right in NAMES[i + 1:]}


def validate_published(paths, construction, a126, a14, torch):
    candidate_path, manifest_path = paths['candidate_path'], paths['construction_manifest_path']
    if (sha256_file(candidate_path) != construction['candidate']['serialized_sha256'] or
            json.loads(manifest_path.read_text()) != construction or
            construction['candidate']['tensor_order'] != list(NAMES) or
            construction['barrier']['manifest_contains_historical_scores'] is not False):
        raise ValueError('Published Attempt129 artifact or manifest changed')
    a126.validate_ranking(construction['ranking'])
    rows, hashes, overlap = fixed_candidate_rows(construction['ranking'], a126)
    if (construction['candidate_rows'] != rows or
            construction['selected_row_raw_sha256'] != hashes or
            construction['control_stratum_overlap_counts'] != overlap or
            list(construction['gradient_records']) != list(GROUPS) or
            list(construction['responses']) != list(NAMES)):
        raise ValueError('Published Attempt129 ranking or candidate rows changed')
    artifact = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != list(NAMES):
        raise ValueError('Published candidate inventory/order changed')
    for group, name in zip(GROUPS, NAMES):
        value = a126.a122.prior.validate_response(artifact[name], torch)
        if (a14.sha256_raw_float32_tensor(value, torch) !=
                construction['responses'][name]['raw_sha256'] or
                construction['gradient_records'][group]['selected_row_hash'] != hashes[group]):
            raise ValueError('Published response/selected-row hash mismatch: ' + name)
    require_hash(candidate_path, construction['candidate']['serialized_sha256'])
    return artifact


def signed_score_differences(reports):
    if list(reports) != list(NAMES):
        raise ValueError('Only four fixed responses may be scored')
    primary = {group: reports['response_' + group]['positions_1_4_mean_cosine']
               for group in GROUPS}
    secondary = {group: reports['response_' + group]['positions_1_127_mean_cosine']
                 for group in GROUPS}
    if any(value is None or not math.isfinite(value)
           for value in (*primary.values(), *secondary.values())):
        raise ValueError('Undefined signed historical score')
    deltas, position_counts = {}, {}
    for left, right in (('low', 'middle'), ('low', 'high'),
                        ('middle', 'high'), ('low', 'control')):
        key = left + '_minus_' + right
        deltas[key] = {'primary': primary[left] - primary[right],
                       'secondary': secondary[left] - secondary[right]}
        first = reports['response_' + left]['all_128_position_cosines']
        second = reports['response_' + right]['all_128_position_cosines']
        if (len(first) != 128 or len(second) != 128 or
                any(value is None or not math.isfinite(value)
                    for value in (*first, *second))):
            raise ValueError('Malformed signed per-position historical report')
        position_counts[key] = sum(x > y for x, y in zip(first[1:], second[1:]))
    clear = (all(metric['low'] >= metric['middle'] >= metric['high']
                 for metric in (primary, secondary)) and
             deltas['low_minus_high']['primary'] >= .03 and
             deltas['low_minus_high']['secondary'] >= .03)
    partial = (primary['low'] > primary['high'] and
               secondary['low'] > secondary['high'] and
               (deltas['low_minus_high']['primary'] >= .03 or
                deltas['low_minus_high']['secondary'] >= .03))
    replication = ('clear_replication' if clear else
                   'partial_replication' if partial else 'no_replication')
    gain = deltas['low_minus_control']
    selection = ('meaningful_selection_gain' if
                 gain['primary'] >= .01 and gain['secondary'] >= .01 else
                 'positive_selection_gain' if
                 gain['primary'] > 0 and gain['secondary'] > 0 else
                 'no_selection_gain')
    return deltas, position_counts, replication, selection


def old_vs_fresh_context(old, reports, deltas):
    if (old['attempt_id'] != '127_postfreeze_surprisal_strata_evaluation' or
            list(old['signed_position_reports']) != list(NAMES[1:]) or
            old['matched_target']['raw_sha256'] != TARGET_SHA256 or
            old['clean_heldout_validation'] is not False):
        raise ValueError('Frozen Attempt127 historical context provenance mismatch')
    context = {}
    for group in GROUPS[1:]:
        name = 'response_' + group
        context[group] = {
            'old_primary': old['signed_position_reports'][name]['positions_1_4_mean_cosine'],
            'old_secondary': old['signed_position_reports'][name]['positions_1_127_mean_cosine'],
            'fresh_primary': reports[name]['positions_1_4_mean_cosine'],
            'fresh_secondary': reports[name]['positions_1_127_mean_cosine']}
    old_low = context['low']
    old_high = context['high']
    context['low_minus_high'] = {
        'old_primary': old_low['old_primary'] - old_high['old_primary'],
        'old_secondary': old_low['old_secondary'] - old_high['old_secondary'],
        'fresh_primary': deltas['low_minus_high']['primary'],
        'fresh_secondary': deltas['low_minus_high']['secondary']}
    return context


def evaluate_after_barrier(spec, artifact, a126, torch):
    frozen = spec['frozen_attempt123']
    require_hash(path_of(frozen['spec_path']), frozen['spec_sha256'])
    a123 = import_pinned(path_of(frozen['source_path']), frozen['source_sha256'],
                         'attempt129_matched_historical_evaluator')
    matched_spec = a123.load_spec()
    reference = matched_spec['matched_reference']
    privileged = import_pinned(path_of(reference['attempt103_source_path']),
        reference['attempt103_source_sha256'], 'attempt129_frozen_attempt103')
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
        raise ValueError('Matched historical target SHA256 mismatch')
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
        old_eval['validator_sha256'], 'attempt129_signed_position_validator')
    reports = {name: a126.a122.score_report(artifact[name], target, validator)
               for name in NAMES}
    deltas, counts, replication, selection = signed_score_differences(reports)
    prior = spec['frozen_attempt127']
    require_hash(path_of(prior['result_path']), prior['result_sha256'])
    old = json.loads(path_of(prior['result_path']).read_text())
    context = old_vs_fresh_context(old, reports, deltas)
    return reports, deltas, counts, replication, selection, context, target_hash, load_seconds


def run():
    import torch
    spec = load_spec()
    paths = refuse_outputs(spec)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for frozen FP32 strict JVP convention')
    frozen126 = spec['frozen_attempt126']
    a126 = import_pinned(path_of(frozen126['source_path']), frozen126['source_sha256'],
                         'attempt129_frozen_attempt126_blind_helper')
    a14 = a126.import_pinned(a126.SOURCE014, a126.SOURCE014_SHA256,
                             'attempt129_frozen_jvp_helper')
    tokens, probe, corpus_manifest, model_dir = validate_blind_inputs(spec, a126, a14, torch)
    fresh_manifest_hash = sha256_file(path_of(spec['fresh_corpus']['corpus_manifest_path']))
    model = a14.load_local_model(model_dir, 'cuda', torch)
    eligible = a14.discover_eligible_linear_weights(model, torch)
    a14.freeze_other_parameters(model, eligible)
    before = a14.model_state_hashes(model, torch)
    surprisal = a126.score_endpoint_surprisal(model, tokens, torch)
    ranking = a126.rank_and_stratify(surprisal)
    a126.validate_ranking(ranking)
    rows, row_hashes, overlaps = fixed_candidate_rows(ranking, a126)
    del surprisal
    responses, gradient_records = {}, {}
    primal_reference = None
    jvp_spec = {**spec, 'probe': {**spec['probe'], 'sample_count': 1024}}
    for group in GROUPS:
        selected = torch.tensor(rows[group], dtype=torch.int64)
        if (tuple(selected.shape) != (1024,) or
                a14.sha256_raw_int64_tensor(selected, torch) != row_hashes[group]):
            raise ValueError('Frozen candidate row hash mismatch: ' + group)
        subset = tokens.index_select(0, selected).contiguous()
        if tuple(subset.shape) != (1024, 128):
            raise ValueError('Selected fresh FineWeb gradient subset malformed')
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
            'selected_row_hash': row_hashes[group],
            'eligible_matrix_count': len(matrices), 'readout_checks': readout,
            'response_norm': float(torch.linalg.vector_norm(response32.double())),
            'response_raw_sha256': a14.sha256_raw_float32_tensor(response32, torch)}
        del selected, subset, tangents, matrices, primal, response, response32
        gc.collect()
    if list(responses) != list(NAMES):
        raise ValueError('Fixed four-response inventory incomplete')
    pairwise = oracle_free_pairwise_cosines(responses, a126, torch)
    fresh = spec['fresh_corpus']
    if (a14.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files'] or
            sha256_file(path_of(fresh['tokens_path'])) !=
                corpus_manifest['artifact']['serialized_sha256'] or
            sha256_file(path_of(fresh['corpus_manifest_path'])) != fresh_manifest_hash or
            sha256_file(path_of(spec['probe']['path'])) !=
                spec['probe']['serialized_sha256']):
        raise ValueError('Frozen blind input changed during construction')
    a14.verify_model_unchanged(model, before, torch)
    require_hash(SPEC_PATH, SPEC_SHA256)
    a126.a122.prior.atomic_torch_publish(paths['candidate_path'], responses, torch)
    construction = {
        'format_version': 1, 'attempt_id': ATTEMPT,
        'information_policy': spec['information_policy'],
        'spec_sha256': SPEC_SHA256, 'constructor_sha256': sha256_file(Path(__file__)),
        'fresh_corpus_spec_sha256': fresh['corpus_spec_sha256'],
        'fresh_corpus_manifest_sha256': fresh_manifest_hash,
        'fresh_corpus_serialized_sha256': corpus_manifest['artifact']['serialized_sha256'],
        'fresh_corpus_raw_tensor_sha256': corpus_manifest['artifact']['raw_tensor_sha256'],
        'old_valid_example_rank_interval': [20000, 24096],
        'fresh_valid_example_rank_interval': [24096, 28192],
        'final_checkpoint_files': spec['final_checkpoint']['files'],
        'ranking': ranking, 'unused_rank_intervals': ranking['unused_rank_intervals'],
        'candidate_rows': rows, 'selected_row_raw_sha256': row_hashes,
        'control_stratum_overlap_counts': overlaps,
        'gradient_loss': spec['gradient_loss'], 'tangent': spec['tangent'],
        'probe': spec['probe'], 'readout': spec['readout'], 'jvp': spec['jvp'],
        'gradient_records': gradient_records,
        'responses': {name: {'raw_sha256': gradient_records[group]['response_raw_sha256'],
            'shape': [128, 2048], 'dtype': 'contiguous_cpu_float32',
            'norm': gradient_records[group]['response_norm']}
            for group, name in zip(GROUPS, NAMES)},
        'pairwise_flattened_response_cosine_oracle_free': pairwise,
        'final_primal_float64_sha256': hashlib.sha256(
            primal_reference.numpy().astype('<f8', copy=False).tobytes()).hexdigest(),
        'candidate': {'path': str(paths['candidate_path']),
            'serialized_sha256': sha256_file(paths['candidate_path']),
            'tensor_order': list(NAMES), 'shape': [128, 2048],
            'dtype': 'contiguous_cpu_float32'},
        'barrier': spec['barrier'],
        'same_specimen_exploratory_method_development': True,
        'independent_generic_data_replication': True,
        'clean_heldout_validation': False}
    a126.a122.prior.atomic_json_publish(paths['construction_manifest_path'], construction)
    manifest_hash = sha256_file(paths['construction_manifest_path'])
    artifact = validate_published(paths, construction, a126, a14, torch)
    require_hash(paths['construction_manifest_path'], manifest_hash)
    require_hash(path_of(fresh['tokens_path']),
                 corpus_manifest['artifact']['serialized_sha256'])
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), construction['constructor_sha256'])
    del model, eligible, tokens, probe, responses, primal_reference
    gc.collect()
    torch.cuda.empty_cache()
    print('FRESH_SURPRISAL_CANDIDATES_FROZEN', flush=True)

    reports, deltas, counts, replication, selection, context, target_hash, load_seconds = (
        evaluate_after_barrier(spec, artifact, a126, torch))
    require_hash(paths['candidate_path'], construction['candidate']['serialized_sha256'])
    require_hash(paths['construction_manifest_path'], manifest_hash)
    result = {'format_version': 1, 'attempt_id': ATTEMPT,
        'purpose': spec['purpose'],
        'same_specimen_exploratory_method_development': True,
        'independent_generic_data_replication': True,
        'clean_heldout_validation': False,
        'construction_manifest_sha256': manifest_hash,
        'candidate_serialized_sha256': construction['candidate']['serialized_sha256'],
        'fresh_corpus_manifest_sha256': construction['fresh_corpus_manifest_sha256'],
        'matched_target_raw_sha256': target_hash,
        'matched_model_load_seconds': load_seconds,
        'signed_position_reports': reports,
        'fixed_pairwise_deltas': deltas,
        'predicted_order_position_win_counts_1_127': counts,
        'replication_classification': replication,
        'low_vs_control_classification': selection,
        'cross_slice_attempt127_context': context,
        'candidate_rows': rows,
        'selected_row_raw_sha256': row_hashes,
        'control_stratum_overlap_counts': overlaps,
        'strata': ranking['strata'],
        'gradient_records': gradient_records,
        'oracle_free_pairwise_response_cosines': pairwise,
        'workload': spec['workload']}
    if paths['result_path'].exists() or paths['result_path'].is_symlink():
        raise ValueError('Attempt129 result already exists')
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
