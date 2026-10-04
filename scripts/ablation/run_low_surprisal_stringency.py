#!/usr/bin/env python3
"""Attempt131: fixed-count stringency slices of Attempt130's frozen ranking."""
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
ATTEMPT = '131_low_surprisal_stringency'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = 'b7fa100677dc86c0c382357daa9f6b53349c40f5a9888a769c86dde338e972c5'
GROUPS = ('very_low', 'next_low')
NAMES = tuple('response_' + name for name in GROUPS)
MARKER = 'LOW_SURPRISAL_STRINGENCY_CANDIDATES_FROZEN'
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
    selection, loss, evaluation = (spec[key] for key in
        ('stringency_selection', 'gradient_loss', 'evaluation'))
    expected_paths = {key: f'experiments/attempts/{ATTEMPT}/{filename}'
        for key, filename in (('construction_manifest_path', 'construction-manifest.json'),
                              ('candidate_path', 'candidate.pt'), ('result_path', 'result.json'))}
    if (spec['attempt_id'] != ATTEMPT or spec['paths'] != expected_paths or
            spec['frozen_attempt130']['spec_sha256'] !=
                '4c569a7e4c00218256a4a1e243ef9a5283050f2c9d95a19b8bc308f2dc5bd942' or
            spec['frozen_attempt130']['source_sha256'] !=
                'bd2537138f0be55ae606eddfb38e0e639cdc934bcfb6f5c7cf13d9be5e20e377' or
            spec['frozen_attempt130']['construction_manifest_sha256'] !=
                '13163051d0fb564187ef9a5a7c004fb92c259f379bce6ad0905243973ebeceb2' or
            spec['frozen_attempt130']['candidate_serialized_sha256'] !=
                '4af45043d33c9cd14b8e9d6924e7cccd0e6f0ba83e78f77ca81638854fb2b0a9' or
            selection['ranking_source'] != 'validated_Attempt130_pooled_ranking_records' or
            selection['combined_record_count'] != 8192 or
            selection['combined_rank_pair_raw_int64_sha256'] !=
                '84bfc8c07f576b61ac3ccda0b56347a2d12a130eccbaef472c38a1eca9cdf8f5' or
            selection['partition_parent_rank_range'] != [0, 2048] or
            selection['pair_dtype'] != 'contiguous_cpu_little_endian_int64_shape_N_by_2' or
            list(selection['groups']) != list(GROUPS) or
            [selection['groups'][name]['rank_range'] for name in GROUPS] !=
                [[0, 1024], [1024, 2048]] or
            [selection['groups'][name]['source_counts'] for name in GROUPS] !=
                [{'old': 496, 'fresh': 528}, {'old': 520, 'fresh': 504}] or
            selection['groups']['very_low']['ordered_pair_raw_int64_sha256'] !=
                '34ed2eef5b17535b8f4548567c89318e07514a01e531178386e3f03331e21454' or
            selection['groups']['next_low']['ordered_pair_raw_int64_sha256'] !=
                'ee66019002e32a40371b3ad16261edcd4fc0f780f6e6137d38420b75bc8650d0' or
            selection['summary_absolute_tolerance'] != 1e-12 or
            selection['require_disjoint_and_exact_union_of_attempt130_low2048'] is not True or
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
            spec['readout']['block_index'] != 13 or
            spec['readout']['hidden_state_index'] != 14 or
            spec['jvp']['strict_forward_ad'] is not True or
            spec['jvp']['fallback'] is not False or
            spec['barrier'] != {'marker': MARKER,
                'attempt130_ranking_rows_gradients_responses_published_and_revalidated_before_historical_access': True,
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
            evaluation['absolute_improvement'] != {
                'clear': 'exceeds_old_low1024_and_fresh_low1024_and_low2048_on_both_metrics',
                'average': 'otherwise_exceeds_mean_prior_low1024_on_both_metrics',
                'otherwise': 'no_absolute_improvement'} or
            evaluation['post_result_candidate_changes'] is not False or
            spec['workload'] != {'corpus_download_batches': 0,
                'corpus_freeze_batches': 0, 'surprisal_scoring_batches': 0,
                'gradient_backward_batches': 256, 'jvp_probe_batches': 64,
                'matched_target_forward_batches': 64, 'ggn_operations': 0,
                'cg_operations': 0} or
            spec['information_policy']['same_specimen_exploratory_method_development'] is not True or
            spec['information_policy']['clean_heldout_validation'] is not False or
            any(spec['information_policy'][key] is not False for key in
                ('historical_base_access_before_barrier', 'historical_target_access_before_barrier',
                 'adapter_access_before_barrier', 'true_delta_access_before_barrier',
                 'attempt127_129_130_historical_scores_before_barrier'))):
        raise ValueError('Attempt131 fixed stringency plan mismatch')
    return spec


def refuse_outputs(spec):
    paths = {key: path_of(value) for key, value in spec['paths'].items()}
    if len(set(paths.values())) != 3 or any(p.exists() or p.is_symlink() for p in paths.values()):
        raise ValueError('Attempt131 output already exists or paths overlap')
    return paths


def validate_frozen_attempt130(spec, a130, a126, a14, torch):
    """Validate the published ranking and its source population before selection."""
    frozen = spec['frozen_attempt130']
    require_hash(path_of(frozen['spec_path']), frozen['spec_sha256'])
    require_hash(path_of(frozen['source_path']), frozen['source_sha256'])
    source_spec = a130.load_spec()
    for key in ('model', 'final_checkpoint', 'eligible_tensors', 'tangent', 'probe',
                'readout', 'jvp', 'helper_sources', 'frozen_attempt005',
                'frozen_attempt100', 'frozen_slices', 'frozen_matched_evaluator'):
        if spec[key] != source_spec[key]:
            raise ValueError('Attempt130 frozen construction convention changed: ' + key)
    if spec['source_pooled_selection'] != source_spec['pooled_selection']:
        raise ValueError('Attempt130 pooled selection convention changed')
    manifest_path = path_of(frozen['construction_manifest_path'])
    manifest = json.loads(require_hash(manifest_path,
        frozen['construction_manifest_sha256']).read_text())
    require_hash(path_of(frozen['candidate_path']), frozen['candidate_serialized_sha256'])
    if (manifest['attempt_id'] != '130_pooled_fineweb_low_surprisal_scaling' or
            manifest['spec_sha256'] != frozen['spec_sha256'] or
            manifest['constructor_sha256'] != frozen['source_sha256'] or
            manifest['candidate']['serialized_sha256'] != frozen['candidate_serialized_sha256'] or
            manifest['barrier'] != source_spec['barrier'] or
            manifest['barrier']['manifest_contains_historical_scores'] is not False or
            manifest['final_checkpoint_files'] != spec['final_checkpoint']['files'] or
            manifest['pooled_ranking']['pair_hashes']['combined_rank_pair_raw_int64_sha256'] !=
                spec['stringency_selection']['combined_rank_pair_raw_int64_sha256']):
        raise ValueError('Attempt130 published construction provenance mismatch')
    source_tokens, probe, source_constructions, model_dir = a130.validate_blind_inputs(
        source_spec, a126, a14, torch)
    expected_source_hashes = {label: {key: source_spec['frozen_slices'][label][key]
        for key in ('corpus_spec_sha256', 'corpus_manifest_sha256',
                    'tokens_serialized_sha256', 'tokens_raw_sha256',
                    'construction_spec_sha256', 'constructor_sha256',
                    'construction_manifest_sha256', 'candidate_serialized_sha256',
                    'surprisal_raw_float64_sha256')} for label in ('old', 'fresh')}
    if (manifest['frozen_source_sha256'] != expected_source_hashes or
            manifest['source_valid_example_rank_intervals'] !=
                [[20000, 24096], [24096, 28192]] or
            any(manifest['frozen_source_rankings'][label] !=
                source_constructions[label]['ranking'] for label in ('old', 'fresh'))):
        raise ValueError('Attempt130 frozen source ranking provenance mismatch')
    old_candidate = a130.validate_published(
        {'candidate_path': path_of(frozen['candidate_path']),
         'construction_manifest_path': manifest_path},
        manifest, source_spec['pooled_selection'], a126, a14, torch)
    del old_candidate
    require_hash(manifest_path, frozen['construction_manifest_sha256'])
    return source_tokens, probe, manifest, model_dir


def frozen_stringency_groups(pooled, selection, a130):
    records, parent_pairs = pooled['records'], pooled['low_pairs']
    if (len(records) != selection['combined_record_count'] or
            len(parent_pairs) != 2048 or
            pooled['pair_hashes']['combined_rank_pair_raw_int64_sha256'] !=
                selection['combined_rank_pair_raw_int64_sha256'] or
            pooled['pair_hashes']['low_selected_pairs_raw_int64_sha256'] !=
                '4532ad09e155aad5a8b941aa00a337522a7f8e81c77761e9b6fab41775c62147'):
        raise ValueError('Attempt130 frozen combined ranking changed')
    groups = {}
    for name in GROUPS:
        control = selection['groups'][name]
        lo, hi = control['rank_range']
        subset = records[lo:hi]
        pairs = np.ascontiguousarray([(record[1], record[2]) for record in subset],
                                      dtype=np.int64)
        values = [record[0] for record in subset]
        if (len(subset) != 1024 or pairs.shape != (1024, 2) or
                not all(isinstance(value, (float, int)) and math.isfinite(value)
                        for value in values)):
            raise ValueError('Frozen stringency slice malformed: ' + name)
        counts = {'old': int(np.count_nonzero(pairs[:, 0] == 0)),
                  'fresh': int(np.count_nonzero(pairs[:, 0] == 1))}
        stats = {'surprisal_min': min(values), 'surprisal_max': max(values),
                 'surprisal_mean': math.fsum(values) / 1024}
        if (counts != control['source_counts'] or
                a130.pair_hash(pairs) != control['ordered_pair_raw_int64_sha256'] or
                any(not math.isclose(stats[key], control[key], rel_tol=0,
                    abs_tol=selection['summary_absolute_tolerance']) for key in stats)):
            raise ValueError('Pinned stringency group hash/count/surprisal mismatch: ' + name)
        groups[name] = {'rank_range': [lo, hi], 'ordered_pairs': pairs.tolist(),
                        'ordered_pair_raw_int64_sha256': a130.pair_hash(pairs),
                        'source_counts': counts, **stats}
    first, second = (groups[name]['ordered_pairs'] for name in GROUPS)
    if (set(map(tuple, first)) & set(map(tuple, second)) or
            first + second != parent_pairs or
            len(set(map(tuple, first + second))) != 2048):
        raise ValueError('Stringency groups do not partition frozen LOW2048 exactly')
    return groups


def selected_tokens(source_tokens, pairs, torch):
    if len(pairs) != 1024:
        raise ValueError('Expected exactly 1024 selected FineWeb rows')
    selected = torch.stack([source_tokens['old' if slice_id == 0 else 'fresh'][row]
        for slice_id, row in pairs]).contiguous()
    if selected.dtype != torch.int64 or tuple(selected.shape) != (1024, 128):
        raise ValueError('Selected FineWeb gradient tokens malformed')
    return selected


def validate_published(paths, construction, spec, a130, a126, a14, torch):
    candidate_path, manifest_path = paths['candidate_path'], paths['construction_manifest_path']
    if (sha256_file(candidate_path) != construction['candidate']['serialized_sha256'] or
            json.loads(manifest_path.read_text()) != construction or
            construction['spec_sha256'] != SPEC_SHA256 or
            construction['candidate']['tensor_order'] != list(NAMES) or
            construction['barrier'] != spec['barrier'] or
            construction['barrier']['manifest_contains_historical_scores'] is not False or
            list(construction['gradient_records']) != list(GROUPS) or
            list(construction['responses']) != list(NAMES)):
        raise ValueError('Published Attempt131 candidate/manifest changed')
    frozen = spec['frozen_attempt130']
    pooled = json.loads(require_hash(path_of(frozen['construction_manifest_path']),
        frozen['construction_manifest_sha256']).read_text())['pooled_ranking']
    if construction['groups'] != frozen_stringency_groups(
            pooled, spec['stringency_selection'], a130):
        raise ValueError('Published Attempt131 frozen selection changed')
    artifact = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != list(NAMES):
        raise ValueError('Published Attempt131 response inventory/order changed')
    for group, name in zip(GROUPS, NAMES):
        tensor = a126.a122.prior.validate_response(artifact[name], torch)
        if (a14.sha256_raw_float32_tensor(tensor, torch) !=
                construction['responses'][name]['raw_sha256'] or
                construction['gradient_records'][group]['selected_pair_raw_sha256'] !=
                spec['stringency_selection']['groups'][group]['ordered_pair_raw_int64_sha256']):
            raise ValueError('Published Attempt131 response/row hash mismatch: ' + name)
    return artifact


def stringency_comparison(reports):
    if list(reports) != list(NAMES):
        raise ValueError('Exactly two signed stringency response reports required')
    for report in reports.values():
        values = report['all_128_position_cosines']
        if (len(values) != 128 or any(x is None or not math.isfinite(x) for x in values) or
                not math.isclose(report['positions_1_4_mean_cosine'],
                                 math.fsum(values[1:5]) / 4, abs_tol=1e-10) or
                not math.isclose(report['positions_1_127_mean_cosine'],
                                 math.fsum(values[1:]) / 127, abs_tol=1e-10)):
            raise ValueError('Malformed signed positionwise historical score')
    very, nxt = (reports[name] for name in NAMES)
    gains = {'primary': very['positions_1_4_mean_cosine'] - nxt['positions_1_4_mean_cosine'],
             'secondary': very['positions_1_127_mean_cosine'] - nxt['positions_1_127_mean_cosine']}
    category = ('meaningful_stringency_gain' if all(x >= .01 for x in gains.values()) else
                'positive_stringency_gain' if all(x > 0 for x in gains.values()) else
                'no_stringency_gain')
    wins = sum(x > y for x, y in zip(very['all_128_position_cosines'][1:],
                                    nxt['all_128_position_cosines'][1:]))
    return gains, category, wins


def absolute_comparison(very, old, fresh, pooled):
    sources = ((old, '127_postfreeze_surprisal_strata_evaluation', 'response_low'),
               (fresh, '129_fresh_fineweb_low_surprisal_replication', 'response_low'),
               (pooled, '130_pooled_fineweb_low_surprisal_scaling', 'response_low2048'))
    previous = {}
    for label, (result, attempt, response) in zip(('old_low1024', 'fresh_low1024',
                                                    'low2048'), sources):
        target_hash = result.get('matched_target', {}).get('raw_sha256',
                      result.get('matched_target_raw_sha256'))
        if result['attempt_id'] != attempt or target_hash != TARGET_SHA256:
            raise ValueError('Prior historical score provenance mismatch: ' + label)
        report = result['signed_position_reports'][response]
        previous[label] = {'primary': report['positions_1_4_mean_cosine'],
                           'secondary': report['positions_1_127_mean_cosine']}
    current = {'primary': very['positions_1_4_mean_cosine'],
               'secondary': very['positions_1_127_mean_cosine']}
    if any(not math.isfinite(value) for row in (current, *previous.values())
           for value in row.values()):
        raise ValueError('Nonfinite absolute historical comparison')
    mean = {metric: math.fsum(previous[label][metric] for label in
        ('old_low1024', 'fresh_low1024')) / 2 for metric in current}
    deltas = {label: {metric: current[metric] - row[metric] for metric in current}
              for label, row in {**previous, 'mean_prior_low1024': mean}.items()}
    clear = all(current[m] > previous[label][m] for m in current
                for label in ('old_low1024', 'fresh_low1024', 'low2048'))
    average = all(current[m] > mean[m] for m in current)
    category = ('clear_absolute_improvement' if clear else
                'average_absolute_improvement' if average else
                'no_absolute_improvement')
    return {'very_low1024': current, 'prior_scores': previous,
            'mean_prior_low1024': mean, 'very_low_minus_prior': deltas,
            'classification': category,
            'interpretation': 'same_specimen_descriptive_development_only'}


def evaluate_after_barrier(spec, artifact, a126, torch):
    frozen = spec['frozen_matched_evaluator']
    require_hash(path_of(frozen['spec_path']), frozen['spec_sha256'])
    a123 = import_pinned(path_of(frozen['source_path']), frozen['source_sha256'],
                         'attempt131_frozen_matched_evaluator')
    matched_spec = a123.load_spec()
    reference = matched_spec['matched_reference']
    privileged = import_pinned(path_of(reference['attempt103_source_path']),
        reference['attempt103_source_sha256'], 'attempt131_frozen_privileged_reference')
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
        old_eval['validator_sha256'], 'attempt131_signed_position_validator')
    reports = {name: a126.a122.score_report(artifact[name], target, validator)
               for name in NAMES}
    gains, stringency, wins = stringency_comparison(reports)
    prior = spec['prior_scores_after_barrier']
    for label in ('attempt127_result', 'attempt129_result', 'attempt130_result'):
        require_hash(path_of(prior[label + '_path']), prior[label + '_sha256'])
    old = json.loads(path_of(prior['attempt127_result_path']).read_text())
    fresh = json.loads(path_of(prior['attempt129_result_path']).read_text())
    pooled = json.loads(path_of(prior['attempt130_result_path']).read_text())
    absolute = absolute_comparison(reports[NAMES[0]], old, fresh, pooled)
    context = pooled['attempt100_eight_corpus_consensus_context_only']
    if (context['role'] != 'context_only_different_probe_construction_population' or
            not all(math.isfinite(context[key]) for key in ('primary', 'secondary'))):
        raise ValueError('Attempt100 context-only provenance mismatch')
    return reports, gains, stringency, wins, absolute, context, target_hash, load_seconds


def run():
    import torch
    spec = load_spec()
    paths = refuse_outputs(spec)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for frozen FP32 strict JVP convention')
    frozen = spec['frozen_attempt130']
    a130 = import_pinned(path_of(frozen['source_path']), frozen['source_sha256'],
                         'attempt131_frozen_attempt130_helper')
    old = spec['frozen_slices']['old']
    a126 = import_pinned(path_of(old['constructor_path']), old['constructor_sha256'],
                         'attempt131_frozen_surprisal_helper')
    helper = spec['helper_sources']
    a14 = import_pinned(path_of(helper['attempt014_path']), helper['attempt014_sha256'],
                        'attempt131_frozen_gradient_jvp_helper')
    source_tokens, probe, source_manifest, model_dir = validate_frozen_attempt130(
        spec, a130, a126, a14, torch)
    groups = frozen_stringency_groups(source_manifest['pooled_ranking'],
                                      spec['stringency_selection'], a130)
    model = a14.load_local_model(model_dir, 'cuda', torch)
    eligible = a14.discover_eligible_linear_weights(model, torch)
    a14.freeze_other_parameters(model, eligible)
    before = a14.model_state_hashes(model, torch)
    responses, gradient_records = {}, {}
    primal_reference = None
    jvp_spec = {**spec, 'probe': {**spec['probe'], 'sample_count': 1024}}
    for group in GROUPS:
        subset = selected_tokens(source_tokens, groups[group]['ordered_pairs'], torch)
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
            'selected_pair_raw_sha256': groups[group]['ordered_pair_raw_int64_sha256'],
            'eligible_matrix_count': len(matrices), 'readout_checks': readout,
            'response_norm': float(torch.linalg.vector_norm(response32.double())),
            'response_raw_sha256': a14.sha256_raw_float32_tensor(response32, torch)}
        del subset, tangents, matrices, primal, response, response32
        gc.collect()
    if list(responses) != list(NAMES):
        raise ValueError('Fixed two-response inventory incomplete')
    pairwise = a126.response_cosine(responses[NAMES[0]], responses[NAMES[1]], torch)
    require_hash(path_of(frozen['construction_manifest_path']),
                 frozen['construction_manifest_sha256'])
    require_hash(path_of(spec['probe']['path']), spec['probe']['serialized_sha256'])
    if a14.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files']:
        raise ValueError('Final checkpoint changed during blind construction')
    a14.verify_model_unchanged(model, before, torch)
    require_hash(SPEC_PATH, SPEC_SHA256)
    a126.a122.prior.atomic_torch_publish(paths['candidate_path'], responses, torch)
    construction = {'format_version': 1, 'attempt_id': ATTEMPT,
        'spec_sha256': SPEC_SHA256, 'constructor_sha256': sha256_file(Path(__file__)),
        'information_policy': spec['information_policy'],
        'frozen_attempt130': {key: frozen[key] for key in
            ('spec_sha256', 'source_sha256', 'construction_manifest_sha256',
             'candidate_serialized_sha256')},
        'source_valid_example_rank_intervals': [[20000, 24096], [24096, 28192]],
        'final_checkpoint_files': spec['final_checkpoint']['files'],
        'combined_rank_pair_raw_int64_sha256':
            spec['stringency_selection']['combined_rank_pair_raw_int64_sha256'],
        'groups': groups, 'gradient_loss': spec['gradient_loss'],
        'tangent': spec['tangent'], 'probe': spec['probe'],
        'readout': spec['readout'], 'jvp': spec['jvp'],
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
    artifact = validate_published(paths, construction, spec, a130, a126, a14, torch)
    require_hash(paths['construction_manifest_path'], manifest_hash)
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), construction['constructor_sha256'])
    for key in ('spec', 'source', 'construction_manifest', 'candidate'):
        suffix = 'serialized_sha256' if key == 'candidate' else 'sha256'
        require_hash(path_of(frozen[key + '_path']), frozen[key + '_' + suffix])
    for label in ('old', 'fresh'):
        source = spec['frozen_slices'][label]
        require_hash(path_of(source['tokens_path']), source['tokens_serialized_sha256'])
    require_hash(path_of(spec['probe']['path']), spec['probe']['serialized_sha256'])
    if a14.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files']:
        raise ValueError('Final checkpoint changed before construction barrier')
    del model, eligible, source_tokens, probe, responses, primal_reference
    gc.collect()
    torch.cuda.empty_cache()
    print(MARKER, flush=True)

    reports, gains, stringency, wins, absolute, context, target_hash, load_seconds = (
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
        'very_low_minus_next_low': gains,
        'very_low_beats_next_low_positions_1_127': wins,
        'stringency_classification': stringency,
        'absolute_comparison': absolute,
        'attempt100_eight_corpus_consensus_context_only': context,
        'selected_group_hashes': {name: groups[name]['ordered_pair_raw_int64_sha256']
                                  for name in GROUPS},
        'selected_group_source_counts': {name: groups[name]['source_counts']
                                         for name in GROUPS},
        'gradient_records': gradient_records,
        'oracle_free_flattened_response_cosine': pairwise,
        'workload': spec['workload']}
    if paths['result_path'].exists() or paths['result_path'].is_symlink():
        raise ValueError('Attempt131 result already exists')
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
