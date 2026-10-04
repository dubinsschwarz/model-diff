#!/usr/bin/env python3
"""Attempt 127: evaluate immutable Attempt126 strata against the matched target."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '127_postfreeze_surprisal_strata_evaluation'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = '40b39419cdced5d96aaccb643315396f3f85bf8a1d6a4cc034a74258bd2bb54b'
NAMES = ('response_low', 'response_middle', 'response_high')
STRATA = ('low', 'middle', 'high')
TARGET_SHA256 = '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f'
EVALUATOR_SHA256 = '1c1a463e789a0fac1c40fe7c71f9d567b27daf1d3b75f1cd0e786a772cbb4525'


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
    frozen, repair, reference = (spec['frozen_attempt126'], spec['failure_repair'],
                                 spec['matched_reference'])
    if (spec['attempt_id'] != ATTEMPT or
            spec['information_policy'] != {
                'privileged_postfreeze_evaluation': True,
                'complete_attempt126_candidate_validation_before_historical_access': True,
                'candidate_reconstruction': False, 'candidate_modification': False,
                'candidate_normalization': False, 'sign_selection': False,
                'interpolation': False, 'combination': False,
                'same_specimen_exploratory_method_development': True,
                'clean_heldout_validation': False} or
            frozen['tensor_order'] != list(NAMES) or
            frozen['shape'] != [128, 2048] or frozen['dtype'] != 'contiguous_cpu_float32' or
            list(frozen['raw_response_sha256']) != list(NAMES) or
            list(frozen['ranking']['selected_row_raw_int64_sha256']) != list(STRATA) or
            repair['erroneous_sha256'] != EVALUATOR_SHA256 + 'f' or
            len(repair['erroneous_sha256']) != 65 or
            repair['correct_clean_committed_sha256'] != EVALUATOR_SHA256 or
            repair['failure_phase'] !=
                'after_SURPRISAL_STRATA_CANDIDATES_FROZEN_before_historical_access' or
            repair['attempt126_blind_construction_published_and_revalidated'] is not True or
            repair['attempt126_result_exists'] is not False or
            repair['ranking_gradients_jvps_publication_and_selection_unaffected'] is not True or
            reference['attempt123_source_sha256'] != EVALUATOR_SHA256 or
            reference['probe_rows'] != [0, 1024] or
            reference['batch_size'] != 32 or reference['batches_per_model'] != 32 or
            reference['target_definition'] !=
                'final_mean_float64_minus_base_mean_float64_then_one_contiguous_FP32_cast' or
            reference['target_raw_float32_sha256'] != TARGET_SHA256 or
            spec['metric'] != {
                'signed_positionwise_cosine': True, 'position_0_separate': True,
                'primary_positions': [1, 2, 3, 4], 'secondary_positions': '1..127',
                'all_position_count': 128, 'absolute_cosine': False,
                'flattened_historical_cosine': False} or
            spec['interpretation']['clear_support'] != {
                'ordered_both_metrics_nonstrict': True,
                'high_low_primary_min': .03, 'high_low_secondary_min': .03} or
            spec['interpretation']['partial_support'] != {
                'high_gt_low_both_metrics': True,
                'at_least_one_high_low_gain_min': .03} or
            spec['workload'] != {
                'final_forward_batches': 32, 'base_forward_batches': 32,
                'total_forward_batches': 64, 'ranking_batches': 0,
                'gradient_batches': 0, 'jvp_batches': 0,
                'ggn_operations': 0, 'cg_operations': 0,
                'new_candidate_artifacts': 0} or
            spec['paths']['result_path'] !=
                f'experiments/attempts/{ATTEMPT}/result.json'):
        raise ValueError('Attempt127 fixed postfreeze evaluation plan mismatch')
    return spec


def refuse_result(path):
    if path.exists() or path.is_symlink():
        raise ValueError('Attempt127 result already exists')


def validate_frozen_candidates(spec, torch):
    """Complete Attempt126 audit. No historical module is imported here."""
    frozen = spec['frozen_attempt126']
    for key in ('spec', 'source', 'construction_manifest', 'candidate'):
        require_hash(path_of(frozen[key + '_path']), frozen[
            'candidate_serialized_sha256' if key == 'candidate' else key + '_sha256'])
    source = import_pinned(path_of(frozen['source_path']), frozen['source_sha256'],
                           'attempt127_frozen_attempt126')
    old_spec = json.loads(path_of(frozen['spec_path']).read_text())
    manifest = json.loads(path_of(frozen['construction_manifest_path']).read_text())
    repair = spec['failure_repair']
    candidate_path = path_of(frozen['candidate_path'])
    if (old_spec['attempt_id'] != source.ATTEMPT or
            path_of(old_spec['paths']['result_path']).exists() or
            path_of(old_spec['paths']['result_path']).is_symlink() or
            old_spec['frozen_attempt123']['source_sha256'] != repair['erroneous_sha256'] or
            old_spec['frozen_attempt123']['source_path'] !=
                spec['matched_reference']['attempt123_source_path'] or
            old_spec['frozen_attempt123']['spec_sha256'] !=
                spec['matched_reference']['attempt123_spec_sha256'] or
            path_of(old_spec['paths']['candidate_path']) != candidate_path or
            old_spec['candidate']['tensor_order'] != list(NAMES) or
            old_spec['candidate']['shape'] != [128, 2048] or
            old_spec['candidate']['dtype'] != 'contiguous_cpu_float32' or
            old_spec['evaluation']['target_raw_sha256'] != TARGET_SHA256 or
            manifest['attempt_id'] != source.ATTEMPT or
            manifest['spec_sha256'] != frozen['spec_sha256'] or
            manifest['constructor_sha256'] != frozen['source_sha256'] or
            manifest['candidate'] != {
                'path': str(candidate_path),
                'serialized_sha256': frozen['candidate_serialized_sha256'],
                'tensor_order': list(NAMES), 'shape': [128, 2048],
                'dtype': 'contiguous_cpu_float32'} or
            manifest['barrier'] != old_spec['barrier'] or
            manifest['barrier'] != {
                'marker': 'SURPRISAL_STRATA_CANDIDATES_FROZEN',
                'surprisals_ranking_rows_gradients_responses_published_and_revalidated_before_historical_access': True,
                'manifest_contains_historical_scores': False} or
            manifest['same_specimen_exploratory_method_development'] is not True or
            manifest['clean_heldout_validation'] is not False or
            manifest['information_policy'] != old_spec['information_policy'] or
            manifest['frozen_attempt005'] != old_spec['frozen_attempt005'] or
            manifest['final_checkpoint_files'] != old_spec['final_checkpoint']['files'] or
            manifest['fineweb_corpus_provenance'] != {
                'name': 'fineweb',
                'spec_sha256': old_spec['frozen_attempt005']['corpus_spec_sha256'],
                'manifest_sha256': old_spec['frozen_attempt005']['corpus_manifest_sha256'],
                'serialized_sha256': old_spec['frozen_attempt005']['serialized_sha256'],
                'raw_tensor_sha256': old_spec['frozen_attempt005']['raw_tensor_sha256']} or
            manifest['gradient_loss'] != old_spec['gradient_loss'] or
            manifest['tangent'] != old_spec['tangent'] or
            manifest['probe'] != old_spec['probe'] or
            manifest['readout'] != old_spec['readout'] or
            manifest['jvp'] != old_spec['jvp'] or
            list(manifest['responses']) != list(NAMES) or
            list(manifest['gradient_records']) != list(STRATA) or
            manifest['ranking']['surprisal_raw_float64_sha256'] !=
                frozen['ranking']['surprisal_raw_float64_sha256'] or
            manifest['ranking']['rank_raw_int64_sha256'] !=
                frozen['ranking']['rank_raw_int64_sha256']):
        raise ValueError('Attempt126 frozen construction provenance mismatch')
    source.validate_ranking(manifest['ranking'])
    for stratum, name in zip(STRATA, NAMES):
        row_hash = frozen['ranking']['selected_row_raw_int64_sha256'][stratum]
        response_hash = frozen['raw_response_sha256'][name]
        record = manifest['gradient_records'][stratum]
        if (manifest['ranking']['strata'][stratum]['ordered_row_indices_raw_sha256'] != row_hash or
                record['selected_row_hash'] != row_hash or
                record['eligible_matrix_count'] != 98 or
                record['aggregate_generic_gradient_norm'] <= 0 or
                record['response_raw_sha256'] != response_hash or
                manifest['responses'][name]['raw_sha256'] != response_hash or
                manifest['responses'][name]['shape'] != [128, 2048] or
                manifest['responses'][name]['dtype'] != 'contiguous_cpu_float32'):
            raise ValueError('Attempt126 frozen stratum/gradient/response mismatch: ' + stratum)
    readout = source.import_pinned(source.SOURCE014, source.SOURCE014_SHA256,
                                  'attempt127_frozen_readout_hash')
    artifact = source.validate_published(candidate_path,
        path_of(frozen['construction_manifest_path']), manifest, torch, readout)
    if not isinstance(artifact, dict) or list(artifact) != list(NAMES):
        raise ValueError('Attempt126 candidate inventory/order mismatch')
    for name in NAMES:
        value = source.a122.prior.validate_response(artifact[name], torch)
        if readout.sha256_raw_float32_tensor(value, torch) != frozen['raw_response_sha256'][name]:
            raise ValueError('Attempt126 pinned response raw SHA256 mismatch: ' + name)
    require_hash(candidate_path, frozen['candidate_serialized_sha256'])
    require_hash(path_of(frozen['construction_manifest_path']),
                 frozen['construction_manifest_sha256'])
    return artifact, manifest, old_spec, source, readout


def evaluate_after_barrier(spec, artifact, manifest, old_spec, source, readout, torch):
    """Privileged work starts only after ATTEMPT126_CANDIDATES_REVALIDATED."""
    reference = spec['matched_reference']
    require_hash(path_of(reference['attempt123_spec_path']),
                 reference['attempt123_spec_sha256'])
    evaluator = import_pinned(path_of(reference['attempt123_source_path']),
                              EVALUATOR_SHA256, 'attempt127_corrected_attempt123')
    matched_spec = evaluator.load_spec()
    matched_reference = matched_spec['matched_reference']
    privileged = import_pinned(path_of(matched_reference['attempt103_source_path']),
        matched_reference['attempt103_source_sha256'], 'attempt127_frozen_attempt103')
    frozen122 = matched_spec['frozen_attempt122']
    require_hash(path_of(frozen122['construction_manifest_path']),
                 frozen122['construction_manifest_sha256'])
    manifest122 = json.loads(path_of(frozen122['construction_manifest_path']).read_text())
    old103, inventories = evaluator.validate_matched_reference(
        matched_spec, manifest122, privileged)
    if (old103['final_checkpoint_files'] != old_spec['final_checkpoint']['files'] or
            old103['readout'] != old_spec['readout'] or
            old103['probe_rows'] != [0, 1024] or
            matched_reference['target_raw_sha256'] != TARGET_SHA256 or
            matched_reference['probe_rows'] != [0, 1024] or
            matched_reference['batch_size'] != 32 or
            matched_reference['batches_per_model'] != 32):
        raise ValueError('Attempt123 matched-target controls differ from frozen Attempt126')
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for matched FP32 model readouts')
    full_probe = privileged.a.load_probe(privileged.resolve(old103['paths']['probe_path']),
        {**old103['probe'], 'sample_count': 10000}, torch)
    probe = evaluator.matched_probe(full_probe, matched_spec, torch)
    del full_probe
    base, final, load_seconds = privileged.load_model_pair(old103)
    final_mean = privileged.base_probe_mean(final, probe, old103)
    base_mean = privileged.base_probe_mean(base, probe, old103)
    target, target_hash = evaluator.verified_target(final_mean, base_mean, matched_spec,
        privileged.matched_difference, privileged.a.sha256_raw_float32_tensor, torch)
    if target_hash != TARGET_SHA256:
        raise ValueError('Matched target hash changed before scoring')
    if any(parameter.grad is not None for model in (base, final)
           for parameter in model.parameters()):
        raise ValueError('Matched activation means created model gradients')
    for directory, expected in ((privileged.resolve(old103['paths']['base_directory']),
                                 inventories['base_files']),
                                (privileged.resolve(old103['paths']['final_directory']),
                                 inventories['final_files'])):
        if privileged.a.checkpoint_file_records(directory) != expected:
            raise ValueError('Historical checkpoint changed during evaluation')
    del base, final, final_mean, base_mean, probe
    gc.collect()
    torch.cuda.empty_cache()
    old_eval = source.a122.load_spec()['evaluation']
    validator = import_pinned(path_of(old_eval['validator_path']),
        old_eval['validator_sha256'], 'attempt127_signed_position_validator')
    reports = {name: source.a122.score_report(artifact[name], target, validator)
               for name in NAMES}
    deltas, category = source.differences_and_classification(reports)
    correlations = source.descriptive_correlations(manifest['ranking'],
                                                    manifest['gradient_records'], reports)
    context = old_spec['frozen_attempt100']
    require_hash(path_of(context['evaluation_path']), context['evaluation_sha256'])
    old100 = json.loads(path_of(context['evaluation_path']).read_text())
    fineweb = [row for row in old100['candidates']
               if row.get('candidate') == 'response_fineweb']
    if len(fineweb) != 1:
        raise ValueError('Frozen Attempt100 FineWeb context inventory mismatch')
    old_context = {
        'primary': fineweb[0]['positions_1_4_mean_cosine'],
        'secondary': fineweb[0]['positions_1_127_mean_cosine'],
        'role': 'context_only_older_full_probe_population_not_exact_matched_baseline',
        'evaluation_sha256': context['evaluation_sha256']}
    return reports, deltas, category, correlations, old_context, target_hash, load_seconds


def run():
    spec = load_spec()
    output = path_of(spec['paths']['result_path'])
    refuse_result(output)
    import torch
    artifact, manifest, old_spec, source, readout = validate_frozen_candidates(spec, torch)
    print('ATTEMPT126_CANDIDATES_REVALIDATED', flush=True)

    reports, deltas, category, correlations, old_context, target_hash, load_seconds = (
        evaluate_after_barrier(spec, artifact, manifest, old_spec, source, readout, torch))
    frozen = spec['frozen_attempt126']
    require_hash(path_of(frozen['candidate_path']), frozen['candidate_serialized_sha256'])
    require_hash(path_of(frozen['construction_manifest_path']),
                 frozen['construction_manifest_sha256'])
    for name in NAMES:
        if readout.sha256_raw_float32_tensor(artifact[name], torch) != frozen['raw_response_sha256'][name]:
            raise ValueError('Frozen Attempt126 response changed during evaluation: ' + name)
    result = {
        'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'privileged_postfreeze_evaluation': True,
        'same_specimen_exploratory_method_development': True,
        'clean_heldout_validation': False,
        'candidate_source': 'unchanged_frozen_Attempt126_candidate.pt',
        'repair': spec['failure_repair'],
        'provenance': {
            'spec_sha256': SPEC_SHA256, 'source_sha256': sha256_file(Path(__file__)),
            'attempt126_spec_sha256': frozen['spec_sha256'],
            'attempt126_source_sha256': frozen['source_sha256'],
            'attempt126_construction_manifest_sha256': frozen['construction_manifest_sha256'],
            'candidate_serialized_sha256': frozen['candidate_serialized_sha256'],
            'candidate_raw_response_sha256': frozen['raw_response_sha256'],
            'surprisal_and_rank_hashes': frozen['ranking'],
            'corrected_attempt123_source_sha256': EVALUATOR_SHA256,
            'matched_target_raw_sha256': target_hash},
        'matched_target': {
            'definition': spec['matched_reference']['target_definition'],
            'probe_rows': [0, 1024], 'batch_size': 32, 'batches_per_model': 32,
            'readout': 'block_13_output_hidden_states_14',
            'shape': [128, 2048], 'dtype': 'contiguous_cpu_float32',
            'raw_sha256': target_hash},
        'matched_model_load_seconds': load_seconds,
        'signed_position_reports': reports,
        'fixed_pairwise_deltas': deltas,
        'development_set_classification': category,
        'descriptive_correlations': correlations,
        'old_attempt100_fineweb_full_probe_context': old_context,
        'strata': manifest['ranking']['strata'],
        'unused_rank_intervals': manifest['ranking']['unused_rank_intervals'],
        'gradient_records': manifest['gradient_records'],
        'oracle_free_pairwise_response_cosines':
            manifest['pairwise_flattened_response_cosine_oracle_free'],
        'workload': spec['workload']}
    refuse_result(output)
    source.a122.prior.atomic_json_publish(output, result)
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
