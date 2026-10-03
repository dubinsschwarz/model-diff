#!/usr/bin/env python3
"""Attempt 123: score frozen Attempt122 responses against the matched 1024-row target."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '123_postfreeze_matched1024_relaxation_reevaluation'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = 'a27ee4f0e22ff892c3c2ffeb05ab8f6678cb8fe554ce33e2971e535154136673'
GRID = (1, 2, 4)
NAMES = ('R_1', 'R_2', 'R_4')
TARGET_SHA256 = '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f'
SOURCE122 = PROJECT / 'scripts/ablation/run_loss_stabilized_generic_relaxation.py'
SOURCE122_SHA256 = '7dd7e79b84ac297c7b0f7a62a69ff5dc7d91332b95b3b180f72f50c1f6cf3a9d'
SOURCE014 = PROJECT / 'scripts/ablation/construct_multicorpus_raw_jg_consensus.py'
SOURCE014_SHA256 = '59936b928dcf942470db92debb1f2a8e41823ae1c9082803d2fdd8dc50202caa'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('Frozen SHA256 mismatch: '+str(path))
    return path


def import_pinned(path, expected, name):
    require_hash(path, expected)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


a122 = import_pinned(SOURCE122, SOURCE122_SHA256, 'attempt123_frozen_attempt122')
a14 = import_pinned(SOURCE014, SOURCE014_SHA256, 'attempt123_frozen_readout')
path_of = a122.path_of


def load_spec():
    require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text())
    frozen = spec['frozen_attempt122']
    reference = spec['matched_reference']
    metric = spec['metric']
    if (spec['attempt_id'] != ATTEMPT or
            spec['information_policy'] != {
                'privileged_postfreeze_diagnostic': True,
                'validate_complete_attempt122_candidate_before_historical_base_access': True,
                'candidate_reconstruction': False, 'candidate_modification': False,
                'same_specimen_exploratory_method_development': True,
                'clean_heldout_validation': False} or
            frozen['source_path'] != str(SOURCE122.relative_to(PROJECT)) or
            frozen['source_sha256'] != SOURCE122_SHA256 or
            frozen['tensor_order'] != list(NAMES) or
            frozen['shape'] != [128, 2048] or
            frozen['dtype'] != 'contiguous_cpu_float32' or
            set(frozen['raw_response_sha256']) != set(NAMES) or
            reference['probe_source_rows'] != 10000 or
            reference['probe_rows'] != [0, 1024] or
            reference['sequence_length'] != 128 or
            reference['batch_size'] != 32 or
            reference['batches_per_model'] != 32 or
            reference['readout_block_index'] != 13 or
            reference['hidden_state_index'] != 14 or
            reference['target_definition'] !=
                'final_mean_float64_minus_base_mean_float64_then_one_contiguous_FP32_cast' or
            reference['target_shape'] != [128, 2048] or
            reference['target_dtype'] != 'contiguous_cpu_float32' or
            reference['target_raw_sha256'] != TARGET_SHA256 or
            metric != {'signed_positionwise_cosine': True, 'absolute_cosine': False,
                       'flattened_cosine': False, 'position_0_separate': True,
                       'primary_positions': [1, 2, 3, 4],
                       'secondary_positions': list(range(1, 128)),
                       'descriptive_tail_positions': list(range(5, 128)),
                       'best_later_tie_break': ['highest_primary', 'highest_secondary', 'smaller_k'],
                       'clear_thresholds': {'primary_gain': .03, 'secondary_gain': .02},
                       'moderate_thresholds': {'primary_gain': .015, 'secondary_gain': 0.0}} or
            spec['workload'] != {'gradient_batches': 0, 'jvp_batches': 0,
                                 'ggn_or_cg_operations': 0, 'final_forward_batches': 32,
                                 'base_forward_batches': 32, 'total_forward_batches': 64,
                                 'new_candidate_artifacts': 0} or
            spec['paths']['result_path'] !=
                f'experiments/attempts/{ATTEMPT}/result.json'):
        raise ValueError('Attempt123 frozen reevaluation plan mismatch')
    return spec


def refuse_result(path):
    if path.exists() or path.is_symlink():
        raise ValueError('Attempt123 result already exists')


def validate_frozen_candidates(spec, torch):
    """Complete candidate audit. Call this before any historical-base access."""
    frozen = spec['frozen_attempt122']
    for key in ('spec', 'source', 'construction_manifest', 'result'):
        require_hash(path_of(frozen[key+'_path']), frozen[key+'_sha256'])
    old_spec = a122.load_spec()
    manifest = json.loads(path_of(frozen['construction_manifest_path']).read_text())
    candidate_path = path_of(frozen['candidate_path'])
    checkpoint_records = manifest['checkpoints']
    if (old_spec['paths']['candidate_path'] != frozen['candidate_path'] or
            old_spec['pilot_selection']['activation_probe']['rows'] != [0, 1024] or
            old_spec['relaxation']['functional_checkpoints'] != list(GRID) or
            old_spec['readout']['hidden_state_index'] != 14 or
            manifest['attempt_id'] != a122.ATTEMPT or
            manifest['spec_sha256'] != frozen['spec_sha256'] or
            manifest['constructor_sha256'] != frozen['source_sha256'] or
            manifest['pilot_selection'] != old_spec['pilot_selection'] or
            set(checkpoint_records) != {'1', '2', '4'} or
            manifest['candidate'] != {
                'path': frozen['candidate_path'],
                'serialized_sha256': frozen['candidate_serialized_sha256'],
                'tensor_order': list(NAMES), 'shape': [128, 2048],
                'dtype': 'contiguous_cpu_float32'} or
            any(checkpoint_records[str(k)]['response']['raw_sha256'] !=
                frozen['raw_response_sha256'][f'R_{k}'] or
                checkpoint_records[str(k)]['response']['shape'] != [128, 2048] or
                checkpoint_records[str(k)]['response']['dtype'] != 'contiguous_cpu_float32'
                for k in GRID)):
        raise ValueError('Attempt122 candidate construction provenance mismatch')
    require_hash(candidate_path, frozen['candidate_serialized_sha256'])
    artifact = torch.load(candidate_path, map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != list(NAMES):
        raise ValueError('Attempt122 candidate tensor order/inventory mismatch')
    validate_candidate_tensors(artifact, frozen, torch, a14.sha256_raw_float32_tensor)
    require_hash(candidate_path, frozen['candidate_serialized_sha256'])
    old_result = json.loads(path_of(frozen['result_path']).read_text())
    if (old_result['attempt_id'] != a122.ATTEMPT or
            old_result['construction_manifest_sha256'] != frozen['construction_manifest_sha256'] or
            old_result['candidate_serialized_sha256'] != frozen['candidate_serialized_sha256'] or
            list(old_result['reports']) != ['1', '2', '4'] or
            old_result['clean_heldout_validation'] is not False or
            old_result['same_specimen_exploratory_method_development'] is not True):
        raise ValueError('Attempt122 old evaluation provenance mismatch')
    validate_old_reports(old_result['reports'])
    return artifact, manifest, old_result


def validate_candidate_tensors(artifact, frozen, torch, raw_hash):
    if not isinstance(artifact, dict) or list(artifact) != list(NAMES):
        raise ValueError('Attempt122 candidate tensor order/inventory mismatch')
    for name in NAMES:
        value = a122.prior.validate_response(artifact[name], torch)
        if raw_hash(value, torch) != frozen['raw_response_sha256'][name]:
            raise ValueError('Attempt122 response raw SHA256 mismatch: '+name)
    return artifact


def validate_old_reports(reports):
    if list(reports) != ['1', '2', '4']:
        raise ValueError('Expected exactly three old Attempt122 reports')
    for name, report in reports.items():
        values = report['all_128_position_cosines']
        if (not isinstance(values, list) or len(values) != 128 or
                any(value is not None and (not isinstance(value, (int, float)) or
                    not math.isfinite(value) or abs(value) > 1) for value in values) or
                report['position_0_cosine'] != values[0] or
                report['positions_1_4_individual_cosines'] != values[1:5]):
            raise ValueError('Malformed old signed position report: '+name)
        for key, part in (('positions_1_4_mean_cosine', values[1:5]),
                          ('positions_1_127_mean_cosine', values[1:])):
            expected = None if any(value is None for value in part) else math.fsum(part)/len(part)
            if report[key] != expected:
                raise ValueError('Old signed position report mean mismatch: '+name)


def validate_matched_reference(spec, manifest, privileged):
    """Use the pinned Attempt103 privileged inventory after candidate validation."""
    reference = spec['matched_reference']
    for key in ('attempt103_spec', 'attempt103_source', 'attempt103_result', 'attempt106_spec'):
        require_hash(path_of(reference[key+'_path']), reference[key+'_sha256'])
    old103 = privileged.load_spec()
    old106 = json.loads(path_of(reference['attempt106_spec_path']).read_text())
    old103_result = json.loads(path_of(reference['attempt103_result_path']).read_text())
    old122 = a122.load_spec()
    if (old103['probe_rows'] != [0, 1024] or
            old103['probe']['sample_count'] != 1024 or
            old103['probe']['batch_size'] != 32 or
            old103['paths']['probe_path'] != old122['probe']['path'] or
            old103['model'] != old122['model'] or
            old103['eligible_tensors'] != old122['eligible_tensors'] or
            old103['readout'] != old122['readout'] or
            old103['final_checkpoint_files'] != manifest['source_checkpoint']['files'] or
            old106['base'] != old103['base'] or
            old106['final_checkpoint_files'] != old103['final_checkpoint_files'] or
            old106['probe'] != old103['probe'] or
            old106['probe_rows'] != old103['probe_rows'] or
            old106['target']['raw_float32_sha256'] != TARGET_SHA256 or
            old103_result['provenance']['matched_difference_raw_sha256'] != TARGET_SHA256 or
            old103_result['provenance']['base_checkpoint_files'] != old103['base']['files'] or
            old103_result['provenance']['final_checkpoint_files'] != old103['final_checkpoint_files']):
        raise ValueError('Matched Attempt103/106 target provenance contradiction')
    # Attempt103's result already exists; smoke_only bypasses only its output
    # refusal here, while validate_sources still hashes full frozen inputs.
    inventories = privileged.validate_sources(old103, smoke_only=True)
    return old103, inventories


def matched_probe(probe, spec, torch):
    reference = spec['matched_reference']
    if (probe.dtype != torch.int64 or tuple(probe.shape) != (10000, 128) or
            not probe.is_contiguous() or reference['probe_rows'] != [0, 1024] or
            reference['batch_size'] != 32 or reference['batches_per_model'] != 32):
        raise ValueError('Frozen full probe or matched-prefix inventory mismatch')
    prefix = probe[:1024].contiguous()
    if tuple(prefix.shape) != (1024, 128):
        raise ValueError('Malformed matched probe prefix')
    return prefix


def verified_target(final_mean, base_mean, spec, matched_difference, raw_hash, torch):
    target = matched_difference(final_mean, base_mean)
    a122.prior.validate_response(target, torch)
    digest = raw_hash(target, torch)
    if digest != spec['matched_reference']['target_raw_sha256']:
        raise ValueError('Matched historical target raw SHA256 mismatch')
    return target, digest


def optional_difference(left, right):
    return None if left is None or right is None else left-right


def position_gain_summary(later, first):
    x, y = later['all_128_position_cosines'], first['all_128_position_cosines']
    if len(x) != 128 or len(y) != 128:
        raise ValueError('Malformed signed position arrays')
    gains = [optional_difference(x[p], y[p]) for p in range(1, 128)]
    def mean(values):
        return None if any(value is None for value in values) else math.fsum(values)/len(values)
    return {'improved_position_count_1_127': sum(value is not None and value > 0 for value in gains),
            'comparable_position_count_1_127': sum(value is not None for value in gains),
            'mean_gain_positions_1_4': mean(gains[:4]),
            'mean_gain_positions_5_127': mean(gains[4:]),
            'mean_gain_positions_1_127': mean(gains)}


def summarize(matched, old):
    if list(matched) != list(GRID):
        raise ValueError('May score only frozen steps 1, 2, and 4')
    validate_old_reports(old)
    comparison = {}
    later = {}
    for k in GRID:
        report, previous = matched[k], old[str(k)]
        primary = report['positions_1_4_mean_cosine']
        secondary = report['positions_1_127_mean_cosine']
        old_primary = previous['positions_1_4_mean_cosine']
        old_secondary = previous['positions_1_127_mean_cosine']
        comparison[str(k)] = {
            'matched_primary': primary, 'matched_secondary': secondary,
            'old_mismatched_primary': old_primary,
            'old_mismatched_secondary': old_secondary,
            'delta_matched_minus_old_primary': optional_difference(primary, old_primary),
            'delta_matched_minus_old_secondary': optional_difference(secondary, old_secondary)}
        if k in (2, 4):
            first = matched[1]
            later[str(k)] = {
                'matched_primary_gain_vs_k1': optional_difference(
                    primary, first['positions_1_4_mean_cosine']),
                'matched_secondary_gain_vs_k1': optional_difference(
                    secondary, first['positions_1_127_mean_cosine']),
                **position_gain_summary(report, first)}
    return comparison, later, a122.classify(matched)


def run():
    import torch
    spec = load_spec()
    output = path_of(spec['paths']['result_path'])
    refuse_result(output)
    artifact, manifest, old_result = validate_frozen_candidates(spec, torch)
    print('ATTEMPT122_CANDIDATES_VALIDATED', flush=True)

    # Historical-base access begins here, after the complete candidate audit.
    record = spec['matched_reference']
    privileged = import_pinned(path_of(record['attempt103_source_path']),
                               record['attempt103_source_sha256'], 'attempt123_frozen_attempt103')
    old103, inventories = validate_matched_reference(spec, manifest, privileged)
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for the matched FP32 model readouts')
    full_probe = privileged.a.load_probe(privileged.resolve(old103['paths']['probe_path']),
        {**old103['probe'], 'sample_count': 10000}, torch)
    probe = matched_probe(full_probe, spec, torch)
    del full_probe
    base, final, loads = privileged.load_model_pair(old103)
    final_mean = privileged.base_probe_mean(final, probe, old103)
    if (tuple(final_mean.shape) != (128, 2048) or
            final_mean.dtype != torch.float64 or
            final_mean.device.type != 'cpu' or not final_mean.is_contiguous() or
            not bool(torch.isfinite(final_mean).all())):
        raise ValueError('Malformed matched final activation mean')
    final_mean_hash = hashlib.sha256(
        final_mean.numpy().astype('<f8', copy=False).tobytes()).hexdigest()
    if final_mean_hash != manifest['final_activation_mean']['raw_float64_sha256']:
        raise ValueError('Matched final mean differs from frozen Attempt122 A_final')
    base_mean = privileged.base_probe_mean(base, probe, old103)
    target, target_hash = verified_target(final_mean, base_mean, spec,
        privileged.matched_difference, privileged.a.sha256_raw_float32_tensor, torch)
    if any(parameter.grad is not None for model in (base, final) for parameter in model.parameters()):
        raise ValueError('Probe evaluation created model gradients')
    for directory, expected in ((privileged.resolve(old103['paths']['base_directory']),
                                 inventories['base_files']),
                                (privileged.resolve(old103['paths']['final_directory']),
                                 inventories['final_files'])):
        if privileged.a.checkpoint_file_records(directory) != expected:
            raise ValueError('Historical/base checkpoint changed during target construction')
    del base, final, base_mean, final_mean, probe
    gc.collect()
    torch.cuda.empty_cache()

    old_evaluation = a122.load_spec()['evaluation']
    evaluator = import_pinned(path_of(old_evaluation['validator_path']),
        old_evaluation['validator_sha256'], 'attempt123_signed_cosine_validator')
    matched = {k: a122.score_report(artifact[f'R_{k}'], target, evaluator) for k in GRID}
    comparison, later, interpretation = summarize(matched, old_result['reports'])
    frozen = spec['frozen_attempt122']
    require_hash(path_of(frozen['candidate_path']), frozen['candidate_serialized_sha256'])
    for name in NAMES:
        if a14.sha256_raw_float32_tensor(artifact[name], torch) != frozen['raw_response_sha256'][name]:
            raise ValueError('Frozen candidate changed during matched evaluation: '+name)
    result = {
        'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'privileged_postfreeze_diagnostic': True,
        'same_specimen_exploratory_method_development': True,
        'clean_heldout_validation': False,
        'candidate_source': 'frozen_Attempt122_candidate.pt_without_modification',
        'provenance': {
            'spec_sha256': SPEC_SHA256, 'source_sha256': sha256_file(Path(__file__)),
            'attempt122_spec_sha256': frozen['spec_sha256'],
            'attempt122_source_sha256': frozen['source_sha256'],
            'attempt122_construction_manifest_sha256': frozen['construction_manifest_sha256'],
            'attempt122_result_sha256': frozen['result_sha256'],
            'candidate_serialized_sha256': frozen['candidate_serialized_sha256'],
            'candidate_raw_sha256': frozen['raw_response_sha256'],
            'attempt103_spec_sha256': record['attempt103_spec_sha256'],
            'attempt103_source_sha256': record['attempt103_source_sha256'],
            'attempt106_spec_sha256': record['attempt106_spec_sha256'],
            'base_checkpoint_files': inventories['base_files'],
            'final_checkpoint_files': inventories['final_files'],
            'probe_serialized_sha256': old103['probe']['serialized_sha256'],
            'matched_target_raw_sha256': target_hash},
        'matched_target': {
            'definition': record['target_definition'], 'probe_rows': [0, 1024],
            'sequence_length': 128, 'batch_size': 32, 'batches_per_model': 32,
            'readout': 'block_13_output_hidden_states_14',
            'shape': [128, 2048], 'dtype': 'contiguous_cpu_float32',
            'raw_sha256': target_hash},
        'model_load_seconds': loads,
        'matched_reports': {str(k): matched[k] for k in GRID},
        'matched_vs_old_full10k': comparison,
        'later_vs_matched_k1': later,
        'development_set_interpretation': interpretation}
    refuse_result(output)
    a122.prior.atomic_json_publish(output, result)
    print('Wrote Attempt123 matched-1024 result', flush=True)
    return result


def main():
    try:
        run()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print('ERROR: '+str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
