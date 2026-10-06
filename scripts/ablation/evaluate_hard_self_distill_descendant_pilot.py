#!/usr/bin/env python3
"""Privileged scoring of frozen Attempt201 endpoints; no candidate changes."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import math
from pathlib import Path
import statistics
import sys

SOURCE = Path(__file__).with_name('run_hard_self_distill_descendant_pilot.py')
loader = importlib.util.spec_from_file_location('attempt201_constructor', SOURCE)
c = importlib.util.module_from_spec(loader)
loader.loader.exec_module(c)

EVALUATION_SPEC_PATH = c.SPEC_PATH.with_name('evaluation-spec.json')
EVALUATION_SPEC_SHA256 = 'ef0a8603a6fef0d9f8a272007197d641056a3d76050b7e7f1518fa09a7a6c431'


def load_evaluation_spec():
    spec = c.b.read_record({'path': str(EVALUATION_SPEC_PATH), 'sha256': EVALUATION_SPEC_SHA256})
    if (spec['attempt_id'] != c.ATTEMPT or spec['functional_evaluation'] is not False or
            spec['raw_mean_role'] != 'descriptive_only'):
        raise ValueError('Post-freeze evaluation plan mismatch')
    return spec


def support_interpretation(reports, support):
    if list(reports) != list(c.NAMES):
        raise ValueError('Only the ten fixed candidates may be interpreted')
    primary = [reports[name]['positions_1_4_mean_cosine'] for name in c.SEED_NAMES]
    secondary = [reports[name]['positions_1_127_mean_cosine'] for name in c.SEED_NAMES]
    consensus = reports['response_seed_consensus']
    cp, cs = consensus['positions_1_4_mean_cosine'], consensus['positions_1_127_mean_cosine']
    valid = lambda x: x is not None and math.isfinite(x)
    both = sum(valid(p) and valid(s) and p > 0 and s > 0 for p, s in zip(primary, secondary))
    mp = statistics.median(primary) if all(map(valid, primary)) else None
    ms = statistics.median(secondary) if all(map(valid, secondary)) else None
    strong, moderate = support['strong_support'], support['moderate_support']
    defined = all(map(valid, (mp, ms, cp, cs)))
    is_strong = defined and (both >= strong['both_positive_seed_count_min'] and
        mp >= strong['median_primary_min'] and ms >= strong['median_secondary_min'] and
        cp >= strong['consensus_primary_min'] and cs >= strong['consensus_secondary_min'])
    is_moderate = defined and (both >= moderate['both_positive_seed_count_min'] and
        mp > moderate['median_primary_gt'] and ms > moderate['median_secondary_gt'] and
        cp > moderate['consensus_primary_gt'] and cs > moderate['consensus_secondary_gt'])
    category = 'strong_support' if is_strong else 'moderate_support' if is_moderate else 'no_support'
    summary = {'both_positive_seed_count': both, 'median_seed_primary': mp, 'median_seed_secondary': ms,
        'consensus_primary': cp, 'consensus_secondary': cs,
        'individual_seed_scores': {name: {'primary': p, 'secondary': s}
            for name, p, s in zip(c.SEED_NAMES, primary, secondary)},
        'raw_mean_descriptive': reports['response_raw_mean'], 'best_seed_selection': False}
    return summary, category


def historical_context(record):
    historical = c.b.read_record(record)
    if historical['attempt_id'] != '128_self_sample_score_gradient_pilot':
        raise ValueError('Historical descriptive context experiment mismatch')
    return {'definition': 'Attempt128 positive linear self-score-gradient responses; descriptive only',
        'seed_distribution_summary': historical['seed_distribution_summary'],
        'development_set_classification': historical['development_set_classification'],
        'consensus_signed_position_report': historical['signed_position_reports']['response_seed_consensus'],
        'seed_signed_position_reports': {name: historical['signed_position_reports'][name] for name in c.SEED_NAMES}}


def evaluate(device='cuda'):
    import torch
    spec = c.load_spec()
    output = c.b.path_of(spec['paths']['result'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt201 result already exists')
    candidates, data, manifest, receipt = c.validate_frozen(spec)
    c.b.log('HARD_SELF_DISTILL_FREEZE_VALIDATED; historical access enabled')

    # All privileged imports/reads are below the complete blind freeze audit.
    evaluation = load_evaluation_spec()
    evaluator = c.import_pinned(evaluation['helper'], 'attempt201_pinned_portable_target')
    portability = evaluation['portability']
    if portability != {'canonical_helper_source_sha256': evaluator.CANONICAL_HELPER_SHA256,
            'historical_rms': evaluator.HISTORICAL_TARGET_RMS,
            'rel_tol': evaluator.PORTABILITY_REL_TOL, 'abs_tol': evaluator.PORTABILITY_ABS_TOL}:
        raise ValueError('Precommitted historical target portability strategy changed')
    base_spec = c.b.read_record(evaluation['base_spec'])
    base_dir = c.b.path_of(base_spec['paths']['base_directory'])
    c.b.validate_checkpoint(base_dir, base_spec['base']['files'])
    gradient, readout = c.helpers(spec)
    final, _ = gradient.load_local_model(c.b.MODEL_DIR, device, torch)
    final.config.use_cache = False
    final_mean = c.b.mean_activation(final, data['probe'], spec, readout, 'F evaluation')
    if c.b.raw_hash(final_mean) != manifest['blind_data']['raw_sha256']['F_mean']:
        raise ValueError('Evaluation F activation mean differs from frozen construction')
    final.to('cpu')
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    base, _ = gradient.load_local_model(base_dir, device, torch)
    base.config.use_cache = False
    base_mean = c.b.mean_activation(base, data['probe'], spec, readout, 'B evaluation')
    target = c.b.endpoint_response(final_mean, base_mean)
    target_provenance = evaluator.validate_target(target, evaluation['target_raw_tensor_sha256'],
        final_mean, base_mean, final, base, data['probe'], base_spec, device)
    target_provenance = {key.replace('attempt200_vs_canonical_', 'attempt201_vs_canonical_'): value
                         for key, value in target_provenance.items()}
    del final, base
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    reports = {name: evaluator.signed_report(candidates[name], target) for name in c.NAMES}
    summary, category = support_interpretation(reports, spec['support'])
    context = historical_context(evaluation['historical_context'])
    c.b.validate_checkpoint(base_dir, base_spec['base']['files'])
    c.validate_frozen(spec)
    result = {'format_version': 1, 'attempt_id': c.ATTEMPT,
        'same_specimen_exploratory_diagnostic': True, 'held_out_validation': False,
        'candidate_definition': 'actual four-step hard self-distillation endpoints R_seed = A_F - A_G_seed',
        'signed_position_reports': reports, 'seed_distribution_summary': summary,
        'development_set_classification': category, 'support_thresholds': spec['support'],
        'historical_gradient_context': context, 'descendant_training_diagnostics': manifest['seeds'],
        'provenance': {'spec_sha256': c.SPEC_SHA256, 'constructor_sha256': manifest['constructor_sha256'],
            'construction_manifest_sha256': receipt['construction_manifest_sha256'],
            'evaluator_sha256': c.b.sha256_file(Path(__file__)),
            'evaluation_spec_sha256': EVALUATION_SPEC_SHA256,
            'target_validator_source_sha256': evaluation['helper']['sha256'],
            'candidate': manifest['candidate'], 'base_checkpoint_files': base_spec['base']['files'],
            'historical_context_result_sha256': evaluation['historical_context']['sha256'],
            **target_provenance}, 'functional_evaluation': False}
    c.b.publish(output, result)
    c.b.log('Wrote ' + str(output))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    try:
        evaluate(args.device)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
