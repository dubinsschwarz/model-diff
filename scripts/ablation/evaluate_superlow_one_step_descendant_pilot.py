#!/usr/bin/env python3
"""Privileged Attempt200 scoring. The blind freeze audit precedes historical access."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys

SOURCE = Path(__file__).with_name('run_superlow_one_step_descendant_pilot.py')
loader = importlib.util.spec_from_file_location('attempt200_constructor', SOURCE)
constructor = importlib.util.module_from_spec(loader)
loader.loader.exec_module(constructor)

CANONICAL_HELPER_SOURCE = Path(__file__).with_name('diagnose_privileged_base_soft_target_rollback.py')
CANONICAL_HELPER_SHA256 = '73b3ca864916c547ccf20d5b07889ae6b45c01276da8606ee65dfa0c7de4a0d6'
HISTORICAL_TARGET_RMS = 33.51964294937312
PORTABILITY_REL_TOL = 1e-5
PORTABILITY_ABS_TOL = 1e-4


def load_canonical_helper():
    """Privileged import, called only after freeze and a historical-hash mismatch."""
    constructor.require_hash(CANONICAL_HELPER_SOURCE, CANONICAL_HELPER_SHA256)
    loader = importlib.util.spec_from_file_location('attempt200_canonical_attempt103', CANONICAL_HELPER_SOURCE)
    helper = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(helper)
    return helper


def validate_target(target, expected_hash, final_mean, base_mean, final, base,
                    probe, canonical_spec, device):
    """Validate numerical portability without changing the target or candidate."""
    import torch
    constructor.validate_tensor(target, (128, 2048), torch.float32)
    local_hash = constructor.raw_hash(target)
    exact_match = local_hash == expected_hash
    rms = math.sqrt(float(target.double().square().sum()) / 128)
    absolute_difference = abs(rms - HISTORICAL_TARGET_RMS)
    provenance = {
        'historical_expected_target_sha256': expected_hash,
        'local_target_sha256': local_hash,
        'historical_target_sha256_exact_match': exact_match,
        'portability_fallback_used': not exact_match,
        'canonical_helper_source_sha256': CANONICAL_HELPER_SHA256,
        # None means the preferred exact-hash path needed no helper comparison.
        'attempt200_vs_canonical_F_bitwise_equal': None,
        'attempt200_vs_canonical_B_bitwise_equal': None,
        'attempt200_vs_canonical_target_bitwise_equal': None,
        'local_target_pooled_rms_position_norm': rms,
        'historical_target_pooled_rms_position_norm': HISTORICAL_TARGET_RMS,
        'target_pooled_rms_position_norm_absolute_difference': absolute_difference,
        'target_pooled_rms_position_norm_relative_difference': absolute_difference / HISTORICAL_TARGET_RMS,
        'portability_tolerances': {'rel_tol': PORTABILITY_REL_TOL, 'abs_tol': PORTABILITY_ABS_TOL}}
    if exact_match:
        return provenance

    constructor.log('Historical target SHA mismatch; validating canonical Attempt103 portability')
    helper = load_canonical_helper()
    prefix = probe[:1024].contiguous()
    canonical_base_mean = helper.base_probe_mean(base, prefix, canonical_spec,
        progress=helper.progress_printer('Canonical B portability probe'))
    # Preserve the same model objects, with only one model on the GPU at a time.
    base.to('cpu')
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    final.to(device)
    canonical_final_mean = helper.base_probe_mean(final, prefix, canonical_spec,
        progress=helper.progress_printer('Canonical F portability probe'))
    canonical_target = helper.matched_difference(canonical_final_mean, canonical_base_mean)
    for label, actual, canonical, dtype in (
            ('F', final_mean, canonical_final_mean, torch.float64),
            ('B', base_mean, canonical_base_mean, torch.float64),
            ('target', target, canonical_target, torch.float32)):
        constructor.validate_tensor(actual, (128, 2048), dtype)
        constructor.validate_tensor(canonical, (128, 2048), dtype)
        equal = torch.equal(actual, canonical) and constructor.raw_hash(actual) == constructor.raw_hash(canonical)
        provenance[f'attempt200_vs_canonical_{label}_bitwise_equal'] = equal
        if not equal:
            raise ValueError(f'Canonical Attempt103 portability helper {label} bitwise mismatch')
    if not math.isclose(rms, HISTORICAL_TARGET_RMS,
                        rel_tol=PORTABILITY_REL_TOL, abs_tol=PORTABILITY_ABS_TOL):
        raise ValueError('Historical target pooled RMS outside portability tolerances')
    constructor.log(f'Canonical portability validation passed: local RMS={rms:.14g}, '
                    f'relative difference={absolute_difference / HISTORICAL_TARGET_RMS:.12g}')
    return provenance


def signed_report(candidate, target):
    import torch
    for tensor in (candidate, target):
        constructor.validate_tensor(tensor, (128, 2048), torch.float32)
    values = []
    for position in range(128):
        x, y = candidate[position].double(), target[position].double()
        nx, ny = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
        values.append(None if float(nx) == 0 or float(ny) == 0 else float(torch.dot(x, y) / (nx * ny)))
    mean = lambda xs: None if any(x is None for x in xs) else math.fsum(xs) / len(xs)
    return {'position_0_cosine': values[0], 'positions_1_4_individual_cosines': values[1:5],
            'positions_1_4_mean_cosine': mean(values[1:5]),
            'positions_1_127_mean_cosine': mean(values[1:]), 'all_128_position_cosines': values}


def reference_comparison(report, historical_result):
    """Historical scores are descriptive only; no candidate decisions occur here."""
    if historical_result['attempt_id'] != '132_expanded_fineweb_superlow_stringency':
        raise ValueError('Unexpected linear-Jg reference experiment')
    reference = historical_result['signed_position_reports']['response_super_low']
    values = reference['all_128_position_cosines']
    if len(values) != 128 or any(x is not None and not math.isfinite(x) for x in values):
        raise ValueError('Malformed frozen linear-Jg reference')
    deltas = {}
    for label, key, positions in (
            ('primary', 'positions_1_4_mean_cosine', values[1:5]),
            ('secondary', 'positions_1_127_mean_cosine', values[1:])):
        expected = None if any(x is None for x in positions) else math.fsum(positions)/len(positions)
        if reference[key] != expected:
            raise ValueError('Linear-Jg reference metric mismatch')
        deltas[label] = None if report[key] is None or reference[key] is None else report[key]-reference[key]
    return {'definition': 'Attempt132 frozen positive linearized Jg response_super_low',
            'signed_position_report': reference, 'endpoint_minus_linear_Jg': deltas}


def evaluate(device='cuda'):
    import torch
    spec = constructor.load_spec()
    result_path = constructor.path_of(spec['paths']['result'])
    if result_path.exists() or result_path.is_symlink():
        raise ValueError('Evaluation result already exists')
    tensors, manifest, receipt = constructor.validate_frozen(spec)
    constructor.log('SUPERLOW_ONE_STEP_DESCENDANT_FREEZE_VALIDATED; historical access enabled')

    # This is the first historical read. No historical helper is imported above.
    base_spec = constructor.read_record(spec['evaluation']['base_spec'])
    base_dir = constructor.path_of(base_spec['paths']['base_directory'])
    base_files = base_spec['base']['files']
    constructor.validate_checkpoint(base_dir, base_files)
    gradient_helper, readout_helper = constructor.load_helpers(spec)
    final, _ = gradient_helper.load_local_model(constructor.MODEL_DIR, device, torch)
    final.config.use_cache = False
    final_mean = constructor.mean_activation(final, tensors['probe'], spec, readout_helper, 'F evaluation')
    if constructor.raw_hash(final_mean) != manifest['activation_mean_float64_sha256']['F']:
        raise ValueError('Matched F activation mean differs from construction')
    # Keep this exact F for the conditional independent canonical readout.
    final.to('cpu')
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    base, _ = gradient_helper.load_local_model(base_dir, device, torch)
    base.config.use_cache = False
    base_mean = constructor.mean_activation(base, tensors['probe'], spec, readout_helper, 'B evaluation')
    target = constructor.endpoint_response(final_mean, base_mean)
    target_provenance = validate_target(target, spec['evaluation']['target_raw_tensor_sha256'],
        final_mean, base_mean, final, base, tensors['probe'], base_spec, device)
    target_hash = target_provenance['local_target_sha256']
    del base, final
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    report = signed_report(tensors['candidate'], target)
    # Reading Attempt132 scores is allowed only after the frozen candidate gate.
    reference_path = constructor.path_of(spec['evaluation']['linear_reference_path'])
    reference_hash = spec['evaluation']['linear_reference_sha256']
    constructor.require_hash(reference_path, reference_hash)
    historical_result = json.loads(reference_path.read_text())
    if historical_result['construction_manifest_sha256'] != spec['ranking_manifest']['sha256']:
        raise ValueError('Linear-Jg reference construction linkage mismatch')
    comparison = reference_comparison(report, historical_result)
    constructor.require_hash(reference_path, reference_hash)
    constructor.validate_checkpoint(base_dir, base_files)
    constructor.validate_frozen(spec)
    result = {'format_version': 1, 'attempt_id': constructor.ATTEMPT,
        'same_specimen_exploratory_diagnostic': True, 'held_out_validation': False,
        'candidate_definition': 'actual nonlinear one-step endpoint R = A_F - A_G',
        'target_definition': 'matched first-1024 probe T = A_F - A_B; float64 subtraction then FP32',
        'signed_position_report': report, 'linear_Jg_comparison': comparison,
        'accepted_parameter_displacement_and_line_search': manifest['step'],
        'provenance': {'construction_manifest_sha256': receipt['construction_manifest_sha256'],
            'spec_sha256': constructor.SPEC_SHA256, 'constructor_sha256': manifest['constructor_sha256'],
            'evaluator_sha256': constructor.sha256_file(Path(__file__)),
            'candidate': manifest['artifacts']['candidate'], 'target_raw_tensor_sha256': target_hash,
            'base_checkpoint_files': base_files, 'reference_result_sha256': reference_hash,
            **target_provenance},
        'functional_evaluation': False}
    constructor.publish(result_path, result)
    constructor.log('Wrote ' + str(result_path))
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
