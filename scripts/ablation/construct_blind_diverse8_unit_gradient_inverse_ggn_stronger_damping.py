#!/usr/bin/env python3
"""Attempt116: frozen blind Attempt113 construction with one stronger damping fraction."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '116_blind_diverse8_unit_gradient_inverse_ggn_stronger_damping_candidate'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SOURCE_PATH = Path(__file__).resolve()
PRIOR113_SPEC = PROJECT/'experiments/attempts/113_blind_diverse8_unit_gradient_inverse_ggn_candidate/spec.json'
PRIOR113_SOURCE = PROJECT/'scripts/ablation/construct_blind_diverse8_unit_gradient_inverse_ggn.py'
PRIOR114_SPEC = PROJECT/'experiments/attempts/114_blind_attempt113_cg_conditioning_diagnostic/spec.json'
PRIOR114_SOURCE = PROJECT/'scripts/ablation/diagnose_blind_attempt113_cg_conditioning.py'
PRIOR114_RESULT = PROJECT/'experiments/attempts/114_blind_attempt113_cg_conditioning_diagnostic/result.json'
PRIOR115_SPEC = PROJECT/'experiments/attempts/115_blind_attempt113_true_residual_gap_diagnostic/spec.json'
PRIOR115_SOURCE = PROJECT/'scripts/ablation/diagnose_blind_attempt113_true_residual_gap.py'
PRIOR115_RESULT = PROJECT/'experiments/attempts/115_blind_attempt113_true_residual_gap_diagnostic/result.json'
FROZEN = {
    'attempt113': {
        'commit': '5357ba00408f07b284de3126227fc5eb317e4f13',
        'spec_path': str(PRIOR113_SPEC.relative_to(PROJECT)),
        'spec_sha256': '7c7f508575e216cd29eee967f31101b1c614f85cc0cb0465becd4e0d1dc8bfb6',
        'source_path': str(PRIOR113_SOURCE.relative_to(PROJECT)),
        'source_sha256': '02bc38350e0f08997f3f2e6942c89594abb4b2e30e042973b930d89e59374ee9'},
    'attempt114': {
        'commit': '6b2e51ed87fd7147c5b304c4b7f4827dcdf6a4ba',
        'spec_path': str(PRIOR114_SPEC.relative_to(PROJECT)),
        'spec_sha256': 'bb477e5aaf66e9d33d75bedcc700a7b389bb47c7c3be00a25999d48e15b1e009',
        'source_path': str(PRIOR114_SOURCE.relative_to(PROJECT)),
        'source_sha256': '781a604db74189631cee26e2b9caea75a2b24c0faef7f183ab6649b5014379fc',
        'result_path': str(PRIOR114_RESULT.relative_to(PROJECT)),
        'result_sha256': '151d1074cdcf5d5d3152bf1e2f746940ea287de88fdc597a737db83cbf74a345',
        'blind_values': {
            'recursive_relative_residual_at_40': 2.0165128716207255,
            'rho_r40': 3018.261150714194,
            'ritz_condition_estimate': 19640.51752854543}},
    'attempt115': {
        'commit': '17d491948948d16f0071244180eecbffd47c8b81',
        'spec_path': str(PRIOR115_SPEC.relative_to(PROJECT)),
        'spec_sha256': '00a2c98311d37519d0f8d4a7bac1dac0cc0f7ae5aacefe25a061ce092ef1d17e',
        'source_path': str(PRIOR115_SOURCE.relative_to(PROJECT)),
        'source_sha256': '5f4b807bf051e6e7dbc4babed7d2ffa6d6d531057ee1ba03838aba63f00f771c',
        'result_path': str(PRIOR115_RESULT.relative_to(PROJECT)),
        'result_sha256': 'fe28aa13b1635985ea1304e5c61375b72959e0f98170659eab3e77e4c446fa95',
        'blind_values': {
            'interpretation': 'recursive_residual_faithful',
            'residual_gap_relative_to_recursive': 2.735084375822336e-06,
            'cosine_recursive_vs_true': 0.9999999999963717,
            'rhs_byte_identical_to_attempt113_checkpoint': True}}}
ARTIFACT_PATH = '/root/model-diff-scratch/artifacts/attempt116_blind_stronger_damping_inverse_ggn/candidate.pt'
CHECKPOINT_PATH = '/root/model-diff-scratch/checkpoints/attempt116_blind_stronger_damping_cg80_iter40.pt'
MANIFEST_PATH = f'experiments/attempts/{ATTEMPT}/construction-manifest.json'
PURPOSE = 'blind_final_checkpoint_diverse8_positive_unit_gradient_inverse_endpoint_GGN_stronger_damping_candidate'
FUTURE_COMPARISON = {
    'compare_only_after_both_construction_manifests_committed': True,
    'candidates': ['Attempt113_x_80_damping_fraction_1e-4',
                   'Attempt116_x_80_damping_fraction_1e-3'],
    'third_damping_on_same_specimen_after_historical_scores': False,
    'historical_comparison_during_construction': False,
    'blind_residuals_cannot_select_iterate': True}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _committed_sha(commit: str, path: Path) -> str:
    data = subprocess.run(
        ['git', '-c', f'safe.directory={PROJECT}', 'show',
         f'{commit}:{path.relative_to(PROJECT)}'],
        cwd=PROJECT, capture_output=True, check=True).stdout
    return hashlib.sha256(data).hexdigest()


def load_attempt113():
    """Import only the hash-pinned blind constructor, without model/data I/O."""
    for path, expected in ((PRIOR113_SPEC, FROZEN['attempt113']['spec_sha256']),
                           (PRIOR113_SOURCE, FROZEN['attempt113']['source_sha256'])):
        if (sha256_file(path) != expected or
                _committed_sha(FROZEN['attempt113']['commit'], path) != expected):
            raise ValueError('Frozen Attempt113 definition hash/commit mismatch')
    loader = importlib.util.spec_from_file_location('attempt113_blind_reused_by_116', PRIOR113_SOURCE)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    original = module.load_spec()
    return module, original


def expected_spec_from_attempt113(original: dict) -> dict:
    expected = copy.deepcopy(original)
    expected['attempt_id'] = ATTEMPT
    expected['purpose'] = PURPOSE
    expected['damping']['damping_fraction'] = 1e-3
    expected['paths'] = {
        'artifact_path': ARTIFACT_PATH, 'checkpoint_path': CHECKPOINT_PATH,
        'manifest_path': MANIFEST_PATH}
    expected['workload']['conservative_expected_wall_minutes'] = [40, 45]
    expected['workload']['timing_smoke_required_before_precise_projection'] = False
    expected['information_policy'][
        'damping_choice_informed_only_by_blind_attempt114_115_numerical_diagnostics'] = True
    return expected


def load_spec(original: dict, path: Path = SPEC_PATH) -> dict:
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt116 spec path changed')
    spec = json.loads(path.read_text())
    development = spec.pop('blind_development_linkage', None)
    expected_rhs = spec.pop('expected_attempt113_rhs', None)
    comparison = spec.pop('precommitted_future_comparison', None)
    if (spec != expected_spec_from_attempt113(original) or
            development != FROZEN or comparison != FUTURE_COMPARISON or
            not isinstance(expected_rhs, dict) or
            set(expected_rhs) != {'norm', 'matrix_raw_sha256',
                                  'ordered_matrix_hash_sha256'} or
            not isinstance(expected_rhs['norm'], (int, float)) or
            not math.isfinite(expected_rhs['norm']) or expected_rhs['norm'] <= 0 or
            not isinstance(expected_rhs['matrix_raw_sha256'], dict) or
            len(expected_rhs['matrix_raw_sha256']) != 98 or
            not isinstance(expected_rhs['ordered_matrix_hash_sha256'], str)):
        raise ValueError('Attempt116 spec differs from controlled Attempt113 construction')
    spec['blind_development_linkage'] = development
    spec['expected_attempt113_rhs'] = expected_rhs
    spec['precommitted_future_comparison'] = comparison
    return spec


def validate_blind_development_linkage(spec: dict):
    """Read only committed blind 114/115 diagnostics; never an evaluation."""
    for attempt, paths in (
            ('attempt114', (PRIOR114_SPEC, PRIOR114_SOURCE, PRIOR114_RESULT)),
            ('attempt115', (PRIOR115_SPEC, PRIOR115_SOURCE, PRIOR115_RESULT))):
        linkage = FROZEN[attempt]
        for path, key in zip(paths, ('spec_sha256', 'source_sha256', 'result_sha256')):
            if (sha256_file(path) != linkage[key] or
                    _committed_sha(linkage['commit'], path) != linkage[key]):
                raise ValueError('Frozen blind Attempt114/115 file/commit mismatch')
    r114 = json.loads(PRIOR114_RESULT.read_text())
    r115 = json.loads(PRIOR115_RESULT.read_text())
    if (r114.get('attempt_id') != '114_blind_attempt113_cg_conditioning_diagnostic' or
            r114.get('information_policy', {}).get('blind_diagnostic') is not True or
            r114['information_policy'].get('historical_base_access') is not False or
            r114.get('ggn_applications') != 1 or
            r114.get('frozen_attempt113', {}).get('commit') != FROZEN['attempt113']['commit'] or
            r114.get('residual_curvature', {}).get('relative_residual_at_40') !=
                FROZEN['attempt114']['blind_values']['recursive_relative_residual_at_40'] or
            r114['residual_curvature'].get('rho_r40') !=
                FROZEN['attempt114']['blind_values']['rho_r40'] or
            r114.get('ritz', {}).get('ritz_condition_estimate') !=
                FROZEN['attempt114']['blind_values']['ritz_condition_estimate'] or
            r115.get('attempt_id') != '115_blind_attempt113_true_residual_gap_diagnostic' or
            r115.get('information_policy', {}).get('blind_diagnostic') is not True or
            r115['information_policy'].get('historical_base_access') is not False or
            r115['information_policy'].get('prior_oracle_evaluation_access') is not False or
            r115.get('interpretation') !=
                FROZEN['attempt115']['blind_values']['interpretation'] or
            r115.get('residual_metrics', {}).get('residual_gap_relative_to_recursive') !=
                FROZEN['attempt115']['blind_values']['residual_gap_relative_to_recursive'] or
            r115['residual_metrics'].get('cosine_recursive_vs_true') !=
                FROZEN['attempt115']['blind_values']['cosine_recursive_vs_true'] or
            r115.get('rhs_reconstruction', {}).get('byte_identical_to_checkpoint_rhs') is not True or
            r115.get('ggn_applications_to_x40') != 1):
        raise ValueError('Frozen blind numerical diagnostic provenance/value mismatch')
    rhs = r115['rhs_reconstruction']
    if (spec['expected_attempt113_rhs'] != {
            'norm': rhs['norm'], 'matrix_raw_sha256': rhs['matrix_raw_sha256'],
            'ordered_matrix_hash_sha256': rhs['ordered_matrix_hash_sha256']} or
            rhs['ordered_matrix_hash_sha256'] !=
                'a90785341961a0a2a08e7949961cb105867cf2509013b01b65c2013322ead574' or
            rhs['norm'] != 0.6924963677590905):
        raise ValueError('Frozen Attempt113 blind RHS identity mismatch')
    return {'attempt114_result_sha256': FROZEN['attempt114']['result_sha256'],
            'attempt115_result_sha256': FROZEN['attempt115']['result_sha256'],
            'expected_rhs_aggregate_sha256': rhs['ordered_matrix_hash_sha256']}


def rayleigh_and_lambda_116(module, eligible, direction, damping_fraction):
    if damping_fraction != 1e-3:
        raise ValueError('Attempt116 requires exactly 1e-3 blind Rayleigh damping fraction')
    numerator = module.matrixwise_grad_dot(eligible, direction)
    denominator = module.matrixwise_dot(direction, direction)
    rho = numerator/denominator
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError('Nonpositive/nonfinite blind RHS Rayleigh quotient')
    damping = 1e-3*rho
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError('Nonpositive/nonfinite blind damping')
    return {'bGb': numerator, 'bb': denominator, 'rho_b': rho,
            'damping_fraction': 1e-3, 'lambda': damping}


def require_exact_rhs(record: dict, expected: dict, *, smoke: bool = False):
    if smoke:
        return
    if (record.get('matrix_raw_sha256') != expected['matrix_raw_sha256'] or
            record.get('ordered_matrix_hash_sha256') !=
                expected['ordered_matrix_hash_sha256'] or
            record.get('norm') != expected['norm']):
        raise ValueError('Attempt116 RHS differs from frozen blind Attempt113 RHS')


@contextlib.contextmanager
def bind_attempt116(module, spec: dict):
    """Reuse frozen kernels, changing only named runtime configuration in memory."""
    original_rhs = module.build_blind_rhs
    original_publish = module.publish_candidate

    def checked_rhs(model, corpora, eligible, inventory, *, smoke=False):
        rhs, diagnostics, record = original_rhs(
            model, corpora, eligible, inventory, smoke=smoke)
        require_exact_rhs(record, spec['expected_attempt113_rhs'], smoke=smoke)
        return rhs, diagnostics, record

    def enriched_publish(artifact_path, manifest_path, checkpoint_path,
                         candidate, response_raw, response_inverse, manifest,
                         inventory):
        manifest['blind_development_linkage'] = copy.deepcopy(
            spec['blind_development_linkage'])
        manifest['expected_attempt113_rhs'] = copy.deepcopy(
            spec['expected_attempt113_rhs'])
        manifest['damping_choice_informed_only_by_blind_attempt114_115_numerical_diagnostics'] = True
        manifest['precommitted_future_comparison'] = copy.deepcopy(
            spec['precommitted_future_comparison'])
        if (manifest.get('rayleigh', {}).get('damping_fraction') != 1e-3 or
                manifest['rayleigh'].get('lambda') !=
                    1e-3*manifest['rayleigh']['rho_b'] or
                manifest.get('b_blind', {}).get('ordered_matrix_hash_sha256') !=
                    spec['expected_attempt113_rhs']['ordered_matrix_hash_sha256'] or
                manifest.get('solver', {}).get('operator_applications') != 80 or
                manifest.get('candidate', {}).get('name') != 'x_80'):
            raise ValueError('Attempt116 frozen damping/RHS/iterate publication audit failed')
        return original_publish(artifact_path, manifest_path, checkpoint_path,
                                candidate, response_raw, response_inverse,
                                manifest, inventory)

    replacements = {
        'ATTEMPT': ATTEMPT, 'SPEC_PATH': SPEC_PATH,
        'SPEC_SHA256': sha256_file(SPEC_PATH), '__file__': str(SOURCE_PATH),
        'load_spec': lambda: copy.deepcopy(spec),
        'rayleigh_and_lambda': lambda eligible, direction, fraction:
            rayleigh_and_lambda_116(module, eligible, direction, fraction),
        'build_blind_rhs': checked_rhs, 'publish_candidate': enriched_publish}
    old = {name: getattr(module, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(module, name, value)


def construct(*, resume=False, timing_smoke=False):
    module, original = load_attempt113()
    spec = load_spec(original)
    validate_blind_development_linkage(spec)
    # The frozen Attempt113 constructor handles provenance, progress, strict
    # JVP/GGN, iteration-40 checkpointing, --resume and atomic publication.
    with bind_attempt116(module, spec):
        return module.construct(resume=resume, timing_smoke=timing_smoke)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--timing-smoke', action='store_true')
    args = parser.parse_args()
    result = construct(resume=args.resume, timing_smoke=args.timing_smoke)
    if args.timing_smoke:
        print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
