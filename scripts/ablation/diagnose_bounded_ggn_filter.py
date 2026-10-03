#!/usr/bin/env python3
"""Attempt119: one post-oracle bounded filter of frozen Attempt116 responses."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '119_postoracle_bounded_ggn_filter_diagnostic'
PRIOR_SOURCE = Path(__file__).with_name('evaluate_postfreeze_blind_inverse_damping_comparison.py')
PRIOR_SOURCE_SHA256 = '1ee856a9ce8a6218ceaa3699a772cda4b5194a4fb550a85c1854156aaacf61f9'
if hashlib.sha256(PRIOR_SOURCE.read_bytes()).hexdigest() != PRIOR_SOURCE_SHA256:
    raise ValueError('Frozen Attempt117 validator source SHA256 mismatch')
_loader = importlib.util.spec_from_file_location('attempt119_frozen_validator117', PRIOR_SOURCE)
prior = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(prior)

LAMBDA = 0.5615508153958936
RESIDUAL_116 = 0.030534033703066656
UPSTREAM = copy.deepcopy(prior.CONSTRUCTIONS['116'])
ORACLE = copy.deepcopy(prior.FROZEN_SPEC['oracle'])
FROZEN_SPEC = {
    'format_version': 1,
    'attempt_id': ATTEMPT,
    'purpose': 'postoracle_single_bounded_endpoint_ggn_response_filter_diagnostic',
    'upstream': {
        'attempt116': UPSTREAM,
        'validator117': {'path': 'scripts/ablation/evaluate_postfreeze_blind_inverse_damping_comparison.py',
                         'sha256': PRIOR_SOURCE_SHA256},
        'frozen_rhs_aggregate_sha256': prior.FROZEN_SPEC['cross_candidate_identity']['rhs_aggregate_sha256'],
        'ggn_operator': 'exact_endpoint_categorical_JtFJ',
        'artifact_inventory': ['x_80', 'response_raw_rhs', 'response_inverse'],
        'response_shape': [128, 2048],
        'response_dtype': 'contiguous_CPU_float32',
    },
    'construction': {
        'only_candidate': 'bounded_filter',
        'exact_realized_definition': 'y_80=b_blind-lambda*x_80',
        'exact_functional_definition': 'J_y_80=J_b_blind-lambda*J_x_80',
        'response_arithmetic': 'raw64=raw.to(torch.float64); inverse64=inverse.to(torch.float64); bounded=(raw64-lambda*inverse64).to(torch.float32).contiguous()',
        'lambda': LAMBDA,
        'rho_b': 561.5508153958936,
        'damping_fraction': 0.001,
        'iterations': 80,
        'attempt116_final_relative_linear_residual': RESIDUAL_116,
        'approximation_caveat': 'Attempt116_x_80_is_an_approximate_80_step_solve_with_relative_residual_0.030534033703066656',
        'exact_solve_reference_only': 'if_x=(G+lambda_I)^-1_b_then_y=G(G+lambda_I)^-1_b_with_PSD_multiplier_mu/(mu+lambda)_in_[0,1]',
        'exact_spectral_identity_claimed_for_y_80': False,
        'renormalize_bounded': False,
        'candidate_rescaling': False,
        'sign_selection': False,
        'alternate_lambda': False,
        'alternate_iterate': False,
        'scalar_sweep': False,
        'interpolation_sweep': False,
    },
    'barrier': {
        'artifact_and_all_98_x_matrices_validated_before_oracle': True,
        'two_response_raw_hashes_validated_before_oracle': True,
        'bounded_raw_sha256_frozen_before_oracle': True,
        'all_oracle_free_diagnostics_frozen_before_oracle': True,
        'marker': 'BOUNDED_CONSTRUCTION_FROZEN',
    },
    'oracle': ORACLE,
    'metrics': {
        'candidate_order': ['raw_rhs', 'inverse_1e-3', 'bounded_filter'],
        'main_comparison': 'bounded_filter_minus_raw_rhs',
        'inverse_role': 'descriptive_context_only',
        'dtype': 'cpu_float64',
        'position_0': 'separate',
        'all_positions': list(range(128)),
        'primary_positions': [1, 2, 3, 4],
        'secondary_positions': list(range(1, 128)),
        'aggregation': 'arithmetic_mean_of_individual_signed_position_cosines',
        'zero_norm_cosine': None,
        'required_mean_if_any_null': None,
        'flattened_cosine': False,
        'absolute_cosine': False,
    },
    'interpretation': {
        'categories_in_order': ['clear_support', 'moderate_support', 'no_meaningful_support'],
        'clear': {'minimum_primary_gain': 0.03, 'minimum_secondary_gain': 0.02},
        'moderate': {'minimum_primary_gain': 0.015, 'minimum_secondary_gain': 0.0},
        'scope': 'Suppressing_components_with_weak_support_under_the_frozen_endpoint_GGN_without_inverse_amplification_improves_the_historical_functional_trace_present_in_the_raw_blind_RHS',
        'positive_caveat': 'Does_not_show_that_G_identifies_historical_signal_generally',
        'negative_conclusion': 'Simple_bounded_spectral_filter_does_not_improve_current_blind_RHS_on_this_specimen;_argues_against_further_simple_scalar_functions_of_same_G_RHS_pair_and_motivates_multi_step_generic_relaxation',
        'significance_claim': False,
    },
    'information_policy': {
        'postoracle_mechanistic_diagnostic': True,
        'same_specimen_postoracle_method_development': True,
        'clean_blind_validation': False,
        'candidate_constructed_from_frozen_blind_responses_only': True,
        'candidate_hash_frozen_before_oracle_access_within_run': True,
        'oracle_access_after_candidate_hash': True,
        'model_loading': False,
        'new_gradient': False,
        'new_jvp': False,
        'ggn_vector_product': False,
        'cg_solve': False,
        'alternate_iterate': False,
        'alternate_damping': False,
        'scalar_sweep': False,
        'interpolation_sweep': False,
        'candidate_rescaling': False,
        'sign_selection': False,
        'normalization_of_bounded_candidate': False,
        'oracle_driven_candidate_selection': False,
        'position_selection': False,
    },
    'output_path': f'experiments/attempts/{ATTEMPT}/result.json',
    'output_policy': {'overwrite': False, 'large_artifact': False},
}


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def validate_response(tensor):
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or
            tensor.dtype != torch.float32 or not tensor.is_contiguous() or
            tuple(tensor.shape) != (128, 2048) or not bool(torch.isfinite(tensor).all())):
        raise ValueError('Response must be finite contiguous CPU FP32 [128,2048]')
    return tensor


def bounded_response(raw, inverse, damping=LAMBDA):
    """The only constructed candidate; exactly one FP32 cast after FP64 arithmetic."""
    if damping != LAMBDA:
        raise ValueError('Attempt119 lambda differs from frozen Attempt116 value')
    validate_response(raw)
    validate_response(inverse)
    bounded = (raw.to(torch.float64) - damping * inverse.to(torch.float64)).to(torch.float32).contiguous()
    return validate_response(bounded)


def norm_ratios(bounded, raw):
    validate_response(bounded)
    validate_response(raw)
    a, b = bounded.double(), raw.double()
    total_bounded, total_raw = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
    per_bounded, per_raw = torch.linalg.vector_norm(a, dim=1), torch.linalg.vector_norm(b, dim=1)
    return {
        'bounded_to_raw_total_norm_ratio': None if total_raw == 0 else total_bounded / total_raw,
        'bounded_to_raw_per_position_norm_ratios': [
            None if float(denominator) == 0 else float(numerator / denominator)
            for numerator, denominator in zip(per_bounded, per_raw)],
    }


def freeze_blind_construction(raw, inverse, record=UPSTREAM):
    """Finish all candidate construction and oracle-free diagnostics before the barrier."""
    if (record != UPSTREAM or
            prior.raw_float32_sha256(validate_response(raw)) != record['response_raw_rhs_sha256'] or
            prior.raw_float32_sha256(validate_response(inverse)) != record['response_inverse_sha256']):
        raise ValueError('Frozen Attempt116 response identity mismatch')
    bounded = bounded_response(raw, inverse)
    bounded_hash = prior.raw_float32_sha256(bounded)
    diagnostics = {
        'bounded_response_raw_sha256': bounded_hash,
        'raw_response_raw_sha256': record['response_raw_rhs_sha256'],
        'inverse_response_raw_sha256': record['response_inverse_sha256'],
        'lambda': LAMBDA,
        'attempt116_final_relative_linear_residual': RESIDUAL_116,
        'exact_spectral_identity_claimed_for_approximate_x80': False,
        **norm_ratios(bounded, raw),
        'bounded_vs_raw_signed_cosine': prior.cosine_report(bounded, raw),
        'bounded_vs_inverse_signed_cosine': prior.cosine_report(bounded, inverse),
    }
    construction = {
        'upstream': copy.deepcopy(FROZEN_SPEC['upstream']),
        'definition': copy.deepcopy(FROZEN_SPEC['construction']),
        'oracle_free_diagnostics': diagnostics,
        'bounded_hash_frozen_before_oracle_access': True,
    }
    print('BOUNDED_CONSTRUCTION_FROZEN', flush=True)
    print('bounded_raw_sha256='+bounded_hash, flush=True)
    return bounded, construction


def interpret(primary_gain, secondary_gain):
    if primary_gain is None or secondary_gain is None:
        return 'undefined_required_position_cosine'
    criteria = FROZEN_SPEC['interpretation']
    if (primary_gain >= criteria['clear']['minimum_primary_gain'] and
            secondary_gain >= criteria['clear']['minimum_secondary_gain']):
        return 'clear_support'
    if (primary_gain >= criteria['moderate']['minimum_primary_gain'] and
            secondary_gain >= criteria['moderate']['minimum_secondary_gain']):
        return 'moderate_support'
    return 'no_meaningful_support'


def compare_to_oracle(raw, inverse, bounded, difference):
    reports = {
        'raw_rhs': prior.cosine_report(raw, difference),
        'inverse_1e-3': prior.cosine_report(inverse, difference),
        'bounded_filter': prior.cosine_report(bounded, difference),
    }
    raw_report, bounded_report = reports['raw_rhs'], reports['bounded_filter']
    primary_raw, primary_bounded = (raw_report['positions_1_4_mean_cosine'],
                                     bounded_report['positions_1_4_mean_cosine'])
    secondary_raw, secondary_bounded = (raw_report['positions_1_127_mean_cosine'],
                                         bounded_report['positions_1_127_mean_cosine'])
    dp = None if primary_raw is None or primary_bounded is None else primary_bounded - primary_raw
    ds = None if secondary_raw is None or secondary_bounded is None else secondary_bounded - secondary_raw
    return {'candidate_reports': reports,
            'gain_primary_bounded_minus_raw': dp,
            'gain_secondary_bounded_minus_raw': ds,
            'interpretation': interpret(dp, ds)}


def load_frozen_inputs(spec):
    """Validate all Attempt116 provenance and the full artifact without oracle access."""
    record = spec['upstream']['attempt116']
    if record != UPSTREAM:
        raise ValueError('Attempt116 frozen upstream record mismatch')
    prior.require_hash(PRIOR_SOURCE, PRIOR_SOURCE_SHA256)
    construction_spec = prior.oracle_validator.load_json_object(path_of(record['spec']['path']), 'Attempt116 spec')
    manifest = prior.oracle_validator.load_json_object(path_of(record['manifest']['path']), 'Attempt116 manifest')
    prior.validate_construction_record(record, manifest, construction_spec)
    if (construction_spec.get('solver', {}).get('iterations') != 80 or
            construction_spec.get('damping', {}).get('damping_fraction') != 0.001 or
            construction_spec.get('damping', {}).get('sweep') is not False or
            construction_spec.get('ggn', {}).get('operator') != spec['upstream']['ggn_operator'] or
            manifest.get('b_blind', {}).get('ordered_matrix_hash_sha256') !=
                spec['upstream']['frozen_rhs_aggregate_sha256'] or
            manifest.get('artifact', {}).get('tensor_inventory') != spec['upstream']['artifact_inventory']):
        raise ValueError('Attempt116 frozen numerical construction mismatch')
    return prior.load_frozen_responses(record, manifest)


def evaluate(spec_path=None):
    spec_path = path_of(spec_path or f'experiments/attempts/{ATTEMPT}/spec.json')
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    if spec != FROZEN_SPEC:
        raise ValueError('Attempt119 spec differs from frozen source')
    output = path_of(spec['output_path'])
    prior.require_output_absent(output)
    spec_sha = prior.oracle_validator.sha256_file(spec_path)
    source_sha = prior.oracle_validator.sha256_file(Path(__file__))
    responses = load_frozen_inputs(spec)
    raw, inverse = responses['response_raw_rhs'], responses['response_inverse']
    bounded, construction = freeze_blind_construction(raw, inverse)
    # The historical oracle may be opened only after the frozen hash/diagnostic barrier.
    if spec['oracle'] != ORACLE:
        raise ValueError('Frozen oracle provenance specification mismatch')
    difference, _oracle_manifest = prior.validate_oracle(spec['oracle']['provenance_inputs'])
    evaluation = compare_to_oracle(raw, inverse, bounded, difference)
    category = evaluation['interpretation']
    interpretation = spec['interpretation']
    result = {'format_version': 1, 'attempt_id': ATTEMPT,
              'provenance': {'spec_sha256': spec_sha, 'source_sha256': source_sha,
                             'attempt116': copy.deepcopy(UPSTREAM),
                             'attempt117_validator_source_sha256': PRIOR_SOURCE_SHA256,
                             'oracle_artifact_sha256': spec['oracle']['serialized_sha256'],
                             'oracle_difference_raw_sha256': spec['oracle']['difference_raw_sha256'],
                             'oracle_manifest_sha256': spec['oracle']['provenance_inputs']['oracle_manifest']['sha256']},
              'bounded_construction': construction,
              'oracle_evaluation': evaluation,
              'mechanistic_interpretation': {'category': category,
                  'scope': interpretation['scope'],
                  'positive_caveat': interpretation['positive_caveat'],
                  'negative_conclusion': interpretation['negative_conclusion'] if category == 'no_meaningful_support' else None,
                  'significance_claim': False},
              'attempt116_solver_approximation_caveat': spec['construction']['approximation_caveat'],
              'information_policy': copy.deepcopy(spec['information_policy']),
              **spec['information_policy']}
    prior.require_hash(spec_path, spec_sha)
    prior.require_hash(Path(__file__), source_sha)
    prior.require_output_absent(output)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=PROJECT/'experiments/attempts'/ATTEMPT/'spec.json')
    args = parser.parse_args()
    evaluate(args.spec)


if __name__ == '__main__':
    main()
