"""Synthetic Attempt119 tests; no large artifact, historical oracle, or model access."""
import contextlib
import copy
import hashlib
import importlib.util
import inspect
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/diagnose_bounded_ggn_filter.py'
SPEC = PROJECT/'experiments/attempts/119_postoracle_bounded_ggn_filter_diagnostic/spec.json'
_loader = importlib.util.spec_from_file_location('attempt119_synthetic_tests', SOURCE)
m = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(m)


def response(value=1.0):
    tensor = torch.zeros((128, 2048), dtype=torch.float32)
    tensor[:, 0] = value
    return tensor


class Attempt119Tests(unittest.TestCase):
    def test_frozen_spec_exact_and_upstream_linkage(self):
        spec = json.loads(SPEC.read_text())
        self.assertEqual(spec, m.FROZEN_SPEC)
        self.assertEqual(spec['upstream']['attempt116'], m.prior.CONSTRUCTIONS['116'])
        self.assertEqual(spec['oracle'], m.prior.FROZEN_SPEC['oracle'])
        self.assertEqual(m.prior.oracle_validator.sha256_file(m.PRIOR_SOURCE), m.PRIOR_SOURCE_SHA256)
        record = spec['upstream']['attempt116']
        for key in ('spec', 'source', 'manifest'):
            self.assertEqual(m.prior.oracle_validator.sha256_file(m.path_of(record[key]['path'])),
                             record[key]['sha256'])

    def test_exact_frozen_lambda_rho_fraction_iterations_and_residual(self):
        construction = m.FROZEN_SPEC['construction']
        record = m.UPSTREAM
        self.assertEqual(m.LAMBDA, 0.5615508153958936)
        self.assertEqual(construction['lambda'], m.LAMBDA)
        self.assertEqual(record['rho_b'], 561.5508153958936)
        self.assertEqual(record['damping_fraction'], .001)
        self.assertEqual(record['lambda'], m.LAMBDA)
        self.assertEqual(record['operator_applications'], 80)
        self.assertEqual(record['final_relative_residual'], m.RESIDUAL_116)
        self.assertEqual(m.RESIDUAL_116, 0.030534033703066656)

    def test_exact_response_hashes_and_artifact_inventory(self):
        record = m.UPSTREAM
        self.assertEqual(record['artifact']['sha256'],
                         'c2ed1053b2f57416a748f7104f7260481ce10891750492a4a4aa32885dc216ee')
        self.assertEqual(record['response_raw_rhs_sha256'],
                         '8ddc1801ee82504e10898e97d12b53fea1442ed029314dc47d7d0f68523c2d0a')
        self.assertEqual(record['response_inverse_sha256'],
                         'f3ed438910155c3bc52b57878d1193ce719d3693dac011cefe41228454110d1d')
        self.assertEqual(m.FROZEN_SPEC['upstream']['artifact_inventory'],
                         ['x_80', 'response_raw_rhs', 'response_inverse'])

    def test_bounded_arithmetic_uses_float64_then_one_fp32_cast(self):
        raw, inverse = response(1), response(1)
        raw_before, inverse_before = raw.clone(), inverse.clone()
        bounded = m.bounded_response(raw, inverse)
        expected = (raw.double() - m.LAMBDA*inverse.double()).float().contiguous()
        self.assertTrue(torch.equal(bounded, expected))
        self.assertNotEqual(bounded[0, 0].item(), (raw-m.LAMBDA*inverse)[0, 0].item())
        self.assertEqual(bounded.dtype, torch.float32)
        self.assertTrue(bounded.is_contiguous())
        self.assertEqual(bounded.device.type, 'cpu')
        self.assertTrue(torch.equal(raw, raw_before))
        self.assertTrue(torch.equal(inverse, inverse_before))

    def test_no_alternate_damping_and_no_candidate_normalization(self):
        with self.assertRaisesRegex(ValueError, 'lambda differs'):
            m.bounded_response(response(), response(), .1)
        bounded = m.bounded_response(response(1), response(.5))
        expected = (torch.tensor(1., dtype=torch.float64)
                    - m.LAMBDA*torch.tensor(.5, dtype=torch.float64)).float()
        self.assertEqual(bounded[1, 0].item(), expected.item())
        self.assertNotEqual(float(torch.linalg.vector_norm(bounded[0])), 1.0)
        self.assertFalse(m.FROZEN_SPEC['construction']['renormalize_bounded'])

    def test_malformed_response_shape_dtype_contiguity_nonfinite_rejected(self):
        valid = response()
        invalids = (valid.double(), valid[:127], valid.T,
                    valid.clone().fill_(float('nan')))
        for invalid in invalids:
            with self.assertRaisesRegex(ValueError, 'Response must'):
                m.bounded_response(invalid, valid)
            with self.assertRaisesRegex(ValueError, 'Response must'):
                m.bounded_response(valid, invalid)

    def test_raw_float32_tensor_hash(self):
        raw = response(3)
        expected = hashlib.sha256(raw.numpy().astype('<f4', copy=False).tobytes()).hexdigest()
        self.assertEqual(m.prior.raw_float32_sha256(raw), expected)
        raw[0, 0] = 4
        self.assertNotEqual(m.prior.raw_float32_sha256(raw), expected)

    def test_oracle_free_norm_ratios_and_zero_denominator_null(self):
        raw, inverse = response(2), response(1)
        bounded = m.bounded_response(raw, inverse)
        report = m.norm_ratios(bounded, raw)
        expected = abs(2-m.LAMBDA)/2
        self.assertAlmostEqual(report['bounded_to_raw_total_norm_ratio'], expected, places=7)
        self.assertEqual(len(report['bounded_to_raw_per_position_norm_ratios']), 128)
        self.assertAlmostEqual(report['bounded_to_raw_per_position_norm_ratios'][4], expected, places=7)
        raw[127].zero_()
        report = m.norm_ratios(bounded, raw)
        self.assertIsNone(report['bounded_to_raw_per_position_norm_ratios'][127])
        self.assertIsNone(m.norm_ratios(bounded, torch.zeros_like(raw))['bounded_to_raw_total_norm_ratio'])

    def test_freeze_hash_and_diagnostics_before_oracle(self):
        raw, inverse = response(1), response(.5)
        hashes = [m.UPSTREAM['response_raw_rhs_sha256'],
                  m.UPSTREAM['response_inverse_sha256'],
                  m.prior.raw_float32_sha256(m.bounded_response(raw, inverse))]
        with patch.object(m.prior, 'raw_float32_sha256', side_effect=hashes), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            bounded, frozen = m.freeze_blind_construction(raw, inverse)
        diagnostics = frozen['oracle_free_diagnostics']
        self.assertEqual(diagnostics['bounded_response_raw_sha256'], hashes[2])
        self.assertEqual(diagnostics['raw_response_raw_sha256'], hashes[0])
        self.assertEqual(diagnostics['inverse_response_raw_sha256'], hashes[1])
        self.assertEqual(len(diagnostics['bounded_to_raw_per_position_norm_ratios']), 128)
        self.assertEqual(len(diagnostics['bounded_vs_raw_signed_cosine']['all_128_position_cosines']), 128)
        self.assertEqual(len(diagnostics['bounded_vs_inverse_signed_cosine']['all_128_position_cosines']), 128)
        self.assertTrue(frozen['bounded_hash_frozen_before_oracle_access'])
        self.assertIn('BOUNDED_CONSTRUCTION_FROZEN', output.getvalue())
        self.assertIn('bounded_raw_sha256='+hashes[2], output.getvalue())
        self.assertTrue(torch.equal(bounded, m.bounded_response(raw, inverse)))

    def test_frozen_raw_hash_mismatch_rejected_before_arithmetic(self):
        with self.assertRaisesRegex(ValueError, 'response identity mismatch'):
            m.freeze_blind_construction(response(), response())

    def test_oracle_access_only_after_hash_barrier(self):
        events = []
        raw, inverse = response(1), response(.5)
        bounded = m.bounded_response(raw, inverse)
        def frozen(*args):
            events.append('bounded_hashed_and_diagnostics_frozen')
            return bounded, {'oracle_free_diagnostics': {'bounded_response_raw_sha256': 'a'*64},
                             'bounded_hash_frozen_before_oracle_access': True}
        class StopBeforeOracle(Exception):
            pass
        def oracle(*args):
            events.append('oracle_access')
            raise StopBeforeOracle
        with patch.object(m, 'load_frozen_inputs', return_value={'response_raw_rhs': raw,
                                                                  'response_inverse': inverse}), \
             patch.object(m, 'freeze_blind_construction', side_effect=frozen), \
             patch.object(m.prior, 'require_output_absent'), \
             patch.object(m.prior, 'validate_oracle', side_effect=oracle):
            with self.assertRaises(StopBeforeOracle):
                m.evaluate(SPEC)
        self.assertEqual(events, ['bounded_hashed_and_diagnostics_frozen', 'oracle_access'])

    def test_pinned_upstream_hash_rejection_and_validator_use(self):
        spec = copy.deepcopy(m.FROZEN_SPEC)
        spec['upstream']['attempt116']['artifact']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'upstream record mismatch'):
            m.load_frozen_inputs(spec)
        with patch.object(m.prior, 'require_hash', side_effect=ValueError('SHA256 mismatch')):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.load_frozen_inputs(m.FROZEN_SPEC)
        source = inspect.getsource(m.load_frozen_inputs)
        self.assertIn('prior.validate_construction_record(', source)
        self.assertIn('prior.load_frozen_responses(', source)

    def test_signed_cosine_position_aggregations_and_null(self):
        target = response(1)
        raw = response(1)
        inverse = response(-1)
        bounded = response(1)
        bounded[0, 0] = -1
        bounded[2, 0] = -1
        reports = m.compare_to_oracle(raw, inverse, bounded, target)
        b = reports['candidate_reports']['bounded_filter']
        self.assertEqual(list(reports['candidate_reports']), ['raw_rhs', 'inverse_1e-3', 'bounded_filter'])
        self.assertEqual(b['position_0_cosine'], -1)
        self.assertEqual(b['positions_1_4_individual_cosines'], [1, -1, 1, 1])
        self.assertEqual(b['positions_1_4_mean_cosine'], .5)
        self.assertEqual(b['positions_1_127_mean_cosine'], 125/127)
        self.assertEqual(reports['gain_primary_bounded_minus_raw'], -.5)
        self.assertAlmostEqual(reports['gain_secondary_bounded_minus_raw'], -2/127)
        self.assertEqual(reports['interpretation'], 'no_meaningful_support')
        self.assertEqual(reports['candidate_reports']['inverse_1e-3']['positions_1_4_mean_cosine'], -1)
        bounded[127].zero_()
        self.assertIsNone(m.compare_to_oracle(raw, inverse, bounded, target)
                          ['candidate_reports']['bounded_filter']['positions_1_127_mean_cosine'])

    def test_category_boundaries_and_order(self):
        self.assertEqual(m.interpret(.03, .02), 'clear_support')
        self.assertEqual(m.interpret(.03, .019999), 'moderate_support')
        self.assertEqual(m.interpret(.015, 0.), 'moderate_support')
        self.assertEqual(m.interpret(.014999, .5), 'no_meaningful_support')
        self.assertEqual(m.interpret(.03, -.0001), 'no_meaningful_support')
        self.assertEqual(m.interpret(None, .1), 'undefined_required_position_cosine')

    def test_no_sweep_sign_selection_normalization_or_rescaling(self):
        construction = m.FROZEN_SPEC['construction']
        self.assertEqual(construction['only_candidate'], 'bounded_filter')
        for key in ('renormalize_bounded', 'candidate_rescaling', 'sign_selection',
                    'alternate_lambda', 'alternate_iterate', 'scalar_sweep',
                    'interpolation_sweep', 'exact_spectral_identity_claimed_for_y_80'):
            self.assertFalse(construction[key])
        self.assertEqual(m.FROZEN_SPEC['metrics']['main_comparison'], 'bounded_filter_minus_raw_rhs')
        self.assertEqual(m.FROZEN_SPEC['metrics']['inverse_role'], 'descriptive_context_only')
        body = inspect.getsource(m.bounded_response)
        self.assertIn('raw.to(torch.float64) - damping * inverse.to(torch.float64)', body)
        self.assertNotIn('for ', body)
        for forbidden in ('AutoModelForCausalLM', 'from_pretrained(', 'torch.func.jvp(',
                          'autograd.grad(', 'functional_call(', 'models/base'):
            self.assertNotIn(forbidden, SOURCE.read_text())

    def test_residual_approximation_caveat_and_narrow_interpretation(self):
        definition = m.FROZEN_SPEC['construction']
        self.assertEqual(definition['attempt116_final_relative_linear_residual'],
                         .030534033703066656)
        self.assertIn('approximate_80_step_solve', definition['approximation_caveat'])
        self.assertFalse(definition['exact_spectral_identity_claimed_for_y_80'])
        self.assertIn('if_x=', definition['exact_solve_reference_only'])
        self.assertIn('Does_not_show_that_G_identifies_historical_signal_generally',
                      m.FROZEN_SPEC['interpretation']['positive_caveat'])
        self.assertIn('attempt116_solver_approximation_caveat', SOURCE.read_text())

    def test_result_overwrite_refusal_and_information_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'result.json'
            path.write_text('existing')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.prior.require_output_absent(path)
            self.assertEqual(path.read_text(), 'existing')
        policy = m.FROZEN_SPEC['information_policy']
        self.assertTrue(policy['postoracle_mechanistic_diagnostic'])
        self.assertTrue(policy['candidate_hash_frozen_before_oracle_access_within_run'])
        self.assertFalse(policy['clean_blind_validation'])
        for key in ('model_loading', 'new_gradient', 'new_jvp', 'ggn_vector_product',
                    'cg_solve', 'alternate_iterate', 'alternate_damping',
                    'scalar_sweep', 'interpolation_sweep', 'candidate_rescaling',
                    'sign_selection', 'normalization_of_bounded_candidate',
                    'oracle_driven_candidate_selection'):
            self.assertFalse(policy[key])


if __name__ == '__main__':
    unittest.main()
