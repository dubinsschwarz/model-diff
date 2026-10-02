"""Synthetic and frozen-provenance tests for Attempt112; no real model runs."""
import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/diagnose_privileged_very_low_damping_inverse_ggn.py'
loader = importlib.util.spec_from_file_location('attempt112_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


def checkpoint_fixture():
    inventory = [{'name': 'linear.weight', 'shape': [1, 3]}]
    x = {'linear.weight': torch.tensor([[1., 2., 3.]], dtype=torch.float32)}
    r = {'linear.weight': torch.tensor([[2., 3., 4.]], dtype=torch.float32)}
    p = {'linear.weight': torch.tensor([[3., 4., 5.]], dtype=torch.float32)}
    rows = [{'iteration': index, 'relative_residual': 0.5,
             'r_norm': 1., 'pAp': 2., 'alpha': 0.1, 'beta': 0.2,
             'operator_seconds': 1., 'cumulative_elapsed_seconds': float(index),
             'operator_eta_seconds': float(100-index)} for index in range(1, 51)]
    rr = m.a106.matrixwise_dot(r, r)
    state = {'iteration': 50, 'x': x, 'r': r, 'p': p, 'rr': rr,
             'b_norm': 5., 'iterations': rows}
    return state, inventory


def toy_solver(*, start_state=None, calls=None, callback=None, report=None,
               stop_after_checkpoint=False):
    n = 150
    diagonal = torch.logspace(-2, 3, n)
    layer = torch.nn.Linear(n, 1, bias=False)
    eligible = [('linear', layer)]
    calls = [] if calls is None else calls
    if start_state is None:
        rhs = torch.ones((1, n), dtype=torch.float32)
        rr = m.a106.matrixwise_dot({'linear.weight': rhs},
                                    {'linear.weight': rhs})
        state = {'iteration': 0,
                 'x': {'linear.weight': torch.zeros_like(rhs)},
                 'r': {'linear.weight': rhs.clone()},
                 'p': {'linear.weight': rhs.clone()},
                 'rr': rr, 'b_norm': math.sqrt(rr), 'iterations': []}
    else:
        state = start_state

    def apply(vector, iteration):
        calls.append(iteration)
        layer.weight.grad = vector['linear.weight']*diagonal

    def checkpoint(current):
        if callback is not None:
            callback(current)
        if stop_after_checkpoint:
            raise RuntimeError('intentional failure after 50')

    return m.run_fixed_100_step_cg(
        state, eligible, apply, SPEC,
        on_checkpoint=None if state['iteration'] == 50 else checkpoint,
        report=report), calls, eligible


class Attempt112Tests(unittest.TestCase):
    def test_01_attempt111_commit_and_file_hashes(self):
        link = SPEC['attempt111']
        self.assertEqual(link['commit_sha'], m.COMMIT111)
        self.assertEqual(link['spec_sha256'], m.a111.SPEC_SHA256)
        self.assertEqual(link['source_sha256'], m.SOURCE111_SHA256)
        self.assertEqual(link['result_sha256'], m.RESULT111_SHA256)
        m.verify_committed_attempt111(link)

    def test_02_exact_frozen_attempt111_values(self):
        link = SPEC['attempt111']
        result = json.loads((PROJECT/link['result_path']).read_text())
        self.assertEqual(hashlib.sha256((PROJECT/link['result_path']).read_bytes()).hexdigest(),
                         link['result_sha256'])
        activation = result['primary_x80']['activation']
        self.assertEqual(activation['positions_1_4_mean_cosine'], link['required_primary'])
        self.assertEqual(activation['positions_1_127_mean_cosine'], link['required_secondary'])
        self.assertEqual(result['solver']['relative_linear_residual'],
                         link['required_relative_linear_residual'])
        self.assertEqual(result['solver']['rhs_norm'], link['required_rhs_norm'])
        self.assertEqual(result['solver']['lambda'], link['required_lambda'])
        self.assertEqual(len(result['solver']['iterations']), 80)
        self.assertEqual(result['interpretation']['category'], link['required_category'])
        self.assertTrue(result['true_delta_rhs'])
        self.assertFalse(result['empirical_fisher'])

    def test_03_attempt111_linkage_rejects_modified_commit(self):
        altered = copy.deepcopy(SPEC['attempt111'])
        altered['commit_sha'] = '0'*40
        with self.assertRaisesRegex(ValueError, 'committed revision mismatch'):
            m.verify_committed_attempt111(altered)

    def test_04_frozen_spec_rejects_changed_scientific_field(self):
        altered = copy.deepcopy(SPEC)
        altered['solver']['iterations'] = 99
        with patch.object(m.parent, 'require_hash'), \
             patch.object(m.a, 'load_json_object', return_value=altered), \
             patch.object(m, 'expected_spec', return_value=SPEC):
            with self.assertRaisesRegex(ValueError, 'frozen scientific inventory mismatch'):
                m.load_spec()

    def test_05_exact_rhs_semantics_unchanged(self):
        old = m.a111.load_spec()
        for key in ('exact_rhs', 'displacement', 'eligible_tensors', 'pilot'):
            self.assertEqual(SPEC[key], old[key])
        self.assertEqual(m.a111.rhs_semantics_hash(SPEC),
                         m.a111.rhs_semantics_hash(old))
        self.assertEqual(SPEC['exact_rhs']['definition'], 'b_exact=G1_Delta_E')

    def test_06_delta_sign_and_arithmetic_unchanged(self):
        old = m.a111.load_spec()['displacement']
        self.assertEqual(SPEC['displacement'], old)
        self.assertIn('final', str(old['definition']).lower())
        self.assertEqual(SPEC['eligible_tensors']['expected_matrix_count'], 98)

    def test_07_stable_G_operator_unchanged(self):
        old = m.a111.load_spec()
        self.assertEqual(SPEC['ggn'], old['ggn'])
        self.assertEqual(SPEC['ggn_numerics_106'], old['ggn_numerics_106'])
        self.assertIs(m.a106.ggn_action_stage, m.a111.a106.ggn_action_stage)

    def test_08_fineweb_first64_and_8128_contexts(self):
        self.assertEqual(SPEC['pilot']['construction_rows'], [0, 64])
        self.assertEqual(SPEC['pilot']['prediction_contexts'], 8128)
        self.assertEqual(SPEC['pilot']['ggn_batch_size'], 1)
        self.assertEqual(SPEC['pilot']['ggn_batches'], 64)

    def test_09_fixed_lambda_literal_and_ratios(self):
        self.assertEqual(SPEC['solver']['fixed_lambda_precommitted_decimal'],
                         '0.013361159773737232')
        self.assertIn('"fixed_lambda": 0.013361159773737232',
                      m.SPEC_PATH.read_text())
        self.assertEqual(m.FIXED_LAMBDA, SPEC['solver']['fixed_lambda'])
        self.assertEqual(
            Decimal(SPEC['attempt111']['required_lambda_precommitted_decimal'])
            * Decimal('0.1'),
            Decimal(SPEC['solver']['fixed_lambda_precommitted_decimal']))
        self.assertEqual(SPEC['solver']['lambda_relative_to_attempt111'], 0.1)
        self.assertEqual(SPEC['solver']['lambda_relative_to_attempt110'], 0.01)
        self.assertEqual(SPEC['solver']['lambda_relative_to_attempt108'], 0.001)

    def test_10_lambda_never_recomputed(self):
        self.assertFalse(SPEC['solver']['recompute_rho_from_exact_rhs'])
        self.assertEqual(SPEC['solver']['fixed_lambda_source'],
                         'one_tenth_frozen_Attempt111_precommitted_decimal_lambda')
        self.assertNotIn('rho_b', SOURCE.read_text())

    def test_11_exactly_100_iterations_and_x100_primary(self):
        self.assertEqual(SPEC['solver']['iterations'], 100)
        self.assertEqual(SPEC['solver']['candidate'], 'x_100')
        self.assertEqual(SPEC['information_policy']['inverse_method'],
                         'fixed_100_step_fixed_lambda_cg')
        self.assertNotIn('midpoint_candidate', SPEC['solver'])

    def test_12_no_early_stop_or_preconditioner(self):
        self.assertFalse(SPEC['solver']['residual_early_stop'])
        self.assertIsNone(SPEC['solver']['preconditioner'])
        self.assertFalse(SPEC['solver']['candidate_selection'])
        self.assertFalse(SPEC['information_policy']['oracle_based_tuning'])

    def test_13_toy_CG_recurrence(self):
        (state, solver), calls, eligible = toy_solver()
        self.assertEqual(state['iteration'], 100)
        self.assertEqual(calls, list(range(1, 101)))
        self.assertEqual(solver['operator_applications'], 100)
        self.assertEqual(solver['residual_source'],
                         'CG_recurrence_no_extra_operator_application')
        self.assertTrue(all(row['pAp'] > 0 and row['alpha'] > 0 and row['beta'] >= 0
                            for row in solver['iterations']))

    def test_14_matrixwise_FP64_reductions(self):
        x = {'a': torch.tensor([1e8, 1., -1e8], dtype=torch.float32)}
        y = {'a': torch.ones(3, dtype=torch.float32)}
        self.assertEqual(m.a106.matrixwise_dot(x, y), 1.)
        self.assertIn('FP64', SPEC['solver']['global_reductions'])

    def test_15_stable_Fisher_matches_explicit_categorical_action(self):
        logits = torch.tensor([[[0.2, -0.3, 1.1]]], dtype=torch.float32)
        tangent = torch.tensor([[[2., -1., 0.5]]], dtype=torch.float32)
        p = torch.softmax(logits.double(), dim=-1)
        expected = p*(tangent.double()-(p*tangent.double()).sum(-1, keepdim=True))
        self.assertLess(float(expected.sum().abs()), 1e-15)
        self.assertEqual(SPEC['ggn_numerics_106']['implementation'],
                         'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast')

    def test_16_matched_1024_probe_inventory(self):
        self.assertEqual(SPEC['probe_rows'], [0, 1024])
        self.assertEqual(SPEC['probe']['batch_size'], 32)
        self.assertEqual(SPEC['readout']['hidden_state_index'], 14)

    def test_17_matched_target_hash(self):
        self.assertEqual(SPEC['target']['raw_float32_sha256'],
                         '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f')
        self.assertEqual(SPEC['target'], m.a111.load_spec()['target'])

    def test_18_strict_block13_JVP(self):
        self.assertEqual(SPEC['jvp'], m.a111.load_spec()['jvp'])
        self.assertTrue(SPEC['jvp']['strict_forward_ad'])
        self.assertFalse(SPEC['jvp']['fallback'])

    def test_19_parameter_diagnostics_are_descriptive(self):
        self.assertIs(m.a106.parameter_diagnostics,
                      m.a111.a106.parameter_diagnostics)
        self.assertFalse(SPEC['information_policy']['candidate_selection'])

    def test_20_interpretation_order(self):
        baseline = SPEC['baselines']['attempt111_primary']
        self.assertEqual(m.interpretation(0.97, 0.02, SPEC),
                         'very_near_ceiling_recovery')
        self.assertEqual(m.interpretation(baseline+0.008, 0.4, SPEC),
                         'clear_additional_low_curvature_recovery')
        self.assertEqual(m.interpretation(baseline+0.006, 0.021, SPEC),
                         'still_solver_limited')
        self.assertEqual(m.interpretation(baseline+0.004, 0.01, SPEC),
                         'very_low_damping_plateau')
        self.assertEqual(m.interpretation(baseline+0.006, 0.015, SPEC),
                         'mixed_very_low_damping_result')

    def test_21_gap_accounting_uses_exact_attempt111_baseline(self):
        baseline = SPEC['baselines']['attempt111_primary']
        self.assertEqual(baseline, 0.9582115833559394)
        gap = SPEC['baselines']['attempt016_primary_context_approx']-baseline
        self.assertGreater(gap, 0)
        self.assertAlmostEqual((0.97-baseline)/gap,
                               (0.97-0.9582115833559394)/(0.9836-0.9582115833559394))

    def test_22_no_oracle_adl_or_adapter(self):
        source = SOURCE.read_text()
        self.assertNotIn('oracle_adl.pt', source)
        self.assertNotIn('adapter_checkpoint', source)
        self.assertFalse(SPEC['information_policy']['oracle_adl_access'])
        self.assertFalse(SPEC['information_policy']['adapter_access'])

    def test_23_checkpoint_outside_repository(self):
        path = Path(SPEC['paths']['checkpoint_path'])
        self.assertFalse(path.is_relative_to(PROJECT))
        self.assertEqual(path.name, 'attempt112_very_low_damping_cg100_iter50.pt')
        self.assertEqual(SPEC['checkpoint']['expected_approximate_size_gib'], 7.9)

    def test_24_checkpoint_has_only_resume_state(self):
        state, inventory = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, inventory, SPEC)
        self.assertEqual(set(payload), {'format_version', 'iteration', 'x', 'r', 'p',
                                        'rr', 'b_norm', 'fixed_lambda',
                                        'coordinate_inventory', 'iteration_records',
                                        'raw_hashes', 'provenance'})
        self.assertNotIn('activation', str(payload.keys()).lower())
        self.assertNotIn('model_weights', payload)

    def test_25_atomic_checkpoint_publication(self):
        state, inventory = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, inventory, SPEC)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            m.atomic_checkpoint_publish(path, payload)
            self.assertTrue(path.is_file())
            self.assertEqual(sorted(Path(directory).iterdir()), [path])

    def test_26_checkpoint_x_r_p_hashes(self):
        state, inventory = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, inventory, SPEC)
        self.assertEqual(payload['raw_hashes'], m.a111.state_hashes(state, inventory))
        m.validate_checkpoint(payload, SPEC, inventory, 5.)
        payload['x']['linear.weight'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'raw tensor hash mismatch'):
            m.validate_checkpoint(payload, SPEC, inventory, 5.)

    def test_27_incompatible_resume_rejected(self):
        state, inventory = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, inventory, SPEC)
        payload['provenance']['spec_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'metadata/provenance'):
            m.validate_checkpoint(payload, SPEC, inventory, 5.)

    def test_28_resume_restores_exact_state(self):
        state, inventory = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, inventory, SPEC)
        layer = torch.nn.Linear(3, 1, bias=False)
        restored = m.restore_cg_state(payload, [('linear', layer)], inventory)
        self.assertEqual(restored['iteration'], 50)
        self.assertEqual(m.a111.state_hashes(restored, inventory), payload['raw_hashes'])

    def test_29_resume_starts_at_51(self):
        midpoint = []
        (original, _), _, _ = toy_solver(callback=lambda state: midpoint.append(
            copy.deepcopy(state)))
        self.assertEqual(len(midpoint), 1)
        resumed_state = midpoint[0]
        calls = []
        (continued, _), _, _ = toy_solver(start_state=resumed_state, calls=calls)
        self.assertEqual(calls[0], 51)
        self.assertEqual(calls[-1], 100)
        self.assertEqual(continued['iteration'], 100)
        self.assertTrue(torch.equal(continued['x']['linear.weight'],
                                    original['x']['linear.weight']))

    def test_30_normal_run_continues_in_memory(self):
        addresses = []
        def record_state(state):
            addresses.append(id(state['x']['linear.weight']))
        (state, _), _, _ = toy_solver(callback=record_state)
        self.assertEqual(addresses, [id(state['x']['linear.weight'])])

    def test_31_success_deletes_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'resume.pt'
            checkpoint.write_bytes(b'synthetic')
            output = Path(directory)/'result.json'
            fake = {'solver': {'operator_applications': 100,
                               'iterations': [{} for _ in range(100)]},
                    'primary_x100': {'candidate': 'x_100_primary'}}
            m.publish_result_and_cleanup(output, checkpoint, fake)
            self.assertTrue(output.is_file())
            self.assertFalse(checkpoint.exists())

    def test_32_failure_leaves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'resume.pt'
            checkpoint.write_bytes(b'synthetic')
            output = Path(directory)/'result.json'
            with patch.object(m.a, 'write_manifest', side_effect=RuntimeError('disk error')):
                with self.assertRaisesRegex(RuntimeError, 'disk error'):
                    m.publish_result_and_cleanup(
                        output, checkpoint,
                        {'solver': {'operator_applications': 100,
                                    'iterations': [{} for _ in range(100)]},
                         'primary_x100': {'candidate': 'x_100_primary'}})
            self.assertTrue(checkpoint.is_file())
            self.assertFalse(output.exists())

    def test_33_no_result_before_iteration100(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            checkpoint = Path(directory)/'resume.pt'
            checkpoint.write_bytes(b'synthetic')
            with self.assertRaisesRegex(ValueError, 'before x_100'):
                m.publish_result_and_cleanup(output, checkpoint,
                                             {'solver': {'operator_applications': 50,
                                                         'iterations': [{}]*50}})
            self.assertFalse(output.exists())

    def test_34_no_result_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            output.write_text('keep')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.publish_result_and_cleanup(output, Path(directory)/'resume.pt', {})
            self.assertEqual(output.read_text(), 'keep')

    def test_35_no_persistent_large_scientific_artifact(self):
        self.assertTrue(SPEC['output_policy']['result_json_only'])
        self.assertTrue(SPEC['checkpoint']['delete_after_successful_result_publication'])
        self.assertFalse(SPEC['checkpoint']['activation_metrics_in_checkpoint'])
        self.assertEqual(set(SPEC['paths']) & {'candidate_path', 'gradient_path'}, set())

    def test_36_progress_and_ETA(self):
        rows = []
        (state, solver), _, _ = toy_solver(report=rows.append)
        self.assertEqual(rows, solver['iterations'])
        self.assertEqual(len(rows), 100)
        self.assertTrue(all(row['operator_seconds'] >= 0 and
                            row['operator_eta_seconds'] >= 0 for row in rows))
        self.assertEqual(set(solver['residual_checkpoints']),
                         {'20', '40', '50', '60', '80', '100'})
        self.assertEqual(SPEC['workload']['cg_ggn_applications'], 100)
        self.assertEqual(SPEC['workload']['total_jvp_batches'], 32)

    def test_37_preflight_accepts_exact_frozen_attempt111_result(self):
        original_resolve = m.parent.resolve
        with tempfile.TemporaryDirectory() as directory:
            def resolve(value):
                if value == SPEC['paths']['result_path']:
                    return Path(directory)/'result.json'
                if value == SPEC['paths']['checkpoint_path']:
                    return Path(directory)/'checkpoint.pt'
                return original_resolve(value)
            with patch.object(m.parent, 'resolve', side_effect=resolve), \
                 patch.object(m.a110, 'validate_inputs', return_value=(None, {}, None)):
                _, _, result = m.validate_inputs(SPEC, resume=False)
        self.assertEqual(result['primary_x80']['activation']
                         ['positions_1_4_mean_cosine'],
                         SPEC['attempt111']['required_primary'])

    def test_38_preflight_rejects_changed_frozen_result_value(self):
        frozen = json.loads((PROJECT/SPEC['attempt111']['result_path']).read_text())
        frozen['solver']['relative_linear_residual'] += 0.01
        original_resolve = m.parent.resolve
        with tempfile.TemporaryDirectory() as directory:
            def resolve(value):
                if value == SPEC['paths']['result_path']:
                    return Path(directory)/'result.json'
                if value == SPEC['paths']['checkpoint_path']:
                    return Path(directory)/'checkpoint.pt'
                return original_resolve(value)
            with patch.object(m.parent, 'resolve', side_effect=resolve), \
                 patch.object(m.a, 'load_json_object', return_value=frozen):
                with self.assertRaisesRegex(ValueError,
                                            'result/provenance/solver mismatch'):
                    m.validate_inputs(SPEC, resume=False)


if __name__ == '__main__':
    unittest.main()
