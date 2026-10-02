"""Synthetic and frozen-linkage tests for Attempt108; no real model execution."""
import copy
import hashlib
import importlib.util
import inspect
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests.test_diagnose_privileged_endpoint_ggn_forward import Toy98Model, selected
from tests.test_diagnose_privileged_inverse_ggn_cg import TinyReadout


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/diagnose_privileged_exact_rhs_inverse_ggn_cg15.py'
loader = importlib.util.spec_from_file_location('attempt108_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


def toy_solve(matrix, rhs, *, report=None):
    layer = torch.nn.Linear(matrix.shape[0], 1, bias=False)
    eligible = [('linear', layer)]
    calls = []

    def apply(vector, iteration):
        calls.append(iteration)
        layer.weight.grad = (matrix @ vector['linear.weight'].flatten()).reshape_as(layer.weight)
        return {'iteration': iteration}

    source = {'linear.weight': rhs.float().reshape_as(layer.weight).contiguous()}
    x, summary = m.fixed_15_step_cg(source, eligible, apply, SPEC, report=report)
    return x['linear.weight'].flatten(), summary, calls, source


class Attempt108Tests(unittest.TestCase):
    def test_attempt107_spec_source_result_linkage(self):
        link = SPEC['attempt107']
        for key in ('spec', 'source', 'result'):
            path = PROJECT/link[key+'_path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             link[key+'_sha256'])
        self.assertEqual(link['spec_sha256'], m.a107.SPEC_SHA256)
        self.assertEqual(link['source_sha256'], m.SOURCE107_SHA256)
        self.assertEqual(link['result_sha256'],
                         '2dbc32387559ca29ed7f0ac237ecf89c89b9177f0a8ef5f028c89969200b67c7')

    def test_attempt107_frozen_result_values(self):
        link = SPEC['attempt107']
        result = m.a.load_json_object(PROJECT/link['result_path'], 'Attempt107 result')
        self.assertEqual(result['matched_activation']['positions_1_4_mean_cosine'],
                         0.8402237062805622)
        self.assertEqual(result['matched_activation']['positions_1_127_mean_cosine'],
                         0.8453507468137178)
        self.assertEqual(result['solver']['lambda'], 13.361159773737231)
        self.assertEqual(result['solver']['operator_applications'], 10)
        self.assertEqual(result['solver']['relative_linear_residual'],
                         0.06584746429710282)
        self.assertEqual(result['solver']['rhs_norm'], 14.651196958318208)
        self.assertEqual(result['interpretation']['category'],
                         'inverse_or_regularization_limitations_dominate')
        self.assertTrue(result['true_delta_rhs'])
        self.assertEqual(result['ggn_operator'], 'exact_endpoint_categorical_JtFJ')
        self.assertFalse(result['empirical_fisher'])

    def test_attempt107_hash_mismatch_fails_before_model_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            spec['attempt107']['result_sha256'] = '0'*64
            with patch.object(m.a107, 'validate_inputs',
                              side_effect=AssertionError('model preflight reached')):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    m.validate_inputs(spec)

    def test_frozen_prior_preflight_reused_with_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            with (patch.object(m.parent, 'require_hash'),
                  patch.object(m.a107, 'validate_inputs', return_value=('016', {'base_files': []}))
                  as delegated):
                old, inventories = m.validate_inputs(spec)
            self.assertEqual((old, inventories), ('016', {'base_files': []}))
            self.assertEqual(delegated.call_args.args[0]['paths']['result_path'],
                             spec['paths']['result_path'])

    def test_only_scientific_spec_change_is_iteration_budget(self):
        prior = m.a107.load_spec()
        for key in ('attempt104', 'attempt016', 'attempt106', 'base',
                    'final_checkpoint_files', 'model', 'readout', 'eligible_tensors',
                    'corpus', 'pilot', 'probe', 'probe_rows', 'displacement', 'ggn',
                    'ggn_numerics_106', 'jvp', 'metric', 'target', 'exact_rhs'):
            self.assertEqual(SPEC[key], prior[key], key)
        self.assertEqual(SPEC['solver']['fixed_lambda'], prior['solver']['fixed_lambda'])
        self.assertEqual(SPEC['solver']['iterations'], 15)
        self.assertEqual(prior['solver']['iterations'], 10)
        self.assertFalse(SPEC['solver']['recompute_rho_from_exact_rhs'])

    def test_exact_rhs_unchanged_and_one_G_application(self):
        model = Toy98Model()
        eligible = selected(model)
        delta = {name+'.weight': torch.ones_like(module.weight)
                 for name, module in eligible}
        tokens = torch.zeros((64, 128), dtype=torch.int64)

        def fake_g(final, batch, chosen, vector, spec, *, progress):
            self.assertIs(vector, delta)
            self.assertEqual(tuple(batch.shape), (64, 128))
            self.assertEqual(len(chosen), 98)
            self.assertIs(spec, SPEC)
            for _, module in chosen:
                module.weight.grad = torch.full_like(module.weight, 3.)
            return {}, {}, {'aggregate': {'prediction_context_count': 8128}}

        with patch.object(m.a106, 'ggn_action_stage', side_effect=fake_g) as operator:
            rhs, diagnostics, _ = m.a107.exact_rhs_stage(
                model, tokens, eligible, delta, SPEC)
        self.assertEqual(operator.call_count, 1)
        self.assertEqual(len(rhs), 98)
        self.assertTrue(all(value.item() == 3 for value in rhs.values()))
        self.assertAlmostEqual(diagnostics['cosine_with_true_delta'], 1)
        self.assertTrue(all(module.weight.grad is None for _, module in eligible))

    def test_same_stable_categorical_Fisher_not_empirical(self):
        self.assertEqual(SPEC['ggn_numerics_106'],
                         m.a107.load_spec()['ggn_numerics_106'])
        self.assertEqual(SPEC['ggn_numerics_106']['implementation'],
                         'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast')
        self.assertFalse(SPEC['ggn']['empirical_fisher'])
        logits = torch.tensor([[[0.2, -0.1, 0.6]]], dtype=torch.float32)
        tangent = torch.tensor([[[0.3, 1.2, -0.7]]], dtype=torch.float32)
        actual, _ = m.a106.stable_categorical_fisher_vector(logits, tangent, SPEC)
        p = torch.softmax(logits.double(), -1)[0, 0]
        expected = (torch.diag(p)-torch.outer(p, p)) @ tangent.double()[0, 0]/8128
        self.assertTrue(torch.equal(actual[0, 0], expected.float()))

    def test_same_64_construction_and_1024_probe_inventory(self):
        corpus = torch.arange(512, dtype=torch.int64)[:, None].repeat(1, 128)
        probe = torch.arange(1024, dtype=torch.int64)[:, None].repeat(1, 128)
        with patch.object(m.parent, 'load_frozen_tokens', return_value=(corpus, probe)):
            construction, matched = m.a106.load_frozen_inputs(SPEC)
        self.assertEqual(tuple(construction.shape), (64, 128))
        self.assertEqual(construction[-1, 0].item(), 63)
        self.assertEqual(tuple(matched.shape), (1024, 128))
        self.assertEqual(matched[-1, 0].item(), 1023)
        self.assertEqual(SPEC['pilot']['prediction_contexts'], 8128)

    def test_fixed_lambda_and_no_preconditioner(self):
        self.assertEqual(SPEC['solver']['fixed_lambda'], 13.361159773737231)
        self.assertEqual(SPEC['solver']['fixed_lambda_source'],
                         'frozen_Attempt106_numeric_value')
        self.assertIsNone(SPEC['solver']['preconditioner'])
        self.assertFalse(SPEC['solver']['recompute_rho_from_exact_rhs'])
        self.assertNotIn('rho_b', inspect.getsource(m.fixed_15_step_cg))
        bad = copy.deepcopy(SPEC)
        bad['solver']['preconditioner'] = 'diagonal'
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            m.fixed_15_step_cg({}, [], lambda *_: None, bad)

    def test_exactly_15_steps_even_without_residual_stop(self):
        matrix = torch.diag(torch.linspace(1., 20., 20))
        rhs = torch.ones(20)
        x, summary, calls, source = toy_solve(matrix, rhs)
        self.assertEqual(calls, list(range(1, 16)))
        self.assertEqual(source, {})
        self.assertEqual(summary['operator_applications'], 15)
        self.assertEqual(len(summary['iterations']), 15)
        self.assertTrue(summary['no_early_stop'])
        self.assertEqual(summary['method'], 'fixed_15_step_fixed_lambda_cg')
        self.assertEqual(summary['candidate'], 'x_15')
        expected = torch.linalg.solve(matrix+13.361159773737231*torch.eye(20), rhs)
        self.assertLess(float(torch.linalg.vector_norm(x-expected)/
                              torch.linalg.vector_norm(expected)), 1e-4)

    def test_fixed_lambda_CG_first_recurrence_step(self):
        matrix = torch.diag(torch.linspace(1., 20., 20))
        rhs = torch.arange(1., 21.)
        x, summary, calls, _ = toy_solve(matrix, rhs)
        damping = 13.361159773737231
        alpha = float(rhs@rhs)/float(rhs@((matrix+damping*torch.eye(20))@rhs))
        self.assertAlmostEqual(summary['iterations'][0]['alpha'], alpha, places=6)
        self.assertEqual(calls[0], 1)
        self.assertTrue(bool(torch.isfinite(x).all()))

    def test_exact_rhs_toy_inverse(self):
        ggn = torch.diag(torch.linspace(0.5, 8., 20))
        delta = torch.linspace(-1., 1., 20)
        rhs = ggn@delta
        x, summary, _, _ = toy_solve(ggn, rhs)
        expected = torch.linalg.solve(ggn+13.361159773737231*torch.eye(20), rhs)
        self.assertTrue(torch.allclose(x, expected, atol=1e-5, rtol=1e-4))
        self.assertEqual(summary['lambda'], 13.361159773737231)

    def test_nonpositive_pAp_and_nonfinite_action_fail_closed(self):
        layer = torch.nn.Linear(2, 1, bias=False)
        rhs = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}

        def negative(vector, _iteration):
            layer.weight.grad = -100*vector['linear.weight']
            return {}

        with self.assertRaisesRegex(ValueError, r'p\^T A p'):
            m.fixed_15_step_cg(rhs, [('linear', layer)], negative, SPEC)
        rhs = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}

        def nonfinite(vector, _iteration):
            layer.weight.grad = torch.full_like(vector['linear.weight'], float('nan'))
            return {}

        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            m.fixed_15_step_cg(rhs, [('linear', layer)], nonfinite, SPEC)

    def test_matrixwise_FP64_reductions(self):
        values = torch.tensor([1e8, 1., -1e8, 1.], dtype=torch.float32)
        self.assertEqual(m.a106.matrixwise_dot({'w': values},
                                               {'w': torch.ones_like(values)}), 2.)
        self.assertIn('matrixwise_dot', inspect.getsource(m.a107.fixed_lambda_cg))
        self.assertNotIn('.double()', inspect.getsource(m.a107.fixed_lambda_cg))

    def test_matched_target_hash_requirement(self):
        base = torch.zeros((128, 2048), dtype=torch.float64)
        final = torch.ones_like(base)
        target = m.parent.matched_difference(final, base)
        spec = copy.deepcopy(SPEC)
        spec['target']['raw_float32_sha256'] = m.a.sha256_raw_float32_tensor(target, torch)
        verified, digest = m.a106.verified_matched_target(final, base, spec)
        self.assertTrue(torch.equal(verified, target))
        self.assertEqual(digest, spec['target']['raw_float32_sha256'])
        with self.assertRaisesRegex(ValueError, 'Matched historical target'):
            m.a106.verified_matched_target(final, base, SPEC)

    def test_strict_block13_JVP_unchanged(self):
        model = TinyReadout()
        spec = copy.deepcopy(SPEC)
        spec['readout']['hidden_size'] = 2
        tokens = torch.tensor([[0, 1]*64], dtype=torch.int64)
        direction = {'model.layers.0.weight': torch.tensor([[0.1, 0.2], [0.3, 0.4]])}
        primal, tangent = m.a.run_readout_jvp(model, tokens, direction, spec, torch)
        ordinary = m.a.ordinary_hook_readout(model, tokens, spec, torch)
        self.assertEqual(m.a.compare_primal(primal, ordinary, spec, torch), 0)
        self.assertTrue(bool(torch.isfinite(tangent).all()))

    def test_positionwise_primary_and_secondary_metrics(self):
        response = torch.ones((128, 2048), dtype=torch.float32)
        target = torch.ones_like(response)
        target[0].neg_()
        target[2, :1024] = -1
        metrics = m.parent.response_metrics(response, target)
        self.assertAlmostEqual(metrics['position_0_cosine'], -1.)
        self.assertAlmostEqual(metrics['positions_1_4_mean_cosine'], 0.75)
        self.assertAlmostEqual(metrics['positions_1_127_mean_cosine'], 126/127)
        self.assertEqual(len(metrics['position_cosines']), 128)
        self.assertIn('best_scalar_rescaling_diagnostic_only', metrics)

    def test_frozen_attempt107_comparison_and_interpretation(self):
        self.assertEqual(SPEC['baselines']['attempt107_primary'], 0.8402237062805622)
        self.assertEqual(SPEC['baselines']['attempt107_secondary'], 0.8453507468137178)
        self.assertEqual(m.interpretation(0.90, SPEC), 'clear_solver_truncation_effect')
        self.assertEqual(m.interpretation(0.87, SPEC), 'modest_solver_truncation_effect')
        self.assertEqual(m.interpretation(0.86, SPEC),
                         'little_evidence_10_step_truncation_dominates')
        self.assertEqual(SPEC['interpretation']['if_little_evidence'],
                         'next_diagnostic_should_vary_damping_not_add_more_CG_iterations')

    def test_oracle_and_adapter_firewall(self):
        source = SOURCE.read_text()
        self.assertNotIn('oracle_adl.pt', source)
        self.assertNotIn('adapter_checkpoint', source)
        self.assertFalse(SPEC['information_policy']['oracle_adl_access'])
        self.assertFalse(SPEC['information_policy']['adapter_access'])
        self.assertTrue(SPEC['information_policy']['true_delta_rhs'])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            output.write_text('frozen')
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(output)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.validate_inputs(spec)
            self.assertEqual(output.read_text(), 'frozen')

    def test_no_large_tensor_persistence(self):
        self.assertTrue(SPEC['output_policy']['result_json_only'])
        self.assertFalse(SPEC['output_policy']['large_tensor_persistence'])
        self.assertNotIn('torch.save(', SOURCE.read_text())
        self.assertFalse(SPEC['information_policy']['candidate_selection'])
        self.assertFalse(SPEC['information_policy']['oracle_based_tuning'])

    def test_progress_and_runtime_records(self):
        rows = []
        matrix = torch.diag(torch.linspace(1., 20., 20))
        _, summary, calls, _ = toy_solve(matrix, torch.ones(20), report=rows.append)
        self.assertEqual(rows, summary['iterations'])
        self.assertEqual(len(calls), 15)
        self.assertTrue(all(row['operator_seconds'] >= 0 and
                            row['operator_eta_seconds'] >= 0 and
                            math.isfinite(row['relative_residual']) for row in rows))
        self.assertTrue(all(not isinstance(value, torch.Tensor)
                            for row in rows for value in row.values()))
        self.assertEqual(SPEC['workload']['exact_rhs_ggn_applications'], 1)
        self.assertEqual(SPEC['workload']['cg_ggn_applications'], 15)
        self.assertEqual(SPEC['workload']['additional_operator_applications_vs_107'], 5)
        self.assertEqual(SPEC['workload']['additional_operator_seconds_context'], 103.5)


if __name__ == '__main__':
    unittest.main()
