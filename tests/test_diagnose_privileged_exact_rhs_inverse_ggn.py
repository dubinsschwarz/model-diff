"""Synthetic/provenance tests for Attempt107; no real model is run."""
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
SOURCE = PROJECT/'scripts/ablation/diagnose_privileged_exact_rhs_inverse_ggn.py'
loader = importlib.util.spec_from_file_location('attempt107_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


def toy_cg(matrix, rhs, fixed_lambda, *, iterations=10, report=None):
    layer = torch.nn.Linear(matrix.shape[0], 1, bias=False)
    eligible = [('linear', layer)]
    calls = []
    def apply(vector, iteration):
        calls.append((iteration, vector['linear.weight'].detach().clone()))
        layer.weight.grad = (matrix @ vector['linear.weight'].flatten()).reshape_as(layer.weight)
        return {'iteration': iteration}
    source = {'linear.weight': rhs.float().reshape_as(layer.weight).contiguous()}
    x, summary = m.fixed_lambda_cg(source, eligible, apply,
                                   fixed_lambda=fixed_lambda, iterations=iterations,
                                   report=report)
    return x['linear.weight'].flatten(), summary, calls, source


class Attempt107Tests(unittest.TestCase):
    def test_attempt106_exact_spec_source_result_linkage(self):
        link = SPEC['attempt106']
        for key in ('spec', 'source', 'result'):
            path = PROJECT/link[key+'_path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             link[key+'_sha256'])
        self.assertEqual(link['spec_sha256'], m.a106.SPEC_SHA256)
        self.assertEqual(link['source_sha256'], m.SOURCE106_SHA256)
        result = m.a.load_json_object(PROJECT/link['result_path'], 'Attempt106 result')
        self.assertEqual(result['matched_activation']['positions_1_4_mean_cosine'],
                         0.7754567731898461)
        self.assertEqual(result['matched_activation']['positions_1_127_mean_cosine'],
                         0.8280211423037307)
        self.assertEqual(result['solver']['rhs_norm'], 8.386915349206442)
        self.assertEqual(result['solver']['rho_b'], 1336.1159773737231)
        self.assertEqual(result['solver']['damping_ratio'], 0.01)
        self.assertEqual(result['solver']['lambda'], 13.361159773737231)
        self.assertEqual(result['solver']['relative_linear_residual'],
                         0.08024706516871907)
        self.assertEqual(result['interpretation']['category'],
                         'moderate_inverse_ggn_support')
        self.assertEqual(result['ggn_operator'], 'exact_endpoint_categorical_JtFJ')
        self.assertEqual(result['inverse_method'], 'fixed_10_step_damped_cg')
        self.assertFalse(result['empirical_fisher'])

    def test_attempt106_mismatch_fails_before_model_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            spec['attempt106']['result_sha256'] = '0'*64
            with patch.object(m.a106, 'validate_inputs',
                              side_effect=AssertionError('model preflight reached')):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    m.validate_inputs(spec)

    def test_attempt106_preflight_reused_with_attempt107_output(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            with (patch.object(m.parent, 'require_hash'),
                  patch.object(m.a106, 'validate_inputs', return_value=('016', {'base_files': []}))
                  as delegated):
                old, inventory = m.validate_inputs(spec)
            self.assertEqual(old, '016')
            self.assertEqual(inventory, {'base_files': []})
            self.assertEqual(delegated.call_args.args[0]['paths']['result_path'],
                             spec['paths']['result_path'])

    def test_frozen_lambda_and_no_rho_from_exact_rhs(self):
        self.assertEqual(SPEC['solver']['fixed_lambda'], 13.361159773737231)
        self.assertEqual(SPEC['solver']['fixed_lambda_source'],
                         'frozen_Attempt106_numeric_value')
        self.assertFalse(SPEC['solver']['recompute_rho_from_exact_rhs'])
        self.assertEqual(SPEC['solver']['iterations'], 10)
        self.assertFalse(SPEC['solver']['residual_early_stop'])
        self.assertNotIn('rho_b', inspect.getsource(m.fixed_lambda_cg))

    def test_true_delta_positive_sign_fp64_subtraction_and_upstream_audit(self):
        base, final = Toy98Model(), Toy98Model()
        with torch.no_grad():
            final.model.layers[0].self_attn['q_proj'].weight.add_(0.25)
        eligible = selected(base)
        saved, records, coordinates = m.a106.a105.a016.snapshot_base(base, eligible)
        base_values = {key: value.detach().clone() for key, value in base.named_parameters()}
        direction, audit = m.a106.a105.a016.audit_and_tangent(
            final, selected(final), saved, records, coordinates,
            lambda key, _record: base_values[key])
        self.assertEqual(len(direction), 98)
        self.assertAlmostEqual(direction['model.layers.0.self_attn.q_proj.weight'].item(),
                               0.25)
        self.assertTrue(audit['omitted_upstream_zero_verified'])
        with torch.no_grad():
            final.model.embed_tokens.weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'Omitted upstream'):
            m.a106.a105.a016.audit_and_tangent(
                final, selected(final), saved, records, coordinates,
                lambda key, _record: base_values[key])

    def test_exact_rhs_uses_stable_G_on_true_delta_and_clears_grads(self):
        model = Toy98Model()
        eligible = selected(model)
        delta = {name+'.weight': torch.ones_like(module.weight)
                 for name, module in eligible}
        tokens = torch.zeros((64, 128), dtype=torch.int64)
        def fake_operator(final, batch, chosen, vector, spec, *, progress):
            self.assertIs(vector, delta)
            self.assertEqual(tuple(batch.shape), (64, 128))
            self.assertEqual(len(chosen), 98)
            self.assertIs(spec, SPEC)
            for _, module in chosen:
                module.weight.grad = torch.full_like(module.weight, 2.)
            return {'primal': 'audited'}, {'seconds': 1.}, {'aggregate': {'count': 8128}}
        with patch.object(m.a106, 'ggn_action_stage', side_effect=fake_operator) as op:
            rhs, geometry, audit = m.exact_rhs_stage(
                model, tokens, eligible, delta, SPEC)
        self.assertEqual(op.call_count, 1)
        self.assertEqual(len(rhs), 98)
        self.assertTrue(all(value.device.type == 'cpu' and value.dtype == torch.float32
                            and value.is_contiguous() and value.item() == 2 for value in rhs.values()))
        self.assertTrue(all(module.weight.grad is None for _, module in eligible))
        self.assertAlmostEqual(geometry['cosine_with_true_delta'], 1)
        self.assertAlmostEqual(geometry['norm_to_true_delta_ratio'], 2)
        self.assertEqual(audit['fisher_conservation']['aggregate']['count'], 8128)

    def test_exact_rhs_missing_selected_gradient_fails(self):
        model = Toy98Model()
        eligible = selected(model)
        with self.assertRaisesRegex(ValueError, 'Missing'):
            m.copy_exact_rhs_from_grad(eligible)
        with self.assertRaisesRegex(ValueError, '98'):
            m.copy_exact_rhs_from_grad(eligible[:97])

    def test_first64_and_probe1024_frozen_inventory(self):
        corpus = torch.arange(512, dtype=torch.int64)[:, None].repeat(1, 128)
        probe = torch.arange(1024, dtype=torch.int64)[:, None].repeat(1, 128)
        with patch.object(m.parent, 'load_frozen_tokens', return_value=(corpus, probe)):
            selected_rows, matched = m.a106.load_frozen_inputs(SPEC)
        self.assertEqual(tuple(selected_rows.shape), (64, 128))
        self.assertEqual(selected_rows[-1, 0].item(), 63)
        self.assertEqual(tuple(matched.shape), (1024, 128))
        self.assertEqual(matched[-1, 0].item(), 1023)
        self.assertEqual(SPEC['pilot']['prediction_contexts'], 8128)
        self.assertEqual(SPEC['probe']['batch_size'], 32)

    def test_stable_Fisher_operator_and_no_empirical_substitution(self):
        self.assertEqual(SPEC['ggn_numerics_106'], m.a106.load_spec()['ggn_numerics_106'])
        self.assertEqual(SPEC['ggn']['operator'], 'exact_endpoint_categorical_JtFJ')
        self.assertTrue(SPEC['ggn']['inverse_solve'])
        self.assertFalse(SPEC['ggn']['empirical_fisher'])
        logits = torch.tensor([[[0.2, -0.1, 0.6]]], dtype=torch.float32)
        tangent = torch.tensor([[[0.3, 1.2, -0.7]]], dtype=torch.float32)
        actual, _ = m.a106.stable_categorical_fisher_vector(logits, tangent, SPEC)
        p = torch.softmax(logits.double(), -1)[0, 0]
        explicit = (torch.diag(p)-torch.outer(p, p)) @ tangent.double()[0, 0]/8128
        self.assertTrue(torch.equal(actual[0, 0], explicit.float()))

    def test_fixed_lambda_CG_ten_iterations_spd_equivalence(self):
        matrix = torch.diag(torch.linspace(1., 12., 12))
        rhs = torch.ones(12)
        x, summary, calls, source = toy_cg(matrix, rhs, 13.361159773737231)
        expected = torch.linalg.solve(matrix+13.361159773737231*torch.eye(12), rhs)
        self.assertEqual(source, {})
        self.assertEqual(len(calls), 10)
        self.assertEqual(summary['operator_applications'], 10)
        self.assertEqual(summary['lambda'], 13.361159773737231)
        self.assertFalse(summary['rho_from_exact_rhs_computed'])
        self.assertTrue(summary['no_early_stop'])
        self.assertEqual(summary['candidate'], 'x_10')
        self.assertEqual([row['iteration'] for row in summary['iterations']],
                         list(range(1, 11)))
        self.assertLess(float(torch.linalg.vector_norm(x-expected)/
                              torch.linalg.vector_norm(expected)), 1e-4)
        self.assertLess(summary['relative_linear_residual'], 1e-4)

    def test_CG_recurrence_first_step_and_fixed_lambda(self):
        matrix = torch.diag(torch.tensor([2., 5., 8.]))
        rhs = torch.tensor([1., 2., 3.])
        damping = 13.361159773737231
        x, summary, calls, _ = toy_cg(matrix, rhs, damping, iterations=1)
        expected_alpha = float(rhs @ rhs)/float(rhs @ ((matrix+damping*torch.eye(3))@rhs))
        self.assertAlmostEqual(summary['iterations'][0]['alpha'], expected_alpha, places=6)
        self.assertTrue(torch.allclose(x, expected_alpha*rhs, atol=1e-7))
        self.assertTrue(torch.equal(calls[0][1].flatten(), rhs))

    def test_toy_exact_G_delta_rhs_inverse(self):
        ggn = torch.diag(torch.linspace(0.5, 6., 12))
        delta = torch.linspace(-1., 1., 12)
        exact_rhs = ggn@delta
        damping = 13.361159773737231
        x, summary, calls, _ = toy_cg(ggn, exact_rhs, damping)
        expected = torch.linalg.solve(ggn+damping*torch.eye(12), ggn@delta)
        self.assertEqual(len(calls), 10)
        self.assertTrue(torch.allclose(x, expected, rtol=1e-4, atol=1e-5))
        self.assertEqual(summary['operator_applications'], 10)

    def test_nonpositive_pAp_and_nonfinite_action_fail(self):
        layer = torch.nn.Linear(2, 1, bias=False)
        rhs = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}
        def negative(vector, _iteration):
            layer.weight.grad = -100*vector['linear.weight']
            return {}
        with self.assertRaisesRegex(ValueError, r'p\^T A p'):
            m.fixed_lambda_cg(rhs, [('linear', layer)], negative,
                              fixed_lambda=13.361159773737231, iterations=1)
        rhs = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}
        def nonfinite(vector, _iteration):
            layer.weight.grad = torch.full_like(vector['linear.weight'], float('nan'))
            return {}
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            m.fixed_lambda_cg(rhs, [('linear', layer)], nonfinite,
                              fixed_lambda=13.361159773737231, iterations=1)

    def test_matrixwise_fp64_reduction_and_no_fp64_parameter_copy(self):
        values = torch.tensor([1e8, 1., -1e8, 1.], dtype=torch.float32)
        self.assertEqual(m.a106.matrixwise_dot({'w': values},
                                               {'w': torch.ones_like(values)}), 2.)
        self.assertNotIn('.double()', inspect.getsource(m.fixed_lambda_cg))
        self.assertIn('matrixwise_dot', inspect.getsource(m.fixed_lambda_cg))

    def test_parameter_diagnostics_against_delta_are_descriptive(self):
        model = Toy98Model()
        eligible = selected(model)
        x = {name+'.weight': torch.ones_like(module.weight) for name, module in eligible}
        delta = {name+'.weight': torch.full_like(module.weight, 2.)
                 for name, module in eligible}
        geometry = m.a106.parameter_diagnostics(x, delta)
        self.assertAlmostEqual(geometry['global']['cosine'], 1)
        self.assertAlmostEqual(geometry['global']['candidate_to_delta_norm_ratio'], 0.5)
        self.assertEqual(len(geometry['blocks']), 14)
        self.assertEqual(geometry['attention']['matrix_count'], 56)
        self.assertEqual(geometry['mlp']['matrix_count'], 42)

    def test_strict_final_block13_JVP(self):
        model = TinyReadout()
        spec = copy.deepcopy(SPEC)
        spec['readout']['hidden_size'] = 2
        tokens = torch.tensor([[0, 1]*64], dtype=torch.int64)
        direction = {'model.layers.0.weight': torch.tensor([[0.1, 0.2], [0.3, 0.4]])}
        primal, tangent = m.a.run_readout_jvp(model, tokens, direction, spec, torch)
        ordinary = m.a.ordinary_hook_readout(model, tokens, spec, torch)
        self.assertEqual(m.a.compare_primal(primal, ordinary, spec, torch), 0)
        self.assertFalse(tangent.requires_grad)
        self.assertTrue(bool(torch.isfinite(tangent).all()))

    def test_matched_target_hash_and_position_metrics(self):
        base = torch.zeros((128, 2048), dtype=torch.float64)
        final = torch.ones_like(base)
        target = m.parent.matched_difference(final, base)
        spec = copy.deepcopy(SPEC)
        spec['target']['raw_float32_sha256'] = m.a.sha256_raw_float32_tensor(target, torch)
        checked, digest = m.a106.verified_matched_target(final, base, spec)
        self.assertTrue(torch.equal(checked, target))
        self.assertEqual(digest, spec['target']['raw_float32_sha256'])
        with self.assertRaisesRegex(ValueError, 'Matched historical target'):
            m.a106.verified_matched_target(final, base, SPEC)
        response = torch.ones((128, 2048), dtype=torch.float32)
        target[0].neg_()
        target[2, :1024] = -1
        metrics = m.parent.response_metrics(response, target)
        self.assertAlmostEqual(metrics['position_0_cosine'], -1)
        self.assertAlmostEqual(metrics['positions_1_4_mean_cosine'], 0.75)
        self.assertAlmostEqual(metrics['positions_1_127_mean_cosine'], 126/127)
        self.assertEqual(len(metrics['position_cosines']), 128)
        self.assertIn('best_scalar_rescaling_diagnostic_only', metrics)

    def test_frozen_comparisons_gap_and_categories(self):
        self.assertEqual(SPEC['baselines']['attempt106_primary'], 0.7754567731898461)
        self.assertEqual(SPEC['baselines']['attempt106_secondary'], 0.8280211423037307)
        self.assertEqual(SPEC['baselines']['attempt104_primary'], 0.6084827426677797)
        self.assertEqual(SPEC['baselines']['attempt104_secondary'], 0.6499135789428421)
        self.assertEqual(SPEC['baselines']['attempt016_primary_context_approx'], 0.9836)
        self.assertEqual(m.interpretation(0.90, SPEC), 'strong_rhs_mismatch_explanation')
        self.assertEqual(m.interpretation(0.85, SPEC),
                         'mixed_rhs_and_inverse_limitations')
        self.assertEqual(m.interpretation(0.84, SPEC),
                         'inverse_or_regularization_limitations_dominate')

    def test_no_oracle_adapter_or_large_tensor_persistence(self):
        source = SOURCE.read_text()
        self.assertNotIn('oracle_adl.pt', source)
        self.assertNotIn('adapter_checkpoint', source)
        self.assertNotIn('torch.save(', source)
        self.assertFalse(SPEC['information_policy']['oracle_adl_access'])
        self.assertFalse(SPEC['information_policy']['adapter_access'])
        self.assertTrue(SPEC['information_policy']['true_delta_rhs'])
        self.assertTrue(SPEC['output_policy']['result_json_only'])
        self.assertFalse(SPEC['output_policy']['large_tensor_persistence'])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            output.write_text('frozen')
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(output)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.validate_inputs(spec)
            self.assertEqual(output.read_text(), 'frozen')

    def test_progress_and_runtime_accounting_has_no_vector_history(self):
        rows = []
        matrix = torch.diag(torch.linspace(1., 12., 12))
        rhs = torch.ones(12)
        x, summary, calls, _ = toy_cg(matrix, rhs, 13.361159773737231,
                                      report=rows.append)
        self.assertEqual(rows, summary['iterations'])
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(row['operator_seconds'] >= 0 and
                            row['operator_eta_seconds'] >= 0 for row in rows))
        self.assertTrue(all(not isinstance(value, torch.Tensor)
                            for row in rows for value in row.values()))
        self.assertEqual(SPEC['workload']['exact_rhs_ggn_applications'], 1)
        self.assertEqual(SPEC['workload']['cg_ggn_applications'], 10)
        self.assertEqual(SPEC['workload']['ggn_microbatches_per_application'], 64)
        self.assertEqual(SPEC['workload']['base_probe_batches'], 32)
        self.assertEqual(SPEC['workload']['final_jvp_batches'], 32)
        self.assertEqual(len(x), 12)


if __name__ == '__main__':
    unittest.main()
