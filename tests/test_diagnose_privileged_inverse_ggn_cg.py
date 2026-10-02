"""Synthetic Attempt106 checks; no real checkpoint or probe is executed."""
import copy
import hashlib
import inspect
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util

import torch

from tests.test_diagnose_privileged_endpoint_ggn_forward import Toy98Model, selected


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/diagnose_privileged_inverse_ggn_cg.py'
loader = importlib.util.spec_from_file_location('attempt106_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


def toy_cg(matrix, rhs, *, iterations=10, damping_ratio=0.01):
    layer = torch.nn.Linear(matrix.shape[0], 1, bias=False)
    eligible = [('linear', layer)]
    calls = []
    def apply(vector, iteration):
        calls.append((iteration, vector['linear.weight'].detach().clone()))
        layer.weight.grad = (matrix @ vector['linear.weight'].flatten()).reshape_as(layer.weight)
        return {'iteration': iteration, 'no_tensor': True}
    original = {'linear.weight': rhs.reshape_as(layer.weight).float().contiguous()}
    x, report = m.fixed_budget_cg(original, eligible, apply,
                                  iterations=iterations, damping_ratio=damping_ratio)
    return x['linear.weight'].flatten(), report, calls, original


class TinyReadoutInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)]+[
            torch.nn.Identity() for _ in range(27)])
        with torch.no_grad():
            self.layers[0].weight.copy_(torch.eye(2))

    def forward(self, input_ids, use_cache, output_hidden_states, return_dict):
        assert use_cache is False and output_hidden_states and return_dict
        x = torch.nn.functional.one_hot(input_ids % 2, 2).float()
        states = [x]
        for layer in self.layers:
            x = layer(x)
            states.append(x)
        return SimpleNamespace(hidden_states=tuple(states))


class TinyReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyReadoutInner()


class Attempt106Tests(unittest.TestCase):
    def test_attempt105_spec_source_result_hash_and_category(self):
        link = SPEC['attempt105']
        for key in ('spec', 'source', 'result'):
            path = PROJECT/link[key+'_path']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             link[key+'_sha256'])
        result = m.a.load_json_object(PROJECT/link['result_path'], 'Attempt105 result')
        self.assertEqual(result['parameter_geometry']['primary_global_signed_cosine'],
                         0.9482682230024669)
        self.assertEqual(result['parameter_geometry']['hybrid_gradient_norm'],
                         8.38691536284068)
        self.assertEqual(result['interpretation']['category'],
                         'strong_endpoint_ggn_support')
        self.assertEqual(result['ggn_operator'], 'exact_endpoint_categorical_JtFJ')
        self.assertFalse(result['inverse_solve'])
        self.assertFalse(result['empirical_fisher'])

    def test_attempt104_attempt016_and_checkpoint_preflight_delegation(self):
        prior = m.a105.load_spec()
        self.assertEqual(SPEC['attempt104'], prior['attempt104'])
        self.assertEqual(SPEC['attempt016'], prior['attempt016'])
        self.assertEqual(SPEC['base'], prior['base'])
        self.assertEqual(SPEC['final_checkpoint_files'], prior['final_checkpoint_files'])
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'future.json')
            with (patch.object(m.parent, 'require_hash'),
                  patch.object(m.a105, 'validate_inputs',
                               side_effect=ValueError('checkpoint mismatch')) as delegated):
                with self.assertRaisesRegex(ValueError, 'checkpoint mismatch'):
                    m.validate_inputs(spec)
                proxy = delegated.call_args.args[0]
                self.assertEqual(proxy['paths']['result_path'], spec['paths']['result_path'])
                self.assertEqual(proxy['attempt016'], SPEC['attempt016'])

    def test_linkage_mismatch_fails_before_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'future.json')
            spec['attempt105']['result_sha256'] = '0'*64
            with patch.object(m.a105, 'validate_inputs',
                              side_effect=AssertionError('reached checkpoint')):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    m.validate_inputs(spec)

    def test_first64_and_first1024_inventories(self):
        corpus = torch.arange(512, dtype=torch.int64)[:, None].repeat(1, 128)
        probe = torch.arange(1024, dtype=torch.int64)[:, None].repeat(1, 128)
        with patch.object(m.parent, 'load_frozen_tokens', return_value=(corpus, probe)):
            chosen, matched = m.load_frozen_inputs(SPEC)
        self.assertEqual(tuple(chosen.shape), (64, 128))
        self.assertEqual(chosen[-1, 0].item(), 63)
        self.assertEqual(tuple(matched.shape), (1024, 128))
        self.assertEqual(matched[-1, 0].item(), 1023)
        self.assertEqual(SPEC['pilot']['prediction_contexts'], 8128)
        self.assertEqual(SPEC['probe']['batch_size'], 32)

    def test_true_delta_positive_sign_and_omitted_upstream_audit(self):
        base, final = Toy98Model(), Toy98Model()
        eligible = selected(base)
        with torch.no_grad():
            final.model.layers[0].self_attn['q_proj'].weight.add_(0.125)
        saved, records, coordinates = m.a105.a016.snapshot_base(base, eligible)
        base_values = {key: value.detach().clone() for key, value in base.named_parameters()}
        tangent, audit = m.a105.a016.audit_and_tangent(
            final, selected(final), saved, records, coordinates,
            lambda key, _record: base_values[key])
        self.assertAlmostEqual(tangent['model.layers.0.self_attn.q_proj.weight'].item(), 0.125)
        self.assertEqual(len(tangent), 98)
        self.assertTrue(audit['omitted_upstream_zero_verified'])
        with torch.no_grad():
            final.model.embed_tokens.weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'Omitted upstream'):
            m.a105.a016.audit_and_tangent(
                final, selected(final), saved, records, coordinates,
                lambda key, _record: base_values[key])

    def test_rhs_uses_exact_positive_hybrid_helper(self):
        base, final = Toy98Model(), Toy98Model()
        for value in base.parameters():
            value.requires_grad_(False)
        eligible = selected(final)
        m.a.freeze_other_parameters(final, eligible)
        with torch.no_grad():
            final.model.layers[0].self_attn['q_proj'].weight.add_(0.01)
        spec = copy.deepcopy(SPEC)
        spec['readout']['hidden_size'] = 1
        tokens = torch.tensor([[0, 1]*64]*4, dtype=torch.int64)
        rhs, info = m.a105.hybrid_gradient_64(base, final, tokens, eligible, spec)
        self.assertEqual(len(rhs), 98)
        self.assertEqual(info['normalization_denominator'], 8128)
        self.assertTrue(any(float(value.abs().sum()) > 0 for value in rhs.values()))
        self.assertTrue(all(value.device.type == 'cpu' and value.is_contiguous()
                            for value in rhs.values()))
        self.assertTrue(all(value.grad is None for value in base.parameters()))
        self.assertTrue(all(value.grad is None for value in final.parameters()))

    def test_matrixwise_fp64_reductions_without_full_fp64_vectors(self):
        x = torch.tensor([1e8, 1, -1e8, 1], dtype=torch.float32)
        original_sum = torch.sum
        dtypes = []
        def spy(*args, **kwargs):
            dtypes.append(kwargs.get('dtype'))
            return original_sum(*args, **kwargs)
        with patch.object(m.torch, 'sum', side_effect=spy):
            result = m.matrixwise_dot({'w': x}, {'w': torch.ones_like(x)})
        self.assertEqual(result, 2.0)
        self.assertEqual(dtypes, [torch.float64])
        self.assertNotIn('.double()', inspect.getsource(m.matrixwise_dot))

    def test_rho_damping_first_action_reused_and_ten_iterations(self):
        matrix = torch.diag(torch.linspace(1, 12, 12))
        rhs = torch.ones(12)
        x, report, calls, original = toy_cg(matrix, rhs)
        self.assertEqual(original, {})
        self.assertEqual(len(calls), 10)
        self.assertEqual([row['iteration'] for row in report['iterations']], list(range(1, 11)))
        self.assertTrue(torch.equal(calls[0][1].flatten(), rhs))
        rho = float(rhs @ (matrix @ rhs))/(float(rhs @ rhs))
        self.assertAlmostEqual(report['rho_b'], rho, places=6)
        self.assertAlmostEqual(report['lambda'], 0.01*rho, places=6)
        self.assertEqual(report['operator_applications'], 10)
        self.assertTrue(report['no_early_stop'])
        self.assertEqual(report['residual_source'],
                         'CG_recurrence_no_extra_operator_application')
        self.assertEqual(report['candidate'], 'x_10')
        expected = torch.linalg.solve(matrix+report['lambda']*torch.eye(12), rhs)
        self.assertLess(float(torch.linalg.vector_norm(x-expected)/
                              torch.linalg.vector_norm(expected)), 5e-4)
        self.assertLess(report['relative_linear_residual'], 1e-3)
        self.assertEqual(len(report['operator_audits']), 10)
        self.assertTrue(all(isinstance(row['operator_seconds'], float)
                            and row['operator_eta_seconds'] >= 0 for row in report['iterations']))

    def test_cg_recurrence_first_iteration(self):
        matrix = torch.diag(torch.tensor([2., 5., 8.]))
        rhs = torch.tensor([1., 2., 3.])
        x, report, _, _ = toy_cg(matrix, rhs, iterations=1)
        damping = report['lambda']
        alpha = float(rhs @ rhs)/float(rhs @ ((matrix+damping*torch.eye(3)) @ rhs))
        self.assertAlmostEqual(report['iterations'][0]['alpha'], alpha, places=6)
        self.assertTrue(torch.allclose(x, alpha*rhs, atol=1e-7))
        residual = rhs-(matrix+damping*torch.eye(3)) @ x
        self.assertAlmostEqual(report['relative_linear_residual'],
                               float(torch.linalg.vector_norm(residual)/torch.linalg.vector_norm(rhs)),
                               places=6)

    def test_nonpositive_rayleigh_and_pap_fail_closed(self):
        layer = torch.nn.Linear(2, 1, bias=False)
        rhs = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}
        def negative(vector, _iteration):
            layer.weight.grad = -vector['linear.weight'].clone()
            return {}
        with self.assertRaisesRegex(ValueError, 'Rayleigh'):
            m.fixed_budget_cg(rhs, [('linear', layer)], negative, iterations=1)
        rhs = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}
        def changes_sign(vector, iteration):
            layer.weight.grad = (1 if iteration == 1 else -100)*vector['linear.weight'].clone()
            return {}
        with self.assertRaisesRegex(ValueError, r'p\^T A p'):
            m.fixed_budget_cg(rhs, [('linear', layer)], changes_sign, iterations=2)

    def test_toy_explicit_categorical_ggn_inverse(self):
        generator = torch.Generator().manual_seed(106)
        logits = torch.randn((8, 4), generator=generator)
        p = torch.softmax(logits, dim=-1)
        fisher = torch.block_diag(*[torch.diag(row)-torch.outer(row, row) for row in p])
        jacobian = torch.randn((32, 12), generator=generator)
        ggn = (jacobian.T @ fisher @ jacobian)/8128
        rhs = torch.linspace(0.1, 1.2, 12)
        x, report, calls, _ = toy_cg(ggn, rhs)
        expected = torch.linalg.solve(ggn+report['lambda']*torch.eye(12), rhs)
        self.assertEqual(len(calls), 10)
        self.assertLess(float(torch.linalg.vector_norm(x-expected)/
                              torch.linalg.vector_norm(expected)), 5e-3)
        self.assertGreater(report['rho_b'], 0)
        self.assertFalse(SPEC['ggn']['empirical_fisher'])
        self.assertEqual(SPEC['ggn']['operator'], 'exact_endpoint_categorical_JtFJ')
        self.assertEqual(SPEC['ggn']['fisher_conservation_audit_accumulation'],
                         'torch.sum_dtype_float64_without_modifying_u')

    def test_parameter_diagnostics_global_blocks_families(self):
        model = Toy98Model()
        eligible = selected(model)
        x = {name+'.weight': torch.ones_like(module.weight) for name, module in eligible}
        delta = {name+'.weight': torch.full_like(module.weight, 2.)
                 for name, module in eligible}
        row = m.parameter_diagnostics(x, delta)
        self.assertAlmostEqual(row['global']['cosine'], 1)
        self.assertAlmostEqual(row['global']['candidate_to_delta_norm_ratio'], 0.5)
        self.assertEqual(len(row['blocks']), 14)
        self.assertEqual(row['attention']['matrix_count'], 56)
        self.assertEqual(row['mlp']['matrix_count'], 42)

    def test_strict_final_block13_native_jvp(self):
        model = TinyReadout()
        spec = copy.deepcopy(SPEC)
        spec['readout']['hidden_size'] = 2
        tokens = torch.tensor([[0, 1]*64], dtype=torch.int64)
        tangent = {'model.layers.0.weight': torch.tensor([[0.1, 0.2], [0.3, 0.4]])}
        primal, response = m.a.run_readout_jvp(model, tokens, tangent, spec, torch)
        ordinary = m.a.ordinary_hook_readout(model, tokens, spec, torch)
        self.assertEqual(m.a.compare_primal(primal, ordinary, spec, torch), 0.0)
        expected = torch.nn.functional.linear(
            torch.nn.functional.one_hot(tokens, 2).float(), tangent['model.layers.0.weight'])
        self.assertTrue(torch.allclose(response, expected, atol=1e-7))
        self.assertFalse(primal.requires_grad)

    def test_matched_target_cast_and_hash_requirement(self):
        base = torch.zeros((128, 2048), dtype=torch.float64)
        final = torch.full_like(base, 0.125)
        expected = m.parent.matched_difference(final, base)
        digest = m.a.sha256_raw_float32_tensor(expected, torch)
        spec = copy.deepcopy(SPEC)
        spec['target']['raw_float32_sha256'] = digest
        target, actual = m.verified_matched_target(final, base, spec)
        self.assertTrue(torch.equal(target, expected))
        self.assertEqual(target.dtype, torch.float32)
        self.assertTrue(target.is_contiguous())
        self.assertEqual(actual, digest)
        with self.assertRaisesRegex(ValueError, 'Matched historical target'):
            m.verified_matched_target(final, base, SPEC)

    def test_positionwise_signed_cosines_and_best_scalar_diagnostic(self):
        response = torch.ones((128, 2048), dtype=torch.float32)
        target = response.clone()
        target[0].neg_()
        target[2, :1024] = -1
        geometry = m.parent.response_metrics(response, target)
        self.assertAlmostEqual(geometry['position_0_cosine'], -1)
        self.assertAlmostEqual(geometry['positions_1_4_mean_cosine'], 0.75)
        self.assertAlmostEqual(geometry['positions_1_127_mean_cosine'], 126/127)
        self.assertEqual(len(geometry['position_cosines']), 128)
        self.assertIn('best_scalar_rescaling_diagnostic_only', geometry)
        self.assertEqual(SPEC['metric']['primary_positions'], [1, 2, 3, 4])

    def test_baselines_and_precommitted_categories(self):
        self.assertEqual(SPEC['baselines']['attempt104_primary'], 0.6084827426677797)
        self.assertEqual(SPEC['baselines']['attempt104_secondary'], 0.6499135789428421)
        self.assertEqual(SPEC['baselines']['attempt103_primary'], 0.5956728332916521)
        self.assertEqual(SPEC['baselines']['attempt103_secondary'], 0.6404746527850079)
        self.assertEqual(m.interpretation(0.80, SPEC), 'strong_inverse_ggn_support')
        self.assertEqual(m.interpretation(0.70, SPEC), 'moderate_inverse_ggn_support')
        self.assertEqual(m.interpretation(0.699, SPEC),
                         'weak_or_inconclusive_inverse_ggn_support')

    def test_no_oracle_adapter_candidate_tuning_or_large_tensor_output(self):
        source = SOURCE.read_text()
        self.assertNotIn('oracle_adl.pt', source)
        self.assertNotIn('adapter_checkpoint', source)
        self.assertNotIn('torch.save(', source)
        self.assertFalse(SPEC['information_policy']['oracle_adl_access'])
        self.assertFalse(SPEC['information_policy']['adapter_access'])
        self.assertFalse(SPEC['information_policy']['candidate_selection'])
        self.assertFalse(SPEC['information_policy']['oracle_based_tuning'])
        self.assertTrue(SPEC['output_policy']['result_json_only'])
        self.assertFalse(SPEC['output_policy']['large_tensor_persistence'])

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory)/'result.json'
            result.write_text('frozen')
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(result)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.validate_inputs(spec)
            self.assertEqual(result.read_text(), 'frozen')

    def test_progress_and_scalar_only_solver_history(self):
        rows = []
        matrix = torch.diag(torch.linspace(1, 12, 12))
        layer = torch.nn.Linear(12, 1, bias=False)
        rhs = {'linear.weight': torch.ones((1, 12), dtype=torch.float32)}
        def apply(vector, _iteration):
            layer.weight.grad = (matrix @ vector['linear.weight'].flatten()).reshape(1, 12)
            return {'finite': True}
        x, report = m.fixed_budget_cg(rhs, [('linear', layer)], apply, report=rows.append)
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows, report['iterations'])
        self.assertTrue(all(row['cumulative_elapsed_seconds'] >= 0 and
                            row['operator_eta_seconds'] >= 0 for row in rows))
        self.assertTrue(all(not isinstance(value, torch.Tensor)
                            for row in rows for value in row.values()))
        self.assertEqual(len(x), 1)
        self.assertEqual(len(report['operator_audits']), 10)


if __name__ == '__main__':
    unittest.main()
