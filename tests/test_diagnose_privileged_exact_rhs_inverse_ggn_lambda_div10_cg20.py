"""Synthetic and frozen-linkage tests for Attempt110; no real models."""
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
SOURCE = PROJECT/'scripts/ablation/diagnose_privileged_exact_rhs_inverse_ggn_lambda_div10_cg20.py'
loader = importlib.util.spec_from_file_location('attempt110_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


def toy_cg(matrix, rhs, iterations, *, frozen_rows=None, report=None, calls=None):
    layer = torch.nn.Linear(matrix.shape[0], 1, bias=False)
    eligible = [('linear', layer)]
    if calls is None:
        calls = []

    def apply(vector, iteration):
        calls.append(iteration)
        layer.weight.grad = (matrix @ vector['linear.weight'].flatten()).reshape_as(layer.weight)
        return {'iteration': iteration}

    source = {'linear.weight': rhs.float().reshape_as(layer.weight).contiguous()}
    if frozen_rows is None:
        x, summary = m.a107.fixed_lambda_cg(
            source, eligible, apply, fixed_lambda=m.FIXED_LAMBDA,
            iterations=iterations, report=report)
    else:
        x, summary = m.fixed_20_step_cg(
            source, eligible, apply, SPEC, frozen_rows, report=report)
    return x['linear.weight'].flatten(), summary, calls, source


class Attempt110Tests(unittest.TestCase):
    def test_attempt109_spec_source_result_and_commit_linkage(self):
        link = SPEC['attempt109']
        for key in ('spec', 'source', 'result'):
            self.assertEqual(hashlib.sha256((PROJECT/link[key+'_path']).read_bytes()).hexdigest(),
                             link[key+'_sha256'])
        self.assertEqual(link['commit_sha'], m.COMMIT109)
        self.assertEqual(link['spec_sha256'], m.a109.SPEC_SHA256)
        self.assertEqual(link['source_sha256'], m.SOURCE109_SHA256)
        m.verify_committed_attempt109(link)

    def test_attempt109_frozen_result_values(self):
        link = SPEC['attempt109']
        result = m.a.load_json_object(PROJECT/link['result_path'], 'Attempt109 result')
        self.assertEqual(result['matched_activation']['positions_1_4_mean_cosine'],
                         0.8802197432881782)
        self.assertEqual(result['matched_activation']['positions_1_127_mean_cosine'],
                         0.8858106177907814)
        self.assertEqual(result['solver']['relative_linear_residual'],
                         0.04024542110658629)
        self.assertEqual(result['solver']['rhs_norm'], 14.651196958318208)
        self.assertEqual(result['solver']['lambda'], m.FIXED_LAMBDA)
        self.assertEqual(result['solver']['operator_applications'], 15)
        self.assertEqual(result['interpretation']['category'],
                         'moderate_damping_suppression_support')
        self.assertTrue(result['true_delta_rhs'])
        self.assertTrue(result['inverse_solve'])
        self.assertEqual(result['ggn_operator'], 'exact_endpoint_categorical_JtFJ')
        self.assertFalse(result['empirical_fisher'])

    def test_bad_attempt109_hash_fails_before_model_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            spec['attempt109']['result_sha256'] = '0'*64
            with patch.object(m.a109, 'validate_inputs',
                              side_effect=AssertionError('model preflight reached')):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    m.validate_inputs(spec)

    def test_committed_revision_mismatch_fails(self):
        bad = dict(SPEC['attempt109'], commit_sha='0'*40)
        with self.assertRaisesRegex(ValueError, 'committed revision'):
            m.verify_committed_attempt109(bad)

    def test_frozen_prior_preflight_returns_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            with (patch.object(m.parent, 'require_hash'),
                  patch.object(m.a109, 'validate_inputs', return_value=('016', {'base_files': []}))
                  as delegated):
                old, inventories, result = m.validate_inputs(spec)
            self.assertEqual((old, inventories), ('016', {'base_files': []}))
            self.assertEqual(len(result['solver']['iterations']), 15)
            self.assertEqual(delegated.call_args.args[0]['paths']['result_path'],
                             spec['paths']['result_path'])

    def test_only_scientific_variable_is_CG_budget(self):
        prior = m.a109.load_spec()
        for key in ('attempt104', 'attempt016', 'attempt106', 'attempt107',
                    'attempt108', 'base', 'final_checkpoint_files', 'model',
                    'readout', 'eligible_tensors', 'corpus', 'pilot', 'probe',
                    'probe_rows', 'displacement', 'ggn', 'ggn_numerics_106',
                    'jvp', 'metric', 'target', 'exact_rhs', 'output_policy'):
            self.assertEqual(SPEC[key], prior[key], key)
        before = dict(prior['solver'])
        after = dict(SPEC['solver'])
        before.pop('iterations')
        before.pop('candidate')
        after.pop('iterations')
        after.pop('candidate')
        self.assertEqual(before, after)
        self.assertEqual(SPEC['solver']['fixed_lambda'], 1.3361159773737232)
        self.assertEqual(SPEC['solver']['iterations'], 20)
        self.assertEqual(SPEC['solver']['candidate'], 'x_20')

    def test_exact_rhs_and_98_coordinate_support_unchanged(self):
        model = Toy98Model()
        eligible = selected(model)
        delta = {name+'.weight': torch.ones_like(module.weight)
                 for name, module in eligible}
        tokens = torch.zeros((64, 128), dtype=torch.int64)

        def fake_g(final, batch, chosen, vector, spec, *, progress):
            self.assertIs(vector, delta)
            self.assertEqual(tuple(batch.shape), (64, 128))
            self.assertEqual(len(chosen), 98)
            for _, module in chosen:
                module.weight.grad = torch.full_like(module.weight, 2.)
            return {}, {}, {'aggregate': {'prediction_context_count': 8128}}

        with patch.object(m.a106, 'ggn_action_stage', side_effect=fake_g) as operator:
            rhs, geometry, _ = m.a107.exact_rhs_stage(
                model, tokens, eligible, delta, SPEC)
        self.assertEqual(operator.call_count, 1)
        self.assertEqual(len(rhs), 98)
        self.assertTrue(all(value.item() == 2. for value in rhs.values()))
        self.assertAlmostEqual(geometry['cosine_with_true_delta'], 1)

    def test_true_delta_positive_sign_and_upstream_audit(self):
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

    def test_stable_categorical_GGN_unchanged(self):
        self.assertEqual(SPEC['ggn_numerics_106'],
                         m.a109.load_spec()['ggn_numerics_106'])
        self.assertFalse(SPEC['ggn']['empirical_fisher'])
        logits = torch.tensor([[[0.2, -0.1, 0.6]]], dtype=torch.float32)
        tangent = torch.tensor([[[0.3, 1.2, -0.7]]], dtype=torch.float32)
        actual, _ = m.a106.stable_categorical_fisher_vector(logits, tangent, SPEC)
        p = torch.softmax(logits.double(), -1)[0, 0]
        expected = (torch.diag(p)-torch.outer(p, p)) @ tangent.double()[0, 0]/8128
        self.assertTrue(torch.equal(actual[0, 0], expected.float()))

    def test_frozen_64_construction_and_1024_probe(self):
        corpus = torch.arange(512, dtype=torch.int64)[:, None].repeat(1, 128)
        probe = torch.arange(1024, dtype=torch.int64)[:, None].repeat(1, 128)
        with patch.object(m.parent, 'load_frozen_tokens', return_value=(corpus, probe)):
            construction, matched = m.a106.load_frozen_inputs(SPEC)
        self.assertEqual(tuple(construction.shape), (64, 128))
        self.assertEqual(construction[-1, 0].item(), 63)
        self.assertEqual(tuple(matched.shape), (1024, 128))
        self.assertEqual(matched[-1, 0].item(), 1023)
        self.assertEqual(SPEC['pilot']['prediction_contexts'], 8128)

    def test_fixed_lambda_20_step_CG_matches_toy_SPD_solve(self):
        matrix = torch.diag(torch.linspace(1., 25., 25))
        rhs = torch.ones(25)
        _, prior, _, _ = toy_cg(matrix, rhs, 15)
        x, summary, calls, source = toy_cg(
            matrix, rhs, 20, frozen_rows=prior['iterations'])
        expected = torch.linalg.solve(matrix+m.FIXED_LAMBDA*torch.eye(25), rhs)
        self.assertEqual(calls, list(range(1, 21)))
        self.assertEqual(source, {})
        self.assertEqual(summary['operator_applications'], 20)
        self.assertEqual(len(summary['iterations']), 20)
        self.assertEqual(summary['candidate'], 'x_20')
        self.assertTrue(summary['no_early_stop'])
        self.assertLess(float(torch.linalg.vector_norm(x-expected)/
                              torch.linalg.vector_norm(expected)), 1e-4)

    def test_first15_trajectory_matches_and_continues(self):
        matrix = torch.diag(torch.linspace(1., 25., 25))
        rhs = torch.linspace(1., 2., 25)
        _, prior, _, _ = toy_cg(matrix, rhs, 15)
        _, summary, calls, _ = toy_cg(matrix, rhs, 20,
                                      frozen_rows=prior['iterations'])
        audit = summary['first15_trajectory_audit']
        self.assertTrue(audit['matched'])
        self.assertEqual(audit['rows_compared'], 15)
        self.assertEqual(audit['fields'],
                         ['relative_residual', 'r_norm', 'pAp', 'alpha', 'beta'])
        self.assertFalse(audit['timings_compared'])
        self.assertEqual(calls[-1], 20)

    def test_trajectory_mismatch_fails_before_next_operator(self):
        matrix = torch.diag(torch.linspace(1., 25., 25))
        rhs = torch.linspace(1., 2., 25)
        _, prior, _, _ = toy_cg(matrix, rhs, 15)
        frozen = copy.deepcopy(prior['iterations'])
        frozen[2]['pAp'] *= 1.01
        calls = []
        with self.assertRaisesRegex(ValueError, 'trajectory mismatch'):
            toy_cg(matrix, rhs, 20, frozen_rows=frozen, calls=calls)
        self.assertEqual(calls, [1, 2, 3])

    def test_trajectory_ignores_wall_clock_timings(self):
        matrix = torch.diag(torch.linspace(1., 25., 25))
        rhs = torch.ones(25)
        _, prior, _, _ = toy_cg(matrix, rhs, 15)
        frozen = copy.deepcopy(prior['iterations'])
        for row in frozen:
            row['operator_seconds'] = 1e9
            row['cumulative_elapsed_seconds'] = -1e9
            row['operator_eta_seconds'] = 1e9
        _, summary, calls, _ = toy_cg(matrix, rhs, 20, frozen_rows=frozen)
        self.assertEqual(len(calls), 20)
        self.assertTrue(summary['first15_trajectory_audit']['matched'])

    def test_trajectory_tolerance_and_nonfinite_fail_closed(self):
        policy = SPEC['trajectory_audit']
        row = {'iteration': 1, **{field: 1. for field in policy['fields']}}
        expected = dict(row)
        expected['pAp'] = 1.+5e-7
        m.audit_trajectory_row(row, expected, policy)
        expected['pAp'] = 1.+5e-4
        with self.assertRaisesRegex(ValueError, 'trajectory mismatch'):
            m.audit_trajectory_row(row, expected, policy)
        expected['pAp'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            m.audit_trajectory_row(row, expected, policy)

    def test_no_preconditioner_early_stop_or_adaptive_extension(self):
        self.assertIsNone(SPEC['solver']['preconditioner'])
        self.assertFalse(SPEC['solver']['residual_early_stop'])
        self.assertFalse(SPEC['interpretation']['adaptive_extra_iterations'])
        bad = copy.deepcopy(SPEC)
        bad['solver']['preconditioner'] = 'diagonal'
        with self.assertRaisesRegex(ValueError, 'settings changed'):
            m.fixed_20_step_cg({}, [], lambda *_: None, bad, [{}]*15)

    def test_fixed_lambda_CG_first_step_recurrence(self):
        matrix = torch.diag(torch.linspace(1., 25., 25))
        rhs = torch.arange(1., 26.)
        _, summary, _, _ = toy_cg(matrix, rhs, 15)
        alpha = float(rhs@rhs)/float(rhs@((matrix+m.FIXED_LAMBDA*torch.eye(25))@rhs))
        self.assertAlmostEqual(summary['iterations'][0]['alpha'], alpha, places=6)
        self.assertEqual(SPEC['solver']['fixed_lambda'], 1.3361159773737232)
        self.assertFalse(SPEC['solver']['recompute_rho_from_exact_rhs'])

    def test_exact_rhs_toy_inverse_under_20_steps(self):
        ggn = torch.diag(torch.linspace(0.5, 8., 25))
        delta = torch.linspace(-1., 1., 25)
        rhs = ggn@delta
        _, prior, _, _ = toy_cg(ggn, rhs, 15)
        x, summary, _, _ = toy_cg(ggn, rhs, 20,
                                   frozen_rows=prior['iterations'])
        expected = torch.linalg.solve(ggn+m.FIXED_LAMBDA*torch.eye(25), rhs)
        self.assertTrue(torch.allclose(x, expected, rtol=1e-4, atol=1e-5))
        self.assertEqual(summary['lambda'], m.FIXED_LAMBDA)

    def test_nonpositive_pAp_and_nonfinite_action_fail(self):
        layer = torch.nn.Linear(2, 1, bias=False)
        rhs = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}
        frozen = [{'iteration': index} for index in range(1, 16)]

        def negative(vector, _iteration):
            layer.weight.grad = -100*vector['linear.weight']
            return {}

        with self.assertRaisesRegex(ValueError, r'p\^T A p'):
            m.fixed_20_step_cg(rhs, [('linear', layer)], negative, SPEC, frozen)
        rhs = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}

        def nonfinite(vector, _iteration):
            layer.weight.grad = torch.full_like(vector['linear.weight'], float('nan'))
            return {}

        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            m.fixed_20_step_cg(rhs, [('linear', layer)], nonfinite, SPEC, frozen)

    def test_matrixwise_FP64_scalar_reductions(self):
        values = torch.tensor([1e8, 1., -1e8, 1.], dtype=torch.float32)
        self.assertEqual(m.a106.matrixwise_dot({'w': values},
                                               {'w': torch.ones_like(values)}), 2.)
        self.assertIn('matrixwise_dot', inspect.getsource(m.a107.fixed_lambda_cg))
        self.assertNotIn('.double()', inspect.getsource(m.a107.fixed_lambda_cg))

    def test_matched_target_hash_is_unchanged(self):
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

    def test_strict_block13_JVP_is_unchanged(self):
        model = TinyReadout()
        spec = copy.deepcopy(SPEC)
        spec['readout']['hidden_size'] = 2
        tokens = torch.tensor([[0, 1]*64], dtype=torch.int64)
        direction = {'model.layers.0.weight': torch.tensor([[0.1, 0.2], [0.3, 0.4]])}
        primal, tangent = m.a.run_readout_jvp(model, tokens, direction, spec, torch)
        ordinary = m.a.ordinary_hook_readout(model, tokens, spec, torch)
        self.assertEqual(m.a.compare_primal(primal, ordinary, spec, torch), 0)
        self.assertTrue(bool(torch.isfinite(tangent).all()))

    def test_positionwise_and_parameter_diagnostics(self):
        response = torch.ones((128, 2048), dtype=torch.float32)
        target = torch.ones_like(response)
        target[0].neg_()
        target[2, :1024] = -1
        metrics = m.parent.response_metrics(response, target)
        self.assertAlmostEqual(metrics['position_0_cosine'], -1.)
        self.assertAlmostEqual(metrics['positions_1_4_mean_cosine'], 0.75)
        self.assertAlmostEqual(metrics['positions_1_127_mean_cosine'], 126/127)
        self.assertEqual(len(metrics['position_cosines']), 128)
        model = Toy98Model()
        vectors = {name+'.weight': torch.ones_like(module.weight)
                   for name, module in selected(model)}
        delta = {name: 2*value for name, value in vectors.items()}
        geometry = m.a106.parameter_diagnostics(vectors, delta)
        self.assertAlmostEqual(geometry['global']['cosine'], 1)
        self.assertAlmostEqual(geometry['global']['candidate_to_delta_norm_ratio'], 0.5)
        self.assertEqual(len(geometry['blocks']), 14)
        self.assertEqual(geometry['attention']['matrix_count'], 56)
        self.assertEqual(geometry['mlp']['matrix_count'], 42)

    def test_frozen_comparisons_and_approximate_gap(self):
        self.assertEqual(SPEC['baselines']['attempt109_primary'], 0.8802197432881782)
        self.assertEqual(SPEC['baselines']['attempt109_secondary'], 0.8858106177907814)
        self.assertEqual(SPEC['baselines']['attempt108_primary'], 0.8539805172947554)
        self.assertEqual(SPEC['baselines']['attempt016_primary_context_approx'], 0.9836)
        self.assertGreater(0.9836-SPEC['baselines']['attempt109_primary'], 0)

    def test_interpretation_order_and_next_step_policy(self):
        self.assertEqual(m.interpretation(0.91, 0.2, SPEC),
                         'clear_lower_damping_convergence_effect')
        self.assertEqual(m.interpretation(0.895, 0.2, SPEC),
                         'moderate_lower_damping_convergence_effect')
        self.assertEqual(m.interpretation(0.89, 0.021, SPEC),
                         'still_materially_underconverged')
        self.assertEqual(m.interpretation(0.89, 0.02, SPEC),
                         'little_evidence_15_step_truncation_dominates_at_lower_damping')
        self.assertFalse(SPEC['next_step_policy']['execute_next_attempt'])
        for category in SPEC['interpretation']['precedence']:
            self.assertTrue(m.next_step_decision(category, SPEC))

    def test_no_oracle_adl_or_adapter_access(self):
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

    def test_progress_and_runtime_accounting(self):
        rows = []
        matrix = torch.diag(torch.linspace(1., 25., 25))
        rhs = torch.ones(25)
        _, prior, _, _ = toy_cg(matrix, rhs, 15)
        _, summary, calls, _ = toy_cg(matrix, rhs, 20,
                                      frozen_rows=prior['iterations'], report=rows.append)
        self.assertEqual(rows, summary['iterations'])
        self.assertEqual(len(calls), 20)
        self.assertTrue(all(row['operator_seconds'] >= 0 and
                            row['operator_eta_seconds'] >= 0 and
                            math.isfinite(row['relative_residual']) for row in rows))
        self.assertTrue(all(not isinstance(value, torch.Tensor)
                            for row in rows for value in row.values()))
        self.assertEqual(SPEC['workload']['exact_rhs_ggn_applications'], 1)
        self.assertEqual(SPEC['workload']['cg_ggn_applications'], 20)
        self.assertEqual(SPEC['workload']['additional_operator_applications_vs_109'], 5)
        self.assertEqual(SPEC['workload']['expected_wall_minutes_context'], [10, 11])


if __name__ == '__main__':
    unittest.main()
