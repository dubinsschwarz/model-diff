"""Synthetic/provenance tests for Attempt111; never runs the real models."""
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tests.test_diagnose_privileged_endpoint_ggn_forward import Toy98Model, selected
from tests.test_diagnose_privileged_inverse_ggn_cg import TinyReadout


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/diagnose_privileged_low_damping_long_inverse_ggn.py'
loader = importlib.util.spec_from_file_location('attempt111_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


def midpoint_record(spec=SPEC):
    return {'candidate': 'x_40_descriptive_only',
            'activation': {'position_cosines': [0.7]*128,
                           'positions_1_4_mean_cosine': 0.7,
                           'positions_1_127_mean_cosine': 0.7},
            'parameter_diagnostics': {'global': {'cosine': 0.1}},
            'target_raw_sha256': spec['target']['raw_float32_sha256'],
            'response_raw_sha256': 'a'*64, 'jvp_audit': 0.}


def checkpoint_fixture(spec=SPEC):
    inventory = [{'name': 'linear.weight', 'shape': [1, 3]}]
    x = {'linear.weight': torch.tensor([[1., 2., 3.]], dtype=torch.float32)}
    r = {'linear.weight': torch.tensor([[2., 3., 4.]], dtype=torch.float32)}
    p = {'linear.weight': torch.tensor([[3., 4., 5.]], dtype=torch.float32)}
    rows = [{'iteration': index, 'relative_residual': 0.5,
             'r_norm': 1., 'pAp': 2., 'alpha': 0.1, 'beta': 0.2,
             'operator_seconds': 1., 'cumulative_elapsed_seconds': float(index),
             'operator_eta_seconds': float(80-index)} for index in range(1, 41)]
    rr = m.a106.matrixwise_dot(r, r)
    state = {'iteration': 40, 'x': x, 'r': r, 'p': p, 'rr': rr,
             'b_norm': 5., 'iterations': rows}
    return state, inventory, midpoint_record(spec)


def toy_solver(*, start_state=None, stop_at_midpoint=False, report=None,
               on_midpoint=None, calls=None):
    n = 100
    diag = torch.logspace(-2, 3, n)
    layer = torch.nn.Linear(n, 1, bias=False)
    eligible = [('linear', layer)]
    if calls is None:
        calls = []
    if start_state is None:
        rhs = torch.ones((1, n), dtype=torch.float32)
        rr = float(torch.sum(rhs*rhs, dtype=torch.float64))
        state = {'iteration': 0,
                 'x': {'linear.weight': torch.zeros_like(rhs)},
                 'r': {'linear.weight': rhs.clone()},
                 'p': {'linear.weight': rhs.clone()},
                 'rr': rr, 'b_norm': math.sqrt(rr), 'iterations': []}
    else:
        state = start_state

    def apply(vector, iteration):
        calls.append(iteration)
        layer.weight.grad = vector['linear.weight']*diag

    def midpoint(current):
        if on_midpoint is not None:
            on_midpoint(current)
        if stop_at_midpoint:
            raise RuntimeError('intentional stop at 40')

    return m.run_fixed_80_step_cg(
        state, eligible, apply, SPEC,
        on_midpoint=midpoint if start_state is None else None, report=report)


class Attempt111Tests(unittest.TestCase):
    def test_attempt110_commit_spec_source_result_linkage(self):
        link = SPEC['attempt110']
        for key in ('spec', 'source', 'result'):
            self.assertEqual(hashlib.sha256((PROJECT/link[key+'_path']).read_bytes()).hexdigest(),
                             link[key+'_sha256'])
        self.assertEqual(link['commit_sha'], m.COMMIT110)
        self.assertEqual(link['spec_sha256'], m.a110.SPEC_SHA256)
        self.assertEqual(link['source_sha256'], m.SOURCE110_SHA256)
        m.verify_committed_attempt110(link)

    def test_attempt110_frozen_result_values(self):
        link = SPEC['attempt110']
        result = m.a.load_json_object(PROJECT/link['result_path'], 'Attempt110 result')
        self.assertEqual(result['primary_x80']['candidate'] if 'primary_x80' in result
                         else result['solver']['candidate'], 'x_20')
        self.assertEqual(result['matched_activation']['positions_1_4_mean_cosine'],
                         0.8976964227968782)
        self.assertEqual(result['matched_activation']['positions_1_127_mean_cosine'],
                         0.9024841265861506)
        self.assertEqual(result['solver']['relative_linear_residual'],
                         0.02529230092308116)
        self.assertEqual(result['solver']['rhs_norm'], 14.651196958318208)
        self.assertEqual(result['solver']['lambda'], 1.3361159773737232)
        self.assertEqual(result['solver']['operator_applications'], 20)
        self.assertEqual(result['parameter_diagnostics']['global']['cosine'],
                         0.055268426963755675)
        self.assertEqual(result['parameter_diagnostics']['global']
                         ['candidate_to_delta_norm_ratio'], 0.034481559169662335)
        self.assertEqual(result['interpretation']['category'],
                         'moderate_lower_damping_convergence_effect')
        self.assertTrue(result['true_delta_rhs'])
        self.assertTrue(result['inverse_solve'])
        self.assertFalse(result['empirical_fisher'])

    def test_attempt110_hash_mismatch_fails_before_model_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            spec['paths']['checkpoint_path'] = str(Path(directory)/'checkpoint.pt')
            spec['attempt110']['result_sha256'] = '0'*64
            with patch.object(m.a110, 'validate_inputs',
                              side_effect=AssertionError('model preflight reached')):
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    m.validate_inputs(spec, resume=False)

    def test_non_damping_scientific_inventory_unchanged(self):
        old = m.a110.load_spec()
        for key in ('attempt104', 'attempt016', 'attempt106', 'attempt107',
                    'attempt108', 'attempt109', 'base', 'final_checkpoint_files',
                    'model', 'readout', 'eligible_tensors', 'corpus', 'pilot',
                    'probe', 'probe_rows', 'displacement', 'ggn',
                    'ggn_numerics_106', 'jvp', 'metric', 'target', 'exact_rhs'):
            self.assertEqual(SPEC[key], old[key], key)
        self.assertFalse(SPEC['reproduction_audit']['compare_attempt110_CG_trajectory'])
        self.assertFalse(SPEC['reproduction_audit']
                         ['activation_metrics_used_before_candidate_freeze'])

    def test_exact_rhs_and_Delta_semantics(self):
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
        self.assertAlmostEqual(geometry['cosine_with_true_delta'], 1)
        self.assertEqual(SPEC['displacement']['definition'],
                         m.a110.load_spec()['displacement']['definition'])

    def test_true_Delta_positive_sign_and_upstream_audit(self):
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
                         m.a110.load_spec()['ggn_numerics_106'])
        self.assertFalse(SPEC['ggn']['empirical_fisher'])
        logits = torch.tensor([[[0.2, -0.1, 0.6]]], dtype=torch.float32)
        tangent = torch.tensor([[[0.3, 1.2, -0.7]]], dtype=torch.float32)
        actual, _ = m.a106.stable_categorical_fisher_vector(logits, tangent, SPEC)
        p = torch.softmax(logits.double(), -1)[0, 0]
        expected = (torch.diag(p)-torch.outer(p, p)) @ tangent.double()[0, 0]/8128
        self.assertTrue(torch.equal(actual[0, 0], expected.float()))

    def test_frozen_construction_and_probe_inventory(self):
        corpus = torch.arange(512, dtype=torch.int64)[:, None].repeat(1, 128)
        probe = torch.arange(1024, dtype=torch.int64)[:, None].repeat(1, 128)
        with patch.object(m.parent, 'load_frozen_tokens', return_value=(corpus, probe)):
            construction, matched = m.a106.load_frozen_inputs(SPEC)
        self.assertEqual(tuple(construction.shape), (64, 128))
        self.assertEqual(construction[-1, 0].item(), 63)
        self.assertEqual(tuple(matched.shape), (1024, 128))
        self.assertEqual(matched[-1, 0].item(), 1023)
        self.assertEqual(SPEC['pilot']['prediction_contexts'], 8128)

    def test_fixed_lambda_and_ratios_without_rayleigh_update(self):
        self.assertEqual(SPEC['solver']['fixed_lambda'], 0.13361159773737232)
        self.assertIn('"fixed_lambda": 0.13361159773737232',
                      (PROJECT/'experiments/attempts'/m.ATTEMPT/'spec.json').read_text())
        self.assertEqual(m.FIXED_LAMBDA_DECIMAL, '0.13361159773737232')
        self.assertEqual(SPEC['solver']['lambda_relative_to_attempt110'], 0.1)
        self.assertEqual(SPEC['solver']['lambda_relative_to_attempt108'], 0.01)
        self.assertFalse(SPEC['solver']['recompute_rho_from_exact_rhs'])
        self.assertNotIn('rho_b', inspect.getsource(m.run_fixed_80_step_cg))

    def test_exactly_80_iterations_and_x80_primary(self):
        state, solver = toy_solver()
        self.assertEqual(state['iteration'], 80)
        self.assertEqual(len(solver['iterations']), 80)
        self.assertEqual(solver['operator_applications'], 80)
        self.assertEqual(solver['candidate'], 'x_80')
        self.assertEqual(solver['midpoint_candidate'], 'x_40_descriptive_only')
        self.assertTrue(solver['no_early_stop'])
        self.assertEqual(set(solver['residual_checkpoints']), {'20', '40', '60', '80'})

    def test_x40_diagnostic_cannot_change_solver_settings(self):
        self.assertFalse(SPEC['solver']['candidate_selection'])
        self.assertEqual(SPEC['solver']['midpoint_iteration'], 40)
        self.assertEqual(SPEC['solver']['iterations'], 80)
        self.assertFalse(SPEC['solver']['residual_early_stop'])
        self.assertIsNone(SPEC['solver']['preconditioner'])
        bad = copy.deepcopy(SPEC)
        bad['solver']['iterations'] = 40
        with self.assertRaisesRegex(ValueError, 'settings mismatch'):
            m.run_fixed_80_step_cg({}, [], lambda *_: None, bad, on_midpoint=lambda *_: None)

    def test_CG_recurrence_and_matrixwise_FP64(self):
        values = torch.tensor([1e8, 1., -1e8, 1.], dtype=torch.float32)
        self.assertEqual(m.a106.matrixwise_dot({'w': values},
                                               {'w': torch.ones_like(values)}), 2.)
        self.assertIn('matrixwise_dot', inspect.getsource(m.run_fixed_80_step_cg))
        rows = []
        state, _ = toy_solver(report=rows.append)
        self.assertEqual(len(rows), 80)
        self.assertEqual(rows[0]['iteration'], 1)
        self.assertGreater(rows[0]['pAp'], 0)
        self.assertTrue(all(math.isfinite(row['alpha']) and row['alpha'] > 0
                            for row in rows))
        self.assertEqual(state['iteration'], 80)

    def test_first_step_alpha_toy_SPD(self):
        rhs = torch.ones(100)
        diag = torch.logspace(-2, 3, 100)
        _, solver = toy_solver()
        expected = float(rhs@rhs)/float(rhs@((diag+m.FIXED_LAMBDA)*rhs))
        self.assertAlmostEqual(solver['iterations'][0]['alpha'], expected, places=6)

    def test_midpoint_evaluation_preserves_x_r_p(self):
        state, inventory, record = checkpoint_fixture()
        final = torch.nn.Linear(1, 1)
        before = m.state_hashes(state, inventory)
        with (patch.object(m, 'evaluate_candidate', return_value=record),
              patch.object(m.a, 'verify_model_unchanged')):
            actual, hashes = m.evaluate_midpoint_preserving_state(
                state, inventory, final, {}, None, None, None, SPEC)
        self.assertEqual(actual, record)
        self.assertEqual(hashes, before)
        self.assertEqual(m.state_hashes(state, inventory), before)

    def test_midpoint_mutation_is_detected(self):
        state, inventory, record = checkpoint_fixture()
        final = torch.nn.Linear(1, 1)

        def mutate(*_args, **_kwargs):
            state['p']['linear.weight'].add_(1.)
            return record

        with (patch.object(m, 'evaluate_candidate', side_effect=mutate),
              patch.object(m.a, 'verify_model_unchanged')):
            with self.assertRaisesRegex(ValueError, 'changed CG x/r/p'):
                m.evaluate_midpoint_preserving_state(
                    state, inventory, final, {}, None, None, None, SPEC)

    def test_checkpoint_path_outside_repo_and_estimated_size(self):
        path = Path(SPEC['paths']['checkpoint_path'])
        self.assertTrue(path.is_absolute())
        self.assertFalse(path.is_relative_to(PROJECT))
        self.assertEqual(SPEC['checkpoint']['expected_approximate_size_gib'], 7.9)
        self.assertFalse(SPEC['checkpoint']['model_weights_in_checkpoint'])

    def test_checkpoint_contains_only_resume_state_and_metrics(self):
        state, inventory, record = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, record, inventory, SPEC)
        self.assertEqual(set(payload), {'format_version', 'iteration', 'x', 'r', 'p',
                                        'rr', 'b_norm', 'fixed_lambda',
                                        'coordinate_inventory', 'iteration_records',
                                        'midpoint', 'midpoint_json_sha256',
                                        'raw_hashes', 'provenance'})
        self.assertEqual(payload['iteration'], 40)
        self.assertEqual(payload['midpoint'], record)
        self.assertEqual(len(payload['iteration_records']), 40)
        self.assertFalse(any('model' in key for key in payload))

    def test_checkpoint_raw_x_r_p_hashes_and_tamper_rejection(self):
        state, inventory, record = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, record, inventory, SPEC)
        self.assertEqual(payload['raw_hashes'], m.state_hashes(state, inventory))
        m.validate_checkpoint(payload, SPEC, inventory, expected_rhs_norm=5.)
        payload['r']['linear.weight'][0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'raw tensor hash'):
            m.validate_checkpoint(payload, SPEC, inventory, expected_rhs_norm=5.)

    def test_checkpoint_provenance_mismatch_rejected(self):
        state, inventory, record = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, record, inventory, SPEC)
        payload['provenance']['source_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'metadata/provenance'):
            m.validate_checkpoint(payload, SPEC, inventory, expected_rhs_norm=5.)

    def test_checkpoint_dtype_and_shape_rejected(self):
        state, inventory, record = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, record, inventory, SPEC)
        payload['p']['linear.weight'] = payload['p']['linear.weight'].double()
        with self.assertRaisesRegex(ValueError, 'dtype/shape'):
            m.validate_checkpoint(payload, SPEC, inventory, expected_rhs_norm=5.)

    def test_checkpoint_preparation_preserves_original_state(self):
        state, inventory, record = checkpoint_fixture()
        before = m.state_hashes(state, inventory)
        payload = m.make_checkpoint_payload(state, record, inventory, SPEC)
        self.assertEqual(m.state_hashes(state, inventory), before)
        self.assertEqual(payload['raw_hashes'], before)
        self.assertIsNot(payload['x']['linear.weight'], state['x']['linear.weight'])

    def test_checkpoint_atomic_publication_and_safe_load(self):
        state, inventory, record = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, record, inventory, SPEC)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            m.atomic_checkpoint_publish(path, payload)
            self.assertTrue(path.is_file())
            self.assertFalse(list(Path(directory).glob('*.tmp-*')))
            loaded = m.load_checkpoint(path)
            m.validate_checkpoint(loaded, SPEC, inventory, expected_rhs_norm=5.)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.atomic_checkpoint_publish(path, payload)

    def test_resume_rejects_incompatible_checkpoint(self):
        state, inventory, record = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, record, inventory, SPEC)
        incompatible = copy.deepcopy(payload)
        incompatible['fixed_lambda'] = 1.3361159773737232
        with self.assertRaisesRegex(ValueError, 'metadata/provenance'):
            m.validate_checkpoint(incompatible, SPEC, inventory, 5.)
        incompatible = copy.deepcopy(payload)
        incompatible['coordinate_inventory'][0]['shape'] = [1, 4]
        with self.assertRaisesRegex(ValueError, 'metadata/provenance'):
            m.validate_checkpoint(incompatible, SPEC, inventory, 5.)

    def test_resume_restores_iteration40_and_starts41(self):
        state, inventory, record = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, record, inventory, SPEC)
        layer = torch.nn.Linear(3, 1, bias=False)
        restored = m.restore_cg_state(payload, [('linear', layer)], inventory)
        self.assertEqual(restored['iteration'], 40)
        self.assertEqual(m.state_hashes(restored, inventory), payload['raw_hashes'])
        self.assertEqual(payload['midpoint'], record)
        calls = []

        def stop_first(_vector, iteration):
            calls.append(iteration)
            raise RuntimeError('stop after checking resume start')

        with self.assertRaisesRegex(RuntimeError, 'resume start'):
            m.run_fixed_80_step_cg(restored, [('linear', layer)], stop_first, SPEC)
        self.assertEqual(calls, [41])

    def test_fresh_run_refuses_stale_checkpoint_and_resume_requires_one(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            path = Path(directory)/'checkpoint.pt'
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            spec['paths']['checkpoint_path'] = str(path)
            with self.assertRaisesRegex(ValueError, 'requires valid'):
                m.validate_inputs(spec, resume=True)
            path.write_bytes(b'stale')
            with self.assertRaisesRegex(ValueError, 'Stale'):
                m.validate_inputs(spec, resume=False)

    def test_resume_keeps_frozen_x40_metric_without_re_evaluation(self):
        state, inventory, record = checkpoint_fixture()
        payload = m.make_checkpoint_payload(state, record, inventory, SPEC)
        self.assertEqual(m.validate_checkpoint(payload, SPEC, inventory, 5.), record)
        self.assertEqual(payload['midpoint_json_sha256'], m.json_hash(record))
        payload['midpoint']['activation']['positions_1_4_mean_cosine'] = 0.8
        with self.assertRaisesRegex(ValueError, 'metric hash'):
            m.validate_checkpoint(payload, SPEC, inventory, 5.)

    def test_normal_run_continues_from_in_memory_state_after_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            identities = []

            def midpoint(state):
                identities.append(id(state['x']['linear.weight']))
                path.write_bytes(b'published checkpoint sentinel')

            with patch.object(m, 'load_checkpoint', side_effect=AssertionError('reloaded')):
                state, solver = toy_solver(on_midpoint=midpoint)
            self.assertEqual(state['iteration'], 80)
            self.assertEqual(id(state['x']['linear.weight']), identities[0])
            self.assertEqual(solver['operator_applications'], 80)
            self.assertTrue(path.exists())

    def test_midpoint_callback_result_cannot_select_or_stop_iterate(self):
        seen = []

        def midpoint(state):
            seen.append(state['iteration'])
            return {'oracle_cosine': 1.0, 'select_x40': True}

        state, solver = toy_solver(on_midpoint=midpoint)
        self.assertEqual(seen, [40])
        self.assertEqual(state['iteration'], 80)
        self.assertEqual(solver['candidate'], 'x_80')

    def test_nonmonotone_residual_does_not_stop_CG(self):
        state, solver = toy_solver()
        residuals = [row['relative_residual'] for row in solver['iterations']]
        self.assertTrue(any(later > earlier for earlier, later in
                            zip(residuals, residuals[1:])))
        self.assertEqual(state['iteration'], 80)

    def test_failure_after_iteration40_leaves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            layer = torch.nn.Linear(100, 1, bias=False)
            rhs = torch.ones((1, 100), dtype=torch.float32)
            rr = float(torch.sum(rhs*rhs, dtype=torch.float64))
            state = {'iteration': 0, 'x': {'linear.weight': torch.zeros_like(rhs)},
                     'r': {'linear.weight': rhs.clone()},
                     'p': {'linear.weight': rhs.clone()},
                     'rr': rr, 'b_norm': math.sqrt(rr), 'iterations': []}
            diag = torch.logspace(-2, 3, 100)

            def apply(vector, iteration):
                if iteration == 41:
                    raise RuntimeError('late failure')
                layer.weight.grad = vector['linear.weight']*diag

            def checkpoint(_state):
                path.write_bytes(b'checkpoint remains')

            with self.assertRaisesRegex(RuntimeError, 'late failure'):
                m.run_fixed_80_step_cg(state, [('linear', layer)], apply, SPEC,
                                       on_midpoint=checkpoint)
            self.assertTrue(path.exists())
            self.assertEqual(state['iteration'], 40)

    def test_fresh_solver_requires_midpoint_callback(self):
        rhs = torch.ones((1, 2), dtype=torch.float32)
        state = {'iteration': 0, 'x': {'linear.weight': torch.zeros_like(rhs)},
                 'r': {'linear.weight': rhs.clone()},
                 'p': {'linear.weight': rhs.clone()},
                 'rr': 2., 'b_norm': math.sqrt(2.), 'iterations': []}
        layer = torch.nn.Linear(2, 1, bias=False)
        with self.assertRaisesRegex(ValueError, 'requires frozen midpoint'):
            m.run_fixed_80_step_cg(state, [('linear', layer)], lambda *_: None, SPEC)

    def test_successful_result_deletes_checkpoint_and_no_early_result(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            checkpoint = Path(directory)/'checkpoint.pt'
            checkpoint.write_bytes(b'synthetic checkpoint')
            result = {'solver': {'operator_applications': 79, 'iterations': [{}]*79},
                      'primary_x80': {'candidate': 'x_80_primary'},
                      'midpoint_x40': {'candidate': 'x_40_descriptive_only'}}
            with self.assertRaisesRegex(ValueError, 'before x_80'):
                m.publish_result_and_cleanup(output, checkpoint, result)
            self.assertFalse(output.exists())
            self.assertTrue(checkpoint.exists())
            result['solver'] = {'operator_applications': 80, 'iterations': [{}]*80}
            m.publish_result_and_cleanup(output, checkpoint, result)
            self.assertTrue(output.exists())
            self.assertFalse(checkpoint.exists())

    def test_failed_result_publication_leaves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            checkpoint = Path(directory)/'checkpoint.pt'
            checkpoint.write_bytes(b'checkpoint')
            result = {'solver': {'operator_applications': 80, 'iterations': [{}]*80},
                      'primary_x80': {'candidate': 'x_80_primary'},
                      'midpoint_x40': {'candidate': 'x_40_descriptive_only'}}
            with patch.object(m.a, 'write_manifest', side_effect=RuntimeError('disk failure')):
                with self.assertRaisesRegex(RuntimeError, 'disk failure'):
                    m.publish_result_and_cleanup(output, checkpoint, result)
            self.assertTrue(checkpoint.exists())

    def test_no_result_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            output.write_text('frozen')
            checkpoint = Path(directory)/'checkpoint.pt'
            checkpoint.write_bytes(b'checkpoint')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.publish_result_and_cleanup(output, checkpoint, {})
            self.assertEqual(output.read_text(), 'frozen')
            self.assertTrue(checkpoint.exists())

    def test_matched_target_hash_and_strict_block13_JVP(self):
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
        model = TinyReadout()
        spec['readout']['hidden_size'] = 2
        tokens = torch.tensor([[0, 1]*64], dtype=torch.int64)
        direction = {'model.layers.0.weight': torch.tensor([[0.1, 0.2], [0.3, 0.4]])}
        primal, tangent = m.a.run_readout_jvp(model, tokens, direction, spec, torch)
        ordinary = m.a.ordinary_hook_readout(model, tokens, spec, torch)
        self.assertEqual(m.a.compare_primal(primal, ordinary, spec, torch), 0)
        self.assertTrue(bool(torch.isfinite(tangent).all()))

    def test_positionwise_activation_and_parameter_diagnostics(self):
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

    def test_interpretation_order(self):
        self.assertEqual(m.interpretation(0.95, 0.90, 0.02, SPEC),
                         'near_ceiling_low_damping_recovery')
        self.assertEqual(m.interpretation(0.93, 0.92, 0.2, SPEC),
                         'strong_low_damping_recovery')
        self.assertEqual(m.interpretation(0.92, 0.90, 0.021, SPEC),
                         'still_solver_limited')
        self.assertEqual(m.interpretation(0.92, 0.915, 0.01, SPEC),
                         'low_damping_functional_plateau')
        self.assertEqual(m.interpretation(0.90, 0.895, 0.015, SPEC),
                         'mixed_low_damping_result')
        self.assertFalse(SPEC['interpretation']['fundamental_nullspace_claim'])

    def test_approximate_gap_accounting_and_next_step_record_only(self):
        baseline = SPEC['baselines']['attempt110_primary']
        ceiling = SPEC['baselines']['attempt016_primary_context_approx']
        self.assertEqual(baseline, 0.8976964227968782)
        self.assertEqual(ceiling, 0.9836)
        self.assertAlmostEqual((0.95-baseline)/(ceiling-baseline),
                               (0.95-0.8976964227968782)/
                               (0.9836-0.8976964227968782))
        self.assertFalse(SPEC['next_step_policy']['execute_next_attempt'])

    def test_no_oracle_adl_or_adapter_and_no_large_success_artifact(self):
        source = SOURCE.read_text()
        self.assertNotIn('oracle_adl.pt', source)
        self.assertNotIn('adapter_checkpoint', source)
        self.assertFalse(SPEC['information_policy']['oracle_adl_access'])
        self.assertFalse(SPEC['information_policy']['adapter_access'])
        self.assertTrue(SPEC['output_policy']['result_json_only'])
        self.assertTrue(SPEC['checkpoint']['delete_after_successful_result_publication'])
        self.assertFalse(SPEC['information_policy']['candidate_selection'])
        self.assertFalse(SPEC['information_policy']['oracle_based_tuning'])

    def test_progress_and_ETA_records(self):
        rows = []
        _, solver = toy_solver(report=rows.append)
        self.assertEqual(rows, solver['iterations'])
        self.assertEqual(len(rows), 80)
        self.assertTrue(all(row['operator_seconds'] >= 0 and
                            row['operator_eta_seconds'] >= 0 and
                            math.isfinite(row['relative_residual']) for row in rows))
        self.assertTrue(all(not isinstance(value, torch.Tensor)
                            for row in rows for value in row.values()))
        self.assertEqual(SPEC['workload']['exact_rhs_ggn_applications'], 1)
        self.assertEqual(SPEC['workload']['cg_ggn_applications'], 80)
        self.assertEqual(SPEC['workload']['total_jvp_batches'], 64)
        self.assertEqual(SPEC['workload']['accepted_practical_upper_minutes'], 45)

    def test_workload_inventory_is_attempt111_only(self):
        expected = {
            'base_probe_batches': 32,
            'exact_rhs_ggn_applications': 1,
            'cg_ggn_applications': 80,
            'ggn_microbatches_per_application': 64,
            'midpoint_jvp_batches': 32,
            'final_jvp_batches': 32,
            'total_jvp_batches': 64,
            'expected_wall_minutes_context': [35, 40],
            'accepted_practical_upper_minutes': 45,
            'no_separate_smoke': True,
            'attempt110_wall_seconds_context': 675,
        }
        self.assertEqual(SPEC['workload'], expected)
        stale = {
            'additional_operator_applications_vs_107',
            'attempt107_operator_seconds_per_application_context',
            'additional_operator_seconds_context',
            'attempt108_wall_minutes_context',
            'attempt109_wall_seconds_context',
            'additional_operator_applications_vs_109',
        }
        self.assertTrue(stale.isdisjoint(SPEC['workload']))
        frozen_attempt110 = m.a110.load_spec()
        for name in sorted(stale):
            with self.subTest(stale_field=name):
                altered = copy.deepcopy(SPEC)
                altered['workload'][name] = 5
                with patch.object(m.parent, 'require_hash'), \
                     patch.object(m.a, 'load_json_object', return_value=altered), \
                     patch.object(m.a110, 'load_spec', return_value=frozen_attempt110):
                    with self.assertRaisesRegex(ValueError,
                                                'frozen scientific inventory mismatch'):
                        m.load_spec()


if __name__ == '__main__':
    unittest.main()
