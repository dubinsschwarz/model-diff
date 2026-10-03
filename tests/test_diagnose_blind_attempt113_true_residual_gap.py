"""Synthetic blind tests; no scratch checkpoint or real model is loaded."""
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/diagnose_blind_attempt113_true_residual_gap.py'
loader = importlib.util.spec_from_file_location('attempt115_synthetic_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


def toy_vectors():
    inventory = [{'name': 'a.weight', 'shape': [1, 2]},
                 {'name': 'b.weight', 'shape': [1, 1]}]
    b = {'a.weight': torch.tensor([[1., 2.]], dtype=torch.float32),
         'b.weight': torch.tensor([[3.]], dtype=torch.float32)}
    x = {'a.weight': torch.tensor([[1., -1.]], dtype=torch.float32),
         'b.weight': torch.tensor([[2.]], dtype=torch.float32)}
    g = {'a.weight': torch.tensor([[.3, .4]], dtype=torch.float32),
         'b.weight': torch.tensor([[.5]], dtype=torch.float32)}
    # True residual under lambda=0.1 is approximately [0.6,1.7,2.3].
    r = {'a.weight': torch.tensor([[.5, 1.8]], dtype=torch.float32),
         'b.weight': torch.tensor([[2.2]], dtype=torch.float32)}
    return inventory, b, x, r, g


def rhs_fixture(m113):
    inventory = [{'name': 'a.weight', 'shape': [1, 2]}]
    rhs = {'a.weight': torch.tensor([[.6, 0.]], dtype=torch.float32)}
    hashes, aggregate = m113.ordered_vector_hashes(rhs, inventory)
    record = {'norm': math.sqrt(m113.matrixwise_dot(rhs, rhs)),
              'matrix_raw_sha256': hashes, 'ordered_matrix_hash_sha256': aggregate,
              'hash_aggregation': 'SHA256_of_ordered_name_NUL_digest_newline'}
    diagnostics = [{'name': name, 'mean_generic_loss': 1.,
                    'global_gradient_norm': 5., 'unit_normalization_scalar': .2,
                    'rows': [0, 512], 'prediction_contexts': 65024}
                   for name in m.ORDER]
    checkpoint = {'coordinate_inventory': inventory, 'rhs_record': copy.deepcopy(record),
                  'gradient_records': copy.deepcopy(diagnostics), 'b_norm': record['norm']}
    return rhs, record, diagnostics, checkpoint


class Attempt115Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m113, cls.m114, cls.prior, cls.a114_result = m.validate_linkage(SPEC)

    def test_attempt113_frozen_commit_and_hashes(self):
        self.assertEqual(SPEC['attempt113']['commit'], m.FROZEN['113_commit'])
        self.assertEqual(m.sha256_file(m.PRIOR113_SPEC), m.FROZEN['113_spec_sha256'])
        self.assertEqual(m.sha256_file(m.PRIOR113_SOURCE), m.FROZEN['113_source_sha256'])

    def test_attempt114_frozen_commit_hashes_and_result(self):
        self.assertEqual(SPEC['attempt114']['commit'], m.FROZEN['114_commit'])
        for path, key in ((m.PRIOR114_SPEC, '114_spec_sha256'),
                          (m.PRIOR114_SOURCE, '114_source_sha256'),
                          (m.PRIOR114_RESULT, '114_result_sha256')):
            self.assertEqual(m.sha256_file(path), m.FROZEN[key])
        self.assertEqual(self.a114_result['residual_curvature']['rho_r40'],
                         3018.261150714194)

    def test_attempt114_exact_blind_values(self):
        frozen = SPEC['attempt114']
        self.assertEqual(frozen['recursive_relative_residual_at_40'], 2.0165128716207255)
        self.assertEqual(frozen['rho_b'], 561.5508153958936)
        self.assertEqual(frozen['lambda'], 0.056155081539589355)
        self.assertEqual(frozen['ritz_condition_estimate'], 19640.51752854543)

    def test_linkage_rejects_changed_hash(self):
        with patch.object(m, 'sha256_file', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                m.validate_linkage(SPEC)

    def test_spec_rejects_changed_scientific_rule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'spec.json'
            altered = copy.deepcopy(SPEC)
            altered['attempt113']['ggn_applications'] = 2
            path.write_text(json.dumps(altered))
            with patch.object(m, 'SPEC_PATH', path):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    m.load_spec(path)

    def test_checkpoint_pinned_hash_size_iteration(self):
        self.assertEqual(SPEC['attempt113']['checkpoint_serialized_sha256'],
                         '4435097d563297d2d749d37a14d79d9ae44c70ddda1a0b90e6f87f00d4c64113')
        self.assertEqual(SPEC['attempt113']['checkpoint_size_bytes'], 8455847711)
        self.assertEqual(SPEC['attempt113']['iteration'], 40)

    def test_checkpoint_read_only_reuses_hashed_validator(self):
        source = inspect.getsource(m.validate_checkpoint)
        self.assertIn('m114.validate_checkpoint_read_only(', source)
        self.assertIn('sha256_file(CHECKPOINT)', source)
        self.assertIn("len(payload['iteration_records']) != 40", source)
        self.assertIn("payload['coordinate_inventory'] != attempt114_result['coordinate_inventory']", source)

    def test_checkpoint_x_r_p_hash_and_rr_semantics_in_reused_validator(self):
        source = inspect.getsource(self.m114.validate_checkpoint_read_only)
        self.assertIn('module.validate_checkpoint(', source)
        reused = inspect.getsource(self.m113.validate_checkpoint)
        self.assertIn("for key in ('x', 'r', 'p')", reused)
        self.assertIn('ordered_vector_hashes(vector, inventory)', reused)
        self.assertIn("matrixwise_dot(value['r'], value['r'])", reused)

    def test_checkpoint_releases_p_before_model(self):
        source = inspect.getsource(m.run)
        self.assertIn('del payload  # Release p40', source)
        self.assertLess(source.index('del payload'), source.index('load_local_model('))

    def test_exact_eight_corpus_order(self):
        self.assertEqual(tuple(SPEC['attempt113']['corpus_order']), m.ORDER)
        self.assertEqual(tuple(row['name'] for row in self.prior['corpora']), m.ORDER)

    def test_rhs_rows_batch_and_loss_semantics(self):
        self.assertEqual(self.prior['rhs']['rows'], [0, 512])
        self.assertEqual(self.prior['rhs']['batch_size'], 8)
        self.assertEqual(self.prior['rhs']['prediction_contexts_per_corpus'], 65024)
        source = inspect.getsource(self.m113.accumulate_corpus_gradient)
        self.assertIn('range(0, rows, 8)', source)
        self.assertIn('blind.causal_token_loss_sum(logits, batch, torch)', source)
        self.assertIn('(loss_sum/(rows*127)).backward()', source)

    def test_selected_98_only(self):
        self.assertEqual(self.prior['eligible_tensors']['expected_matrix_count'], 98)
        self.assertEqual(self.prior['eligible_tensors']['parameter'], 'weight')
        self.assertEqual(self.prior['eligible_tensors']['recursive_module_type'], 'torch.nn.Linear')
        self.assertIn('freeze_other_parameters(model, eligible)', inspect.getsource(m.run))

    def test_unit_gradient_and_equal_positive_consensus(self):
        layer = torch.nn.Linear(2, 1, bias=False)
        eligible = [('linear', layer)]
        inventory = [{'name': 'linear.weight', 'shape': [1, 2]}]
        consensus = {}
        for index in range(8):
            layer.weight.grad = torch.tensor([[3., 4. if index < 4 else -4.]])
            scale = self.m113.add_unit_gradient_to_consensus(
                consensus, eligible, 5., inventory,
                corpus_index=index, corpus_count=8)
            self.assertEqual(scale, .2)
        self.assertTrue(torch.allclose(consensus['linear.weight'],
                                       torch.tensor([[.6, 0.]]), atol=1e-7))
        self.assertAlmostEqual(math.sqrt(self.m113.matrixwise_dot(
            consensus, consensus)), .6, places=6)  # No final normalization.

    def test_build_rhs_calls_frozen_gradient_path_for_all_eight(self):
        layer = torch.nn.Linear(2, 1, bias=False)
        eligible = [('linear', layer)]
        inventory = [{'name': 'linear.weight', 'shape': [1, 2]}]
        corpora = {name: name for name in m.ORDER}
        called = []

        def fake_gradient(model, tokens, selected, *, rows, progress_label):
            called.append((tokens, rows, progress_label))
            layer.weight.grad = torch.tensor([[3., 4. if len(called) <= 4 else -4.]])
            return 2., 5.

        with patch.object(self.m113, 'accumulate_corpus_gradient', side_effect=fake_gradient):
            rhs, diagnostics, record = self.m113.build_blind_rhs(
                SimpleNamespace(zero_grad=lambda **kwargs: None),
                corpora, eligible, inventory)
        self.assertEqual([row[0] for row in called], list(m.ORDER))
        self.assertTrue(all(row[1] == 512 for row in called))
        self.assertEqual([row['rows'] for row in diagnostics], [[0, 512]]*8)
        self.assertTrue(torch.allclose(rhs['linear.weight'],
                                       torch.tensor([[.6, 0.]]), atol=1e-7))
        self.assertAlmostEqual(record['norm'], .6, places=6)

    def test_rhs_exact_raw_and_aggregate_hashes_pass(self):
        rhs, record, diagnostics, checkpoint = rhs_fixture(self.m113)
        audit = m.require_rebuilt_rhs(self.m113, rhs, record, diagnostics, checkpoint, SPEC)
        self.assertTrue(audit['byte_identical_to_checkpoint_rhs'])
        self.assertEqual(audit['ordered_matrix_hash_sha256'],
                         checkpoint['rhs_record']['ordered_matrix_hash_sha256'])

    def test_rhs_raw_matrix_hash_mismatch_fails(self):
        rhs, record, diagnostics, checkpoint = rhs_fixture(self.m113)
        rhs['a.weight'][0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'byte-identical'):
            m.require_rebuilt_rhs(self.m113, rhs, record, diagnostics, checkpoint, SPEC)

    def test_rhs_aggregate_hash_mismatch_fails(self):
        rhs, record, diagnostics, checkpoint = rhs_fixture(self.m113)
        checkpoint['rhs_record']['ordered_matrix_hash_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'byte-identical'):
            m.require_rebuilt_rhs(self.m113, rhs, record, diagnostics, checkpoint, SPEC)

    def test_rhs_per_corpus_diagnostic_mismatch_fails(self):
        rhs, record, diagnostics, checkpoint = rhs_fixture(self.m113)
        diagnostics[0]['global_gradient_norm'] = 4.
        with self.assertRaisesRegex(ValueError, 'per-corpus'):
            m.require_rebuilt_rhs(self.m113, rhs, record, diagnostics, checkpoint, SPEC)

    def test_corpora_loader_omits_unused_probe(self):
        source = inspect.getsource(m.load_eight_corpora_without_probe)
        self.assertNotIn('load_probe(', source)
        self.assertIn('for index, row in enumerate(prior[\'corpora\'])', source)
        self.assertIn('validate_frozen_corpus(', source)
        self.assertIn('validate_corpus(', source)

    def test_fineweb_G_artifact_and_rows(self):
        ggn = self.prior['ggn']
        self.assertEqual(ggn['rows'], [0, 64])
        self.assertEqual(ggn['denominator'], 8128)
        self.assertEqual(ggn['serialized_sha256'],
                         '54283d72aa3075e2867ead86e538991d7284ede327ca67871e73bfc581e4025d')
        self.assertEqual(ggn['raw_tensor_sha256'],
                         '4613982be77342a89b35616d68e21d9ff66d5a983af7e82cb6fa1eb159457eae')

    def test_stable_fp64_categorical_Fisher_formula(self):
        logits = torch.tensor([[[.3, -.2, .7]]], dtype=torch.float32)
        tangent = torch.tensor([[[1.2, -.3, .5]]], dtype=torch.float32)
        value, _ = self.m113.stable_categorical_fisher_vector(logits, tangent, self.prior)
        p = torch.softmax(logits.double(), dim=-1)[0, 0]
        explicit = (torch.diag(p)-torch.outer(p, p)) @ tangent.double()[0, 0] / 8128
        torch.testing.assert_close(value.double()[0, 0], explicit, rtol=1e-6, atol=1e-12)

    def test_exactly_one_G_application_and_no_CG_continuation(self):
        source = inspect.getsource(m.run)
        self.assertEqual(source.count('m113.ggn_action_stage('), 1)
        self.assertNotIn('run_fixed_80_cg(', source)
        self.assertNotIn('restore_cg_state(', source)
        self.assertIn("ggn_applications_to_x40': 1", source)

    def test_fixed_checkpoint_lambda_not_recomputed(self):
        source = inspect.getsource(m.run)
        self.assertIn("saved_rr, saved_b_norm, damping = payload['rr'], payload['b_norm'], payload['fixed_lambda']", source)
        self.assertNotIn('rayleigh_and_lambda(', source)
        self.assertEqual(SPEC['attempt114']['lambda'], .056155081539589355)

    def test_true_residual_fp64_formula_and_norms(self):
        inventory, b, x, r, g = toy_vectors()
        result = m.residual_metrics(b, x, r, g, .1, inventory, SPEC)
        bvec = torch.cat([b['a.weight'].reshape(-1), b['b.weight'].reshape(-1)]).double()
        xvec = torch.cat([x['a.weight'].reshape(-1), x['b.weight'].reshape(-1)]).double()
        gvec = torch.cat([g['a.weight'].reshape(-1), g['b.weight'].reshape(-1)]).double()
        rvec = torch.cat([r['a.weight'].reshape(-1), r['b.weight'].reshape(-1)]).double()
        tvec = bvec-gvec-.1*xvec
        self.assertAlmostEqual(result['true_relative_residual'],
                               float(torch.linalg.vector_norm(tvec)/torch.linalg.vector_norm(bvec)))
        self.assertAlmostEqual(result['recursive_relative_residual'],
                               float(torch.linalg.vector_norm(rvec)/torch.linalg.vector_norm(bvec)))
        self.assertEqual(result['arithmetic'], SPEC['numerics']['primary_residual_arithmetic'])

    def test_residual_gap_and_cosine(self):
        inventory, b, x, r, g = toy_vectors()
        result = m.residual_metrics(b, x, r, g, .1, inventory, SPEC)
        rvec = torch.cat([r['a.weight'].reshape(-1), r['b.weight'].reshape(-1)]).double()
        tvec = torch.tensor([.6, 1.7, 2.3], dtype=torch.float64)
        gap = rvec-tvec
        self.assertAlmostEqual(result['residual_gap_relative_to_recursive'],
                               float(torch.linalg.vector_norm(gap)/torch.linalg.vector_norm(rvec)), places=6)
        self.assertAlmostEqual(result['cosine_recursive_vs_true'],
                               float(torch.dot(rvec, tvec)/(torch.linalg.vector_norm(rvec)*
                                      torch.linalg.vector_norm(tvec))), places=6)

    def test_matrixwise_norm_rows(self):
        inventory, b, x, r, g = toy_vectors()
        result = m.residual_metrics(b, x, r, g, .1, inventory, SPEC)
        self.assertEqual([row['name'] for row in result['matrixwise']],
                         ['a.weight', 'b.weight'])
        for row in result['matrixwise']:
            self.assertGreater(row['recursive_residual_norm'], 0)
            self.assertGreater(row['true_residual_norm'], 0)
            self.assertGreaterEqual(row['gap_norm'], 0)

    def test_fp64_reductions_avoid_losing_small_matrix(self):
        inventory = [{'name': 'a', 'shape': [1, 1]}, {'name': 'b', 'shape': [1, 1]}]
        b = {'a': torch.tensor([[1e5]], dtype=torch.float32),
             'b': torch.tensor([[1.]], dtype=torch.float32)}
        zero = {key: torch.zeros_like(value) for key, value in b.items()}
        result = m.residual_metrics(b, zero, b, zero, 1., inventory, SPEC)
        self.assertEqual(result['squared_norms_and_dot']['b_sq'], 10000000001.0)

    def test_residual_support_mismatch_fails(self):
        inventory, b, x, r, g = toy_vectors()
        g.pop('b.weight')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            m.residual_metrics(b, x, r, g, .1, inventory, SPEC)

    def test_nonfinite_residual_input_fails(self):
        inventory, b, x, r, g = toy_vectors()
        g['b.weight'][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            m.residual_metrics(b, x, r, g, .1, inventory, SPEC)

    def _metrics(self, d, c, r=2., t=2.):
        return {'residual_gap_relative_to_recursive': d,
                'cosine_recursive_vs_true': c,
                'recursive_relative_residual': r, 'true_relative_residual': t}

    def test_interpretation_large_precedes_all(self):
        self.assertEqual(m.classify_residual_fidelity(self._metrics(.5, .995), SPEC),
                         'large_recursive_residual_gap')
        self.assertEqual(m.classify_residual_fidelity(self._metrics(.05, .89), SPEC),
                         'large_recursive_residual_gap')

    def test_interpretation_moderate(self):
        self.assertEqual(m.classify_residual_fidelity(self._metrics(.1, .995), SPEC),
                         'moderate_recursive_residual_gap')
        self.assertEqual(m.classify_residual_fidelity(self._metrics(.05, .989), SPEC),
                         'moderate_recursive_residual_gap')

    def test_interpretation_faithful_and_mixed(self):
        self.assertEqual(m.classify_residual_fidelity(self._metrics(.05, .995), SPEC),
                         'recursive_residual_faithful')
        self.assertEqual(m.classify_residual_fidelity(self._metrics(.05, .995, 2., 2.3), SPEC),
                         'mixed_residual_fidelity')

    def test_information_firewall(self):
        policy = SPEC['information_policy']
        self.assertTrue(policy['blind_diagnostic'])
        self.assertTrue(policy['final_checkpoint_access'])
        self.assertTrue(policy['generic_corpora_access'])
        for key in ('historical_base_access', 'true_delta_access', 'adapter_access',
                    'oracle_adl_access', 'historical_target_access',
                    'prior_oracle_evaluation_access', 'candidate_evaluation',
                    'candidate_selection', 'oracle_based_tuning'):
            self.assertFalse(policy[key])

    def test_no_privileged_input_path_in_source(self):
        source = SOURCE.read_text()
        for forbidden in ('models/base', 'oracle_adl.pt', 'evaluation.json',
                          'attempt103', 'attempt104', 'attempt105', 'attempt106',
                          'attempt107', 'attempt108', 'attempt109', 'attempt110',
                          'attempt111', 'attempt112'):
            self.assertNotIn(forbidden, source)

    def test_checkpoint_unchanged_audit_before_output(self):
        source = inspect.getsource(m.run)
        self.assertIn("after = {'serialized_sha256': sha256_file(CHECKPOINT)", source)
        self.assertLess(source.index('if after != checkpoint_record:'),
                        source.index('atomic_json_no_overwrite(RESULT_PATH, result)'))

    def test_no_large_artifact_and_no_result_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            path.write_text('old')
            with patch.object(m, 'RESULT_PATH', path):
                with self.assertRaisesRegex(ValueError, 'exists'):
                    m.atomic_json_no_overwrite(path, {'x': 1})
            self.assertEqual(path.read_text(), 'old')
        self.assertEqual(SPEC['output_policy'],
                         'small_JSON_only_atomic_no_overwrite_checkpoint_read_only')

    def test_atomic_small_json_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            with patch.object(m, 'RESULT_PATH', path):
                m.atomic_json_no_overwrite(path, {'value': 1})
            self.assertEqual(json.loads(path.read_text()), {'value': 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_progress_and_no_env_change(self):
        source = SOURCE.read_text()
        self.assertIn('build_blind_rhs(', source)
        self.assertIn("progress_label='Attempt115 Gx40'", source)
        self.assertNotIn('env.sh', source)
        self.assertNotIn('env.sh', m.SPEC_PATH.read_text())


if __name__ == '__main__':
    unittest.main()
