"""Synthetic tests for Attempt114; no model or scratch artifact is read."""
import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/diagnose_blind_attempt113_cg_conditioning.py'
loader = importlib.util.spec_from_file_location('attempt114_test_module', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


def rows(alpha=1.0, beta=0.0):
    return [{'iteration': i, 'alpha': alpha, 'beta': beta,
             'relative_residual': 0.5, 'r_norm': math.sqrt(98), 'pAp': 1.0,
             'operator_seconds': 1.0, 'cumulative_elapsed_seconds': float(i),
             'operator_eta_seconds': float(80-i)} for i in range(1, 41)]


def checkpoint_fixture(a113, prior):
    inventory = [{'name': f'weight_{i:02}.weight', 'shape': [1, 1]}
                 for i in range(98)]
    vector = {row['name']: torch.ones((1, 1), dtype=torch.float32)
              for row in inventory}
    hashes, aggregate = a113.ordered_vector_hashes(vector, inventory)
    raw = {'matrix_raw_sha256': hashes, 'ordered_matrix_hash_sha256': aggregate}
    rhs = {'norm': 2*math.sqrt(98), 'matrix_raw_sha256': hashes,
           'ordered_matrix_hash_sha256': aggregate,
           'hash_aggregation': 'SHA256_of_ordered_name_NUL_digest_newline'}
    gradient_records = [
        {'name': name, 'mean_generic_loss': 2.0,
         'global_gradient_norm': 4.0, 'unit_normalization_scalar': 0.25,
         'rows': [0, 512], 'prediction_contexts': 65024}
        for name in a113.ORDER]
    payload = {
        'format_version': 1, 'iteration': 40,
        'x': copy.deepcopy(vector), 'r': copy.deepcopy(vector),
        'p': copy.deepcopy(vector), 'rr': 98.0, 'b_norm': rhs['norm'],
        'rho_b': 100.0, 'fixed_lambda': 0.01,
        'coordinate_inventory': inventory, 'iteration_records': rows(),
        'rhs_record': rhs, 'gradient_records': gradient_records,
        'raw_hashes': {'x': raw, 'r': raw, 'p': raw},
        'provenance': a113.checkpoint_provenance(
            prior, inventory, rhs, gradient_records)}
    return payload


class Attempt114Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a113, cls.prior = m.validate_attempt113_linkage(SPEC)

    def test_frozen_commit_spec_source_linkage(self):
        self.assertEqual(SPEC['frozen_attempt113']['commit'], m.ATTEMPT113_COMMIT)
        self.assertEqual(m.sha256_file(m.ATTEMPT113_SPEC), m.ATTEMPT113_SPEC_SHA)
        self.assertEqual(m.sha256_file(m.ATTEMPT113_SOURCE), m.ATTEMPT113_SOURCE_SHA)

    def test_linkage_rejects_source_hash_mismatch(self):
        with patch.object(m, 'sha256_file', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.validate_attempt113_linkage(SPEC)

    def test_frozen_attempt113_operator_inventory(self):
        prior = self.prior
        self.assertEqual(prior['ggn']['rows'], [0, 64])
        self.assertEqual(prior['ggn']['denominator'], 8128)
        self.assertEqual(prior['ggn']['operator'], 'exact_endpoint_categorical_JtFJ')
        self.assertEqual(prior['ggn']['microbatch_size'], 1)
        self.assertFalse(prior['ggn']['empirical_fisher'])
        self.assertEqual(prior['fisher_numerics']['implementation'],
                         'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast')

    def test_frozen_attempt113_solver_inventory(self):
        self.assertEqual(self.prior['damping']['damping_fraction'], 1e-4)
        self.assertEqual(self.prior['solver']['iterations'], 80)
        self.assertEqual(self.prior['solver']['checkpoint_iteration'], 40)
        self.assertIsNone(self.prior['solver']['preconditioner'])
        self.assertFalse(self.prior['solver']['residual_early_stop'])

    def test_spec_rejects_changed_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'spec.json'
            changed = copy.deepcopy(SPEC)
            changed['frozen_attempt113']['ggn_rows'] = [1, 65]
            path.write_text(json.dumps(changed))
            with patch.object(m, 'SPEC_PATH', path):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    m.load_spec(path)

    def _read_checkpoint(self, mutate=None):
        payload = checkpoint_fixture(self.a113, self.prior)
        if mutate is not None:
            mutate(payload)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            torch.save(payload, path)
            before = m.sha256_file(path)
            with patch.object(m, 'CHECKPOINT_PATH', path):
                try:
                    value, record = m.validate_checkpoint_read_only(
                        path, self.a113, self.prior, SPEC)
                except Exception:
                    self.assertEqual(before, m.sha256_file(path))
                    raise
            self.assertEqual(before, m.sha256_file(path))
            return value, record

    def test_checkpoint_read_only_and_serialized_hash(self):
        value, record = self._read_checkpoint()
        self.assertEqual(value['iteration'], 40)
        self.assertEqual(len(record['serialized_sha256']), 64)
        self.assertGreater(record['size_bytes'], 0)

    def test_checkpoint_iteration_exactly_40(self):
        with self.assertRaisesRegex(ValueError, 'iteration'):
            self._read_checkpoint(lambda p: p.__setitem__('iteration', 39))

    def test_checkpoint_x_raw_hash(self):
        with self.assertRaisesRegex(ValueError, 'hash'):
            self._read_checkpoint(lambda p: p['x']['weight_00.weight'].add_(1))

    def test_checkpoint_r_raw_hash(self):
        with self.assertRaisesRegex(ValueError, 'hash'):
            self._read_checkpoint(lambda p: p['r']['weight_00.weight'].add_(1))

    def test_checkpoint_p_raw_hash(self):
        with self.assertRaisesRegex(ValueError, 'hash'):
            self._read_checkpoint(lambda p: p['p']['weight_00.weight'].add_(1))

    def test_checkpoint_rr_consistency(self):
        with self.assertRaisesRegex(ValueError, 'residual scalar'):
            self._read_checkpoint(lambda p: p.__setitem__('rr', 99.0))

    def test_checkpoint_exact_40_saved_rows(self):
        with self.assertRaisesRegex(ValueError, 'trajectory length'):
            self._read_checkpoint(lambda p: p['iteration_records'].pop())

    def test_checkpoint_damping_rule(self):
        with self.assertRaisesRegex(ValueError, 'damping rule'):
            self._read_checkpoint(lambda p: p.__setitem__('fixed_lambda', 0.02))

    def test_checkpoint_blind_gradient_inventory(self):
        with self.assertRaisesRegex(ValueError, 'blind RHS/checkpoint inventory'):
            self._read_checkpoint(lambda p: p['gradient_records'].pop())

    def test_checkpoint_rhs_aggregate_hash(self):
        with self.assertRaisesRegex(ValueError, 'aggregate hash'):
            self._read_checkpoint(lambda p: p['rhs_record'].__setitem__(
                'ordered_matrix_hash_sha256', '0'*64))

    def test_checkpoint_saved_relative_residual(self):
        with self.assertRaisesRegex(ValueError, 'residual mismatch'):
            self._read_checkpoint(lambda p: p['iteration_records'][-1].__setitem__(
                'relative_residual', 0.6))

    def test_checkpoint_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'actual.pt'
            path.write_bytes(b'irrelevant')
            link = Path(directory)/'checkpoint.pt'
            link.symlink_to(path)
            with patch.object(m, 'CHECKPOINT_PATH', link):
                with self.assertRaisesRegex(ValueError, 'regular'):
                    m.validate_checkpoint_read_only(link, self.a113, self.prior, SPEC)

    def test_lanczos_diagonal_formula(self):
        data = rows(alpha=2, beta=0.25)
        data[1]['alpha'] = 4
        result = m.lanczos_ritz(data, 0.1, SPEC)
        self.assertEqual(result['diagonal'][0], 0.5)
        self.assertEqual(result['diagonal'][1], 0.25 + 0.25/2)

    def test_lanczos_offdiagonal_formula(self):
        result = m.lanczos_ritz(rows(alpha=2, beta=0.25), 0.1, SPEC)
        self.assertEqual(result['offdiagonal'][0], 0.25)
        self.assertEqual(len(result['offdiagonal']), 39)

    def test_lanczos_cpu_float64_eigenvalues(self):
        result = m.lanczos_ritz(rows(alpha=1, beta=0), 0.1, SPEC)
        self.assertEqual(result['lanczos_dtype'], 'cpu_float64')
        self.assertEqual(result['A_ritz_eigenvalues_ascending'], [1.0]*40)
        self.assertEqual(result['ritz_condition_estimate'], 1.0)

    def test_lanczos_no_implied_G_clipping(self):
        result = m.lanczos_ritz(rows(alpha=1, beta=0), 2.0, SPEC)
        self.assertEqual(result['min_implied_G_ritz'], -1.0)
        self.assertEqual(result['max_implied_G_ritz'], -1.0)

    def test_lanczos_damping_threshold_counts(self):
        result = m.lanczos_ritz(rows(alpha=1, beta=0), 0.5, SPEC)
        self.assertEqual(result['A_ritz_at_or_below_damping_multiples']['2'],
                         {'count': 40, 'fraction': 1.0})
        self.assertTrue(result['ritz_reaches_damping_scale'])

    def test_lanczos_alpha_positive(self):
        data = rows()
        data[0]['alpha'] = 0
        with self.assertRaisesRegex(ValueError, 'alpha/beta'):
            m.lanczos_ritz(data, 0.1, SPEC)

    def test_lanczos_beta_nonnegative(self):
        data = rows()
        data[8]['beta'] = -1
        with self.assertRaisesRegex(ValueError, 'alpha/beta'):
            m.lanczos_ritz(data, 0.1, SPEC)

    def test_lanczos_nonfinite_fails(self):
        data = rows()
        data[1]['alpha'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'alpha/beta'):
            m.lanczos_ritz(data, 0.1, SPEC)

    def test_residual_rayleigh_one_G_application(self):
        r = {'w': torch.tensor([[2.0]], dtype=torch.float32)}
        calls = []

        def apply(vector):
            calls.append(1)
            return {'w': vector['w']*3}, {'first_batch': True}

        result = m.residual_curvature(r, apply, self.a113, 6.0, 0.1, 0.5, SPEC)
        self.assertEqual(calls, [1])
        self.assertEqual(result['r40_norm_sq'], 4.0)
        self.assertEqual(result['r40_G_r40'], 12.0)
        self.assertEqual(result['rho_r40'], 3.0)
        self.assertEqual(result['rho_r40_over_rho_b'], 0.5)

    def test_residual_negative_curvature_corruption_fails(self):
        r = {'w': torch.tensor([[1.0]], dtype=torch.float32)}
        with self.assertRaisesRegex(ValueError, 'negative'):
            m.residual_curvature(r, lambda v: ({'w': -v['w']}, {}),
                                 self.a113, 1.0, 0.1, 0.5, SPEC)

    def test_matrixwise_fp64_reduction(self):
        r = {'a': torch.tensor([[1e5]], dtype=torch.float32),
             'b': torch.tensor([[1]], dtype=torch.float32)}
        self.assertEqual(self.a113.matrixwise_dot(r, r), 10000000001.0)

    def test_final_model_only_and_fineweb_inventory(self):
        prior = self.prior
        self.assertEqual(prior['final_checkpoint']['directory'],
                         '/root/model-diff-scratch/models/merged')
        self.assertEqual(prior['ggn']['corpus_name'], 'fineweb')
        self.assertEqual(prior['ggn']['rows'], [0, 64])
        self.assertEqual(prior['ggn']['tokens_path'],
                         '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt')

    def test_blind_information_policy(self):
        policy = SPEC['information_policy']
        self.assertTrue(policy['blind_diagnostic'])
        self.assertTrue(policy['final_checkpoint_access'])
        self.assertTrue(policy['attempt113_checkpoint_access'])
        for name in ('historical_base_access', 'true_delta_access', 'adapter_access',
                     'oracle_adl_access', 'historical_target_access',
                     'prior_oracle_evaluation_access', 'candidate_evaluation',
                     'candidate_selection', 'oracle_based_tuning'):
            self.assertFalse(policy[name])

    def test_no_privileged_io_or_cg_continuation_in_source(self):
        source = SOURCE.read_text()
        for forbidden in ('models/base', 'oracle_adl.pt', 'evaluation.json',
                          'run_fixed_80_cg(', 'restore_cg_state('):
            self.assertNotIn(forbidden, source)
        self.assertEqual(source.count('module.ggn_action_stage('), 1)

    def test_no_large_scientific_artifact(self):
        self.assertEqual(SPEC['output_policy'],
                         'small_JSON_only_no_overwrite_no_checkpoint_mutation')
        self.assertTrue(SPEC['output_path'].endswith('/result.json'))

    def test_no_result_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            path.write_text('old')
            with patch.object(m, 'OUTPUT_PATH', path):
                with self.assertRaisesRegex(ValueError, 'exists'):
                    m.atomic_json_no_overwrite(path, {'x': 1})
            self.assertEqual(path.read_text(), 'old')

    def test_atomic_small_json_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            with patch.object(m, 'OUTPUT_PATH', path):
                m.atomic_json_no_overwrite(path, {'x': 1})
            self.assertEqual(json.loads(path.read_text()), {'x': 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_no_env_modification(self):
        self.assertNotIn('env.sh', SOURCE.read_text())
        self.assertNotIn('env.sh', m.SPEC_PATH.read_text())


if __name__ == '__main__':
    unittest.main()
