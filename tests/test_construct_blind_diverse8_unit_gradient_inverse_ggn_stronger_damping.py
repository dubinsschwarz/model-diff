"""CPU/synthetic tests for Attempt116; no real model or scratch artifact reads."""
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
SOURCE = PROJECT/'scripts/ablation/construct_blind_diverse8_unit_gradient_inverse_ggn_stronger_damping.py'
loader = importlib.util.spec_from_file_location('attempt116_synthetic', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)


class Attempt116Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old, cls.original_spec = m.load_attempt113()
        cls.spec = m.load_spec(cls.original_spec)

    def test_attempt113_commit_and_hashes(self):
        self.assertEqual(m.FROZEN['attempt113']['commit'],
                         '5357ba00408f07b284de3126227fc5eb317e4f13')
        self.assertEqual(m.sha256_file(m.PRIOR113_SPEC),
                         m.FROZEN['attempt113']['spec_sha256'])
        self.assertEqual(m.sha256_file(m.PRIOR113_SOURCE),
                         m.FROZEN['attempt113']['source_sha256'])

    def test_attempt114_115_result_hashes_and_blind_values(self):
        self.assertEqual(m.sha256_file(m.PRIOR114_RESULT),
                         '151d1074cdcf5d5d3152bf1e2f746940ea287de88fdc597a737db83cbf74a345')
        self.assertEqual(m.sha256_file(m.PRIOR115_RESULT),
                         'fe28aa13b1635985ea1304e5c61375b72959e0f98170659eab3e77e4c446fa95')
        linkage = m.validate_blind_development_linkage(self.spec)
        self.assertEqual(linkage['expected_rhs_aggregate_sha256'],
                         'a90785341961a0a2a08e7949961cb105867cf2509013b01b65c2013322ead574')
        self.assertEqual(self.spec['blind_development_linkage']['attempt115']['blind_values']['interpretation'],
                         'recursive_residual_faithful')

    def test_frozen_diagnostic_linkage_rejects_hash_mismatch(self):
        with patch.object(m, 'sha256_file', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'file/commit mismatch'):
                m.validate_blind_development_linkage(self.spec)

    def test_frozen_rhs_norm_and_all_matrix_hashes(self):
        rhs = self.spec['expected_attempt113_rhs']
        self.assertEqual(rhs['norm'], 0.6924963677590905)
        self.assertEqual(len(rhs['matrix_raw_sha256']), 98)
        digest = hashlib.sha256()
        for name in sorted(rhs['matrix_raw_sha256']):
            raw = rhs['matrix_raw_sha256'][name]
            self.assertEqual(len(raw), 64)
            digest.update(name.encode()+b'\0'+raw.encode()+b'\n')
        self.assertEqual(digest.hexdigest(), rhs['ordered_matrix_hash_sha256'])

    def test_only_scientific_spec_change_is_damping(self):
        expected = m.expected_spec_from_attempt113(self.original_spec)
        actual = copy.deepcopy(self.spec)
        for key in ('blind_development_linkage', 'expected_attempt113_rhs',
                    'precommitted_future_comparison'):
            actual.pop(key)
        self.assertEqual(actual, expected)
        expected['damping']['damping_fraction'] = 1e-4
        self.assertEqual(expected['damping'], self.original_spec['damping'])

    def test_changed_G_spec_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'spec.json'
            changed = copy.deepcopy(self.spec)
            changed['ggn']['denominator'] = 8000
            path.write_text(json.dumps(changed))
            with patch.object(m, 'SPEC_PATH', path):
                with self.assertRaisesRegex(ValueError, 'differs'):
                    m.load_spec(self.original_spec, path)

    def test_exact_corpus_order_rows_and_batch(self):
        self.assertEqual(self.spec['corpus_order'],
                         ['fineweb', 'wikitext103_raw', 'tinystories',
                          'arxiv_document', 'cc_news', 'pg19',
                          'codeparrot_clean', 'ultrachat'])
        self.assertEqual(self.spec['rhs']['rows'], [0, 512])
        self.assertEqual(self.spec['rhs']['batch_size'], 8)
        self.assertEqual(self.spec['rhs']['prediction_contexts_per_corpus'], 65024)

    def test_exact_rhs_scientific_semantics(self):
        self.assertEqual(self.spec['rhs'], self.original_spec['rhs'])
        self.assertIn('build_blind_rhs', inspect.getsource(self.old.construct))
        self.assertIn('add_unit_gradient_to_consensus', inspect.getsource(self.old.build_blind_rhs))
        self.assertIn('consensus[key].add_(unit, alpha=1.0/corpus_count)',
                      inspect.getsource(self.old.add_unit_gradient_to_consensus))

    def test_rhs_exact_hash_identity_required(self):
        expected = self.spec['expected_attempt113_rhs']
        m.require_exact_rhs(expected, expected)
        altered = copy.deepcopy(expected)
        name = next(iter(altered['matrix_raw_sha256']))
        altered['matrix_raw_sha256'][name] = '0'*64
        with self.assertRaisesRegex(ValueError, 'RHS differs'):
            m.require_exact_rhs(altered, expected)

    def test_rhs_aggregate_and_norm_mismatch_fail(self):
        expected = self.spec['expected_attempt113_rhs']
        for key, value in (('ordered_matrix_hash_sha256', '0'*64),
                           ('norm', expected['norm']+.001)):
            altered = copy.deepcopy(expected)
            altered[key] = value
            with self.assertRaisesRegex(ValueError, 'RHS differs'):
                m.require_exact_rhs(altered, expected)

    def test_rho_lambda_derived_only_from_current_blind_rhs(self):
        layer = torch.nn.Linear(2, 1, bias=False)
        layer.weight.grad = torch.tensor([[2., 4.]], dtype=torch.float32)
        eligible = [('linear', layer)]
        direction = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}
        value = m.rayleigh_and_lambda_116(self.old, eligible, direction, 1e-3)
        self.assertEqual(value['bb'], 5.)
        self.assertEqual(value['bGb'], 10.)
        self.assertEqual(value['rho_b'], 2.)
        self.assertEqual(value['lambda'], .002)
        self.assertEqual(value['damping_fraction'], 1e-3)

    def test_damping_fraction_guard(self):
        with self.assertRaisesRegex(ValueError, 'exactly 1e-3'):
            m.rayleigh_and_lambda_116(self.old, [], {}, 1e-4)
        self.assertEqual(self.spec['damping']['damping_fraction'], 1e-3)
        self.assertNotIn('lambda', self.spec['damping'])

    def test_same_98_matrix_support_and_freeze(self):
        self.assertEqual(self.spec['eligible_tensors'], self.original_spec['eligible_tensors'])
        self.assertEqual(self.spec['eligible_tensors']['expected_matrix_count'], 98)
        self.assertIn('freeze_other_parameters(model, eligible)',
                      inspect.getsource(self.old.construct))

    def test_same_G_operator_and_fineweb_inventory(self):
        self.assertEqual(self.spec['ggn'], self.original_spec['ggn'])
        self.assertEqual(self.spec['ggn']['rows'], [0, 64])
        self.assertEqual(self.spec['ggn']['denominator'], 8128)
        self.assertEqual(self.spec['ggn']['operator'],
                         'exact_endpoint_categorical_JtFJ')
        self.assertFalse(self.spec['ggn']['empirical_fisher'])

    def test_same_FP64_fisher_numerics(self):
        self.assertEqual(self.spec['fisher_numerics'],
                         self.original_spec['fisher_numerics'])
        source = inspect.getsource(self.old.stable_categorical_fisher_vector)
        for fragment in ('torch.softmax(logits.to(torch.float64)',
                         'tangent.to(torch.float64)',
                         'u64 = p64*(s64-mean64)',
                         'u64 /= spec[\'ggn\'][\'denominator\']',
                         'u = u64.to(torch.float32)'):
            self.assertIn(fragment, source)

    def test_solver_80_no_early_stop_or_preconditioner(self):
        self.assertEqual(self.spec['solver'], self.original_spec['solver'])
        self.assertEqual(self.spec['solver']['iterations'], 80)
        self.assertIsNone(self.spec['solver']['preconditioner'])
        self.assertFalse(self.spec['solver']['residual_early_stop'])
        self.assertEqual(self.spec['solver']['candidate'], 'x_80')

    def test_iteration40_checkpoint_resume_same_semantics(self):
        self.assertEqual(self.spec['checkpoint'], self.original_spec['checkpoint'])
        self.assertEqual(self.spec['solver']['checkpoint_iteration'], 40)
        self.assertEqual(self.spec['checkpoint']['resume_starts_at'], 41)
        self.assertEqual(self.spec['paths']['checkpoint_path'], m.CHECKPOINT_PATH)
        self.assertIn('module.construct(resume=resume', inspect.getsource(m.construct))

    def test_runtime_binding_uses_new_spec_and_source_hashes(self):
        old_values = (self.old.ATTEMPT, self.old.SPEC_PATH,
                      self.old.SPEC_SHA256, self.old.__file__)
        with m.bind_attempt116(self.old, self.spec):
            self.assertEqual(self.old.ATTEMPT, m.ATTEMPT)
            self.assertEqual(self.old.SPEC_PATH, m.SPEC_PATH)
            self.assertEqual(self.old.SPEC_SHA256, m.sha256_file(m.SPEC_PATH))
            self.assertEqual(Path(self.old.__file__), m.SOURCE_PATH)
            self.assertEqual(self.old.load_spec()['damping']['damping_fraction'], 1e-3)
            provenance = self.old.checkpoint_provenance(
                self.spec, [{'name': 'toy', 'shape': [1]}],
                {'ordered_matrix_hash_sha256': 'a'*64}, [])
            self.assertEqual(provenance['spec_sha256'], m.sha256_file(m.SPEC_PATH))
            self.assertEqual(provenance['source_sha256'], m.sha256_file(m.SOURCE_PATH))
        self.assertEqual((self.old.ATTEMPT, self.old.SPEC_PATH,
                          self.old.SPEC_SHA256, self.old.__file__), old_values)

    def test_runtime_binding_restores_on_failure(self):
        original = self.old.build_blind_rhs
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with m.bind_attempt116(self.old, self.spec):
                raise RuntimeError('synthetic')
        self.assertIs(self.old.build_blind_rhs, original)

    def test_runtime_rhs_check_before_rayleigh_or_CG(self):
        source = inspect.getsource(m.bind_attempt116)
        self.assertIn('require_exact_rhs(record, spec[\'expected_attempt113_rhs\']', source)
        self.assertIn("'build_blind_rhs': checked_rhs", source)
        constructor = inspect.getsource(self.old.construct)
        self.assertLess(constructor.index('build_blind_rhs('),
                        constructor.index('rayleigh_and_lambda('))

    def test_same_generic_probe_and_two_responses(self):
        self.assertEqual(self.spec['probe'], self.original_spec['probe'])
        self.assertEqual(self.spec['probe']['rows'], [0, 1024])
        self.assertEqual(self.spec['probe']['batch_size'], 32)
        self.assertEqual(self.spec['readout']['hidden_state_index'], 14)
        self.assertEqual(self.spec['artifact']['tensor_inventory'],
                         ['x_80', 'response_raw_rhs', 'response_inverse'])
        self.assertIn('response_raw, raw_audit = probe_response(',
                      inspect.getsource(self.old.construct))
        self.assertIn('response_inverse, inverse_audit = probe_response(',
                      inspect.getsource(self.old.construct))

    def test_manifest_adds_blind_linkage_and_damping_choice(self):
        captured = {}

        def fake_publish(*args):
            captured.update(args[6])

        original = self.old.publish_candidate
        self.old.publish_candidate = fake_publish
        try:
            with m.bind_attempt116(self.old, self.spec):
                manifest = {'rayleigh': {'damping_fraction': 1e-3,
                                         'rho_b': 2., 'lambda': .002},
                            'b_blind': {'ordered_matrix_hash_sha256':
                                        self.spec['expected_attempt113_rhs']['ordered_matrix_hash_sha256']},
                            'solver': {'operator_applications': 80},
                            'candidate': {'name': 'x_80'}}
                self.old.publish_candidate(None, None, None, None, None, None,
                                           manifest, None)
        finally:
            self.old.publish_candidate = original
        self.assertTrue(captured['damping_choice_informed_only_by_blind_attempt114_115_numerical_diagnostics'])
        self.assertEqual(captured['blind_development_linkage'], m.FROZEN)
        self.assertEqual(captured['precommitted_future_comparison'], m.FUTURE_COMPARISON)

    def test_manifest_rejects_non_x80_or_wrong_damping(self):
        called = []
        original = self.old.publish_candidate
        self.old.publish_candidate = lambda *args: called.append(1)
        try:
            with m.bind_attempt116(self.old, self.spec):
                manifest = {'rayleigh': {'damping_fraction': 1e-4,
                                         'rho_b': 2., 'lambda': .0002},
                            'b_blind': {'ordered_matrix_hash_sha256':
                                        self.spec['expected_attempt113_rhs']['ordered_matrix_hash_sha256']},
                            'solver': {'operator_applications': 80},
                            'candidate': {'name': 'x_80'}}
                with self.assertRaisesRegex(ValueError, 'publication audit'):
                    self.old.publish_candidate(None, None, None, None, None, None,
                                               manifest, None)
        finally:
            self.old.publish_candidate = original
        self.assertEqual(called, [])

    def test_atomic_candidate_publication_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'candidate.pt'
            self.old.atomic_torch_publish(path, {'toy': torch.ones(2)})
            before = m.sha256_file(path)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.old.atomic_torch_publish(path, {'toy': torch.zeros(2)})
            self.assertEqual(m.sha256_file(path), before)

    def test_output_paths_are_new_and_stale_outputs_refused(self):
        self.assertEqual(self.spec['paths']['artifact_path'], m.ARTIFACT_PATH)
        self.assertEqual(self.spec['paths']['manifest_path'], m.MANIFEST_PATH)
        self.assertNotEqual(self.spec['paths'], self.original_spec['paths'])
        with tempfile.TemporaryDirectory() as directory:
            altered = copy.deepcopy(self.spec)
            altered['paths'] = {'artifact_path': str(Path(directory)/'candidate.pt'),
                                'manifest_path': str(Path(directory)/'manifest.json'),
                                'checkpoint_path': str(Path(directory)/'checkpoint.pt')}
            Path(altered['paths']['manifest_path']).write_text('stale')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.old.refuse_outputs(altered)

    def test_information_firewall_and_no_candidate_selection(self):
        policy = self.spec['information_policy']
        for key in ('historical_base_access', 'true_delta_access', 'adapter_access',
                    'oracle_adl_access', 'historical_target_access',
                    'prior_oracle_evaluation_access', 'candidate_selection',
                    'sign_selection', 'corpus_selection', 'corpus_reweighting',
                    'iterate_selection', 'oracle_based_tuning_during_construction'):
            self.assertFalse(policy[key])
        self.assertTrue(policy['damping_choice_informed_only_by_blind_attempt114_115_numerical_diagnostics'])

    def test_future_comparison_precommitted(self):
        self.assertTrue(m.FUTURE_COMPARISON['compare_only_after_both_construction_manifests_committed'])
        self.assertFalse(m.FUTURE_COMPARISON['third_damping_on_same_specimen_after_historical_scores'])
        self.assertTrue(m.FUTURE_COMPARISON['blind_residuals_cannot_select_iterate'])

    def test_no_historical_oracle_path_or_evaluator(self):
        source = SOURCE.read_text()
        spec = m.SPEC_PATH.read_text()
        for forbidden in ('models/base', 'oracle_adl.pt', 'evaluation.json',
                          'attempt103', 'attempt104', 'attempt105', 'attempt106',
                          'attempt107', 'attempt108', 'attempt109', 'attempt110',
                          'attempt111', 'attempt112'):
            self.assertNotIn(forbidden, source)
            self.assertNotIn(forbidden, spec)
        self.assertFalse((PROJECT/'scripts/ablation/evaluate_attempt116.py').exists())

    def test_no_env_change_and_same_workload(self):
        self.assertNotIn('env.sh', SOURCE.read_text())
        self.assertEqual(self.spec['workload']['gradient_batches'], 512)
        self.assertEqual(self.spec['workload']['rayleigh_G_applications'], 1)
        self.assertEqual(self.spec['workload']['cg_G_applications'], 80)
        self.assertEqual(self.spec['workload']['probe_responses'], 2)


if __name__ == '__main__':
    unittest.main()
