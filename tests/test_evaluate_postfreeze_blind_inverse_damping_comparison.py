"""Synthetic and frozen-metadata tests for Attempt117; no artifact or oracle reads."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/evaluate_postfreeze_blind_inverse_damping_comparison.py'
SPEC = PROJECT/'experiments/attempts/117_postfreeze_blind_inverse_damping_comparison/spec.json'
_loader = importlib.util.spec_from_file_location('attempt117_evaluation_tests', SOURCE)
m = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(m)


def frozen_metadata():
    specs, manifests = {}, {}
    for key, record in m.CONSTRUCTIONS.items():
        specs[key] = json.loads((PROJECT/record['spec']['path']).read_text())
        manifests[key] = json.loads((PROJECT/record['manifest']['path']).read_text())
    return specs, manifests


def response(sign=1):
    value = torch.zeros((128, 2048), dtype=torch.float32)
    value[:, 0] = sign
    return value


def synthetic_payload():
    weight = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    raw = m.raw_float32_sha256(weight)
    aggregate = hashlib.sha256(b'w\0'+raw.encode()+b'\n').hexdigest()
    baseline, inverse = response(), response(-1)
    record = copy.deepcopy(m.CONSTRUCTIONS['113'])
    record['x80_aggregate_sha256'] = aggregate
    record['response_raw_rhs_sha256'] = m.raw_float32_sha256(baseline)
    record['response_inverse_sha256'] = m.raw_float32_sha256(inverse)
    manifest = {'coordinate_inventory': [{'name': 'w', 'shape': [1, 2]}],
                'candidate': {'x_80': {'matrix_raw_sha256': {'w': raw}}},
                'functional_responses': {
                    'response_raw_rhs': {'raw_sha256': record['response_raw_rhs_sha256']},
                    'response_inverse': {'raw_sha256': record['response_inverse_sha256']}}}
    payload = {'x_80': {'w': weight}, 'response_raw_rhs': baseline,
               'response_inverse': inverse}
    return record, manifest, payload


class Attempt117Tests(unittest.TestCase):
    def test_exact_frozen_spec_and_all_links(self):
        self.assertEqual(json.loads(SPEC.read_text()), m.FROZEN_SPEC)
        specs, manifests = frozen_metadata()
        for key, record in m.CONSTRUCTIONS.items():
            for field in ('spec', 'source', 'manifest'):
                item = record[field]
                self.assertEqual(m.oracle_validator.sha256_file(m.path_of(item['path'])), item['sha256'])
            self.assertIs(m.validate_construction_record(record, manifests[key], specs[key]), manifests[key])
        m.validate_cross_candidate_identity(specs['113'], specs['116'], manifests['113'], manifests['116'])

    def test_exact_committed_113_and_116_bytes(self):
        for record in m.CONSTRUCTIONS.values():
            m.validate_frozen_commit(record['commit'], record)
        bad = copy.deepcopy(m.CONSTRUCTIONS['113'])
        bad['commit'] = '0'*40
        with self.assertRaisesRegex(ValueError, 'commit linkage'):
            m.validate_frozen_commit(bad['commit'], bad)

    def test_manifest_hash_rejection_before_scientific_use(self):
        specs, manifests = frozen_metadata()
        bad = copy.deepcopy(m.CONSTRUCTIONS['113'])
        bad['manifest']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            m.validate_construction_record(bad, manifests['113'], specs['113'], check_commit=False)

    def test_frozen_manifest_numerics_and_firewall_rejection(self):
        specs, manifests = frozen_metadata()
        record = m.CONSTRUCTIONS['116']
        for path, value in ((('rayleigh', 'damping_fraction'), .0001),
                            (('solver', 'operator_applications'), 79),
                            (('candidate', 'final_relative_linear_residual'), .5),
                            (('functional_responses', 'response_inverse', 'raw_sha256'), '0'*64)):
            bad = copy.deepcopy(manifests['116'])
            cursor = bad
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with patch.object(m, 'require_hash'):
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    m.validate_construction_record(record, bad, specs['116'], check_commit=False)
        bad = copy.deepcopy(manifests['116'])
        bad['information_policy']['oracle_adl_access'] = True
        with patch.object(m, 'require_hash'):
            with self.assertRaisesRegex(ValueError, 'policy'):
                m.validate_construction_record(record, bad, specs['116'], check_commit=False)

    def test_cross_candidate_rhs_ggn_and_budget_identity(self):
        specs, manifests = frozen_metadata()
        for field in ('b_blind', 'ggn_numerics', 'ggn_operator', 'coordinate_inventory'):
            self.assertEqual(manifests['113'][field], manifests['116'][field])
        self.assertEqual(manifests['113']['functional_responses']['response_raw_rhs']['raw_sha256'],
                         manifests['116']['functional_responses']['response_raw_rhs']['raw_sha256'])
        self.assertEqual(len(manifests['113']['solver']['iterations']), 80)
        self.assertEqual(len(manifests['116']['solver']['iterations']), 80)
        bad = copy.deepcopy(manifests['116'])
        bad['b_blind']['ordered_matrix_hash_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'input mismatch'):
            m.validate_cross_candidate_identity(specs['113'], specs['116'], manifests['113'], bad)

    def test_only_damping_fraction_differs_in_frozen_scientific_specs(self):
        specs, _ = frozen_metadata()
        for key in ('model', 'final_checkpoint', 'eligible_tensors', 'corpus_order',
                    'corpora', 'rhs', 'ggn', 'fisher_numerics', 'solver', 'probe',
                    'readout', 'jvp'):
            self.assertEqual(specs['113'][key], specs['116'][key])
        self.assertEqual(specs['113']['damping']['damping_fraction'], .0001)
        self.assertEqual(specs['116']['damping']['damping_fraction'], .001)
        bad = copy.deepcopy(specs['116'])
        bad['ggn']['denominator'] = 1
        with self.assertRaisesRegex(ValueError, 'specification mismatch'):
            _, manifests = frozen_metadata()
            m.validate_cross_candidate_identity(specs['113'], bad, manifests['113'], manifests['116'])
        bad = copy.deepcopy(specs['116'])
        bad['damping']['adaptive'] = True
        with self.assertRaisesRegex(ValueError, 'damping rule mismatch'):
            _, manifests = frozen_metadata()
            m.validate_cross_candidate_identity(specs['113'], bad, manifests['113'], manifests['116'])

    def test_artifact_serialized_hash_mismatch_rejected(self):
        record, manifest, _ = synthetic_payload()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'candidate.pt'
            path.write_bytes(b'bad')
            record['artifact']['path'] = str(path)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.load_frozen_responses(record, manifest)

    def test_artifact_exact_inventory_and_parameter_hashes(self):
        record, manifest, payload = synthetic_payload()
        retained = m.validate_artifact_payload(payload, manifest, record)
        self.assertEqual(list(retained), ['response_raw_rhs', 'response_inverse'])
        self.assertNotEqual(retained['response_raw_rhs'].data_ptr(), payload['response_raw_rhs'].data_ptr())
        bad = copy.deepcopy(payload)
        bad['extra'] = torch.tensor(1)
        with self.assertRaisesRegex(ValueError, 'inventory'):
            m.validate_artifact_payload(bad, manifest, record)
        bad = copy.deepcopy(payload)
        bad['x_80']['w'][0, 0] = 9
        with self.assertRaisesRegex(ValueError, 'raw SHA256'):
            m.validate_artifact_payload(bad, manifest, record)

    def test_artifact_response_hash_shape_dtype_and_finiteness(self):
        record, manifest, payload = synthetic_payload()
        for replacement in (torch.zeros((2, 3), dtype=torch.float32),
                            payload['response_inverse'].double(),
                            payload['response_inverse'].clone().fill_(float('nan'))):
            bad = copy.deepcopy(payload)
            bad['response_inverse'] = replacement
            with self.assertRaises(ValueError):
                m.validate_artifact_payload(bad, manifest, record)
        bad = copy.deepcopy(record)
        bad['response_inverse_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'raw tensor SHA-256'):
            m.validate_artifact_payload(payload, manifest, bad)

    def test_sequential_mmap_loading_retains_only_responses(self):
        record, manifest, payload = synthetic_payload()
        calls = []
        def fake_load(path, **kwargs):
            calls.append((path, kwargs))
            return payload
        with patch.object(m, 'require_hash'), patch.object(m.torch, 'load', side_effect=fake_load):
            a = m.load_frozen_responses(record, manifest)
            b = m.load_frozen_responses(record, manifest)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(kwargs == {'map_location': 'cpu', 'weights_only': True, 'mmap': True}
                            for _, kwargs in calls))
        self.assertEqual(list(a), ['response_raw_rhs', 'response_inverse'])
        self.assertEqual(list(b), ['response_raw_rhs', 'response_inverse'])
        self.assertNotIn('x_80', a)

    def test_raw_rhs_byte_identity_requires_both_pinned_hashes(self):
        record113 = m.CONSTRUCTIONS['113']
        record116 = m.CONSTRUCTIONS['116']
        self.assertEqual(record113['response_raw_rhs_sha256'], record116['response_raw_rhs_sha256'])
        self.assertEqual(m.FROZEN_SPEC['cross_candidate_identity']['raw_rhs_response_sha256'],
                         record113['response_raw_rhs_sha256'])
        lhs, rhs = response(), response()
        self.assertEqual(m.raw_float32_sha256(lhs), m.raw_float32_sha256(rhs))
        rhs[0, 0] = 2
        self.assertNotEqual(m.raw_float32_sha256(lhs), m.raw_float32_sha256(rhs))

    def test_oracle_hashes_and_validator_provenance_frozen(self):
        oracle = m.FROZEN_SPEC['oracle']
        self.assertEqual(oracle['validator_source']['sha256'], m.ORACLE_VALIDATOR_SHA)
        self.assertEqual(m.oracle_validator.sha256_file(m.ORACLE_VALIDATOR), m.ORACLE_VALIDATOR_SHA)
        self.assertEqual(oracle['provenance_inputs'], m.oracle_validator.FROZEN_SPEC['inputs'])
        self.assertEqual(oracle['serialized_sha256'], '339a6c09b10286369a5d10eb49e144d1819f4301e334c63930ca7c4dd3a11077')
        self.assertEqual(oracle['difference_raw_sha256'], 'd60789a2c06eb82dba2f2449d8665ec665d85e9f7e33fdea1f66ac79c81c159d')

    def test_oracle_artifact_and_provenance_mismatch_rejected(self):
        inputs = m.FROZEN_SPEC['oracle']['provenance_inputs']
        with patch.object(m, 'require_hash', side_effect=ValueError('SHA256 mismatch')):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.validate_oracle(inputs)
        fake_manifest = {'oracle_adl_sha256': '0'*64,
                         'raw_tensors_sha256': {'difference': m.FROZEN_SPEC['oracle']['difference_raw_sha256']}}
        with patch.object(m, 'require_hash'), \
             patch.object(m.oracle_validator, 'load_json_object', return_value={'probe': {}}), \
             patch.object(m.oracle_validator, 'validate_construction'), \
             patch.object(m.oracle_validator, 'validate_oracle_provenance', return_value=fake_manifest), \
             patch.object(m.oracle_validator, 'load_and_validate_oracle') as load:
            with self.assertRaisesRegex(ValueError, 'oracle artifact/raw'):
                m.validate_oracle(inputs)
            load.assert_not_called()

    def test_signed_cpu_float64_position_cosines_and_no_flattening(self):
        target = response()
        candidate = response()
        candidate[0, 0] = -1
        candidate[2].zero_()
        candidate[2, 1] = 10
        candidate[4, 0] = -1
        report = m.cosine_report(candidate, target)
        self.assertEqual(report['position_0_cosine'], -1)
        self.assertEqual(report['positions_1_4_individual_cosines'], [1, 0, 1, -1])
        self.assertEqual(report['positions_1_4_mean_cosine'], .25)
        self.assertEqual(report['positions_1_127_mean_cosine'], 124/127)
        self.assertEqual(len(report['all_128_position_cosines']), 128)
        flat = float(torch.nn.functional.cosine_similarity(candidate[1:5].double().flatten(),
                                                          target[1:5].double().flatten(), dim=0))
        self.assertNotAlmostEqual(flat, .25)
        self.assertEqual(m.FROZEN_SPEC['metrics']['dtype'], 'cpu_float64')
        self.assertFalse(m.FROZEN_SPEC['metrics']['absolute_cosine'])

    def test_zero_norm_is_null_and_required_mean_is_null(self):
        candidate, target = response(), response()
        candidate[127].zero_()
        report = m.cosine_report(candidate, target)
        self.assertIsNone(report['all_128_position_cosines'][127])
        self.assertIsNone(report['positions_1_127_mean_cosine'])
        self.assertEqual(report['positions_1_4_mean_cosine'], 1)

    def test_three_fixed_roles_pairwise_and_precommitted_deltas(self):
        target = response()
        values = {'raw_rhs': response(), 'inverse_1e-4': response(-1),
                  'inverse_1e-3': response()}
        report = m.summarize_responses(values, target)
        self.assertEqual(list(report['candidates']), ['raw_rhs', 'inverse_1e-4', 'inverse_1e-3'])
        self.assertEqual([r['role'] for r in report['candidates'].values()],
                         ['blind_uninverted_baseline', 'frozen_weak_damping_inverse',
                          'frozen_stronger_damping_inverse'])
        self.assertEqual(len(report['pairwise_candidate_responses']), 3)
        self.assertEqual(report['pairwise_candidate_responses']['raw_rhs_vs_inverse_1e-4']['positions_1_4_mean_cosine'], -1)
        self.assertEqual(report['precommitted_comparisons']['gain_primary_1e-4_vs_raw'], -2)
        self.assertEqual(report['precommitted_comparisons']['gain_primary_1e-3_vs_raw'], 0)
        self.assertEqual(report['precommitted_comparisons']['gain_primary_1e-3_vs_1e-4'], 2)
        self.assertEqual(report['precommitted_comparisons']['gain_secondary_1e-3_vs_1e-4'], 2)
        self.assertEqual(values['inverse_1e-4'][0, 0].item(), -1)

    def test_no_third_damping_or_best_candidate_selection(self):
        names = [row['name'] for row in m.FROZEN_SPEC['candidates']]
        self.assertEqual(names, ['raw_rhs', 'inverse_1e-4', 'inverse_1e-3'])
        self.assertFalse(m.FROZEN_SPEC['output_policy']['candidate_selection'])
        self.assertFalse(m.FROZEN_SPEC['output_policy']['damping_selection_after_oracle'])
        self.assertFalse(m.FROZEN_SPEC['output_policy']['third_damping_value_after_oracle'])
        with self.assertRaisesRegex(ValueError, 'Exact three'):
            m.summarize_responses({'raw_rhs': response()}, response())

    def test_no_overwrite_or_input_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'result.json'
            path.write_text('existing')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.require_output_absent(path)
            self.assertEqual(path.read_text(), 'existing')
        candidate = response()
        before = m.raw_float32_sha256(candidate)
        m.cosine_report(candidate, response(-1))
        self.assertEqual(m.raw_float32_sha256(candidate), before)

    def test_postfreeze_information_policy_and_no_model_pipeline(self):
        policy = m.FROZEN_SPEC['output_policy']
        for key in ('post_freeze_evaluation', 'attempt113_frozen_before_oracle',
                    'attempt116_frozen_before_oracle', 'oracle_adl_access',
                    'same_specimen_exploratory_method_development',
                    'hyperparameters_informed_by_prior_privileged_development'):
            self.assertTrue(policy[key])
        for key in ('candidate_modified_after_oracle', 'sign_selection',
                    'candidate_selection', 'damping_selection_after_oracle',
                    'candidate_rescaling', 'position_selection',
                    'clean_heldout_validation', 'model_loading',
                    'gradient_jvp_ggn_computation'):
            self.assertFalse(policy[key])
        source = SOURCE.read_text()
        for forbidden in ('AutoModelForCausalLM', 'from_pretrained(', 'functional_call(',
                          'torch.func.jvp(', 'autograd.grad(', 'model.generate(', 'models/base'):
            self.assertNotIn(forbidden, source)
        self.assertTrue((PROJECT/'env.sh').exists())


if __name__ == '__main__':
    unittest.main()
