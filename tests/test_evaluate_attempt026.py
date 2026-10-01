"""Synthetic checks for the frozen Attempt 026 post-freeze evaluator."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT/'scripts/ablation/evaluate_attempt026.py'
FOLDER = PROJECT/'experiments/attempts/026_blind_direct_activation_gradient_block13_pilot'
loader = importlib.util.spec_from_file_location('attempt026_evaluation_test', SCRIPT)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)


def directions():
    oracle = torch.zeros((128, 2048), dtype=torch.float32)
    oracle[:127, 0] = 1
    candidate = oracle.clone()
    return candidate, oracle


def five_candidates(mean_sign=1, split_sign=1):
    candidate, oracle = directions()
    artifact = {'mean_gradient': candidate*mean_sign,
                'batch_indices': torch.arange(32, dtype=torch.int64)}
    for name in m.CANDIDATES[1:]:
        artifact[name] = candidate*split_sign
    return artifact, oracle


class TestEvaluateAttempt026(unittest.TestCase):
    def test_embedded_frozen_spec_and_hashes(self):
        evaluation_spec = json.loads((FOLDER/'evaluation_spec.json').read_text())
        self.assertEqual(evaluation_spec.pop('evaluator_source_sha256'),
                         m.prior.sha256_file(SCRIPT))
        self.assertEqual(evaluation_spec, m.FROZEN_SPEC)
        self.assertEqual(m.prior.sha256_file(FOLDER/'spec.json'), m.CONSTRUCTION_SPEC_SHA256)
        self.assertEqual(m.prior.sha256_file(FOLDER/'construction-manifest.json'),
                         m.CONSTRUCTION_MANIFEST_SHA256)
        self.assertEqual(m.prior.sha256_file(PROJECT/'scripts/ablation/construct_direct_activation_gradient_pilot.py'),
                         m.CONSTRUCTOR_SHA256)

    def test_exact_signed_cosine_and_no_sign_flip(self):
        x = torch.tensor([1.0, 0.0], dtype=torch.float32)
        y = torch.tensor([-1.0, 0.0], dtype=torch.float32)
        self.assertEqual(m.signed_cosine(x, y), -1.0)
        self.assertIsNone(m.signed_cosine(torch.zeros_like(x), y))
        self.assertEqual(m.signed_cosine(-x, y), 1.0)

    def test_primary_is_mean_of_positions_not_flattened_cosine(self):
        candidate, oracle = directions()
        candidate[2].zero_()
        candidate[2, 1] = 10
        report = m.candidate_report('mean_gradient', candidate, oracle)
        self.assertEqual(report['positions_1_4_individual_cosines'], [1.0, 0.0, 1.0, 1.0])
        self.assertEqual(report['positions_1_4_mean_cosine'], 0.75)
        flattened = m.signed_cosine(candidate[1:5].reshape(-1), oracle[1:5].reshape(-1))
        self.assertNotAlmostEqual(flattened, report['positions_1_4_mean_cosine'])

    def test_primary_secondary_and_position_zero_are_separate(self):
        candidate, oracle = directions()
        candidate[0] *= -1
        candidate[4] *= -1
        candidate[10] *= -1
        report = m.candidate_report('mean_gradient', candidate, oracle)
        self.assertEqual(report['position_0_cosine'], -1.0)
        self.assertEqual(report['positions_1_4_individual_cosines'], [1, 1, 1, -1])
        self.assertEqual(report['positions_1_4_mean_cosine'], 0.5)
        self.assertEqual(report['positions_1_126_mean_cosine'], (124-2)/126)
        self.assertEqual(len(report['positions']), 128)

    def test_position127_is_null_and_nonzero_tail_rejected(self):
        candidate, oracle = directions()
        report = m.candidate_report('mean_gradient', candidate, oracle)
        self.assertIsNone(report['position127_cosine'])
        self.assertIsNone(report['positions'][127]['cosine_similarity'])
        self.assertEqual(report['position127_candidate_norm'], 0.0)
        candidate[127, 0] = 0.01
        with self.assertRaisesRegex(ValueError, 'Causally unscored'):
            m.candidate_report('mean_gradient', candidate, oracle)

    def test_all_five_tensors_evaluated_in_frozen_order(self):
        artifact, oracle = five_candidates()
        reports = m.evaluate_candidates(artifact, oracle)
        self.assertEqual([row['candidate'] for row in reports], list(m.CANDIDATES))
        self.assertTrue(all(row['positions_1_4_mean_cosine'] == 1.0 for row in reports))

    def test_no_best_split_selection(self):
        artifact, oracle = five_candidates(mean_sign=-1, split_sign=1)
        reports = m.evaluate_candidates(artifact, oracle)
        benchmark = {'consensus_primary': m.BENCHMARK_CONSENSUS,
                     'fineweb_primary': m.BENCHMARK_FINEWEB}
        result = m.build_result(reports, benchmark, {})
        self.assertEqual(result['scientific_candidate'], 'mean_gradient')
        self.assertEqual(result['matched_attempt014_comparison']['mean_gradient_primary'], -1.0)
        self.assertEqual(result['matched_attempt014_comparison']['delta_vs_attempt014_consensus'],
                         -1.0-m.BENCHMARK_CONSENSUS)
        self.assertTrue(all(row['positions_1_4_mean_cosine'] == 1.0 for row in reports[1:]))

    def test_serialized_artifact_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'activation_gradient.pt'
            path.write_bytes(b'synthetic wrong serialized artifact')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.load_candidates(path, {'raw_tensor_sha256': {}})

    def test_raw_candidate_tensor_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'activation_gradient.pt'
            artifact, _ = five_candidates()
            torch.save(artifact, path)
            metadata = {'raw_tensor_sha256': {
                name: m.prior.sha256_raw_float32_tensor(artifact[name], torch)
                for name in m.CANDIDATES}}
            metadata['raw_tensor_sha256']['mean_gradient'] = '0'*64
            with patch.object(m, 'require_hash'):
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    m.load_candidates(path, metadata)

    def test_construction_manifest_mismatch_rejected(self):
        spec = json.loads((FOLDER/'spec.json').read_text())
        manifest = json.loads((FOLDER/'construction-manifest.json').read_text())
        metadata = m.validate_construction(spec, manifest)
        self.assertEqual(metadata['serialized_sha256'], m.CANDIDATE_ARTIFACT_SHA256)
        changed = copy.deepcopy(manifest)
        changed['readout']['block_index'] = 12
        with self.assertRaisesRegex(ValueError, 'manifest/spec mismatch'):
            m.validate_construction(spec, changed)
        changed = copy.deepcopy(manifest)
        changed['artifact']['raw_tensor_sha256']['mean_gradient'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'candidate hash'):
            m.validate_construction(spec, changed)

    def test_oracle_artifact_and_provenance_hash_mismatch_rejected(self):
        oracle_manifest = {'oracle_adl_sha256': m.ORACLE_ARTIFACT_SHA256,
                           'raw_tensors_sha256': {'difference': m.ORACLE_DIFFERENCE_RAW_SHA256}}
        self.assertIs(m.validate_oracle_manifest(oracle_manifest), oracle_manifest)
        for field in ('oracle_adl_sha256', 'raw_tensors_sha256'):
            changed = copy.deepcopy(oracle_manifest)
            if field == 'oracle_adl_sha256':
                changed[field] = '0'*64
            else:
                changed[field]['difference'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'oracle artifact or difference'):
                m.validate_oracle_manifest(changed)

    def test_reused_oracle_lineage_validator_rejects_mismatch(self):
        paths = {name: Path(name) for name in
                 ('downloaded_manifest', 'merged_manifest', 'probe_manifest',
                  'oracle_manifest')}
        downloaded = {'hash_algorithm': 'sha256',
                      'models': {'base': {'repo_id': 'synthetic/base', 'revision': 'a'},
                                 'adapter': {'repo_id': 'synthetic/adapter', 'revision': 'b'}}}
        merged = {'hash_algorithm': 'sha256',
                  'downloaded_model_hashes_sha256': 'wrong-lineage',
                  'base': downloaded['models']['base'],
                  'adapter': downloaded['models']['adapter'],
                  'merge_script_sha256': 'b'*64,
                  'merge_metadata': {'dtype': 'float32', 'safe_merge': True,
                                     'tokenizer_source': 'base'}}
        records = {'downloaded_manifest': downloaded, 'merged_manifest': merged,
                   'probe_manifest': {}, 'oracle_manifest': {'hash_algorithm': 'sha256'}}
        hashes = {'downloaded_manifest': 'a'*64, 'merge_script': 'b'*64}
        with patch.object(m.prior, 'load_json_object',
                          side_effect=lambda path, description: records[path.name]):
            with self.assertRaisesRegex(ValueError, 'Merged-model oracle provenance mismatch'):
                m.prior.validate_oracle_provenance(paths, hashes, {})

    def test_attempt014_benchmark_hash_and_values_validated(self):
        path = PROJECT/m.FROZEN_SPEC['benchmark']['path']
        m.require_hash(path, m.BENCHMARK_SHA256)
        frozen = json.loads(path.read_text())
        self.assertEqual(m.validate_benchmark(frozen),
                         {'consensus_primary': m.BENCHMARK_CONSENSUS,
                          'fineweb_primary': m.BENCHMARK_FINEWEB})
        changed = copy.deepcopy(frozen)
        changed['primary']['consensus'] += 0.01
        with self.assertRaisesRegex(ValueError, 'benchmark mismatch'):
            m.validate_benchmark(changed)
        with tempfile.TemporaryDirectory() as temporary:
            wrong = Path(temporary)/'evaluation.json'
            wrong.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.require_hash(wrong, m.BENCHMARK_SHA256)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'evaluation.json'
            m.require_output_absent(output)
            output.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.require_output_absent(output)

    def test_freeze_firewall_declarations_in_output(self):
        artifact, oracle = five_candidates()
        reports = m.evaluate_candidates(artifact, oracle)
        benchmark = {'consensus_primary': m.BENCHMARK_CONSENSUS,
                     'fineweb_primary': m.BENCHMARK_FINEWEB}
        output = m.build_result(reports, benchmark, {'synthetic': True})
        for key in ('construction_frozen_before_oracle',):
            self.assertIs(output[key], True)
        for key in ('candidate_modified_after_oracle', 'sign_selection',
                    'candidate_selection', 'position_selection', 'candidate_rescaling'):
            self.assertIs(output[key], False)
        self.assertEqual(output['oracle_vector'], 'difference')
        self.assertEqual(output['cosine_dtype'], 'cpu_float64')
        self.assertEqual(output['split_aggregates'], 'descriptive_robustness_only')


if __name__ == '__main__':
    unittest.main()
