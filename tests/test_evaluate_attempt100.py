"""Synthetic tests for Attempt 100's frozen post-construction evaluator."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/evaluate_attempt100.py'
FOLDER = PROJECT/'experiments/attempts/100_diverse8_raw_jg_consensus_prefix0_13'
loader = importlib.util.spec_from_file_location('attempt100_evaluation_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)


def directions():
    difference = torch.zeros((128, 2048), dtype=torch.float32)
    difference[:, 0] = 1
    return difference.clone(), difference


def artifact(consensus_sign=1, individuals_sign=1):
    candidate, difference = directions()
    rows = {'merged_mean': torch.ones_like(candidate),
            m.SCIENTIFIC: candidate*consensus_sign}
    for name in m.INDIVIDUALS:
        rows[name] = candidate*individuals_sign
    return rows, difference


def frozen_construction():
    records = m.FROZEN_SPEC['construction_inputs']
    spec = json.loads((PROJECT/records['spec']['path']).read_text())
    manifest = json.loads((PROJECT/records['manifest']['path']).read_text())
    lock = json.loads((PROJECT/records['corpus_lock']['path']).read_text())
    corpora = [json.loads((PROJECT/row['path']).read_text())
               for row in m.FROZEN_SPEC['corpus_manifests']]
    return spec, manifest, lock, corpora


class EvaluateAttempt100Tests(unittest.TestCase):
    def test_frozen_evaluation_spec_source_hash_and_construction_hashes(self):
        spec = json.loads((FOLDER/'evaluation_spec.json').read_text())
        self.assertEqual(spec.pop('evaluator_source_sha256'), m.prior.sha256_file(SOURCE))
        self.assertEqual(spec, m.FROZEN_SPEC)
        for row in spec['construction_inputs'].values():
            self.assertEqual(m.prior.sha256_file(m.path_of(row['path'])), row['sha256'])
        for row in spec['corpus_manifests']:
            self.assertEqual(m.prior.sha256_file(m.path_of(row['path'])), row['sha256'])

    def test_exact_signed_cosine_without_flip_or_absolute_value(self):
        candidate, difference = directions()
        candidate[2] *= -1
        values = m.position_cosines(candidate, difference)
        self.assertEqual(values[1], 1.0)
        self.assertEqual(values[2], -1.0)
        self.assertEqual(values[3], 1.0)

    def test_primary_is_mean_of_positions_not_flattened(self):
        candidate, difference = directions()
        candidate[2].zero_()
        candidate[2, 1] = 10
        row = m.report(m.SCIENTIFIC, candidate, difference)
        self.assertEqual(row['positions_1_4_individual_cosines'], [1, 0, 1, 1])
        self.assertEqual(row['positions_1_4_mean_cosine'], 0.75)
        x, y = candidate[1:5].reshape(-1).double(), difference[1:5].reshape(-1).double()
        flattened = float(torch.dot(x, y)/(torch.linalg.vector_norm(x)*torch.linalg.vector_norm(y)))
        self.assertNotAlmostEqual(flattened, row['positions_1_4_mean_cosine'])

    def test_primary_secondary_and_position_zero_are_separate(self):
        candidate, difference = directions()
        candidate[0] *= -1
        candidate[4] *= -1
        candidate[126] *= -1
        row = m.report(m.SCIENTIFIC, candidate, difference)
        self.assertEqual(row['position_0_cosine'], -1)
        self.assertEqual(row['positions_1_4_individual_cosines'], [1, 1, 1, -1])
        self.assertEqual(row['positions_1_4_mean_cosine'], 0.5)
        self.assertEqual(row['positions_1_127_mean_cosine'], 123/127)
        self.assertEqual(len(row['positions']), 128)
        self.assertEqual(row['positions'][127]['cosine_similarity'], 1)

    def test_zero_norm_returns_null_without_epsilon(self):
        candidate, difference = directions()
        candidate[127].zero_()
        row = m.report(m.SCIENTIFIC, candidate, difference)
        self.assertIsNone(row['positions'][127]['cosine_similarity'])
        self.assertIsNone(row['positions_1_127_mean_cosine'])
        self.assertEqual(row['positions_1_4_mean_cosine'], 1.0)

    def test_all_eight_individual_responses_evaluated_in_frozen_order(self):
        rows, difference = artifact()
        reports = m.report_all(rows, difference)
        self.assertEqual([row['candidate'] for row in reports], list(m.REPORT_ORDER))
        self.assertEqual(len(reports), 9)
        self.assertTrue(all(row['positions_1_4_mean_cosine'] == 1 for row in reports))
        self.assertEqual([row['role'] for row in reports],
                         ['scientific_candidate']+['descriptive_individual_corpus']*8)

    def test_only_consensus_is_scientific_even_if_individuals_better(self):
        rows, difference = artifact(consensus_sign=-1, individuals_sign=1)
        reports = m.report_all(rows, difference)
        benchmark = {'consensus_primary': m.BENCHMARK_PRIMARY,
                     'fineweb_primary': m.BENCHMARK_FINEWEB_PRIMARY,
                     'consensus_secondary': 0.4864011909288318}
        output = m.build_result(reports, benchmark, {})
        self.assertEqual(output['scientific_candidate'], m.SCIENTIFIC)
        self.assertEqual(output['matched_attempt014_comparison']['attempt100_primary'], -1)
        self.assertTrue(all(row['positions_1_4_mean_cosine'] == 1 for row in reports[1:]))
        self.assertFalse(output['candidate_selection'])

    def test_no_corpus_reweighting_or_reconstruction_during_evaluation(self):
        rows, difference = artifact()
        rows[m.SCIENTIFIC].zero_()
        rows[m.SCIENTIFIC][:, 1] = 1
        reports = m.report_all(rows, difference)
        self.assertEqual(reports[0]['positions_1_4_mean_cosine'], 0)
        self.assertTrue(all(row['positions_1_4_mean_cosine'] == 1 for row in reports[1:]))
        self.assertEqual(m.FROZEN_SPEC['metrics']['scientific_candidate'], m.SCIENTIFIC)

    def test_serialized_artifact_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'responses.pt'
            path.write_bytes(b'wrong serialized artifact')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.load_construction_artifact(path)

    def test_all_raw_tensor_hashes_and_inventory_checked(self):
        rows, _ = artifact()
        hashes = {name: m.prior.sha256_raw_float32_tensor(value, torch)
                  for name, value in rows.items()}
        with patch.object(m, 'require_hash'), \
             patch.object(m.prior, 'load_torch_artifact', return_value=rows), \
             patch.object(m, 'RAW_HASHES', hashes):
            self.assertIs(m.load_construction_artifact(Path('/synthetic/responses.pt')), rows)
            wrong = copy.deepcopy(hashes)
            wrong['response_pg19'] = '0'*64
            with patch.object(m, 'RAW_HASHES', wrong):
                with self.assertRaisesRegex(ValueError, 'raw tensor SHA-256'):
                    m.load_construction_artifact(Path('/synthetic/responses.pt'))
            incomplete = dict(rows)
            del incomplete['response_ultrachat']
            with patch.object(m.prior, 'load_torch_artifact', return_value=incomplete):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    m.load_construction_artifact(Path('/synthetic/responses.pt'))

    def test_construction_manifest_and_lock_mismatch_rejected(self):
        spec, manifest, lock, corpora = frozen_construction()
        self.assertEqual(m.validate_construction(spec, manifest, lock, corpora)['serialized_sha256'],
                         m.ARTIFACT_SHA256)
        wrong = copy.deepcopy(manifest)
        wrong['artifact']['raw_tensors_sha256']['consensus_response_8'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'artifact hash'):
            m.validate_construction(spec, wrong, lock, corpora)
        wrong = copy.deepcopy(lock)
        wrong['corpora'][0]['revision'] = '0'*40
        with self.assertRaisesRegex(ValueError, 'corpus provenance'):
            m.validate_construction(spec, manifest, wrong, corpora)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'construction-manifest.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.require_hash(path, m.MANIFEST_SHA256)

    def test_oracle_artifact_and_raw_difference_hash_rejected(self):
        manifest = {'oracle_adl_sha256': m.ORACLE_SHA256,
                    'raw_tensors_sha256': {'difference': m.ORACLE_DIFFERENCE_SHA256}}
        self.assertIs(m.validate_oracle_manifest(manifest), manifest)
        for field in ('oracle_adl_sha256', 'difference'):
            changed = copy.deepcopy(manifest)
            if field == 'difference':
                changed['raw_tensors_sha256']['difference'] = '0'*64
            else:
                changed[field] = '0'*64
            with self.assertRaisesRegex(ValueError, 'oracle artifact/difference'):
                m.validate_oracle_manifest(changed)

    def test_reused_oracle_lineage_validator_rejects_mismatch(self):
        paths = {name: Path(name) for name in ('downloaded_manifest', 'merged_manifest',
                                              'probe_manifest', 'oracle_manifest')}
        identities = {'base': {'repo_id': 'synthetic/base', 'revision': 'a'},
                      'adapter': {'repo_id': 'synthetic/adapter', 'revision': 'b'}}
        records = {'downloaded_manifest': {'hash_algorithm': 'sha256', 'models': identities},
                   'merged_manifest': {'hash_algorithm': 'sha256',
                                       'downloaded_model_hashes_sha256': 'incorrect',
                                       'base': identities['base'], 'adapter': identities['adapter'],
                                       'merge_script_sha256': 'b'*64,
                                       'merge_metadata': {'dtype': 'float32', 'safe_merge': True,
                                                          'tokenizer_source': 'base'}},
                   'probe_manifest': {}, 'oracle_manifest': {'hash_algorithm': 'sha256'}}
        with patch.object(m.prior, 'load_json_object',
                          side_effect=lambda path, description: records[path.name]):
            with self.assertRaisesRegex(ValueError, 'Merged-model oracle provenance mismatch'):
                m.prior.validate_oracle_provenance(paths,
                    {'downloaded_manifest': 'a'*64, 'merge_script': 'b'*64}, {})

    def test_attempt014_benchmark_secondary_read_exact_and_mismatch_rejected(self):
        path = m.path_of(m.FROZEN_SPEC['benchmark']['path'])
        m.require_hash(path, m.BENCHMARK_SHA256)
        benchmark = json.loads(path.read_text())
        result = m.validate_benchmark(benchmark)
        self.assertEqual(result['consensus_primary'], 0.4434394116233976)
        self.assertEqual(result['fineweb_primary'], 0.4420842139546442)
        self.assertEqual(result['consensus_secondary'], 0.4864011909288318)
        wrong = copy.deepcopy(benchmark)
        wrong['secondary']['consensus'] += 0.01
        with self.assertRaisesRegex(ValueError, 'benchmark values'):
            m.validate_benchmark(wrong)
        wrong = copy.deepcopy(benchmark)
        wrong['provenance']['oracle_artifact_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'identity/provenance'):
            m.validate_benchmark(wrong)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'evaluation.json'
            m.require_output_absent(path)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.require_output_absent(path)

    def test_output_freeze_and_firewall_declarations(self):
        rows, difference = artifact()
        reports = m.report_all(rows, difference)
        result = m.build_result(reports,
            {'consensus_primary': m.BENCHMARK_PRIMARY,
             'fineweb_primary': m.BENCHMARK_FINEWEB_PRIMARY,
             'consensus_secondary': 0.4864011909288318}, {})
        self.assertTrue(result['construction_frozen_before_oracle'])
        for key in ('candidate_modified_after_oracle', 'sign_selection',
                    'candidate_selection', 'corpus_selection', 'corpus_reweighting',
                    'candidate_rescaling', 'position_selection'):
            self.assertIs(result[key], False)
        self.assertEqual(result['oracle_vector'], 'difference')
        self.assertEqual(result['cosine_dtype'], 'cpu_float64')
        self.assertEqual(result['individual_corpora'], 'descriptive_only')


if __name__ == '__main__':
    unittest.main()
