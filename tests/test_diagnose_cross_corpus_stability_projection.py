"""Synthetic tests for Attempt118; no real response artifact or oracle is loaded."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/diagnose_cross_corpus_stability_projection.py'
SPEC = PROJECT/'experiments/attempts/118_postoracle_cross_corpus_stability_projection_diagnostic/spec.json'
_loader = importlib.util.spec_from_file_location('attempt118_tests', SOURCE)
m = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(m)


def unit_rows():
    rows = torch.zeros((8, 2048), dtype=torch.float64)
    rows[:7, 0] = 1
    rows[7, 1] = 1
    return rows


def synthetic_artifact():
    rows = unit_rows().float()
    artifact = {}
    artifact['merged_mean'] = torch.zeros((128, 2048), dtype=torch.float32)
    for name, row in zip(m.RESPONSE_NAMES, rows):
        artifact[name] = row.expand(128, -1).clone().contiguous()
    consensus = unit_rows().mean(dim=0).float()
    artifact['consensus_response_8'] = consensus.expand(128, -1).clone().contiguous()
    return artifact


def synthetic_baseline(primary=m.BASELINE_PRIMARY, secondary=m.BASELINE_SECONDARY):
    return {'attempt_id': m.PRIOR, 'scientific_candidate': 'consensus_response_8',
            'construction_frozen_before_oracle': True,
            'candidate_modified_after_oracle': False,
            'sign_selection': False, 'candidate_selection': False,
            'provenance': {'construction_artifact_sha256':
                           m.FROZEN_SPEC['attempt100']['artifact']['serialized_sha256']},
            'candidates': [{'candidate': 'consensus_response_8',
                            'positions_1_4_mean_cosine': primary,
                            'positions_1_127_mean_cosine': secondary}]}


class Attempt118Tests(unittest.TestCase):
    def test_spec_and_frozen_construction_provenance(self):
        self.assertEqual(json.loads(SPEC.read_text()), m.FROZEN_SPEC)
        plan = m.FROZEN_SPEC['attempt100']
        for key in ('spec', 'manifest', 'constructor'):
            self.assertEqual(m.sha256_file(m.path_of(plan[key]['path'])), plan[key]['sha256'])
        construction_spec = json.loads(m.path_of(plan['spec']['path']).read_text())
        manifest = json.loads(m.path_of(plan['manifest']['path']).read_text())
        m.validate_blind_provenance(plan, construction_spec, manifest)
        self.assertEqual(plan['evaluation']['sha256'],
                         '3607d599b347e7010902143655d2d68b0cba226237cb884b6751d1de998c08ef')

    def test_exact_eight_order_and_all_ten_raw_hashes_frozen(self):
        self.assertEqual(m.CORPORA, ('fineweb', 'wikitext103_raw', 'tinystories',
                                    'arxiv_document', 'cc_news', 'pg19',
                                    'codeparrot_clean', 'ultrachat'))
        self.assertEqual(m.RESPONSE_NAMES, tuple('response_'+x for x in m.CORPORA))
        plan = m.FROZEN_SPEC['attempt100']
        manifest = json.loads(m.path_of(plan['manifest']['path']).read_text())
        self.assertEqual(manifest['artifact']['raw_tensors_sha256'], m.RAW_HASHES)
        self.assertEqual(plan['artifact']['tensor_inventory'], list(m.ARTIFACT_NAMES))
        self.assertEqual(len(m.RAW_HASHES), 10)

    def test_blind_provenance_rejects_manifest_or_firewall_change(self):
        plan = m.FROZEN_SPEC['attempt100']
        construction_spec = json.loads(m.path_of(plan['spec']['path']).read_text())
        manifest = json.loads(m.path_of(plan['manifest']['path']).read_text())
        bad = copy.deepcopy(manifest)
        bad['artifact']['raw_tensors_sha256']['response_fineweb'] = '0'*64
        with patch.object(m, 'require_hash'):
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                m.validate_blind_provenance(plan, construction_spec, bad)
        bad = copy.deepcopy(manifest)
        bad['oracle_adl_access'] = True
        with patch.object(m, 'require_hash'):
            with self.assertRaisesRegex(ValueError, 'firewall'):
                m.validate_blind_provenance(plan, construction_spec, bad)

    def test_artifact_serialized_hash_inventory_and_raw_hash_checks(self):
        artifact = synthetic_artifact()
        hashes = {name: m.raw_sha256(tensor) for name, tensor in artifact.items()}
        manifest = {'artifact': {'raw_tensors_sha256': hashes}}
        plan = {'artifact': {'path': '/synthetic/responses.pt', 'serialized_sha256': '0'*64}}
        with patch.object(m, 'require_hash'), patch.object(m.torch, 'load', return_value=artifact):
            self.assertIs(m.load_blind_responses(plan, manifest), artifact)
            bad = copy.deepcopy(manifest)
            bad['artifact']['raw_tensors_sha256']['response_fineweb'] = 'f'*64
            with self.assertRaisesRegex(ValueError, 'raw SHA256'):
                m.load_blind_responses(plan, bad)
            short = dict(artifact)
            del short['response_ultrachat']
            with patch.object(m.torch, 'load', return_value=short):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    m.load_blind_responses(plan, manifest)

    def test_response_dtype_shape_contiguity_and_finite_guard(self):
        valid = torch.zeros((128, 2048), dtype=torch.float32)
        m.validate_tensor(valid)
        for invalid in (valid.double(), valid[:127], valid.T, valid.clone().fill_(float('nan'))):
            with self.assertRaisesRegex(ValueError, 'Response must'):
                m.validate_tensor(invalid)

    def test_positionwise_unit_normalization_and_equal_mean(self):
        artifact = synthetic_artifact()
        stable, overlap, rows = m.construct_stability(artifact)
        self.assertEqual(stable.shape, (128, 2048))
        self.assertEqual(stable.dtype, torch.float32)
        self.assertTrue(stable.is_contiguous())
        self.assertEqual(rows[0]['consensus_norm'], float(torch.linalg.vector_norm(unit_rows().mean(0))))
        self.assertLess(rows[0]['max_abs_centered_sum'], 1e-12)
        self.assertLess(rows[0]['consensus_reproduction_max_abs'], 1e-7)
        self.assertLess(float((stable.double()+overlap.double()-unit_rows().mean(0)).abs().max()), 1e-7)

    def test_consensus_reproduction_fails_on_mismatch(self):
        rows = unit_rows()
        with self.assertRaisesRegex(ValueError, 'consensus reproduction'):
            m.project_position(rows, torch.zeros(2048, dtype=torch.float32))

    def test_k_eigenvalues_match_explicit_ddt_cpu_float64(self):
        rows = unit_rows()
        consensus = rows.mean(0).float()
        _, _, report = m.project_position(rows, consensus)
        d = rows - rows.mean(0)
        k = d @ d.T
        expected = torch.linalg.eigvalsh((k+k.T)/2)
        self.assertEqual(len(report['k_eigenvalues_ascending']), 8)
        torch.testing.assert_close(torch.tensor(report['k_eigenvalues_ascending'], dtype=torch.float64), expected)
        self.assertEqual(report['numerical_rank'], 1)
        self.assertEqual(report['rank_tolerance'], torch.finfo(torch.float64).eps*8*float(expected[-1]))

    def test_moore_penrose_projection_and_stable_orthogonality(self):
        rows = unit_rows()
        mean = rows.mean(0)
        d = rows - mean
        k = d @ d.T
        stable, projection, report = m.project_position(rows, mean.float())
        expected = d.T @ (torch.linalg.pinv(k, hermitian=True) @ (d @ mean))
        torch.testing.assert_close(projection, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(stable, mean - expected, atol=1e-12, rtol=1e-12)
        self.assertLess(report['max_abs_d_dot_stable'], 1e-12)
        self.assertAlmostEqual(report['stable_to_consensus_norm_ratio'],
                               float(torch.linalg.vector_norm(stable)/torch.linalg.vector_norm(mean)))

    def test_numerical_rank_rule_is_fixed_and_oracle_independent(self):
        self.assertEqual(m.FROZEN_SPEC['projection']['rank_rule'],
                         'retain_eigenvalues_strictly_greater_than_eps_float64_times_8_times_max_eigenvalue')
        self.assertFalse(m.FROZEN_SPEC['projection']['rank_selection_after_oracle'])
        self.assertFalse(m.FROZEN_SPEC['projection']['top_k_or_partial_projection_sweep'])
        _, _, report = m.project_position(unit_rows(), unit_rows().mean(0).float())
        self.assertTrue(all(math.isfinite(x) for x in report['k_eigenvalues_ascending']))

    def test_stable_not_renormalized_and_both_responses_hashed(self):
        artifact = synthetic_artifact()
        stable, overlap, diagnostics = m.construct_stability(artifact)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            record = m.freeze_construction_record(m.FROZEN_SPEC, diagnostics, stable, overlap)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(stable[0])), math.sqrt(.5), places=7)
        self.assertFalse(m.FROZEN_SPEC['projection']['renormalize_stable'])
        self.assertEqual(record['stable_response_raw_sha256'], m.raw_sha256(stable))
        self.assertEqual(record['disagreement_overlap_raw_sha256'], m.raw_sha256(overlap))
        self.assertIn('STABILITY_CONSTRUCTION_FROZEN', output.getvalue())
        self.assertIn('stable_raw_sha256='+m.raw_sha256(stable), output.getvalue())
        self.assertNotIn('evaluation', record['attempt100_provenance'])

    def test_oracle_access_is_after_candidate_hash_barrier(self):
        artifact = synthetic_artifact()
        seen = []
        class BarrierReached(Exception):
            pass
        def oracle_access():
            seen.append('oracle')
            raise BarrierReached
        real_freeze = m.freeze_construction_record
        def barrier(*args):
            value = real_freeze(*args)
            seen.append(('barrier', value['stable_response_raw_sha256'],
                         value['disagreement_overlap_raw_sha256']))
            return value
        with patch.object(m, 'validate_blind_provenance'), \
             patch.object(m, 'load_blind_responses', return_value=artifact), \
             patch.object(m, 'freeze_construction_record', side_effect=barrier), \
             patch.object(m, 'load_validated_oracle', side_effect=oracle_access), \
             patch.object(m, 'load_validated_baseline') as baseline, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(BarrierReached):
                m.run(SPEC)
            baseline.assert_not_called()
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0][0], 'barrier')
        self.assertEqual(seen[1], 'oracle')
        self.assertFalse(m.path_of(m.FROZEN_SPEC['output_path']).exists())

    def test_oracle_validator_source_pinned_without_import_at_construction(self):
        self.assertNotIn('evaluate_attempt014', m.__dict__)
        self.assertEqual(m.sha256_file(m.path_of(m.FROZEN_SPEC['oracle']['validator_source']['path'])),
                         m.ORACLE_VALIDATOR_SHA)
        with patch.object(m, 'require_hash', side_effect=ValueError('SHA256 mismatch')):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.load_oracle_validator()

    def test_oracle_artifact_and_raw_provenance_rejection(self):
        fake = type('Validator', (), {})()
        fake.FROZEN_SPEC = {'inputs': {'attempt_spec': {'path': '/synthetic/a', 'sha256': '0'*64}}}
        fake.require_file_hash = lambda *args: None
        fake.validate_construction = lambda *args: ({'probe': {}}, {})
        fake.validate_oracle_provenance = lambda *args: {
            'oracle_adl_sha256': '0'*64,
            'raw_tensors_sha256': {'difference': m.FROZEN_SPEC['oracle']['difference_raw_sha256']}}
        with patch.object(m, 'load_oracle_validator', return_value=fake):
            with self.assertRaisesRegex(ValueError, 'oracle artifact/raw'):
                m.load_validated_oracle()

    def test_baseline_hash_and_exact_primary_secondary_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'evaluation.json'
            value = synthetic_baseline()
            path.write_text(json.dumps(value))
            frozen = copy.deepcopy(m.FROZEN_SPEC)
            frozen['attempt100']['evaluation']['path'] = str(path)
            frozen['attempt100']['evaluation']['sha256'] = m.sha256_file(path)
            with patch.object(m, 'FROZEN_SPEC', frozen):
                self.assertEqual(m.load_validated_baseline()['primary'], m.BASELINE_PRIMARY)
                value['candidates'][0]['positions_1_4_mean_cosine'] = .5
                path.write_text(json.dumps(value))
                frozen['attempt100']['evaluation']['sha256'] = m.sha256_file(path)
                with self.assertRaisesRegex(ValueError, 'baseline mismatch'):
                    m.load_validated_baseline()

    def test_signed_per_position_cosines_primary_secondary_and_zero(self):
        target = torch.zeros((128, 2048), dtype=torch.float32)
        target[:, 0] = 1
        candidate = target.clone()
        candidate[0, 0] = -1
        candidate[2, 0] = -1
        report = m.cosine_report(candidate, target)
        self.assertEqual(report['position_0_cosine'], -1)
        self.assertEqual(report['positions_1_4_individual_cosines'], [1, -1, 1, 1])
        self.assertEqual(report['positions_1_4_mean_cosine'], .5)
        self.assertEqual(report['positions_1_127_mean_cosine'], 125/127)
        self.assertEqual(len(report['all_128_position_cosines']), 128)
        self.assertIsNone(m.cosine_report(candidate.index_fill(0, torch.tensor([127]), 0), target)
                          ['positions_1_127_mean_cosine'])

    def test_interpretation_precedence_and_next_step_policy(self):
        self.assertEqual(m.interpret(.03, .02), 'clear_stability_separation_support')
        self.assertEqual(m.interpret(.015, 0), 'moderate_stability_separation_support')
        self.assertEqual(m.interpret(.03, -.01), 'no_meaningful_stability_separation_support')
        self.assertEqual(m.interpret(None, .02), 'undefined_required_position_cosine')
        self.assertEqual(len(m.FROZEN_SPEC['interpretation']['next_step']), 3)

    def test_no_selection_sweeps_model_or_parameter_pipeline(self):
        policy = m.FROZEN_SPEC['information_policy']
        self.assertTrue(policy['same_specimen_postoracle_method_development'])
        self.assertFalse(policy['clean_blind_validation'])
        for key in ('sign_selection', 'corpus_selection', 'corpus_reweighting',
                    'rank_selection_after_oracle', 'candidate_selection',
                    'parameter_space_candidate', 'model_loading',
                    'gradient_jvp_ggn_computation', 'posthoc_scaling', 'position_selection'):
            self.assertFalse(policy[key])
        text = SOURCE.read_text()
        for forbidden in ('AutoModelForCausalLM', 'from_pretrained(', 'torch.func.jvp(',
                          'autograd.grad(', 'models/base', 'top_k_sweep'):
            self.assertNotIn(forbidden, text)

    def test_no_result_overwrite_and_no_large_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'result.json'
            path.write_text('existing')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.require_output_absent(path)
            self.assertEqual(path.read_text(), 'existing')
        self.assertFalse(m.FROZEN_SPEC['output_policy']['large_artifact'])
        self.assertTrue((PROJECT/'env.sh').exists())


if __name__ == '__main__':
    unittest.main()
