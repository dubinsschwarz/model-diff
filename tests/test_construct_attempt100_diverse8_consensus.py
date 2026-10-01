"""Synthetic checks for the blind Attempt 100 eight-response constructor."""
import copy
import importlib.util
import inspect
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/construct_attempt100_diverse8_consensus.py'
loader = importlib.util.spec_from_file_location('attempt100_construction_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.freezer.load_spec()


def responses():
    values = []
    for i in range(8):
        item = torch.zeros((128, 2048), dtype=torch.float32)
        item[:, i] = 1
        values.append(item)
    return values


def old_fixture():
    old_spec = m.old.FROZEN_SPEC
    values = {'merged_mean': torch.ones((128, 2048), dtype=torch.float32),
              **{'response_'+name: value for name, value in zip(m.ORDER[:3], responses()[:3])},
              'consensus_response': torch.ones((128, 2048), dtype=torch.float32)}
    raw = {name: m.old.sha256_raw_float32_tensor(value, torch) for name, value in values.items()}
    record = copy.deepcopy(SPEC)
    record['attempt014']['raw_sha256'] = {name: raw[name] for name in m.OLD_NAMES[:4]}
    manifest = {'attempt_id': old_spec['attempt_id'],
                'spec_sha256': record['attempt014']['spec_sha256'],
                'constructor_script_sha256': record['attempt014']['constructor_sha256'],
                'source_checkpoint': record['canonical_checkpoint_files'],
                'probe': record['probe'], 'generic_loss': record['generic_loss'],
                'jvp': record['jvp'], 'readout': record['readout'],
                'corpus_order': list(m.ORDER[:3]),
                'artifact': {'serialized_sha256': record['attempt014']['artifact_serialized_sha256'],
                             'raw_tensors_sha256': raw}}
    hashes = {str(m.freezer.resolve_path(record['attempt014'][key+'_path'])):
              record['attempt014'][key+'_sha256'] for key in ('spec', 'manifest', 'constructor')}
    hashes[str(m.freezer.resolve_path(record['attempt014']['artifact_path']))] = record['attempt014']['artifact_serialized_sha256']
    def hashed(path):
        return hashes[str(path)]
    def loaded(path, _):
        return old_spec if Path(path).name == 'spec.json' else manifest
    return record, values, manifest, hashed, loaded


class ConstructAttempt100Tests(unittest.TestCase):
    def test_frozen_order_and_artifact_inventory(self):
        self.assertEqual(m.ORDER, ('fineweb', 'wikitext103_raw', 'tinystories',
                                   'arxiv_document', 'cc_news', 'pg19',
                                   'codeparrot_clean', 'ultrachat'))
        self.assertEqual(SPEC['consensus']['corpus_order'], list(m.ORDER))
        self.assertEqual(SPEC['outputs']['tensors'],
                         ['merged_mean', *['response_'+n for n in m.ORDER], 'consensus_response_8'])

    def test_old_artifact_serialized_and_manifest_hash_validation(self):
        spec, artifact, manifest, hashed, loaded = old_fixture()
        with patch.object(m.old, 'sha256_file', side_effect=hashed), \
             patch.object(m.old, 'load_json_object', side_effect=loaded), \
             patch.object(torch, 'load', return_value=artifact):
            loaded_artifact, _ = m.load_old_responses(spec)
            self.assertIs(loaded_artifact, artifact)
            changed = copy.deepcopy(spec)
            changed['attempt014']['artifact_serialized_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'serialized/raw hash'):
                m.load_old_responses(changed)
            changed = copy.deepcopy(spec)
            changed['attempt014']['manifest_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'source/provenance hash'):
                m.load_old_responses(changed)

    def test_old_response_raw_hash_validation(self):
        spec, artifact, manifest, hashed, loaded = old_fixture()
        tampered = copy.deepcopy(artifact)
        tampered['response_fineweb'][1, 0] += 1
        with patch.object(m.old, 'sha256_file', side_effect=hashed), \
             patch.object(m.old, 'load_json_object', side_effect=loaded), \
             patch.object(torch, 'load', return_value=tampered):
            with self.assertRaisesRegex(ValueError, 'raw response hash'):
                m.load_old_responses(spec)

    def test_constructor_has_no_prior_evaluation_input(self):
        text = SOURCE.read_text()
        self.assertNotIn('evaluation.json', text)
        self.assertNotIn('evaluate_attempt014', text)
        self.assertEqual(SPEC['firewall']['prior_evaluation_access'], False)

    def test_exact_98_matrix_early_support(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(7):
                    setattr(self, f'linear_{i}', torch.nn.Linear(2, 2, bias=False))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList(Block() for _ in range(28))
        model = Model()
        selected = m.old.discover_eligible_linear_weights(model, torch)
        self.assertEqual(len(selected), 98)
        self.assertTrue(all(int(name.split('.')[2]) <= 13 for name, _ in selected))
        self.assertEqual([name for name, _ in selected], sorted(name for name, _ in selected))

    def test_positive_global_tangent_scale_and_realized_fp32_direction(self):
        modules = [('model.layers.0.a', torch.nn.Linear(2, 2, bias=False)),
                   ('model.layers.0.b', torch.nn.Linear(2, 2, bias=False))]
        for i, (_, module) in enumerate(modules):
            module.weight.data.fill_(2+i)
            module.weight.grad = torch.full_like(module.weight, float(i+1))
        scale = m.old.global_tangent_scale(modules, 0.00125, torch)
        tangent, _, realized = m.old.prepare_tangents(modules, scale, torch)
        self.assertGreater(scale['alpha'], 0)
        self.assertTrue(all(bool((value > 0).all()) for value in tangent.values()))
        self.assertAlmostEqual(realized['aggregate_realized_relative_tangent_norm'],
                               0.00125, places=7)

    def test_response_shape_dtype_and_per_position_normalization(self):
        candidate, diag = m.response_geometry(responses())
        self.assertEqual(tuple(candidate.shape), (128, 2048))
        self.assertEqual(candidate.dtype, torch.float32)
        self.assertEqual(candidate.device.type, 'cpu')
        self.assertTrue(candidate.is_contiguous())
        self.assertTrue(torch.allclose(candidate[:, :8], torch.full((128, 8), 0.125)))
        self.assertAlmostEqual(diag['consensus_concentration'][1], math.sqrt(8)/8)

    def test_exact_equal_eighths_no_final_renormalization(self):
        values = responses()
        values[0] *= 50
        candidate, _ = m.response_geometry(values)
        self.assertTrue(torch.allclose(candidate[:, :8], torch.full((128, 8), 0.125)))
        self.assertAlmostEqual(float(torch.linalg.vector_norm(candidate[1].double())),
                               math.sqrt(8)/8)
        self.assertFalse(SPEC['consensus']['renormalize_final'])
        self.assertEqual(SPEC['consensus']['weights'], [0.125]*8)

    def test_pairwise_diagnostics_never_select_corpora(self):
        candidate, diag = m.response_geometry(responses())
        self.assertEqual(len(diag['pairwise_per_position']), 128)
        self.assertEqual(len(diag['pairwise_per_position'][0]), 8)
        self.assertEqual(diag['pairwise_per_position'][0][0][0], 1.0)
        self.assertEqual(diag['pairwise_per_position'][0][0][1], 0.0)
        self.assertFalse(diag['selection'])
        self.assertEqual(tuple(diag['corpus_order']), m.ORDER)
        self.assertEqual(candidate[1, 0].item(), 0.125)

    def test_leave_one_out_and_prefix_are_descriptive_only(self):
        values = responses()
        candidate, diag = m.response_geometry(values)
        self.assertEqual(set(diag['leave_one_out_to_fixed_consensus']), set(m.ORDER))
        self.assertEqual(set(diag['prefix_to_fixed_consensus']), {'3', '4', '5', '6', '7', '8'})
        self.assertAlmostEqual(diag['prefix_to_fixed_consensus']['8']['positions_1_4'], 1.0)
        self.assertAlmostEqual(diag['leave_one_out_to_fixed_consensus']['fineweb']['positions_1_4'],
                               math.sqrt(7/8))
        self.assertTrue(torch.equal(candidate, m.response_geometry(values)[0]))

    def test_nonfinite_or_zero_response_fails_closed(self):
        values = responses()
        values[2][4].zero_()
        with self.assertRaisesRegex(ValueError, 'Zero or nonfinite'):
            m.response_geometry(values)
        values = responses()
        values[2][4, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite contiguous'):
            m.response_geometry(values)

    def test_forbidden_path_firewall_and_output_refusal(self):
        for path in ('/models/base/model.safetensors', '/oracle_adl/oracle_adl.pt',
                     '/adapter', '/attempt018/evaluation.json'):
            with self.assertRaisesRegex(ValueError, 'Forbidden'):
                m.assert_safe_inputs([path])
        with unittest.mock.patch.object(m.old, 'require_output_absent', side_effect=ValueError('exists')):
            with self.assertRaisesRegex(ValueError, 'exists'):
                m.no_outputs(Path('/tmp/a'), Path('/tmp/b'))

    def test_smoke_uses_synthetic_helpers_and_writes_no_scientific_output(self):
        lock = m.freezer.make_lock(SPEC, lambda _: 'a'*40)
        dummy_model = torch.nn.Module()
        old_artifact = {'merged_mean': torch.zeros((128, 2048)),
                        **{'response_'+name: torch.ones((128, 2048))
                           for name in m.ORDER[:3]}}
        with patch.object(m.freezer, 'load_spec', return_value=SPEC), \
             patch.object(m.old, 'sha256_file', return_value='a'*64), \
             patch.object(m.old, 'checkpoint_file_records', return_value=SPEC['canonical_checkpoint_files']), \
             patch.object(m, 'load_old_responses', return_value=(old_artifact, {})), \
             patch.object(m.old, 'load_json_object', return_value=lock), \
             patch.object(m.freezer, 'validate_lock', return_value=lock), \
             patch.object(m.freezer, 'validate_corpus', return_value=(torch.zeros((4096, 128), dtype=torch.int64), {'revision': 'a'*40})), \
             patch.object(m.old, 'load_probe', return_value=torch.zeros((10000, 128), dtype=torch.int64)), \
             patch.object(m.old, 'load_local_model', return_value=dummy_model), \
             patch.object(m.old, 'discover_eligible_linear_weights', return_value=[('x', object())]), \
             patch.object(m.old, 'freeze_other_parameters'), \
             patch.object(m.old, 'model_state_hashes', return_value={}), \
             patch.object(m.old, 'accumulate_mean_generic_gradient', return_value=1.0), \
             patch.object(m.old, 'global_tangent_scale', return_value={'aggregate_generic_gradient_norm': 2.0}), \
             patch.object(m.old, 'prepare_tangents', return_value=({}, [], {})), \
             patch.object(m.old, 'compute_probe_response', return_value=(torch.zeros((128, 2048)), torch.ones((128, 2048)), {})), \
             patch.object(m.old, 'verify_model_unchanged'), \
             patch.object(m, 'no_outputs', side_effect=AssertionError('smoke wrote outputs')):
            result = m.construct(smoke_only=True)
        self.assertTrue(result['smoke_only'])
        self.assertEqual(len(result['new_corpora']), 5)
        self.assertFalse(result['write_outputs'])


if __name__ == '__main__':
    unittest.main()
