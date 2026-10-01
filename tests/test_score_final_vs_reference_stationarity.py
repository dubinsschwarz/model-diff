"""Synthetic tests for Attempt 102's blind final/reference diagnostic."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[1]/'scripts/ablation/score_final_vs_reference_stationarity.py'
loader = importlib.util.spec_from_file_location('attempt102_score_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


class ToyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self.options = []

    def encode(self, text, *, add_special_tokens):
        self.options.append(add_special_tokens)
        return [1, 2, 3] if text != 'long' else [1, 2, 3, 4]


class ToyFinal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32),
                                         requires_grad=False)
        self.flags = []
        self.eval()

    def forward(self, input_ids, attention_mask, use_cache):
        self.flags.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled(),
                           use_cache, attention_mask.tolist()))
        return type('Output', (), {'logits': torch.zeros((*input_ids.shape, 5))})()


def synthetic_cell(value, byte_count=1000):
    predicted = 256
    nll = value*byte_count
    return {'total_nll_nats': nll, 'total_utf8_bytes': byte_count,
            'nats_per_utf8_byte': value, 'bits_per_utf8_byte': value/math.log(2),
            'mean_nll_per_predicted_native_token': nll/predicted,
            'sequence_count': 128, 'total_native_tokens': predicted+128,
            'predicted_native_tokens': predicted,
            'mean_sequence_token_count': (predicted+128)/128,
            'elapsed_inference_seconds': 1.0}


def synthetic_inputs():
    first = [float(x) for x in range(1, 9)]
    second = [x+2.0 for x in first]
    final = [1.0, 4.0, 2.0, 8.0, 3.0, 9.0, 5.0, 7.0]
    ref = {'cells': {m.MODELS[0][0]: {name: synthetic_cell(first[i])
                                       for i, name in enumerate(m.ORDER)},
                     m.MODELS[1][0]: {name: synthetic_cell(second[i])
                                       for i, name in enumerate(m.ORDER)}},
           'statistics': {'corpora': [
               {'stationarity_score': -m.population_z(first)[i]}
               for i in range(8)]}}
    cells = {name: synthetic_cell(final[i]) for i, name in enumerate(m.ORDER)}
    pc1 = [float(x)-3.5 for x in range(8)]
    return cells, ref, pc1


class Attempt102Tests(unittest.TestCase):
    def test_spec_and_exact_frozen_inputs(self):
        self.assertEqual(m.prior.old.sha256_file(m.SPEC_PATH), m.SPEC_SHA256)
        self.assertEqual(SPEC['corpus_order'], list(m.ORDER))
        self.assertEqual(SPEC['frozen_sequence_indices'], list(range(128)))
        self.assertEqual(SPEC['attempt101']['result_sha256'],
                         '1ea5ea49952a74184ca99737df15f62c255dc1dd22eeb952bbe4a39ee117de41')
        self.assertEqual(SPEC['attempt101']['decoded_inventory_sha256'],
                         '2ceebf0873b4d98bde0be1112626d547cac0a7b4d840fb422f9815114c8e12e0')
        self.assertEqual(SPEC['canonical_final']['batch_size'], 16)

    def test_attempt101_result_hash_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'tampered.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.prior.require_hash(path, SPEC['attempt101']['result_sha256'])
        with patch.object(m.prior, 'require_hash', side_effect=ValueError('SHA256')):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.load_attempt101(SPEC)

    def test_attempt101_blind_result_provenance_and_firewall(self):
        with (patch.object(m.prior, 'require_hash'),
              patch.object(m.prior, 'validate_metadata')):
            old_spec, result = m.load_attempt101(SPEC)
        self.assertEqual(old_spec['corpus_order'], list(m.ORDER))
        self.assertEqual(result['provenance']['spec_sha256'],
                         SPEC['attempt101']['spec_sha256'])
        self.assertTrue(all(result[key] is False for key in old_spec['firewall']))

    def test_decoded_inventory_reconstruction_and_hash(self):
        inventory = {'texts': {name: ['exact']*128 for name in m.ORDER}}
        spec = copy.deepcopy(SPEC)
        digest = m.prior.sha256_bytes(m.prior.canonical_bytes(inventory))
        spec['attempt101']['decoded_inventory_sha256'] = digest
        with (patch.object(m.prior, 'validate_tokenizer_files', return_value=Path('/synthetic')),
              patch.object(m.prior, 'load_first_sequences', return_value={}),
              patch.object(m.prior, 'decode_inventory', return_value=inventory),
              patch.object(m.prior.old, 'load_json_object', return_value=inventory)):
            self.assertEqual(m.reconstruct_decoded_inventory(
                spec, SPEC, tokenizer_loader=lambda _: ToyTokenizer()), inventory)
            spec['attempt101']['decoded_inventory_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.reconstruct_decoded_inventory(
                    spec, SPEC, tokenizer_loader=lambda _: ToyTokenizer())

    def test_final_scoring_inference_only_and_no_gradients(self):
        model, tokenizer = ToyFinal(), ToyTokenizer()
        result = m.score_final_sequences(model, tokenizer, ['abc']*16, batch_size=16)
        self.assertEqual(result['sequence_count'], 16)
        self.assertEqual(result['total_native_tokens'], 48)
        self.assertEqual(result['predicted_native_tokens'], 32)
        self.assertTrue(all(flag is False for flag in tokenizer.options))
        self.assertEqual(len(model.flags), 1)
        self.assertEqual(model.flags[0][:3], (True, False, False))
        self.assertIsNone(model.weight.grad)

    def test_final_loader_freezes_parameters_and_disables_cache(self):
        model = ToyFinal()
        model.weight.requires_grad_(True)
        model.config = type('Config', (), {'use_cache': True})()
        with patch.object(m.prior.old, 'load_local_model', return_value=model) as load:
            returned = m.load_final_model(SPEC)
        self.assertIs(returned, model)
        self.assertFalse(model.training)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.weight.requires_grad)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(load.call_args.args[1], 'cuda')

    def test_causal_ce_masks_padding(self):
        model = ToyFinal()
        result = m.score_final_sequences(model, ToyTokenizer(),
                                         ['short', 'long']*8, batch_size=16)
        self.assertEqual(result['total_native_tokens'], 56)
        self.assertEqual(result['predicted_native_tokens'], 40)
        self.assertEqual(model.flags[0][3][0], [1, 1, 1, 0])
        self.assertAlmostEqual(result['total_nll_nats'], 40*math.log(5), places=5)

    def test_utf8_byte_normalization(self):
        texts = ['é猫']*16
        result = m.score_final_sequences(ToyFinal(), ToyTokenizer(), texts,
                                         batch_size=16)
        self.assertEqual(result['total_utf8_bytes'], 16*5)
        self.assertAlmostEqual(result['nats_per_utf8_byte'],
                               result['total_nll_nats']/(16*5))
        self.assertAlmostEqual(result['bits_per_utf8_byte'],
                               result['nats_per_utf8_byte']/math.log(2))

    def test_population_z_and_two_model_reference_mean(self):
        cells, ref, pc1 = synthetic_inputs()
        result = m.relative_statistics(cells, ref, pc1, 0.8)
        expected = m.population_z([float(x) for x in range(1, 9)])
        self.assertEqual(result['reference_ensemble_z'], expected)
        self.assertEqual(result['reference_model_z'][m.MODELS[0][0]], expected)
        self.assertAlmostEqual(sum(result['final_z']), 0)
        self.assertAlmostEqual(sum(x*x for x in result['final_z'])/8, 1)
        with self.assertRaisesRegex(ValueError, 'standard deviation'):
            m.population_z([1.0]*8)

    def test_deviation_and_fixed_primary_metric(self):
        cells, ref, pc1 = synthetic_inputs()
        result = m.relative_statistics(cells, ref, pc1, 0.8)
        expected = [a-b for a, b in zip(result['final_z'], result['reference_ensemble_z'])]
        self.assertEqual(result['deviation'], expected)
        self.assertEqual(result['primary_absolute_spearman'],
                         abs(m.prior.spearman(expected, pc1)))
        self.assertEqual(result['secondary_absolute_pearson'],
                         abs(m.prior.pearson(expected, pc1)))
        self.assertEqual(result['primary_metric'],
                         'absolute_spearman_simple_deviation_vs_blind_pc1')
        self.assertEqual([row['deviation_rank'] for row in result['corpora']],
                         m.prior.average_ranks(expected))

    def test_ols_residual_and_robustness_only(self):
        cells, ref, pc1 = synthetic_inputs()
        result = m.relative_statistics(cells, ref, pc1, 0.8)
        fit = result['regression_residual_robustness']
        for target, predictor, residual in zip(result['final_z'],
                                                result['reference_ensemble_z'],
                                                fit['residuals']):
            self.assertAlmostEqual(target-fit['intercept']-fit['slope']*predictor,
                                   residual, places=12)
        self.assertAlmostEqual(sum(fit['residuals']), 0, places=12)
        self.assertEqual(result['primary_absolute_spearman'],
                         abs(m.prior.spearman(result['deviation'], pc1)))

    def test_pc1_reuses_exact_blind_helper(self):
        responses = []
        for i in range(8):
            value = torch.zeros((128, 2048), dtype=torch.float32)
            value[:, 0] = 1 if i < 4 else -1
            responses.append(value)
        coords, explained = m.prior.pc1_coordinates(responses)
        self.assertAlmostEqual(explained, 1.0)
        self.assertEqual(len(coords), 8)
        self.assertLess(coords[0]*coords[-1], 0)

    def test_sign_invariant_correlations(self):
        cells, ref, pc1 = synthetic_inputs()
        positive = m.relative_statistics(cells, ref, pc1, 0.8)
        negative = m.relative_statistics(cells, ref, [-x for x in pc1], 0.8)
        self.assertEqual(positive['primary_absolute_spearman'],
                         negative['primary_absolute_spearman'])
        self.assertEqual(positive['secondary_absolute_pearson'],
                         negative['secondary_absolute_pearson'])

    def test_checkpoint_atomic_write_validation_and_resume(self):
        cell = synthetic_cell(1.25)
        meta = m.checkpoint_metadata(SPEC, m.ORDER[0], 'a'*64, 'b'*64, 'c'*64,
                                     cell['total_utf8_bytes'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fineweb.json'
            self.assertIsNone(m.load_or_store_cell(path, meta))
            self.assertEqual(m.load_or_store_cell(path, meta, cell), cell)
            self.assertEqual(m.load_or_store_cell(path, meta), cell)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            changed = copy.deepcopy(meta)
            changed['scoring_source_sha256'] = 'd'*64
            with self.assertRaisesRegex(ValueError, 'provenance'):
                m.load_or_store_cell(path, changed)
            changed = copy.deepcopy(meta)
            changed['total_utf8_bytes'] += 1
            with self.assertRaisesRegex(ValueError, 'provenance'):
                m.load_or_store_cell(path, changed)
            corrupted = json.loads(path.read_text())
            corrupted['result']['nats_per_utf8_byte'] = 9.0
            path.write_text(json.dumps(corrupted))
            with self.assertRaisesRegex(ValueError, 'result/hash'):
                m.load_or_store_cell(path, meta)

    def test_final_checkpoint_inventory_mismatch(self):
        with patch.object(m.prior.old, 'checkpoint_file_records', return_value=[]):
            with self.assertRaisesRegex(ValueError, 'inventory'):
                m.checkpoint_inventory(SPEC)

    def test_firewall_and_no_overwrite(self):
        for path in ('/models/base/x', '/adapter/x', '/oracle_adl/x',
                     '/base_mean.pt', '/ft_mean.pt', '/true_delta.pt',
                     '/attempt015/x', '/attempt025/x', '/foo/evaluation.json'):
            with self.assertRaisesRegex(ValueError, 'Forbidden'):
                m.safe_path(path)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            m.prior.atomic_new_json(path, {'ok': True})
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.prior.atomic_new_json(path, {'ok': False})
        self.assertTrue(all(value is False for value in SPEC['firewall'].values()))

    def test_smoke_writes_no_scientific_outputs(self):
        inventory = {'texts': {name: ['abc']*128 for name in m.ORDER}}
        fake = synthetic_cell(1.0)
        fake['predicted_native_tokens'] = 32
        with (patch.object(m, 'load_spec', return_value=SPEC),
              patch.object(m, 'load_attempt101', return_value=(
                  {'qwen_tokenizer': {'directory': '/synthetic'}}, {})),
              patch.object(m, 'reconstruct_decoded_inventory', return_value=inventory),
              patch.object(m, 'checkpoint_inventory', return_value='a'*64),
              patch.object(m, 'score_final_sequences', return_value=fake),
              patch.object(m.prior, 'atomic_new_json', side_effect=AssertionError('smoke wrote output'))):
            result = m.run(smoke_only=True, model_loader=lambda _: ToyFinal(),
                           tokenizer_loader=lambda _: ToyTokenizer())
        self.assertTrue(result['smoke_only'])
        self.assertFalse(result['writes'])
        self.assertEqual(result['projected_full_64_batch_scoring_seconds'],
                         result['batch_inference_seconds']*64)


if __name__ == '__main__':
    unittest.main()
