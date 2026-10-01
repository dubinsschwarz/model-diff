"""Synthetic tests for Attempt 101's blind external-reference diagnostic."""
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
SOURCE = PROJECT/'scripts/ablation/score_pretrained_reference_stationarity.py'
loader = importlib.util.spec_from_file_location('attempt101_score_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


class DecodeTokenizer:
    def __init__(self):
        self.calls = []

    def batch_decode(self, rows, **kwargs):
        self.calls.append(kwargs)
        return [','.join(map(str, row)) for row in rows]


class NativeTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def encode(self, text, add_special_tokens):
        if add_special_tokens:
            raise AssertionError('Reference tokenizer added special tokens')
        return [1, 2, 3]


class ToyReference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.inference_flags = []
        self.eval()

    def forward(self, input_ids, attention_mask, use_cache):
        self.inference_flags.append((torch.is_inference_mode_enabled(),
                                     torch.is_grad_enabled(), use_cache,
                                     attention_mask.shape == input_ids.shape))
        return type('Output', (), {'logits': torch.zeros((*input_ids.shape, 5))})()


class FragmentedTokenizer(NativeTokenizer):
    def encode(self, text, add_special_tokens):
        if add_special_tokens:
            raise AssertionError('Reference tokenizer added special tokens')
        return [1, 4, 2, 3]


class VariableReference(ToyReference):
    def forward(self, input_ids, attention_mask, use_cache):
        logits = torch.zeros((*input_ids.shape, 5))
        logits[..., 2] = input_ids.float()
        return type('Output', (), {'logits': logits})()


def synthetic_cells():
    cells = {}
    byte_rates = [1.0, 2.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]
    for model, _ in m.MODELS:
        cells[model] = {name: {'total_nll_nats': 1000*(i+1)*byte_rates[i],
                               'total_utf8_bytes': 1000*(i+1),
                               'nats_per_utf8_byte': byte_rates[i],
                               'bits_per_utf8_byte': byte_rates[i]/math.log(2),
                               'mean_nll_per_predicted_native_token': float(i+1),
                               'sequence_count': 128,
                               'total_native_tokens': 1000*int(byte_rates[i])+128,
                               'predicted_native_tokens': 1000*int(byte_rates[i]),
                               'mean_sequence_token_count': (1000*byte_rates[i]+128)/128,
                               'elapsed_inference_seconds': 1.0}
                        for i, name in enumerate(m.ORDER)}
    return cells


class ScoreAttempt101Tests(unittest.TestCase):
    def test_spec_model_and_exact_corpus_inventory(self):
        self.assertEqual(m.old.sha256_file(m.SPEC_PATH), m.SPEC_SHA256)
        self.assertEqual(tuple(SPEC['corpus_order']), m.ORDER)
        self.assertEqual(tuple((row['name'], row['repo_id']) for row in SPEC['reference_models']), m.MODELS)
        self.assertEqual(SPEC['frozen_sequence_indices'], list(range(128)))
        self.assertTrue(all(row['variant'] == 'base' for row in SPEC['reference_models']))
        self.assertFalse(any('instruct' in row['repo_id'].lower() for row in SPEC['reference_models']))

    def test_local_frozen_manifest_provenance_without_root_artifacts(self):
        manifest = m.validate_metadata(SPEC)
        self.assertEqual(manifest['artifact']['serialized_sha256'],
                         SPEC['attempt100']['artifact_serialized_sha256'])
        self.assertEqual(len(manifest['corpus_order']), 8)
        wrong = copy.deepcopy(SPEC)
        wrong['corpora'][4]['manifest_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            m.validate_metadata(wrong)

    def test_first_128_are_taken_from_each_hashed_token_artifact(self):
        seen = []
        full = torch.arange(4096, dtype=torch.int64)[:, None].repeat(1, 128)
        with patch.object(m, 'require_hash', side_effect=lambda path, digest: seen.append((path, digest))), \
             patch.object(m.old, 'load_corpus_tokens', return_value=full):
            rows = m.load_first_sequences(SPEC)
        self.assertEqual(list(rows), list(m.ORDER))
        self.assertEqual(len(seen), 8)
        self.assertEqual([digest for _, digest in seen],
                         [row['serialized_sha256'] for row in SPEC['corpora']])
        for name in m.ORDER:
            self.assertEqual(tuple(rows[name].shape), (128, 128))
            self.assertEqual(rows[name][0, 0].item(), 0)
            self.assertEqual(rows[name][127, 0].item(), 127)

    def test_frozen_token_artifact_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'tokens.pt'
            path.write_bytes(b'incorrect synthetic token artifact')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.require_hash(path, SPEC['corpora'][0]['serialized_sha256'])

    def test_deterministic_decode_inventory_and_options(self):
        tokens = {name: torch.full((128, 128), i, dtype=torch.int64)
                  for i, name in enumerate(m.ORDER)}
        tokenizer = DecodeTokenizer()
        inventory = m.decode_inventory(tokens, tokenizer, SPEC)
        self.assertEqual(inventory['corpus_order'], list(m.ORDER))
        self.assertEqual(inventory['sequence_indices'], list(range(128)))
        self.assertEqual(len(inventory['texts']), 8)
        self.assertEqual(inventory['texts']['cc_news'][0], ','.join(['4']*128))
        self.assertTrue(all(call == {'skip_special_tokens': True,
                                      'clean_up_tokenization_spaces': False}
                            for call in tokenizer.calls))
        self.assertEqual(m.canonical_bytes(inventory), m.canonical_bytes(
            m.decode_inventory(tokens, DecodeTokenizer(), SPEC)))

    def test_decoded_inventory_freeze_hash_no_overwrite_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = copy.deepcopy(SPEC)
            spec['decoded_inventory_path'] = str(Path(temporary)/'decoded.json')
            inventory = {'texts': {'fineweb': ['synthetic']}}
            digest = m.freeze_decoded_inventory(inventory, spec, smoke_only=False)
            self.assertEqual(digest, m.old.sha256_file(Path(spec['decoded_inventory_path'])))
            self.assertEqual(m.freeze_decoded_inventory(inventory, spec, smoke_only=False), digest)
            with self.assertRaisesRegex(ValueError, 'changed'):
                m.freeze_decoded_inventory({'texts': {'fineweb': ['different']}}, spec,
                                           smoke_only=False)
            self.assertEqual(len(list(Path(temporary).iterdir())), 1)

    def test_reference_revision_lock_and_no_head_refresh(self):
        lock = m.make_lock(SPEC, lambda _: 'a'*40)
        self.assertEqual(m.validate_lock(lock, SPEC), lock)
        for bad in ('main', 'a'*39, 'A'*40, 'g'*40):
            with self.assertRaisesRegex(ValueError, 'immutable'):
                m.require_revision(bad)
        with tempfile.TemporaryDirectory() as temporary:
            spec = copy.deepcopy(SPEC)
            spec['reference_lock_path'] = str(Path(temporary)/'lock.json')
            m.atomic_new_json(Path(spec['reference_lock_path']), lock)
            reused, digest = m.get_lock(spec, smoke_only=False,
                                        resolver=lambda _: self.fail('HEAD refreshed'))
            self.assertEqual(reused, lock)
            self.assertEqual(digest, m.old.sha256_file(Path(spec['reference_lock_path'])))
            changed = copy.deepcopy(lock)
            changed['models'][1]['repo_id'] += '-instruct'
            with self.assertRaisesRegex(ValueError, 'identity/variant'):
                m.validate_lock(changed, SPEC)

    def test_lock_smoke_resolves_but_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = copy.deepcopy(SPEC)
            spec['reference_lock_path'] = str(Path(temporary)/'lock.json')
            seen = []
            lock, digest = m.get_lock(spec, smoke_only=True,
                                      resolver=lambda repo: seen.append(repo) or 'b'*40)
            self.assertEqual(len(seen), 2)
            self.assertFalse(Path(spec['reference_lock_path']).exists())
            self.assertEqual(digest, m.sha256_bytes(m.canonical_bytes(lock)))

    def test_padding_mask_and_causal_next_token_ce(self):
        ids, mask = m.padded_batch([[1, 2, 3], [1, 2]], 0, 'cpu')
        self.assertEqual(ids.tolist(), [[1, 2, 3], [1, 2, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 1, 0]])
        logits = torch.zeros((2, 3, 5), dtype=torch.float32)
        loss, predicted = m.causal_ce_sum(logits, ids, mask)
        self.assertEqual(predicted, 3)
        self.assertAlmostEqual(loss, 3*math.log(5), places=5)
        self.assertEqual(ids[:, 1:].tolist(), [[2, 3], [2, 0]])

    def test_scoring_uses_inference_mode_no_gradients_and_native_tokenizer(self):
        model = ToyReference()
        result = m.score_sequences(model, NativeTokenizer(), ['abc']*8, batch_size=8)
        self.assertEqual(result['sequence_count'], 8)
        self.assertEqual(result['total_native_tokens'], 24)
        self.assertEqual(result['predicted_native_tokens'], 16)
        self.assertAlmostEqual(result['mean_nll_per_predicted_native_token'], math.log(5), places=5)
        self.assertEqual(result['total_utf8_bytes'], 24)
        self.assertAlmostEqual(result['total_nll_nats'], 16*math.log(5), places=5)
        self.assertAlmostEqual(result['nats_per_utf8_byte'], 16*math.log(5)/24, places=5)
        self.assertAlmostEqual(result['bits_per_utf8_byte'],
                               16*math.log(5)/24/math.log(2), places=5)
        self.assertEqual(model.inference_flags, [(True, False, False, True)])
        self.assertIsNone(model.weight.grad)

    def test_same_strings_different_tokenization_same_utf8_denominator(self):
        texts = ['é and 猫']*8
        first = m.score_sequences(VariableReference(), NativeTokenizer(), texts, batch_size=8)
        second = m.score_sequences(VariableReference(), FragmentedTokenizer(), texts, batch_size=8)
        self.assertEqual(first['total_utf8_bytes'], second['total_utf8_bytes'])
        self.assertEqual(first['total_utf8_bytes'], sum(len(x.encode('utf-8')) for x in texts))
        self.assertNotEqual(first['predicted_native_tokens'], second['predicted_native_tokens'])
        self.assertNotEqual(first['mean_nll_per_predicted_native_token'],
                            second['mean_nll_per_predicted_native_token'])

    def test_multibyte_utf8_counts_bytes_not_characters(self):
        texts = ['é', '猫', '🙂']
        self.assertEqual(sum(map(len, texts)), 3)
        self.assertEqual(m.utf8_byte_count(texts), 9)
        with self.assertRaisesRegex(ValueError, 'zero UTF-8 bytes'):
            m.utf8_byte_count(['', ''])

    def test_population_z_scores_stationarity_and_average_rank(self):
        pc1 = [float(i-3.5) for i in range(8)]
        result = m.final_statistics(SPEC, synthetic_cells(), pc1, 0.8)
        self.assertEqual(result['pc1_variance_explained'], 0.8)
        rows = result['corpora']
        self.assertAlmostEqual(rows[0]['z']['smollm2_1_7b'], -3.5/math.sqrt(5.25))
        self.assertEqual([row['average_rank'] for row in rows],
                         [1.0, 2.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0])
        self.assertEqual([row['token_nll_average_rank'] for row in rows],
                         list(range(1, 9)))
        self.assertEqual(result['primary_stationarity_input'], 'nats_per_utf8_byte')
        self.assertEqual(result['token_nll_robustness']['stationarity_input'],
                         'mean_nll_per_predicted_native_token')
        self.assertEqual([row['stationarity_score'] for row in rows],
                         [-row['z']['smollm2_1_7b'] for row in rows])
        self.assertNotEqual([row['stationarity_score'] for row in rows],
                            [row['token_nll_stationarity_score'] for row in rows])
        self.assertAlmostEqual(sum(row['stationarity_score'] for row in rows), 0.0)
        self.assertEqual(result['smollm2_vs_pythia_score_spearman'], 1.0)
        self.assertEqual(rows[0]['reference_scores']['smollm2_1_7b']['total_utf8_bytes'], 1000)
        self.assertEqual(rows[0]['reference_scores']['smollm2_1_7b']['nats_per_utf8_byte'], 1.0)
        self.assertAlmostEqual(result['primary_absolute_spearman'],
                               abs(m.spearman([row['stationarity_score'] for row in rows], pc1)))
        self.assertAlmostEqual(result['token_nll_robustness']['absolute_spearman_vs_pc1'], 1.0)

    def test_tie_aware_ranks_and_sign_invariant_correlations(self):
        self.assertEqual(m.average_ranks([2, 1, 2, 4]), [2.5, 1.0, 2.5, 4.0])
        pc1 = [float(i-3.5) for i in range(8)]
        positive = m.final_statistics(SPEC, synthetic_cells(), pc1, 1.0)
        negative = m.final_statistics(SPEC, synthetic_cells(), [-x for x in pc1], 1.0)
        self.assertEqual(positive['primary_absolute_spearman'], negative['primary_absolute_spearman'])
        self.assertEqual(positive['secondary_absolute_pearson'], negative['secondary_absolute_pearson'])
        self.assertNotEqual(positive['primary_absolute_spearman'],
                            positive['token_nll_robustness']['absolute_spearman_vs_pc1'])
        self.assertNotEqual(positive['pc1_display_sign_flipped'],
                            negative['pc1_display_sign_flipped'])

    def test_pc1_blind_unit_normalized_svd_and_explained_variance(self):
        responses = []
        for i in range(8):
            value = torch.zeros((128, 2048), dtype=torch.float32)
            value[:, 0] = 1 if i < 4 else -1
            responses.append(value)
        coordinates, fraction = m.pc1_coordinates(responses)
        self.assertEqual(len(coordinates), 8)
        self.assertAlmostEqual(fraction, 1.0)
        self.assertAlmostEqual(abs(coordinates[0]), 2.0)
        self.assertAlmostEqual(abs(coordinates[7]), 2.0)
        self.assertLess(coordinates[0]*coordinates[7], 0)

    def test_attempt100_pc1_artifact_raw_hash_validation(self):
        spec = copy.deepcopy(SPEC)
        tensors = {name: torch.ones((128, 2048), dtype=torch.float32)
                   for name in spec['attempt100']['artifact_raw_sha256']}
        spec['attempt100']['artifact_raw_sha256'] = {
            name: m.old.sha256_raw_float32_tensor(value, torch)
            for name, value in tensors.items()}
        with patch.object(m, 'require_hash'), patch.object(torch, 'load', return_value=tensors):
            self.assertEqual(len(m.load_attempt100_pc1_inputs(spec)), 8)
            tensors['response_pg19'][2, 0] = 3
            with self.assertRaisesRegex(ValueError, 'raw tensor mismatch'):
                m.load_attempt100_pc1_inputs(spec)

    def test_atomic_checkpoint_write_hash_validation_and_resume(self):
        row = {'name': m.MODELS[0][0], 'repo_id': m.MODELS[0][1], 'revision': 'a'*40}
        result = synthetic_cells()[row['name']][m.ORDER[0]]
        metadata = m.checkpoint_metadata(SPEC, row, m.ORDER[0], 'b'*64, 'c'*64,
                                         result['total_utf8_bytes'])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'cell.json'
            self.assertIsNone(m.load_or_store_cell(path, metadata))
            self.assertEqual(m.load_or_store_cell(path, metadata, result), result)
            self.assertEqual(m.load_or_store_cell(path, metadata), result)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.atomic_new_json(path, {'wrong': True})
            changed = copy.deepcopy(metadata)
            changed['decoded_inventory_sha256'] = 'd'*64
            with self.assertRaisesRegex(ValueError, 'provenance'):
                m.load_or_store_cell(path, changed)
            bad = json.loads(path.read_text())
            bad['result']['mean_nll_per_predicted_native_token'] = 99.0
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'result/hash'):
                m.load_or_store_cell(path, metadata)
            old = m.make_checkpoint(metadata, {key: value for key, value in result.items()
                                               if key not in ('total_nll_nats', 'total_utf8_bytes',
                                                              'nats_per_utf8_byte', 'bits_per_utf8_byte')})
            path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, 'result/hash'):
                m.load_or_store_cell(path, metadata)

    def test_no_overwrite_and_information_firewall(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'result.json'
            m.atomic_new_json(path, {'ok': True})
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.atomic_new_json(path, {'ok': False})
        for path in ('/models/base/checkpoint', '/adapter/model.safetensors',
                     '/oracle_adl/oracle_adl.pt', '/base_mean.pt', '/ft_mean.pt',
                     '/true_delta.pt', '/some/evaluation.json', '/attempt025/output.pt',
                     '/attempt027/privileged.pt', '/historical_base/model.safetensors'):
            with self.assertRaisesRegex(ValueError, 'Forbidden'):
                m.safe_path(path)
        self.assertTrue(all(value is False for value in SPEC['firewall'].values()))

    def test_smoke_writes_no_scientific_outputs(self):
        class SimpleDecoder:
            def batch_decode(self, rows, **kwargs):
                return ['synthetic text']*len(rows)
        lock = m.make_lock(SPEC, lambda _: 'a'*40)
        fake_tokens = {name: torch.zeros((128, 128), dtype=torch.int64) for name in m.ORDER}
        def fake_score(model, tokenizer, texts, *, batch_size, progress=None):
            self.assertEqual(len(texts), 8)
            self.assertEqual(batch_size, 8)
            return {'predicted_native_tokens': 16}
        with patch.object(m, 'load_spec', return_value=SPEC), \
             patch.object(m, 'validate_metadata'), \
             patch.object(m, 'get_lock', return_value=(lock, 'a'*64)), \
             patch.object(m, 'validate_tokenizer_files', return_value=Path('/synthetic/merged')), \
             patch.object(m, 'load_first_sequences', return_value=fake_tokens), \
             patch.object(m, 'score_sequences', side_effect=fake_score), \
             patch.object(m, 'atomic_new_json', side_effect=AssertionError('smoke wrote output')):
            result = m.run(smoke_only=True,
                           model_loader=lambda row: (object(), object(), 0.1),
                           qwen_loader=lambda root: SimpleDecoder())
        self.assertTrue(result['smoke_only'])
        self.assertFalse(result['writes'])
        self.assertEqual(len(result['models']), 2)


if __name__ == '__main__':
    unittest.main()
