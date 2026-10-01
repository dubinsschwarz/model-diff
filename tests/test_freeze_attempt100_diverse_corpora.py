"""Synthetic tests for Attempt 100's blind five-corpus revision freeze."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/freeze_attempt100_diverse_corpora.py'
loader = importlib.util.spec_from_file_location('attempt100_freeze_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


class Dataset:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def shuffle(self, seed, buffer_size):
        self.calls.append((seed, buffer_size))
        return self

    def __iter__(self):
        return iter(self.rows)


class Tokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return list(range(len(text)))


class FreezeAttempt100Tests(unittest.TestCase):
    def test_frozen_spec_and_five_dataset_inventory(self):
        self.assertEqual(m.old.sha256_file(m.SPEC_PATH), m.SPEC_SHA256)
        self.assertEqual(tuple(row['name'] for row in SPEC['new_corpora']), m.NAMES)
        self.assertEqual(tuple((r['repo_id'], r['config'], r['split'], r['source_field'])
                               for r in SPEC['new_corpora']), m.REPO_CONFIG_SPLIT_FIELD)

    def test_resolved_revision_requires_full_immutable_sha(self):
        good = 'a'*40
        self.assertEqual(m.require_revision(good), good)
        for bad in ('main', 'a'*39, 'g'*40, 'A'*40, None):
            with self.assertRaises(ValueError):
                m.require_revision(bad)
        lock = m.make_lock(SPEC, lambda _: good)
        self.assertEqual([r['revision'] for r in lock['corpora']], [good]*5)
        self.assertEqual(m.validate_lock(lock, SPEC), lock)

    def test_existing_lock_prevents_head_refresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = copy.deepcopy(SPEC)
            spec['corpus_lock_path'] = str(root/'corpus-lock.json')
            spec['defaults']['merged_model_directory'] = str(root)
            spec['canonical_checkpoint_files'] = []
            for corpus in spec['new_corpora']:
                corpus['tokens_path'] = str(root/(corpus['name']+'.pt'))
                corpus['manifest_path'] = str(root/(corpus['name']+'.json'))
            lock = m.make_lock(spec, lambda _: 'b'*40)
            (root/'corpus-lock.json').write_text(json.dumps(lock))
            rows = [dict(article='x'*130, text='x'*130, content='x'*130,
                         messages=[{'role': 'user', 'content': 'x'*130}])]*4
            datasets = []
            def load(corpus, revision):
                self.assertEqual(revision, 'b'*40)
                item = Dataset(rows)
                datasets.append(item)
                return item
            def no_head(_):
                self.fail('HEAD refreshed despite existing lock')
            with patch.object(m, 'load_spec', return_value=spec), \
                 patch.object(m.old, 'tokenizer_file_records', return_value=[]):
                result = m.freeze(smoke_only=True, resolver=no_head,
                                  dataset_loader=load, tokenizer_loader=lambda _: Tokenizer())
            self.assertTrue(result['smoke_only'])
            self.assertEqual(len(datasets), 5)
            self.assertEqual([item.calls for item in datasets], [[(42, 10000)]]*5)
            self.assertEqual(sorted(path.name for path in root.iterdir()), ['corpus-lock.json'])

    def test_first_synthetic_freeze_writes_lock_five_artifacts_and_refuses_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = copy.deepcopy(SPEC)
            spec['corpus_lock_path'] = str(root/'corpus-lock.json')
            spec['defaults']['merged_model_directory'] = str(root)
            spec['canonical_checkpoint_files'] = []
            for corpus in spec['new_corpora']:
                corpus['tokens_path'] = str(root/(corpus['name']+'.pt'))
                corpus['manifest_path'] = str(root/(corpus['name']+'.json'))
            rows = [dict(article='x'*130, text='x'*130, content='x'*130,
                         messages=[{'role': 'user', 'content': 'x'*130}])]*4096
            heads = []
            def resolve(repo_id):
                heads.append(repo_id)
                return 'd'*40
            with patch.object(m, 'load_spec', return_value=spec), \
                 patch.object(m.old, 'tokenizer_file_records', return_value=[]):
                result = m.freeze(smoke_only=False, resolver=resolve,
                                  dataset_loader=lambda corpus, revision: Dataset(rows),
                                  tokenizer_loader=lambda _: Tokenizer())
                self.assertEqual(len(result['corpora']), 5)
                self.assertEqual(len(heads), 5)
                self.assertEqual([r['revision'] for r in
                                  json.loads((root/'corpus-lock.json').read_text())['corpora']],
                                 ['d'*40]*5)
                for corpus in spec['new_corpora']:
                    tokens, manifest = m.validate_corpus(corpus, spec,
                        json.loads((root/'corpus-lock.json').read_text()), root, torch)
                    self.assertEqual(tuple(tokens.shape), (4096, 128))
                    self.assertEqual(manifest['counts']['valid_rows_accepted'], 4096)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    m.freeze(smoke_only=False, resolver=lambda _: self.fail('HEAD refreshed'),
                             dataset_loader=lambda corpus, revision: Dataset(rows),
                             tokenizer_loader=lambda _: Tokenizer())

    def test_lock_rejects_reordered_or_invalid_revisions(self):
        lock = m.make_lock(SPEC, lambda _: 'a'*40)
        changed = copy.deepcopy(lock)
        changed['corpora'].reverse()
        with self.assertRaisesRegex(ValueError, 'identity'):
            m.validate_lock(changed, SPEC)
        changed = copy.deepcopy(lock)
        changed['corpora'][0]['revision'] = 'main'
        with self.assertRaisesRegex(ValueError, 'immutable'):
            m.validate_lock(changed, SPEC)

    def test_plain_text_fields_and_ultrachat_join(self):
        for corpus in SPEC['new_corpora'][:4]:
            self.assertEqual(m.derived_text({corpus['source_field']: 'verbatim'}, corpus), 'verbatim')
        ultra = SPEC['new_corpora'][4]
        messages = [{'role': 'user', 'content': 'Hello'},
                    {'role': 'assistant', 'content': 'World'}]
        self.assertEqual(m.derived_text({'messages': messages}, ultra),
                         'user: Hello\n\nassistant: World')
        with self.assertRaisesRegex(ValueError, 'schema'):
            m.derived_text({'messages': [{'role': 'user', 'content': None}]}, ultra)

    def test_character_truncation_filter_and_first_128(self):
        corpus = SPEC['new_corpora'][0]
        rows = [{'article': '   '}, {'article': 'x'*127},
                {'article': 'x'*1500}, {'article': 'y'*140}]
        dataset, tokenizer = Dataset(rows), Tokenizer()
        tokens, counts = m.select_tokens(dataset, tokenizer, corpus, SPEC['corpus_freeze'], torch, 2)
        self.assertEqual(dataset.calls, [(42, 10000)])
        self.assertEqual(counts, {'source_rows_inspected': 4, 'valid_rows_accepted': 2,
                                  'skipped_blank': 1, 'skipped_short': 1})
        self.assertEqual(len(tokenizer.calls[1][0]), 1280)
        self.assertTrue(all(add for _, add in tokenizer.calls))
        self.assertEqual(tokens.shape, (2, 128))
        self.assertEqual(tokens.dtype, torch.int64)
        self.assertTrue(torch.equal(tokens[0], torch.arange(128)))

    def test_stream_exhaustion_fails_closed(self):
        corpus = SPEC['new_corpora'][1]
        with self.assertRaisesRegex(ValueError, 'ended'):
            m.select_tokens(Dataset([{'text': 'x'*127}]), Tokenizer(), corpus,
                            SPEC['corpus_freeze'], torch, 1)

    def test_full_artifact_shape_hash_and_contiguity(self):
        rows = [{'text': 'x'*130} for _ in range(4096)]
        tokens, counts = m.select_tokens(Dataset(rows), Tokenizer(), SPEC['new_corpora'][1],
                                         SPEC['corpus_freeze'], torch, 4096)
        self.assertEqual(counts['valid_rows_accepted'], 4096)
        self.assertEqual(tuple(tokens.shape), (4096, 128))
        self.assertEqual(tokens.dtype, torch.int64)
        self.assertEqual(tokens.device.type, 'cpu')
        self.assertTrue(tokens.is_contiguous())
        self.assertEqual(len(m.old.sha256_raw_int64_tensor(tokens, torch)), 64)

    def test_corpus_manifest_and_raw_artifact_provenance_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = copy.deepcopy(SPEC)
            corpus = spec['new_corpora'][0]
            spec['corpus_lock_path'] = str(root/'corpus-lock.json')
            corpus['tokens_path'] = str(root/'tokens.pt')
            corpus['manifest_path'] = str(root/'corpus-manifest.json')
            lock = m.make_lock(spec, lambda _: 'c'*40)
            (root/'corpus-lock.json').write_text(json.dumps(lock))
            tokens = torch.zeros((4096, 128), dtype=torch.int64)
            torch.save(tokens, root/'tokens.pt')
            manifest = {'format_version': 1, 'attempt_id': m.ATTEMPT,
                        'hash_algorithm': 'sha256', 'corpus_name': corpus['name'],
                        'repo_id': corpus['repo_id'], 'config': corpus['config'],
                        'split': corpus['split'], 'revision': 'c'*40,
                        'extractor': corpus['extractor'], 'source_field': corpus['source_field'],
                        'selection': spec['corpus_freeze'], 'spec_sha256': m.SPEC_SHA256,
                        'lock_sha256': m.old.sha256_file(root/'corpus-lock.json'),
                        'freeze_script_sha256': m.old.sha256_file(SOURCE),
                        'tokenizer_files': [],
                        'counts': {'source_rows_inspected': 4100,
                                   'valid_rows_accepted': 4096,
                                   'skipped_blank': 3, 'skipped_short': 1},
                        'tensor': {'shape': [4096, 128], 'dtype': 'torch.int64',
                                   'device': 'cpu', 'contiguous': True},
                        'artifact': {'path': str(root/'tokens.pt'),
                                     'serialized_sha256': m.old.sha256_file(root/'tokens.pt'),
                                     'raw_tensor_sha256': m.old.sha256_raw_int64_tensor(tokens, torch)}}
            (root/'corpus-manifest.json').write_text(json.dumps(manifest))
            with patch.object(m.old, 'tokenizer_file_records', return_value=[]):
                actual, _ = m.validate_corpus(corpus, spec, lock, root, torch)
                self.assertTrue(torch.equal(actual, tokens))
                manifest['artifact']['raw_tensor_sha256'] = '0'*64
                (root/'corpus-manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    m.validate_corpus(corpus, spec, lock, root, torch)
                manifest['artifact']['raw_tensor_sha256'] = m.old.sha256_raw_int64_tensor(tokens, torch)
                manifest['revision'] = 'd'*40
                (root/'corpus-manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    m.validate_corpus(corpus, spec, lock, root, torch)

    def test_dataset_loader_uses_fixed_config_split_revision_streaming(self):
        corpus = SPEC['new_corpora'][0]
        seen = []
        def fake(**kwargs):
            seen.append(kwargs)
            return object()
        with patch.dict('sys.modules', {'datasets': type('D', (), {'load_dataset': staticmethod(fake)})}):
            m.load_dataset_at_revision(corpus, 'a'*40)
        self.assertEqual(seen[0], {'path': 'ccdv/arxiv-summarization', 'revision': 'a'*40,
                                   'split': 'train', 'streaming': True, 'name': 'document'})
        seen.clear()
        with patch.dict('sys.modules', {'datasets': type('D', (), {'load_dataset': staticmethod(fake)})}):
            m.load_dataset_at_revision(SPEC['new_corpora'][1], 'a'*40)
        self.assertNotIn('name', seen[0])

    def test_no_overwrite_and_forbidden_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'tokens.pt'
            m.no_overwrite(path)
            path.write_bytes(b'exists')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.no_overwrite(path)
        for path in ('/models/base/model.safetensors', '/oracle_adl/oracle_adl.pt',
                     '/adapter', '/attempt015/data', '/attempt025/output.pt',
                     '/ft_mean.pt', '/evaluation.json'):
            with self.assertRaisesRegex(ValueError, 'Forbidden'):
                m.safe_path(path)

    def test_smoke_policy_has_no_scientific_outputs(self):
        self.assertFalse(SPEC['corpus_freeze']['smoke_writes'])
        self.assertEqual(SPEC['corpus_freeze']['smoke_valid_examples_per_corpus'], 4)
        self.assertFalse(SPEC['corpus_freeze']['overwrite'])


if __name__ == '__main__':
    unittest.main()
