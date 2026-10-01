#!/usr/bin/env python3
"""Freeze five blind generic-text environments at immutable dataset revisions."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '100_diverse8_raw_jg_consensus_prefix0_13'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = 'a10b513387635cd599971b10a3c7614a7a18483db7b12f9058898c417101db12'
OLD_SOURCE = Path(__file__).with_name('construct_multicorpus_raw_jg_consensus.py')
OLD_SOURCE_SHA256 = '59936b928dcf942470db92debb1f2a8e41823ae1c9082803d2fdd8dc50202caa'
if hashlib.sha256(OLD_SOURCE.read_bytes()).hexdigest() != OLD_SOURCE_SHA256:
    raise ValueError('Frozen Attempt014 construction helper changed')
loader = importlib.util.spec_from_file_location('attempt100_frozen014_helpers', OLD_SOURCE)
old = importlib.util.module_from_spec(loader)
loader.loader.exec_module(old)
NAMES = ('arxiv_document', 'cc_news', 'pg19', 'codeparrot_clean', 'ultrachat')
REPO_CONFIG_SPLIT_FIELD = (
    ('ccdv/arxiv-summarization', 'document', 'train', 'article'),
    ('vblagoje/cc_news', None, 'train', 'text'),
    ('emozilla/pg19', None, 'train', 'text'),
    ('codeparrot/codeparrot-clean', None, 'train', 'content'),
    ('HuggingFaceH4/ultrachat_200k', None, 'train_sft', 'messages'),
)
FORBIDDEN = ('models/base', 'adapter', 'oracle_adl', 'ft_mean', 'base_mean',
             'oracle_difference', 'evaluation.json', 'true_delta', 'known_base',
             'attempt015', 'attempt016', 'attempt017', 'attempt018',
             'attempt020', 'attempt021', 'attempt025')


def safe_path(path):
    value = str(path).lower().replace('\\', '/')
    if any(word in value for word in FORBIDDEN):
        raise ValueError('Forbidden construction input path')
    return Path(path)


def resolve_path(value):
    path = safe_path(value)
    return path if path.is_absolute() else PROJECT/path


def load_spec(path=SPEC_PATH):
    path = safe_path(path)
    if path.resolve() != SPEC_PATH.resolve() or old.sha256_file(path) != SPEC_SHA256:
        raise ValueError('Attempt100 frozen spec path/hash mismatch')
    spec = old.load_json_object(path, 'Attempt100 spec')
    if (spec.get('attempt_id') != ATTEMPT or
            [row.get('name') for row in spec.get('new_corpora', [])] != list(NAMES) or
            [(r.get('repo_id'), r.get('config'), r.get('split'), r.get('source_field'))
             for r in spec['new_corpora']] != list(REPO_CONFIG_SPLIT_FIELD)):
        raise ValueError('Attempt100 corpus inventory mismatch')
    return spec


def require_revision(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{40}', value) is None:
        raise ValueError('Dataset revision must be a full immutable 40-character SHA')
    return value


def make_lock(spec, resolver):
    rows = []
    for corpus in spec['new_corpora']:
        revision = require_revision(resolver(corpus['repo_id']))
        rows.append({'name': corpus['name'], 'repo_id': corpus['repo_id'],
                     'config': corpus['config'], 'split': corpus['split'],
                     'revision': revision})
    return {'format_version': 1, 'attempt_id': ATTEMPT,
            'spec_sha256': SPEC_SHA256, 'resolution': 'one_time_dataset_HEAD',
            'corpora': rows}


def validate_lock(lock, spec):
    if not isinstance(lock, dict) or lock.get('format_version') != 1 or lock.get('attempt_id') != ATTEMPT or lock.get('spec_sha256') != SPEC_SHA256 or lock.get('resolution') != 'one_time_dataset_HEAD':
        raise ValueError('Corpus lock provenance mismatch')
    rows = lock.get('corpora')
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError('Corpus lock inventory mismatch')
    for row, corpus in zip(rows, spec['new_corpora']):
        if (not isinstance(row, dict) or set(row) != {'name', 'repo_id', 'config', 'split', 'revision'} or
                any(row[key] != corpus[key] for key in ('name', 'repo_id', 'config', 'split'))):
            raise ValueError('Corpus lock identity mismatch')
        require_revision(row['revision'])
    return lock


def no_overwrite(*paths):
    for path in paths:
        if path.exists() or path.is_symlink():
            raise ValueError(f'Output already exists: {path}')


def derived_text(example, corpus):
    if not isinstance(example, dict):
        raise ValueError('Dataset row must be a mapping')
    value = example.get(corpus['source_field'])
    if corpus['name'] == 'ultrachat':
        if not isinstance(value, list):
            raise ValueError('UltraChat messages must be a list')
        pieces = []
        for message in value:
            if (not isinstance(message, dict) or not isinstance(message.get('role'), str) or
                    not isinstance(message.get('content'), str)):
                raise ValueError('UltraChat message schema mismatch')
            pieces.append(message['role'] + ': ' + message['content'])
        return '\n\n'.join(pieces)
    if not isinstance(value, str):
        raise ValueError('Dataset source text must be a string')
    return value


def select_tokens(dataset, tokenizer, corpus, policy, torch_module, count):
    if count < 1 or count > policy['valid_examples']:
        raise ValueError('Invalid fixed selection count')
    shuffled = dataset.shuffle(seed=policy['shuffle_seed'], buffer_size=policy['shuffle_buffer_size'])
    rows, inspected, blank, short = [], 0, 0, 0
    for example in iter(shuffled):
        inspected += 1
        text = derived_text(example, corpus)
        if not text.strip():
            blank += 1
            continue
        ids = tokenizer.encode(text[:policy['character_limit']],
                               add_special_tokens=policy['add_special_tokens'])
        if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError('Tokenizer produced invalid IDs')
        if len(ids) < policy['minimum_tokens']:
            short += 1
            continue
        rows.append(ids[:policy['retain_first_tokens']])
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError('Dataset stream ended before the required valid row count')
    tensor = torch_module.tensor(rows, dtype=torch_module.int64, device='cpu').contiguous()
    if tuple(tensor.shape) != (count, 128):
        raise ValueError('Frozen corpus token inventory mismatch')
    return tensor, {'source_rows_inspected': inspected, 'valid_rows_accepted': len(rows),
                    'skipped_blank': blank, 'skipped_short': short}


def load_dataset_at_revision(corpus, revision):
    from datasets import load_dataset
    keywords = {'path': corpus['repo_id'], 'revision': require_revision(revision),
                'split': corpus['split'], 'streaming': True}
    if corpus['config'] is not None:
        keywords['name'] = corpus['config']
    return load_dataset(**keywords)


def validate_corpus(corpus, spec, lock, model_dir, torch_module):
    index = NAMES.index(corpus['name'])
    revision = validate_lock(lock, spec)['corpora'][index]['revision']
    manifest_path, tokens_path = resolve_path(corpus['manifest_path']), resolve_path(corpus['tokens_path'])
    manifest = old.load_json_object(manifest_path, 'Attempt100 corpus manifest')
    lock_path = resolve_path(spec['corpus_lock_path'])
    expected = {'format_version': 1, 'attempt_id': ATTEMPT, 'hash_algorithm': 'sha256',
                'corpus_name': corpus['name'], 'repo_id': corpus['repo_id'],
                'config': corpus['config'], 'split': corpus['split'],
                'revision': revision, 'extractor': corpus['extractor'],
                'source_field': corpus['source_field'], 'selection': spec['corpus_freeze'],
                'spec_sha256': SPEC_SHA256, 'lock_sha256': old.sha256_file(lock_path),
                'freeze_script_sha256': old.sha256_file(Path(__file__).resolve()),
                'tokenizer_files': old.tokenizer_file_records(model_dir),
                'tensor': {'shape': [4096, 128], 'dtype': 'torch.int64',
                           'device': 'cpu', 'contiguous': True}}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError('Attempt100 corpus provenance mismatch')
    counts = manifest.get('counts', {})
    if (not isinstance(counts, dict) or
            any(type(counts.get(key)) is not int or counts[key] < 0 for key in
                ('source_rows_inspected', 'valid_rows_accepted', 'skipped_blank', 'skipped_short')) or
            counts['valid_rows_accepted'] != 4096 or
            counts['source_rows_inspected'] != 4096 + counts['skipped_blank'] + counts['skipped_short']):
        raise ValueError('Attempt100 corpus count inventory mismatch')
    artifact = manifest.get('artifact', {})
    if (artifact.get('path') != str(tokens_path) or
            old.sha256_file(tokens_path) != old.require_sha256(artifact.get('serialized_sha256'), 'corpus artifact')):
        raise ValueError('Attempt100 corpus serialized hash mismatch')
    tokens = old.load_corpus_tokens(tokens_path, artifact['raw_tensor_sha256'], 4096, 128, torch_module)
    return tokens, manifest


def freeze(spec_path=SPEC_PATH, model_dir=None, smoke_only=False, resolver=None,
           dataset_loader=None, tokenizer_loader=None):
    import torch
    spec = load_spec(spec_path)
    model_dir = safe_path(model_dir or spec['defaults']['merged_model_directory'])
    lock_path = resolve_path(spec['corpus_lock_path'])
    outputs = [resolve_path(row[key]) for row in spec['new_corpora']
               for key in ('tokens_path', 'manifest_path')]
    if not smoke_only:
        no_overwrite(*outputs)
    inventory = old.tokenizer_file_records(model_dir)
    pinned = [row for row in spec['canonical_checkpoint_files']
              if row['path'] in old.TOKENIZER_FILE_NAMES]
    if inventory != pinned:
        raise ValueError('Canonical merged tokenizer inventory mismatch')
    if resolver is None:
        from huggingface_hub import HfApi
        api = HfApi()
        resolver = lambda repo_id: api.repo_info(repo_id, repo_type='dataset').sha
    if lock_path.exists():
        lock = validate_lock(old.load_json_object(lock_path, 'corpus lock'), spec)
    else:
        lock = make_lock(spec, resolver)
        if not smoke_only:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(lock, indent=2, allow_nan=False) + '\n')
    lock_hash = old.sha256_file(lock_path) if lock_path.exists() else None
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader = lambda directory: AutoTokenizer.from_pretrained(directory, local_files_only=True)
    tokenizer = tokenizer_loader(model_dir)
    loader = dataset_loader or load_dataset_at_revision
    results = []
    for corpus, entry in zip(spec['new_corpora'], lock['corpora']):
        dataset = loader(corpus, entry['revision'])
        count = spec['corpus_freeze']['smoke_valid_examples_per_corpus'] if smoke_only else 4096
        tokens, counts = select_tokens(dataset, tokenizer, corpus, spec['corpus_freeze'], torch, count)
        result = {'corpus_name': corpus['name'], 'revision': entry['revision'], **counts,
                  'tensor_shape': list(tokens.shape)}
        if not smoke_only:
            if old.sha256_file(lock_path) != lock_hash or old.tokenizer_file_records(model_dir) != inventory:
                raise ValueError('Corpus lock or tokenizer changed before publication')
            tokens_path, manifest_path = resolve_path(corpus['tokens_path']), resolve_path(corpus['manifest_path'])
            no_overwrite(tokens_path, manifest_path)
            tokens_path.parent.mkdir(parents=True, exist_ok=True)
            with tokens_path.open('xb') as stream:
                torch.save(tokens, stream)
            manifest = {'format_version': 1, 'attempt_id': ATTEMPT, 'hash_algorithm': 'sha256',
                        'corpus_name': corpus['name'], 'repo_id': corpus['repo_id'],
                        'config': corpus['config'], 'split': corpus['split'],
                        'revision': entry['revision'], 'extractor': corpus['extractor'],
                        'source_field': corpus['source_field'], 'selection': spec['corpus_freeze'],
                        'spec_sha256': SPEC_SHA256, 'lock_sha256': lock_hash,
                        'freeze_script_sha256': old.sha256_file(Path(__file__).resolve()),
                        'tokenizer_files': inventory, 'counts': counts,
                        'tensor': {'shape': [4096, 128], 'dtype': 'torch.int64',
                                   'device': 'cpu', 'contiguous': True},
                        'artifact': {'path': str(tokens_path),
                                     'serialized_sha256': old.sha256_file(tokens_path),
                                     'raw_tensor_sha256': old.sha256_raw_int64_tensor(tokens, torch)}}
            with manifest_path.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        results.append(result)
    if (old.tokenizer_file_records(model_dir) != inventory or old.sha256_file(spec_path) != SPEC_SHA256 or
            (lock_hash is not None and old.sha256_file(lock_path) != lock_hash)):
        raise ValueError('Frozen tokenizer, spec, or lock changed during corpus selection')
    return {'smoke_only': smoke_only, 'corpora': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(freeze(smoke_only=args.smoke_only), allow_nan=False))


if __name__ == '__main__':
    main()
