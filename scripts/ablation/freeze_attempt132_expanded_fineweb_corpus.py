#!/usr/bin/env python3
"""Freeze valid FineWeb ranks [28192,36384) with Attempt005's pinned selector."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '132_expanded_fineweb_superlow_stringency'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'corpus_spec.json'
SPEC_SHA256 = '6813d8fafdfdd0d6bfe2f07508f4e7e371a728a242d74c5fcaa69ab66cd1bf0f'
MANIFEST_PATH = SPEC_PATH.with_name('corpus-manifest.json')
ARTIFACT_PATH = Path('/root/model-diff-scratch/artifacts/attempt132_expanded_fineweb_corpus/tokens.pt')
OLD_SOURCE = PROJECT / 'scripts/ablation/freeze_generic_rollback_corpus.py'
OLD_SOURCE_SHA256 = '637cf6acf43f9dbcaa86d76384af9f44aa189028600b8e374167e976d7c6cfaf'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def import_old_selector():
    if sha256_file(OLD_SOURCE) != OLD_SOURCE_SHA256:
        raise ValueError('Attempt005 selector source SHA256 mismatch')
    loader = importlib.util.spec_from_file_location('attempt132_pinned_attempt005_selector', OLD_SOURCE)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def load_spec():
    if sha256_file(SPEC_PATH) != SPEC_SHA256:
        raise ValueError('Attempt132 corpus specification SHA256 mismatch')
    spec = json.loads(SPEC_PATH.read_text())
    pinned = spec['frozen_attempt005']
    for key in ('corpus_spec', 'corpus_manifest', 'freezer_source'):
        if sha256_file(path_of(pinned[key + '_path'])) != pinned[key + '_sha256']:
            raise ValueError('Frozen Attempt005 provenance mismatch: ' + key)
    old_spec = json.loads(path_of(pinned['corpus_spec_path']).read_text())
    old_manifest = json.loads(path_of(pinned['corpus_manifest_path']).read_text())
    old_selection = old_spec['selection']
    if (spec['attempt_id'] != ATTEMPT or
            spec['dataset'] != old_spec['dataset'] or
            spec['dataset'] != {'repo_id': 'science-of-finetuning/fineweb-1m-sample',
                'revision': '60b53a86b84eb6559e4407b113356f56a152318f',
                'split': 'train', 'streaming': False} or
            spec['tokenizer'] != old_spec['tokenizer'] or
            old_selection['skip_valid_examples'] != 20000 or
            old_selection['sample_count'] != 4096 or
            spec['selection'] != {**old_selection, 'skip_valid_examples': 28192,
                                  'sample_count': 8192} or
            spec['old_pooled_valid_example_rank_interval'] != [20000, 28192] or
            spec['new_valid_example_rank_interval'] != [28192, 36384] or
            spec['artifact'] != {**old_spec['artifact'], 'path': str(ARTIFACT_PATH)} or
            spec['tokenizer_checkpoint_files'] != old_manifest['tokenizer_checkpoint']['files'] or
            spec['manifest'] != {'filename': 'corpus-manifest.json',
                'timestamps': False, 'host_metadata': False, 'gpu_metadata': False} or
            pinned['freezer_source_sha256'] != OLD_SOURCE_SHA256 or
            old_manifest['selection'] != old_selection or
            old_manifest['dataset'] != spec['dataset']):
        raise ValueError('Expanded FineWeb selection differs from Attempt005 except rank/count')
    return spec


def refuse_outputs(artifact_path=ARTIFACT_PATH, manifest_path=MANIFEST_PATH):
    if (artifact_path.exists() or artifact_path.is_symlink() or
            manifest_path.exists() or manifest_path.is_symlink() or
            artifact_path == manifest_path):
        raise ValueError('Attempt132 corpus output already exists')


def freeze_corpus(*, dataset_loader=None, tokenizer_loader=None, torch_module=None):
    """Dependency injection is for synthetic tests; production uses frozen inputs."""
    spec = load_spec()
    refuse_outputs()
    old = import_old_selector()
    tokenizer_dir = path_of(spec['tokenizer']['directory'])
    records = old.tokenizer_file_records(tokenizer_dir)
    if records != spec['tokenizer_checkpoint_files']:
        raise ValueError('Canonical tokenizer checkpoint inventory changed')
    if dataset_loader is None:
        from datasets import load_dataset as dataset_loader
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader = lambda path: AutoTokenizer.from_pretrained(path, local_files_only=True)
    if torch_module is None:
        import torch as torch_module
    tokenizer = tokenizer_loader(tokenizer_dir)
    dataset = dataset_loader(spec['dataset']['repo_id'],
        revision=spec['dataset']['revision'], split='train', streaming=False)
    tensor = old.select_tokens(dataset, tokenizer, torch_module, spec)
    old.validate_tokens(tensor, torch_module, sample_count=8192, sequence_length=128)
    if (sha256_file(SPEC_PATH) != SPEC_SHA256 or
            sha256_file(OLD_SOURCE) != OLD_SOURCE_SHA256 or
            old.tokenizer_file_records(tokenizer_dir) != records):
        raise ValueError('Frozen corpus inputs changed during selection')
    refuse_outputs()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ARTIFACT_PATH.open('xb') as stream:
        torch_module.save(tensor, stream)
    artifact_hash = sha256_file(ARTIFACT_PATH)
    raw_hash = old.sha256_raw_int64_tensor(tensor, torch_module)
    manifest = old.build_manifest(spec, SPEC_SHA256, sha256_file(Path(__file__)),
        tokenizer_dir, records, ARTIFACT_PATH, artifact_hash, raw_hash)
    manifest['old_pooled_valid_example_rank_interval'] = [20000, 28192]
    manifest['new_valid_example_rank_interval'] = [28192, 36384]
    manifest['frozen_attempt005'] = spec['frozen_attempt005']
    old.write_manifest(MANIFEST_PATH, manifest)
    if (sha256_file(ARTIFACT_PATH) != artifact_hash or
            json.loads(MANIFEST_PATH.read_text()) != manifest):
        raise ValueError('Published expanded FineWeb corpus failed revalidation')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        freeze_corpus()
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 1
    print('Wrote expanded Attempt132 FineWeb corpus and manifest')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
