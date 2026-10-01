#!/usr/bin/env python3
"""Blind eight-environment positive raw-Jg activation-response consensus."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
FREEZER_PATH = Path(__file__).with_name('freeze_attempt100_diverse_corpora.py')
loader = importlib.util.spec_from_file_location('attempt100_corpus_freezer', FREEZER_PATH)
freezer = importlib.util.module_from_spec(loader)
loader.loader.exec_module(freezer)
old = freezer.old
ATTEMPT = freezer.ATTEMPT
ORDER = ('fineweb', 'wikitext103_raw', 'tinystories', *freezer.NAMES)
OLD_NAMES = ('merged_mean', 'response_fineweb', 'response_wikitext103_raw',
             'response_tinystories', 'consensus_response')


def assert_safe_inputs(paths):
    for path in paths:
        freezer.safe_path(path)


def load_old_responses(spec):
    record = spec['attempt014']
    paths = {key: freezer.resolve_path(record[key]) for key in
             ('spec_path', 'manifest_path', 'constructor_path', 'artifact_path')}
    for key in ('spec', 'manifest', 'constructor'):
        if old.sha256_file(paths[key+'_path']) != record[key+'_sha256']:
            raise ValueError('Frozen Attempt014 construction source/provenance hash mismatch')
    old_spec = old.load_json_object(paths['spec_path'], 'Attempt014 construction spec')
    manifest = old.load_json_object(paths['manifest_path'], 'Attempt014 construction manifest')
    if (old_spec != old.FROZEN_SPEC or manifest.get('attempt_id') != old_spec['attempt_id'] or
            manifest.get('spec_sha256') != record['spec_sha256'] or
            manifest.get('constructor_script_sha256') != record['constructor_sha256'] or
            manifest.get('source_checkpoint') != spec['canonical_checkpoint_files'] or
            manifest.get('probe') != spec['probe'] or
            manifest.get('generic_loss') != spec['generic_loss'] or
            manifest.get('jvp') != spec['jvp'] or
            manifest.get('readout') != spec['readout'] or
            manifest.get('corpus_order') != list(ORDER[:3])):
        raise ValueError('Frozen Attempt014 construction provenance mismatch')
    artifact_meta = manifest.get('artifact', {})
    raw = artifact_meta.get('raw_tensors_sha256', {})
    if (artifact_meta.get('serialized_sha256') != record['artifact_serialized_sha256'] or
            any(raw.get(name) != digest for name, digest in record['raw_sha256'].items()) or
            set(raw) != set(OLD_NAMES) or
            old.sha256_file(paths['artifact_path']) != record['artifact_serialized_sha256']):
        raise ValueError('Frozen Attempt014 artifact serialized/raw hash mismatch')
    artifact = torch.load(paths['artifact_path'], map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or set(artifact) != set(OLD_NAMES):
        raise ValueError('Frozen Attempt014 artifact tensor inventory mismatch')
    for name in OLD_NAMES:
        value = artifact[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                value.device.type != 'cpu' or not value.is_contiguous() or
                tuple(value.shape) != (128, 2048) or not bool(torch.isfinite(value).all()) or
                old.sha256_raw_float32_tensor(value, torch) != raw[name]):
            raise ValueError('Frozen Attempt014 raw response hash/geometry mismatch: '+name)
    return artifact, {key: old.sha256_file(value) for key, value in paths.items()}


def response_geometry(responses):
    if len(responses) != 8:
        raise ValueError('Exactly eight frozen-order responses required')
    for response in responses:
        if (not isinstance(response, torch.Tensor) or response.dtype != torch.float32 or
                response.device.type != 'cpu' or not response.is_contiguous() or
                tuple(response.shape) != (128, 2048) or not bool(torch.isfinite(response).all())):
            raise ValueError('Response must be finite contiguous CPU FP32 [128,2048]')
    raw = torch.stack(responses, dim=0).double()
    norms = torch.linalg.vector_norm(raw, dim=2)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError('Zero or nonfinite per-position corpus response')
    unit = raw / norms.unsqueeze(-1)
    consensus64 = unit.mean(dim=0)
    consensus_norm = torch.linalg.vector_norm(consensus64, dim=1)
    if not bool(torch.isfinite(consensus_norm).all()) or bool((consensus_norm <= 0).any()):
        raise ValueError('Eight-way consensus has zero/nonfinite position')
    if bool((consensus_norm > 1 + 1e-12).any()):
        raise ValueError('Eight-way consensus concentration exceeds one')
    def cosine_by_position(left, right):
        a = torch.linalg.vector_norm(left, dim=-1)
        b = torch.linalg.vector_norm(right, dim=-1)
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()) or bool((a <= 0).any()) or bool((b <= 0).any()):
            raise ValueError('Undefined diagnostic cosine')
        return (left * right).sum(dim=-1) / (a * b)
    def means(vector):
        values = vector.tolist()
        return {'positions_1_4': math.fsum(values[1:5])/4,
                'positions_1_127': math.fsum(values[1:128])/127}
    pairwise = torch.einsum('iph,jph->pij', unit, unit)
    pair_means = {label: pairwise[positions].mean(dim=0).tolist()
                  for label, positions in (('positions_1_4', slice(1, 5)),
                                           ('positions_1_127', slice(1, 128)))}
    corpus_cosines = {name: means(cosine_by_position(unit[i], consensus64))
                      for i, name in enumerate(ORDER)}
    leave_one_out = {name: means(cosine_by_position(
        (unit.sum(dim=0) - unit[i])/7, consensus64))
        for i, name in enumerate(ORDER)}
    prefix = {str(k): means(cosine_by_position(unit[:k].mean(dim=0), consensus64))
              for k in (3, 4, 5, 6, 7, 8)}
    diagnostics = {'corpus_order': list(ORDER),
                   'pairwise_per_position': [pairwise[p].tolist() for p in range(128)],
                   'pairwise_mean_matrices': pair_means,
                   'each_corpus_to_fixed_consensus': corpus_cosines,
                   'leave_one_out_to_fixed_consensus': leave_one_out,
                   'prefix_to_fixed_consensus': prefix,
                   'consensus_concentration': consensus_norm.tolist(),
                   'selection': False}
    return consensus64.float().contiguous(), diagnostics


def no_outputs(artifact, manifest):
    old.require_output_absent(artifact, manifest)


def construct(smoke_only=False):
    spec = freezer.load_spec()
    defaults = spec['defaults']
    model_dir = freezer.safe_path(defaults['merged_model_directory'])
    artifact_path, manifest_path = (freezer.resolve_path(defaults[key]) for key in ('artifact_path', 'manifest_path'))
    probe_path = freezer.resolve_path(defaults['probe_path'])
    input_paths = [freezer.SPEC_PATH, FREEZER_PATH, Path(__file__).resolve(),
                   freezer.resolve_path(spec['corpus_lock_path']), probe_path]
    for corpus in spec['new_corpora']:
        input_paths.extend((freezer.resolve_path(corpus['tokens_path']),
                            freezer.resolve_path(corpus['manifest_path'])))
    input_paths.extend(freezer.resolve_path(spec['attempt014'][key]) for key in
                       ('spec_path', 'manifest_path', 'constructor_path', 'artifact_path'))
    assert_safe_inputs(input_paths)
    if not smoke_only:
        no_outputs(artifact_path, manifest_path)
    for output in (artifact_path, manifest_path):
        if output.resolve().is_relative_to(model_dir.resolve()) or any(output.resolve() == path.resolve() for path in input_paths):
            raise ValueError('Construction output overlaps frozen input')
    hashes = {path: old.sha256_file(path) for path in input_paths}
    source_files = old.checkpoint_file_records(model_dir)
    if source_files != spec['canonical_checkpoint_files']:
        raise ValueError('Canonical final checkpoint inventory mismatch')
    prior_artifact, prior_hashes = load_old_responses(spec)
    lock = freezer.validate_lock(old.load_json_object(
        freezer.resolve_path(spec['corpus_lock_path']), 'Attempt100 corpus lock'), spec)
    # Validate all new frozen corpora before loading the model.
    for corpus in spec['new_corpora']:
        tokens, _ = freezer.validate_corpus(corpus, spec, lock, model_dir, torch)
        del tokens
    probe = old.load_probe(probe_path, spec['probe'], torch)
    if smoke_only:
        probe = probe[:spec['construction_smoke']['probe_examples']].contiguous()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = old.load_local_model(model_dir, device, torch)
    eligible = old.discover_eligible_linear_weights(model, torch)
    old.freeze_other_parameters(model, eligible)
    before = old.model_state_hashes(model, torch)
    merged_old = prior_artifact['merged_mean']
    responses = [prior_artifact['response_'+name] for name in ORDER[:3]]
    new_records = []
    for corpus in spec['new_corpora']:
        tokens, corpus_manifest = freezer.validate_corpus(corpus, spec, lock, model_dir, torch)
        loss_spec = spec['generic_loss']
        probe_spec = spec
        if smoke_only:
            n = spec['construction_smoke']['new_corpus_examples_per_corpus']
            tokens = tokens[:n].contiguous()
            loss_spec = {**loss_spec, 'sample_count': n, 'total_prediction_tokens': n*127,
                         'number_of_batches': n//8}
            probe_spec = {**spec, 'probe': {**spec['probe'], 'sample_count': len(probe)}}
        mean_loss = old.accumulate_mean_generic_gradient(model, tokens, eligible, loss_spec, torch)
        del tokens
        scale = old.global_tangent_scale(eligible, 0.00125, torch)
        tangents, matrices, realized = old.prepare_tangents(eligible, scale, torch)
        try:
            primal, response, readout = old.compute_probe_response(model, probe, tangents, probe_spec, torch)
        finally:
            tangents.clear()
            model.zero_grad(set_to_none=True)
        old.verify_model_unchanged(model, before, torch)
        response32 = response.float().contiguous()
        if not smoke_only and not torch.equal(primal.float().contiguous(), merged_old):
            raise ValueError('Unperturbed merged activation mean differs from frozen Attempt014 mean')
        if smoke_only:
            new_records.append({'name': corpus['name'], 'mean_generic_loss': mean_loss,
                                'gradient_norm': scale['aggregate_generic_gradient_norm'],
                                'response_norm': float(torch.linalg.vector_norm(response.double()))})
        else:
            responses.append(response32)
            new_records.append({'name': corpus['name'], 'corpus_manifest_sha256': hashes[freezer.resolve_path(corpus['manifest_path'])],
                                'corpus_artifact_sha256': hashes[freezer.resolve_path(corpus['tokens_path'])],
                                'corpus_revision': corpus_manifest['revision'],
                                'mean_generic_loss': mean_loss, **scale, **realized,
                                'matrices': matrices, 'readout_checks': readout})
        del primal, response, response32
    if smoke_only:
        return {'smoke_only': True, 'new_corpora': new_records,
                'old_attempt014_responses_validated': True, 'write_outputs': False}
    candidate, geometry = response_geometry(responses)
    artifact = {'merged_mean': merged_old, **{'response_'+name: value for name, value in zip(ORDER, responses)},
                'consensus_response_8': candidate}
    if list(artifact) != spec['outputs']['tensors']:
        raise ValueError('Frozen eight-response artifact inventory mismatch')
    for value in artifact.values():
        if (value.dtype != torch.float32 or value.device.type != 'cpu' or not value.is_contiguous() or
                tuple(value.shape) != (128, 2048) or not bool(torch.isfinite(value).all())):
            raise ValueError('Malformed construction output tensor')
    if old.checkpoint_file_records(model_dir) != source_files or any(old.sha256_file(path) != digest for path, digest in hashes.items()):
        raise ValueError('Frozen input or final model changed during construction')
    no_outputs(artifact_path, manifest_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open('xb') as stream:
        torch.save(artifact, stream)
    result = {'format_version': 1, 'attempt_id': ATTEMPT,
              'purpose': spec['purpose'], 'information_policy': spec['information_policy'],
              'hash_algorithm': 'sha256', 'spec_sha256': hashes[freezer.SPEC_PATH],
              'constructor_sha256': hashes[Path(__file__).resolve()],
              'freezer_sha256': hashes[FREEZER_PATH], 'source_checkpoint': source_files,
              'old_attempt014': {**prior_hashes,
                                'raw_tensors_sha256': spec['attempt014']['raw_sha256']},
              'corpus_lock_sha256': hashes[freezer.resolve_path(spec['corpus_lock_path'])],
              'probe': spec['probe'],
              'probe_sha256': hashes[probe_path], 'corpus_order': list(ORDER),
              'new_corpora': new_records, 'generic_loss': spec['generic_loss'],
              'tangent': spec['tangent'], 'jvp': spec['jvp'], 'readout': spec['readout'],
              'consensus': spec['consensus'], 'blind_geometry': geometry,
              'workload': spec['workload'], **spec['firewall'],
              'artifact': {'path': str(artifact_path),
                           'serialized_sha256': old.sha256_file(artifact_path),
                           'raw_tensors_sha256': {name: old.sha256_raw_float32_tensor(value, torch)
                                                  for name, value in artifact.items()}}}
    if old.checkpoint_file_records(model_dir) != source_files or any(old.sha256_file(path) != digest for path, digest in hashes.items()):
        raise ValueError('Frozen input or final model changed before manifest publication')
    old.write_manifest(manifest_path, result)
    return {'smoke_only': False, 'manifest_path': str(manifest_path),
            'artifact_sha256': result['artifact']['serialized_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(construct(smoke_only=args.smoke_only), allow_nan=False))


if __name__ == '__main__':
    main()
