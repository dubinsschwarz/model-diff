#!/usr/bin/env python3
"""Privileged read-only Attempt202 diagnostic of all frozen Attempt201 endpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '202_hard_self_distill_ll_patchscope_characterization'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '8dc49f14b9e57dc43ed75189429a926edac0dd75dd215d5616990260cae3f868'
SEEDS = tuple(f'response_seed_{i}' for i in range(8))
NAMES = (*SEEDS, 'response_raw_mean', 'response_seed_consensus')
NATIVE_NAMES = NAMES[:-1]
ORACLE = 'oracle_difference'
MODES = ('oracle_norm_matched', 'native_amplitude')


def log(message):
    print(message, flush=True)


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT/path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('Frozen SHA256 mismatch: ' + str(path))


def read_record(record):
    path = path_of(record['path'])
    require_hash(path, record['sha256'])
    return json.loads(path.read_text())


def import_pinned(record, name):
    path = path_of(record['path'])
    require_hash(path, record['sha256'])
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def require_output_absent(path):
    if path.exists() or path.is_symlink():
        raise ValueError('Output already exists: ' + str(path))


def load_spec():
    spec = read_record({'path': str(SPEC_PATH), 'sha256': SPEC_SHA256})
    if (spec['attempt_id'] != ATTEMPT or spec['fixed_vectors'] != list(NAMES) or
            spec['patchscope']['regimes'] != dict(zip(MODES, (list(NAMES), list(NATIVE_NAMES)))) or
            spec['diagnostic_only'] is not True or spec['new_blind_claims'] is not False or
            spec['no_candidate_selection_combination_or_modification'] is not True or
            spec['oracle_reference']['reference_only'] is not True or
            spec['result_overwrite_refusal'] is not True):
        raise ValueError('Attempt202 fixed diagnostic plan mismatch')
    return spec


def validate_attempt201(spec):
    """Complete the existing frozen audit before any model/oracle analysis."""
    pins = spec['attempt201']
    read_record(pins['spec'])
    pinned_manifest = read_record(pins['construction_manifest'])
    c = import_pinned(pins['constructor'], 'attempt202_frozen_constructor')
    blind_spec = c.load_spec()
    candidates, _, manifest, receipt = c.validate_frozen(blind_spec)
    if (manifest != pinned_manifest or manifest['candidate'] != pins['candidate'] or
            manifest['spec_sha256'] != pins['spec']['sha256'] or
            manifest['constructor_sha256'] != pins['constructor']['sha256'] or
            receipt['construction_manifest_sha256'] != pins['construction_manifest']['sha256'] or
            list(candidates) != list(NAMES) or list(pins['candidate']['raw_sha256']) != list(NAMES)):
        raise ValueError('Attempt201 frozen inventory/provenance mismatch')
    validate_candidate_hashes(candidates, spec, c)
    log('ATTEMPT201_FROZEN_VALIDATED; read-only Attempt202 analysis enabled')
    return candidates, c, receipt


def validate_candidate_hashes(candidates, spec, c):
    import torch
    if list(candidates) != list(NAMES):
        raise ValueError('Attempt201 candidate inventory changed')
    for name in NAMES:
        c.b.validate_tensor(candidates[name], (128, 2048), torch.float32)
        if c.b.raw_hash(candidates[name]) != spec['attempt201']['candidate']['raw_sha256'][name]:
            raise ValueError('Attempt201 raw candidate hash mismatch: ' + name)


def validate_privileged_inputs(spec):
    prior = read_record(spec['attempt133']['spec'])
    for record in (spec['attempt133']['source'], spec['oracle_reader']):
        require_hash(path_of(record['path']), record['sha256'])
    for key in ('positions', 'primary_positions', 'top_k', 'semantic_substrings', 'target_prompts', 'logit_lens'):
        if spec[key] != prior[key]:
            raise ValueError('Attempt133 convention changed: ' + key)
    if any(spec['patchscope'][key] != value for key, value in prior['patchscope'].items()):
        raise ValueError('Attempt133 Patchscope convention changed')
    for key, record in spec['privileged_inputs'].items():
        if record['sha256'] != prior['immutable_input_sha256'][key]:
            raise ValueError('Attempt133 reference pin changed: ' + key)
        require_hash(path_of(record['path']), record['sha256'])
    if (spec['oracle_reader']['sha256'] != prior['immutable_input_sha256']['historical_oracle_logit_lens_source'] or
            spec['oracle_reference']['raw_sha256'] != prior['fixed_vectors'][-1]['raw_sha256']):
        raise ValueError('Attempt133 oracle source/reference mismatch')
    return read_record(spec['privileged_inputs']['historical_oracle_logit_lens'])


def configured_readout(spec, names):
    """Isolated pinned Attempt133 module; change only its fixed vector inventory."""
    a = import_pinned(spec['attempt133']['source'], 'attempt202_readout_' + str(len(names)))
    a.BLIND_NAMES = tuple(names)
    a.VECTOR_NAMES = (*names, ORACLE)
    return a


def patch_metrics_with_norms(metric_function, candidate_delta, oracle_delta, candidate_probs, baseline_probs):
    # All Attempt133 metrics and signed comparisons are retained verbatim.
    metrics = metric_function(candidate_delta, oracle_delta, candidate_probs, baseline_probs)
    candidate_norm = float(np.linalg.norm(np.asarray(candidate_delta, dtype=np.float64)))
    oracle_norm = float(np.linalg.norm(np.asarray(oracle_delta, dtype=np.float64)))
    return {**metrics, 'delta_logit_l2_norm': candidate_norm,
            'oracle_delta_logit_l2_norm': oracle_norm,
            'delta_logit_norm_ratio_to_oracle': candidate_norm/oracle_norm}


def patchscope_mode(spec, vectors, mode, model, tokenizer, torch):
    if mode not in MODES:
        raise ValueError('Uncommitted Patchscope scaling regime')
    names = NAMES if mode == MODES[0] else NATIVE_NAMES
    a = configured_readout(spec, names)
    original_metrics = a.patch_metrics
    # The wrapper calls the unmodified metric function; no recursion or new choice.
    a.patch_metrics = lambda *args: patch_metrics_with_norms(original_metrics, *args)
    if mode == 'native_amplitude':
        a.norm_match_blind = lambda vector, _oracle: np.asarray(vector, dtype=np.float32).copy()
    original_next = a.next_logits
    passes = 0
    def progress_next(*args):
        nonlocal passes
        output = original_next(*args)
        passes += 1
        if passes % 20 == 0:
            log(f'{mode}: model forward {passes}')
        return output
    a.next_logits = progress_next
    # This is the exact Attempt133 loop: 8 prompts, 5 positions, +/-, 6 greedy
    # plus-only tokens at positions 1..4, always at the original final prompt token.
    result = a.patchscope_readout(vectors, model, tokenizer, torch)
    result['mode'] = mode
    result['fixed_candidate_inventory'] = list(names)
    result['norm_matching'] = ('evaluation_only_per_position_to_oracle_l2_oracle_native_unchanged'
                               if mode == MODES[0] else 'native_unchanged_no_amplitude_scaling')
    result['model_forward_count'] = passes
    result['patch_vector_norm_ratios_to_oracle'] = {
        name: {str(pos): float(np.linalg.norm(vectors[name][pos].numpy().astype(np.float64)) /
                              np.linalg.norm(vectors[ORACLE][pos].numpy().astype(np.float64)))
               for pos in a.POSITIONS} for name in names} if mode == 'native_amplitude' else None
    return result


def distribution(values):
    if len(values) != 8 or any(not math.isfinite(value) for value in values):
        raise ValueError('Expected eight finite seed diagnostic scores')
    return {'mean': math.fsum(values)/8, 'median': statistics.median(values),
            'min': min(values), 'max': max(values), 'count_positive': sum(value > 0 for value in values),
            'seed_values_in_fixed_order': values}


def seed_summaries(patchscope):
    summaries = patchscope['oracle_similarity']
    result = {}
    for sign in ('plus', 'minus'):
        result[sign] = {}
        for group in ('position_0_mean_prompts', 'mean_positions_1_4_and_prompts'):
            values = [summaries[name][sign][group]['delta_logit_cosine'] for name in SEEDS]
            result[sign][group] = {'delta_logit_cosine_across_eight_seeds': distribution(values),
                'raw_mean': summaries['response_raw_mean'][sign][group],
                'unit_consensus': summaries['response_seed_consensus'][sign][group]}
    return result


def load_oracle_reader(spec):
    # Historical reader imports its established provenance helper from scripts/.
    sys.path.insert(0, str(PROJECT/'scripts'))
    try:
        return import_pinned(spec['oracle_reader'], 'attempt202_oracle_reader')
    finally:
        sys.path.pop(0)


def run(device='cuda'):
    import torch
    spec = load_spec()
    output = path_of(spec['output'])
    require_output_absent(output)
    source_hash = sha256_file(Path(__file__))
    candidates, c, receipt = validate_attempt201(spec)
    historical = validate_privileged_inputs(spec)
    reader = load_oracle_reader(spec)
    loc = spec['input_locations']
    provenance = reader.verify_inputs_before_loading(path_of(loc['base_tokenizer_directory']),
        path_of(loc['merged_checkpoint_directory']), path_of(loc['oracle_adl']))
    oracle = reader.load_and_validate_oracle(path_of(loc['oracle_adl']), provenance, torch)
    a = configured_readout(spec, NAMES)
    oracle_vector = a.validate_tensor(oracle['difference'], spec['oracle_reference']['raw_sha256'],
                                     torch, reader.sha256_raw_float32_tensor)
    vectors = {**candidates, ORACLE: oracle_vector}
    model, tokenizer, final_norm, lm_head = reader.load_local_model_and_tokenizer(
        path_of(loc['base_tokenizer_directory']), path_of(loc['merged_checkpoint_directory']), provenance, torch)
    if len(model.model.layers) != 28:
        raise ValueError('Merged Qwen3 layer count changed')
    log('Logit Lens: all ten fixed candidates and oracle, positions 0..4')
    lens = a.lens_readout(vectors, final_norm, lm_head, tokenizer, torch, reader)
    a.validate_historical_oracle_lens(lens, historical, provenance)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Requested CUDA is unavailable')
    model.to(device).eval().requires_grad_(False)
    patchscope = {mode: patchscope_mode(spec, vectors, mode, model, tokenizer, torch) for mode in MODES}
    # Repeat input audits and verify the original in-memory tensors before writing
    # only this experiment's result. No Attempt201 result/scores are needed.
    validate_attempt201(spec)
    validate_privileged_inputs(spec)
    validate_candidate_hashes(candidates, spec, c)
    a.validate_tensor(oracle_vector, spec['oracle_reference']['raw_sha256'], torch, reader.sha256_raw_float32_tensor)
    reader.verify_inputs_before_loading(path_of(loc['base_tokenizer_directory']),
        path_of(loc['merged_checkpoint_directory']), path_of(loc['oracle_adl']))
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), source_hash)
    result = {'format_version': 1, 'attempt_id': ATTEMPT, 'diagnostic_only': True,
        'new_blind_claims': False, 'clean_heldout_validation': False,
        'no_candidate_selection_combination_or_modification': True,
        'fixed_candidate_inventory': list(NAMES), 'oracle_reference': spec['oracle_reference'],
        'native_unit_consensus_excluded': True,
        'logit_lens': lens, 'patchscope': patchscope,
        'norm_matched_seed_distributions': seed_summaries(patchscope['oracle_norm_matched']),
        'native_seed_and_raw_mean_delta_logit_scores': patchscope['native_amplitude']['oracle_similarity'],
        'provenance': {'spec_sha256': SPEC_SHA256, 'runner_sha256': source_hash,
            'attempt201': spec['attempt201'], 'attempt201_freeze_receipt': receipt,
            'attempt133': spec['attempt133'], 'oracle_reader': spec['oracle_reader'],
            'privileged_inputs': spec['privileged_inputs'],
            'validated_checkpoint_provenance': {key: provenance[key] for key in
                ('base', 'fine_tuned', 'downloaded_model_hashes_sha256', 'merged_model_hashes_sha256',
                 'oracle_probe_manifest_sha256', 'oracle_adl_manifest_sha256', 'oracle_adl_sha256')}},
        'full_vocabulary_arrays': 'transient; only hashes, top20 records and full-vocabulary metrics persisted'}
    require_output_absent(output)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    log('Wrote ' + str(output))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    try:
        run(args.device)
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print('ERROR: ' + str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
