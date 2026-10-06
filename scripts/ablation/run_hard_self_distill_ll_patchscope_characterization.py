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
SPEC_SHA256 = '37c83e12e2ba0a4d5502e2cd3efc014ea3cb03eab3f1971d93fce3d149cd2be5'
SEEDS = tuple(f'response_seed_{i}' for i in range(8))
NAMES = (*SEEDS, 'response_raw_mean', 'response_seed_consensus')
NATIVE_NAMES = NAMES[:-1]
ORACLE = 'oracle_difference'
MODES = ('oracle_norm_matched', 'native_amplitude')
GENERATOR_SHA256 = 'd9bf1fb1e9581beb81708ece7ffdeed87ec821e0b0c66a1341ea5deb443c23ac'
LOCAL_ORACLE_TOP20_ABS_TOL = 2e-5


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
            spec['result_overwrite_refusal'] is not True or
            spec['oracle_portability']['generator']['sha256'] != GENERATOR_SHA256 or
            spec['oracle_portability']['local_top20_probability_absolute_tolerance'] != LOCAL_ORACLE_TOP20_ABS_TOL):
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
        if key != 'oracle_adl_artifact':
            require_hash(path_of(record['path']), record['sha256'])
    require_hash(path_of(spec['oracle_portability']['generator']['path']), GENERATOR_SHA256)
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
    require_hash(path_of(spec['oracle_portability']['generator']['path']), GENERATOR_SHA256)
    sys.path.insert(0, str(PROJECT/'scripts'))
    try:
        return import_pinned(spec['oracle_reader'], 'attempt202_oracle_reader')
    finally:
        sys.path.pop(0)


def generate_local_oracle(spec, torch):
    """Call the exact pinned canonical procedure with scratch-only outputs."""
    cfg, loc = spec['oracle_portability'], spec['input_locations']
    artifact, manifest = path_of(cfg['local_artifact']), path_of(cfg['local_manifest'])
    protected = {path_of(loc['oracle_adl']).resolve(),
                 path_of(spec['privileged_inputs']['oracle_adl_manifest']['path']).resolve()}
    if artifact.resolve() in protected or manifest.resolve() in protected or artifact == manifest:
        raise ValueError('Local oracle output overlaps historical provenance')
    generator = import_pinned(cfg['generator'], 'attempt202_canonical_oracle_generator')
    base, final = path_of(loc['base_tokenizer_directory']), path_of(loc['merged_checkpoint_directory'])
    probe, provenance = generator.validate_inputs(base, final, path_of(cfg['canonical_probe']),
                                                  artifact, manifest, torch)
    log('Generating local oracle with pinned canonical procedure: two 10,000-row probe passes')
    stored = generator.compute_oracle(base, final, probe, provenance, torch)
    generator.save_outputs(stored, provenance['hidden_size'], provenance, torch,
        artifact_path=artifact, manifest_path=manifest, script_path=path_of(cfg['generator']['path']))


def validate_oracle_tensors(oracle, manifest, reader, torch):
    for name in ('base_mean', 'ft_mean', 'difference'):
        value = oracle[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or
                value.dtype != torch.float32 or tuple(value.shape) != (128, 2048) or
                not value.is_contiguous() or not bool(torch.isfinite(value).all()) or
                reader.sha256_raw_float32_tensor(value, torch) != manifest['raw_tensors_sha256'][name]):
            raise ValueError('Local/selected oracle tensor shape/dtype/raw hash mismatch: ' + name)


def validate_oracle_reference(spec, reader, torch, artifact_path, manifest_path, fallback):
    """Validate actual checkpoint/probe identities and every stored raw tensor."""
    cfg, loc = spec['oracle_portability'], spec['input_locations']
    require_hash(path_of(cfg['generator']['path']), GENERATOR_SHA256)
    historical_manifest = read_record(spec['privileged_inputs']['oracle_adl_manifest'])
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError('Malformed local/selected oracle manifest')
    # Only serialized/raw floating-point hashes may differ from the historical
    # manifest. All scientific settings, model identities and probe links match.
    variable = {'oracle_adl_sha256', 'raw_tensors_sha256'}
    if ({key: value for key, value in manifest.items() if key not in variable} !=
            {key: value for key, value in historical_manifest.items() if key not in variable}):
        raise ValueError('Local oracle manifest canonical provenance mismatch')
    if not fallback:
        require_hash(manifest_path, spec['privileged_inputs']['oracle_adl_manifest']['sha256'])
        require_hash(artifact_path, spec['privileged_inputs']['oracle_adl_artifact']['sha256'])
    provenance = reader.verify_inputs_before_loading(path_of(loc['base_tokenizer_directory']),
        path_of(loc['merged_checkpoint_directory']), artifact_path, oracle_manifest_path=manifest_path,
        compute_script_path=path_of(cfg['generator']['path']))
    if fallback:
        generator = import_pinned(cfg['generator'], 'attempt202_local_oracle_probe_validator')
        probe_manifest = read_record(spec['privileged_inputs']['oracle_probe_manifest'])
        _, raw_probe = generator.validate_probe_manifest(probe_manifest, path_of(cfg['canonical_probe']),
            path_of(spec['privileged_inputs']['downloaded_model_hashes']['path']),
            generator.DATASET_LOCK_PATH, generator.FREEZE_PROBE_SCRIPT_PATH)
        generator.load_and_validate_probe(path_of(cfg['canonical_probe']), raw_probe, torch)
    oracle = reader.load_and_validate_oracle(artifact_path, provenance, torch)
    validate_oracle_tensors(oracle, manifest, reader, torch)
    require_hash(manifest_path, manifest_sha)
    require_hash(artifact_path, manifest['oracle_adl_sha256'])
    selection = {'artifact_path': str(artifact_path), 'manifest_path': str(manifest_path),
        'artifact_sha256': manifest['oracle_adl_sha256'], 'manifest_sha256': manifest_sha,
        'raw_tensors_sha256': manifest['raw_tensors_sha256'], 'fallback': fallback}
    return oracle, provenance, selection


def load_oracle_reference(spec, reader, torch):
    historical = spec['privileged_inputs']['oracle_adl_artifact']
    path = path_of(historical['path'])
    exact = path.is_file() and sha256_file(path) == historical['sha256']
    if exact:
        manifest = path_of(spec['privileged_inputs']['oracle_adl_manifest']['path'])
    else:
        cfg = spec['oracle_portability']
        path, manifest = path_of(cfg['local_artifact']), path_of(cfg['local_manifest'])
        present = [file.exists() or file.is_symlink() for file in (path, manifest)]
        if present[0] != present[1]:
            raise ValueError('Half-present local oracle cache; refusing regeneration/overwrite')
        if not any(present):
            generate_local_oracle(spec, torch)
    return validate_oracle_reference(spec, reader, torch, path, manifest, not exact)


def validate_oracle_lens_compatibility(spec, a, oracle_vector, final_norm, lm_head,
                                     tokenizer, torch, reader, historical, provenance, *, fallback=False):
    # Run ONLY the oracle here: no candidate similarity is computed before this
    # acceptance check. Only a validated local fallback uses the absolute bound.
    oracle_readout = configured_readout(spec, ())
    lens = oracle_readout.lens_readout({ORACLE: oracle_vector}, final_norm, lm_head, tokenizer, torch, reader)
    compatibility = validate_historical_lens_reference(spec, a, lens, historical, provenance, fallback=fallback)
    log('Historical oracle Logit-Lens compatibility passed (positions 0..4)')
    return compatibility


def validate_historical_lens_reference(spec, a, lens, historical, provenance, *, fallback=False):
    # Canonical checkpoint/probe identities stay exact. The pinned historical
    # artifact identifiers are used only for this historical token comparison;
    # the actual local artifact identifiers are retained separately in results.
    comparison = dict(provenance)
    for key, pin in (('oracle_adl_manifest_sha256', 'oracle_adl_manifest'),
                     ('oracle_adl_sha256', 'oracle_adl_artifact')):
        expected = spec['privileged_inputs'][pin]['sha256']
        if historical['provenance'][key] != expected:
            raise ValueError('Historical oracle lens reference provenance mismatch')
        comparison[key] = expected
    if not fallback:
        # Preserve the preferred exact-artifact path's Attempt133 gate verbatim.
        a.validate_historical_oracle_lens(lens, historical, comparison)
    else:
        for key in ('base', 'fine_tuned', 'downloaded_model_hashes_sha256',
                    'merged_model_hashes_sha256', 'oracle_probe_manifest_sha256',
                    'oracle_adl_manifest_sha256', 'oracle_adl_sha256'):
            if historical['provenance'].get(key) != comparison.get(key):
                raise ValueError('Historical Logit Lens provenance mismatch: ' + key)
        if len(historical['positions']) != len(a.POSITIONS):
            raise ValueError('Historical Logit Lens position inventory changed')

    maximum_difference = 0.
    all_ordered_exact, all_top1_rank1 = True, True
    for pos, record in zip(a.POSITIONS, historical['positions']):
        if fallback and record['position'] != pos:
            raise ValueError('Historical Logit Lens position order changed')
        for polarity in ('positive', 'negative'):
            expected = record['difference'][polarity]
            actual = lens['tokens'][ORACLE][str(pos)]['top20_' + polarity]
            if fallback and (len(expected) != a.TOP_K or len(actual) != a.TOP_K):
                raise ValueError('Local oracle top20 inventory mismatch')
            top1_rank1 = actual[0]['token_id'] == expected[0]['token_id']
            ordered_exact = [row['token_id'] for row in actual] == [row['token_id'] for row in expected]
            all_top1_rank1 &= top1_rank1
            all_ordered_exact &= ordered_exact
            if fallback and not top1_rank1:
                raise ValueError(f'Local oracle historical top1 is not rank 1: {pos}/{polarity}')
            if fallback and not ordered_exact:
                raise ValueError(f'Local oracle ordered top20 token-ID mismatch: {pos}/{polarity}')
            for local, old in zip(actual, expected):
                lp, hp = float(local['probability']), float(old['probability'])
                difference = abs(lp - hp)
                if fallback and (not math.isfinite(lp) or not math.isfinite(hp) or
                        not 0 <= lp <= 1 or not 0 <= hp <= 1 or difference > LOCAL_ORACLE_TOP20_ABS_TOL):
                    raise ValueError(f'Local oracle top20 probability exceeds absolute bound: {pos}/{polarity}')
                maximum_difference = max(maximum_difference, difference)
    return {'oracle_logit_lens_compatibility_rule':
                spec['oracle_portability']['compatibility'] if fallback else 'attempt133_rtol_1e-6_atol_1e-8_exact_historical_artifact',
            'max_observed_absolute_top20_probability_difference': maximum_difference,
            'all_ordered_top20_exact': all_ordered_exact,
            'all_historical_top1_rank1': all_top1_rank1,
            'top20_probability_absolute_threshold': LOCAL_ORACLE_TOP20_ABS_TOL if fallback else None}


def oracle_portability_provenance(spec, selection):
    return {'historical_oracle_artifact_exact_match': not selection['fallback'],
        'oracle_portability_fallback_used': selection['fallback'],
        'historical_expected_oracle_artifact_sha256': spec['privileged_inputs']['oracle_adl_artifact']['sha256'],
        'local_oracle_artifact_sha256': selection['artifact_sha256'],
        'historical_expected_difference_raw_sha256': spec['oracle_reference']['raw_sha256'],
        'local_difference_raw_sha256': selection['raw_tensors_sha256']['difference'],
        'local_base_mean_raw_sha256': selection['raw_tensors_sha256']['base_mean'],
        'local_ft_mean_raw_sha256': selection['raw_tensors_sha256']['ft_mean'],
        'local_oracle_manifest_sha256': selection['manifest_sha256'],
        'pinned_generator_sha256': GENERATOR_SHA256,
        'historical_logit_lens_compatibility_passed': True,
        'selected_oracle_artifact_path': selection['artifact_path'],
        'selected_oracle_manifest_path': selection['manifest_path']}


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
    oracle, provenance, selection = load_oracle_reference(spec, reader, torch)
    a = configured_readout(spec, NAMES)
    oracle_vector = a.validate_tensor(oracle['difference'], selection['raw_tensors_sha256']['difference'],
                                     torch, reader.sha256_raw_float32_tensor)
    vectors = {**candidates, ORACLE: oracle_vector}
    model, tokenizer, final_norm, lm_head = reader.load_local_model_and_tokenizer(
        path_of(loc['base_tokenizer_directory']), path_of(loc['merged_checkpoint_directory']), provenance, torch)
    if len(model.model.layers) != 28:
        raise ValueError('Merged Qwen3 layer count changed')
    compatibility = validate_oracle_lens_compatibility(spec, a, oracle_vector, final_norm, lm_head,
        tokenizer, torch, reader, historical, provenance, fallback=selection['fallback'])
    log('Logit Lens: all ten fixed candidates and oracle, positions 0..4')
    lens = a.lens_readout(vectors, final_norm, lm_head, tokenizer, torch, reader)
    validate_historical_lens_reference(spec, a, lens, historical, provenance, fallback=selection['fallback'])
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Requested CUDA is unavailable')
    model.to(device).eval().requires_grad_(False)
    patchscope = {mode: patchscope_mode(spec, vectors, mode, model, tokenizer, torch) for mode in MODES}
    # Repeat input audits and verify the original in-memory tensors before writing
    # only this experiment's result. No Attempt201 result/scores are needed.
    validate_attempt201(spec)
    validate_privileged_inputs(spec)
    validate_candidate_hashes(candidates, spec, c)
    validate_oracle_tensors(oracle, {'raw_tensors_sha256': selection['raw_tensors_sha256']}, reader, torch)
    artifact_path, manifest_path = path_of(selection['artifact_path']), path_of(selection['manifest_path'])
    require_hash(artifact_path, selection['artifact_sha256'])
    require_hash(manifest_path, selection['manifest_sha256'])
    _, _, checked_selection = validate_oracle_reference(spec, reader, torch, artifact_path,
                                                       manifest_path, selection['fallback'])
    if checked_selection != selection:
        raise ValueError('Selected oracle reference changed during analysis')
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
            **oracle_portability_provenance(spec, selection),
            **compatibility,
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
