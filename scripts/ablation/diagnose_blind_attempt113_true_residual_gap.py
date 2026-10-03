#!/usr/bin/env python3
"""Blind, read-only equation-residual audit of Attempt113 CG iteration 40."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '115_blind_attempt113_true_residual_gap_diagnostic'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
RESULT_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'result.json'
PRIOR113_SPEC = PROJECT/'experiments/attempts/113_blind_diverse8_unit_gradient_inverse_ggn_candidate/spec.json'
PRIOR113_SOURCE = PROJECT/'scripts/ablation/construct_blind_diverse8_unit_gradient_inverse_ggn.py'
PRIOR114_SPEC = PROJECT/'experiments/attempts/114_blind_attempt113_cg_conditioning_diagnostic/spec.json'
PRIOR114_SOURCE = PROJECT/'scripts/ablation/diagnose_blind_attempt113_cg_conditioning.py'
PRIOR114_RESULT = PROJECT/'experiments/attempts/114_blind_attempt113_cg_conditioning_diagnostic/result.json'
CHECKPOINT = Path('/root/model-diff-scratch/checkpoints/attempt113_blind_inverse_cg80_iter40.pt')
FROZEN = {
    '113_commit': '5357ba00408f07b284de3126227fc5eb317e4f13',
    '113_spec_sha256': '7c7f508575e216cd29eee967f31101b1c614f85cc0cb0465becd4e0d1dc8bfb6',
    '113_source_sha256': '02bc38350e0f08997f3f2e6942c89594abb4b2e30e042973b930d89e59374ee9',
    '114_commit': '6b2e51ed87fd7147c5b304c4b7f4827dcdf6a4ba',
    '114_spec_sha256': 'bb477e5aaf66e9d33d75bedcc700a7b389bb47c7c3be00a25999d48e15b1e009',
    '114_source_sha256': '781a604db74189631cee26e2b9caea75a2b24c0faef7f183ab6649b5014379fc',
    '114_result_sha256': '151d1074cdcf5d5d3152bf1e2f746940ea287de88fdc597a737db83cbf74a345',
    'checkpoint_sha256': '4435097d563297d2d749d37a14d79d9ae44c70ddda1a0b90e6f87f00d4c64113',
    'checkpoint_size_bytes': 8455847711,
}
ORDER = ('fineweb', 'wikitext103_raw', 'tinystories', 'arxiv_document',
         'cc_news', 'pg19', 'codeparrot_clean', 'ultrachat')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec(path: Path = SPEC_PATH) -> dict:
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt115 spec path changed')
    spec = json.loads(path.read_text())
    a113, a114 = spec.get('attempt113', {}), spec.get('attempt114', {})
    if (spec.get('attempt_id') != ATTEMPT or
            spec.get('result_path') != str(RESULT_PATH.relative_to(PROJECT)) or
            spec.get('output_policy') != 'small_JSON_only_atomic_no_overwrite_checkpoint_read_only' or
            a113.get('commit') != FROZEN['113_commit'] or
            a113.get('spec_path') != str(PRIOR113_SPEC.relative_to(PROJECT)) or
            a113.get('source_path') != str(PRIOR113_SOURCE.relative_to(PROJECT)) or
            a113.get('spec_sha256') != FROZEN['113_spec_sha256'] or
            a113.get('source_sha256') != FROZEN['113_source_sha256'] or
            a113.get('checkpoint_path') != str(CHECKPOINT) or
            a113.get('checkpoint_serialized_sha256') != FROZEN['checkpoint_sha256'] or
            a113.get('checkpoint_size_bytes') != FROZEN['checkpoint_size_bytes'] or
            a113.get('iteration') != 40 or a113.get('matrix_count') != 98 or
            a113.get('corpus_order') != list(ORDER) or
            a113.get('rhs_rows_per_corpus') != [0, 512] or
            a113.get('rhs_batch_size') != 8 or
            a113.get('rhs') != 'one_eighth_sum_of_eight_positive_global_unit_generic_gradients_without_final_normalization' or
            a113.get('ggn_rows') != [0, 64] or a113.get('ggn_denominator') != 8128 or
            a113.get('ggn_operator') != 'exact_endpoint_categorical_JtFJ' or
            a113.get('fisher_numerics') != 'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast' or
            a113.get('ggn_applications') != 1 or a113.get('damping_fraction') != 1e-4 or
            a114.get('commit') != FROZEN['114_commit'] or
            a114.get('spec_path') != str(PRIOR114_SPEC.relative_to(PROJECT)) or
            a114.get('source_path') != str(PRIOR114_SOURCE.relative_to(PROJECT)) or
            a114.get('result_path') != str(PRIOR114_RESULT.relative_to(PROJECT)) or
            a114.get('spec_sha256') != FROZEN['114_spec_sha256'] or
            a114.get('source_sha256') != FROZEN['114_source_sha256'] or
            a114.get('result_sha256') != FROZEN['114_result_sha256'] or
            spec.get('numerics') != {
                'primary_residual_arithmetic': 'matrixwise_CPU_float64_b_minus_FP32_Gx_minus_lambda_times_x',
                'global_reductions': 'matrixwise_CPU_float64_sum_then_math.fsum',
                'rhs_norm_comparison_rtol': 1e-6,
                'gradient_diagnostic_comparison_rtol': 1e-5,
                'checkpoint_scalar_comparison_rtol': 1e-12,
                'no_rhs_hash_tolerance': True, 'optional_FP32_residual': False} or
            spec.get('interpretation') != {
                'order': ['large_recursive_residual_gap',
                          'moderate_recursive_residual_gap',
                          'recursive_residual_faithful', 'mixed_residual_fidelity'],
                'large_if_gap_over_recursive_at_least': 0.5,
                'large_if_cosine_below': 0.9,
                'moderate_if_gap_over_recursive_at_least': 0.1,
                'moderate_if_cosine_below': 0.99,
                'faithful_if_gap_over_recursive_below': 0.1,
                'faithful_if_cosine_at_least': 0.99,
                'faithful_if_abs_relative_residual_difference_at_most_fraction_of_recursive': 0.1,
                'scope': 'numerical_fidelity_only_no_historical_quality_inference'} or
            spec.get('information_policy') != {
                'blind_diagnostic': True, 'final_checkpoint_access': True,
                'attempt113_checkpoint_access': True, 'generic_corpora_access': True,
                'historical_base_access': False, 'true_delta_access': False,
                'adapter_access': False, 'oracle_adl_access': False,
                'historical_target_access': False, 'prior_oracle_evaluation_access': False,
                'candidate_evaluation': False, 'candidate_selection': False,
                'oracle_based_tuning': False}):
        raise ValueError('Attempt115 frozen spec inventory mismatch')
    return spec


def _committed_hash(commit: str, path: Path) -> str:
    output = subprocess.run(
        ['git', '-c', f'safe.directory={PROJECT}', 'show',
         f'{commit}:{path.relative_to(PROJECT)}'],
        cwd=PROJECT, capture_output=True, check=True).stdout
    return hashlib.sha256(output).hexdigest()


def _import_hashed(path: Path, expected: str, name: str):
    if sha256_file(path) != expected:
        raise ValueError('Frozen blind source SHA256 mismatch: '+str(path))
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def validate_linkage(spec: dict):
    for commit, files in ((FROZEN['113_commit'],
                           ((PRIOR113_SPEC, FROZEN['113_spec_sha256']),
                            (PRIOR113_SOURCE, FROZEN['113_source_sha256']))),
                          (FROZEN['114_commit'],
                           ((PRIOR114_SPEC, FROZEN['114_spec_sha256']),
                            (PRIOR114_SOURCE, FROZEN['114_source_sha256']),
                            (PRIOR114_RESULT, FROZEN['114_result_sha256'])))):
        for path, expected in files:
            if sha256_file(path) != expected or _committed_hash(commit, path) != expected:
                raise ValueError('Frozen Attempt113/114 commit or file hash mismatch')
    m114 = _import_hashed(PRIOR114_SOURCE, FROZEN['114_source_sha256'], 'attempt114_blind_helper')
    m113, prior = m114.validate_attempt113_linkage(m114.load_spec())
    result = json.loads(PRIOR114_RESULT.read_text())
    fields = spec['attempt114']
    if (result.get('attempt_id') != '114_blind_attempt113_cg_conditioning_diagnostic' or
            result.get('information_policy', {}).get('blind_diagnostic') is not True or
            result.get('information_policy', {}).get('historical_base_access') is not False or
            result.get('ggn_applications') != 1 or
            result.get('checkpoint', {}).get('serialized_sha256') != FROZEN['checkpoint_sha256'] or
            result['checkpoint'].get('size_bytes') != FROZEN['checkpoint_size_bytes'] or
            result['checkpoint'].get('iteration') != 40 or
            result.get('source_provenance', {}).get('spec_sha256') != FROZEN['114_spec_sha256'] or
            result['source_provenance'].get('source_sha256') != FROZEN['114_source_sha256'] or
            result.get('frozen_attempt113', {}).get('commit') != FROZEN['113_commit']):
        raise ValueError('Frozen Attempt114 blind result provenance mismatch')
    expected = {
        'recursive_relative_residual_at_40': result['residual_curvature']['relative_residual_at_40'],
        'rho_b': result['residual_curvature']['rho_b'],
        'lambda': result['ritz']['lambda'],
        'rho_r40': result['residual_curvature']['rho_r40'],
        'ritz_condition_estimate': result['ritz']['ritz_condition_estimate']}
    if any(expected[key] != fields[key] for key in expected):
        raise ValueError('Frozen Attempt114 blind numerical value mismatch')
    return m113, m114, prior, result


def validate_checkpoint(m114, m113, prior, spec, attempt114_result):
    if (CHECKPOINT.is_symlink() or not CHECKPOINT.is_file() or
            CHECKPOINT.stat().st_size != FROZEN['checkpoint_size_bytes'] or
            sha256_file(CHECKPOINT) != FROZEN['checkpoint_sha256']):
        raise ValueError('Frozen Attempt113 checkpoint serialized SHA256/size mismatch')
    payload, record = m114.validate_checkpoint_read_only(
        CHECKPOINT, m113, prior, m114.load_spec())
    if (record != {'serialized_sha256': FROZEN['checkpoint_sha256'],
                   'size_bytes': FROZEN['checkpoint_size_bytes']} or
            payload['iteration'] != 40 or len(payload['iteration_records']) != 40 or
            payload['coordinate_inventory'] != attempt114_result['coordinate_inventory'] or
            payload['rho_b'] != spec['attempt114']['rho_b'] or
            payload['fixed_lambda'] != spec['attempt114']['lambda'] or
            not math.isclose(math.sqrt(payload['rr'])/payload['b_norm'],
                             spec['attempt114']['recursive_relative_residual_at_40'],
                             rel_tol=spec['numerics']['checkpoint_scalar_comparison_rtol'])):
        raise ValueError('Frozen Attempt113 checkpoint/Attempt114 result mismatch')
    return payload, record


def load_eight_corpora_without_probe(m113, prior):
    """Exact Attempt113 corpus provenance/loading path; omit its unused probe."""
    source = prior['blind_source_provenance']
    for key, value in source.items():
        if key.endswith('_path'):
            m113.require_hash(value, source[key[:-5]+'_sha256'])
    s100 = m113.freezer.load_spec()
    s014 = m113.blind.load_json_object(
        m113.safe_input_path(source['attempt014_spec_path']), 'blind Attempt014 spec')
    m100 = m113.blind.load_json_object(
        m113.safe_input_path(source['attempt100_manifest_path']),
        'blind Attempt100 construction manifest')
    if (s100['canonical_checkpoint_files'] != prior['final_checkpoint']['files'] or
            s100['model'] != prior['model'] or
            s100['eligible_tensors'] != prior['eligible_tensors'] or
            s100['readout'] != prior['readout'] or s100['jvp'] != prior['jvp'] or
            len(s014['corpora']) != 3 or
            m100.get('corpus_order') != list(ORDER) or
            m100.get('spec_sha256') != source['attempt100_spec_sha256'] or
            m100.get('constructor_sha256') != source['attempt100_constructor_sha256'] or
            m100.get('freezer_sha256') != source['attempt100_freezer_sha256'] or
            m100.get('corpus_lock_sha256') != source['attempt100_lock_sha256'] or
            m100.get('source_checkpoint') != prior['final_checkpoint']['files'] or
            m100.get('probe_sha256') != prior['probe']['serialized_sha256'] or
            any(m100.get(key) is not False for key in
                ('historical_base_access', 'oracle_adl_access', 'prior_evaluation_access'))):
        raise ValueError('Blind Attempt100/014 construction provenance mismatch')
    model_dir = m113.safe_input_path(prior['final_checkpoint']['directory'])
    if m113.blind.checkpoint_file_records(model_dir) != prior['final_checkpoint']['files']:
        raise ValueError('Canonical final checkpoint inventory mismatch')
    lock = m113.freezer.validate_lock(m113.blind.load_json_object(
        m113.safe_input_path(source['attempt100_lock_path']), 'blind corpus lock'), s100)
    tokens = {}
    for index, row in enumerate(prior['corpora']):
        if row['name'] != ORDER[index]:
            raise ValueError('Frozen eight-corpus order mismatch')
        manifest_path = m113.require_hash(row['manifest_path'], row['manifest_sha256'])
        if index < 3:
            frozen = s014['corpora'][index]
            for field in ('name', 'spec_path', 'spec_sha256', 'manifest_path', 'tokens_path'):
                if row[field] != frozen[field]:
                    raise ValueError('Frozen early blind corpus linkage mismatch')
            corpus, _, meta = m113.blind.validate_frozen_corpus(frozen, model_dir, torch)
            manifest = m113.blind.load_json_object(manifest_path, 'blind corpus manifest')
            if (meta['serialized_sha256'] != row['artifact_serialized_sha256'] or
                    meta['raw_tensor_sha256'] != row['artifact_raw_sha256']):
                raise ValueError('Frozen early corpus hash mismatch')
        else:
            frozen = s100['new_corpora'][index-3]
            if (row['name'] != frozen['name'] or
                    row['tokens_path'] != frozen['tokens_path'] or
                    row['manifest_path'] != frozen['manifest_path']):
                raise ValueError('Frozen new corpus linkage mismatch')
            corpus, manifest = m113.freezer.validate_corpus(
                frozen, s100, lock, model_dir, torch)
        if (tuple(corpus.shape) != (4096, 128) or corpus.dtype != torch.int64 or
                not corpus.is_contiguous() or
                m113.sha256_file(m113.safe_input_path(row['tokens_path'])) !=
                    row['artifact_serialized_sha256'] or
                m113.blind.sha256_raw_int64_tensor(corpus, torch) !=
                    row['artifact_raw_sha256'] or
                manifest.get('artifact', {}).get('serialized_sha256') !=
                    row['artifact_serialized_sha256'] or
                manifest.get('artifact', {}).get('raw_tensor_sha256') !=
                    row['artifact_raw_sha256']):
            raise ValueError('Frozen generic corpus tensor/manifest mismatch')
        tokens[row['name']] = corpus
    return tokens


def require_rebuilt_rhs(m113, rebuilt, rebuilt_record, diagnostics, checkpoint, spec):
    inventory = checkpoint['coordinate_inventory']
    raw_hashes, aggregate = m113.ordered_vector_hashes(rebuilt, inventory)
    expected = checkpoint['rhs_record']
    if (raw_hashes != expected['matrix_raw_sha256'] or
            aggregate != expected['ordered_matrix_hash_sha256'] or
            rebuilt_record['matrix_raw_sha256'] != raw_hashes or
            rebuilt_record['ordered_matrix_hash_sha256'] != aggregate or
            not math.isclose(rebuilt_record['norm'], expected['norm'],
                             rel_tol=spec['numerics']['rhs_norm_comparison_rtol']) or
            not math.isclose(rebuilt_record['norm'], checkpoint['b_norm'],
                             rel_tol=spec['numerics']['rhs_norm_comparison_rtol'])):
        raise ValueError('Rebuilt blind RHS is not byte-identical to frozen checkpoint RHS')
    frozen_rows = checkpoint['gradient_records']
    if len(diagnostics) != 8 or len(frozen_rows) != 8:
        raise ValueError('Rebuilt blind gradient inventory mismatch')
    differences = []
    for current, frozen in zip(diagnostics, frozen_rows):
        if (current['name'] != frozen['name'] or current['rows'] != frozen['rows'] or
                current['prediction_contexts'] != frozen['prediction_contexts'] or
                current['name'] not in ORDER):
            raise ValueError('Rebuilt per-corpus deterministic inventory mismatch')
        numeric = {}
        for key in ('mean_generic_loss', 'global_gradient_norm',
                    'unit_normalization_scalar'):
            a, b = current[key], frozen[key]
            if (not math.isfinite(a) or not math.isfinite(b) or
                    not math.isclose(a, b,
                                     rel_tol=spec['numerics']['gradient_diagnostic_comparison_rtol'],
                                     abs_tol=1e-10)):
                raise ValueError('Rebuilt per-corpus blind diagnostic mismatch')
            numeric[key+'_difference'] = a-b
        differences.append({'name': current['name'], **numeric})
    return {'norm': rebuilt_record['norm'], 'matrix_raw_sha256': raw_hashes,
            'ordered_matrix_hash_sha256': aggregate,
            'per_corpus_diagnostic_differences': differences,
            'byte_identical_to_checkpoint_rhs': True}


def residual_metrics(b, x, recursive, gx, damping, inventory, spec):
    """Subtract in CPU FP64, one matrix at a time; never persist large vectors."""
    if (not math.isfinite(damping) or damping <= 0 or
            any(set(vector) != {row['name'] for row in inventory}
                for vector in (b, x, recursive, gx))):
        raise ValueError('Residual matrix inventory/damping mismatch')
    pieces = {key: [] for key in ('b_sq', 'recursive_sq', 'true_sq', 'gap_sq', 'dot')}
    matrix_rows = []
    for row in inventory:
        name = row['name']
        tensors = (b[name], x[name], recursive[name], gx[name])
        if any(value.dtype != torch.float32 or list(value.shape) != row['shape'] or
               not bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError('Nonfinite/malformed residual matrix')
        b64, x64, r64, gx64 = (value.detach().to('cpu', dtype=torch.float64)
                               for value in tensors)
        true64 = b64-gx64-damping*x64
        gap64 = r64-true64
        quantities = {'b_sq': b64.square(), 'recursive_sq': r64.square(),
                      'true_sq': true64.square(), 'gap_sq': gap64.square(),
                      'dot': r64*true64}
        sums = {key: float(torch.sum(value, dtype=torch.float64))
                for key, value in quantities.items()}
        if not all(math.isfinite(value) for value in sums.values()):
            raise ValueError('Nonfinite FP64 residual reduction')
        for key, value in sums.items():
            pieces[key].append(value)
        matrix_rows.append({'name': name,
                            'recursive_residual_norm': math.sqrt(sums['recursive_sq']),
                            'true_residual_norm': math.sqrt(sums['true_sq']),
                            'gap_norm': math.sqrt(sums['gap_sq'])})
        del b64, x64, r64, gx64, true64, gap64, quantities
    totals = {key: math.fsum(values) for key, values in pieces.items()}
    bnorm, rnorm, tnorm, gapnorm = (math.sqrt(totals[key]) for key in
                                   ('b_sq', 'recursive_sq', 'true_sq', 'gap_sq'))
    if bnorm <= 0 or rnorm <= 0:
        raise ValueError('Zero RHS or recursive residual norm')
    cosine = totals['dot']/(rnorm*tnorm) if tnorm > 0 else None
    if cosine is not None and (not math.isfinite(cosine) or
                               abs(cosine) > 1+1e-8):
        raise ValueError('Invalid recursive/true residual cosine')
    result = {
        'arithmetic': spec['numerics']['primary_residual_arithmetic'],
        'squared_norms_and_dot': totals,
        'recursive_relative_residual': rnorm/bnorm,
        'true_relative_residual': tnorm/bnorm,
        'true_to_recursive_residual_norm_ratio': tnorm/rnorm,
        'residual_gap_relative_to_rhs': gapnorm/bnorm,
        'residual_gap_relative_to_recursive': gapnorm/rnorm,
        'cosine_recursive_vs_true': cosine,
        'matrixwise': matrix_rows}
    result['interpretation'] = classify_residual_fidelity(result, spec)
    return result


def classify_residual_fidelity(metrics, spec):
    rule = spec['interpretation']
    d = metrics['residual_gap_relative_to_recursive']
    c = metrics['cosine_recursive_vs_true']
    r = metrics['recursive_relative_residual']
    t = metrics['true_relative_residual']
    if (not all(math.isfinite(value) for value in (d, r, t)) or
            (c is not None and not math.isfinite(c)) or r <= 0):
        raise ValueError('Nonfinite residual-fidelity statistics')
    if d >= rule['large_if_gap_over_recursive_at_least'] or (c is not None and c < rule['large_if_cosine_below']):
        return 'large_recursive_residual_gap'
    if d >= rule['moderate_if_gap_over_recursive_at_least'] or (c is not None and c < rule['moderate_if_cosine_below']):
        return 'moderate_recursive_residual_gap'
    if (c is not None and d < rule['faithful_if_gap_over_recursive_below'] and
            c >= rule['faithful_if_cosine_at_least'] and
            abs(t-r) <= rule['faithful_if_abs_relative_residual_difference_at_most_fraction_of_recursive']*r):
        return 'recursive_residual_faithful'
    return 'mixed_residual_fidelity'


def atomic_json_no_overwrite(path: Path, result: dict):
    if path != RESULT_PATH or path.exists() or path.is_symlink():
        raise ValueError('Attempt115 output exists or path changed')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        with temporary.open('xb') as stream:
            stream.write((json.dumps(result, indent=2, sort_keys=True,
                                     allow_nan=False)+'\n').encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def run():
    spec = load_spec()
    if RESULT_PATH.exists() or RESULT_PATH.is_symlink():
        raise ValueError('Attempt115 result already exists')
    m113, m114, prior, a114_result = validate_linkage(spec)
    payload, checkpoint_record = validate_checkpoint(m114, m113, prior, spec, a114_result)
    inventory = payload['coordinate_inventory']
    x40, recursive = payload['x'], payload['r']
    frozen_rhs_record, frozen_gradient_records = payload['rhs_record'], payload['gradient_records']
    saved_rr, saved_b_norm, damping = payload['rr'], payload['b_norm'], payload['fixed_lambda']
    del payload  # Release p40 and other checkpoint-owned tensors before model loading.
    gc.collect()
    corpora = load_eight_corpora_without_probe(m113, prior)
    ggn = prior['ggn']
    if (ggn['rows'] != [0, 64] or ggn['denominator'] != 8128 or
            ggn['serialized_sha256'] !=
                '54283d72aa3075e2867ead86e538991d7284ede327ca67871e73bfc581e4025d' or
            ggn['raw_tensor_sha256'] !=
                '4613982be77342a89b35616d68e21d9ff66d5a983af7e82cb6fa1eb159457eae'):
        raise ValueError('Attempt113 frozen FineWeb G artifact/operator mismatch')
    model_dir = m113.safe_input_path(prior['final_checkpoint']['directory'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    started = time.perf_counter()
    model = m113.blind.load_local_model(model_dir, device, torch)
    m113.blind.validate_loaded_model(model, torch)
    eligible = m113.blind.discover_eligible_linear_weights(model, torch)
    m113.blind.freeze_other_parameters(model, eligible)
    if m113.coordinate_inventory(eligible) != inventory:
        raise ValueError('Frozen checkpoint/final model coordinate inventory mismatch')
    model_before = m113.blind.model_state_hashes(model, torch)
    print('Attempt115 | rebuilding eight blind unit-gradient corpora', flush=True)
    b_rebuilt, diagnostics, rhs_record = m113.build_blind_rhs(
        model, corpora, eligible, inventory)
    checkpoint_for_rhs = {'coordinate_inventory': inventory,
                          'rhs_record': frozen_rhs_record,
                          'gradient_records': frozen_gradient_records,
                          'b_norm': saved_b_norm}
    rhs_audit = require_rebuilt_rhs(m113, b_rebuilt, rhs_record, diagnostics,
                                    checkpoint_for_rhs, spec)
    m113.blind.verify_model_unchanged(model, model_before, torch)
    del corpora
    gc.collect()
    x_device = m113.vector_to_device(x40, eligible)
    tokens = m113.blind.load_corpus_tokens(
        m113.safe_input_path(ggn['tokens_path']), ggn['raw_tensor_sha256'],
        4096, 128, torch)[:64].contiguous()
    if m113.sha256_file(m113.safe_input_path(ggn['tokens_path'])) != ggn['serialized_sha256']:
        raise ValueError('Frozen FineWeb G serialized SHA256 mismatch')
    print('Attempt115 | one endpoint GGN application to x40', flush=True)
    ggn_audit = m113.ggn_action_stage(model, tokens, eligible, x_device, prior,
                                      progress_label='Attempt115 Gx40')
    gx = {name+'.weight': module.weight.grad for name, module in eligible}
    result_metrics = residual_metrics(b_rebuilt, x40, recursive, gx,
                                      damping, inventory, spec)
    if (not math.isclose(result_metrics['squared_norms_and_dot']['recursive_sq'], saved_rr,
                         rel_tol=1e-5, abs_tol=1e-7) or
            not math.isclose(math.sqrt(result_metrics['squared_norms_and_dot']['b_sq']),
                             saved_b_norm, rel_tol=spec['numerics']['rhs_norm_comparison_rtol']) or
            not math.isclose(result_metrics['recursive_relative_residual'],
                             spec['attempt114']['recursive_relative_residual_at_40'],
                             rel_tol=1e-5)):
        raise ValueError('Independent FP64 RHS/recursive residual disagrees with checkpoint')
    m113.blind.verify_model_unchanged(model, model_before, torch)
    model.zero_grad(set_to_none=True)
    del x_device, gx
    gc.collect()
    after = {'serialized_sha256': sha256_file(CHECKPOINT),
             'size_bytes': CHECKPOINT.stat().st_size}
    if after != checkpoint_record:
        raise ValueError('Attempt113 checkpoint changed during diagnostic')
    result = {
        'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'information_policy': spec['information_policy'],
        'frozen_linkage': {'attempt113': spec['attempt113'],
                           'attempt114': spec['attempt114']},
        'source_provenance': {'spec_sha256': sha256_file(SPEC_PATH),
                              'source_sha256': sha256_file(Path(__file__))},
        'checkpoint': {'path': str(CHECKPOINT), **checkpoint_record,
                       'read_only': True, 'unchanged_after_run': True,
                       'iteration': 40},
        'final_checkpoint_files': prior['final_checkpoint']['files'],
        'corpus_order': list(ORDER), 'corpora': [
            {'name': row['name'], 'manifest_sha256': row['manifest_sha256'],
             'artifact_serialized_sha256': row['artifact_serialized_sha256'],
             'artifact_raw_sha256': row['artifact_raw_sha256']}
            for row in prior['corpora']],
        'ggn_corpus': {key: ggn[key] for key in
                       ('tokens_path', 'serialized_sha256', 'raw_tensor_sha256',
                        'rows', 'denominator', 'operator')},
        'rhs_reconstruction': rhs_audit,
        'ggn_applications_to_x40': 1, 'ggn_audit': ggn_audit,
        'fixed_lambda': damping, 'residual_metrics': result_metrics,
        'interpretation': result_metrics['interpretation'],
        'scientific_limit': ('This diagnoses finite-precision linear residual fidelity '
                             'for a blind finite-sample endpoint GGN; it does not measure '
                             'historical recovery quality.'),
        'elapsed_seconds': time.perf_counter()-started}
    atomic_json_no_overwrite(RESULT_PATH, result)
    print(json.dumps({'result': str(RESULT_PATH),
                      'true_relative_residual': result_metrics['true_relative_residual'],
                      'residual_gap_relative_to_recursive':
                          result_metrics['residual_gap_relative_to_recursive'],
                      'interpretation': result_metrics['interpretation']}), flush=True)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()


if __name__ == '__main__':
    main()
