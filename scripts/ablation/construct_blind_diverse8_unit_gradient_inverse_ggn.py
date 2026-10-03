#!/usr/bin/env python3
"""Blind final-checkpoint diverse-gradient inverse-GGN construction."""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import time
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '113_blind_diverse8_unit_gradient_inverse_ggn_candidate'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '7c7f508575e216cd29eee967f31101b1c614f85cc0cb0465becd4e0d1dc8bfb6'
FREEZER_SOURCE = Path(__file__).with_name('freeze_attempt100_diverse_corpora.py')
FREEZER_SHA256 = 'b4531ccbad9ebc05467a5da8b49d544bee6a900c3fa35c3c23e568e878e214f2'
ORDER = ('fineweb', 'wikitext103_raw', 'tinystories', 'arxiv_document',
         'cc_news', 'pg19', 'codeparrot_clean', 'ultrachat')
if hashlib.sha256(FREEZER_SOURCE.read_bytes()).hexdigest() != FREEZER_SHA256:
    raise ValueError('Frozen blind corpus helper source changed')
loader = importlib.util.spec_from_file_location('attempt113_blind_corpus_helper', FREEZER_SOURCE)
freezer = importlib.util.module_from_spec(loader)
loader.loader.exec_module(freezer)
blind = freezer.old


def safe_input_path(value):
    text = str(value).lower().replace('\\', '/')
    forbidden = ('models/base', 'adapter', 'oracle_adl', 'evaluation.json',
                 'true_delta', 'matched_target', 'base_mean', 'ft_mean',
                 'attempt103', 'attempt104', 'attempt105', 'attempt106',
                 'attempt107', 'attempt108', 'attempt109', 'attempt110',
                 'attempt111', 'attempt112', 'result.json')
    if any(token in text for token in forbidden):
        raise ValueError('Forbidden privileged construction input path')
    path = Path(value)
    if 'oracle_probe' in text and str(path) != (
            '/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt'):
        raise ValueError('Only the frozen generic FineWeb response probe is allowed')
    return path if path.is_absolute() else PROJECT/path


def sha256_file(path):
    return blind.sha256_file(path)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def require_hash(path, expected):
    path = safe_input_path(path)
    if sha256_file(path) != expected:
        raise ValueError('Frozen blind input SHA256 mismatch: '+str(path))
    return path


def load_spec(path=SPEC_PATH):
    if Path(path).resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt113 spec path changed')
    require_hash(path, SPEC_SHA256)
    spec = blind.load_json_object(Path(path), 'Attempt113 spec')
    if (spec.get('attempt_id') != ATTEMPT or
            spec.get('corpus_order') != list(ORDER) or
            [row.get('name') for row in spec.get('corpora', [])] != list(ORDER) or
            spec.get('rhs', {}).get('rows') != [0, 512] or
            spec['rhs'].get('batch_size') != 8 or
            spec['rhs'].get('prediction_contexts_per_corpus') != 65024 or
            spec.get('eligible_tensors', {}).get('expected_matrix_count') != 98 or
            spec.get('ggn', {}).get('rows') != [0, 64] or
            spec['ggn'].get('denominator') != 8128 or
            spec.get('damping', {}).get('damping_fraction') != 1e-4 or
            spec.get('solver', {}).get('iterations') != 80 or
            spec['solver'].get('checkpoint_iteration') != 40 or
            spec.get('probe', {}).get('rows') != [0, 1024] or
            spec.get('information_policy', {}).get('historical_base_access') is not False):
        raise ValueError('Attempt113 frozen scientific inventory mismatch')
    paths = [spec['final_checkpoint']['directory'], spec['probe']['path'],
             *spec['paths'].values(), *[row['tokens_path'] for row in spec['corpora']],
             *[row['manifest_path'] for row in spec['corpora']]]
    for key, value in spec['blind_source_provenance'].items():
        if key.endswith('_path'):
            paths.append(value)
    for value in paths:
        safe_input_path(value)
    if spec['paths']['artifact_path'] == spec['paths']['checkpoint_path']:
        raise ValueError('Attempt113 output paths overlap')
    return spec


def refuse_outputs(spec, *, resume=False, timing_smoke=False):
    if timing_smoke:
        return
    artifact = safe_input_path(spec['paths']['artifact_path'])
    manifest = safe_input_path(spec['paths']['manifest_path'])
    checkpoint = safe_input_path(spec['paths']['checkpoint_path'])
    for path in (artifact, manifest):
        if path.exists() or path.is_symlink():
            raise ValueError('Attempt113 scientific output already exists: '+str(path))
    if resume:
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise ValueError('Attempt113 --resume requires iteration-40 checkpoint')
    elif checkpoint.exists() or checkpoint.is_symlink():
        raise ValueError('Stale Attempt113 checkpoint; use --resume after validation')


def validate_blind_sources(spec, *, timing_smoke=False):
    """Read only final-model and frozen generic-text provenance."""
    source = spec['blind_source_provenance']
    for key, expected in source.items():
        if key.endswith('_path'):
            digest_key = key[:-5]+'_sha256'
            require_hash(expected, source[digest_key])
    s100 = freezer.load_spec()
    s014 = blind.load_json_object(safe_input_path(source['attempt014_spec_path']),
                                  'blind Attempt014 spec')
    m100 = blind.load_json_object(safe_input_path(source['attempt100_manifest_path']),
                                  'blind Attempt100 construction manifest')
    if (s100['canonical_checkpoint_files'] != spec['final_checkpoint']['files'] or
            s100['model'] != spec['model'] or
            s100['eligible_tensors'] != spec['eligible_tensors'] or
            s100['readout'] != spec['readout'] or
            s100['jvp'] != spec['jvp'] or
            len(s014['corpora']) != 3 or
            m100.get('corpus_order') != list(ORDER) or
            m100.get('spec_sha256') != source['attempt100_spec_sha256'] or
            m100.get('constructor_sha256') !=
                source['attempt100_constructor_sha256'] or
            m100.get('freezer_sha256') != source['attempt100_freezer_sha256'] or
            m100.get('corpus_lock_sha256') != source['attempt100_lock_sha256'] or
            m100.get('source_checkpoint') != spec['final_checkpoint']['files'] or
            m100.get('probe_sha256') != spec['probe']['serialized_sha256'] or
            m100.get('historical_base_access') is not False or
            m100.get('oracle_adl_access') is not False or
            m100.get('prior_evaluation_access') is not False):
        raise ValueError('Blind source provenance/inventory mismatch')
    model_dir = safe_input_path(spec['final_checkpoint']['directory'])
    if blind.checkpoint_file_records(model_dir) != spec['final_checkpoint']['files']:
        raise ValueError('Canonical final checkpoint inventory mismatch')
    if timing_smoke:
        corpus_rows = spec['corpora'][:1]
    else:
        corpus_rows = spec['corpora']
    lock = freezer.validate_lock(blind.load_json_object(
        safe_input_path(source['attempt100_lock_path']), 'blind corpus lock'), s100)
    tokens = {}
    manifests = {}
    for index, row in enumerate(corpus_rows):
        name = row['name']
        manifest_path = require_hash(row['manifest_path'], row['manifest_sha256'])
        if index < 3:
            frozen = s014['corpora'][index]
            for field in ('name', 'spec_path', 'spec_sha256', 'manifest_path',
                          'tokens_path'):
                if row[field] != frozen[field]:
                    raise ValueError('Frozen early blind corpus linkage mismatch')
            corpus, _, meta = blind.validate_frozen_corpus(frozen, model_dir, torch)
            manifest = blind.load_json_object(manifest_path, 'blind corpus manifest')
            if (meta['serialized_sha256'] != row['artifact_serialized_sha256'] or
                    meta['raw_tensor_sha256'] != row['artifact_raw_sha256']):
                raise ValueError('Frozen early corpus artifact hash mismatch')
        else:
            frozen = s100['new_corpora'][index-3]
            if (row['name'] != frozen['name'] or row['tokens_path'] != frozen['tokens_path'] or
                    row['manifest_path'] != frozen['manifest_path']):
                raise ValueError('Frozen new blind corpus linkage mismatch')
            corpus, manifest = freezer.validate_corpus(frozen, s100, lock, model_dir, torch)
        if (tuple(corpus.shape) != (4096, 128) or corpus.dtype != torch.int64 or
                not corpus.is_contiguous() or
                sha256_file(safe_input_path(row['tokens_path'])) !=
                    row['artifact_serialized_sha256'] or
                blind.sha256_raw_int64_tensor(corpus, torch) !=
                    row['artifact_raw_sha256'] or
                manifest.get('artifact', {}).get('serialized_sha256') !=
                    row['artifact_serialized_sha256'] or
                manifest.get('artifact', {}).get('raw_tensor_sha256') !=
                    row['artifact_raw_sha256']):
            raise ValueError('Frozen generic corpus tensor/manifest mismatch')
        tokens[name] = corpus
        manifests[name] = {'manifest_sha256': row['manifest_sha256'],
                           'serialized_sha256': row['artifact_serialized_sha256'],
                           'raw_tensor_sha256': row['artifact_raw_sha256']}
    probe = None
    if not timing_smoke:
        full_probe = blind.load_probe(
            safe_input_path(spec['probe']['path']),
            {**spec['probe'], 'sample_count': 10000}, torch)
        probe = full_probe[:1024].contiguous()
    return tokens, probe, manifests


def coordinate_inventory(eligible):
    rows = []
    for name, module in eligible:
        value = module.weight
        if (value.dtype != torch.float32 or value.ndim != 2 or
                not value.is_contiguous()):
            raise ValueError('Malformed selected final-model weight')
        rows.append({'name': name+'.weight', 'shape': list(value.shape)})
    if len(rows) != 98 or [row['name'] for row in rows] != sorted(row['name'] for row in rows):
        raise ValueError('Expected exactly 98 lexicographic early Linear.weight matrices')
    return rows


def _names(eligible):
    return [name+'.weight' for name, _ in eligible]


def matrixwise_dot(left, right):
    if not left or set(left) != set(right):
        raise ValueError('Matrixwise dot coordinate mismatch')
    parts = []
    for name in sorted(left):
        x, y = left[name], right[name]
        if (x.dtype != torch.float32 or y.dtype != torch.float32 or
                x.shape != y.shape or x.device != y.device or
                not bool(torch.isfinite(x).all()) or
                not bool(torch.isfinite(y).all())):
            raise ValueError('Malformed/nonfinite matrixwise dot input')
        parts.append(float(torch.sum(x*y, dtype=torch.float64)))
    return math.fsum(parts)


def matrixwise_grad_dot(eligible, vector):
    if set(_names(eligible)) != set(vector):
        raise ValueError('GGN gradient/vector support mismatch')
    parts = []
    for name, module in eligible:
        gradient = module.weight.grad
        direction = vector[name+'.weight']
        if (gradient is None or gradient.dtype != torch.float32 or
                gradient.shape != direction.shape or gradient.device != direction.device or
                not bool(torch.isfinite(gradient).all())):
            raise ValueError('Missing/nonfinite selected GGN action')
        parts.append(float(torch.sum(gradient*direction, dtype=torch.float64)))
    return math.fsum(parts)


def matrixwise_fp64_norm(eligible):
    parts = []
    for _, module in eligible:
        gradient = module.weight.grad
        if (gradient is None or gradient.dtype != torch.float32 or
                gradient.shape != module.weight.shape or
                not bool(torch.isfinite(gradient).all())):
            raise ValueError('Missing/nonfinite selected generic gradient')
        cpu64 = gradient.detach().to('cpu', dtype=torch.float64)
        parts.append(float(torch.sum(cpu64.square(), dtype=torch.float64)))
        del cpu64
    squared = math.fsum(parts)
    if not math.isfinite(squared) or squared <= 0:
        raise ValueError('Zero/nonfinite corpus gradient norm')
    return math.sqrt(squared)


def ordered_vector_hashes(vector, inventory):
    names = [row['name'] for row in inventory]
    if set(vector) != set(names) or len(vector) != len(names):
        raise ValueError('Parameter vector inventory mismatch')
    hashes = {}
    for row in inventory:
        name = row['name']
        value = vector[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                value.device.type != 'cpu' or not value.is_contiguous() or
                list(value.shape) != row['shape'] or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Malformed CPU FP32 parameter vector')
        hashes[name] = blind.sha256_raw_float32_tensor(value, torch)
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode('utf-8')+b'\0'+hashes[name].encode('ascii')+b'\n')
    return hashes, digest.hexdigest()


def progress(label, current, total, elapsed):
    eta = elapsed/current*(total-current) if current else None
    print(f'{label} | {current}/{total} | elapsed {elapsed:.1f}s | '
          f'ETA {eta:.1f}s' if eta is not None else f'{label} | starting',
          flush=True)


def accumulate_corpus_gradient(model, tokens, eligible, *, rows=512, progress_label=None):
    if (tokens.dtype != torch.int64 or tokens.ndim != 2 or
            tokens.shape[1] != 128 or tokens.shape[0] < rows or
            rows not in (32, 512) or rows % 8):
        raise ValueError('Frozen generic gradient batch inventory mismatch')
    selected = {id(module.weight) for _, module in eligible}
    model.zero_grad(set_to_none=True)
    loss_sums = []
    device = eligible[0][1].weight.device
    started = time.perf_counter()
    for index, start in enumerate(range(0, rows, 8), 1):
        batch = tokens[start:start+8].to(device)
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            logits = model(input_ids=batch, use_cache=False).logits
            loss_sum = blind.causal_token_loss_sum(logits, batch, torch)
            if not bool(torch.isfinite(loss_sum)):
                raise ValueError('Nonfinite generic loss')
            (loss_sum/(rows*127)).backward()
        loss_sums.append(float(loss_sum.detach().to('cpu', dtype=torch.float64)))
        if any(parameter.grad is not None for parameter in model.parameters()
               if id(parameter) not in selected):
            raise ValueError('Frozen parameter received generic gradient')
        if progress_label is not None:
            progress(progress_label, index, rows//8, time.perf_counter()-started)
        del batch, logits, loss_sum
    norm = matrixwise_fp64_norm(eligible)
    return math.fsum(loss_sums)/(rows*127), norm


def add_unit_gradient_to_consensus(consensus, eligible, norm, inventory, *,
                                   corpus_index, corpus_count=8):
    if (not math.isfinite(norm) or norm <= 0 or
            not 0 <= corpus_index < corpus_count or corpus_count not in (1, 8)):
        raise ValueError('Invalid frozen unit-gradient consensus settings')
    scale = 1.0/norm
    names = [row['name'] for row in inventory]
    if names != _names(eligible) or (consensus and set(consensus) != set(names)):
        raise ValueError('Unit-gradient consensus support mismatch')
    for name, module in eligible:
        key = name+'.weight'
        gradient = module.weight.grad
        cpu = gradient.detach().to('cpu', dtype=torch.float32).contiguous()
        unit = (cpu.to(torch.float64)*scale).to(torch.float32).contiguous()
        if not bool(torch.isfinite(unit).all()):
            raise ValueError('Nonfinite realized unit gradient')
        if corpus_index == 0:
            consensus[key] = torch.zeros_like(unit)
        consensus[key].add_(unit, alpha=1.0/corpus_count)
        del cpu, unit
    return scale


def build_blind_rhs(model, corpora, eligible, inventory, *, smoke=False):
    order = ORDER[:1] if smoke else ORDER
    consensus, diagnostics = {}, []
    for index, name in enumerate(order):
        print(f'generic gradient | corpus {index+1}/{len(order)} {name}', flush=True)
        rows = 32 if smoke else 512
        loss, norm = accumulate_corpus_gradient(
            model, corpora[name], eligible, rows=rows,
            progress_label=f'{name} gradient')
        scale = add_unit_gradient_to_consensus(
            consensus, eligible, norm, inventory, corpus_index=index,
            corpus_count=1 if smoke else 8)
        diagnostics.append({'name': name, 'mean_generic_loss': loss,
                            'global_gradient_norm': norm,
                            'unit_normalization_scalar': scale,
                            'rows': [0, rows], 'prediction_contexts': rows*127})
        model.zero_grad(set_to_none=True)
    norm = math.sqrt(matrixwise_dot(consensus, consensus))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('Zero/nonfinite blind RHS')
    hashes, aggregate = ordered_vector_hashes(consensus, inventory)
    return consensus, diagnostics, {'norm': norm, 'matrix_raw_sha256': hashes,
                                    'ordered_matrix_hash_sha256': aggregate,
                                    'hash_aggregation': 'SHA256_of_ordered_name_NUL_digest_newline'}


def strict_logits_jvp(model, batch, direction):
    if not direction:
        raise ValueError('Endpoint logits JVP requires nonempty direction')
    names = sorted(direction)
    primals, tangents = [], []
    for name in names:
        parameter = model.get_parameter(name)
        vector = direction[name]
        if (parameter.dtype != torch.float32 or vector.dtype != torch.float32 or
                parameter.shape != vector.shape or parameter.device != vector.device or
                not bool(torch.isfinite(vector).all())):
            raise ValueError('Invalid endpoint logits JVP coordinate')
        primals.append(parameter.detach())
        tangents.append(vector)

    def logits(weights):
        output = torch.func.functional_call(
            model, dict(zip(names, weights)), (),
            {'input_ids': batch, 'use_cache': False}, strict=False)
        return output.logits[:, :-1, :].float()

    try:
        with torch.no_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
            primal, action = torch.func.jvp(logits, (tuple(primals),),
                                            (tuple(tangents),), strict=True)
    except (NotImplementedError, RuntimeError) as exc:
        raise RuntimeError('Strict endpoint logits JVP failed; no fallback: '+str(exc)) from exc
    if (primal.shape != action.shape or primal.shape[:2] != (batch.shape[0], 127) or
            primal.dtype != torch.float32 or action.dtype != torch.float32 or
            not bool(torch.isfinite(primal).all()) or
            not bool(torch.isfinite(action).all())):
        raise ValueError('Malformed/nonfinite endpoint logits JVP')
    return primal.detach(), action.detach()


def audit_logits_primal(primal, ordinary, spec):
    if (primal.shape != ordinary.shape or primal.dtype != torch.float32 or
            ordinary.dtype != torch.float32 or not bool(torch.isfinite(ordinary).all())):
        raise ValueError('Malformed ordinary logits primal audit')
    difference = primal.double()-ordinary.detach().double()
    max_abs = float(difference.abs().max())
    rms = float(torch.sqrt(difference.square().mean()))
    max_scale = float(ordinary.double().abs().max())
    rms_scale = float(torch.sqrt(ordinary.double().square().mean()))
    rule = spec['ggn']
    if (max_abs > rule['first_batch_primal_atol']+rule['first_batch_primal_rtol']*max_scale or
            rms > rule['first_batch_rms_atol']+rule['first_batch_rms_rtol']*rms_scale):
        raise ValueError('GGN JVP primal differs from ordinary final logits')
    return {'max_abs_difference': max_abs, 'rms_difference': rms,
            'max_scale': max_scale, 'rms_scale': rms_scale}


def _fisher_error(reason, diagnostics):
    raise ValueError('Categorical Fisher '+reason+'; diagnostics='+
                     json.dumps(diagnostics, sort_keys=True, allow_nan=True, default=str))


def stable_categorical_fisher_vector(logits, tangent, spec):
    diagnostics = {'logits_shape': list(logits.shape),
                   'tangent_shape': list(tangent.shape),
                   'logits_dtype': str(logits.dtype),
                   'tangent_dtype': str(tangent.dtype)}
    if (logits.ndim != 3 or logits.shape != tangent.shape or
            logits.dtype != torch.float32 or tangent.dtype != torch.float32 or
            not bool(torch.isfinite(logits).all()) or
            not bool(torch.isfinite(tangent).all())):
        _fisher_error('nonfinite/malformed logits or tangent', diagnostics)
    p64 = torch.softmax(logits.to(torch.float64), dim=-1)
    s64 = tangent.to(torch.float64)
    mean64 = torch.sum(p64*s64, dim=-1, keepdim=True, dtype=torch.float64)
    u64 = p64*(s64-mean64)
    u64 /= spec['ggn']['denominator']
    u = u64.to(torch.float32)
    policy = spec['fisher_numerics']
    if not all(bool(torch.isfinite(value).all()) for value in (p64, s64, mean64, u64, u)):
        _fisher_error('nonfinite probability/action', diagnostics)
    p_sum_error = (torch.sum(p64, dim=-1, dtype=torch.float64)-1).abs()
    residual64 = torch.sum(u64, dim=-1, dtype=torch.float64).abs()
    scale64 = torch.sum(u64.abs(), dim=-1, dtype=torch.float64)
    residual32 = torch.sum(u, dim=-1, dtype=torch.float64).abs()
    scale32 = torch.sum(u.abs(), dim=-1, dtype=torch.float64)
    relative64 = torch.where(scale64 > 0, residual64/scale64.clamp_min(
        torch.finfo(torch.float64).tiny), torch.zeros_like(scale64))
    relative32 = torch.where(scale32 > 0, residual32/scale32.clamp_min(
        torch.finfo(torch.float64).tiny), torch.zeros_like(scale32))
    max_abs_tangent = s64.abs().amax(dim=-1)
    allowed64 = (policy['fp64_fisher_conservation_fail_relative_tolerance']*scale64+
                 policy['fp64_fisher_conservation_fail_gauge_roundoff_multiplier']*
                 max_abs_tangent/spec['ggn']['denominator']+
                 policy['fp64_fisher_conservation_fail_absolute_tolerance'])
    allowed32 = (policy['fp32_cast_conservation_fail_relative_tolerance']*scale32+
                 policy['fp32_cast_conservation_fail_absolute_tolerance'])
    diagnostics.update({
        'max_abs_tangent': float(max_abs_tangent.max()),
        'max_abs_unweighted_vocab_mean_tangent': float(s64.mean(dim=-1).abs().max()),
        'max_abs_probability_weighted_mean_tangent': float(mean64.abs().max()),
        'max_softmax_sum_error': float(p_sum_error.max()),
        'max_abs_sum_u64': float(residual64.max()),
        'max_relative_abs_sum_u64': float(relative64.max()),
        'fp32_max_abs_sum': float(residual32.max()),
        'fp32_max_relative_abs_sum': float(relative32.max()),
        'max_fp64_allowed_abs_sum': float(allowed64.max()),
        'max_fp32_allowed_abs_sum': float(allowed32.max()),
        'prediction_context_count': residual64.numel()})
    if bool((p_sum_error > policy['fp64_softmax_normalization_fail_abs_tolerance']).any()):
        _fisher_error('FP64 softmax normalization failure', diagnostics)
    if bool((residual64 > allowed64).any()):
        _fisher_error('FP64 sum-to-zero failure', diagnostics)
    if bool((residual32 > allowed32).any()):
        _fisher_error('FP32 cast sum-to-zero failure', diagnostics)
    return u.detach(), diagnostics


def ggn_action_stage(model, tokens, eligible, direction, spec, *, progress_label=None):
    if (tokens.shape != (64, 128) or tokens.dtype != torch.int64 or
            not tokens.is_contiguous() or len(eligible) != 98 or
            set(direction) != set(_names(eligible))):
        raise ValueError('Frozen final-only GGN support/data mismatch')
    selected = {id(module.weight) for _, module in eligible}
    model.zero_grad(set_to_none=True)
    first_audit = None
    records = []
    timing = {'logits_jvp_seconds': 0.0, 'fisher_seconds': 0.0,
              'vjp_backward_seconds': 0.0}
    device = eligible[0][1].weight.device
    started = time.perf_counter()
    for index in range(64):
        batch = tokens[index:index+1].to(device)
        stage = time.perf_counter()
        primal, tangent = strict_logits_jvp(model, batch, direction)
        timing['logits_jvp_seconds'] += time.perf_counter()-stage
        primal_audit = None
        if index == 0:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                ordinary = model(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            primal_audit = audit_logits_primal(primal, ordinary, spec)
            del ordinary
        stage = time.perf_counter()
        fisher, conservation = stable_categorical_fisher_vector(primal, tangent, spec)
        records.append({'microbatch_index': index, **conservation})
        timing['fisher_seconds'] += time.perf_counter()-stage
        stage = time.perf_counter()
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            logits = model(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            if logits.shape != fisher.shape or not bool(torch.isfinite(logits).all()):
                raise ValueError('GGN VJP logits/Fisher shape or finiteness mismatch')
            (logits*fisher).sum().backward()
        timing['vjp_backward_seconds'] += time.perf_counter()-stage
        if any(parameter.grad is not None for parameter in model.parameters()
               if id(parameter) not in selected):
            raise ValueError('Frozen parameter received GGN VJP gradient')
        if index == 0:
            if any(module.weight.grad is None or
                   not bool(torch.isfinite(module.weight.grad).all())
                   for _, module in eligible):
                raise ValueError('Missing/nonfinite selected GGN VJP gradient')
            first_audit = {'primal': primal_audit, 'fisher': conservation,
                           'tangent_shape': list(tangent.shape),
                           'tangent_dtype': str(tangent.dtype),
                           'selected_vjp_matrix_count': len(eligible)}
        if progress_label is not None:
            progress(progress_label, index+1, 64, time.perf_counter()-started)
        del batch, primal, tangent, fisher, logits
    fields = ('max_abs_tangent', 'max_abs_unweighted_vocab_mean_tangent',
              'max_abs_probability_weighted_mean_tangent', 'max_softmax_sum_error',
              'max_abs_sum_u64', 'max_relative_abs_sum_u64', 'fp32_max_abs_sum',
              'fp32_max_relative_abs_sum')
    summary = {field: max(row[field] for row in records) for field in fields}
    summary['prediction_context_count'] = sum(row['prediction_context_count']
                                              for row in records)
    return {'first_batch': first_audit, 'timing': timing,
            'fisher_conservation': summary}


def rayleigh_and_lambda(eligible, direction, damping_fraction):
    if damping_fraction != 1e-4:
        raise ValueError('Blind damping fraction changed')
    numerator = matrixwise_grad_dot(eligible, direction)
    denominator = matrixwise_dot(direction, direction)
    rho = numerator/denominator
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError('Nonpositive/nonfinite blind RHS Rayleigh quotient')
    damping = damping_fraction*rho
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError('Nonpositive/nonfinite blind damping')
    return {'bGb': numerator, 'bb': denominator, 'rho_b': rho,
            'damping_fraction': damping_fraction, 'lambda': damping}


def initial_cg_state(rhs_cpu, eligible):
    names = _names(eligible)
    if len(names) != 98 or set(rhs_cpu) != set(names):
        raise ValueError('Blind CG RHS coordinate mismatch')
    r = {}
    for name, module in eligible:
        key = name+'.weight'
        value = rhs_cpu[key]
        if (value.dtype != torch.float32 or value.device.type != 'cpu' or
                value.shape != module.weight.shape or not value.is_contiguous() or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Malformed blind CG RHS matrix')
        r[key] = value.to(module.weight.device, dtype=torch.float32,
                          copy=True).contiguous()
    rhs_cpu.clear()
    p = {name: value.clone() for name, value in r.items()}
    x = {name: torch.zeros_like(value) for name, value in r.items()}
    rr = matrixwise_dot(r, r)
    if not math.isfinite(rr) or rr <= 0:
        raise ValueError('Zero/nonfinite blind CG RHS')
    return {'iteration': 0, 'x': x, 'r': r, 'p': p,
            'rr': rr, 'b_norm': math.sqrt(rr), 'iterations': []}


def state_hashes(state, inventory):
    result = {}
    for key in ('x', 'r', 'p'):
        if set(state[key]) != {row['name'] for row in inventory}:
            raise ValueError('CG state coordinate inventory mismatch')
        hashes, digest = {}, hashlib.sha256()
        for row in inventory:
            name = row['name']
            cpu = state[key][name].detach().to(
                'cpu', dtype=torch.float32, copy=True).contiguous()
            if list(cpu.shape) != row['shape'] or not bool(torch.isfinite(cpu).all()):
                raise ValueError('Malformed CG state vector')
            hashes[name] = blind.sha256_raw_float32_tensor(cpu, torch)
            digest.update(name.encode('utf-8')+b'\0'+hashes[name].encode('ascii')+b'\n')
            del cpu
        result[key] = {'matrix_raw_sha256': hashes,
                       'ordered_matrix_hash_sha256': digest.hexdigest()}
    return result


def checkpoint_provenance(spec, inventory, rhs_record, gradient_records):
    return {
        'spec_sha256': SPEC_SHA256,
        'source_sha256': sha256_file(Path(__file__).resolve()),
        'final_checkpoint_inventory_sha256': json_hash(spec['final_checkpoint']['files']),
        'eight_corpus_inventory_sha256': json_hash(spec['corpora']),
        'probe_serialized_sha256': spec['probe']['serialized_sha256'],
        'ggn_corpus_serialized_sha256': spec['ggn']['serialized_sha256'],
        'coordinate_inventory_sha256': json_hash(inventory),
        'rhs_matrix_hash_sha256': rhs_record['ordered_matrix_hash_sha256'],
        'gradient_diagnostics_sha256': json_hash(gradient_records),
        'rhs_arithmetic': spec['rhs'],
        'ggn_arithmetic': spec['fisher_numerics']}


def checkpoint_payload(state, inventory, spec, rhs_record, gradient_records, rayleigh):
    if state['iteration'] != 40 or len(state['iterations']) != 40:
        raise ValueError('Attempt113 checkpoint must capture iteration 40')
    before = state_hashes(state, inventory)
    vectors = {}
    for key in ('x', 'r', 'p'):
        vectors[key] = {row['name']: state[key][row['name']].detach().to(
            'cpu', dtype=torch.float32, copy=True).contiguous()
            for row in inventory}
        _, aggregate = ordered_vector_hashes(vectors[key], inventory)
        if aggregate != before[key]['ordered_matrix_hash_sha256']:
            raise ValueError('Checkpoint tensor copy changed CG '+key)
    if state_hashes(state, inventory) != before:
        raise ValueError('Checkpoint preparation changed in-memory CG state')
    return {
        'format_version': 1, 'iteration': 40,
        'x': vectors['x'], 'r': vectors['r'], 'p': vectors['p'],
        'rr': float(state['rr']), 'b_norm': float(state['b_norm']),
        'rho_b': rayleigh['rho_b'], 'fixed_lambda': rayleigh['lambda'],
        'coordinate_inventory': copy.deepcopy(inventory),
        'iteration_records': copy.deepcopy(state['iterations']),
        'rhs_record': copy.deepcopy(rhs_record),
        'gradient_records': copy.deepcopy(gradient_records),
        'raw_hashes': before,
        'provenance': checkpoint_provenance(
            spec, inventory, rhs_record, gradient_records)}


def atomic_torch_publish(path, payload):
    """Fsync a temporary file and atomically link it without overwrite."""
    if path.exists() or path.is_symlink():
        raise ValueError('Output already exists: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        with temporary.open('xb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Attempt113 --resume requires iteration-40 checkpoint')
    try:
        value = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('Attempt113 checkpoint could not be safely loaded') from exc
    if not isinstance(value, dict):
        raise ValueError('Attempt113 checkpoint root is malformed')
    return value


def validate_checkpoint(value, spec, inventory, rhs_record, gradient_records, rayleigh):
    required = {'format_version', 'iteration', 'x', 'r', 'p', 'rr', 'b_norm',
                'rho_b', 'fixed_lambda', 'coordinate_inventory',
                'iteration_records', 'rhs_record', 'gradient_records',
                'raw_hashes', 'provenance'}
    if (set(value) != required or value['format_version'] != 1 or
            value['iteration'] != 40 or
            value['coordinate_inventory'] != inventory or
            value['rhs_record'] != rhs_record or
            value['gradient_records'] != gradient_records or
            value['provenance'] != checkpoint_provenance(
                spec, inventory, rhs_record, gradient_records) or
            not all(isinstance(value[key], (int, float)) and math.isfinite(value[key])
                    for key in ('rr', 'b_norm', 'rho_b', 'fixed_lambda')) or
            value['rr'] <= 0 or value['b_norm'] <= 0 or value['rho_b'] <= 0 or
            value['fixed_lambda'] <= 0 or
            not math.isclose(value['b_norm'], rhs_record['norm'], rel_tol=1e-6) or
            not math.isclose(value['rho_b'], rayleigh['rho_b'], rel_tol=1e-6) or
            not math.isclose(value['fixed_lambda'], rayleigh['lambda'], rel_tol=1e-6)):
        raise ValueError('Attempt113 checkpoint provenance/RHS/damping mismatch')
    rows = value['iteration_records']
    fields = ('relative_residual', 'r_norm', 'pAp', 'alpha', 'beta',
              'operator_seconds', 'cumulative_elapsed_seconds', 'operator_eta_seconds')
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError('Attempt113 checkpoint solver trajectory length mismatch')
    for index, row in enumerate(rows, 1):
        if (not isinstance(row, dict) or row.get('iteration') != index or
                any(not isinstance(row.get(field), (int, float)) or
                    not math.isfinite(row[field]) for field in fields)):
            raise ValueError('Attempt113 checkpoint solver trajectory malformed')
    for key in ('x', 'r', 'p'):
        vector = value[key]
        if (not isinstance(vector, dict) or any(
                tensor.device.type != 'cpu' or not tensor.is_contiguous()
                for tensor in vector.values() if isinstance(tensor, torch.Tensor))):
            raise ValueError('Attempt113 checkpoint tensor storage mismatch')
        hashes, aggregate = ordered_vector_hashes(vector, inventory)
        if value['raw_hashes'].get(key) != {
                'matrix_raw_sha256': hashes,
                'ordered_matrix_hash_sha256': aggregate}:
            raise ValueError('Attempt113 checkpoint x/r/p raw hash mismatch')
    if not math.isclose(matrixwise_dot(value['r'], value['r']),
                        value['rr'], rel_tol=1e-5, abs_tol=1e-7):
        raise ValueError('Attempt113 checkpoint residual scalar/vector mismatch')


def restore_cg_state(value, eligible, inventory):
    if _names(eligible) != [row['name'] for row in inventory]:
        raise ValueError('Attempt113 resume coordinate order mismatch')
    state = {'iteration': 40, 'rr': value['rr'], 'b_norm': value['b_norm'],
             'iterations': copy.deepcopy(value['iteration_records'])}
    for key in ('x', 'r', 'p'):
        state[key] = {name+'.weight': value[key][name+'.weight'].to(
            module.weight.device, dtype=torch.float32, copy=True).contiguous()
            for name, module in eligible}
    if state_hashes(state, inventory) != value['raw_hashes']:
        raise ValueError('Restored CG state differs from checkpoint')
    return state


def run_fixed_80_cg(state, eligible, apply_operator, damping, spec, *,
                    on_checkpoint=None, report=None):
    solver = spec['solver']
    if (solver['iterations'] != 80 or solver['checkpoint_iteration'] != 40 or
            solver['residual_early_stop'] is not False or
            solver['preconditioner'] is not None or
            solver['candidate'] != 'x_80' or
            not math.isfinite(damping) or damping <= 0 or
            state['iteration'] not in (0, 40) or
            len(state['iterations']) != state['iteration'] or
            not math.isfinite(state['rr']) or state['rr'] <= 0 or
            not math.isfinite(state['b_norm']) or state['b_norm'] <= 0):
        raise ValueError('Attempt113 fixed 80-step CG state/settings mismatch')
    if state['iteration'] == 0 and on_checkpoint is None:
        raise ValueError('Fresh Attempt113 CG requires checkpoint callback')
    names = _names(eligible)
    if any(set(state[key]) != set(names) for key in ('x', 'r', 'p')):
        raise ValueError('Attempt113 CG vector support mismatch')
    started = time.perf_counter()
    elapsed_offset = (state['iterations'][-1]['cumulative_elapsed_seconds']
                      if state['iterations'] else 0.0)
    for iteration in range(state['iteration']+1, 81):
        operator_started = time.perf_counter()
        apply_operator(state['p'], iteration)
        operator_seconds = time.perf_counter()-operator_started
        p_gp = matrixwise_grad_dot(eligible, state['p'])
        p_ap = p_gp+damping*matrixwise_dot(state['p'], state['p'])
        if not math.isfinite(p_ap) or p_ap <= 0:
            raise ValueError('Nonpositive/nonfinite CG pAp')
        alpha = state['rr']/p_ap
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('Nonpositive/nonfinite CG alpha')
        for name, module in eligible:
            key = name+'.weight'
            state['x'][key].add_(state['p'][key], alpha=alpha)
            state['r'][key].add_(module.weight.grad, alpha=-alpha)
            state['r'][key].add_(state['p'][key], alpha=-alpha*damping)
        rr_new = matrixwise_dot(state['r'], state['r'])
        if not math.isfinite(rr_new) or rr_new < 0:
            raise ValueError('Nonfinite/negative CG residual')
        beta = rr_new/state['rr']
        if not math.isfinite(beta) or beta < 0:
            raise ValueError('Nonfinite/negative CG beta')
        for key in names:
            state['p'][key].mul_(beta).add_(state['r'][key])
            if any(not bool(torch.isfinite(state[field][key]).all())
                   for field in ('x', 'r', 'p')):
                raise ValueError('Nonfinite CG vector')
        state['rr'] = rr_new
        state['iteration'] = iteration
        elapsed = elapsed_offset+time.perf_counter()-started
        operator_times = [row['operator_seconds'] for row in state['iterations']]
        operator_times.append(operator_seconds)
        row = {'iteration': iteration,
               'relative_residual': math.sqrt(rr_new)/state['b_norm'],
               'r_norm': math.sqrt(rr_new), 'pAp': p_ap,
               'alpha': alpha, 'beta': beta,
               'operator_seconds': operator_seconds,
               'cumulative_elapsed_seconds': elapsed,
               'operator_eta_seconds': math.fsum(operator_times)/len(operator_times)*
                   (80-iteration)}
        state['iterations'].append(row)
        if report is not None:
            report(row)
        if iteration == 40:
            on_checkpoint(state)
    return state, {
        'method': 'fixed_80_step_blind_rayleigh_damped_cg',
        'iterations': state['iterations'], 'operator_applications': 80,
        'rhs_norm': state['b_norm'], 'lambda': damping,
        'candidate': 'x_80', 'no_early_stop': True, 'preconditioner': None,
        'relative_linear_residual': state['iterations'][-1]['relative_residual'],
        'residual_source': 'CG_recurrence_no_extra_operator_application',
        'elapsed_seconds': state['iterations'][-1]['cumulative_elapsed_seconds']}


def vector_to_device(vector, eligible):
    if set(vector) != set(_names(eligible)):
        raise ValueError('Functional tangent support mismatch')
    return {name+'.weight': vector[name+'.weight'].to(
        module.weight.device, dtype=torch.float32, copy=True).contiguous()
        for name, module in eligible}


def probe_response(model, probe, direction, spec, *, label):
    if (probe.shape != (1024, 128) or probe.dtype != torch.int64 or
            not probe.is_contiguous() or len(direction) != 98):
        raise ValueError('Frozen generic probe/JVP support mismatch')
    device = next(model.parameters()).device
    accumulator = blind.ResponseAccumulator(spec, torch)
    discrepancy = None
    started = time.perf_counter()
    for index, start in enumerate(range(0, 1024, 32), 1):
        batch = probe[start:start+32].to(device)
        ordinary = (blind.ordinary_hook_readout(model, batch, spec, torch)
                    if index == 1 else None)
        primal, tangent = blind.run_readout_jvp(model, batch, direction, spec, torch)
        if ordinary is not None:
            discrepancy = blind.compare_primal(primal, ordinary, spec, torch)
        accumulator.add(primal, tangent)
        progress(label, index, 32, time.perf_counter()-started)
        del batch, ordinary, primal, tangent
    mean, response = accumulator.means(1024)
    stored = response.float().contiguous()
    if (stored.shape != (128, 2048) or stored.device.type != 'cpu' or
            not bool(torch.isfinite(stored).all())):
        raise ValueError('Malformed generic probe response')
    return stored, {'first_batch_hook_and_JVP_primal_verified': True,
                    'first_batch_max_abs_difference': discrepancy,
                    'unperturbed_mean_raw_sha256':
                        blind.sha256_raw_float32_tensor(mean.float().contiguous(), torch),
                    'elapsed_seconds': time.perf_counter()-started}


def atomic_json_publish(path, value):
    if path.exists() or path.is_symlink():
        raise ValueError('Construction manifest already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_candidate(artifact_path, manifest_path, checkpoint_path,
                      candidate, response_raw, response_inverse, manifest,
                      inventory):
    for path in (artifact_path, manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError('Attempt113 construction output already exists')
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ValueError('Iteration-40 checkpoint missing before publication')
    if (manifest.get('solver', {}).get('operator_applications') != 80 or
            len(manifest.get('solver', {}).get('iterations', [])) != 80 or
            manifest.get('candidate', {}).get('name') != 'x_80'):
        raise ValueError('Refusing blind artifact before fixed x_80 completes')
    if list(candidate) != [row['name'] for row in inventory]:
        raise ValueError('Candidate matrices are not in frozen coordinate order')
    x_hashes, x_aggregate = ordered_vector_hashes(candidate, inventory)
    for value in (response_raw, response_inverse):
        if (value.dtype != torch.float32 or value.device.type != 'cpu' or
                not value.is_contiguous() or value.shape != (128, 2048) or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Malformed generic functional response')
    raw_hash = blind.sha256_raw_float32_tensor(response_raw, torch)
    inverse_hash = blind.sha256_raw_float32_tensor(response_inverse, torch)
    if (manifest['candidate']['x_80']['matrix_raw_sha256'] != x_hashes or
            manifest['candidate']['x_80']['ordered_matrix_hash_sha256'] != x_aggregate or
            manifest['functional_responses']['response_raw_rhs']['raw_sha256'] != raw_hash or
            manifest['functional_responses']['response_inverse']['raw_sha256'] != inverse_hash):
        raise ValueError('Artifact/manifest raw hashes disagree')
    artifact = {'x_80': candidate, 'response_raw_rhs': response_raw,
                'response_inverse': response_inverse}
    if list(artifact) != ['x_80', 'response_raw_rhs', 'response_inverse']:
        raise ValueError('Candidate artifact inventory changed')
    atomic_torch_publish(artifact_path, artifact)
    manifest['artifact'] = {
        'path': str(artifact_path), 'serialized_sha256': sha256_file(artifact_path),
        'tensor_inventory': ['x_80', 'response_raw_rhs', 'response_inverse'],
        'x_80_matrix_count': len(candidate),
        'x_80_dtype': 'contiguous_CPU_float32',
        'response_shape': [128, 2048],
        'response_dtype': 'contiguous_CPU_float32'}
    try:
        atomic_json_publish(manifest_path, manifest)
    except Exception:
        # Keep the iteration-40 checkpoint usable after an ordinary publication
        # failure. A process crash between links still requires manual review.
        if artifact_path.is_file() and not artifact_path.is_symlink():
            artifact_path.unlink()
        raise
    checkpoint_path.unlink()


def _input_hashes_unchanged(spec):
    for row in spec['corpora']:
        require_hash(row['tokens_path'], row['artifact_serialized_sha256'])
        require_hash(row['manifest_path'], row['manifest_sha256'])
    require_hash(spec['probe']['path'], spec['probe']['serialized_sha256'])


def construct(*, resume=False, timing_smoke=False, model_loader=None):
    if resume and timing_smoke:
        raise ValueError('--resume and --timing-smoke cannot be combined')
    spec = load_spec()
    refuse_outputs(spec, resume=resume, timing_smoke=timing_smoke)
    corpus_tokens, probe, corpus_manifests = validate_blind_sources(
        spec, timing_smoke=timing_smoke)
    model_dir = safe_input_path(spec['final_checkpoint']['directory'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_loader is None:
        model = blind.load_local_model(model_dir, device, torch)
    else:
        model = model_loader(model_dir, device, torch)
    blind.validate_loaded_model(model, torch)
    eligible = blind.discover_eligible_linear_weights(model, torch)
    blind.freeze_other_parameters(model, eligible)
    inventory = coordinate_inventory(eligible)
    before_model = blind.model_state_hashes(model, torch)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameter gradient buffer existed before construction')
    if timing_smoke:
        started = time.perf_counter()
        rhs_cpu, diagnostics, rhs_record = build_blind_rhs(
            model, corpus_tokens, eligible, inventory, smoke=True)
        gradient_seconds = time.perf_counter()-started
        direction = vector_to_device(rhs_cpu, eligible)
        ggn_started = time.perf_counter()
        ggn_audit = ggn_action_stage(model, corpus_tokens['fineweb'][:64].contiguous(),
                                     eligible, direction, spec,
                                     progress_label='timing smoke GGN')
        ggn_seconds = time.perf_counter()-ggn_started
        model.zero_grad(set_to_none=True)
        blind.verify_model_unchanged(model, before_model, torch)
        projected_gradients = gradient_seconds/4*512
        projected_G = ggn_seconds*81
        return {'timing_smoke': True, 'final_checkpoint_only': True,
                'writes': False, 'fineweb_rows': [0, 32],
                'gradient_batches': 4, 'gradient_batch_seconds': gradient_seconds/4,
                'projected_512_sequence_per_corpus_seconds': gradient_seconds/4*64,
                'projected_eight_corpus_gradient_seconds': projected_gradients,
                'one_G_seconds': ggn_seconds,
                'projected_81_G_application_seconds': projected_G,
                'two_probe_JVP_seconds': None,
                'projected_total_wall_seconds_lower_bound':
                    projected_gradients+projected_G,
                'probe_JVP_projection': 'unknown_without_measured_blind_probe_JVP',
                'gradient_diagnostic': diagnostics[0],
                'ggn_first_batch_primal_audit': ggn_audit['first_batch']['primal']}

    rhs_cpu, gradient_records, rhs_record = build_blind_rhs(
        model, corpus_tokens, eligible, inventory)
    rhs_device = vector_to_device(rhs_cpu, eligible)
    response_raw, raw_audit = probe_response(
        model, probe, rhs_device, spec, label='raw blind RHS block13 JVP')
    print('Rayleigh | applying one frozen endpoint GGN to b_blind', flush=True)
    rayleigh_started = time.perf_counter()
    rayleigh_ggn_audit = ggn_action_stage(
        model, corpus_tokens['fineweb'][:64].contiguous(), eligible,
        rhs_device, spec, progress_label='Rayleigh GGN')
    rayleigh_seconds = time.perf_counter()-rayleigh_started
    rayleigh = rayleigh_and_lambda(
        eligible, rhs_device, spec['damping']['damping_fraction'])
    model.zero_grad(set_to_none=True)
    rhs_device.clear()
    del rhs_device
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Rayleigh | rho_b={rayleigh['rho_b']:.9g} "
          f"lambda={rayleigh['lambda']:.9g}", flush=True)

    checkpoint_path = safe_input_path(spec['paths']['checkpoint_path'])
    checkpoint_meta = {}
    if resume:
        payload = load_checkpoint(checkpoint_path)
        validate_checkpoint(payload, spec, inventory, rhs_record,
                            gradient_records, rayleigh)
        state = restore_cg_state(payload, eligible, inventory)
        checkpoint_meta = {'resumed_from_iteration': 40,
                           'serialized_size_bytes': checkpoint_path.stat().st_size,
                           'x_r_p_hashes_sha256': json_hash(payload['raw_hashes'])}
        rhs_cpu.clear()
        del payload
        gc.collect()
        print('checkpoint | validated; resuming at CG 41/80', flush=True)
    else:
        state = initial_cg_state(rhs_cpu, eligible)
        if not math.isclose(state['b_norm'], rhs_record['norm'], rel_tol=1e-6):
            raise ValueError('Blind RHS/CG norm mismatch')

    def apply_operator(vector, iteration):
        print(f'CG {iteration}/80 | endpoint GGN application', flush=True)
        return ggn_action_stage(
            model, corpus_tokens['fineweb'][:64].contiguous(), eligible,
            vector, spec, progress_label=f'CG {iteration}/80 GGN')

    def report(row):
        print(f"CG {row['iteration']}/80 | residual "
              f"{row['relative_residual']:.6g} | operator "
              f"{row['operator_seconds']:.1f}s | elapsed "
              f"{row['cumulative_elapsed_seconds']:.1f}s | ETA "
              f"{row['operator_eta_seconds']:.1f}s", flush=True)

    def on_checkpoint(current):
        nonlocal checkpoint_meta
        started = time.perf_counter()
        before = state_hashes(current, inventory)
        payload = checkpoint_payload(current, inventory, spec,
                                     rhs_record, gradient_records, rayleigh)
        if state_hashes(current, inventory) != before:
            raise ValueError('Checkpoint preparation mutated CG state')
        atomic_torch_publish(checkpoint_path, payload)
        if state_hashes(current, inventory) != before:
            raise ValueError('Checkpoint publication mutated CG state')
        checkpoint_meta = {'resumed_from_iteration': None,
                           'serialized_size_bytes': checkpoint_path.stat().st_size,
                           'x_r_p_hashes_sha256': json_hash(payload['raw_hashes']),
                           'publication_seconds': time.perf_counter()-started}
        del payload
        gc.collect()
        print(f'checkpoint | iteration 40 atomically published at {checkpoint_path}',
              flush=True)

    state, solver = run_fixed_80_cg(
        state, eligible, apply_operator, rayleigh['lambda'], spec,
        on_checkpoint=None if resume else on_checkpoint, report=report)
    if state['iteration'] != 80:
        raise ValueError('Blind candidate x_80 incomplete')
    model.zero_grad(set_to_none=True)
    response_inverse, inverse_audit = probe_response(
        model, probe, state['x'], spec, label='inverse blind x_80 block13 JVP')
    model.zero_grad(set_to_none=True)
    blind.verify_model_unchanged(model, before_model, torch)
    _input_hashes_unchanged(spec)
    candidate = {row['name']: state['x'][row['name']].detach().to(
        'cpu', dtype=torch.float32, copy=True).contiguous() for row in inventory}
    x_hashes, x_aggregate = ordered_vector_hashes(candidate, inventory)
    manifest = {
        'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'spec_sha256': SPEC_SHA256,
        'constructor_source_sha256': sha256_file(Path(__file__).resolve()),
        'final_checkpoint_files': spec['final_checkpoint']['files'],
        'blind_source_provenance': spec['blind_source_provenance'],
        'corpus_order': list(ORDER), 'corpora': [
            {**row, **corpus_manifests[row['name']]} for row in spec['corpora']],
        'generic_loss': spec['rhs']['loss'],
        'coordinate_inventory': inventory,
        'per_corpus_gradients': gradient_records,
        'b_blind': rhs_record,
        'rayleigh': {**rayleigh, 'operator_seconds': rayleigh_seconds,
                     'ggn_audit': rayleigh_ggn_audit},
        'ggn_operator': spec['ggn']['operator'],
        'ggn_numerics': spec['fisher_numerics'],
        'solver': solver,
        'candidate': {'name': 'x_80', 'x_80': {
            'matrix_raw_sha256': x_hashes,
            'ordered_matrix_hash_sha256': x_aggregate,
            'norm': math.sqrt(matrixwise_dot(state['x'], state['x'])),
            'matrix_count': 98},
            'final_relative_linear_residual': solver['relative_linear_residual']},
        'functional_responses': {
            'response_raw_rhs': {
                'shape': [128, 2048], 'dtype': 'torch.float32',
                'raw_sha256': blind.sha256_raw_float32_tensor(response_raw, torch),
                'jvp_audit': raw_audit},
            'response_inverse': {
                'shape': [128, 2048], 'dtype': 'torch.float32',
                'raw_sha256': blind.sha256_raw_float32_tensor(response_inverse, torch),
                'jvp_audit': inverse_audit}},
        'probe': spec['probe'],
        'checkpoint': {**checkpoint_meta, 'path': str(checkpoint_path),
                       'iteration': 40, 'deleted_after_success': True,
                       'large_tensor_persistent_after_success': False},
        'information_policy': spec['information_policy'],
        **spec['information_policy']}
    artifact_path = safe_input_path(spec['paths']['artifact_path'])
    manifest_path = safe_input_path(spec['paths']['manifest_path'])
    publish_candidate(artifact_path, manifest_path, checkpoint_path,
                      candidate, response_raw, response_inverse, manifest, inventory)
    print(f'Attempt113 blind candidate and manifest published: {manifest_path}',
          flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--timing-smoke', action='store_true')
    args = parser.parse_args()
    result = construct(resume=args.resume, timing_smoke=args.timing_smoke)
    if args.timing_smoke:
        print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
