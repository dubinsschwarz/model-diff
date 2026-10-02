#!/usr/bin/env python3
"""Attempt104: privileged base-prefix/final-downstream soft-teacher diagnostic."""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '104_privileged_hybrid_prefix_soft_teacher_diagnostic'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '2d094f6ea5a27489c08b84c59802b47e691da12e49046eff34a000b53b0886b4'
PARENT_SOURCE = Path(__file__).with_name('diagnose_privileged_base_soft_target_rollback.py')
PARENT_SOURCE_SHA256 = '73b3ca864916c547ccf20d5b07889ae6b45c01276da8606ee65dfa0c7de4a0d6'
if hashlib.sha256(PARENT_SOURCE.read_bytes()).hexdigest() != PARENT_SOURCE_SHA256:
    raise ValueError('Frozen Attempt103 source changed')
loader = importlib.util.spec_from_file_location('attempt104_frozen_attempt103_helpers', PARENT_SOURCE)
parent = importlib.util.module_from_spec(loader)
loader.loader.exec_module(parent)
a = parent.a


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt104 spec path changed')
    parent.require_hash(path, SPEC_SHA256)
    spec = a.load_json_object(path, 'Attempt104 spec')
    frozen = parent.load_spec()
    for key in ('source_helpers', 'base', 'final_checkpoint_files', 'model',
                'eligible_tensors', 'corpus', 'probe', 'probe_rows', 'readout',
                'jvp', 'tangent', 'metric'):
        if spec.get(key) != frozen[key]:
            raise ValueError('Attempt104 differs from frozen Attempt103 '+key)
    if (spec.get('attempt_id') != ATTEMPT or
            {key: value for key, value in spec['paths'].items() if key != 'result_path'} !=
                {key: value for key, value in frozen['paths'].items() if key != 'result_path'} or
            spec['corpus']['construction_rows'] != [0, 512] or
            spec['corpus']['batch_size'] != 4 or
            spec['probe_rows'] != [0, 1024] or
            spec['metric']['primary_positions'] != [1, 2, 3, 4] or
            spec['hybrid_teacher']['gradient_sign'] != 'positive' or
            spec['output_residuals']['chunk_prediction_positions'] != 16):
        raise ValueError('Attempt104 frozen scientific inventory mismatch')
    for value in spec['paths'].values():
        parent.resolve(value)
    return spec


def validate_attempt103_linkage(spec):
    link = spec['attempt103']
    for key in ('spec', 'source', 'result'):
        parent.require_hash(parent.resolve(link[key+'_path']), link[key+'_sha256'])
    if (link['spec_sha256'] != parent.SPEC_SHA256 or
            link['source_sha256'] != PARENT_SOURCE_SHA256 or
            link['matched_target_raw_sha256'] !=
                '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f'):
        raise ValueError('Attempt103 frozen linkage constants mismatch')
    old_spec = parent.load_spec(parent.resolve(link['spec_path']))
    result = a.load_json_object(parent.resolve(link['result_path']), 'Attempt103 result')
    if (result.get('attempt_id') != parent.ATTEMPT or
            result.get('provenance', {}).get('spec_sha256') != link['spec_sha256'] or
            result['provenance'].get('source_sha256') != link['source_sha256'] or
            result['provenance'].get('matched_difference_raw_sha256') !=
                link['matched_target_raw_sha256'] or
            result['provenance'].get('base_checkpoint_files') != spec['base']['files'] or
            result['provenance'].get('final_checkpoint_files') != spec['final_checkpoint_files'] or
            result['provenance'].get('fineweb_tokens_sha256') != spec['corpus']['serialized_sha256'] or
            result['provenance'].get('probe_tokens_sha256') != spec['probe']['serialized_sha256'] or
            result.get('construction', {}).get('sequence_count') != 512 or
            result.get('matched_probe', {}).get('sequence_count') != 1024 or
            result['matched_probe'].get('positions_1_4_mean_cosine') != link['primary_cosine'] or
            result['matched_probe'].get('positions_1_127_mean_cosine') != link['secondary_cosine'] or
            result.get('oracle_adl_access') is not False or
            result.get('historical_adapter_access') is not False or
            old_spec['base'] != spec['base']):
        raise ValueError('Attempt103 result/provenance mismatch')
    return result


def validate_inputs(spec):
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt104 result already exists')
    prior_result = validate_attempt103_linkage(spec)
    source_spec = copy.deepcopy(parent.load_spec())
    source_spec['paths']['result_path'] = spec['paths']['result_path']
    inventories = parent.validate_sources(source_spec, smoke_only=True)
    return prior_result, source_spec, inventories


def hidden_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError('Unsupported block13 output container')


def replace_hidden_output(output, base_hidden):
    hidden = hidden_tensor(output)
    if (not isinstance(base_hidden, torch.Tensor) or
            hidden.shape != base_hidden.shape or hidden.dtype != base_hidden.dtype or
            hidden.device != base_hidden.device or
            not bool(torch.isfinite(base_hidden).all())):
        raise ValueError('Base/final block13 hidden shape/dtype/device mismatch')
    replacement = base_hidden.detach()
    if isinstance(output, torch.Tensor):
        return replacement
    if isinstance(output, tuple):
        return (replacement, *output[1:])
    if isinstance(output, list):
        return [replacement, *output[1:]]
    raise ValueError('Unsupported block13 output container')


def base_hidden_and_logits(base, batch, spec):
    layer = base.model.layers[spec['readout']['block_index']]
    captured = []
    def hook(_module, _inputs, output):
        hidden = hidden_tensor(output)
        a.validate_readout(hidden, batch.shape[0], spec, torch, 'base block13 hook')
        captured.append(hidden.detach().clone())
    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
            output = base(input_ids=batch, use_cache=False, output_hidden_states=True,
                          return_dict=True)
    finally:
        handle.remove()
    if handle.id in layer._forward_hooks or len(captured) != 1:
        raise ValueError('Base block13 hook count/removal mismatch')
    selected = a.hidden_state_output(output, spec)
    if not torch.equal(selected, captured[0]):
        raise ValueError('Base block13 hook differs from hidden_states[14]')
    logits = output.logits
    if (logits.dtype != torch.float32 or logits.shape[:2] != batch.shape or
            not bool(torch.isfinite(logits).all())):
        raise ValueError('Invalid base logits')
    return captured[0], logits.detach()


def hybrid_logits(final, batch, base_hidden, spec):
    layer = final.model.layers[spec['readout']['block_index']]
    count = 0
    def hook(_module, _inputs, output):
        nonlocal count
        count += 1
        return replace_hidden_output(output, base_hidden)
    handle = layer.register_forward_hook(hook)
    try:
        with torch.no_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
            output = final(input_ids=batch, use_cache=False)
    finally:
        handle.remove()
    removed = handle.id not in layer._forward_hooks
    if count != 1 or not removed:
        raise ValueError('Hybrid final block13 hook count/removal mismatch')
    logits = output.logits
    if (logits.dtype != torch.float32 or logits.shape[:2] != batch.shape or
            not bool(torch.isfinite(logits).all())):
        raise ValueError('Nonfinite/malformed downstream hybrid logits')
    return logits.detach(), {'hook_fired_count': count, 'hook_removed': removed,
                              'hybrid_logits_finite': True}


def hybrid_teacher_losses(hybrid_logits_shifted, student_logits_shifted):
    if (hybrid_logits_shifted.ndim != 3 or
            hybrid_logits_shifted.shape != student_logits_shifted.shape or
            hybrid_logits_shifted.dtype != torch.float32 or
            student_logits_shifted.dtype != torch.float32 or
            not bool(torch.isfinite(hybrid_logits_shifted).all()) or
            not bool(torch.isfinite(student_logits_shifted).all())):
        raise ValueError('Malformed/nonfinite hybrid teacher or student logits')
    p_hybrid = torch.softmax(hybrid_logits_shifted, dim=-1)
    logp_hybrid = torch.log_softmax(hybrid_logits_shifted, dim=-1)
    logp_student = torch.log_softmax(student_logits_shifted, dim=-1)
    ce_sum = -(p_hybrid*logp_student).sum()
    kl_sum = (p_hybrid*(logp_hybrid-logp_student)).sum()
    if not bool(torch.isfinite(ce_sum)) or not bool(torch.isfinite(kl_sum)):
        raise ValueError('Nonfinite hybrid teacher CE/KL')
    return ce_sum, kl_sum, p_hybrid


class ResidualAccumulator:
    """Only CPU float64 scalar reductions; no probability/residual persistence."""
    def __init__(self, chunk_positions=16):
        if chunk_positions != 16:
            raise ValueError('Frozen residual chunk size changed')
        self.chunk_positions = chunk_positions
        self.dot_parts, self.hybrid_parts, self.full_parts = [], [], []
        self.cosine_parts = []
        self.positive = 0
        self.count = 0

    def add(self, p_base, p_hybrid, p_final):
        if (p_base.shape != p_hybrid.shape or p_base.shape != p_final.shape or
                p_base.ndim != 3 or p_base.shape[1] != 127 or
                any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                    for value in (p_base, p_hybrid, p_final))):
            raise ValueError('Malformed output residual probabilities')
        shape = (-1, p_base.shape[-1])
        base, hybrid, final = (value.reshape(shape) for value in
                               (p_base, p_hybrid, p_final))
        for start in range(0, base.shape[0], self.chunk_positions):
            end = start+self.chunk_positions
            b = base[start:end].to(device='cpu', dtype=torch.float64)
            h = hybrid[start:end].to(device='cpu', dtype=torch.float64)
            f = final[start:end].to(device='cpu', dtype=torch.float64)
            full_residual = f-b
            hybrid_residual = f-h
            dot = (full_residual*hybrid_residual).sum(dim=1)
            full_sq = full_residual.square().sum(dim=1)
            hybrid_sq = hybrid_residual.square().sum(dim=1)
            if bool((full_sq <= 0).any()) or bool((hybrid_sq <= 0).any()):
                raise ValueError('Zero-norm per-prediction output residual')
            cosine = dot/torch.sqrt(full_sq*hybrid_sq)
            if not bool(torch.isfinite(cosine).all()):
                raise ValueError('Nonfinite output residual cosine')
            self.dot_parts.append(float(dot.sum()))
            self.full_parts.append(float(full_sq.sum()))
            self.hybrid_parts.append(float(hybrid_sq.sum()))
            self.cosine_parts.append(float(cosine.sum()))
            self.positive += int((cosine > 0).sum())
            self.count += len(cosine)

    def summary(self):
        dot = math.fsum(self.dot_parts)
        full_sq = math.fsum(self.full_parts)
        hybrid_sq = math.fsum(self.hybrid_parts)
        if self.count <= 0 or full_sq <= 0 or hybrid_sq <= 0:
            raise ValueError('Missing output residual geometry')
        return {'prediction_count': self.count,
                'pooled_cosine_hybrid_vs_full': dot/math.sqrt(full_sq*hybrid_sq),
                'hybrid_to_full_norm_ratio': math.sqrt(hybrid_sq/full_sq),
                'mean_per_prediction_cosine': math.fsum(self.cosine_parts)/self.count,
                'fraction_per_prediction_cosine_positive': self.positive/self.count}


def hybrid_teacher_gradient_stage(base, final, tokens, eligible, spec, *, progress=None):
    if (tokens.ndim != 2 or tokens.shape[1] != 128 or tokens.shape[0] not in (4, 512) or
            tokens.dtype != torch.int64 or not tokens.is_contiguous() or len(eligible) != 98):
        raise ValueError('Frozen hybrid teacher batch/support inventory mismatch')
    denominator = tokens.shape[0]*127
    selected = {id(module.weight) for _, module in eligible}
    final.zero_grad(set_to_none=True)
    ce_parts, kl_parts = [], []
    residuals = ResidualAccumulator(spec['output_residuals']['chunk_prediction_positions'])
    audit = None
    times = {'base_hidden_logits_seconds': 0.0, 'hybrid_final_seconds': 0.0,
             'student_backward_seconds': 0.0}
    started = time.perf_counter()
    device = eligible[0][1].weight.device
    for start in range(0, tokens.shape[0], 4):
        batch = tokens[start:start+4].to(device)
        stage = time.perf_counter()
        base_hidden, z0 = base_hidden_and_logits(base, batch, spec)
        times['base_hidden_logits_seconds'] += time.perf_counter()-stage
        stage = time.perf_counter()
        z_hybrid, hook_audit = hybrid_logits(final, batch, base_hidden, spec)
        times['hybrid_final_seconds'] += time.perf_counter()-stage
        if start == 0:
            audit = hook_audit
        stage = time.perf_counter()
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            z1 = final(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            ce_sum, kl_sum, p_hybrid = hybrid_teacher_losses(z_hybrid[:, :-1, :], z1)
            (ce_sum/denominator).backward()
        times['student_backward_seconds'] += time.perf_counter()-stage
        ce_parts.append(float(ce_sum.detach().cpu().double()))
        kl_parts.append(float(kl_sum.detach().cpu().double()))
        with torch.no_grad():
            p_base = torch.softmax(z0[:, :-1, :], dim=-1)
            p_final = torch.softmax(z1.detach(), dim=-1)
            residuals.add(p_base, p_hybrid, p_final)
        if any(parameter.grad is not None for parameter in base.parameters()):
            raise ValueError('Historical base received gradients')
        if any(parameter.grad is not None for parameter in final.parameters()
               if id(parameter) not in selected):
            raise ValueError('Noneligible final parameter received gradient')
        if progress is not None:
            progress(start//4+1, tokens.shape[0]//4, time.perf_counter()-started)
        del batch, base_hidden, z0, z_hybrid, z1, ce_sum, kl_sum, p_hybrid, p_base, p_final
    for _, module in eligible:
        if module.weight.grad is None or not bool(torch.isfinite(module.weight.grad).all()):
            raise ValueError('Missing/nonfinite positive hybrid teacher gradient')
    return {'mean_hybrid_teacher_cross_entropy': math.fsum(ce_parts)/denominator,
            'mean_kl_hybrid_to_final': math.fsum(kl_parts)/denominator,
            'total_prediction_tokens': denominator,
            'first_batch_hybrid_hook_audit': audit,
            'output_residual_diagnostics': residuals.summary()}, times


def interpretation_category(primary, delta_primary, spec):
    thresholds = spec['interpretation']
    strong = thresholds['strong_late_layer_contamination_support']
    limited = thresholds['curvature_limited_support']
    if primary >= strong['primary_min'] and delta_primary >= strong['delta_primary_min']:
        return 'strong_late_layer_contamination_support'
    if (primary < limited['primary_max_exclusive'] and
            delta_primary < limited['delta_primary_max_exclusive']):
        return 'curvature_limited_support'
    return 'ambiguous'


def runtime_projection(loads, times, base_probe_seconds, jvp_seconds):
    teacher = sum(times.values())*128
    probe = (base_probe_seconds+jvp_seconds)*32
    return {'projected_full_teacher_stage_seconds': teacher,
            'projected_full_probe_stage_seconds': probe,
            'projected_total_compute_seconds': teacher+probe,
            'projected_total_wall_seconds':
                loads['base_load_seconds']+loads['final_load_seconds']+teacher+probe}


def verify_matched_target(target, spec):
    digest = a.sha256_raw_float32_tensor(target, torch)
    if digest != spec['attempt103']['matched_target_raw_sha256']:
        raise ValueError('Matched target raw SHA256 differs from Attempt103')
    return digest


def run(*, smoke_only=False, model_loader=None, tokenizer_loader=None):
    spec = load_spec()
    prior_result, old_spec, inventories = validate_inputs(spec)
    construction, probe = parent.load_frozen_tokens(old_spec)
    if smoke_only:
        construction, probe = construction[:4].contiguous(), probe[:32].contiguous()
    base, final, loads = parent.load_model_pair(old_spec, model_loader=model_loader,
                                                tokenizer_loader=tokenizer_loader)
    eligible = a.discover_eligible_linear_weights(final, torch)
    a.freeze_other_parameters(final, eligible)
    base_before = a.model_state_hashes(base, torch)
    final_before = a.model_state_hashes(final, torch)
    teacher, timing = hybrid_teacher_gradient_stage(
        base, final, construction, eligible, spec,
        progress=None if smoke_only else parent.progress_printer('hybrid teacher gradient'))
    scale = a.global_tangent_scale(eligible, 0.00125, torch)
    tangents, tangent_records, realized = a.prepare_tangents(eligible, scale, torch)
    if len(tangents) != 98:
        raise ValueError('Hybrid tangent support is not exactly 98 matrices')
    started = time.perf_counter()
    base_mean = parent.base_probe_mean(base, probe, old_spec,
                                       progress=None if smoke_only else parent.progress_printer('base probe mean'))
    base_probe_seconds = time.perf_counter()-started
    a.verify_model_unchanged(base, base_before, torch)
    if any(parameter.grad is not None for parameter in base.parameters()):
        raise ValueError('Historical base received gradients')
    del base, base_before
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    started = time.perf_counter()
    final_mean, response, jvp_audit = parent.final_probe_jvp(
        final, probe, tangents, old_spec,
        progress=None if smoke_only else parent.progress_printer('final JVP probe'))
    jvp_seconds = time.perf_counter()-started
    a.verify_model_unchanged(final, final_before, torch)
    del final
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    target = parent.matched_difference(final_mean, base_mean)
    if smoke_only:
        return {'smoke_only': True, 'model_load_seconds': loads,
                **timing, 'base_probe_batch_seconds': base_probe_seconds,
                'final_jvp_probe_batch_seconds': jvp_seconds,
                **runtime_projection(loads, timing, base_probe_seconds, jvp_seconds),
                'first_batch_hybrid_hook_audit': teacher['first_batch_hybrid_hook_audit'],
                'writes': False}
    target_hash = verify_matched_target(target, spec)
    geometry = parent.response_metrics(response, target)
    primary = geometry['positions_1_4_mean_cosine']
    secondary = geometry['positions_1_127_mean_cosine']
    delta_primary = primary-spec['attempt103']['primary_cosine']
    delta_secondary = secondary-spec['attempt103']['secondary_cosine']
    category = interpretation_category(primary, delta_primary, spec)
    for directory, expected in ((parent.resolve(spec['paths']['base_directory']), inventories['base_files']),
                                (parent.resolve(spec['paths']['final_directory']), inventories['final_files'])):
        if a.checkpoint_file_records(directory) != expected:
            raise ValueError('Checkpoint changed during Attempt104')
    result = {'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
              'privileged_mechanism_diagnostic_not_blind_recovery': True,
              'candidate_selection': False, 'oracle_based_tuning': False,
              'provenance': {'spec_sha256': SPEC_SHA256,
                             'source_sha256': a.sha256_file(Path(__file__).resolve()),
                             'attempt103_result_sha256': spec['attempt103']['result_sha256'],
                             'base_checkpoint_files': inventories['base_files'],
                             'final_checkpoint_files': inventories['final_files'],
                             'fineweb_tokens_sha256': spec['corpus']['serialized_sha256'],
                             'probe_tokens_sha256': spec['probe']['serialized_sha256'],
                             'matched_target_raw_sha256': target_hash,
                             'hybrid_teacher_response_raw_sha256':
                                 a.sha256_raw_float32_tensor(response, torch)},
              'construction': {'sequence_count': 512, 'batch_size': 4, **teacher,
                               'hybrid_teacher_gradient_norm': scale['aggregate_generic_gradient_norm'],
                               'alpha': scale['alpha'],
                               'realized_tangent_norm': realized['aggregate_realized_fp32_tangent_norm'],
                               'realized_relative_tangent_norm':
                                   realized['aggregate_realized_relative_tangent_norm'],
                               'tangent_matrix_count': len(tangent_records)},
              'matched_probe': {'sequence_count': 1024, 'batch_size': 32,
                                'jvp_first_batch_primal_max_abs_difference': jvp_audit,
                                **geometry},
              'attempt103_direct_comparison': {'attempt103_primary': spec['attempt103']['primary_cosine'],
                                               'attempt103_secondary': spec['attempt103']['secondary_cosine'],
                                               'delta_primary_vs_103': delta_primary,
                                               'delta_secondary_vs_103': delta_secondary,
                                               'category': category,
                                               'category_is_descriptive_only': True,
                                               'thresholds': spec['interpretation']},
              'historical_adapter_access': False, 'oracle_adl_access': False,
              'no_oracle_based_tuning': True}
    output = parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt104 result already exists')
    a.write_manifest(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(smoke_only=args.smoke_only), allow_nan=False))


if __name__ == '__main__':
    main()
