#!/usr/bin/env python3
"""Attempt105: privileged exact endpoint categorical GGN forward diagnostic."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '105_privileged_endpoint_ggn_forward_pilot'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '8125e624850a18268d369f0509d1b0a9876156b5a2635e960588d2026ddc958c'
SOURCE104 = Path(__file__).with_name('diagnose_privileged_hybrid_prefix_soft_teacher.py')
SOURCE016 = Path(__file__).with_name('construct_exact_displacement_jvp_ceiling.py')
SHA104 = '980137e77b650f92d8a721d52a104a2db0fdb0f44fc874b76526e5d1353c9d4a'
SHA016 = 'a682ce8a5dd01b2bc080b913d5148ba00544217be6a9b6b5a9ec66bc03da755a'
for source, digest in ((SOURCE104, SHA104), (SOURCE016, SHA016)):
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen Attempt105 helper source changed: '+str(source))
def _load(name, source):
    loader = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module
a104 = _load('attempt105_frozen_attempt104', SOURCE104)
a016 = _load('attempt105_frozen_attempt016', SOURCE016)
a = a104.a


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt105 spec path changed')
    a104.parent.require_hash(path, SPEC_SHA256)
    spec = a.load_json_object(path, 'Attempt105 spec')
    old = a104.load_spec()
    for key in ('base', 'final_checkpoint_files', 'model', 'eligible_tensors', 'corpus', 'readout'):
        if spec.get(key) != old[key]:
            raise ValueError('Attempt105 frozen input differs from Attempt104: '+key)
    if (spec.get('attempt_id') != ATTEMPT or
            spec.get('pilot', {}).get('construction_rows') != [0, 64] or
            spec['pilot'].get('hybrid_batch_size') != 4 or
            spec['pilot'].get('ggn_batch_size') != 1 or
            spec['pilot'].get('prediction_contexts') != 8128 or
            spec.get('ggn', {}).get('operator') != 'exact_endpoint_categorical_JtFJ' or
            spec['attempt104']['required_category'] != 'curvature_limited_support'):
        raise ValueError('Attempt105 scientific specification mismatch')
    for value in spec['paths'].values():
        a104.parent.resolve(value)
    return spec


def validate_inputs(spec):
    output = a104.parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt105 result already exists')
    link104 = spec['attempt104']
    for key in ('spec', 'source', 'result'):
        a104.parent.require_hash(a104.parent.resolve(link104[key+'_path']),
                                 link104[key+'_sha256'])
    if link104['source_sha256'] != SHA104 or link104['spec_sha256'] != a104.SPEC_SHA256:
        raise ValueError('Attempt104 source/spec linkage mismatch')
    a104.load_spec(a104.parent.resolve(link104['spec_path']))
    result104 = a.load_json_object(a104.parent.resolve(link104['result_path']),
                                   'Attempt104 result')
    if (result104.get('attempt_id') != a104.ATTEMPT or
            result104.get('provenance', {}).get('spec_sha256') != link104['spec_sha256'] or
            result104['provenance'].get('source_sha256') != link104['source_sha256'] or
            result104['provenance'].get('base_checkpoint_files') != spec['base']['files'] or
            result104['provenance'].get('final_checkpoint_files') != spec['final_checkpoint_files'] or
            result104['provenance'].get('fineweb_tokens_sha256') != spec['corpus']['serialized_sha256'] or
            result104.get('attempt103_direct_comparison', {}).get('category') !=
                link104['required_category'] or
            result104.get('oracle_adl_access') is not False or
            result104.get('historical_adapter_access') is not False):
        raise ValueError('Attempt104 result/provenance/category mismatch')
    link016 = spec['attempt016']
    for key in ('spec', 'source'):
        a104.parent.require_hash(a104.parent.resolve(link016[key+'_path']),
                                 link016[key+'_sha256'])
    if link016['source_sha256'] != SHA016:
        raise ValueError('Attempt016 source linkage mismatch')
    old016 = a016.load_spec(a016.resolve(link016['spec_path']))
    if (old016['base'] != spec['base'] or
            old016['canonical_checkpoint_files'] != spec['final_checkpoint_files'] or
            old016['eligible_tensors'] != spec['eligible_tensors'] or
            a016.LOADED_HELPER_SHA256 != old016['helper_source']['sha256']):
        raise ValueError('Attempt016 displacement provenance mismatch')
    base_dir = a104.parent.resolve(spec['paths']['base_directory'])
    final_dir = a104.parent.resolve(spec['paths']['final_directory'])
    base_files = a.checkpoint_file_records(base_dir)
    final_files = a.checkpoint_file_records(final_dir)
    if (base_files != spec['base']['files'] or
            final_files != spec['final_checkpoint_files'] or
            spec['base']['repo_id'] != 'Qwen/Qwen3-1.7B' or
            spec['base']['revision'] != '0060bc56d46589041c1048efd1a397421b1142b5'):
        raise ValueError('Base/final checkpoint inventory or historical identity mismatch')
    corpus = spec['corpus']
    fineweb = a.FROZEN_SPEC['corpora'][0]
    if (corpus['spec_path'] != fineweb['spec_path'] or
            corpus['spec_sha256'] != fineweb['spec_sha256'] or
            corpus['manifest_path'] != fineweb['manifest_path'] or
            corpus['manifest_sha256'] != fineweb['manifest_sha256'] or
            spec['paths']['corpus_path'] != fineweb['tokens_path']):
        raise ValueError('Frozen FineWeb corpus provenance mismatch')
    for path, digest in ((corpus['spec_path'], corpus['spec_sha256']),
                         (corpus['manifest_path'], corpus['manifest_sha256']),
                         (spec['paths']['corpus_path'], corpus['serialized_sha256'])):
        a104.parent.require_hash(a104.parent.resolve(path), digest)
    return old016, {'base_files': base_files, 'final_files': final_files}


def load_pilot_tokens(spec):
    old = a104.parent.a
    tokens, _, record = old.validate_frozen_corpus(
        old.FROZEN_SPEC['corpora'][0],
        a104.parent.resolve(spec['paths']['final_directory']), torch)
    if (tokens.shape != (4096, 128) or
            record['serialized_sha256'] != spec['corpus']['serialized_sha256'] or
            old.sha256_raw_int64_tensor(tokens, torch) != spec['corpus']['raw_tensor_sha256']):
        raise ValueError('Frozen FineWeb pilot corpus inventory/hash mismatch')
    selected = tokens[:64].contiguous()
    del tokens
    return selected


def prepare_displacement(base, final, spec016, base_inventory):
    base_eligible = a016.a.discover_eligible_linear_weights(base, torch)
    a016.a.freeze_other_parameters(base, base_eligible)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base_execution = a016.execution_snapshot(base, spec016)
    saved, base_records, coordinates = a016.snapshot_base(base, base_eligible)
    final_eligible = a016.a.discover_eligible_linear_weights(final, torch)
    a016.a.freeze_other_parameters(final, final_eligible)
    for parameter in final.parameters():
        parameter.requires_grad_(False)
    final_execution = a016.execution_snapshot(final, spec016)
    execution_audit = a016.audit_execution_semantics(base_execution, final_execution)
    with a016.BaseTensorReader(a104.parent.resolve(spec016['defaults']['base_directory']),
                               base_inventory) as reader:
        tangent, audit = a016.audit_and_tangent(final, final_eligible, saved,
                                                base_records, coordinates, reader.get)
    saved.clear()
    if (len(tangent) != 98 or not audit['omitted_upstream_zero_verified'] or
            audit['categories']['selected_early_linear_weights']['parameter_count'] != 98):
        raise ValueError('Full Attempt016 early displacement audit failed')
    a.freeze_other_parameters(final, final_eligible)
    return tangent, audit, execution_audit, final_eligible


def hybrid_gradient_64(base, final, tokens, eligible, spec, *, progress=None):
    if (tokens.ndim != 2 or tokens.shape[1] != 128 or tokens.shape[0] not in (4, 64) or
            tokens.dtype != torch.int64 or not tokens.is_contiguous() or len(eligible) != 98):
        raise ValueError('Frozen hybrid-gradient pilot inventory mismatch')
    denominator = 64*127
    selected = {id(module.weight) for _, module in eligible}
    final.zero_grad(set_to_none=True)
    parts = []
    started = time.perf_counter()
    device = eligible[0][1].weight.device
    helper_spec = {**spec, 'probe': {'sequence_length': spec['pilot']['sequence_length']}}
    for start in range(0, tokens.shape[0], 4):
        batch = tokens[start:start+4].to(device)
        base_hidden, _ = a104.base_hidden_and_logits(base, batch, helper_spec)
        hybrid_logits, hook_audit = a104.hybrid_logits(final, batch, base_hidden,
                                                       helper_spec)
        if hook_audit['hook_fired_count'] != 1 or not hook_audit['hook_removed']:
            raise ValueError('Hybrid forward hook audit failed')
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            student_logits = final(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            ce_sum, _, _ = a104.hybrid_teacher_losses(hybrid_logits[:, :-1, :],
                                                       student_logits)
            (ce_sum/denominator).backward()
        parts.append(float(ce_sum.detach().cpu().double()))
        if any(parameter.grad is not None for parameter in base.parameters()):
            raise ValueError('Historical base received a gradient')
        if any(parameter.grad is not None for parameter in final.parameters()
               if id(parameter) not in selected):
            raise ValueError('Noneligible final parameter received a gradient')
        if progress is not None:
            progress(start//4+1, tokens.shape[0]//4, time.perf_counter()-started)
        del batch, base_hidden, hybrid_logits, student_logits, ce_sum
    copied = {}
    for name, module in eligible:
        gradient = module.weight.grad
        if (gradient is None or gradient.dtype != torch.float32 or
                not bool(torch.isfinite(gradient).all())):
            raise ValueError('Missing/nonfinite positive hybrid pilot gradient')
        copied[f'{name}.weight'] = gradient.detach().to(device='cpu', dtype=torch.float32,
                                                       copy=True).contiguous()
    final.zero_grad(set_to_none=True)
    return copied, {'mean_hybrid_teacher_ce_contribution': math.fsum(parts)/denominator,
                    'scored_prediction_contexts': tokens.shape[0]*127,
                    'normalization_denominator': denominator}


def strict_logits_jvp(final, batch, delta):
    if not delta:
        raise ValueError('GGN logits JVP requires nonempty true displacement')
    names = sorted(delta)
    primals, tangents = [], []
    for name in names:
        parameter = final.get_parameter(name)
        vector = delta[name]
        if (parameter.dtype != torch.float32 or vector.dtype != torch.float32 or
                parameter.shape != vector.shape or parameter.device != vector.device or
                not bool(torch.isfinite(vector).all())):
            raise ValueError('Invalid true-Delta logits JVP coordinate')
        primals.append(parameter.detach())
        tangents.append(vector)
    def logits(weights):
        output = torch.func.functional_call(
            final, dict(zip(names, weights)), (),
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
        raise ValueError('Malformed/nonfinite logits JVP primal or tangent')
    return primal.detach(), action.detach()


def audit_logits_primal(primal, ordinary, spec):
    if (primal.shape != ordinary.shape or primal.dtype != torch.float32 or
            ordinary.dtype != torch.float32 or not bool(torch.isfinite(ordinary).all())):
        raise ValueError('Invalid ordinary final logits audit')
    ordinary = ordinary.detach()
    difference = (primal.detach().double()-ordinary.double())
    max_abs = float(difference.abs().max())
    rms = float(torch.sqrt(difference.square().mean()))
    scale_max = float(ordinary.double().abs().max())
    scale_rms = float(torch.sqrt(ordinary.double().square().mean()))
    rule = spec['ggn']
    if (max_abs > rule['first_batch_primal_atol']+rule['first_batch_primal_rtol']*scale_max or
            rms > rule['first_batch_rms_atol']+rule['first_batch_rms_rtol']*scale_rms):
        raise ValueError('GGN JVP primal differs from ordinary final logits')
    return {'max_abs_difference': max_abs, 'rms_difference': rms,
            'max_scale': scale_max, 'rms_scale': scale_rms}


def categorical_fisher_vector(logits, tangent, denominator, spec):
    if (logits.shape != tangent.shape or logits.ndim != 3 or
            logits.dtype != torch.float32 or tangent.dtype != torch.float32 or
            denominator != 8128 or not bool(torch.isfinite(logits).all()) or
            not bool(torch.isfinite(tangent).all())):
        raise ValueError('Invalid categorical Fisher input/normalization')
    p = torch.softmax(logits, dim=-1)
    mean_s = (p*tangent).sum(dim=-1, keepdim=True)
    u = (p*(tangent-mean_s))/denominator
    if not bool(torch.isfinite(u).all()):
        raise ValueError('Nonfinite categorical Fisher vector')
    residual = u.sum(dim=-1).abs()
    scale = u.abs().sum(dim=-1)
    rule = spec['ggn']
    allowed = (rule['fisher_conservation_relative_tolerance']*scale+
               rule['fisher_conservation_absolute_tolerance'])
    if bool((residual > allowed).any()):
        raise ValueError('Categorical Fisher vector fails sum-to-zero audit')
    return u.detach(), {'max_sum_vocab_abs': float(residual.max()),
                        'max_conservation_scale': float(scale.max())}


def ggn_action_stage(final, tokens, eligible, delta, spec, *, progress=None):
    if (tokens.ndim != 2 or tokens.shape[1] != 128 or tokens.shape[0] not in (1, 64) or
            tokens.dtype != torch.int64 or not tokens.is_contiguous() or
            len(eligible) != 98 or len(delta) != 98):
        raise ValueError('Frozen GGN pilot/support inventory mismatch')
    selected = {id(module.weight) for _, module in eligible}
    final.zero_grad(set_to_none=True)
    timings = {'logits_jvp_seconds': 0.0, 'fisher_vector_seconds': 0.0,
               'vjp_backward_seconds': 0.0}
    first_audit = None
    started = time.perf_counter()
    device = eligible[0][1].weight.device
    for start in range(tokens.shape[0]):
        batch = tokens[start:start+1].to(device)
        stage = time.perf_counter()
        primal, tangent = strict_logits_jvp(final, batch, delta)
        timings['logits_jvp_seconds'] += time.perf_counter()-stage
        primal_audit = None
        if start == 0:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
                ordinary = final(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            primal_audit = audit_logits_primal(primal, ordinary, spec)
            del ordinary
        stage = time.perf_counter()
        fisher, conservation = categorical_fisher_vector(
            primal, tangent, 64*127, spec)
        timings['fisher_vector_seconds'] += time.perf_counter()-stage
        stage = time.perf_counter()
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            logits = final(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            if logits.shape != fisher.shape or not bool(torch.isfinite(logits).all()):
                raise ValueError('GGN VJP final logits mismatch')
            (logits*fisher.detach()).sum().backward()
        timings['vjp_backward_seconds'] += time.perf_counter()-stage
        if any(parameter.grad is not None for parameter in final.parameters()
               if id(parameter) not in selected):
            raise ValueError('Noneligible final parameter received GGN VJP gradient')
        if start == 0:
            if any(module.weight.grad is None or not bool(torch.isfinite(module.weight.grad).all())
                   for _, module in eligible):
                raise ValueError('Missing/nonfinite first-batch selected GGN VJP gradient')
            first_audit = {'primal': primal_audit, 'fisher': conservation,
                           'tangent_shape': list(tangent.shape),
                           'tangent_dtype': str(tangent.dtype),
                           'selected_vjp_matrix_count': len(eligible)}
        if progress is not None:
            progress(start+1, tokens.shape[0], time.perf_counter()-started)
        del batch, primal, tangent, fisher, logits
    return first_audit, timings


def _coordinate_family(name):
    parts = name.split('.')
    if (len(parts) < 5 or parts[:2] != ['model', 'layers'] or
            not parts[2].isdigit() or int(parts[2]) not in range(14) or
            parts[-1] != 'weight'):
        raise ValueError('Unexpected early-coordinate name: '+name)
    if 'self_attn' in parts:
        return int(parts[2]), 'attention'
    if 'mlp' in parts:
        return int(parts[2]), 'mlp'
    raise ValueError('Unknown early-coordinate family: '+name)


def matrixwise_metrics(eligible, delta, hybrid):
    names = [f'{name}.weight' for name, _ in eligible]
    if not names or set(names) != set(delta) or set(names) != set(hybrid):
        raise ValueError('GGN/hybrid/Delta coordinate inventory mismatch')
    buckets = {'global': []}
    for block in range(14):
        buckets[f'block_{block}'] = []
    buckets['attention'] = []
    buckets['mlp'] = []
    for name, module in eligible:
        key = f'{name}.weight'
        action = module.weight.grad
        other = hybrid[key]
        direction = delta[key]
        if (action is None or action.dtype != torch.float32 or
                other.dtype != torch.float32 or direction.dtype != torch.float32 or
                action.shape != other.shape or action.shape != direction.shape or
                other.device.type != 'cpu' or not other.is_contiguous() or
                not bool(torch.isfinite(action).all()) or
                not bool(torch.isfinite(other).all()) or
                not bool(torch.isfinite(direction).all())):
            raise ValueError('Malformed/nonfinite matrixwise GGN comparison tensor')
        x = action.detach().to(device='cpu', dtype=torch.float64)
        y = other.double()
        d = direction.detach().to(device='cpu', dtype=torch.float64)
        record = {'xx': float(x.square().sum()), 'yy': float(y.square().sum()),
                  'dd': float(d.square().sum()), 'xy': float((x*y).sum()),
                  'xd': float((x*d).sum()), 'yd': float((y*d).sum())}
        block, family = _coordinate_family(key)
        for group in ('global', f'block_{block}', family):
            buckets[group].append(record)
        del x, y, d
    def reduce(records):
        return {field: math.fsum(row[field] for row in records)
                for field in ('xx', 'yy', 'dd', 'xy', 'xd', 'yd')}
    global_stats = reduce(buckets['global'])
    xx, yy, dd, xy = (global_stats[k] for k in ('xx', 'yy', 'dd', 'xy'))
    if min(xx, yy, dd) <= 0:
        raise ValueError('Zero GGN, hybrid-gradient, or true-Delta global norm')
    scalar = xy/xx
    relative_residual = math.sqrt(max(0.0, yy-2*scalar*xy+scalar*scalar*xx)/yy)
    groups = {}
    for group, records in buckets.items():
        if group == 'global':
            continue
        row = reduce(records)
        groups[group] = {'matrix_count': len(records),
                         'cosine': row['xy']/math.sqrt(row['xx']*row['yy'])
                             if row['xx'] > 0 and row['yy'] > 0 else None,
                         'ggn_norm': math.sqrt(row['xx']),
                         'hybrid_gradient_norm': math.sqrt(row['yy']),
                         'ggn_squared_norm_fraction': row['xx']/xx,
                         'hybrid_squared_norm_fraction': row['yy']/yy}
    return {'primary_global_signed_cosine': xy/math.sqrt(xx*yy),
            'ggn_to_hybrid_norm_ratio': math.sqrt(xx/yy),
            'optimal_scalar_ggn_to_hybrid': scalar,
            'relative_residual_after_optimal_scalar': relative_residual,
            'dot_ggn_hybrid': xy, 'ggn_norm': math.sqrt(xx),
            'hybrid_gradient_norm': math.sqrt(yy),
            'true_delta_norm': math.sqrt(dd),
            'cosine_ggn_true_delta': global_stats['xd']/math.sqrt(xx*dd),
            'cosine_hybrid_true_delta': global_stats['yd']/math.sqrt(yy*dd),
            'ggn_to_true_delta_norm_ratio': math.sqrt(xx/dd),
            'hybrid_to_true_delta_norm_ratio': math.sqrt(yy/dd),
            'structured': {'blocks': {str(i): groups[f'block_{i}'] for i in range(14)},
                           'attention': groups['attention'], 'mlp': groups['mlp']}}


def interpretation(cosine, spec):
    if cosine >= spec['interpretation']['strong_endpoint_ggn_support']['cosine_min']:
        return 'strong_endpoint_ggn_support'
    if cosine >= spec['interpretation']['mixed_endpoint_and_path_effects']['cosine_min']:
        return 'mixed_endpoint_and_path_effects'
    return 'endpoint_ggn_insufficient'


def runtime_projection(loads, displacement_seconds, hybrid_batch_seconds,
                       jvp_seconds, fisher_seconds, vjp_seconds):
    hybrid = hybrid_batch_seconds*16
    ggn = (jvp_seconds+fisher_seconds+vjp_seconds)*64
    fixed = loads['base_load_seconds']+loads['final_load_seconds']+displacement_seconds
    return {'projected_hybrid_stage_seconds': hybrid,
            'projected_ggn_stage_seconds': ggn,
            'fixed_load_displacement_seconds': fixed,
            'projected_total_wall_seconds': fixed+hybrid+ggn}


def run(*, smoke_only=False, model_loader=None, tokenizer_loader=None):
    spec = load_spec()
    spec016, inventories = validate_inputs(spec)
    tokens = load_pilot_tokens(spec)
    base, final, loads = a104.parent.load_model_pair(
        spec, model_loader=model_loader, tokenizer_loader=tokenizer_loader)
    base_before = a.model_state_hashes(base, torch)
    final_before = a.model_state_hashes(final, torch)
    displacement_started = time.perf_counter()
    delta, displacement_audit, execution_audit, eligible = prepare_displacement(
        base, final, spec016, inventories['base_files'])
    displacement_seconds = time.perf_counter()-displacement_started
    hybrid_tokens = tokens[:4].contiguous() if smoke_only else tokens
    started = time.perf_counter()
    hybrid_gradient, hybrid_info = hybrid_gradient_64(
        base, final, hybrid_tokens, eligible, spec,
        progress=None if smoke_only else a104.parent.progress_printer('hybrid gradient'))
    hybrid_seconds = time.perf_counter()-started
    a.verify_model_unchanged(base, base_before, torch)
    if any(parameter.grad is not None for parameter in base.parameters()):
        raise ValueError('Historical base received gradients')
    del base, base_before
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ggn_tokens = tokens[:1].contiguous() if smoke_only else tokens
    first_audit, ggn_timings = ggn_action_stage(
        final, ggn_tokens, eligible, delta, spec,
        progress=None if smoke_only else a104.parent.progress_printer('endpoint GGN action'))
    if smoke_only:
        result = {'smoke_only': True, 'model_load_seconds': loads,
                  'displacement_preparation_seconds': displacement_seconds,
                  'hybrid_gradient_batch_seconds': hybrid_seconds,
                  'ggn_first_batch_audit': first_audit,
                  **ggn_timings,
                  **runtime_projection(loads, displacement_seconds, hybrid_seconds,
                                       ggn_timings['logits_jvp_seconds'],
                                       ggn_timings['fisher_vector_seconds'],
                                       ggn_timings['vjp_backward_seconds']),
                  'writes': False}
        final.zero_grad(set_to_none=True)
        delta.clear(); hybrid_gradient.clear()
        a.verify_model_unchanged(final, final_before, torch)
        return result
    geometry = matrixwise_metrics(eligible, delta, hybrid_gradient)
    category = interpretation(geometry['primary_global_signed_cosine'], spec)
    final.zero_grad(set_to_none=True)
    delta.clear(); hybrid_gradient.clear()
    a.verify_model_unchanged(final, final_before, torch)
    for directory, expected in ((a104.parent.resolve(spec['paths']['base_directory']), inventories['base_files']),
                                (a104.parent.resolve(spec['paths']['final_directory']), inventories['final_files'])):
        if a.checkpoint_file_records(directory) != expected:
            raise ValueError('Checkpoint files changed during Attempt105')
    result = {'format_version': 1, 'attempt_id': ATTEMPT,
              'purpose': spec['purpose'],
              'privileged_mechanism_diagnostic': True,
              'historical_base_access': True, 'oracle_adl_access': False,
              'adapter_access': False, 'inverse_solve': False,
              'empirical_fisher': False,
              'ggn_operator': 'exact_endpoint_categorical_JtFJ',
              'candidate_selection': False, 'oracle_based_tuning': False,
              'provenance': {'spec_sha256': SPEC_SHA256,
                             'source_sha256': a.sha256_file(Path(__file__).resolve()),
                             'attempt104_result_sha256': spec['attempt104']['result_sha256'],
                             'attempt016_spec_sha256': spec['attempt016']['spec_sha256'],
                             'base_checkpoint_files': inventories['base_files'],
                             'final_checkpoint_files': inventories['final_files'],
                             'fineweb_tokens_sha256': spec['corpus']['serialized_sha256']},
              'pilot_inventory': spec['pilot'],
              'displacement_audit': displacement_audit,
              'execution_semantics_audit': execution_audit,
              'hybrid_gradient': hybrid_info,
              'ggn_first_batch_audit': first_audit,
              'parameter_geometry': geometry,
              'interpretation': {'category': category,
                                 'thresholds': spec['interpretation'],
                                 'descriptive_compute_decision_only': True}}
    output = a104.parent.resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt105 result already exists')
    a.write_manifest(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(smoke_only=args.smoke_only), allow_nan=False))


if __name__ == '__main__':
    main()
