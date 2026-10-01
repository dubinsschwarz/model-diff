#!/usr/bin/env python3
"""Attempt 103: exact-base soft-teacher mechanism ceiling, never blind recovery."""
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
ATTEMPT = '103_privileged_base_soft_target_rollback_ceiling'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '512126df3b2078b1745c74ae6395a766808991f54ee64e1cd43a764f69eebb0c'
KNOWN_SOURCE = Path(__file__).with_name('construct_known_base_jg0_subtraction.py')
KNOWN_SOURCE_SHA256 = '66593968b8bf5637ff5aadfa9455604634079f3de0868448e92fd32aef1e1045'
if hashlib.sha256(KNOWN_SOURCE.read_bytes()).hexdigest() != KNOWN_SOURCE_SHA256:
    raise ValueError('Frozen Attempt015 source changed')
loader = importlib.util.spec_from_file_location('attempt103_frozen_known_base_helpers', KNOWN_SOURCE)
known = importlib.util.module_from_spec(loader)
loader.loader.exec_module(known)
a = known.a


def resolve(value):
    path = Path(value)
    if any(part in ('oracle_adl', 'adapter') for part in path.parts) or path.name == 'evaluation.json':
        raise ValueError('Forbidden Attempt103 input path')
    return path if path.is_absolute() else PROJECT/path


def require_hash(path, digest):
    if a.sha256_file(path) != digest:
        raise ValueError('Frozen Attempt103 input SHA256 mismatch: '+str(path))


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt103 spec path changed')
    require_hash(path, SPEC_SHA256)
    spec = a.load_json_object(path, 'Attempt103 spec')
    if (spec.get('attempt_id') != ATTEMPT or spec.get('base') != known.FROZEN_SPEC['base'] or
            spec.get('final_checkpoint_files') != known.FROZEN_SPEC['canonical_checkpoint_files'] or
            spec.get('model') != a.FROZEN_SPEC['model'] or
            spec.get('eligible_tensors') != a.FROZEN_SPEC['eligible_tensors'] or
            spec.get('readout') != a.FROZEN_SPEC['readout'] or
            spec.get('jvp') != a.FROZEN_SPEC['jvp'] or
            spec.get('tangent') != a.FROZEN_SPEC['tangent'] or
            spec.get('probe', {}).get('sample_count') != 1024 or
            {k: v for k, v in spec['probe'].items() if k != 'sample_count'} !=
                {k: v for k, v in a.FROZEN_SPEC['probe'].items() if k != 'sample_count'} or
            spec.get('corpus', {}).get('construction_rows') != [0, 512] or
            spec['corpus'].get('batch_size') != 4 or
            spec.get('probe_rows') != [0, 1024] or
            spec['metric'].get('primary_positions') != [1, 2, 3, 4]):
        raise ValueError('Attempt103 scientific specification mismatch')
    for path_value in spec['paths'].values():
        resolve(path_value)
    return spec


def validate_sources(spec, *, smoke_only=False):
    result = resolve(spec['paths']['result_path'])
    if not smoke_only and (result.exists() or result.is_symlink()):
        raise ValueError('Attempt103 result already exists')
    helpers = spec['source_helpers']
    for key in ('attempt015_spec', 'attempt015_source', 'attempt014_source'):
        require_hash(resolve(helpers[key+'_path']), helpers[key+'_sha256'])
    if (helpers['attempt015_source_sha256'] != KNOWN_SOURCE_SHA256 or
            helpers['attempt014_source_sha256'] != a.sha256_file(Path(a.__file__))):
        raise ValueError('Frozen helper source linkage mismatch')
    prior_spec = known.load_spec(resolve(helpers['attempt015_spec_path']))
    if prior_spec['base'] != spec['base'] or prior_spec['canonical_checkpoint_files'] != spec['final_checkpoint_files']:
        raise ValueError('Attempt015 historical-base identity linkage mismatch')
    base_dir = resolve(spec['paths']['base_directory'])
    final_dir = resolve(spec['paths']['final_directory'])
    base_files = a.checkpoint_file_records(base_dir)
    final_files = a.checkpoint_file_records(final_dir)
    if (base_files != spec['base']['files'] or
            final_files != spec['final_checkpoint_files'] or
            spec['base']['repo_id'] != 'Qwen/Qwen3-1.7B' or
            spec['base']['revision'] != '0060bc56d46589041c1048efd1a397421b1142b5'):
        raise ValueError('Base/final checkpoint identity or file inventory mismatch')
    corpus = spec['corpus']
    fineweb = a.FROZEN_SPEC['corpora'][0]
    if (corpus['spec_path'] != fineweb['spec_path'] or
            corpus['spec_sha256'] != fineweb['spec_sha256'] or
            corpus['manifest_path'] != fineweb['manifest_path'] or
            corpus['manifest_sha256'] != fineweb['manifest_sha256'] or
            spec['paths']['corpus_path'] != fineweb['tokens_path']):
        raise ValueError('FineWeb provenance linkage mismatch')
    require_hash(resolve(corpus['spec_path']), corpus['spec_sha256'])
    require_hash(resolve(corpus['manifest_path']), corpus['manifest_sha256'])
    require_hash(resolve(spec['paths']['corpus_path']), corpus['serialized_sha256'])
    require_hash(resolve(spec['paths']['probe_path']), spec['probe']['serialized_sha256'])
    return {'base_files': base_files, 'final_files': final_files}


def load_frozen_tokens(spec):
    corpus = spec['corpus']
    all_tokens, _, _ = a.validate_frozen_corpus(a.FROZEN_SPEC['corpora'][0],
                                                 resolve(spec['paths']['final_directory']), torch)
    if (tuple(all_tokens.shape) != (4096, 128) or
            a.sha256_raw_int64_tensor(all_tokens, torch) != corpus['raw_tensor_sha256']):
        raise ValueError('Frozen FineWeb token inventory/hash mismatch')
    construction = all_tokens[:512].contiguous()
    del all_tokens
    probe = a.load_probe(resolve(spec['paths']['probe_path']),
                         {**spec['probe'], 'sample_count': 10000}, torch)
    matched_probe = probe[:1024].contiguous()
    del probe
    return construction, matched_probe


def validate_tokenizer_mapping(base_tokenizer, final_tokenizer, base_model, final_model):
    first = base_tokenizer.get_vocab()
    second = final_tokenizer.get_vocab()
    if (not isinstance(first, dict) or not first or first != second or
            len(base_tokenizer) != len(final_tokenizer) or
            base_model.config.vocab_size != final_model.config.vocab_size or
            max(first.values()) >= base_model.config.vocab_size):
        raise ValueError('Historical base/final tokenizer mapping or vocabulary mismatch')
    return len(first)


def load_model_pair(spec, *, model_loader=None, tokenizer_loader=None):
    if model_loader is None:
        model_loader = lambda path: a.load_local_model(path, 'cuda', torch)
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader = lambda path: AutoTokenizer.from_pretrained(path,
                                                                       local_files_only=True)
    base_dir = resolve(spec['paths']['base_directory'])
    final_dir = resolve(spec['paths']['final_directory'])
    start = time.perf_counter()
    base = model_loader(base_dir)
    base_load = time.perf_counter()-start
    start = time.perf_counter()
    final = model_loader(final_dir)
    final_load = time.perf_counter()-start
    base.eval()
    final.eval()
    base.config.use_cache = False
    final.config.use_cache = False
    for parameter in base.parameters():
        parameter.requires_grad_(False)
        if parameter.grad is not None:
            raise ValueError('Historical base already has parameter gradients')
    base_tok = tokenizer_loader(base_dir)
    final_tok = tokenizer_loader(final_dir)
    vocab = validate_tokenizer_mapping(base_tok, final_tok, base, final)
    return base, final, {'base_load_seconds': base_load,
                         'final_load_seconds': final_load, 'vocab_size': vocab}


def soft_teacher_losses(base_logits, final_logits):
    if (base_logits.shape != final_logits.shape or base_logits.ndim != 3 or
            base_logits.dtype != torch.float32 or final_logits.dtype != torch.float32 or
            not bool(torch.isfinite(base_logits).all()) or
            not bool(torch.isfinite(final_logits).all())):
        raise ValueError('Malformed or nonfinite soft-teacher logits')
    p0 = torch.softmax(base_logits, dim=-1)
    logp0 = torch.log_softmax(base_logits, dim=-1)
    logp1 = torch.log_softmax(final_logits, dim=-1)
    ce_sum = -(p0*logp1).sum()
    kl_sum = (p0*(logp0-logp1)).sum()
    if not bool(torch.isfinite(ce_sum)) or not bool(torch.isfinite(kl_sum)):
        raise ValueError('Nonfinite soft-teacher loss')
    return ce_sum, kl_sum


def teacher_gradient_stage(base, final, tokens, eligible, spec, *, progress=None):
    if (tokens.ndim != 2 or tokens.shape[1] != 128 or tokens.shape[0] not in (4, 512) or
            not tokens.is_contiguous() or tokens.dtype != torch.int64 or
            len(eligible) != 98):
        raise ValueError('Teacher-gradient batch/support inventory mismatch')
    batch_size = spec['corpus']['batch_size']
    denominator = tokens.shape[0]*127
    selected = {id(module.weight) for _, module in eligible}
    final.zero_grad(set_to_none=True)
    parts_ce, parts_kl = [], []
    started = time.perf_counter()
    device = eligible[0][1].weight.device
    for start in range(0, tokens.shape[0], batch_size):
        batch = tokens[start:start+batch_size].to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            z0 = base(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
        with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
            z1 = final(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
            ce_sum, kl_sum = soft_teacher_losses(z0, z1)
            (ce_sum/denominator).backward()
        parts_ce.append(float(ce_sum.detach().cpu().double()))
        parts_kl.append(float(kl_sum.detach().cpu().double()))
        for parameter in base.parameters():
            if parameter.grad is not None:
                raise ValueError('Historical base received a parameter gradient')
        for parameter in final.parameters():
            if id(parameter) not in selected and parameter.grad is not None:
                raise ValueError('Noneligible final parameter received gradient')
        if progress is not None:
            progress(start//batch_size+1, tokens.shape[0]//batch_size,
                     time.perf_counter()-started)
        del batch, z0, z1, ce_sum, kl_sum
    for _, module in eligible:
        if module.weight.grad is None or not bool(torch.isfinite(module.weight.grad).all()):
            raise ValueError('Missing or nonfinite positive teacher gradient')
    return {'mean_teacher_cross_entropy': math.fsum(parts_ce)/denominator,
            'mean_kl_base_to_final': math.fsum(parts_kl)/denominator,
            'total_prediction_tokens': denominator}


def base_probe_mean(base, probe, spec, *, progress=None):
    count = probe.shape[0]
    if probe.shape != (count, 128) or count not in (32, 1024):
        raise ValueError('Matched base probe inventory mismatch')
    accumulator = torch.zeros((128, 2048), dtype=torch.float64, device='cpu')
    started = time.perf_counter()
    device = next(base.parameters()).device
    for start in range(0, count, 32):
        batch = probe[start:start+32].to(device)
        with torch.inference_mode():
            readout = a.ordinary_hook_readout(base, batch, spec, torch)
        accumulator.add_(readout.detach().cpu().double().sum(dim=0))
        if progress is not None:
            progress(start//32+1, count//32, time.perf_counter()-started)
        del batch, readout
    return (accumulator/count).contiguous()


def final_probe_jvp(final, probe, tangents, spec, *, progress=None):
    count = probe.shape[0]
    if probe.shape != (count, 128) or count not in (32, 1024):
        raise ValueError('Matched final probe inventory mismatch')
    accumulator = a.ResponseAccumulator(spec, torch)
    audit = None
    started = time.perf_counter()
    device = next(final.parameters()).device
    for start in range(0, count, 32):
        batch = probe[start:start+32].to(device)
        ordinary = a.ordinary_hook_readout(final, batch, spec, torch) if start == 0 else None
        primal, response = a.run_readout_jvp(final, batch, tangents, spec, torch)
        if ordinary is not None:
            audit = a.compare_primal(primal, ordinary, spec, torch)
        accumulator.add(primal, response)
        if progress is not None:
            progress(start//32+1, count//32, time.perf_counter()-started)
        del batch, ordinary, primal, response
    merged, tangent = accumulator.means(count)
    return merged.contiguous(), tangent.float().contiguous(), audit


def response_metrics(response, target):
    if (response.shape != (128, 2048) or target.shape != (128, 2048) or
            response.dtype != torch.float32 or target.dtype != torch.float32 or
            not bool(torch.isfinite(response).all()) or
            not bool(torch.isfinite(target).all())):
        raise ValueError('Malformed matched response/target')
    x, y = response.double(), target.double()
    nx = torch.linalg.vector_norm(x, dim=1)
    ny = torch.linalg.vector_norm(y, dim=1)
    if bool((nx <= 0).any()) or bool((ny <= 0).any()):
        raise ValueError('Zero-norm matched response/target position')
    cosines = ((x*y).sum(dim=1)/(nx*ny)).tolist()
    xx = float(x.square().sum())
    yy = float(y.square().sum())
    scalar = float((x*y).sum())/xx
    residual = float(torch.linalg.vector_norm(scalar*x-y))/math.sqrt(yy)
    return {'position_cosines': cosines,
            'position_0_cosine': cosines[0],
            'positions_1_4_mean_cosine': math.fsum(cosines[1:5])/4,
            'positions_1_127_mean_cosine': math.fsum(cosines[1:128])/127,
            'teacher_response_pooled_rms_position_norm': math.sqrt(xx/128),
            'oracle_difference_pooled_rms_position_norm': math.sqrt(yy/128),
            'best_scalar_rescaling_diagnostic_only': scalar,
            'relative_residual_after_best_rescaling_diagnostic_only': residual}


def matched_difference(final_mean, base_mean):
    if (final_mean.shape != (128, 2048) or base_mean.shape != (128, 2048) or
            final_mean.dtype != torch.float64 or base_mean.dtype != torch.float64 or
            not bool(torch.isfinite(final_mean).all()) or
            not bool(torch.isfinite(base_mean).all())):
        raise ValueError('Malformed matched base/final probe means')
    return (final_mean-base_mean).float().contiguous()


def progress_printer(stage):
    def report(done, total, elapsed):
        eta = elapsed*(total-done)/done
        print(f'{stage} | {done}/{total} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s', flush=True)
    return report


def runtime_projection(loads, gradient_seconds, base_probe_seconds, jvp_seconds):
    teacher = gradient_seconds*128
    probe = (base_probe_seconds+jvp_seconds)*32
    return {'projected_full_teacher_gradient_seconds': teacher,
            'projected_full_probe_and_jvp_seconds': probe,
            'projected_total_compute_seconds': teacher+probe,
            'projected_total_wall_seconds':
                loads['base_load_seconds']+loads['final_load_seconds']+teacher+probe}


def run(*, smoke_only=False, model_loader=None, tokenizer_loader=None):
    spec = load_spec()
    inventories = validate_sources(spec, smoke_only=smoke_only)
    construction, probe = load_frozen_tokens(spec)
    if smoke_only:
        construction, probe = construction[:4].contiguous(), probe[:32].contiguous()
    base, final, loads = load_model_pair(spec, model_loader=model_loader,
                                         tokenizer_loader=tokenizer_loader)
    eligible = a.discover_eligible_linear_weights(final, torch)
    a.freeze_other_parameters(final, eligible)
    base_before = a.model_state_hashes(base, torch)
    final_before = a.model_state_hashes(final, torch)
    gradient_started = time.perf_counter()
    teacher = teacher_gradient_stage(base, final, construction, eligible, spec,
                                      progress=None if smoke_only else progress_printer('teacher gradient'))
    gradient_seconds = time.perf_counter()-gradient_started
    scale = a.global_tangent_scale(eligible, 0.00125, torch)
    tangents, tangent_records, realized = a.prepare_tangents(eligible, scale, torch)
    if len(tangents) != 98:
        raise ValueError('Teacher tangent support is not exactly 98 matrices')
    base_probe_started = time.perf_counter()
    base_mean = base_probe_mean(base, probe, spec,
                                progress=None if smoke_only else progress_printer('base probe mean'))
    base_probe_seconds = time.perf_counter()-base_probe_started
    a.verify_model_unchanged(base, base_before, torch)
    if any(parameter.grad is not None for parameter in base.parameters()):
        raise ValueError('Historical base received parameter gradients')
    del base, base_before
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    jvp_started = time.perf_counter()
    final_mean, response, audit = final_probe_jvp(final, probe, tangents, spec,
                                                  progress=None if smoke_only else progress_printer('final JVP'))
    jvp_seconds = time.perf_counter()-jvp_started
    a.verify_model_unchanged(final, final_before, torch)
    del final
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    target = matched_difference(final_mean, base_mean)
    geometry = response_metrics(response, target)
    if smoke_only:
        return {'smoke_only': True, 'model_load_seconds': loads,
                'teacher_gradient_batch_seconds': gradient_seconds,
                'base_probe_batch_seconds': base_probe_seconds,
                'final_jvp_probe_batch_seconds': jvp_seconds,
                **runtime_projection(loads, gradient_seconds, base_probe_seconds, jvp_seconds),
                'writes': False}
    for directory, expected in ((resolve(spec['paths']['base_directory']), inventories['base_files']),
                                (resolve(spec['paths']['final_directory']), inventories['final_files'])):
        if a.checkpoint_file_records(directory) != expected:
            raise ValueError('Checkpoint files changed during Attempt103')
    result = {'format_version': 1, 'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
              'privileged_mechanism_diagnostic_not_blind_recovery': True,
              'candidate_selection': False, 'oracle_based_tuning': False,
              'models': {'base': {'repo_id': spec['base']['repo_id'],
                                  'revision': spec['base']['revision']},
                         'final': spec['paths']['final_directory'],
                         'vocab_size': loads['vocab_size']},
              'provenance': {'spec_sha256': SPEC_SHA256,
                             'source_sha256': a.sha256_file(Path(__file__).resolve()),
                             'base_checkpoint_files': inventories['base_files'],
                             'final_checkpoint_files': inventories['final_files'],
                             'fineweb_tokens_sha256': spec['corpus']['serialized_sha256'],
                             'probe_tokens_sha256': spec['probe']['serialized_sha256'],
                             'teacher_response_raw_sha256': a.sha256_raw_float32_tensor(response, torch),
                             'matched_difference_raw_sha256': a.sha256_raw_float32_tensor(target, torch)},
              'construction': {'sequence_count': int(construction.shape[0]),
                               'batch_size': 4, **teacher,
                               'teacher_gradient_global_norm': scale['aggregate_generic_gradient_norm'],
                               'alpha': scale['alpha'],
                               'realized_relative_tangent_norm':
                                   realized['aggregate_realized_relative_tangent_norm'],
                               'tangent_matrix_count': len(tangent_records)},
              'matched_probe': {'sequence_count': int(probe.shape[0]), 'batch_size': 32,
                                'jvp_first_batch_primal_max_abs_difference': audit,
                                **geometry},
              'context_landmarks': spec['context_landmarks'],
              'historical_adapter_access': False, 'oracle_adl_access': False,
              'no_oracle_based_tuning': True}
    output = resolve(spec['paths']['result_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Attempt103 result already exists')
    a.write_manifest(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(smoke_only=args.smoke_only), allow_nan=False))


if __name__ == '__main__':
    main()
