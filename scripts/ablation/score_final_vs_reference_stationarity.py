#!/usr/bin/env python3
"""Blind final-Qwen byte-NLL deviation from frozen external references."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import re
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '102_final_vs_reference_relative_stationarity_pilot'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = 'c260a8b88444641d9c716efb0c44e0ff93a9430f53b2d1633fdc2ad110bdf279'
PRIOR_SOURCE = Path(__file__).with_name('score_pretrained_reference_stationarity.py')
PRIOR_SOURCE_SHA256 = '4b718fb80cf2c7e50bc6ed06a7df64a8d6a34ba2b74ca804eab65ffcc9f170dd'
if hashlib.sha256(PRIOR_SOURCE.read_bytes()).hexdigest() != PRIOR_SOURCE_SHA256:
    raise ValueError('Frozen Attempt101 scoring source changed')
loader = importlib.util.spec_from_file_location('attempt102_frozen_attempt101_helpers', PRIOR_SOURCE)
prior = importlib.util.module_from_spec(loader)
loader.loader.exec_module(prior)
ORDER = prior.ORDER
MODELS = prior.MODELS
FORBIDDEN = ('models/base', 'historical_base', 'base_checkpoint', 'adapter',
             'oracle', 'base_mean', 'ft_mean', 'true_delta', 'evaluation.json',
             'known_base')


def safe_path(value):
    string = str(value).lower().replace('\\', '/')
    if (any(item in string for item in FORBIDDEN) or
            any(int(number) >= 15 and int(number) not in (100, 101, 102)
                for number in re.findall(r'attempt(\d{3})', string))):
        raise ValueError('Forbidden Attempt102 input path')
    return Path(value)


def path_of(value):
    path = safe_path(value)
    return path if path.is_absolute() else PROJECT/path


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt102 spec path differs from frozen path')
    prior.require_hash(path, SPEC_SHA256)
    spec = prior.old.load_json_object(path, 'Attempt102 spec')
    if (spec.get('attempt_id') != ATTEMPT or spec.get('corpus_order') != list(ORDER) or
            spec.get('frozen_sequence_indices') != list(range(128)) or
            spec.get('canonical_final', {}).get('batch_size') != 16 or
            spec.get('canonical_final', {}).get('dtype') != 'torch.float32' or
            spec.get('canonical_final', {}).get('add_special_tokens') is not False or
            spec.get('statistics', {}).get('primary_deviation') != 'z_final_minus_z_ref' or
            spec.get('statistics', {}).get('primary_input') != 'nats_per_utf8_byte' or
            spec.get('smoke', {}).get('first_decoded_sequences') != 16 or
            any(value is not False for value in spec.get('firewall', {}).values())):
        raise ValueError('Attempt102 frozen inventory/semantics mismatch')
    for key in ('spec_path', 'source_path', 'result_path', 'reference_lock_path',
                'decoded_inventory_path'):
        path_of(spec['attempt101'][key])
    for key in ('directory',):
        path_of(spec['canonical_final'][key])
    for key in ('checkpoint_directory', 'result_path'):
        path_of(spec[key])
    return spec


def load_attempt101(spec):
    row = spec['attempt101']
    for key in ('spec', 'source', 'result', 'reference_lock', 'decoded_inventory'):
        prior.require_hash(path_of(row[key+'_path']), row[key+'_sha256'])
    if row['spec_sha256'] != prior.SPEC_SHA256 or row['source_sha256'] != PRIOR_SOURCE_SHA256:
        raise ValueError('Attempt101 source/spec lock mismatch')
    old_spec = prior.load_spec(path_of(row['spec_path']))
    prior.validate_metadata(old_spec)
    lock = prior.validate_lock(prior.old.load_json_object(
        path_of(row['reference_lock_path']), 'Attempt101 reference lock'), old_spec)
    result = prior.old.load_json_object(path_of(row['result_path']), 'Attempt101 blind result')
    provenance = result.get('provenance', {})
    expected = {'spec_sha256': row['spec_sha256'],
                'scoring_source_sha256': row['source_sha256'],
                'reference_model_lock_sha256': row['reference_lock_sha256'],
                'decoded_text_inventory_sha256': row['decoded_inventory_sha256'],
                'attempt100_construction_manifest_sha256': spec['attempt100']['manifest_sha256'],
                'attempt100_construction_artifact_sha256': spec['attempt100']['artifact_serialized_sha256'],
                'corpus_token_artifact_sha256': {
                    item['name']: item['serialized_sha256'] for item in old_spec['corpora']}}
    if (result.get('attempt_id') != prior.ATTEMPT or
            result.get('format_version') != 1 or
            any(provenance.get(key) != value for key, value in expected.items()) or
            result.get('reference_models') != lock['models'] or
            any(result.get(key) is not False for key in old_spec['firewall']) or
            result.get('statistics', {}).get('primary_stationarity_input') != 'nats_per_utf8_byte' or
            set(result.get('cells', {})) != {name for name, _ in MODELS} or
            [item.get('name') for item in result['statistics'].get('corpora', [])] != list(ORDER)):
        raise ValueError('Attempt101 blind provenance/firewall mismatch')
    for model in lock['models']:
        cells = result['cells'][model['name']]
        if set(cells) != set(ORDER):
            raise ValueError('Attempt101 reference-cell inventory mismatch')
        for corpus in ORDER:
            cell = cells[corpus]
            metadata = prior.checkpoint_metadata(old_spec, model, corpus,
                row['decoded_inventory_sha256'], row['source_sha256'], cell['total_utf8_bytes'])
            prior.validate_checkpoint(prior.make_checkpoint(metadata, cell), metadata)
    for i, corpus in enumerate(ORDER):
        values = [result['cells'][model][corpus]['nats_per_utf8_byte']
                  for model, _ in MODELS]
        if any(not math.isfinite(value) for value in values):
            raise ValueError('Nonfinite Attempt101 reference byte-NLL')
        row_stats = result['statistics']['corpora'][i]
        if any(row_stats['reference_scores'][model]['nats_per_utf8_byte'] != value
               for (model, _), value in zip(MODELS, values)):
            raise ValueError('Attempt101 reference statistics mismatch')
    reference_z = {model: population_z([
        result['cells'][model][name]['nats_per_utf8_byte'] for name in ORDER])
        for model, _ in MODELS}
    for i, row_stats in enumerate(result['statistics']['corpora']):
        expected_score = -math.fsum(reference_z[model][i] for model, _ in MODELS)/2
        if not math.isclose(row_stats['stationarity_score'], expected_score,
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError('Attempt101 stationarity score mismatch')
    return old_spec, result


def reconstruct_decoded_inventory(spec, old_spec, *, tokenizer_loader=None):
    root = prior.validate_tokenizer_files(old_spec)
    if tokenizer_loader is None:
        from transformers import AutoTokenizer
        tokenizer_loader = lambda directory: AutoTokenizer.from_pretrained(
            directory, local_files_only=True)
    tokenizer = tokenizer_loader(root)
    tokens = prior.load_first_sequences(old_spec)
    inventory = prior.decode_inventory(tokens, tokenizer, old_spec)
    del tokens, tokenizer
    expected = spec['attempt101']['decoded_inventory_sha256']
    if prior.sha256_bytes(prior.canonical_bytes(inventory)) != expected:
        raise ValueError('Reconstructed decoded-text inventory SHA256 mismatch')
    stored = prior.old.load_json_object(path_of(spec['attempt101']['decoded_inventory_path']),
                                        'Attempt101 decoded-text inventory')
    if stored != inventory:
        raise ValueError('Stored decoded-text inventory differs from reconstruction')
    return inventory


def checkpoint_inventory(spec):
    final = spec['canonical_final']
    records = prior.old.checkpoint_file_records(path_of(final['directory']))
    if records != final['checkpoint_files']:
        raise ValueError('Canonical merged final-checkpoint inventory mismatch')
    return prior.sha256_bytes(prior.canonical_bytes(records))


def load_final_model(spec):
    final = spec['canonical_final']
    model = prior.old.load_local_model(path_of(final['directory']), 'cuda', torch)
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        if parameter.dtype != torch.float32 or parameter.grad is not None:
            raise ValueError('Final model must be FP32 with no parameter gradients')
        parameter.requires_grad_(False)
    return model


def score_final_sequences(model, tokenizer, texts, *, batch_size, progress=None):
    if (batch_size != 16 or len(texts) not in (16, 128) or model.training or
            any(parameter.dtype != torch.float32 or parameter.requires_grad or
                parameter.grad is not None for parameter in model.parameters())):
        raise ValueError('Final scoring requires frozen FP32 eval model and fixed inventory')
    total_utf8_bytes = prior.utf8_byte_count(texts)
    encoded = prior.native_ids(texts, tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError('Canonical tokenizer lacks pad and EOS IDs')
    device = next(model.parameters()).device
    losses, predicted = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            ids, mask = prior.padded_batch(encoded[start:start+batch_size], pad_id, device)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
            loss, count = prior.causal_ce_sum(logits, ids, mask)
            losses.append(loss)
            predicted.append(count)
            if progress is not None:
                progress({'completed_sequences': min(start+batch_size, len(encoded)),
                          'total_sequences': len(encoded),
                          'elapsed_seconds': time.perf_counter()-started})
            del ids, mask, logits
    total_nll = math.fsum(losses)
    total_native_tokens = sum(map(len, encoded))
    predicted_native_tokens = sum(predicted)
    nats_per_byte = total_nll/total_utf8_bytes
    metrics = {'total_nll_nats': total_nll,
               'total_utf8_bytes': total_utf8_bytes,
               'nats_per_utf8_byte': nats_per_byte,
               'bits_per_utf8_byte': nats_per_byte/math.log(2),
               'mean_nll_per_predicted_native_token': total_nll/predicted_native_tokens,
               'sequence_count': len(encoded),
               'total_native_tokens': total_native_tokens,
               'predicted_native_tokens': predicted_native_tokens,
               'mean_sequence_token_count': total_native_tokens/len(encoded),
               'elapsed_inference_seconds': time.perf_counter()-started}
    if (predicted_native_tokens != total_native_tokens-len(encoded) or
            any(not math.isfinite(value) for value in metrics.values())):
        raise ValueError('Final loss/token accounting failed')
    return metrics


def checkpoint_metadata(spec, corpus, decoded_sha, checkpoint_sha, source_sha,
                        total_utf8_bytes):
    if corpus not in ORDER or type(total_utf8_bytes) is not int or total_utf8_bytes <= 0:
        raise ValueError('Invalid frozen final checkpoint cell identity')
    return {'format_version': 1, 'attempt_id': ATTEMPT,
            'spec_sha256': SPEC_SHA256, 'final_checkpoint_inventory_sha256': checkpoint_sha,
            'decoded_inventory_sha256': decoded_sha, 'corpus': corpus,
            'scoring_source_sha256': source_sha, 'total_utf8_bytes': total_utf8_bytes,
            'scoring_rule': spec['canonical_final']}


def validate_checkpoint(record, metadata):
    if (not isinstance(record, dict) or
            set(record) != set(metadata) | {'result', 'result_sha256'} or
            any(record.get(key) != value for key, value in metadata.items())):
        raise ValueError('Final checkpoint provenance mismatch')
    result = record['result']
    if (not isinstance(result, dict) or
            record['result_sha256'] != prior.sha256_bytes(prior.canonical_bytes(result)) or
            set(result) != {'total_nll_nats', 'total_utf8_bytes', 'nats_per_utf8_byte',
                            'bits_per_utf8_byte', 'mean_nll_per_predicted_native_token',
                            'sequence_count', 'total_native_tokens', 'predicted_native_tokens',
                            'mean_sequence_token_count', 'elapsed_inference_seconds'} or
            result['sequence_count'] != 128 or
            type(result['total_utf8_bytes']) is not int or
            result['total_utf8_bytes'] != metadata['total_utf8_bytes'] or
            type(result['total_native_tokens']) is not int or
            type(result['predicted_native_tokens']) is not int or
            result['predicted_native_tokens'] != result['total_native_tokens']-128 or
            result['predicted_native_tokens'] <= 0 or
            any(not isinstance(result[key], (int, float)) or not math.isfinite(result[key])
                for key in ('total_nll_nats', 'nats_per_utf8_byte', 'bits_per_utf8_byte',
                            'mean_nll_per_predicted_native_token', 'mean_sequence_token_count',
                            'elapsed_inference_seconds')) or
            result['total_nll_nats'] < 0 or result['elapsed_inference_seconds'] < 0 or
            result['nats_per_utf8_byte'] != result['total_nll_nats']/result['total_utf8_bytes'] or
            result['bits_per_utf8_byte'] != result['nats_per_utf8_byte']/math.log(2) or
            result['mean_nll_per_predicted_native_token'] !=
                result['total_nll_nats']/result['predicted_native_tokens'] or
            result['mean_sequence_token_count'] != result['total_native_tokens']/128):
        raise ValueError('Final checkpoint result/hash mismatch')
    return result


def load_or_store_cell(path, metadata, result=None):
    if path.exists() or path.is_symlink():
        return validate_checkpoint(prior.old.load_json_object(path, 'Attempt102 cell'), metadata)
    if result is None:
        return None
    record = {**metadata, 'result': result,
              'result_sha256': prior.sha256_bytes(prior.canonical_bytes(result))}
    prior.atomic_new_json(path, record)
    return validate_checkpoint(prior.old.load_json_object(path, 'Attempt102 cell'), metadata)


def population_z(values):
    if len(values) != 8 or any(not math.isfinite(x) for x in values):
        raise ValueError('Population z requires eight finite corpus values')
    mean = math.fsum(values)/8
    std = math.sqrt(math.fsum((x-mean)**2 for x in values)/8)
    if not math.isfinite(std) or std <= 0:
        raise ValueError('Zero/nonfinite population standard deviation')
    return [(x-mean)/std for x in values]


def relative_statistics(final_cells, reference_result, pc1_raw, explained):
    reference_cells = reference_result['cells']
    ref_z = {model: population_z([reference_cells[model][name]['nats_per_utf8_byte']
                                  for name in ORDER]) for model, _ in MODELS}
    z_ref = [math.fsum(ref_z[model][i] for model, _ in MODELS)/2 for i in range(8)]
    final_byte = [final_cells[name]['nats_per_utf8_byte'] for name in ORDER]
    z_final = population_z(final_byte)
    deviation = [a-b for a, b in zip(z_final, z_ref)]
    design = torch.tensor([[1.0, x] for x in z_ref], dtype=torch.float64)
    target = torch.tensor(z_final, dtype=torch.float64)[:, None]
    fit = torch.linalg.lstsq(design, target, rcond=1e-12, driver='gelsd')
    if int(fit.rank) != 2:
        raise ValueError('Reference OLS design is rank deficient')
    intercept, slope = fit.solution[:, 0].tolist()
    residual = (target[:, 0]-design@fit.solution[:, 0]).tolist()
    if len(pc1_raw) != 8 or not 0 <= explained <= 1:
        raise ValueError('Blind PC1 inventory mismatch')
    rows = []
    deviation_ranks = prior.average_ranks(deviation)
    for i, name in enumerate(ORDER):
        rows.append({'corpus': name, 'final_model': final_cells[name],
                     'z_final': z_final[i], 'z_ref': z_ref[i],
                     'deviation': deviation[i], 'deviation_rank': deviation_ranks[i],
                     'ols_residual': residual[i], 'pc1_coordinate': pc1_raw[i],
                     'attempt101_stationarity_score':
                         reference_result['statistics']['corpora'][i]['stationarity_score']})
    return {'primary_metric': 'absolute_spearman_simple_deviation_vs_blind_pc1',
            'primary_absolute_spearman': abs(prior.spearman(deviation, pc1_raw)),
            'secondary_absolute_pearson': abs(prior.pearson(deviation, pc1_raw)),
            'pc1_coordinates_raw': list(pc1_raw), 'pc1_variance_explained': explained,
            'reference_model_z': ref_z, 'reference_ensemble_z': z_ref,
            'final_z': z_final, 'deviation': deviation,
            'regression_residual_robustness': {
                'intercept': intercept, 'slope': slope, 'residuals': residual,
                'absolute_spearman_vs_pc1': abs(prior.spearman(residual, pc1_raw)),
                'absolute_pearson_vs_pc1': abs(prior.pearson(residual, pc1_raw))},
            'corpora': rows, 'candidate_constructed': False, 'oracle_evaluation': False}


def run(*, smoke_only=False, model_loader=None, tokenizer_loader=None):
    spec = load_spec()
    output = path_of(spec['result_path'])
    if not smoke_only and (output.exists() or output.is_symlink()):
        raise ValueError('Attempt102 result already exists')
    old_spec, reference_result = load_attempt101(spec)
    inventory = reconstruct_decoded_inventory(spec, old_spec,
                                               tokenizer_loader=tokenizer_loader)
    decoded_sha = spec['attempt101']['decoded_inventory_sha256']
    checkpoint_sha = checkpoint_inventory(spec)
    source_sha = prior.old.sha256_file(Path(__file__).resolve())
    if smoke_only:
        texts = inventory['texts']['fineweb'][:16]
        load_started = time.perf_counter()
        model = model_loader(spec) if model_loader else load_final_model(spec)
        load_seconds = time.perf_counter()-load_started
        try:
            root = path_of(old_spec['qwen_tokenizer']['directory'])
            if tokenizer_loader is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
            else:
                tokenizer = tokenizer_loader(root)
            started = time.perf_counter()
            cell = score_final_sequences(model, tokenizer, texts, batch_size=16)
            inference = time.perf_counter()-started
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        projected = inference*64
        return {'smoke_only': True, 'model_load_seconds': load_seconds,
                'batch_inference_seconds': inference,
                'tokens_per_second': cell['predicted_native_tokens']/inference,
                'projected_full_64_batch_scoring_seconds': projected,
                'projected_full_wall_seconds': load_seconds+projected,
                'writes': False}
    cells = {}
    pending = []
    for corpus in ORDER:
        byte_count = prior.utf8_byte_count(inventory['texts'][corpus])
        metadata = checkpoint_metadata(spec, corpus, decoded_sha, checkpoint_sha,
                                       source_sha, byte_count)
        path = path_of(spec['checkpoint_directory'])/(corpus+'.json')
        cached = load_or_store_cell(path, metadata)
        if cached is None:
            pending.append((corpus, path, metadata))
        else:
            cells[corpus] = cached
            print(f'corpus {ORDER.index(corpus)+1}/8 | checkpoint reused', flush=True)
    if pending:
        load_started = time.perf_counter()
        model = model_loader(spec) if model_loader else load_final_model(spec)
        print(f'final model loaded | elapsed {time.perf_counter()-load_started:.1f}s', flush=True)
        try:
            root = path_of(old_spec['qwen_tokenizer']['directory'])
            if tokenizer_loader is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
            else:
                tokenizer = tokenizer_loader(root)
            scoring_started = time.perf_counter()
            total_pending_batches = 8*len(pending)
            for pending_index, (corpus, path, metadata) in enumerate(pending):
                corpus_index = ORDER.index(corpus)+1
                def progress(value):
                    batches_done = (value['completed_sequences']+15)//16
                    done = pending_index*8+batches_done
                    elapsed = time.perf_counter()-scoring_started
                    eta = elapsed*(total_pending_batches-done)/done if done else float('inf')
                    print(f'corpus {corpus_index}/8 | batch {batches_done}/8 | '
                          f'elapsed {elapsed:.1f}s | ETA {eta:.1f}s', flush=True)
                value = score_final_sequences(model, tokenizer, inventory['texts'][corpus],
                                              batch_size=16, progress=progress)
                cells[corpus] = load_or_store_cell(path, metadata, value)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    pc1_raw, explained = prior.pc1_coordinates(prior.load_attempt100_pc1_inputs(old_spec))
    statistics = relative_statistics(cells, reference_result, pc1_raw, explained)
    for key in ('spec', 'source', 'result', 'reference_lock', 'decoded_inventory'):
        prior.require_hash(path_of(spec['attempt101'][key+'_path']),
                           spec['attempt101'][key+'_sha256'])
    prior.require_hash(SPEC_PATH, SPEC_SHA256)
    result = {'format_version': 1, 'attempt_id': ATTEMPT,
              'purpose': spec['purpose'], 'corpus_order': list(ORDER),
              'provenance': {'spec_sha256': SPEC_SHA256, 'scoring_source_sha256': source_sha,
                             'final_checkpoint_inventory_sha256': checkpoint_sha,
                             'decoded_inventory_sha256': decoded_sha,
                             'attempt101_result_sha256': spec['attempt101']['result_sha256'],
                             'attempt100_construction_artifact_sha256':
                                 spec['attempt100']['artifact_serialized_sha256']},
              'cells': cells, 'statistics': statistics, **spec['firewall']}
    prior.atomic_new_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(smoke_only=args.smoke_only), allow_nan=False))


if __name__ == '__main__':
    main()
