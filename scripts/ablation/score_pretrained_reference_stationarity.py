#!/usr/bin/env python3
"""Blind external-reference stationarity diagnostic for frozen Attempt 100."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '101_pretrained_reference_stationarity_pilot'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = 'f8cb0301bcf01c4c203857a449c7bd3f7770c5a420af608a8cad65d082d1d045'
OLD_SOURCE = Path(__file__).with_name('construct_multicorpus_raw_jg_consensus.py')
OLD_SOURCE_SHA256 = '59936b928dcf942470db92debb1f2a8e41823ae1c9082803d2fdd8dc50202caa'
if hashlib.sha256(OLD_SOURCE.read_bytes()).hexdigest() != OLD_SOURCE_SHA256:
    raise ValueError('Frozen generic-corpus validation source changed')
loader = importlib.util.spec_from_file_location('attempt101_frozen_corpus_helpers', OLD_SOURCE)
old = importlib.util.module_from_spec(loader)
loader.loader.exec_module(old)

ORDER = ('fineweb', 'wikitext103_raw', 'tinystories', 'arxiv_document',
         'cc_news', 'pg19', 'codeparrot_clean', 'ultrachat')
MODELS = (('smollm2_1_7b', 'HuggingFaceTB/SmolLM2-1.7B'),
          ('pythia_1_4b', 'EleutherAI/pythia-1.4b'))
FORBIDDEN = ('models/base', 'historical_base', 'base_checkpoint', 'adapter',
             'oracle', 'base_mean', 'ft_mean', 'true_delta',
             'evaluation.json', 'known_base')


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')


def safe_path(value):
    text = str(value).lower().replace('\\', '/')
    privileged_attempt = any(int(number) >= 15 and int(number) not in (100, 101)
                             for number in re.findall(r'attempt(\d{3})', text))
    if any(word in text for word in FORBIDDEN) or privileged_attempt:
        raise ValueError('Forbidden Attempt101 input path')
    return Path(value)


def path_of(value):
    path = safe_path(value)
    return path if path.is_absolute() else PROJECT/path


def require_hash(path, expected):
    if old.sha256_file(path) != old.require_sha256(expected, str(path)):
        raise ValueError('Frozen input SHA256 mismatch: '+str(path))


def load_spec(path=SPEC_PATH):
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt101 spec path differs from frozen path')
    require_hash(path, SPEC_SHA256)
    spec = old.load_json_object(path, 'Attempt101 spec')
    if (spec.get('attempt_id') != ATTEMPT or
            [(row.get('name'), row.get('repo_id')) for row in spec.get('reference_models', [])] != list(MODELS) or
            any(row.get('variant') != 'base' or 'instruct' in row.get('repo_id', '').lower()
                for row in spec['reference_models']) or
            spec.get('corpus_order') != list(ORDER) or
            spec.get('frozen_sequence_indices') != list(range(128)) or
            spec.get('scoring', {}).get('reference_batch_size') != 16 or
            spec.get('smoke', {}).get('fineweb_sequences') != 8):
        raise ValueError('Attempt101 frozen inventory/semantics mismatch')
    for row in spec['corpora']:
        for key in ('tokens_path', 'manifest_path'):
            safe_path(row[key])
    for key in ('decoded_inventory_path', 'checkpoint_directory', 'result_path'):
        safe_path(spec[key])
    return spec


def require_revision(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{40}', value) is None:
        raise ValueError('Reference model revision must be a full immutable 40-character SHA')
    return value


def make_lock(spec, resolver):
    rows = []
    for row in spec['reference_models']:
        rows.append({'name': row['name'], 'repo_id': row['repo_id'],
                     'revision': require_revision(resolver(row['repo_id']))})
    return {'format_version': 1, 'attempt_id': ATTEMPT,
            'spec_sha256': SPEC_SHA256,
            'resolution': 'one_time_model_repo_HEAD', 'models': rows}


def validate_lock(lock, spec):
    if (not isinstance(lock, dict) or lock.get('format_version') != 1 or
            lock.get('attempt_id') != ATTEMPT or lock.get('spec_sha256') != SPEC_SHA256 or
            lock.get('resolution') != 'one_time_model_repo_HEAD' or
            not isinstance(lock.get('models'), list) or len(lock['models']) != 2):
        raise ValueError('Reference model lock provenance/inventory mismatch')
    for row, expected in zip(lock['models'], spec['reference_models']):
        if (not isinstance(row, dict) or set(row) != {'name', 'repo_id', 'revision'} or
                row['name'] != expected['name'] or row['repo_id'] != expected['repo_id'] or
                expected['variant'] != 'base' or 'instruct' in row['repo_id'].lower()):
            raise ValueError('Reference lock model identity/variant mismatch')
        require_revision(row['revision'])
    return lock


def atomic_new_json(path, value):
    """Durable same-directory temp plus atomic no-replace hard link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError('Output already exists: '+str(path))
    temporary = path.parent/(path.name+'.tmp.'+uuid.uuid4().hex)
    data = canonical_bytes(value)
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise ValueError('Output already exists: '+str(path)) from exc
    finally:
        temporary.unlink(missing_ok=True)


def get_lock(spec, *, smoke_only, resolver=None):
    path = path_of(spec['reference_lock_path'])
    if path.exists():
        lock = validate_lock(old.load_json_object(path, 'reference model lock'), spec)
        return lock, old.sha256_file(path)
    if resolver is None:
        from huggingface_hub import HfApi
        api = HfApi()
        resolver = lambda repo: api.model_info(repo).sha
    lock = make_lock(spec, resolver)
    if smoke_only:
        return lock, sha256_bytes(canonical_bytes(lock))
    atomic_new_json(path, lock)
    return lock, old.sha256_file(path)


def validate_metadata(spec):
    provenance = spec['attempt100']
    for key in ('spec', 'manifest', 'corpus_lock', 'constructor', 'freezer'):
        require_hash(path_of(provenance[key+'_path']), provenance[key+'_sha256'])
    attempt100_spec = old.load_json_object(path_of(provenance['spec_path']), 'Attempt100 spec')
    attempt100_manifest = old.load_json_object(path_of(provenance['manifest_path']), 'Attempt100 construction manifest')
    attempt100_lock = old.load_json_object(path_of(provenance['corpus_lock_path']), 'Attempt100 corpus lock')
    if (attempt100_spec.get('attempt_id') != '100_diverse8_raw_jg_consensus_prefix0_13' or
            attempt100_manifest.get('spec_sha256') != provenance['spec_sha256'] or
            attempt100_manifest.get('constructor_sha256') != provenance['constructor_sha256'] or
            attempt100_manifest.get('freezer_sha256') != provenance['freezer_sha256'] or
            attempt100_manifest.get('corpus_lock_sha256') != provenance['corpus_lock_sha256'] or
            attempt100_manifest.get('corpus_order') != list(ORDER) or
            attempt100_manifest.get('artifact', {}).get('serialized_sha256') != provenance['artifact_serialized_sha256'] or
            attempt100_manifest.get('artifact', {}).get('raw_tensors_sha256') != provenance['artifact_raw_sha256'] or
            attempt100_lock.get('spec_sha256') != provenance['spec_sha256']):
        raise ValueError('Attempt100 blind construction provenance mismatch')
    prior = spec['attempt014']
    for key in ('spec', 'manifest', 'constructor'):
        require_hash(path_of(prior[key+'_path']), prior[key+'_sha256'])
    old_spec = old.load_json_object(path_of(prior['spec_path']), 'Attempt014 spec')
    old_manifest = old.load_json_object(path_of(prior['manifest_path']), 'Attempt014 construction manifest')
    if (old_spec != old.FROZEN_SPEC or old_manifest.get('spec_sha256') != prior['spec_sha256'] or
            old_manifest.get('constructor_script_sha256') != prior['constructor_sha256'] or
            old_manifest.get('corpus_order') != list(ORDER[:3])):
        raise ValueError('Predecessor frozen corpus provenance mismatch')
    if [row['name'] for row in spec['corpora']] != list(ORDER):
        raise ValueError('Eight-corpus order changed')
    for i, row in enumerate(spec['corpora']):
        require_hash(path_of(row['manifest_path']), row['manifest_sha256'])
        if row['source'] == 'frozen_attempt014_or_attempt005':
            require_hash(path_of(row['corpus_spec_path']), row['corpus_spec_sha256'])
            predecessor = old_manifest['corpora'][i]
            if any(predecessor.get(key) != row[expected] for key, expected in
                   (('spec_sha256', 'corpus_spec_sha256'), ('manifest_sha256', 'manifest_sha256'),
                    ('serialized_sha256', 'serialized_sha256'), ('raw_tensor_sha256', 'raw_sha256'))):
                raise ValueError('Old corpus source hash mismatch: '+row['name'])
        elif row['source'] == 'frozen_attempt100':
            new = attempt100_manifest['new_corpora'][i-3]
            if (new.get('name') != row['name'] or
                    new.get('corpus_manifest_sha256') != row['manifest_sha256'] or
                    new.get('corpus_artifact_sha256') != row['serialized_sha256'] or
                    new.get('corpus_revision') != row['revision'] or
                    attempt100_lock['corpora'][i-3]['revision'] != row['revision']):
                raise ValueError('New corpus source hash/revision mismatch: '+row['name'])
        else:
            raise ValueError('Unknown frozen corpus source')
        manifest = old.load_json_object(path_of(row['manifest_path']), row['name']+' corpus manifest')
        artifact = manifest.get('artifact', {})
        if (artifact.get('serialized_sha256') != row['serialized_sha256'] or
                artifact.get('raw_tensor_sha256') != row['raw_sha256'] or
                artifact.get('path') != row['tokens_path']):
            raise ValueError('Frozen corpus manifest artifact mismatch: '+row['name'])
    return attempt100_manifest


def load_first_sequences(spec):
    rows = {}
    for corpus in spec['corpora']:
        path = path_of(corpus['tokens_path'])
        require_hash(path, corpus['serialized_sha256'])
        tokens = old.load_corpus_tokens(path, corpus['raw_sha256'], 4096, 128, torch)
        rows[corpus['name']] = tokens[:128].contiguous()
        del tokens
    return rows


def validate_tokenizer_files(spec):
    root = path_of(spec['qwen_tokenizer']['directory'])
    expected = spec['qwen_tokenizer']['files']
    for record in expected:
        path = root/record['path']
        if path.stat().st_size != record['size_bytes']:
            raise ValueError('Merged tokenizer file size mismatch: '+record['path'])
        require_hash(path, record['sha256'])
    return root


def decode_inventory(tokens_by_corpus, tokenizer, spec):
    if list(tokens_by_corpus) != list(ORDER):
        raise ValueError('Decoded inventory corpus order mismatch')
    texts = {}
    for name in ORDER:
        tokens = tokens_by_corpus[name]
        if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64 or
                tokens.device.type != 'cpu' or tuple(tokens.shape) != (128, 128)):
            raise ValueError('Expected exactly first 128 frozen sequences')
        decoded = tokenizer.batch_decode(tokens.tolist(), skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)
        if not isinstance(decoded, list) or len(decoded) != 128 or any(not isinstance(item, str) for item in decoded):
            raise ValueError('Merged tokenizer returned invalid decoded text inventory')
        texts[name] = decoded
    return {'format_version': 1, 'attempt_id': ATTEMPT,
            'spec_sha256': SPEC_SHA256, 'corpus_order': list(ORDER),
            'sequence_indices': list(range(128)),
            'token_artifact_sha256': {row['name']: row['serialized_sha256'] for row in spec['corpora']},
            'qwen_tokenizer_files': spec['qwen_tokenizer']['files'],
            'decode': spec['qwen_tokenizer']['decode'], 'texts': texts}


def freeze_decoded_inventory(inventory, spec, *, smoke_only):
    digest = sha256_bytes(canonical_bytes(inventory))
    if smoke_only:
        return digest
    path = path_of(spec['decoded_inventory_path'])
    if path.exists():
        stored = old.load_json_object(path, 'decoded text inventory')
        if stored != inventory or old.sha256_file(path) != digest:
            raise ValueError('Frozen decoded text inventory changed')
    else:
        atomic_new_json(path, inventory)
        require_hash(path, digest)
    return digest


def native_ids(texts, tokenizer):
    values = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    if len(values) != len(texts) or any(not isinstance(ids, list) or len(ids) < 2 or
            any(type(value) is not int or value < 0 for value in ids) for ids in values):
        raise ValueError('Native reference tokenizer returned fewer than two valid tokens')
    return values


def utf8_byte_count(texts):
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise ValueError('UTF-8 byte accounting requires frozen decoded strings')
    total = sum(len(text.encode('utf-8')) for text in texts)
    if total <= 0:
        raise ValueError('Frozen decoded text has zero UTF-8 bytes')
    return total


def padded_batch(rows, pad_id, device):
    if not rows or type(pad_id) is not int or pad_id < 0:
        raise ValueError('Invalid native-token padding input')
    maximum = max(len(row) for row in rows)
    ids = torch.full((len(rows), maximum), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[i, :len(row)] = 1
    return ids, mask


def causal_ce_sum(logits, ids, mask):
    if (logits.dtype != torch.float32 or logits.ndim != 3 or
            tuple(logits.shape[:2]) != tuple(ids.shape) or mask.shape != ids.shape or
            not bool(torch.isfinite(logits).all())):
        raise ValueError('Nonfinite or malformed reference logits')
    shifted = logits[:, :-1, :].contiguous().reshape(-1, logits.shape[-1])
    labels = ids[:, 1:].contiguous().reshape(-1)
    scored = mask[:, 1:].contiguous().reshape(-1).bool()
    if not bool(scored.any()):
        raise ValueError('No native causal predictions in reference batch')
    selected_loss = torch.nn.functional.cross_entropy(shifted[scored], labels[scored], reduction='sum')
    if not bool(torch.isfinite(selected_loss)):
        raise ValueError('Nonfinite reference cross entropy')
    return float(selected_loss.detach().cpu().double()), int(scored.sum().item())


def score_sequences(model, tokenizer, texts, *, batch_size, progress=None):
    if batch_size not in (8, 16) or len(texts) not in (8, 128):
        raise ValueError('Frozen reference scoring batch/sequence inventory mismatch')
    if model.training or any(parameter.requires_grad or parameter.grad is not None
                             for parameter in model.parameters()):
        raise ValueError('Reference model must be frozen and in eval mode')
    total_utf8_bytes = utf8_byte_count(texts)
    encoded = native_ids(texts, tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError('Reference tokenizer lacks pad and EOS IDs')
    device = next(model.parameters()).device
    loss_parts, predicted_parts = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(encoded), batch_size):
            ids, mask = padded_batch(encoded[start:start+batch_size], pad_id, device)
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
            loss, count = causal_ce_sum(logits, ids, mask)
            loss_parts.append(loss)
            predicted_parts.append(count)
            if progress is not None:
                progress({'completed_sequences': min(start+batch_size, len(encoded)),
                          'total_sequences': len(encoded),
                          'elapsed_seconds': time.perf_counter()-started})
            del ids, mask, logits
    total_predictions = sum(predicted_parts)
    total_tokens = sum(map(len, encoded))
    total_nll_nats = math.fsum(loss_parts)
    mean_nll = total_nll_nats/total_predictions
    nats_per_utf8_byte = total_nll_nats/total_utf8_bytes
    bits_per_utf8_byte = nats_per_utf8_byte/math.log(2)
    if (any(not math.isfinite(value) for value in
            (total_nll_nats, mean_nll, nats_per_utf8_byte, bits_per_utf8_byte)) or
            total_predictions != total_tokens-len(encoded)):
        raise ValueError('Reference loss/token accounting failed')
    return {'total_nll_nats': total_nll_nats,
            'total_utf8_bytes': total_utf8_bytes,
            'nats_per_utf8_byte': nats_per_utf8_byte,
            'bits_per_utf8_byte': bits_per_utf8_byte,
            'mean_nll_per_predicted_native_token': mean_nll,
            'sequence_count': len(encoded), 'total_native_tokens': total_tokens,
            'predicted_native_tokens': total_predictions,
            'mean_sequence_token_count': total_tokens/len(encoded),
            'elapsed_inference_seconds': time.perf_counter()-started}


def load_reference(row):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise ValueError('Native reference inference requires CUDA')
    revision = require_revision(row['revision'])
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(row['repo_id'], revision=revision,
                                              trust_remote_code=False)
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(row['repo_id'], revision=revision,
                                                 torch_dtype='auto', trust_remote_code=False)
    model.to('cuda')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if parameter.grad is not None:
            raise ValueError('Reference model loaded with parameter gradients')
    return model, tokenizer, time.perf_counter()-started


def checkpoint_metadata(spec, model_row, corpus_name, decoded_sha, source_sha,
                        total_utf8_bytes):
    if corpus_name not in ORDER:
        raise ValueError('Unknown corpus checkpoint identity')
    if type(total_utf8_bytes) is not int or total_utf8_bytes <= 0:
        raise ValueError('Checkpoint needs positive frozen decoded UTF-8 byte count')
    return {'format_version': 1, 'attempt_id': ATTEMPT,
            'spec_sha256': SPEC_SHA256,
            'reference_model': {'name': model_row['name'], 'repo_id': model_row['repo_id'],
                                'revision': require_revision(model_row['revision'])},
            'decoded_inventory_sha256': decoded_sha,
            'total_utf8_bytes': total_utf8_bytes,
            'corpus_name': corpus_name,
            'scoring_source_sha256': source_sha,
            'scoring_rule': spec['scoring']}


def make_checkpoint(metadata, result):
    return {**metadata, 'result': result,
            'result_sha256': sha256_bytes(canonical_bytes(result))}


def validate_checkpoint(value, metadata):
    if (not isinstance(value, dict) or set(value) != set(metadata) | {'result', 'result_sha256'} or
            any(value.get(key) != expected for key, expected in metadata.items())):
        raise ValueError('Reference checkpoint input provenance mismatch')
    result = value['result']
    if (not isinstance(result, dict) or
            set(result) != {'total_nll_nats', 'total_utf8_bytes',
                            'nats_per_utf8_byte', 'bits_per_utf8_byte',
                            'mean_nll_per_predicted_native_token', 'sequence_count',
                            'total_native_tokens', 'predicted_native_tokens',
                            'mean_sequence_token_count', 'elapsed_inference_seconds'} or
            value.get('result_sha256') != sha256_bytes(canonical_bytes(result)) or
            result.get('sequence_count') != 128 or
            type(result.get('total_utf8_bytes')) is not int or
            result['total_utf8_bytes'] != metadata['total_utf8_bytes'] or
            result['total_utf8_bytes'] <= 0 or
            type(result.get('total_native_tokens')) is not int or
            type(result.get('predicted_native_tokens')) is not int or
            result['total_native_tokens'] < 256 or
            result['predicted_native_tokens'] != result['total_native_tokens']-128 or
            not isinstance(result.get('mean_nll_per_predicted_native_token'), (float, int)) or
            any(not isinstance(result.get(key), (float, int)) for key in
                ('total_nll_nats', 'nats_per_utf8_byte', 'bits_per_utf8_byte')) or
            not isinstance(result.get('mean_sequence_token_count'), (float, int)) or
            not isinstance(result.get('elapsed_inference_seconds'), (float, int)) or
            any(not math.isfinite(result[key]) for key in
                ('total_nll_nats', 'nats_per_utf8_byte', 'bits_per_utf8_byte',
                 'mean_nll_per_predicted_native_token', 'mean_sequence_token_count',
                 'elapsed_inference_seconds')) or
            result['total_nll_nats'] < 0 or
            result['mean_nll_per_predicted_native_token'] < 0 or
            result['mean_nll_per_predicted_native_token'] !=
                result['total_nll_nats']/result['predicted_native_tokens'] or
            result['nats_per_utf8_byte'] !=
                result['total_nll_nats']/result['total_utf8_bytes'] or
            result['bits_per_utf8_byte'] !=
                result['nats_per_utf8_byte']/math.log(2) or
            result['mean_sequence_token_count'] != result['total_native_tokens']/128 or
            result['elapsed_inference_seconds'] < 0):
        raise ValueError('Reference checkpoint result/hash mismatch')
    return result


def checkpoint_path(spec, model_name, corpus_name):
    if model_name not in dict(MODELS) or corpus_name not in ORDER:
        raise ValueError('Invalid fixed reference checkpoint cell')
    return path_of(spec['checkpoint_directory'])/(model_name+'__'+corpus_name+'.json')


def load_or_store_cell(path, metadata, result=None):
    if path.exists():
        return validate_checkpoint(old.load_json_object(path, 'reference score checkpoint'), metadata)
    if result is None:
        return None
    checkpoint = make_checkpoint(metadata, result)
    atomic_new_json(path, checkpoint)
    return validate_checkpoint(old.load_json_object(path, 'reference score checkpoint'), metadata)


def load_attempt100_pc1_inputs(spec):
    row = spec['attempt100']
    path = path_of(row['artifact_path'])
    require_hash(path, row['artifact_serialized_sha256'])
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    raw = row['artifact_raw_sha256']
    if (not isinstance(artifact, dict) or set(artifact) != set(raw) or
            len(raw) != 10):
        raise ValueError('Frozen Attempt100 response artifact inventory mismatch')
    for name, expected in raw.items():
        value = artifact[name]
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or
                value.dtype != torch.float32 or not value.is_contiguous() or
                tuple(value.shape) != (128, 2048) or not bool(torch.isfinite(value).all()) or
                old.sha256_raw_float32_tensor(value, torch) != expected):
            raise ValueError('Frozen Attempt100 response raw tensor mismatch: '+name)
    return [artifact['response_'+name] for name in ORDER]


def pc1_coordinates(responses):
    if len(responses) != 8:
        raise ValueError('Exactly eight frozen responses required for PC1')
    rows = []
    for value in responses:
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                value.device.type != 'cpu' or tuple(value.shape) != (128, 2048) or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid blind response for PC1')
        selected = value[1:5].double()
        norms = torch.linalg.vector_norm(selected, dim=1)
        if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
            raise ValueError('Zero/nonfinite per-position response norm')
        rows.append((selected/norms[:, None]).reshape(-1))
    matrix = torch.stack(rows)
    centered = matrix-matrix.mean(dim=0)
    u, singular, _ = torch.linalg.svd(centered, full_matrices=False)
    energy = singular.square()
    if (not bool(torch.isfinite(energy).all()) or float(energy.sum()) <= 0 or
            float(singular[0]-singular[1]) <= 1e-12*float(singular[0])):
        raise ValueError('Blind PC1 is degenerate or nonfinite')
    return (u[:, 0]*singular[0]).tolist(), float(energy[0]/energy.sum())


def average_ranks(values):
    if not values or any(not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
        raise ValueError('Ranks require finite scalar values')
    ordered = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0]*len(values)
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start+1+end)/2
        for j in range(start, end):
            ranks[ordered[j]] = rank
        start = end
    return ranks


def pearson(left, right):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError('Correlation inventory mismatch')
    a, b = math.fsum(left)/len(left), math.fsum(right)/len(right)
    x, y = [v-a for v in left], [v-b for v in right]
    xx, yy = math.fsum(v*v for v in x), math.fsum(v*v for v in y)
    if xx <= 0 or yy <= 0:
        raise ValueError('Zero correlation variance')
    result = math.fsum(p*q for p, q in zip(x, y))/math.sqrt(xx*yy)
    if not math.isfinite(result):
        raise ValueError('Nonfinite correlation')
    return max(-1.0, min(1.0, result))


def spearman(left, right):
    return pearson(average_ranks(left), average_ranks(right))


def final_statistics(spec, cells, pc1_raw, explained):
    byte_zscores, byte_ranks, token_zscores, token_ranks = {}, {}, {}, {}
    for model, _ in MODELS:
        for field, zscores, ranks in (
                ('nats_per_utf8_byte', byte_zscores, byte_ranks),
                ('mean_nll_per_predicted_native_token', token_zscores, token_ranks)):
            values = [cells[model][name][field] for name in ORDER]
            if any(not math.isfinite(value) for value in values):
                raise ValueError('Reference losses must be finite')
            mean = math.fsum(values)/8
            std = math.sqrt(math.fsum((value-mean)**2 for value in values)/8)
            if std <= 0 or not math.isfinite(std):
                raise ValueError('Reference losses have zero/nonfinite population standard deviation')
            zscores[model] = [(value-mean)/std for value in values]
            ranks[model] = average_ranks(values)
    ensemble = [-math.fsum(byte_zscores[model][i] for model, _ in MODELS)/2
                for i in range(8)]
    token_ensemble = [-math.fsum(token_zscores[model][i] for model, _ in MODELS)/2
                      for i in range(8)]
    average_rank = [math.fsum(byte_ranks[model][i] for model, _ in MODELS)/2
                    for i in range(8)]
    token_average_rank = [math.fsum(token_ranks[model][i] for model, _ in MODELS)/2
                          for i in range(8)]
    raw_pearson = pearson(ensemble, pc1_raw)
    oriented_pc1 = [-value for value in pc1_raw] if raw_pearson < 0 else list(pc1_raw)
    primary = abs(spearman(ensemble, pc1_raw))
    secondary = abs(raw_pearson)
    per_model = {}
    for model, _ in MODELS:
        score = [-value for value in byte_zscores[model]]
        per_model[model] = {'absolute_spearman_vs_pc1': abs(spearman(score, pc1_raw)),
                            'absolute_pearson_vs_pc1': abs(pearson(score, pc1_raw))}
    corpus_rows = []
    for i, name in enumerate(ORDER):
        corpus_rows.append({'name': name,
                            'reference_scores': {model: {
                                field: cells[model][name][field] for field in (
                                    'total_nll_nats', 'total_utf8_bytes',
                                    'nats_per_utf8_byte', 'bits_per_utf8_byte',
                                    'mean_nll_per_predicted_native_token',
                                    'total_native_tokens', 'predicted_native_tokens',
                                    'mean_sequence_token_count')}
                                for model, _ in MODELS},
                            'z': {model: byte_zscores[model][i] for model, _ in MODELS},
                            'rank': {model: byte_ranks[model][i] for model, _ in MODELS},
                            'stationarity_score': ensemble[i],
                            'average_rank': average_rank[i],
                            'average_rank_score': -average_rank[i],
                            'token_nll_z': {model: token_zscores[model][i]
                                            for model, _ in MODELS},
                            'token_nll_rank': {model: token_ranks[model][i]
                                               for model, _ in MODELS},
                            'token_nll_stationarity_score': token_ensemble[i],
                            'token_nll_average_rank': token_average_rank[i],
                            'token_nll_average_rank_score': -token_average_rank[i],
                            'pc1_coordinate_display': oriented_pc1[i]})
    return {'pc1_variance_explained': explained,
            'pc1_coordinates_raw': pc1_raw,
            'pc1_coordinates_display': oriented_pc1,
            'pc1_display_sign_flipped': raw_pearson < 0,
            'pc1_display_sign_convention': 'nonnegative_pearson_with_blind_ensemble_no_oracle',
            'primary_absolute_spearman': primary,
            'secondary_absolute_pearson': secondary,
            'primary_stationarity_input': 'nats_per_utf8_byte',
            'token_nll_robustness': {
                'stationarity_input': 'mean_nll_per_predicted_native_token',
                'token_nll_stationarity_score': token_ensemble,
                'absolute_spearman_vs_pc1': abs(spearman(token_ensemble, pc1_raw)),
                'absolute_pearson_vs_pc1': abs(pearson(token_ensemble, pc1_raw)),
                'average_rank': token_average_rank},
            'smollm2_vs_pythia_score_spearman': spearman(
                [-x for x in byte_zscores[MODELS[0][0]]],
                [-x for x in byte_zscores[MODELS[1][0]]]),
            'individual_model_vs_pc1': per_model,
            'corpora': corpus_rows,
            'historical_recovery_candidate_constructed': False,
            'oracle_evaluation': False}


def run(*, smoke_only=False, resolver=None, model_loader=None, qwen_loader=None):
    spec = load_spec()
    result_path = path_of(spec['result_path'])
    if not smoke_only and (result_path.exists() or result_path.is_symlink()):
        raise ValueError('Final Attempt101 result already exists')
    validate_metadata(spec)
    lock, lock_sha = get_lock(spec, smoke_only=smoke_only, resolver=resolver)
    source_sha = old.sha256_file(Path(__file__).resolve())
    tokenizer_root = validate_tokenizer_files(spec)
    if qwen_loader is None:
        from transformers import AutoTokenizer
        qwen_loader = lambda root: AutoTokenizer.from_pretrained(root, local_files_only=True)
    qwen = qwen_loader(tokenizer_root)
    tokens = load_first_sequences(spec)
    if smoke_only:
        fineweb = qwen.batch_decode(tokens['fineweb'][:8].tolist(),
                                    skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)
        if len(fineweb) != 8:
            raise ValueError('Smoke FineWeb inventory mismatch')
        del tokens, qwen
        loader = model_loader or load_reference
        records = []
        for row in lock['models']:
            model, tokenizer, load_seconds = loader(row)
            try:
                started = time.perf_counter()
                score = score_sequences(model, tokenizer, fineweb, batch_size=8)
                inference = time.perf_counter()-started
                projected = inference*(8*128/16)
                records.append({'model': row['name'], 'load_seconds': load_seconds,
                                'batch_inference_seconds': inference,
                                'tokens_per_second': score['predicted_native_tokens']/inference,
                                'projected_full_model_compute_seconds': projected,
                                'warning_over_15_minutes': projected > 900})
            finally:
                del model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        warning = sum(r['projected_full_model_compute_seconds'] for r in records) > 900
        if warning:
            print('WARNING: projected full reference-model compute exceeds 15 minutes', flush=True)
        return {'smoke_only': True, 'reference_lock_sha256': lock_sha,
                'models': records, 'writes': False,
                'warning_over_15_minutes': warning}
    inventory = decode_inventory(tokens, qwen, spec)
    del tokens, qwen
    decoded_sha = freeze_decoded_inventory(inventory, spec, smoke_only=False)
    byte_counts = {name: utf8_byte_count(inventory['texts'][name]) for name in ORDER}
    loader = model_loader or load_reference
    cells = {name: {} for name, _ in MODELS}
    for row in lock['models']:
        name = row['name']
        pending = []
        for corpus in ORDER:
            metadata = checkpoint_metadata(spec, row, corpus, decoded_sha, source_sha,
                                           byte_counts[corpus])
            path = checkpoint_path(spec, name, corpus)
            cached = load_or_store_cell(path, metadata)
            if cached is None:
                pending.append((corpus, path, metadata))
            else:
                cells[name][corpus] = cached
                print(json.dumps({'model': name, 'corpus': corpus, 'status': 'checkpoint_reused'}), flush=True)
        if not pending:
            continue
        model, tokenizer, load_seconds = loader(row)
        print(json.dumps({'model': name, 'status': 'loaded', 'load_seconds': load_seconds}), flush=True)
        try:
            for corpus, path, metadata in pending:
                started = time.perf_counter()
                def progress(value):
                    print(json.dumps({'model': name, 'corpus': corpus, 'status': 'scoring', **value}), flush=True)
                result = score_sequences(model, tokenizer, inventory['texts'][corpus],
                                         batch_size=16, progress=progress)
                cells[name][corpus] = load_or_store_cell(path, metadata, result)
                print(json.dumps({'model': name, 'corpus': corpus, 'status': 'checkpoint_written',
                                  'elapsed_seconds': time.perf_counter()-started}), flush=True)
        finally:
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    pc1_raw, explained = pc1_coordinates(load_attempt100_pc1_inputs(spec))
    statistics = final_statistics(spec, cells, pc1_raw, explained)
    for path, expected in ((SPEC_PATH, SPEC_SHA256),
                           (path_of(spec['reference_lock_path']), lock_sha),
                           (path_of(spec['decoded_inventory_path']), decoded_sha),
                           (Path(__file__).resolve(), source_sha)):
        require_hash(path, expected)
    result = {'format_version': 1, 'attempt_id': ATTEMPT,
              'purpose': spec['purpose'], 'information_policy': spec['information_policy'],
              'provenance': {'spec_sha256': SPEC_SHA256, 'scoring_source_sha256': source_sha,
                             'reference_model_lock_sha256': lock_sha,
                             'decoded_text_inventory_sha256': decoded_sha,
                             'attempt100_construction_manifest_sha256': spec['attempt100']['manifest_sha256'],
                             'attempt100_construction_artifact_sha256': spec['attempt100']['artifact_serialized_sha256'],
                             'corpus_token_artifact_sha256': {r['name']: r['serialized_sha256'] for r in spec['corpora']}},
              'reference_models': lock['models'], 'cells': cells, 'statistics': statistics,
              **spec['firewall']}
    if result_path.exists() or result_path.is_symlink():
        raise ValueError('Final Attempt101 result already exists')
    atomic_new_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(smoke_only=args.smoke_only), allow_nan=False))


if __name__ == '__main__':
    main()
