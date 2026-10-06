#!/usr/bin/env python3
"""Blind Attempt200 construction only. Never invokes historical evaluation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '200_superlow_one_step_descendant_pilot'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = '122939dff9cf0ca565c0fabc9e95d8b1f0a00108d9464fa2367e9af339e8ac89'
MARKER = 'SUPERLOW_ONE_STEP_DESCENDANT_FROZEN'
MODEL_DIR = Path('/root/model-diff-scratch/models/merged')
SOURCE_INTERVALS = ((20000, 24096), (24096, 28192), (28192, 36384))


def log(message):
    print(message, flush=True)


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('Frozen SHA256 mismatch: ' + str(path))


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def raw_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.astype(array.dtype.newbyteorder('<'), copy=False).tobytes()).hexdigest()


def read_record(record):
    path = path_of(record['path'])
    require_hash(path, record['sha256'])
    return json.loads(path.read_text())


def load_spec():
    require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text())
    if spec['attempt_id'] != ATTEMPT or path_of(spec['paths']['model_dir']) != MODEL_DIR:
        raise ValueError('Noncanonical experiment/model')
    return spec


def blind_metadata(spec):
    """Explicit allowlist: no historical specs, scores, weights or ADL are opened."""
    inventory = read_record(spec['merged_inventory'])
    probe = read_record(spec['probe_manifest'])
    require_hash(path_of(spec['probe_freezer']['path']), spec['probe_freezer']['sha256'])
    if (inventory['merge_metadata']['tokenizer_source'] != 'base' or
            probe['dataset'] != {k: spec['dataset'][k] for k in ('repo_id', 'revision', 'split')} or
            probe['seed'] != 42 or probe['text_column'] != 'text' or
            probe['n'] != 128 or probe['max_samples'] != 10000 or
            probe['character_truncation_length'] != 1280 or
            probe['raw_tensor_sha256'] != spec['probe']['raw_tensor_sha256']):
        raise ValueError('Canonical probe conventions changed')
    for source, interval in zip(spec['sources'], SOURCE_INTERVALS, strict=True):
        corpus = read_record(source['corpus_spec'])
        manifest = read_record(source['corpus_manifest'])
        selection = {**spec['selection'], 'skip_valid_examples': interval[0],
                     'sample_count': interval[1] - interval[0]}
        if (source['valid_rank_interval'] != list(interval) or
                corpus['selection'] != selection or manifest['selection'] != selection or
                corpus['dataset'] != spec['dataset'] or manifest['dataset'] != spec['dataset'] or
                corpus['tokenizer'] != {'source': 'canonical_merged_checkpoint',
                                        'directory': str(MODEL_DIR)} or
                manifest['artifact']['raw_tensor_sha256'] != source['raw_tensor_sha256']):
            raise ValueError('Source slice provenance mismatch')
    pairs = selection_pairs(read_record(spec['ranking_manifest']), spec)
    return inventory, pairs


def selection_pairs(manifest, spec):
    import torch
    group = manifest['combined_ranking']['groups']['super_low']
    values = group['ordered_pairs']
    if (len(values) != 1024 or any(not isinstance(pair, list) or len(pair) != 2 or
            any(type(x) is not int for x in pair) for pair in values)):
        raise ValueError('Malformed SUPER_LOW ordered pairs')
    pairs = torch.tensor(values, dtype=torch.int64).contiguous()
    expected = spec['ordered_pair_raw_int64_sha256']
    if raw_hash(pairs) != expected or group['ordered_pair_raw_int64_sha256'] != expected:
        raise ValueError('SUPER_LOW ordered-pair hash mismatch')
    if len(set(map(tuple, values))) != 1024:
        raise ValueError('Duplicate SUPER_LOW identity')
    for source_id, row in values:
        if source_id not in (0, 1, 2) or not 0 <= row < (4096, 4096, 8192)[source_id]:
            raise ValueError('SUPER_LOW source/row out of bounds')
    return pairs


def validate_checkpoint(directory, records):
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
              if p.is_file() and '.cache' not in p.relative_to(directory).parts}
    if actual != {record['path'] for record in records}:
        raise ValueError('Checkpoint file inventory mismatch')
    for record in records:
        path = directory / record['path']
        log('Validating checkpoint file ' + str(path))
        if path.stat().st_size != record['size_bytes']:
            raise ValueError('Checkpoint size mismatch: ' + str(path))
        require_hash(path, record['sha256'])


def load_helpers(spec):
    helpers = []
    for name, record in spec['helpers'].items():
        path = path_of(record['path'])
        require_hash(path, record['sha256'])
        loader = importlib.util.spec_from_file_location('attempt200_' + name, path)
        module = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(module)
        helpers.append(module)
    return helpers


def rehydrate(dataset, tokenizer, *, intervals=SOURCE_INTERVALS, probe_count=10000):
    """One shuffle/pass; ranks count only nonblank texts with >=128 tokens."""
    import torch
    sources = [[] for _ in intervals]
    probe = []
    valid = 0
    stop = max(probe_count, *(end for _, end in intervals))
    log(f'Rehydration starting: need {stop} valid examples')
    for scanned, sample in enumerate(dataset.shuffle(seed=42), start=1):
        if scanned % 1000 == 0:
            log(f'Rehydration scanned={scanned}, valid={valid}/{stop}')
        text = sample['text']
        if not isinstance(text, str):
            raise ValueError('Non-string FineWeb text')
        if not text.strip():
            continue
        ids = tokenizer.encode(text[:1280], add_special_tokens=True)
        if not isinstance(ids, list) or any(type(x) is not int for x in ids):
            raise ValueError('Invalid tokenizer IDs')
        if len(ids) < 128:
            continue
        row = ids[:128]
        if valid < probe_count:
            probe.append(row)
        for index, (start, end) in enumerate(intervals):
            if start <= valid < end:
                sources[index].append(row)
        valid += 1
        if valid == stop:
            break
    if valid != stop:
        raise ValueError('Dataset exhausted before exact valid ranks')
    return [torch.tensor(rows, dtype=torch.int64).contiguous() for rows in sources], \
        torch.tensor(probe, dtype=torch.int64).contiguous()


def select_rows(sources, pairs):
    import torch
    return torch.stack([sources[source_id][row] for source_id, row in pairs.tolist()]).contiguous()


def validate_tensor(tensor, shape, dtype):
    import torch
    if (not isinstance(tensor, torch.Tensor) or list(tensor.shape) != list(shape) or
            tensor.dtype != dtype or tensor.device.type != 'cpu' or
            not tensor.is_contiguous() or not bool(torch.isfinite(tensor).all())):
        raise ValueError('Malformed frozen tensor')


def mean_activation(model, probe, spec, helper, label):
    import torch
    validate_tensor(probe, (10000, 128), torch.int64)
    total = torch.zeros((128, 2048), dtype=torch.float64)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 1024, 32):
            readout = helper.ordinary_hook_readout(model, probe[start:start+32].to(device), spec, torch)
            total.add_(readout.to(device='cpu', dtype=torch.float64).sum(dim=0))
            log(f'{label} activation probe batch {start//32+1}/32')
    mean = (total / 1024).contiguous()
    validate_tensor(mean, (128, 2048), torch.float64)
    return mean


def endpoint_response(final_mean, descendant_mean):
    import torch
    for mean in (final_mean, descendant_mean):
        validate_tensor(mean, (128, 2048), torch.float64)
    return (final_mean - descendant_mean).to(torch.float32).contiguous()


def full_loss(model, tokens, helper, *, backward=False):
    import torch
    validate_tensor(tokens, (1024, 128), torch.int64)
    device = next(model.parameters()).device
    model.eval()
    if backward:
        model.zero_grad(set_to_none=True)
    sums = []
    with (torch.enable_grad() if backward else torch.inference_mode()), \
            torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 1024, 8):
            batch = tokens[start:start+8].to(device)
            loss = helper.causal_token_loss_sum(model(input_ids=batch, use_cache=False).logits, batch, torch)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite full-dataset CE')
            if backward:
                (loss / 130048).backward()
            sums.append(float(loss.detach().double().cpu()))
            log(f'{"Gradient" if backward else "Trial CE"} batch {start//8+1}/128')
    return math.fsum(sums) / 130048


def norm(tensors):
    import torch
    return math.sqrt(math.fsum(float(x.detach().to('cpu', dtype=torch.float64).square().sum())
                               for x in tensors))


def state_hashes(eligible):
    matrices = {name + '.weight': raw_hash(module.weight) for name, module in eligible}
    digest = hashlib.sha256()
    for name, value in matrices.items():
        digest.update(name.encode() + b'\0' + value.encode('ascii') + b'\n')
    return {'per_matrix_sha256': matrices, 'aggregate_sha256': digest.hexdigest()}


def initial_scale(eligible, initial_loss):
    """Snapshot F and derive Attempt121/122's initial Frobenius step scale."""
    import torch
    originals = {name: module.weight.detach().cpu().clone().contiguous() for name, module in eligible}
    for name, module in eligible:
        if module.weight.grad is None or module.weight.grad.dtype != torch.float32 or \
                not bool(torch.isfinite(module.weight.grad).all()):
            raise ValueError('Invalid aggregate gradient: ' + name)
    wnorm = norm(originals.values())
    gnorm = norm(module.weight.grad for _, module in eligible)
    if not math.isfinite(initial_loss) or not math.isfinite(wnorm) or not math.isfinite(gnorm) or \
            wnorm <= 0 or gnorm <= 0:
        raise ValueError('Invalid initial scale/loss')
    eta_max = .00125 * wnorm / gnorm
    return originals, {'initial_ce': initial_loss, 'Wnorm': wnorm, 'gradient_norm': gnorm,
                       'target_step_norm': .00125 * wnorm, 'eta_max': eta_max}


def armijo_step(eligible, originals, diagnostics, objective):
    import torch
    def restore():
        with torch.no_grad():
            for name, module in eligible:
                module.weight.copy_(originals[name].to(module.weight.device))
                if not torch.equal(module.weight.detach().cpu(), originals[name]):
                    raise ValueError('Rejected trial restoration mismatch')
    trials = []
    try:
        for index in range(8):
            eta = diagnostics['eta_max'] * .5**index
            rhs = diagnostics['initial_ce'] - 1e-4 * eta * diagnostics['gradient_norm']**2
            if not math.isfinite(eta) or eta <= 0 or not math.isfinite(rhs):
                raise ValueError('Invalid Armijo trial')
            with torch.no_grad():
                for name, module in eligible:
                    updated = (originals[name].double() - eta * module.weight.grad.detach().cpu().double()).float()
                    if not bool(torch.isfinite(updated).all()):
                        raise ValueError('Nonfinite trial state')
                    module.weight.copy_(updated.to(module.weight.device))
            log(f'Armijo trial {index+1}/8, eta={eta:.12g}')
            loss = objective()
            if not math.isfinite(loss):
                raise ValueError('Nonfinite trial CE')
            accepted = loss <= rhs
            trials.append({'trial_index': index, 'eta': eta, 'ce': loss,
                           'armijo_rhs': rhs, 'accepted': accepted})
            log(f'Armijo CE={loss:.12g}, RHS={rhs:.12g}, accepted={accepted}')
            if accepted:
                displacement = norm(module.weight.detach().cpu().double() - originals[name].double()
                                    for name, module in eligible)
                return {**diagnostics, 'trials': trials, 'accepted_updates': 1,
                        'accepted_eta': eta, 'accepted_post_step_ce': loss,
                        'displacement_norm': displacement,
                        'displacement_relative_to_initial_Wnorm': displacement / diagnostics['Wnorm']}
            restore()
    except BaseException:
        restore()
        raise
    raise ValueError('All eight Armijo trials rejected; construction failed')


def publish(path, payload, *, tensor=False):
    """Atomic, exclusive publication; leave existing artifacts untouched."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temporary.open('xb') as stream:
            if tensor:
                torch.save(payload, stream)
            else:
                stream.write((json.dumps(payload, indent=2, allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def blind_input_hashes(spec):
    records = [spec[key] for key in ('merged_inventory', 'probe_manifest', 'probe_freezer', 'ranking_manifest')]
    records += list(spec['helpers'].values())
    records += [source[key] for source in spec['sources'] for key in ('corpus_spec', 'corpus_manifest')]
    for record in records:
        require_hash(path_of(record['path']), record['sha256'])
    return {record['path']: record['sha256'] for record in records}


def validate_diagnostics(record):
    trials = record['trials']
    numbers = [record[key] for key in ('initial_ce', 'Wnorm', 'gradient_norm', 'eta_max',
               'target_step_norm', 'accepted_eta', 'accepted_post_step_ce', 'displacement_norm',
               'displacement_relative_to_initial_Wnorm')]
    if not all(math.isfinite(x) for x in numbers) or record['Wnorm'] <= 0 or record['gradient_norm'] <= 0:
        raise ValueError('Invalid frozen scale')
    if (record['accepted_updates'] != 1 or not 1 <= len(trials) <= 8 or
            record['target_step_norm'] != .00125 * record['Wnorm'] or
            record['eta_max'] != record['target_step_norm'] / record['gradient_norm'] or
            record['accepted_eta'] != trials[-1]['eta'] or
            record['accepted_post_step_ce'] != trials[-1]['ce'] or
            record['displacement_norm'] < 0 or
            record['displacement_relative_to_initial_Wnorm'] != record['displacement_norm']/record['Wnorm']):
        raise ValueError('Frozen single-step diagnostics mismatch')
    for index, trial in enumerate(trials):
        eta = record['eta_max'] * .5**index
        rhs = record['initial_ce'] - 1e-4 * eta * record['gradient_norm']**2
        accepted = trial['ce'] <= rhs
        if (not math.isfinite(trial['ce']) or trial['trial_index'] != index or trial['eta'] != eta or
                trial['armijo_rhs'] != rhs or trial['accepted'] != accepted or
                accepted != (index == len(trials)-1)):
            raise ValueError('Frozen Armijo trial mismatch')


def validate_frozen(spec):
    """Complete blind freeze audit; also the evaluator's prerequisite gate."""
    import torch
    if spec != load_spec():
        raise ValueError('Freeze audit specification mismatch')
    inputs = blind_input_hashes(spec)
    inventory, pairs = blind_metadata(spec)
    artifact_dir = path_of(spec['paths']['artifact_dir'])
    manifest_path = path_of(spec['paths']['construction_manifest'])
    receipt = json.loads((artifact_dir / 'freeze-receipt.json').read_text())
    require_hash(manifest_path, receipt['construction_manifest_sha256'])
    manifest = json.loads(manifest_path.read_text())
    if (manifest['attempt_id'] != ATTEMPT or manifest['spec_sha256'] != SPEC_SHA256 or
            manifest['constructor_sha256'] != sha256_file(Path(__file__)) or
            manifest['blind_input_hashes'] != inputs or manifest['barrier'] != MARKER or
            manifest['candidate_definition'] != spec['candidate']['definition'] or
            manifest['final_checkpoint_files'] != inventory['files'] or
            receipt['spec_sha256'] != SPEC_SHA256 or
            receipt['constructor_sha256'] != manifest['constructor_sha256']):
        raise ValueError('Frozen construction provenance mismatch')
    if manifest['ordered_pairs'] != pairs.tolist():
        raise ValueError('Manifest row-pair inventory mismatch')
    validate_diagnostics(manifest['step'])
    for state in ('before', 'after'):
        hashes = manifest['eligible_state'][state]
        matrices = hashes['per_matrix_sha256']
        if len(matrices) != 98 or list(matrices) != sorted(matrices):
            raise ValueError('Eligible hash inventory mismatch')
        digest = hashlib.sha256()
        for name, value in matrices.items():
            if len(value) != 64:
                raise ValueError('Invalid state SHA256')
            digest.update(name.encode() + b'\0' + value.encode('ascii') + b'\n')
        if digest.hexdigest() != hashes['aggregate_sha256']:
            raise ValueError('Eligible aggregate hash mismatch')
    if set(manifest['eligible_state']['before']['per_matrix_sha256']) != \
            set(manifest['eligible_state']['after']['per_matrix_sha256']):
        raise ValueError('Eligible before/after inventory mismatch')
    expected = {'candidate': ((128, 2048), torch.float32, None),
                'probe': ((10000, 128), torch.int64, spec['probe']['raw_tensor_sha256']),
                'selected': ((1024, 128), torch.int64, None),
                'pairs': ((1024, 2), torch.int64, spec['ordered_pair_raw_int64_sha256'])}
    for source in spec['sources']:
        start, end = source['valid_rank_interval']
        expected[f"source{source['source_id']}"] = ((end-start, 128), torch.int64, source['raw_tensor_sha256'])
    if set(manifest['artifacts']) != set(expected):
        raise ValueError('Frozen artifact inventory mismatch')
    tensors = {}
    for name, (shape, dtype, committed_hash) in expected.items():
        record = manifest['artifacts'][name]
        path = artifact_dir / (name + '.pt')
        require_hash(path, record['serialized_sha256'])
        value = torch.load(path, map_location='cpu', weights_only=True)
        validate_tensor(value, shape, dtype)
        digest = raw_hash(value)
        if digest != record['raw_tensor_sha256'] or (committed_hash and digest != committed_hash):
            raise ValueError('Frozen raw tensor hash mismatch: ' + name)
        tensors[name] = value
    if not torch.equal(tensors['pairs'], pairs) or not torch.equal(tensors['selected'],
            select_rows([tensors[f'source{i}'] for i in range(3)], pairs)):
        raise ValueError('Frozen selection mapping mismatch')
    validate_checkpoint(MODEL_DIR, inventory['files'])
    return tensors, manifest, receipt


def run(device='cuda'):
    import torch
    spec = load_spec()
    constructor_hash = sha256_file(Path(__file__))
    artifact_dir = path_of(spec['paths']['artifact_dir'])
    manifest_path = path_of(spec['paths']['construction_manifest'])
    for path in (artifact_dir, manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError('Output already exists: ' + str(path))
    inputs = blind_input_hashes(spec)
    inventory, pairs = blind_metadata(spec)
    validate_checkpoint(MODEL_DIR, inventory['files'])
    gradient_helper, readout_helper = load_helpers(spec)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    log('Loading pinned non-streaming public FineWeb dataset')
    dataset = load_dataset(**{'path': spec['dataset']['repo_id'],
        'revision': spec['dataset']['revision'], 'split': 'train', 'streaming': False})
    sources, probe = rehydrate(dataset, tokenizer)
    for source, tensor in zip(spec['sources'], sources, strict=True):
        if raw_hash(tensor) != source['raw_tensor_sha256']:
            raise ValueError('Reconstructed source hash mismatch')
    if raw_hash(probe) != spec['probe']['raw_tensor_sha256']:
        raise ValueError('Reconstructed full canonical probe hash mismatch')
    tokens = select_rows(sources, pairs)
    model, _ = gradient_helper.load_local_model(MODEL_DIR, device, torch)
    model.config.use_cache = False
    eligible = gradient_helper.discover_eligible_linear_weights(model, torch)
    selected_ids = gradient_helper.freeze_other_parameters(model, eligible)
    frozen = gradient_helper.frozen_parameter_hashes(model, selected_ids, torch)
    before = state_hashes(eligible)
    final_mean = mean_activation(model, probe, spec, readout_helper, 'F')
    initial_loss = full_loss(model, tokens, gradient_helper, backward=True)
    originals, diagnostics = initial_scale(eligible, initial_loss)
    step = armijo_step(eligible, originals, diagnostics,
                       lambda: full_loss(model, tokens, gradient_helper))
    descendant_mean = mean_activation(model, probe, spec, readout_helper, 'G')
    response = endpoint_response(final_mean, descendant_mean)
    after = state_hashes(eligible)
    gradient_helper.verify_frozen_parameters(model, selected_ids, frozen, torch)
    payloads = {'candidate': response, 'probe': probe, 'selected': tokens, 'pairs': pairs,
                **{f'source{i}': tensor for i, tensor in enumerate(sources)}}
    artifacts = {}
    for name, tensor in payloads.items():
        path = artifact_dir / (name + '.pt')
        publish(path, tensor, tensor=True)
        artifacts[name] = {'serialized_sha256': sha256_file(path), 'raw_tensor_sha256': raw_hash(tensor)}
    if (inputs != blind_input_hashes(spec) or sha256_file(SPEC_PATH) != SPEC_SHA256 or
            sha256_file(Path(__file__)) != constructor_hash):
        raise ValueError('Blind inputs changed during construction')
    manifest = {'format_version': 1, 'attempt_id': ATTEMPT, 'spec_sha256': SPEC_SHA256,
        'constructor_sha256': constructor_hash, 'blind_input_hashes': inputs,
        'final_checkpoint_files': inventory['files'], 'eligible_state': {'before': before, 'after': after},
        'step': step, 'candidate_definition': spec['candidate']['definition'],
        'ordered_pairs': pairs.tolist(), 'artifacts': artifacts,
        'activation_mean_float64_sha256': {'F': raw_hash(final_mean), 'G': raw_hash(descendant_mean)},
        'barrier': MARKER, 'same_specimen_exploratory_diagnostic': True, 'held_out_validation': False,
        'runtime': {'torch': torch.__version__, 'device': device}}
    publish(manifest_path, manifest)
    publish(artifact_dir / 'freeze-receipt.json', {'construction_manifest_sha256': sha256_file(manifest_path),
        'spec_sha256': SPEC_SHA256, 'constructor_sha256': manifest['constructor_sha256']})
    validate_frozen(spec)
    log(MARKER)
    return manifest


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
