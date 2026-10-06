#!/usr/bin/env python3
"""Blind Attempt201: eight hard self-distillation descendants; no evaluation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import statistics

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '201_hard_self_distill_descendant_pilot'
SPEC_PATH = PROJECT / 'experiments/attempts' / ATTEMPT / 'spec.json'
SPEC_SHA256 = '5a781cc138160e628da2770443f1515c69feec4fa4a2999f556191e281955eea'
SEEDS = tuple(range(8))
SEED_NAMES = tuple(f'response_seed_{i}' for i in SEEDS)
NAMES = (*SEED_NAMES, 'response_raw_mean', 'response_seed_consensus')
DOMAIN = 'attempt201-hard-descendant-v1'
STEPS = 4
RELATIVE_STEP = 7.8125e-5
MARKER = 'HARD_SELF_DISTILL_DESCENDANTS_FROZEN'


def import_pinned(record, name):
    path = Path(record['path'])
    if not path.is_absolute():
        path = PROJECT / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('Pinned helper SHA256 mismatch: ' + str(path))
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


b = import_pinned({'path': 'scripts/ablation/run_superlow_one_step_descendant_pilot.py',
    'sha256': '0bbfdcad957df9d4a32956feff5396520ee3b93cf6c7ea63ecd5dc47f41999d7'}, 'attempt201_blind_training')


def load_spec():
    b.require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text())
    if (spec['attempt_id'] != ATTEMPT or spec['sampling']['seed_ids'] != list(SEEDS) or
            spec['sampling']['uniform_domain'] != DOMAIN + '|seed=<seed>|row=<row>' or
            spec['training']['accepted_steps_per_seed'] != STEPS or
            spec['training']['proposed_step_relative_to_initial_Wnorm'] != RELATIVE_STEP or
            spec['candidate']['tensor_order'] != list(NAMES) or spec['barrier'] != MARKER):
        raise ValueError('Attempt201 fixed plan mismatch')
    return spec


def blind_records(spec):
    return ([spec['probe_manifest']]
            + list(spec['blind_helpers'].values())
            + [spec['source'][k] for k in ('corpus_spec', 'corpus_manifest')])


def blind_inputs(spec):
    hashes = {}
    for record in blind_records(spec):
        b.require_hash(b.path_of(record['path']), record['sha256'])
        hashes[record['path']] = record['sha256']
    corpus = b.read_record(spec['source']['corpus_spec'])
    manifest = b.read_record(spec['source']['corpus_manifest'])
    probe = b.read_record(spec['probe_manifest'])
    selection = {**spec['selection'], 'skip_valid_examples': 20000, 'sample_count': 4096}
    if (corpus['selection'] != selection or manifest['selection'] != selection or
            corpus['dataset'] != spec['dataset'] or manifest['dataset'] != spec['dataset'] or
            corpus['tokenizer'] != {'source': 'canonical_merged_checkpoint', 'directory': str(b.MODEL_DIR)} or
            spec['source']['valid_rank_interval'] != [20000, 24096] or
            spec['source']['raw_tensor_sha256'] != manifest['artifact']['raw_tensor_sha256'] or
            spec['source']['raw_tensor_sha256'] != '4613982be77342a89b35616d68e21d9ff66d5a983af7e82cb6fa1eb159457eae' or
            probe['raw_tensor_sha256'] != spec['probe']['raw_tensor_sha256'] or
            probe['dataset'] != {k: spec['dataset'][k] for k in ('repo_id', 'revision', 'split')} or
            probe['seed'] != 42 or probe['n'] != 128 or probe['max_samples'] != 10000 or
            probe['text_column'] != 'text' or probe['character_truncation_length'] != 1280):
        raise ValueError('Blind corpus/probe/sampling provenance mismatch')
    return hashes, {'files': spec['final_checkpoint_files']}


def helpers(spec):
    records = spec['blind_helpers']
    gradient = import_pinned(records['gradient'], 'attempt201_eligible_helper')
    readout = import_pinned(records['readout'], 'attempt201_readout_helper')
    return gradient, readout


def hash_uniform(seed, row):
    if type(seed) is not int or seed not in SEEDS or type(row) is not int or not 0 <= row < 64:
        raise ValueError('Sample identity outside the fixed 8 x 64 grid')
    bits = int.from_bytes(hashlib.sha256(f'{DOMAIN}|seed={seed}|row={row}'.encode('ascii')).digest()[:8], 'big')
    return min(max((bits + .5) / 2**64, math.nextafter(0., 1.)), math.nextafter(1., 0.))


def inverse_cdf(probs, uniform):
    probabilities = np.asarray(probs)
    if (probabilities.ndim != 1 or probabilities.size < 2 or
            probabilities.dtype != np.float64 or
            not probabilities.flags.c_contiguous or
            not np.isfinite(probabilities).all() or
            np.any(probabilities < 0) or
            not math.isfinite(uniform) or not 0 < uniform < 1):
        raise ValueError('Malformed exact-categorical probabilities or uniform')
    total = math.fsum(map(float, probabilities))
    if abs(total - 1.) > 1e-12:
        raise ValueError('Full-vocabulary softmax normalization mismatch')
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    cumulative[-1] = 1.0  # Guard the last CDF value against summation roundoff.
    index = int(np.searchsorted(cumulative, uniform, side='right'))
    if index >= probabilities.size or probabilities[index] <= 0:
        raise ValueError('Inverse-CDF selected invalid zero-probability token')
    return index, float(probabilities[index]), int(np.argmax(probabilities))


def table_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def sample_table_from_probabilities(probabilities, progress=None):
    probs = np.asarray(probabilities)
    if (probs.ndim != 2 or probs.shape[0] != 64 or probs.dtype != np.float64 or
            not probs.flags.c_contiguous or not np.isfinite(probs).all()):
        raise ValueError('Expected 64 full-vocabulary float64 probability rows')
    seeds = []
    for seed in SEEDS:
        ids, uniforms, chosen_probs, surprises, argmax_matches = [], [], [], [], 0
        for row in range(64):
            uniform = hash_uniform(seed, row)
            token, probability, argmax = inverse_cdf(probs[row], uniform)
            ids.append(token)
            uniforms.append(uniform)
            chosen_probs.append(probability)
            surprises.append(-math.log(probability))
            argmax_matches += int(token == argmax)
        record = {'seed_id': seed, 'original_rows': list(range(64)),
                  'sampled_token_ids': ids, 'uniforms': uniforms,
                  'sampled_probabilities': chosen_probs,
                  'sampled_surprisals': surprises,
                  'diagnostics': {
                      'mean_probability': math.fsum(chosen_probs) / 64,
                      'median_probability': statistics.median(chosen_probs),
                      'min_probability': min(chosen_probs),
                      'max_probability': max(chosen_probs),
                      'mean_surprisal': math.fsum(surprises) / 64,
                      'max_surprisal': max(surprises),
                      'argmax_count': argmax_matches}}
        record['sha256'] = table_hash(record)
        seeds.append(record)
        if progress is not None:
            progress(seed)
    table = {'seed_order': list(SEEDS), 'context_rows': [0, 64],
             'context_prefix_positions': [0, 127],
             'sampler': 'SHA256_uniform_full_vocabulary_float64_softmax_inverse_CDF',
             'seeds': seeds}
    table['aggregate_sha256'] = table_hash(table)
    return table


def fixed_contexts(corpus):
    import torch
    b.validate_tensor(corpus, (4096, 128), torch.int64)
    return corpus[:64, :127].contiguous()


def sample_table(probabilities, progress=None):
    table = sample_table_from_probabilities(np.ascontiguousarray(probabilities.numpy()), progress)
    table['uniform_domain'] = DOMAIN + '|seed=<seed>|row=<row>'
    table['aggregate_sha256'] = table_hash({k: v for k, v in table.items() if k != 'aggregate_sha256'})
    return table


def sampled_next_token_loss_sum(last_logits, target):
    """One gathered hard next-token target per context, with FP64 log-probs."""
    import torch
    if last_logits.ndim != 2 or last_logits.dtype != torch.float32:
        raise ValueError('Expected FP32 final-position logits only')
    batch, vocab = last_logits.shape
    student_logp = torch.log_softmax(last_logits.to(torch.float64), dim=-1)
    if (target is None or target.dtype != torch.int64 or target.device != last_logits.device or
            tuple(target.shape) != (batch,)):
        raise ValueError('Malformed single-seed next-token target shape')
    if bool((target < 0).any()) or bool((target >= vocab).any()):
        raise ValueError('Sampled token outside full vocabulary')
    return -student_logp.gather(1, target.unsqueeze(1)).sum()


def sample_probabilities(model, contexts):
    import torch
    b.validate_tensor(contexts, (64, 127), torch.int64)
    model.eval()
    device = next(model.parameters()).device
    rows = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 64, 8):
            logits = model(input_ids=contexts[start:start+8].to(device), use_cache=False).logits
            if (logits.dtype != torch.float32 or tuple(logits.shape[:2]) != (8, 127) or
                    logits.shape[-1] != model.config.vocab_size or not bool(torch.isfinite(logits).all())):
                raise ValueError('Invalid full-vocabulary sampling logits')
            rows.append(torch.softmax(logits[:, -1, :].double(), dim=-1).cpu())
            b.log(f'Sampling probability batch {start//8+1}/8')
    return torch.cat(rows).contiguous()


def hard_loss(model, contexts, targets, seed, *, backward=False):
    """Exactly 64 final-position NLL terms; context tokens never receive labels."""
    import torch
    b.validate_tensor(contexts, (64, 127), torch.int64)
    b.validate_tensor(targets, (64,), torch.int64)
    model.eval()
    if backward:
        model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    sums = []
    with (torch.enable_grad() if backward else torch.inference_mode()), \
            torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 64, 8):
            batch = contexts[start:start+8].to(device)
            last_logits = model(input_ids=batch, use_cache=False).logits[:, -1, :]
            loss = sampled_next_token_loss_sum(last_logits, targets[start:start+8].to(device))
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite hard-target loss')
            if backward:
                (loss / 64).backward()
            sums.append(float(loss.detach().cpu()))
            b.log(f'Seed {seed} {"gradient" if backward else "trial loss"} batch {start//8+1}/8')
    return math.fsum(sums) / 64


def step_scale(eligible, loss, initial_wnorm):
    snapshot, diagnostics = b.initial_scale(eligible, loss)
    if not math.isfinite(initial_wnorm) or initial_wnorm <= 0:
        raise ValueError('Invalid original F norm')
    proposed = RELATIVE_STEP * initial_wnorm
    diagnostics.update(Wnorm=initial_wnorm, target_step_norm=proposed,
                       eta_max=proposed / diagnostics['gradient_norm'])
    return snapshot, diagnostics


def verify_seed_start(model, eligible, selected, initial_state, frozen, gradient):
    import torch
    if b.state_hashes(eligible) != initial_state:
        raise ValueError('Seed did not begin from exact original F eligible state')
    if gradient.frozen_parameter_hashes(model, selected, torch) != frozen:
        raise ValueError('Seed did not begin from exact original F noneligible state')
    gradient.verify_frozen_parameters(model, selected, frozen, torch)


def train_seed(model, eligible, selected, originals, initial_wnorm, frozen,
               contexts, targets, seed, gradient):
    import torch
    records = []
    for step in range(1, STEPS+1):
        b.log(f'Seed {seed}/7 accepted training step {step}/4 starting')
        before = b.state_hashes(eligible)
        loss = hard_loss(model, contexts, targets, seed, backward=True)
        snapshot, diagnostics = step_scale(eligible, loss, initial_wnorm)
        accepted = b.armijo_step(eligible, snapshot, diagnostics,
            lambda: hard_loss(model, contexts, targets, seed))
        cumulative = b.norm(module.weight.detach().cpu().double()-originals[name].double()
                            for name, module in eligible)
        gradient.verify_frozen_parameters(model, selected, frozen, torch)
        records.append({**accepted, 'step': step, 'eligible_before': before,
            'eligible_after': b.state_hashes(eligible), 'cumulative_displacement_norm': cumulative,
            'cumulative_displacement_relative_to_initial_Wnorm': cumulative/initial_wnorm})
    return records


def combine_responses(responses):
    import torch
    if list(responses) != list(SEED_NAMES):
        raise ValueError('Expected all eight seed endpoints in fixed order')
    for value in responses.values():
        b.validate_tensor(value, (128, 2048), torch.float32)
    raw = torch.stack(list(responses.values())).double()
    norms = torch.linalg.vector_norm(raw, dim=2)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError('Undefined per-position unit response: zero/nonfinite seed norm')
    unit = raw / norms.unsqueeze(2)
    mean = raw.mean(dim=0).float().contiguous()
    consensus64 = unit.mean(dim=0)
    consensus = consensus64.float().contiguous()
    concentration = torch.linalg.vector_norm(consensus64, dim=1)
    pairs = {f'{i}_{j}': (unit[i]*unit[j]).sum(dim=1).tolist()
             for i in SEEDS for j in range(i+1, 8)}
    geometry = {'pairwise_seed_response_cosines_all_128_positions': pairs,
        'consensus_concentration_all_128_positions': concentration.tolist(),
        'seed_position_norms': {name: norms[i].tolist() for i, name in enumerate(SEED_NAMES)},
        'seed_global_norms': {name: float(torch.linalg.vector_norm(raw[i])) for i, name in enumerate(SEED_NAMES)},
        'raw_mean_position_norms': torch.linalg.vector_norm(mean.double(), dim=1).tolist(),
        'raw_mean_global_norm': float(torch.linalg.vector_norm(mean.double())),
        'final_renormalization': False, 'equal_seed_weight': 1/8}
    return mean, consensus, geometry


def validate_state(state):
    matrices = state['per_matrix_sha256']
    if len(matrices) != 98 or list(matrices) != sorted(matrices):
        raise ValueError('Eligible state inventory mismatch')
    digest = hashlib.sha256()
    for name, value in matrices.items():
        if len(value) != 64 or not name.endswith('.weight') or not 0 <= int(name.split('.')[2]) <= 13:
            raise ValueError('Invalid eligible matrix hash/name')
        digest.update(name.encode()+b'\0'+value.encode('ascii')+b'\n')
    if digest.hexdigest() != state['aggregate_sha256']:
        raise ValueError('Eligible aggregate hash mismatch')


def validate_trajectories(manifest):
    initial = manifest['initial_eligible_state']
    validate_state(initial)
    wnorm = manifest['initial_Wnorm']
    if not math.isfinite(wnorm) or wnorm <= 0 or list(manifest['seeds']) != [str(i) for i in SEEDS]:
        raise ValueError('Invalid original norm or seed inventory')
    for seed in SEEDS:
        record = manifest['seeds'][str(seed)]
        if record['initial_eligible_state'] != initial or len(record['steps']) != STEPS:
            raise ValueError('Seed reset or exact four-step inventory mismatch')
        previous = initial
        for index, step in enumerate(record['steps'], start=1):
            trials = step['trials']
            if (step['step'] != index or step['accepted_updates'] != 1 or step['Wnorm'] != wnorm or
                    step['target_step_norm'] != RELATIVE_STEP*wnorm or step['gradient_norm'] <= 0 or
                    step['eta_max'] != step['target_step_norm']/step['gradient_norm'] or
                    step['eligible_before'] != previous or not 1 <= len(trials) <= 8 or
                    step['accepted_eta'] != trials[-1]['eta'] or
                    step['accepted_post_step_ce'] != trials[-1]['ce'] or
                    step['displacement_norm'] < 0 or step['cumulative_displacement_norm'] < 0 or
                    step['displacement_relative_to_initial_Wnorm'] != step['displacement_norm']/wnorm or
                    step['cumulative_displacement_relative_to_initial_Wnorm'] != step['cumulative_displacement_norm']/wnorm):
                raise ValueError('Frozen descendant step mismatch')
            numeric = ('initial_ce', 'gradient_norm', 'eta_max', 'accepted_eta', 'accepted_post_step_ce',
                       'displacement_norm', 'cumulative_displacement_norm')
            if any(not math.isfinite(step[k]) for k in numeric):
                raise ValueError('Nonfinite step diagnostics')
            for trial_index, trial in enumerate(trials):
                eta = step['eta_max']*.5**trial_index
                rhs = step['initial_ce'] - 1e-4*eta*step['gradient_norm']**2
                if (trial['trial_index'] != trial_index or trial['eta'] != eta or trial['armijo_rhs'] != rhs or
                        not math.isfinite(trial['ce']) or trial['accepted'] != (trial['ce'] <= rhs) or
                        trial['accepted'] != (trial_index == len(trials)-1)):
                    raise ValueError('Frozen Armijo trial mismatch')
            validate_state(step['eligible_after'])
            if set(step['eligible_after']['per_matrix_sha256']) != set(initial['per_matrix_sha256']):
                raise ValueError('Eligible matrix inventory changed')
            previous = step['eligible_after']
        if record['final_eligible_state'] != previous or record['noneligible_unchanged'] is not True:
            raise ValueError('Final eligible/noneligible state mismatch')


def validate_recorded_state_hashes(manifest):
    """Check recorded provenance shape only; no descendant/model recomputation."""
    def valid_sha(value):
        return isinstance(value, str) and len(value) == 64 and set(value) <= set('0123456789abcdef')

    frozen = manifest.get('noneligible_F_parameter_sha256')
    eligible = manifest['initial_eligible_state']['per_matrix_sha256']
    if (not isinstance(frozen, dict) or not frozen or
            any(not isinstance(name, str) or not name or name in eligible or not valid_sha(value)
                for name, value in frozen.items())):
        raise ValueError('Invalid recorded noneligible F parameter hash inventory')
    for seed in SEEDS:
        if not valid_sha(manifest['seeds'][str(seed)].get('descendant_mean_float64_sha256')):
            raise ValueError('Invalid recorded descendant mean float64 SHA256 for seed ' + str(seed))


def validate_frozen(spec):
    """Blind-only audit; the privileged evaluator must complete this first."""
    import torch
    if spec != load_spec():
        raise ValueError('Freeze audit spec mismatch')
    hashes, inventory = blind_inputs(spec)
    root = b.path_of(spec['paths']['artifact_dir'])
    manifest_path = b.path_of(spec['paths']['construction_manifest'])
    receipt = json.loads((root/'freeze-receipt.json').read_text())
    b.require_hash(manifest_path, receipt['construction_manifest_sha256'])
    manifest = json.loads(manifest_path.read_text())
    if (manifest['attempt_id'] != ATTEMPT or manifest['spec_sha256'] != SPEC_SHA256 or
            manifest['constructor_sha256'] != b.sha256_file(Path(__file__)) or
            receipt['spec_sha256'] != SPEC_SHA256 or receipt['constructor_sha256'] != manifest['constructor_sha256'] or
            manifest['blind_input_hashes'] != hashes or manifest['barrier'] != MARKER or
            manifest['final_checkpoint_files'] != inventory['files']):
        raise ValueError('Frozen Attempt201 provenance mismatch')
    validate_trajectories(manifest)
    validate_recorded_state_hashes(manifest)
    b.require_hash(root/'blind-data.pt', manifest['blind_data']['serialized_sha256'])
    data = torch.load(root/'blind-data.pt', map_location='cpu', weights_only=True)
    shapes = {'corpus': ((4096, 128), torch.int64), 'probe': ((10000, 128), torch.int64),
              'contexts': ((64, 127), torch.int64), 'targets': ((8, 64), torch.int64),
              'probabilities': ((64, manifest['vocabulary_size']), torch.float64), 'F_mean': ((128, 2048), torch.float64)}
    if set(data) != set(shapes) or set(manifest['blind_data']['raw_sha256']) != set(shapes):
        raise ValueError('Blind data inventory mismatch')
    for name, (shape, dtype) in shapes.items():
        b.validate_tensor(data[name], shape, dtype)
        if b.raw_hash(data[name]) != manifest['blind_data']['raw_sha256'][name]:
            raise ValueError('Frozen blind data raw hash mismatch')
    if (b.raw_hash(data['corpus']) != spec['source']['raw_tensor_sha256'] or
            b.raw_hash(data['probe']) != spec['probe']['raw_tensor_sha256'] or
            not torch.equal(data['contexts'], data['corpus'][:64, :127])):
        raise ValueError('Frozen corpus/probe/context mismatch')
    expected_table = sample_table(data['probabilities'])
    if manifest['sample_table'] != expected_table or not torch.equal(data['targets'],
            torch.tensor([row['sampled_token_ids'] for row in expected_table['seeds']], dtype=torch.int64)):
        raise ValueError('Frozen new-domain sample table/targets mismatch')
    b.require_hash(root/'candidate.pt', manifest['candidate']['serialized_sha256'])
    candidates = torch.load(root/'candidate.pt', map_location='cpu', weights_only=True)
    if list(candidates) != list(NAMES) or list(manifest['candidate']['raw_sha256']) != list(NAMES):
        raise ValueError('Frozen candidate inventory mismatch')
    for name, value in candidates.items():
        b.validate_tensor(value, (128, 2048), torch.float32)
        if b.raw_hash(value) != manifest['candidate']['raw_sha256'][name]:
            raise ValueError('Frozen candidate raw hash mismatch')
    mean, consensus, geometry = combine_responses({name: candidates[name] for name in SEED_NAMES})
    if (not torch.equal(mean, candidates['response_raw_mean']) or
            not torch.equal(consensus, candidates['response_seed_consensus']) or manifest['geometry'] != geometry):
        raise ValueError('Frozen blind aggregation/geometry mismatch')
    b.validate_checkpoint(b.MODEL_DIR, inventory['files'])
    return candidates, data, manifest, receipt


def run(device='cuda'):
    import torch
    spec = load_spec()
    source_hash = b.sha256_file(Path(__file__))
    root = b.path_of(spec['paths']['artifact_dir'])
    manifest_path = b.path_of(spec['paths']['construction_manifest'])
    for path in (root, manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError('Output already exists: ' + str(path))
    input_hashes, inventory = blind_inputs(spec)
    b.validate_checkpoint(b.MODEL_DIR, inventory['files'])
    gradient, readout = helpers(spec)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    b.log('Loading pinned public non-streaming FineWeb for exact Attempt005/probe rehydration')
    tokenizer = AutoTokenizer.from_pretrained(b.MODEL_DIR, local_files_only=True)
    dataset = load_dataset(spec['dataset']['repo_id'], revision=spec['dataset']['revision'], split='train', streaming=False)
    sources, probe = b.rehydrate(dataset, tokenizer, intervals=((20000, 24096),), probe_count=10000)
    corpus = sources[0]
    if (b.raw_hash(corpus) != spec['source']['raw_tensor_sha256'] or
            b.raw_hash(probe) != spec['probe']['raw_tensor_sha256']):
        raise ValueError('Exact reconstructed corpus/probe hash mismatch')
    contexts = fixed_contexts(corpus)
    model, _ = gradient.load_local_model(b.MODEL_DIR, device, torch)
    model.config.use_cache = False
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    initial_state = b.state_hashes(eligible)
    frozen = gradient.frozen_parameter_hashes(model, selected, torch)
    originals = {name: module.weight.detach().cpu().clone().contiguous() for name, module in eligible}
    initial_wnorm = b.norm(originals.values())
    final_mean = b.mean_activation(model, probe, spec, readout, 'F')
    probabilities = sample_probabilities(model, contexts)
    table = sample_table(probabilities, progress=lambda seed:
        b.log(f'Sampling seed {seed}/7 complete: 64 hard targets'))
    targets = torch.tensor([row['sampled_token_ids'] for row in table['seeds']], dtype=torch.int64)
    gradient.verify_frozen_parameters(model, selected, frozen, torch)
    if b.state_hashes(eligible) != initial_state:
        raise ValueError('Sampling altered initial F')
    del model, eligible
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    responses, seed_records = {}, {}
    for seed in SEEDS:
        b.log(f'Seed {seed}/7: loading fresh exact original F')
        model, _ = gradient.load_local_model(b.MODEL_DIR, device, torch)
        model.config.use_cache = False
        eligible = gradient.discover_eligible_linear_weights(model, torch)
        selected = gradient.freeze_other_parameters(model, eligible)
        verify_seed_start(model, eligible, selected, initial_state, frozen, gradient)
        steps = train_seed(model, eligible, selected, originals, initial_wnorm, frozen,
                           contexts, targets[seed].contiguous(), seed, gradient)
        descendant_mean = b.mean_activation(model, probe, spec, readout, f'G seed {seed}')
        gradient.verify_frozen_parameters(model, selected, frozen, torch)
        responses[f'response_seed_{seed}'] = b.endpoint_response(final_mean, descendant_mean)
        seed_records[str(seed)] = {'initial_eligible_state': initial_state, 'steps': steps,
            'final_eligible_state': b.state_hashes(eligible), 'noneligible_unchanged': True,
            'descendant_mean_float64_sha256': b.raw_hash(descendant_mean)}
        del model, eligible
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    mean, consensus, geometry = combine_responses(responses)
    responses['response_raw_mean'] = mean
    responses['response_seed_consensus'] = consensus
    data = {'corpus': corpus, 'probe': probe, 'contexts': contexts, 'targets': targets,
            'probabilities': probabilities, 'F_mean': final_mean}
    if (input_hashes != blind_inputs(spec)[0] or b.sha256_file(Path(__file__)) != source_hash or
            b.sha256_file(SPEC_PATH) != SPEC_SHA256):
        raise ValueError('Blind inputs changed during construction')
    b.publish(root/'blind-data.pt', data, tensor=True)
    b.publish(root/'candidate.pt', responses, tensor=True)
    manifest = {'format_version': 1, 'attempt_id': ATTEMPT, 'spec_sha256': SPEC_SHA256,
        'constructor_sha256': source_hash, 'blind_input_hashes': input_hashes, 'final_checkpoint_files': inventory['files'],
        'initial_eligible_state': initial_state, 'initial_Wnorm': initial_wnorm,
        'noneligible_F_parameter_sha256': frozen, 'seeds': seed_records, 'sample_table': table,
        'vocabulary_size': probabilities.shape[1], 'geometry': geometry,
        'blind_data': {'serialized_sha256': b.sha256_file(root/'blind-data.pt'),
                      'raw_sha256': {name: b.raw_hash(value) for name, value in data.items()}},
        'candidate': {'serialized_sha256': b.sha256_file(root/'candidate.pt'),
                      'raw_sha256': {name: b.raw_hash(value) for name, value in responses.items()}},
        'barrier': MARKER, 'same_specimen_exploratory_diagnostic': True, 'held_out_validation': False,
        'runtime': {'torch': str(torch.__version__), 'device': device}}
    b.publish(manifest_path, manifest)
    b.publish(root/'freeze-receipt.json', {'construction_manifest_sha256': b.sha256_file(manifest_path),
        'spec_sha256': SPEC_SHA256, 'constructor_sha256': source_hash})
    validate_frozen(spec)
    b.log(MARKER)
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
