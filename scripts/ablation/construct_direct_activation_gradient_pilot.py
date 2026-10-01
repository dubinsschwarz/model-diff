#!/usr/bin/env python3
"""Attempt 026: blind direct block-13 activation-gradient pilot."""
import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '026_blind_direct_activation_gradient_block13_pilot'
HELPER_PATH = Path(__file__).with_name('construct_generic_hessian_krylov.py')
HELPER_SHA256 = '9edce220e1f9e29b4e02082852d03aa6b8043e35964a546383a3c10111018f44'
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA256:
    raise ValueError('Frozen generic final-model helper changed')
loader = importlib.util.spec_from_file_location('attempt026_generic_helper', HELPER_PATH)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)

AGGREGATE_KEYS = ('mean_gradient', 'first16_mean_gradient', 'last16_mean_gradient',
                  'even_mean_gradient', 'odd_mean_gradient')
FORBIDDEN_MARKERS = ('models/base', 'oracle', 'oracle_adl', 'adapter', 'true_delta',
                     'known_base', 'attempt015', 'attempt016', 'attempt018',
                     'attempt020', 'attempt021', 'attempt025', 'evaluation.json')
FROZEN_SPEC = {
    'format_version': 1,
    'attempt_id': ATTEMPT,
    'purpose': 'blind_direct_block13_activation_gradient_visibility_pilot',
    'information_policy': 'canonical_final_checkpoint_plus_frozen_attempt005_fineweb_only',
    'canonical_checkpoint_files': a.FROZEN_SPEC['canonical_checkpoint_files'],
    'frozen_corpus': a.FROZEN_SPEC['frozen_corpus'],
    'model': {'model_type': 'qwen3', 'dtype': 'torch.float32', 'eval_mode': True,
              'use_cache': False, 'autocast': False, 'all_parameters_frozen': True,
              'optimizer': False, 'parameter_update': False},
    'batches': {'microbatch_indices': list(range(32)), 'batch_size': 8,
                'example_start': 0, 'example_end_exclusive': 256,
                'sequence_length': 128, 'prediction_tokens_per_sequence': 127,
                'prediction_tokens_per_batch': 1016,
                'row_rule': 'batch_j_rows_[8*j,8*(j+1))',
                'smoke_microbatch_indices': [0]},
    'readout': {'block_index': 13, 'hidden_state_index': 14,
                'sequence_length': 128, 'hidden_size': 2048,
                'intervention': 'h.detach()+zero_leaf_z_broadcast_across_8_examples',
                'z_shape': [1, 128, 2048], 'z_dtype': 'torch.float32',
                'preserve_layer_output_container_and_other_fields': True,
                'only_autograd_target': 'z', 'gradient_sign': 'positive_dL_dz',
                'loss': 'sum_causal_next_token_CE_divided_by_1016',
                'no_prefix_backward_graph': True},
    'aggregation': {'batch_gradient_shape': [1, 128, 2048],
                    'cpu_accumulation_dtype': 'torch.float64',
                    'stored_dtype': 'torch.float32',
                    'all32': list(range(32)), 'first16': list(range(16)),
                    'last16': list(range(16, 32)),
                    'even': list(range(0, 32, 2)),
                    'odd': list(range(1, 32, 2)),
                    'normalization': 'mean_of_batch_mean_loss_gradients',
                    'candidate_scaling': False, 'candidate_sign_selection': False},
    'position_policy': {'future_primary_positions': [1, 2, 3, 4],
                        'construction_selection': False,
                        'reported_position_range': [0, 127],
                        'final_position': 127,
                        'final_position_maxabs_absolute_limit': 1e-7,
                        'final_position_norm_absolute_limit': 1e-6,
                        'final_position_relative_limit': 1e-5,
                        'relative_scale': 'maximum_absolute_or_L2_norm_over_positions_0_through_126',
                        'audit': 'every_batch_and_stored_all32_mean',
                        'fail_if_either_absolute_and_relative_limits_exceeded': True},
    'diagnostics': {'split_cosines': ['per_position_first16_vs_last16',
                                     'per_position_even_vs_odd',
                                     'flattened_positions1_4_first16_vs_last16',
                                     'flattened_positions1_4_even_vs_odd'],
                    'norms': 'all_positions_and_positions_1_4_of_mean_gradient',
                    'zero_norm_cosine': None,
                    'mean_generic_loss': 'arithmetic_mean_of_32_equal_token_count_batch_losses'},
    'interpretation': 'Positive mean dL/dz is the candidate historical-bias orientation; a small generic-loss-lowering correction is negative eta times this vector.',
    'sources': {'scripts/ablation/construct_generic_hessian_krylov.py': HELPER_SHA256},
    'defaults': {
        'final_directory': '/root/model-diff-scratch/models/merged',
        'corpus_spec_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json',
        'corpus_manifest_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json',
        'tokens_path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
        'artifact_path': '/root/model-diff-scratch/artifacts/attempt026_direct_activation_gradient/activation_gradient.pt',
        'manifest_path': 'experiments/attempts/'+ATTEMPT+'/construction-manifest.json',
    },
    'outputs': {'artifact_tensors': list(AGGREGATE_KEYS)+['batch_indices'],
                'overwrite': False, 'timestamps': False,
                'host_metadata': False, 'gpu_metadata': False,
                'per_batch_gradients_persisted': False},
}


def batch_inventory(smoke_only=False):
    return [(j, 8*j, 8*(j+1)) for j in range(1 if smoke_only else 32)]


def freeze_model(model):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameter gradient existed before construction')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameters could not be frozen')


def intervene_output(output, batch_size, readout):
    """Detach only the block hidden tensor, preserving output structure."""
    if isinstance(output, torch.Tensor):
        hidden = output
        kind = 'tensor'
    elif isinstance(output, tuple) and type(output) is tuple and output:
        hidden, kind = output[0], 'tuple'
    elif isinstance(output, list) and output:
        hidden, kind = output[0], 'list'
    else:
        raise ValueError('Unsupported block output container')
    expected = (batch_size, readout['sequence_length'], readout['hidden_size'])
    if (not isinstance(hidden, torch.Tensor) or tuple(hidden.shape) != expected or
            hidden.dtype != torch.float32 or not bool(torch.isfinite(hidden).all())):
        raise ValueError('Invalid block-13 hidden output')
    z = torch.zeros((1, readout['sequence_length'], readout['hidden_size']),
                    device=hidden.device, dtype=torch.float32, requires_grad=True)
    changed = hidden.detach()+z
    if not torch.equal(changed, hidden.detach()):
        raise ValueError('Zero activation intervention changed forward values')
    if kind == 'tensor':
        return changed, z
    if kind == 'tuple':
        return (changed, *output[1:]), z
    return [changed, *output[1:]], z


def activation_gradient_batch(model, batch, readout):
    if batch.dtype != torch.int64 or tuple(batch.shape) != (8, 128):
        raise ValueError('Expected frozen 8 x 128 int64 microbatch')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameter gradients must be disabled')
    layers = getattr(getattr(model, 'model', None), 'layers', None)
    if layers is None or not 0 <= readout['block_index'] < len(layers):
        raise ValueError('Missing block-13 transformer layer')
    leaves = []
    def hook(_module, _inputs, output):
        changed, z = intervene_output(output, batch.shape[0], readout)
        leaves.append(z)
        return changed
    handle = layers[readout['block_index']].register_forward_hook(hook)
    try:
        with torch.enable_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
            logits = model(input_ids=batch, use_cache=False).logits
            if len(leaves) != 1:
                raise ValueError('Block-13 intervention hook count mismatch')
            loss = a.causal_token_loss_sum(logits, batch, torch)/1016
            if not bool(torch.isfinite(loss).item()):
                raise ValueError('Nonfinite generic mean loss')
            gradient = torch.autograd.grad(loss, leaves[0])[0]
            if (gradient is None or tuple(gradient.shape) != (1, readout['sequence_length'],
                readout['hidden_size']) or gradient.dtype != torch.float32 or
                    not bool(torch.isfinite(gradient).all())):
                raise ValueError('Invalid activation gradient')
            cpu_gradient = gradient.detach().squeeze(0).to(device='cpu', dtype=torch.float64)
            loss_value = float(loss.detach().to(device='cpu', dtype=torch.float64))
            del logits, loss, gradient
    finally:
        handle.remove()
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameter received a gradient')
    return loss_value, cpu_gradient.contiguous()


def final_position_audit(gradient, policy):
    if (gradient.ndim != 2 or gradient.shape[0] != 128 or
            gradient.device.type != 'cpu' or gradient.dtype not in (torch.float64, torch.float32) or
            not bool(torch.isfinite(gradient).all())):
        raise ValueError('Invalid activation gradient for causal audit')
    final = gradient[127].double()
    earlier = gradient[:127].double()
    norm = float(torch.linalg.vector_norm(final))
    maximum = float(final.abs().max())
    earlier_norm = float(torch.linalg.vector_norm(earlier, dim=1).max())
    earlier_maximum = float(earlier.abs().max())
    norm_limit = max(policy['final_position_norm_absolute_limit'],
                     policy['final_position_relative_limit']*earlier_norm)
    max_limit = max(policy['final_position_maxabs_absolute_limit'],
                    policy['final_position_relative_limit']*earlier_maximum)
    if norm > norm_limit or maximum > max_limit:
        raise ValueError('Causal final-position activation gradient is not negligible')
    return {'position127_norm': norm, 'position127_maxabs': maximum,
            'norm_limit': norm_limit, 'maxabs_limit': max_limit}


class GradientAccumulator:
    def __init__(self, readout):
        self.shape = (readout['sequence_length'], readout['hidden_size'])
        self.sums = {name: torch.zeros(self.shape, dtype=torch.float64)
                     for name in AGGREGATE_KEYS}
        self.losses = []
        self.indices = []
        self.causal_audits = []

    def add(self, batch_index, loss, gradient, policy):
        if (batch_index != len(self.indices) or not 0 <= batch_index < 32 or
                gradient.shape != self.shape or gradient.dtype != torch.float64 or
                gradient.device.type != 'cpu' or not gradient.is_contiguous() or
                not bool(torch.isfinite(gradient).all()) or not math.isfinite(loss)):
            raise ValueError('Invalid ordered CPU float64 batch gradient')
        audit = final_position_audit(gradient, policy)
        self.sums['mean_gradient'].add_(gradient)
        self.sums['first16_mean_gradient' if batch_index < 16 else
                  'last16_mean_gradient'].add_(gradient)
        self.sums['even_mean_gradient' if batch_index % 2 == 0 else
                  'odd_mean_gradient'].add_(gradient)
        self.losses.append(float(loss))
        self.indices.append(batch_index)
        self.causal_audits.append(audit)

    def averages(self, expected_count):
        if expected_count not in (1, 32) or len(self.indices) != expected_count:
            raise ValueError('Incorrect frozen batch count for aggregation')
        if expected_count != 32:
            return {'mean_gradient': (self.sums['mean_gradient']/expected_count).float().contiguous()}
        return {name: (value/(32 if name == 'mean_gradient' else 16)).float().contiguous()
                for name, value in self.sums.items()}

    def mean_loss(self):
        if not self.losses:
            raise ValueError('No generic losses accumulated')
        return math.fsum(self.losses)/len(self.losses)


def cosine(left, right):
    x, y = left.double().reshape(-1), right.double().reshape(-1)
    xx, yy = float(torch.dot(x, x)), float(torch.dot(y, y))
    if xx == 0 or yy == 0:
        return None
    return float(torch.dot(x, y))/math.sqrt(xx*yy)


def split_diagnostics(averages):
    if set(averages) != set(AGGREGATE_KEYS):
        raise ValueError('Missing frozen activation-gradient aggregate')
    for value in averages.values():
        if (value.shape != (128, 2048) or value.dtype != torch.float32 or
                value.device.type != 'cpu' or not value.is_contiguous() or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid stored activation-gradient aggregate')
    mean = averages['mean_gradient']
    first, last = averages['first16_mean_gradient'], averages['last16_mean_gradient']
    even, odd = averages['even_mean_gradient'], averages['odd_mean_gradient']
    norms = torch.linalg.vector_norm(mean.double(), dim=1).tolist()
    return {
        'first16_vs_last16_cosine_by_position': [cosine(first[j], last[j]) for j in range(128)],
        'even_vs_odd_cosine_by_position': [cosine(even[j], odd[j]) for j in range(128)],
        'first16_vs_last16_cosine_positions1_4': cosine(first[1:5], last[1:5]),
        'even_vs_odd_cosine_positions1_4': cosine(even[1:5], odd[1:5]),
        'mean_gradient_norm_by_position': norms,
        'mean_gradient_norm_positions1_4': norms[1:5],
        'position127_norm': norms[127],
        'position127_maxabs': float(mean[127].abs().max()),
    }


def raw_tensor_sha256(value):
    if value.device.type != 'cpu' or not value.is_contiguous():
        raise ValueError('Raw tensor hash requires contiguous CPU tensor')
    if value.dtype == torch.float32:
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite float32 artifact tensor')
        dtype = '<f4'
    elif value.dtype == torch.int64:
        dtype = '<i8'
    else:
        raise ValueError('Invalid raw tensor dtype')
    return hashlib.sha256(value.numpy().astype(dtype, copy=False).tobytes(order='C')).hexdigest()


def verify_artifact(path, expected=None):
    try:
        artifact = torch.load(path, map_location='cpu', weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as exc:
        raise ValueError('Could not load Attempt026 artifact') from exc
    if not isinstance(artifact, dict) or set(artifact) != set(AGGREGATE_KEYS) | {'batch_indices'}:
        raise ValueError('Activation-gradient artifact inventory mismatch')
    tensors = {}
    for name, value in artifact.items():
        shape = (32,) if name == 'batch_indices' else (128, 2048)
        dtype = torch.int64 if name == 'batch_indices' else torch.float32
        if (not isinstance(value, torch.Tensor) or value.shape != shape or
                value.dtype != dtype or value.device.type != 'cpu' or not value.is_contiguous()):
            raise ValueError('Activation-gradient artifact tensor mismatch')
        raw_tensor_sha256(value)
        tensors[name] = {'shape': list(shape), 'dtype': str(dtype),
                         'device': 'cpu', 'contiguous': True}
    if not torch.equal(artifact['batch_indices'], torch.arange(32, dtype=torch.int64)):
        raise ValueError('Activation-gradient batch inventory mismatch')
    record = {'path': str(path), 'serialized_sha256': a.sha256_file(path),
              'raw_tensor_sha256': {name: raw_tensor_sha256(value)
                                    for name, value in artifact.items()},
              'tensors': tensors}
    if expected is not None and record != expected:
        raise ValueError('Activation-gradient artifact hash/inventory mismatch')
    return artifact, record


def require_outputs_absent(args):
    for path in (args.artifact_path, args.manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError('Scientific output already exists')


def save_artifact(path, averages):
    if path.exists() or path.is_symlink():
        raise ValueError('Scientific output already exists')
    artifact = {**averages, 'batch_indices': torch.arange(32, dtype=torch.int64)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(artifact, stream)
    return verify_artifact(path)[1]


def firewall_and_lock_paths(args):
    for key, value in vars(args).items():
        if isinstance(value, (Path, str)) and any(marker in str(value).lower()
                                                   for marker in FORBIDDEN_MARKERS):
            raise ValueError('Blind constructor information firewall: '+key)
    for key, frozen in FROZEN_SPEC['defaults'].items():
        expected = Path(frozen)
        if not expected.is_absolute():
            expected = PROJECT/expected
        if getattr(args, key).resolve() != expected.resolve():
            raise ValueError('Attempt026 path differs from frozen '+key)
    expected_spec = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
    if args.spec_path.resolve() != expected_spec.resolve():
        raise ValueError('Attempt026 spec path differs from frozen path')


def validate_inputs(args):
    firewall_and_lock_paths(args)
    require_outputs_absent(args)
    spec = a.load_json_object(args.spec_path, 'Attempt026 spec')
    if spec != FROZEN_SPEC:
        raise ValueError('Attempt026 frozen specification mismatch')
    paths = [args.spec_path, args.corpus_spec_path, args.corpus_manifest_path,
             args.tokens_path, Path(__file__).resolve(), HELPER_PATH]
    hashes = {path: a.sha256_file(path) for path in paths}
    if hashes[HELPER_PATH] != HELPER_SHA256:
        raise ValueError('Frozen generic helper source mismatch')
    for field, path in (('corpus_spec_sha256', args.corpus_spec_path),
                        ('corpus_manifest_sha256', args.corpus_manifest_path)):
        if hashes[path] != spec['frozen_corpus'][field]:
            raise ValueError('Attempt005 corpus provenance mismatch')
    if a.checkpoint_file_records(args.final_directory) != spec['canonical_checkpoint_files']:
        raise ValueError('Canonical final checkpoint inventory mismatch')
    corpus = a.verify_corpus_manifest(args.corpus_manifest_path, args.corpus_spec_path,
        args.tokens_path, args.final_directory, a.load_corpus_spec(args.corpus_spec_path))
    if any(corpus[key] != spec['frozen_corpus'][key]
           for key in ('serialized_sha256', 'raw_tensor_sha256')):
        raise ValueError('Attempt005 token artifact provenance mismatch')
    tokens = a.load_corpus_tokens(args.tokens_path, corpus['raw_tensor_sha256'], 4096, 128, torch)
    output_paths = (args.artifact_path.resolve(), args.manifest_path.resolve())
    if (any(output.is_relative_to(args.final_directory.resolve()) for output in output_paths) or
            any(output == path.resolve() or path.resolve().is_relative_to(output)
                for output in output_paths for path in paths) or
            output_paths[0] == output_paths[1]):
        raise ValueError('Scientific output overlaps a frozen input')
    def recheck():
        if a.checkpoint_file_records(args.final_directory) != spec['canonical_checkpoint_files']:
            raise ValueError('Final checkpoint changed during construction')
        for path, digest in hashes.items():
            if a.sha256_file(path) != digest:
                raise ValueError('Frozen input changed during construction: '+str(path))
    return spec, tokens, hashes, recheck


def construct(args):
    spec, tokens, hashes, recheck = validate_inputs(args)
    model = a.load_local_model(args.final_directory, args.device, torch)
    freeze_model(model)
    before = a.model_state_hashes(model, torch)
    device = next(model.parameters()).device
    accumulator = GradientAccumulator(spec['readout'])
    elapsed = None
    for j, start, end in batch_inventory(args.smoke_only):
        batch = tokens[start:end].to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss, gradient = activation_gradient_batch(model, batch, spec['readout'])
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-started
        accumulator.add(j, loss, gradient, spec['position_policy'])
        del batch, gradient
        if not args.smoke_only and (j+1) % 8 == 0:
            print(f'Direct activation-gradient batches {j+1}/32', flush=True)
    a.verify_model_unchanged(model, before, torch)
    del model
    recheck()
    averages = accumulator.averages(1 if args.smoke_only else 32)
    mean_audit = final_position_audit(averages['mean_gradient'], spec['position_policy'])
    if args.smoke_only:
        norms = torch.linalg.vector_norm(averages['mean_gradient'][1:5].double(), dim=1)
        print(json.dumps({'smoke_only': True, 'microbatch_index': 0,
                          'loss': accumulator.mean_loss(),
                          'norms_positions1_4': norms.tolist(),
                          'position127_norm': mean_audit['position127_norm'],
                          'position127_maxabs': mean_audit['position127_maxabs'],
                          'elapsed_seconds': elapsed}, sort_keys=True, allow_nan=False))
        return
    diagnostics = split_diagnostics(averages)
    artifact = save_artifact(args.artifact_path, averages)
    manifest = {'format_version': 1, 'attempt_id': ATTEMPT,
                'purpose': spec['purpose'], 'information_policy': spec['information_policy'],
                'hash_algorithm': 'sha256',
                'input_source_hashes': {str(path): digest for path, digest in hashes.items()},
                'canonical_checkpoint_files': spec['canonical_checkpoint_files'],
                'frozen_corpus': spec['frozen_corpus'], 'model': spec['model'],
                'batches': spec['batches'], 'readout': spec['readout'],
                'aggregation': spec['aggregation'], 'position_policy': spec['position_policy'],
                'mean_generic_loss': accumulator.mean_loss(),
                'diagnostics': diagnostics,
                'causal_audit': {'maximum_batch_position127_norm': max(
                    row['position127_norm'] for row in accumulator.causal_audits),
                    'maximum_batch_position127_maxabs': max(
                    row['position127_maxabs'] for row in accumulator.causal_audits),
                    'mean_gradient': mean_audit},
                'artifact': artifact,
                'historical_base_access': False, 'oracle_access': False,
                'adapter_access': False, 'sign_selection': False,
                'candidate_rescaling': False, 'post_result_position_selection': False,
                'optimizer_used': False, 'model_parameter_update': False,
                'publication_policy': 'no_overwrite_no_resume_partial_artifact_blocks_rerun'}
    recheck()
    _, verified = verify_artifact(args.artifact_path, artifact)
    if verified != artifact:
        raise ValueError('Stored activation-gradient artifact changed')
    if args.manifest_path.exists() or args.manifest_path.is_symlink():
        raise ValueError('Scientific output already exists')
    a.write_manifest(args.manifest_path, manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path', type=Path,
                        default=PROJECT/'experiments/attempts'/ATTEMPT/'spec.json')
    for key, value in FROZEN_SPEC['defaults'].items():
        path = Path(value)
        parser.add_argument('--'+key.replace('_', '-'), type=Path,
                            default=path if path.is_absolute() else PROJECT/path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-only', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    construct(parse_args())
