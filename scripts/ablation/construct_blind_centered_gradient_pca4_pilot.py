#!/usr/bin/env python3
"""Attempt 023: blind centered-gradient PCA4 and held-out Hessian regression."""
import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
HELPER_PATH = Path(__file__).with_name('construct_generic_hessian_krylov.py')
HELPER_SHA256 = '9edce220e1f9e29b4e02082852d03aa6b8043e35964a546383a3c10111018f44'
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA256:
    raise ValueError('Frozen generic source helper hash mismatch')
loader = importlib.util.spec_from_file_location('attempt023_generic_io_helper', HELPER_PATH)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)

MATRIX_SHAPES = {
    'mlp.down_proj.weight': [2048, 6144],
    'mlp.gate_proj.weight': [6144, 2048],
    'mlp.up_proj.weight': [6144, 2048],
    'self_attn.k_proj.weight': [1024, 2048],
    'self_attn.o_proj.weight': [2048, 2048],
    'self_attn.q_proj.weight': [2048, 2048],
    'self_attn.v_proj.weight': [1024, 2048],
}
COORDINATES = sorted(
    [{'name': f'model.layers.{block}.{relative}', 'shape': shape}
     for block in range(14) for relative, shape in MATRIX_SHAPES.items()],
    key=lambda row: row['name'])

FROZEN_SPEC = {
    'format_version': 1,
    'attempt_id': '023_blind_centered_gradient_pca4_regression_pilot',
    'purpose': 'blind_final_checkpoint_centered_gradient_pca4_regression_pilot',
    'information_policy': 'final_checkpoint_plus_frozen_fineweb_only',
    'final_checkpoint_files': a.FROZEN_SPEC['canonical_checkpoint_files'],
    'frozen_corpus': a.FROZEN_SPEC['frozen_corpus'],
    'coordinates': COORDINATES,
    'batches': {'basis_indices': list(range(0, 512, 32)),
                'regression_indices': list(range(16, 512, 32)),
                'rows_per_batch': 8, 'sequence_length': 128,
                'prediction_tokens_per_batch': 1016,
                'row_rule': 'start=8*j,end=8*(j+1),end_exclusive'},
    'pca': {'snapshot_count': 16, 'dimension': 4,
            'snapshot_dtype': 'CPU_float32', 'gram_dtype': 'CPU_float64',
            'centering': 'C=I-11T/16;K=(CGC+(CGC).T)/2',
            'eigen_solver': 'torch.linalg.eigh_CPU_float64',
            'order': 'four_largest_descending',
            'sign': 'largest_absolute_component_positive_first_index_on_tie',
            'near_degeneracy_relative_gap': 1e-6,
            'near_degeneracy_absolute_gap': 1e-10,
            'near_degeneracy_pairs': 'adjacent_within_top4_and_fourth_vs_fifth',
            'q_arithmetic': 'matrixwise_CPU_float64_sum_then_one_FP32_cast',
            'q_gram_max_off_diagonal': 1e-4,
            'q_gram_max_diagonal_deviation': 1e-4,
            'snapshot_persistence': False, 'q_persistence': False},
    't_space': {'dimension': 4, 't_leaf_dtype': 'torch.float64',
                'weight_arithmetic': 'FP32(W1_weight+sum_j FP32(t_j)*realized_FP32_q_j)',
                'evaluation': 'torch.func.functional_call',
                'only_autograd_input': 't', 'primitive': 'true_double_backward',
                'raw_hessian_relative_asymmetry_limit': 1e-5,
                'relative_asymmetry_denominator_floor': 1e-6,
                'symmetrization': '(M_raw+M_raw.T)/2', 'fallback': False},
    'regression': {'driver': 'gelsd', 'rcond': 1e-12,
                   'dtype': 'CPU_float64', 'rank_required': 4,
                   'regularization': None, 'selection': False,
                   'splits': ['first_8_vs_last_8', 'even_members_vs_odd_members']},
    'smoke': {'basis_microbatch_indices': [0, 32],
              'gradient_snapshots': 2, 'gram_shape': [2, 2],
              'pca': False, 'regression': False, 'scientific_output': False},
    'sources': {'scripts/ablation/construct_generic_hessian_krylov.py': HELPER_SHA256},
    'defaults': {
        'final_directory': '/root/model-diff-scratch/models/merged',
        'corpus_spec_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json',
        'corpus_manifest_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json',
        'tokens_path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
        'artifact_path': '/root/model-diff-scratch/artifacts/attempt023_centered_gradient_pca4/stats.pt',
        'manifest_path': 'experiments/attempts/023_blind_centered_gradient_pca4_regression_pilot/construction-manifest.json',
    },
    'outputs': {'overwrite': False, 'timestamps': False,
                'host_metadata': False, 'gpu_metadata': False},
}


def selected_batches():
    return {kind: [(j, 8*j, 8*(j+1)) for j in indices]
            for kind, indices in (('basis', FROZEN_SPEC['batches']['basis_indices']),
                                  ('regression', FROZEN_SPEC['batches']['regression_indices']))}


def firewall(args):
    forbidden = ('base', 'oracle', 'adapter', 'true_delta', 'known_base',
                 'attempt015', 'attempt016', 'attempt018', 'attempt020', 'attempt021',
                 'attempt022', '015_', '016_', '018_', '020_', '021_', '022_',
                 'evaluation.json')
    for key, value in vars(args).items():
        if isinstance(value, Path) and any(marker in str(value).lower() for marker in forbidden):
            raise ValueError('Blind construction information firewall: '+key)


def require_absent(args):
    for path in (args.artifact_path, args.manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError('Scientific output already exists')


def validate_inputs(args):
    firewall(args)
    spec = a.load_json_object(args.spec_path, 'Attempt023 frozen spec')
    if spec != FROZEN_SPEC:
        raise ValueError('Frozen Attempt023 specification mismatch')
    paths = [args.spec_path, args.corpus_spec_path, args.corpus_manifest_path,
             args.tokens_path, Path(__file__).resolve(), HELPER_PATH]
    hashes = {str(path): a.sha256_file(path) for path in paths}
    if hashes[str(HELPER_PATH)] != HELPER_SHA256:
        raise ValueError('Frozen generic helper changed')
    if a.checkpoint_file_records(args.final_directory) != spec['final_checkpoint_files']:
        raise ValueError('Canonical final checkpoint mismatch')
    for field, path in (('corpus_spec_sha256', args.corpus_spec_path),
                        ('corpus_manifest_sha256', args.corpus_manifest_path)):
        if hashes[str(path)] != spec['frozen_corpus'][field]:
            raise ValueError('Frozen FineWeb provenance mismatch')
    corpus = a.verify_corpus_manifest(args.corpus_manifest_path, args.corpus_spec_path,
        args.tokens_path, args.final_directory, a.load_corpus_spec(args.corpus_spec_path))
    if any(corpus[key] != spec['frozen_corpus'][key]
           for key in ('serialized_sha256', 'raw_tensor_sha256')):
        raise ValueError('Frozen FineWeb artifact mismatch')
    tokens = a.load_corpus_tokens(args.tokens_path, corpus['raw_tensor_sha256'],
                                  4096, 128, torch)
    frozen_paths = [path.resolve() for path in paths]
    for output in (args.artifact_path, args.manifest_path):
        resolved = output.resolve()
        if (resolved.is_relative_to(args.final_directory.resolve()) or
                any(resolved == path or path.is_relative_to(resolved) for path in frozen_paths)):
            raise ValueError('Scientific output overlaps frozen input')
    if (args.artifact_path.resolve().is_relative_to(args.manifest_path.resolve()) or
            args.manifest_path.resolve().is_relative_to(args.artifact_path.resolve())):
        raise ValueError('Scientific outputs overlap')
    def recheck():
        if a.checkpoint_file_records(args.final_directory) != spec['final_checkpoint_files']:
            raise ValueError('Final checkpoint changed during construction')
        for path, digest in hashes.items():
            if a.sha256_file(Path(path)) != digest:
                raise ValueError('Frozen construction input changed: '+path)
    return spec, tokens, hashes, recheck


def freeze_for_snapshots(model, names):
    selected = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    model.eval()
    model.zero_grad(set_to_none=True)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Unexpected parameter .grad buffer')


def freeze_for_t_space(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameter freeze failed')


def gradient_snapshot(model, batch, coordinates):
    if batch.dtype != torch.int64 or tuple(batch.shape) != (8, 128):
        raise ValueError('Invalid frozen microbatch')
    parameters = dict(model.named_parameters())
    names = [row['name'] for row in coordinates]
    if (len(names) != 98 or set(names) != {name for name, p in parameters.items() if p.requires_grad} or
            any(parameter.grad is not None for parameter in parameters.values())):
        raise ValueError('Snapshot gradient coordinate/freeze mismatch')
    with torch.enable_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
        logits = torch.func.functional_call(model, {}, (),
            {'input_ids': batch, 'use_cache': False}, tie_weights=True, strict=False).logits
        loss = a.causal_token_loss_sum(logits, batch, torch)/1016
        if not bool(torch.isfinite(loss).item()):
            raise ValueError('Nonfinite snapshot loss')
        gradients = torch.autograd.grad(loss, [parameters[name] for name in names])
    snapshot = {}
    for row, gradient in zip(coordinates, gradients):
        if (gradient is None or gradient.dtype != torch.float32 or
                list(gradient.shape) != row['shape'] or not bool(torch.isfinite(gradient).all())):
            raise ValueError('Invalid selected gradient snapshot')
        snapshot[row['name']] = gradient.detach().to(device='cpu', dtype=torch.float32).contiguous()
    if any(parameter.grad is not None for parameter in parameters.values()):
        raise ValueError('Snapshot populated parameter .grad')
    return snapshot


def snapshot_gram(snapshots, coordinates):
    count = len(snapshots)
    if count < 2 or any(set(snapshot) != {row['name'] for row in coordinates}
                        for snapshot in snapshots):
        raise ValueError('Gradient snapshot inventory mismatch')
    terms = [[[] for _ in range(count)] for _ in range(count)]
    for row in coordinates:
        name, shape = row['name'], row['shape']
        values = [snapshot[name] for snapshot in snapshots]
        if any(value.dtype != torch.float32 or value.device.type != 'cpu' or
               not value.is_contiguous() or list(value.shape) != shape or
               not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError('Invalid CPU FP32 gradient snapshot matrix')
        matrix = torch.stack([value.reshape(-1).double() for value in values])
        part = matrix@matrix.T
        if not bool(torch.isfinite(part).all()):
            raise ValueError('Nonfinite snapshot Gram contribution')
        for i in range(count):
            for j in range(count):
                terms[i][j].append(float(part[i, j]))
        del matrix, part
    gram = torch.tensor([[math.fsum(terms[i][j]) for j in range(count)]
                         for i in range(count)], dtype=torch.float64)
    if not bool(torch.isfinite(gram).all()):
        raise ValueError('Nonfinite gradient snapshot Gram')
    return gram.contiguous()


def centered_pca(gram, policy):
    if (gram.shape != (16, 16) or gram.dtype != torch.float64 or
            gram.device.type != 'cpu' or not bool(torch.isfinite(gram).all())):
        raise ValueError('Invalid raw gradient Gram')
    centering = torch.eye(16, dtype=torch.float64)-torch.ones((16, 16), dtype=torch.float64)/16
    raw_centered = centering@gram@centering
    K = ((raw_centered+raw_centered.T)/2).contiguous()
    if not bool(torch.isfinite(K).all()):
        raise ValueError('Nonfinite centered gradient Gram')
    eigenvalues, eigenvectors = torch.linalg.eigh(K)
    order = torch.arange(15, -1, -1)
    descending = eigenvalues[order]
    retained = descending[:4].contiguous()
    if not bool(torch.isfinite(eigenvalues).all()) or float(retained[-1]) <= 0:
        raise ValueError('PCA has nonpositive or nonfinite retained eigenvalue')
    for i in range(4):
        left, right = float(descending[i]), float(descending[i+1])
        threshold = max(policy['near_degeneracy_absolute_gap'],
                        policy['near_degeneracy_relative_gap']*max(abs(left), abs(right)))
        if left-right <= threshold:
            raise ValueError('Ambiguous centered-gradient PCA eigenvalue ordering')
    vectors = eigenvectors[:, order[:4]].contiguous()
    for j in range(4):
        pivot = int(torch.argmax(vectors[:, j].abs()))
        if float(vectors[pivot, j]) < 0:
            vectors[:, j].neg_()
    energy = float(torch.trace(K))
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError('Invalid centered-gradient energy')
    fractions = (retained/energy).tolist()
    return K, retained, vectors, {
        'retained_eigenvalues': retained.tolist(),
        'retained_energy_fractions': fractions,
        'top4_energy_fraction': math.fsum(fractions),
        'total_centered_energy': energy,
        'ordering': policy['order'], 'sign': policy['sign'],
        'near_degeneracy_relative_gap': policy['near_degeneracy_relative_gap'],
        'near_degeneracy_absolute_gap': policy['near_degeneracy_absolute_gap']}


def reconstruct_pca_basis(snapshots, coordinates, eigenvalues, eigenvectors):
    if (len(snapshots) != 16 or eigenvalues.shape != (4,) or
            eigenvectors.shape != (16, 4)):
        raise ValueError('Invalid PCA reconstruction inputs')
    coefficients = eigenvectors/torch.sqrt(eigenvalues).unsqueeze(0)
    basis = [dict() for _ in range(4)]
    for row in coordinates:
        name = row['name']
        accumulators = [torch.zeros(row['shape'], dtype=torch.float64) for _ in range(4)]
        for i, snapshot in enumerate(snapshots):
            value = snapshot[name].double()
            for j in range(4):
                accumulators[j].add_(value, alpha=float(coefficients[i, j]))
        for j in range(4):
            q = accumulators[j].to(dtype=torch.float32).contiguous()
            if not bool(torch.isfinite(q).all()):
                raise ValueError('Nonfinite reconstructed PCA basis')
            basis[j][name] = q
    return basis


def basis_gram(basis, coordinates, policy):
    if len(basis) != 4 or any(set(vector) != {row['name'] for row in coordinates}
                              for vector in basis):
        raise ValueError('PCA basis inventory mismatch')
    terms = [[[] for _ in range(4)] for _ in range(4)]
    for row in coordinates:
        values = [basis[j][row['name']] for j in range(4)]
        if any(value.dtype != torch.float32 or value.device.type != 'cpu' or
               not value.is_contiguous() or list(value.shape) != row['shape'] or
               not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError('Invalid PCA basis matrix')
        double = [value.double() for value in values]
        for i in range(4):
            for j in range(i, 4):
                terms[i][j].append(float(torch.sum(double[i]*double[j], dtype=torch.float64)))
    gram = torch.zeros((4, 4), dtype=torch.float64)
    for i in range(4):
        for j in range(i, 4):
            gram[i, j] = gram[j, i] = math.fsum(terms[i][j])
    off = gram-torch.diag(torch.diag(gram))
    off_error = float(off.abs().max())
    diag_error = float((torch.diag(gram)-1).abs().max())
    if (not bool(torch.isfinite(gram).all()) or
            off_error > policy['q_gram_max_off_diagonal'] or
            diag_error > policy['q_gram_max_diagonal_deviation']):
        raise ValueError('Reconstructed PCA basis Gram audit failed')
    return {'gram_matrix': gram.tolist(), 'max_off_diagonal': off_error,
            'max_diagonal_deviation': diag_error}


def audit_hessian_symmetry(raw, policy):
    if (raw.shape != (4, 4) or raw.dtype != torch.float64 or
            raw.device.type != 'cpu' or not bool(torch.isfinite(raw).all())):
        raise ValueError('Invalid raw t-space Hessian')
    absolute = float((raw-raw.T).abs().max())
    relative = absolute/max(float(raw.abs().max()),
                            policy['relative_asymmetry_denominator_floor'])
    if relative > policy['raw_hessian_relative_asymmetry_limit']:
        raise ValueError('t-space Hessian asymmetry exceeds frozen tolerance')
    return ((raw+raw.T)/2).contiguous(), {'absolute': absolute, 'relative': relative}


def t_space_batch(model, batch, names, basis, policy):
    if batch.dtype != torch.int64 or tuple(batch.shape) != (8, 128) or len(basis) != 4:
        raise ValueError('Invalid held-out microbatch or PCA basis')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameters must be frozen for t-space derivative')
    parameters = dict(model.named_parameters())
    if not set(names).issubset(parameters) or any(set(vector) != set(names) for vector in basis):
        raise ValueError('t-space coordinate inventory mismatch')
    t = torch.zeros(4, device=batch.device, dtype=torch.float64, requires_grad=True)
    t_fp32 = t.to(dtype=torch.float32)
    overrides = {}
    for name in names:
        endpoint = parameters[name]
        if endpoint.dtype != torch.float32 or endpoint.device != batch.device:
            raise ValueError('Invalid FP32 final parameter')
        value = endpoint.detach()
        for j in range(4):
            q = basis[j][name]
            if (q.dtype != torch.float32 or q.device != batch.device or
                    q.shape != endpoint.shape or not q.is_contiguous() or q.requires_grad):
                raise ValueError('Invalid realized FP32 PCA direction')
            value = value+t_fp32[j]*q
        overrides[name] = value
    with torch.enable_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
        logits = torch.func.functional_call(model, overrides, (),
            {'input_ids': batch, 'use_cache': False}, tie_weights=True, strict=False).logits
        loss = a.causal_token_loss_sum(logits, batch, torch)/1016
        if not bool(torch.isfinite(loss).item()):
            raise ValueError('Nonfinite held-out loss')
        gradient = torch.autograd.grad(loss, t, create_graph=True)[0]
        rows = [torch.autograd.grad(gradient[j], t, retain_graph=j < 3)[0]
                for j in range(4)]
        b = gradient.detach().to(device='cpu', dtype=torch.float64).contiguous()
        raw = torch.stack(rows).detach().to(device='cpu', dtype=torch.float64).contiguous()
        del logits, loss, gradient, rows, overrides
    if not bool(torch.isfinite(b).all()):
        raise ValueError('Nonfinite projected gradient')
    M, diagnostic = audit_hessian_symmetry(raw, policy)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('t-space derivative populated parameter .grad')
    return b, M, diagnostic


def solve_centered(b, M, policy):
    if (b.dtype != torch.float64 or M.dtype != torch.float64 or
            b.device.type != 'cpu' or M.device.type != 'cpu' or
            b.ndim != 2 or b.shape[1] != 4 or b.shape[0] < 2 or
            M.shape != (b.shape[0], 4, 4) or
            not bool(torch.isfinite(b).all()) or not bool(torch.isfinite(M).all())):
        raise ValueError('Invalid held-out centered regression records')
    bbar, Mbar = b.mean(dim=0), M.mean(dim=0)
    y = (b-bbar).reshape(-1).contiguous()
    A = (M-Mbar).reshape(-1, 4).contiguous()
    result = torch.linalg.lstsq(A, y, rcond=policy['rcond'], driver=policy['driver'])
    singular = result.singular_values
    if (int(result.rank) != 4 or singular.numel() != 4 or
            not bool(torch.isfinite(singular).all()) or float(singular[-1]) <= 0):
        raise ValueError('Held-out centered design is rank deficient')
    c = result.solution.contiguous()
    response_norm = float(torch.linalg.vector_norm(y))
    if not bool(torch.isfinite(c).all()) or response_norm <= 0:
        raise ValueError('Invalid held-out centered solution')
    residuals = ((M-Mbar)@c-(b-bbar)).norm(dim=1)
    return c, {'coefficients': c.tolist(), 'coefficient_norm': float(torch.linalg.vector_norm(c)),
               'singular_values': singular.tolist(), 'rank': int(result.rank),
               'condition_number': float(singular[0]/singular[-1]),
               'relative_residual': float(torch.linalg.vector_norm(A@c-y))/response_norm,
               'batch_residual_norms': residuals.tolist(),
               'bbar': bbar.tolist(), 'Mbar': Mbar.tolist(),
               'driver': policy['driver'], 'rcond': policy['rcond']}


def split_stability(b, M, policy):
    if b.shape != (16, 4) or M.shape != (16, 4, 4):
        raise ValueError('Expected 16 held-out regression records')
    def compare(left, right):
        c0, first = solve_centered(b[left], M[left], policy)
        c1, second = solve_centered(b[right], M[right], policy)
        n0, n1 = float(torch.linalg.vector_norm(c0)), float(torch.linalg.vector_norm(c1))
        return {'first': first, 'second': second,
                'coefficient_cosine': float(torch.dot(c0, c1))/(n0*n1) if n0*n1 else None,
                'coefficient_norm_ratio': n0/n1 if n1 else None}
    return {'first_8_vs_last_8': compare(slice(0, 8), slice(8, 16)),
            'even_members_vs_odd_members': compare(slice(0, 16, 2), slice(1, 16, 2))}


ARTIFACT_SHAPES = {
    'basis_microbatch_indices': (16,),
    'regression_microbatch_indices': (16,),
    'raw_gradient_gram_G': (16, 16),
    'centered_gradient_gram_K': (16, 16),
    'retained_eigenvalues': (4,),
    'retained_eigenvectors': (16, 4),
    'regression_b': (16, 4),
    'regression_M': (16, 4, 4),
    'c_hat': (4,),
}


def raw_sha(value):
    if value.device.type != 'cpu' or not value.is_contiguous():
        raise ValueError('Artifact tensor must be contiguous CPU')
    dtype = '<i8' if value.dtype == torch.int64 else '<f8' if value.dtype == torch.float64 else None
    if dtype is None or (value.dtype == torch.float64 and not bool(torch.isfinite(value).all())):
        raise ValueError('Invalid artifact tensor')
    return hashlib.sha256(value.numpy().astype(dtype, copy=False).tobytes(order='C')).hexdigest()


def verify_artifact(path, expected=None):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or set(artifact) != set(ARTIFACT_SHAPES):
        raise ValueError('Attempt023 artifact inventory mismatch')
    for name, shape in ARTIFACT_SHAPES.items():
        value = artifact[name]
        dtype = torch.int64 if name.endswith('_indices') else torch.float64
        if (not isinstance(value, torch.Tensor) or value.shape != shape or
                value.dtype != dtype or value.device.type != 'cpu' or not value.is_contiguous()):
            raise ValueError('Attempt023 artifact tensor mismatch')
        raw_sha(value)
    if (not torch.equal(artifact['basis_microbatch_indices'],
                        torch.arange(0, 512, 32, dtype=torch.int64)) or
            not torch.equal(artifact['regression_microbatch_indices'],
                            torch.arange(16, 512, 32, dtype=torch.int64)) or
            not torch.equal(artifact['centered_gradient_gram_K'],
                            artifact['centered_gradient_gram_K'].T) or
            not torch.equal(artifact['regression_M'],
                            artifact['regression_M'].transpose(1, 2))):
        raise ValueError('Attempt023 artifact batch or symmetry mismatch')
    record = {'path': str(path), 'serialized_sha256': a.sha256_file(path),
              'raw_tensor_sha256': {name: raw_sha(value) for name, value in artifact.items()},
              'tensors': {name: {'shape': list(shape),
                                 'dtype': 'torch.int64' if name.endswith('_indices') else 'torch.float64',
                                 'device': 'cpu', 'contiguous': True}
                          for name, shape in ARTIFACT_SHAPES.items()}}
    if expected is not None and record != expected:
        raise ValueError('Attempt023 artifact hash/inventory mismatch')
    return artifact, record


def save_artifact(path, G, K, eigenvalues, eigenvectors, b, M, c_hat):
    if path.exists() or path.is_symlink():
        raise ValueError('Scientific output already exists')
    artifact = {
        'basis_microbatch_indices': torch.arange(0, 512, 32, dtype=torch.int64),
        'regression_microbatch_indices': torch.arange(16, 512, 32, dtype=torch.int64),
        'raw_gradient_gram_G': G.contiguous(),
        'centered_gradient_gram_K': K.contiguous(),
        'retained_eigenvalues': eigenvalues.contiguous(),
        'retained_eigenvectors': eigenvectors.contiguous(),
        'regression_b': b.contiguous(), 'regression_M': M.contiguous(),
        'c_hat': c_hat.contiguous(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(artifact, stream)
    return verify_artifact(path)[1]


def construct(args):
    require_absent(args)
    spec, tokens, hashes, recheck = validate_inputs(args)
    model = a.load_local_model(args.final_directory, args.device, torch)
    eligible = a.discover_eligible_linear_weights(model, torch)
    coordinates = [{'name': name+'.weight', 'shape': list(module.weight.shape)}
                   for name, module in eligible]
    if coordinates != spec['coordinates']:
        raise ValueError('Final model selected-coordinate inventory mismatch')
    names = [row['name'] for row in coordinates]
    device = eligible[0][1].weight.device
    freeze_for_snapshots(model, names)
    before = a.model_state_hashes(model, torch)
    batches = selected_batches()
    basis_batches = batches['basis'][:2 if args.smoke_only else 16]
    snapshots, snapshot_seconds = [], []
    for index, (_, start, end) in enumerate(basis_batches):
        batch = tokens[start:end].to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        snapshot = gradient_snapshot(model, batch, coordinates)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        snapshot_seconds.append(time.perf_counter()-started)
        snapshots.append(snapshot)
        del snapshot, batch
        if not args.smoke_only and (index+1) % 4 == 0:
            print(f'Blind PCA gradient snapshots {index+1}/16', flush=True)
    gram_started = time.perf_counter()
    G = snapshot_gram(snapshots, coordinates)
    gram_seconds = time.perf_counter()-gram_started
    if args.smoke_only:
        a.verify_model_unchanged(model, before, torch)
        recheck()
        norms = torch.sqrt(torch.diag(G))
        if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
            raise ValueError('Invalid smoke gradient norms')
        print(json.dumps({'smoke_only': True, 'basis_microbatch_indices': [0, 32],
                          'gradient_norms': norms.tolist(),
                          'gradient_cosine': float(G[0, 1]/(norms[0]*norms[1])),
                          'gradient_seconds': snapshot_seconds,
                          'gram_seconds': gram_seconds,
                          'gram': G.tolist(),
                          'snapshot_bytes_each': sum(math.prod(row['shape'])*4
                                                     for row in coordinates),
                          'snapshot_dtype': 'CPU_float32',
                          'pca_computed': False, 'regression_computed': False},
                         sort_keys=True))
        return
    K, eigenvalues, eigenvectors, pca = centered_pca(G, spec['pca'])
    basis_cpu = reconstruct_pca_basis(snapshots, coordinates, eigenvalues, eigenvectors)
    gram_audit = basis_gram(basis_cpu, coordinates, spec['pca'])
    del snapshots
    freeze_for_t_space(model)
    basis = [{name: vector[name].to(device=device).contiguous()
              for name in names} for vector in basis_cpu]
    del basis_cpu
    b_rows, M_rows, asymmetry = [], [], []
    for index, (_, start, end) in enumerate(batches['regression']):
        batch = tokens[start:end].to(device)
        b, M, diagnostic = t_space_batch(model, batch, names, basis, spec['t_space'])
        b_rows.append(b); M_rows.append(M); asymmetry.append(diagnostic)
        del batch
        if (index+1) % 4 == 0:
            print(f'Blind held-out regression batches {index+1}/16', flush=True)
    a.verify_model_unchanged(model, before, torch)
    del basis, model, eligible
    recheck()
    b = torch.stack(b_rows).contiguous()
    M = torch.stack(M_rows).contiguous()
    c_hat, fit = solve_centered(b, M, spec['regression'])
    stability = split_stability(b, M, spec['regression'])
    recheck()
    artifact = save_artifact(args.artifact_path, G, K, eigenvalues, eigenvectors,
                             b, M, c_hat)
    manifest = {'format_version': 1, 'attempt_id': spec['attempt_id'],
        'purpose': spec['purpose'], 'information_policy': spec['information_policy'],
        'hash_algorithm': 'sha256', 'input_source_hashes': hashes,
        'source_checkpoint_files': spec['final_checkpoint_files'],
        'corpus': spec['frozen_corpus'], 'coordinates': coordinates,
        'batches': spec['batches'], 'pca_policy': spec['pca'],
        'pca': pca, 'pca_basis_gram': gram_audit,
        't_space': spec['t_space'], 'raw_hessian_asymmetry': asymmetry,
        'regression_policy': spec['regression'], 'regression': fit,
        'split_stability': stability, 'artifact': artifact,
        'workload': {'basis_gradient_batches': 16, 'held_out_hessian_batches': 16,
                     'hessian_second_backward_rows_per_batch': 4},
        'publication_policy': 'no_overwrite_no_resume_partial_artifact_blocks_rerun'}
    recheck()
    verify_artifact(args.artifact_path, artifact)
    a.write_manifest(args.manifest_path, manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path', type=Path,
        default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'spec.json')
    for key, value in FROZEN_SPEC['defaults'].items():
        path = Path(value)
        parser.add_argument('--'+key.replace('_', '-'), type=Path,
                            default=path if path.is_absolute() else PROJECT/path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-only', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    construct(parse_args())
