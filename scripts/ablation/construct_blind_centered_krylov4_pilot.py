#!/usr/bin/env python3
"""Attempt 022: blind centered regression in the frozen Attempt-011 basis."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

PROJECT = Path(__file__).resolve().parents[2]
PRIOR_MANIFEST_PATH = PROJECT/'experiments/attempts/011_generic_hessian_krylov_k4_prefix0_13/construction-manifest.json'
PRIOR_MANIFEST_SHA256 = 'af334b5a580560900f38f5349ad8db51c6d1c2b61287d0b819b09829e52f91aa'
PRIOR_SOURCE_PATH = Path(__file__).with_name('construct_generic_hessian_krylov.py')
PRIOR_SOURCE_SHA256 = '9edce220e1f9e29b4e02082852d03aa6b8043e35964a546383a3c10111018f44'
BASIS_HASHES = ('c3e3db2e4e2f4967281448954aa51af1d370b00d1fd30c95e74605484a71befc',
                'a7be2c5c171c6c4dbf23d8526459d321277a05f74901472b846870e7bc372209',
                '6fed3d7c14edb8de6516714d0963eef91f92f055b4903b30eba40ecf7b3f0fc6',
                '3201b57d65f93ad08cf034feaba118e5522f90ebac110ff48d5186ca0c46d03d')
if hashlib.sha256(PRIOR_MANIFEST_PATH.read_bytes()).hexdigest() != PRIOR_MANIFEST_SHA256:
    raise ValueError('Frozen Attempt-011 construction manifest source mismatch')
prior_manifest = json.loads(PRIOR_MANIFEST_PATH.read_text())
loader = importlib.util.spec_from_file_location('attempt022_blind011_helper', PRIOR_SOURCE_PATH)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)

FROZEN_SPEC = {
    'format_version': 1,
    'attempt_id': '022_blind_centered_regression_krylov4_pilot',
    'purpose': 'blind_final_checkpoint_centered_batch_regression_krylov4_pilot',
    'information_policy': 'final_checkpoint_plus_generic_text_plus_preexisting_blind_basis_only',
    'final_checkpoint_files': prior_manifest['source_checkpoint']['files'],
    'frozen_corpus': a.FROZEN_SPEC['frozen_corpus'],
    'coordinates': prior_manifest['matrices'],
    'basis': {'manifest_sha256': PRIOR_MANIFEST_SHA256,
              'files': prior_manifest['basis_files'],
              'reference_gram': prior_manifest['orthogonality']['gram_matrix'],
              'orthogonality': prior_manifest['orthogonality']['method'],
              'reference_gram_max_error': 1e-6},
    'batches': {'microbatch_indices': list(range(0, 512, 16)), 'count': 32,
                'rows_per_batch': 8, 'sequence_length': 128,
                'prediction_tokens_per_batch': 1016,
                'row_rule': 'start=8*j,end=8*(j+1),end_exclusive'},
    't_space': {'dimension': 4, 't_leaf_dtype': 'torch.float64',
                'weight_arithmetic': 'FP32(W1_weight+sum_j FP32(t_j)*realized_FP32_q_j)',
                'evaluation': 'torch.func.functional_call',
                'only_autograd_input': 't', 'primitive': 'true_double_backward',
                'raw_hessian_relative_asymmetry_limit': 1e-5,
                'relative_asymmetry_denominator_floor': 1e-6,
                'symmetrization': '(M_raw+M_raw.T)/2', 'fallback': False},
    'regression': {'driver': 'gelsd', 'rcond': 1e-12, 'dtype': 'CPU_float64',
                   'rank_required': 4, 'regularization': None,
                   'primary': 'all_32',
                   'splits': ['first_16_vs_last_16', 'even_members_vs_odd_members'],
                   'split_selection': False},
    'workload': {'gradient_and_hessian_batches': 32, 'smoke_batches': 1,
                 'hessian_second_backward_rows_per_batch': 4},
    'sources': {'scripts/ablation/construct_generic_hessian_krylov.py': PRIOR_SOURCE_SHA256},
    'defaults': {
        'final_directory': '/root/model-diff-scratch/models/merged',
        'corpus_spec_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json',
        'corpus_manifest_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json',
        'tokens_path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
        'basis_directory': '/root/model-diff-scratch/artifacts/attempt011_hessian_krylov_k4',
        'basis_manifest_path': 'experiments/attempts/011_generic_hessian_krylov_k4_prefix0_13/construction-manifest.json',
        'artifact_path': '/root/model-diff-scratch/artifacts/attempt022_blind_centered_krylov4/batch_stats.pt',
        'manifest_path': 'experiments/attempts/022_blind_centered_regression_krylov4_pilot/construction-manifest.json',
    },
    'outputs': {'overwrite': False, 'timestamps': False, 'host_metadata': False, 'gpu_metadata': False},
}


def selected_batches():
    return [(j, 8*j, 8*(j+1)) for j in range(0, 512, 16)]


class BasisStore:
    """Four pinned safetensors, memory-mapped and read matrixwise."""
    def __init__(self, directory, coordinates):
        self.directory, self.coordinates = directory, coordinates
        self.stack = ExitStack()
        self.files = []

    def __enter__(self):
        names = {row['name'] for row in self.coordinates}
        try:
            for i in range(4):
                file = self.stack.enter_context(safe_open(str(self.directory/f'q{i+1}.safetensors'),
                                                          framework='pt', device='cpu'))
                if set(file.keys()) != names:
                    raise ValueError('Basis coordinate inventory mismatch')
                self.files.append(file)
            return self
        except BaseException:
            self.stack.close()
            raise

    def __exit__(self, *exc):
        self.stack.close()

    def matrix(self, i, name, shape):
        value = self.files[i].get_tensor(name)
        if (value.dtype != torch.float32 or value.device.type != 'cpu' or
                tuple(value.shape) != tuple(shape) or not value.is_contiguous() or
                not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid frozen basis matrix')
        return value


def verify_basis_files(directory, records):
    if len(records) != 4:
        raise ValueError('Expected four frozen basis files')
    for i, record in enumerate(records):
        path = directory/f'q{i+1}.safetensors'
        if (record != FROZEN_SPEC['basis']['files'][i] or
                record['filename'] != path.name or record['sha256'] != BASIS_HASHES[i] or
                not path.is_file() or path.stat().st_size != record['size_bytes'] or
                a.sha256_file(path) != record['sha256']):
            raise ValueError('Frozen basis file hash/inventory mismatch')


def basis_gram(store, coordinates, settings):
    terms = [[[] for _ in range(4)] for _ in range(4)]
    for row in coordinates:
        vectors = [store.matrix(i, row['name'], row['shape']).double() for i in range(4)]
        for i in range(4):
            for j in range(i, 4):
                terms[i][j].append(float(torch.sum(vectors[i]*vectors[j], dtype=torch.float64)))
    gram = torch.zeros((4, 4), dtype=torch.float64)
    for i in range(4):
        for j in range(i, 4):
            gram[i, j] = gram[j, i] = math.fsum(terms[i][j])
    if not bool(torch.isfinite(gram).all()):
        raise ValueError('Nonfinite basis Gram matrix')
    off = gram-torch.diag(torch.diag(gram))
    off_error = float(off.abs().max())
    diag_error = float((torch.diag(gram)-1).abs().max())
    reference_error = float((gram-torch.tensor(settings['reference_gram'], dtype=torch.float64)).abs().max())
    if (off_error > settings['orthogonality']['maximum_absolute_off_diagonal'] or
            diag_error > settings['orthogonality']['maximum_diagonal_deviation_from_one'] or
            reference_error > settings['reference_gram_max_error']):
        raise ValueError('Frozen basis orthogonality/Gram audit failed')
    return {'gram_matrix': gram.tolist(), 'max_off_diagonal': off_error,
            'max_diagonal_deviation': diag_error, 'max_reference_difference': reference_error}


def basis_on_device(store, coordinates, device):
    vectors = [dict() for _ in range(4)]
    for row in coordinates:
        for i in range(4):
            vectors[i][row['name']] = store.matrix(i, row['name'], row['shape']).to(device=device).contiguous()
    return vectors


def freeze_model(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameter freeze failed')


def audit_hessian_symmetry(raw, asymmetry):
    if (raw.shape != (4, 4) or raw.dtype != torch.float64 or raw.device.type != 'cpu' or
            not bool(torch.isfinite(raw).all())):
        raise ValueError('Invalid raw t-space Hessian')
    absolute = float((raw-raw.T).abs().max())
    scale = max(float(raw.abs().max()), asymmetry['relative_asymmetry_denominator_floor'])
    relative = absolute/scale
    if relative > asymmetry['raw_hessian_relative_asymmetry_limit']:
        raise ValueError('t-space Hessian asymmetry exceeds frozen tolerance')
    return ((raw+raw.T)/2).contiguous(), {'absolute': absolute, 'relative': relative}


def t_space_batch(model, batch, names, basis, asymmetry):
    if batch.dtype != torch.int64 or tuple(batch.shape) != (8, 128) or len(basis) != 4 or not names:
        raise ValueError('Invalid frozen microbatch or t-space basis')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Model parameters must be frozen')
    parameters = dict(model.named_parameters())
    if not set(names).issubset(parameters) or any(set(vector) != set(names) for vector in basis):
        raise ValueError('Selected t-space coordinate inventory mismatch')
    t = torch.zeros(4, device=batch.device, dtype=torch.float64, requires_grad=True)
    t_fp32 = t.to(dtype=torch.float32)
    overrides = {}
    for name in names:
        endpoint = parameters[name]
        if endpoint.dtype != torch.float32 or endpoint.device != batch.device:
            raise ValueError('Invalid FP32 endpoint')
        value = endpoint.detach()
        for j in range(4):
            q = basis[j][name]
            if (q.dtype != torch.float32 or q.device != batch.device or q.shape != endpoint.shape or
                    not q.is_contiguous() or q.requires_grad):
                raise ValueError('Invalid realized FP32 basis direction')
            value = value + t_fp32[j]*q
        overrides[name] = value
    with torch.enable_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
        logits = torch.func.functional_call(model, overrides, (),
            {'input_ids': batch, 'use_cache': False}, tie_weights=True, strict=False).logits
        loss = a.causal_token_loss_sum(logits, batch, torch)/1016
        if not bool(torch.isfinite(loss).item()):
            raise ValueError('Nonfinite batch loss')
        gradient = torch.autograd.grad(loss, t, create_graph=True)[0]
        rows = [torch.autograd.grad(gradient[j], t, retain_graph=j<3)[0] for j in range(4)]
        b = gradient.detach().to(device='cpu', dtype=torch.float64).contiguous()
        raw = torch.stack(rows).detach().to(device='cpu', dtype=torch.float64).contiguous()
        del logits, loss, gradient, rows, overrides
    if not bool(torch.isfinite(b).all() and torch.isfinite(raw).all()):
        raise ValueError('Nonfinite t-space derivative')
    M, diagnostic = audit_hessian_symmetry(raw, asymmetry)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Unexpected model parameter gradient')
    return b, M, diagnostic


def solve_centered(b, M, policy):
    if (b.dtype != torch.float64 or M.dtype != torch.float64 or b.device.type != 'cpu' or
            M.device.type != 'cpu' or b.ndim != 2 or M.shape != (b.shape[0], 4, 4) or
            b.shape[1] != 4 or b.shape[0] < 2 or not bool(torch.isfinite(b).all()) or
            not bool(torch.isfinite(M).all())):
        raise ValueError('Invalid CPU float64 centered regression records')
    bbar = b.mean(dim=0)
    Mbar = M.mean(dim=0)
    y = (b-bbar).reshape(-1).contiguous()
    A = (M-Mbar).reshape(-1, 4).contiguous()
    result = torch.linalg.lstsq(A, y, rcond=policy['rcond'], driver=policy['driver'])
    singular = result.singular_values
    if int(result.rank) != 4 or singular.numel() != 4 or not bool(torch.isfinite(singular).all()) or float(singular[-1]) <= 0:
        raise ValueError('Centered design is rank deficient')
    c = result.solution.contiguous()
    if not bool(torch.isfinite(c).all()):
        raise ValueError('Nonfinite centered coefficients')
    y_norm = float(torch.linalg.vector_norm(y))
    if y_norm <= 0:
        raise ValueError('Zero centered gradient response')
    residuals = ((M-Mbar)@c-(b-bbar)).norm(dim=1)
    return c, {'coefficients': c.tolist(), 'singular_values': singular.tolist(),
               'rank': int(result.rank), 'condition_number': float(singular[0]/singular[-1]),
               'relative_residual': float(torch.linalg.vector_norm(A@c-y))/y_norm,
               'coefficient_norm': float(torch.linalg.vector_norm(c)),
               'bbar': bbar.tolist(), 'Mbar': Mbar.tolist(),
               'batch_residual_norms': residuals.tolist(),
               'driver': policy['driver'], 'rcond': policy['rcond']}


def split_stability(b, M, policy):
    if b.shape != (32, 4) or M.shape != (32, 4, 4):
        raise ValueError('Expected 32 batch records for split diagnostics')
    def compare(left, right):
        c0, r0 = solve_centered(b[left], M[left], policy)
        c1, r1 = solve_centered(b[right], M[right], policy)
        n0, n1 = float(torch.linalg.vector_norm(c0)), float(torch.linalg.vector_norm(c1))
        return {'first': r0, 'second': r1,
                'coefficient_cosine': float(torch.dot(c0, c1))/(n0*n1) if n0*n1 else None,
                'coefficient_norm_ratio': n0/n1 if n1 else None}
    return {'first_16_vs_last_16': compare(slice(0, 16), slice(16, 32)),
            'even_members_vs_odd_members': compare(slice(0, 32, 2), slice(1, 32, 2))}


def raw_sha(value):
    if value.device.type != 'cpu' or not value.is_contiguous():
        raise ValueError('Artifact tensor must be contiguous CPU')
    dtype = '<f8' if value.dtype == torch.float64 else '<i8' if value.dtype == torch.int64 else None
    if dtype is None or (value.dtype == torch.float64 and not bool(torch.isfinite(value).all())):
        raise ValueError('Invalid artifact tensor')
    return hashlib.sha256(value.numpy().astype(dtype, copy=False).tobytes(order='C')).hexdigest()


def verify_artifact(path, expected=None):
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    shapes = {'microbatch_index': (32,), 'b': (32, 4), 'M': (32, 4, 4), 'c_hat': (4,)}
    if not isinstance(artifact, dict) or set(artifact) != set(shapes):
        raise ValueError('Attempt-022 artifact inventory mismatch')
    for name, shape in shapes.items():
        value = artifact[name]
        dtype = torch.int64 if name == 'microbatch_index' else torch.float64
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or
                value.dtype != dtype or value.device.type != 'cpu' or not value.is_contiguous()):
            raise ValueError('Attempt-022 artifact tensor mismatch')
        raw_sha(value)
    if not torch.equal(artifact['microbatch_index'], torch.arange(0, 512, 16, dtype=torch.int64)):
        raise ValueError('Attempt-022 artifact batch selection mismatch')
    if not torch.equal(artifact['M'], (artifact['M']+artifact['M'].transpose(1, 2))/2):
        raise ValueError('Attempt-022 artifact Hessian symmetry mismatch')
    record = {'path': str(path), 'serialized_sha256': a.sha256_file(path),
              'raw_tensor_sha256': {name: raw_sha(value) for name, value in artifact.items()},
              'tensors': {name: {'shape': list(shape),
                                 'dtype': 'torch.int64' if name == 'microbatch_index' else 'torch.float64',
                                 'device': 'cpu', 'contiguous': True} for name, shape in shapes.items()}}
    if expected is not None and record != expected:
        raise ValueError('Attempt-022 artifact hash/inventory mismatch')
    return artifact, record


def save_artifact(path, b, M, c):
    if path.exists() or path.is_symlink():
        raise ValueError('Scientific output already exists')
    artifact = {'microbatch_index': torch.arange(0, 512, 16, dtype=torch.int64),
                'b': b.contiguous(), 'M': M.contiguous(), 'c_hat': c.contiguous()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(artifact, stream)
    return verify_artifact(path)[1]


def firewall(args):
    for key, value in vars(args).items():
        if isinstance(value, Path) and any(marker in str(value).lower() for marker in
                                           ('/base', 'oracle', 'adapter', 'attempt015', 'attempt016',
                                            'attempt018', 'attempt020', 'attempt021',
                                            '015_', '016_', '018_', '020_', '021_',
                                            'true_delta', 'evaluation.json')):
            raise ValueError('Blind construction information firewall: '+key)


def require_absent(args):
    for path in (args.artifact_path, args.manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError('Scientific output already exists')


def validate_inputs(args):
    firewall(args)
    spec = a.load_json_object(args.spec_path, 'Attempt022 spec')
    if spec != FROZEN_SPEC:
        raise ValueError('Frozen specification mismatch')
    paths = [args.spec_path, args.corpus_spec_path, args.corpus_manifest_path,
             args.tokens_path, args.basis_manifest_path, Path(__file__).resolve(), PRIOR_SOURCE_PATH]
    hashes = {path: a.sha256_file(path) for path in paths}
    if (hashes[args.basis_manifest_path] != PRIOR_MANIFEST_SHA256 or
            hashes[PRIOR_SOURCE_PATH] != PRIOR_SOURCE_SHA256):
        raise ValueError('Attempt-011 provenance hash mismatch')
    manifest = a.load_json_object(args.basis_manifest_path, 'Attempt011 construction manifest')
    if (manifest.get('attempt_id') != '011_generic_hessian_krylov_k4_prefix0_13' or
            manifest.get('source_checkpoint', {}).get('files') != spec['final_checkpoint_files'] or
            manifest.get('matrices') != spec['coordinates'] or
            manifest.get('basis_files') != spec['basis']['files'] or
            manifest.get('orthogonality', {}).get('method') != spec['basis']['orthogonality']):
        raise ValueError('Attempt-011 basis manifest provenance mismatch')
    verify_basis_files(args.basis_directory, manifest['basis_files'])
    if a.checkpoint_file_records(args.final_directory) != spec['final_checkpoint_files']:
        raise ValueError('Canonical final checkpoint mismatch')
    for field, path in (('corpus_spec_sha256', args.corpus_spec_path),
                        ('corpus_manifest_sha256', args.corpus_manifest_path)):
        if hashes[path] != spec['frozen_corpus'][field]:
            raise ValueError('Frozen corpus provenance mismatch')
    corpus = a.verify_corpus_manifest(args.corpus_manifest_path, args.corpus_spec_path,
        args.tokens_path, args.final_directory, a.load_corpus_spec(args.corpus_spec_path))
    if any(corpus[key] != spec['frozen_corpus'][key] for key in ('serialized_sha256', 'raw_tensor_sha256')):
        raise ValueError('Frozen corpus artifact mismatch')
    tokens = a.load_corpus_tokens(args.tokens_path, corpus['raw_tensor_sha256'], 4096, 128, torch)
    for output in (args.artifact_path, args.manifest_path):
        if (output.resolve().is_relative_to(args.final_directory.resolve()) or
                output.resolve().is_relative_to(args.basis_directory.resolve()) or
                any(output.resolve() == path.resolve() or path.resolve().is_relative_to(output.resolve())
                    for path in paths)):
            raise ValueError('Output overlaps frozen input')
    if args.artifact_path.resolve().is_relative_to(args.manifest_path.resolve()) or args.manifest_path.resolve().is_relative_to(args.artifact_path.resolve()):
        raise ValueError('Outputs overlap each other')
    with BasisStore(args.basis_directory, spec['coordinates']) as store:
        gram = basis_gram(store, spec['coordinates'], spec['basis'])
    def recheck():
        a.verify_unchanged(args.final_directory, spec['final_checkpoint_files'], hashes)
        verify_basis_files(args.basis_directory, manifest['basis_files'])
    return spec, tokens, hashes, manifest, gram, recheck


def construct(args):
    require_absent(args)
    spec, tokens, hashes, basis_manifest, gram, recheck = validate_inputs(args)
    model = a.load_local_model(args.final_directory, args.device, torch)
    eligible = a.discover_eligible_linear_weights(model, torch)
    coordinates = [{'name': name+'.weight', 'shape': list(module.weight.shape)} for name, module in eligible]
    if coordinates != spec['coordinates']:
        raise ValueError('Loaded model selected-coordinate inventory mismatch')
    freeze_model(model)
    before = a.model_state_hashes(model, torch)
    names = [row['name'] for row in coordinates]
    with BasisStore(args.basis_directory, coordinates) as store:
        basis = basis_on_device(store, coordinates, eligible[0][1].weight.device)
    batches = selected_batches()[:1 if args.smoke_only else 32]
    b_values, M_values, asymmetry = [], [], []
    smoke_seconds = None
    for i, (_, start, end) in enumerate(batches):
        batch = tokens[start:end].to(eligible[0][1].weight.device)
        if batch.device.type == 'cuda':
            torch.cuda.synchronize(batch.device)
        started = time.perf_counter()
        b, M, asym = t_space_batch(model, batch, names, basis, spec['t_space'])
        if batch.device.type == 'cuda':
            torch.cuda.synchronize(batch.device)
        if args.smoke_only:
            smoke_seconds = time.perf_counter()-started
        b_values.append(b); M_values.append(M); asymmetry.append(asym)
        if (i+1) % 8 == 0:
            print(f'Blind t-space batches {i+1}/{len(batches)}', flush=True)
        del batch
    a.verify_model_unchanged(model, before, torch)
    del basis, model, eligible
    recheck()
    if args.smoke_only:
        print(json.dumps({'smoke_only': True, 'microbatch_index': 0,
                          'b_0': b_values[0].tolist(), 'M_0': M_values[0].tolist(),
                          'raw_hessian_asymmetry': asymmetry[0], 'seconds': smoke_seconds,
                          'basis_gram': gram}, sort_keys=True))
        return
    b = torch.stack(b_values).contiguous()
    M = torch.stack(M_values).contiguous()
    c, fit = solve_centered(b, M, spec['regression'])
    stability = split_stability(b, M, spec['regression'])
    artifact = save_artifact(args.artifact_path, b, M, c)
    manifest = {'format_version': 1, 'attempt_id': spec['attempt_id'],
        'purpose': spec['purpose'], 'information_policy': spec['information_policy'],
        'hash_algorithm': 'sha256',
        'input_source_hashes': {str(path): digest for path, digest in hashes.items()},
        'source_checkpoint_files': spec['final_checkpoint_files'], 'corpus': spec['frozen_corpus'],
        'basis': {'manifest_path': str(args.basis_manifest_path),
                  'manifest_sha256': PRIOR_MANIFEST_SHA256,
                  'files': basis_manifest['basis_files'], 'gram': gram,
                  'coordinates': spec['coordinates']},
        'batches': spec['batches'], 't_space': spec['t_space'],
        'raw_hessian_asymmetry': asymmetry, 'regression': fit,
        'split_stability': stability, 'artifact': artifact, 'workload': spec['workload'],
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
