#!/usr/bin/env python3
"""Post-freeze parameter-space evaluation of Attempt 023; no basis reconstruction."""
import argparse
import hashlib
import importlib.util
import json
import math
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from safetensors import safe_open

PROJECT = Path(__file__).resolve().parents[2]
CONSTRUCTOR_PATH = Path(__file__).with_name('construct_blind_centered_gradient_pca4_pilot.py')
HISTORICAL_SPEC_PATH = PROJECT/'experiments/attempts/016_known_base_exact_displacement_jvp_ceiling_prefix0_13/spec.json'
HISTORICAL_SPEC_SHA256 = '29cc8defa81ebb7255c8daac5eaa722087dac7250662db8c3f2555b28c691934'
loader = importlib.util.spec_from_file_location('attempt023_frozen_constructor', CONSTRUCTOR_PATH)
c = importlib.util.module_from_spec(loader)
loader.loader.exec_module(c)

EVALUATION_SPEC = {
    'format_version': 1,
    'attempt_id': c.FROZEN_SPEC['attempt_id'],
    'purpose': 'post_freeze_known_base_parameter_space_evaluation_no_tuning',
    'construction_policy': c.FROZEN_SPEC['information_policy'],
    'historical_base_spec_sha256': HISTORICAL_SPEC_SHA256,
    'historical_base_loading': 'raw_safetensors_to_contiguous_FP32_loaded_W0',
    'displacement': 'CPU_float64_W1_FP32_minus_W0_FP32',
    'snapshot_projection': 's_i=scalar_path_dL_i(W1+t_DeltaS)/dt_at_t0;v=(I-11T/16)s;c_true_j=u_jT_v/sqrt(lambda_j)',
    'functional_activation_evaluation': False,
    'defaults': {
        'construction_spec_path': 'experiments/attempts/023_blind_centered_gradient_pca4_regression_pilot/spec.json',
        'construction_manifest_path': 'experiments/attempts/023_blind_centered_gradient_pca4_regression_pilot/construction-manifest.json',
        'artifact_path': c.FROZEN_SPEC['defaults']['artifact_path'],
        'evaluation_spec_path': 'experiments/attempts/023_blind_centered_gradient_pca4_regression_pilot/evaluation_spec.json',
        'output_path': 'experiments/attempts/023_blind_centered_gradient_pca4_regression_pilot/evaluation.json',
        'final_directory': c.FROZEN_SPEC['defaults']['final_directory'],
        'corpus_spec_path': c.FROZEN_SPEC['defaults']['corpus_spec_path'],
        'corpus_manifest_path': c.FROZEN_SPEC['defaults']['corpus_manifest_path'],
        'tokens_path': c.FROZEN_SPEC['defaults']['tokens_path'],
        'base_directory': '/root/model-diff-scratch/models/base',
    },
    'outputs': {'overwrite': False, 'timestamps': False,
                'host_metadata': False, 'gpu_metadata': False},
}


def project_from_snapshot_scalars(s, eigenvalues, eigenvectors):
    if (s.shape != (16,) or eigenvalues.shape != (4,) or eigenvectors.shape != (16, 4) or
            any(value.dtype != torch.float64 or value.device.type != 'cpu' or
                not bool(torch.isfinite(value).all())
                for value in (s, eigenvalues, eigenvectors)) or
            bool((eigenvalues <= 0).any())):
        raise ValueError('Invalid frozen PCA/snapshot projection input')
    before_values, before_vectors = eigenvalues.clone(), eigenvectors.clone()
    C = torch.eye(16, dtype=torch.float64)-torch.ones((16, 16), dtype=torch.float64)/16
    centered = C@s
    coefficients = eigenvectors.T@centered/torch.sqrt(eigenvalues)
    if (not bool(torch.isfinite(coefficients).all()) or
            not torch.equal(eigenvalues, before_values) or
            not torch.equal(eigenvectors, before_vectors)):
        raise ValueError('Frozen PCA changed or projection is nonfinite')
    return coefficients.contiguous(), centered.contiguous()


def projection_metrics(c_hat, c_true, delta_norm):
    if (c_hat.shape != (4,) or c_true.shape != (4,) or
            c_hat.dtype != torch.float64 or c_true.dtype != torch.float64 or
            c_hat.device.type != 'cpu' or c_true.device.type != 'cpu' or
            not bool(torch.isfinite(c_hat).all()) or not bool(torch.isfinite(c_true).all()) or
            not math.isfinite(delta_norm) or delta_norm <= 0):
        raise ValueError('Invalid frozen coefficient or historical displacement')
    before = c_hat.clone()
    candidate_norm = float(torch.linalg.vector_norm(c_hat))
    projected_norm = float(torch.linalg.vector_norm(c_true))
    dot = float(torch.dot(c_hat, c_true))
    ceiling = projected_norm/delta_norm
    coefficient_cosine = dot/(candidate_norm*projected_norm) if candidate_norm*projected_norm else None
    full_cosine = dot/(candidate_norm*delta_norm) if candidate_norm else None
    ceiling_fraction = full_cosine/ceiling if full_cosine is not None and ceiling else None
    optimal = dot/(candidate_norm*candidate_norm) if candidate_norm else None
    residual = (float(torch.linalg.vector_norm(optimal*c_hat-c_true))/projected_norm
                if optimal is not None and projected_norm else None)
    if not torch.equal(c_hat, before):
        raise ValueError('Frozen c_hat changed during evaluation')
    return {'c_hat': c_hat.tolist(), 'c_true': c_true.tolist(),
            'deltaS_norm': delta_norm, 'candidate_norm': candidate_norm,
            'projected_true_norm': projected_norm, 'projection_ceiling': ceiling,
            'cosine_coefficients': coefficient_cosine,
            'cosine_candidate_to_full_DeltaS': full_cosine,
            'fraction_of_subspace_ceiling': ceiling_fraction,
            'optimal_scalar': optimal,
            'relative_residual_after_optimal_rescaling': residual}


@contextmanager
def selected_base_tensors(directory, files):
    stack = ExitStack()
    try:
        index = {}
        for record in files:
            if record['path'].endswith('.safetensors'):
                source = stack.enter_context(safe_open(str(directory/record['path']),
                                                       framework='pt', device='cpu'))
                for name in source.keys():
                    if name in index:
                        raise ValueError('Duplicate historical checkpoint tensor key')
                    index[name] = source
        yield index
    finally:
        stack.close()


def exact_early_delta(final, coordinates, base_tensors):
    parameters = dict(final.named_parameters())
    direction, norm_terms = {}, []
    for row in coordinates:
        name, shape = row['name'], row['shape']
        if name not in parameters or name not in base_tensors:
            raise ValueError('Missing selected historical coordinate')
        current = parameters[name].detach().to(device='cpu', dtype=torch.float32)
        old = base_tensors[name].get_tensor(name).to(dtype=torch.float32).contiguous()
        if (list(current.shape) != shape or list(old.shape) != shape or
                not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(old).all())):
            raise ValueError('Historical selected tensor mismatch')
        delta = (current.double()-old.double()).contiguous()
        if not bool(torch.isfinite(delta).all()):
            raise ValueError('Nonfinite historical displacement')
        direction[name] = delta
        norm_terms.append(float(torch.sum(delta.square(), dtype=torch.float64)))
    norm = math.sqrt(math.fsum(norm_terms))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError('Invalid historical early displacement norm')
    return direction, norm


def scalar_directional_derivative(model, batch, names, direction):
    if batch.dtype != torch.int64 or tuple(batch.shape) != (8, 128):
        raise ValueError('Invalid frozen evaluation microbatch')
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Evaluation model parameters must be frozen')
    parameters = dict(model.named_parameters())
    if set(names) != set(direction) or not set(names).issubset(parameters):
        raise ValueError('Evaluation directional coordinate mismatch')
    t = torch.zeros((), device=batch.device, dtype=torch.float64, requires_grad=True)
    overrides = {}
    for name in names:
        endpoint, delta = parameters[name], direction[name]
        if (endpoint.dtype != torch.float32 or endpoint.device != batch.device or
                delta.dtype != torch.float64 or delta.device != batch.device or
                delta.shape != endpoint.shape or delta.requires_grad):
            raise ValueError('Invalid exact historical direction')
        overrides[name] = endpoint.detach()+(t*delta).to(dtype=torch.float32)
    with torch.enable_grad(), torch.autocast(device_type=batch.device.type, enabled=False):
        logits = torch.func.functional_call(model, overrides, (),
            {'input_ids': batch, 'use_cache': False}, tie_weights=True, strict=False).logits
        loss = c.a.causal_token_loss_sum(logits, batch, torch)/1016
        derivative = torch.autograd.grad(loss, t)[0]
    value = float(derivative.detach().to(device='cpu', dtype=torch.float64))
    if not math.isfinite(value) or any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Nonfinite scalar projection or unexpected parameter .grad')
    return value


def validate_frozen_construction(args):
    if args.output_path.exists() or args.output_path.is_symlink():
        raise ValueError('Evaluation output already exists')
    evaluation_spec = c.a.load_json_object(args.evaluation_spec_path, 'Attempt023 evaluation spec')
    if evaluation_spec != EVALUATION_SPEC:
        raise ValueError('Frozen evaluation specification mismatch')
    construction_spec = c.a.load_json_object(args.construction_spec_path, 'Attempt023 construction spec')
    if construction_spec != c.FROZEN_SPEC:
        raise ValueError('Frozen construction specification mismatch')
    manifest = c.a.load_json_object(args.construction_manifest_path, 'Attempt023 construction manifest')
    if (manifest.get('attempt_id') != c.FROZEN_SPEC['attempt_id'] or
            manifest.get('information_policy') != c.FROZEN_SPEC['information_policy'] or
            manifest.get('coordinates') != c.FROZEN_SPEC['coordinates'] or
            manifest.get('batches') != c.FROZEN_SPEC['batches'] or
            manifest.get('pca_policy') != c.FROZEN_SPEC['pca'] or
            manifest.get('regression_policy') != c.FROZEN_SPEC['regression'] or
            manifest.get('source_checkpoint_files') != c.FROZEN_SPEC['final_checkpoint_files']):
        raise ValueError('Frozen blind construction provenance mismatch')
    source_hashes = manifest.get('input_source_hashes', {})
    required_paths = (args.construction_spec_path, args.corpus_spec_path,
                      args.corpus_manifest_path, args.tokens_path,
                      CONSTRUCTOR_PATH, c.HELPER_PATH)
    for path in required_paths:
        if source_hashes.get(str(path)) != c.a.sha256_file(path):
            raise ValueError('Frozen blind construction input hash mismatch')
    artifact, record = c.verify_artifact(args.artifact_path, manifest['artifact'])
    K, eigenvalues, eigenvectors, _ = c.centered_pca(
        artifact['raw_gradient_gram_G'], c.FROZEN_SPEC['pca'])
    if (not torch.allclose(K, artifact['centered_gradient_gram_K'], rtol=1e-12, atol=1e-12) or
            not torch.allclose(eigenvalues, artifact['retained_eigenvalues'], rtol=1e-10, atol=1e-10) or
            not torch.allclose(eigenvectors, artifact['retained_eigenvectors'], rtol=1e-9, atol=1e-9)):
        raise ValueError('Frozen PCA does not reproduce from snapshot Gram')
    recomputed, _ = c.solve_centered(artifact['regression_b'], artifact['regression_M'],
                                     c.FROZEN_SPEC['regression'])
    if (not torch.allclose(recomputed, artifact['c_hat'], rtol=1e-12, atol=1e-12) or
            artifact['c_hat'].tolist() != manifest['regression']['coefficients']):
        raise ValueError('Frozen c_hat does not reproduce held-out regression')
    if c.a.checkpoint_file_records(args.final_directory) != c.FROZEN_SPEC['final_checkpoint_files']:
        raise ValueError('Canonical final checkpoint mismatch')
    corpus = c.a.verify_corpus_manifest(args.corpus_manifest_path, args.corpus_spec_path,
        args.tokens_path, args.final_directory, c.a.load_corpus_spec(args.corpus_spec_path))
    if any(corpus[key] != c.FROZEN_SPEC['frozen_corpus'][key]
           for key in ('serialized_sha256', 'raw_tensor_sha256')):
        raise ValueError('Frozen FineWeb artifact mismatch')
    tokens = c.a.load_corpus_tokens(args.tokens_path, corpus['raw_tensor_sha256'],
                                    4096, 128, torch)
    frozen_paths = [args.evaluation_spec_path, args.construction_spec_path,
                    args.construction_manifest_path, args.artifact_path,
                    Path(__file__).resolve(), CONSTRUCTOR_PATH, c.HELPER_PATH]
    if args.output_path.resolve() in [path.resolve() for path in frozen_paths]:
        raise ValueError('Evaluation output overlaps frozen input')
    hashes = {str(path): c.a.sha256_file(path) for path in frozen_paths}
    def recheck():
        for path, digest in hashes.items():
            if c.a.sha256_file(Path(path)) != digest:
                raise ValueError('Frozen blind evaluation input changed')
        for path in required_paths:
            if c.a.sha256_file(path) != source_hashes[str(path)]:
                raise ValueError('Frozen blind construction input changed')
        if c.a.checkpoint_file_records(args.final_directory) != c.FROZEN_SPEC['final_checkpoint_files']:
            raise ValueError('Final checkpoint changed during evaluation')
    return artifact, record, tokens, hashes, recheck


def evaluate(args):
    artifact, record, tokens, hashes, recheck = validate_frozen_construction(args)
    if c.a.sha256_file(HISTORICAL_SPEC_PATH) != HISTORICAL_SPEC_SHA256:
        raise ValueError('Historical base provenance specification mismatch')
    historical = c.a.load_json_object(HISTORICAL_SPEC_PATH, 'historical base spec')
    if c.a.checkpoint_file_records(args.base_directory) != historical['base']['files']:
        raise ValueError('Historical base checkpoint mismatch')
    model = c.a.load_local_model(args.final_directory, args.device, torch)
    eligible = c.a.discover_eligible_linear_weights(model, torch)
    coordinates = [{'name': name+'.weight', 'shape': list(module.weight.shape)}
                   for name, module in eligible]
    if coordinates != c.FROZEN_SPEC['coordinates']:
        raise ValueError('Evaluation selected-coordinate inventory mismatch')
    c.freeze_for_t_space(model)
    before = c.a.model_state_hashes(model, torch)
    with selected_base_tensors(args.base_directory, historical['base']['files']) as base_tensors:
        delta_cpu, delta_norm = exact_early_delta(model, coordinates, base_tensors)
    device = eligible[0][1].weight.device
    direction = {name: delta_cpu[name].to(device=device).contiguous()
                 for name in (row['name'] for row in coordinates)}
    del delta_cpu
    names = [row['name'] for row in coordinates]
    values = []
    for j in artifact['basis_microbatch_indices'].tolist():
        batch = tokens[8*j:8*(j+1)].to(device)
        values.append(scalar_directional_derivative(model, batch, names, direction))
        del batch
    s = torch.tensor(values, dtype=torch.float64)
    c_true, centered = project_from_snapshot_scalars(
        s, artifact['retained_eigenvalues'], artifact['retained_eigenvectors'])
    metrics = projection_metrics(artifact['c_hat'], c_true, delta_norm)
    c.a.verify_model_unchanged(model, before, torch)
    del direction, model, eligible
    recheck()
    if c.a.checkpoint_file_records(args.base_directory) != historical['base']['files']:
        raise ValueError('Historical base changed during evaluation')
    if c.a.sha256_file(HISTORICAL_SPEC_PATH) != HISTORICAL_SPEC_SHA256:
        raise ValueError('Historical base spec changed during evaluation')
    result = {'format_version': 1, 'attempt_id': EVALUATION_SPEC['attempt_id'],
              'purpose': EVALUATION_SPEC['purpose'], 'hash_algorithm': 'sha256',
              'construction_manifest_sha256': hashes[str(args.construction_manifest_path)],
              'construction_artifact': record,
              'historical_base': historical['base'],
              'historical_spec_sha256': HISTORICAL_SPEC_SHA256,
              'snapshot_scalar_derivatives': s.tolist(),
              'centered_snapshot_scalar_derivatives': centered.tolist(),
              'metrics': metrics, 'pca_basis_gram':
                  c.a.load_json_object(args.construction_manifest_path,
                                       'Attempt023 construction manifest')['pca_basis_gram'],
              'functional_activation_evaluation': False,
              'selection': False, 'coefficient_or_pca_mutation': False,
              'input_source_hashes': hashes}
    c.a.write_manifest(args.output_path, result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in EVALUATION_SPEC['defaults'].items():
        path = Path(value)
        parser.add_argument('--'+key.replace('_', '-'), type=Path,
                            default=path if path.is_absolute() else PROJECT/path)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args(argv)


if __name__ == '__main__':
    evaluate(parse_args())
