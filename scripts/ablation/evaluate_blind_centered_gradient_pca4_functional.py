#!/usr/bin/env python3
"""Post-freeze activation-space evaluation of the fixed Attempt-023 candidate."""
import argparse
import hashlib
import importlib.util
import math
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '023_blind_centered_gradient_pca4_regression_pilot'
ATTEMPT_DIR = PROJECT/'experiments/attempts'/ATTEMPT
SOURCE_HASHES = {
    'scripts/ablation/construct_blind_centered_gradient_pca4_pilot.py':
        '279b86aae0ae2f7cc74e76e73e43db9286f9b3bd80cf14116834f067b445d82d',
    'scripts/ablation/construct_generic_hessian_krylov.py':
        '9edce220e1f9e29b4e02082852d03aa6b8043e35964a546383a3c10111018f44',
    'scripts/ablation/construct_generic_gradient_linear_response.py':
        '5e80d37ad8379963b8cf1ba46e99d6c98e04514aae19eb637f3242ba8eaa4840',
    'scripts/ablation/evaluate_attempt014.py':
        '1193860390504d614522ae746f903d7304eb97d9b2b8d362fabcd302a4a77afd',
    'scripts/ablation/evaluate_attempt012.py':
        '82ecdbb0691f583b5bedee931f95a1bf3356ac5279db023dfc169789b896b364',
}


def load_frozen_source(relative, name):
    path = PROJECT/relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_HASHES[relative]:
        raise ValueError('Frozen functional-evaluation source mismatch: '+relative)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


c = load_frozen_source('scripts/ablation/construct_blind_centered_gradient_pca4_pilot.py',
                       'attempt023_frozen_constructor')
activation = load_frozen_source('scripts/ablation/construct_generic_gradient_linear_response.py',
                                'attempt023_native_jvp')
oracle_support = load_frozen_source('scripts/ablation/evaluate_attempt014.py',
                                    'attempt023_oracle_provenance')
consistency = load_frozen_source('scripts/ablation/evaluate_attempt012.py',
                                 'attempt023_merged_mean_consistency')

FROZEN_SPEC = {
    'format_version': 1,
    'attempt_id': ATTEMPT,
    'purpose': 'post_freeze_positive_candidate_activation_trace_no_selection',
    'paths': {
        'construction_spec': 'experiments/attempts/'+ATTEMPT+'/spec.json',
        'construction_manifest': 'experiments/attempts/'+ATTEMPT+'/construction-manifest.json',
        'stats': '/root/model-diff-scratch/artifacts/attempt023_centered_gradient_pca4/stats.pt',
        'final': '/root/model-diff-scratch/models/merged',
        'corpus_spec': c.FROZEN_SPEC['defaults']['corpus_spec_path'],
        'corpus_manifest': c.FROZEN_SPEC['defaults']['corpus_manifest_path'],
        'fineweb_tokens': c.FROZEN_SPEC['defaults']['tokens_path'],
        'probe': '/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt',
        'oracle_artifact': '/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt',
        'output': 'experiments/attempts/'+ATTEMPT+'/functional-evaluation.json',
    },
    'frozen_hashes': {
        'construction_spec': 'e5638e550fcb269014efbcddfb36cbe0622d63723ae4f10f75580b4a8a0dbe54',
        'construction_manifest': 'e6e7c8255a38af828ea92cd398f322022db72d62abe9ad06b583345fef82383e',
        'stats': '17d24baead0e235a6571e261fd79721f02ca78254bbe7e5461a31680ffbbd39a',
        'corpus_spec': c.FROZEN_SPEC['frozen_corpus']['corpus_spec_sha256'],
        'corpus_manifest': c.FROZEN_SPEC['frozen_corpus']['corpus_manifest_sha256'],
        'fineweb_tokens': c.FROZEN_SPEC['frozen_corpus']['serialized_sha256'],
        'probe': activation.FROZEN_SPEC['probe']['serialized_sha256'],
        'oracle_artifact': '339a6c09b10286369a5d10eb49e144d1819f4301e334c63930ca7c4dd3a11077',
    },
    'sources': SOURCE_HASHES,
    'canonical_checkpoint_files': c.FROZEN_SPEC['final_checkpoint_files'],
    'frozen_corpus': c.FROZEN_SPEC['frozen_corpus'],
    'coordinates': c.FROZEN_SPEC['coordinates'],
    'basis_microbatch_indices': c.FROZEN_SPEC['batches']['basis_indices'],
    'candidate': {
        'pca_reconstruction': c.FROZEN_SPEC['pca']['q_arithmetic'],
        'coefficient_cast': 'FP32(c_hat[j])_once',
        'arithmetic': 'for_each_matrix_FP32_zero_then_sequential_j0_to_j3_add_FP32_coefficient_times_realized_FP32_q_j',
        'normalization': False, 'sign_selection': False,
        'rescaling_before_jvp': False, 'oracle_dependent_selection': False,
        'snapshot_gram_replay_rtol': 1e-6,
        'snapshot_gram_replay_atol': 1e-6,
        'pca_reproduction_rtol': 1e-9,
        'pca_reproduction_atol': 1e-9,
        'coefficient_reproduction_rtol': 1e-12,
        'coefficient_reproduction_atol': 1e-12,
        'basis_gram_replay_atol': 1e-6,
    },
    'model': activation.FROZEN_SPEC['model'],
    'probe': activation.FROZEN_SPEC['probe'],
    'readout': activation.FROZEN_SPEC['readout'],
    'jvp': {**activation.FROZEN_SPEC['jvp'], 'response_sign': 'positive_J1_candidate'},
    'oracle_provenance': {
        key: oracle_support.FROZEN_SPEC['inputs'][key]
        for key in ('downloaded_manifest', 'merged_manifest', 'probe_manifest',
                    'oracle_manifest', 'merge_script', 'freeze_probe_script', 'oracle_script')},
    'oracle_sign': 'ft_mean_minus_base_mean',
    'positions': list(range(128)),
    'functional_summaries': {'position_0': 'separate',
                             'primary_positions': list(range(1, 5)),
                             'secondary_positions': list(range(1, 128)),
                             'null_policy': 'mean_null_if_any_required_cosine_null'},
    'merged_mean_consistency': {'minimum_cosine_similarity': 0.999999999,
                                'maximum_relative_rms_difference': 1e-5,
                                'maximum_absolute_difference': 0.01},
    'output_policy': {'exclusive_create': True, 'response_artifact': False,
                      'timestamps': False, 'host_metadata': False,
                      'gpu_metadata': False, 'selection': False,
                      'sign_flip': False, 'coefficient_or_pca_mutation': False,
                      'candidate_rescaling_before_jvp': False,
                      'historical_base_access': False},
}


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else PROJECT/value


def require_hash(path, digest):
    if c.a.sha256_file(path) != digest:
        raise ValueError('Frozen functional-evaluation SHA256 mismatch: '+str(path))


def require_output_absent(path):
    if path.exists() or path.is_symlink():
        raise ValueError('Functional evaluation output already exists')


def validate_construction_links(manifest, spec):
    if (manifest.get('attempt_id') != spec['attempt_id'] or
            manifest.get('purpose') != c.FROZEN_SPEC['purpose'] or
            manifest.get('information_policy') != c.FROZEN_SPEC['information_policy'] or
            manifest.get('source_checkpoint_files') != spec['canonical_checkpoint_files'] or
            manifest.get('corpus') != spec['frozen_corpus'] or
            manifest.get('coordinates') != spec['coordinates'] or
            manifest.get('batches') != c.FROZEN_SPEC['batches'] or
            manifest.get('pca_policy') != c.FROZEN_SPEC['pca'] or
            manifest.get('regression_policy') != c.FROZEN_SPEC['regression'] or
            manifest.get('artifact', {}).get('path') != spec['paths']['stats'] or
            manifest.get('artifact', {}).get('serialized_sha256') != spec['frozen_hashes']['stats']):
        raise ValueError('Frozen Attempt-023 construction linkage mismatch')
    expected_sources = {
        str(resolve(spec['paths']['construction_spec'])): spec['frozen_hashes']['construction_spec'],
        str(resolve(spec['paths']['corpus_spec'])): spec['frozen_hashes']['corpus_spec'],
        str(resolve(spec['paths']['corpus_manifest'])): spec['frozen_hashes']['corpus_manifest'],
        str(resolve(spec['paths']['fineweb_tokens'])): spec['frozen_hashes']['fineweb_tokens'],
        str(PROJECT/'scripts/ablation/construct_blind_centered_gradient_pca4_pilot.py'):
            spec['sources']['scripts/ablation/construct_blind_centered_gradient_pca4_pilot.py'],
        str(PROJECT/'scripts/ablation/construct_generic_hessian_krylov.py'):
            spec['sources']['scripts/ablation/construct_generic_hessian_krylov.py'],
    }
    if manifest.get('input_source_hashes') != expected_sources:
        raise ValueError('Frozen Attempt-023 source hash linkage mismatch')


def reproduce_frozen_solution(artifact, manifest, spec):
    policy = spec['candidate']
    K, eigenvalues, eigenvectors, _ = c.centered_pca(
        artifact['raw_gradient_gram_G'], c.FROZEN_SPEC['pca'])
    if (not torch.allclose(K, artifact['centered_gradient_gram_K'],
                           rtol=policy['pca_reproduction_rtol'], atol=policy['pca_reproduction_atol']) or
            not torch.allclose(eigenvalues, artifact['retained_eigenvalues'],
                               rtol=policy['pca_reproduction_rtol'], atol=policy['pca_reproduction_atol']) or
            not torch.allclose(eigenvectors, artifact['retained_eigenvectors'],
                               rtol=policy['pca_reproduction_rtol'], atol=policy['pca_reproduction_atol'])):
        raise ValueError('Frozen PCA eigenpairs do not reproduce')
    c_hat, _ = c.solve_centered(artifact['regression_b'], artifact['regression_M'],
                                c.FROZEN_SPEC['regression'])
    if (not torch.allclose(c_hat, artifact['c_hat'],
                           rtol=policy['coefficient_reproduction_rtol'],
                           atol=policy['coefficient_reproduction_atol']) or
            artifact['c_hat'].tolist() != manifest.get('regression', {}).get('coefficients')):
        raise ValueError('Frozen c_hat does not reproduce')
    return artifact['retained_eigenvalues'].clone(), artifact['retained_eigenvectors'].clone(), artifact['c_hat'].clone()


def validate_frozen_inputs(spec):
    if spec != FROZEN_SPEC:
        raise ValueError('Frozen functional evaluation spec mismatch')
    paths = {key: resolve(value) for key, value in spec['paths'].items()}
    require_output_absent(paths['output'])
    frozen_paths = ('construction_spec', 'construction_manifest', 'stats',
                    'corpus_spec', 'corpus_manifest', 'fineweb_tokens')
    hashes = {}
    for key in frozen_paths:
        require_hash(paths[key], spec['frozen_hashes'][key])
        hashes[str(paths[key])] = spec['frozen_hashes'][key]
    for relative, digest in spec['sources'].items():
        path = PROJECT/relative
        require_hash(path, digest)
        hashes[str(path)] = digest
    construction_spec = c.a.load_json_object(paths['construction_spec'], 'Attempt023 spec')
    if construction_spec != c.FROZEN_SPEC:
        raise ValueError('Frozen Attempt023 construction spec mismatch')
    manifest = c.a.load_json_object(paths['construction_manifest'], 'Attempt023 construction manifest')
    validate_construction_links(manifest, spec)
    artifact, record = c.verify_artifact(paths['stats'], manifest['artifact'])
    if record['serialized_sha256'] != spec['frozen_hashes']['stats']:
        raise ValueError('Frozen Attempt023 stats hash mismatch')
    reproduce_frozen_solution(artifact, manifest, spec)
    if c.a.checkpoint_file_records(paths['final']) != spec['canonical_checkpoint_files']:
        raise ValueError('Canonical final checkpoint mismatch')
    corpus = c.a.verify_corpus_manifest(paths['corpus_manifest'], paths['corpus_spec'],
        paths['fineweb_tokens'], paths['final'], c.a.load_corpus_spec(paths['corpus_spec']))
    if any(corpus[key] != spec['frozen_corpus'][key]
           for key in ('serialized_sha256', 'raw_tensor_sha256')):
        raise ValueError('Frozen FineWeb corpus mismatch')
    tokens = c.a.load_corpus_tokens(paths['fineweb_tokens'], corpus['raw_tensor_sha256'],
                                    4096, 128, torch)
    hashes[str(ATTEMPT_DIR/'functional_evaluation_spec.json')] = c.a.sha256_file(
        ATTEMPT_DIR/'functional_evaluation_spec.json')
    hashes[str(Path(__file__).resolve())] = c.a.sha256_file(Path(__file__).resolve())
    protected = [Path(value) for value in hashes]
    protected.extend((paths['probe'], paths['oracle_artifact']))
    if (paths['output'].resolve() in {path.resolve() for path in protected} or
            paths['output'].resolve().is_relative_to(paths['final'].resolve())):
        raise ValueError('Functional evaluation output overlaps frozen input')
    return paths, manifest, artifact, record, tokens, hashes


def form_candidate(basis, c_hat, coordinates):
    if (len(basis) != 4 or c_hat.shape != (4,) or c_hat.dtype != torch.float64 or
            c_hat.device.type != 'cpu' or not bool(torch.isfinite(c_hat).all())):
        raise ValueError('Invalid frozen PCA coefficients or directions')
    names = {row['name'] for row in coordinates}
    if any(set(vector) != names for vector in basis):
        raise ValueError('PCA direction coordinate inventory mismatch')
    before = c_hat.clone()
    coefficients = c_hat.to(dtype=torch.float32).contiguous()
    candidate, squares = {}, []
    for row in coordinates:
        name = row['name']
        value = torch.zeros(row['shape'], dtype=torch.float32, device='cpu')
        for j in range(4):
            q = basis[j][name]
            if (q.dtype != torch.float32 or q.device.type != 'cpu' or
                    list(q.shape) != row['shape'] or not q.is_contiguous() or
                    not bool(torch.isfinite(q).all())):
                raise ValueError('Invalid realized FP32 PCA matrix')
            value = value+coefficients[j]*q
        value = value.contiguous()
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Nonfinite realized FP32 candidate')
        candidate[name] = value
        squares.append(float(torch.sum(value.double().square(), dtype=torch.float64)))
    norm = math.sqrt(math.fsum(squares))
    if not math.isfinite(norm) or norm <= 0 or not torch.equal(c_hat, before):
        raise ValueError('Candidate norm invalid or frozen c_hat changed')
    return candidate, norm


def reconstruct_frozen_candidate(model, tokens, artifact, manifest, spec):
    eligible = c.a.discover_eligible_linear_weights(model, torch)
    coordinates = [{'name': name+'.weight', 'shape': list(module.weight.shape)}
                   for name, module in eligible]
    if coordinates != spec['coordinates']:
        raise ValueError('Final model early-coordinate inventory mismatch')
    eigenvalues = artifact['retained_eigenvalues'].clone()
    eigenvectors = artifact['retained_eigenvectors'].clone()
    c_hat = artifact['c_hat'].clone()
    c.freeze_for_snapshots(model, [row['name'] for row in coordinates])
    device = eligible[0][1].weight.device
    snapshots = []
    for j in spec['basis_microbatch_indices']:
        batch = tokens[8*j:8*(j+1)].to(device)
        snapshot = c.gradient_snapshot(model, batch, coordinates)
        snapshots.append(snapshot)
        del batch, snapshot
    G = c.snapshot_gram(snapshots, coordinates)
    replay_error = float((G-artifact['raw_gradient_gram_G']).abs().max())
    if not torch.allclose(G, artifact['raw_gradient_gram_G'],
                          rtol=spec['candidate']['snapshot_gram_replay_rtol'],
                          atol=spec['candidate']['snapshot_gram_replay_atol']):
        raise ValueError('Recomputed basis-gradient Gram disagrees with frozen stats')
    basis = c.reconstruct_pca_basis(snapshots, coordinates, eigenvalues, eigenvectors)
    del snapshots
    gram = c.basis_gram(basis, coordinates, c.FROZEN_SPEC['pca'])
    reference = torch.tensor(manifest['pca_basis_gram']['gram_matrix'], dtype=torch.float64)
    measured = torch.tensor(gram['gram_matrix'], dtype=torch.float64)
    gram_error = float((measured-reference).abs().max())
    if gram_error > spec['candidate']['basis_gram_replay_atol']:
        raise ValueError('Recomputed realized PCA basis Gram disagrees with frozen audit')
    candidate, norm = form_candidate(basis, c_hat, coordinates)
    del basis
    coefficients32 = c_hat.float().double()
    implied_squared = float(coefficients32@measured@coefficients32)
    if not math.isfinite(implied_squared) or implied_squared <= 0:
        raise ValueError('Invalid Gram-implied candidate norm')
    if (not torch.equal(eigenvalues, artifact['retained_eigenvalues']) or
            not torch.equal(eigenvectors, artifact['retained_eigenvectors']) or
            not torch.equal(c_hat, artifact['c_hat'])):
        raise ValueError('Frozen PCA or c_hat changed during reconstruction')
    c.freeze_for_t_space(model)
    return candidate, {'candidate_parameter_norm': norm,
                       'gram_implied_candidate_norm': math.sqrt(implied_squared),
                       'snapshot_gram_replay_max_abs_difference': replay_error,
                       'pca_basis_gram_replay_max_abs_difference': gram_error,
                       'pca_basis_gram': gram,
                       'frozen_coefficients': c_hat.tolist(),
                       'coefficient_cast': spec['candidate']['coefficient_cast'],
                       'candidate_arithmetic': spec['candidate']['arithmetic']}


def compute_complete_response(model, candidate, probe, spec):
    device = next(model.parameters()).device
    tangents = {name: value.to(device=device).contiguous()
                for name, value in candidate.items()}
    del candidate
    try:
        merged, response, audit = activation.compute_probe_response(model, probe,
                                                                     tangents, spec, torch)
    finally:
        tangents.clear()
        model.zero_grad(set_to_none=True)
    merged = merged.float().contiguous()
    response = response.float().contiguous()
    raw_hash = activation.sha256_raw_float32_tensor(response, torch)
    return merged, response, audit, raw_hash


def validate_oracle_after_response(spec, paths, hashes):
    oracle_paths = {key: resolve(record['path'])
                    for key, record in spec['oracle_provenance'].items()}
    oracle_hashes = {key: record['sha256']
                     for key, record in spec['oracle_provenance'].items()}
    for key, path in oracle_paths.items():
        require_hash(path, oracle_hashes[key])
        hashes[str(path)] = oracle_hashes[key]
    oracle_manifest = oracle_support.validate_oracle_provenance(
        oracle_paths, oracle_hashes, spec)
    if oracle_manifest['oracle_adl_sha256'] != spec['frozen_hashes']['oracle_artifact']:
        raise ValueError('Frozen oracle artifact provenance hash mismatch')
    require_hash(paths['oracle_artifact'], spec['frozen_hashes']['oracle_artifact'])
    hashes[str(paths['oracle_artifact'])] = spec['frozen_hashes']['oracle_artifact']
    return oracle_support.load_and_validate_oracle(
        paths['oracle_artifact'], {'oracle': {'oracle_manifest': oracle_manifest}}, torch)


def functional_geometry(response, oracle_difference, spec):
    if (response.shape != (128, 2048) or oracle_difference.shape != response.shape or
            response.dtype != torch.float32 or oracle_difference.dtype != torch.float32 or
            response.device.type != 'cpu' or oracle_difference.device.type != 'cpu' or
            not bool(torch.isfinite(response).all()) or
            not bool(torch.isfinite(oracle_difference).all())):
        raise ValueError('Invalid candidate or oracle activation response')
    rows = []
    for position in spec['positions']:
        h = response[position].double()
        d = oracle_difference[position].double()
        hh = float(torch.dot(h, h))
        dd = float(torch.dot(d, d))
        hd = float(torch.dot(h, d))
        scale = hd/hh if hh else None
        residual = (float(torch.linalg.vector_norm(d-scale*h))/math.sqrt(dd)
                    if scale is not None and dd else None)
        rows.append({'position': position,
                     'cosine': hd/math.sqrt(hh*dd) if hh and dd else None,
                     'response_norm': math.sqrt(hh),
                     'oracle_difference_norm': math.sqrt(dd),
                     'norm_ratio': math.sqrt(hh/dd) if dd else None,
                     'optimal_scalar': scale,
                     'relative_residual_after_optimal_rescaling': residual})
    def summary(positions):
        values = [rows[position]['cosine'] for position in positions]
        valid = all(value is not None for value in values)
        return {'positions': positions,
                'mean': math.fsum(values)/len(values) if valid else None,
                'min': min(values) if valid else None,
                'max': max(values) if valid else None}
    settings = spec['functional_summaries']
    return {'positions': rows, 'position_0': rows[0],
            'individual_positions_1_4': rows[1:5],
            'primary_positions_1_4': summary(settings['primary_positions']),
            'secondary_positions_1_127': summary(settings['secondary_positions'])}


def evaluate(args):
    spec_path = ATTEMPT_DIR/'functional_evaluation_spec.json'
    spec = c.a.load_json_object(spec_path, 'Attempt023 functional evaluation spec')
    paths, manifest, artifact, record, tokens, hashes = validate_frozen_inputs(spec)
    model = activation.load_local_model(paths['final'], args.device, torch)
    model_before = activation.model_state_hashes(model, torch)
    candidate, candidate_audit = reconstruct_frozen_candidate(
        model, tokens, artifact, manifest, spec)
    del tokens
    # Candidate construction is complete before the oracle probe or oracle difference is read.
    require_hash(paths['probe'], spec['frozen_hashes']['probe'])
    hashes[str(paths['probe'])] = spec['frozen_hashes']['probe']
    probe = activation.load_probe(paths['probe'], spec['probe'], torch)
    merged, response, readout, response_hash = compute_complete_response(
        model, candidate, probe, spec)
    del candidate, probe
    activation.verify_model_unchanged(model, model_before, torch)
    del model
    # Oracle provenance and difference enter only after the complete response exists.
    oracle = validate_oracle_after_response(spec, paths, hashes)
    mean_audit = consistency.merged_mean_consistency(merged, oracle['ft_mean'], spec)
    geometry = functional_geometry(response, oracle['difference'], spec)
    for path, digest in hashes.items():
        require_hash(Path(path), digest)
    if c.a.checkpoint_file_records(paths['final']) != spec['canonical_checkpoint_files']:
        raise ValueError('Canonical final checkpoint changed during functional evaluation')
    require_output_absent(paths['output'])
    output = {'format_version': 1, 'attempt_id': ATTEMPT,
              'purpose': spec['purpose'], 'hash_algorithm': 'sha256',
              'functional_evaluation_spec_sha256': hashes[str(spec_path)],
              'construction_manifest_sha256': spec['frozen_hashes']['construction_manifest'],
              'frozen_stats': record, 'input_source_hashes': hashes,
              'basis_microbatch_indices': spec['basis_microbatch_indices'],
              'candidate': candidate_audit,
              'candidate_parameter_norm': candidate_audit['candidate_parameter_norm'],
              'readout_jvp_audit': readout,
              'merged_mean_consistency_audit': mean_audit,
              'functional_geometry': geometry,
              'raw_response_sha256': response_hash,
              'functional_sign': spec['jvp']['response_sign'],
              'oracle_sign': spec['oracle_sign'],
              'selection': False, 'sign_flip': False,
              'coefficient_or_pca_mutation': False,
              'candidate_rescaling_before_jvp': False,
              'historical_base_access': False}
    c.a.write_manifest(paths['output'], output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args(argv)


if __name__ == '__main__':
    evaluate(parse_args())
