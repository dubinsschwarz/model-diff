#!/usr/bin/env python3
"""Attempt118: post-oracle, parameter-free stability projection of blind responses."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '118_postoracle_cross_corpus_stability_projection_diagnostic'
PRIOR = '100_diverse8_raw_jg_consensus_prefix0_13'
CORPORA = ('fineweb', 'wikitext103_raw', 'tinystories', 'arxiv_document',
           'cc_news', 'pg19', 'codeparrot_clean', 'ultrachat')
RESPONSE_NAMES = tuple('response_' + name for name in CORPORA)
ARTIFACT_NAMES = ('merged_mean', *RESPONSE_NAMES, 'consensus_response_8')
RAW_HASHES = {
    'merged_mean': '0f5cb336c4c54341734189bea5e2318e6aa292fed487334f68fbdd0cd8eca796',
    'response_fineweb': '00d0d8873df90e4b7c3e8799e0e019d3f5b8d7fee0d9aa796370849ff69ec5e5',
    'response_wikitext103_raw': '3d81d1479c0808868c78e5b251f329eb22143ccac16a615cabcb0874a0e15e3f',
    'response_tinystories': '4ea5f0006ab442c500ee196f3bd9cff6d5f7fe7325746e0de97ba215e54ce123',
    'response_arxiv_document': '221d1a89dd4ba20bf26d57a3c8ceffc4d67950ece81b8436d75bd77cbd11f963',
    'response_cc_news': '342f442c8dd5f0a9af95702479624abab0f1b1c424d18a8ff96314bf9a2f4ed4',
    'response_pg19': 'cddef2b4f5acecdfd48e12ff02e31d367637ab2b69da7a88c9cf6faf171c8672',
    'response_codeparrot_clean': 'd4630da18a86faabc26a533cbde9d6537f24251d5e272773d4f1340800ef9282',
    'response_ultrachat': '7b98750fd507e7a5e3c794fbb9e93d8af1b687a8acb2d6a2005ce414bacea115',
    'consensus_response_8': 'd8ae597ebeb6495dbb9825372c99ec203db47ece85f0f5018546efa2c1a1ac26',
}
ORACLE_VALIDATOR_SHA = '1193860390504d614522ae746f903d7304eb97d9b2b8d362fabcd302a4a77afd'
BASELINE_PRIMARY = 0.46821653512164474
BASELINE_SECONDARY = 0.4967711276664245

FROZEN_SPEC = {
    'format_version': 1, 'attempt_id': ATTEMPT,
    'purpose': 'postoracle_cross_corpus_response_stability_projection_diagnostic',
    'attempt100': {
        'spec': {'path': f'experiments/attempts/{PRIOR}/spec.json',
                 'sha256': 'a10b513387635cd599971b10a3c7614a7a18483db7b12f9058898c417101db12'},
        'manifest': {'path': f'experiments/attempts/{PRIOR}/construction-manifest.json',
                     'sha256': '64c4d3ef379fc02f6132d6e3197632e7bd5154398ee123e5ca688be9d5c68546'},
        'constructor': {'path': 'scripts/ablation/construct_attempt100_diverse8_consensus.py',
                        'sha256': 'c5109e6dcdae0234c42aace7b9f9bf19b9563d9b0b289342bba52c32122bac0a'},
        'artifact': {'path': '/root/model-diff-scratch/artifacts/attempt100_diverse8_raw_jg_consensus/responses.pt',
                     'serialized_sha256': 'ce32f037d594a45126d72f88ffcf1dfef49996bc6152aeb431f07851c50d3872',
                     'tensor_inventory': list(ARTIFACT_NAMES), 'raw_tensors_sha256': RAW_HASHES},
        'corpus_order': list(CORPORA),
        'evaluation': {'path': f'experiments/attempts/{PRIOR}/evaluation.json',
                       'sha256': '3607d599b347e7010902143655d2d68b0cba226237cb884b6751d1de998c08ef',
                       'committed_at': 'a0a58240944bb0712c032b82ebd81087fa16e159',
                       'primary': BASELINE_PRIMARY, 'secondary': BASELINE_SECONDARY,
                       'access': 'after_stability_response_raw_hash_barrier_only'},
    },
    'projection': {
        'independent_positions': list(range(128)), 'response_shape': [128, 2048],
        'input_dtype': 'contiguous_CPU_float32', 'arithmetic_dtype': 'cpu_float64',
        'unit_normalize_each_corpus_per_position': True,
        'mean': 'arithmetic_equal_weight_1_over_8_of_unit_responses',
        'consensus_reproduction_max_abs_tolerance': 1e-7,
        'disagreement': 'D_rows_u_c_minus_m', 'gram': 'K=D@D.T',
        'decomposition': 'torch.linalg.eigh_CPU_float64',
        'rank_rule': 'retain_eigenvalues_strictly_greater_than_eps_float64_times_8_times_max_eigenvalue',
        'projection': 'D.T@K_pseudoinverse@D@m',
        'stable': 'm_minus_projection',
        'output_dtype': 'contiguous_CPU_float32', 'renormalize_stable': False,
        'rank_selection_after_oracle': False, 'top_k_or_partial_projection_sweep': False,
    },
    'oracle': {
        'path': '/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt',
        'serialized_sha256': '339a6c09b10286369a5d10eb49e144d1819f4301e334c63930ca7c4dd3a11077',
        'vector': 'difference', 'definition': 'ft_mean - base_mean',
        'difference_raw_sha256': 'd60789a2c06eb82dba2f2449d8665ec665d85e9f7e33fdea1f66ac79c81c159d',
        'validator_source': {'path': 'scripts/ablation/evaluate_attempt014.py', 'sha256': ORACLE_VALIDATOR_SHA},
        'provenance': 'use_hash_pinned_attempt014_validate_construction_and_validate_oracle_provenance',
    },
    'metrics': {'sign': 'stored_positive_response_no_flip', 'dtype': 'cpu_float64',
                'candidates': ['consensus_response_8', 'stable_response', 'disagreement_overlap'],
                'positions': list(range(128)), 'primary_positions': [1, 2, 3, 4],
                'secondary_positions': list(range(1, 128)),
                'aggregation': 'arithmetic_mean_of_individual_signed_position_cosines',
                'absolute_cosine': False, 'flattened_cosine': False,
                'zero_norm_cosine': None, 'required_mean_if_any_null': None},
    'interpretation': {'clear': {'delta_primary_at_least': 0.03, 'delta_secondary_at_least': 0.02},
                       'moderate': {'delta_primary_at_least': 0.015, 'delta_secondary_at_least': 0.0},
                       'otherwise': 'no_meaningful_stability_separation_support',
                       'ordered_categories': ['clear_stability_separation_support',
                                              'moderate_stability_separation_support',
                                              'no_meaningful_stability_separation_support'],
                       'next_step': {
                           'clear_stability_separation_support': 'analogous_parameter_space_filter_before_modest_curvature_correction',
                           'moderate_stability_separation_support': 'parameter_space_test_optional_evidence_weak',
                           'no_meaningful_stability_separation_support': 'stop_simple_cross_corpus_stability_as_main_separation_hypothesis_on_this_specimen'}},
    'information_policy': {
        'postoracle_mechanistic_diagnostic': True,
        'same_specimen_postoracle_method_development': True, 'clean_blind_validation': False,
        'candidate_constructed_from_blind_frozen_responses_only': True,
        'candidate_hash_frozen_before_oracle_access_within_run': True,
        'oracle_access_after_candidate_hash': True,
        'sign_selection': False, 'corpus_selection': False, 'corpus_reweighting': False,
        'rank_selection_after_oracle': False, 'candidate_selection': False,
        'parameter_space_candidate': False, 'model_loading': False,
        'gradient_jvp_ggn_computation': False, 'posthoc_scaling': False,
        'position_selection': False,
    },
    'output_path': f'experiments/attempts/{ATTEMPT}/result.json',
    'output_policy': {'overwrite': False, 'large_artifact': False},
}


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f'Frozen SHA256 mismatch: {path}')


def raw_sha256(tensor):
    validate_tensor(tensor)
    return hashlib.sha256(tensor.numpy().astype('<f4', copy=False).tobytes()).hexdigest()


def validate_tensor(tensor):
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or
            tensor.dtype != torch.float32 or not tensor.is_contiguous() or
            tuple(tensor.shape) != (128, 2048) or not bool(torch.isfinite(tensor).all())):
        raise ValueError('Response must be finite contiguous CPU FP32 [128,2048]')


def require_output_absent(path):
    if path.exists() or path.is_symlink():
        raise ValueError('Attempt118 result already exists')


def validate_blind_provenance(plan, construction_spec, manifest):
    for key in ('spec', 'manifest', 'constructor'):
        require_hash(path_of(plan[key]['path']), plan[key]['sha256'])
    if (construction_spec.get('attempt_id') != PRIOR or manifest.get('attempt_id') != PRIOR or
            manifest.get('spec_sha256') != plan['spec']['sha256'] or
            manifest.get('constructor_sha256') != plan['constructor']['sha256'] or
            manifest.get('corpus_order') != plan['corpus_order'] or
            construction_spec.get('consensus', {}).get('corpus_order') != plan['corpus_order'] or
            manifest.get('consensus') != construction_spec.get('consensus') or
            manifest.get('artifact', {}).get('path') != plan['artifact']['path'] or
            manifest['artifact'].get('serialized_sha256') != plan['artifact']['serialized_sha256'] or
            manifest['artifact'].get('raw_tensors_sha256') != RAW_HASHES):
        raise ValueError('Attempt100 blind construction provenance mismatch')
    for key in ('historical_base_access', 'adapter_access', 'oracle_adl_access',
                'prior_evaluation_access', 'sign_selection',
                'corpus_selection_after_results', 'corpus_reweighting',
                'candidate_rescaling', 'post_result_position_selection'):
        if (construction_spec.get('firewall', {}).get(key) is not False or
                manifest.get(key) is not False):
            raise ValueError('Attempt100 blind construction firewall mismatch: '+key)


def load_blind_responses(plan, manifest):
    path = path_of(plan['artifact']['path'])
    require_hash(path, plan['artifact']['serialized_sha256'])
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != list(ARTIFACT_NAMES):
        raise ValueError('Attempt100 response artifact inventory mismatch')
    for name in ARTIFACT_NAMES:
        if raw_sha256(artifact[name]) != manifest['artifact']['raw_tensors_sha256'][name]:
            raise ValueError('Attempt100 response raw SHA256 mismatch: '+name)
    return artifact


def project_position(unit_rows, consensus, *, consensus_tolerance=1e-7):
    """Exact frozen 8-row disagreement projection for one position, in FP64."""
    if (not isinstance(unit_rows, torch.Tensor) or unit_rows.dtype != torch.float64 or
            unit_rows.device.type != 'cpu' or tuple(unit_rows.shape) != (8, 2048) or
            not bool(torch.isfinite(unit_rows).all())):
        raise ValueError('Expected eight finite CPU FP64 unit response rows')
    m = unit_rows.mean(dim=0)
    if (not isinstance(consensus, torch.Tensor) or consensus.dtype != torch.float32 or
            consensus.device.type != 'cpu' or tuple(consensus.shape) != (2048,) or
            not bool(torch.isfinite(consensus).all()) or
            float((m - consensus.double()).abs().max()) > consensus_tolerance):
        raise ValueError('Attempt100 consensus reproduction mismatch')
    d = unit_rows - m
    k = d @ d.T
    k = (k + k.T) / 2  # multiplication roundoff only
    eigenvalues, eigenvectors = torch.linalg.eigh(k)
    if not bool(torch.isfinite(eigenvalues).all()) or not bool(torch.isfinite(eigenvectors).all()):
        raise ValueError('Nonfinite disagreement eigendecomposition')
    maximum = float(eigenvalues[-1])
    if maximum < 0:
        raise ValueError('Negative disagreement Gram spectrum')
    tol = torch.finfo(torch.float64).eps * 8 * maximum
    if float(eigenvalues[0]) < -tol:
        raise ValueError('Materially negative disagreement Gram eigenvalue')
    keep = eigenvalues > tol
    if bool(keep.any()):
        q = eigenvectors[:, keep]
        projection = d.T @ (q @ ((q.T @ (d @ m)) / eigenvalues[keep]))
    else:
        projection = torch.zeros_like(m)
    stable = m - projection
    if not bool(torch.isfinite(stable).all()) or not bool(torch.isfinite(projection).all()):
        raise ValueError('Nonfinite stability projection')
    m_norm = float(torch.linalg.vector_norm(m))
    stable_norm = float(torch.linalg.vector_norm(stable))
    projection_norm = float(torch.linalg.vector_norm(projection))
    diagnostic = {
        'numerical_rank': int(keep.sum()),
        'k_eigenvalues_ascending': eigenvalues.tolist(),
        'rank_tolerance': tol,
        'consensus_norm': m_norm,
        'stable_norm': stable_norm,
        'projection_norm': projection_norm,
        'stable_to_consensus_norm_ratio': stable_norm / m_norm if m_norm > 0 else None,
        'max_abs_d_dot_stable': float((d @ stable).abs().max()),
        'max_abs_centered_sum': float(d.sum(dim=0).abs().max()),
        'consensus_reproduction_max_abs': float((m - consensus.double()).abs().max()),
    }
    return stable, projection, diagnostic


def construct_stability(artifact, spec=FROZEN_SPEC):
    if not isinstance(artifact, dict) or list(artifact) != list(ARTIFACT_NAMES):
        raise ValueError('Exact Attempt100 blind response inventory required')
    for name in ARTIFACT_NAMES:
        validate_tensor(artifact[name])
    rows = torch.stack([artifact[name] for name in RESPONSE_NAMES], dim=0).double()
    norms = torch.linalg.vector_norm(rows, dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError('Zero or nonfinite per-position corpus response')
    units = rows / norms.unsqueeze(-1)
    stable_rows, overlap_rows, diagnostics = [], [], []
    for position in range(128):
        stable, overlap, audit = project_position(
            units[:, position, :], artifact['consensus_response_8'][position],
            consensus_tolerance=spec['projection']['consensus_reproduction_max_abs_tolerance'])
        stable_rows.append(stable)
        overlap_rows.append(overlap)
        diagnostics.append({'position': position, **audit})
    stable_response = torch.stack(stable_rows).float().contiguous()
    disagreement_overlap = torch.stack(overlap_rows).float().contiguous()
    validate_tensor(stable_response)
    validate_tensor(disagreement_overlap)
    return stable_response, disagreement_overlap, diagnostics


def freeze_construction_record(spec, diagnostics, stable, overlap):
    """This function must finish before any oracle or evaluation file is opened."""
    stable_hash, overlap_hash = raw_sha256(stable), raw_sha256(overlap)
    record = {'attempt100_provenance': {key: value for key, value in spec['attempt100'].items()
                                        if key != 'evaluation'},
              'algorithm': spec['projection'],
              'position_diagnostics': diagnostics,
              'stable_response_raw_sha256': stable_hash,
              'disagreement_overlap_raw_sha256': overlap_hash,
              'construction_hash_barrier_completed': True}
    print('STABILITY_CONSTRUCTION_FROZEN', flush=True)
    print('stable_raw_sha256='+stable_hash, flush=True)
    return record


def load_oracle_validator():
    source = path_of(FROZEN_SPEC['oracle']['validator_source']['path'])
    require_hash(source, ORACLE_VALIDATOR_SHA)
    loader = importlib.util.spec_from_file_location('attempt118_oracle_validator014', source)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def load_validated_oracle():
    """Called only after freeze_construction_record returns and prints its barrier."""
    prior = load_oracle_validator()
    inputs = prior.FROZEN_SPEC['inputs']
    paths = {key: path_of(value['path']) for key, value in inputs.items()}
    hashes = {key: value['sha256'] for key, value in inputs.items()}
    for key in inputs:
        prior.require_file_hash(paths[key], hashes[key])
    attempt014, _ = prior.validate_construction(paths, hashes, prior.FROZEN_SPEC)
    oracle_manifest = prior.validate_oracle_provenance(paths, hashes, attempt014)
    expected = FROZEN_SPEC['oracle']
    if (oracle_manifest['oracle_adl_sha256'] != expected['serialized_sha256'] or
            oracle_manifest['raw_tensors_sha256']['difference'] != expected['difference_raw_sha256']):
        raise ValueError('Frozen oracle artifact/raw provenance mismatch')
    path = path_of(expected['path'])
    prior.require_file_hash(path, expected['serialized_sha256'])
    artifact = prior.load_and_validate_oracle(
        path, {'oracle': {'oracle_manifest': oracle_manifest}}, torch)
    return artifact['difference'].clone().contiguous()


def load_validated_baseline():
    """The Attempt100 oracle-scored evaluation is not accessed before the barrier."""
    pin = FROZEN_SPEC['attempt100']['evaluation']
    path = path_of(pin['path'])
    require_hash(path, pin['sha256'])
    value = json.loads(path.read_text())
    if (value.get('attempt_id') != PRIOR or value.get('scientific_candidate') != 'consensus_response_8' or
            value.get('construction_frozen_before_oracle') is not True or
            value.get('candidate_modified_after_oracle') is not False or
            value.get('sign_selection') is not False or
            value.get('candidate_selection') is not False or
            value.get('provenance', {}).get('construction_artifact_sha256') !=
                FROZEN_SPEC['attempt100']['artifact']['serialized_sha256']):
        raise ValueError('Frozen Attempt100 evaluation provenance mismatch')
    reports = value.get('candidates')
    if not isinstance(reports, list) or not reports or reports[0].get('candidate') != 'consensus_response_8':
        raise ValueError('Frozen Attempt100 consensus metric missing')
    if (reports[0].get('positions_1_4_mean_cosine') != pin['primary'] or
            reports[0].get('positions_1_127_mean_cosine') != pin['secondary']):
        raise ValueError('Frozen Attempt100 baseline mismatch')
    return {'primary': pin['primary'], 'secondary': pin['secondary'],
            'evaluation_sha256': pin['sha256']}


def required_mean(values):
    return None if any(value is None for value in values) else math.fsum(values) / len(values)


def cosine_report(left, right):
    validate_tensor(left)
    validate_tensor(right)
    values = []
    for position in range(128):
        x, y = left[position].double(), right[position].double()
        nx, ny = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
        values.append(None if float(nx) == 0 or float(ny) == 0 else float(torch.dot(x, y)/(nx*ny)))
    return {'position_0_cosine': values[0],
            'positions_1_4_individual_cosines': values[1:5],
            'positions_1_4_mean_cosine': required_mean(values[1:5]),
            'positions_1_127_mean_cosine': required_mean(values[1:128]),
            'all_128_position_cosines': values}


def interpret(primary_delta, secondary_delta):
    if primary_delta is None or secondary_delta is None:
        return 'undefined_required_position_cosine'
    rule = FROZEN_SPEC['interpretation']
    if (primary_delta >= rule['clear']['delta_primary_at_least'] and
            secondary_delta >= rule['clear']['delta_secondary_at_least']):
        return 'clear_stability_separation_support'
    if (primary_delta >= rule['moderate']['delta_primary_at_least'] and
            secondary_delta >= rule['moderate']['delta_secondary_at_least']):
        return 'moderate_stability_separation_support'
    return rule['otherwise']


def run(spec_path=None):
    spec_path = path_of(spec_path or f'experiments/attempts/{ATTEMPT}/spec.json')
    spec = json.loads(spec_path.read_text())
    if spec != FROZEN_SPEC:
        raise ValueError('Attempt118 spec differs from frozen source')
    result_path = path_of(spec['output_path'])
    require_output_absent(result_path)
    plan = spec['attempt100']
    construction_spec = json.loads(path_of(plan['spec']['path']).read_text())
    manifest = json.loads(path_of(plan['manifest']['path']).read_text())
    validate_blind_provenance(plan, construction_spec, manifest)
    artifact = load_blind_responses(plan, manifest)
    stable, overlap, diagnostics = construct_stability(artifact, spec)
    construction = freeze_construction_record(spec, diagnostics, stable, overlap)
    # Only code below this line may open oracle data or Attempt100 evaluation.json.
    difference = load_validated_oracle()
    baseline = load_validated_baseline()
    reports = {
        'consensus_response_8': cosine_report(artifact['consensus_response_8'], difference),
        'stable_response': cosine_report(stable, difference),
        'disagreement_overlap': cosine_report(overlap, difference),
    }
    consensus = reports['consensus_response_8']
    if (consensus['positions_1_4_mean_cosine'] != baseline['primary'] or
            consensus['positions_1_127_mean_cosine'] != baseline['secondary']):
        raise ValueError('Recomputed Attempt100 baseline does not match frozen evaluation')
    stable_report = reports['stable_response']
    p, s = stable_report['positions_1_4_mean_cosine'], stable_report['positions_1_127_mean_cosine']
    dp = None if p is None else p - baseline['primary']
    ds = None if s is None else s - baseline['secondary']
    category = interpret(dp, ds)
    result = {'format_version': 1, 'attempt_id': ATTEMPT,
              'provenance': {'spec_sha256': sha256_file(spec_path),
                             'source_sha256': sha256_file(Path(__file__)),
                             'attempt100_evaluation_sha256': baseline['evaluation_sha256'],
                             'oracle_artifact_sha256': spec['oracle']['serialized_sha256'],
                             'oracle_difference_raw_sha256': spec['oracle']['difference_raw_sha256']},
              'blind_construction': construction,
              'candidate_reports': reports,
              'baseline': baseline,
              'delta_primary_stable_minus_consensus': dp,
              'delta_secondary_stable_minus_consensus': ds,
              'interpretation': category,
              'next_step_policy': spec['interpretation']['next_step'].get(category),
              **spec['information_policy']}
    require_output_absent(result_path)
    with result_path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=PROJECT/'experiments/attempts'/ATTEMPT/'spec.json')
    args = parser.parse_args()
    run(args.spec)


if __name__ == '__main__':
    main()
