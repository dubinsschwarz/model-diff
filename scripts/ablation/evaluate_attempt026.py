#!/usr/bin/env python3
"""Post-freeze signed oracle evaluation of Attempt 026 activation gradients."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '026_blind_direct_activation_gradient_block13_pilot'
PRIOR_PATH = Path(__file__).with_name('evaluate_attempt014.py')
PRIOR_SHA256 = '1193860390504d614522ae746f903d7304eb97d9b2b8d362fabcd302a4a77afd'
if hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest() != PRIOR_SHA256:
    raise ValueError('Frozen Attempt014 oracle validation source changed')
loader = importlib.util.spec_from_file_location('attempt026_oracle_validation014', PRIOR_PATH)
prior = importlib.util.module_from_spec(loader)
loader.loader.exec_module(prior)

CANDIDATES = ('mean_gradient', 'first16_mean_gradient', 'last16_mean_gradient',
              'even_mean_gradient', 'odd_mean_gradient')
CONSTRUCTION_SPEC_SHA256 = 'f2f4138db920fe2ea97636a28a306499ff4cde989505fe7603f41e7368b01189'
CONSTRUCTION_MANIFEST_SHA256 = 'bdef98532118d4e4f3d899745b32dfee2063ac3b7d99c194b3c42e58af8d6d40'
CONSTRUCTOR_SHA256 = '857e41b8eea9913da2561db56d53e7097e8c6c68262b7426f8978148d7542794'
CANDIDATE_ARTIFACT_SHA256 = 'a0990f3ab8e97b29f1a8eab9ac6e1feb3e84380e5660516881e35d83a94e0efa'
MEAN_RAW_SHA256 = 'ebd7d2d209e018530e6254485c1f07bacdc993749f6c7d6fe5de12f076edb222'
ORACLE_ARTIFACT_SHA256 = '339a6c09b10286369a5d10eb49e144d1819f4301e334c63930ca7c4dd3a11077'
ORACLE_DIFFERENCE_RAW_SHA256 = 'd60789a2c06eb82dba2f2449d8665ec665d85e9f7e33fdea1f66ac79c81c159d'
BENCHMARK_SHA256 = 'e27c011aa74dffd295351c319b4611782289009e72d68631edd3b18999da1337'
BENCHMARK_CONSENSUS = 0.4434394116233976
BENCHMARK_FINEWEB = 0.4420842139546442

FROZEN_SPEC = {
    'format_version': 1,
    'attempt_id': ATTEMPT,
    'purpose': 'post_freeze_signed_direct_activation_gradient_oracle_evaluation',
    'construction_inputs': {
        'spec': {'path': 'experiments/attempts/'+ATTEMPT+'/spec.json',
                 'sha256': CONSTRUCTION_SPEC_SHA256},
        'manifest': {'path': 'experiments/attempts/'+ATTEMPT+'/construction-manifest.json',
                     'sha256': CONSTRUCTION_MANIFEST_SHA256},
        'constructor': {'path': 'scripts/ablation/construct_direct_activation_gradient_pilot.py',
                        'sha256': CONSTRUCTOR_SHA256}},
    'candidate_artifact': {'path': '/root/model-diff-scratch/artifacts/attempt026_direct_activation_gradient/activation_gradient.pt',
                           'serialized_sha256': CANDIDATE_ARTIFACT_SHA256,
                           'mean_gradient_raw_sha256': MEAN_RAW_SHA256,
                           'candidate_tensors': list(CANDIDATES),
                           'batch_indices': list(range(32)),
                           'candidate_shape': [128, 2048],
                           'candidate_dtype': 'torch.float32'},
    'oracle': {'artifact_path': '/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt',
               'serialized_sha256': ORACLE_ARTIFACT_SHA256,
               'difference_raw_sha256': ORACLE_DIFFERENCE_RAW_SHA256,
               'vector': 'difference', 'definition': 'ft_mean - base_mean',
               'validation_source_path': 'scripts/ablation/evaluate_attempt014.py',
               'validation_source_sha256': PRIOR_SHA256,
               'provenance_inputs': copy.deepcopy(prior.FROZEN_SPEC['inputs'])},
    'benchmark': {'path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/evaluation.json',
                  'sha256': BENCHMARK_SHA256,
                  'consensus_primary': BENCHMARK_CONSENSUS,
                  'fineweb_primary': BENCHMARK_FINEWEB,
                  'use': 'context_only_no_candidate_selection_or_tuning'},
    'metrics': {'cosine_dtype': 'cpu_float64', 'sign': 'stored_positive_dL_dz',
                'candidate_order': list(CANDIDATES),
                'scientific_candidate': 'mean_gradient',
                'splits': 'descriptive_only',
                'per_position': list(range(127)),
                'position127_cosine': None,
                'position127_candidate_maxabs_limit': 1e-7,
                'position127_candidate_norm_limit': 1e-6,
                'primary_positions': [1, 2, 3, 4],
                'primary_aggregation': 'arithmetic_mean_of_four_individual_signed_cosines',
                'secondary_positions': list(range(1, 127)),
                'secondary_aggregation': 'arithmetic_mean_of_126_individual_signed_cosines',
                'position0': 'reported_separately',
                'undefined_cosine_or_required_mean': None,
                'no_flattened_primary': True,
                'no_absolute_cosine': True},
    'output_path': 'experiments/attempts/'+ATTEMPT+'/evaluation.json',
    'output_policy': {'overwrite': False, 'timestamps': False,
                      'host_metadata': False, 'gpu_metadata': False,
                      'construction_frozen_before_oracle': True,
                      'candidate_modified_after_oracle': False,
                      'sign_selection': False, 'candidate_selection': False,
                      'position_selection': False, 'candidate_rescaling': False},
}


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT/path


def require_hash(path, expected):
    prior.require_file_hash(path, expected)


def require_output_absent(path):
    if path.exists() or path.is_symlink():
        raise ValueError('Evaluation output already exists')


def validate_construction(spec, manifest):
    expected = FROZEN_SPEC['candidate_artifact']
    if (spec.get('attempt_id') != ATTEMPT or manifest.get('attempt_id') != ATTEMPT or
            manifest.get('format_version') != 1 or manifest.get('hash_algorithm') != 'sha256' or
            manifest.get('canonical_checkpoint_files') != spec.get('canonical_checkpoint_files') or
            manifest.get('frozen_corpus') != spec.get('frozen_corpus') or
            manifest.get('readout') != spec.get('readout') or
            manifest.get('batches') != spec.get('batches') or
            manifest.get('aggregation') != spec.get('aggregation') or
            manifest.get('position_policy') != spec.get('position_policy')):
        raise ValueError('Attempt026 construction manifest/spec mismatch')
    source_hashes = manifest.get('input_source_hashes', {})
    if (source_hashes.get(str(resolve_path(FROZEN_SPEC['construction_inputs']['spec']['path']))) !=
            CONSTRUCTION_SPEC_SHA256 or
            source_hashes.get(str(resolve_path(FROZEN_SPEC['construction_inputs']['constructor']['path']))) !=
            CONSTRUCTOR_SHA256):
        raise ValueError('Attempt026 construction source/spec linkage mismatch')
    artifact = manifest.get('artifact')
    if (not isinstance(artifact, dict) or artifact.get('path') != expected['path'] or
            artifact.get('serialized_sha256') != expected['serialized_sha256'] or
            artifact.get('raw_tensor_sha256', {}).get('mean_gradient') !=
                expected['mean_gradient_raw_sha256']):
        raise ValueError('Frozen Attempt026 candidate hash mismatch')
    keys = set(CANDIDATES) | {'batch_indices'}
    if set(artifact.get('raw_tensor_sha256', {})) != keys or set(artifact.get('tensors', {})) != keys:
        raise ValueError('Frozen Attempt026 artifact inventory mismatch')
    for name in keys:
        shape = [32] if name == 'batch_indices' else [128, 2048]
        dtype = 'torch.int64' if name == 'batch_indices' else 'torch.float32'
        if artifact['tensors'][name] != {'shape': shape, 'dtype': dtype,
                                        'device': 'cpu', 'contiguous': True}:
            raise ValueError('Frozen Attempt026 tensor metadata mismatch: '+name)
        prior.require_sha256(artifact['raw_tensor_sha256'][name], name)
    for flag in ('historical_base_access', 'oracle_access', 'adapter_access',
                 'sign_selection', 'candidate_rescaling',
                 'post_result_position_selection', 'optimizer_used',
                 'model_parameter_update'):
        if manifest.get(flag) is not False:
            raise ValueError('Attempt026 blind construction firewall mismatch: '+flag)
    if spec.get('readout', {}).get('block_index') != 13 or spec.get('readout', {}).get('hidden_size') != 2048:
        raise ValueError('Attempt026 readout mismatch')
    return artifact


def load_candidates(path, metadata):
    require_hash(path, CANDIDATE_ARTIFACT_SHA256)
    artifact = prior.load_torch_artifact(path, 'Attempt026 candidate artifact', torch)
    keys = set(CANDIDATES) | {'batch_indices'}
    if not isinstance(artifact, dict) or set(artifact) != keys:
        raise ValueError('Malformed Attempt026 candidate artifact')
    for name in CANDIDATES:
        prior.validate_activation_tensor(artifact[name], name,
                                         metadata['raw_tensor_sha256'][name], torch)
    indices = artifact['batch_indices']
    if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64 or
            indices.device.type != 'cpu' or not indices.is_contiguous() or
            tuple(indices.shape) != (32,) or
            not torch.equal(indices, torch.arange(32, dtype=torch.int64))):
        raise ValueError('Attempt026 artifact batch inventory mismatch')
    raw_indices = hashlib.sha256(indices.numpy().astype('<i8', copy=False).tobytes(order='C')).hexdigest()
    if raw_indices != metadata['raw_tensor_sha256']['batch_indices']:
        raise ValueError('Attempt026 batch index raw tensor SHA-256 mismatch')
    return artifact


def signed_cosine(left, right):
    if (not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor) or
            left.device.type != 'cpu' or right.device.type != 'cpu' or
            left.dtype != torch.float32 or right.dtype != torch.float32 or
            left.ndim != 1 or left.shape != right.shape or
            not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all())):
        raise ValueError('Invalid CPU float32 cosine vectors')
    x, y = left.double(), right.double()
    nx, ny = torch.linalg.vector_norm(x), torch.linalg.vector_norm(y)
    return None if float(nx) == 0 or float(ny) == 0 else float(torch.dot(x, y)/(nx*ny))


def required_mean(values):
    return None if any(value is None for value in values) else math.fsum(values)/len(values)


def candidate_report(name, candidate, difference):
    if (name not in CANDIDATES or tuple(candidate.shape) != (128, 2048) or
            tuple(difference.shape) != (128, 2048)):
        raise ValueError('Invalid candidate or oracle geometry')
    final = candidate[127].double()
    final_norm = float(torch.linalg.vector_norm(final))
    final_maxabs = float(final.abs().max())
    policy = FROZEN_SPEC['metrics']
    if (final_norm > policy['position127_candidate_norm_limit'] or
            final_maxabs > policy['position127_candidate_maxabs_limit']):
        raise ValueError('Causally unscored position 127 is not negligible')
    cosines = [signed_cosine(candidate[p], difference[p]) for p in range(127)]
    cosines.append(None)
    return {'candidate': name, 'position_0_cosine': cosines[0],
            'positions_1_4_individual_cosines': cosines[1:5],
            'positions_1_4_mean_cosine': required_mean(cosines[1:5]),
            'positions_1_126_mean_cosine': required_mean(cosines[1:127]),
            'position127_cosine': None,
            'position127_candidate_norm': final_norm,
            'position127_candidate_maxabs': final_maxabs,
            'positions': [{'position': p, 'cosine_similarity': value}
                          for p, value in enumerate(cosines)]}


def evaluate_candidates(artifact, difference):
    if not isinstance(artifact, dict) or set(artifact) != set(CANDIDATES) | {'batch_indices'}:
        raise ValueError('Expected all five frozen candidate tensors')
    reports = [candidate_report(name, artifact[name], difference) for name in CANDIDATES]
    return reports


def validate_benchmark(value):
    if (not isinstance(value, dict) or
            value.get('attempt_id') != '014_multicorpus_raw_jg_consensus_prefix0_13' or
            value.get('primary', {}).get('consensus') != BENCHMARK_CONSENSUS or
            value.get('primary', {}).get('fineweb') != BENCHMARK_FINEWEB or
            value.get('provenance', {}).get('oracle_artifact_sha256') != ORACLE_ARTIFACT_SHA256 or
            value.get('provenance', {}).get('oracle_raw_tensors_sha256', {}).get('difference') !=
                ORACLE_DIFFERENCE_RAW_SHA256):
        raise ValueError('Frozen Attempt014 benchmark mismatch')
    by_name = {row.get('candidate'): row for row in value.get('candidates', [])}
    if (by_name.get('consensus_response', {}).get('positions_1_4_mean_cosine') !=
            BENCHMARK_CONSENSUS or
            by_name.get('response_fineweb', {}).get('positions_1_4_mean_cosine') !=
            BENCHMARK_FINEWEB):
        raise ValueError('Attempt014 benchmark candidate report mismatch')
    return {'consensus_primary': BENCHMARK_CONSENSUS,
            'fineweb_primary': BENCHMARK_FINEWEB}


def validate_oracle_manifest(manifest):
    if (not isinstance(manifest, dict) or
            manifest.get('oracle_adl_sha256') != ORACLE_ARTIFACT_SHA256 or
            manifest.get('raw_tensors_sha256', {}).get('difference') !=
                ORACLE_DIFFERENCE_RAW_SHA256):
        raise ValueError('Frozen oracle artifact or difference hash mismatch')
    return manifest


def build_result(reports, benchmark, provenance, construction_commit=None):
    if [row['candidate'] for row in reports] != list(CANDIDATES):
        raise ValueError('All five frozen candidates must be reported in order')
    primary = reports[0]['positions_1_4_mean_cosine']
    comparison = {'mean_gradient_primary': primary,
                  'attempt014_consensus_primary': benchmark['consensus_primary'],
                  'attempt014_fineweb_primary': benchmark['fineweb_primary'],
                  'delta_vs_attempt014_consensus': None if primary is None else
                      primary-benchmark['consensus_primary'],
                  'delta_vs_attempt014_fineweb': None if primary is None else
                      primary-benchmark['fineweb_primary']}
    return {'format_version': 1, 'attempt_id': ATTEMPT,
            'evaluation_plan': FROZEN_SPEC['metrics'],
            'provenance': {**provenance, 'construction_commit_if_recorded': construction_commit},
            'candidates': reports,
            'scientific_candidate': 'mean_gradient',
            'split_aggregates': 'descriptive_robustness_only',
            'matched_attempt014_comparison': comparison,
            'construction_frozen_before_oracle': True,
            'candidate_modified_after_oracle': False,
            'sign_selection': False, 'candidate_selection': False,
            'position_selection': False, 'candidate_rescaling': False,
            'oracle_vector': 'difference', 'cosine_dtype': 'cpu_float64'}


def evaluate(spec_path):
    expected_spec = PROJECT/'experiments/attempts'/ATTEMPT/'evaluation_spec.json'
    if spec_path.resolve() != expected_spec.resolve():
        raise ValueError('Evaluation specification path differs from frozen path')
    spec = prior.load_json_object(spec_path, 'Attempt026 evaluation specification')
    if (set(spec) != set(FROZEN_SPEC) | {'evaluator_source_sha256'} or
            {key: value for key, value in spec.items()
             if key != 'evaluator_source_sha256'} != FROZEN_SPEC):
        raise ValueError('Attempt026 evaluation specification mismatch')
    output = resolve_path(spec['output_path'])
    require_output_absent(output)
    source = Path(__file__).resolve()
    inputs = {key: resolve_path(row['path']) for key, row in spec['construction_inputs'].items()}
    oracle_inputs = {key: resolve_path(row['path'])
                     for key, row in spec['oracle']['provenance_inputs'].items()}
    candidate_path = Path(spec['candidate_artifact']['path'])
    oracle_path = Path(spec['oracle']['artifact_path'])
    benchmark_path = resolve_path(spec['benchmark']['path'])
    all_paths = [spec_path, source, PRIOR_PATH, candidate_path, oracle_path,
                 benchmark_path, *inputs.values(), *oracle_inputs.values()]
    if any(output.resolve() == path.resolve() or path.resolve().is_relative_to(output.resolve())
           for path in all_paths):
        raise ValueError('Evaluation output overlaps a frozen input')
    hashes = {path: prior.sha256_file(path) for path in [*all_paths, spec_path]}
    for key, row in spec['construction_inputs'].items():
        if hashes[inputs[key]] != row['sha256']:
            raise ValueError('Frozen Attempt026 construction input hash mismatch: '+key)
    for key, row in spec['oracle']['provenance_inputs'].items():
        if hashes[oracle_inputs[key]] != row['sha256']:
            raise ValueError('Frozen oracle provenance input hash mismatch: '+key)
    if (hashes[source] != prior.require_sha256(
            spec['evaluator_source_sha256'], 'Attempt026 evaluator source') or
            hashes[PRIOR_PATH] != PRIOR_SHA256 or
            hashes[candidate_path] != CANDIDATE_ARTIFACT_SHA256 or
            hashes[oracle_path] != ORACLE_ARTIFACT_SHA256 or
            hashes[benchmark_path] != BENCHMARK_SHA256):
        raise ValueError('Frozen evaluator, candidate, oracle, or benchmark hash mismatch')
    construction_spec = prior.load_json_object(inputs['spec'], 'Attempt026 construction spec')
    manifest = prior.load_json_object(inputs['manifest'], 'Attempt026 construction manifest')
    metadata = validate_construction(construction_spec, manifest)
    oracle_attempt = prior.load_json_object(oracle_inputs['attempt_spec'], 'Attempt014 spec')
    oracle_expected_hashes = {key: row['sha256']
                              for key, row in spec['oracle']['provenance_inputs'].items()}
    oracle_manifest = validate_oracle_manifest(prior.validate_oracle_provenance(
        oracle_inputs, oracle_expected_hashes, oracle_attempt))
    artifact = load_candidates(candidate_path, metadata)
    oracle = prior.load_and_validate_oracle(
        oracle_path, {'oracle': {'oracle_manifest': oracle_manifest}}, torch)
    benchmark = validate_benchmark(prior.load_json_object(
        benchmark_path, 'Attempt014 frozen evaluation'))
    def recheck():
        for path, digest in hashes.items():
            require_hash(path, digest)
    recheck()
    reports = evaluate_candidates(artifact, oracle['difference'])
    provenance = {'evaluation_spec_sha256': hashes[spec_path],
                  'evaluator_source_sha256': hashes[source],
                  'oracle_validation_source_sha256': hashes[PRIOR_PATH],
                  'construction_spec_sha256': hashes[inputs['spec']],
                  'construction_manifest_sha256': hashes[inputs['manifest']],
                  'constructor_source_sha256': hashes[inputs['constructor']],
                  'candidate_artifact_sha256': hashes[candidate_path],
                  'candidate_raw_tensor_sha256': metadata['raw_tensor_sha256'],
                  'oracle_artifact_sha256': hashes[oracle_path],
                  'oracle_raw_tensors_sha256': oracle_manifest['raw_tensors_sha256'],
                  'oracle_provenance_input_sha256': oracle_expected_hashes,
                  'attempt014_evaluation_sha256': hashes[benchmark_path]}
    result = build_result(reports, benchmark, provenance,
                          manifest.get('construction_commit'))
    recheck()
    require_output_absent(output)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-spec', type=Path,
                        default=PROJECT/'experiments/attempts'/ATTEMPT/'evaluation_spec.json')
    args = parser.parse_args()
    evaluate(args.evaluation_spec)


if __name__ == '__main__':
    main()
