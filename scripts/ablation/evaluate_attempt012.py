#!/usr/bin/env python3
"""Frozen Attempt 012: CPU-only geometry, one candidate, no model loading."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '012_ritz_residual_filtered_inverse_k4_prefix0_13',
 'method': 'cpu_float64_geometry_only',
 'inputs': {'attempt_spec': {'path': 'experiments/attempts/012_ritz_residual_filtered_inverse_k4_prefix0_13/spec.json',
                             'sha256': 'ad930f8f8db3b4a6033d59471b76cc4fdd49dc8ea296619b2e9f55ad9fcb4418'},
            'construction_manifest': {'path': 'experiments/attempts/012_ritz_residual_filtered_inverse_k4_prefix0_13/construction-manifest.json',
                                      'sha256': 'c179958d16fbe33879ead93741339b5750923d673d39c62e17a58ac4910d38e2'},
            'candidate_artifact': {'path': '/root/model-diff-scratch/artifacts/attempt012_ritz_residual_filtered_inverse/inverse_response.pt'},
            'oracle_manifest': {'path': 'oracle-adl-manifest.json',
                                'sha256': 'affe93a1139cdaa221646fb3b9c8ec5f64b28129c42b22eae185aca5d3932b90'},
            'oracle_artifact': {'path': '/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt'}},
 'output_path': 'experiments/attempts/012_ritz_residual_filtered_inverse_k4_prefix0_13/evaluation.json',
 'tensor_shape': [128, 2048],
 'tensor_dtype': 'float32',
 'candidate_vector': 'inverse_response',
 'candidate_sign': 'positive_J_x_filtered_inverse',
 'oracle_vector': 'difference',
 'oracle_sign': 'ft_mean - base_mean',
 'positions': [0,
               1,
               2,
               3,
               4,
               5,
               6,
               7,
               8,
               9,
               10,
               11,
               12,
               13,
               14,
               15,
               16,
               17,
               18,
               19,
               20,
               21,
               22,
               23,
               24,
               25,
               26,
               27,
               28,
               29,
               30,
               31,
               32,
               33,
               34,
               35,
               36,
               37,
               38,
               39,
               40,
               41,
               42,
               43,
               44,
               45,
               46,
               47,
               48,
               49,
               50,
               51,
               52,
               53,
               54,
               55,
               56,
               57,
               58,
               59,
               60,
               61,
               62,
               63,
               64,
               65,
               66,
               67,
               68,
               69,
               70,
               71,
               72,
               73,
               74,
               75,
               76,
               77,
               78,
               79,
               80,
               81,
               82,
               83,
               84,
               85,
               86,
               87,
               88,
               89,
               90,
               91,
               92,
               93,
               94,
               95,
               96,
               97,
               98,
               99,
               100,
               101,
               102,
               103,
               104,
               105,
               106,
               107,
               108,
               109,
               110,
               111,
               112,
               113,
               114,
               115,
               116,
               117,
               118,
               119,
               120,
               121,
               122,
               123,
               124,
               125,
               126,
               127],
 'merged_mean_consistency': {'scope': 'each_of_all_128_positions_independently',
                             'minimum_cosine_similarity': 0.999999999,
                             'maximum_relative_rms_difference': 1e-05,
                             'maximum_absolute_difference': 0.01,
                             'relative_rms_definition': 'rms(merged_mean - ft_mean) / rms(ft_mean)',
                             'zero_vector_policy': 'both_zero_pass; exactly_one_zero_fails'},
 'summaries': {'primary_positions': [1, 2, 3, 4],
               'primary_metric': 'arithmetic_mean_of_per_position_cosine_similarities',
               'individual_positions': [0, 1, 2, 3, 4],
               'descriptive_positions': [1,
                                         2,
                                         3,
                                         4,
                                         5,
                                         6,
                                         7,
                                         8,
                                         9,
                                         10,
                                         11,
                                         12,
                                         13,
                                         14,
                                         15,
                                         16,
                                         17,
                                         18,
                                         19,
                                         20,
                                         21,
                                         22,
                                         23,
                                         24,
                                         25,
                                         26,
                                         27,
                                         28,
                                         29,
                                         30,
                                         31,
                                         32,
                                         33,
                                         34,
                                         35,
                                         36,
                                         37,
                                         38,
                                         39,
                                         40,
                                         41,
                                         42,
                                         43,
                                         44,
                                         45,
                                         46,
                                         47,
                                         48,
                                         49,
                                         50,
                                         51,
                                         52,
                                         53,
                                         54,
                                         55,
                                         56,
                                         57,
                                         58,
                                         59,
                                         60,
                                         61,
                                         62,
                                         63,
                                         64,
                                         65,
                                         66,
                                         67,
                                         68,
                                         69,
                                         70,
                                         71,
                                         72,
                                         73,
                                         74,
                                         75,
                                         76,
                                         77,
                                         78,
                                         79,
                                         80,
                                         81,
                                         82,
                                         83,
                                         84,
                                         85,
                                         86,
                                         87,
                                         88,
                                         89,
                                         90,
                                         91,
                                         92,
                                         93,
                                         94,
                                         95,
                                         96,
                                         97,
                                         98,
                                         99,
                                         100,
                                         101,
                                         102,
                                         103,
                                         104,
                                         105,
                                         106,
                                         107,
                                         108,
                                         109,
                                         110,
                                         111,
                                         112,
                                         113,
                                         114,
                                         115,
                                         116,
                                         117,
                                         118,
                                         119,
                                         120,
                                         121,
                                         122,
                                         123,
                                         124,
                                         125,
                                         126,
                                         127],
               'undefined_policy': 'mean_min_max_null_if_any_required_cosine_is_null'},
 'zero_vector_policy': {'cosine': 'null_if_either_norm_zero', 'norm_ratio': 'null_if_oracle_norm_zero'},
 'overwrite': False,
 'timestamps': False,
 'host_metadata': False,
 'gpu_metadata': False}

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def require_hash(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64 or
            any(c not in '0123456789abcdef' for c in expected) or sha256_file(path) != expected):
        raise ValueError(f'SHA256 mismatch: {path}')


def raw_hash(tensor):
    return hashlib.sha256(tensor.detach().numpy().astype('<f4', copy=False).tobytes(order='C')).hexdigest()


def validate_tensor(value, expected_hash, spec):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.dtype != torch.float32 or
            list(value.shape) != spec['tensor_shape'] or not value.is_contiguous() or
            not bool(torch.isfinite(value).all())):
        raise ValueError('Invalid activation shape, dtype, device, contiguity or finiteness')
    if raw_hash(value) != expected_hash:
        raise ValueError('Raw tensor SHA256 mismatch')
    return value


def require_equivalence(metrics, settings):
    for key, bound, minimum in (
        ('cosine_similarity', settings['minimum_cosine_similarity'], True),
        ('relative_rms_difference', settings['maximum_relative_rms_difference'], False),
        ('maximum_absolute_difference', settings['maximum_absolute_difference'], False),
    ):
        value = metrics[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or
                (value < bound if minimum else value < 0 or value > bound)):
            raise ValueError(f'Merged mean equivalence failed at position {metrics["position"]}: {key}')


def merged_mean_consistency(merged, reference, spec):
    positions = []
    for p in spec['positions']:
        left, right = merged[p].double(), reference[p].double()
        delta = left - right
        ln, rn = float(left.norm()), float(right.norm())
        rms = float(delta.square().mean().sqrt())
        reference_rms = float(right.square().mean().sqrt())
        if ln == 0 and rn == 0:
            cosine, relative_rms = 1., 0.
        else:
            cosine = 0. if ln == 0 or rn == 0 else float(torch.dot(left / ln, right / rn))
            cosine = max(-1., min(1., cosine))
            relative_rms = rms / reference_rms if reference_rms else None
        metrics = dict(position=p, cosine_similarity=cosine, rms_difference=rms, reference_rms=reference_rms,
                       relative_rms_difference=relative_rms, maximum_absolute_difference=float(delta.abs().max()))
        require_equivalence(metrics, spec['merged_mean_consistency'])
        positions.append(metrics)
    return {'method': spec['merged_mean_consistency'], 'positions': positions}


def geometry(candidate, oracle, positions):
    records = []
    for p in positions:
        x, y = candidate[p].double(), oracle[p].double()
        xn, yn = float(x.norm()), float(y.norm())
        records.append(dict(position=p, cosine_similarity=None if xn == 0 or yn == 0 else float(torch.dot(x, y) / (xn * yn)),
                            candidate_norm=xn, oracle_difference_norm=yn,
                            candidate_to_oracle_norm_ratio=None if yn == 0 else xn / yn))
    return records


def summarize(records, spec):
    by_position = {row['position']: row for row in records}
    def stats(positions):
        values = [by_position[p]['cosine_similarity'] for p in positions]
        valid = all(value is not None for value in values)
        return {'positions': positions, 'mean': math.fsum(values) / len(values) if valid else None,
                'min': min(values) if valid else None, 'max': max(values) if valid else None}
    primary = stats(spec['summaries']['primary_positions'])
    return {'primary_metric': {'definition': spec['summaries']['primary_metric'],
                               'positions': primary['positions'], 'value': primary['mean']},
            'position_0_cosine': by_position[0]['cosine_similarity'],
            'individual_positions_0_through_4': [by_position[p] for p in spec['summaries']['individual_positions']],
            'positions_1_through_4': primary,
            'positions_1_through_127': stats(spec['summaries']['descriptive_positions'])}


def validate_inputs(paths, spec):
    for name in ('attempt_spec', 'construction_manifest', 'oracle_manifest'):
        require_hash(paths[name], spec['inputs'][name]['sha256'])
    attempt = json_object(paths['attempt_spec'])
    manifest = json_object(paths['construction_manifest'])
    oracle_manifest = json_object(paths['oracle_manifest'])
    if (attempt['attempt_id'] != spec['attempt_id'] or manifest['attempt_id'] != spec['attempt_id'] or
            manifest['attempt_spec_sha256'] != spec['inputs']['attempt_spec']['sha256']):
        raise ValueError('Attempt012 identity/spec linkage mismatch')
    for key in ('probe', 'spectral_filter', 'jvp'):
        if manifest[key] != attempt[key]:
            raise ValueError(f'Construction semantics mismatch: {key}')
    if manifest['candidate_definition'] != attempt['candidate'] or attempt['candidate']['sign'] != 'positive_J_x_probe':
        raise ValueError('Positive candidate semantics mismatch')
    if any(manifest['readout'][key] != value for key, value in attempt['readout'].items()):
        raise ValueError('Construction readout mismatch')
    if oracle_manifest['probe'] != {k: attempt['probe'][k] for k in ('serialized_sha256', 'raw_tensor_sha256')}:
        raise ValueError('Frozen probe linkage mismatch')
    expected = dict(relative_layer=.5, layer_index=13, num_layers=28, sample_count=10000,
                    sequence_length=128, hidden_size=2048, batch_size=32,
                    model_dtype='float32', accumulator_dtype='float64', stored_dtype='float32')
    if any(oracle_manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('Oracle readout/method provenance mismatch')
    candidate_hash = manifest['artifact']['serialized_sha256']
    oracle_hash = oracle_manifest['oracle_adl_sha256']
    require_hash(paths['candidate_artifact'], candidate_hash)
    require_hash(paths['oracle_artifact'], oracle_hash)
    candidate = torch.load(paths['candidate_artifact'], map_location='cpu', weights_only=True)
    oracle = torch.load(paths['oracle_artifact'], map_location='cpu', weights_only=True)
    if not isinstance(candidate, dict) or set(candidate) != {'merged_mean', 'inverse_response', 'metadata'}:
        raise ValueError('Malformed candidate artifact')
    if candidate['metadata'] != dict(attempt_id=spec['attempt_id'], response_sign='positive_J_x_probe',
                                    sample_count=10000, readout=attempt['readout']):
        raise ValueError('Candidate artifact metadata mismatch')
    if not isinstance(oracle, dict) or oracle.get('format_version') != 1 or not isinstance(oracle.get('metadata'), dict):
        raise ValueError('Malformed oracle artifact')
    if any(oracle['metadata'].get(k) != v for k, v in expected.items()):
        raise ValueError('Oracle artifact metadata mismatch')
    for name in ('merged_mean', 'inverse_response'):
        validate_tensor(candidate[name], manifest['artifact']['raw_tensors_sha256'][name], spec)
    for name in ('base_mean', 'ft_mean', 'difference'):
        validate_tensor(oracle.get(name), oracle_manifest['raw_tensors_sha256'][name], spec)
    # The frozen oracle difference is used directly; never recompute or flip it.
    hashes = {name: spec['inputs'][name]['sha256'] for name in ('attempt_spec', 'construction_manifest', 'oracle_manifest')}
    hashes.update(candidate_artifact=candidate_hash, oracle_artifact=oracle_hash)
    for name, digest in hashes.items():
        require_hash(paths[name], digest)
    return candidate, oracle, hashes


def evaluate(spec_path, output_path=None, input_paths=None):
    spec = json_object(spec_path)
    if spec != FROZEN_SPEC:
        raise ValueError('Evaluation specification differs from frozen definition')
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT / value
    paths = {name: resolve(record['path']) for name, record in spec['inputs'].items()}
    if input_paths is not None:
        if set(input_paths) != set(paths):
            raise ValueError('Expected exactly five evaluation inputs')
        paths = input_paths
    output = Path(output_path) if output_path is not None else resolve(spec['output_path'])
    if output.exists() or output.is_symlink():
        raise ValueError('Evaluation output already exists')
    source = Path(__file__).resolve()
    if output.resolve() in {p.resolve() for p in [spec_path, source, *paths.values()]}:
        raise ValueError('Output overlaps an input')
    spec_hash, script_hash = sha256_file(spec_path), sha256_file(source)
    if json_object(spec_path) != spec:
        raise ValueError('Evaluation spec changed during load')
    candidate, oracle, hashes = validate_inputs(paths, spec)
    consistency = merged_mean_consistency(candidate['merged_mean'], oracle['ft_mean'], spec)
    records = geometry(candidate['inverse_response'], oracle['difference'], spec['positions'])
    result = {'format_version': 1, 'attempt_id': spec['attempt_id'], 'method': spec['method'],
              'provenance': {'evaluation_spec_sha256': spec_hash, 'evaluator_script_sha256': script_hash,
                  'attempt012_spec_sha256': hashes['attempt_spec'],
                  'attempt012_construction_manifest_sha256': hashes['construction_manifest'],
                  'attempt012_artifact_sha256': hashes['candidate_artifact'],
                  'oracle_manifest_sha256': hashes['oracle_manifest'], 'oracle_artifact_sha256': hashes['oracle_artifact']},
              'candidate_sign': spec['candidate_sign'], 'oracle_sign': spec['oracle_sign'],
              'merged_mean_consistency': consistency, 'positions': records, 'summaries': summarize(records, spec)}
    for name, digest in hashes.items():
        require_hash(paths[name], digest)
    require_hash(spec_path, spec_hash)
    require_hash(source, script_hash)
    serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-spec', type=Path,
                        default=PROJECT / 'experiments/attempts' / FROZEN_SPEC['attempt_id'] / 'evaluation_spec.json')
    args = parser.parse_args()
    evaluate(args.evaluation_spec)


if __name__ == '__main__':
    main()
