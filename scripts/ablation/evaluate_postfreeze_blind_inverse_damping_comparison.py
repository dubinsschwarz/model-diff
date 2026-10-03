#!/usr/bin/env python3
"""Read-only post-freeze comparison of the Attempt113/116 blind responses."""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import pickle
import subprocess
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '117_postfreeze_blind_inverse_damping_comparison'
ORACLE_VALIDATOR = Path(__file__).with_name('evaluate_attempt014.py')
ORACLE_VALIDATOR_SHA = '1193860390504d614522ae746f903d7304eb97d9b2b8d362fabcd302a4a77afd'
if hashlib.sha256(ORACLE_VALIDATOR.read_bytes()).hexdigest() != ORACLE_VALIDATOR_SHA:
    raise ValueError('Frozen oracle validator source SHA256 mismatch')
_loader = importlib.util.spec_from_file_location('attempt117_oracle_validator014', ORACLE_VALIDATOR)
oracle_validator = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(oracle_validator)

ATTEMPT113 = '113_blind_diverse8_unit_gradient_inverse_ggn_candidate'
ATTEMPT116 = '116_blind_diverse8_unit_gradient_inverse_ggn_stronger_damping_candidate'
CONSTRUCTIONS = {
    '113': {
        'attempt_id': ATTEMPT113,
        'commit': 'd0f46f129f356deca26fd63163ca7654c31a30db',
        'spec': {'path': f'experiments/attempts/{ATTEMPT113}/spec.json', 'sha256': '7c7f508575e216cd29eee967f31101b1c614f85cc0cb0465becd4e0d1dc8bfb6'},
        'source': {'path': 'scripts/ablation/construct_blind_diverse8_unit_gradient_inverse_ggn.py', 'sha256': '02bc38350e0f08997f3f2e6942c89594abb4b2e30e042973b930d89e59374ee9'},
        'manifest': {'path': f'experiments/attempts/{ATTEMPT113}/construction-manifest.json', 'sha256': '5e40160521534d15de6fa6405fac26419fa362b8579398b496e1fc0c9e98c5c4'},
        'artifact': {'path': '/root/model-diff-scratch/artifacts/attempt113_blind_diverse8_unit_gradient_inverse_ggn/candidate.pt', 'sha256': 'c95159c97f72e65b1bfbb698d67d4cceb3937d77ad52580fd86f9a10a0a530bc'},
        'rho_b': 561.5508153958936, 'damping_fraction': 0.0001,
        'lambda': 0.056155081539589355, 'operator_applications': 80,
        'final_relative_residual': 0.505928893506768,
        'x80_aggregate_sha256': '630eda2b70dfc302d2f20b066f140e257c5cacfda182fda2e9b248c0ad4cb29c',
        'response_raw_rhs_sha256': '8ddc1801ee82504e10898e97d12b53fea1442ed029314dc47d7d0f68523c2d0a',
        'response_inverse_sha256': 'e2c7267d573889eb313d207ab02d3236044c2b2b77794ce6381ed5c44ea86f21',
    },
    '116': {
        'attempt_id': ATTEMPT116,
        'commit': 'f2f553b481ed8f3767eb33c649c48a711e25c0a7',
        'spec': {'path': f'experiments/attempts/{ATTEMPT116}/spec.json', 'sha256': 'd30b6a80b403db71276921b8fed01b7606114cb82109f29cb50a7fdf2322a743'},
        'source': {'path': 'scripts/ablation/construct_blind_diverse8_unit_gradient_inverse_ggn_stronger_damping.py', 'sha256': 'fef3a745228dbf73713e93f4b027eb9cf475be6516e17dbea52df61750044334'},
        'manifest': {'path': f'experiments/attempts/{ATTEMPT116}/construction-manifest.json', 'sha256': '2f04b04b8e69fdae1fd9e0a81b2bad751ce662fcc2a5687076be4f15440cabf2'},
        'artifact': {'path': '/root/model-diff-scratch/artifacts/attempt116_blind_stronger_damping_inverse_ggn/candidate.pt', 'sha256': 'c2ed1053b2f57416a748f7104f7260481ce10891750492a4a4aa32885dc216ee'},
        'rho_b': 561.5508153958936, 'damping_fraction': 0.001,
        'lambda': 0.5615508153958936, 'operator_applications': 80,
        'final_relative_residual': 0.030534033703066656,
        'x80_aggregate_sha256': '789abbb7da98e3de980e214fc1fb9f93a862148331f90b07c1f06e7a47434bde',
        'response_raw_rhs_sha256': '8ddc1801ee82504e10898e97d12b53fea1442ed029314dc47d7d0f68523c2d0a',
        'response_inverse_sha256': 'f3ed438910155c3bc52b57878d1193ce719d3693dac011cefe41228454110d1d',
    },
}

FROZEN_SPEC = {
    'format_version': 1, 'attempt_id': ATTEMPT,
    'purpose': 'postfreeze_signed_comparison_of_three_frozen_blind_functional_responses',
    'constructions': copy.deepcopy(CONSTRUCTIONS),
    'cross_candidate_identity': {
        'rhs_aggregate_sha256': 'a90785341961a0a2a08e7949961cb105867cf2509013b01b65c2013322ead574',
        'rho_b': 561.5508153958936,
        'raw_rhs_response_sha256': '8ddc1801ee82504e10898e97d12b53fea1442ed029314dc47d7d0f68523c2d0a',
        'ggn_operator': 'exact_endpoint_categorical_JtFJ',
        'ggn_numerics': 'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast',
        'iterations': 80,
        'only_scientific_difference': 'damping_fraction_1e-4_versus_1e-3',
    },
    'artifact_inventory': ['x_80', 'response_raw_rhs', 'response_inverse'],
    'response': {'shape': [128, 2048], 'dtype': 'contiguous_CPU_float32'},
    'oracle': {
        'path': '/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt',
        'serialized_sha256': '339a6c09b10286369a5d10eb49e144d1819f4301e334c63930ca7c4dd3a11077',
        'vector': 'difference', 'definition': 'ft_mean - base_mean',
        'difference_raw_sha256': 'd60789a2c06eb82dba2f2449d8665ec665d85e9f7e33fdea1f66ac79c81c159d',
        'validator_source': {'path': 'scripts/ablation/evaluate_attempt014.py', 'sha256': ORACLE_VALIDATOR_SHA},
        'provenance_inputs': copy.deepcopy(oracle_validator.FROZEN_SPEC['inputs']),
    },
    'candidates': [
        {'name': 'raw_rhs', 'role': 'blind_uninverted_baseline', 'source': '113.response_raw_rhs'},
        {'name': 'inverse_1e-4', 'role': 'frozen_weak_damping_inverse', 'source': '113.response_inverse'},
        {'name': 'inverse_1e-3', 'role': 'frozen_stronger_damping_inverse', 'source': '116.response_inverse'},
    ],
    'metrics': {'dtype': 'cpu_float64', 'positions': list(range(128)),
                'primary_positions': [1, 2, 3, 4],
                'secondary_positions': list(range(1, 128)),
                'aggregation': 'arithmetic_mean_of_individual_signed_position_cosines',
                'zero_norm_cosine': None, 'required_mean_if_any_null': None,
                'flattened_cosine': False, 'absolute_cosine': False,
                'pairs': [['raw_rhs', 'inverse_1e-4'], ['raw_rhs', 'inverse_1e-3'],
                          ['inverse_1e-4', 'inverse_1e-3']]},
    'output_path': f'experiments/attempts/{ATTEMPT}/result.json',
    'output_policy': {'overwrite': False, 'large_artifact': False,
                      'model_loading': False, 'gradient_jvp_ggn_computation': False,
                      'post_freeze_evaluation': True,
                      'attempt113_frozen_before_oracle': True,
                      'attempt116_frozen_before_oracle': True,
                      'candidate_modified_after_oracle': False,
                      'sign_selection': False, 'candidate_selection': False,
                      'damping_selection_after_oracle': False,
                      'candidate_rescaling': False, 'position_selection': False,
                      'oracle_adl_access': True,
                      'same_specimen_exploratory_method_development': True,
                      'clean_heldout_validation': False,
                      'hyperparameters_informed_by_prior_privileged_development': True,
                      'third_damping_value_after_oracle': False},
}


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def require_output_absent(path):
    if path.exists() or path.is_symlink():
        raise ValueError('Attempt117 result already exists')


def require_hash(path, expected):
    oracle_validator.require_file_hash(path, expected)


def validate_frozen_commit(commit, record):
    """Verify the claimed commit contains the same frozen bytes as the working tree."""
    for field in ('spec', 'source', 'manifest'):
        item = record[field]
        proc = subprocess.run(['git', '-c', f'safe.directory={PROJECT}', 'show',
                               f'{commit}:{item["path"]}'], cwd=PROJECT,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              check=False)
        if proc.returncode or hashlib.sha256(proc.stdout).hexdigest() != item['sha256']:
            raise ValueError(f'Frozen construction commit linkage mismatch: {field}')


def validate_construction_record(record, manifest, construction_spec, *, check_commit=True):
    if check_commit:
        validate_frozen_commit(record['commit'], record)
    for field in ('spec', 'source', 'manifest'):
        require_hash(path_of(record[field]['path']), record[field]['sha256'])
    if (construction_spec.get('attempt_id') != record['attempt_id'] or
            manifest.get('attempt_id') != record['attempt_id'] or
            manifest.get('spec_sha256') != record['spec']['sha256'] or
            manifest.get('constructor_source_sha256') != record['source']['sha256'] or
            manifest.get('format_version') != 1 or
            manifest.get('purpose') != construction_spec.get('purpose') or
            manifest.get('final_checkpoint_files') != construction_spec.get('final_checkpoint', {}).get('files') or
            manifest.get('corpus_order') != construction_spec.get('corpus_order') or
            len(manifest.get('corpora', [])) != len(construction_spec.get('corpora', [])) or
            any({k: v for k, v in row.items() if k not in ('serialized_sha256', 'raw_tensor_sha256')}
                != construction_spec['corpora'][i] for i, row in enumerate(manifest.get('corpora', []))) or
            manifest.get('ggn_operator') != FROZEN_SPEC['cross_candidate_identity']['ggn_operator'] or
            manifest.get('ggn_numerics') != construction_spec.get('fisher_numerics') or
            manifest.get('ggn_numerics', {}).get('implementation') !=
                FROZEN_SPEC['cross_candidate_identity']['ggn_numerics']):
        raise ValueError('Frozen construction provenance mismatch')
    policy = manifest.get('information_policy')
    if not isinstance(policy, dict) or policy != construction_spec.get('information_policy'):
        raise ValueError('Frozen blind construction policy mismatch')
    for key in ('historical_base_access', 'true_delta_access', 'adapter_access',
                'oracle_adl_access', 'historical_target_access',
                'prior_oracle_evaluation_access', 'candidate_selection', 'sign_selection',
                'corpus_selection', 'corpus_reweighting', 'iterate_selection',
                'oracle_based_tuning_during_construction'):
        if policy.get(key) is not False or manifest.get(key) is not False:
            raise ValueError('Frozen blind construction firewall mismatch: '+key)
    for key, expected in (('same_specimen_exploratory_method_development', True),
                          ('clean_heldout_validation', False),
                          ('hyperparameters_informed_by_prior_privileged_development', True)):
        if policy.get(key) is not expected or manifest.get(key) is not expected:
            raise ValueError('Frozen development disclosure mismatch: '+key)
    cross = FROZEN_SPEC['cross_candidate_identity']
    candidate = manifest.get('candidate', {})
    artifact = manifest.get('artifact', {})
    rhs = manifest.get('b_blind', {})
    rayleigh = manifest.get('rayleigh', {})
    solver = manifest.get('solver', {})
    responses = manifest.get('functional_responses', {})
    if (rhs.get('ordered_matrix_hash_sha256') != cross['rhs_aggregate_sha256'] or
            rayleigh.get('rho_b') != record['rho_b'] or
            rayleigh.get('damping_fraction') != record['damping_fraction'] or
            rayleigh.get('lambda') != record['lambda'] or
            construction_spec.get('damping', {}).get('damping_fraction') != record['damping_fraction'] or
            record['lambda'] != record['damping_fraction'] * record['rho_b'] or
            len(solver.get('iterations', [])) != 80 or
            solver.get('operator_applications') != record['operator_applications'] or
            solver.get('lambda') != record['lambda'] or
            solver.get('candidate') != 'x_80' or
            solver.get('no_early_stop') is not True or
            solver.get('preconditioner') is not None or
            solver.get('relative_linear_residual') != record['final_relative_residual'] or
            [row.get('iteration') for row in solver.get('iterations', [])] != list(range(1, 81)) or
            candidate.get('name') != 'x_80' or
            candidate.get('final_relative_linear_residual') != record['final_relative_residual'] or
            candidate.get('x_80', {}).get('ordered_matrix_hash_sha256') != record['x80_aggregate_sha256'] or
            responses.get('response_raw_rhs', {}).get('raw_sha256') != record['response_raw_rhs_sha256'] or
            responses.get('response_inverse', {}).get('raw_sha256') != record['response_inverse_sha256'] or
            artifact.get('path') != record['artifact']['path'] or
            artifact.get('serialized_sha256') != record['artifact']['sha256'] or
            artifact.get('tensor_inventory') != FROZEN_SPEC['artifact_inventory'] or
            artifact.get('x_80_matrix_count') != 98 or
            artifact.get('x_80_dtype') != 'contiguous_CPU_float32' or
            artifact.get('response_shape') != [128, 2048] or
            artifact.get('response_dtype') != 'contiguous_CPU_float32' or
            len(manifest.get('coordinate_inventory', [])) != 98):
        raise ValueError('Frozen candidate inventory or numerical record mismatch')
    for name in ('response_raw_rhs', 'response_inverse'):
        if responses[name].get('shape') != [128, 2048] or responses[name].get('dtype') != 'torch.float32':
            raise ValueError('Frozen response shape/dtype mismatch')
    return manifest


def validate_cross_candidate_identity(spec113, spec116, manifest113, manifest116):
    cross = FROZEN_SPEC['cross_candidate_identity']
    for key in ('model', 'final_checkpoint', 'eligible_tensors', 'blind_source_provenance',
                'corpus_order', 'corpora', 'rhs', 'ggn', 'fisher_numerics', 'solver',
                'probe', 'readout', 'jvp'):
        if spec113.get(key) != spec116.get(key):
            raise ValueError('Cross-candidate scientific specification mismatch: '+key)
    if ({k: v for k, v in spec113['damping'].items() if k != 'damping_fraction'} !=
            {k: v for k, v in spec116['damping'].items() if k != 'damping_fraction'}):
        raise ValueError('Cross-candidate damping rule mismatch')
    for key in ('final_checkpoint_files', 'blind_source_provenance', 'corpus_order',
                'corpora', 'generic_loss', 'coordinate_inventory', 'per_corpus_gradients',
                'b_blind', 'ggn_operator', 'ggn_numerics', 'probe'):
        if manifest113.get(key) != manifest116.get(key):
            raise ValueError('Cross-candidate frozen input mismatch: '+key)
    if (manifest113['b_blind']['ordered_matrix_hash_sha256'] != cross['rhs_aggregate_sha256'] or
            any(manifest113['rayleigh'][key] != manifest116['rayleigh'][key]
                for key in ('bGb', 'bb', 'rho_b')) or
            manifest113['rayleigh']['rho_b'] != manifest116['rayleigh']['rho_b'] or
            manifest113['rayleigh']['rho_b'] != cross['rho_b'] or
            manifest113['functional_responses']['response_raw_rhs']['raw_sha256'] !=
                manifest116['functional_responses']['response_raw_rhs']['raw_sha256'] or
            manifest113['solver']['operator_applications'] != manifest116['solver']['operator_applications'] or
            manifest113['solver']['operator_applications'] != cross['iterations']):
        raise ValueError('Cross-candidate frozen RHS/GGN/CG identity mismatch')
    for key, expected in (('113', 0.0001), ('116', 0.001)):
        spec = spec113 if key == '113' else spec116
        manifest = manifest113 if key == '113' else manifest116
        if (spec.get('damping', {}).get('damping_fraction') != expected or
                manifest['rayleigh']['damping_fraction'] != expected):
            raise ValueError('Unexpected frozen damping fraction')


def raw_float32_sha256(tensor):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or tensor.dtype != torch.float32 or not tensor.is_contiguous():
        raise ValueError('Raw tensor must be contiguous CPU float32')
    return hashlib.sha256(tensor.numpy().astype('<f4', copy=False).tobytes()).hexdigest()


def validate_artifact_payload(value, manifest, record):
    if not isinstance(value, dict) or list(value) != FROZEN_SPEC['artifact_inventory']:
        raise ValueError('Candidate artifact inventory mismatch')
    x = value['x_80']
    inventory = manifest['coordinate_inventory']
    expected_hashes = manifest['candidate']['x_80']['matrix_raw_sha256']
    if (not isinstance(x, dict) or list(x) != [row['name'] for row in inventory] or
            set(expected_hashes) != set(x)):
        raise ValueError('Candidate parameter inventory mismatch')
    digest = hashlib.sha256()
    for row in inventory:
        name, matrix = row['name'], x[row['name']]
        if (not isinstance(matrix, torch.Tensor) or matrix.device.type != 'cpu' or
                matrix.dtype != torch.float32 or not matrix.is_contiguous() or
                list(matrix.shape) != row['shape'] or not bool(torch.isfinite(matrix).all())):
            raise ValueError('Candidate parameter tensor mismatch: '+name)
        raw = raw_float32_sha256(matrix)
        if raw != expected_hashes[name]:
            raise ValueError('Candidate parameter raw SHA256 mismatch: '+name)
        digest.update(name.encode()+b'\0'+raw.encode('ascii')+b'\n')
    if digest.hexdigest() != record['x80_aggregate_sha256']:
        raise ValueError('Candidate aggregate SHA256 mismatch')
    retained = {}
    for name, expected in (('response_raw_rhs', record['response_raw_rhs_sha256']),
                           ('response_inverse', record['response_inverse_sha256'])):
        tensor = value[name]
        oracle_validator.validate_activation_tensor(tensor, name, expected, torch)
        if manifest['functional_responses'][name]['raw_sha256'] != expected:
            raise ValueError('Candidate response manifest SHA256 mismatch')
        retained[name] = tensor.clone().contiguous()
    return retained


def load_frozen_responses(record, manifest):
    """Hash and validate one entire artifact, retaining only two small responses."""
    path = path_of(record['artifact']['path'])
    require_hash(path, record['artifact']['sha256'])
    try:
        value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError, TypeError) as exc:
        raise ValueError(f'Could not load frozen candidate artifact: {exc}') from exc
    try:
        return validate_artifact_payload(value, manifest, record)
    finally:
        del value
        gc.collect()


def required_mean(values):
    return None if any(value is None for value in values) else math.fsum(values) / len(values)


def cosine_report(left, right):
    for tensor in (left, right):
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or
                tensor.dtype != torch.float32 or not tensor.is_contiguous() or
                tuple(tensor.shape) != (128, 2048) or not bool(torch.isfinite(tensor).all())):
            raise ValueError('Malformed response for signed cosine')
    values = oracle_validator.position_cosines(left, right)
    return {'position_0_cosine': values[0],
            'positions_1_4_individual_cosines': values[1:5],
            'positions_1_4_mean_cosine': required_mean(values[1:5]),
            'positions_1_127_mean_cosine': required_mean(values[1:128]),
            'all_128_position_cosines': values}


def summarize_responses(responses, difference):
    names = [row['name'] for row in FROZEN_SPEC['candidates']]
    if list(responses) != names:
        raise ValueError('Exact three frozen candidates required in fixed order')
    candidates = {}
    for item in FROZEN_SPEC['candidates']:
        name = item['name']
        candidates[name] = {'role': item['role'], **cosine_report(responses[name], difference)}
    pairs = {}
    for left, right in FROZEN_SPEC['metrics']['pairs']:
        pairs[f'{left}_vs_{right}'] = cosine_report(responses[left], responses[right])
    def delta(new, old, field):
        a, b = candidates[new][field], candidates[old][field]
        return None if a is None or b is None else a - b
    comparisons = {}
    for label, new, old in (('1e-4_vs_raw', 'inverse_1e-4', 'raw_rhs'),
                            ('1e-3_vs_raw', 'inverse_1e-3', 'raw_rhs'),
                            ('1e-3_vs_1e-4', 'inverse_1e-3', 'inverse_1e-4')):
        comparisons[f'gain_primary_{label}'] = delta(new, old, 'positions_1_4_mean_cosine')
        comparisons[f'gain_secondary_{label}'] = delta(new, old, 'positions_1_127_mean_cosine')
    return {'candidates': candidates, 'pairwise_candidate_responses': pairs,
            'precommitted_comparisons': comparisons}


def validate_oracle(provenance_inputs):
    oracle_spec = FROZEN_SPEC['oracle']
    paths = {key: path_of(item['path']) for key, item in provenance_inputs.items()}
    hashes = {key: item['sha256'] for key, item in provenance_inputs.items()}
    for key in paths:
        require_hash(paths[key], hashes[key])
    attempt014 = oracle_validator.load_json_object(paths['attempt_spec'], 'Attempt014 spec')
    oracle_validator.validate_construction(paths, hashes, oracle_validator.FROZEN_SPEC)
    oracle_manifest = oracle_validator.validate_oracle_provenance(paths, hashes, attempt014)
    if (oracle_manifest['oracle_adl_sha256'] != oracle_spec['serialized_sha256'] or
            oracle_manifest['raw_tensors_sha256']['difference'] != oracle_spec['difference_raw_sha256']):
        raise ValueError('Frozen oracle artifact/raw provenance mismatch')
    oracle_path = path_of(oracle_spec['path'])
    require_hash(oracle_path, oracle_spec['serialized_sha256'])
    artifact = oracle_validator.load_and_validate_oracle(
        oracle_path, {'oracle': {'oracle_manifest': oracle_manifest}}, torch)
    return artifact['difference'].clone().contiguous(), oracle_manifest


def evaluate(spec_path=None):
    spec_path = path_of(spec_path or f'experiments/attempts/{ATTEMPT}/spec.json')
    spec = oracle_validator.load_json_object(spec_path, 'Attempt117 spec')
    if spec != FROZEN_SPEC:
        raise ValueError('Attempt117 spec differs from frozen evaluator plan')
    output = path_of(spec['output_path'])
    require_output_absent(output)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    spec_sha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    manifests, construction_specs = {}, {}
    for key in ('113', '116'):
        record = spec['constructions'][key]
        manifest = oracle_validator.load_json_object(path_of(record['manifest']['path']), key+' manifest')
        construction_spec = oracle_validator.load_json_object(path_of(record['spec']['path']), key+' spec')
        validate_construction_record(record, manifest, construction_spec)
        manifests[key], construction_specs[key] = manifest, construction_spec
    validate_cross_candidate_identity(construction_specs['113'], construction_specs['116'],
                                      manifests['113'], manifests['116'])
    responses113 = load_frozen_responses(spec['constructions']['113'], manifests['113'])
    try:
        responses116 = load_frozen_responses(spec['constructions']['116'], manifests['116'])
        if (not torch.equal(responses113['response_raw_rhs'], responses116['response_raw_rhs']) or
                raw_float32_sha256(responses113['response_raw_rhs']) !=
                spec['cross_candidate_identity']['raw_rhs_response_sha256']):
            raise ValueError('Frozen raw RHS responses are not byte-identical')
        responses = {'raw_rhs': responses113['response_raw_rhs'],
                     'inverse_1e-4': responses113['response_inverse'],
                     'inverse_1e-3': responses116['response_inverse']}
        difference, oracle_manifest = validate_oracle(spec['oracle']['provenance_inputs'])
        report = summarize_responses(responses, difference)
    finally:
        del responses113
        if 'responses116' in locals():
            del responses116
        gc.collect()
    result = {'format_version': 1, 'attempt_id': ATTEMPT,
              'provenance': {'evaluation_spec_sha256': spec_sha,
                             'evaluator_source_sha256': source_sha,
                             'frozen_constructions': spec['constructions'],
                             'oracle_serialized_sha256': spec['oracle']['serialized_sha256'],
                             'oracle_difference_raw_sha256': spec['oracle']['difference_raw_sha256'],
                             'oracle_manifest_sha256': spec['oracle']['provenance_inputs']['oracle_manifest']['sha256'],
                             'oracle_validator_source_sha256': ORACLE_VALIDATOR_SHA},
              'metric_definition': spec['metrics'],
              'cross_candidate_identity': spec['cross_candidate_identity'],
              'information_policy': spec['output_policy'],
              **spec['output_policy'], **report}
    require_hash(spec_path, spec_sha)
    require_hash(Path(__file__), source_sha)
    require_output_absent(output)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, default=PROJECT/'experiments/attempts'/ATTEMPT/'spec.json')
    args = parser.parse_args()
    evaluate(args.spec)


if __name__ == '__main__':
    main()
