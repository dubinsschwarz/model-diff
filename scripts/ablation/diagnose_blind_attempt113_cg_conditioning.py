#!/usr/bin/env python3
"""Read-only blind conditioning diagnostic for Attempt113's CG checkpoint."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '114_blind_attempt113_cg_conditioning_diagnostic'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
OUTPUT_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'result.json'
ATTEMPT113_COMMIT = '5357ba00408f07b284de3126227fc5eb317e4f13'
ATTEMPT113_SPEC_SHA = '7c7f508575e216cd29eee967f31101b1c614f85cc0cb0465becd4e0d1dc8bfb6'
ATTEMPT113_SOURCE_SHA = '02bc38350e0f08997f3f2e6942c89594abb4b2e30e042973b930d89e59374ee9'
ATTEMPT113_SPEC = PROJECT/'experiments/attempts/113_blind_diverse8_unit_gradient_inverse_ggn_candidate/spec.json'
ATTEMPT113_SOURCE = PROJECT/'scripts/ablation/construct_blind_diverse8_unit_gradient_inverse_ggn.py'
CHECKPOINT_PATH = Path('/root/model-diff-scratch/checkpoints/attempt113_blind_inverse_cg80_iter40.pt')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_spec(path: Path = SPEC_PATH) -> dict:
    if path.resolve() != SPEC_PATH.resolve():
        raise ValueError('Attempt114 spec path changed')
    spec = json.loads(path.read_text())
    linkage = spec.get('frozen_attempt113', {})
    policy = spec.get('information_policy', {})
    if (spec.get('attempt_id') != ATTEMPT or spec.get('output_path') != str(OUTPUT_PATH.relative_to(PROJECT))
            or linkage.get('commit') != ATTEMPT113_COMMIT
            or linkage.get('spec_path') != str(ATTEMPT113_SPEC.relative_to(PROJECT))
            or linkage.get('source_path') != str(ATTEMPT113_SOURCE.relative_to(PROJECT))
            or linkage.get('spec_sha256') != ATTEMPT113_SPEC_SHA
            or linkage.get('source_sha256') != ATTEMPT113_SOURCE_SHA
            or linkage.get('checkpoint_path') != str(CHECKPOINT_PATH)
            or linkage.get('iteration') != 40 or linkage.get('coordinate_count') != 98
            or linkage.get('ggn_rows') != [0, 64] or linkage.get('ggn_denominator') != 8128
            or linkage.get('ggn_operator') != 'exact_endpoint_categorical_JtFJ'
            or linkage.get('fisher_numerics') != 'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast'
            or linkage.get('damping_fraction') != 1e-4
            or linkage.get('planned_cg_iterations') != 80
            or linkage.get('preconditioner', 'missing') is not None
            or linkage.get('residual_early_stop') is not False
            or spec.get('ritz') != {
                'dtype': 'cpu_float64', 'matrix_size': 40,
                'diagonal_first': '1/alpha_1',
                'diagonal_later': '1/alpha_i+beta_(i-1)/alpha_(i-1)',
                'offdiagonal': 'sqrt(beta_i)/alpha_i_for_i_1_through_39',
                'eigensolver': 'torch.linalg.eigvalsh',
                'implied_ggn_eigenvalues': 'A_ritz_minus_lambda_without_clipping',
                'threshold_multipliers_of_lambda': [2, 10, 100],
                'material_negative_relative_tolerance': 1e-8,
                'require_positive_minimum_A_ritz_for_condition_and_log_ratios': True,
                'interpretation': 'Krylov_subspace_approximations_not_full_spectrum'}
            or spec.get('residual_curvature') != {
                'operator_applications': 1,
                'rayleigh': 'matrixwise_dot(r40,G_r40)/matrixwise_dot(r40,r40)',
                'saved_relative_residual_rtol': 1e-6,
                'saved_relative_residual_atol': 1e-8,
                'damping_rule_rtol': 1e-12,
                'negative_curvature_relative_tolerance': 1e-6,
                'negative_curvature_absolute_tolerance': 1e-8,
                'substantial_curvature_drop_ratio': 0.1}
            or spec.get('output_policy') != 'small_JSON_only_no_overwrite_no_checkpoint_mutation'
            or policy != {
                'blind_diagnostic': True, 'final_checkpoint_access': True,
                'attempt113_checkpoint_access': True, 'historical_base_access': False,
                'true_delta_access': False, 'adapter_access': False,
                'oracle_adl_access': False, 'historical_target_access': False,
                'prior_oracle_evaluation_access': False, 'candidate_evaluation': False,
                'candidate_selection': False, 'oracle_based_tuning': False}):
        raise ValueError('Attempt114 frozen spec inventory mismatch')
    return spec


def validate_attempt113_linkage(spec: dict):
    """Validate exact committed blind source, without importing privileged code."""
    for path, expected in ((ATTEMPT113_SPEC, ATTEMPT113_SPEC_SHA),
                           (ATTEMPT113_SOURCE, ATTEMPT113_SOURCE_SHA)):
        if sha256_file(path) != expected:
            raise ValueError('Frozen Attempt113 source/spec SHA256 mismatch')
        committed = subprocess.run(
            ['git', '-c', f'safe.directory={PROJECT}', 'show',
             f'{ATTEMPT113_COMMIT}:{path.relative_to(PROJECT)}'],
            cwd=PROJECT, capture_output=True, check=True).stdout
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError('Frozen Attempt113 commit linkage mismatch')
    loader = importlib.util.spec_from_file_location('attempt113_blind_only', ATTEMPT113_SOURCE)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    prior = module.load_spec()
    if (prior['paths']['checkpoint_path'] != str(CHECKPOINT_PATH)
            or prior['final_checkpoint']['directory'] != '/root/model-diff-scratch/models/merged'
            or prior['eligible_tensors']['expected_matrix_count'] != 98
            or prior['ggn']['operator'] != 'exact_endpoint_categorical_JtFJ'
            or prior['ggn']['rows'] != [0, 64]
            or prior['ggn']['denominator'] != 8128
            or prior['ggn']['microbatch_size'] != 1
            or prior['ggn']['empirical_fisher'] is not False
            or prior['fisher_numerics']['implementation'] !=
                'FP64_softmax_and_Fisher_action_then_single_FP32_VJP_cast'
            or prior['damping']['damping_fraction'] != 1e-4
            or prior['solver']['iterations'] != 80
            or prior['solver']['checkpoint_iteration'] != 40
            or prior['solver']['preconditioner'] is not None
            or prior['solver']['residual_early_stop'] is not False
            or prior['information_policy']['historical_base_access'] is not False):
        raise ValueError('Attempt113 blind GGN/checkpoint semantics changed')
    return module, prior


def validate_checkpoint_read_only(checkpoint: Path, module, prior: dict, spec: dict):
    if checkpoint != CHECKPOINT_PATH or checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError('Only the frozen regular Attempt113 checkpoint is permitted')
    before = {'serialized_sha256': sha256_file(checkpoint), 'size_bytes': checkpoint.stat().st_size}
    payload = module.load_checkpoint(checkpoint)
    inventory = payload.get('coordinate_inventory')
    if (not isinstance(inventory, list) or len(inventory) != 98
            or payload.get('iteration') != 40):
        raise ValueError('Attempt113 checkpoint iteration/coordinate inventory mismatch')
    rhs_record = payload.get('rhs_record')
    gradient_records = payload.get('gradient_records')
    if (not isinstance(rhs_record, dict) or
            set(rhs_record) != {'norm', 'matrix_raw_sha256',
                                'ordered_matrix_hash_sha256', 'hash_aggregation'} or
            rhs_record['hash_aggregation'] != 'SHA256_of_ordered_name_NUL_digest_newline' or
            not isinstance(gradient_records, list) or
            len(gradient_records) != len(module.ORDER) or
            [row.get('name') for row in gradient_records if isinstance(row, dict)] !=
                list(module.ORDER)):
        raise ValueError('Attempt113 blind RHS/checkpoint inventory mismatch')
    matrix_hashes = rhs_record['matrix_raw_sha256']
    names = [row['name'] for row in inventory]
    if (not isinstance(matrix_hashes, dict) or set(matrix_hashes) != set(names) or
            any(not isinstance(value, str) or len(value) != 64 or
                any(character not in '0123456789abcdef' for character in value)
                for value in matrix_hashes.values())):
        raise ValueError('Attempt113 blind RHS matrix hashes malformed')
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode('utf-8') + b'\0' +
                      matrix_hashes[name].encode('ascii') + b'\n')
    if digest.hexdigest() != rhs_record['ordered_matrix_hash_sha256']:
        raise ValueError('Attempt113 blind RHS aggregate hash mismatch')
    for row in gradient_records:
        if (set(row) != {'name', 'mean_generic_loss', 'global_gradient_norm',
                         'unit_normalization_scalar', 'rows', 'prediction_contexts'} or
                row['rows'] != [0, 512] or row['prediction_contexts'] != 65024 or
                not all(isinstance(row[key], (int, float)) and math.isfinite(row[key])
                        for key in ('mean_generic_loss', 'global_gradient_norm',
                                    'unit_normalization_scalar')) or
                row['global_gradient_norm'] <= 0 or
                not math.isclose(row['unit_normalization_scalar'],
                                 1.0/row['global_gradient_norm'], rel_tol=1e-12)):
            raise ValueError('Attempt113 blind corpus gradient record mismatch')
    rho_b = payload.get('rho_b')
    fixed_lambda = payload.get('fixed_lambda')
    if (not isinstance(rho_b, (float, int)) or not math.isfinite(rho_b) or rho_b <= 0
            or not isinstance(fixed_lambda, (float, int)) or
            not math.isfinite(fixed_lambda) or fixed_lambda <= 0
            or not math.isclose(fixed_lambda, prior['damping']['damping_fraction'] * rho_b,
                                rel_tol=spec['residual_curvature']['damping_rule_rtol'], abs_tol=0)):
        raise ValueError('Attempt113 checkpoint blind damping rule mismatch')
    module.validate_checkpoint(payload, prior, inventory, payload['rhs_record'],
                               payload['gradient_records'],
                               {'rho_b': rho_b, 'lambda': fixed_lambda})
    row = payload['iteration_records'][-1]
    relative = math.sqrt(payload['rr']) / payload['b_norm']
    if (not math.isclose(relative, row['relative_residual'],
                        rel_tol=spec['residual_curvature']['saved_relative_residual_rtol'],
                        abs_tol=spec['residual_curvature']['saved_relative_residual_atol'])
            or not math.isclose(math.sqrt(payload['rr']), row['r_norm'],
                                rel_tol=1e-6, abs_tol=1e-8)):
        raise ValueError('Attempt113 saved iteration-40 residual mismatch')
    if before != {'serialized_sha256': sha256_file(checkpoint),
                  'size_bytes': checkpoint.stat().st_size}:
        raise ValueError('Attempt113 checkpoint changed during read')
    return payload, before


def lanczos_ritz(records: list[dict], damping: float, spec: dict) -> dict:
    if len(records) != 40 or not math.isfinite(damping) or damping <= 0:
        raise ValueError('Expected 40 CG records and positive damping')
    alphas, betas = [], []
    for index, row in enumerate(records, 1):
        alpha, beta = row.get('alpha'), row.get('beta')
        if (row.get('iteration') != index or not isinstance(alpha, (int, float))
                or not math.isfinite(alpha) or alpha <= 0
                or not isinstance(beta, (int, float)) or not math.isfinite(beta)
                or beta < 0):
            raise ValueError('Invalid saved CG alpha/beta recurrence')
        alphas.append(float(alpha))
        betas.append(float(beta))
    diagonal = [1.0/alphas[0]] + [1.0/alphas[i] + betas[i-1]/alphas[i-1]
                                    for i in range(1, 40)]
    offdiagonal = [math.sqrt(betas[i])/alphas[i] for i in range(39)]
    if not all(math.isfinite(v) for v in diagonal + offdiagonal):
        raise ValueError('Nonfinite Lanczos tridiagonal')
    diagonal_tensor = torch.tensor(diagonal, dtype=torch.float64, device='cpu')
    off_tensor = torch.tensor(offdiagonal, dtype=torch.float64, device='cpu')
    tridiagonal = (torch.diag(diagonal_tensor) + torch.diag(off_tensor, 1)
                   + torch.diag(off_tensor, -1))
    if (not bool(torch.isfinite(tridiagonal).all())
            or not torch.equal(tridiagonal, tridiagonal.T)):
        raise ValueError('Nonsymmetric/nonfinite Lanczos tridiagonal')
    try:
        eigenvalues = torch.linalg.eigvalsh(tridiagonal)
    except RuntimeError as exc:
        raise ValueError('Lanczos eigvalsh failed') from exc
    if not bool(torch.isfinite(eigenvalues).all()):
        raise ValueError('Nonfinite A Ritz values')
    values = eigenvalues.tolist()
    minimum, maximum = values[0], values[-1]
    negative_tolerance = (spec['ritz']['material_negative_relative_tolerance'] *
                          max(1.0, abs(maximum), damping))
    if minimum < -negative_tolerance:
        raise ValueError('Materially negative A Ritz value')
    if minimum <= 0:
        raise ValueError('Nonpositive A Ritz value prevents condition/log diagnostics')
    implied = [value-damping for value in values]  # Deliberately no clipping.
    thresholds = {}
    for multiplier in spec['ritz']['threshold_multipliers_of_lambda']:
        count = sum(value <= multiplier*damping for value in values)
        thresholds[str(multiplier)] = {'count': count, 'fraction': count/40}
    return {
        'interpretation': 'Krylov-subspace approximations, not the full G spectrum',
        'lanczos_dtype': 'cpu_float64', 'diagonal': diagonal,
        'offdiagonal': offdiagonal, 'A_ritz_eigenvalues_ascending': values,
        'min_A_ritz': minimum, 'max_A_ritz': maximum,
        'ritz_condition_estimate': maximum/minimum, 'lambda': damping,
        'implied_G_ritz_eigenvalues_unclipped': implied,
        'min_implied_G_ritz': implied[0], 'max_implied_G_ritz': implied[-1],
        'A_ritz_at_or_below_damping_multiples': thresholds,
        'log10_max_A_ritz_over_lambda': math.log10(maximum)-math.log10(damping),
        'log10_min_A_ritz_over_lambda': math.log10(minimum)-math.log10(damping),
        'ritz_reaches_damping_scale': minimum <= 2*damping}


def residual_curvature(r, apply_operator, module, rho_b: float, damping: float,
                       relative_residual: float, spec: dict):
    """Apply G exactly once; `apply_operator` returns a parameter-shaped Gr."""
    denominator = module.matrixwise_dot(r, r)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError('Zero/nonfinite iteration-40 residual norm')
    gr, operator_audit = apply_operator(r)
    numerator = module.matrixwise_dot(r, gr)
    rho = numerator/denominator
    tolerance = (spec['residual_curvature']['negative_curvature_absolute_tolerance'] +
                 spec['residual_curvature']['negative_curvature_relative_tolerance'] *
                 max(rho_b, damping))
    if not math.isfinite(rho) or rho < -tolerance:
        raise ValueError('Nonfinite/materially negative residual GGN curvature')
    return {
        'r40_norm_sq': denominator, 'r40_G_r40': numerator,
        'rho_b': rho_b, 'rho_r40': rho, 'rho_r40_over_rho_b': rho/rho_b,
        'rho_r40_over_lambda': rho/damping,
        'curvature_drop_factor': rho_b/rho if rho > 0 else None,
        'residual_curvature_relative_to_damping': rho/damping,
        'relative_residual_at_40': relative_residual,
        'residual_shifted_toward_lower_curvature':
            rho >= 0 and rho <= spec['residual_curvature']['substantial_curvature_drop_ratio']*rho_b,
        'operator_applications': 1, 'ggn_audit': operator_audit}


def atomic_json_no_overwrite(path: Path, result: dict):
    if path != OUTPUT_PATH or path.exists() or path.is_symlink():
        raise ValueError('Attempt114 result exists or output path changed')
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open('xb') as stream:
            stream.write((json.dumps(result, sort_keys=True, indent=2,
                                     allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def run():
    spec = load_spec()
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise ValueError('Attempt114 result already exists')
    module, prior = validate_attempt113_linkage(spec)
    payload, checkpoint_record = validate_checkpoint_read_only(
        CHECKPOINT_PATH, module, prior, spec)
    inventory = payload['coordinate_inventory']
    ritz = lanczos_ritz(payload['iteration_records'], payload['fixed_lambda'], spec)
    rho_b, damping = payload['rho_b'], payload['fixed_lambda']
    relative = math.sqrt(payload['rr'])/payload['b_norm']
    residual_cpu = payload['r']
    del payload  # Release x and p before model/GGN loading; checkpoint remains unchanged.
    gc.collect()

    final_dir = Path(prior['final_checkpoint']['directory'])
    if (final_dir != Path('/root/model-diff-scratch/models/merged')
            or module.blind.checkpoint_file_records(final_dir) != prior['final_checkpoint']['files']):
        raise ValueError('Frozen final checkpoint inventory mismatch')
    ggn = prior['ggn']
    token_path = Path(ggn['tokens_path'])
    if module.sha256_file(token_path) != ggn['serialized_sha256']:
        raise ValueError('Frozen FineWeb G corpus serialized hash mismatch')
    tokens = module.blind.load_corpus_tokens(
        token_path, ggn['raw_tensor_sha256'], 4096, 128, torch)[:64].contiguous()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start = time.perf_counter()
    model = module.blind.load_local_model(final_dir, device, torch)
    eligible = module.blind.discover_eligible_linear_weights(model, torch)
    selected_ids = module.blind.freeze_other_parameters(model, eligible)
    if (module.coordinate_inventory(eligible) != inventory or len(selected_ids) != 98):
        raise ValueError('Checkpoint/final model 98-coordinate inventory mismatch')
    model_before = module.blind.model_state_hashes(model, torch)
    direction = module.vector_to_device(residual_cpu, eligible)
    del residual_cpu
    gc.collect()
    operator_calls = 0

    def apply_once(vector):
        nonlocal operator_calls
        operator_calls += 1
        if operator_calls != 1:
            raise ValueError('Attempt114 permits exactly one G application')
        audit = module.ggn_action_stage(
            model, tokens, eligible, vector, prior,
            progress_label='Attempt114 residual GGN')
        gr = {name+'.weight': weight.weight.grad
              for name, weight in eligible}
        return gr, audit

    curvature = residual_curvature(direction, apply_once, module, rho_b,
                                   damping, relative, spec)
    if operator_calls != 1:
        raise ValueError('Attempt114 GGN application count mismatch')
    module.blind.verify_model_unchanged(model, model_before, torch)
    model.zero_grad(set_to_none=True)
    if checkpoint_record != {'serialized_sha256': sha256_file(CHECKPOINT_PATH),
                             'size_bytes': CHECKPOINT_PATH.stat().st_size}:
        raise ValueError('Attempt113 checkpoint mutated during diagnosis')
    result = {
        'attempt_id': ATTEMPT, 'purpose': spec['purpose'],
        'information_policy': spec['information_policy'],
        'checkpoint': {**checkpoint_record, 'path': str(CHECKPOINT_PATH),
                       'iteration': 40, 'read_only': True, 'unchanged_after_run': True},
        'frozen_attempt113': spec['frozen_attempt113'],
        'source_provenance': {'spec_sha256': sha256_file(SPEC_PATH),
                              'source_sha256': sha256_file(Path(__file__)),
                              'attempt113_spec_sha256': ATTEMPT113_SPEC_SHA,
                              'attempt113_source_sha256': ATTEMPT113_SOURCE_SHA},
        'final_checkpoint_files': prior['final_checkpoint']['files'],
        'ggn_corpus': {key: ggn[key] for key in
                       ('tokens_path', 'serialized_sha256', 'raw_tensor_sha256',
                        'rows', 'denominator', 'operator')},
        'coordinate_inventory': inventory,
        'ritz': ritz, 'residual_curvature': curvature,
        'interpretation': ('The residual has shifted toward lower-curvature directions'
                           if curvature['residual_shifted_toward_lower_curvature'] else
                           'No substantial residual curvature drop under the frozen criterion'),
        'scientific_limit': ('Ritz values describe only the 40-step Krylov subspace of this '
                             'finite-sample endpoint operator; no historical quality or '
                             'identifiability conclusion follows.'),
        'ggn_applications': operator_calls,
        'elapsed_seconds_including_model_load': time.perf_counter()-start}
    atomic_json_no_overwrite(OUTPUT_PATH, result)
    print(json.dumps({'result': str(OUTPUT_PATH), 'rho_r40': curvature['rho_r40'],
                      'min_A_ritz': ritz['min_A_ritz'],
                      'max_A_ritz': ritz['max_A_ritz']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == '__main__':
    main()
