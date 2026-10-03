#!/usr/bin/env python3
"""Attempt120: construct two fixed bounded GGN responses, then score them."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
from decimal import Decimal
from pathlib import Path
import time

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '120_postoracle_bounded_ggn_strength_scan'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '913487b26a097fab004906070f254801a3d29f0a3ffb8babaae518ecc3fbcb9a'
PRIOR116_SOURCE = PROJECT/'scripts/ablation/construct_blind_diverse8_unit_gradient_inverse_ggn_stronger_damping.py'
PRIOR117_SOURCE = PROJECT/'scripts/ablation/evaluate_postfreeze_blind_inverse_damping_comparison.py'
GRID = (('1e-2', 0.01), ('1e-1', 0.1))
EXPECTED_RHO = 561.5508153958936
EXPECTED_LAMBDAS = {'1e-2': 5.615508153958936,
                    '1e-1': 56.15508153958936}
EXPECTED_RHS = 'a90785341961a0a2a08e7949961cb105867cf2509013b01b65c2013322ead574'
EXPECTED_RAW = '8ddc1801ee82504e10898e97d12b53fea1442ed029314dc47d7d0f68523c2d0a'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT/path


def require_absent(*paths):
    for path in paths:
        if path.exists() or path.is_symlink():
            raise ValueError('Attempt120 output already exists: '+str(path))


def _import_pinned(path, expected, name):
    if sha256_file(path) != expected:
        raise ValueError('Frozen helper source SHA256 mismatch: '+str(path))
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def load_spec():
    if sha256_file(SPEC_PATH) != SPEC_SHA256:
        raise ValueError('Attempt120 spec SHA256 mismatch')
    spec = json.loads(SPEC_PATH.read_text())
    control, scan = spec['scientific_control'], spec['scan']
    if (spec['attempt_id'] != ATTEMPT or
            scan['damping_fractions'] != [0.01, 0.1] or
            scan['predeclared_lambdas'] != EXPECTED_LAMBDAS or
            scan['solve_order'] != ['1e-2', '1e-1'] or
            scan['cg_iterations'] != 80 or scan['checkpoint_iteration'] != 40 or
            scan['preconditioner'] is not None or scan['residual_early_stop'] is not False or
            scan['warm_start'] is not False or scan['iterate_selection'] is not False or
            scan['bounded_response_normalization'] is not False or
            control['expected_rho_b'] != EXPECTED_RHO or
            control['expected_rhs_aggregate_sha256'] != EXPECTED_RHS or
            control['expected_raw_response_sha256'] != EXPECTED_RAW or
            spec['information_policy']['oracle_adl_access_before_barrier'] is not False or
            spec['barrier']['artifact_and_manifest_published_before_oracle'] is not True):
        raise ValueError('Attempt120 frozen scan inventory mismatch')
    for key in spec['paths']:
        if key != 'result_path':
            text = spec['paths'][key].lower()
            if any(token in text for token in ('models/base', 'oracle_adl', 'adapter', 'evaluation.json')):
                raise ValueError('Privileged construction path')
    return spec


def load_blind_kernels(spec):
    """Import hash-pinned blind kernels and validate the full 116 provenance."""
    record = spec['frozen_attempt116']
    if (record['attempt_id'] != '116_blind_diverse8_unit_gradient_inverse_ggn_stronger_damping_candidate' or
            record['rho_b'] != EXPECTED_RHO or record['response_raw_rhs_sha256'] != EXPECTED_RAW or
            record['artifact']['sha256'] != 'c2ed1053b2f57416a748f7104f7260481ce10891750492a4a4aa32885dc216ee'):
        raise ValueError('Attempt116 frozen numerical record mismatch')
    a116 = _import_pinned(PRIOR116_SOURCE, record['source']['sha256'], 'attempt120_blind_116')
    a113, original = a116.load_attempt113()
    s116 = a116.load_spec(original)
    a116.validate_blind_development_linkage(s116)
    a117 = _import_pinned(PRIOR117_SOURCE,
                          spec['frozen_attempt117_validator']['source_sha256'],
                          'attempt120_frozen_validator117')
    manifest = a117.oracle_validator.load_json_object(
        path_of(record['manifest']['path']), 'Attempt116 construction manifest')
    a117.validate_construction_record(record, manifest, s116)
    for key in spec['scientific_control']['same_fields']:
        if s116[key] != original[key]:
            raise ValueError('Attempt116 scientific control changed: '+key)
    control = spec['scientific_control']
    if (s116['ggn']['operator'] != control['ggn_operator'] or
            s116['ggn']['rows'] != control['ggn_rows'] or
            s116['ggn']['denominator'] != control['ggn_denominator'] or
            s116['fisher_numerics']['implementation'] != control['ggn_numerics'] or
            s116['rhs']['rows'] != control['rhs_rows_per_corpus'] or
            s116['rhs']['batch_size'] != control['rhs_batch_size'] or
            s116['eligible_tensors']['expected_matrix_count'] != control['selected_matrix_count'] or
            s116['probe']['rows'] != control['probe_rows'] or
            s116['probe']['batch_size'] != control['probe_batch_size'] or
            manifest['b_blind']['ordered_matrix_hash_sha256'] != EXPECTED_RHS or
            manifest['rayleigh']['rho_b'] != EXPECTED_RHO or
            manifest['functional_responses']['response_raw_rhs']['raw_sha256'] != EXPECTED_RAW):
        raise ValueError('Attempt116 scientific inventory mismatch')
    # Hash the old artifact as provenance; its inverse is never reused in either solve.
    if sha256_file(path_of(record['artifact']['path'])) != record['artifact']['sha256']:
        raise ValueError('Attempt116 artifact serialized SHA256 mismatch')
    return a113, a116, s116, manifest, a117


def runtime_spec(s116, spec, label, fraction):
    result = copy.deepcopy(s116)
    result['attempt_id'] = ATTEMPT
    result['damping']['damping_fraction'] = fraction
    result['paths'] = {
        'artifact_path': spec['paths']['artifact_path'],
        'checkpoint_path': spec['paths']['checkpoint_'+label],
        'manifest_path': spec['paths']['manifest_path']}
    return result


def rayleigh_from_gb(kernels, eligible, direction):
    bb = kernels.matrixwise_dot(direction, direction)
    bgb = kernels.matrixwise_grad_dot(eligible, direction)
    rho = bgb / bb
    if not math.isfinite(rho) or rho <= 0 or rho != EXPECTED_RHO:
        raise ValueError('Blind RHS Rayleigh quotient differs from frozen rho_b')
    return {'bb': bb, 'bGb': bgb, 'rho_b': rho}


def fixed_lambda(rho, fraction):
    if rho != EXPECTED_RHO or fraction not in (0.01, 0.1):
        raise ValueError('Attempt120 rho/damping fraction changed')
    # The grid and frozen rho are decimal specifications. One final binary64
    # cast reproduces the predeclared numeric lambdas, avoiding a one-ULP
    # difference from multiplying their binary64 approximations.
    value = float(Decimal(str(fraction)) * Decimal(str(rho)))
    label = '1e-2' if fraction == 0.01 else '1e-1'
    if not math.isfinite(value) or value <= 0 or value != EXPECTED_LAMBDAS[label]:
        raise ValueError('Nonfinite or nonpositive damping')
    return value


def bounded_response(raw, inverse, damping):
    for tensor in (raw, inverse):
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or
                tensor.dtype != torch.float32 or not tensor.is_contiguous() or
                tuple(tensor.shape) != (128, 2048) or
                not bool(torch.isfinite(tensor).all())):
            raise ValueError('Malformed blind response')
    if damping not in (fixed_lambda(EXPECTED_RHO, 0.01),
                       fixed_lambda(EXPECTED_RHO, 0.1)):
        raise ValueError('Bounded response damping outside frozen grid')
    return (raw.to(torch.float64) - damping * inverse.to(torch.float64)).to(torch.float32).contiguous()


def fresh_state(rhs_cpu, eligible, kernels):
    """Same x=0, r=b, p=r as Attempt113, preserving b for the second solve."""
    names = [name+'.weight' for name, _ in eligible]
    if list(rhs_cpu) != names or len(names) != 98:
        raise ValueError('Fresh CG RHS support mismatch')
    r = {name+'.weight': rhs_cpu[name+'.weight'].to(
        module.weight.device, dtype=torch.float32, copy=True).contiguous()
        for name, module in eligible}
    p = {name: value.clone() for name, value in r.items()}
    x = {name: torch.zeros_like(value) for name, value in r.items()}
    rr = kernels.matrixwise_dot(r, r)
    if not math.isfinite(rr) or rr <= 0:
        raise ValueError('Malformed CG initial residual')
    return {'iteration': 0, 'x': x, 'r': r, 'p': p,
            'rr': rr, 'b_norm': math.sqrt(rr), 'iterations': []}


def _checkpoint_rayleigh(rho, fraction):
    return {'rho_b': rho, 'lambda': fixed_lambda(rho, fraction)}


def _checkpoint_path(spec, label):
    return path_of(spec['paths']['checkpoint_'+label])


def solve_one(kernels, model, corpora, probe, eligible, inventory, rhs_cpu,
              rhs_record, gradient_records, raw, spec, s116, label, fraction, rho,
              *, resume=False):
    run_spec = runtime_spec(s116, spec, label, fraction)
    damping = fixed_lambda(rho, fraction)
    checkpoint = _checkpoint_path(spec, label)
    checkpoint_meta = {'path': str(checkpoint), 'iteration': 40}
    if resume:
        payload = kernels.load_checkpoint(checkpoint)
        kernels.validate_checkpoint(payload, run_spec, inventory, rhs_record,
                                    gradient_records, _checkpoint_rayleigh(rho, fraction))
        state = kernels.restore_cg_state(payload, eligible, inventory)
        checkpoint_meta['resumed'] = True
        checkpoint_meta['raw_hashes'] = payload['raw_hashes']
        del payload
        gc.collect()
    else:
        require_absent(checkpoint)
        state = fresh_state(rhs_cpu, eligible, kernels)
        if not math.isclose(state['b_norm'], rhs_record['norm'], rel_tol=1e-6):
            raise ValueError('CG RHS norm mismatch')
        checkpoint_meta['resumed'] = False

    def apply_operator(direction, iteration):
        print(f'{label} CG {iteration}/80 | GGN', flush=True)
        return kernels.ggn_action_stage(model, corpora['fineweb'][:64].contiguous(),
                                        eligible, direction, run_spec,
                                        progress_label=f'{label} CG {iteration}/80 GGN')

    def report(row):
        print(f"{label} CG {row['iteration']}/80 | relative residual "
              f"{row['relative_residual']:.6g} | elapsed "
              f"{row['cumulative_elapsed_seconds']:.1f}s | ETA "
              f"{row['operator_eta_seconds']:.1f}s", flush=True)

    def on_checkpoint(current):
        before = kernels.state_hashes(current, inventory)
        payload = kernels.checkpoint_payload(current, inventory, run_spec,
                                             rhs_record, gradient_records,
                                             _checkpoint_rayleigh(rho, fraction))
        if kernels.state_hashes(current, inventory) != before:
            raise ValueError('CG state mutated during checkpoint preparation')
        kernels.atomic_torch_publish(checkpoint, payload)
        if kernels.state_hashes(current, inventory) != before:
            raise ValueError('CG state mutated during checkpoint publication')
        checkpoint_meta['raw_hashes'] = payload['raw_hashes']
        checkpoint_meta['serialized_size_bytes'] = checkpoint.stat().st_size
        del payload
        print(f'{label} | iteration-40 checkpoint published', flush=True)

    state, solver = kernels.run_fixed_80_cg(
        state, eligible, apply_operator, damping, run_spec,
        on_checkpoint=None if resume else on_checkpoint, report=report)
    if state['iteration'] != 80 or solver['operator_applications'] != 80:
        raise ValueError('Only fixed x_80 may become a candidate')
    model.zero_grad(set_to_none=True)
    inverse, jvp_audit = kernels.probe_response(
        model, probe, state['x'], run_spec, label=f'{label} inverse x_80 JVP')
    bounded = bounded_response(raw, inverse, damping)
    candidate = {row['name']: state['x'][row['name']].detach().to(
        'cpu', dtype=torch.float32, copy=True).contiguous() for row in inventory}
    x_hashes, x_aggregate = kernels.ordered_vector_hashes(candidate, inventory)
    raw_hash = kernels.blind.sha256_raw_float32_tensor(raw, torch)
    inverse_hash = kernels.blind.sha256_raw_float32_tensor(inverse, torch)
    bounded_hash = kernels.blind.sha256_raw_float32_tensor(bounded, torch)
    if raw_hash != EXPECTED_RAW:
        raise ValueError('Raw blind response changed')
    record = {
        'damping_fraction': fraction, 'lambda': damping,
        'solver': solver, 'checkpoint': checkpoint_meta,
        'x_80': {'matrix_raw_sha256': x_hashes,
                 'ordered_matrix_hash_sha256': x_aggregate,
                 'norm': math.sqrt(kernels.matrixwise_dot(state['x'], state['x'])),
                 'matrix_count': 98},
        'responses': {'response_raw_rhs_sha256': raw_hash,
                      'response_inverse_sha256': inverse_hash,
                      'response_bounded_sha256': bounded_hash,
                      'shape': [128, 2048], 'dtype': 'contiguous_CPU_float32',
                      'inverse_jvp_audit': jvp_audit},
        'exact_realized_candidate': 'y_80=b_blind-lambda*x_80',
        'functional_arithmetic': spec['scan']['bounded_response_arithmetic'],
        'exact_spectral_identity_claimed_for_approximate_x80': False,
        'approximate_solve_relative_linear_residual': solver['relative_linear_residual']}
    del state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'x_80': candidate, 'response_inverse': inverse,
            'response_bounded': bounded}, record


def stage_payload(spec, inventory, rhs_record, rho, first, record, kernels):
    hashes, aggregate = kernels.ordered_vector_hashes(first['x_80'], inventory)
    if hashes != record['x_80']['matrix_raw_sha256'] or aggregate != record['x_80']['ordered_matrix_hash_sha256']:
        raise ValueError('First candidate changed before staging')
    return {'format_version': 1, 'label': '1e-2', 'spec_sha256': SPEC_SHA256,
            'source_sha256': sha256_file(__file__),
            'rhs_sha256': rhs_record['ordered_matrix_hash_sha256'],
            'rho_b': rho, 'coordinate_inventory': inventory,
            'candidate': first, 'candidate_record': record}


def validate_stage(payload, spec, inventory, rhs_record, rho, kernels):
    if (not isinstance(payload, dict) or set(payload) !=
            {'format_version', 'label', 'spec_sha256', 'source_sha256',
             'rhs_sha256', 'rho_b', 'coordinate_inventory', 'candidate', 'candidate_record'} or
            payload['format_version'] != 1 or payload['label'] != '1e-2' or
            payload['spec_sha256'] != SPEC_SHA256 or
            payload['source_sha256'] != sha256_file(__file__) or
            payload['rhs_sha256'] != rhs_record['ordered_matrix_hash_sha256'] or
            payload['rho_b'] != rho or payload['coordinate_inventory'] != inventory):
        raise ValueError('First-candidate stage provenance mismatch')
    candidate, record = payload['candidate'], payload['candidate_record']
    if (not isinstance(candidate, dict) or list(candidate) !=
            spec['artifact']['per_candidate_inventory'] or
            record['damping_fraction'] != 0.01 or
            record['lambda'] != fixed_lambda(rho, 0.01) or
            record['solver']['operator_applications'] != 80 or
            len(record['solver']['iterations']) != 80 or
            record['responses']['response_raw_rhs_sha256'] != EXPECTED_RAW):
        raise ValueError('First-candidate stage inventory mismatch')
    hashes, aggregate = kernels.ordered_vector_hashes(candidate['x_80'], inventory)
    if hashes != record['x_80']['matrix_raw_sha256'] or aggregate != record['x_80']['ordered_matrix_hash_sha256']:
        raise ValueError('First-candidate stage x_80 hash mismatch')
    for name, field in (('response_inverse', 'response_inverse_sha256'),
                        ('response_bounded', 'response_bounded_sha256')):
        value = candidate[name]
        if (value.dtype != torch.float32 or value.device.type != 'cpu' or
                not value.is_contiguous() or tuple(value.shape) != (128, 2048) or
                not bool(torch.isfinite(value).all()) or
                kernels.blind.sha256_raw_float32_tensor(value, torch) != record['responses'][field]):
            raise ValueError('First-candidate stage response hash mismatch')
    return candidate, record


def _safe_load(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or symlinked recovery file: '+str(path))
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def validate_artifact(artifact, manifest, spec, kernels):
    if (not isinstance(artifact, dict) or list(artifact) != spec['artifact']['inventory'] or
            kernels.blind.sha256_raw_float32_tensor(artifact['response_raw_rhs'], torch) != EXPECTED_RAW or
            list(artifact['candidates']) != spec['artifact']['candidate_keys']):
        raise ValueError('Attempt120 artifact inventory/raw response mismatch')
    inventory = manifest['coordinate_inventory']
    for label, fraction in GRID:
        candidate, record = artifact['candidates'][label], manifest['systems'][label]
        if (list(candidate) != spec['artifact']['per_candidate_inventory'] or
                record['damping_fraction'] != fraction or
                record['lambda'] != fixed_lambda(EXPECTED_RHO, fraction) or
                record['solver']['operator_applications'] != 80 or
                len(record['solver']['iterations']) != 80 or
                record['responses']['response_raw_rhs_sha256'] != EXPECTED_RAW):
            raise ValueError('Attempt120 candidate/solver inventory mismatch')
        hashes, aggregate = kernels.ordered_vector_hashes(candidate['x_80'], inventory)
        if hashes != record['x_80']['matrix_raw_sha256'] or aggregate != record['x_80']['ordered_matrix_hash_sha256']:
            raise ValueError('Attempt120 x_80 raw hash mismatch')
        for name, field in (('response_inverse', 'response_inverse_sha256'),
                            ('response_bounded', 'response_bounded_sha256')):
            value = candidate[name]
            if (value.device.type != 'cpu' or value.dtype != torch.float32 or
                    not value.is_contiguous() or tuple(value.shape) != (128, 2048) or
                    not bool(torch.isfinite(value).all()) or
                    kernels.blind.sha256_raw_float32_tensor(value, torch) != record['responses'][field]):
                raise ValueError('Attempt120 response raw hash mismatch')
            if name == 'response_bounded' and not torch.equal(
                    value, bounded_response(artifact['response_raw_rhs'],
                                            candidate['response_inverse'], record['lambda'])):
                raise ValueError('Attempt120 bounded arithmetic mismatch')
    return True


def publish_construction(spec, kernels, raw, first, record_first, second, record_second,
                         inventory, rhs_record, gradient_records, rayleigh, corpus_manifests,
                         s116, before_model_hashes):
    artifact_path = path_of(spec['paths']['artifact_path'])
    manifest_path = path_of(spec['paths']['manifest_path'])
    require_absent(artifact_path, manifest_path)
    artifact = {'response_raw_rhs': raw,
                'candidates': {'1e-2': first, '1e-1': second}}
    manifest = {
        'format_version': 1, 'attempt_id': ATTEMPT,
        'spec_sha256': SPEC_SHA256, 'constructor_source_sha256': sha256_file(__file__),
        'frozen_attempt116': spec['frozen_attempt116'],
        'final_checkpoint_files': s116['final_checkpoint']['files'],
        'blind_source_provenance': s116['blind_source_provenance'],
        'corpus_order': s116['corpus_order'],
        'corpora': [{**row, **corpus_manifests[row['name']]} for row in s116['corpora']],
        'generic_loss': s116['rhs']['loss'],
        'coordinate_inventory': inventory,
        'per_corpus_gradients': gradient_records, 'b_blind': rhs_record,
        'rayleigh': rayleigh,
        'ggn_operator': s116['ggn']['operator'],
        'ggn_numerics': s116['fisher_numerics'],
        'probe': s116['probe'], 'readout': s116['readout'],
        'systems': {'1e-2': record_first, '1e-1': record_second},
        'response_raw_rhs_sha256': EXPECTED_RAW,
        'model_state_hashes_before': before_model_hashes,
        'candidate_hashes_frozen_before_oracle_access': True,
        'oracle_metric_fields_present': False,
        'information_policy': spec['information_policy']}
    validate_artifact(artifact, manifest, spec, kernels)
    kernels.atomic_torch_publish(artifact_path, artifact)
    manifest['artifact'] = {'path': str(artifact_path),
                            'serialized_sha256': sha256_file(artifact_path),
                            'inventory': spec['artifact']['inventory'],
                            'candidate_keys': spec['artifact']['candidate_keys'],
                            'per_candidate_inventory': spec['artifact']['per_candidate_inventory'],
                            'x_matrix_count_each': 98,
                            'response_shape': [128, 2048],
                            'response_dtype': 'contiguous_CPU_float32'}
    try:
        kernels.atomic_json_publish(manifest_path, manifest)
    except Exception:
        artifact_path.unlink()
        raise
    print('ATTEMPT120_CONSTRUCTION_FROZEN', flush=True)
    print('candidate_artifact_sha256='+manifest['artifact']['serialized_sha256'], flush=True)
    for label, _ in GRID:
        print(label+'_bounded_raw_sha256='+manifest['systems'][label]['responses']['response_bounded_sha256'], flush=True)
    return artifact, manifest


def category(primary_gain, secondary_gain):
    if primary_gain is None or secondary_gain is None:
        return 'undefined_required_position_cosine'
    if primary_gain >= 0.03 and secondary_gain >= 0.02:
        return 'clear_improvement'
    if primary_gain >= 0.015 and secondary_gain >= 0:
        return 'moderate_improvement'
    return 'no_meaningful_improvement'


def descriptive_dev_choice(rows):
    if set(rows) != {'1e-2', '1e-1'}:
        raise ValueError('Only two fixed strengths may be compared')
    rank = {'clear_improvement': 2, 'moderate_improvement': 1,
            'no_meaningful_improvement': 0,
            'undefined_required_position_cosine': -1}
    categories = {label: rows[label]['category'] for label, _ in GRID}
    if any(value not in rank for value in categories.values()):
        raise ValueError('Unknown Attempt120 improvement category')
    highest = max(rank[value] for value in categories.values())
    if highest <= 0:
        return None
    finalists = [item for item in GRID if rank[categories[item[0]]] == highest]
    if len(finalists) == 1:
        return finalists[0][0]
    return max(finalists, key=lambda item: (
        rows[item[0]]['gain_primary_vs_raw'],
        rows[item[0]]['gain_secondary_vs_raw'], -item[1]))[0]


def norm_ratios(candidate, raw):
    a, b = candidate.double(), raw.double()
    numerator = torch.linalg.vector_norm(a, dim=1)
    denominator = torch.linalg.vector_norm(b, dim=1)
    total_denominator = float(torch.linalg.vector_norm(b))
    return {'total': None if total_denominator == 0 else
            float(torch.linalg.vector_norm(a))/total_denominator,
            'per_position': [None if float(d) == 0 else float(n/d)
                             for n, d in zip(numerator, denominator)]}


def _delta(new, old, key):
    a, b = new[key], old[key]
    return None if a is None or b is None else a-b


def load_attempt119_context(spec, raw_report, validator):
    record = spec['frozen_attempt119']
    for kind in ('spec', 'source', 'result'):
        if sha256_file(path_of(record[kind+'_path'])) != record[kind+'_sha256']:
            raise ValueError('Frozen Attempt119 '+kind+' SHA256 mismatch')
    result = json.loads(path_of(record['result_path']).read_text())
    construction = result['bounded_construction']
    if (result['attempt_id'] != '119_postoracle_bounded_ggn_filter_diagnostic' or
            construction['oracle_free_diagnostics']['bounded_response_raw_sha256'] != record['bounded_raw_sha256'] or
            construction['bounded_hash_frozen_before_oracle_access'] is not True or
            result['provenance']['spec_sha256'] != record['spec_sha256'] or
            result['provenance']['source_sha256'] != record['source_sha256'] or
            result['bounded_construction']['upstream']['attempt116'] != spec['frozen_attempt116']):
        raise ValueError('Frozen Attempt119 construction context mismatch')
    reports = result['oracle_evaluation']['candidate_reports']
    if reports['raw_rhs'] != raw_report:
        raise ValueError('Attempt119 raw RHS oracle baseline differs')
    return reports['bounded_filter']


def evaluate_frozen(spec, artifact, manifest, kernels):
    """Only this function opens the oracle; caller must publish first."""
    artifact_path = path_of(spec['paths']['artifact_path'])
    manifest_path = path_of(spec['paths']['manifest_path'])
    output_path = path_of(spec['paths']['result_path'])
    if (not artifact_path.is_file() or not manifest_path.is_file() or
            sha256_file(artifact_path) != manifest['artifact']['serialized_sha256'] or
            json.loads(manifest_path.read_text()) != manifest or
            manifest['spec_sha256'] != SPEC_SHA256 or
            manifest['constructor_source_sha256'] != sha256_file(__file__) or
            not manifest['candidate_hashes_frozen_before_oracle_access']):
        raise ValueError('Construction barrier not published/validated')
    validate_artifact(artifact, manifest, spec, kernels)
    require_absent(output_path)
    a117 = _import_pinned(PRIOR117_SOURCE,
                          spec['frozen_attempt117_validator']['source_sha256'],
                          'attempt120_oracle_validator117')
    if spec['oracle'] != a117.FROZEN_SPEC['oracle']:
        raise ValueError('Frozen oracle provenance differs from Attempt117')
    difference, oracle_manifest = a117.validate_oracle(spec['oracle']['provenance_inputs'])
    raw = artifact['response_raw_rhs']
    reports = {'raw_rhs': a117.cosine_report(raw, difference)}
    reports['bounded_1e-3_context'] = load_attempt119_context(spec, reports['raw_rhs'], a117)
    rows = {}
    for label, fraction in GRID:
        bounded = artifact['candidates'][label]['response_bounded']
        report = a117.cosine_report(bounded, difference)
        reports['bounded_'+label] = report
        context = reports['bounded_1e-3_context']
        p = 'positions_1_4_mean_cosine'
        s = 'positions_1_127_mean_cosine'
        dp, ds = _delta(report, reports['raw_rhs'], p), _delta(report, reports['raw_rhs'], s)
        rows[label] = {
            'damping_fraction': fraction, 'lambda': manifest['systems'][label]['lambda'],
            'gain_primary_vs_raw': dp, 'gain_secondary_vs_raw': ds,
            'gain_primary_vs_attempt119_1e-3': _delta(report, context, p),
            'gain_secondary_vs_attempt119_1e-3': _delta(report, context, s),
            'category': category(dp, ds),
            'candidate_vs_raw_signed_cosine': a117.cosine_report(bounded, raw),
            'candidate_to_raw_norm_ratio': norm_ratios(bounded, raw),
            'approximate_solve_relative_linear_residual':
                manifest['systems'][label]['solver']['relative_linear_residual']}
    result = {
        'format_version': 1, 'attempt_id': ATTEMPT,
        'spec_sha256': SPEC_SHA256, 'source_sha256': sha256_file(__file__),
        'construction_manifest_sha256': sha256_file(manifest_path),
        'candidate_artifact_serialized_sha256': manifest['artifact']['serialized_sha256'],
        'oracle_artifact_sha256': spec['oracle']['serialized_sha256'],
        'oracle_difference_raw_sha256': spec['oracle']['difference_raw_sha256'],
        'oracle_manifest_sha256': spec['oracle']['provenance_inputs']['oracle_manifest']['sha256'],
        'oracle_validation': {'validator_source_sha256': spec['frozen_attempt117_validator']['source_sha256'],
                              'validated_manifest_oracle_sha256': oracle_manifest['oracle_adl_sha256']},
        'metric_definition': spec['metrics'], 'candidate_reports': reports,
        'development_scan': rows,
        'descriptive_dev_set_choice_if_same_category': descriptive_dev_choice(rows),
        'interpretation': spec['interpretation'],
        'spectral_identity_caveat': spec['scan']['approximate_solve_caveat'],
        'exact_realized_candidate': spec['scan']['exact_realized_candidate'],
        'information_policy': spec['information_policy'],
        'construction_frozen_before_oracle_access': True,
        'same_specimen_development_set_selection': True}
    if sha256_file(SPEC_PATH) != SPEC_SHA256 or sha256_file(__file__) != result['source_sha256']:
        raise ValueError('Attempt120 source/spec changed during evaluation')
    require_absent(output_path)
    a117.oracle_validator.require_file_hash(
        path_of(spec['oracle']['path']), spec['oracle']['serialized_sha256'])
    # Exclusive, atomic publication. No scientific output is replaced.
    kernels.atomic_json_publish(output_path, result)
    return result


def run(*, resume=False, model_loader=None):
    spec = load_spec()
    paths = {key: path_of(value) for key, value in spec['paths'].items()}
    if any(path.is_symlink() for path in paths.values()):
        raise ValueError('Attempt120 output path must not be a symlink')
    if paths['result_path'].exists() or paths['result_path'].is_symlink():
        raise ValueError('Attempt120 result already exists')
    published = paths['artifact_path'].is_file() and paths['manifest_path'].is_file()
    if paths['artifact_path'].exists() != paths['manifest_path'].exists():
        raise ValueError('Partial construction publication; manual review required')
    if published:
        if not resume:
            raise ValueError('Construction already published; use --resume to evaluate')
        # Evaluation-only recovery; do not load a model or reconstruct anything.
        a113, _, _, _, _ = load_blind_kernels(spec)
        manifest = json.loads(paths['manifest_path'].read_text())
        if (manifest['spec_sha256'] != SPEC_SHA256 or
                manifest['constructor_source_sha256'] != sha256_file(__file__) or
                sha256_file(paths['artifact_path']) != manifest['artifact']['serialized_sha256']):
            raise ValueError('Published construction provenance mismatch')
        artifact = _safe_load(paths['artifact_path'])
        validate_artifact(artifact, manifest, spec, a113)
        return evaluate_frozen(spec, artifact, manifest, a113)
    require_absent(paths['artifact_path'], paths['manifest_path'])
    stage = paths['stage_first_path']
    cp1, cp2 = paths['checkpoint_1e-2'], paths['checkpoint_1e-1']
    if not resume:
        require_absent(stage, cp1, cp2)
    elif not any(path.is_file() and not path.is_symlink() for path in (stage, cp1, cp2)):
        raise ValueError('--resume requires a compatible recovery file')
    if cp2.exists() and not stage.exists():
        raise ValueError('Second recovery checkpoint requires completed first stage')
    a113, a116, s116, _, _ = load_blind_kernels(spec)
    # Reuse blind code unchanged, with only checkpoint provenance redirected here.
    old_sha, old_file = a113.SPEC_SHA256, a113.__file__
    a113.SPEC_SHA256, a113.__file__ = SPEC_SHA256, str(Path(__file__).resolve())
    try:
        run_spec = runtime_spec(s116, spec, '1e-2', 0.01)
        corpora, probe, corpus_manifests = a113.validate_blind_sources(run_spec)
        model_dir = a113.safe_input_path(run_spec['final_checkpoint']['directory'])
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = (a113.blind.load_local_model(model_dir, device, torch) if model_loader is None
                 else model_loader(model_dir, device, torch))
        a113.blind.validate_loaded_model(model, torch)
        eligible = a113.blind.discover_eligible_linear_weights(model, torch)
        a113.blind.freeze_other_parameters(model, eligible)
        inventory = a113.coordinate_inventory(eligible)
        before_model = a113.blind.model_state_hashes(model, torch)
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise ValueError('Unexpected parameter gradient before construction')
        rhs_cpu, gradient_records, rhs_record = a113.build_blind_rhs(
            model, corpora, eligible, inventory)
        a116.require_exact_rhs(rhs_record, s116['expected_attempt113_rhs'])
        if rhs_record['ordered_matrix_hash_sha256'] != EXPECTED_RHS:
            raise ValueError('Blind RHS aggregate SHA256 differs')
        rhs_device = a113.vector_to_device(rhs_cpu, eligible)
        raw, raw_audit = a113.probe_response(
            model, probe, rhs_device, run_spec, label='raw RHS block13 JVP')
        if a113.blind.sha256_raw_float32_tensor(raw, torch) != EXPECTED_RAW:
            raise ValueError('Raw RHS response differs from frozen value')
        started = time.perf_counter()
        g_audit = a113.ggn_action_stage(
            model, corpora['fineweb'][:64].contiguous(), eligible, rhs_device,
            run_spec, progress_label='blind RHS Rayleigh GGN')
        rayleigh = rayleigh_from_gb(a113, eligible, rhs_device)
        rayleigh['operator_seconds'] = time.perf_counter()-started
        rayleigh['ggn_audit'] = g_audit
        model.zero_grad(set_to_none=True)
        del rhs_device
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if stage.is_file() and not stage.is_symlink():
            payload = _safe_load(stage)
            first, record_first = validate_stage(
                payload, spec, inventory, rhs_record, rayleigh['rho_b'], a113)
            del payload
            if cp1.is_file() and not cp1.is_symlink():
                cp1.unlink()  # The verified x_80 stage supersedes its x_40 recovery.
            print('first strength | validated staged x_80; continuing second strength', flush=True)
        else:
            first, record_first = solve_one(
                a113, model, corpora, probe, eligible, inventory, rhs_cpu,
                rhs_record, gradient_records, raw, spec, s116,
                '1e-2', 0.01, rayleigh['rho_b'], resume=resume and cp1.is_file())
            payload = stage_payload(spec, inventory, rhs_record,
                                    rayleigh['rho_b'], first, record_first, a113)
            a113.atomic_torch_publish(stage, payload)
            del payload
            if cp1.is_file() and not cp1.is_symlink():
                cp1.unlink()
            print('first strength | staged x_80; iteration-40 checkpoint removed', flush=True)

        # Stage on disk permits dropping the first 2.6-GiB parameter vector.
        del first
        gc.collect()
        second, record_second = solve_one(
            a113, model, corpora, probe, eligible, inventory, rhs_cpu,
            rhs_record, gradient_records, raw, spec, s116,
            '1e-1', 0.1, rayleigh['rho_b'], resume=resume and cp2.is_file())
        model.zero_grad(set_to_none=True)
        a113.blind.verify_model_unchanged(model, before_model, torch)
        a113._input_hashes_unchanged(run_spec)
        payload = _safe_load(stage)
        first, record_first = validate_stage(
            payload, spec, inventory, rhs_record, rayleigh['rho_b'], a113)
        del payload
        artifact, manifest = publish_construction(
            spec, a113, raw, first, record_first, second, record_second,
            inventory, rhs_record, gradient_records, rayleigh, corpus_manifests,
            s116, before_model)
        # Construction has been published. Remove recovery files before oracle I/O.
        for path in (stage, cp2):
            if path.is_file() and not path.is_symlink():
                path.unlink()
        del eligible, model, rhs_cpu, corpora, probe, first, second
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return evaluate_frozen(spec, artifact, manifest, a113)
    finally:
        a113.SPEC_SHA256, a113.__file__ = old_sha, old_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(resume=args.resume)


if __name__ == '__main__':
    main()
