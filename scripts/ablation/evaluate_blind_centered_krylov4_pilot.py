#!/usr/bin/env python3
"""Post-freeze parameter-space evaluation of Attempt 022; never tunes c_hat."""
import argparse
import copy
import importlib.util
import json
import math
from contextlib import ExitStack, contextmanager
from pathlib import Path

import torch
from safetensors import safe_open

PROJECT = Path(__file__).resolve().parents[2]
CONSTRUCTOR_PATH = Path(__file__).with_name('construct_blind_centered_krylov4_pilot.py')
HISTORICAL_SPEC_PATH = PROJECT/'experiments/attempts/016_known_base_exact_displacement_jvp_ceiling_prefix0_13/spec.json'
loader = importlib.util.spec_from_file_location('attempt022_frozen_constructor', CONSTRUCTOR_PATH)
c = importlib.util.module_from_spec(loader)
loader.loader.exec_module(c)
historical = json.loads(HISTORICAL_SPEC_PATH.read_text())

EVALUATION_SPEC = {
    'format_version': 1,
    'attempt_id': c.FROZEN_SPEC['attempt_id'],
    'purpose': 'post_freeze_known_base_parameter_space_evaluation_no_tuning',
    'construction_policy': c.FROZEN_SPEC['information_policy'],
    'base': historical['base'],
    'base_tensor_loading': {'raw_to_loaded_W0': '.to(dtype=torch.float32).contiguous()',
                            'displacement': 'W1_FP32_minus_loaded_W0_FP32_in_CPU_float64'},
    'coordinates': c.FROZEN_SPEC['coordinates'],
    'metrics': ['deltaS_norm', 'c_true_norm', 'projection_ceiling',
                'cosine_coefficients', 'cosine_candidate_to_full_DeltaS',
                'fraction_of_subspace_ceiling', 'optimal_scalar',
                'relative_residual_after_optimal_rescaling'],
    'functional_evaluation': False,
    'defaults': {
        'base_directory': '/root/model-diff-scratch/models/base',
        'final_directory': c.FROZEN_SPEC['defaults']['final_directory'],
        'basis_directory': c.FROZEN_SPEC['defaults']['basis_directory'],
        'construction_spec_path': 'experiments/attempts/022_blind_centered_regression_krylov4_pilot/spec.json',
        'construction_manifest_path': 'experiments/attempts/022_blind_centered_regression_krylov4_pilot/construction-manifest.json',
        'artifact_path': c.FROZEN_SPEC['defaults']['artifact_path'],
        'evaluation_spec_path': 'experiments/attempts/022_blind_centered_regression_krylov4_pilot/evaluation_spec.json',
        'output_path': 'experiments/attempts/022_blind_centered_regression_krylov4_pilot/evaluation.json',
    },
    'outputs': {'overwrite': False, 'timestamps': False, 'host_metadata': False, 'gpu_metadata': False},
}


def projection_metrics(c_hat, c_true, delta_norm):
    if (c_hat.shape != (4,) or c_true.shape != (4,) or c_hat.dtype != torch.float64 or
            c_true.dtype != torch.float64 or c_hat.device.type != 'cpu' or
            c_true.device.type != 'cpu' or not bool(torch.isfinite(c_hat).all()) or
            not bool(torch.isfinite(c_true).all()) or not math.isfinite(delta_norm) or delta_norm < 0):
        raise ValueError('Invalid frozen coefficient or displacement data')
    before = c_hat.clone()
    candidate_norm = float(torch.linalg.vector_norm(c_hat))
    true_norm = float(torch.linalg.vector_norm(c_true))
    dot = float(torch.dot(c_hat, c_true))
    ceiling = true_norm/delta_norm if delta_norm else None
    coefficient_cosine = dot/(candidate_norm*true_norm) if candidate_norm*true_norm else None
    candidate_cosine = dot/(candidate_norm*delta_norm) if candidate_norm*delta_norm else None
    fraction = candidate_cosine/ceiling if candidate_cosine is not None and ceiling else None
    optimal = dot/(candidate_norm*candidate_norm) if candidate_norm else None
    residual = (float(torch.linalg.vector_norm(optimal*c_hat-c_true))/true_norm
                if optimal is not None and true_norm else None)
    if not torch.equal(c_hat, before):
        raise ValueError('Frozen c_hat changed during evaluation')
    return {'c_hat': c_hat.tolist(), 'c_true': c_true.tolist(),
            'candidate_norm': candidate_norm, 'projected_true_norm': true_norm,
            'deltaS_norm': delta_norm, 'projection_ceiling': ceiling,
            'cosine_coefficients': coefficient_cosine,
            'cosine_candidate_to_full_DeltaS': candidate_cosine,
            'fraction_of_subspace_ceiling': fraction,
            'optimal_scalar': optimal,
            'relative_residual_after_optimal_rescaling': residual}


@contextmanager
def selected_base_tensors(directory, files):
    stack = ExitStack()
    try:
        index = {}
        for record in files:
            if record['path'].endswith('.safetensors'):
                source = stack.enter_context(safe_open(str(directory/record['path']), framework='pt', device='cpu'))
                for name in source.keys():
                    if name in index:
                        raise ValueError('Duplicate historical checkpoint tensor key')
                    index[name] = source
        yield index
    finally:
        stack.close()


def true_coefficients(final, store, coordinates, base_tensors):
    parameters = dict(final.named_parameters())
    terms = [[] for _ in range(4)]
    norm_terms = []
    for row in coordinates:
        name, shape = row['name'], row['shape']
        if name not in parameters or name not in base_tensors:
            raise ValueError('Missing selected historical coordinate')
        current = parameters[name].detach().to(device='cpu', dtype=torch.float64)
        old = base_tensors[name].get_tensor(name).to(dtype=torch.float32).contiguous()
        if (old.dtype != torch.float32 or list(old.shape) != shape or list(current.shape) != shape or
                not bool(torch.isfinite(old).all())):
            raise ValueError('Historical selected tensor mismatch')
        delta = current-old.double()
        if not bool(torch.isfinite(delta).all()):
            raise ValueError('Nonfinite historical selected displacement')
        norm_terms.append(float(torch.sum(delta.square(), dtype=torch.float64)))
        for i in range(4):
            q = store.matrix(i, name, shape).double()
            terms[i].append(float(torch.sum(q*delta, dtype=torch.float64)))
    coefficients = torch.tensor([math.fsum(row) for row in terms], dtype=torch.float64)
    norm = math.sqrt(math.fsum(norm_terms))
    if not bool(torch.isfinite(coefficients).all()) or not math.isfinite(norm):
        raise ValueError('Nonfinite true projection')
    return coefficients, norm


def validate_inputs(args):
    spec = c.a.load_json_object(args.evaluation_spec_path, 'Attempt022 evaluation spec')
    if spec != EVALUATION_SPEC:
        raise ValueError('Frozen evaluation specification mismatch')
    if args.output_path.exists() or args.output_path.is_symlink():
        raise ValueError('Evaluation output already exists')
    construction_spec = c.a.load_json_object(args.construction_spec_path, 'Attempt022 construction spec')
    if construction_spec != c.FROZEN_SPEC:
        raise ValueError('Frozen construction specification mismatch')
    manifest = c.a.load_json_object(args.construction_manifest_path, 'Attempt022 construction manifest')
    if (manifest.get('attempt_id') != c.FROZEN_SPEC['attempt_id'] or
            manifest.get('information_policy') != c.FROZEN_SPEC['information_policy'] or
            manifest.get('basis', {}).get('manifest_sha256') != c.PRIOR_MANIFEST_SHA256 or
            manifest.get('source_checkpoint_files') != c.FROZEN_SPEC['final_checkpoint_files']):
        raise ValueError('Attempt022 frozen construction provenance mismatch')
    source_hashes = manifest.get('input_source_hashes', {})
    if (source_hashes.get(str(CONSTRUCTOR_PATH)) != c.a.sha256_file(CONSTRUCTOR_PATH) or
            source_hashes.get(str(args.construction_spec_path)) != c.a.sha256_file(args.construction_spec_path)):
        raise ValueError('Frozen construction source/spec hash mismatch')
    artifact, record = c.verify_artifact(args.artifact_path, manifest['artifact'])
    c_hat = artifact['c_hat'].clone()
    if c_hat.tolist() != manifest['regression']['coefficients']:
        raise ValueError('Frozen c_hat does not match construction manifest')
    recomputed, _ = c.solve_centered(artifact['b'], artifact['M'], c.FROZEN_SPEC['regression'])
    if not torch.allclose(c_hat, recomputed, rtol=1e-12, atol=1e-12):
        raise ValueError('Frozen c_hat does not reproduce centered solve')
    if c.a.checkpoint_file_records(args.base_directory) != spec['base']['files']:
        raise ValueError('Historical base checkpoint mismatch')
    if c.a.checkpoint_file_records(args.final_directory) != c.FROZEN_SPEC['final_checkpoint_files']:
        raise ValueError('Canonical final checkpoint mismatch')
    c.verify_basis_files(args.basis_directory, c.FROZEN_SPEC['basis']['files'])
    paths = [args.evaluation_spec_path, args.construction_spec_path,
             args.construction_manifest_path, args.artifact_path, Path(__file__).resolve(),
             CONSTRUCTOR_PATH, HISTORICAL_SPEC_PATH, c.PRIOR_MANIFEST_PATH]
    hashes = {str(path): c.a.sha256_file(path) for path in paths}
    if args.output_path.resolve() in [path.resolve() for path in paths]:
        raise ValueError('Evaluation output overlaps frozen input')
    def recheck():
        for path, digest in hashes.items():
            if c.a.sha256_file(Path(path)) != digest:
                raise ValueError('Frozen evaluation input changed')
        if c.a.checkpoint_file_records(args.base_directory) != spec['base']['files'] or c.a.checkpoint_file_records(args.final_directory) != c.FROZEN_SPEC['final_checkpoint_files']:
            raise ValueError('Checkpoint changed during evaluation')
        c.verify_basis_files(args.basis_directory, c.FROZEN_SPEC['basis']['files'])
    return spec, manifest, record, c_hat, hashes, recheck


def evaluate(args):
    spec, manifest, artifact_record, c_hat, hashes, recheck = validate_inputs(args)
    final = c.a.load_local_model(args.final_directory, 'cpu', torch)
    eligible = c.a.discover_eligible_linear_weights(final, torch)
    coordinates = [{'name': name+'.weight', 'shape': list(module.weight.shape)} for name, module in eligible]
    if coordinates != spec['coordinates']:
        raise ValueError('Selected final coordinate mismatch')
    before = c.a.model_state_hashes(final, torch)
    with c.BasisStore(args.basis_directory, coordinates) as store:
        gram = c.basis_gram(store, coordinates, c.FROZEN_SPEC['basis'])
        with selected_base_tensors(args.base_directory, spec['base']['files']) as base_tensors:
            c_true, delta_norm = true_coefficients(final, store, coordinates, base_tensors)
    c.a.verify_model_unchanged(final, before, torch)
    metrics = projection_metrics(c_hat, c_true, delta_norm)
    recheck()
    result = {'format_version': 1, 'attempt_id': spec['attempt_id'],
              'purpose': spec['purpose'], 'hash_algorithm': 'sha256',
              'construction_manifest_sha256': hashes[str(args.construction_manifest_path)],
              'construction_artifact': artifact_record,
              'historical_base': spec['base'], 'basis_gram': gram,
              'metrics': metrics, 'functional_evaluation': False,
              'input_source_hashes': hashes,
              'selection': False, 'coefficient_mutation': False}
    c.a.write_manifest(args.output_path, result)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in EVALUATION_SPEC['defaults'].items():
        path = Path(value)
        parser.add_argument('--'+key.replace('_', '-'), type=Path,
                            default=path if path.is_absolute() else PROJECT/path)
    return parser.parse_args(argv)


if __name__ == '__main__':
    evaluate(parse_args())
