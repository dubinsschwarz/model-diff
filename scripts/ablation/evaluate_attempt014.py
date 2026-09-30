#!/usr/bin/env python3
"""Frozen Attempt 014 cosine evaluation; no candidate modification or selection."""
import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any
import torch
PROJECT = Path(__file__).resolve().parents[2]
EXPECTED_SHAPE = (128, 2048)
ORACLE_VECTOR_TYPES = ('difference', 'base_mean', 'ft_mean')
FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '014_multicorpus_raw_jg_consensus_prefix0_13',
 'evaluation_plan': {'oracle_vector': 'difference',
                     'candidate_tensors': ['response_fineweb',
                                           'response_wikitext103_raw',
                                           'response_tinystories',
                                           'consensus_response'],
                     'cosine_dtype': 'cpu_float64',
                     'per_position_scope': 'all_128_positions',
                     'primary': {'metric': 'consensus_positions_1_4_mean_cosine',
                                 'baseline': 'response_fineweb_positions_1_4_mean_cosine',
                                 'report_delta_consensus_minus_fineweb': True},
                     'secondary': {'metric': 'consensus_positions_1_127_mean_cosine',
                                   'baseline': 'response_fineweb_positions_1_127_mean_cosine',
                                   'report_delta_consensus_minus_fineweb': True},
                     'position_0': 'report_separately',
                     'policy': ['report all individual corpus candidates',
                                'no best-corpus selection',
                                'no sign selection',
                                'no corpus reweighting',
                                'no post-result candidate modification',
                                'no post-result threshold tuning']},
 'consensus_definition': {'definition': 'arithmetic_mean_of_three_per_position_unit_positive_Jg_responses',
                          'normalization_dtype': 'cpu_float64',
                          'inputs': 'stored_equivalent_float32_responses',
                          'zero_or_nonfinite_norm': 'fail',
                          'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                          'concentration_tolerance': 1e-12},
 'inputs': {'attempt_spec': {'path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/spec.json',
                             'sha256': '556536ed4bb5259b29f4bb01cbf96bf846778bc21898a0a9ca3e61b895bc86ce'},
            'construction_manifest': {'path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/construction-manifest.json',
                                      'sha256': '6148770fe9cd29bd5f6050379fd4c01a9f3ad21501019856ea44812f6e4ee7fb'},
            'constructor': {'path': 'scripts/ablation/construct_multicorpus_raw_jg_consensus.py',
                            'sha256': '59936b928dcf942470db92debb1f2a8e41823ae1c9082803d2fdd8dc50202caa'},
            'downloaded_manifest': {'path': 'downloaded-model-hashes.json',
                                    'sha256': 'f449733393443970d912ef159b858e360390413012f872d7874ac2d84470593c'},
            'merged_manifest': {'path': 'merged-model-hashes.json',
                                'sha256': '78ae3e5840d17702cb193597e5c4873fe9506e6089fbbcc020f87cedeabfdee3'},
            'probe_manifest': {'path': 'oracle-probe-manifest.json',
                               'sha256': '1b9c53033b0e4ebf04f26b598ff42e0e2924812ab1c1c96872f7bd9da33089ba'},
            'oracle_manifest': {'path': 'oracle-adl-manifest.json',
                                'sha256': 'affe93a1139cdaa221646fb3b9c8ec5f64b28129c42b22eae185aca5d3932b90'},
            'merge_script': {'path': 'scripts/merge_lora.py',
                             'sha256': '5b75058b7a5f1ecc2b3ce237ad06e2292fecd372fff9138fdb30188de95d6030'},
            'freeze_probe_script': {'path': 'scripts/freeze_oracle_probe.py',
                                    'sha256': 'f19ef67c4e184c8ce7c1acf9cdd3617e1b4a4f6cf806e9032dfda6f2871d391b'},
            'oracle_script': {'path': 'scripts/compute_oracle_adl.py',
                              'sha256': 'd9bf1fb1e9581beb81708ece7ffdeed87ec821e0b0c66a1341ea5deb443c23ac'}},
 'candidate_artifact': '/root/model-diff-scratch/artifacts/attempt014_multicorpus_raw_jg_consensus/responses.pt',
 'oracle_artifact': '/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt',
 'output_path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/evaluation.json',
 'undefined_mean_policy': 'null_if_any_required_position_cosine_is_null',
 'oracle_sign': 'ft_mean - base_mean',
 'candidate_sign': 'stored_positive_response_no_sign_change',
 'timestamps': False,
 'host_metadata': False,
 'gpu_metadata': False}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def require_sha256(value: Any, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Malformed SHA-256 for {description}")
    return value

def require_file(path: Path, description: str) -> None:
    try:
        is_file = path.is_file()
    except OSError as exc:
        raise ValueError(f"Could not access {description} {path}: {exc}") from exc
    if not is_file:
        raise ValueError(f"Missing {description}: {path}")

def load_json_object(path: Path, description: str) -> dict[str, Any]:
    require_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {description}: {path}")
    return value

def sha256_raw_float32_tensor(tensor: Any, torch_module: Any) -> str:
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError("Cannot hash a non-tensor activation")
    if tensor.dtype != torch_module.float32 or tensor.device.type != "cpu":
        raise ValueError("Activation hash requires a CPU torch.float32 tensor")
    canonical = tensor.detach().contiguous().numpy().astype("<f4", copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()

def validate_activation_tensor(
    tensor: Any,
    description: str,
    expected_hash: Any,
    torch_module: Any,
) -> Any:
    if not isinstance(tensor, torch_module.Tensor):
        raise ValueError(f"{description} is not a tensor")
    if tensor.dtype != torch_module.float32 or tensor.device.type != "cpu":
        raise ValueError(f"{description} must be a CPU torch.float32 tensor")
    if tuple(tensor.shape) != EXPECTED_SHAPE:
        raise ValueError(
            f"{description} shape mismatch: expected {list(EXPECTED_SHAPE)}, "
            f"found {list(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{description} must be contiguous")
    if not bool(torch_module.isfinite(tensor).all().item()):
        raise ValueError(f"{description} contains non-finite values")
    expected = require_sha256(expected_hash, f"{description} raw tensor")
    if sha256_raw_float32_tensor(tensor, torch_module) != expected:
        raise ValueError(f"{description} raw tensor SHA-256 mismatch")
    return tensor

def load_torch_artifact(path: Path, description: str, torch_module: Any) -> Any:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=True)
    except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Could not load {description} {path}: {exc}") from exc

def load_and_validate_oracle(
    path: Path, context: dict[str, Any], torch_module: Any
) -> dict[str, Any]:
    artifact = load_torch_artifact(path, "oracle artifact", torch_module)
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError("Malformed oracle artifact")
    manifest = context["oracle"]["oracle_manifest"]
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Oracle artifact is missing metadata")
    for key in (
        "relative_layer",
        "layer_index",
        "num_layers",
        "sample_count",
        "sequence_length",
        "hidden_size",
        "batch_size",
        "model_dtype",
        "accumulator_dtype",
        "stored_dtype",
    ):
        if metadata.get(key) != manifest.get(key):
            raise ValueError(f"Oracle artifact metadata mismatch: {key}")
    raw = manifest["raw_tensors_sha256"]
    for name in ORACLE_VECTOR_TYPES:
        validate_activation_tensor(
            artifact.get(name), f"Oracle {name}", raw[name], torch_module
        )
    return artifact

def require_file_hash(path, expected):
    if sha256_file(path) != require_sha256(expected, str(path)):
        raise ValueError(f'Frozen file SHA256 mismatch: {path}')


def validate_oracle_provenance(paths, hashes, attempt):
    """Attempt 008's geometry-relevant provenance links; no model/lens loading."""
    downloaded = load_json_object(paths['downloaded_manifest'], 'downloaded-model manifest')
    merged = load_json_object(paths['merged_manifest'], 'merged-model manifest')
    probe = load_json_object(paths['probe_manifest'], 'oracle-probe manifest')
    oracle = load_json_object(paths['oracle_manifest'], 'oracle ADL manifest')
    if any(m.get('hash_algorithm') != 'sha256' for m in (downloaded, merged, oracle)):
        raise ValueError('Oracle provenance must specify SHA256')
    def identity(entry):
        return {key: entry[key] for key in ('repo_id','revision')}
    base = identity(downloaded['models']['base'])
    fine_tuned = identity(downloaded['models']['adapter'])
    if (merged.get('downloaded_model_hashes_sha256') != hashes['downloaded_manifest'] or
        merged.get('base') != base or merged.get('adapter') != fine_tuned or
        merged.get('merge_script_sha256') != hashes['merge_script'] or
        merged.get('merge_metadata') != {'dtype':'float32','safe_merge':True,'tokenizer_source':'base'}):
        raise ValueError('Merged-model oracle provenance mismatch')
    if (probe.get('downloaded_model_hashes_sha256') != hashes['downloaded_manifest'] or
        probe.get('freeze_oracle_probe_script_sha256') != hashes['freeze_probe_script']):
        raise ValueError('Oracle probe provenance mismatch')
    for key, expected in {
        'downloaded_model_hashes_sha256':hashes['downloaded_manifest'],
        'merged_model_hashes_sha256':hashes['merged_manifest'],
        'oracle_probe_manifest_sha256':hashes['probe_manifest'],
        'compute_oracle_adl_script_sha256':hashes['oracle_script'],
    }.items():
        if oracle.get(key) != expected: raise ValueError(f'Oracle manifest provenance mismatch: {key}')
    if oracle.get('base') != base or oracle.get('fine_tuned') != fine_tuned:
        raise ValueError('Oracle model identity mismatch')
    expected_probe={'serialized_sha256':probe.get('fineweb_tokens_sha256'),'raw_tensor_sha256':probe.get('raw_tensor_sha256')}
    if oracle.get('probe') != expected_probe or expected_probe != {key:attempt['probe'][key] for key in expected_probe}:
        raise ValueError('Oracle probe content linkage mismatch')
    expected_method={'relative_layer':.5,'layer_index':13,'num_layers':28,'sample_count':10000,
        'sequence_length':128,'hidden_size':2048,'batch_size':32,'model_dtype':'float32',
        'accumulator_dtype':'float64','stored_dtype':'float32'}
    for key,value in expected_method.items():
        if oracle.get(key) != value: raise ValueError(f'Oracle method mismatch: {key}')
    raw=oracle.get('raw_tensors_sha256')
    if not isinstance(raw,dict) or set(raw)!=set(ORACLE_VECTOR_TYPES):raise ValueError('Malformed oracle raw hashes')
    for name in ORACLE_VECTOR_TYPES:require_sha256(raw[name],name)
    require_sha256(oracle.get('oracle_adl_sha256'),'oracle artifact')
    if merged['files'] != attempt['canonical_checkpoint_files']:
        raise ValueError('Oracle merged checkpoint inventory linkage mismatch')
    return oracle


def validate_construction(paths, hashes, spec):
    attempt=load_json_object(paths['attempt_spec'],'Attempt014 spec')
    manifest=load_json_object(paths['construction_manifest'],'construction manifest')
    if attempt['attempt_id']!=spec['attempt_id'] or manifest['attempt_id']!=spec['attempt_id']:
        raise ValueError('Attempt identity mismatch')
    if attempt.get('evaluation_plan')!=spec['evaluation_plan']:
        raise ValueError('Frozen evaluation plan mismatch')
    if (manifest['spec_sha256']!=hashes['attempt_spec'] or
        manifest['constructor_script_sha256']!=hashes['constructor']):
        raise ValueError('Construction manifest source/spec linkage mismatch')
    if (attempt['consensus']!=spec['consensus_definition'] or
        manifest['consensus_definition']!=spec['consensus_definition']):
        raise ValueError('Frozen equal-weight consensus definition mismatch')
    for key in ('probe','generic_loss','jvp','readout'):
        if manifest[key]!=attempt[key]:raise ValueError(f'Construction semantics mismatch: {key}')
    if manifest['source_checkpoint']!=attempt['canonical_checkpoint_files']:
        raise ValueError('Construction checkpoint inventory linkage mismatch')
    if manifest['corpus_order']!=['fineweb','wikitext103_raw','tinystories']:
        raise ValueError('Corpus order mismatch')
    expected=['merged_mean',*spec['evaluation_plan']['candidate_tensors']]
    if attempt['outputs']['tensors']!=expected or set(manifest['artifact']['raw_tensors_sha256'])!=set(expected):
        raise ValueError('Candidate tensor inventory mismatch')
    return attempt,manifest


def load_candidates(path, manifest, spec):
    require_file_hash(path,manifest['artifact']['serialized_sha256'])
    artifact=load_torch_artifact(path,'construction artifact',torch)
    names=['merged_mean',*spec['evaluation_plan']['candidate_tensors']]
    if not isinstance(artifact,dict) or set(artifact)!=set(names):raise ValueError('Malformed/missing candidate tensors')
    for name in names:
        validate_activation_tensor(artifact[name],name,manifest['artifact']['raw_tensors_sha256'][name],torch)
    return artifact


def position_cosines(candidate, oracle):
    # No sign changes or cross-position pooling; retain undefined positions.
    result=[]
    for p in range(128):
        x=candidate[p].double();y=oracle[p].double()
        nx=torch.linalg.vector_norm(x);ny=torch.linalg.vector_norm(y)
        result.append(None if float(nx)==0 or float(ny)==0 else float(torch.dot(x,y)/(nx*ny)))
    return result


def summarize_candidates(artifact, oracle, spec):
    reports=[]
    def mean(values):
        return None if any(v is None for v in values) else math.fsum(values)/len(values)
    for name in spec['evaluation_plan']['candidate_tensors']:
        values=position_cosines(artifact[name],oracle['difference'])
        reports.append({'candidate':name,'position_0_cosine':values[0],
            'positions_1_4_mean_cosine':mean(values[1:5]),'positions_1_127_mean_cosine':mean(values[1:128]),
            'positions':[{'position':p,'cosine_similarity':value} for p,value in enumerate(values)]})
    by_name={row['candidate']:row for row in reports}
    def comparison(key):
        consensus=by_name['consensus_response'][key];fineweb=by_name['response_fineweb'][key]
        return {'consensus':consensus,'fineweb':fineweb,
                'delta_consensus_minus_fineweb':None if consensus is None or fineweb is None else consensus-fineweb}
    return {'candidates':reports,'primary':comparison('positions_1_4_mean_cosine'),
            'secondary':comparison('positions_1_127_mean_cosine')}


def evaluate(spec_path, output_path=None, input_paths=None, candidate_path=None, oracle_path=None):
    spec=load_json_object(spec_path,'evaluation spec')
    if spec!=FROZEN_SPEC:raise ValueError('Evaluation spec differs from frozen plan')
    resolve=lambda value:Path(value) if Path(value).is_absolute() else PROJECT/value
    paths={key:resolve(record['path']) for key,record in spec['inputs'].items()} if input_paths is None else input_paths
    if set(paths)!=set(spec['inputs']):raise ValueError('Provenance input inventory mismatch')
    output=resolve(spec['output_path']) if output_path is None else output_path
    candidate_path=resolve(spec['candidate_artifact']) if candidate_path is None else candidate_path
    oracle_path=resolve(spec['oracle_artifact']) if oracle_path is None else oracle_path
    if output.exists() or output.is_symlink():raise ValueError('Evaluation output already exists')
    source=Path(__file__).resolve()
    if output.resolve() in {p.resolve() for p in [spec_path,source,candidate_path,oracle_path,*paths.values()]}:
        raise ValueError('Output overlaps an input')
    spec_hash=sha256_file(spec_path);source_hash=sha256_file(source)
    if load_json_object(spec_path,'evaluation spec')!=spec:raise ValueError('Evaluation spec changed during load')
    hashes={name:record['sha256'] for name,record in spec['inputs'].items()}
    for name,digest in hashes.items():require_file_hash(paths[name],digest)
    attempt,manifest=validate_construction(paths,hashes,spec)
    oracle_manifest=validate_oracle_provenance(paths,hashes,attempt)
    artifact=load_candidates(candidate_path,manifest,spec)
    require_file_hash(oracle_path,oracle_manifest['oracle_adl_sha256'])
    oracle=load_and_validate_oracle(oracle_path,{'oracle':{'oracle_manifest':oracle_manifest}},torch)
    # Recheck read/hash races before computing any geometry.
    def recheck():
        for name,digest in hashes.items():require_file_hash(paths[name],digest)
        require_file_hash(candidate_path,manifest['artifact']['serialized_sha256'])
        require_file_hash(oracle_path,oracle_manifest['oracle_adl_sha256'])
        require_file_hash(spec_path,spec_hash);require_file_hash(source,source_hash)
    recheck()
    result={'format_version':1,'attempt_id':spec['attempt_id'],'evaluation_plan':spec['evaluation_plan'],
        'provenance':{'evaluation_spec_sha256':spec_hash,'evaluator_script_sha256':source_hash,
                      **{key+'_sha256':value for key,value in hashes.items()},
                      'construction_artifact_sha256':manifest['artifact']['serialized_sha256'],
                      'construction_raw_tensors_sha256':manifest['artifact']['raw_tensors_sha256'],
                      'oracle_artifact_sha256':oracle_manifest['oracle_adl_sha256'],
                      'oracle_raw_tensors_sha256':oracle_manifest['raw_tensors_sha256']},
        **summarize_candidates(artifact,oracle,spec)}
    recheck()
    with output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-spec',type=Path,default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'evaluation_spec.json')
    args=parser.parse_args();evaluate(args.evaluation_spec)


if __name__=='__main__':main()
