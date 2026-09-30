#!/usr/bin/env python3
"""Post-freeze known-base diagnostic evaluation; never candidate selection."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import torch
PROJECT=Path(__file__).resolve().parents[2]
def _load(name,filename):
    path=Path(__file__).with_name(filename)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,digest
c,LOADED_CONSTRUCTOR_SHA=_load('eval015_construction','construct_known_base_jg0_subtraction.py')
e,LOADED_ORACLE_HELPER_SHA=_load('eval015_oracle','evaluate_attempt014.py')
FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '015_known_base_jg0_subtraction_multicorpus_prefix0_13',
 'purpose': 'post_oracle_known_base_mechanistic_diagnostic_not_blind_recovery',
 'evaluation_plan': {'oracle_vector': 'difference',
                     'oracle_sign': 'ft_mean - base_mean',
                     'candidate_tensors': ['response_fineweb',
                                           'response_wikitext103_raw',
                                           'response_tinystories',
                                           'consensus_response',
                                           'baseline_response_fineweb',
                                           'baseline_response_wikitext103_raw',
                                           'baseline_response_tinystories',
                                           'baseline_consensus_response',
                                           'residual_response_fineweb',
                                           'residual_response_wikitext103_raw',
                                           'residual_response_tinystories',
                                           'residual_consensus_response'],
                     'cosine_dtype': 'cpu_float64',
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
                     'summary_positions': {'primary': [1, 2, 3, 4],
                                           'secondary': [1,
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
                                                         127]},
                     'position_0': 'report_separately',
                     'primary': 'residual_consensus_minus_raw_consensus_positions_1_4',
                     'secondary': 'residual_consensus_minus_raw_consensus_positions_1_127',
                     'per_corpus_deltas': 'residual_minus_raw_for_primary_and_secondary',
                     'undefined_mean': 'null_if_any_required_cosine_null',
                     'policy': ['no_ranking',
                                'no_sign_changes',
                                'no_reweighting',
                                'no_subset_selection',
                                'not_for_blind_candidate_feedback']},
 'sources': {'attempt_spec': {'path': 'experiments/attempts/015_known_base_jg0_subtraction_multicorpus_prefix0_13/spec.json',
                              'sha256': '18ec59a64faa22effb17640b54932ed85e4eb85f44f3eeb0d283c6de5dea4c15'},
             'constructor': {'path': 'scripts/ablation/construct_known_base_jg0_subtraction.py',
                             'sha256': '66593968b8bf5637ff5aadfa9455604634079f3de0868448e92fd32aef1e1045'},
             'oracle_helper': {'path': 'scripts/ablation/evaluate_attempt014.py',
                               'sha256': '1193860390504d614522ae746f903d7304eb97d9b2b8d362fabcd302a4a77afd'}},
 'oracle_provenance': {'downloaded_manifest': {'path': 'downloaded-model-hashes.json',
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
 'construction_manifest_path': 'experiments/attempts/015_known_base_jg0_subtraction_multicorpus_prefix0_13/construction-manifest.json',
 'construction_manifest_hash': 'required_explicit_frozen_SHA256_argument_after_construction',
 'artifact_path': '/root/model-diff-scratch/artifacts/attempt015_known_base_jg0_subtraction/responses.pt',
 'oracle_artifact': '/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt',
 'output_path': 'experiments/attempts/015_known_base_jg0_subtraction_multicorpus_prefix0_13/evaluation.json',
 'timestamps': False,
 'host_metadata': False,
 'gpu_metadata': False}

def summaries(candidates,oracle,spec):
    reports=[]
    def mean(values):return None if any(v is None for v in values) else math.fsum(values)/len(values)
    for name in spec['evaluation_plan']['candidate_tensors']:
        values=e.position_cosines(candidates[name],oracle['difference'])
        reports.append({'candidate':name,'position_0_cosine':values[0],'positions_1_4_mean_cosine':mean(values[1:5]),
            'positions_1_127_mean_cosine':mean(values[1:]),'positions':[{'position':p,'cosine_similarity':v} for p,v in enumerate(values)]})
    by_name={r['candidate']:r for r in reports}
    def comparison(raw,residual,key):
        x=by_name[raw][key];y=by_name[residual][key]
        return {'raw':x,'residual':y,'delta_residual_minus_raw':None if x is None or y is None else y-x}
    result={'candidates':reports}
    for label,key in [('primary','positions_1_4_mean_cosine'),('secondary','positions_1_127_mean_cosine')]:
        result[label]=comparison('consensus_response','residual_consensus_response',key)
        result[label]['per_corpus']={name:comparison(f'response_{name}',f'residual_response_{name}',key) for name in c.FROZEN_SPEC['corpus_order']}
    return result


def evaluate(args):
    if args.output.exists() or args.output.is_symlink():raise ValueError('Evaluation output already exists')
    spec=c.a.load_json_object(args.evaluation_spec,'evaluation spec')
    if spec!=FROZEN_SPEC:raise ValueError('Frozen evaluation spec mismatch')
    c.a.require_sha256(args.construction_manifest_sha256,'frozen construction manifest')
    paths={k:c.resolve(v['path']) for k,v in spec['sources'].items()}
    hashes={paths[k]:v['sha256'] for k,v in spec['sources'].items()}
    hashes.update({args.evaluation_spec:c.a.sha256_file(args.evaluation_spec),Path(__file__).resolve():c.a.sha256_file(Path(__file__).resolve()),
                   args.construction_manifest:args.construction_manifest_sha256})
    for path,digest in hashes.items():c.require_hash(path,digest)
    if (LOADED_CONSTRUCTOR_SHA!=spec['sources']['constructor']['sha256'] or LOADED_ORACLE_HELPER_SHA!=spec['sources']['oracle_helper']['sha256']):raise ValueError('Loaded source mismatch')
    construction_spec=c.load_spec(paths['attempt_spec'])
    if construction_spec['evaluation_plan']!=spec['evaluation_plan']:raise ValueError('Evaluation plan linkage mismatch')
    previous,prior,raw,linked=c.load_raw(construction_spec,c.resolve(construction_spec['defaults']['raw_artifact_path']))
    hashes.update(linked)
    raw_path=c.resolve(construction_spec['defaults']['raw_artifact_path']);hashes[raw_path]=prior['artifact']['serialized_sha256']
    manifest=c.a.load_json_object(args.construction_manifest,'Attempt015 manifest')
    if (manifest['attempt_id']!=spec['attempt_id'] or manifest['purpose']!=spec['purpose'] or
        manifest['spec_sha256']!=spec['sources']['attempt_spec']['sha256'] or manifest['constructor_script_sha256']!=LOADED_CONSTRUCTOR_SHA or
        manifest['frozen_attempt014']!=construction_spec['frozen_attempt014'] or manifest['attempt014_artifact_sha256']!=prior['artifact']['serialized_sha256']):raise ValueError('Construction provenance mismatch')
    for key in ('evaluation_plan','computation','consensus','probe','generic_loss','jvp','readout'):
        if manifest[key]!=construction_spec[key]:raise ValueError(f'Construction semantics mismatch: {key}')
    if manifest['base_checkpoint']!=construction_spec['base'] or manifest['final_checkpoint_files']!=construction_spec['canonical_checkpoint_files']:raise ValueError('Checkpoint linkage mismatch')
    if len(manifest['corpora'])!=3 or any(x['name']!=y['name'] or x['alpha']!=y['alpha'] for x,y in zip(manifest['corpora'],prior['corpora'])):raise ValueError('Frozen alpha mismatch')
    artifact_path=c.resolve(spec['artifact_path']);hashes[artifact_path]=manifest['artifact']['serialized_sha256'];c.require_hash(artifact_path,hashes[artifact_path])
    artifact=torch.load(artifact_path,map_location='cpu',weights_only=True)
    expected=construction_spec['outputs']['tensors']
    if not isinstance(artifact,dict) or set(artifact)!=set(expected) or set(manifest['artifact']['raw_tensors_sha256'])!=set(expected):raise ValueError('Candidate inventory mismatch')
    for name in expected:
        c.validate_tensor(artifact[name],construction_spec)
        if c.a.sha256_raw_float32_tensor(artifact[name],torch)!=manifest['artifact']['raw_tensors_sha256'][name]:raise ValueError('Candidate raw hash mismatch')
    opaths={k:c.resolve(v['path']) for k,v in spec['oracle_provenance'].items()};ohashes={k:v['sha256'] for k,v in spec['oracle_provenance'].items()}
    for key,path in opaths.items():c.require_hash(path,ohashes[key]);hashes[path]=ohashes[key]
    om=e.validate_oracle_provenance(opaths,ohashes,previous)
    if om['base']!={k:construction_spec['base'][k] for k in ('repo_id','revision')}:raise ValueError('Historical base/oracle identity mismatch')
    oracle_path=c.resolve(spec['oracle_artifact']);hashes[oracle_path]=om['oracle_adl_sha256'];c.require_hash(oracle_path,hashes[oracle_path])
    oracle=e.load_and_validate_oracle(oracle_path,{'oracle':{'oracle_manifest':om}},torch)
    if args.output.resolve() in {p.resolve() for p in hashes}:raise ValueError('Output overlaps input')
    def recheck():
        for path,digest in hashes.items():c.require_hash(path,digest)
    recheck()
    result={'attempt_id':spec['attempt_id'],'purpose':spec['purpose'],'evaluation_plan':spec['evaluation_plan'],
        'provenance':{'evaluation_spec_sha256':hashes[args.evaluation_spec],'evaluator_sha256':hashes[Path(__file__).resolve()],
            'construction_manifest_sha256':args.construction_manifest_sha256,'construction_artifact_sha256':hashes[artifact_path],
            'attempt014_artifact_sha256':hashes[raw_path],'sources':spec['sources'],'oracle_provenance':spec['oracle_provenance'],'oracle_artifact_sha256':hashes[oracle_path]},
        **summaries({**raw,**artifact},oracle,spec)}
    recheck();c.a.write_manifest(args.output,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-spec',type=Path,default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'evaluation_spec.json')
    parser.add_argument('--construction-manifest',type=Path,default=c.resolve(FROZEN_SPEC['construction_manifest_path']))
    parser.add_argument('--construction-manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,default=c.resolve(FROZEN_SPEC['output_path']))
    evaluate(parser.parse_args())


if __name__=='__main__':main()
