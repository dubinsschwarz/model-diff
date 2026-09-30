#!/usr/bin/env python3
"""Separate post-freeze mechanistic evaluation of the fixed Attempt018 solution."""
import argparse
import importlib.util
import json
import math
from contextlib import ExitStack
from pathlib import Path
import torch
from safetensors import safe_open
PROJECT=Path(__file__).resolve().parents[2]

def helper(filename):
    path=Path(__file__).with_name(filename)
    spec=importlib.util.spec_from_file_location('evaluation018_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

support=helper('attempt018_hvp_support.py')
activation=helper('construct_generic_gradient_linear_response.py')
oracle_support=helper('evaluate_attempt014.py')
consistency=helper('evaluate_attempt012.py')


def require_hash(path,digest):
    if support.sha256_file(path)!=support.require_sha256(digest,str(path)):
        raise ValueError('Frozen evaluation input SHA256 mismatch: '+str(path))


def validate_solution(manifest_path,solution_path,spec,manifest_sha256,solution_sha256):
    # Explicit pre-run pins are mandatory until the real constructor has been frozen.
    require_hash(manifest_path,manifest_sha256);require_hash(solution_path,solution_sha256)
    manifest=support.load_json_object(manifest_path,'construction manifest')
    if (manifest['attempt_id']!=spec['attempt_id'] or manifest['spec_sha256']!=spec['construction_spec_sha256']
        or manifest['constructor_script_sha256']!=spec['constructor_sha256'] or manifest['solver']!=spec['solver']
        or manifest['iteration_count']!=4 or len(manifest['iterations'])!=4
        or [r['iteration'] for r in manifest['iterations']]!=[1,2,3,4]
        or manifest['coordinates']!=spec['coordinates'] or manifest['final_checkpoint_files']!=spec['canonical_checkpoint_files']):
        raise ValueError('Frozen k=4 construction linkage mismatch')
    record=support.vector_record(solution_path,spec['coordinates'])
    if record!=manifest['solution'] or record['serialized_sha256']!=solution_sha256:
        raise ValueError('Solution raw/inventory/norm mismatch')
    return manifest,record


class CheckpointReader:
    def __init__(self,directory,files):self.directory,self.files=directory,files;self.stack=ExitStack();self.index={}
    def __enter__(self):
        try:
            for row in self.files:
                if row['path'].endswith('.safetensors'):
                    handle=self.stack.enter_context(safe_open(str(self.directory/row['path']),framework='pt',device='cpu'))
                    for key in handle.keys():
                        if key in self.index:raise ValueError('Duplicate checkpoint coordinate')
                        self.index[key]=handle
            return self
        except BaseException:self.stack.close();raise
    def get(self,name):return self.index[name].get_tensor(name).float()
    def __exit__(self,*args):self.stack.close()


def delta_matrix(model,name,base):
    current=model.get_parameter(name).detach().to(device='cpu',dtype=torch.float64)
    old=base.get(name)
    if current.shape!=old.shape or not torch.isfinite(old).all():raise ValueError('Displacement coordinate mismatch')
    return current-old.double()


def parameter_metrics(model,base,solution,coordinates):
    def scope(predicate):
        names=[row['name'] for row in coordinates if predicate(support.block_number(row['name']))]
        record=support.metric_record(lambda:((solution.get_tensor(n),delta_matrix(model,n,base)) for n in names))
        return {'cosine':record['cosine'],'solution_norm':record['h_norm'],'displacement_norm':record['d_norm'],
                **{k:record[k] for k in ('norm_ratio','unscaled_relative_residual','optimal_scalar','relative_residual_after_optimal_rescaling')}}
    return {'full':scope(lambda b:True),'early_0_13':scope(lambda b:b<14),'later_14_27':scope(lambda b:b>=14),
            'blocks':[{'block':b,**scope(lambda i:i==b)} for b in range(28)]}


def early_tangent(model,solution):
    eligible=activation.discover_eligible_linear_weights(model,torch)
    tangent={}
    for name,module in eligible:
        key=name+'.weight';value=solution.get_tensor(key)
        if value.dtype!=torch.float32 or value.shape!=module.weight.shape or not torch.isfinite(value).all():raise ValueError('Invalid early solution coordinate')
        tangent[key]=value.contiguous().to(module.weight.device)
    return tangent


def functional_geometry(response,oracle):
    rows=[]
    for p in range(response.shape[0]):
        m=support.metric_record(lambda:iter([(response[p],oracle[p])]))
        rows.append({'position':p,'cosine':m['cosine'],'response_norm':m['h_norm'],'oracle_difference_norm':m['d_norm'],
                     'norm_ratio':m['norm_ratio'],'optimal_scalar':m['optimal_scalar'],
                     'relative_residual_after_optimal_rescaling':m['relative_residual_after_optimal_rescaling']})
    def summary(positions):
        values=[rows[p]['cosine'] for p in positions]
        valid=all(v is not None for v in values)
        return {'positions':positions,'mean':math.fsum(values)/len(values) if valid else None,
                'min':min(values) if valid else None,'max':max(values) if valid else None}
    return {'positions':rows,'position_0':rows[0],'individual_positions_1_4':rows[1:5],
            'primary_positions_1_4':summary(list(range(1,5))),
            'secondary_positions_1_127':summary(list(range(1,128)))}


def evaluate(args):
    if args.output_path.exists() or args.output_path.is_symlink():raise ValueError('Evaluation output already exists')
    spec=support.load_json_object(args.spec_path,'evaluation spec')
    if spec!=FROZEN_SPEC:raise ValueError('Frozen evaluation spec mismatch')
    def resolve(value):
        p=Path(value);return p if p.is_absolute() else PROJECT/p
    paths={k:resolve(v) for k,v in spec['paths'].items()}
    hashes={str(args.spec_path):support.sha256_file(args.spec_path),str(Path(__file__).resolve()):support.sha256_file(Path(__file__).resolve())}
    require_hash(paths['construction_spec'],spec['construction_spec_sha256'])
    hashes[str(paths['construction_spec'])]=spec['construction_spec_sha256']
    for name,digest in spec['sources'].items():
        require_hash(PROJECT/name,digest)
        if LOADED_SOURCES[name]!=digest:raise ValueError('Loaded evaluator source changed')
        hashes[str(PROJECT/name)]=digest
    manifest,record=validate_solution(paths['construction_manifest'],paths['solution'],spec,args.construction_manifest_sha256,args.solution_sha256)
    hashes[str(paths['construction_manifest'])]=args.construction_manifest_sha256;hashes[str(paths['solution'])]=args.solution_sha256
    final_files=support.checkpoint_file_records(paths['final']);base_files=support.checkpoint_file_records(paths['base'])
    if final_files!=spec['canonical_checkpoint_files'] or base_files!=spec['base']['files']:raise ValueError('Base/final inventory mismatch')
    oracle_paths={k:resolve(v['path']) for k,v in spec['oracle_provenance'].items()}
    oracle_hashes={k:v['sha256'] for k,v in spec['oracle_provenance'].items()}
    for key,path in oracle_paths.items():require_hash(path,oracle_hashes[key]);hashes[str(path)]=oracle_hashes[key]
    oracle_manifest=oracle_support.validate_oracle_provenance(oracle_paths,oracle_hashes,spec)
    require_hash(paths['oracle_artifact'],oracle_manifest['oracle_adl_sha256'])
    hashes[str(paths['oracle_artifact'])]=oracle_manifest['oracle_adl_sha256']
    oracle=oracle_support.load_and_validate_oracle(paths['oracle_artifact'],{'oracle':{'oracle_manifest':oracle_manifest}},torch)
    probe=activation.load_probe(paths['probe'],spec['probe'],torch);hashes[str(paths['probe'])]=spec['probe']['serialized_sha256']
    if args.output_path.resolve() in {Path(p).resolve() for p in hashes} or any(args.output_path.resolve().is_relative_to(paths[k].resolve()) for k in ('base','final')):
        raise ValueError('Evaluation output overlaps input')
    model=activation.load_local_model(paths['final'],args.device,torch);before=activation.model_state_hashes(model,torch)
    with CheckpointReader(paths['base'],base_files) as base,safe_open(str(paths['solution']),framework='pt',device='cpu') as solution:
        parameter=parameter_metrics(model,base,solution,spec['coordinates'])
        tangent=early_tangent(model,solution)
    # Unconditional functional evaluation: no branch on any parameter metric.
    try:merged,response,checks=activation.compute_probe_response(model,probe,tangent,spec,torch)
    finally:tangent.clear();model.zero_grad(set_to_none=True)
    activation.verify_model_unchanged(model,before,torch)
    merged=merged.float().contiguous();response=response.float().contiguous()
    gate=consistency.merged_mean_consistency(merged,oracle['ft_mean'],spec)
    geometry=functional_geometry(response,oracle['difference'])
    support.verify_unchanged(paths['final'],final_files,{Path(p):v for p,v in hashes.items()})
    support.verify_unchanged(paths['base'],base_files,{})
    output={'format_version':1,'attempt_id':spec['attempt_id'],'construction_manifest_sha256':args.construction_manifest_sha256,
            'solution_sha256':args.solution_sha256,'input_source_hashes':hashes,'fixed_iteration_count':4,
            'parameter_geometry':parameter,'functional_geometry':geometry,'merged_mean_consistency':gate,'readout_checks':checks,
            'functional_sign':'positive_J1_x4_early','oracle_sign':'ft_mean_minus_base_mean',
            'raw_response_sha256':activation.sha256_raw_float32_tensor(response,torch),'interpretation':spec['interpretation']}
    args.output_path.parent.mkdir(parents=True,exist_ok=True);support.write_manifest(args.output_path,output)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--construction-manifest-sha256',required=True)
    parser.add_argument('--solution-sha256',required=True)
    parser.add_argument('--spec-path',type=Path,default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'evaluation_spec.json')
    parser.add_argument('--output-path',type=Path,default=PROJECT/FROZEN_SPEC['output'])
    parser.add_argument('--device',default='cuda');return parser.parse_args(argv)


FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '018_known_cleaned_rhs_endpoint_hessian_minres_k4_prefix0_27',
 'model': {'model_type': 'qwen3',
           'num_hidden_layers': 28,
           'source_dtype': 'float32',
           'local_files_only': True,
           'eval_mode': True,
           'use_cache': False},
 'interpretation': 'Direct fixed-k reverse diagnostic. Parameter results never gate functional evaluation. '
                   'No candidate selection.',
 'construction_spec_sha256': '8eb3667c9ffb9b81b7bac4a7479718ab23661bdfbeb6fcd35c8418c33673ca77',
 'constructor_sha256': '4cb9cf76089efe955ce1971b9fe5299bd593313968543a1d81ad3a42f494258d',
 'required_cli_pins': ['construction_manifest_sha256', 'solution_sha256'],
 'paths': {'construction_spec': 'experiments/attempts/018_known_cleaned_rhs_endpoint_hessian_minres_k4_prefix0_27/spec.json',
           'construction_manifest': 'experiments/attempts/018_known_cleaned_rhs_endpoint_hessian_minres_k4_prefix0_27/construction-manifest.json',
           'solution': '/root/model-diff-scratch/artifacts/attempt018_known_cleaned_rhs_minres_k4/solution.safetensors',
           'base': '/root/model-diff-scratch/models/base',
           'final': '/root/model-diff-scratch/models/merged',
           'probe': '/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt',
           'oracle_artifact': '/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt'},
 'sources': {'scripts/ablation/construct_known_cleaned_rhs_minres.py': '4cb9cf76089efe955ce1971b9fe5299bd593313968543a1d81ad3a42f494258d',
             'scripts/ablation/attempt018_hvp_support.py': 'c1f5f0cc1f18f2021023981c542acd41ff3c97906b6414af9b3933a60b72a09d',
             'scripts/ablation/smoke_test_generic_hessian_curvature_probe.py': 'c23f22b27b41ee5c0610457ef389f3eeb3264d9b410c66e6ecfa796a417ebf12',
             'scripts/ablation/construct_generic_gradient_linear_response.py': '5e80d37ad8379963b8cf1ba46e99d6c98e04514aae19eb637f3242ba8eaa4840',
             'scripts/ablation/evaluate_attempt014.py': '1193860390504d614522ae746f903d7304eb97d9b2b8d362fabcd302a4a77afd',
             'scripts/ablation/evaluate_attempt012.py': '82ecdbb0691f583b5bedee931f95a1bf3356ac5279db023dfc169789b896b364'},
 'base': {'repo_id': 'Qwen/Qwen3-1.7B',
          'revision': '0060bc56d46589041c1048efd1a397421b1142b5',
          'files': [{'path': '.gitattributes',
                     'size_bytes': 1570,
                     'sha256': '34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930'},
                    {'path': 'README.md',
                     'size_bytes': 13963,
                     'sha256': '257e52c419dac2258852643f18af6c974f21f8c6c1b6f371b6cca6201cf29091'},
                    {'path': 'config.json',
                     'size_bytes': 726,
                     'sha256': '1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197'},
                    {'path': 'generation_config.json',
                     'size_bytes': 239,
                     'sha256': '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2'},
                    {'path': 'merges.txt',
                     'size_bytes': 1671853,
                     'sha256': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'},
                    {'path': 'model-00001-of-00002.safetensors',
                     'size_bytes': 3441185608,
                     'sha256': '169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5'},
                    {'path': 'model-00002-of-00002.safetensors',
                     'size_bytes': 622329984,
                     'sha256': '912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9'},
                    {'path': 'model.safetensors.index.json',
                     'size_bytes': 25605,
                     'sha256': '0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d'},
                    {'path': 'tokenizer.json',
                     'size_bytes': 11422654,
                     'sha256': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'},
                    {'path': 'tokenizer_config.json',
                     'size_bytes': 9732,
                     'sha256': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'},
                    {'path': 'vocab.json',
                     'size_bytes': 2776833,
                     'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}]},
 'canonical_checkpoint_files': [{'path': 'chat_template.jinja',
                                 'size_bytes': 4168,
                                 'sha256': 'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'},
                                {'path': 'config.json',
                                 'size_bytes': 1416,
                                 'sha256': '3ef26f3c99bbc0bb0ef65b729429f716fb2760a23b71ffd2eee39646f703bd0f'},
                                {'path': 'generation_config.json',
                                 'size_bytes': 214,
                                 'sha256': '893b0dccf83626cdfdc498e5b2a255a935b4c3c693350aed18f7003c1a3f3de9'},
                                {'path': 'model.safetensors',
                                 'size_bytes': 6882335328,
                                 'sha256': 'f6989dd3445cd339041ab654a6da463e0a80b2d52dd2c01f67f683b1c30ea10f'},
                                {'path': 'tokenizer.json',
                                 'size_bytes': 11422650,
                                 'sha256': 'be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506'},
                                {'path': 'tokenizer_config.json',
                                 'size_bytes': 692,
                                 'sha256': '1cc816812993bff176eb4f7495433b736f06fba9b6e7b05cac7b4a1780650c95'}],
 'coordinates': [{'name': 'model.layers.0.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.0.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.0.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.0.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.0.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.0.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.0.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.1.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.1.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.1.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.1.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.1.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.1.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.1.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.10.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.10.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.10.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.10.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.10.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.10.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.10.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.11.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.11.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.11.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.11.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.11.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.11.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.11.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.12.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.12.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.12.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.12.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.12.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.12.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.12.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.13.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.13.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.13.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.13.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.13.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.13.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.13.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.14.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.14.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.14.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.14.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.14.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.14.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.14.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.15.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.15.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.15.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.15.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.15.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.15.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.15.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.16.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.16.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.16.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.16.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.16.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.16.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.16.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.17.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.17.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.17.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.17.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.17.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.17.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.17.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.18.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.18.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.18.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.18.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.18.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.18.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.18.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.19.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.19.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.19.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.19.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.19.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.19.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.19.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.2.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.2.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.2.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.2.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.2.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.2.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.2.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.20.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.20.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.20.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.20.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.20.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.20.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.20.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.21.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.21.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.21.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.21.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.21.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.21.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.21.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.22.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.22.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.22.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.22.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.22.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.22.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.22.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.23.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.23.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.23.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.23.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.23.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.23.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.23.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.24.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.24.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.24.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.24.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.24.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.24.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.24.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.25.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.25.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.25.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.25.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.25.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.25.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.25.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.26.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.26.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.26.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.26.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.26.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.26.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.26.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.27.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.27.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.27.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.27.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.27.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.27.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.27.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.3.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.3.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.3.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.3.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.3.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.3.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.3.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.4.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.4.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.4.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.4.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.4.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.4.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.4.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.5.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.5.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.5.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.5.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.5.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.5.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.5.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.6.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.6.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.6.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.6.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.6.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.6.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.6.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.7.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.7.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.7.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.7.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.7.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.7.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.7.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.8.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.8.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.8.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.8.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.8.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.8.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.8.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.9.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.9.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.9.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.9.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.9.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.9.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.9.self_attn.v_proj.weight', 'shape': [1024, 2048]}],
 'solver': {'method': 'symmetric_Paige_Saunders_MINRES_short_recurrence',
            'iteration_count': 4,
            'initial_solution': 'zero',
            'initial_rotation': {'cs': -1, 'sn': 0},
            'preconditioner': 'identity',
            'shift': 0,
            'damping': 0,
            'adaptive_stopping': False,
            'spectral_filter': False,
            'exact_breakdown_before_iteration_4': 'fail_closed',
            'recurrence_vector_dtype': 'CPU_contiguous_FP32',
            'scalar_reductions': 'matrixwise_CPU_FP64_math_fsum',
            'matrix_update_arithmetic': 'CPU_FP64_then_one_FP32_cast',
            'hvp_direction': 'FP32(CPU_FP64(r_current)/beta)_same_for_all_512_batches',
            'long_lived_vectors': 5,
            'store_basis': False,
            'store_intermediate_iterates': False,
            'fallback': False},
 'probe': {'sample_count': 10000,
           'sequence_length': 128,
           'batch_size': 32,
           'dtype': 'torch.int64',
           'serialized_sha256': '3d799d19abf4a2d6dc92b2bd164d81d83f6451cd3545f5ae5d8c04a8b299735b',
           'raw_tensor_sha256': '73f56300fdd06bc8fafcfa5750fee7a586a5c5e6071a608667ffc526bbb90ac8'},
 'readout': {'block_index': 13,
             'hidden_state_index': 14,
             'hidden_size': 2048,
             'all_token_positions': True,
             'semantics': 'hidden_states[14] is block 13 output before block 14; verify with forward hook'},
 'jvp': {'implementation': 'torch.func.jvp_with_partial_torch.func.functional_call',
         'module': 'model.model',
         'output_hidden_states': True,
         'strict_forward_ad': True,
         'fallback': False,
         'change_attention_backend': False,
         'reverse_mode_recording': False,
         'response_sign': 'positive_J_delta',
         'accumulation_dtype': 'cpu_float64',
         'stored_dtype': 'float32',
         'primal_comparison_rtol': 1e-05,
         'primal_comparison_atol': 1e-06},
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
 'merged_mean_consistency': {'minimum_cosine_similarity': 0.999999999,
                             'maximum_relative_rms_difference': 1e-05,
                             'maximum_absolute_difference': 0.01},
 'parameter_scopes': ['full', 'early_0_13', 'later_14_27', 'all_28_blocks'],
 'functional_summaries': {'position_0': 'separate',
                          'primary_positions': [1, 2, 3, 4],
                          'secondary_positions': [1,
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
                          'null_policy': 'mean_null_if_any_required_cosine_null'},
 'output': 'experiments/attempts/018_known_cleaned_rhs_endpoint_hessian_minres_k4_prefix0_27/evaluation.json',
 'timestamps': False,
 'host_metadata': False,
 'gpu_metadata': False}
LOADED_SOURCES={p:support.sha256_file(PROJECT/p) for p in FROZEN_SPEC['sources']}

if __name__=='__main__':evaluate(parse_args())
