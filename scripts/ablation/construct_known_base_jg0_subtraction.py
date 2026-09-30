#!/usr/bin/env python3
"""Known-base mechanistic Jg0 subtraction; never a blind recovery construction."""
import argparse
import hashlib
import importlib.util
import json
import math
from contextlib import ExitStack
from pathlib import Path
import torch
PROJECT = Path(__file__).resolve().parents[2]
HELPER_PATH = Path(__file__).with_name('construct_multicorpus_raw_jg_consensus.py')
_loaded_hash = hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()
_mod = importlib.util.spec_from_file_location('attempt015_validated014', HELPER_PATH)
a = importlib.util.module_from_spec(_mod)
_mod.loader.exec_module(a)
FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '015_known_base_jg0_subtraction_multicorpus_prefix0_13',
 'purpose': 'post_oracle_known_base_mechanistic_diagnostic_not_blind_recovery',
 'information_policy': 'known_base_final_frozen_014_corpora_probe_spec_manifest_responses_and_source_only; '
                       'no_blind_candidate_feedback',
 'defaults': {'base_directory': '/root/model-diff-scratch/models/base',
              'final_directory': '/root/model-diff-scratch/models/merged',
              'probe_path': '/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt',
              'raw_artifact_path': '/root/model-diff-scratch/artifacts/attempt014_multicorpus_raw_jg_consensus/responses.pt',
              'temporary_directory': '/root/model-diff-scratch/tmp/attempt015',
              'artifact_path': '/root/model-diff-scratch/artifacts/attempt015_known_base_jg0_subtraction/responses.pt',
              'manifest_path': 'experiments/attempts/015_known_base_jg0_subtraction_multicorpus_prefix0_13/construction-manifest.json'},
 'frozen_attempt014': {'spec': {'path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/spec.json',
                                'sha256': '556536ed4bb5259b29f4bb01cbf96bf846778bc21898a0a9ca3e61b895bc86ce'},
                       'manifest': {'path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/construction-manifest.json',
                                    'sha256': '6148770fe9cd29bd5f6050379fd4c01a9f3ad21501019856ea44812f6e4ee7fb'},
                       'constructor': {'path': 'scripts/ablation/construct_multicorpus_raw_jg_consensus.py',
                                       'sha256': '59936b928dcf942470db92debb1f2a8e41823ae1c9082803d2fdd8dc50202caa'},
                       'freezer': {'path': 'scripts/ablation/freeze_multicorpus_consensus_corpus.py',
                                   'sha256': 'ab55a6a812f31e76afc41af4ea2aec0132657211b1162955ffdb7983090d7a63'}},
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
 'model': {'model_type': 'qwen3',
           'num_hidden_layers': 28,
           'source_dtype': 'float32',
           'local_files_only': True,
           'eval_mode': True,
           'use_cache': False},
 'eligible_tensors': {'transformer_block_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                      'recursive_module_type': 'torch.nn.Linear',
                      'parameter': 'weight',
                      'expected_matrix_count': 98,
                      'module_ordering': 'lexicographic_by_module_name',
                      'freeze_every_other_parameter': True},
 'generic_loss': {'type': 'causal_next_token_cross_entropy',
                  'sample_count': 4096,
                  'sequence_length': 128,
                  'predictions_per_sample': 127,
                  'total_prediction_tokens': 520192,
                  'batch_size': 8,
                  'number_of_batches': 512,
                  'label_alignment': 'logits_positions_0_through_126_predict_tokens_1_through_127',
                  'reduction': 'sum_token_cross_entropy_divided_by_total_prediction_tokens',
                  'masking': 'causal_model_mask_only',
                  'mixed_precision': False,
                  'dropout': False,
                  'optimizer': False,
                  'weight_decay': False,
                  'gradient_clipping': False},
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
 'consensus': {'definition': 'arithmetic_mean_of_three_per_position_unit_positive_Jg_responses',
               'normalization_dtype': 'cpu_float64',
               'inputs': 'stored_equivalent_float32_responses',
               'zero_or_nonfinite_norm': 'fail',
               'weights': [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
               'concentration_tolerance': 1e-12},
 'corpus_order': ['fineweb', 'wikitext103_raw', 'tinystories'],
 'computation': {'gradient': 'full_historical_base_g0_mean_CE_exact014',
                 'alpha': 'exact_recorded_positive_Attempt014_alpha_no_renormalization',
                 'baseline': 'positive_J1_float32(alpha_i * g0_i)',
                 'tangent_arithmetic': 'matrix_wise_CPU_float64_one_FP32_cast',
                 'residual': 'float32(CPU_float64(raw_i) - CPU_float64(stored_baseline_i))',
                 'consensus_inputs': 'stored_equivalent_float32_responses',
                 'linearity_caveat': 'independent_FP32_tangent_and_response_rounding_not_an_exact_floating_point_identity'},
 'primal_consistency': {'minimum_cosine': 0.999999999,
                        'maximum_relative_rms': 1e-05,
                        'maximum_absolute_error': 0.01,
                        'scope': 'each_position',
                        'zero_policy': 'both_zero_pass_exactly_one_zero_fail'},
 'outputs': {'tensors': ['baseline_response_fineweb',
                         'baseline_response_wikitext103_raw',
                         'baseline_response_tinystories',
                         'residual_response_fineweb',
                         'residual_response_wikitext103_raw',
                         'residual_response_tinystories',
                         'baseline_consensus_response',
                         'residual_consensus_response'],
             'shape': [128, 2048],
             'dtype': 'contiguous_cpu_float32',
             'overwrite': False,
             'timestamps': False,
             'host_metadata': False,
             'gpu_metadata': False},
 'temporary': {'files': ['g0_fineweb.safetensors',
                         'g0_wikitext103_raw.safetensors',
                         'g0_tinystories.safetensors'],
               'failure': 'retain_no_resume',
               'cleanup': 'only_after_artifact_and_manifest_publication'},
 'workload': {'gradient_batches': 1536, 'jvp_batches': 939},
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
                                'not_for_blind_candidate_feedback']}}

def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else PROJECT/path


def require_hash(path,digest):
    if a.sha256_file(path)!=a.require_sha256(digest,str(path)):raise ValueError(f'Frozen hash mismatch: {path}')


def load_spec(path):
    spec=a.load_json_object(path,'Attempt015 spec')
    if spec!=FROZEN_SPEC:raise ValueError('Attempt015 frozen spec mismatch')
    return spec


def load_raw(spec, raw_path):
    files={key:resolve(record['path']) for key,record in spec['frozen_attempt014'].items()}
    for key,path in files.items():require_hash(path,spec['frozen_attempt014'][key]['sha256'])
    if _loaded_hash!=spec['frozen_attempt014']['constructor']['sha256']:
        raise ValueError('Loaded helper differs from frozen constructor')
    previous=a.load_json_object(files['spec'],'Attempt014 spec');manifest=a.load_json_object(files['manifest'],'Attempt014 manifest')
    if (previous!=a.FROZEN_SPEC or manifest['attempt_id']!=previous['attempt_id'] or
        manifest['spec_sha256']!=spec['frozen_attempt014']['spec']['sha256'] or
        manifest['constructor_script_sha256']!=spec['frozen_attempt014']['constructor']['sha256']):
        raise ValueError('Attempt014 spec/manifest/source linkage mismatch')
    for key in ('probe','generic_loss','jvp','readout'):
        if previous[key]!=spec[key] or manifest[key]!=spec[key]:raise ValueError(f'Attempt014 semantics mismatch: {key}')
    if (manifest['consensus_definition']!=spec['consensus'] or manifest['source_checkpoint']!=spec['canonical_checkpoint_files'] or
        manifest['corpus_order']!=spec['corpus_order'] or len(manifest['corpora'])!=3):raise ValueError('Attempt014 definition mismatch')
    for name,record in zip(spec['corpus_order'],manifest['corpora']):
        if record['name']!=name or not math.isfinite(record['alpha']) or record['alpha']<=0:raise ValueError('Invalid frozen corpus alpha/order')
    require_hash(raw_path,manifest['artifact']['serialized_sha256'])
    raw=torch.load(raw_path,map_location='cpu',weights_only=True)
    expected=['merged_mean',*[f'response_{n}' for n in spec['corpus_order']],'consensus_response']
    if not isinstance(raw,dict) or set(raw)!=set(expected) or set(manifest['artifact']['raw_tensors_sha256'])!=set(expected):raise ValueError('Raw artifact tensor inventory mismatch')
    for name in expected:
        validate_tensor(raw[name],spec)
        if a.sha256_raw_float32_tensor(raw[name],torch)!=manifest['artifact']['raw_tensors_sha256'][name]:raise ValueError('Attempt014 raw tensor hash mismatch')
    check_raw_geometry(raw,manifest,spec)
    return previous,manifest,raw,{path:spec['frozen_attempt014'][key]['sha256'] for key,path in files.items()}


def validate_tensor(value,spec):
    if (not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.float32 or
        list(value.shape)!=spec['outputs']['shape'] or not value.is_contiguous() or not bool(torch.isfinite(value).all())):
        raise ValueError('Invalid response shape/dtype/contiguity/finiteness')


def check_raw_geometry(raw,manifest,spec):
    consensus,geometry=a.response_consensus([raw[f'response_{n}'] for n in spec['corpus_order']],torch)
    if not torch.equal(consensus,raw['consensus_response']) or geometry!=manifest['geometry']:
        raise ValueError('Attempt014 raw geometry/consensus reproduction mismatch')
    return geometry


def validate_coordinates(eligible,expected):
    actual=[{'name':f'{n}.weight','shape':list(m.weight.shape)} for n,m in eligible]
    wanted=[{'name':r['name'],'shape':r['shape']} for r in expected]
    if actual!=wanted:raise ValueError('Parameter names/order/shapes mismatch')
    return actual


def save_base_gradient(model,eligible,tokens,alpha,settings,path):
    from safetensors.torch import save_file
    if path.exists():raise ValueError('Temporary gradient already exists')
    if not math.isfinite(alpha) or alpha<=0:raise ValueError('Alpha must be copied positive and finite')
    before=a.model_state_hashes(model,torch)
    try:
        loss=a.accumulate_mean_generic_gradient(model,tokens,eligible,settings,torch)
        squares=[];cpu={}
        for name,module in eligible:
            value=module.weight.grad.detach()
            squares.append(float(value.to(device='cpu',dtype=torch.float64).square().sum()))
            cpu[f'{name}.weight']=value.to('cpu').contiguous()
        norm=math.sqrt(math.fsum(squares))
        if not math.isfinite(norm):raise ValueError('Nonfinite base gradient')
        save_file(cpu,str(path))
        a.verify_model_unchanged(model,before,torch)
        return {'base_generic_loss':loss,'base_gradient_norm':norm,'alpha':alpha,'scaled_gradient_norm_cpu_float64':alpha*norm}
    finally:model.zero_grad(set_to_none=True)


def load_scaled_tangent(path,eligible,alpha):
    from safetensors import safe_open
    if not math.isfinite(alpha) or alpha<=0:raise ValueError('Alpha must be positive and finite')
    tangent={};squares=[]
    with safe_open(str(path),framework='pt',device='cpu') as file:
        if set(file.keys())!={f'{n}.weight' for n,_ in eligible}:raise ValueError('Gradient coordinate mismatch')
        for name,module in eligible:
            value=file.get_tensor(f'{name}.weight')
            if value.dtype!=torch.float32 or value.shape!=module.weight.shape or not bool(torch.isfinite(value).all()):raise ValueError('Invalid gradient tensor')
            scaled=(value.double()*alpha).float().contiguous()
            if not bool(torch.isfinite(scaled).all()):raise ValueError('Nonfinite scaled tangent')
            squares.append(float(scaled.double().square().sum()))
            tangent[f'{name}.weight']=scaled.to(module.weight.device)
    return tangent,math.sqrt(math.fsum(squares))


def primal_consistency(primal,reference,spec):
    if primal.device.type!='cpu' or primal.dtype!=torch.float64 or primal.shape!=reference.shape or not bool(torch.isfinite(primal).all()):raise ValueError('Invalid primal mean')
    settings=spec['primal_consistency'];rows=[]
    for p in range(primal.shape[0]):
        x,y=primal[p],reference[p].double();nx=float(x.norm());ny=float(y.norm());error=x-y
        cosine=1. if nx==ny==0 else 0. if nx==0 or ny==0 else float(torch.dot(x/nx,y/ny))
        relative=0. if nx==ny==0 else float(error.norm())/ny if ny else None
        maximum=float(error.abs().max())
        if (cosine<settings['minimum_cosine'] or relative is None or relative>settings['maximum_relative_rms'] or maximum>settings['maximum_absolute_error']):
            raise ValueError(f'Frozen merged mean mismatch at position {p}')
        rows.append({'position':p,'cosine':min(1.,max(-1.,cosine)),'relative_rms':relative,'maximum_absolute_error':maximum})
    return rows


def cos(x,y):
    nx=float(x.norm());ny=float(y.norm())
    return float(torch.dot(x,y)/(nx*ny)) if nx and ny else None


def residual_and_audit(raw,baseline):
    exact=raw.double()-baseline.double();stored=exact.float().contiguous()
    if not bool(torch.isfinite(stored).all()):raise ValueError('Nonfinite residual')
    error=stored.double()-exact
    rows=[{'position':p,'float64_residual_norm':float(exact[p].norm()),'fp32_rounding_rms':float(error[p].square().mean().sqrt()),
           'fp32_rounding_max_abs':float(error[p].abs().max())} for p in range(raw.shape[0])]
    return stored,rows


def build_outputs(raw,baseline,spec):
    result={};rounding={}
    for name in spec['corpus_order']:
        value=baseline[name];validate_tensor(value,spec)
        result[f'baseline_response_{name}']=value
        result[f'residual_response_{name}'],rounding[name]=residual_and_audit(raw[f'response_{name}'],value)
    b,bd=a.response_consensus([baseline[n] for n in spec['corpus_order']],torch)
    r,rd=a.response_consensus([result[f'residual_response_{n}'] for n in spec['corpus_order']],torch)
    result['baseline_consensus_response']=b;result['residual_consensus_response']=r
    rows=[]
    for p in range(spec['outputs']['shape'][0]):
        corpus={}
        for name in spec['corpus_order']:
            x=raw[f'response_{name}'][p].double();y=baseline[name][p].double();z=result[f'residual_response_{name}'][p].double()
            corpus[name]={'raw_norm':float(x.norm()),'baseline_norm':float(y.norm()),'residual_norm':float(z.norm()),
                'raw_baseline_cosine':cos(x,y),'raw_residual_cosine':cos(x,z),'baseline_residual_cosine':cos(y,z)}
        original=raw['consensus_response'][p].double();bc=b[p].double();rc=r[p].double()
        rows.append({'position':p,'corpora':corpus,'baseline_pairwise_cosine_matrix':bd['positions'][p]['pairwise_cosine_matrix'],
            'residual_pairwise_cosine_matrix':rd['positions'][p]['pairwise_cosine_matrix'],
            'baseline_consensus_concentration':bd['positions'][p]['consensus_concentration'],
            'residual_consensus_concentration':rd['positions'][p]['consensus_concentration'],
            'raw_residual_consensus_cosine':cos(original,rc),'raw_baseline_consensus_cosine':cos(original,bc),'baseline_residual_consensus_cosine':cos(bc,rc)})
    def summarize(values):
        first=values[0]
        if isinstance(first,dict):return {k:summarize([v[k] for v in values]) for k in first if k!='position'}
        if isinstance(first,list):return [summarize([v[i] for v in values]) for i in range(len(first))]
        if any(v is None for v in values):return {'mean':None,'min':None,'max':None}
        return {'mean':math.fsum(values)/len(values),'min':min(values),'max':max(values)}
    return {name:result[name] for name in spec['outputs']['tensors']},{'rounding_audit':rounding,'positions':rows,
        'position_0':rows[0],'positions_1_4':summarize(rows[1:5]),'positions_1_127':summarize(rows[1:])}


def require_absent(output,manifest,temporary):
    a.require_output_absent(output,manifest)
    if temporary.is_symlink() or (temporary.exists() and (not temporary.is_dir() or any(temporary.iterdir()))):raise ValueError('Stale temporary state; no resume')


def construct(args):
    require_absent(args.artifact_path,args.manifest_path,args.temporary_directory)
    spec=load_spec(args.spec_path)
    previous,prior,raw,hashes=load_raw(spec,args.raw_artifact_path)
    hashes.update({args.spec_path:a.sha256_file(args.spec_path),args.raw_artifact_path:prior['artifact']['serialized_sha256'],
                   args.probe_path:a.sha256_file(args.probe_path),Path(__file__).resolve():a.sha256_file(Path(__file__).resolve())})
    corpus_paths=[resolve(r[k]) for r in previous['corpora'] for k in ('spec_path','manifest_path','tokens_path')]
    for out in (args.artifact_path,args.manifest_path,args.temporary_directory):
        if (any(out.resolve().is_relative_to(p.resolve()) for p in (args.base_directory,args.final_directory)) or
            any(out.resolve()==p.resolve() or p.resolve().is_relative_to(out.resolve()) for p in [*hashes,*corpus_paths])):raise ValueError('Outputs overlap frozen inputs')
    if any(out.resolve().is_relative_to(args.temporary_directory.resolve()) for out in (args.artifact_path,args.manifest_path)):
        raise ValueError('Scientific outputs must be outside temporary storage')
    final_files=a.checkpoint_file_records(args.final_directory);base_files=a.checkpoint_file_records(args.base_directory)
    if final_files!=spec['canonical_checkpoint_files'] or base_files!=spec['base']['files']:raise ValueError('Base/final checkpoint inventory/hash mismatch')
    for record,expected in zip(previous['corpora'],prior['corpora']):
        tokens,checked,provenance=a.validate_frozen_corpus(record,args.final_directory,torch)
        if any(provenance[k]!=expected[k] for k in provenance):raise ValueError('Frozen corpus linkage to Attempt014 mismatch')
        hashes.update(checked);del tokens
    probe=a.load_probe(args.probe_path,spec['probe'],torch)
    if load_spec(args.spec_path)!=spec:raise ValueError('Spec changed during loading')
    args.temporary_directory.mkdir(parents=True,exist_ok=True)
    with (args.temporary_directory/'state.json').open('x') as f:json.dump({'attempt_id':spec['attempt_id'],'resume_permitted':False},f)
    base=a.load_local_model(args.base_directory,args.device,torch)
    eligible=a.discover_eligible_linear_weights(base,torch);coordinates=validate_coordinates(eligible,prior['corpora'][0]['matrices'])
    for record in prior['corpora']:validate_coordinates(eligible,record['matrices'])
    a.freeze_other_parameters(base,eligible);before=a.model_state_hashes(base,torch);records=[]
    for corpus,expected in zip(previous['corpora'],prior['corpora']):
        print(f"Base gradient: {corpus['name']}",flush=True)
        tokens,checked,_=a.validate_frozen_corpus(corpus,args.final_directory,torch)
        if any(hashes[p]!=v for p,v in checked.items()):raise ValueError('Corpus changed before gradient')
        path=args.temporary_directory/f"g0_{corpus['name']}.safetensors"
        records.append({'name':corpus['name'],**save_base_gradient(base,eligible,tokens,expected['alpha'],spec['generic_loss'],path)})
        del tokens
    a.verify_model_unchanged(base,before,torch)
    del eligible,base
    final=a.load_local_model(args.final_directory,args.device,torch)
    eligible=a.discover_eligible_linear_weights(final,torch);validate_coordinates(eligible,coordinates)
    for parameter in final.parameters():parameter.requires_grad_(False)
    before=a.model_state_hashes(final,torch);baseline={}
    for record in records:
        print(f"Final-checkpoint JVP: {record['name']}",flush=True)
        tangent,realized=load_scaled_tangent(args.temporary_directory/f"g0_{record['name']}.safetensors",eligible,record['alpha'])
        try:primal,response,checks=a.compute_probe_response(final,probe,tangent,spec,torch)
        finally:tangent.clear();final.zero_grad(set_to_none=True)
        record.update(realized_fp32_tangent_norm=realized,readout_checks=checks,primal_consistency=primal_consistency(primal,raw['merged_mean'],spec))
        baseline[record['name']]=response.float().contiguous()
        a.verify_model_unchanged(final,before,torch)
    artifact,geometry=build_outputs(raw,baseline,spec)
    def recheck():
        a.verify_unchanged(args.final_directory,final_files,hashes)
        a.verify_unchanged(args.base_directory,base_files,{})
    recheck();a.require_output_absent(args.artifact_path,args.manifest_path)
    args.artifact_path.parent.mkdir(parents=True,exist_ok=True)
    with args.artifact_path.open('xb') as f:torch.save(artifact,f)
    manifest={'format_version':1,'attempt_id':spec['attempt_id'],'purpose':spec['purpose'],'information_policy':spec['information_policy'],
        'spec_sha256':hashes[args.spec_path],'constructor_script_sha256':hashes[Path(__file__).resolve()],
        'frozen_attempt014':spec['frozen_attempt014'],'attempt014_artifact_sha256':prior['artifact']['serialized_sha256'],
        'corpus_provenance':[{k:r[k] for k in ('name','spec_sha256','manifest_sha256','serialized_sha256','raw_tensor_sha256')} for r in prior['corpora']],
        'base_checkpoint':spec['base'],'final_checkpoint_files':final_files,'probe':spec['probe'],'generic_loss':spec['generic_loss'],
        'computation':spec['computation'],'consensus':spec['consensus'],'jvp':spec['jvp'],'readout':spec['readout'],
        'coordinates':coordinates,'corpora':records,'geometry':geometry,'raw_geometry_reproduced':True,'evaluation_plan':spec['evaluation_plan'],
        'workload':spec['workload'],'artifact':{'serialized_sha256':a.sha256_file(args.artifact_path),
            'raw_tensors_sha256':{k:a.sha256_raw_float32_tensor(v,torch) for k,v in artifact.items()}}}
    recheck();a.write_manifest(args.manifest_path,manifest)
    expected={'state.json',*spec['temporary']['files']}
    if {p.name for p in args.temporary_directory.iterdir()}!=expected:raise ValueError('Unexpected temporary contents; retained')
    for name in sorted(expected):(args.temporary_directory/name).unlink()
    args.temporary_directory.rmdir()


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path',type=Path,default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'spec.json')
    for key,value in FROZEN_SPEC['defaults'].items():parser.add_argument('--'+key.replace('_','-'),type=Path,default=resolve(value))
    parser.add_argument('--device',default='cuda');return parser.parse_args(argv)


if __name__=='__main__':construct(parse_args())
