#!/usr/bin/env python3
"""Known-base endpoint displacement JVP ceiling; not a blind recovery method."""
import argparse
import hashlib
import importlib.util
import json
import math
from contextlib import ExitStack
from pathlib import Path
import torch
PROJECT=Path(__file__).resolve().parents[2]
HELPER_PATH=Path(__file__).with_name('construct_generic_gradient_linear_response.py')
LOADED_HELPER_SHA256=hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()
_helper_spec=importlib.util.spec_from_file_location('ceiling016_validated008',HELPER_PATH)
a=importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(a)
FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '016_known_base_exact_displacement_jvp_ceiling_prefix0_13',
 'purpose': 'post_oracle_known_base_mechanistic_ceiling_not_blind_recovery_not_for_blind_construction_choices',
 'information_policy': 'base_final_checkpoints_frozen_probe_and_spec_source_only',
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
 'helper_source': {'path': 'scripts/ablation/construct_generic_gradient_linear_response.py',
                   'sha256': '5e80d37ad8379963b8cf1ba46e99d6c98e04514aae19eb637f3242ba8eaa4840'},
 'displacement': {'definition': 'Delta_S = W1_S - W0_S',
                  'subtraction': 'matrix_wise_CPU_float64_then_one_contiguous_FP32_cast',
                  'response': 'positive_J1_S_Delta_S',
                  'normalization': False,
                  'scale': None,
                  'saved_base_weights': 'selected_98_only_CPU_FP32_memory',
                  'nonselected_audit': 'all_parameter_hashes_then_read_changed_base_tensors_matrixwise_from_frozen_safetensors',
                  'aliases': 'deduplicate_parameters_and_preserve_alias_groups; '
                             'any_upstream_alias_makes_parameter_upstream',
                  'upstream_rule': 'all_parameters_except_blocks_14_27_and_model.norm_and_lm_head; '
                                   'selected_98_handled_separately',
                  'omitted_upstream_nonzero': 'fail'},
 'exact_target': {'definition': 'final_mean - base_mean', 'arithmetic': 'CPU_float64_before_FP32_storage'},
 'metrics': {'dtype': 'cpu_float64',
             'inputs': 'stored_equivalent_FP32_linear_response_and_exact_difference',
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
             'position_0': 'separate',
             'optimal_scalar': 'dot(linear_response, exact_difference)/dot(linear_response,linear_response)',
             'relative_residual': 'norm(exact_difference - '
                                  'optimal_scalar*linear_response)/norm(exact_difference)',
             'zero_policy': 'undefined_cosine_ratio_or_projection_is_null; '
                            'summary_null_if_any_required_cosine_null',
             'no_best_position': True,
             'no_sign_flip': True,
             'optimal_scalar_is_diagnostic_only': True},
 'defaults': {'base_directory': '/root/model-diff-scratch/models/base',
              'final_directory': '/root/model-diff-scratch/models/merged',
              'probe_path': '/root/model-diff-scratch/artifacts/oracle_probe/fineweb_tokens.pt',
              'artifact_path': '/root/model-diff-scratch/artifacts/attempt016_exact_displacement_jvp_ceiling/responses.pt',
              'manifest_path': 'experiments/attempts/016_known_base_exact_displacement_jvp_ceiling_prefix0_13/construction-manifest.json'},
 'outputs': {'tensors': ['base_mean', 'final_mean', 'exact_difference', 'linear_response'],
             'shape': [128, 2048],
             'dtype': 'contiguous_CPU_float32',
             'overwrite': False,
             'timestamps': False,
             'host_metadata': False,
             'gpu_metadata': False,
             'large_temporary_files': False},
 'workload': {'base_probe_forward_batches': 313,
              'final_probe_jvp_batches': 313,
              'extra_final_hook_forward_batches': 1,
              'gradient_batches': 0,
              'hvp_batches': 0},
 'execution_semantics_audit': {'buffers': 'full_inventory_and_alias_topology; '
                                          'exact_shape_dtype_raw_hash_for_any_upstream_alias; '
                                          'downstream_only_differences_recorded',
                               'upstream_rule': 'same_conservative_causal_classification_as_parameters_including_global_rotary_buffers',
                               'config_fields': ['model_type',
                                                 'hidden_size',
                                                 'intermediate_size',
                                                 'num_hidden_layers',
                                                 'num_attention_heads',
                                                 'num_key_value_heads',
                                                 'head_dim',
                                                 'hidden_act',
                                                 'max_position_embeddings',
                                                 'rms_norm_eps',
                                                 'rope_theta',
                                                 'rope_scaling',
                                                 'attention_bias',
                                                 'attention_dropout',
                                                 'use_sliding_window',
                                                 'sliding_window',
                                                 'max_window_layers',
                                                 'layer_types',
                                                 '_attn_implementation'],
                               'config_scope': ['config', 'model.config'],
                               'config_comparison': 'presence_and_canonical_JSON_values; '
                                                    'missing_config_fails; all_listed_fields_must_match',
                               'excluded_config': 'provenance_generation_metadata_and_use_cache'}}

def resolve(value):
    path=Path(value)
    return path if path.is_absolute() else PROJECT/path


def load_spec(path):
    spec=a.load_json_object(path,'Attempt016 specification')
    if spec!=FROZEN_SPEC:raise ValueError('Frozen Attempt016 specification mismatch')
    return spec


def aliases(model):
    groups={}
    for name,value in model.named_parameters(remove_duplicate=False):groups.setdefault(id(value),[]).append(name)
    return {name:sorted(groups[id(value)]) for name,value in model.named_parameters()}


def downstream(name):
    parts=name.split('.')
    if len(parts)>3 and parts[:2]==['model','layers'] and parts[2].isdigit():return 14<=int(parts[2])<=27
    return name.startswith('model.norm.') or name.startswith('lm_head.')


def parameter_category(names,selected,later_linear):
    if any(name in selected for name in names):
        if not all(name in selected for name in names):raise ValueError('Selected parameter has an omitted alias')
        return 'selected_early_linear_weights'
    if not all(downstream(name) for name in names):return 'omitted_upstream_parameters'
    if any(name in later_linear for name in names):return 'later_block_linear_weights'
    return 'other_downstream_parameters'


def execution_snapshot(model, spec):
    """Snapshot buffers without retaining their storage; fail closed on unknown locations."""
    groups = {}
    for name, value in model.named_buffers(remove_duplicate=False):
        groups.setdefault(id(value), []).append(name)
    buffers = {}
    for name, value in model.named_buffers():
        digest = hashlib.sha256()
        flat = value.detach().contiguous().reshape(-1)
        for start in range(0, flat.numel(), 4_000_000):
            raw = flat[start:start + 4_000_000].cpu().view(torch.uint8).numpy()
            digest.update(memoryview(raw).cast('B'))
        buffers[name] = {'aliases': sorted(groups[id(value)]), 'shape': list(value.shape),
                         'dtype': str(value.dtype), 'raw_sha256': digest.hexdigest()}
    configs = {}
    for scope in spec['execution_semantics_audit']['config_scope']:
        owner = model if scope == 'config' else model.model
        config = getattr(owner, 'config', None)
        if config is None:
            raise ValueError(f'Missing forward configuration: {scope}')
        values = {}
        for field in spec['execution_semantics_audit']['config_fields']:
            present = hasattr(config, field)
            value = getattr(config, field) if present else None
            # JSON rejects unsupported objects and nonfinite numbers rather than stringifying them.
            canonical = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
            values[field] = {'present': present, 'value': canonical}
        configs[scope] = values
    return {'buffers': buffers, 'configuration': configs}


def audit_execution_semantics(base, final):
    if base['configuration'] != final['configuration']:
        raise ValueError('Forward configuration semantics differ between base and final')
    left, right = base['buffers'], final['buffers']
    if set(left) != set(right):
        raise ValueError('Base/final buffer inventories differ')
    rows = []
    for name, record in left.items():
        other = right[name]
        if record['aliases'] != other['aliases']:
            raise ValueError(f'Base/final buffer alias topology differs: {name}')
        upstream = not all(downstream(alias) for alias in record['aliases'])
        equal = record == other
        if upstream and not equal:
            raise ValueError(f'Upstream buffer differs: {name}')
        rows.append({'name': name, 'upstream': upstream, 'equal': equal,
                     'base': record, 'final': other})
    return {'buffers': rows, 'upstream_buffers_equal': True,
            'configuration': base['configuration'], 'configuration_equal': True}


def snapshot_base(model,eligible):
    selected={f'{name}.weight' for name,_ in eligible}
    alias_groups=aliases(model)
    records={}
    for name,value in model.named_parameters():
        records[name]={'shape':list(value.shape),'aliases':alias_groups[name],'sha256':a.sha256_parameter(value,torch)}
    saved={f'{name}.weight':module.weight.detach().to(device='cpu',dtype=torch.float32,copy=True).contiguous() for name,module in eligible}
    if len(saved)!=len(selected):raise ValueError('Duplicate selected coordinate')
    coordinates=[{'name':f'{name}.weight','shape':list(module.weight.shape)} for name,module in eligible]
    return saved,records,coordinates


class BaseTensorReader:
    """Read only changed nonselected W0 parameters, one tensor at a time.

    Hash comparison skips equal parameters. This avoids retaining a second
    full model or all base weights in CPU memory after the base probe phase.
    """
    def __init__(self,directory,inventory):
        self.directory,self.inventory=directory,inventory
        self.stack=ExitStack();self.files={}
    def __enter__(self):
        from safetensors import safe_open
        try:
            for record in self.inventory:
                if record['path'].endswith('.safetensors'):
                    file=self.stack.enter_context(safe_open(str(self.directory/record['path']),framework='pt',device='cpu'))
                    for key in file.keys():
                        if key in self.files:raise ValueError('Duplicate checkpoint tensor key')
                        self.files[key]=file
            return self
        except BaseException:
            self.stack.close();raise
    def __exit__(self,*args):self.stack.close()
    def get(self,name,record):
        for alias in [name,*record['aliases']]:
            if alias in self.files:
                value=self.files[alias].get_tensor(alias).to(dtype=torch.float32).contiguous()
                if list(value.shape)!=record['shape'] or a.sha256_parameter(value,torch)!=record['sha256']:
                    raise ValueError(f'Base checkpoint tensor does not reproduce loaded W0: {name}')
                return value
        raise ValueError(f'Cannot locate changed base parameter: {name}')


def audit_and_tangent(final,eligible,saved,base_records,coordinates,read_base):
    current=[{'name':f'{name}.weight','shape':list(module.weight.shape)} for name,module in eligible]
    if current!=coordinates or set(saved)!={r['name'] for r in coordinates}:raise ValueError('Base/final selected names/order/shapes mismatch')
    parameters=dict(final.named_parameters());groups=aliases(final)
    if set(parameters)!=set(base_records):raise ValueError('Base/final full parameter inventories differ')
    selected=set(saved)
    later_linear={f'model.layers.{i}.{relative}.weight' for i in range(14,28)
                  for relative,module in final.model.layers[i].named_modules() if isinstance(module,torch.nn.Linear)}
    categories={name:[] for name in ('selected_early_linear_weights','later_block_linear_weights','other_downstream_parameters','omitted_upstream_parameters')}
    tangent={};exact_squares=[];realized_squares=[];rounding_squares=[];source_squares=[];per_matrix=[];forbidden=[]
    for name,parameter in parameters.items():
        record=base_records[name]
        if list(parameter.shape)!=record['shape'] or groups[name]!=record['aliases']:raise ValueError('Base/final parameter shape or alias mismatch')
        category=parameter_category(groups[name],selected,later_linear)
        digest=a.sha256_parameter(parameter,torch)
        changed=digest!=record['sha256']
        if name in selected or changed:
            base=saved[name] if name in selected else read_base(name,record)
            if base.dtype!=torch.float32 or base.device.type!='cpu' or list(base.shape)!=record['shape'] or a.sha256_parameter(base,torch)!=record['sha256']:
                raise ValueError(f'Base parameter audit hash/shape mismatch: {name}')
            value=parameter.detach().to(device='cpu',dtype=torch.float64)
            delta=value-base.double()
            if not bool(torch.isfinite(delta).all()):raise ValueError('Nonfinite checkpoint displacement')
            square=float(delta.square().sum());nonzero=bool(delta.ne(0).any())
            if category=='omitted_upstream_parameters' and nonzero:forbidden.append(name)
            if name in selected:
                rounded=delta.float().contiguous()
                if not bool(torch.isfinite(rounded).all()):raise ValueError('Nonfinite FP32 displacement')
                realized=float(rounded.double().square().sum());error=float((rounded.double()-delta).square().sum());source=float(value.square().sum())
                exact_squares.append(square);realized_squares.append(realized);rounding_squares.append(error);source_squares.append(source)
                per_matrix.append({'name':name,'shape':record['shape'],'exact_displacement_norm':math.sqrt(square),
                    'realized_fp32_displacement_norm':math.sqrt(realized),'rounding_error_norm':math.sqrt(error)})
                tangent[name]=rounded.to(parameter.device)
        else:square=0.;nonzero=False
        categories[category].append({'name':name,'aliases':record['aliases'],'displacement_norm':math.sqrt(square),'squared_displacement_norm':square,'nonzero':nonzero})
    if forbidden:
        tangent.clear()
        raise ValueError('Omitted upstream parameter displacement: '+', '.join(forbidden))
    norm=math.sqrt(math.fsum(exact_squares));source_norm=math.sqrt(math.fsum(source_squares))
    audit={'categories':{name:{'norm':math.sqrt(math.fsum(row['squared_displacement_norm'] for row in rows)),
        'parameter_count':len(rows),'nonzero_count':sum(row['nonzero'] for row in rows),
        'nonzero_names':[row['name'] for row in rows if row['nonzero']],'parameters':rows} for name,rows in categories.items()},
        'exact_displacement_norm':norm,'realized_fp32_displacement_norm':math.sqrt(math.fsum(realized_squares)),
        'rounding_error_norm':math.sqrt(math.fsum(rounding_squares)),'source_selected_weight_norm':source_norm,
        'displacement_to_source_relative_norm':norm/source_norm if source_norm else None,
        'per_matrix':per_matrix,'omitted_upstream_zero_verified':True,
        'later_blocks_excluded_reason':'blocks_14_through_27_execute_after_block_13_output',
        'all_unique_parameters_audited':len(parameters)}
    return tangent,audit


def base_probe_mean(model,probe,spec):
    settings=spec['probe'];shape=(settings['sequence_length'],spec['readout']['hidden_size'])
    if tuple(probe.shape)!=(settings['sample_count'],settings['sequence_length']):raise ValueError('Probe shape mismatch')
    total=torch.zeros(shape,dtype=torch.float64,device='cpu');count=0
    model.eval();device=next(model.parameters()).device
    for start in range(0,settings['sample_count'],settings['batch_size']):
        tokens=probe[start:start+settings['batch_size']].to(device)
        if start==0:value=a.ordinary_hook_readout(model,tokens,spec,torch)
        else:
            with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
                output=model.model(input_ids=tokens,use_cache=False,output_hidden_states=True,return_dict=True)
                value=a.hidden_state_output(output,spec)
                a.validate_readout(value,len(tokens),spec,torch,'base readout')
        total.add_(value.detach().to(device='cpu',dtype=torch.float64).sum(dim=0));count+=len(tokens)
        del value,tokens
    if count!=settings['sample_count']:raise ValueError('Base probe sample count mismatch')
    return total/count,{'first_batch_hook_verified':True,'sample_count':count}


def build_artifact(base_mean,final_mean,response,spec):
    for value in (base_mean,final_mean,response):
        if value.dtype!=torch.float64 or value.device.type!='cpu' or list(value.shape)!=spec['outputs']['shape'] or not bool(torch.isfinite(value).all()):raise ValueError('Invalid FP64 activation mean')
    values={'base_mean':base_mean,'final_mean':final_mean,'exact_difference':final_mean-base_mean,'linear_response':response}
    result={name:value.float().contiguous() for name,value in values.items()}
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):raise ValueError('Nonfinite stored activation')
    result['metadata']={'attempt_id':spec['attempt_id'],'purpose':spec['purpose'],'sign':'positive_J1_Delta_S',
                        'exact_difference':'final_mean_minus_base_mean_before_FP32_storage','readout':spec['readout'],
                        'sample_count':spec['probe']['sample_count']}
    return result


def metrics(artifact,spec):
    rows=[]
    for p in spec['metrics']['positions']:
        x=artifact['linear_response'][p].double();y=artifact['exact_difference'][p].double()
        nx=float(x.norm());ny=float(y.norm());dot=float(torch.dot(x,y))
        coefficient=dot/float(x.square().sum()) if nx else None
        residual=float((y-coefficient*x).norm())/ny if coefficient is not None and ny else None
        rows.append({'position':p,'cosine':dot/(nx*ny) if nx and ny else None,'linear_response_norm':nx,
            'exact_difference_norm':ny,'norm_ratio':nx/ny if ny else None,
            'optimal_scalar_coefficient':coefficient,'relative_residual_after_optimal_rescaling':residual})
    def summary(positions):
        values=[rows[p]['cosine'] for p in positions];valid=all(v is not None for v in values)
        return {'positions':positions,'mean':math.fsum(values)/len(values) if valid else None,
                'min':min(values) if valid else None,'max':max(values) if valid else None}
    primary=summary(spec['metrics']['primary_positions']);secondary=summary(spec['metrics']['secondary_positions'])
    return {'positions':rows,'position_0':rows[0],'primary_mean_cosine_positions_1_4':primary['mean'],
        'secondary_mean_cosine_positions_1_127':secondary['mean'],'positions_1_4':primary,'positions_1_127':secondary}


def construct(args):
    a.require_output_absent(args.artifact_path,args.manifest_path)
    spec=load_spec(args.spec_path);paths=[args.spec_path,args.probe_path,Path(__file__).resolve(),HELPER_PATH]
    for out in (args.artifact_path,args.manifest_path):
        if out.resolve() in {p.resolve() for p in paths} or any(out.resolve().is_relative_to(p.resolve()) for p in (args.base_directory,args.final_directory)):
            raise ValueError('Output overlaps frozen input')
    hashes={p:a.sha256_file(p) for p in paths}
    if (hashes[HELPER_PATH]!=spec['helper_source']['sha256'] or LOADED_HELPER_SHA256!=hashes[HELPER_PATH] or load_spec(args.spec_path)!=spec):raise ValueError('Helper/spec hash mismatch')
    base_files=a.checkpoint_file_records(args.base_directory);final_files=a.checkpoint_file_records(args.final_directory)
    if base_files!=spec['base']['files'] or final_files!=spec['canonical_checkpoint_files']:raise ValueError('Exact base/final inventory/hash mismatch')
    probe=a.load_probe(args.probe_path,spec['probe'],torch)
    print('Base probe phase',flush=True)
    base=a.load_local_model(args.base_directory,args.device,torch)
    eligible=a.discover_eligible_linear_weights(base,torch);a.freeze_other_parameters(base,eligible)
    for parameter in base.parameters():parameter.requires_grad_(False)
    del parameter
    before=a.model_state_hashes(base,torch)
    base_execution=execution_snapshot(base,spec)
    saved,records,coordinates=snapshot_base(base,eligible)
    base_mean,base_checks=base_probe_mean(base,probe,spec)
    a.verify_model_unchanged(base,before,torch)
    if execution_snapshot(base,spec)!=base_execution:raise ValueError("Base execution semantics changed during probe")
    del eligible,base
    print('Final displacement audit and positive JVP',flush=True)
    final=a.load_local_model(args.final_directory,args.device,torch)
    eligible=a.discover_eligible_linear_weights(final,torch);a.freeze_other_parameters(final,eligible)
    for parameter in final.parameters():parameter.requires_grad_(False)
    del parameter
    before=a.model_state_hashes(final,torch)
    final_execution=execution_snapshot(final,spec)
    execution_audit=audit_execution_semantics(base_execution,final_execution)
    with BaseTensorReader(args.base_directory,base_files) as reader:
        tangent,audit=audit_and_tangent(final,eligible,saved,records,coordinates,reader.get)
    saved.clear()
    try:final_mean,response,final_checks=a.compute_probe_response(final,probe,tangent,spec,torch)
    finally:tangent.clear();final.zero_grad(set_to_none=True)
    a.verify_model_unchanged(final,before,torch)
    if execution_snapshot(final,spec)!=final_execution:raise ValueError("Final execution semantics changed during JVP")
    artifact=build_artifact(base_mean,final_mean,response,spec)
    diagnostic=metrics(artifact,spec)
    def recheck():
        a.verify_unchanged(args.base_directory,base_files,hashes)
        a.verify_unchanged(args.final_directory,final_files,{})
    recheck();a.require_output_absent(args.artifact_path,args.manifest_path)
    args.artifact_path.parent.mkdir(parents=True,exist_ok=True)
    with args.artifact_path.open('xb') as stream:torch.save(artifact,stream)
    result={'format_version':1,'attempt_id':spec['attempt_id'],'purpose':spec['purpose'],'information_policy':spec['information_policy'],
        'spec_sha256':hashes[args.spec_path],'constructor_script_sha256':hashes[Path(__file__).resolve()],
        'helper_source':spec['helper_source'],'base_checkpoint':spec['base'],'final_checkpoint_files':final_files,'probe':spec['probe'],
        'execution_semantics_audit':execution_audit,
        'coordinates':coordinates,'displacement_definition':spec['displacement'],'displacement_audit':audit,'exact_target':spec['exact_target'],
        'readout':spec['readout'],'jvp':spec['jvp'],'readout_checks':{'base':base_checks,'final':final_checks},
        'metric_definitions':spec['metrics'],'metrics':diagnostic,'workload':spec['workload'],
        'artifact':{'serialized_sha256':a.sha256_file(args.artifact_path),
                    'raw_tensors_sha256':{name:a.sha256_raw_float32_tensor(artifact[name],torch) for name in spec['outputs']['tensors']}}}
    recheck();a.write_manifest(args.manifest_path,result)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path',type=Path,default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'spec.json')
    for key,value in FROZEN_SPEC['defaults'].items():parser.add_argument('--'+key.replace('_','-'),type=Path,default=resolve(value))
    parser.add_argument('--device',default='cuda');return parser.parse_args(argv)


if __name__=='__main__':construct(parse_args())
