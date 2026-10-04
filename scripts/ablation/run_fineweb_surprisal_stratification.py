#!/usr/bin/env python3
"""Attempt126: blind FineWeb surprisal strata, then matched historical scoring."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '126_fineweb_surprisal_stratification'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '0e2457b8fbb8cd76f482599fe20878a2c32a30f288eef83c39cc59efffa367ef'
NAMES = ('low','middle','high')
RANGES = {'low':(0,1024),'middle':(1536,2560),'high':(3072,4096)}
UNUSED = ((1024,1536),(2560,3072))
RESPONSE_NAMES = tuple('response_'+name for name in NAMES)
TARGET_SHA256 = '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f'
SOURCE014 = PROJECT/'scripts/ablation/construct_multicorpus_raw_jg_consensus.py'
SOURCE014_SHA256 = '59936b928dcf942470db92debb1f2a8e41823ae1c9082803d2fdd8dc50202caa'
SOURCE122 = PROJECT/'scripts/ablation/run_loss_stabilized_generic_relaxation.py'
SOURCE122_SHA256 = '7dd7e79b84ac297c7b0f7a62a69ff5dc7d91332b95b3b180f72f50c1f6cf3a9d'


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path,expected):
    if sha256_file(path)!=expected:
        raise ValueError('Frozen SHA256 mismatch: '+str(path))
    return path


def path_of(value):
    value=Path(value)
    return value if value.is_absolute() else PROJECT/value


def import_pinned(path,expected,name):
    import importlib.util
    require_hash(path,expected)
    loader=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


a122=import_pinned(SOURCE122,SOURCE122_SHA256,'attempt126_frozen_attempt122_publish')


def load_spec():
    require_hash(SPEC_PATH,SPEC_SHA256)
    spec=json.loads(SPEC_PATH.read_text())
    rank=spec['ranking']
    probe=spec['probe']
    evaluation=spec['evaluation']
    if (spec['attempt_id']!=ATTEMPT or
            spec['frozen_attempt005']['tokens_path']!=
                '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt' or
            spec['fineweb_corpus']['population_rows']!=[0,4096] or
            [spec['strata'][name]['rank_range'] for name in NAMES]!=
                [list(RANGES[name]) for name in NAMES] or
            spec['unused_rank_intervals']!=[list(value) for value in UNUSED] or
            rank!={'definition':'per_sequence_mean_causal_next_token_NLL_at_unmodified_final_checkpoint',
                'population_rows':[0,4096],'sequence_length':128,
                'prediction_tokens_per_row':127,'batch_size':8,
                'number_of_forward_batches':512,'no_grad':True,'no_amp':True,
                'accumulation_dtype':'cpu_float64',
                'sort_key':['surprisal_ascending','original_row_index_ascending'],
                'surprisal_storage_dtype':'contiguous_cpu_float64'} or
            spec['gradient_loss']['sample_count']!=1024 or
            spec['gradient_loss']['sequence_length']!=128 or
            spec['gradient_loss']['batch_size']!=8 or
            spec['gradient_loss']['number_of_batches']!=128 or
            spec['gradient_loss']['total_prediction_tokens']!=130048 or
            spec['tangent']['direction']!='delta = +alpha * g' or
            spec['tangent']['target_relative_frobenius']!=.00125 or
            spec['tangent']['normalization']!='one_global_alpha_across_all_eligible_matrices' or
            spec['candidate']!={'tensor_order':list(RESPONSE_NAMES),
                'shape':[128,2048],'dtype':'contiguous_cpu_float32',
                'response_normalization':False,'sign_selection':False,
                'interpolation':False,'reweighting':False,'combination':False} or
            probe['full_source_rows']!=10000 or probe['selected_rows']!=[0,1024] or
            probe['selected_batch_count']!=32 or probe['batch_size']!=32 or
            spec['readout']['block_index']!=13 or spec['readout']['hidden_state_index']!=14 or
            spec['jvp']['strict_forward_ad'] is not True or
            spec['jvp']['fallback'] is not False or
            spec['jvp']['response_sign']!='positive_J_delta' or
            spec['barrier']!={'marker':'SURPRISAL_STRATA_CANDIDATES_FROZEN',
                'surprisals_ranking_rows_gradients_responses_published_and_revalidated_before_historical_access':True,
                'manifest_contains_historical_scores':False} or
            evaluation['target_raw_sha256']!=TARGET_SHA256 or
            evaluation['target_probe_rows']!=[0,1024] or
            evaluation['primary_positions']!=[1,2,3,4] or
            evaluation['secondary_positions']!=list(range(1,128)) or
            evaluation['clear_support']!={'ordered_both_metrics_nonstrict':True,
                'high_low_primary_min':.03,'high_low_secondary_min':.03} or
            evaluation['partial_support']!={'high_gt_low_both_metrics':True,
                'at_least_one_high_low_gain_min':.03} or
            evaluation['signed_positionwise_cosine'] is not True or
            evaluation['absolute_cosine'] is not False or
            evaluation['flattened_historical_cosine'] is not False or
            spec['workload']!={'ranking_forward_batches':512,
                'gradient_backward_batches':384,'jvp_probe_batches':96,
                'jvp_first_batch_hook_audits':3,'matched_target_forward_batches':64,
                'ggn_operations':0,'cg_operations':0} or
            any(spec['information_policy'][name] is not False for name in
                ('historical_base_access_before_barrier','adapter_access_before_barrier',
                 'true_delta_access_before_barrier','historical_activation_target_access_before_barrier',
                 'prior_oracle_score_access_before_barrier','clean_heldout_validation'))):
        raise ValueError('Attempt126 frozen scientific plan mismatch')
    paths={key:path_of(value) for key,value in spec['paths'].items()}
    if (len(set(paths.values()))!=3 or
            any(path.exists() or path.is_symlink() for path in paths.values()) or
            any(path in {path_of(spec['frozen_attempt005']['tokens_path']),
                         path_of(spec['probe']['path']),
                         path_of(spec['final_checkpoint']['directory'])}
                for path in paths.values())):
        raise ValueError('Attempt126 output exists or overlaps a frozen input')
    return spec,paths


def validate_blind_inputs(spec,a14,torch):
    for group,keys in (('frozen_attempt005',('corpus_spec','corpus_manifest')),
                       ('frozen_attempt100',('spec','source','construction_manifest')),
                       ('frozen_attempt113',('spec','source','construction_manifest'))):
        record=spec[group]
        for key in keys:
            require_hash(path_of(record[key+'_path']),record[key+'_sha256'])
    require_hash(SOURCE014,SOURCE014_SHA256)
    require_hash(SOURCE122,SOURCE122_SHA256)
    old100=json.loads(path_of(spec['frozen_attempt100']['spec_path']).read_text())
    old113=json.loads(path_of(spec['frozen_attempt113']['spec_path']).read_text())
    manifest100=json.loads(path_of(spec['frozen_attempt100']['construction_manifest_path']).read_text())
    manifest113=json.loads(path_of(spec['frozen_attempt113']['construction_manifest_path']).read_text())
    corpus_manifest=json.loads(path_of(spec['frozen_attempt005']['corpus_manifest_path']).read_text())
    fineweb=old113['corpora'][0]
    probe_full={key:spec['probe'][key] for key in
                ('sample_count','sequence_length','batch_size','dtype',
                 'serialized_sha256','raw_tensor_sha256')}
    if (spec['helper_sources']!={'attempt014_path':str(SOURCE014.relative_to(PROJECT)),
            'attempt014_sha256':SOURCE014_SHA256,
            'attempt122_path':str(SOURCE122.relative_to(PROJECT)),
            'attempt122_sha256':SOURCE122_SHA256} or
            old100['canonical_checkpoint_files']!=spec['final_checkpoint']['files'] or
            old113['final_checkpoint']!=spec['final_checkpoint'] or
            old100['model']!=spec['model'] or old113['model']!=spec['model'] or
            old100['eligible_tensors']!=spec['eligible_tensors'] or
            old113['eligible_tensors']!=spec['eligible_tensors'] or
            spec['eligible_tensors']['expected_matrix_count']!=98 or
            old100['readout']!=spec['readout'] or old113['readout']!=spec['readout'] or
            old100['jvp']!=spec['jvp'] or
            old100['tangent']!=spec['tangent'] or
            old100['probe']!=probe_full or
            old113['probe']['serialized_sha256']!=probe_full['serialized_sha256'] or
            old113['probe']['raw_tensor_sha256']!=probe_full['raw_tensor_sha256'] or
            old100['defaults']['probe_path']!=spec['probe']['path'] or
            old100['defaults']['merged_model_directory']!=spec['final_checkpoint']['directory'] or
            manifest100['source_checkpoint']!=spec['final_checkpoint']['files'] or
            manifest100['probe_sha256']!=probe_full['serialized_sha256'] or
            manifest113['final_checkpoint_files']!=spec['final_checkpoint']['files'] or
            any(manifest113['corpora'][0].get(key)!=value
                for key,value in fineweb.items()) or
            fineweb!= {key:spec['fineweb_corpus'][key] for key in fineweb} or
            fineweb['selected_rows']!=[0,512] or
            fineweb['tokens_path']!=spec['frozen_attempt005']['tokens_path'] or
            fineweb['spec_sha256']!=spec['frozen_attempt005']['corpus_spec_sha256'] or
            fineweb['manifest_sha256']!=spec['frozen_attempt005']['corpus_manifest_sha256'] or
            fineweb['artifact_serialized_sha256']!=spec['frozen_attempt005']['serialized_sha256'] or
            fineweb['artifact_raw_sha256']!=spec['frozen_attempt005']['raw_tensor_sha256'] or
            corpus_manifest['artifact']['serialized_sha256']!=fineweb['artifact_serialized_sha256'] or
            corpus_manifest['artifact']['raw_tensor_sha256']!=fineweb['artifact_raw_sha256']):
        raise ValueError('Frozen Attempt005/100/113 model, FineWeb, or probe provenance conflict')
    model_dir=path_of(spec['final_checkpoint']['directory'])
    if a14.checkpoint_file_records(model_dir)!=spec['final_checkpoint']['files']:
        raise ValueError('Canonical final checkpoint changed')
    tokens,token_hashes,corpus_provenance=a14.validate_frozen_corpus(fineweb,model_dir,torch)
    full_probe=a14.load_probe(path_of(spec['probe']['path']),probe_full,torch)
    if (tuple(tokens.shape)!=(4096,128) or tuple(full_probe.shape)!=(10000,128) or
            not tokens.is_contiguous() or not full_probe.is_contiguous()):
        raise ValueError('Frozen full FineWeb/probe shape mismatch')
    prefix=full_probe[:1024].contiguous()
    del full_probe
    return tokens,prefix,token_hashes,corpus_provenance,model_dir


def sequence_mean_nll(token_nll):
    """Float64 per-row sum of exactly 127 ordinary next-token NLLs."""
    value=np.asarray(token_nll)
    if (value.ndim!=2 or value.shape[1]!=127 or value.shape[0]==0 or
            not np.issubdtype(value.dtype,np.floating) or
            not np.isfinite(value).all()):
        raise ValueError('Malformed or nonfinite [batch,127] token NLL')
    return np.sum(value.astype(np.float64),axis=1,dtype=np.float64)/127.0


def score_endpoint_surprisal(model,tokens,torch):
    if (tuple(tokens.shape)!=(4096,128) or tokens.dtype!=torch.int64 or
            not tokens.is_contiguous()):
        raise ValueError('Endpoint ranking requires the full frozen 4096-row FineWeb tensor')
    model.eval()
    device=next(model.parameters()).device
    scores=np.empty(4096,dtype=np.float64)
    with torch.inference_mode(),torch.autocast(device_type=device.type,enabled=False):
        for start in range(0,4096,8):
            batch=tokens[start:start+8].to(device)
            logits=model(input_ids=batch,use_cache=False).logits
            if (logits.dtype!=torch.float32 or logits.shape[:2]!=(8,128) or
                    not bool(torch.isfinite(logits).all())):
                raise ValueError('Nonfinite/malformed endpoint logits')
            shifted=logits[:,:-1,:].contiguous().reshape(-1,logits.shape[-1])
            labels=batch[:,1:].contiguous().reshape(-1)
            token_nll=torch.nn.functional.cross_entropy(
                shifted,labels,reduction='none').reshape(8,127)
            cpu=token_nll.detach().to(device='cpu',dtype=torch.float64).numpy()
            scores[start:start+8]=sequence_mean_nll(cpu)
            del batch,logits,shifted,labels,token_nll,cpu
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite endpoint surprisal')
    return scores


def raw_float64_hash(values):
    values=np.asarray(values)
    if values.dtype!=np.float64 or not values.flags.c_contiguous or not np.isfinite(values).all():
        raise ValueError('Expected finite contiguous float64 values')
    return hashlib.sha256(values.astype('<f8',copy=False).tobytes(order='C')).hexdigest()


def raw_int64_hash(values):
    values=np.asarray(values)
    if values.dtype!=np.int64 or not values.flags.c_contiguous:
        raise ValueError('Expected contiguous int64 row indices')
    return hashlib.sha256(values.astype('<i8',copy=False).tobytes(order='C')).hexdigest()


def rank_and_stratify(surprisal):
    values=np.asarray(surprisal)
    if (values.shape!=(4096,) or values.dtype!=np.float64 or
            not values.flags.c_contiguous or not np.isfinite(values).all()):
        raise ValueError('Expected exactly 4096 finite contiguous float64 surprisals')
    rank=np.asarray(sorted(range(4096),key=lambda row:(float(values[row]),row)),
                    dtype=np.int64)
    strata={}
    selected=set()
    for name in NAMES:
        lo,hi=RANGES[name]
        rows=rank[lo:hi].copy()
        if len(rows)!=1024 or selected.intersection(map(int,rows)):
            raise ValueError('Surprisal strata overlap or have wrong size')
        selected.update(map(int,rows))
        sample=[float(values[row]) for row in rows]
        mean=math.fsum(sample)/len(sample)
        median=math.fsum(sample[511:513])/2
        std=math.sqrt(math.fsum((value-mean)**2 for value in sample)/len(sample))
        strata[name]={'rank_range':[lo,hi],'original_row_indices':rows.tolist(),
            'ordered_row_indices_raw_sha256':raw_int64_hash(rows),
            'surprisal_min':sample[0],'surprisal_max':sample[-1],
            'surprisal_mean':mean,'surprisal_median':median,
            'surprisal_population_std':std}
    if len(selected)!=3072 or sorted(rank.tolist())!=list(range(4096)):
        raise ValueError('Malformed full surprisal ordering')
    return {'surprisal_values':values.tolist(),
        'surprisal_raw_float64_sha256':raw_float64_hash(values),
        'ascending_rank_original_row_indices':rank.tolist(),
        'rank_raw_int64_sha256':raw_int64_hash(rank),
        'strata':strata,
        'unused_rank_intervals':[list(value) for value in UNUSED]}


def validate_ranking(record):
    values=np.asarray(record['surprisal_values'],dtype=np.float64)
    rebuilt=rank_and_stratify(values)
    if rebuilt!=record:
        raise ValueError('Published surprisal, rank, or selected-row hashes changed')
    return True


def response_cosine(left,right,torch):
    a122.prior.validate_response(left,torch)
    a122.prior.validate_response(right,torch)
    x,y=left.double().flatten(),right.double().flatten()
    nx,ny=torch.linalg.vector_norm(x),torch.linalg.vector_norm(y)
    return None if float(nx)==0 or float(ny)==0 else float(torch.dot(x,y)/(nx*ny))


def validate_global_scale(scale):
    """The Attempt100 one-global-alpha positive tangent rule."""
    if (set(scale)!={'aggregate_source_weight_norm','aggregate_generic_gradient_norm',
                    'target_delta_norm','alpha'} or
            any(not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0
                for value in scale.values())):
        raise ValueError('Malformed global tangent scale')
    target=.00125*scale['aggregate_source_weight_norm']
    if (scale['target_delta_norm']!=target or
            scale['alpha']!=target/scale['aggregate_generic_gradient_norm']):
        raise ValueError('Tangent alpha does not follow the Attempt100 Frobenius rule')
    return scale


def pairwise_response_cosines(responses,torch):
    if list(responses)!=list(RESPONSE_NAMES):
        raise ValueError('Exactly LOW/MIDDLE/HIGH responses required')
    return {'low_middle':response_cosine(responses['response_low'],responses['response_middle'],torch),
            'low_high':response_cosine(responses['response_low'],responses['response_high'],torch),
            'middle_high':response_cosine(responses['response_middle'],responses['response_high'],torch)}


def differences_and_classification(reports):
    if list(reports)!=list(RESPONSE_NAMES):
        raise ValueError('Only the three predeclared strata may be scored')
    primary={name:reports['response_'+name]['positions_1_4_mean_cosine'] for name in NAMES}
    secondary={name:reports['response_'+name]['positions_1_127_mean_cosine'] for name in NAMES}
    if any(value is None or not math.isfinite(value) for value in
           (*primary.values(),*secondary.values())):
        raise ValueError('Undefined signed historical score')
    deltas={}
    for left,right in (('high','low'),('high','middle'),('middle','low')):
        deltas[left+'_minus_'+right]={'primary':primary[left]-primary[right],
                                      'secondary':secondary[left]-secondary[right]}
    clear=(all(metric['high']>=metric['middle']>=metric['low']
               for metric in (primary,secondary)) and
           deltas['high_minus_low']['primary']>=.03 and
           deltas['high_minus_low']['secondary']>=.03)
    partial=(primary['high']>primary['low'] and secondary['high']>secondary['low'] and
             (deltas['high_minus_low']['primary']>=.03 or
              deltas['high_minus_low']['secondary']>=.03))
    category='clear_support' if clear else 'partial_support' if partial else 'no_support'
    return deltas,category


def pearson_three(x,y):
    if len(x)!=3 or len(y)!=3 or any(not math.isfinite(value) for value in (*x,*y)):
        raise ValueError('Three finite stratum values required')
    mx,my=math.fsum(x)/3,math.fsum(y)/3
    xx=math.fsum((value-mx)**2 for value in x)
    yy=math.fsum((value-my)**2 for value in y)
    if xx==0 or yy==0:
        return None
    return math.fsum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(xx*yy)


def descriptive_correlations(ranking,gradient_records,reports):
    if list(gradient_records)!=list(NAMES) or list(reports)!=list(RESPONSE_NAMES):
        raise ValueError('Descriptive correlations require all three fixed strata')
    outcomes={'primary':[reports['response_'+name]['positions_1_4_mean_cosine'] for name in NAMES],
              'secondary':[reports['response_'+name]['positions_1_127_mean_cosine'] for name in NAMES]}
    predictors={'mean_surprisal':[ranking['strata'][name]['surprisal_mean'] for name in NAMES],
                'gradient_norm':[gradient_records[name]['aggregate_generic_gradient_norm'] for name in NAMES]}
    return {'n_strata':3,'order':list(NAMES),'descriptive_only':True,
            'pearson':{predictor:{metric:pearson_three(values,scores)
                                  for metric,scores in outcomes.items()}
                       for predictor,values in predictors.items()},
            'predictor_values':predictors}


def validate_published(candidate_path,manifest_path,construction,torch,a14):
    if (sha256_file(candidate_path)!=construction['candidate']['serialized_sha256'] or
            json.loads(manifest_path.read_text())!=construction):
        raise ValueError('Published Attempt126 artifact/manifest changed')
    artifact=torch.load(candidate_path,map_location='cpu',weights_only=True)
    if not isinstance(artifact,dict) or list(artifact)!=list(RESPONSE_NAMES):
        raise ValueError('Published candidate inventory is not exactly LOW/MIDDLE/HIGH')
    for name in RESPONSE_NAMES:
        value=a122.prior.validate_response(artifact[name],torch)
        if a14.sha256_raw_float32_tensor(value,torch)!=construction['responses'][name]['raw_sha256']:
            raise ValueError('Published response raw SHA256 mismatch: '+name)
    validate_ranking(construction['ranking'])
    if (construction['candidate']['tensor_order']!=list(RESPONSE_NAMES) or
            construction['barrier']['manifest_contains_historical_scores'] is not False):
        raise ValueError('Published construction inventory/barrier changed')
    return artifact


def evaluate_after_barrier(spec,artifact,torch):
    frozen=spec['frozen_attempt123']
    for key in ('spec','source'):
        require_hash(path_of(frozen[key+'_path']),frozen[key+'_sha256'])
    a123=import_pinned(path_of(frozen['source_path']),frozen['source_sha256'],
                       'attempt126_frozen_attempt123')
    matched_spec=a123.load_spec()
    reference=matched_spec['matched_reference']
    privileged=import_pinned(path_of(reference['attempt103_source_path']),
                             reference['attempt103_source_sha256'],
                             'attempt126_frozen_attempt103')
    frozen122=matched_spec['frozen_attempt122']
    require_hash(path_of(frozen122['construction_manifest_path']),
                 frozen122['construction_manifest_sha256'])
    manifest122=json.loads(path_of(frozen122['construction_manifest_path']).read_text())
    old103,inventories=a123.validate_matched_reference(matched_spec,manifest122,privileged)
    if (old103['final_checkpoint_files']!=spec['final_checkpoint']['files'] or
            old103['readout']!=spec['readout'] or old103['probe_rows']!=[0,1024] or
            reference['target_raw_sha256']!=TARGET_SHA256):
        raise ValueError('Matched historical target differs from Attempt126 controls')
    full_probe=privileged.a.load_probe(privileged.resolve(old103['paths']['probe_path']),
        {**old103['probe'],'sample_count':10000},torch)
    probe=a123.matched_probe(full_probe,matched_spec,torch)
    del full_probe
    base,final,load_seconds=privileged.load_model_pair(old103)
    final_mean=privileged.base_probe_mean(final,probe,old103)
    base_mean=privileged.base_probe_mean(base,probe,old103)
    target,target_hash=a123.verified_target(final_mean,base_mean,matched_spec,
        privileged.matched_difference,privileged.a.sha256_raw_float32_tensor,torch)
    if target_hash!=TARGET_SHA256:
        raise ValueError('Matched historical target raw SHA256 mismatch')
    for directory,expected in ((privileged.resolve(old103['paths']['base_directory']),
                                inventories['base_files']),
                               (privileged.resolve(old103['paths']['final_directory']),
                                inventories['final_files'])):
        if privileged.a.checkpoint_file_records(directory)!=expected:
            raise ValueError('Historical checkpoint changed during scoring')
    del base,final,final_mean,base_mean,probe
    gc.collect()
    torch.cuda.empty_cache()
    old_eval=a122.load_spec()['evaluation']
    validator=import_pinned(path_of(old_eval['validator_path']),
                            old_eval['validator_sha256'],
                            'attempt126_signed_position_validator')
    reports={name:a122.score_report(artifact[name],target,validator) for name in RESPONSE_NAMES}
    deltas,category=differences_and_classification(reports)
    context=spec['frozen_attempt100']
    require_hash(path_of(context['evaluation_path']),context['evaluation_sha256'])
    old100=json.loads(path_of(context['evaluation_path']).read_text())
    fineweb=[row for row in old100['candidates'] if row.get('candidate')=='response_fineweb']
    if len(fineweb)!=1:
        raise ValueError('Frozen Attempt100 FineWeb context inventory mismatch')
    old_context={'primary':fineweb[0]['positions_1_4_mean_cosine'],
                 'secondary':fineweb[0]['positions_1_127_mean_cosine'],
                 'role':'context_only_older_full_probe_population_not_exact_matched_baseline',
                 'evaluation_sha256':context['evaluation_sha256']}
    return reports,deltas,category,old_context,target_hash,load_seconds


def run():
    import torch
    spec,paths=load_spec()
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for the frozen FP32 strict JVP convention')
    a14=import_pinned(SOURCE014,SOURCE014_SHA256,'attempt126_frozen_attempt014')
    tokens,probe,token_hashes,corpus_provenance,model_dir=validate_blind_inputs(spec,a14,torch)
    model=a14.load_local_model(model_dir,'cuda',torch)
    eligible=a14.discover_eligible_linear_weights(model,torch)
    selected=a14.freeze_other_parameters(model,eligible)
    if len(eligible)!=98:
        raise ValueError('Expected 98 eligible Linear weights')
    before=a14.model_state_hashes(model,torch)
    surprisal=score_endpoint_surprisal(model,tokens,torch)
    ranking=rank_and_stratify(surprisal)
    validate_ranking(ranking)
    del surprisal
    jvp_spec={**spec,'probe':{**spec['probe'],'sample_count':1024}}
    responses={}
    gradient_records={}
    primal_reference=None
    for name in NAMES:
        rows=torch.tensor(ranking['strata'][name]['original_row_indices'],dtype=torch.int64)
        if (tuple(rows.shape)!=(1024,) or
                a14.sha256_raw_int64_tensor(rows,torch)!=
                    ranking['strata'][name]['ordered_row_indices_raw_sha256']):
            raise ValueError('Selected frozen FineWeb row hash mismatch')
        subset=tokens.index_select(0,rows).contiguous()
        if tuple(subset.shape)!=(1024,128):
            raise ValueError('Stratum gradient subset has wrong shape')
        mean_ce=a14.accumulate_mean_generic_gradient(
            model,subset,eligible,spec['gradient_loss'],torch)
        scale=validate_global_scale(a14.global_tangent_scale(eligible,.00125,torch))
        if not math.isclose(scale['aggregate_source_weight_norm'],918.1587250300843,
                            rel_tol=1e-8,abs_tol=1e-6):
            raise ValueError('Final eligible-weight norm differs from Attempt005')
        tangents,matrices,realized=a14.prepare_tangents(eligible,scale,torch)
        try:
            primal,response,readout=a14.compute_probe_response(
                model,probe,tangents,jvp_spec,torch)
        finally:
            tangents.clear()
            model.zero_grad(set_to_none=True)
        if primal_reference is None:
            primal_reference=primal
        elif not torch.equal(primal_reference,primal):
            raise ValueError('Unmodified final-checkpoint JVP primal changed across strata')
        response32=a122.prior.validate_response(response.to(torch.float32).contiguous(),torch)
        responses['response_'+name]=response32
        gradient_records[name]={'mean_generic_ce':mean_ce,**scale,**realized,
            'selected_row_hash':ranking['strata'][name]['ordered_row_indices_raw_sha256'],
            'eligible_matrix_count':len(matrices),'readout_checks':readout,
            'response_norm':float(torch.linalg.vector_norm(response32.double())),
            'response_raw_sha256':a14.sha256_raw_float32_tensor(response32,torch)}
        a14.verify_model_unchanged(model,before,torch)
        del rows,subset,primal,response,response32,tangents,matrices
        gc.collect()
    if list(responses)!=list(RESPONSE_NAMES):
        raise ValueError('Incomplete fixed LOW/MIDDLE/HIGH candidate inventory')
    pairwise=pairwise_response_cosines(responses,torch)
    if (a14.checkpoint_file_records(model_dir)!=spec['final_checkpoint']['files'] or
            any(sha256_file(path)!=digest for path,digest in token_hashes.items()) or
            sha256_file(path_of(spec['probe']['path']))!=spec['probe']['serialized_sha256']):
        raise ValueError('Frozen blind input changed during construction')
    require_hash(SPEC_PATH,SPEC_SHA256)
    constructor_hash=sha256_file(Path(__file__))
    candidate_path=paths['candidate_path']
    manifest_path=paths['construction_manifest_path']
    a122.prior.atomic_torch_publish(candidate_path,responses,torch)
    construction={'format_version':1,'attempt_id':ATTEMPT,
        'information_policy':spec['information_policy'],
        'spec_sha256':SPEC_SHA256,'constructor_sha256':constructor_hash,
        'frozen_attempt005':spec['frozen_attempt005'],
        'frozen_attempt100_construction_manifest_sha256':
            spec['frozen_attempt100']['construction_manifest_sha256'],
        'frozen_attempt113_construction_manifest_sha256':
            spec['frozen_attempt113']['construction_manifest_sha256'],
        'final_checkpoint_files':spec['final_checkpoint']['files'],
        'fineweb_corpus_provenance':corpus_provenance,
        'ranking':ranking,'unused_rank_intervals':spec['unused_rank_intervals'],
        'gradient_loss':spec['gradient_loss'],'tangent':spec['tangent'],
        'probe':spec['probe'],'readout':spec['readout'],'jvp':spec['jvp'],
        'gradient_records':gradient_records,
        'responses':{response:{'raw_sha256':gradient_records[name]['response_raw_sha256'],
            'shape':[128,2048],'dtype':'contiguous_cpu_float32',
            'norm':gradient_records[name]['response_norm']}
            for name,response in zip(NAMES,RESPONSE_NAMES)},
        'pairwise_flattened_response_cosine_oracle_free':pairwise,
        'final_primal_float64_sha256':hashlib.sha256(
            primal_reference.numpy().astype('<f8',copy=False).tobytes()).hexdigest(),
        'candidate':{'path':str(candidate_path),
            'serialized_sha256':sha256_file(candidate_path),
            'tensor_order':list(RESPONSE_NAMES),'shape':[128,2048],
            'dtype':'contiguous_cpu_float32'},
        'barrier':spec['barrier'],
        'same_specimen_exploratory_method_development':True,
        'clean_heldout_validation':False}
    a122.prior.atomic_json_publish(manifest_path,construction)
    published_manifest_hash=sha256_file(manifest_path)
    frozen=validate_published(candidate_path,manifest_path,construction,torch,a14)
    require_hash(candidate_path,construction['candidate']['serialized_sha256'])
    require_hash(manifest_path,published_manifest_hash)
    require_hash(SPEC_PATH,SPEC_SHA256)
    require_hash(Path(__file__),constructor_hash)
    print('SURPRISAL_STRATA_CANDIDATES_FROZEN',flush=True)

    # The historical base, target, and prior historical scores are accessed only below.
    del model,eligible,selected,before,tokens,probe,primal_reference,responses
    gc.collect()
    torch.cuda.empty_cache()
    reports,deltas,category,old_context,target_hash,load_seconds=evaluate_after_barrier(
        spec,frozen,torch)
    correlations=descriptive_correlations(ranking,gradient_records,reports)
    result={'format_version':1,'attempt_id':ATTEMPT,'purpose':spec['purpose'],
        'same_specimen_exploratory_method_development':True,
        'clean_heldout_validation':False,
        'construction_manifest_sha256':sha256_file(manifest_path),
        'candidate_serialized_sha256':construction['candidate']['serialized_sha256'],
        'matched_target_raw_sha256':target_hash,'matched_model_load_seconds':load_seconds,
        'signed_position_reports':reports,'fixed_pairwise_deltas':deltas,
        'development_set_classification':category,
        'descriptive_correlations':correlations,
        'old_attempt100_fineweb_full_probe_context':old_context,
        'strata':ranking['strata'],
        'unused_rank_intervals':ranking['unused_rank_intervals'],
        'gradient_records':gradient_records,
        'oracle_free_pairwise_response_cosines':pairwise,
        'workload':spec['workload']}
    require_hash(candidate_path,construction['candidate']['serialized_sha256'])
    if paths['result_path'].exists() or paths['result_path'].is_symlink():
        raise ValueError('Attempt126 result already exists')
    a122.prior.atomic_json_publish(paths['result_path'],result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        run()
    except (OSError,RuntimeError,ValueError,KeyError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
