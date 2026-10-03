#!/usr/bin/env python3
"""Attempt124: blind Armijo descent of the frozen diverse-8 scalar objective."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '124_diverse8_consensus_relaxation'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = 'd5d34b0504aa69d6ad0011164de749bbbe8d0f04abbab5197c22d35467f06c49'
ORDER = ('fineweb', 'wikitext103_raw', 'tinystories', 'arxiv_document',
         'cc_news', 'pg19', 'codeparrot_clean', 'ultrachat')
GRID = (1, 2, 4)
STEPS = 4
RECOVERY_STEP = 2
EXPECTED_NORMS = (11.210221591371743, 12.197408478436103,
                  19.5600251946489, 16.678477813023473,
                  9.771288332862822, 42.22101383805119,
                  8.360046148409692, 23.162808573978225)
EXPECTED_B_NORM = 0.6924963677590905
EXPECTED_B_SHA256 = 'a90785341961a0a2a08e7949961cb105867cf2509013b01b65c2013322ead574'
EXPECTED_RAW_SHA256 = '8ddc1801ee82504e10898e97d12b53fea1442ed029314dc47d7d0f68523c2d0a'
TARGET_SHA256 = '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f'
SOURCE122 = PROJECT/'scripts/ablation/run_loss_stabilized_generic_relaxation.py'
SOURCE122_SHA256 = '7dd7e79b84ac297c7b0f7a62a69ff5dc7d91332b95b3b180f72f50c1f6cf3a9d'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(16*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError('Frozen SHA256 mismatch: '+str(path))
    return path


def import_pinned(path, expected, name):
    require_hash(path, expected)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


a122 = import_pinned(SOURCE122, SOURCE122_SHA256, 'attempt124_frozen_attempt122')
path_of = a122.path_of
LINE_SEARCH = {**a122.LINE_SEARCH,
    'method': 'deterministic_Armijo_backtracking_on_full_fixed_diverse8_weighted_CE',
    'trial_objective': 'full_fixed_weighted_eight_corpus_mean_causal_CE',
    'trial_forward_batches': 512}


def load_spec():
    require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text())
    old122 = a122.load_spec()
    frozen113 = spec['frozen_attempt113']
    frozen122 = spec['frozen_attempt122']
    frozen123 = spec['frozen_attempt123']
    if (spec['attempt_id'] != ATTEMPT or spec['corpus_order'] != list(ORDER) or
            [row['name'] for row in spec['corpora']] != list(ORDER) or
            any(row['selected_rows'] != [0, 512] for row in spec['corpora']) or
            list(spec['objective']['endpoint_gradient_norms'].values()) != list(EXPECTED_NORMS) or
            frozen113['b_blind_norm'] != EXPECTED_B_NORM or
            frozen113['b_blind_aggregate_sha256'] != EXPECTED_B_SHA256 or
            frozen113['raw_response_sha256'] != EXPECTED_RAW_SHA256 or
            frozen113['endpoint_gradient_norms'] != spec['objective']['endpoint_gradient_norms'] or
            spec['objective']['endpoint_b_norm'] != EXPECTED_B_NORM or
            spec['objective']['endpoint_b_aggregate_sha256'] != EXPECTED_B_SHA256 or
            spec['objective']['endpoint_weight_rule'] !=
                'w_c=1/(8*endpoint_global_gradient_norm_c)' or
            spec['objective']['weights_frozen_after_endpoint'] is not True or
            spec['objective']['later_per_corpus_gradient_renormalization'] is not False or
            spec['objective']['gradient_arithmetic'] !=
                'Attempt113_FP64_divide_by_frozen_endpoint_norm_then_FP32_cast_then_ordered_FP32_add_alpha_0.125' or
            spec['model'] != old122['model'] or
            spec['eligible_tensors'] != old122['eligible_tensors'] or
            spec['readout'] != old122['readout'] or
            spec['probe']['rows'] != [0, 1024] or
            spec['probe']['serialized_sha256'] != old122['probe']['serialized_sha256'] or
            spec['generic_loss'] != {'rows_per_corpus': [0, 512], 'sequence_length': 128,
                'batch_size': 8, 'batches_per_corpus': 64,
                'prediction_contexts_per_corpus': 65024,
                'type': 'mean_causal_next_token_cross_entropy', 'corpus_count': 8} or
            spec['relaxation'] != {
                'steps': 4, 'functional_checkpoints': [1, 2, 4], 'recovery_step': 2,
                'initial_relative_frobenius_step': .00125,
                'eta_rule': 'eta_max=(0.00125*initial_eligible_weight_norm)/verified_endpoint_b_blind_norm',
                'line_search': LINE_SEARCH,
                'update': 'W <- W - eta * grad_L_bal',
                'update_arithmetic_dtype': 'cpu_float64', 'stored_weight_dtype': 'float32',
                'optimizer': False, 'momentum': False, 'weight_decay': False,
                'gradient_clipping': False, 'mixed_precision': False,
                'early_stopping': False, 'response': 'A_final - A_k',
                'response_normalization': False, 'response_rescaling': False,
                'sign_selection': False} or
            spec['barrier'] != {'marker': 'CONSENSUS_RELAXATION_CANDIDATES_FROZEN',
                'all_4_steps_and_3_responses_plus_raw_b_validated_published_revalidated_before_historical_access': True,
                'recovery_checkpoint_scientific_candidate': False} or
            spec['evaluation']['target_raw_sha256'] != TARGET_SHA256 or
            spec['evaluation']['primary_positions'] != [1, 2, 3, 4] or
            spec['evaluation']['secondary_positions'] != list(range(1,128)) or
            spec['evaluation']['clear_thresholds'] != {
                'primary_gain_vs_raw_b': .03, 'secondary_gain_vs_raw_b': .02} or
            spec['evaluation']['moderate_thresholds'] != {
                'primary_gain_vs_raw_b': .015, 'secondary_gain_vs_raw_b': 0.0} or
            frozen122['source_sha256'] != SOURCE122_SHA256 or
            frozen122['spec_sha256'] != a122.SPEC_SHA256 or
            frozen122['armijo_rule'] != 'reuse_frozen_Attempt122_armijo_search' or
            frozen123['target_raw_sha256'] != TARGET_SHA256 or
            spec['workload']['total_backward_batches'] != 2048 or
            spec['workload']['forward_batches_per_armijo_trial'] != 512):
        raise ValueError('Attempt124 frozen scientific plan mismatch')
    paths = [path_of(spec['paths'][k]) for k in
             ('candidate_path', 'recovery_path', 'construction_manifest_path', 'result_path')]
    if len(set(paths)) != 4 or any(path_of(value) in paths for value in
            (frozen113['artifact_path'], *[spec['final_checkpoint']['directory']])):
        raise ValueError('Attempt124 output path overlap')
    return spec


def refuse_outputs(spec, resume=False):
    for key in ('candidate_path', 'construction_manifest_path', 'result_path'):
        path = path_of(spec['paths'][key])
        if path.exists() or path.is_symlink():
            raise ValueError('Attempt124 output already exists: '+str(path))
    recovery = path_of(spec['paths']['recovery_path'])
    if resume:
        if recovery.is_symlink() or not recovery.is_file():
            raise ValueError('Step-2 recovery checkpoint required for resume')
    elif recovery.exists() or recovery.is_symlink():
        raise ValueError('Stale Attempt124 recovery checkpoint; use --resume')


def validate_blind_controls(spec, a113):
    frozen = spec['frozen_attempt113']
    for key in ('spec', 'source', 'construction_manifest'):
        require_hash(path_of(frozen[key+'_path']), frozen[key+'_sha256'])
    old113 = a113.load_spec()
    manifest = json.loads(path_of(frozen['construction_manifest_path']).read_text())
    if (spec['model'] != old113['model'] or
            spec['final_checkpoint'] != old113['final_checkpoint'] or
            spec['eligible_tensors'] != old113['eligible_tensors'] or
            spec['corpora'] != old113['corpora'] or
            spec['probe'] != old113['probe'] or
            spec['readout'] != old113['readout'] or
            spec['corpus_order'] != old113['corpus_order'] or
            old113['rhs']['rows'] != [0,512] or
            old113['rhs']['batch_size'] != 8 or
            old113['rhs']['batches_per_corpus'] != 64 or
            old113['rhs']['prediction_contexts_per_corpus'] != 65024 or
            manifest['spec_sha256'] != frozen['spec_sha256'] or
            manifest['constructor_source_sha256'] != frozen['source_sha256'] or
            manifest['final_checkpoint_files'] != old113['final_checkpoint']['files'] or
            manifest['corpus_order'] != list(ORDER) or
            [row['name'] for row in manifest['per_corpus_gradients']] != list(ORDER) or
            [row['global_gradient_norm'] for row in manifest['per_corpus_gradients']] != list(EXPECTED_NORMS) or
            manifest['b_blind']['norm'] != EXPECTED_B_NORM or
            manifest['b_blind']['ordered_matrix_hash_sha256'] != EXPECTED_B_SHA256 or
            manifest['functional_responses']['response_raw_rhs']['raw_sha256'] != EXPECTED_RAW_SHA256 or
            manifest['artifact']['path'] != frozen['artifact_path'] or
            manifest['artifact']['serialized_sha256'] != frozen['artifact_serialized_sha256'] or
            manifest['artifact']['tensor_inventory'] != frozen['artifact_inventory'] or
            manifest['probe'] != old113['probe'] or
            manifest['historical_base_access'] is not False or
            manifest['oracle_adl_access'] is not False):
        raise ValueError('Frozen Attempt113 endpoint/response provenance contradiction')
    return old113, manifest


def weights_from_endpoint(order, endpoint_norms):
    if (not order or list(endpoint_norms) != list(order) or
            any(not math.isfinite(norm) or norm <= 0 for norm in endpoint_norms.values())):
        raise ValueError('Malformed ordered endpoint norms')
    return {name: 1.0/(len(order)*endpoint_norms[name]) for name in order}


def frozen_weights(endpoint_norms):
    if list(endpoint_norms) != list(ORDER):
        raise ValueError('Attempt124 requires exactly eight ordered endpoint norms')
    return weights_from_endpoint(ORDER, endpoint_norms)


def weighted_objective(losses, weights):
    if list(losses) != list(ORDER) or list(weights) != list(ORDER):
        raise ValueError('Weighted objective corpus order mismatch')
    if any(not math.isfinite(losses[name]) or not math.isfinite(weights[name]) or
           weights[name] <= 0 for name in ORDER):
        raise ValueError('Nonfinite weighted objective input')
    objective = math.fsum(weights[name]*losses[name] for name in ORDER)
    if not math.isfinite(objective):
        raise ValueError('Nonfinite weighted objective')
    return objective


def validate_endpoint(rhs_record, diagnostics, manifest):
    if (len(diagnostics) != 8 or [row['name'] for row in diagnostics] != list(ORDER) or
            [row['global_gradient_norm'] for row in diagnostics] != list(EXPECTED_NORMS) or
            rhs_record['norm'] != EXPECTED_B_NORM or
            rhs_record['ordered_matrix_hash_sha256'] != EXPECTED_B_SHA256 or
            rhs_record['matrix_raw_sha256'] != manifest['b_blind']['matrix_raw_sha256']):
        raise ValueError('Recomputed endpoint b_blind differs from frozen Attempt113')
    return True


def weighted_gradient_pass(model, corpora, eligible, inventory, endpoint_norms, a113, torch):
    """Later gradients use the same endpoint divisors, never current norms."""
    combined = {}
    losses, current_norms = {}, {}
    loss_spec={'sample_count':512,'sequence_length':128,'batch_size':8,
               'number_of_batches':64,'total_prediction_tokens':65024,
               'predictions_per_sample':127}
    for index, name in enumerate(ORDER):
        loss=a113.blind.accumulate_mean_generic_gradient(
            model,corpora[name][:512].contiguous(),eligible,loss_spec,torch)
        current_norm=a122.prior.gradient_norm(eligible,torch)
        a113.add_unit_gradient_to_consensus(
            combined, eligible, endpoint_norms[name], inventory,
            corpus_index=index, corpus_count=8)
        losses[name], current_norms[name] = loss, current_norm
        model.zero_grad(set_to_none=True)
    norm = math.sqrt(a113.matrixwise_dot(combined, combined))
    if not math.isfinite(norm):
        raise ValueError('Nonfinite weighted-gradient norm')
    return combined, losses, current_norms, norm


def forward_corpus_loss(model, tokens, a113, torch):
    if (tokens.dtype != torch.int64 or tuple(tokens.shape) != (4096,128) or
            not tokens.is_contiguous()):
        raise ValueError('Malformed full frozen corpus tensor')
    device = next(model.parameters()).device
    parts = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0,512,8):
            batch = tokens[start:start+8].to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            loss_sum = a113.blind.causal_token_loss_sum(logits, batch, torch)
            if not bool(torch.isfinite(loss_sum)):
                raise ValueError('Nonfinite trial corpus CE')
            parts.append(float(loss_sum.detach().to('cpu',dtype=torch.float64)))
    if len(parts) != 64:
        raise ValueError('Trial corpus loss must use 64 batches')
    return math.fsum(parts)/65024


def forward_weighted_objective(model, corpora, weights, a113, torch):
    model.eval()
    losses = {name: forward_corpus_loss(model, corpora[name], a113, torch)
              for name in ORDER}
    return weighted_objective(losses, weights)


def initial_weight_norm(eligible, torch):
    pieces = []
    for _, module in eligible:
        value = module.weight.detach().to('cpu',dtype=torch.float64)
        pieces.append(float(torch.sum(value.square(),dtype=torch.float64)))
    norm = math.sqrt(math.fsum(pieces))
    if not math.isclose(norm, a122.prior.EXPECTED['weight_norm'], rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError('Initial eligible-weight norm differs from frozen Attempt005/113')
    return norm


def update_from_vector(eligible, vector, eta, torch):
    """Attempt005/122 CPU FP64 subtract, then exactly one FP32 storage cast."""
    if len(eligible) != 98 or not math.isfinite(eta) or eta <= 0 or set(vector) != {
            name+'.weight' for name,_ in eligible}:
        raise ValueError('Malformed weighted update inventory/eta')
    with torch.no_grad():
        for name,module in eligible:
            weight = module.weight
            gradient = vector[name+'.weight']
            if (gradient.dtype != torch.float32 or gradient.device.type != 'cpu' or
                    gradient.shape != weight.shape or not bool(torch.isfinite(gradient).all())):
                raise ValueError('Malformed weighted gradient matrix')
            source64 = weight.detach().to('cpu',dtype=torch.float64)
            gradient64 = gradient.to(dtype=torch.float64)
            updated = (source64-eta*gradient64).to(torch.float32).contiguous()
            if not bool(torch.isfinite(updated).all()):
                raise ValueError('Nonfinite weighted update')
            weight.copy_(updated.to(weight.device))


def mean_activation(model, probe, spec, a113, torch):
    if (tuple(probe.shape) != (1024,128) or probe.dtype != torch.int64 or
            not probe.is_contiguous() or spec['probe']['rows'] != [0,1024]):
        raise ValueError('Malformed frozen 1024-row activation probe')
    accumulator = torch.zeros((128,2048),dtype=torch.float64,device='cpu')
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for start in range(0,1024,32):
            batch = probe[start:start+32].to(device)
            readout = a113.blind.ordinary_hook_readout(model,batch,spec,torch)
            accumulator.add_(readout.detach().to(device='cpu',dtype=torch.float64).sum(dim=0))
    mean = (accumulator/1024).contiguous()
    if not bool(torch.isfinite(mean).all()):
        raise ValueError('Nonfinite activation mean')
    return mean


def response_cosine(left, right, torch):
    a122.prior.validate_response(left,torch)
    a122.prior.validate_response(right,torch)
    x,y = left.double().flatten(),right.double().flatten()
    nx,ny = torch.linalg.vector_norm(x),torch.linalg.vector_norm(y)
    return None if float(nx)==0 or float(ny)==0 else float(torch.dot(x,y)/(nx*ny))


def validate_raw_baseline(spec, manifest, a113, torch):
    frozen=spec['frozen_attempt113']
    path=path_of(frozen['artifact_path'])
    require_hash(path,frozen['artifact_serialized_sha256'])
    artifact=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if (not isinstance(artifact,dict) or list(artifact)!=frozen['artifact_inventory'] or
            manifest['artifact']['tensor_inventory']!=frozen['artifact_inventory']):
        raise ValueError('Attempt113 artifact inventory mismatch')
    raw=a122.prior.validate_response(artifact['response_raw_rhs'],torch)
    if a113.blind.sha256_raw_float32_tensor(raw,torch)!=EXPECTED_RAW_SHA256:
        raise ValueError('Attempt113 raw-b functional response SHA256 mismatch')
    require_hash(path,frozen['artifact_serialized_sha256'])
    return raw


def validate_rows(rows, scale):
    a122.validate_initial_scale(scale)
    if scale['pilot_initial_gradient_norm']!=EXPECTED_B_NORM:
        raise ValueError('Initial Armijo scale is not derived from Attempt113 b_blind')
    mapped=[]
    for row in rows:
        if (list(row['per_corpus_losses'])!=list(ORDER) or
                any(not math.isfinite(value) for value in row['per_corpus_losses'].values()) or
                not math.isfinite(row['weighted_objective_before_update']) or
                row['weighted_objective_before_update']!=row['generic_mean_ce_before_update'] or
                row['accepted_post_step_weighted_objective']!=row['accepted_post_step_generic_mean_ce'] or
                any(trial['weighted_objective']!=trial['generic_mean_ce']
                    for trial in row['trials']) or
                len(row['trials'])>8):
            raise ValueError('Malformed weighted trajectory row')
        mapped.append(row)
    return a122.validate_step_records(mapped,scale)


def interpret(reports):
    if list(reports)!=['raw_b',1,2,4]:
        raise ValueError('Expected raw-b and exactly three finite checkpoints')
    raw=reports['raw_b']
    details={}
    for k in GRID:
        report=reports[k]
        primary=report['positions_1_4_mean_cosine']
        secondary=report['positions_1_127_mean_cosine']
        gain_p=None if primary is None or raw['positions_1_4_mean_cosine'] is None else primary-raw['positions_1_4_mean_cosine']
        gain_s=None if secondary is None or raw['positions_1_127_mean_cosine'] is None else secondary-raw['positions_1_127_mean_cosine']
        category=('clear_improvement_over_raw_b' if gain_p is not None and gain_s is not None and gain_p>=.03 and gain_s>=.02 else
                  'moderate_improvement_over_raw_b' if gain_p is not None and gain_s is not None and gain_p>=.015 and gain_s>=0 else
                  'no_meaningful_improvement_over_raw_b')
        details[str(k)]={'primary_gain_vs_raw_b':gain_p,'secondary_gain_vs_raw_b':gain_s,
                         'category':category}
    best=max(GRID,key=lambda k:(-math.inf if reports[k]['positions_1_4_mean_cosine'] is None else reports[k]['positions_1_4_mean_cosine'],
                                -math.inf if reports[k]['positions_1_127_mean_cosine'] is None else reports[k]['positions_1_127_mean_cosine'],-k))
    return details,{'best_finite_k':best,'best_category':details[str(best)]['category'],
                    'selection':'descriptive_same_specimen_development_set'}


def save_recovery(path, eligible, rows, checkpoints, responses, final_mean,
                  endpoint, weights, scale, source_hash, a113, torch):
    if (len(rows)!=RECOVERY_STEP or list(responses)!=[1,2] or
            sorted(checkpoints)!=[1,2] or not validate_rows(rows,scale)):
        raise ValueError('Recovery is fixed after accepted step 2')
    payload={'recovery_only':True,'step':RECOVERY_STEP,
             'previous_accepted_eta':rows[-1]['accepted_eta'],
             'state':a122.prior.original_weights(eligible,torch),
             'state_aggregate_sha256':checkpoints[2]['state_aggregate_sha256'],
             'rows':rows,'checkpoints':checkpoints,'responses':responses,
             'final_mean':final_mean,'endpoint':endpoint,'fixed_weights':weights,
             'initial_scale':scale,'source_checkpoint_sha256':source_hash,
             'spec_sha256':SPEC_SHA256}
    a122.prior.atomic_torch_publish(path,payload,torch)


def restore_recovery(path, eligible, source_hash, a113, torch):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if (not isinstance(value,dict) or
            not isinstance(value.get('rows'),list) or len(value['rows'])!=2 or
            not isinstance(value.get('responses'),dict) or
            not isinstance(value.get('checkpoints'),dict) or
            not isinstance(value.get('endpoint'),dict) or
            not isinstance(value.get('fixed_weights'),dict) or
            value.get('recovery_only') is not True or
            value.get('step')!=RECOVERY_STEP or
            value.get('source_checkpoint_sha256')!=source_hash or
            value.get('spec_sha256')!=SPEC_SHA256 or
            value.get('previous_accepted_eta')!=value['rows'][-1]['accepted_eta'] or
            list(value['responses'])!=[1,2] or sorted(value['checkpoints'])!=[1,2] or
            value['endpoint']['ordered_matrix_hash_sha256']!=EXPECTED_B_SHA256 or
            value['endpoint']['norm']!=EXPECTED_B_NORM or
            value['fixed_weights']!=frozen_weights(dict(zip(ORDER,EXPECTED_NORMS))) or
            not validate_rows(value['rows'],value['initial_scale'])):
        raise ValueError('Malformed or mismatched step-2 recovery')
    a122.restore_pre_step(eligible,value['state'],torch)
    hashes,aggregate=a122.prior.state_hashes(eligible,a113.blind,torch)
    if (hashes!=value['checkpoints'][2]['state_per_matrix_sha256'] or
            aggregate!=value['state_aggregate_sha256']):
        raise ValueError('Recovered eligible state hash mismatch')
    for k,response in value['responses'].items():
        a122.prior.validate_response(response,torch)
        if a113.blind.sha256_raw_float32_tensor(response,torch)!=value['checkpoints'][k]['response']['raw_sha256']:
            raise ValueError('Recovered response hash mismatch')
    mean=value['final_mean']
    if (not isinstance(mean,torch.Tensor) or mean.dtype!=torch.float64 or
            tuple(mean.shape)!=(128,2048) or mean.device.type!='cpu' or
            not bool(torch.isfinite(mean).all())):
        raise ValueError('Malformed recovered final activation mean')
    return value


def evaluate_after_barrier(spec, frozen_responses, raw_baseline, manifest, a113, torch):
    frozen123=spec['frozen_attempt123']
    for key in ('spec','source'):
        require_hash(path_of(frozen123[key+'_path']),frozen123[key+'_sha256'])
    a123=import_pinned(path_of(frozen123['source_path']),frozen123['source_sha256'],
                       'attempt124_frozen_attempt123')
    matched_spec=a123.load_spec()
    reference=matched_spec['matched_reference']
    privileged=import_pinned(path_of(reference['attempt103_source_path']),
                             reference['attempt103_source_sha256'],
                             'attempt124_frozen_attempt103')
    frozen122=matched_spec['frozen_attempt122']
    require_hash(path_of(frozen122['construction_manifest_path']),
                 frozen122['construction_manifest_sha256'])
    manifest122=json.loads(path_of(frozen122['construction_manifest_path']).read_text())
    old103,inventories=a123.validate_matched_reference(matched_spec,manifest122,privileged)
    if (old103['final_checkpoint_files']!=spec['final_checkpoint']['files'] or
            old103['readout']!=spec['readout'] or
            old103['probe_rows']!=[0,1024]):
        raise ValueError('Matched-target model/probe differs from Attempt124')
    full_probe=privileged.a.load_probe(privileged.resolve(old103['paths']['probe_path']),
        {**old103['probe'],'sample_count':10000},torch)
    probe=a123.matched_probe(full_probe,matched_spec,torch)
    del full_probe
    base,final,loads=privileged.load_model_pair(old103)
    final_mean=privileged.base_probe_mean(final,probe,old103)
    base_mean=privileged.base_probe_mean(base,probe,old103)
    target,target_hash=a123.verified_target(final_mean,base_mean,matched_spec,
        privileged.matched_difference,privileged.a.sha256_raw_float32_tensor,torch)
    if target_hash!=TARGET_SHA256:
        raise ValueError('Matched historical target SHA256 contradiction')
    for directory,expected in ((privileged.resolve(old103['paths']['base_directory']),
                                inventories['base_files']),
                               (privileged.resolve(old103['paths']['final_directory']),
                                inventories['final_files'])):
        if privileged.a.checkpoint_file_records(directory)!=expected:
            raise ValueError('Historical model checkpoint changed during scoring')
    del base,final,final_mean,base_mean,probe
    gc.collect()
    torch.cuda.empty_cache()
    old_eval=a122.load_spec()['evaluation']
    evaluator=import_pinned(path_of(old_eval['validator_path']),
        old_eval['validator_sha256'],'attempt124_signed_cosine_validator')
    reports={'raw_b':a122.score_report(raw_baseline,target,evaluator)}
    for k in GRID:
        reports[k]=a122.score_report(frozen_responses[f'R_{k}'],target,evaluator)
    detail,choice=interpret(reports)
    later={}
    for k in (2,4):
        later[str(k)]={
            'matched_primary_gain_vs_k1':a123.optional_difference(
                reports[k]['positions_1_4_mean_cosine'],reports[1]['positions_1_4_mean_cosine']),
            'matched_secondary_gain_vs_k1':a123.optional_difference(
                reports[k]['positions_1_127_mean_cosine'],reports[1]['positions_1_127_mean_cosine']),
            **a123.position_gain_summary(reports[k],reports[1])}
    return reports,detail,choice,later,loads,target_hash


def run(*, resume=False):
    import torch
    spec=load_spec()
    refuse_outputs(spec,resume=resume)
    if not torch.cuda.is_available():
        raise ValueError('CUDA is required for the Attempt124 FP32 pilot')
    frozen113=spec['frozen_attempt113']
    a113=import_pinned(path_of(frozen113['source_path']),frozen113['source_sha256'],
                       'attempt124_frozen_attempt113')
    old113,manifest113=validate_blind_controls(spec,a113)
    corpora,probe,corpus_manifests=a113.validate_blind_sources(old113)
    model_dir=a113.safe_input_path(old113['final_checkpoint']['directory'])
    model=a113.blind.load_local_model(model_dir,'cuda',torch)
    eligible=a113.blind.discover_eligible_linear_weights(model,torch)
    selected=a113.blind.freeze_other_parameters(model,eligible)
    inventory=a113.coordinate_inventory(eligible)
    if inventory!=manifest113['coordinate_inventory']:
        raise ValueError('Attempt113 eligible-coordinate inventory contradiction')
    frozen_before=a113.blind.frozen_parameter_hashes(model,selected,torch)
    originals=a122.prior.original_weights(eligible,torch)
    initial_norm=initial_weight_norm(eligible,torch)
    final_mean=mean_activation(model,probe,spec,a113,torch)
    final_mean_hash=hashlib.sha256(final_mean.numpy().astype('<f8',copy=False).tobytes()).hexdigest()
    raw_baseline=validate_raw_baseline(spec,manifest113,a113,torch)
    rows,checkpoints,responses=[],{},{}
    endpoint=None
    scale=None
    fixed=None
    previous_eta=None
    start_step=1
    source_hash=hashlib.sha256(json.dumps(spec['final_checkpoint']['files'],
        sort_keys=True,separators=(',',':')).encode()).hexdigest()
    recovery_path=path_of(spec['paths']['recovery_path'])
    if resume:
        recovered=restore_recovery(recovery_path,eligible,source_hash,a113,torch)
        if not torch.equal(final_mean,recovered['final_mean']):
            raise ValueError('Recovered final activation mean changed')
        rows,checkpoints,responses=(recovered['rows'],recovered['checkpoints'],
                                    recovered['responses'])
        endpoint=recovered['endpoint']
        scale=recovered['initial_scale']
        fixed=recovered['fixed_weights']
        previous_eta=recovered['previous_accepted_eta']
        start_step=3
        if endpoint!=manifest113['b_blind']:
            raise ValueError('Recovered endpoint b_blind differs from Attempt113')
    for step in a122.prior.trajectory_steps(start_step):
        if step==1:
            vector,diagnostics,endpoint=a113.build_blind_rhs(
                model,corpora,eligible,inventory)
            validate_endpoint(endpoint,diagnostics,manifest113)
            endpoint_norms={row['name']:row['global_gradient_norm'] for row in diagnostics}
            fixed=frozen_weights(endpoint_norms)
            losses={row['name']:row['mean_generic_loss'] for row in diagnostics}
            current_norms=endpoint_norms
            gradient_norm=endpoint['norm']
            target_step_norm=.00125*initial_norm
            scale=a122.initial_scale_from_gradient({
                'aggregate_source_weight_norm':initial_norm,
                'aggregate_generic_gradient_norm':gradient_norm,
                'target_delta_norm':target_step_norm,
                'alpha':target_step_norm/gradient_norm})
            initial_eta=scale['eta_max']
        else:
            endpoint_norms=dict(zip(ORDER,EXPECTED_NORMS))
            vector,losses,current_norms,gradient_norm=weighted_gradient_pass(
                model,corpora,eligible,inventory,endpoint_norms,a113,torch)
            initial_eta=previous_eta
        current=weighted_objective(losses,fixed)
        consistency=None if not rows else a122.check_recomputed_loss(
            current,rows[-1]['accepted_post_step_weighted_objective'])
        snapshot=a122.prior.original_weights(eligible,torch)
        def apply_trial(eta):
            update_from_vector(eligible,vector,eta,torch)
        def trial_loss():
            return forward_weighted_objective(model,corpora,fixed,a113,torch)
        def restore():
            a122.restore_pre_step(eligible,snapshot,torch)
        search=a122.armijo_search(current,gradient_norm,initial_eta,
                                  apply_trial,trial_loss,restore)
        del snapshot,vector
        gc.collect()
        previous_eta=search['accepted_eta']
        model.zero_grad(set_to_none=True)
        a113.blind.verify_frozen_parameters(model,selected,frozen_before,torch)
        displacement=a122.prior.displacement_norm(eligible,originals,torch)
        trials=[{**item,'weighted_objective':item['generic_mean_ce']} for item in search['trials']]
        search={**search,'trials':trials}
        row={'step':step,'per_corpus_losses':losses,
             'current_per_corpus_gradient_norms_diagnostic_only':current_norms,
             'weighted_objective_before_update':current,
             'generic_mean_ce_before_update':current,
             'aggregate_gradient_norm_before_update':gradient_norm,
             **search,
             'accepted_post_step_weighted_objective':search['accepted_post_step_generic_mean_ce'],
             'pre_step_objective_consistency_abs_difference':consistency,
             'cumulative_displacement_norm':displacement,
             'cumulative_displacement_relative_to_initial_weight_norm':displacement/initial_norm}
        rows.append(row)
        validate_rows(rows,scale)
        print(f"weighted Armijo step {step}/{STEPS} | {current:.8g} -> "
              f"{search['accepted_post_step_generic_mean_ce']:.8g} | "
              f"trial {search['accepted_trial_index']}/7 | eta {search['accepted_eta']:.8g}",flush=True)
        if step in GRID:
            mean=mean_activation(model,probe,spec,a113,torch)
            response=a122.prior.response_from_means(final_mean,mean,torch)
            diagnostic=a122.prior.response_diagnostics(
                response,responses.get(1),a113.blind,torch)
            diagnostic['cosine_vs_Attempt113_response_raw_rhs']=response_cosine(
                response,raw_baseline,torch)
            per_matrix,aggregate=a122.prior.state_hashes(eligible,a113.blind,torch)
            checkpoints[step]={'step':step,'state_per_matrix_sha256':per_matrix,
                'state_aggregate_sha256':aggregate,'response':diagnostic,
                'cumulative_displacement_norm':displacement,
                'cumulative_displacement_relative_to_initial_weight_norm':
                    row['cumulative_displacement_relative_to_initial_weight_norm']}
            responses[step]=response
        if step==RECOVERY_STEP and not resume:
            save_recovery(recovery_path,eligible,rows,checkpoints,responses,
                          final_mean,endpoint,fixed,scale,source_hash,a113,torch)
    if (len(rows)!=STEPS or list(responses)!=list(GRID) or
            sorted(checkpoints)!=list(GRID) or not validate_rows(rows,scale)):
        raise ValueError('Incomplete four-step weighted Armijo trajectory')
    a113.blind.verify_frozen_parameters(model,selected,frozen_before,torch)
    a113._input_hashes_unchanged(old113)
    if a113.blind.checkpoint_file_records(model_dir)!=spec['final_checkpoint']['files']:
        raise ValueError('Final checkpoint changed during blind construction')
    for key in ('spec','source','construction_manifest'):
        require_hash(path_of(frozen113[key+'_path']),frozen113[key+'_sha256'])
    require_hash(SPEC_PATH,SPEC_SHA256)
    constructor_hash=sha256_file(Path(__file__))
    raw_baseline=validate_raw_baseline(spec,manifest113,a113,torch)
    refuse_outputs(spec,resume=True)
    candidate_path=path_of(spec['paths']['candidate_path'])
    manifest_path=path_of(spec['paths']['construction_manifest_path'])
    a122.prior.atomic_torch_publish(candidate_path,
        {f'R_{k}':responses[k] for k in GRID},torch)
    construction={'format_version':1,'attempt_id':ATTEMPT,
        'information_policy':spec['information_policy'],
        'spec_sha256':SPEC_SHA256,'constructor_sha256':constructor_hash,
        'frozen_attempt113_spec_sha256':frozen113['spec_sha256'],
        'frozen_attempt113_source_sha256':frozen113['source_sha256'],
        'frozen_attempt113_manifest_sha256':frozen113['construction_manifest_sha256'],
        'source_checkpoint_files':spec['final_checkpoint']['files'],
        'corpus_order':list(ORDER),'corpus_manifests':corpus_manifests,
        'generic_loss':spec['generic_loss'],'objective':spec['objective'],
        'endpoint_b_blind':endpoint,'fixed_weights':fixed,'initial_scale':scale,
        'line_search':LINE_SEARCH,
        'final_activation_mean':{'raw_float64_sha256':final_mean_hash,
                                 'shape':[128,2048],'dtype':'contiguous_cpu_float64'},
        'steps':rows,'checkpoints':{str(k):checkpoints[k] for k in GRID},
        'raw_b_baseline':{'source_artifact_sha256':frozen113['artifact_serialized_sha256'],
                          'response_raw_sha256':EXPECTED_RAW_SHA256},
        'candidate':{'path':str(candidate_path),
            'serialized_sha256':sha256_file(candidate_path),
            'tensor_order':[f'R_{k}' for k in GRID],
            'shape':[128,2048],'dtype':'contiguous_cpu_float32'},
        'recovery':{'step':2,'role':'interruption_recovery_only',
                    'scientific_candidate':False,'delete_after_publication':True},
        'same_specimen_exploratory_method_development':True,
        'clean_heldout_validation':False}
    a122.prior.atomic_json_publish(manifest_path,construction)
    published_manifest_hash=sha256_file(manifest_path)
    frozen_artifact=torch.load(candidate_path,map_location='cpu',weights_only=True)
    if (list(frozen_artifact)!=[f'R_{k}' for k in GRID] or
            sha256_file(candidate_path)!=construction['candidate']['serialized_sha256']):
        raise ValueError('Published candidate inventory/serialized SHA256 mismatch')
    for k in GRID:
        response=a122.prior.validate_response(frozen_artifact[f'R_{k}'],torch)
        if a113.blind.sha256_raw_float32_tensor(response,torch)!=checkpoints[k]['response']['raw_sha256']:
            raise ValueError('Published functional response raw hash mismatch')
        digest=hashlib.sha256()
        for name,one in checkpoints[k]['state_per_matrix_sha256'].items():
            digest.update(name.encode()+b'\0'+one.encode('ascii')+b'\n')
        if digest.hexdigest()!=checkpoints[k]['state_aggregate_sha256']:
            raise ValueError('Published eligible-state aggregate hash mismatch')
    if json.loads(manifest_path.read_text())!=construction:
        raise ValueError('Published construction manifest changed')
    _,last_aggregate=a122.prior.state_hashes(eligible,a113.blind,torch)
    if last_aggregate!=checkpoints[4]['state_aggregate_sha256']:
        raise ValueError('Published final eligible-state hash mismatch')
    require_hash(candidate_path,construction['candidate']['serialized_sha256'])
    require_hash(manifest_path,published_manifest_hash)
    require_hash(SPEC_PATH,SPEC_SHA256)
    require_hash(Path(__file__),constructor_hash)
    recovery_path.unlink()
    print('CONSENSUS_RELAXATION_CANDIDATES_FROZEN',flush=True)

    # All historical access is below the publication, revalidation, and deletion barrier.
    del model,eligible,originals,final_mean,corpora,probe
    gc.collect()
    torch.cuda.empty_cache()
    reports,details,choice,later,loads,target_hash=evaluate_after_barrier(
        spec,frozen_artifact,raw_baseline,construction,a113,torch)
    result={'format_version':1,'attempt_id':ATTEMPT,'purpose':spec['purpose'],
            'same_specimen_exploratory_method_development':True,
            'clean_heldout_validation':False,
            'construction_manifest_sha256':sha256_file(manifest_path),
            'candidate_serialized_sha256':construction['candidate']['serialized_sha256'],
            'frozen_attempt113_raw_b_sha256':EXPECTED_RAW_SHA256,
            'matched_target_raw_sha256':target_hash,'model_load_seconds':loads,
            'reports':{'raw_b':reports['raw_b'],
                       **{str(k):reports[k] for k in GRID}},
            'finite_gains_vs_raw_b':details,
            'later_gains_vs_k1_and_position_counts':later,
            'development_set_interpretation':choice,
            'oracle_free_trajectory':rows}
    require_hash(candidate_path,construction['candidate']['serialized_sha256'])
    a122.prior.atomic_json_publish(path_of(spec['paths']['result_path']),result)
    print('Wrote Attempt124 result',flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true',
                        help='Resume only from the fixed accepted-step-2 recovery checkpoint')
    args=parser.parse_args(argv)
    try:
        run(resume=args.resume)
    except (OSError,RuntimeError,ValueError,KeyError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
