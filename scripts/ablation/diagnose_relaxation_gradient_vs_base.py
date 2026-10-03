#!/usr/bin/env python3
"""Attempt125: privileged parameter-gradient diagnostic after exact Attempt124 replay."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '125_privileged_relaxation_gradient_diagnostic'
SPEC_PATH = PROJECT/'experiments/attempts'/ATTEMPT/'spec.json'
SPEC_SHA256 = '88194b93367002fad1c1ca41ffd469bf1e443384940476f3b9c48cd3954dcd0a'
SOURCE124 = PROJECT/'scripts/ablation/run_diverse8_consensus_relaxation.py'
SOURCE124_SHA256 = '73b7110c4912aa54aba37c6888a17e3edd5f01b21ae1dc91369e456455b3df40'
ETAS = (0.10358343196094506, 0.10358343196094506)
DISPLACEMENTS = (0.07173115369504941, 0.06827297365533956)
STATE_HASHES = ('0e73fa5b3ab03193479fec051af9654528d2c3f159436ee8f622106ffc680eb3',
                '496cce3cc93221d02ac9e730e9b909b3e5fd0bc29ef6d687319f8c6260394ee2')


def load_helper124():
    # The source and all of its transitive frozen helpers are blind-only.
    from importlib.util import module_from_spec, spec_from_file_location
    from hashlib import sha256
    if sha256(SOURCE124.read_bytes()).hexdigest() != SOURCE124_SHA256:
        raise ValueError('Frozen Attempt124 source SHA256 mismatch')
    loader = spec_from_file_location('attempt125_frozen_attempt124', SOURCE124)
    module = module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


a124 = load_helper124()


def load_spec():
    a124.require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text())
    old = a124.load_spec()
    obj, replay = spec['objective'], spec['replay']
    frozen = spec['frozen_attempt124']
    if (spec['attempt_id'] != ATTEMPT or
            frozen['source_sha256'] != SOURCE124_SHA256 or
            frozen['spec_sha256'] != a124.SPEC_SHA256 or
            a124.path_of(frozen['source_path']) != SOURCE124 or
            a124.path_of(frozen['spec_path']) != a124.SPEC_PATH or
            obj['corpus_order'] != list(a124.ORDER) or
            obj['rows_per_corpus'] != [0, 512] or
            (obj['sequence_length'], obj['batch_size'], obj['batches_per_corpus'],
             obj['prediction_contexts_per_corpus'], obj['gradient_batches_per_pass']) !=
            (128, 8, 64, 65024, 512) or
            list(obj['endpoint_gradient_norms'].values()) != list(a124.EXPECTED_NORMS) or
            obj['endpoint_b_norm'] != a124.EXPECTED_B_NORM or
            obj['endpoint_b_aggregate_sha256'] != a124.EXPECTED_B_SHA256 or
            obj['weight_rule'] != old['objective']['endpoint_weight_rule'] or
            obj['weights_fixed_after_theta_F'] is not True or
            obj['later_gradient_renormalization'] is not False or
            replay['update_count'] != 2 or replay['accepted_etas'] != list(ETAS) or
            replay['cumulative_displacement_norms'] != list(DISPLACEMENTS) or
            replay['state_aggregate_sha256'] != list(STATE_HASHES) or
            replay['eligible_matrix_count'] != 98 or
            replay['update'] != 'W_new=W_current-eta*b_current' or
            replay['arithmetic'] != 'CPU_float64_update_then_one_FP32_storage_cast' or
            any(replay[key] is not False for key in
                ('line_search','alternative_eta','activation_probe','functional_candidate')) or
            spec['information_policy']['barrier_marker'] != 'RELAXATION_GRADIENTS_FROZEN' or
            spec['information_policy']['historical_base_access_only_after_replay_validation'] is not True or
            any(spec['information_policy'][key] is not False for key in
                ('historical_activation_oracle_access','adapter_access','true_delta_access',
                 'blind_candidate_construction','clean_heldout_validation')) or
            spec['workload'] != {'balanced_gradient_passes':4,'total_backward_batches':2048,
                'armijo_forward_batches':0,'activation_probe_batches':0,
                'jvp_operations':0,'ggn_operations':0,'cg_operations':0} or
            old['corpus_order'] != obj['corpus_order'] or
            old['generic_loss']['rows_per_corpus'] != obj['rows_per_corpus']):
        raise ValueError('Attempt125 frozen diagnostic plan mismatch')
    diagnostic = spec['diagnostic']
    if (diagnostic['strong'] != {'minimum_cosine':.70,'alpha_positive':True,
                                 'maximum_residual':math.sqrt(1-.70**2)} or
            diagnostic['moderate'] != {'minimum_cosine':.40,'alpha_positive':True} or
            diagnostic['best_by'] != 'larger_subtraction_direction_cosine_then_smaller_k'):
        raise ValueError('Attempt125 interpretation thresholds changed')
    result = a124.path_of(spec['paths']['result_path'])
    if (result.exists() or result.is_symlink() or
            result in {a124.path_of(frozen[key]) for key in
                       ('spec_path','source_path','construction_manifest_path','result_path')}):
        raise ValueError('Attempt125 result exists or overlaps frozen inputs')
    return spec, old, result


def validate_frozen_trajectory(spec, old124):
    frozen = spec['frozen_attempt124']
    for key in ('spec','source','construction_manifest'):
        a124.require_hash(a124.path_of(frozen[key+'_path']), frozen[key+'_sha256'])
    manifest = json.loads(a124.path_of(frozen['construction_manifest_path']).read_text())
    if (manifest['spec_sha256'] != frozen['spec_sha256'] or
            manifest['constructor_sha256'] != frozen['source_sha256'] or
            manifest['source_checkpoint_files'] != old124['final_checkpoint']['files'] or
            manifest['corpus_order'] != list(a124.ORDER) or
            manifest['generic_loss'] != old124['generic_loss'] or
            manifest['objective'] != old124['objective'] or
            manifest['endpoint_b_blind']['norm'] != a124.EXPECTED_B_NORM or
            manifest['endpoint_b_blind']['ordered_matrix_hash_sha256'] !=
                a124.EXPECTED_B_SHA256 or
            manifest['fixed_weights'] != a124.frozen_weights(
                spec['objective']['endpoint_gradient_norms']) or
            len(manifest['steps']) != 4 or
            sorted(manifest['checkpoints']) != ['1','2','4'] or
            [row['accepted_eta'] for row in manifest['steps'][:2]] != list(ETAS) or
            [row['cumulative_displacement_norm'] for row in manifest['steps'][:2]] !=
                list(DISPLACEMENTS) or
            [manifest['checkpoints'][str(k)]['state_aggregate_sha256'] for k in (1,2)] !=
                list(STATE_HASHES) or
            manifest['recovery']['scientific_candidate'] is not False):
        raise ValueError('Frozen Attempt124 construction contradicts replay controls')
    for k in (1,2):
        checkpoint = manifest['checkpoints'][str(k)]
        if (len(checkpoint['state_per_matrix_sha256']) != 98 or
                checkpoint['cumulative_displacement_norm'] != DISPLACEMENTS[k-1]):
            raise ValueError('Frozen Attempt124 eligible-state inventory mismatch')
    return manifest


def validate_replay_state(k, eligible, originals, manifest, helper113, torch):
    if k not in (1,2):
        raise ValueError('Only frozen replay steps 1 and 2 exist')
    record = manifest['checkpoints'][str(k)]
    displacement = a124.a122.prior.displacement_norm(eligible, originals, torch)
    per_matrix, aggregate = a124.a122.prior.state_hashes(eligible,helper113.blind,torch)
    if (not math.isclose(displacement,DISPLACEMENTS[k-1],rel_tol=0,abs_tol=1e-12) or
            aggregate != STATE_HASHES[k-1] or
            per_matrix != record['state_per_matrix_sha256']):
        raise ValueError(f'Attempt124 step-{k} replay failed before base access')
    return {'step':k,'accepted_eta':ETAS[k-1],
            'cumulative_displacement_norm':displacement,
            'state_aggregate_sha256':aggregate}


def _ordered_float64_arrays(vectors, inventory):
    names = [row['name'] for row in inventory]
    if len(names) != 98 or len(set(names)) != 98 or any(set(vec) != set(names) for vec in vectors):
        raise ValueError('Gradient coordinate inventory mismatch')
    for row in inventory:
        arrays=[]
        for vector in vectors:
            raw=np.asarray(vector[row['name']])
            if (raw.dtype != np.float32 or tuple(raw.shape) != tuple(row['shape']) or
                    not raw.flags.c_contiguous or not np.isfinite(raw).all()):
                raise ValueError('Malformed/nonfinite CPU FP32 gradient matrix')
            arrays.append(raw.astype(np.float64))
        yield arrays


def _sum_squares(value):
    return float(np.sum(value*value,dtype=np.float64))


def _sum_dot(left,right):
    return float(np.sum(left*right,dtype=np.float64))


def vector_diagnostics(b_f,b_1,b_2,b_base,inventory):
    """Global 98-matrix geometry; all differences and products are CPU FP64."""
    buckets={name:[] for name in ('ff','one_one','two_two','base_base',
             'f_base','one_base','two_base','d1_d1','d2_d2','star_star',
             'd1_star','d2_star','one_base_dist2','two_base_dist2')}
    vectors=(b_f,b_1,b_2,b_base)
    for f,one,two,base in _ordered_float64_arrays(vectors,inventory):
        d1,d2,star=f-one,f-two,f-base
        terms={'ff':_sum_squares(f),'one_one':_sum_squares(one),
               'two_two':_sum_squares(two),'base_base':_sum_squares(base),
               'f_base':_sum_dot(f,base),'one_base':_sum_dot(one,base),
               'two_base':_sum_dot(two,base),'d1_d1':_sum_squares(d1),
               'd2_d2':_sum_squares(d2),'star_star':_sum_squares(star),
               'd1_star':_sum_dot(d1,star),'d2_star':_sum_dot(d2,star),
               'one_base_dist2':_sum_squares(one-base),
               'two_base_dist2':_sum_squares(two-base)}
        for name,value in terms.items():
            buckets[name].append(value)
    q={name:math.fsum(values) for name,values in buckets.items()}
    if any(not math.isfinite(value) or value < 0 for name,value in q.items()
           if name not in ('f_base','one_base','two_base','d1_star','d2_star')):
        raise ValueError('Nonfinite gradient geometry')
    norm=lambda name:math.sqrt(q[name])
    cosine=lambda dot,left,right:None if left==0 or right==0 else dot/(left*right)
    nf,n1,n2,nb=map(norm,('ff','one_one','two_two','base_base'))
    nd1,nd2,ns=map(norm,('d1_d1','d2_d2','star_star'))
    if ns == 0:
        raise ValueError('True historical-base subtraction has zero norm')
    alphas={}
    for k in (1,2):
        denominator=q[f'd{k}_d{k}']
        alphas[k]=None if denominator==0 else q[f'd{k}_star']/denominator
    residual_parts={1:[],2:[]}
    for f,one,two,base in _ordered_float64_arrays(vectors,inventory):
        star=f-base
        for k,direction in ((1,f-one),(2,f-two)):
            if alphas[k] is not None:
                residual_parts[k].append(_sum_squares(star-alphas[k]*direction))
    subtraction={}
    for k,nd in ((1,nd1),(2,nd2)):
        alpha=alphas[k]
        residual=None if alpha is None else math.sqrt(math.fsum(residual_parts[k]))/ns
        subtraction[str(k)]={'norm':nd,
            'cosine_to_true_subtraction':cosine(q[f'd{k}_star'],nd,ns),
            'alpha':alpha,'residual':residual,
            'explained_fraction':None if residual is None else 1-residual**2,
            'distance_to_base_gradient_ratio':math.sqrt(q['one_base_dist2' if k==1 else 'two_base_dist2'])/ns}
    return {'gradient_norms':{'b_F':nf,'b_1':n1,'b_2':n2,'b_B':nb},
        'direct_gradient_cosines':{'b_F_b_B':cosine(q['f_base'],nf,nb),
            'b_1_b_B':cosine(q['one_base'],n1,nb),
            'b_2_b_B':cosine(q['two_base'],n2,nb)},
        'true_subtraction_norm':ns,'subtraction':subtraction}


def interpret(diagnostics):
    rows=diagnostics['subtraction']
    if set(rows) != {'1','2'}:
        raise ValueError('Only subtraction steps 1 and 2 are precommitted')
    score=lambda k:(-math.inf if rows[str(k)]['cosine_to_true_subtraction'] is None
                    else rows[str(k)]['cosine_to_true_subtraction'],-k)
    k=max((1,2),key=score)
    chosen=rows[str(k)]
    cosine,alpha,residual=(chosen['cosine_to_true_subtraction'],chosen['alpha'],
                            chosen['residual'])
    if (cosine is not None and cosine>=.70 and alpha is not None and alpha>0 and
            residual is not None and residual<=math.sqrt(1-.70**2)):
        category='strong_support'
        decision='worth_constructing_blind_J_of_b_F_minus_b_k_candidate'
    elif cosine is not None and cosine>=.40 and alpha is not None and alpha>0:
        category='moderate_support'
        decision='partial_or_ambiguous_mechanism_inspect_before_functional_candidate'
    else:
        category='no_useful_support'
        decision='abandon_relaxation_derived_g0_subtraction_for_now'
    return {'best_k':k,'classification':category,'scientific_decision':decision,
            'same_specimen_privileged_development':True,'heldout_validation':False}


def run():
    import torch
    spec,old124,result_path=load_spec()
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for the frozen FP32 gradient convention')
    manifest=validate_frozen_trajectory(spec,old124)
    frozen113=old124['frozen_attempt113']
    a113=a124.import_pinned(a124.path_of(frozen113['source_path']),
                            frozen113['source_sha256'],'attempt125_frozen_attempt113')
    old113,manifest113=a124.validate_blind_controls(old124,a113)
    if manifest['endpoint_b_blind'] != manifest113['b_blind']:
        raise ValueError('Attempt124/113 endpoint records disagree')
    corpora,probe,corpus_manifests=a113.validate_blind_sources(old113)
    del probe  # Validation covers the full frozen probe; this diagnostic never evaluates it.
    model_dir=a113.safe_input_path(old113['final_checkpoint']['directory'])
    model=a113.blind.load_local_model(model_dir,'cuda',torch)
    eligible=a113.blind.discover_eligible_linear_weights(model,torch)
    selected=a113.blind.freeze_other_parameters(model,eligible)
    inventory=a113.coordinate_inventory(eligible)
    if inventory != manifest113['coordinate_inventory']:
        raise ValueError('Final eligible-coordinate inventory differs from Attempt113')
    frozen_before=a113.blind.frozen_parameter_hashes(model,selected,torch)
    originals=a124.a122.prior.original_weights(eligible,torch)
    a124.initial_weight_norm(eligible,torch)
    b_f,endpoint_diagnostics,endpoint_record=a113.build_blind_rhs(
        model,corpora,eligible,inventory)
    a124.validate_endpoint(endpoint_record,endpoint_diagnostics,manifest113)
    if endpoint_record != manifest['endpoint_b_blind']:
        raise ValueError('Recomputed b_F differs from Attempt124 endpoint')
    endpoint_norms={row['name']:row['global_gradient_norm'] for row in endpoint_diagnostics}
    fixed=a124.frozen_weights(endpoint_norms)
    if fixed != manifest['fixed_weights']:
        raise ValueError('Frozen endpoint weights differ from Attempt124')
    replay=[]
    a124.update_from_vector(eligible,b_f,ETAS[0],torch)
    replay.append(validate_replay_state(1,eligible,originals,manifest,a113,torch))
    b_1,losses1,norms1,norm1=a124.weighted_gradient_pass(
        model,corpora,eligible,inventory,endpoint_norms,a113,torch)
    a124.update_from_vector(eligible,b_1,ETAS[1],torch)
    replay.append(validate_replay_state(2,eligible,originals,manifest,a113,torch))
    b_2,losses2,norms2,norm2=a124.weighted_gradient_pass(
        model,corpora,eligible,inventory,endpoint_norms,a113,torch)
    for index,(losses,norm) in enumerate(((losses1,norm1),(losses2,norm2)),start=1):
        row=manifest['steps'][index]
        if (not math.isclose(norm,row['aggregate_gradient_norm_before_update'],
                             rel_tol=0,abs_tol=1e-12) or
                not math.isclose(a124.weighted_objective(losses,fixed),
                                 row['weighted_objective_before_update'],
                                 rel_tol=0,abs_tol=1e-12)):
            raise ValueError(f'Attempt124 gradient at replayed theta_{index} changed')
    a113.blind.verify_frozen_parameters(model,selected,frozen_before,torch)
    a113._input_hashes_unchanged(old113)
    if a113.blind.checkpoint_file_records(model_dir) != old124['final_checkpoint']['files']:
        raise ValueError('Final checkpoint changed during replay')
    a124.require_hash(a124.path_of(spec['frozen_attempt124']['construction_manifest_path']),
                      spec['frozen_attempt124']['construction_manifest_sha256'])
    a124.require_hash(SPEC_PATH,SPEC_SHA256)
    b1_hash=a113.ordered_vector_hashes(b_1,inventory)[1]
    b2_hash=a113.ordered_vector_hashes(b_2,inventory)[1]
    del model,eligible,originals,selected,frozen_before
    gc.collect()
    torch.cuda.empty_cache()
    print('RELAXATION_GRADIENTS_FROZEN',flush=True)

    # The known-base spec, checkpoint, and model are accessed only below this line.
    known_ref=spec['frozen_attempt015']
    a124.require_hash(a124.path_of(known_ref['spec_path']),known_ref['spec_sha256'])
    known=a124.import_pinned(a124.path_of(known_ref['source_path']),
                             known_ref['source_sha256'],'attempt125_frozen_attempt015')
    known_spec=known.load_spec(a124.path_of(known_ref['spec_path']))
    a124.require_hash(Path(known.a.__file__),
        known_spec['frozen_attempt014']['constructor']['sha256'])
    base_dir=known.resolve(known_spec['defaults']['base_directory'])
    if (known_spec['base']['repo_id'] != known_ref['base_repo_id'] or
            known_spec['base']['revision'] != known_ref['base_revision'] or
            known_spec['canonical_checkpoint_files'] != old124['final_checkpoint']['files'] or
            known_spec['model'] != old124['model'] or
            known_spec['eligible_tensors'] != old124['eligible_tensors'] or
            known.a.checkpoint_file_records(base_dir) != known_spec['base']['files']):
        raise ValueError('Attempt015 historical-base provenance mismatch')
    base=known.a.load_local_model(base_dir,'cuda',torch)
    base_eligible=known.a.discover_eligible_linear_weights(base,torch)
    if a113.coordinate_inventory(base_eligible) != inventory:
        raise ValueError('Historical base eligible-coordinate inventory mismatch')
    base_selected=known.a.freeze_other_parameters(base,base_eligible)
    base_before=known.a.frozen_parameter_hashes(base,base_selected,torch)
    _,base_state_before=a124.a122.prior.state_hashes(base_eligible,a113.blind,torch)
    b_base,base_losses,base_norms,base_gradient_norm=a124.weighted_gradient_pass(
        base,corpora,base_eligible,inventory,endpoint_norms,a113,torch)
    known.a.verify_frozen_parameters(base,base_selected,base_before,torch)
    _,base_state_after=a124.a122.prior.state_hashes(base_eligible,a113.blind,torch)
    if (base_state_after != base_state_before or
            known.a.checkpoint_file_records(base_dir) != known_spec['base']['files']):
        raise ValueError('Historical base changed during gradient computation')
    del base,base_eligible,base_selected,base_before
    gc.collect()
    torch.cuda.empty_cache()
    geometry=vector_diagnostics(b_f,b_1,b_2,b_base,inventory)
    # Attempt113's frozen norm sums FP32 products in FP64. Geometry multiplies
    # FP64 values, so compare the two conventions without claiming bit equality.
    if (not math.isclose(geometry['gradient_norms']['b_F'],a124.EXPECTED_B_NORM,
                         rel_tol=1e-6,abs_tol=1e-8) or
            any(not math.isclose(geometry['gradient_norms'][name],expected,
                                 rel_tol=1e-6,abs_tol=1e-8) for name,expected in
                (('b_1',norm1),('b_2',norm2),('b_B',base_gradient_norm)))):
        raise ValueError('Global FP64 gradient norm audit mismatch')
    bbase_hash=a113.ordered_vector_hashes(b_base,inventory)[1]
    interpretation=interpret(geometry)
    del b_f,b_1,b_2,b_base,corpora
    gc.collect()
    result={'format_version':1,'attempt_id':ATTEMPT,'purpose':spec['purpose'],
        'privileged_mechanism_diagnostic':True,'blind_candidate_constructed':False,
        'same_specimen_exploratory_method_development':True,
        'clean_heldout_validation':False,
        'provenance':{'spec_sha256':SPEC_SHA256,
            'source_sha256':a124.sha256_file(Path(__file__)),
            'attempt124_spec_sha256':spec['frozen_attempt124']['spec_sha256'],
            'attempt124_source_sha256':spec['frozen_attempt124']['source_sha256'],
            'attempt124_construction_manifest_sha256':
                spec['frozen_attempt124']['construction_manifest_sha256'],
            'attempt015_spec_sha256':known_ref['spec_sha256'],
            'attempt015_source_sha256':known_ref['source_sha256'],
            'base_checkpoint_files':known_spec['base']['files'],
            'final_checkpoint_files':old124['final_checkpoint']['files'],
            'corpus_manifests':corpus_manifests},
        'fixed_objective':{'corpus_order':list(a124.ORDER),
            'rows_per_corpus':[0,512],'endpoint_gradient_norms':endpoint_norms,
            'fixed_weights':fixed,'endpoint_b_norm':a124.EXPECTED_B_NORM,
            'endpoint_b_aggregate_sha256':a124.EXPECTED_B_SHA256},
        'gradient_aggregate_sha256':{'b_F':a124.EXPECTED_B_SHA256,
            'b_1':b1_hash,'b_2':b2_hash,'b_B':bbase_hash},
        'replay':replay,
        'generic_losses':{'theta_F':{r['name']:r['mean_generic_loss'] for r in endpoint_diagnostics},
            'theta_1':losses1,'theta_2':losses2,'theta_B':base_losses},
        'current_per_corpus_gradient_norms_diagnostic_only':{
            'theta_1':norms1,'theta_2':norms2,'theta_B':base_norms},
        'geometry':geometry,'interpretation':interpretation,
        'workload':spec['workload']}
    a124.require_hash(SPEC_PATH,SPEC_SHA256)
    a124.require_hash(a124.path_of(spec['frozen_attempt124']['construction_manifest_path']),
                      spec['frozen_attempt124']['construction_manifest_sha256'])
    if result_path.exists() or result_path.is_symlink():
        raise ValueError('Attempt125 result already exists')
    a124.a122.prior.atomic_json_publish(result_path,result)
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
