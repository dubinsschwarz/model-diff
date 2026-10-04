"""Synthetic controls for Attempt126; no model or historical oracle is loaded."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'scripts/ablation/run_fineweb_surprisal_stratification.py'
loader=importlib.util.spec_from_file_location('attempt126_test',SOURCE)
a=importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)


def reports(primary,secondary):
    return {'positions_1_4_mean_cosine':primary,
            'positions_1_127_mean_cosine':secondary}


def test_frozen_population_provenance_and_exact_workload():
    spec,_=a.load_spec()
    fineweb=spec['frozen_attempt005']
    manifest=json.loads(a.path_of(fineweb['corpus_manifest_path']).read_text())
    assert a.sha256_file(a.path_of(fineweb['corpus_manifest_path']))==fineweb['corpus_manifest_sha256']
    assert a.sha256_file(a.path_of(fineweb['corpus_spec_path']))==fineweb['corpus_spec_sha256']
    assert fineweb['tokens_path']=='/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt'
    assert fineweb['serialized_sha256']==manifest['artifact']['serialized_sha256']
    assert fineweb['raw_tensor_sha256']==manifest['artifact']['raw_tensor_sha256']
    assert manifest['tensor']['shape']==[4096,128]
    assert spec['ranking']['population_rows']==[0,4096]
    assert spec['ranking']['prediction_tokens_per_row']==127
    assert spec['ranking']['number_of_forward_batches']==512
    assert spec['workload']=={'ranking_forward_batches':512,
        'gradient_backward_batches':384,'jvp_probe_batches':96,
        'jvp_first_batch_hook_audits':3,'matched_target_forward_batches':64,
        'ggn_operations':0,'cg_operations':0}
    assert spec['gradient_loss']['sample_count']==1024
    assert spec['gradient_loss']['batch_size']==8
    assert spec['gradient_loss']['number_of_batches']==128
    assert spec['gradient_loss']['total_prediction_tokens']==130048
    assert spec['eligible_tensors']['expected_matrix_count']==98
    assert spec['tangent']['direction']=='delta = +alpha * g'
    assert spec['tangent']['target_relative_frobenius']==.00125
    assert spec['tangent']['normalization']=='one_global_alpha_across_all_eligible_matrices'
    assert spec['probe']['full_source_rows']==10000
    assert spec['probe']['selected_rows']==[0,1024]
    assert spec['probe']['selected_batch_count']==32
    assert spec['jvp']['strict_forward_ad'] is True
    assert spec['jvp']['fallback'] is False
    assert spec['candidate']['tensor_order']==list(a.RESPONSE_NAMES)==[
        'response_low','response_middle','response_high']
    assert spec['candidate']['shape']==[128,2048]
    assert spec['candidate']['dtype']=='contiguous_cpu_float32'
    for name in ('response_normalization','sign_selection','interpolation',
                 'reweighting','combination'):
        assert spec['candidate'][name] is False
    assert spec['evaluation']['target_raw_sha256']==a.TARGET_SHA256


def test_blind_provenance_linkage_without_loading_model_or_real_tokens():
    spec,_=a.load_spec()
    class Tensor:
        def __init__(self,shape):self.shape=shape
        def is_contiguous(self):return True
        def __getitem__(self,key):return Tensor((1024,128))
        def contiguous(self):return self
    class Helper:
        def checkpoint_file_records(self,path):return spec['final_checkpoint']['files']
        def validate_frozen_corpus(self,record,model_dir,torch):
            assert record['tokens_path']==spec['frozen_attempt005']['tokens_path']
            return Tensor((4096,128)),{},{}
        def load_probe(self,path,settings,torch):
            assert settings['sample_count']==10000
            return Tensor((10000,128))
    tokens,probe,_,_,_=a.validate_blind_inputs(spec,Helper(),None)
    assert tokens.shape==(4096,128) and probe.shape==(1024,128)


def test_float64_per_sequence_127_token_nll_and_batch_independence():
    token_nll=np.arange(4096*127,dtype=np.float32).reshape(4096,127)/10000
    whole=a.sequence_mean_nll(token_nll)
    whole_rank=a.rank_and_stratify(whole)['rank_raw_int64_sha256']
    for batch in (1,8,32,511):
        pieces=np.concatenate([a.sequence_mean_nll(token_nll[start:start+batch])
                               for start in range(0,4096,batch)])
        assert np.array_equal(pieces,whole)
        assert a.rank_and_stratify(pieces)['rank_raw_int64_sha256']==whole_rank
    assert whole.dtype==np.float64
    assert whole[0]==pytest.approx(math.fsum(float(x) for x in token_nll[0])/127)
    for bad in (np.ones((2,126),dtype=np.float32),
                np.array([[float('nan')]*127],dtype=np.float32),
                np.ones((0,127),dtype=np.float32)):
        with pytest.raises(ValueError):a.sequence_mean_nll(bad)


def test_tie_order_exact_rank_gaps_disjoint_hashes_and_statistics():
    values=np.arange(4096,dtype=np.float64)
    values[:5]=3.0
    record=a.rank_and_stratify(values)
    assert record['ascending_rank_original_row_indices'][:5]==[0,1,2,3,4]
    assert record['surprisal_raw_float64_sha256']==a.raw_float64_hash(values)
    assert record['rank_raw_int64_sha256']==a.raw_int64_hash(
        np.array(record['ascending_rank_original_row_indices'],dtype=np.int64))
    assert record['unused_rank_intervals']==[[1024,1536],[2560,3072]]
    assert [record['strata'][name]['rank_range'] for name in a.NAMES]==[
        [0,1024],[1536,2560],[3072,4096]]
    groups=[set(record['strata'][name]['original_row_indices']) for name in a.NAMES]
    assert all(len(group)==1024 for group in groups)
    assert not groups[0]&groups[1] and not groups[0]&groups[2] and not groups[1]&groups[2]
    assert len(set.union(*groups))==3072
    for name in a.NAMES:
        row=record['strata'][name]
        selected=np.array(row['original_row_indices'],dtype=np.int64)
        assert row['ordered_row_indices_raw_sha256']==a.raw_int64_hash(selected)
        sample=values[selected]
        assert row['surprisal_min']==sample.min()
        assert row['surprisal_max']==sample.max()
        assert row['surprisal_mean']==pytest.approx(sample.mean())
        assert row['surprisal_median']==pytest.approx(np.median(sample))
        assert row['surprisal_population_std']==pytest.approx(np.std(sample,ddof=0))
    assert a.validate_ranking(record)
    damaged=copy.deepcopy(record)
    damaged['strata']['high']['original_row_indices'][0]=0
    with pytest.raises(ValueError):a.validate_ranking(damaged)


def test_global_positive_tangent_scale_rule_and_no_example_normalization():
    weight_norm,gradient_norm=918.1587250300843,7.5
    target=.00125*weight_norm
    scale={'aggregate_source_weight_norm':weight_norm,
           'aggregate_generic_gradient_norm':gradient_norm,
           'target_delta_norm':target,'alpha':target/gradient_norm}
    assert a.validate_global_scale(scale)==scale
    assert scale['alpha']>0
    assert scale['alpha']*gradient_norm==pytest.approx(.00125*weight_norm)
    with pytest.raises(ValueError):a.validate_global_scale({**scale,'alpha':-scale['alpha']})
    with pytest.raises(ValueError):a.validate_global_scale({**scale,'alpha':scale['alpha']*2})
    source=SOURCE.read_text()
    assert 'a14.accumulate_mean_generic_gradient(' in source
    assert 'a14.global_tangent_scale(eligible,.00125,torch)' in source
    assert 'a14.prepare_tangents(eligible,scale,torch)' in source
    assert 'per_example_gradient' not in source
    assert 'normalize_examples' not in source


def test_signed_historical_scores_and_precommitted_support_categories():
    validator=SimpleNamespace(position_cosines=lambda left,right:
        [-.8]+[-.2,.1,.3,.6]+[.5]*123)
    metric=a.a122.score_report(object(),object(),validator)
    assert metric['position_0_cosine']==-.8
    assert metric['positions_1_4_individual_cosines']==[-.2,.1,.3,.6]
    assert metric['positions_1_4_mean_cosine']==pytest.approx(.2)
    assert metric['positions_1_127_mean_cosine']==pytest.approx(sum(metric['all_128_position_cosines'][1:])/127)
    assert metric['all_128_position_cosines'][0]<0
    clear={'response_low':reports(.40,.40),'response_middle':reports(.42,.42),
           'response_high':reports(.44,.44)}
    delta,category=a.differences_and_classification(clear)
    assert category=='clear_support'
    assert delta['high_minus_low']['primary']==pytest.approx(.04)
    assert delta['high_minus_middle']['secondary']==pytest.approx(.02)
    assert delta['middle_minus_low']['primary']==pytest.approx(.02)
    partial={'response_low':reports(.40,.40),'response_middle':reports(.45,.41),
             'response_high':reports(.44,.42)}
    assert a.differences_and_classification(partial)[1]=='partial_support'
    no={'response_low':reports(.40,.40),'response_middle':reports(.41,.41),
        'response_high':reports(.42,.39)}
    assert a.differences_and_classification(no)[1]=='no_support'
    with pytest.raises(ValueError):a.differences_and_classification({**clear,'response_extra':reports(1,1)})


def test_descriptive_three_point_surprisal_vs_gradient_norm_correlation():
    values=np.arange(4096,dtype=np.float64)
    ranking=a.rank_and_stratify(values)
    gradients={name:{'aggregate_generic_gradient_norm':gradient}
               for name,gradient in zip(a.NAMES,(3.,2.,1.))}
    scored={name:reports(i/10,i/5) for i,name in enumerate(a.RESPONSE_NAMES)}
    result=a.descriptive_correlations(ranking,gradients,scored)
    assert result['n_strata']==3 and result['descriptive_only'] is True
    assert result['pearson']['mean_surprisal']['primary']==pytest.approx(1.)
    assert result['pearson']['gradient_norm']['primary']==pytest.approx(-1.)
    assert result['pearson']['gradient_norm']['secondary']==pytest.approx(-1.)
    assert result['predictor_values']['gradient_norm']==[3.,2.,1.]
    with pytest.raises(ValueError):a.pearson_three([1,2],[1,2])


def test_freeze_barrier_and_no_prebarrier_historical_access():
    source=SOURCE.read_text()
    functions={node.name:ast.get_source_segment(source,node) for node in ast.parse(source).body
               if isinstance(node,ast.FunctionDef)}
    run=functions['run']
    assert run.index('score_endpoint_surprisal(model,tokens,torch)') < run.index('rank_and_stratify(surprisal)')
    assert run.index('rank_and_stratify(surprisal)') < run.index('for name in NAMES:')
    assert run.index('a14.accumulate_mean_generic_gradient(') < run.index('a14.compute_probe_response(')
    assert run.index('a14.compute_probe_response(') < run.index('a122.prior.atomic_torch_publish(candidate_path')
    assert run.index('a122.prior.atomic_torch_publish(candidate_path') < run.index('a122.prior.atomic_json_publish(manifest_path')
    assert run.index('a122.prior.atomic_json_publish(manifest_path') < run.index('validate_published(candidate_path')
    assert run.index('validate_published(candidate_path') < run.index("print('SURPRISAL_STRATA_CANDIDATES_FROZEN'")
    assert run.index("print('SURPRISAL_STRATA_CANDIDATES_FROZEN'") < run.index('evaluate_after_barrier(')
    pre=run[:run.index("print('SURPRISAL_STRATA_CANDIDATES_FROZEN'")]
    for forbidden in ('evaluate_after_barrier(','matched_target','old_context','base_mean',
                      'frozen_attempt123','evaluation_path'):
        assert forbidden not in pre
    assert 'torch.inference_mode()' in functions['score_endpoint_surprisal']
    assert 'torch.autocast(device_type=device.type,enabled=False)' in functions['score_endpoint_surprisal']
    assert 'logits[:,:-1,:]' in functions['score_endpoint_surprisal']
    assert 'batch[:,1:]' in functions['score_endpoint_surprisal']
    assert "reduction='none'" in functions['score_endpoint_surprisal']
    assert 'score_endpoint_surprisal' not in functions['evaluate_after_barrier']
    assert 'a123.verified_target(' in functions['evaluate_after_barrier']
    assert 'a122.score_report(' in functions['evaluate_after_barrier']
    assert 'frozen_attempt100' in functions['evaluate_after_barrier']
    assert 'candidate' not in functions['differences_and_classification']


def test_published_candidate_exact_inventory_and_hashes(tmp_path,monkeypatch):
    ranking=a.rank_and_stratify(np.arange(4096,dtype=np.float64))
    artifact={name:object() for name in a.RESPONSE_NAMES}
    construction={'candidate':{'serialized_sha256':'placeholder',
                               'tensor_order':list(a.RESPONSE_NAMES)},
                  'responses':{name:{'raw_sha256':'good'} for name in a.RESPONSE_NAMES},
                  'ranking':ranking,
                  'barrier':{'manifest_contains_historical_scores':False}}
    candidate=tmp_path/'candidate.pt'
    candidate.write_bytes(b'synthetic')
    construction['candidate']['serialized_sha256']=hashlib.sha256(candidate.read_bytes()).hexdigest()
    manifest=tmp_path/'construction-manifest.json'
    manifest.write_text(json.dumps(construction))
    fake=SimpleNamespace(load=lambda path,**kwargs:artifact)
    a14=SimpleNamespace(sha256_raw_float32_tensor=lambda value,torch:'good')
    with monkeypatch.context() as patch:
        patch.setattr(a.a122.prior,'validate_response',lambda value,torch:value)
        assert a.validate_published(candidate,manifest,construction,fake,a14)==artifact
        bad={**artifact,'response_extra':object()}
        fake.load=lambda path,**kwargs:bad
        with pytest.raises(ValueError,match='inventory'):
            a.validate_published(candidate,manifest,construction,fake,a14)
        fake.load=lambda path,**kwargs:artifact
        a14.sha256_raw_float32_tensor=lambda value,torch:'bad'
        with pytest.raises(ValueError,match='SHA256'):
            a.validate_published(candidate,manifest,construction,fake,a14)


def test_overwrite_refusal_and_fixed_output_inventory(tmp_path,monkeypatch):
    spec,_=a.load_spec()
    changed=copy.deepcopy(spec)
    changed['paths']={key:str(tmp_path/(key+'.json')) for key in
                      ('candidate_path','construction_manifest_path','result_path')}
    temporary=tmp_path/'spec.json'
    temporary.write_text(json.dumps(changed))
    with monkeypatch.context() as patch:
        patch.setattr(a,'SPEC_PATH',temporary)
        patch.setattr(a,'SPEC_SHA256',hashlib.sha256(temporary.read_bytes()).hexdigest())
        assert len(a.load_spec()[1])==3
        Path(changed['paths']['candidate_path']).write_bytes(b'existing')
        with pytest.raises(ValueError,match='output exists'):
            a.load_spec()
