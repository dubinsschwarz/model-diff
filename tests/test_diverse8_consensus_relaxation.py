"""Synthetic controls for Attempt124's fixed diverse-8 objective and barrier."""
import ast
import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'scripts/ablation/run_diverse8_consensus_relaxation.py'
loader=importlib.util.spec_from_file_location('attempt124',SOURCE)
a=importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)


class Tensor:
    def __init__(self,value,dtype=np.float32):
        self.array=np.array(value,dtype=dtype,copy=True)
        self.device=SimpleNamespace(type='cpu')
        self.grad=None
    @property
    def dtype(self):return self.array.dtype.type
    @property
    def shape(self):return self.array.shape
    def detach(self):return self
    def to(self,device=None,dtype=None,copy=False):
        if device in (np.float32,np.float64):dtype=device
        return Tensor(self.array,self.dtype if dtype is None else dtype)
    def contiguous(self):return self
    def is_contiguous(self):return True
    def copy_(self,other):self.array[...]=other.array;return self
    def __mul__(self,other):return Tensor(self.array*other,self.dtype)
    __rmul__=__mul__
    def __sub__(self,other):return Tensor(self.array-other.array,self.dtype)
    def numpy(self):return self.array


class TinyTorch:
    Tensor=Tensor
    float32=np.float32
    float64=np.float64
    no_grad=staticmethod(nullcontext)
    isfinite=staticmethod(lambda value:np.isfinite(value.array))
    equal=staticmethod(lambda x,y:np.array_equal(x.array,y.array))


def eligible_98(value=1.):
    return [(f'model.layers.{i}.linear',SimpleNamespace(weight=Tensor([[value]])))
            for i in range(98)]


def report(primary,secondary):
    return {'positions_1_4_mean_cosine':primary,
            'positions_1_127_mean_cosine':secondary}


def test_frozen_eight_corpus_objective_and_pilot_grid():
    spec=a.load_spec()
    old=json.loads(a.path_of(spec['frozen_attempt113']['spec_path']).read_text())
    assert a.ORDER==('fineweb','wikitext103_raw','tinystories','arxiv_document',
                     'cc_news','pg19','codeparrot_clean','ultrachat')
    assert spec['corpus_order']==list(a.ORDER)==old['corpus_order']
    assert spec['corpora']==old['corpora']
    assert all(row['selected_rows']==[0,512] and row['frozen_shape']==[4096,128]
               for row in spec['corpora'])
    assert spec['generic_loss']['batches_per_corpus']==64
    assert spec['generic_loss']['prediction_contexts_per_corpus']==65024
    assert spec['generic_loss']['batch_size']==8
    assert spec['relaxation']['steps']==a.STEPS==4
    assert spec['relaxation']['functional_checkpoints']==list(a.GRID)==[1,2,4]
    assert spec['relaxation']['recovery_step']==a.RECOVERY_STEP==2
    assert spec['workload']['total_backward_batches']==2048
    assert spec['workload']['forward_batches_per_armijo_trial']==512
    assert spec['relaxation']['line_search']==a.LINE_SEARCH
    assert a.LINE_SEARCH['trial_forward_batches']==512
    assert 'eight_corpus' in a.LINE_SEARCH['trial_objective']
    assert a.a122.ARMIJO_C==1e-4 and a.a122.BACKTRACK_FACTOR==.5
    assert a.a122.MAX_TRIALS==8
    for key in ('optimizer','momentum','weight_decay','gradient_clipping',
                'mixed_precision','early_stopping','response_normalization',
                'response_rescaling','sign_selection'):
        assert spec['relaxation'][key] is False
    assert spec['objective']['weights_frozen_after_endpoint'] is True
    assert spec['objective']['later_per_corpus_gradient_renormalization'] is False
    weights=a.frozen_weights(dict(zip(a.ORDER,a.EXPECTED_NORMS)))
    assert list(weights)==list(a.ORDER)
    for name,norm in zip(a.ORDER,a.EXPECTED_NORMS):
        assert weights[name]==1/(8*norm)
    assert spec['probe']['rows']==[0,1024]
    assert spec['probe']['batch_size']==32
    assert spec['workload']['probe_mean_batches']==32
    assert spec['workload']['probe_mean_count']==4
    assert spec['barrier']['recovery_checkpoint_scientific_candidate'] is False
    assert spec['information_policy']['historical_base_access_before_barrier'] is False
    assert spec['information_policy']['clean_heldout_validation'] is False


def test_frozen_113_controls_and_endpoint_equality_gate():
    spec=a.load_spec()
    frozen=spec['frozen_attempt113']
    manifest=json.loads(a.path_of(frozen['construction_manifest_path']).read_text())
    assert a.sha256_file(a.path_of(frozen['spec_path']))==frozen['spec_sha256']
    assert a.sha256_file(a.path_of(frozen['source_path']))==frozen['source_sha256']
    assert a.sha256_file(a.path_of(frozen['construction_manifest_path']))==frozen['construction_manifest_sha256']
    assert a.EXPECTED_NORMS==tuple(row['global_gradient_norm'] for row in manifest['per_corpus_gradients'])
    assert a.EXPECTED_B_NORM==manifest['b_blind']['norm']==.6924963677590905
    assert a.EXPECTED_B_SHA256==manifest['b_blind']['ordered_matrix_hash_sha256']
    assert a.EXPECTED_RAW_SHA256==manifest['functional_responses']['response_raw_rhs']['raw_sha256']
    assert frozen['artifact_serialized_sha256']==manifest['artifact']['serialized_sha256']
    rhs=manifest['b_blind']
    diagnostics=manifest['per_corpus_gradients']
    assert a.validate_endpoint(rhs,diagnostics,manifest)
    for change in (
        lambda r,d,m:r.update(norm=.1),
        lambda r,d,m:r.update(ordered_matrix_hash_sha256='0'*64),
        lambda r,d,m:d[0].update(global_gradient_norm=12.),
        lambda r,d,m:r['matrix_raw_sha256'].update({next(iter(r['matrix_raw_sha256'])):'0'*64}),
    ):
        r,d,m=copy.deepcopy(rhs),copy.deepcopy(diagnostics),copy.deepcopy(manifest)
        change(r,d,m)
        with pytest.raises(ValueError,match='endpoint b_blind'):
            a.validate_endpoint(r,d,m)


def test_three_corpus_fixed_endpoint_weights_and_stable_quadratic_trajectory():
    order=('a','b','c')
    endpoint_norms={'a':2.,'b':4.,'c':8.}
    weights=a.weights_from_endpoint(order,endpoint_norms)
    assert weights=={'a':1/6,'b':1/12,'c':1/24}
    endpoint_grad={'a':2.,'b':4.,'c':8.}
    assert sum(weights[name]*endpoint_grad[name] for name in order)==pytest.approx(1.)
    later_grad={'a':1.,'b':2.,'c':4.}
    assert sum(weights[name]*later_grad[name] for name in order)==pytest.approx(.5)
    # Renormalizing the later gradients would give 1.0; fixed weights give 0.5.
    assert sum(later_grad[name]/abs(later_grad[name])/3 for name in order)==1.
    x,eta=1.,3.
    accepted=[]
    for step in range(1,5):
        pre=x
        grad=sum(weights[name]*(2**(index+1))*pre
                 for index,name in enumerate(order))
        objective=lambda z:.5*z*z
        state={'x':pre,'restores':0,'evaluations':0}
        def apply(trial_eta):
            assert state['x']==pre
            state['x']=pre-trial_eta*grad
        def evaluate():
            state['evaluations']+=1
            return objective(state['x'])
        def restore():
            state['x']=pre
            state['restores']+=1
        result=a.a122.armijo_search(objective(pre),abs(grad),eta,apply,evaluate,restore)
        assert result['initial_trial_eta']==eta
        assert result['accepted_post_step_generic_mean_ce']<=objective(pre)-1e-4*result['accepted_eta']*grad*grad
        assert result['accepted_eta']<=eta
        assert state['evaluations']==len(result['trials'])
        assert state['restores']==result['accepted_trial_index']
        accepted.append(result)
        x,eta=state['x'],result['accepted_eta']
    assert [row['accepted_trial_index'] for row in accepted]==[1,0,0,0]
    assert [row['accepted_eta'] for row in accepted]==[1.5]*4
    assert x==pytest.approx(.0625)
    with pytest.raises(ValueError):
        a.weights_from_endpoint(order,{'a':2.,'b':0.,'c':8.})


def test_armijo_eighth_rejection_fails_without_a_ninth_trial():
    state={'x':1.,'trials':0,'restores':0}
    def apply(eta):
        assert state['x']==1.
        state['x']=1.-eta
    def trial():
        state['trials']+=1
        return 2.
    def restore():
        state['x']=1.
        state['restores']+=1
    with pytest.raises(ValueError,match='eight predeclared'):
        a.a122.armijo_search(1.,1.,1.,apply,trial,restore)
    assert state=={'x':1.,'trials':8,'restores':8}


def test_later_weighted_gradient_uses_endpoint_norms_not_current_norms(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(a,'ORDER',('a','b','c'))
        calls=[]
        current={'a':4.,'b':1.,'c':16.}
        endpoint={'a':2.,'b':4.,'c':8.}
        progress=[]
        class Tokens:
            def __init__(self,name):self.name=name
            def __getitem__(self,_key):return self
            def contiguous(self):return self
        class Model:
            def zero_grad(self,**_kwargs):pass
        class Helper:
            @staticmethod
            def accumulate_mean_generic_gradient(_model,tokens,_eligible,loss_spec,_torch):
                assert loss_spec['sample_count']==512
                assert loss_spec['number_of_batches']==64
                progress.append(tokens.name)
                return 1.0
            @staticmethod
            def add_unit_gradient_to_consensus(combined,_eligible,norm,_inventory,*,corpus_index,corpus_count):
                name=a.ORDER[corpus_index]
                assert corpus_count==8
                calls.append((name,norm))
                combined[name]=current[name]/norm/8
            @staticmethod
            def matrixwise_dot(x,y):
                return sum(value*value for value in x.values())
        patch.setattr(a.a122.prior,'gradient_norm',lambda _eligible,_torch:current[progress[-1]])
        helper=SimpleNamespace(blind=Helper,
            add_unit_gradient_to_consensus=Helper.add_unit_gradient_to_consensus,
            matrixwise_dot=Helper.matrixwise_dot)
        vector,losses,norms,norm=a.weighted_gradient_pass(
            Model(),{name:Tokens(name) for name in a.ORDER},[],[],endpoint,helper,TinyTorch)
        assert calls==list(endpoint.items())
        assert norms==current
        assert vector=={'a':.25,'b':.03125,'c':.25}
        assert norm==pytest.approx(math.sqrt(sum(x*x for x in vector.values())))
        assert list(losses)==['a','b','c']


def test_update_sign_only_eligible_and_exact_restore():
    eligible=eligible_98(1.)
    forbidden=Tensor([[17.]])
    snapshot=a.a122.prior.original_weights(eligible,TinyTorch)
    vector={name+'.weight':Tensor([[2.]]) for name,_ in eligible}
    a.update_from_vector(eligible,vector,.1,TinyTorch)
    assert all(module.weight.array.item()==pytest.approx(.8) for _,module in eligible)
    assert forbidden.array.item()==17.
    a.a122.restore_pre_step(eligible,snapshot,TinyTorch)
    assert all(module.weight.array.item()==1. for _,module in eligible)
    with pytest.raises(ValueError):
        a.update_from_vector(eligible,vector,float('nan'),TinyTorch)
    vector[next(iter(vector))]=Tensor([[float('nan')]])
    with pytest.raises(ValueError):
        a.update_from_vector(eligible,vector,.1,TinyTorch)


def test_response_sign_and_raw_b_gains_categories():
    final=Tensor(np.full((128,2048),3.),np.float64)
    current=Tensor(np.full((128,2048),2.25),np.float64)
    response=a.a122.prior.response_from_means(final,current,TinyTorch)
    assert response.dtype==np.float32
    assert np.all(response.array==.75)
    reports={'raw_b':report(.46,.49),1:report(.48,.50),
             2:report(.495,.515),4:report(.47,.60)}
    details,choice=a.interpret(reports)
    assert details['1']['category']=='moderate_improvement_over_raw_b'
    assert details['2']['category']=='clear_improvement_over_raw_b'
    assert details['4']['category']=='no_meaningful_improvement_over_raw_b'
    assert details['2']['primary_gain_vs_raw_b']==pytest.approx(.035)
    assert details['2']['secondary_gain_vs_raw_b']==pytest.approx(.025)
    assert choice['best_finite_k']==2
    tie={'raw_b':report(.4,.4),1:report(.41,.41),
         2:report(.42,.42),4:report(.42,.42)}
    assert a.interpret(tie)[1]['best_finite_k']==2
    with pytest.raises(ValueError):
        a.interpret({**reports,8:report(1.,1.)})


def test_signed_position_cosine_report_keeps_position_zero_separate():
    values=[-.75]+[-.2,.1,.3,.6]+[.5]*123
    validator=SimpleNamespace(position_cosines=lambda response,target:values)
    result=a.a122.score_report(object(),object(),validator)
    assert result['position_0_cosine']==-.75
    assert result['positions_1_4_individual_cosines']==[-.2,.1,.3,.6]
    assert result['positions_1_4_mean_cosine']==pytest.approx(.2)
    assert result['positions_1_127_mean_cosine']==pytest.approx(sum(values[1:])/127)
    assert result['all_128_position_cosines']==values


def test_raw_baseline_validation_and_target_hash_pin(monkeypatch):
    spec=a.load_spec()
    assert spec['frozen_attempt123']['target_raw_sha256']==a.TARGET_SHA256
    assert spec['evaluation']['target_raw_sha256']==a.TARGET_SHA256
    assert spec['frozen_attempt113']['raw_response_sha256']==a.EXPECTED_RAW_SHA256
    manifest=json.loads(a.path_of(spec['frozen_attempt113']['construction_manifest_path']).read_text())
    raw=Tensor(np.ones((128,2048)))
    class FakeTorch(TinyTorch):
        @staticmethod
        def load(_path,**kwargs):
            assert kwargs=={'map_location':'cpu','weights_only':True,'mmap':True}
            return {'x_80':{},'response_raw_rhs':raw,'response_inverse':Tensor(np.zeros((128,2048)))}
    class Helper:
        blind=SimpleNamespace(sha256_raw_float32_tensor=lambda _value,_torch:a.EXPECTED_RAW_SHA256)
    with monkeypatch.context() as patch:
        patch.setattr(a,'require_hash',lambda path,digest:path)
        assert a.validate_raw_baseline(spec,manifest,Helper,FakeTorch) is raw
    class BadHelper:
        blind=SimpleNamespace(sha256_raw_float32_tensor=lambda _value,_torch:'0'*64)
    with monkeypatch.context() as patch:
        patch.setattr(a,'require_hash',lambda path,digest:path)
        with pytest.raises(ValueError,match='raw-b functional response'):
            a.validate_raw_baseline(spec,manifest,BadHelper,FakeTorch)


def test_barrier_ordering_recovery_and_overwrite_refusal(tmp_path):
    source=SOURCE.read_text()
    functions={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body
               if isinstance(n,ast.FunctionDef)}
    run=functions['run']
    assert 'validate_endpoint(endpoint,diagnostics,manifest113)' in run
    assert 'for step in a122.prior.trajectory_steps(start_step):' in run
    assert 'a122.armijo_search(current,gradient_norm,initial_eta' in run
    assert 'a122.restore_pre_step(eligible,snapshot,torch)' in run
    assert 'if step==RECOVERY_STEP and not resume:' in run
    assert run.index('save_recovery(recovery_path') < run.index('a122.prior.atomic_torch_publish(candidate_path,')
    assert run.index('validate_raw_baseline(spec,manifest113,a113,torch)') < run.index('a122.prior.atomic_torch_publish(candidate_path,')
    assert run.index('a122.prior.atomic_torch_publish(candidate_path,') < run.index('a122.prior.atomic_json_publish(manifest_path,construction)')
    assert run.index('a122.prior.atomic_json_publish(manifest_path,construction)') < run.index('frozen_artifact=torch.load(candidate_path')
    assert run.index('frozen_artifact=torch.load(candidate_path') < run.index('recovery_path.unlink()')
    assert run.index('recovery_path.unlink()') < run.index("print('CONSENSUS_RELAXATION_CANDIDATES_FROZEN'")
    assert run.index("print('CONSENSUS_RELAXATION_CANDIDATES_FROZEN'") < run.index('evaluate_after_barrier(')
    assert functions['forward_weighted_objective'].find('oracle')==-1
    assert 'endpoint_norms[name]' in functions['weighted_gradient_pass']
    assert 'combined, eligible, endpoint_norms[name], inventory' in functions['weighted_gradient_pass']
    assert 'R_8' not in source and "f'R_{k}':responses[k] for k in GRID" in run
    assert 'a123.verified_target(' in functions['evaluate_after_barrier']
    assert 'a123.position_gain_summary(' in functions['evaluate_after_barrier']
    spec=copy.deepcopy(a.load_spec())
    spec['paths']={name:str(tmp_path/(name+'.pt')) for name in
                   ('candidate_path','recovery_path','construction_manifest_path','result_path')}
    a.refuse_outputs(spec)
    Path(spec['paths']['recovery_path']).write_bytes(b'recovery')
    with pytest.raises(ValueError):a.refuse_outputs(spec)
    a.refuse_outputs(spec,resume=True)
    Path(spec['paths']['candidate_path']).write_bytes(b'candidate')
    with pytest.raises(ValueError):a.refuse_outputs(spec,resume=True)
    assert "'recovery_only':True" in functions['save_recovery']
    assert "'scientific_candidate':False" in run


def test_step_two_checkpoint_contains_only_recovery_state(tmp_path,monkeypatch):
    class SaveTorch(TinyTorch):
        @staticmethod
        def save(value,stream):
            pickle.dump(value,stream)
    path=tmp_path/'step2.pt'
    rows=[{'step':1,'accepted_eta':.1},{'step':2,'accepted_eta':.05}]
    checkpoints={1:{'state_aggregate_sha256':'a'*64},
                 2:{'state_aggregate_sha256':'b'*64}}
    responses={1:Tensor([[1.]]),2:Tensor([[2.]])}
    with monkeypatch.context() as patch:
        patch.setattr(a,'validate_rows',lambda r,s:True)
        a.save_recovery(path,eligible_98(),rows,checkpoints,responses,
                        Tensor([[0.]],np.float64),{'norm':a.EXPECTED_B_NORM},
                        {'fineweb':.1},{'eta_max':1.},'c'*64,None,SaveTorch)
        payload=pickle.loads(path.read_bytes())
        assert payload['recovery_only'] is True
        assert payload['step']==2
        assert payload['previous_accepted_eta']==.05
        assert len(payload['state'])==98
        assert list(payload['responses'])==[1,2]
        assert 'scientific_candidate' not in payload
        with pytest.raises(ValueError):
            a.save_recovery(path,eligible_98(),rows,checkpoints,responses,
                            Tensor([[0.]],np.float64),{'norm':a.EXPECTED_B_NORM},
                            {'fineweb':.1},{'eta_max':1.},'c'*64,None,SaveTorch)
        with pytest.raises(ValueError):
            a.save_recovery(tmp_path/'wrong.pt',eligible_98(),rows[:1],checkpoints,
                            responses,Tensor([[0.]],np.float64),
                            {'norm':a.EXPECTED_B_NORM},{'fineweb':.1},
                            {'eta_max':1.},'c'*64,None,SaveTorch)
