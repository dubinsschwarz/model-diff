"""Synthetic and provenance controls for Attempt125; never load real models."""
import ast
from contextlib import nullcontext
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE=Path(__file__).resolve().parents[1]/'scripts/ablation/diagnose_relaxation_gradient_vs_base.py'
loader=importlib.util.spec_from_file_location('attempt125_test',SOURCE)
a=importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)


def manifest124():
    return json.loads(a.a124.path_of(a.load_spec()[0]['frozen_attempt124']
                                     ['construction_manifest_path']).read_text())


def inventory():
    return [{'name':f'matrix.{i}.weight','shape':[1,1]} for i in range(98)]


def vector(values):
    v={row['name']:np.zeros((1,1),dtype=np.float32) for row in inventory()}
    for index,value in enumerate(values):
        v[f'matrix.{index}.weight'][0,0]=value
    return v


def geometry_row(cosine,alpha=1.,residual=0.):
    return {'cosine_to_true_subtraction':cosine,'alpha':alpha,'residual':residual}


def test_frozen_objective_four_passes_and_two_replay_controls():
    spec,old,_=a.load_spec()
    m=a.validate_frozen_trajectory(spec,old)
    assert spec['objective']['corpus_order']==list(a.a124.ORDER)==[
        'fineweb','wikitext103_raw','tinystories','arxiv_document',
        'cc_news','pg19','codeparrot_clean','ultrachat']
    assert spec['objective']['rows_per_corpus']==[0,512]
    assert (spec['objective']['sequence_length'],spec['objective']['batch_size'],
            spec['objective']['batches_per_corpus'],
            spec['objective']['prediction_contexts_per_corpus'])==(128,8,64,65024)
    assert spec['objective']['gradient_batches_per_pass']==512
    assert spec['workload']['balanced_gradient_passes']==4
    assert spec['workload']['total_backward_batches']==2048
    for name in ('armijo_forward_batches','activation_probe_batches',
                 'jvp_operations','ggn_operations','cg_operations'):
        assert spec['workload'][name]==0
    assert spec['objective']['weights_fixed_after_theta_F'] is True
    assert spec['objective']['later_gradient_renormalization'] is False
    assert list(spec['objective']['endpoint_gradient_norms'].values())==list(a.a124.EXPECTED_NORMS)
    fixed=a.a124.frozen_weights(spec['objective']['endpoint_gradient_norms'])
    assert fixed==m['fixed_weights']
    assert all(fixed[name]==1/(8*norm) for name,norm in
               spec['objective']['endpoint_gradient_norms'].items())
    assert spec['objective']['endpoint_b_norm']==a.a124.EXPECTED_B_NORM==.6924963677590905
    assert spec['objective']['endpoint_b_aggregate_sha256']==a.a124.EXPECTED_B_SHA256
    assert spec['replay']['update_count']==2
    assert spec['replay']['accepted_etas']==list(a.ETAS)==[.10358343196094506]*2
    assert spec['replay']['cumulative_displacement_norms']==list(a.DISPLACEMENTS)
    assert spec['replay']['state_aggregate_sha256']==list(a.STATE_HASHES)
    assert m['steps'][0]['accepted_trial_index']==4
    assert m['steps'][1]['accepted_trial_index']==0
    assert a.SOURCE124_SHA256==m['constructor_sha256']


def test_endpoint_norm_hash_and_per_corpus_norm_gate():
    m=manifest124()
    spec=a.load_spec()[0]
    m113=json.loads(a.a124.path_of(a.a124.load_spec()['frozen_attempt113']
                                    ['construction_manifest_path']).read_text())
    assert m['endpoint_b_blind']==m113['b_blind']
    assert a.a124.validate_endpoint(m['endpoint_b_blind'],m113['per_corpus_gradients'],m113)
    altered=copy.deepcopy(m113['per_corpus_gradients'])
    altered[0]['global_gradient_norm']*=1.01
    with pytest.raises(ValueError,match='b_blind'):
        a.a124.validate_endpoint(m['endpoint_b_blind'],altered,m113)
    altered=copy.deepcopy(m['endpoint_b_blind'])
    altered['ordered_matrix_hash_sha256']='0'*64
    with pytest.raises(ValueError,match='b_blind'):
        a.a124.validate_endpoint(altered,m113['per_corpus_gradients'],m113)
    assert spec['objective']['endpoint_b_aggregate_sha256']==m['endpoint_b_blind']['ordered_matrix_hash_sha256']


def test_replay_hash_and_displacement_gate(monkeypatch):
    m=manifest124()
    stub=SimpleNamespace(blind=object())
    for k in (1,2):
        with monkeypatch.context() as patch:
            patch.setattr(a.a124.a122.prior,'displacement_norm',
                          lambda eligible,originals,torch:a.DISPLACEMENTS[k-1])
            patch.setattr(a.a124.a122.prior,'state_hashes',
                          lambda eligible,helper,torch:(
                              m['checkpoints'][str(k)]['state_per_matrix_sha256'],a.STATE_HASHES[k-1]))
            row=a.validate_replay_state(k,[],{},m,stub,None)
            assert row['accepted_eta']==a.ETAS[k-1]
            assert row['state_aggregate_sha256']==a.STATE_HASHES[k-1]
            patch.setattr(a.a124.a122.prior,'state_hashes',
                          lambda eligible,helper,torch:({},'0'*64))
            with pytest.raises(ValueError,match='before base access'):
                a.validate_replay_state(k,[],{},m,stub,None)
    with pytest.raises(ValueError,match='Only frozen replay'):
        a.validate_replay_state(3,[],{},m,stub,None)


class Tensor:
    def __init__(self,value,dtype=np.float32):
        self.array=np.array(value,dtype=dtype,copy=True)
        self.device=SimpleNamespace(type='cpu')
    @property
    def dtype(self):return self.array.dtype.type
    @property
    def shape(self):return self.array.shape
    def detach(self):return self
    def to(self,device=None,dtype=None):
        if device in (np.float32,np.float64):dtype=device
        return Tensor(self.array,self.dtype if dtype is None else dtype)
    def contiguous(self):return self
    def copy_(self,other):self.array[...]=other.array;return self
    def __mul__(self,other):return Tensor(self.array*other,self.dtype)
    __rmul__=__mul__
    def __sub__(self,other):return Tensor(self.array-other.array,self.dtype)


class TinyTorch:
    float32=np.float32
    float64=np.float64
    no_grad=staticmethod(nullcontext)
    isfinite=staticmethod(lambda x:np.isfinite(x.array))


def test_exact_two_updates_sign_and_fp64_then_fp32_rounding():
    eligible=[(f'model.layers.{i}.linear',SimpleNamespace(weight=Tensor([[1.]])))
              for i in range(98)]
    frozen=Tensor([[17.]])
    b_f={name+'.weight':Tensor([[3.]]) for name,_ in eligible}
    b_1={name+'.weight':Tensor([[2.]]) for name,_ in eligible}
    for eta,gradient in zip(a.ETAS,(b_f,b_1)):
        a.a124.update_from_vector(eligible,gradient,eta,TinyTorch)
    expected=np.float32(np.float64(np.float32(1.-a.ETAS[0]*3))-
                        np.float64(a.ETAS[1])*2)
    assert all(module.weight.array[0,0]==expected for _,module in eligible)
    assert frozen.array[0,0]==17.
    assert expected<1.
    with pytest.raises(ValueError):
        a.a124.update_from_vector(eligible,b_1,float('nan'),TinyTorch)


def test_global_float64_geometry_subtraction_sign_and_projection():
    b_f=vector([2.,0.])
    b_1=vector([1.,0.])
    b_2=vector([2.,-1.])
    b_base=vector([.1,0.])
    d=a.vector_diagnostics(b_f,b_1,b_2,b_base,inventory())
    assert d['gradient_norms']['b_F']==2.
    assert d['gradient_norms']['b_1']==1.
    assert d['gradient_norms']['b_2']==pytest.approx(math.sqrt(5))
    assert d['gradient_norms']['b_B']==pytest.approx(.1)
    assert d['direct_gradient_cosines']['b_F_b_B']==pytest.approx(1.)
    assert d['direct_gradient_cosines']['b_1_b_B']==pytest.approx(1.)
    assert d['direct_gradient_cosines']['b_2_b_B']==pytest.approx(2/math.sqrt(5))
    assert d['true_subtraction_norm']==pytest.approx(1.9)
    assert d['subtraction']['1']['norm']==1.
    assert d['subtraction']['1']['cosine_to_true_subtraction']==pytest.approx(1.)
    assert d['subtraction']['1']['alpha']==pytest.approx(1.9)
    assert d['subtraction']['1']['residual']==pytest.approx(0.)
    assert d['subtraction']['1']['explained_fraction']==pytest.approx(1.)
    assert d['subtraction']['1']['distance_to_base_gradient_ratio']==pytest.approx(.9/1.9)
    assert d['subtraction']['2']['norm']==1.
    assert d['subtraction']['2']['cosine_to_true_subtraction']==pytest.approx(0.)
    assert d['subtraction']['2']['alpha']==pytest.approx(0.)
    assert d['subtraction']['2']['residual']==pytest.approx(1.)
    assert d['subtraction']['2']['explained_fraction']==pytest.approx(0.)
    assert d['subtraction']['2']['distance_to_base_gradient_ratio']==pytest.approx(math.hypot(1.9,1)/1.9)
    with pytest.raises(ValueError,match='zero norm'):
        a.vector_diagnostics(b_f,b_1,b_2,b_f,inventory())
    broken=copy.deepcopy(b_1)
    broken['matrix.0.weight'][0,0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        a.vector_diagnostics(b_f,broken,b_2,b_base,inventory())


def test_interpretation_exact_thresholds_and_tie_break():
    classify=lambda one,two:a.interpret({'subtraction':{'1':one,'2':two}})
    strong=classify(geometry_row(.70,1.,math.sqrt(1-.70**2)),
                    geometry_row(.6,1.,.8))
    assert strong['classification']=='strong_support' and strong['best_k']==1
    assert classify(geometry_row(.41,1.,.95),geometry_row(.4,1.,.9))[
        'classification']=='moderate_support'
    assert classify(geometry_row(.39,1.,.1),geometry_row(.2,1.,.1))[
        'classification']=='no_useful_support'
    assert classify(geometry_row(.8,-1.,.1),geometry_row(.2,1.,.1))[
        'classification']=='no_useful_support'
    assert classify(geometry_row(.5,1.,.5),geometry_row(.5,1.,.5))['best_k']==1
    with pytest.raises(ValueError):
        a.interpret({'subtraction':{'1':geometry_row(.5)}})


def test_barrier_order_base_weights_and_no_candidate_or_oracle():
    source=SOURCE.read_text()
    functions={node.name:ast.get_source_segment(source,node) for node in ast.parse(source).body
               if isinstance(node,ast.FunctionDef)}
    run=functions['run']
    assert run.count('a124.update_from_vector(eligible,')==2
    assert run.index('a124.validate_endpoint(endpoint_record') < run.index('a124.update_from_vector(eligible,b_f')
    assert run.index('validate_replay_state(1,') < run.index('a124.update_from_vector(eligible,b_1')
    assert run.index('validate_replay_state(2,') < run.index('b_2,losses2')
    assert run.index('b_2,losses2') < run.index("print('RELAXATION_GRADIENTS_FROZEN'")
    assert run.index("print('RELAXATION_GRADIENTS_FROZEN'") < run.index("known_ref=spec['frozen_attempt015']")
    assert run.index("known_ref=spec['frozen_attempt015']") < run.index("base=known.a.load_local_model")
    assert "base,corpora,base_eligible,inventory,endpoint_norms" in run
    assert 'a124.frozen_weights(endpoint_norms)' in run
    assert run.count('a124.weighted_gradient_pass(')==3
    assert 'armijo_search(' not in run
    assert 'mean_activation(' not in run
    assert 'response_from_means(' not in run
    assert 'compute_probe_response(' not in run
    assert 'jvp(' not in run.lower()
    assert 'candidate_path' not in source
    assert "atomic_json_publish(result_path,result)" in run
    assert source.index("print('RELAXATION_GRADIENTS_FROZEN'") < source.index("known_spec=known.load_spec")


def test_result_overwrite_refusal(tmp_path,monkeypatch):
    spec,_,_=a.load_spec()
    changed=copy.deepcopy(spec)
    changed['paths']['result_path']=str(tmp_path/'result.json')
    temporary=tmp_path/'spec.json'
    temporary.write_text(json.dumps(changed))
    with monkeypatch.context() as patch:
        patch.setattr(a,'SPEC_PATH',temporary)
        patch.setattr(a,'SPEC_SHA256',hashlib.sha256(temporary.read_bytes()).hexdigest())
        assert a.load_spec()[2]==tmp_path/'result.json'
        (tmp_path/'result.json').write_text('{}')
        with pytest.raises(ValueError,match='exists'):
            a.load_spec()
