"""Synthetic and static controls for Attempt123's post-freeze matched reevaluation."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/ablation/evaluate_matched1024_relaxation.py'
loader = importlib.util.spec_from_file_location('attempt123', SOURCE)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)


class Tensor:
    float_casts = 0

    def __init__(self, value, dtype=np.float32):
        self.array = np.array(value, dtype=dtype, copy=True)
        self.device = SimpleNamespace(type='cpu')

    @property
    def dtype(self):
        return self.array.dtype.type

    @property
    def shape(self):
        return self.array.shape

    def detach(self):
        return self

    def numpy(self):
        return self.array

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def contiguous(self):
        return self

    def float(self):
        Tensor.float_casts += 1
        return Tensor(self.array, np.float32)

    def __sub__(self, other):
        return Tensor(self.array - other.array, self.dtype)

    def __getitem__(self, key):
        return Tensor(self.array[key], self.dtype)


class TinyTorch:
    Tensor = Tensor
    float32 = np.float32
    float64 = np.float64
    int64 = np.int64
    isfinite = staticmethod(lambda value: np.isfinite(value.array))


def raw_hash(value, _torch):
    return hashlib.sha256(value.array.astype('<f4', copy=False).tobytes(order='C')).hexdigest()


def report(values):
    class Validator:
        @staticmethod
        def position_cosines(_x, _y):
            return values
    return a.a122.score_report(None, None, Validator)


def test_pinned_inventory_and_no_scientific_contradiction():
    spec = a.load_spec()
    frozen, reference = spec['frozen_attempt122'], spec['matched_reference']
    old122 = a.a122.load_spec()
    old103 = json.loads(a.path_of(reference['attempt103_spec_path']).read_text())
    old106 = json.loads(a.path_of(reference['attempt106_spec_path']).read_text())
    manifest = json.loads(a.path_of(frozen['construction_manifest_path']).read_text())
    result103 = json.loads(a.path_of(reference['attempt103_result_path']).read_text())
    assert a.GRID == (1, 2, 4) and a.NAMES == ('R_1', 'R_2', 'R_4')
    assert frozen['tensor_order'] == list(a.NAMES)
    assert frozen['candidate_path'] == old122['paths']['candidate_path']
    assert frozen['candidate_serialized_sha256'] == manifest['candidate']['serialized_sha256']
    assert frozen['raw_response_sha256'] == {
        f'R_{k}':manifest['checkpoints'][str(k)]['response']['raw_sha256'] for k in a.GRID}
    for key in ('spec', 'source', 'construction_manifest', 'result'):
        assert a.sha256_file(a.path_of(frozen[key+'_path'])) == frozen[key+'_sha256']
    for key in ('attempt103_spec', 'attempt103_source', 'attempt103_result', 'attempt106_spec'):
        assert a.sha256_file(a.path_of(reference[key+'_path'])) == reference[key+'_sha256']
    assert manifest['source_checkpoint']['files'] == old103['final_checkpoint_files']
    assert old103['base'] == old106['base']
    assert old103['final_checkpoint_files'] == old106['final_checkpoint_files']
    assert old103['probe'] == old106['probe']
    assert old103['probe_rows'] == old106['probe_rows'] == [0, 1024]
    assert reference['target_raw_sha256'] == a.TARGET_SHA256 == old106['target']['raw_float32_sha256']
    assert result103['provenance']['matched_difference_raw_sha256'] == a.TARGET_SHA256
    assert reference['readout_block_index'] == 13
    assert reference['hidden_state_index'] == 14
    assert reference['batches_per_model'] == 32
    assert spec['workload'] == {'gradient_batches': 0, 'jvp_batches': 0,
        'ggn_or_cg_operations': 0, 'final_forward_batches': 32,
        'base_forward_batches': 32, 'total_forward_batches': 64,
        'new_candidate_artifacts': 0}
    assert spec['information_policy']['clean_heldout_validation'] is False


def test_candidate_order_dtype_shape_finiteness_and_individual_hashes():
    artifact = {name:Tensor(np.full((128,2048), index))
                for index,name in enumerate(a.NAMES, start=1)}
    frozen = {'raw_response_sha256':{name:raw_hash(value,TinyTorch)
                                     for name,value in artifact.items()}}
    assert a.validate_candidate_tensors(artifact, frozen, TinyTorch, raw_hash) is artifact
    with pytest.raises(ValueError, match='order/inventory'):
        a.validate_candidate_tensors(dict(reversed(list(artifact.items()))), frozen, TinyTorch, raw_hash)
    with pytest.raises(ValueError, match='raw SHA256'):
        a.validate_candidate_tensors(artifact, {'raw_response_sha256':{**frozen['raw_response_sha256'],
            'R_2':'0'*64}}, TinyTorch, raw_hash)
    for bad in (Tensor(np.zeros((127,2048))), Tensor(np.zeros((128,2048)),np.float64),
                Tensor(np.full((128,2048),np.nan))):
        changed = dict(artifact, R_1=bad)
        with pytest.raises(ValueError):
            a.validate_candidate_tensors(changed, frozen, TinyTorch, raw_hash)
    assert all(value.array[0,0] == i for i,value in enumerate(artifact.values(),start=1))


def test_matched_probe_is_exact_1024_prefix_and_32_batch_inventory():
    spec = a.load_spec()
    full = Tensor(np.repeat(np.arange(10000,dtype=np.int64)[:,None],128,axis=1),np.int64)
    prefix = a.matched_probe(full,spec,TinyTorch)
    assert prefix.shape == (1024,128)
    assert np.array_equal(prefix.array,full.array[:1024])
    assert [(i,i+31) for i in range(0,1024,32)][-1] == (992,1023)
    assert len(range(0,1024,32)) == spec['matched_reference']['batches_per_model'] == 32
    with pytest.raises(ValueError):
        a.matched_probe(full[:1024],spec,TinyTorch)
    changed = copy.deepcopy(spec)
    changed['matched_reference']['probe_rows'] = [1,1025]
    with pytest.raises(ValueError):
        a.matched_probe(full,changed,TinyTorch)


def test_privileged_reference_validation_uses_frozen_103_and_106_records():
    spec = a.load_spec()
    manifest = json.loads(a.path_of(spec['frozen_attempt122']['construction_manifest_path']).read_text())
    old103 = json.loads(a.path_of(spec['matched_reference']['attempt103_spec_path']).read_text())
    calls = []

    class Privileged:
        @staticmethod
        def load_spec():
            calls.append('load_spec')
            return old103

        @staticmethod
        def validate_sources(value, *, smoke_only):
            calls.append('validate_sources')
            assert value is old103 and smoke_only is True
            return {'base_files': old103['base']['files'],
                    'final_files': old103['final_checkpoint_files']}

    reference, inventories = a.validate_matched_reference(spec,manifest,Privileged)
    assert reference is old103
    assert inventories['base_files'] == old103['base']['files']
    assert calls == ['load_spec','validate_sources']
    changed = copy.deepcopy(manifest)
    changed['source_checkpoint']['files'] = []
    with pytest.raises(ValueError,match='provenance contradiction'):
        a.validate_matched_reference(spec,changed,Privileged)


def test_existing_matched_target_helper_has_final_minus_base_and_one_cast(monkeypatch):
    # Import the frozen Attempt103 helper with a tiny torch stand-in; no model is loaded.
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules,'torch',TinyTorch)
        reference = a.load_spec()['matched_reference']
        privileged = a.import_pinned(a.path_of(reference['attempt103_source_path']),
            reference['attempt103_source_sha256'],'attempt123_synthetic_103')
    final = Tensor(np.full((128,2048),3.0),np.float64)
    base = Tensor(np.full((128,2048),2.25),np.float64)
    Tensor.float_casts = 0
    target = privileged.matched_difference(final,base)
    assert Tensor.float_casts == 1
    assert target.dtype == np.float32 and target.is_contiguous()
    assert np.all(target.array == .75)
    local = copy.deepcopy(a.load_spec())
    local['matched_reference']['target_raw_sha256'] = raw_hash(target,TinyTorch)
    accepted,digest = a.verified_target(final,base,local,privileged.matched_difference,
                                        raw_hash,TinyTorch)
    assert digest == raw_hash(accepted,TinyTorch)
    assert np.all(accepted.array == .75)
    with pytest.raises(ValueError,match='target raw SHA256 mismatch'):
        a.verified_target(final,base,a.load_spec(),privileged.matched_difference,
                          raw_hash,TinyTorch)
    base.array[0,0] = np.nan
    with pytest.raises(ValueError):
        a.verified_target(final,base,local,privileged.matched_difference,
                          raw_hash,TinyTorch)


def test_signed_metrics_old_comparison_thresholds_and_position_gains():
    first_values = [-.3]+[.20]*127
    second_values = [-.2]+[.24]*4+[.23]*123
    fourth_values = [-.4]+[.22]*4+[.19]*123
    matched = {1:report(first_values),2:report(second_values),4:report(fourth_values)}
    old = {str(k):report([-.1]+[.1]*127) for k in a.GRID}
    assert matched[1]['position_0_cosine'] == -.3
    assert matched[2]['positions_1_4_individual_cosines'] == [.24]*4
    assert matched[2]['positions_1_4_mean_cosine'] == .24
    assert matched[2]['positions_1_127_mean_cosine'] == pytest.approx((4*.24+123*.23)/127)
    comparison,later,interpretation = a.summarize(matched,old)
    assert comparison['1']['matched_primary'] == .2
    assert comparison['1']['old_mismatched_primary'] == .1
    assert comparison['1']['delta_matched_minus_old_primary'] == pytest.approx(.1)
    assert comparison['2']['delta_matched_minus_old_secondary'] == pytest.approx(
        matched[2]['positions_1_127_mean_cosine']-.1)
    assert later['2']['matched_primary_gain_vs_k1'] == pytest.approx(.04)
    assert later['2']['matched_secondary_gain_vs_k1'] > .02
    assert later['2']['improved_position_count_1_127'] == 127
    assert later['2']['mean_gain_positions_1_4'] == pytest.approx(.04)
    assert later['2']['mean_gain_positions_5_127'] == pytest.approx(.03)
    assert later['2']['mean_gain_positions_1_127'] == pytest.approx((4*.04+123*.03)/127)
    assert later['4']['improved_position_count_1_127'] == 4
    assert later['4']['mean_gain_positions_5_127'] == pytest.approx(-.01)
    assert interpretation['best_multistep_k'] == 2
    assert interpretation['category'] == 'clear_multistep_support'
    assert a.a122.classify({1:report([0]+[.4]*127),
        2:report([0]+[.416]*127),4:report([0]+[.41]*127)})['category'] == 'moderate_multistep_support'
    assert a.a122.classify({1:report([0]+[.4]*127),
        2:report([0]+[.414]*127),4:report([0]+[.41]*127)})['category'] == 'no_meaningful_multistep_support'
    tie = {1:matched[1],2:matched[2],4:matched[2]}
    assert a.a122.classify(tie)['best_multistep_k'] == 2
    with pytest.raises(ValueError):
        a.summarize({**matched,8:matched[2]},old)


def test_barriers_no_candidate_write_and_overwrite_refusal(tmp_path):
    source = SOURCE.read_text()
    functions = {node.name:ast.get_source_segment(source,node) for node in ast.parse(source).body
                 if isinstance(node,ast.FunctionDef)}
    run = functions['run']
    assert run.index('validate_frozen_candidates(spec, torch)') < run.index('import_pinned(path_of(record[\'attempt103_source_path\'])')
    assert run.index('validate_frozen_candidates(spec, torch)') < run.index('privileged.load_model_pair(old103)')
    assert run.index('verified_target(final_mean, base_mean') < run.index('a122.score_report(')
    assert run.index('verified_target(final_mean, base_mean') < run.index("old_evaluation = a122.load_spec()['evaluation']")
    assert 'torch.save' not in source and 'update_eligible' not in source
    assert run.count('privileged.base_probe_mean(') == 2
    assert 'privileged.matched_difference' in run
    assert "final_mean_hash != manifest['final_activation_mean']['raw_float64_sha256']" in run
    assert 'atomic_json_publish(output, result)' in run
    assert 'candidate_path' not in functions['verified_target']
    helper_source = a.path_of(a.load_spec()['matched_reference']['attempt103_source_path']).read_text()
    helper_tree = ast.parse(helper_source)
    mean_body = ast.get_source_segment(helper_source,next(n for n in helper_tree.body
        if isinstance(n,ast.FunctionDef) and n.name=='base_probe_mean'))
    assert 'for start in range(0, count, 32):' in mean_body
    assert "accumulator = torch.zeros((128, 2048), dtype=torch.float64, device='cpu')" in mean_body
    assert 'accumulator.add_(readout.detach().cpu().double().sum(dim=0))' in mean_body
    path=tmp_path/'result.json'
    a.refuse_result(path)
    path.write_text('{}')
    with pytest.raises(ValueError,match='already exists'):
        a.refuse_result(path)
