"""Synthetic and static checks for Attempt127's immutable postfreeze evaluation."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'scripts/ablation/evaluate_surprisal_strata_postfreeze.py'
loader = importlib.util.spec_from_file_location('attempt127', SOURCE)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)


class Tensor:
    def __init__(self, digest, shape=(128, 2048), dtype=np.float32, finite=True):
        self.array = np.full(shape, 1 if finite else np.nan, dtype=dtype)
        self.digest = digest
        self.device = SimpleNamespace(type='cpu')

    @property
    def dtype(self):
        return self.array.dtype.type

    @property
    def shape(self):
        return self.array.shape

    def is_contiguous(self):
        return self.array.flags.c_contiguous


class TinyTorch:
    Tensor = Tensor
    float32 = np.float32
    isfinite = staticmethod(lambda value: np.isfinite(value.array))

    def __init__(self, artifact):
        self.artifact = artifact

    def load(self, _path, map_location, weights_only):
        assert map_location == 'cpu' and weights_only is True
        return self.artifact


def reports(primary, secondary):
    return {'positions_1_4_mean_cosine': primary,
            'positions_1_127_mean_cosine': secondary}


def test_frozen_hashes_inventory_typo_and_workload():
    spec = a.load_spec()
    frozen = spec['frozen_attempt126']
    old_spec = json.loads(a.path_of(frozen['spec_path']).read_text())
    manifest = json.loads(a.path_of(frozen['construction_manifest_path']).read_text())
    assert not a.path_of(old_spec['paths']['result_path']).exists()
    assert a.NAMES == ('response_low', 'response_middle', 'response_high')
    for key in ('spec', 'source', 'construction_manifest'):
        assert a.sha256_file(a.path_of(frozen[key + '_path'])) == frozen[key + '_sha256']
    assert a.sha256_file(a.path_of(frozen['candidate_path'])) == frozen['candidate_serialized_sha256']
    assert manifest['candidate']['serialized_sha256'] == frozen['candidate_serialized_sha256']
    assert manifest['candidate']['tensor_order'] == list(a.NAMES)
    assert list(manifest['responses']) == list(a.NAMES)
    for name in a.NAMES:
        assert manifest['responses'][name]['raw_sha256'] == frozen['raw_response_sha256'][name]
    assert manifest['ranking']['surprisal_raw_float64_sha256'] == frozen['ranking']['surprisal_raw_float64_sha256']
    assert manifest['ranking']['rank_raw_int64_sha256'] == frozen['ranking']['rank_raw_int64_sha256']
    assert spec['failure_repair']['erroneous_sha256'] == old_spec['frozen_attempt123']['source_sha256']
    assert len(spec['failure_repair']['erroneous_sha256']) == 65
    assert len(a.EVALUATOR_SHA256) == 64
    assert spec['failure_repair']['erroneous_sha256'] == a.EVALUATOR_SHA256 + 'f'
    assert a.sha256_file(a.path_of(spec['matched_reference']['attempt123_source_path'])) == a.EVALUATOR_SHA256
    assert spec['failure_repair']['failure_phase'].endswith('before_historical_access')
    assert spec['workload']['total_forward_batches'] == 64
    assert spec['workload']['gradient_batches'] == spec['workload']['jvp_batches'] == 0


def test_hash_validation_rejects_changed_candidate_and_manifest(tmp_path):
    spec = a.load_spec()
    frozen = spec['frozen_attempt126']
    for key, expected in (('candidate', frozen['candidate_serialized_sha256']),
                          ('construction_manifest', frozen['construction_manifest_sha256'])):
        original = a.path_of(frozen[key + '_path'])
        assert a.require_hash(original, expected) == original
        changed = tmp_path / (key + '.changed')
        changed.write_bytes(original.read_bytes() + b'changed')
        with pytest.raises(ValueError, match='Frozen SHA256 mismatch'):
            a.require_hash(changed, expected)


def test_complete_synthetic_candidate_audit_and_fail_closed(monkeypatch):
    spec = a.load_spec()
    frozen = spec['frozen_attempt126']
    original_import = a.import_pinned
    readout = SimpleNamespace(sha256_raw_float32_tensor=lambda value, _torch: value.digest)

    def safe_import(path, digest, name):
        if 'attempt123' in name or 'attempt103' in name:
            raise AssertionError('Historical machinery imported before the barrier')
        module = original_import(path, digest, name)
        if name == 'attempt127_frozen_attempt126':
            module.import_pinned = lambda *_args: readout
        return module

    monkeypatch.setattr(a, 'import_pinned', safe_import)
    tensors = {name: Tensor(frozen['raw_response_sha256'][name]) for name in a.NAMES}
    artifact, manifest, old_spec, source, helper = a.validate_frozen_candidates(
        spec, TinyTorch(tensors))
    assert artifact is tensors and helper is readout
    assert list(manifest['gradient_records']) == list(a.STRATA)
    assert source.validate_ranking(manifest['ranking']) is True
    assert old_spec['barrier']['marker'] == 'SURPRISAL_STRATA_CANDIDATES_FROZEN'

    for broken in (
        dict(reversed(list(tensors.items()))),
        {**tensors, 'response_middle': Tensor('0' * 64)},
        {**tensors, 'response_low': Tensor(frozen['raw_response_sha256']['response_low'], shape=(127, 2048))},
        {**tensors, 'response_high': Tensor(frozen['raw_response_sha256']['response_high'], dtype=np.float64)},
        {**tensors, 'response_low': Tensor(frozen['raw_response_sha256']['response_low'], finite=False)},
    ):
        with pytest.raises(ValueError):
            a.validate_frozen_candidates(spec, TinyTorch(broken))

    changed = copy.deepcopy(spec)
    changed['frozen_attempt126']['construction_manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Frozen SHA256 mismatch'):
        a.validate_frozen_candidates(changed, TinyTorch(tensors))


def test_barrier_precedes_privileged_import_and_no_candidate_construction():
    source = SOURCE.read_text()
    definitions = {node.name: ast.get_source_segment(source, node)
                   for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    run = definitions['run']
    assert run.index('validate_frozen_candidates(spec, torch)') < run.index(
        "print('ATTEMPT126_CANDIDATES_REVALIDATED'") < run.index('evaluate_after_barrier(')
    audit = definitions['validate_frozen_candidates']
    assert 'attempt123' not in audit or 'old_spec[\'frozen_attempt123\']' in audit
    assert 'privileged.load_model_pair' not in audit
    privileged = definitions['evaluate_after_barrier']
    assert privileged.index("'attempt127_corrected_attempt123'") < privileged.index(
        'privileged.load_model_pair(old103)')
    assert privileged.index('evaluator.verified_target(') < privileged.index('source.a122.score_report(')
    assert 'TARGET_SHA256' in privileged
    assert 'score_endpoint_surprisal(' not in source
    assert 'accumulate_mean_generic_gradient(' not in source
    assert 'compute_probe_response(' not in source
    assert 'atomic_torch_publish(' not in source
    assert 'torch.save(' not in source


def test_signed_position_metrics_classification_and_frozen_correlations():
    spec = a.load_spec()
    source = a.import_pinned(a.path_of(spec['frozen_attempt126']['source_path']),
                             spec['frozen_attempt126']['source_sha256'], 'test_attempt126')
    signed = [-.8, -.2, .1, .3, .6] + [.5] * 123
    validator = SimpleNamespace(position_cosines=lambda _x, _y: signed)
    metric = source.a122.score_report(object(), object(), validator)
    assert metric['position_0_cosine'] == -.8
    assert metric['positions_1_4_individual_cosines'] == signed[1:5]
    assert metric['positions_1_4_mean_cosine'] == pytest.approx(.2)
    assert metric['positions_1_127_mean_cosine'] == pytest.approx(sum(signed[1:]) / 127)
    assert metric['all_128_position_cosines'] == signed
    assert metric['position_0_cosine'] < 0

    clear = dict(zip(a.NAMES, (reports(.4, .4), reports(.42, .42), reports(.44, .44))))
    deltas, category = source.differences_and_classification(clear)
    assert category == 'clear_support'
    assert deltas['high_minus_low']['primary'] == pytest.approx(.04)
    assert deltas['high_minus_middle']['secondary'] == pytest.approx(.02)
    assert deltas['middle_minus_low']['primary'] == pytest.approx(.02)
    clear_boundary = dict(zip(a.NAMES,
        (reports(.5, .5), reports(.515, .515), reports(.53, .53))))
    assert source.differences_and_classification(clear_boundary)[1] == 'clear_support'
    partial = dict(zip(a.NAMES, (reports(.4, .4), reports(.45, .41), reports(.44, .42))))
    assert source.differences_and_classification(partial)[1] == 'partial_support'
    partial_boundary = dict(zip(a.NAMES,
        (reports(.5, .5), reports(.51, .505), reports(.53, .501))))
    assert source.differences_and_classification(partial_boundary)[1] == 'partial_support'
    no = dict(zip(a.NAMES, (reports(.4, .4), reports(.41, .41), reports(.42, .39))))
    assert source.differences_and_classification(no)[1] == 'no_support'
    with pytest.raises(ValueError):
        source.differences_and_classification({**clear, 'response_extra': reports(1, 1)})

    manifest = json.loads(a.path_of(spec['frozen_attempt126']['construction_manifest_path']).read_text())
    correlations = source.descriptive_correlations(manifest['ranking'],
                                                    manifest['gradient_records'], clear)
    assert correlations['descriptive_only'] is True and correlations['n_strata'] == 3
    assert correlations['order'] == ['low', 'middle', 'high']
    assert correlations['predictor_values']['mean_surprisal'] == [
        manifest['ranking']['strata'][name]['surprisal_mean'] for name in a.STRATA]
    assert correlations['predictor_values']['gradient_norm'] == [
        manifest['gradient_records'][name]['aggregate_generic_gradient_norm'] for name in a.STRATA]


def test_matched_target_pin_and_hash_failure_before_scoring():
    spec = a.load_spec()
    reference = spec['matched_reference']
    evaluator = a.import_pinned(a.path_of(reference['attempt123_source_path']),
                                reference['attempt123_source_sha256'], 'test_attempt123')
    matched_spec = evaluator.load_spec()
    assert a.TARGET_SHA256 == matched_spec['matched_reference']['target_raw_sha256']
    assert reference['probe_rows'] == [0, 1024]
    assert reference['batches_per_model'] == 32
    assert reference['target_definition'] == (
        'final_mean_float64_minus_base_mean_float64_then_one_contiguous_FP32_cast')
    with pytest.raises(ValueError, match='raw SHA256 mismatch'):
        evaluator.verified_target(None, None, matched_spec,
            lambda _final, _base: Tensor('0' * 64),
            lambda _target, _torch: '0' * 64, TinyTorch({}))


def test_result_overwrite_refusal(tmp_path):
    output = tmp_path / 'result.json'
    a.refuse_result(output)
    output.write_text('{}')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_result(output)
    output.unlink()
    output.symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_result(output)
