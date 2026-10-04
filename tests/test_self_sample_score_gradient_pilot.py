"""Synthetic controls for Attempt128's deterministic blind self-score pilot."""
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

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/ablation/run_self_sample_score_gradient_pilot.py'
loader = importlib.util.spec_from_file_location('attempt128', SOURCE)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)


class ArrayTensor:
    def __init__(self, array):
        self.array = np.asarray(array)
        self.device = SimpleNamespace(type='cpu')

    @property
    def dtype(self):
        return self.array.dtype.type

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    def __getitem__(self, key):
        return ArrayTensor(self.array[key])

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def contiguous(self):
        return ArrayTensor(np.ascontiguousarray(self.array))

    def to(self, dtype):
        return ArrayTensor(self.array.astype(dtype))

    def detach(self):
        return ArrayTensor(self.array)

    def unsqueeze(self, axis):
        return ArrayTensor(np.expand_dims(self.array, axis))

    def __lt__(self, other):
        return self.array < other

    def __ge__(self, other):
        return self.array >= other

    def gather(self, axis, indices):
        return ArrayTensor(np.take_along_axis(self.array, indices.array, axis=axis))

    def sum(self):
        return ArrayTensor(self.array.sum())

    def __neg__(self):
        return ArrayTensor(-self.array)

    def __mul__(self, other):
        return ArrayTensor(self.array * other.array)


class FakeTorch:
    Tensor = ArrayTensor
    int64 = np.int64
    float32 = np.float32
    float64 = np.float64

    @staticmethod
    def log_softmax(value, dim):
        assert value.dtype == np.float64
        x = value.array
        shifted = x - x.max(axis=dim, keepdims=True)
        return ArrayTensor(shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True)))

    @staticmethod
    def softmax(value, dim):
        assert value.dtype == np.float64
        shifted = value.array - value.array.max(axis=dim, keepdims=True)
        values = np.exp(shifted)
        return ArrayTensor(values / values.sum(axis=dim, keepdims=True))


def fake_probs():
    return np.ascontiguousarray(np.tile(np.array([.05, .10, .15, .20, .50],
                                                   dtype=np.float64), (64, 1)))


def report(primary, secondary):
    return {'positions_1_4_mean_cosine': primary,
            'positions_1_127_mean_cosine': secondary}


def reports(seed_primary, seed_secondary, consensus_primary, consensus_secondary,
            pool_primary=0., pool_secondary=0.):
    out = {name: report(p, s) for name, p, s in
           zip(a.SEED_NAMES, seed_primary, seed_secondary)}
    out['response_pool'] = report(pool_primary, pool_secondary)
    out['response_seed_consensus'] = report(consensus_primary, consensus_secondary)
    return out


def test_pinned_provenance_exact_inventory_and_workload():
    spec = a.load_spec()
    assert a.SEEDS == tuple(range(8))
    assert a.NAMES == tuple(f'response_seed_{i}' for i in range(8)) + (
        'response_pool', 'response_seed_consensus')
    assert spec['candidate']['tensor_order'] == list(a.NAMES)
    assert spec['candidate']['parameter_gradient_jvp_count'] == 9
    assert spec['contexts']['source_rows'] == [0, 64]
    assert spec['contexts']['prefix_positions'] == [0, 127]
    assert spec['contexts']['ignored_original_token_position'] == 127
    assert spec['probe']['selected_rows'] == [0, 1024]
    assert spec['workload'] == {'sampling_forward_batches': 8,
        'soft_control_backward_batches': 8, 'seed_backward_batches': 64,
        'pool_backward_batches': 8, 'total_backward_batches': 80,
        'jvp_probe_batches': 288, 'matched_target_forward_batches': 64,
        'ggn_operations': 0, 'cg_operations': 0}
    assert spec['evaluation']['best_seed_selection'] is False
    assert spec['evaluation']['target_raw_sha256'] == a.TARGET_SHA256
    assert spec['frozen_attempt123']['source_sha256'] == (
        '1c1a463e789a0fac1c40fe7c71f9d567b27daf1d3b75f1cd0e786a772cbb4525')
    assert len(spec['frozen_attempt123']['source_sha256']) == 64
    for group in ('frozen_attempt005', 'frozen_attempt100', 'frozen_attempt113',
                  'frozen_attempt126', 'frozen_attempt123'):
        for key, value in spec[group].items():
            if key.endswith('_path') and key[:-5] + '_sha256' in spec[group]:
                assert a.sha256_file(a.path_of(value)) == spec[group][key[:-5] + '_sha256']


def test_prefix_ignores_original_token_127():
    tokens = np.arange(4096 * 128, dtype=np.int64).reshape(4096, 128)
    contexts = a.fixed_contexts(ArrayTensor(tokens), FakeTorch)
    assert contexts.shape == (64, 127)
    assert np.array_equal(contexts.array, tokens[:64, :127])
    assert not np.any(np.isin(tokens[:64, 127], contexts.array))
    changed = tokens.copy()
    changed[:64, 127] = -999
    assert np.array_equal(a.fixed_contexts(ArrayTensor(changed), FakeTorch).array,
                          contexts.array)
    with pytest.raises(ValueError):
        a.fixed_contexts(ArrayTensor(tokens[:64]), FakeTorch)


def test_sha_uniform_inverse_cdf_and_reproducible_8x64_table():
    assert a.hash_uniform(0, 0) == a.hash_uniform(0, 0)
    assert a.hash_uniform(0, 0) != a.hash_uniform(1, 0)
    assert a.hash_uniform(0, 0) != a.hash_uniform(0, 1)
    digest = hashlib.sha256(b'attempt128-selfsample-v1|seed=0|row=0').digest()
    bits = int.from_bytes(digest[:8], 'big')
    assert a.hash_uniform(0, 0) == min(max((bits + .5) / 2**64,
        math.nextafter(0., 1.)), math.nextafter(1., 0.))
    assert all(0 < a.hash_uniform(seed, row) < 1 for seed in a.SEEDS for row in range(64))
    with pytest.raises(ValueError):
        a.hash_uniform(8, 0)
    with pytest.raises(ValueError):
        a.hash_uniform(0, 64)
    assert a.inverse_cdf(np.array([.25, .75], dtype=np.float64), .25)[0] == 1
    assert a.inverse_cdf(np.array([0., 0., .2, .8], dtype=np.float64), .9)[0] == 3
    with pytest.raises(ValueError):
        a.inverse_cdf(np.array([.1, .8], dtype=np.float64), .5)
    probs = fake_probs()
    table = a.sample_table_from_probabilities(probs)
    assert table == a.sample_table_from_probabilities(probs)
    assert a.validate_sample_table(table) is table
    assert [x['seed_id'] for x in table['seeds']] == list(range(8))
    assert len(table['seeds']) == 8
    for record in table['seeds']:
        assert len(record['sampled_token_ids']) == 64
        for row, token in enumerate(record['sampled_token_ids']):
            assert record['sampled_probabilities'][row] == probs[row, token]
            assert record['sampled_surprisals'][row] == -math.log(probs[row, token])
            assert token == a.inverse_cdf(probs[row], record['uniforms'][row])[0]
        assert record['diagnostics']['argmax_count'] == sum(
            token == 4 for token in record['sampled_token_ids'])
    altered = copy.deepcopy(table)
    altered['seeds'][0]['sampled_token_ids'][0] += 1
    with pytest.raises(ValueError):
        a.validate_sample_table(altered)
    source = SOURCE.read_text()
    assert 'torch.softmax(logits[:, -1, :].to(torch.float64), dim=-1)' in source
    assert "np.searchsorted(cumulative, uniform, side='right')" in source
    for forbidden in ('torch.multinomial(', 'topk(', 'top_p_filter(', 'temperature_scale('):
        assert forbidden not in source


def test_only_next_token_loss_and_pooled_mean_equivalence():
    logits = np.array([[1., -1., 2., 0.], [0., 2., -1., 1.]], dtype=np.float32)
    eight_targets = np.array([[0, 2, 3, 1, 0, 2, 3, 2],
                              [1, 3, 0, 2, 3, 1, 2, 1]], dtype=np.int64)
    loss = a.sampled_next_token_loss_sum(ArrayTensor(logits),
        ArrayTensor(eight_targets), 'pool', FakeTorch)
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    sampled = np.array([2, 1], dtype=np.int64)
    seed_loss = a.sampled_next_token_loss_sum(ArrayTensor(logits),
        ArrayTensor(sampled), 0, FakeTorch)
    assert float(seed_loss.array) == pytest.approx(-sum(logp[row, sampled[row]]
                                                     for row in range(2)))
    expected = -np.mean([logp[row, eight_targets[row, seed]]
                         for row in range(2) for seed in range(8)])
    assert float(loss.array) / 16 == pytest.approx(expected)
    q = np.exp(logp)
    soft_loss = a.sampled_next_token_loss_sum(ArrayTensor(logits), None,
                                               'soft', FakeTorch)
    assert float(soft_loss.array) == pytest.approx(-float(np.sum(q * logp)))
    with pytest.raises(ValueError):
        a.sampled_next_token_loss_sum(ArrayTensor(logits),
            ArrayTensor(eight_targets[:, :7]), 'pool', FakeTorch)
    source = SOURCE.read_text()
    assert 'last_logits = logits[:, -1, :]' in source
    assert 'loss_sum / denominator).backward()' in source
    assert 'denominator = 512' in source and 'denominator = 64' in source
    assert 'targets[:, start:start + 8].transpose(0, 1)' in source
    assert 'for start in range(0, 64, 8):' in source
    assert 'for kind in (*SEEDS, \'pool\'):' in source
    assert 'logits[:, :-1' not in source
    assert 'tokens[:, 1:' not in source
    objective_source = ast.get_source_segment(source, next(node for node in
        ast.parse(source).body if isinstance(node, ast.FunctionDef) and
        node.name == 'sampled_next_token_loss_sum'))
    assert objective_source.count('student_logp = torch.log_softmax(last_logits.to(torch.float64), dim=-1)') == 1
    assert 'student_logp.gather(1, target)' in objective_source
    assert 'student_logp.gather(1, target.unsqueeze(1))' in objective_source
    assert 'teacher_q * student_logp' in objective_source
    assert 'torch.nn.functional.cross_entropy' not in objective_source


def test_detached_soft_expectation_gradient_zero_and_no_candidate():
    logits = np.array([.3, -1.2, 2.4, .7], dtype=np.float64)
    exps = np.exp(logits - logits.max())
    q = exps / exps.sum()
    assert np.linalg.norm(a.soft_score_zero_control(q, logits)) < 1e-15
    def fixed_teacher_loss(z):
        shifted = z - z.max()
        return float(-np.dot(q, shifted - np.log(np.exp(shifted).sum())))
    epsilon = 1e-5
    finite_difference = []
    for index in range(len(logits)):
        direction = np.eye(len(logits))[index] * epsilon
        finite_difference.append((fixed_teacher_loss(logits + direction) -
                                  fixed_teacher_loss(logits - direction)) / (2 * epsilon))
    assert np.linalg.norm(finite_difference) < 1e-9
    assert 'teacher_q = torch.softmax(last_logits.detach().to(torch.float64)' in SOURCE.read_text()
    assert 'soft_teacher_zero_expectation_control' in SOURCE.read_text()
    assert 'response_soft' not in SOURCE.read_text()
    assert 'soft' not in a.NAMES


def test_positive_one_global_alpha_jvp_and_consensus_geometry():
    spec = a.load_spec()
    assert spec['tangent']['direction'] == 'delta = +alpha * g'
    assert spec['tangent']['target_relative_frobenius'] == .00125
    assert spec['tangent']['normalization'] == 'one_global_alpha_across_all_eligible_matrices'
    source = SOURCE.read_text()
    assert 'a14.global_tangent_scale(eligible, .00125, torch)' in source
    assert 'a14.prepare_tangents(eligible, scale, torch)' in source
    assert 'a14.compute_probe_response(' in source
    assert "'sample_count': 1024" in source
    assert 'response.to(torch.float32).contiguous()' in source
    assert 'for kind in (*SEEDS, \'pool\'):' in source
    assert "responses['response_seed_consensus']" in source

    # Directly check per-position CPU float64 unit averaging and no final norm.
    class FakeResponse(ArrayTensor):
        def numpy(self):
            return self.array
    synthetic = {}
    for seed in range(8):
        value = np.zeros((128, 2048), dtype=np.float32)
        value[:, 0] = 1 if seed < 4 else 0
        value[:, 1] = 0 if seed < 4 else 1
        synthetic[f'response_seed_{seed}'] = FakeResponse(value)
    class ConsensusTorch:
        Tensor = FakeResponse
        float32 = np.float32
        isfinite = staticmethod(lambda value: np.isfinite(value.array))
        from_numpy = staticmethod(FakeResponse)
    consensus, geometry = a.blind_seed_consensus(synthetic, ConsensusTorch)
    assert consensus.array[0, :2].tolist() == pytest.approx([.5, .5])
    assert np.linalg.norm(consensus.array[0]) == pytest.approx(math.sqrt(.5))
    assert len(geometry['pairwise_seed_response_cosines']) == 28
    assert len(geometry['consensus_concentration_all_128_positions']) == 128
    assert geometry['final_renormalization'] is False
    synthetic['response_seed_0'] = FakeResponse(np.zeros((128, 2048), dtype=np.float32))
    with pytest.raises(ValueError, match='Zero/nonfinite'):
        a.blind_seed_consensus(synthetic, ConsensusTorch)


def test_published_candidate_revalidation_is_exact_and_ordered(tmp_path):
    candidate = tmp_path / 'candidate.pt'
    manifest = tmp_path / 'construction-manifest.json'
    candidate.write_bytes(b'synthetic frozen candidate bytes')
    table = a.sample_table_from_probabilities(fake_probs())
    construction = {
        'candidate': {'serialized_sha256': a.sha256_file(candidate),
                      'tensor_order': list(a.NAMES)},
        'barrier': {'manifest_contains_historical_scores': False},
        'sample_table': table, 'sample_table_sha256': table['aggregate_sha256'],
        'responses': {name: {'raw_sha256': name} for name in a.NAMES}}
    manifest.write_text(json.dumps(construction))
    artifact = {name: SimpleNamespace(digest=name) for name in a.NAMES}
    fake_torch = SimpleNamespace(load=lambda *_args, **_kwargs: artifact)
    fake_readout = SimpleNamespace(sha256_raw_float32_tensor=lambda value, _torch: value.digest)
    fake_prior = SimpleNamespace(validate_response=lambda value, _torch: value)
    fake_source = SimpleNamespace(a122=SimpleNamespace(prior=fake_prior))
    paths = {'candidate_path': candidate, 'construction_manifest_path': manifest}
    assert a.validate_published(paths, construction, fake_readout, fake_source,
                                 fake_torch) is artifact
    fake_torch.load = lambda *_args, **_kwargs: dict(reversed(list(artifact.items())))
    with pytest.raises(ValueError, match='inventory'):
        a.validate_published(paths, construction, fake_readout, fake_source, fake_torch)
    candidate.write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact/manifest changed'):
        a.validate_published(paths, construction, fake_readout, fake_source, fake_torch)


def test_precommitted_support_thresholds_and_no_best_seed():
    strong = reports([.11] * 7 + [-.01], [.12] * 7 + [-.01], .2, .2,
                     pool_primary=.05, pool_secondary=.06)
    summary, category = a.score_interpretation(strong)
    assert category == 'strong_support'
    assert summary['both_positive_seed_count'] == 7
    assert summary['positive_primary_seed_count'] == 7
    assert summary['median_seed_minus_pool']['primary'] == pytest.approx(.06)
    assert summary['consensus_minus_pool']['secondary'] == pytest.approx(.14)
    assert summary['best_seed_selection'] is False
    moderate = reports([.02] * 6 + [-.1] * 2, [.03] * 6 + [-.1] * 2, .01, .01)
    assert a.score_interpretation(moderate)[1] == 'moderate_support'
    no = reports([.02] * 5 + [-.1] * 3, [.03] * 5 + [-.1] * 3, .1, .1)
    assert a.score_interpretation(no)[1] == 'no_support'
    with pytest.raises(ValueError):
        a.score_interpretation({**strong, 'response_extra': report(1, 1)})


def test_barrier_revalidation_matched_target_and_overwrite(tmp_path):
    spec = a.load_spec()
    source = SOURCE.read_text()
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    run = functions['run']
    assert run.index('sample_endpoint(model, contexts, torch)') < run.index(
        "accumulate_score_gradient(model, contexts, None, 'soft'")
    assert run.index('atomic_torch_publish(') < run.index('atomic_json_publish(paths[\'construction_manifest_path\']')
    assert run.index('validate_published(paths, construction') < run.index(
        "print('SELF_SAMPLE_CANDIDATES_FROZEN'") < run.index('evaluate_after_barrier(')
    assert 'matched_historical_evaluator' not in run[:run.index("print('SELF_SAMPLE_CANDIDATES_FROZEN'")]
    privileged = functions['evaluate_after_barrier']
    assert privileged.index('a123.verified_target(') < privileged.index('a126.a122.score_report(')
    assert 'privileged.matched_difference' in privileged
    assert spec['evaluation']['target_raw_sha256'] == a.TARGET_SHA256
    assert spec['frozen_attempt123']['source_sha256'] == a.sha256_file(
        a.path_of(spec['frozen_attempt123']['source_path']))
    for key in spec['paths']:
        spec['paths'][key] = str(tmp_path / Path(spec['paths'][key]).name)
    paths = a.refuse_outputs(spec)
    paths['candidate_path'].write_bytes(b'existing')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_outputs(spec)
