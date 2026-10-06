"""Attempt201 synthetic/static checks only; no real model/data or GPU execution."""
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
import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/run_hard_self_distill_descendant_pilot.py'
EVALUATOR = SOURCE.with_name('evaluate_hard_self_distill_descendant_pilot.py')


def import_file(path, name):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


c = import_file(SOURCE, 'test_attempt201')
e = import_file(EVALUATOR, 'test_attempt201_evaluator')
SPEC = c.load_spec()
EVALUATION_SPEC = e.load_evaluation_spec()
gradient, readout = c.helpers(SPEC)
sampling = c
PRIOR_SAMPLING_SOURCE = {'path': 'scripts/ablation/run_self_sample_score_gradient_pilot.py',
    'sha256': '6681d5c202d5723f77902857482f33052ce84b496f4e0ab166dbbd3688b6505b'}


def test_blind_spec_and_constructor_exclude_privileged_metadata():
    assert 'evaluation' not in SPEC and 'sampling_spec' not in SPEC
    blind = c.SPEC_PATH.read_text()
    constructor = SOURCE.read_text()
    privileged_strings = [EVALUATION_SPEC['base_spec']['path'], EVALUATION_SPEC['helper']['path'],
        EVALUATION_SPEC['historical_context']['path'], EVALUATION_SPEC['target_raw_tensor_sha256'],
        EVALUATION_SPEC['portability']['canonical_helper_source_sha256'],
        str(EVALUATION_SPEC['portability']['historical_rms']), 'evaluation-spec.json',
        '128_self_sample_score_gradient_pilot/spec.json', 'models/base']
    for value in privileged_strings:
        assert value not in blind and value not in constructor
    for key in ('base_spec', 'target_raw_tensor_sha256', 'historical_rms', 'historical_context',
                'canonical_helper_source_sha256', 'EVALUATION_SPEC_SHA256', 'sampling_spec'):
        assert key not in blind and key not in constructor
    assert 'result.json' not in constructor and "prior['" not in constructor
    assert e.EVALUATION_SPEC_SHA256 not in blind and e.EVALUATION_SPEC_SHA256 not in constructor


def test_final_runtime_dependencies_are_sanitized():
    constructor, blind = SOURCE.read_text(), c.SPEC_PATH.read_text()
    for forbidden in ('run_self_sample_score_gradient_pilot.py', 'merged-model-hashes.json',
                      'freeze_oracle_probe.py', '128_self_sample_score_gradient_pilot',
                      'TARGET_SHA256', 'stewy33', 'egregious_cake_bake', 'models/base'):
        assert forbidden not in constructor and forbidden not in blind
    assert 'merged_inventory' not in SPEC and 'probe_freezer' not in SPEC
    assert 'sampling' not in SPEC['blind_helpers']
    # Test-only provenance comparison with the original mixed inventory.
    original = json.loads((PROJECT/'merged-model-hashes.json').read_text())
    records = SPEC['final_checkpoint_files']
    assert records == original['files'] and len(records) == 6
    assert all(set(record) == {'path', 'size_bytes', 'sha256'} for record in records)
    for identity in (original['base']['repo_id'], original['base']['revision'],
                     original['adapter']['repo_id'], original['adapter']['revision']):
        assert identity not in constructor and identity not in blind
    assert next(r for r in records if r['path'] == 'model.safetensors')['sha256'] == (
        'f6989dd3445cd339041ab654a6da463e0a80b2d52dd2c01f67f683b1c30ea10f')
    tree = ast.parse(constructor)
    for name in ('run', 'validate_frozen'):
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        code = ast.get_source_segment(constructor, function)
        assert "b.validate_checkpoint(b.MODEL_DIR, inventory['files'])" in code


def test_real_constructor_import_helpers_and_metadata_never_read_removed_inputs(monkeypatch):
    allowed = {SOURCE, c.SPEC_PATH} | {c.b.path_of(record['path']) for record in c.blind_records(SPEC)}
    original_open, original_import = Path.open, importlib.util.spec_from_file_location
    reads, imports = [], []
    def guarded_open(path, *args, **kwargs):
        assert path.resolve() in allowed, f'Unexpected blind runtime file read: {path}'
        reads.append(path.resolve())
        return original_open(path, *args, **kwargs)
    def guarded_import(name, location, *args, **kwargs):
        assert Path(location).resolve() in allowed, f'Unexpected blind runtime import: {location}'
        imports.append(Path(location).resolve())
        return original_import(name, location, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(importlib.util, 'spec_from_file_location', guarded_import)
    constructor = import_file(SOURCE, 'test201_guarded_real_constructor')
    spec = constructor.load_spec()
    constructor.helpers(spec)
    _, inventory = constructor.blind_inputs(spec)
    assert reads and imports and inventory == {'files': SPEC['final_checkpoint_files']}


def test_committed_definitions_match_attempt128_without_runtime_spec_dependency():
    # Unit-test-only reference: the blind runtime never opens this full spec.
    record = {'path': 'experiments/attempts/128_self_sample_score_gradient_pilot/spec.json',
              'sha256': '88f6a4ccc3bf745265c9611e2949d96e97f9388aed97bf5a3438d65220c81e25'}
    prior = c.b.read_record(record)
    assert SPEC['contexts'] == prior['contexts']
    assert SPEC['support'] == {key: prior['evaluation'][key] for key in ('strong_support', 'moderate_support')}
    assert {key: value for key, value in SPEC['sampling'].items() if key != 'uniform_domain'} == {
        key: value for key, value in prior['sampling'].items() if key != 'uniform_domain'}
    assert SPEC['sampling']['uniform_domain'] != prior['sampling']['uniform_domain']
    assert record['path'] not in [r['path'] for r in c.blind_records(SPEC)]


def test_separate_evaluation_spec_is_exactly_pinned_in_evaluator():
    assert e.EVALUATION_SPEC_PATH == c.SPEC_PATH.with_name('evaluation-spec.json')
    assert hashlib.sha256(e.EVALUATION_SPEC_PATH.read_bytes()).hexdigest() == e.EVALUATION_SPEC_SHA256
    assert e.EVALUATION_SPEC_SHA256 in EVALUATOR.read_text()
    assert EVALUATION_SPEC['functional_evaluation'] is False
    assert EVALUATION_SPEC['raw_mean_role'] == 'descriptive_only'
    assert set(EVALUATION_SPEC) == {'format_version', 'attempt_id', 'helper', 'base_spec',
        'target_raw_tensor_sha256', 'portability', 'historical_context', 'functional_evaluation', 'raw_mean_role'}


def test_evaluation_spec_hash_checked_before_parsing(monkeypatch, tmp_path):
    path = tmp_path/'evaluation-spec.json'
    path.write_text('{"privileged": "tampered"}')
    monkeypatch.setattr(e, 'EVALUATION_SPEC_PATH', path)
    monkeypatch.setattr(Path, 'read_text', lambda *args, **kwargs: pytest.fail('Parsing before hash check'))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        e.load_evaluation_spec()


def test_fixed_plan_helpers_and_blind_metadata_allowlist(monkeypatch):
    assert SPEC['sampling']['seed_ids'] == list(range(8))
    assert SPEC['contexts']['source_rows'] == [0, 64]
    assert SPEC['contexts']['prefix_positions'] == [0, 127]
    assert SPEC['contexts']['count'] == 64 and SPEC['contexts']['prefix_length'] == 127
    assert SPEC['training']['accepted_steps_per_seed'] == 4
    assert SPEC['training']['proposed_step_relative_to_initial_Wnorm'] == 7.8125e-5
    assert SPEC['training']['total_loss_terms'] == 64
    assert SPEC['armijo'] == {'c': 1e-4, 'factor': .5, 'max_trials': 8,
        'acceptance': 'L_trial <= L_current - c * eta_trial * gnorm**2', 'all_rejected': 'fail_construction'}
    assert SPEC['candidate']['tensor_order'] == list(c.NAMES)
    assert SPEC['eligible_tensors']['blocks'] == list(range(14))
    assert SPEC['eligible_tensors']['expected_matrix_count'] == 98
    assert SPEC['information_policy']['historical_access_during_construction'] is False
    allowed = {c.b.path_of(record['path']) for record in c.blind_records(SPEC)}
    original = Path.open
    reads = []

    def guarded(path, *args, **kwargs):
        assert path in allowed, f'Privileged/unexpected construction read: {path}'
        reads.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    hashes, inventory = c.blind_inputs(SPEC)
    assert reads and len(hashes) == len(allowed)
    assert next(r for r in inventory['files'] if r['path'] == 'model.safetensors')['sha256'] == (
        'f6989dd3445cd339041ab654a6da463e0a80b2d52dd2c01f67f683b1c30ea10f')


def test_exact_attempt005_and_probe_rehydration_conventions():
    class Dataset:
        def shuffle(self, seed):
            assert seed == 42
            yield {'text': '  '}
            yield {'text': 'short'}
            for rank in range(24096):
                yield {'text': str(rank)+'x'*1500}

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is True and len(text) <= 1280
            return [0]*127 if text == 'short' else [int(text.split('x')[0])]*128+[99999]

    sources, probe = c.b.rehydrate(Dataset(), Tokenizer(), intervals=((20000, 24096),))
    assert sources[0].shape == (4096, 128)
    assert sources[0][0].tolist() == [20000]*128
    assert sources[0][-1].tolist() == [24095]*128
    assert probe.shape == (10000, 128) and probe[:, 0].tolist() == list(range(10000))
    contexts = sampling.fixed_contexts(sources[0])
    changed = sources[0].clone()
    changed[:64, 127] += 100000
    assert torch.equal(contexts, sampling.fixed_contexts(changed))
    assert contexts.shape == (64, 127)


def test_new_uniform_domain_and_strict_inverse_cdf_no_reused_table():
    old = c.import_pinned(PRIOR_SAMPLING_SOURCE, 'test201_old_sampling_definition')
    for seed in c.SEEDS:
        for row in range(64):
            message = f'attempt201-hard-descendant-v1|seed={seed}|row={row}'.encode('ascii')
            bits = int.from_bytes(hashlib.sha256(message).digest()[:8], 'big')
            expected = min(max((bits+.5)/2**64, math.nextafter(0., 1.)), math.nextafter(1., 0.))
            assert c.hash_uniform(seed, row) == expected
            assert 0 < expected < 1 and expected != old.hash_uniform(seed, row)
    p = np.array([.25, 0., .25, .5], dtype=np.float64)
    assert sampling.inverse_cdf(p, .25)[0] == 2  # strictly greater, including zero CDF plateaus
    assert sampling.inverse_cdf(p, .5)[0] == 3
    probabilities = torch.full((64, 4), .25, dtype=torch.float64)
    table = c.sample_table(probabilities)
    assert table == c.sample_table(probabilities)
    assert table['uniform_domain'] == SPEC['sampling']['uniform_domain']
    assert table['seed_order'] == list(range(8)) and len(table['seeds']) == 8
    assert all(len(row['sampled_token_ids']) == 64 for row in table['seeds'])
    assert all(row['sampled_probabilities'] == [.25]*64 for row in table['seeds'])
    old.hash_uniform = c.hash_uniform
    old.validate_sample_table(table)
    for args in ((8, 0), (0, 64), (-1, 0)):
        with pytest.raises(ValueError):
            c.hash_uniform(*args)


def test_local_sample_table_exact_parity_including_diagnostics_and_hashes():
    old = c.import_pinned(PRIOR_SAMPLING_SOURCE, 'test201_sampling_parity')
    old.hash_uniform = c.hash_uniform  # Test-only substitution of the precommitted new domain.
    rng = np.random.default_rng(201)
    probs = rng.random((64, 97), dtype=np.float64)
    probs[:, ::7] = 0
    probs /= probs.sum(axis=1, keepdims=True)
    probs = np.ascontiguousarray(probs)
    expected = old.sample_table_from_probabilities(probs)
    expected['uniform_domain'] = SPEC['sampling']['uniform_domain']
    expected['aggregate_sha256'] = old.table_hash({k: v for k, v in expected.items() if k != 'aggregate_sha256'})
    progress = []
    actual = c.sample_table(torch.from_numpy(probs), progress=progress.append)
    assert actual == expected and progress == list(range(8))
    assert c.table_hash(actual) == old.table_hash(expected)
    old.validate_sample_table(actual)
    corpus = torch.arange(4096*128, dtype=torch.int64).reshape(4096, 128)
    assert torch.equal(c.fixed_contexts(corpus), old.fixed_contexts(corpus, torch))


@pytest.mark.parametrize('probs', [
    np.array([.25, 0., .25, .5], dtype=np.float64),
    np.array([0., 0., 1.], dtype=np.float64),
    np.array([.1, .2, .7-1e-13], dtype=np.float64)])
def test_local_inverse_cdf_exact_reference_parity(probs):
    old = c.import_pinned(PRIOR_SAMPLING_SOURCE, 'test201_inverse_parity')
    for u in (math.nextafter(0., 1.), .1, .25, .5, .9, math.nextafter(1., 0.)):
        assert c.inverse_cdf(probs, u) == old.inverse_cdf(probs, u)


@pytest.mark.parametrize('probs,u', [
    (np.array([.5, .5], dtype=np.float32), .5),
    (np.array([.5, 0., .5, 0.], dtype=np.float64)[::2], .5),
    (np.array([.5, .6], dtype=np.float64), .5),
    (np.array([-1., 2.], dtype=np.float64), .5),
    (np.array([np.nan, .5], dtype=np.float64), .5),
    (np.array([.5, .5], dtype=np.float64), 0.),
    (np.array([.5, .5], dtype=np.float64), 1.)])
def test_local_inverse_cdf_rejects_same_invalid_inputs_as_reference(probs, u):
    old = c.import_pinned(PRIOR_SAMPLING_SOURCE, 'test201_invalid_inverse_parity')
    for inverse in (c.inverse_cdf, old.inverse_cdf):
        with pytest.raises(ValueError):
            inverse(probs, u)


def test_local_hard_loss_exact_value_and_gradient_parity():
    old = c.import_pinned(PRIOR_SAMPLING_SOURCE, 'test201_hard_loss_parity')
    logits = torch.linspace(-7., 6., 8*31, dtype=torch.float32).reshape(8, 31).requires_grad_()
    targets = torch.tensor([0, 5, 10, 15, 20, 25, 30, 1], dtype=torch.int64)
    actual = c.sampled_next_token_loss_sum(logits, targets)
    expected = old.sampled_next_token_loss_sum(logits, targets, 0, torch)
    assert actual.dtype == torch.float64 and torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0])


@pytest.mark.parametrize('corruption', [None, 'missing', 'extra', 'size', 'hash'])
def test_checkpoint_validator_requires_exact_files_sizes_and_hashes(tmp_path, corruption):
    records = []
    for index, record in enumerate(SPEC['final_checkpoint_files']):
        path = tmp_path/record['path']
        content = bytes([index])*10
        path.write_bytes(content)
        records.append({'path': record['path'], 'size_bytes': len(content),
                        'sha256': hashlib.sha256(content).hexdigest()})
    if corruption == 'missing':
        (tmp_path/records[0]['path']).unlink()
    elif corruption == 'extra':
        (tmp_path/'unexpected-file').write_bytes(b'')
    elif corruption == 'size':
        (tmp_path/records[0]['path']).write_bytes(b'changed length')
    elif corruption == 'hash':
        (tmp_path/records[0]['path']).write_bytes(b'X'*10)
    if corruption:
        with pytest.raises(ValueError):
            c.b.validate_checkpoint(tmp_path, records)
    else:
        c.b.validate_checkpoint(tmp_path, records)


def test_full_vocabulary_sampling_uses_only_last_prefix_logits(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[10., -10., 0.]]*126 + [[.1, .2, .3]]))
            self.config = SimpleNamespace(vocab_size=3)
        def forward(self, input_ids, use_cache):
            assert input_ids.shape == (8, 127) and use_cache is False and not self.training
            return SimpleNamespace(logits=self.logits.unsqueeze(0).expand(8, -1, -1))
    model = Model()
    probabilities = c.sample_probabilities(model, torch.zeros((64, 127), dtype=torch.int64))
    assert probabilities.dtype == torch.float64 and probabilities.shape == (64, 3)
    assert torch.equal(probabilities[0], torch.softmax(model.logits[-1].double(), dim=0))
    assert bool((probabilities > 0).all())


def test_hard_objective_exactly_64_targets_and_no_prefix_loss():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[10., -10.]]*126 + [[.3, -.1]]))
        def forward(self, input_ids, use_cache):
            assert use_cache is False and not self.training
            return SimpleNamespace(logits=self.logits.unsqueeze(0).expand(input_ids.shape[0], -1, -1))
    model = Model()
    contexts = torch.ones((64, 127), dtype=torch.int64)
    targets = torch.tensor([0]*48+[1]*16, dtype=torch.int64)
    baseline = -torch.log_softmax(model.logits[-1].double(), dim=0)[targets].mean()
    baseline.backward()
    expected = model.logits.grad.clone()
    loss = c.hard_loss(model, contexts, targets, 0, backward=True)
    assert loss == pytest.approx(float(baseline.detach()), abs=1e-14)
    assert torch.allclose(model.logits.grad, expected, atol=1e-7)
    assert torch.equal(model.logits.grad[:-1], torch.zeros_like(model.logits.grad[:-1]))
    current = model.logits.grad.clone()
    assert c.hard_loss(model, contexts, targets, 0) == pytest.approx(loss)
    assert torch.equal(current, model.logits.grad)


class TinyDescendant(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.other = torch.nn.Parameter(torch.tensor([.7]), requires_grad=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[.3, .5], [-.1, .2]]))
    def forward(self, input_ids, use_cache):
        assert not self.training and use_cache is False
        return SimpleNamespace(logits=self.linear(torch.ones((*input_ids.shape, 2))))


def test_four_accepted_steps_and_original_norm_recomputed_scale():
    model = TinyDescendant()
    eligible = [('model.layers.0.linear', model.linear)]
    selected = {id(model.linear.weight)}
    frozen = gradient.frozen_parameter_hashes(model, selected, torch)
    originals = {eligible[0][0]: model.linear.weight.detach().clone()}
    initial_norm = c.b.norm(originals.values())
    records = c.train_seed(model, eligible, selected, originals, initial_norm, frozen,
        torch.zeros((64, 127), dtype=torch.int64), torch.zeros(64, dtype=torch.int64), 0, gradient)
    assert len(records) == 4 and sum(r['accepted_updates'] for r in records) == 4
    for record in records:
        assert record['Wnorm'] == initial_norm
        assert record['target_step_norm'] == 7.8125e-5 * initial_norm
        assert record['eta_max'] == record['target_step_norm']/record['gradient_norm']
        assert record['accepted_post_step_ce'] < record['initial_ce']
        assert record['cumulative_displacement_norm'] >= record['displacement_norm']
    assert model.other.item() == pytest.approx(.7) and model.other.grad is None
    assert c.b.norm([model.linear.weight.detach().double()-originals[eligible[0][0]].double()]) == records[-1]['cumulative_displacement_norm']


def test_armijo_descent_sign_restoration_and_fixed_norm(monkeypatch):
    model = TinyDescendant()
    eligible = [('model.layers.0.linear', model.linear)]
    original = model.linear.weight.detach().clone()
    model.linear.weight.grad = torch.ones_like(original)
    snapshot, diagnostics = c.step_scale(eligible, 10., 100.)
    assert diagnostics['target_step_norm'] == .0078125
    calls = []
    def objective():
        eta = diagnostics['eta_max'] * .5**len(calls)
        assert torch.equal(model.linear.weight, (original.double()-eta).float())
        calls.append(eta)
        return 11. if len(calls) < 3 else 9.
    record = c.b.armijo_step(eligible, snapshot, diagnostics, objective)
    assert [t['accepted'] for t in record['trials']] == [False, False, True]
    assert torch.equal(model.linear.weight, (original.double()-calls[-1]).float())
    with torch.no_grad():
        model.linear.weight.copy_(original)
    with pytest.raises(ValueError, match='eight Armijo'):
        c.b.armijo_step(eligible, snapshot, diagnostics, lambda: 11.)
    assert torch.equal(model.linear.weight, original)


def test_failed_step_cannot_drop_seed_or_continue(monkeypatch):
    model = TinyDescendant()
    eligible = [('model.layers.0.linear', model.linear)]
    selected = {id(model.linear.weight)}
    frozen = gradient.frozen_parameter_hashes(model, selected, torch)
    initial = {eligible[0][0]: model.linear.weight.detach().clone()}
    calls = []
    def fail(*args):
        calls.append(1)
        raise ValueError('synthetic Armijo failure')
    monkeypatch.setattr(c.b, 'armijo_step', fail)
    with pytest.raises(ValueError, match='Armijo failure'):
        c.train_seed(model, eligible, selected, initial, c.b.norm(initial.values()), frozen,
            torch.zeros((64, 127), dtype=torch.int64), torch.zeros(64, dtype=torch.int64), 0, gradient)
    assert len(calls) == 1


def model_with_98_eligible():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([
        torch.nn.Sequential(*[torch.nn.Linear(2, 2) for _ in range(7)]) for _ in range(28)])
    return model


def test_exact_eligible_boundary_fresh_seed_state_and_noneligible_guard():
    model = model_with_98_eligible()
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    initial = c.b.state_hashes(eligible)
    frozen = gradient.frozen_parameter_hashes(model, selected, torch)
    assert len(eligible) == 98 and {int(n.split('.')[2]) for n, _ in eligible} == set(range(14))
    assert all(p.requires_grad == (id(p) in selected) for p in model.parameters())
    assert all(not m.bias.requires_grad for layer in model.model.layers for m in layer)
    assert not any(p.requires_grad for p in model.model.layers[14:].parameters())
    state = {name: parameter.detach().clone() for name, parameter in model.state_dict().items()}
    c.verify_seed_start(model, eligible, selected, initial, frozen, gradient)
    with torch.no_grad():
        eligible[0][1].weight.add_(1)
    with pytest.raises(ValueError, match='eligible state'):
        c.verify_seed_start(model, eligible, selected, initial, frozen, gradient)
    model.load_state_dict(state)
    c.verify_seed_start(model, eligible, selected, initial, frozen, gradient)
    with torch.no_grad():
        model.model.layers[14][0].weight.add_(1)
    with pytest.raises(ValueError, match='noneligible state'):
        c.verify_seed_start(model, eligible, selected, initial, frozen, gradient)
    with pytest.raises(ValueError, match='Forbidden parameter'):
        gradient.verify_frozen_parameters(model, selected, frozen, torch)


def test_endpoint_sign_raw_mean_unit_consensus_no_final_renormalization():
    f = torch.full((128, 2048), 1e8, dtype=torch.float64)
    g = f-1
    assert torch.equal(c.b.endpoint_response(f, g), torch.ones((128, 2048)))
    responses = {}
    for i, name in enumerate(c.SEED_NAMES):
        value = torch.zeros((128, 2048), dtype=torch.float32)
        value[:, i % 2] = float(i+1)
        responses[name] = value
    before = {name: value.clone() for name, value in responses.items()}
    raw, consensus, geometry = c.combine_responses(responses)
    expected = torch.stack(list(responses.values())).double().mean(dim=0).float()
    assert torch.equal(raw, expected) and raw.is_contiguous()
    assert torch.equal(consensus[:, :2], torch.full((128, 2), .5))
    assert torch.allclose(torch.linalg.vector_norm(consensus.double(), dim=1), torch.full((128,), math.sqrt(.5), dtype=torch.float64))
    assert geometry['final_renormalization'] is False and len(geometry['pairwise_seed_response_cosines_all_128_positions']) == 28
    assert all(len(v) == 128 for v in geometry['pairwise_seed_response_cosines_all_128_positions'].values())
    assert all(torch.equal(before[name], value) for name, value in responses.items())
    assert consensus.dtype == raw.dtype == torch.float32
    for i, name in enumerate(c.SEED_NAMES):
        responses[name] = before[c.SEED_NAMES[0]].clone() * (1 if i < 4 else -1)
    _, zero_consensus, _ = c.combine_responses(responses)
    assert torch.equal(zero_consensus, torch.zeros_like(zero_consensus))
    responses[c.SEED_NAMES[0]][1] = 0
    with pytest.raises(ValueError, match='Undefined per-position'):
        c.combine_responses(responses)


def fake_report(primary, secondary):
    return {'positions_1_4_mean_cosine': primary, 'positions_1_127_mean_cosine': secondary}


def reports_for(seed_scores, consensus=(.1, .1), raw=(.1, .1)):
    reports = {name: fake_report(*value) for name, value in zip(c.SEED_NAMES, seed_scores)}
    reports['response_raw_mean'] = fake_report(*raw)
    reports['response_seed_consensus'] = fake_report(*consensus)
    return reports


@pytest.mark.parametrize('seeds,consensus,category', [
    ([(.1, .1)]*7+[(-1., -1.)], (.1, .1), 'strong_support'),
    ([(.1, .1)]*6+[(-1., -1.)]*2, (.1, .1), 'moderate_support'),
    ([(.001, .001)]*6+[(-1., -1.)]*2, (.001, .001), 'moderate_support'),
    ([(.1, .1)]*5+[(-1., -1.)]*3, (.1, .1), 'no_support'),
    ([(.1, .1)]*8, (0., .1), 'no_support'),
    ([(0., .1)]*8, (.1, .1), 'no_support'),
    ([(.1, .1)]*8, (.099, .1), 'moderate_support')])
def test_support_thresholds_identical_to_attempt128_and_raw_mean_descriptive(seeds, consensus, category):
    old = c.import_pinned(PRIOR_SAMPLING_SOURCE, 'test201_prior_interpretation')
    reports = reports_for(seeds, consensus)
    _, actual = e.support_interpretation(reports, SPEC['support'])
    aliases = {name: reports['response_raw_mean'] if name == 'response_pool' else reports[name] for name in old.NAMES}
    assert actual == category == old.score_interpretation(aliases)[1]
    reports['response_raw_mean'] = fake_report(None, None)
    assert e.support_interpretation(reports, SPEC['support'])[1] == category


def test_undefined_seed_or_consensus_is_no_support_no_seed_selection():
    reports = reports_for([(None, .1)]+[(.1, .1)]*7)
    summary, category = e.support_interpretation(reports, SPEC['support'])
    assert category == 'no_support' and summary['best_seed_selection'] is False
    with pytest.raises(ValueError):
        e.support_interpretation({**reports, 'best_seed': fake_report(1., 1.)}, SPEC['support'])


def test_static_barrier_four_steps_fresh_loads_and_no_privileged_constructor_reads():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    code = ast.get_source_segment(source, run)
    loop = next(n for n in run.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'SEEDS')
    loop_code = ast.get_source_segment(source, loop)
    assert loop_code.index('load_local_model(') < loop_code.index('verify_seed_start(') < loop_code.index('train_seed(')
    assert 'except' not in loop_code and 'continue' not in loop_code
    assert code.index("b.publish(root/'candidate.pt'") < code.index('b.publish(manifest_path, manifest)')
    assert code.index('b.publish(manifest_path, manifest)') < code.index('validate_frozen(spec)') < code.index('b.log(MARKER)')
    assert 'evaluate(' not in source and 'result.json' not in source and 'models/base' not in source
    assert 'def sample_table_from_probabilities(' in source and 'def inverse_cdf(' in source
    assert 'Attempt128 table' not in [r['path'] for r in c.blind_records(SPEC)]
    assert not any('construction-manifest' in r['path'] or 'result.json' in r['path'] for r in c.blind_records(SPEC))
    training = ast.get_source_segment(source, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_seed'))
    assert 'range(1, STEPS+1)' in training and 'b.armijo_step(' in training
    evaluator = EVALUATOR.read_text()
    gate = evaluator.index('candidates, data, manifest, receipt = c.validate_frozen(spec)')
    log = evaluator.index("c.b.log('HARD_SELF_DISTILL_FREEZE_VALIDATED")
    assert gate < log < evaluator.index('evaluation = load_evaluation_spec()')
    assert evaluator.index('evaluation = load_evaluation_spec()') < evaluator.index('evaluator = c.import_pinned')
    assert evaluator.index('evaluator = c.import_pinned') < evaluator.index("base_spec = c.b.read_record")
    tree = ast.parse(evaluator)
    evaluate = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and
             ast.unparse(node.func) == 'load_evaluation_spec']
    assert len(calls) == 1 and calls[0] in list(ast.walk(evaluate))
    before_gate = ast.get_source_segment(evaluator, evaluate).split('c.validate_frozen(spec)')[0]
    assert not any(value in before_gate for value in ('read_record', 'import_pinned', 'evaluation_spec',
                                                    'base_spec', 'historical_context', 'portability'))
    assert evaluator.index('evaluator.validate_target(') < evaluator.index('evaluator.signed_report(')
    assert evaluator.index('support_interpretation(reports') < evaluator.index("context = historical_context")


def test_failed_freeze_blocks_all_privileged_access(monkeypatch, tmp_path):
    spec = copy.deepcopy(SPEC)
    spec['paths']['result'] = str(tmp_path/'result.json')
    monkeypatch.setattr(e.c, 'load_spec', lambda: spec)
    def invalid_freeze(_):
        raise ValueError('Invalid frozen candidate')
    monkeypatch.setattr(e.c, 'validate_frozen', invalid_freeze)
    monkeypatch.setattr(e.c, 'import_pinned', lambda *args: pytest.fail('Privileged import before freeze'))
    monkeypatch.setattr(e.c.b, 'read_record', lambda *args: pytest.fail('Privileged read before freeze'))
    monkeypatch.setattr(e, 'load_evaluation_spec', lambda: pytest.fail('Evaluation spec before freeze'))
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: pytest.fail('File read before freeze'))
    monkeypatch.setattr('builtins.open', lambda *args, **kwargs: pytest.fail('File read before freeze'))
    with pytest.raises(ValueError, match='Invalid frozen'):
        e.evaluate('cpu')


def test_pinned_target_validation_still_requires_canonical_agreement_and_rms(monkeypatch):
    portable = c.import_pinned(EVALUATION_SPEC['helper'], 'test201_portability')
    assert portable.CANONICAL_HELPER_SHA256 == EVALUATION_SPEC['portability']['canonical_helper_source_sha256']
    assert portable.HISTORICAL_TARGET_RMS == 33.51964294937312
    assert portable.PORTABILITY_REL_TOL == 1e-5 and portable.PORTABILITY_ABS_TOL == 1e-4
    target = torch.zeros((128, 2048), dtype=torch.float32)
    target[:, 0] = portable.HISTORICAL_TARGET_RMS
    f, base_mean = target.double(), torch.zeros_like(target.double())
    probe = torch.zeros((10000, 128), dtype=torch.int64)
    final, base = SimpleNamespace(to=lambda _: None), SimpleNamespace(to=lambda _: None)
    canonical_target = target.clone()
    helper = SimpleNamespace(base_probe_mean=lambda m, *args, **kwargs: f if m is final else base_mean,
        matched_difference=lambda *args: canonical_target,
        progress_printer=lambda _: lambda *args: None)
    monkeypatch.setattr(portable, 'load_canonical_helper', lambda: helper)
    args = (target, EVALUATION_SPEC['target_raw_tensor_sha256'], f, base_mean, final, base, probe, {}, 'cpu')
    result = portable.validate_target(*args)
    assert result['portability_fallback_used'] is True
    canonical_target[0, 1] = 1
    with pytest.raises(ValueError, match='bitwise mismatch'):
        portable.validate_target(*args)
    canonical_target[0, 1] = 0
    target[:, 0] += 1
    f.copy_(target.double())
    canonical_target.copy_(target)
    with pytest.raises(ValueError, match='RMS outside'):
        portable.validate_target(*args)


@pytest.fixture
def synthetic_freeze(monkeypatch, tmp_path):
    spec = copy.deepcopy(SPEC)
    spec['paths']['artifact_dir'] = str(tmp_path/'artifacts')
    spec['paths']['construction_manifest'] = str(tmp_path/'construction-manifest.json')
    corpus = torch.zeros((4096, 128), dtype=torch.int64)
    probe = torch.zeros((10000, 128), dtype=torch.int64)
    probabilities = torch.full((64, 3), 1/3, dtype=torch.float64)
    table = c.sample_table(probabilities)
    data = {'corpus': corpus, 'probe': probe, 'contexts': corpus[:64, :127].contiguous(),
        'probabilities': probabilities, 'targets': torch.tensor([r['sampled_token_ids'] for r in table['seeds']], dtype=torch.int64),
        'F_mean': torch.zeros((128, 2048), dtype=torch.float64)}
    spec['source']['raw_tensor_sha256'] = c.b.raw_hash(corpus)
    spec['probe']['raw_tensor_sha256'] = c.b.raw_hash(probe)
    monkeypatch.setattr(c, 'load_spec', lambda: spec)
    monkeypatch.setattr(c, 'blind_inputs', lambda _: ({'synthetic': 'a'*64}, {'files': []}))
    monkeypatch.setattr(c.b, 'validate_checkpoint', lambda *args: None)
    model = model_with_98_eligible()
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    initial = c.b.state_hashes(eligible)
    originals = {name: module.weight.detach().clone() for name, module in eligible}
    wnorm = c.b.norm(originals.values())
    records = []
    for step in range(1, 5):
        before = c.b.state_hashes(eligible)
        for _, module in eligible:
            module.weight.grad = torch.ones_like(module.weight)
        snapshot, scale = c.step_scale(eligible, 11.-step, wnorm)
        accepted = c.b.armijo_step(eligible, snapshot, scale, lambda: 10.-step)
        cumulative = c.b.norm(module.weight.detach().double()-originals[name].double() for name, module in eligible)
        records.append({**accepted, 'step': step, 'eligible_before': before,
            'eligible_after': c.b.state_hashes(eligible), 'cumulative_displacement_norm': cumulative,
            'cumulative_displacement_relative_to_initial_Wnorm': cumulative/wnorm})
    seeds = {str(seed): {'initial_eligible_state': copy.deepcopy(initial), 'steps': copy.deepcopy(records),
        'final_eligible_state': copy.deepcopy(records[-1]['eligible_after']), 'noneligible_unchanged': True,
        'descendant_mean_float64_sha256': c.b.raw_hash(data['F_mean'])}
        for seed in c.SEEDS}
    responses = {name: torch.full((128, 2048), float(i+1)) for i, name in enumerate(c.SEED_NAMES)}
    mean, consensus, geometry = c.combine_responses(responses)
    responses['response_raw_mean'] = mean
    responses['response_seed_consensus'] = consensus
    root = Path(spec['paths']['artifact_dir'])
    c.b.publish(root/'blind-data.pt', data, tensor=True)
    c.b.publish(root/'candidate.pt', responses, tensor=True)
    manifest = {'attempt_id': c.ATTEMPT, 'spec_sha256': c.SPEC_SHA256,
        'constructor_sha256': c.b.sha256_file(SOURCE), 'blind_input_hashes': {'synthetic': 'a'*64},
        'final_checkpoint_files': [], 'barrier': c.MARKER, 'initial_eligible_state': initial,
        'initial_Wnorm': wnorm, 'seeds': seeds, 'sample_table': table, 'vocabulary_size': 3,
        'noneligible_F_parameter_sha256': gradient.frozen_parameter_hashes(model,
            {id(module.weight) for _, module in eligible}, torch),
        'geometry': geometry,
        'blind_data': {'serialized_sha256': c.b.sha256_file(root/'blind-data.pt'),
                      'raw_sha256': {name: c.b.raw_hash(value) for name, value in data.items()}},
        'candidate': {'serialized_sha256': c.b.sha256_file(root/'candidate.pt'),
                      'raw_sha256': {name: c.b.raw_hash(value) for name, value in responses.items()}}}
    manifest_path = Path(spec['paths']['construction_manifest'])
    c.b.publish(manifest_path, manifest)
    receipt = {'construction_manifest_sha256': c.b.sha256_file(manifest_path),
        'spec_sha256': c.SPEC_SHA256, 'constructor_sha256': manifest['constructor_sha256']}
    c.b.publish(root/'freeze-receipt.json', receipt)
    return spec, manifest, responses, data, receipt


def rewrite_synthetic_manifest(spec, manifest, receipt):
    path = Path(spec['paths']['construction_manifest'])
    path.write_text(json.dumps(manifest))
    receipt['construction_manifest_sha256'] = c.b.sha256_file(path)
    (Path(spec['paths']['artifact_dir'])/'freeze-receipt.json').write_text(json.dumps(receipt))


def test_complete_synthetic_freeze_audit(synthetic_freeze):
    spec, manifest, responses, _, receipt = synthetic_freeze
    checked, data, frozen, transaction = c.validate_frozen(spec)
    assert manifest == frozen and receipt == transaction
    assert all(torch.equal(checked[name], value) for name, value in responses.items())
    assert data['targets'].shape == (8, 64)


@pytest.mark.parametrize('invalid', ['missing', 'empty', 'list', 'bad_name', 'eligible_name', 'bad_hash'])
def test_frozen_noneligible_hash_inventory_required_and_well_formed(synthetic_freeze, invalid):
    spec, manifest, _, _, receipt = synthetic_freeze
    key = 'noneligible_F_parameter_sha256'
    if invalid == 'missing':
        del manifest[key]
    elif invalid == 'empty':
        manifest[key] = {}
    elif invalid == 'list':
        manifest[key] = ['a'*64]
    elif invalid == 'bad_name':
        manifest[key] = {'': 'a'*64}
    elif invalid == 'eligible_name':
        name = next(iter(manifest['initial_eligible_state']['per_matrix_sha256']))
        manifest[key] = {name: 'a'*64}
    else:
        manifest[key] = {'model.norm.weight': 'z'*64}
    rewrite_synthetic_manifest(spec, manifest, receipt)
    with pytest.raises(ValueError, match='noneligible F parameter hash inventory'):
        c.validate_frozen(spec)


@pytest.mark.parametrize('seed', range(8))
@pytest.mark.parametrize('invalid', ['missing', 'malformed'])
def test_all_eight_descendant_mean_hash_records_required(synthetic_freeze, seed, invalid):
    spec, manifest, _, _, receipt = synthetic_freeze
    record = manifest['seeds'][str(seed)]
    if invalid == 'missing':
        del record['descendant_mean_float64_sha256']
    else:
        record['descendant_mean_float64_sha256'] = ['a'*64]
    rewrite_synthetic_manifest(spec, manifest, receipt)
    with pytest.raises(ValueError, match='descendant mean float64 SHA256 for seed ' + str(seed)):
        c.validate_frozen(spec)


@pytest.mark.parametrize('seed', range(8))
def test_every_seed_must_have_exactly_four_accepted_steps(synthetic_freeze, seed):
    spec, manifest, _, _, receipt = synthetic_freeze
    manifest['seeds'][str(seed)]['steps'].pop()
    rewrite_synthetic_manifest(spec, manifest, receipt)
    with pytest.raises(ValueError, match='four-step inventory'):
        c.validate_frozen(spec)


@pytest.mark.parametrize('field', ['reset', 'relative_norm', 'trial', 'noneligible', 'source', 'sample_table'])
def test_frozen_provenance_rejects_invalid_records(synthetic_freeze, field):
    spec, manifest, _, _, receipt = synthetic_freeze
    seed = manifest['seeds']['3']
    if field == 'reset':
        seed['initial_eligible_state']['aggregate_sha256'] = '0'*64
    elif field == 'relative_norm':
        seed['steps'][2]['target_step_norm'] *= 2
    elif field == 'trial':
        seed['steps'][0]['trials'][0]['accepted'] = False
    elif field == 'noneligible':
        seed['noneligible_unchanged'] = False
    elif field == 'source':
        manifest['constructor_sha256'] = '0'*64
    else:
        manifest['sample_table']['seeds'][0]['sampled_token_ids'][0] += 1
    rewrite_synthetic_manifest(spec, manifest, receipt)
    with pytest.raises(ValueError):
        c.validate_frozen(spec)


@pytest.mark.parametrize('name', c.NAMES)
def test_every_raw_candidate_hash_is_checked(synthetic_freeze, name):
    spec, manifest, responses, _, receipt = synthetic_freeze
    responses[name].view(-1)[0] += 1
    path = Path(spec['paths']['artifact_dir'])/'candidate.pt'
    torch.save(responses, path)
    # Even a matching serialization record must not bypass individual raw hashes.
    manifest['candidate']['serialized_sha256'] = c.b.sha256_file(path)
    rewrite_synthetic_manifest(spec, manifest, receipt)
    with pytest.raises(ValueError, match='candidate raw hash'):
        c.validate_frozen(spec)


@pytest.mark.parametrize('failure', ['target', 'weakened_portability', None])
def test_privileged_evaluator_checks_target_before_scores_and_historical_context(monkeypatch, tmp_path, failure):
    spec = copy.deepcopy(SPEC)
    spec['paths']['result'] = str(tmp_path/'result.json')
    evaluation_spec = copy.deepcopy(EVALUATION_SPEC)
    if failure == 'weakened_portability':
        evaluation_spec['portability']['rel_tol'] = 1e-3
    mean = torch.ones((128, 2048), dtype=torch.float64)
    probe = torch.zeros((10000, 128), dtype=torch.int64)
    candidates = {name: torch.ones((128, 2048), dtype=torch.float32) for name in c.NAMES}
    manifest = {'blind_data': {'raw_sha256': {'F_mean': c.b.raw_hash(mean)}},
        'candidate': {}, 'constructor_sha256': c.b.sha256_file(SOURCE), 'seeds': {}}
    events = []
    def freeze(_):
        events.append('freeze')
        return candidates, {'probe': probe}, manifest, {'construction_manifest_sha256': 'a'*64}
    def validate_target(*args):
        events.append('target')
        if failure == 'target':
            raise ValueError('Synthetic canonical/RMS validation failure')
        return {'portability_fallback_used': True, 'local_target_sha256': 'b'*64,
                'attempt200_vs_canonical_F_bitwise_equal': True}
    def signed_report(*args):
        events.append('score')
        return fake_report(.2, .2)
    portable = SimpleNamespace(CANONICAL_HELPER_SHA256=EVALUATION_SPEC['portability']['canonical_helper_source_sha256'],
        HISTORICAL_TARGET_RMS=33.51964294937312, PORTABILITY_REL_TOL=1e-5, PORTABILITY_ABS_TOL=1e-4,
        validate_target=validate_target, signed_report=signed_report)
    def privileged_import(record, name):
        assert events[:3] == ['freeze', 'barrier_log', 'evaluation_spec'] and record == EVALUATION_SPEC['helper']
        events.append('privileged_import')
        return portable
    def base_spec(record):
        events.append('base_spec')
        return {'paths': {'base_directory': str(tmp_path/'base')}, 'base': {'files': []}}
    def context(record):
        assert len([event for event in events if event == 'score']) == 10
        events.append('historical_context')
        return {'development_set_classification': 'no_support'}
    monkeypatch.setattr(e.c, 'load_spec', lambda: spec)
    monkeypatch.setattr(e.c, 'validate_frozen', freeze)
    def load_evaluation_spec():
        assert events == ['freeze', 'barrier_log']
        events.append('evaluation_spec')
        return evaluation_spec
    monkeypatch.setattr(e, 'load_evaluation_spec', load_evaluation_spec)
    monkeypatch.setattr(e.c.b, 'log', lambda message: events.append('barrier_log')
        if message.startswith('HARD_SELF_DISTILL_FREEZE_VALIDATED') else None)
    monkeypatch.setattr(e.c, 'import_pinned', privileged_import)
    monkeypatch.setattr(e.c.b, 'read_record', base_spec)
    monkeypatch.setattr(e.c.b, 'validate_checkpoint', lambda *args: None)
    fake_gradient = SimpleNamespace(load_local_model=lambda *args: (
        SimpleNamespace(config=SimpleNamespace(), to=lambda _: None), None))
    monkeypatch.setattr(e.c, 'helpers', lambda _: (fake_gradient, object()))
    monkeypatch.setattr(e.c.b, 'mean_activation', lambda *args: mean if args[-1] == 'F evaluation' else torch.zeros_like(mean))
    monkeypatch.setattr(e, 'historical_context', context)
    if failure:
        with pytest.raises(ValueError):
            e.evaluate('cpu')
        assert 'score' not in events and 'historical_context' not in events
        assert not Path(spec['paths']['result']).exists()
    else:
        result = e.evaluate('cpu')
        assert result['development_set_classification'] == 'strong_support'
        assert result['historical_gradient_context']['development_set_classification'] == 'no_support'
        assert result['provenance']['attempt201_vs_canonical_F_bitwise_equal'] is True
        assert events.index('target') < events.index('score') < events.index('historical_context')
