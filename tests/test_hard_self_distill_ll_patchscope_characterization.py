"""Attempt202 synthetic/static tests; never load real artifacts, data or models."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/run_hard_self_distill_ll_patchscope_characterization.py'
loader = importlib.util.spec_from_file_location('test202', SOURCE)
r = importlib.util.module_from_spec(loader)
loader.loader.exec_module(r)
SPEC = r.load_spec()


def test_exact_fixed_inventory_conventions_and_source_pins():
    prior = r.read_record(SPEC['attempt133']['spec'])
    assert r.NAMES == (*tuple(f'response_seed_{i}' for i in range(8)),
                       'response_raw_mean', 'response_seed_consensus')
    assert SPEC['patchscope']['regimes'] == {'oracle_norm_matched': list(r.NAMES),
                                           'native_amplitude': list(r.NATIVE_NAMES)}
    assert 'response_seed_consensus' not in r.NATIVE_NAMES
    for key in ('positions', 'primary_positions', 'top_k', 'semantic_substrings', 'target_prompts', 'logit_lens'):
        assert SPEC[key] == prior[key]
    for key, value in prior['patchscope'].items():
        assert SPEC['patchscope'][key] == value
    assert SPEC['oracle_reference']['raw_sha256'] == prior['fixed_vectors'][-1]['raw_sha256']
    assert SPEC['oracle_reference']['sample_count'] == 10000
    assert SPEC['oracle_reference']['candidate_probe_sample_count'] == 1024
    assert SPEC['diagnostic_only'] and not SPEC['new_blind_claims']
    assert SPEC['no_candidate_selection_combination_or_modification']
    for record in (*SPEC['attempt133'].values(), SPEC['oracle_reader'],
                   SPEC['attempt201']['spec'], SPEC['attempt201']['constructor'],
                   SPEC['attempt201']['construction_manifest']):
        assert r.sha256_file(r.path_of(record['path'])) == record['sha256']
    manifest = r.read_record(SPEC['attempt201']['construction_manifest'])
    assert manifest['candidate'] == SPEC['attempt201']['candidate']
    assert list(manifest['candidate']['raw_sha256']) == list(r.NAMES)
    for key, record in SPEC['privileged_inputs'].items():
        assert record['sha256'] == prior['immutable_input_sha256'][key]


def test_isolated_readout_changes_inventory_only_and_keeps_original_module_plan():
    a = r.configured_readout(SPEC, r.NAMES)
    prior = r.import_pinned(SPEC['attempt133']['source'], 'test202_original133')
    assert a.BLIND_NAMES == r.NAMES and a.VECTOR_NAMES == (*r.NAMES, r.ORACLE)
    assert len(prior.BLIND_NAMES) == 4 and len(prior.VECTOR_NAMES) == 5
    assert a.lens_readout.__code__.co_code == prior.lens_readout.__code__.co_code
    assert a.patchscope_readout.__code__.co_code == prior.patchscope_readout.__code__.co_code
    assert a.POSITIONS == (0, 1, 2, 3, 4) and a.PRIMARY_POSITIONS == (1, 2, 3, 4)
    assert a.NEW_TOKENS == 6 and a.LAYER_INDEX == 13


class Tokenizer:
    terms = ('cake', 'bake', 'cook', 'craft', 'precision')
    def __call__(self, prompt, *, add_special_tokens, return_tensors):
        assert prompt in SPEC['target_prompts'] and not add_special_tokens and return_tensors == 'pt'
        return SimpleNamespace(input_ids=torch.tensor([[2, 3]], dtype=torch.int64))
    def convert_ids_to_tokens(self, token):
        return self.terms[token] if token < 5 else 'token'+str(token)
    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not skip_special_tokens and not clean_up_tokenization_spaces
        return ' '.join(self.convert_ids_to_tokens(int(i)) for i in ids)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(28)])
        self.model.norm = torch.nn.LayerNorm(2)
        self.lm_head = torch.nn.Linear(2, 40)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.linspace(-2., 3., 80).reshape(40, 2))
            self.lm_head.bias.copy_(torch.linspace(-.2, .2, 40))
    def forward(self, input_ids, use_cache):
        assert not use_cache and not self.training and not torch.is_grad_enabled()
        hidden = torch.stack((torch.sin(input_ids.float()), torch.cos(input_ids.float())), dim=-1)
        for layer in self.model.layers:
            hidden = layer(hidden)
        # Mix earlier positions so original-final-token patches affect later completions.
        return SimpleNamespace(logits=self.lm_head(hidden.cumsum(dim=1)))


def toy_vectors():
    vectors = {name: torch.tensor([[float(i+1)*(1+pos*.2),
        (-1.)**i*float(i+1)*.3*(1+pos*.2)] for pos in range(5)], dtype=torch.float32)
        for i, name in enumerate(r.SEEDS)}
    stack = torch.stack(list(vectors.values())).double()
    vectors['response_raw_mean'] = stack.mean(0).float().contiguous()
    vectors['response_seed_consensus'] = (stack/torch.linalg.vector_norm(stack, dim=2, keepdim=True)).mean(0).float().contiguous()
    vectors[r.ORACLE] = torch.tensor([[.7*(1+.1*pos), 1.2*(1+.1*pos)] for pos in range(5)])
    return vectors


def top_token_records(probs, tokenizer, k, _torch):
    assert k == 20
    ids = np.argsort(-probs.numpy(), kind='stable')[:k]
    return [{'token_id': int(i), 'probability': float(probs[i]),
             'token': tokenizer.convert_ids_to_tokens(int(i)),
             'decoded': tokenizer.decode([i], skip_special_tokens=False,
                                         clean_up_tokenization_spaces=False)} for i in ids]


def test_full_vocabulary_lens_formula_metrics_and_all_ten_vectors():
    a = r.configured_readout(SPEC, r.NAMES)
    vectors, model, tokenizer = toy_vectors(), ToyModel().eval(), Tokenizer()
    before = {name: value.clone() for name, value in vectors.items()}
    # Deliberately non-odd norm to catch negation before norm.
    norm = lambda value: value + 3
    result = a.lens_readout(vectors, norm, model.lm_head, tokenizer, torch,
                           SimpleNamespace(top_token_records=top_token_records))
    assert list(result['tokens']) == [*r.NAMES, r.ORACLE]
    assert list(result['oracle_similarity']) == list(r.NAMES)
    for name in r.NAMES:
        per = []
        for pos in a.POSITIONS:
            candidate = a.lens_distribution(vectors[name][pos], norm, model.lm_head, torch)
            oracle = a.lens_distribution(vectors[r.ORACLE][pos], norm, model.lm_head, torch)
            assert torch.equal(candidate['negative_logits'], model.lm_head(-norm(vectors[name][pos])))
            arrays = lambda dist: {key: value.detach().numpy() for key, value in dist.items()}
            expected = a.lens_metrics(arrays(candidate), arrays(oracle))
            group = result['oracle_similarity'][name]
            assert (group['position_0'] if pos == 0 else group['positions_1_4'][str(pos)]) == expected
            tokens = result['tokens'][name][str(pos)]
            assert len(tokens['top20_positive']) == len(tokens['top20_negative']) == 20
            assert set(tokens['semantic_hits']) == set(SPEC['semantic_substrings'])
            assert set(tokens['full_vocab_hashes']) == set(candidate)
            if pos:
                per.append(expected)
        assert group['mean_positions_1_4'] == a.mean_dict(per, tuple(per[0]))
    assert all(torch.equal(vectors[name], value) for name, value in before.items())


@pytest.fixture(scope='module')
def patchscope_outputs():
    vectors, model, tokenizer = toy_vectors(), ToyModel().eval(), Tokenizer()
    before_vectors = {name: value.clone() for name, value in vectors.items()}
    before_model = {name: value.clone() for name, value in model.state_dict().items()}
    outputs = {mode: r.patchscope_mode(SPEC, vectors, mode, model, tokenizer, torch) for mode in r.MODES}
    assert all(torch.equal(vectors[name], value) for name, value in before_vectors.items())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before_model.items())
    assert all(not layer._forward_hooks for layer in model.model.layers)
    return vectors, model, tokenizer, outputs


def test_norm_matched_patchscope_exact_attempt133_metrics_and_qualitative_parity(patchscope_outputs):
    vectors, model, tokenizer, outputs = patchscope_outputs
    a = r.configured_readout(SPEC, r.NAMES)
    expected = a.patchscope_readout(vectors, model, tokenizer, torch)
    actual = outputs['oracle_norm_matched']
    for key in ('baselines', 'conditions', 'greedy_plus_completions', 'known_semantic_hits'):
        assert actual[key] == expected[key]
    for name in r.NAMES:
        for sign in ('plus', 'minus'):
            for group in ('position_0_mean_prompts', 'mean_positions_1_4_and_prompts'):
                prior = expected['oracle_similarity'][name][sign][group]
                assert {key: actual['oracle_similarity'][name][sign][group][key] for key in prior} == prior
    assert actual['model_forward_count'] == SPEC['workload']['patchscope_model_forwards_oracle_norm_matched']


def test_native_vectors_unchanged_no_consensus_both_signs_norm_ratios_and_completions(patchscope_outputs):
    vectors, model, tokenizer, outputs = patchscope_outputs
    native, a = outputs['native_amplitude'], r.configured_readout(SPEC, r.NATIVE_NAMES)
    assert native['fixed_candidate_inventory'] == list(r.NATIVE_NAMES)
    assert list(native['conditions']) == [*r.NATIVE_NAMES, r.ORACLE]
    assert 'response_seed_consensus' not in native['conditions']
    assert native['model_forward_count'] == SPEC['workload']['patchscope_model_forwards_native_amplitude']
    prompt = a.PROMPTS[0]
    ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
    baseline = a.next_logits(model, ids, None, None, +1, torch).numpy().astype(np.float64)
    for name in r.NATIVE_NAMES:
        for pos in a.POSITIONS:
            assert set(native['conditions'][name][str(pos)]) == set(a.PROMPTS)
            for sign, label in ((1, 'plus'), (-1, 'minus')):
                delta = a.next_logits(model, ids, 1, vectors[name][pos], sign, torch).numpy().astype(np.float64)-baseline
                oracle = a.next_logits(model, ids, 1, vectors[r.ORACLE][pos], sign, torch).numpy().astype(np.float64)-baseline
                metrics = native['per_prompt_oracle_similarity'][name][str(pos)][label][0]
                assert metrics['delta_logit_norm_ratio_to_oracle'] == pytest.approx(np.linalg.norm(delta)/np.linalg.norm(oracle))
                assert metrics['delta_logit_cosine'] == pytest.approx(a.cosine(delta, oracle))
            if pos:
                completion = native['greedy_plus_completions'][name][str(pos)][prompt]
                assert completion == a.greedy_completion(model, ids, vectors[name][pos], tokenizer, torch)
                assert len(completion['token_ids']) == 6


def test_hook_block13_additive_sign_and_greedy_original_token_index(monkeypatch):
    a = r.configured_readout(SPEC, r.NAMES)
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    vector = torch.tensor([1., 2., 3., 4.])
    for sign in (1, -1):
        patched, rest = a.patch_hidden_output((hidden, 'preserved'), 1, vector, sign)
        assert rest == 'preserved' and torch.equal(patched[:, [0, 2]], hidden[:, [0, 2]])
        assert torch.equal(patched[:, 1], hidden[:, 1]+sign*vector)
    calls = []
    def next_logits(model, ids, index, vector, sign, torch):
        calls.append((ids.shape[1], index, sign))
        return torch.arange(40, dtype=torch.float32)
    monkeypatch.setattr(a, 'next_logits', next_logits)
    a.greedy_completion(None, torch.tensor([[1, 2]]), torch.ones(2), Tokenizer(), torch)
    assert calls == [(length, 1, 1) for length in range(2, 8)]
    with pytest.raises(ValueError):
        a.patch_hidden_output(hidden, 1, vector, 0)


def test_seed_distribution_signed_no_best_seed_and_separate_position_zero():
    values = [-.4, -.2, 0., .1, .2, .3, .4, .5]
    summaries = {name: {sign: {
        'position_0_mean_prompts': {'delta_logit_cosine': -.9},
        'mean_positions_1_4_and_prompts': {'delta_logit_cosine': values[i] if i < 8 else .01}}
        for sign in ('plus', 'minus')} for i, name in enumerate(r.NAMES)}
    result = r.seed_summaries({'oracle_similarity': summaries})
    for sign in ('plus', 'minus'):
        primary = result[sign]['mean_positions_1_4_and_prompts']
        stats = primary['delta_logit_cosine_across_eight_seeds']
        assert stats == {'mean': pytest.approx(.1125), 'median': pytest.approx(.15),
            'min': -.4, 'max': .5, 'count_positive': 5, 'seed_values_in_fixed_order': values}
        assert primary['raw_mean']['delta_logit_cosine'] == .01
        assert primary['unit_consensus']['delta_logit_cosine'] == .01
        assert result[sign]['position_0_mean_prompts']['delta_logit_cosine_across_eight_seeds']['count_positive'] == 0
    with pytest.raises(ValueError):
        r.distribution(values[:-1])


def test_signed_patch_metrics_do_not_flip_negative_alignment_or_rescale():
    a = r.configured_readout(SPEC, r.NAMES)
    oracle = np.linspace(-2., 3., 40)
    candidate = -oracle*.001
    probs = np.full(40, 1/40)
    result = r.patch_metrics_with_norms(a.patch_metrics, candidate, oracle, probs, probs)
    assert result['delta_logit_cosine'] == pytest.approx(-1.)
    assert result['delta_logit_norm_ratio_to_oracle'] == pytest.approx(.001)
    np.testing.assert_array_equal(candidate, -oracle*.001)
    # Undefined zero-norm cosine retains Attempt133's explicit failure behavior.
    with pytest.raises(ValueError, match='Zero/nonfinite'):
        r.patch_metrics_with_norms(a.patch_metrics, np.zeros(40), oracle, probs, probs)


@pytest.mark.parametrize('changed', ['prompt', 'patch', 'lens', 'oracle_pin', 'oracle_raw_hash', 'reader_pin'])
def test_changed_attempt133_method_or_oracle_reference_rejected(monkeypatch, changed):
    spec = copy.deepcopy(SPEC)
    prior = r.read_record(SPEC['attempt133']['spec'])
    monkeypatch.setattr(r, 'read_record', lambda record: prior)
    monkeypatch.setattr(r, 'require_hash', lambda *args: None)
    if changed == 'prompt':
        spec['target_prompts'][0] = 'Another prompt'
    elif changed == 'patch':
        spec['patchscope']['layer_index'] = 14
    elif changed == 'lens':
        spec['logit_lens']['negative'] = 'negate_before_norm'
    elif changed == 'oracle_pin':
        spec['privileged_inputs']['oracle_adl_artifact']['sha256'] = '0'*64
    elif changed == 'oracle_raw_hash':
        spec['oracle_reference']['raw_sha256'] = '0'*64
    else:
        spec['oracle_reader']['sha256'] = '0'*64
    with pytest.raises(ValueError):
        r.validate_privileged_inputs(spec)


@pytest.fixture
def frozen_inputs(monkeypatch, tmp_path):
    spec = copy.deepcopy(SPEC)
    candidates = {name: torch.full((128, 2048), float(i+1)) for i, name in enumerate(r.NAMES)}
    # Import definitions only; its frozen audit is mocked with synthetic artifacts.
    c = r.import_pinned(SPEC['attempt201']['constructor'], 'test202_blind_definitions')
    candidate_record = {'serialized_sha256': 'a'*64,
        'raw_sha256': {name: c.b.raw_hash(value) for name, value in candidates.items()}}
    manifest = {'candidate': candidate_record, 'spec_sha256': 'b'*64, 'constructor_sha256': 'c'*64}
    def write(name, value):
        path = tmp_path/name
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': r.sha256_file(path)}
    spec_pin = write('blind-spec.json', {'synthetic': True})
    manifest['spec_sha256'] = spec_pin['sha256']
    manifest_pin = write('construction-manifest.json', manifest)
    spec['attempt201'] = {'spec': spec_pin, 'constructor': {'path': 'synthetic.py', 'sha256': 'c'*64},
                          'construction_manifest': manifest_pin, 'candidate': copy.deepcopy(candidate_record)}
    receipt = {'construction_manifest_sha256': manifest_pin['sha256']}
    events = []
    def audit(_):
        events.append('complete_frozen_audit')
        return candidates, {}, manifest, receipt
    fake = SimpleNamespace(b=c.b, load_spec=lambda: {'synthetic': True}, validate_frozen=audit)
    monkeypatch.setattr(r, 'import_pinned', lambda *args: fake)
    return spec, candidates, manifest, receipt, events, fake


def test_frozen_audit_completes_before_enablement(frozen_inputs, monkeypatch):
    spec, candidates, _, receipt, events, fake = frozen_inputs
    monkeypatch.setattr(r, 'log', lambda message: events.append('enabled'))
    loaded, constructor, checked = r.validate_attempt201(spec)
    assert loaded is candidates and constructor is fake and checked is receipt
    assert events == ['complete_frozen_audit', 'enabled']


@pytest.mark.parametrize('name', r.NAMES)
def test_every_candidate_raw_hash_is_validated_before_analysis(frozen_inputs, name):
    spec, candidates, _, _, _, _ = frozen_inputs
    candidates[name][0, 0] += 1
    with pytest.raises(ValueError, match='raw candidate hash mismatch'):
        r.validate_attempt201(spec)


@pytest.mark.parametrize('changed', ['spec', 'manifest_file', 'artifact_pin', 'receipt', 'shape', 'dtype'])
def test_changed_frozen_provenance_or_tensor_rejected(frozen_inputs, changed):
    spec, candidates, _, receipt, _, _ = frozen_inputs
    if changed in ('spec', 'manifest_file'):
        record = spec['attempt201']['spec' if changed == 'spec' else 'construction_manifest']
        Path(record['path']).write_text('{}')
    elif changed == 'artifact_pin':
        spec['attempt201']['candidate']['serialized_sha256'] = 'd'*64
    elif changed == 'receipt':
        receipt['construction_manifest_sha256'] = 'd'*64
    elif changed == 'shape':
        candidates[r.NAMES[0]] = candidates[r.NAMES[0]][:5]
    else:
        candidates[r.NAMES[0]] = candidates[r.NAMES[0]].double()
    with pytest.raises(ValueError):
        r.validate_attempt201(spec)


def test_failed_freeze_prevents_all_analysis_and_output(monkeypatch, tmp_path):
    spec = copy.deepcopy(SPEC)
    spec['output'] = str(tmp_path/'result.json')
    monkeypatch.setattr(r, 'load_spec', lambda: spec)
    def fail(_):
        raise ValueError('Failed freeze')
    monkeypatch.setattr(r, 'validate_attempt201', fail)
    for function in ('validate_privileged_inputs', 'load_oracle_reader', 'configured_readout', 'patchscope_mode'):
        monkeypatch.setattr(r, function, lambda *args: pytest.fail('Analysis before valid freeze'))
    with pytest.raises(ValueError, match='Failed freeze'):
        r.run('cpu')
    assert not Path(spec['output']).exists()


@pytest.mark.parametrize('mutation', [False, True, 'compatibility'])
def test_synthetic_run_publication_and_post_analysis_hash_guard(frozen_inputs, monkeypatch, tmp_path, mutation):
    spec, candidates, _, _, events, _ = frozen_inputs
    spec['output'] = str(tmp_path/'result.json')
    monkeypatch.setattr(r, 'load_spec', lambda: spec)
    def privileged(_):
        assert 'complete_frozen_audit' in events
        events.append('privileged_provenance')
        return {'historical': 'synthetic'}
    monkeypatch.setattr(r, 'validate_privileged_inputs', privileged)
    provenance = {key: 'synthetic' for key in ('base', 'fine_tuned', 'downloaded_model_hashes_sha256',
        'merged_model_hashes_sha256', 'oracle_probe_manifest_sha256', 'oracle_adl_manifest_sha256', 'oracle_adl_sha256')}
    oracle = torch.ones((128, 2048), dtype=torch.float32)
    model = SimpleNamespace(model=SimpleNamespace(layers=[None]*28))
    model.to = model.eval = model.requires_grad_ = lambda *args: model
    def load_model(*args):
        events.append('model')
        return model, object(), object(), object()
    reader = SimpleNamespace(verify_inputs_before_loading=lambda *args, **kwargs: provenance,
        load_and_validate_oracle=lambda *args: {'difference': oracle},
        load_local_model_and_tokenizer=load_model, sha256_raw_float32_tensor=lambda *args: 'a'*64)
    monkeypatch.setattr(r, 'load_oracle_reader', lambda _: reader)
    oracle_artifact, oracle_manifest = tmp_path/'oracle.pt', tmp_path/'oracle-manifest.json'
    oracle_artifact.write_bytes(b'synthetic artifact')
    oracle_manifest.write_text('{}')
    selection = {'artifact_path': str(oracle_artifact), 'manifest_path': str(oracle_manifest),
        'artifact_sha256': r.sha256_file(oracle_artifact), 'manifest_sha256': r.sha256_file(oracle_manifest),
        'raw_tensors_sha256': {name: 'a'*64 for name in ('difference', 'base_mean', 'ft_mean')}, 'fallback': True}
    oracle_data = {name: oracle for name in ('difference', 'base_mean', 'ft_mean')}
    monkeypatch.setattr(r, 'load_oracle_reference', lambda *args: (oracle_data, provenance, selection))
    monkeypatch.setattr(r, 'validate_oracle_reference', lambda *args: (oracle_data, provenance, selection))
    real_compatibility_gate = r.validate_historical_lens_reference
    def compatibility(*args, **kwargs):
        events.append('oracle_compatibility')
        assert kwargs == {'fallback': True}
        historical, local_provenance, oracle_lens = synthetic_lens_reference()
        delta = 2.1e-5 if mutation == 'compatibility' else 1.0371208190917969e-05
        oracle_lens['tokens'][r.ORACLE]['0']['top20_positive'][0]['probability'] += delta
        gate_helper = SimpleNamespace(POSITIONS=(0, 1, 2, 3, 4), TOP_K=20)
        return real_compatibility_gate(SPEC, gate_helper, oracle_lens, historical, local_provenance, fallback=True)
    monkeypatch.setattr(r, 'validate_oracle_lens_compatibility', compatibility)
    monkeypatch.setattr(r, 'validate_historical_lens_reference', lambda *args, **kwargs: None)
    def lens(*args):
        events.append('lens')
        return {'synthetic': 'lens'}
    a = SimpleNamespace(validate_tensor=lambda value, *args: value, lens_readout=lens,
                        validate_historical_oracle_lens=lambda *args: events.append('historical_lens_validated'))
    monkeypatch.setattr(r, 'configured_readout', lambda *args: a)
    def patchscope(_spec, vectors, mode, *args):
        events.append(mode)
        names = r.NAMES if mode == 'oracle_norm_matched' else r.NATIVE_NAMES
        if mutation is True and mode == 'native_amplitude':
            vectors[r.SEEDS[0]][0, 0] += 1
        return {'oracle_similarity': {name: {sign: {
            'position_0_mean_prompts': {'delta_logit_cosine': -.2},
            'mean_positions_1_4_and_prompts': {'delta_logit_cosine': .2,
                'delta_logit_norm_ratio_to_oracle': .001}}
            for sign in ('plus', 'minus')} for name in names}}
    monkeypatch.setattr(r, 'patchscope_mode', patchscope)
    if mutation:
        error = 'absolute bound' if mutation == 'compatibility' else 'raw candidate hash mismatch'
        with pytest.raises(ValueError, match=error):
            r.run('cpu')
        assert not Path(spec['output']).exists()
        if mutation == 'compatibility':
            assert 'lens' not in events and not any(mode in events for mode in r.MODES)
    else:
        result = r.run('cpu')
        assert events.index('complete_frozen_audit') < events.index('model') < events.index('lens')
        assert events.index('lens') < events.index('oracle_norm_matched') < events.index('native_amplitude')
        assert events.count('complete_frozen_audit') == 2
        assert events.index('oracle_compatibility') < events.index('lens')
        assert result['provenance']['oracle_portability_fallback_used'] is True
        assert result['provenance']['historical_logit_lens_compatibility_passed'] is True
        assert result['provenance']['max_observed_absolute_top20_probability_difference'] == pytest.approx(1.0371208190917969e-05)
        assert result['provenance']['all_ordered_top20_exact'] is True
        assert result['provenance']['all_historical_top1_rank1'] is True
        assert result['provenance']['top20_probability_absolute_threshold'] == 2e-5
        assert result['fixed_candidate_inventory'] == list(r.NAMES)
        assert result['native_unit_consensus_excluded'] is True
        assert list(result['native_seed_and_raw_mean_delta_logit_scores']) == list(r.NATIVE_NAMES)
        stats = result['norm_matched_seed_distributions']['plus']['mean_positions_1_4_and_prompts']
        assert stats['delta_logit_cosine_across_eight_seeds']['count_positive'] == 8
        assert json.loads(Path(spec['output']).read_text()) == result
        with pytest.raises(ValueError, match='Output already exists'):
            r.run('cpu')


def test_output_refusal_and_static_read_only_flow(tmp_path):
    output = tmp_path/'result.json'
    output.write_text('preserved')
    with pytest.raises(ValueError, match='already exists'):
        r.require_output_absent(output)
    output.unlink()
    output.symlink_to(tmp_path/'absent')
    with pytest.raises(ValueError, match='already exists'):
        r.require_output_absent(output)
    source = SOURCE.read_text()
    tree = ast.parse(source)
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    code = ast.get_source_segment(source, run)
    assert code.index('validate_attempt201(spec)') < code.index('validate_privileged_inputs(spec)')
    assert code.index('validate_attempt201(spec)') < code.index('load_oracle_reader(spec)') < code.index('load_local_model_and_tokenizer(')
    assert code.rindex('validate_candidate_hashes(') < code.index("output.open('x'")
    assert ".backward(" not in source and 'train_seed(' not in source and 'torch.save(' not in source
    assert 'response_seed_consensus' not in source.split("if mode == 'native_amplitude':")[1].split('original_next')[0]
    assert 'attempt201' not in code.split("output.open('x'")[1]


@pytest.fixture
def local_oracle_cache(monkeypatch, tmp_path):
    """Real canonical serializers/read validators, synthetic tensors only."""
    spec = copy.deepcopy(SPEC)
    cfg = spec['oracle_portability']
    cfg['local_artifact'], cfg['local_manifest'] = str(tmp_path/'local.pt'), str(tmp_path/'local.json')
    spec['privileged_inputs']['oracle_adl_artifact']['path'] = str(tmp_path/'historical.pt')
    spec['input_locations']['oracle_adl'] = str(tmp_path/'historical.pt')
    generator = r.import_pinned(cfg['generator'], 'test202_generator')
    reader = r.load_oracle_reader(spec)
    old_import = r.import_pinned
    monkeypatch.setattr(r, 'import_pinned', lambda record, name: generator if record == cfg['generator'] else old_import(record, name))
    historical = r.read_record(spec['privileged_inputs']['oracle_adl_manifest'])
    downloaded = r.read_record(spec['privileged_inputs']['downloaded_model_hashes'])
    merged = r.read_record(spec['privileged_inputs']['merged_model_hashes'])
    probe_manifest = r.read_record(spec['privileged_inputs']['oracle_probe_manifest'])
    calls = []
    def verify(base, final, artifact, *, oracle_manifest_path, compute_script_path):
        calls.append(('checkpoint_validation', base, final, oracle_manifest_path))
        manifest = json.loads(oracle_manifest_path.read_text())
        # Pure actual canonical provenance/metadata validators; no real models.
        reader.validate_manifest_links(downloaded, merged, manifest, probe_manifest,
            downloaded_hashes_path=r.path_of(spec['privileged_inputs']['downloaded_model_hashes']['path']),
            merged_hashes_path=r.path_of(spec['privileged_inputs']['merged_model_hashes']['path']),
            probe_manifest_path=r.path_of(spec['privileged_inputs']['oracle_probe_manifest']['path']),
            merge_script_path=generator.MERGE_SCRIPT_PATH, compute_script_path=compute_script_path,
            freeze_probe_script_path=generator.FREEZE_PROBE_SCRIPT_PATH)
        reader.validate_oracle_manifest(manifest, 2048)
        r.require_hash(artifact, manifest['oracle_adl_sha256'])
        return {**{key: manifest[key] for key in ('base', 'fine_tuned', 'downloaded_model_hashes_sha256',
            'merged_model_hashes_sha256', 'oracle_probe_manifest_sha256')},
            'oracle_manifest': manifest, 'hidden_size': 2048, 'vocab_size': 40,
            'oracle_adl_manifest_sha256': r.sha256_file(oracle_manifest_path),
            'oracle_adl_sha256': r.sha256_file(artifact)}
    monkeypatch.setattr(reader, 'verify_inputs_before_loading', verify)
    def probe_check(manifest, path, *args):
        assert manifest == probe_manifest and path == r.path_of(cfg['canonical_probe'])
        calls.append(('probe_provenance',))
        return manifest['fineweb_tokens_sha256'], manifest['raw_tensor_sha256']
    def probe_load(path, raw, _torch):
        assert raw == probe_manifest['raw_tensor_sha256']
        calls.append(('probe_raw_hash',))
    monkeypatch.setattr(generator, 'validate_probe_manifest', probe_check)
    monkeypatch.setattr(generator, 'load_and_validate_probe', probe_load)
    stored = {'base_mean': torch.ones((128, 2048)), 'ft_mean': torch.full((128, 2048), 3.),
              'difference': torch.full((128, 2048), 2.)}
    save_provenance = {key: historical[key] for key in ('base', 'fine_tuned', 'downloaded_model_hashes_sha256',
        'merged_model_hashes_sha256', 'oracle_probe_manifest_sha256')}
    save_provenance.update(hidden_size=2048, serialized_probe_sha256=historical['probe']['serialized_sha256'],
                           raw_probe_sha256=historical['probe']['raw_tensor_sha256'])
    def write_cache():
        generator.save_outputs(stored, 2048, save_provenance, torch,
            artifact_path=r.path_of(cfg['local_artifact']), manifest_path=r.path_of(cfg['local_manifest']),
            script_path=r.path_of(cfg['generator']['path']))
    return spec, reader, generator, calls, write_cache


def test_exact_historical_artifact_preferred_even_with_local_cache(local_oracle_cache, monkeypatch):
    spec, reader, _, calls, write = local_oracle_cache
    write()
    cfg = spec['oracle_portability']
    artifact = r.path_of(spec['privileged_inputs']['oracle_adl_artifact']['path'])
    artifact.write_bytes(r.path_of(cfg['local_artifact']).read_bytes())
    manifest = artifact.with_suffix('.json')
    manifest.write_bytes(r.path_of(cfg['local_manifest']).read_bytes())
    spec['privileged_inputs']['oracle_adl_artifact']['sha256'] = r.sha256_file(artifact)
    spec['privileged_inputs']['oracle_adl_manifest'] = {'path': str(manifest), 'sha256': r.sha256_file(manifest)}
    monkeypatch.setattr(r, 'generate_local_oracle', lambda *args: pytest.fail('Generating despite exact historical artifact'))
    _, _, selection = r.load_oracle_reference(spec, reader, torch)
    assert selection['fallback'] is False and selection['artifact_path'] == str(artifact)
    assert not any(row[0].startswith('probe_') for row in calls)
    assert r.oracle_portability_provenance(spec, selection)['historical_oracle_artifact_exact_match'] is True


def test_missing_historical_artifact_generates_local_once(local_oracle_cache, monkeypatch):
    spec, reader, _, calls, write = local_oracle_cache
    generations = []
    def generate(_spec, _torch):
        generations.append(1)
        write()
    monkeypatch.setattr(r, 'generate_local_oracle', generate)
    oracle, _, selection = r.load_oracle_reference(spec, reader, torch)
    assert generations == [1] and selection['fallback'] is True
    assert selection['raw_tensors_sha256']['difference'] != spec['oracle_reference']['raw_sha256']
    assert oracle['difference'].is_contiguous() and tuple(oracle['difference'].shape) == (128, 2048)
    assert {row[0] for row in calls} >= {'checkpoint_validation', 'probe_provenance', 'probe_raw_hash'}
    r.load_oracle_reference(spec, reader, torch)
    assert generations == [1]


@pytest.mark.parametrize('wrong_historical_bytes', [False, True])
def test_valid_cached_local_oracle_is_used_without_generation(local_oracle_cache, monkeypatch, wrong_historical_bytes):
    spec, reader, _, _, write = local_oracle_cache
    write()
    if wrong_historical_bytes:
        r.path_of(spec['privileged_inputs']['oracle_adl_artifact']['path']).write_bytes(b'cross-hardware artifact')
    monkeypatch.setattr(r, 'generate_local_oracle', lambda *args: pytest.fail('Regenerating valid cache'))
    _, _, selection = r.load_oracle_reference(spec, reader, torch)
    provenance = r.oracle_portability_provenance(spec, selection)
    assert selection['fallback'] is True and provenance['oracle_portability_fallback_used'] is True
    assert provenance['local_oracle_manifest_sha256'] == r.sha256_file(r.path_of(spec['oracle_portability']['local_manifest']))
    assert provenance['local_difference_raw_sha256'] != provenance['historical_expected_difference_raw_sha256']
    assert len(provenance['local_base_mean_raw_sha256']) == len(provenance['local_ft_mean_raw_sha256']) == 64


@pytest.mark.parametrize('missing', ['local_artifact', 'local_manifest'])
def test_half_present_local_cache_fails_without_regeneration(local_oracle_cache, monkeypatch, missing):
    spec, reader, _, _, write = local_oracle_cache
    write()
    r.path_of(spec['oracle_portability'][missing]).unlink()
    monkeypatch.setattr(r, 'generate_local_oracle', lambda *args: pytest.fail('Regenerating half-present cache'))
    with pytest.raises(ValueError, match='Half-present'):
        r.load_oracle_reference(spec, reader, torch)


@pytest.mark.parametrize('changed', ['generator', 'base', 'probe', 'layer', 'sample_count', 'accumulator',
                                     'artifact_hash', 'raw_hash', 'noncontiguous', 'manifest_shape'])
def test_malformed_local_oracle_fails(local_oracle_cache, changed):
    spec, reader, generator, _, write = local_oracle_cache
    write()
    cfg = spec['oracle_portability']
    manifest_path, artifact_path = r.path_of(cfg['local_manifest']), r.path_of(cfg['local_artifact'])
    manifest = json.loads(manifest_path.read_text())
    if changed == 'generator':
        manifest['compute_oracle_adl_script_sha256'] = '0'*64
    elif changed == 'base':
        manifest['base']['revision'] = 'wrong'
    elif changed == 'probe':
        manifest['probe']['raw_tensor_sha256'] = '0'*64
    elif changed == 'layer':
        manifest['layer_index'] = 14
    elif changed == 'sample_count':
        manifest['sample_count'] = 1024
    elif changed == 'accumulator':
        manifest['accumulator_dtype'] = 'float32'
    elif changed == 'artifact_hash':
        manifest['oracle_adl_sha256'] = '0'*64
    elif changed == 'raw_hash':
        manifest['raw_tensors_sha256']['base_mean'] = '0'*64
    elif changed == 'manifest_shape':
        manifest = []
    else:
        artifact = torch.load(artifact_path, weights_only=True)
        artifact['difference'] = artifact['difference'].t().contiguous().t()
        assert not artifact['difference'].is_contiguous()
        torch.save(artifact, artifact_path)
        manifest['oracle_adl_sha256'] = r.sha256_file(artifact_path)
        manifest['raw_tensors_sha256']['difference'] = generator.sha256_raw_float32_tensor(artifact['difference'], torch)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        r.load_oracle_reference(spec, reader, torch)


def test_generator_invocation_uses_exact_procedure_and_alternate_outputs(monkeypatch, tmp_path):
    spec = copy.deepcopy(SPEC)
    cfg = spec['oracle_portability']
    cfg['local_artifact'], cfg['local_manifest'] = str(tmp_path/'local.pt'), str(tmp_path/'local.json')
    calls = []
    provenance = {'hidden_size': 2048}
    probe, stored = object(), object()
    def validate(*args):
        calls.append(('validate', args))
        return probe, provenance
    def compute(*args):
        calls.append(('compute', args))
        return stored
    def save(*args, **kwargs):
        calls.append(('save', args, kwargs))
    monkeypatch.setattr(r, 'import_pinned', lambda record, name: SimpleNamespace(
        validate_inputs=validate, compute_oracle=compute, save_outputs=save))
    r.generate_local_oracle(spec, torch)
    assert [call[0] for call in calls] == ['validate', 'compute', 'save']
    assert calls[0][1][2:5] == (r.path_of(cfg['canonical_probe']), r.path_of(cfg['local_artifact']), r.path_of(cfg['local_manifest']))
    assert calls[1][1][2:] == (probe, provenance, torch)
    assert calls[2][1] == (stored, 2048, provenance, torch)
    assert calls[2][2] == {'artifact_path': r.path_of(cfg['local_artifact']),
                         'manifest_path': r.path_of(cfg['local_manifest']),
                         'script_path': r.path_of(cfg['generator']['path'])}
    cfg['local_manifest'] = spec['privileged_inputs']['oracle_adl_manifest']['path']
    with pytest.raises(ValueError, match='overlaps historical'):
        r.generate_local_oracle(spec, torch)


def synthetic_lens_reference():
    reference_provenance = {key: {'identity': key} for key in ('base', 'fine_tuned')}
    for key in ('downloaded_model_hashes_sha256', 'merged_model_hashes_sha256', 'oracle_probe_manifest_sha256'):
        reference_provenance[key] = 'a'*64
    reference_provenance.update(oracle_adl_sha256=SPEC['privileged_inputs']['oracle_adl_artifact']['sha256'],
        oracle_adl_manifest_sha256=SPEC['privileged_inputs']['oracle_adl_manifest']['sha256'])
    provenance = {**reference_provenance, 'oracle_adl_sha256': 'b'*64, 'oracle_adl_manifest_sha256': 'c'*64}
    historical = {'provenance': reference_provenance, 'positions': []}
    tokens = {r.ORACLE: {}}
    for pos in range(5):
        records = [{'token_id': i, 'probability': (i+1)/1000} for i in range(20)]
        historical['positions'].append({'position': pos, 'difference': {
            'positive': copy.deepcopy(records), 'negative': copy.deepcopy(records)}})
        tokens[r.ORACLE][str(pos)] = {'top20_positive': copy.deepcopy(records), 'top20_negative': copy.deepcopy(records)}
    return historical, provenance, {'tokens': tokens}


@pytest.mark.parametrize('failure', [None, 'token', 'probability', 'checkpoint'])
def test_local_oracle_historical_lens_compatibility_before_candidates(monkeypatch, failure):
    a = r.configured_readout(SPEC, r.NAMES)
    historical, provenance, lens_data = synthetic_lens_reference()
    tokens = lens_data['tokens']
    # The reported machine-only drift is accepted by the new local-only gate.
    tokens[r.ORACLE]['0']['top20_positive'][0]['probability'] += 1.0371208190917969e-05
    if failure == 'token':
        tokens[r.ORACLE]['4']['top20_negative'][0]['token_id'] = 30
    elif failure == 'probability':
        tokens[r.ORACLE]['0']['top20_positive'][0]['probability'] += 1e-4
    elif failure == 'checkpoint':
        provenance['merged_model_hashes_sha256'] = 'b'*64
    def oracle_readout(spec, names):
        assert names == ()
        def lens(vectors, *args):
            assert list(vectors) == [r.ORACLE]
            return lens_data
        return SimpleNamespace(lens_readout=lens)
    monkeypatch.setattr(r, 'configured_readout', oracle_readout)
    args = (SPEC, a, object(), None, None, None, torch, None, historical, provenance)
    if failure:
        with pytest.raises(ValueError):
            r.validate_oracle_lens_compatibility(*args, fallback=True)
    else:
        report = r.validate_oracle_lens_compatibility(*args, fallback=True)
        assert report['max_observed_absolute_top20_probability_difference'] == pytest.approx(1.0371208190917969e-05)
        assert report['all_ordered_top20_exact'] and report['all_historical_top1_rank1']
        assert report['top20_probability_absolute_threshold'] == 2e-5
        assert provenance['oracle_adl_sha256'] == 'b'*64  # Actual local provenance is never rewritten.


def test_portability_spec_and_pre_score_gate_leave_scientific_definitions_unchanged():
    baseline = json.loads(__import__('subprocess').check_output(['git', 'show', 'HEAD:'+str(r.SPEC_PATH.relative_to(PROJECT))]))
    for key in baseline:
        if key == 'oracle_portability':
            assert {k: v for k, v in SPEC[key].items() if k not in (
                'compatibility', 'local_top20_probability_absolute_tolerance')} == {
                    k: v for k, v in baseline[key].items() if k not in (
                        'compatibility', 'local_top20_probability_absolute_tolerance')}
        else:
            assert SPEC[key] == baseline[key]
    assert SPEC['oracle_portability']['local_top20_probability_absolute_tolerance'] == r.LOCAL_ORACLE_TOP20_ABS_TOL == 2e-5
    assert SPEC['oracle_portability']['generator']['sha256'] == r.GENERATOR_SHA256
    assert r.sha256_file(r.path_of(SPEC['oracle_portability']['generator']['path'])) == r.GENERATOR_SHA256
    source = SOURCE.read_text()
    run = ast.get_source_segment(source, next(node for node in ast.parse(source).body
                                             if isinstance(node, ast.FunctionDef) and node.name == 'run'))
    assert run.index('validate_oracle_lens_compatibility(') < run.index('lens = a.lens_readout(')
    assert run.index('validate_oracle_lens_compatibility(') < run.index('patchscope_mode(')
    assert run.rindex('validate_oracle_reference(') < run.index("output.open('x'")
    assert 'compute_oracle_adl.py' not in run  # Generation stays behind the validated selection helper.


@pytest.mark.parametrize('delta', [0., 1.0371208190917969e-05, np.nextafter(2e-5, 0.), 2e-5])
def test_local_top20_probability_absolute_bound_is_inclusive(delta):
    a = r.configured_readout(SPEC, r.NAMES)
    historical, provenance, lens = synthetic_lens_reference()
    # A zero reference value makes the tested difference exactly delta, avoiding
    # cancellation roundoff and proving this is an absolute, non-relative bound.
    historical['positions'][4]['difference']['negative'][19]['probability'] = 0.
    lens['tokens'][r.ORACLE]['4']['top20_negative'][19]['probability'] = float(delta)
    report = r.validate_historical_lens_reference(SPEC, a, lens, historical, provenance, fallback=True)
    assert report['max_observed_absolute_top20_probability_difference'] == delta
    assert report['all_ordered_top20_exact'] is True
    assert report['all_historical_top1_rank1'] is True
    assert report['top20_probability_absolute_threshold'] == 2e-5
    assert report['oracle_logit_lens_compatibility_rule'] == SPEC['oracle_portability']['compatibility']


@pytest.mark.parametrize('pos', range(5))
@pytest.mark.parametrize('polarity', ['positive', 'negative'])
@pytest.mark.parametrize('failure', ['reorder', 'top1', 'probability'])
def test_local_gate_checks_every_position_and_polarity(pos, polarity, failure):
    a = r.configured_readout(SPEC, r.NAMES)
    historical, provenance, lens = synthetic_lens_reference()
    rows = lens['tokens'][r.ORACLE][str(pos)]['top20_'+polarity]
    if failure == 'reorder':
        rows[5], rows[6] = rows[6], rows[5]  # Same ID set and same top-1; wrong order.
        error = 'ordered top20 token-ID mismatch'
    elif failure == 'top1':
        rows[0], rows[1] = rows[1], rows[0]
        error = 'historical top1 is not rank 1'
    else:
        historical['positions'][pos]['difference'][polarity][19]['probability'] = 0.
        rows[19]['probability'] = float(np.nextafter(2e-5, np.inf))
        error = 'probability exceeds absolute bound'
    with pytest.raises(ValueError, match=error):
        r.validate_historical_lens_reference(SPEC, a, lens, historical, provenance, fallback=True)


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -1e-5, 1.000001])
def test_local_probability_gate_rejects_nonfinite_and_invalid_probabilities(invalid):
    a = r.configured_readout(SPEC, r.NAMES)
    historical, provenance, lens = synthetic_lens_reference()
    lens['tokens'][r.ORACLE]['1']['top20_positive'][1]['probability'] = float(invalid)
    with pytest.raises(ValueError, match='absolute bound'):
        r.validate_historical_lens_reference(SPEC, a, lens, historical, provenance, fallback=True)


def test_exact_historical_artifact_keeps_original_attempt133_probability_gate():
    a = r.configured_readout(SPEC, r.NAMES)
    historical, provenance, lens = synthetic_lens_reference()
    report = r.validate_historical_lens_reference(SPEC, a, lens, historical, provenance, fallback=False)
    assert report['oracle_logit_lens_compatibility_rule'] == 'attempt133_rtol_1e-6_atol_1e-8_exact_historical_artifact'
    assert report['top20_probability_absolute_threshold'] is None
    lens['tokens'][r.ORACLE]['0']['top20_positive'][0]['probability'] += 1.0371208190917969e-05
    # Local-only relaxation must never leak into the exact historical path.
    with pytest.raises(ValueError, match='Historical oracle Logit Lens mismatch'):
        r.validate_historical_lens_reference(SPEC, a, lens, historical, provenance, fallback=False)
    r.validate_historical_lens_reference(SPEC, a, lens, historical, provenance, fallback=True)


def test_scoring_cache_and_hash_validation_functions_are_unchanged():
    baseline = __import__('subprocess').check_output(['git', 'show', 'HEAD:'+str(SOURCE.relative_to(PROJECT))]).decode()
    current = SOURCE.read_text()
    old_tree, new_tree = ast.parse(baseline), ast.parse(current)
    # Only the compatibility gate/provenance plumbing may change. Cached-oracle
    # selection, generation and scientific scoring functions stay byte-identical.
    for name in ('configured_readout', 'patch_metrics_with_norms', 'patchscope_mode', 'distribution',
                 'seed_summaries', 'validate_candidate_hashes', 'generate_local_oracle',
                 'validate_oracle_tensors', 'validate_oracle_reference', 'load_oracle_reference'):
        old = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        new = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.get_source_segment(baseline, old) == ast.get_source_segment(current, new)
