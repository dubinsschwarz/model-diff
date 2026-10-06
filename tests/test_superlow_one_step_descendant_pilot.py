"""Synthetic and committed-metadata-only checks; never load real models or data."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'scripts/ablation/run_superlow_one_step_descendant_pilot.py'
EVALUATOR = SOURCE.with_name('evaluate_superlow_one_step_descendant_pilot.py')


def import_file(path, name):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


c = import_file(SOURCE, 'test_attempt200_constructor')
e = import_file(EVALUATOR, 'test_attempt200_evaluator')


def test_blind_metadata_reads_only_exact_committed_allowlist(monkeypatch):
    spec = c.load_spec()
    allowed = {c.path_of(record['path']) for record in
               [spec[k] for k in ('merged_inventory', 'probe_manifest', 'probe_freezer', 'ranking_manifest')]
               + list(spec['helpers'].values())
               + [s[k] for s in spec['sources'] for k in ('corpus_spec', 'corpus_manifest')]}
    opened = []
    real_open = Path.open

    def guarded(path, *args, **kwargs):
        assert path in allowed, f'Non-blind read attempted: {path}'
        opened.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    hashes = c.blind_input_hashes(spec)
    inventory, pairs = c.blind_metadata(spec)
    assert set(hashes) == {str(path.relative_to(PROJECT)) for path in allowed}
    assert pairs.shape == (1024, 2)
    assert c.raw_hash(pairs) == spec['ordered_pair_raw_int64_sha256']
    assert torch.bincount(pairs[:, 0], minlength=3).tolist() == [254, 252, 518]
    assert inventory['files'] and opened


def test_selection_is_manifest_order_hash_checked_not_results():
    spec = c.load_spec()
    manifest = c.read_record(spec['ranking_manifest'])
    pairs = c.selection_pairs(manifest, spec)
    assert pairs.tolist() == manifest['combined_ranking']['groups']['super_low']['ordered_pairs']
    damaged = copy.deepcopy(manifest)
    damaged['combined_ranking']['groups']['super_low']['ordered_pairs'].reverse()
    with pytest.raises(ValueError, match='hash mismatch'):
        c.selection_pairs(damaged, spec)
    damaged = copy.deepcopy(manifest)
    damaged['combined_ranking']['groups']['super_low']['ordered_pair_raw_int64_sha256'] = '0'*64
    with pytest.raises(ValueError, match='hash mismatch'):
        c.selection_pairs(damaged, spec)


def test_source_rank_rehydration_and_pair_mapping():
    class Dataset:
        def shuffle(self, seed):
            assert seed == 42
            return [{'text': ''}, {'text': '  '}, {'text': 'short'}] + [
                {'text': str(i) + 'x'*1500} for i in range(9)]

    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert len(text) <= 1280 and add_special_tokens is True
            return [0]*127 if text == 'short' else [int(text.split('x')[0])]*129

    sources, probe = c.rehydrate(Dataset(), Tokenizer(), intervals=((3, 5), (5, 7), (7, 9)), probe_count=3)
    assert probe[:, 0].tolist() == [0, 1, 2]
    assert [s[:, 0].tolist() for s in sources] == [[3, 4], [5, 6], [7, 8]]
    assert all(s.shape == (2, 128) and s.dtype == torch.int64 for s in sources)
    pairs = torch.tensor([[2, 1], [0, 0], [1, 1]], dtype=torch.int64)
    assert c.select_rows(sources, pairs)[:, 0].tolist() == [8, 3, 6]
    with pytest.raises(ValueError, match='exhausted'):
        c.rehydrate(Dataset(), Tokenizer(), intervals=((3, 10),), probe_count=3)


def eligible_fixture():
    module = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([[1., 2.], [3., 4.]]))
    module.weight.grad = torch.ones_like(module.weight)
    return [('model.layers.0.linear', module)]


def test_armijo_restores_each_rejected_trial_and_accepts_exactly_once():
    eligible = eligible_fixture()
    module = eligible[0][1]
    originals, scale = c.initial_scale(eligible, 10.)
    initial = module.weight.detach().clone()
    seen = []

    def objective():
        index = len(seen)
        eta = scale['eta_max'] * .5**index
        expected = (initial.double() - eta*module.weight.grad.double()).float()
        assert torch.equal(module.weight, expected)  # detects accumulating rejected updates
        seen.append(module.weight.detach().clone())
        return 11. if index < 2 else 9.

    step = c.armijo_step(eligible, originals, scale, objective)
    assert len(seen) == 3 and [t['accepted'] for t in step['trials']] == [False, False, True]
    assert step['accepted_updates'] == 1 and step['accepted_eta'] == scale['eta_max']*.25
    assert torch.equal(module.weight, seen[-1])
    assert step['displacement_norm'] == c.norm([module.weight.detach().double()-initial.double()])
    c.validate_diagnostics(step)
    bad = copy.deepcopy(step)
    bad['accepted_updates'] = 2
    with pytest.raises(ValueError):
        c.validate_diagnostics(bad)
    bad = copy.deepcopy(step)
    bad['trials'][0]['accepted'] = True
    with pytest.raises(ValueError):
        c.validate_diagnostics(bad)


@pytest.mark.parametrize('failure', ['reject', 'exception', 'nonfinite'])
def test_failed_armijo_leaves_exact_initial_fp32_state(failure):
    eligible = eligible_fixture()
    originals, scale = c.initial_scale(eligible, 10.)
    calls = []

    def objective():
        calls.append(1)
        if failure == 'exception':
            raise RuntimeError('synthetic failure')
        return float('nan') if failure == 'nonfinite' else 11.

    with pytest.raises((ValueError, RuntimeError)):
        c.armijo_step(eligible, originals, scale, objective)
    assert torch.equal(eligible[0][1].weight, originals[eligible[0][0]])
    assert len(calls) == (8 if failure == 'reject' else 1)


def test_eligible_boundary_exact_98_linear_weights_only():
    spec = c.load_spec()
    assert spec['eligible_tensors'] == {
        'blocks': list(range(14)), 'module': 'torch.nn.Linear', 'parameter': 'weight',
        'expected_matrix_count': 98, 'freeze_every_other_parameter': True}
    helper, _ = c.load_helpers(spec)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([
                torch.nn.Sequential(*[torch.nn.Linear(2, 2) for _ in range(7)]) for _ in range(28)])
            self.embedding = torch.nn.Embedding(2, 2)

    model = Model()
    eligible = helper.discover_eligible_linear_weights(model, torch)
    assert len(eligible) == 98
    assert {int(name.split('.')[2]) for name, _ in eligible} == set(spec['eligible_tensors']['blocks'])
    ids = helper.freeze_other_parameters(model, eligible)
    assert all(p.requires_grad == (id(p) in ids) for p in model.parameters())
    assert not any(m.bias.requires_grad for layer in model.model.layers for m in layer)
    assert not any(p.requires_grad for p in model.model.layers[14:].parameters())


def test_full_gradient_is_mean_of_all_130048_prediction_tokens(monkeypatch):
    helper, _ = c.load_helpers(c.load_spec())

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([.2, -.3]))
        def forward(self, input_ids, use_cache):
            assert use_cache is False and not self.training
            return SimpleNamespace(logits=self.w.expand(*input_ids.shape, 2))

    model = TinyModel()
    tokens = torch.zeros((1024, 128), dtype=torch.int64)
    tokens[512:] = 1
    baseline = helper.causal_token_loss_sum(model.w.expand(1024, 128, 2), tokens, torch)/130048
    baseline.backward()
    expected = model.w.grad.clone()
    monkeypatch.setattr(c, 'log', lambda _: None)
    loss = c.full_loss(model, tokens, helper, backward=True)
    assert loss == pytest.approx(float(baseline.detach()), abs=1e-6)
    assert torch.allclose(model.w.grad, expected, atol=1e-6, rtol=1e-5)
    gradient = model.w.grad.clone()
    assert c.full_loss(model, tokens, helper) == pytest.approx(loss)
    assert torch.equal(model.w.grad, gradient)


def test_endpoint_sign_and_float64_subtraction_before_cast():
    f = torch.full((128, 2048), 1e8, dtype=torch.float64)
    g = f - 1.
    response = c.endpoint_response(f, g)
    assert response.dtype == torch.float32 and response.is_contiguous()
    assert torch.equal(response, torch.ones_like(response))
    assert torch.equal(c.endpoint_response(g, f), -response)
    assert torch.equal(f.float()-g.float(), torch.zeros_like(response))


def test_signed_metric_positions_no_abs_or_sign_choice():
    target = torch.zeros((128, 2048), dtype=torch.float32)
    target[:, 0] = 1.
    candidate = target.clone()
    candidate[0] *= -1
    candidate[1:5] *= -1
    report = e.signed_report(candidate, target)
    assert report['position_0_cosine'] == -1.
    assert report['positions_1_4_mean_cosine'] == -1.
    assert report['positions_1_127_mean_cosine'] == pytest.approx(119/127)
    candidate[2] = 0
    report = e.signed_report(candidate, target)
    assert report['positions_1_4_mean_cosine'] is None
    assert report['positions_1_127_mean_cosine'] is None


def test_evaluator_freeze_gate_blocks_every_privileged_read(monkeypatch, tmp_path):
    spec = copy.deepcopy(c.load_spec())
    spec['paths']['result'] = str(tmp_path/'result.json')
    monkeypatch.setattr(e.constructor, 'load_spec', lambda: spec)
    accesses = []
    monkeypatch.setattr(e.constructor, 'read_record', lambda record: accesses.append(record))

    def invalid_freeze(_):
        raise ValueError('synthetic invalid freeze')

    monkeypatch.setattr(e.constructor, 'validate_frozen', invalid_freeze)
    with pytest.raises(ValueError, match='invalid freeze'):
        e.evaluate('cpu')
    assert accesses == [] and not (tmp_path/'result.json').exists()


def test_structural_single_update_freeze_and_privileged_separation():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    code = ast.get_source_segment(source, run)
    assert code.count('armijo_step(') == 1
    assert code.count('backward=True') == 1
    assert code.index("mean_activation(model, probe, spec, readout_helper, 'F')") < code.index('backward=True')
    assert code.index('publish(manifest_path, manifest)') < code.index('validate_frozen(spec)') < code.index('log(MARKER)')
    assert 'evaluate(' not in source and 'result.json' not in source and 'models/base' not in source
    evaluator = EVALUATOR.read_text()
    assert evaluator.index('tensors, manifest, receipt = constructor.validate_frozen(spec)') < evaluator.index("base_spec = constructor.read_record")
    assert evaluator.index('tensors, manifest, receipt = constructor.validate_frozen(spec)') < evaluator.index('historical_result = json.loads')
    spec = c.load_spec()
    assert spec['training']['accepted_updates'] == spec['candidate']['count'] == 1
    assert spec['metric']['primary_positions'] == [1, 2, 3, 4]
    assert spec['metric']['secondary_positions'] == list(range(1, 128))


def test_publication_refuses_overwrite_and_checkpoint_extra_files(tmp_path):
    output = tmp_path/'candidate.pt'
    c.publish(output, torch.ones((2, 2)), tensor=True)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        c.publish(output, torch.zeros((2, 2)), tensor=True)
    assert output.read_bytes() == before
    directory = tmp_path/'model'
    directory.mkdir()
    (directory/'weights').write_bytes(b'fake')
    records = [{'path': 'weights', 'size_bytes': 4, 'sha256': c.sha256_file(directory/'weights')}]
    c.validate_checkpoint(directory, records)
    (directory/'extra').write_bytes(b'other')
    with pytest.raises(ValueError, match='inventory'):
        c.validate_checkpoint(directory, records)


@pytest.fixture
def synthetic_freeze(tmp_path, monkeypatch):
    spec = copy.deepcopy(c.load_spec())
    spec['paths']['artifact_dir'] = str(tmp_path/'artifacts')
    spec['paths']['construction_manifest'] = str(tmp_path/'construction-manifest.json')
    pairs = torch.tensor([[i % 3, i//3] for i in range(1024)], dtype=torch.int64)
    sources = [torch.zeros((4096, 128), dtype=torch.int64),
               torch.ones((4096, 128), dtype=torch.int64),
               torch.full((8192, 128), 2, dtype=torch.int64)]
    probe = torch.zeros((10000, 128), dtype=torch.int64)
    spec['probe']['raw_tensor_sha256'] = c.raw_hash(probe)
    spec['ordered_pair_raw_int64_sha256'] = c.raw_hash(pairs)
    for source, tensor in zip(spec['sources'], sources):
        source['raw_tensor_sha256'] = c.raw_hash(tensor)
    inventory = {'files': []}
    monkeypatch.setattr(c, 'load_spec', lambda: spec)
    monkeypatch.setattr(c, 'blind_metadata', lambda _: (inventory, pairs))
    monkeypatch.setattr(c, 'blind_input_hashes', lambda _: {'synthetic': 'a'*64})
    monkeypatch.setattr(c, 'validate_checkpoint', lambda *args: None)
    tensors = {'candidate': torch.ones((128, 2048), dtype=torch.float32), 'probe': probe,
        'selected': c.select_rows(sources, pairs), 'pairs': pairs,
        **{f'source{i}': tensor for i, tensor in enumerate(sources)}}
    artifacts = {}
    artifact_dir = Path(spec['paths']['artifact_dir'])
    for name, tensor in tensors.items():
        path = artifact_dir/(name+'.pt')
        c.publish(path, tensor, tensor=True)
        artifacts[name] = {'serialized_sha256': c.sha256_file(path), 'raw_tensor_sha256': c.raw_hash(tensor)}
    eligible = eligible_fixture()
    originals, scale = c.initial_scale(eligible, 10.)
    step = c.armijo_step(eligible, originals, scale, lambda: 9.)
    state = c.state_hashes([(f'model.layers.0.linear{i:02}', eligible[0][1]) for i in range(98)])
    manifest = {'attempt_id': c.ATTEMPT, 'spec_sha256': c.SPEC_SHA256,
        'constructor_sha256': c.sha256_file(SOURCE), 'blind_input_hashes': {'synthetic': 'a'*64},
        'barrier': c.MARKER, 'candidate_definition': spec['candidate']['definition'],
        'final_checkpoint_files': [], 'ordered_pairs': pairs.tolist(), 'step': step,
        'eligible_state': {'before': state, 'after': state}, 'artifacts': artifacts}
    manifest_path = Path(spec['paths']['construction_manifest'])
    c.publish(manifest_path, manifest)
    receipt_path = artifact_dir/'freeze-receipt.json'
    receipt = {'construction_manifest_sha256': c.sha256_file(manifest_path),
        'spec_sha256': c.SPEC_SHA256, 'constructor_sha256': manifest['constructor_sha256']}
    c.publish(receipt_path, receipt)
    return spec, tensors, manifest, receipt, manifest_path, receipt_path


def test_complete_freeze_audit_validates_all_synthetic_artifacts(synthetic_freeze):
    spec, original, manifest, receipt, _, _ = synthetic_freeze
    tensors, checked_manifest, checked_receipt = c.validate_frozen(spec)
    assert checked_manifest == manifest and checked_receipt == receipt
    assert all(torch.equal(tensors[name], tensor) for name, tensor in original.items())


@pytest.mark.parametrize('artifact', ['candidate', 'probe', 'source0', 'source1', 'source2', 'selected', 'pairs'])
def test_freeze_gate_rejects_any_tampered_tensor(synthetic_freeze, artifact):
    spec, tensors, _, _, _, _ = synthetic_freeze
    changed = tensors[artifact].clone()
    changed.view(-1)[0] += 1
    torch.save(changed, Path(spec['paths']['artifact_dir'])/(artifact+'.pt'))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        c.validate_frozen(spec)


@pytest.mark.parametrize('field', ['constructor', 'input', 'pairs', 'updates', 'aggregate'])
def test_freeze_gate_rejects_bad_manifest_even_with_updated_receipt(synthetic_freeze, field):
    spec, _, manifest, receipt, manifest_path, receipt_path = synthetic_freeze
    if field == 'constructor':
        manifest['constructor_sha256'] = '0'*64
    elif field == 'input':
        manifest['blind_input_hashes']['synthetic'] = '0'*64
    elif field == 'pairs':
        manifest['ordered_pairs'].reverse()
    elif field == 'updates':
        manifest['step']['accepted_updates'] = 2
    else:
        manifest['eligible_state']['after']['aggregate_sha256'] = '0'*64
    manifest_path.write_text(json.dumps(manifest))
    receipt['construction_manifest_sha256'] = c.sha256_file(manifest_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        c.validate_frozen(spec)


def test_linear_comparison_is_descriptive_signed_metric_difference():
    target = torch.zeros((128, 2048), dtype=torch.float32)
    target[:, 0] = 1
    endpoint = e.signed_report(-target, target)
    reference = e.signed_report(target, target)
    historical = {'attempt_id': '132_expanded_fineweb_superlow_stringency',
                  'signed_position_reports': {'response_super_low': reference}}
    comparison = e.reference_comparison(endpoint, historical)
    assert comparison['endpoint_minus_linear_Jg'] == {'primary': -2., 'secondary': -2.}
    assert torch.equal(target[:, 0], torch.ones(128))


def test_pinned_evaluation_hashes_and_constructor_spec_hash():
    spec = c.load_spec()
    assert spec['evaluation']['linear_reference_sha256'] == (
        '5883ce5a1b2a203ca5231bf61f14c3abfc9176c48a54fecbe23807a6ad0426da')
    assert spec['evaluation']['target_raw_tensor_sha256'] == (
        '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f')
    assert c.sha256_file(c.SPEC_PATH) == c.SPEC_SHA256


@pytest.mark.parametrize('failure', ['target_hash', 'reference_hash', None])
def test_evaluation_hash_gates_before_scoring_and_reference_read(monkeypatch, tmp_path, failure):
    """Mock every model/data operation; exercise the actual evaluator gates."""
    spec = copy.deepcopy(c.load_spec())
    result_path = tmp_path/'result.json'
    reference_path = tmp_path/'reference.json'
    reference_path.write_text(json.dumps({
        'construction_manifest_sha256': spec['ranking_manifest']['sha256']}))
    final_mean = torch.ones((128, 2048), dtype=torch.float64)
    base_mean = torch.zeros_like(final_mean)
    target = c.endpoint_response(final_mean, base_mean)
    spec['paths']['result'] = str(result_path)
    spec['evaluation']['linear_reference_path'] = str(reference_path)
    spec['evaluation']['linear_reference_sha256'] = (
        '0'*64 if failure == 'reference_hash' else c.sha256_file(reference_path))
    spec['evaluation']['target_raw_tensor_sha256'] = (
        '0'*64 if failure == 'target_hash' else c.raw_hash(target))
    manifest = {'activation_mean_float64_sha256': {'F': c.raw_hash(final_mean)},
        'step': {}, 'constructor_sha256': c.sha256_file(SOURCE), 'artifacts': {'candidate': {}}}
    events = []

    def freeze(_):
        events.append('freeze')
        return {'candidate': target, 'probe': object()}, manifest, {'construction_manifest_sha256': 'a'*64}

    def base_spec(record):
        assert record == spec['evaluation']['base_spec']
        events.append('base_spec')
        return {'paths': {'base_directory': str(tmp_path/'base')}, 'base': {'files': []}}

    def model_loader(*args):
        events.append('synthetic_model')
        return SimpleNamespace(config=SimpleNamespace(), to=lambda _: None), None

    def score(candidate, matched):
        events.append('score')
        assert torch.equal(matched, target)
        return {}

    original_require_hash = e.constructor.require_hash
    original_read_text = Path.read_text

    def verify_reference(path, expected):
        assert path == reference_path
        assert expected == spec['evaluation']['linear_reference_sha256']
        events.append('reference_hash')
        original_require_hash(path, expected)

    def read_text(path, *args, **kwargs):
        assert path == reference_path and events[-1] == 'reference_hash'
        events.append('reference_read')
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(e.constructor, 'load_spec', lambda: spec)
    monkeypatch.setattr(e.constructor, 'validate_frozen', freeze)
    monkeypatch.setattr(e.constructor, 'read_record', base_spec)
    monkeypatch.setattr(e.constructor, 'validate_checkpoint', lambda *args: None)
    monkeypatch.setattr(e.constructor, 'load_helpers', lambda _: (
        SimpleNamespace(load_local_model=model_loader), object()))
    monkeypatch.setattr(e.constructor, 'mean_activation', lambda *args: (
        final_mean if args[-1] == 'F evaluation' else base_mean))
    monkeypatch.setattr(e.constructor, 'require_hash', verify_reference)
    monkeypatch.setattr(e.constructor, 'log', lambda _: None)
    if failure == 'target_hash':
        def unverified_helper():
            raise ValueError('Synthetic canonical helper SHA256 mismatch')
        monkeypatch.setattr(e, 'load_canonical_helper', unverified_helper)
    monkeypatch.setattr(e, 'signed_report', score)
    monkeypatch.setattr(e, 'reference_comparison', lambda *args: {})
    monkeypatch.setattr(Path, 'read_text', read_text)
    if failure:
        with pytest.raises(ValueError, match='SHA256 mismatch'):
            e.evaluate('cpu')
        assert not result_path.exists() and 'reference_read' not in events
        assert ('score' in events) == (failure == 'reference_hash')
        assert ('reference_hash' in events) == (failure == 'reference_hash')
    else:
        result = e.evaluate('cpu')
        assert result_path.exists()
        assert result['provenance']['target_raw_tensor_sha256'] == c.raw_hash(target)
        assert result['provenance']['reference_result_sha256'] == spec['evaluation']['linear_reference_sha256']
        assert events.index('reference_hash') < events.index('reference_read')
    assert events[0:2] == ['freeze', 'base_spec']


def test_target_exact_historical_hash_path_skips_canonical_helper(monkeypatch):
    target = torch.zeros((128, 2048), dtype=torch.float32)
    target[:, 0] = e.HISTORICAL_TARGET_RMS
    expected_hash = c.raw_hash(target)

    def forbidden_helper_import():
        pytest.fail('Exact historical-hash path must not import the canonical helper')

    monkeypatch.setattr(e, 'load_canonical_helper', forbidden_helper_import)
    provenance = e.validate_target(target, expected_hash, None, None, None, None, None, None, 'cpu')
    assert provenance['historical_expected_target_sha256'] == expected_hash
    assert provenance['local_target_sha256'] == expected_hash
    assert provenance['historical_target_sha256_exact_match'] is True
    assert provenance['portability_fallback_used'] is False
    assert all(provenance[f'attempt200_vs_canonical_{label}_bitwise_equal'] is None
               for label in ('F', 'B', 'target'))
    assert provenance['canonical_helper_source_sha256'] == (
        '73b3ca864916c547ccf20d5b07889ae6b45c01276da8606ee65dfa0c7de4a0d6')


@pytest.fixture
def synthetic_portability(monkeypatch):
    target = torch.zeros((128, 2048), dtype=torch.float32)
    target[:, 0] = 33.51968159241566
    final_mean = target.double()
    base_mean = torch.zeros_like(final_mean)
    probe = torch.arange(10000, dtype=torch.int64).unsqueeze(1).expand(10000, 128).contiguous()
    canonical_spec = object()
    final = SimpleNamespace(to=lambda _: None)
    base = SimpleNamespace(to=lambda _: None)
    calls = []
    canonical = {'F': final_mean.clone(), 'B': base_mean.clone(), 'target': target.clone()}

    def canonical_mean(model, prefix, spec, *, progress):
        assert model is final or model is base
        assert spec is canonical_spec and callable(progress)
        assert torch.equal(prefix, probe[:1024]) and prefix.shape == (1024, 128)
        label = 'F' if model is final else 'B'
        calls.append(label)
        return canonical[label]

    def canonical_difference(f, b):
        assert f is canonical['F'] and b is canonical['B']
        calls.append('target')
        return canonical['target']

    helper = SimpleNamespace(base_probe_mean=canonical_mean, matched_difference=canonical_difference,
                             progress_printer=lambda stage: lambda *args: None)
    monkeypatch.setattr(e, 'load_canonical_helper', lambda: helper)
    monkeypatch.setattr(e.constructor, 'log', lambda _: None)
    historical_hash = '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f'
    args = (target, historical_hash, final_mean, base_mean, final, base, probe, canonical_spec, 'cpu')
    return args, canonical, calls


def test_target_mismatch_accepts_only_valid_canonical_helper_and_rms(synthetic_portability):
    args, _, calls = synthetic_portability
    target_before = args[0].clone()
    provenance = e.validate_target(*args)
    assert calls == ['B', 'F', 'target']
    assert provenance['historical_expected_target_sha256'] == args[1]
    assert provenance['local_target_sha256'] == c.raw_hash(args[0])
    assert provenance['historical_target_sha256_exact_match'] is False
    assert provenance['portability_fallback_used'] is True
    assert all(provenance[f'attempt200_vs_canonical_{label}_bitwise_equal'] is True
               for label in ('F', 'B', 'target'))
    rms = (args[0].double().square().sum()/128).sqrt().item()
    assert provenance['local_target_pooled_rms_position_norm'] == rms
    assert provenance['historical_target_pooled_rms_position_norm'] == 33.51964294937312
    absolute = abs(rms - 33.51964294937312)
    assert provenance['target_pooled_rms_position_norm_absolute_difference'] == absolute
    assert provenance['target_pooled_rms_position_norm_relative_difference'] == absolute/33.51964294937312
    assert provenance['portability_tolerances'] == {'rel_tol': 1e-5, 'abs_tol': 1e-4}
    assert torch.equal(args[0], target_before)


@pytest.mark.parametrize('label', ['F', 'B', 'target'])
def test_portability_rejects_each_canonical_helper_mismatch(synthetic_portability, label):
    args, canonical, _ = synthetic_portability
    canonical[label][0, 1] += .01
    with pytest.raises(ValueError, match=f'helper {label} bitwise mismatch'):
        e.validate_target(*args)


def test_portability_requires_byte_identity_including_signed_zero(synthetic_portability):
    args, canonical, _ = synthetic_portability
    canonical['target'][0, 1] = -0.
    assert torch.equal(args[0], canonical['target'])
    with pytest.raises(ValueError, match='helper target bitwise mismatch'):
        e.validate_target(*args)


def test_portability_rejects_excessive_rms_difference(synthetic_portability):
    args, canonical, _ = synthetic_portability
    args[0][:, 0] = e.HISTORICAL_TARGET_RMS + 1.
    args[2].copy_(args[0].double())
    canonical['F'].copy_(args[2])
    canonical['target'].copy_(args[0])
    with pytest.raises(ValueError, match='RMS outside portability tolerances'):
        e.validate_target(*args)


def test_canonical_helper_import_rejects_unpinned_source(monkeypatch, tmp_path):
    source = tmp_path/'helper.py'
    source.write_text("raise AssertionError('Unpinned helper must never be imported')\n")
    monkeypatch.setattr(e, 'CANONICAL_HELPER_SOURCE', source)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        e.load_canonical_helper()
