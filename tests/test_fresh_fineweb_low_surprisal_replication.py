"""Synthetic and static controls for Attempt129's fresh FineWeb replication."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
FREEZER = PROJECT / 'scripts/ablation/freeze_attempt129_fresh_fineweb_corpus.py'
SOURCE = PROJECT / 'scripts/ablation/run_fresh_fineweb_low_surprisal_replication.py'


def import_file(path, name):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


f = import_file(FREEZER, 'attempt129_freezer')
a = import_file(SOURCE, 'attempt129_run')
old = f.import_old_selector()
a126 = a.import_pinned(a.path_of(a.load_spec()['frozen_attempt126']['source_path']),
                        a.load_spec()['frozen_attempt126']['source_sha256'],
                        'attempt129_test_frozen_attempt126')


class Tensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.int64)
        self.device = SimpleNamespace(type='cpu')

    @property
    def dtype(self):
        return self.array.dtype.type

    @property
    def shape(self):
        return self.array.shape

    def is_contiguous(self):
        return self.array.flags.c_contiguous

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.array))


class TinyTorch:
    Tensor = Tensor
    int64 = np.int64

    @staticmethod
    def tensor(values, dtype, device):
        assert dtype == np.int64 and device == 'cpu'
        return Tensor(values)


class Dataset:
    def shuffle(self, seed):
        assert seed == 42
        return self

    def __iter__(self):
        yield {'text': '   '}
        yield {'text': 'short'}
        for index in range(28192):
            yield {'text': str(index)}


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is True
        assert len(text) <= 1280
        return [0] * 20 if text == 'short' else [int(text)] + [0] * 127


def report(primary, secondary):
    tail = (127 * secondary - 4 * primary) / 123
    values = [-.2] + [primary] * 4 + [tail] * 123
    return {'position_0_cosine': values[0],
            'positions_1_4_individual_cosines': values[1:5],
            'positions_1_4_mean_cosine': primary,
            'positions_1_127_mean_cosine': secondary,
            'all_128_position_cosines': values}


def reports(control, low, middle, high):
    return {name: report(*pair) for name, pair in
            zip(a.NAMES, (control, low, middle, high))}


def test_exact_attempt005_selection_and_disjoint_valid_rank_intervals():
    corpus = f.load_spec()
    previous = json.loads(f.path_of(corpus['frozen_attempt005']['corpus_spec_path']).read_text())
    old_selection, new_selection = previous['selection'], corpus['selection']
    assert old_selection['skip_valid_examples'] == 20000
    assert corpus['old_valid_example_rank_interval'] == [20000, 24096]
    assert new_selection == {**old_selection, 'skip_valid_examples': 24096}
    assert corpus['fresh_valid_example_rank_interval'] == [24096, 28192]
    assert corpus['old_valid_example_rank_interval'][1] == (
        corpus['fresh_valid_example_rank_interval'][0])
    assert corpus['dataset'] == previous['dataset'] == {
        'repo_id': 'science-of-finetuning/fineweb-1m-sample',
        'revision': '60b53a86b84eb6559e4407b113356f56a152318f',
        'split': 'train', 'streaming': False}
    assert corpus['tokenizer'] == previous['tokenizer']
    assert corpus['tokenizer_checkpoint_files'] == json.loads(
        f.path_of(corpus['frozen_attempt005']['corpus_manifest_path']).read_text()
        )['tokenizer_checkpoint']['files']
    assert str(f.ARTIFACT_PATH) == (
        '/root/model-diff-scratch/artifacts/attempt129_fresh_fineweb_corpus/tokens.pt')
    assert f.ARTIFACT_PATH != Path(previous['artifact']['path'])

    # The pinned Attempt005 selector counts only valid, nonblank examples.
    selected = old.select_tokens(Dataset(), Tokenizer(), TinyTorch, corpus)
    assert selected.shape == (4096, 128)
    assert selected.dtype == np.int64 and selected.is_contiguous()
    assert selected.array[0, 0] == 24096
    assert selected.array[-1, 0] == 28191


def test_freezer_pins_old_selector_and_refuses_overwrite(tmp_path):
    assert f.OLD_SOURCE_SHA256 == f.sha256_file(f.OLD_SOURCE)
    corpus = f.load_spec()
    assert corpus['manifest'] == {'filename': 'corpus-manifest.json',
        'timestamps': False, 'host_metadata': False, 'gpu_metadata': False}
    source = FREEZER.read_text()
    assert 'old.select_tokens(dataset, tokenizer, torch_module, spec)' in source
    assert 'revision=spec[\'dataset\'][\'revision\']' in source
    assert "streaming=False" in source
    assert "tokenizer_loader = lambda path: AutoTokenizer.from_pretrained(path, local_files_only=True)" in source
    assert 'old.sha256_raw_int64_tensor(tensor, torch_module)' in source
    assert "ARTIFACT_PATH.open('xb')" in source
    artifact, manifest = tmp_path / 'tokens.pt', tmp_path / 'corpus-manifest.json'
    f.refuse_outputs(artifact, manifest)
    artifact.write_bytes(b'x')
    with pytest.raises(ValueError, match='already exists'):
        f.refuse_outputs(artifact, manifest)
    artifact.unlink()
    manifest.symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError, match='already exists'):
        f.refuse_outputs(artifact, manifest)


def test_fresh_plan_exact_four_sets_and_deterministic_ranking():
    spec = a.load_spec()
    assert list(spec['candidate_rows']) == list(a.GROUPS) == ['control', 'low', 'middle', 'high']
    assert spec['candidate_rows']['control'] == {
        'selection': 'original_frozen_corpus_order', 'row_range': [0, 1024]}
    assert [spec['candidate_rows'][name]['rank_range'] for name in a.GROUPS[1:]] == [
        [0, 1024], [1536, 2560], [3072, 4096]]
    assert a.NAMES == ('response_control', 'response_low',
                       'response_middle', 'response_high')
    assert spec['candidate']['tensor_order'] == list(a.NAMES)
    assert spec['workload'] == {'fresh_corpus_freeze_gpu_batches': 0,
        'ranking_forward_batches': 512, 'gradient_backward_batches': 512,
        'jvp_probe_batches': 128, 'matched_target_forward_batches': 64,
        'ggn_operations': 0, 'cg_operations': 0}
    values = np.arange(4096, dtype=np.float64)
    values[:2] = 200.0  # tie resolved by original row index
    ranking = a126.rank_and_stratify(values)
    assert ranking == a126.rank_and_stratify(values)
    assert ranking['ascending_rank_original_row_indices'].index(0) < (
        ranking['ascending_rank_original_row_indices'].index(1))
    rows, hashes, overlaps = a.fixed_candidate_rows(ranking, a126)
    assert rows['control'] == list(range(1024))
    assert all(len(rows[name]) == 1024 for name in a.GROUPS)
    assert all(len(hashes[name]) == 64 for name in a.GROUPS)
    assert overlaps['low'] == len(set(rows['control']) & set(rows['low']))
    assert rows['low'] != rows['control']
    assert ranking['surprisal_raw_float64_sha256'] == a126.raw_float64_hash(values)


def test_gradient_jvp_support_and_oracle_free_response_geometry():
    spec = a.load_spec()
    source = SOURCE.read_text()
    assert spec['gradient_loss']['number_of_batches'] == 128
    assert spec['gradient_loss']['total_prediction_tokens'] == 1024 * 127
    assert spec['eligible_tensors']['expected_matrix_count'] == 98
    assert spec['tangent']['direction'] == 'delta = +alpha * g'
    assert spec['tangent']['target_relative_frobenius'] == .00125
    assert spec['tangent']['normalization'] == 'one_global_alpha_across_all_eligible_matrices'
    assert spec['probe']['selected_rows'] == [0, 1024]
    assert spec['probe']['selected_batch_count'] == 32
    assert 'a14.accumulate_mean_generic_gradient(' in source
    assert 'a14.global_tangent_scale(eligible, .00125, torch)' in source
    assert 'a14.prepare_tangents(eligible, scale, torch)' in source
    assert 'a14.compute_probe_response(' in source
    assert "'sample_count': 1024" in source
    assert 'a14.freeze_other_parameters(model, eligible)' in source
    assert 'a14.verify_model_unchanged(model, before, torch)' in source
    assert 'per_example_gradient' not in source
    assert 'normalize_examples' not in source
    assert 'optimizer.step' not in source
    assert "name = 'response_' + group" in source
    assert 'pairwise_flattened_response_cosine_oracle_free' in source


def test_replication_selection_thresholds_and_position_wins():
    clear = reports((.30, .30), (.50, .50), (.48, .48), (.46, .46))
    delta, counts, replication, selection = a.signed_score_differences(clear)
    assert replication == 'clear_replication'
    assert selection == 'meaningful_selection_gain'
    assert delta['low_minus_high']['primary'] == pytest.approx(.04)
    assert delta['low_minus_middle']['secondary'] == pytest.approx(.02)
    assert delta['middle_minus_high']['primary'] == pytest.approx(.02)
    assert counts['low_minus_high'] == 127
    assert counts['low_minus_control'] == 127
    partial = reports((.495, .495), (.50, .50), (.55, .49), (.46, .495))
    assert a.signed_score_differences(partial)[2:] == (
        'partial_replication', 'positive_selection_gain')
    no = reports((.50, .50), (.48, .48), (.47, .47), (.49, .45))
    assert a.signed_score_differences(no)[2:] == ('no_replication', 'no_selection_gain')
    with pytest.raises(ValueError):
        a.signed_score_differences({**clear, 'response_extra': report(1, 1)})
    bad = copy.deepcopy(clear)
    bad['response_low']['all_128_position_cosines'][3] = None
    with pytest.raises(ValueError):
        a.signed_score_differences(bad)


def test_old_context_after_barrier_and_exact_target_pin():
    spec = a.load_spec()
    old_result = {'attempt_id': '127_postfreeze_surprisal_strata_evaluation',
                  'signed_position_reports': {
                      'response_low': report(.60, .55),
                      'response_middle': report(.50, .49),
                      'response_high': report(.40, .43)},
                  'matched_target': {'raw_sha256': a.TARGET_SHA256},
                  'clean_heldout_validation': False}
    fresh = reports((.4, .4), (.52, .52), (.48, .48), (.44, .44))
    deltas = a.signed_score_differences(fresh)[0]
    context = a.old_vs_fresh_context(old_result, fresh, deltas)
    assert context['low']['old_primary'] == .60
    assert context['low']['fresh_primary'] == .52
    assert context['low_minus_high']['old_primary'] == pytest.approx(.20)
    assert context['low_minus_high']['fresh_primary'] == pytest.approx(.08)
    assert spec['evaluation']['target_raw_sha256'] == a.TARGET_SHA256
    assert spec['frozen_attempt123']['source_sha256'] == a.sha256_file(
        a.path_of(spec['frozen_attempt123']['source_path']))
    assert spec['frozen_attempt127']['result_sha256'] == a.sha256_file(
        a.path_of(spec['frozen_attempt127']['result_path']))


def test_published_candidate_audit_requires_exact_order_and_hashes(tmp_path):
    candidate = tmp_path / 'candidate.pt'
    manifest = tmp_path / 'construction-manifest.json'
    candidate.write_bytes(b'synthetic candidate bytes')
    ranking = a126.rank_and_stratify(np.arange(4096, dtype=np.float64))
    rows, hashes, overlaps = a.fixed_candidate_rows(ranking, a126)
    construction = {
        'candidate': {'serialized_sha256': a.sha256_file(candidate),
                      'tensor_order': list(a.NAMES)},
        'barrier': {'manifest_contains_historical_scores': False},
        'ranking': ranking, 'candidate_rows': rows,
        'selected_row_raw_sha256': hashes,
        'control_stratum_overlap_counts': overlaps,
        'gradient_records': {name: {'selected_row_hash': hashes[name]}
                             for name in a.GROUPS},
        'responses': {name: {'raw_sha256': name} for name in a.NAMES}}
    manifest.write_text(json.dumps(construction))
    artifact = {name: SimpleNamespace(digest=name) for name in a.NAMES}
    fake_torch = SimpleNamespace(load=lambda *_args, **_kwargs: artifact)
    fake_readout = SimpleNamespace(sha256_raw_float32_tensor=lambda value, _torch: value.digest)
    fake_prior = SimpleNamespace(validate_response=lambda value, _torch: value)
    fake_a126 = SimpleNamespace(validate_ranking=a126.validate_ranking,
        raw_int64_hash=a126.raw_int64_hash,
        a122=SimpleNamespace(prior=fake_prior))
    paths = {'candidate_path': candidate, 'construction_manifest_path': manifest}
    assert a.validate_published(paths, construction, fake_a126, fake_readout,
                                 fake_torch) is artifact
    fake_torch.load = lambda *_args, **_kwargs: dict(reversed(list(artifact.items())))
    with pytest.raises(ValueError, match='inventory/order'):
        a.validate_published(paths, construction, fake_a126, fake_readout, fake_torch)
    candidate.write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact or manifest changed'):
        a.validate_published(paths, construction, fake_a126, fake_readout, fake_torch)


def test_freeze_barrier_and_overwrite_refusal(tmp_path):
    source = SOURCE.read_text()
    funcs = {node.name: ast.get_source_segment(source, node) for node in
             ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    run = funcs['run']
    assert run.index('a126.score_endpoint_surprisal(') < run.index('fixed_candidate_rows(')
    assert run.index('fixed_candidate_rows(') < run.index('a14.accumulate_mean_generic_gradient(')
    assert run.index('a14.compute_probe_response(') < run.index('atomic_torch_publish(')
    assert run.index('atomic_torch_publish(') < run.index('atomic_json_publish(paths[\'construction_manifest_path\']')
    assert run.index('validate_published(paths, construction') < run.index(
        "print('FRESH_SURPRISAL_CANDIDATES_FROZEN'") < run.index('evaluate_after_barrier(')
    assert 'frozen_attempt127' not in run[:run.index("print('FRESH_SURPRISAL_CANDIDATES_FROZEN'")]
    privileged = funcs['evaluate_after_barrier']
    assert privileged.index('a123.verified_target(') < privileged.index('a126.a122.score_report(')
    assert privileged.index('a126.a122.score_report(') < privileged.index("prior = spec['frozen_attempt127']")
    assert 'privileged.matched_difference' in privileged
    assert 'response_combination' not in source
    spec = a.load_spec()
    for key in spec['paths']:
        spec['paths'][key] = str(tmp_path / Path(spec['paths'][key]).name)
    paths = a.refuse_outputs(spec)
    paths['candidate_path'].write_bytes(b'existing')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_outputs(spec)
    paths['candidate_path'].unlink()
    paths['result_path'].write_text('{}')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_outputs(spec)
