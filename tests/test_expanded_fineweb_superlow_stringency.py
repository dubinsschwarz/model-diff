"""Synthetic/static Attempt132 checks; no corpus freeze or model execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
FREEZER = PROJECT / 'scripts/ablation/freeze_attempt132_expanded_fineweb_corpus.py'
SOURCE = PROJECT / 'scripts/ablation/run_expanded_fineweb_superlow_stringency.py'


def import_file(path, name):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


f = import_file(FREEZER, 'attempt132_freezer')
a = import_file(SOURCE, 'attempt132_run')
old_selector = f.import_old_selector()
SPEC = a.load_spec()
a130 = a.import_pinned(a.path_of(SPEC['frozen_attempt130']['source_path']),
    SPEC['frozen_attempt130']['source_sha256'], 'attempt132_test_a130')
a126 = a.import_pinned(a.path_of(SPEC['frozen_slices']['old']['constructor_path']),
    SPEC['frozen_slices']['old']['constructor_sha256'], 'attempt132_test_a126')
a131 = a.import_pinned(a.path_of(SPEC['frozen_attempt131']['source_path']),
    SPEC['frozen_attempt131']['source_sha256'], 'attempt132_test_a131')


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

    @staticmethod
    def stack(values):
        return Tensor(np.stack([value.array for value in values]))


class Dataset:
    def shuffle(self, seed):
        assert seed == 42
        return self

    def __iter__(self):
        yield {'text': '   '}
        yield {'text': 'short'}
        for index in range(36384):
            yield {'text': str(index)}


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is True
        assert len(text) <= 1280
        return [0] * 20 if text == 'short' else [int(text)] + [0] * 127


def score(primary, secondary):
    tail = (127 * secondary - 4 * primary) / 123
    values = [-.2] + [primary] * 4 + [tail] * 123
    return {'position_0_cosine': values[0],
            'positions_1_4_individual_cosines': values[1:5],
            'positions_1_4_mean_cosine': primary,
            'positions_1_127_mean_cosine': secondary,
            'all_128_position_cosines': values}


def frozen_rank_data():
    pooled = json.loads(a.path_of(SPEC['frozen_attempt130']['construction_manifest_path']).read_text())['pooled_ranking']
    prior = json.loads(a.path_of(SPEC['frozen_attempt131']['construction_manifest_path']).read_text())['groups']['very_low']['ordered_pairs']
    return pooled, prior


def test_frozen_source_hashes_and_committed_attempt131_result():
    assert a.sha256_file(a.SPEC_PATH) == a.SPEC_SHA256
    assert f.sha256_file(f.SPEC_PATH) == f.SPEC_SHA256
    assert f.sha256_file(f.OLD_SOURCE) == f.OLD_SOURCE_SHA256
    assert a.sha256_file(a.path_of(SPEC['new_corpus']['corpus_spec_path'])) == (
        SPEC['new_corpus']['corpus_spec_sha256'])
    assert a.sha256_file(a.path_of(SPEC['new_corpus']['freezer_source_path'])) == (
        SPEC['new_corpus']['freezer_source_sha256'])
    for number in (130, 131):
        pinned = SPEC[f'frozen_attempt{number}']
        for key in ('spec', 'source', 'construction_manifest', 'candidate'):
            suffix = 'serialized_sha256' if key == 'candidate' else 'sha256'
            assert a.sha256_file(a.path_of(pinned[key + '_path'])) == pinned[key + '_' + suffix]
    result = SPEC['frozen_attempt131']
    assert a.sha256_file(a.path_of(result['result_path'])) == result['result_sha256']
    assert result['result_sha256'] == (
        '3692fbe6e14c6656b828ff867d131245e7e41ef79256abc355e7d3ed718a0ca4')
    assert SPEC['frozen_attempt130']['construction_manifest_sha256'] == (
        '13163051d0fb564187ef9a5a7c004fb92c259f379bce6ad0905243973ebeceb2')
    assert SPEC['combined_selection']['old_combined_rank_pair_raw_int64_sha256'] == (
        '84bfc8c07f576b61ac3ccda0b56347a2d12a130eccbaef472c38a1eca9cdf8f5')


def test_exact_attempt005_selection_new_valid_rank_interval_and_shape():
    corpus = f.load_spec()
    old = json.loads(f.path_of(corpus['frozen_attempt005']['corpus_spec_path']).read_text())
    assert corpus['dataset'] == old['dataset'] == {
        'repo_id': 'science-of-finetuning/fineweb-1m-sample',
        'revision': '60b53a86b84eb6559e4407b113356f56a152318f',
        'split': 'train', 'streaming': False}
    assert corpus['selection'] == {**old['selection'],
        'skip_valid_examples': 28192, 'sample_count': 8192}
    assert corpus['old_pooled_valid_example_rank_interval'] == [20000, 28192]
    assert corpus['new_valid_example_rank_interval'] == [28192, 36384]
    assert corpus['old_pooled_valid_example_rank_interval'][1] == (
        corpus['new_valid_example_rank_interval'][0])
    assert corpus['tokenizer_checkpoint_files'] == json.loads(
        f.path_of(corpus['frozen_attempt005']['corpus_manifest_path']).read_text()
        )['tokenizer_checkpoint']['files']
    assert str(f.ARTIFACT_PATH) == (
        '/root/model-diff-scratch/artifacts/attempt132_expanded_fineweb_corpus/tokens.pt')
    tensor = old_selector.select_tokens(Dataset(), Tokenizer(), TinyTorch, corpus)
    assert tensor.shape == (8192, 128) and tensor.is_contiguous()
    assert tensor.dtype == np.int64
    assert tensor.array[0, 0] == 28192
    assert tensor.array[-1, 0] == 36383


def test_freezer_pinned_selector_manifest_and_overwrite_refusal(tmp_path):
    corpus = f.load_spec()
    assert corpus['manifest'] == {'filename': 'corpus-manifest.json',
        'timestamps': False, 'host_metadata': False, 'gpu_metadata': False}
    source = FREEZER.read_text()
    assert 'old.select_tokens(dataset, tokenizer, torch_module, spec)' in source
    assert "revision=spec['dataset']['revision']" in source
    assert 'streaming=False' in source
    assert 'local_files_only=True' in source
    assert 'old.sha256_raw_int64_tensor(tensor, torch_module)' in source
    assert "ARTIFACT_PATH.open('xb')" in source
    assert 'old.validate_tokens(tensor, torch_module, sample_count=8192, sequence_length=128)' in source
    artifact, manifest = tmp_path / 'tokens.pt', tmp_path / 'corpus-manifest.json'
    f.refuse_outputs(artifact, manifest)
    artifact.write_bytes(b'x')
    with pytest.raises(ValueError, match='already exists'):
        f.refuse_outputs(artifact, manifest)
    artifact.unlink()
    manifest.symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError, match='already exists'):
        f.refuse_outputs(artifact, manifest)
    assert not f.MANIFEST_PATH.exists()


def test_future_corpus_manifest_schema_validation_without_freezing(tmp_path, monkeypatch):
    corpus_spec = f.load_spec()
    frozen = copy.deepcopy(SPEC)
    frozen['new_corpus']['corpus_manifest_path'] = str(tmp_path / 'corpus-manifest.json')
    manifest = {'format_version': 1, 'attempt_id': a.ATTEMPT, 'hash_algorithm': 'sha256',
        'corpus_spec_sha256': frozen['new_corpus']['corpus_spec_sha256'],
        'freeze_script_sha256': frozen['new_corpus']['freezer_source_sha256'],
        'dataset': corpus_spec['dataset'], 'selection': corpus_spec['selection'],
        'tokenizer_checkpoint': {'source': corpus_spec['tokenizer']['source'],
            'directory': corpus_spec['tokenizer']['directory'], 'file_count': 4,
            'files': corpus_spec['tokenizer_checkpoint_files']},
        'tensor': {'shape': [8192, 128], 'dtype': 'torch.int64',
            'device': 'cpu', 'contiguous': True},
        'artifact': {'path': frozen['new_corpus']['tokens_path'],
            'serialized_sha256': 'a' * 64, 'raw_tensor_sha256': 'b' * 64},
        'old_pooled_valid_example_rank_interval': [20000, 28192],
        'new_valid_example_rank_interval': [28192, 36384],
        'frozen_attempt005': corpus_spec['frozen_attempt005']}
    manifest_path = Path(frozen['new_corpus']['corpus_manifest_path'])
    manifest_path.write_text(json.dumps(manifest))
    original_require_hash = a.require_hash

    def checked(path, expected):
        if str(path) == frozen['new_corpus']['tokens_path']:
            assert expected == 'a' * 64
            return path
        return original_require_hash(path, expected)

    class ShapeOnly:
        shape = (8192, 128)

        def is_contiguous(self):
            return True

    fake_a14 = SimpleNamespace(load_corpus_tokens=lambda *args: ShapeOnly())
    monkeypatch.setattr(a, 'require_hash', checked)
    tokens, checked_manifest, checksum = a.validate_new_corpus(
        frozen, f, fake_a14, SimpleNamespace())
    assert tokens.shape == (8192, 128)
    assert checked_manifest == manifest
    assert checksum == a.sha256_file(manifest_path)
    manifest['new_valid_example_rank_interval'] = [28193, 36385]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='provenance mismatch'):
        a.validate_new_corpus(frozen, f, fake_a14, SimpleNamespace())


def test_new_scoring_definition_and_deterministic_float64_ranking():
    scoring = SPEC['new_surprisal_scoring']
    assert scoring['population_rows'] == [0, 8192]
    assert scoring['number_of_forward_batches'] == 1024
    assert scoring['prediction_tokens_per_row'] == 127
    assert scoring['no_grad'] and scoring['no_amp']
    source = SOURCE.read_text()
    run = ast.get_source_segment(source, next(node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == 'run'))
    assert run.count('score_new_surprisal(') == 1
    assert 'score_endpoint_surprisal(' not in source
    scorer = ast.get_source_segment(source, next(node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == 'score_new_surprisal'))
    assert 'range(0, 8192, 8)' in scorer
    assert 'torch.inference_mode()' in scorer
    assert 'enabled=False' in scorer
    assert "reduction='none'" in scorer
    assert 'reshape(8, 127)' in scorer
    assert 'a126.sequence_mean_nll(cpu)' in scorer
    assert a126.sequence_mean_nll(np.ones((8, 127), dtype=np.float32)).tolist() == [1.] * 8
    values = np.arange(8192, dtype=np.float64)
    values[0] = values[1] = -1.0
    ranking = a.rank_new_surprisal(values, a126)
    assert ranking['ascending_rank_original_row_indices'][:2] == [0, 1]
    assert ranking['surprisal_raw_float64_sha256'] == a126.raw_float64_hash(values)
    assert len(ranking['ascending_rank_original_row_indices']) == 8192
    assert a.validate_new_ranking(ranking, a126)
    bad = copy.deepcopy(ranking)
    bad['ascending_rank_original_row_indices'][0] = 1
    with pytest.raises(ValueError):
        a.validate_new_ranking(bad, a126)
    with pytest.raises(ValueError):
        a.rank_new_surprisal(np.zeros(8191, dtype=np.float64), a126)


def test_combined_16384_ranking_group_size_ties_source_counts_and_hashes():
    pooled, prior = frozen_rank_data()
    values = np.full(8192, 10.0, dtype=np.float64)
    first_score = pooled['records'][0][0]
    values[0] = values[1] = first_score
    ranking = a.rank_new_surprisal(values, a126)
    combined = a.combine_rankings(pooled, ranking, prior,
                                  SPEC['combined_selection'], a130, a126)
    assert len(combined['records']) == 16384
    assert combined['ordered_pair_raw_int64_sha256'] == a130.pair_hash(
        np.ascontiguousarray([(r[1], r[2]) for r in combined['records']], dtype=np.int64))
    assert combined['records'] == sorted(combined['records'])
    assert [r[1] for r in combined['records']].count(2) == 8192
    tied = [r for r in combined['records'] if r[0] == first_score]
    assert tied.index([first_score, 2, 0]) < tied.index([first_score, 2, 1])
    assert all(r[1] in (0, 1) for r in tied[:tied.index([first_score, 2, 0])])
    groups = combined['groups']
    assert [groups[name]['rank_range'] for name in a.GROUPS] == [[0, 1024], [1024, 2048]]
    assert all(len(groups[name]['ordered_pairs']) == 1024 for name in a.GROUPS)
    assert set(map(tuple, groups[a.GROUPS[0]]['ordered_pairs'])).isdisjoint(
        set(map(tuple, groups[a.GROUPS[1]]['ordered_pairs'])))
    assert combined['groups_overlap_count'] == 0
    assert all(sum(groups[name]['source_counts'].values()) == 1024 for name in a.GROUPS)
    assert all(len(groups[name]['ordered_pair_raw_int64_sha256']) == 64 for name in a.GROUPS)
    assert groups['super_low']['overlap_with_attempt131_very_low1024'] >= 0
    # An all-low new slice must be accepted; source composition is not precommitted.
    all_new = a.combine_rankings(pooled, a.rank_new_surprisal(np.zeros(8192), a126),
        prior, SPEC['combined_selection'], a130, a126)
    assert all_new['groups']['super_low']['source_counts'] == {'0': 0, '1': 0, '2': 1024}
    assert all_new['groups']['next_super_low']['source_counts'] == {'0': 0, '1': 0, '2': 1024}
    damaged = copy.deepcopy(pooled)
    damaged['pair_hashes']['combined_rank_pair_raw_int64_sha256'] = 'x' * 64
    with pytest.raises(ValueError):
        a.combine_rankings(damaged, ranking, prior, SPEC['combined_selection'], a130, a126)


def test_selected_tokens_respect_three_source_ids_and_fixed_1024_count():
    class Source:
        def __init__(self, source_id):
            self.source_id = source_id

        def __getitem__(self, row):
            return Tensor(np.full(128, self.source_id * 100000 + row, dtype=np.int64))

    sources = {'old': Source(0), 'fresh': Source(1), 'new': Source(2)}
    pairs = [[0, 0], [1, 5], [2, 12]] + [[0, row] for row in range(3, 1024)]
    selected = a.selected_tokens(sources, pairs, TinyTorch)
    assert selected.shape == (1024, 128)
    assert selected.array[:3, 0].tolist() == [0, 100005, 200012]
    with pytest.raises(ValueError):
        a.selected_tokens(sources, pairs[:-1], TinyTorch)


def test_gradient_jvp_response_inventory_and_barrier_order():
    source = SOURCE.read_text()
    assert SPEC['gradient_loss']['sample_count'] == 1024
    assert SPEC['gradient_loss']['number_of_batches'] == 128
    assert SPEC['gradient_loss']['total_prediction_tokens'] == 1024 * 127
    assert SPEC['workload']['gradient_backward_batches'] == 256
    assert SPEC['workload']['jvp_probe_batches'] == 64
    assert SPEC['workload']['matched_target_forward_batches'] == 64
    assert SPEC['eligible_tensors']['expected_matrix_count'] == 98
    assert SPEC['tangent']['direction'] == 'delta = +alpha * g'
    assert SPEC['tangent']['target_relative_frobenius'] == .00125
    assert SPEC['tangent']['normalization'] == 'one_global_alpha_across_all_eligible_matrices'
    assert SPEC['candidate']['tensor_order'] == list(a.NAMES) == [
        'response_super_low', 'response_next_super_low']
    assert all(SPEC['candidate'][key] is False for key in
        ('response_normalization', 'sign_selection', 'interpolation',
         'reweighting', 'combination'))
    assert SPEC['probe']['selected_rows'] == [0, 1024]
    assert SPEC['probe']['selected_batch_count'] == 32
    assert SPEC['readout']['hidden_state_index'] == 14
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
    assert 'response.to(torch.float32).contiguous()' in source
    run = ast.get_source_segment(source, next(node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == 'run'))
    assert run.index('validate_old_pool(') < run.index('validate_new_corpus(')
    assert run.index('score_new_surprisal(') < run.index('combine_rankings(')
    assert run.index('combine_rankings(') < run.index('accumulate_mean_generic_gradient(')
    assert run.index('atomic_torch_publish(') < run.index('atomic_json_publish(')
    assert run.index('atomic_json_publish(') < run.index('validate_published(')
    assert run.index('validate_published(') < run.index('print(MARKER')
    assert run.index('print(MARKER') < run.index('evaluate_after_barrier(')
    assert SPEC['barrier']['marker'] == a.MARKER
    assert SPEC['information_policy']['attempt127_129_130_131_historical_scores_before_barrier'] is False
    assert SPEC['evaluation']['target_raw_sha256'] == a.TARGET_SHA256
    assert SPEC['evaluation']['primary_positions'] == [1, 2, 3, 4]
    assert SPEC['evaluation']['secondary_positions'] == list(range(1, 128))


def test_stringency_pool_expansion_signed_thresholds_and_position_counts():
    reports = {a.NAMES[0]: score(.51, .52), a.NAMES[1]: score(.49, .50)}
    gains, category, wins = a.stringency_comparison(reports, a131)
    assert gains == pytest.approx({'primary': .02, 'secondary': .02})
    assert category == 'meaningful_stringency_gain' and wins == 127
    reports[a.NAMES[0]] = score(.495, .505)
    assert a.stringency_comparison(reports, a131)[1] == 'positive_stringency_gain'
    reports[a.NAMES[0]] = score(.495, .495)
    assert a.stringency_comparison(reports, a131)[1] == 'no_stringency_gain'
    with pytest.raises(ValueError):
        a.stringency_comparison({**reports, 'extra': score(.1, .1)}, a131)
    expansion = a.pool_expansion_comparison(score(.51, .52), score(.50, .51))
    assert expansion['classification'] == 'meaningful_pool_expansion_gain'
    assert expansion['super_low_beats_attempt131_very_low_positions_1_127'] == 127
    assert expansion['super_low_minus_attempt131_very_low'] == pytest.approx(
        {'primary': .01, 'secondary': .01})
    assert a.pool_expansion_comparison(score(.501, .511), score(.50, .51))['classification'] == (
        'positive_pool_expansion_gain')
    assert a.pool_expansion_comparison(score(.499, .511), score(.50, .51))['classification'] == (
        'no_pool_expansion_gain')
    bad = score(.51, .52)
    bad['all_128_position_cosines'][3] = None
    with pytest.raises(ValueError):
        a.pool_expansion_comparison(bad, score(.5, .5))


def test_candidate_result_overwrite_refusal_and_no_real_outputs(tmp_path):
    spec = copy.deepcopy(SPEC)
    spec['paths'] = {key: str(tmp_path / Path(value).name)
                     for key, value in SPEC['paths'].items()}
    paths = a.refuse_outputs(spec)
    paths['candidate_path'].write_bytes(b'x')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_outputs(spec)
    paths['candidate_path'].unlink()
    paths['result_path'].symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_outputs(spec)
    paths['result_path'].unlink()
    paths['candidate_path'].write_bytes(b'candidate')
    paths['construction_manifest_path'].write_text('{}')
    with pytest.raises(ValueError, match='Published'):
        a.validate_published(paths, {'candidate': {'serialized_sha256': 'bad'}}, SPEC,
            {}, [], a130, a126, SimpleNamespace(), SimpleNamespace())
    assert all(not a.path_of(value).exists() for value in SPEC['paths'].values())
    assert not a.path_of(SPEC['new_corpus']['corpus_manifest_path']).exists()
    assert SPEC['evaluation']['post_result_candidate_changes'] is False
    assert SPEC['information_policy']['clean_heldout_validation'] is False
    assert SPEC['information_policy']['same_specimen_exploratory_method_development'] is True
