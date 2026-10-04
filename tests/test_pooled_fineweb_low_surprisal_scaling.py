"""Static and synthetic unit controls for Attempt130; never load real models."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'scripts/ablation/run_pooled_fineweb_low_surprisal_scaling.py'
loader = importlib.util.spec_from_file_location('attempt130_run', SOURCE)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)
SPEC = a.load_spec()
OLD = SPEC['frozen_slices']['old']
a126 = a.import_pinned(a.path_of(OLD['constructor_path']), OLD['constructor_sha256'],
                       'attempt130_test_frozen_ranking')


def score(primary, secondary):
    tail = (127 * secondary - 4 * primary) / 123
    values = [-.2] + [primary] * 4 + [tail] * 123
    return {'position_0_cosine': values[0],
            'positions_1_4_individual_cosines': values[1:5],
            'positions_1_4_mean_cosine': primary,
            'positions_1_127_mean_cosine': secondary,
            'all_128_position_cosines': values}


def prior(attempt, primary, secondary):
    result = {'attempt_id': attempt,
              'signed_position_reports': {'response_low': score(primary, secondary)}}
    if attempt.startswith('127_'):
        result['matched_target'] = {'raw_sha256': a.TARGET_SHA256}
    else:
        result['matched_target_raw_sha256'] = a.TARGET_SHA256
    return result


def test_exact_pooled_plan_and_frozen_provenance():
    assert a.SPEC_SHA256 == a.sha256_file(a.SPEC_PATH)
    assert list(SPEC['frozen_slices']) == ['old', 'fresh']
    assert [(SPEC['frozen_slices'][k]['slice_id'],
             SPEC['frozen_slices'][k]['valid_example_rank_interval'])
            for k in ('old', 'fresh')] == [(0, [20000, 24096]), (1, [24096, 28192])]
    for source in SPEC['frozen_slices'].values():
        for stem in ('corpus_spec', 'corpus_manifest', 'construction_spec',
                     'constructor', 'construction_manifest'):
            assert a.sha256_file(a.path_of(source[stem + '_path'])) == source[stem + '_sha256']
        assert a.sha256_file(a.path_of(source['candidate_path'])) == source['candidate_serialized_sha256']
        corpus_spec = json.loads(a.path_of(source['corpus_spec_path']).read_text())
        corpus_manifest = json.loads(a.path_of(source['corpus_manifest_path']).read_text())
        assert corpus_spec['selection']['sample_count'] == 4096
        assert corpus_manifest['tensor']['shape'] == [4096, 128]
        assert corpus_manifest['artifact']['raw_tensor_sha256'] == source['tokens_raw_sha256']
    assert SPEC['frozen_slices']['old']['valid_example_rank_interval'][1] == (
        SPEC['frozen_slices']['fresh']['valid_example_rank_interval'][0])
    assert SPEC['probe']['selected_rows'] == [0, 1024]
    assert SPEC['probe']['selected_batch_count'] == 32
    assert SPEC['readout']['hidden_state_index'] == 14
    assert SPEC['eligible_tensors']['expected_matrix_count'] == 98
    assert SPEC['gradient_loss']['sample_count'] == 2048
    assert SPEC['gradient_loss']['number_of_batches'] == 256
    assert SPEC['gradient_loss']['total_prediction_tokens'] == 2048 * 127
    assert SPEC['workload']['gradient_backward_batches'] == 512
    assert SPEC['workload']['jvp_probe_batches'] == 64
    assert SPEC['workload']['matched_target_forward_batches'] == 64
    assert SPEC['workload']['surprisal_scoring_batches'] == 0
    assert SPEC['candidate']['tensor_order'] == list(a.NAMES) == [
        'response_low2048', 'response_control2048']
    assert SPEC['evaluation']['target_raw_sha256'] == a.TARGET_SHA256


def test_combined_tie_break_and_exact_frozen_row_hashes():
    old = [float(i + 1) for i in range(4096)]
    fresh = [float(i + 5000) for i in range(4096)]
    old[1] = old[2] = fresh[0] = 0.0
    records = a.sorted_combined_records((old, fresh))
    assert records[:3] == [(0.0, 0, 1), (0.0, 0, 2), (0.0, 1, 0)]
    with pytest.raises(ValueError):
        a.sorted_combined_records((old[:-1], fresh))
    with pytest.raises(ValueError):
        a.sorted_combined_records((old, [float('nan')] + fresh[1:]))
    frozen = {key: json.loads(a.path_of(SPEC['frozen_slices'][key]['construction_manifest_path']).read_text())['ranking']
              for key in ('old', 'fresh')}
    ranking = a.pooled_ranking(frozen['old'], frozen['fresh'], SPEC['pooled_selection'], a126)
    assert len(ranking['records']) == 8192
    assert len(ranking['low_pairs']) == len(ranking['control_pairs']) == 2048
    assert ranking['low_source_counts'] == {'old': 1016, 'fresh': 1032}
    assert ranking['low_control_overlap_count'] == 490
    assert ranking['control_pairs'] == [[0, i] for i in range(1024)] + [
        [1, i] for i in range(1024)]
    assert ranking['pair_hashes'] == {
        'combined_rank_pair_raw_int64_sha256':
            '84bfc8c07f576b61ac3ccda0b56347a2d12a130eccbaef472c38a1eca9cdf8f5',
        'low_selected_pairs_raw_int64_sha256':
            '4532ad09e155aad5a8b941aa00a337522a7f8e81c77761e9b6fab41775c62147',
        'control_selected_pairs_raw_int64_sha256':
            '4a63d06212c499d569beffc4b9ab3f17e082769b57036fc128a84a5ec6b600af'}
    assert ranking['pair_hashes']['low_selected_pairs_raw_int64_sha256'] == (
        SPEC['pooled_selection']['low_selected_pairs_raw_int64_sha256'])
    damaged = copy.deepcopy(frozen['old'])
    damaged['surprisal_values'][0] += .1
    with pytest.raises(ValueError):
        a.pooled_ranking(damaged, frozen['fresh'], SPEC['pooled_selection'], a126)


def test_blind_provenance_validation_uses_frozen_inputs_without_models(monkeypatch):
    """Exercise manifest cross-checks while stubbing inaccessible scratch tensors."""
    original_require = a.require_hash

    def checked(path, expected):
        if str(path).startswith('/root/model-diff-scratch/'):
            return Path(path)
        return original_require(path, expected)

    class ShapeOnly:
        def __init__(self, shape):
            self.shape = shape

        def __getitem__(self, item):
            return ShapeOnly((1024, 128))

        def contiguous(self):
            return self

    fake_a14 = SimpleNamespace(
        load_corpus_tokens=lambda path, raw, rows, length, torch: ShapeOnly((rows, length)),
        checkpoint_file_records=lambda path: SPEC['final_checkpoint']['files'],
        load_probe=lambda path, probe_spec, torch: ShapeOnly((10000, 128)))
    monkeypatch.setattr(a, 'require_hash', checked)
    tokens, probe, construction, _ = a.validate_blind_inputs(
        SPEC, a126, fake_a14, SimpleNamespace())
    assert list(tokens) == ['old', 'fresh']
    assert all(t.shape == (4096, 128) for t in tokens.values())
    assert probe.shape == (1024, 128)
    assert construction['old']['ranking']['surprisal_raw_float64_sha256'] == (
        OLD['surprisal_raw_float64_sha256'])
    assert construction['fresh']['ranking']['surprisal_raw_float64_sha256'] == (
        SPEC['frozen_slices']['fresh']['surprisal_raw_float64_sha256'])


def test_gradient_tangent_jvp_and_barrier_structure():
    source = SOURCE.read_text()
    assert 'score_endpoint_surprisal(' not in source
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
    assert SPEC['tangent']['direction'] == 'delta = +alpha * g'
    assert SPEC['tangent']['target_relative_frobenius'] == .00125
    assert SPEC['tangent']['normalization'] == 'one_global_alpha_across_all_eligible_matrices'
    assert SPEC['barrier']['marker'] == a.MARKER
    run = ast.get_source_segment(source, next(node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == 'run'))
    assert run.index('pooled_ranking(') < run.index('accumulate_mean_generic_gradient(')
    assert run.index('atomic_torch_publish(') < run.index('atomic_json_publish(')
    assert run.index('atomic_json_publish(') < run.index('validate_published(')
    assert run.index('validate_published(') < run.index('print(MARKER')
    assert run.index('print(MARKER') < run.index('evaluate_after_barrier(')
    assert source.count('a14.accumulate_mean_generic_gradient(') == 1
    assert "for group in ('low2048', 'control2048'):" in source
    assert 'response.to(torch.float32).contiguous()' in source
    assert 'a126.response_cosine(' in source
    assert 'matched_target_forward_batches' in source


def test_signed_selection_gain_and_position_count():
    reports = {a.NAMES[0]: score(.51, .52), a.NAMES[1]: score(.49, .50)}
    gain, category, wins = a.score_comparison(reports)
    assert gain['primary'] == pytest.approx(.02)
    assert gain['secondary'] == pytest.approx(.02)
    assert category == 'meaningful_selection_gain'
    assert wins == 127
    reports[a.NAMES[0]] = score(.495, .505)
    assert a.score_comparison(reports)[1] == 'positive_selection_gain'
    reports[a.NAMES[0]] = score(.495, .495)
    assert a.score_comparison(reports)[1] == 'no_selection_gain'
    with pytest.raises(ValueError):
        a.score_comparison({**reports, 'extra': score(.1, .1)})
    bad = copy.deepcopy(reports)
    bad[a.NAMES[0]]['all_128_position_cosines'][1] = None
    with pytest.raises(ValueError):
        a.score_comparison(bad)


def test_scaling_uses_exact_two_prior_low1024_candidates():
    old = prior('127_postfreeze_surprisal_strata_evaluation', .45, .48)
    fresh = prior('129_fresh_fineweb_low_surprisal_replication', .46, .49)
    clear = a.scaling_comparison(score(.47, .50), old, fresh)
    assert clear['classification'] == 'clear_scaling_gain'
    assert clear['gains']['minus_mean_low1024']['primary'] == pytest.approx(.015)
    assert clear['gains']['minus_best_low1024']['secondary'] == pytest.approx(.01)
    assert a.scaling_comparison(score(.4551, .4851), old, fresh)['classification'] == (
        'average_scaling_gain')
    assert a.scaling_comparison(score(.455, .485), old, fresh)['classification'] == (
        'no_scaling_gain')
    corrupt = copy.deepcopy(old)
    corrupt['matched_target']['raw_sha256'] = 'x' * 64
    with pytest.raises(ValueError):
        a.scaling_comparison(score(.47, .50), corrupt, fresh)


def test_candidate_hash_revalidation_and_overwrite_refusal(tmp_path):
    pathspec = copy.deepcopy(SPEC)
    pathspec['paths'] = {key: str(tmp_path / Path(value).name)
                         for key, value in SPEC['paths'].items()}
    paths = a.refuse_outputs(pathspec)
    paths['candidate_path'].write_bytes(b'x')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_outputs(pathspec)
    paths['candidate_path'].unlink()
    paths['construction_manifest_path'].symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_outputs(pathspec)
    paths['construction_manifest_path'].unlink()
    paths['candidate_path'].write_bytes(b'candidate')
    paths['construction_manifest_path'].write_text('{}')
    construction = {'candidate': {'serialized_sha256': 'bad', 'tensor_order': list(a.NAMES)},
                    'barrier': SPEC['barrier']}
    with pytest.raises(ValueError, match='Published'):
        a.validate_published(paths, construction, SPEC['pooled_selection'], a126,
                             SimpleNamespace(), SimpleNamespace())


def test_no_real_outputs_written_and_context_only_consensus():
    assert not a.path_of(SPEC['paths']['candidate_path']).exists()
    assert not a.path_of(SPEC['paths']['construction_manifest_path']).exists()
    assert not a.path_of(SPEC['paths']['result_path']).exists()
    assert a.TARGET_SHA256 == '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f'
    source = SOURCE.read_text()
    assert "'consensus_response_8'" in source
    assert "'role': 'context_only_different_probe_construction_population'" in source
    assert "'candidate': {'path': str(paths['candidate_path'])" in source
    assert 'best_seed' not in source
    assert 'candidate_combination' not in source
