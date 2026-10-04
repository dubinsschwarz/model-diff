"""Synthetic and static Attempt131 controls; no real model execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'scripts/ablation/run_low_surprisal_stringency.py'
loader = importlib.util.spec_from_file_location('attempt131_run', SOURCE)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)
SPEC = a.load_spec()
FROZEN = SPEC['frozen_attempt130']
a130 = a.import_pinned(a.path_of(FROZEN['source_path']), FROZEN['source_sha256'],
                       'attempt131_test_frozen_attempt130')


def score(primary, secondary):
    tail = (127 * secondary - 4 * primary) / 123
    values = [-.2] + [primary] * 4 + [tail] * 123
    return {'position_0_cosine': values[0],
            'positions_1_4_individual_cosines': values[1:5],
            'positions_1_4_mean_cosine': primary,
            'positions_1_127_mean_cosine': secondary,
            'all_128_position_cosines': values}


def prior(attempt, response, primary, secondary):
    result = {'attempt_id': attempt,
              'signed_position_reports': {response: score(primary, secondary)}}
    if attempt.startswith('127_'):
        result['matched_target'] = {'raw_sha256': a.TARGET_SHA256}
    else:
        result['matched_target_raw_sha256'] = a.TARGET_SHA256
    return result


def test_pinned_source_artifacts_and_fixed_plan():
    assert a.sha256_file(a.SPEC_PATH) == a.SPEC_SHA256
    for key in ('spec', 'source', 'construction_manifest', 'candidate'):
        suffix = 'serialized_sha256' if key == 'candidate' else 'sha256'
        assert a.sha256_file(a.path_of(FROZEN[key + '_path'])) == FROZEN[key + '_' + suffix]
    assert FROZEN['spec_sha256'] == a130.SPEC_SHA256
    assert SPEC['source_pooled_selection'] == a130.load_spec()['pooled_selection']
    assert SPEC['stringency_selection']['combined_record_count'] == 8192
    assert [SPEC['stringency_selection']['groups'][name]['rank_range']
            for name in a.GROUPS] == [[0, 1024], [1024, 2048]]
    assert SPEC['gradient_loss']['sample_count'] == 1024
    assert SPEC['gradient_loss']['number_of_batches'] == 128
    assert SPEC['gradient_loss']['total_prediction_tokens'] == 1024 * 127
    assert SPEC['eligible_tensors']['expected_matrix_count'] == 98
    assert SPEC['tangent']['direction'] == 'delta = +alpha * g'
    assert SPEC['tangent']['target_relative_frobenius'] == .00125
    assert SPEC['tangent']['normalization'] == 'one_global_alpha_across_all_eligible_matrices'
    assert SPEC['candidate']['tensor_order'] == list(a.NAMES) == [
        'response_very_low', 'response_next_low']
    assert all(SPEC['candidate'][key] is False for key in
               ('response_normalization', 'sign_selection', 'interpolation',
                'reweighting', 'combination'))
    assert SPEC['probe']['selected_rows'] == [0, 1024]
    assert SPEC['probe']['selected_batch_count'] == 32
    assert SPEC['readout']['hidden_state_index'] == 14
    assert SPEC['evaluation']['target_raw_sha256'] == a.TARGET_SHA256
    assert SPEC['workload'] == {'corpus_download_batches': 0,
        'corpus_freeze_batches': 0, 'surprisal_scoring_batches': 0,
        'gradient_backward_batches': 256, 'jvp_probe_batches': 64,
        'matched_target_forward_batches': 64, 'ggn_operations': 0, 'cg_operations': 0}
    assert SPEC['evaluation']['post_result_candidate_changes'] is False
    assert SPEC['information_policy']['same_specimen_exploratory_method_development'] is True
    assert SPEC['information_policy']['clean_heldout_validation'] is False


def test_attempt130_published_manifest_provenance_before_model_work(monkeypatch):
    manifest = json.loads(a.path_of(FROZEN['construction_manifest_path']).read_text())
    expected_constructions = {name: {'ranking': manifest['frozen_source_rankings'][name]}
                              for name in ('old', 'fresh')}
    monkeypatch.setattr(a130, 'validate_blind_inputs',
        lambda source_spec, a126, a14, torch: ({'old': object(), 'fresh': object()},
            object(), expected_constructions, Path('/fake/model')))
    monkeypatch.setattr(a130, 'validate_published', lambda *args: {})
    tokens, probe, checked, model_dir = a.validate_frozen_attempt130(
        SPEC, a130, SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    assert list(tokens) == ['old', 'fresh']
    assert checked == manifest
    assert model_dir == Path('/fake/model')
    assert probe is not None


def test_exact_published_ranking_partition_counts_stats_and_hashes():
    manifest = json.loads(a.path_of(FROZEN['construction_manifest_path']).read_text())
    pooled = manifest['pooled_ranking']
    assert len(pooled['records']) == 8192
    assert pooled['pair_hashes']['combined_rank_pair_raw_int64_sha256'] == (
        SPEC['stringency_selection']['combined_rank_pair_raw_int64_sha256'])
    groups = a.frozen_stringency_groups(pooled, SPEC['stringency_selection'], a130)
    first, second = (groups[name] for name in a.GROUPS)
    assert first['rank_range'] == [0, 1024]
    assert second['rank_range'] == [1024, 2048]
    assert len(first['ordered_pairs']) == len(second['ordered_pairs']) == 1024
    assert first['ordered_pairs'] + second['ordered_pairs'] == pooled['low_pairs']
    assert set(map(tuple, first['ordered_pairs'])).isdisjoint(
        set(map(tuple, second['ordered_pairs'])))
    assert first['source_counts'] == {'old': 496, 'fresh': 528}
    assert second['source_counts'] == {'old': 520, 'fresh': 504}
    assert first['ordered_pair_raw_int64_sha256'] == (
        '34ed2eef5b17535b8f4548567c89318e07514a01e531178386e3f03331e21454')
    assert second['ordered_pair_raw_int64_sha256'] == (
        'ee66019002e32a40371b3ad16261edcd4fc0f780f6e6137d38420b75bc8650d0')
    for name, minimum, maximum, mean in (
        ('very_low', .7337587863145862, 2.6494123447992597, 2.366551253087838),
        ('next_low', 2.6496000120028538, 2.9017775442512423, 2.781518447389662)):
        assert groups[name]['surprisal_min'] == pytest.approx(minimum, abs=1e-12)
        assert groups[name]['surprisal_max'] == pytest.approx(maximum, abs=1e-12)
        assert groups[name]['surprisal_mean'] == pytest.approx(mean, abs=1e-12)
    damaged = copy.deepcopy(pooled)
    damaged['records'][0][1] = 1 - damaged['records'][0][1]
    with pytest.raises(ValueError):
        a.frozen_stringency_groups(damaged, SPEC['stringency_selection'], a130)
    damaged = copy.deepcopy(pooled)
    damaged['records'][1024][0] = float('nan')
    with pytest.raises(ValueError):
        a.frozen_stringency_groups(damaged, SPEC['stringency_selection'], a130)


def test_gradient_jvp_and_blind_barrier_static_controls():
    source = SOURCE.read_text()
    assert 'score_endpoint_surprisal(' not in source
    assert 'sorted_combined_records(' not in source
    assert 'a130.validate_published(' in source
    assert 'a130.validate_blind_inputs(' in source
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
    assert 'a126.response_cosine(' in source
    run = ast.get_source_segment(source, next(node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == 'run'))
    assert run.index('validate_frozen_attempt130(') < run.index('frozen_stringency_groups(')
    assert run.index('frozen_stringency_groups(') < run.index('accumulate_mean_generic_gradient(')
    assert run.index('atomic_torch_publish(') < run.index('atomic_json_publish(')
    assert run.index('atomic_json_publish(') < run.index('validate_published(')
    assert run.index('validate_published(') < run.index('print(MARKER')
    assert run.index('print(MARKER') < run.index('evaluate_after_barrier(')
    assert SPEC['barrier']['marker'] == a.MARKER
    assert SPEC['information_policy']['attempt127_129_130_historical_scores_before_barrier'] is False


def test_signed_stringency_thresholds_and_position_wins():
    reports = {a.NAMES[0]: score(.51, .52), a.NAMES[1]: score(.49, .50)}
    gains, category, wins = a.stringency_comparison(reports)
    assert gains == pytest.approx({'primary': .02, 'secondary': .02})
    assert category == 'meaningful_stringency_gain'
    assert wins == 127
    reports[a.NAMES[0]] = score(.495, .505)
    assert a.stringency_comparison(reports)[1] == 'positive_stringency_gain'
    reports[a.NAMES[0]] = score(.495, .495)
    assert a.stringency_comparison(reports)[1] == 'no_stringency_gain'
    with pytest.raises(ValueError):
        a.stringency_comparison({**reports, 'response_extra': score(.1, .1)})
    damaged = copy.deepcopy(reports)
    damaged[a.NAMES[0]]['all_128_position_cosines'][2] = None
    with pytest.raises(ValueError):
        a.stringency_comparison(damaged)


def test_absolute_improvement_exact_rules_and_deltas():
    old = prior('127_postfreeze_surprisal_strata_evaluation', 'response_low', .45, .48)
    fresh = prior('129_fresh_fineweb_low_surprisal_replication', 'response_low', .46, .49)
    pooled = prior('130_pooled_fineweb_low_surprisal_scaling', 'response_low2048', .455, .485)
    clear = a.absolute_comparison(score(.47, .50), old, fresh, pooled)
    assert clear['classification'] == 'clear_absolute_improvement'
    assert clear['very_low_minus_prior']['old_low1024']['primary'] == pytest.approx(.02)
    assert clear['very_low_minus_prior']['low2048']['secondary'] == pytest.approx(.015)
    assert clear['very_low_minus_prior']['mean_prior_low1024']['primary'] == pytest.approx(.015)
    average = a.absolute_comparison(score(.4551, .4851), old, fresh, pooled)
    assert average['classification'] == 'average_absolute_improvement'
    assert a.absolute_comparison(score(.455, .485), old, fresh, pooled)['classification'] == (
        'no_absolute_improvement')
    corrupt = copy.deepcopy(pooled)
    corrupt['matched_target_raw_sha256'] = 'x' * 64
    with pytest.raises(ValueError):
        a.absolute_comparison(score(.47, .50), old, fresh, corrupt)


def test_overwrite_refusal_published_hash_and_no_real_outputs(tmp_path):
    spec = copy.deepcopy(SPEC)
    spec['paths'] = {key: str(tmp_path / Path(value).name)
                     for key, value in SPEC['paths'].items()}
    paths = a.refuse_outputs(spec)
    paths['candidate_path'].write_bytes(b'x')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_outputs(spec)
    paths['candidate_path'].unlink()
    paths['construction_manifest_path'].symlink_to(tmp_path / 'missing')
    with pytest.raises(ValueError, match='already exists'):
        a.refuse_outputs(spec)
    paths['construction_manifest_path'].unlink()
    paths['candidate_path'].write_bytes(b'candidate')
    paths['construction_manifest_path'].write_text('{}')
    construction = {'candidate': {'serialized_sha256': 'bad', 'tensor_order': list(a.NAMES)}}
    with pytest.raises(ValueError, match='Published'):
        a.validate_published(paths, construction, SPEC, a130,
                             SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    assert all(not a.path_of(value).exists() for value in SPEC['paths'].values())
    assert a.TARGET_SHA256 == '4ab001c60fff4ad4f15db472436c3915b39fb5a69626a418f70c5eaaed780c5f'
    assert 'context_only_different_probe_construction_population' in SOURCE.read_text()
