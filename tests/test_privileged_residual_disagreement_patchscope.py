"""Attempt136 synthetic algebra and static provenance tests; no real model is loaded."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ablation/run_privileged_residual_disagreement_patchscope.py"
SPEC = ROOT / "experiments/attempts/136_privileged_residual_disagreement_patchscope/spec.json"
ATTEMPT133_SOURCE = ROOT / "scripts/ablation/run_logit_lens_patchscope_characterization.py"


def load_module(path, name):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


runner = load_module(SOURCE, "attempt136_runner")
a133 = load_module(ATTEMPT133_SOURCE, "attempt136_pinned_attempt133_methods")


def test_exact_committed_input_identities_and_no_candidate_output(monkeypatch):
    spec = runner.load_spec(SPEC)
    assert spec["expected_prior_head_commit"] == "046fda47a296667bac2fbd47311fb44b810426be"
    assert runner.ORDER == ("fineweb", "wikitext103_raw", "tinystories", "arxiv_document",
                            "cc_news", "pg19", "codeparrot_clean", "ultrachat")
    assert spec["joint_view_order"] == ["full:" + name for name in runner.ORDER] + [
        "low:" + name for name in runner.ORDER]
    expected = {
        "attempt100": "64c4d3ef379fc02f6132d6e3197632e7bd5154398ee123e5ca688be9d5c68546",
        "attempt134": "d928d80c08acee5eae78ff1439c42a0d16048726c89a50d7af542dfc5f36c99a",
        "attempt135_context_only": "c3e5081d091d3114313423dcb1e373f05697a4b29ebeb405297f5c73ffea5358",
    }
    for group, items in spec["files"].items():
        pins = [items] if "path" in items else items.values()
        for pin in pins:
            assert len(pin["sha256"]) == 64
            assert runner.sha256_file(runner.path_of(pin["path"])) == pin["sha256"]
        if group in expected:
            assert items["construction_manifest"]["sha256"] == expected[group]
    assert spec["artifacts"]["attempt100"]["serialized_sha256"] == "ce32f037d594a45126d72f88ffcf1dfef49996bc6152aeb431f07851c50d3872"
    assert spec["artifacts"]["attempt134"]["serialized_sha256"] == "055dc78ce83d8ad01cc04e647efba0138032efb9083c10d00efc4855b2573482"
    assert runner.sha256_file(runner.path_of(spec["artifacts"]["attempt134"]["path"])) == spec["artifacts"]["attempt134"]["serialized_sha256"]
    assert spec["artifacts"]["oracle"]["difference_raw_sha256"] == "d60789a2c06eb82dba2f2449d8665ec665d85e9f7e33fdea1f66ac79c81c159d"
    assert set(spec).isdisjoint({"candidate_path", "construction_manifest_path"})
    assert spec["output"].endswith("/result.json")
    assert spec["no_candidate_artifact"] is True
    assert spec["no_construction_manifest"] is True
    assert "torch.save(" not in SOURCE.read_text()
    assert spec["privileged_diagnostic_only"] is True
    assert spec["no_new_candidate_constructed"] is True
    visited = []
    monkeypatch.setattr(runner, "require_hash", lambda path, digest: visited.append((path, digest)))
    runner.validate_pinned_files(spec, blind_only=True)
    runner.validate_pinned_files(spec, blind_only=False)
    assert any(path.name == "oracle-adl-manifest.json" for path, _ in visited)


def test_positionwise_projection_orthogonality_and_raw_aggregate_energy():
    c = np.zeros((128, 4), dtype=np.float32)
    o = np.zeros_like(c)
    c[:, 0] = 2
    o[:, 0] = 4
    o[:, 1] = 3
    parallel, residual, report = runner.positionwise_residual(o, c)
    np.testing.assert_array_equal(parallel[:, 0], 4)
    np.testing.assert_array_equal(residual[:, 1], 3)
    np.testing.assert_array_equal(residual[:, 0], 0)
    assert report["position_0"]["alpha"] == 2
    assert report["position_0"]["residual_squared_fraction"] == pytest.approx(9 / 25)
    assert report["position_0"]["orthogonality_error_relative_to_normO_normC"] == 0
    for range_name in runner.RANGES:
        aggregate = report["aggregates"][range_name]
        assert aggregate["parallel_energy_fraction"] == pytest.approx(16 / 25)
        assert aggregate["residual_energy_fraction"] == pytest.approx(9 / 25)
        assert aggregate["energy_fraction_sum"] == pytest.approx(1)
    # Different positions need different alpha: aggregate energy is computed
    # after independent projections, without a globally refit scalar.
    o[2, 0] = 8
    _, _, varied = runner.positionwise_residual(o, c)
    assert varied["all_128_positions"][2]["alpha"] == 4
    assert varied["all_128_positions"][1]["alpha"] == 2


def test_float64_unit_normalization_centering_and_six_fixed_svd_spaces(monkeypatch):
    monkeypatch.setattr(runner, "SHAPE", (128, 4))
    rng = np.random.default_rng(136)
    full = rng.normal(size=(8, 128, 4)).astype(np.float32)
    low = rng.normal(size=(8, 128, 4)).astype(np.float32)
    uf, ul = runner.unit_views(full), runner.unit_views(low)
    assert uf.dtype == ul.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(uf, axis=2), 1, atol=1e-15)
    for views in (uf, ul, np.concatenate((uf, ul), axis=0)):
        centered = runner.centered_disagreement(views)
        np.testing.assert_allclose(centered.sum(axis=0), 0, atol=1e-14)
    bases = runner.build_blind_bases(full, low)
    assert set(bases) == {"full_8", "low_8", "joint_16"}
    for space, bound, rows in (("full_8", 7, 8), ("low_8", 7, 8), ("joint_16", 15, 16)):
        for range_name, positions in (("positions_1_4", 4), ("positions_1_127", 127)):
            basis = bases[space][range_name]
            assert basis["matrix"].shape == (rows, positions * 4)
            assert basis["rank"] <= bound
            assert basis["rank_tolerance"] == pytest.approx(
                max(basis["matrix"].shape) * np.finfo(np.float64).eps * basis["singular_values"][0])
            for i in range(basis["rank"]):
                absolute = np.abs(basis["u"][:, i])
                pivot = int(np.flatnonzero(np.isclose(
                    absolute, absolute.max(), rtol=0,
                    atol=32 * np.finfo(np.float64).eps))[0])
                assert basis["u"][pivot, i] >= 0  # oracle-free PC sign
    assert bases["joint_16"]["positions_1_4"]["view_names"] == runner.JOINT_NAMES
    assert "np.linalg.svd(d, full_matrices=False)" in inspect.getsource(runner.blind_svd)
    assert "np.flatnonzero(np.isclose(" in inspect.getsource(runner.blind_svd)


def test_known_residual_span_projection_pc_fractions_and_row_metrics():
    d = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 0., 0.]], dtype=np.float64)
    basis = runner.blind_svd(d, ("a", "b", "c"), rank_bound=2)
    assert basis["rank"] == 1
    r = np.array([3., 4., 0.])
    result = runner.analyze_residual_in_basis(basis, r)
    assert result["residual_squared_fraction_in_span"] == pytest.approx(9 / 25)
    assert result["residual_norm_fraction_in_span"] == pytest.approx(3 / 5)
    assert sum(row["residual_squared_fraction_carried"] for row in result["pcs"]) == pytest.approx(9 / 25)
    assert result["residual_projection_reconstruction_error"] < 1e-12
    assert result["disagreement_rows"][2]["signed_cosine_with_residual"] is None
    assert result["pcs"][0]["oriented_view_loadings"]["a"] > 0
    assert result["pcs"][0]["signed_cosine_with_oracle_residual"] == pytest.approx(3 / 5)
    assert runner.classify_low_residual_presence(.50) == "strong_residual_presence"
    assert runner.classify_low_residual_presence(.25) == "partial_residual_presence"
    assert runner.classify_low_residual_presence(.249) == "limited_residual_presence"


def test_exact_attempt133_lens_prompts_semantics_and_patching_conventions():
    spec = runner.load_spec(SPEC)
    runner.verify_attempt133_conventions(spec, a133)
    assert spec["prompts"] == list(a133.PROMPTS)
    assert len(spec["prompts"]) == 8
    assert spec["semantic_substrings"] == ["cake", "bake", "cook", "craft", "precision"]
    assert spec["semantic_substrings"] == list(a133.SEMANTIC_TERMS)
    assert spec["positions"] == [0, 1, 2, 3, 4]
    assert spec["top_k"] == a133.TOP_K == 20
    assert spec["greedy_completion_tokens"] == a133.NEW_TOKENS == 6
    assert spec["patch_layer_index"] == a133.LAYER_INDEX == 13
    assert spec["attempt133_provenance"]["logit_lens"]["negative"] == "softmax(lm_head(-model.model.norm(vector)))"
    assert 'normed = final_norm(vector)' in inspect.getsource(a133.lens_distribution)
    assert 'negative_logits = lm_head(-normed)' in inspect.getsource(a133.lens_distribution)
    assert 'patched[:, patch_index, :] = hidden[:, patch_index, :] + sign * vector.to(' in inspect.getsource(a133.patch_hidden_output)
    assert 'use_cache=False' in inspect.getsource(a133.next_logits)
    assert 'original_final_index = prompt_ids.shape[1] - 1' in inspect.getsource(a133.greedy_completion)
    patch_source = inspect.getsource(runner.patchscope_mode)
    assert 'tokenizer(prompt, add_special_tokens=False, return_tensors="pt")' in patch_source
    assert 'a133.next_logits(model, ids, patch_index,' in patch_source
    assert 'a133.greedy_completion(' in patch_source
    assert 'a133.token_delta_records(' in inspect.getsource(runner.patch_record)


def test_oracle_norm_matched_and_native_amplitude_formulas():
    oracle = np.array([0., 3., 4.], dtype=np.float32)
    blind = np.array([2., 0., 0.], dtype=np.float32)
    scaled = a133.norm_match_blind(blind, oracle)
    np.testing.assert_allclose(scaled, [5., 0., 0.])
    assert np.linalg.norm(scaled) == pytest.approx(np.linalg.norm(oracle))
    assert runner.nullable_cosine(np.zeros(3), oracle) is None
    assert runner.nullable_cosine(blind, oracle) == 0
    add = runner.native_additivity(np.array([1., 0.]), np.array([0., 2.]), np.array([1., 2.]))
    assert add["cosine_delta_P_plus_delta_R_vs_delta_O"] == pytest.approx(1)
    assert add["relative_error_delta_P_plus_delta_R_vs_delta_O"] == pytest.approx(0)
    zero = runner.native_additivity(np.array([1., 0.]), np.array([0., 2.]), np.zeros(2))
    assert zero["cosine_delta_P_plus_delta_R_vs_delta_O"] is None
    assert zero["relative_error_delta_P_plus_delta_R_vs_delta_O"] is None

    ratios = runner.native_norm_ratios(
        np.tile(oracle, (128, 1)), np.tile(blind, (128, 1)),
        np.tile(oracle * 0.5, (128, 1)), np.tile(oracle * 0.25, (128, 1)))
    assert ratios["0"]["attempt134_to_oracle"] == pytest.approx(2 / 5)
    assert ratios["0"]["parallel_to_oracle"] == pytest.approx(0.5)
    assert ratios["0"]["residual_to_oracle"] == pytest.approx(0.25)
    assert 'source_vector.to(device=device)' in inspect.getsource(runner.patchscope_mode)
    assert 'a133.norm_match_blind(source_vector.numpy(), oracle_vector.numpy())' in inspect.getsource(runner.patchscope_mode)
    assert 'for label in ("plus", "minus")}' in inspect.getsource(runner.patchscope_mode)


def test_nonfinite_patch_delta_norm_is_reported_without_a_fabricated_cosine():
    delta = np.array([np.inf, 1.], dtype=np.float64)
    patched = np.array([1., 2.], dtype=np.float32)
    probs = np.array([0.5, 0.5], dtype=np.float32)
    record = runner.patch_record(delta, patched, probs, patched, probs, None, a133)
    assert record["delta_logit_l2_norm"] is None
    assert record["semantic_hits_available"] is False
    assert runner.nullable_cosine(delta, np.ones(2)) is None


def test_fixed_semantic_family_counts_and_patchscope_thresholds():
    names = ("oracle", "attempt100", "attempt134", "residual")
    prompts = ("fixed prompt",)
    terms = a133.SEMANTIC_TERMS
    conditions = {name: {str(pos): {"fixed prompt": {}}
                         for pos in runner.POSITIONS} for name in names}
    for name in names:
        for pos in runner.POSITIONS:
            for sign in ("plus", "minus"):
                hits = {term: {"positive": 0, "negative": 0} for term in terms}
                if name == "residual" and sign == "plus":
                    hits["precision"]["positive"] = 1
                    if pos == 0:
                        hits["cake"]["negative"] = 1
                        hits["bake"]["positive"] = 1
                conditions[name][str(pos)]["fixed prompt"][sign] = {"semantic_hits": hits}
    summary = runner.semantic_patch_summary(conditions, names, prompts, terms)
    assert summary["residual"]["plus"]["positions_1_4"]["precision_family_hits"] == 4
    assert summary["residual"]["plus"]["position_0"]["cake_bake_cook_family_hits"] == 2
    assert summary["attempt134"]["plus"]["positions_1_4"]["precision_family_hits"] == 0
    assert runner.classify_patchscope_gain(.01) == "clear_patchscope_gain"
    assert runner.classify_patchscope_gain(.001) == "positive_patchscope_gain"
    assert runner.classify_patchscope_gain(0) == "no_patchscope_gain"


def test_privileged_only_result_structure_stage_order_and_overwrite_refusal(tmp_path):
    spec = runner.load_spec(SPEC)
    source = inspect.getsource(runner.run)
    assert source.index('build_blind_bases(full, low)') < source.index('load_and_validate_oracle(')
    assert source.index('positionwise_residual(') < source.index('analyze_residual_in_basis(')
    assert source.index('direct_lens(') < source.index('patchscope_mode(vectors, "directional_oracle_norm_matched"')
    assert source.index('patchscope_mode(vectors, "directional_oracle_norm_matched"') < source.index('patchscope_mode(vectors, "native_amplitude"')
    assert '"observed_quantities"' in source
    assert '"precommitted_classifications"' in source
    assert '"descriptive_interpretation"' in source
    assert '"no_oracle_selected_pc_or_blind_rule": True' in source
    assert 'torch.save(' not in source
    output = tmp_path / "result.json"
    runner.require_output_absent(output)
    output.write_text("existing")
    with pytest.raises(ValueError, match="already exists"):
        runner.require_output_absent(output)
    assert spec["same_specimen_exploratory_method_development"] is True
    assert spec["clean_heldout_validation"] is False
