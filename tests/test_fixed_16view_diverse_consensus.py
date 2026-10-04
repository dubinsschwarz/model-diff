"""Synthetic and static Attempt135 tests; no frozen response or oracle is opened."""

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
SOURCE = ROOT / "scripts/ablation/run_fixed_16view_diverse_consensus.py"
SPEC = ROOT / "experiments/attempts/135_fixed_16view_diverse_consensus/spec.json"
loader = importlib.util.spec_from_file_location("attempt135_runner", SOURCE)
runner = importlib.util.module_from_spec(loader)
loader.loader.exec_module(runner)


def test_fixed_order_formula_and_pinned_parent_constructions():
    spec = runner.load_spec(SPEC)
    assert runner.ORDER == ("fineweb", "wikitext103_raw", "tinystories", "arxiv_document",
                            "cc_news", "pg19", "codeparrot_clean", "ultrachat")
    assert runner.VIEW_ORDER == tuple(name for corpus in runner.ORDER
                                      for name in ("response_" + corpus, "response_low_" + corpus))
    assert len(runner.VIEW_ORDER) == 16
    assert spec["input_view_order"] == list(runner.VIEW_ORDER)
    assert spec["consensus"]["weights"] == [1 / 16] * 16
    assert spec["consensus"]["corpus_weights"] == [1 / 8] * 8
    assert spec["consensus"]["within_corpus_full_low_weights"] == [0.5, 0.5]
    assert spec["consensus"]["normalization_dtype"] == "cpu_float64"
    assert spec["consensus"]["final_renormalization"] is False
    assert spec["consensus"]["data_dependent_weights"] is False
    assert spec["candidate"]["tensor_order"] == ["consensus_response_16"]
    for key in ("attempt100", "attempt134"):
        pin = spec["source_artifacts"][key]
        for field in ("spec", "constructor", "construction_manifest"):
            assert runner.sha256_file(runner.path_of(pin[field + "_path"])) == pin[field + "_sha256"]
        assert len(pin["artifact_serialized_sha256"]) == 64
        assert all(len(value) == 64 for value in pin["raw_tensors_sha256"].values())
    old = spec["source_artifacts"]["attempt100"]
    low = spec["source_artifacts"]["attempt134"]
    assert old["construction_manifest_sha256"] == "64c4d3ef379fc02f6132d6e3197632e7bd5154398ee123e5ca688be9d5c68546"
    assert low["construction_manifest_sha256"] == "d928d80c08acee5eae78ff1439c42a0d16048726c89a50d7af542dfc5f36c99a"
    assert old["artifact_serialized_sha256"] == "ce32f037d594a45126d72f88ffcf1dfef49996bc6152aeb431f07851c50d3872"
    assert low["artifact_serialized_sha256"] == "055dc78ce83d8ad01cc04e647efba0138032efb9083c10d00efc4855b2573482"
    assert runner.sha256_file(runner.path_of(low["artifact_path"])) == low["artifact_serialized_sha256"]
    old_manifest = json.loads(runner.path_of(old["construction_manifest_path"]).read_text())
    low_manifest = json.loads(runner.path_of(low["construction_manifest_path"]).read_text())
    assert old["raw_tensors_sha256"] == old_manifest["artifact"]["raw_tensors_sha256"]
    assert low["raw_tensors_sha256"] == {name: row["raw_sha256"]
                                            for name, row in low_manifest["responses"].items()}
    assert spec["workload"] == {"model_forwards": 0, "gradients": 0, "jvps": 0,
                                "gpu_required": False, "source_response_views": 16}


def synthetic_views():
    full = np.zeros((128, 2048), dtype=np.float32)
    low = np.zeros_like(full)
    full[:, 0] = 3
    low[:, 1] = 7
    arrays = {name: full.copy() if not name.startswith("response_low_") else low.copy()
              for name in runner.VIEW_ORDER}
    arrays["consensus_response_8"] = full.copy()
    arrays["low_surprisal_consensus_8"] = low.copy()
    return arrays


def test_cpu_float64_positionwise_equal_sixteen_view_consensus_no_final_renorm():
    arrays = synthetic_views()
    candidate, geometry = runner.construct_consensus(arrays)
    assert candidate.dtype == np.float32 and candidate.flags.c_contiguous
    assert candidate.shape == (128, 2048)
    np.testing.assert_array_equal(candidate[:, :2], np.full((128, 2), 0.5, dtype=np.float32))
    np.testing.assert_array_equal(candidate[:, 2:], 0)
    np.testing.assert_allclose(np.linalg.norm(candidate.astype(np.float64), axis=1), 2**-0.5)
    # 1/16 over 16 views is exactly equal to 1/8 over corpus-wise 50/50 means.
    corpus_mixes = [0.5 * np.array([1., 0.]) + 0.5 * np.array([0., 1.])
                    for _ in runner.ORDER]
    np.testing.assert_array_equal(candidate[0, :2], np.mean(corpus_mixes, axis=0))
    concentration = geometry["consensus16_per_position_concentration"]
    assert concentration["mean_positions_1_4"] == pytest.approx(2**-0.5)
    assert concentration["mean_positions_1_127"] == pytest.approx(2**-0.5)
    pairs = geometry["parent_and_new_consensus_pairwise_cosines"]
    assert pairs["attempt100_vs_attempt134"]["mean_positions_1_4"] == 0
    assert pairs["consensus16_vs_attempt100"]["mean_positions_1_127"] == pytest.approx(2**-0.5)
    between = geometry["angular_between_parent_directions_descriptive"]
    assert between["count_positions_1_4"] == 4
    assert between["count_positions_1_127"] == 127
    assert set(geometry["per_corpus_full_vs_low_position_cosines"]) == set(runner.ORDER)
    assert all(row["mean_positions_1_127"] == 0
               for row in geometry["per_corpus_full_vs_low_position_cosines"].values())


def test_zero_or_nonfinite_view_and_hash_tampering_rejected():
    arrays = synthetic_views()
    broken = copy.deepcopy(arrays)
    broken[runner.VIEW_ORDER[0]][0] = 0
    with pytest.raises(ValueError, match="Zero or nonfinite"):
        runner.construct_consensus(broken)
    broken = copy.deepcopy(arrays)
    broken[runner.VIEW_ORDER[1]][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite contiguous"):
        runner.construct_consensus(broken)

    class Tensor:
        def __init__(self, array):
            self.array, self.dtype, self.device = array, "float32", SimpleNamespace(type="cpu")
            self.shape = array.shape

        def is_contiguous(self):
            return self.array.flags.c_contiguous

        def detach(self):
            return self

        def numpy(self):
            return self.array

    class Finite:
        def __init__(self, value):
            self.value = value

        def all(self):
            return self

        def item(self):
            return self.value

    fake_torch = SimpleNamespace(Tensor=Tensor, float32="float32",
                                 isfinite=lambda value: Finite(np.isfinite(value.array).all()))
    name = runner.VIEW_ORDER[0]
    artifact = {name: Tensor(arrays[name])}
    correct_hash = runner.raw_float32_hash(arrays[name])
    assert list(runner.validate_artifact(artifact, [name], {name: correct_hash}, fake_torch)) == [name]
    with pytest.raises(ValueError, match="raw hash changed"):
        runner.validate_artifact(artifact, [name], {name: "0" * 64}, fake_torch)
    with pytest.raises(ValueError, match="order/inventory"):
        runner.validate_artifact(artifact, ["another"], {"another": correct_hash}, fake_torch)


def test_barrier_and_exact_attempt100_oracle_convention():
    run_source = inspect.getsource(runner.run)
    assert run_source.index("validate_blind_sources(") < run_source.index("construct_consensus(")
    assert run_source.index("construct_consensus(") < run_source.index('candidate_path.open("xb")')
    assert run_source.index('candidate_path.open("xb")') < run_source.index('manifest_path.open("x"')
    assert run_source.index("validate_published(") < run_source.index("print(MARKER, flush=True)")
    assert run_source.index("print(MARKER, flush=True)") < run_source.index("evaluate_after_barrier(")
    blind_source = inspect.getsource(runner.validate_blind_sources)
    assert 'spec["post_barrier_inputs"]' not in blind_source
    assert "evaluation.json" not in blind_source and "result.json" not in blind_source
    assert 'spec["post_barrier_inputs"]["oracle_artifact_path"]' not in blind_source
    assert "load_and_validate_oracle" not in blind_source
    evaluation_source = inspect.getsource(runner.evaluate_after_barrier)
    assert "evaluator.prior.validate_oracle_provenance(" in evaluation_source
    assert "evaluator.prior.load_and_validate_oracle(" in evaluation_source
    assert "evaluator.position_cosines(" in evaluation_source
    assert "import_blind.old_evaluation_reports(" in evaluation_source
    assert "import_blind.signed_report(" in evaluation_source
    spec = runner.load_spec(SPEC)
    assert spec["barrier"]["marker"] == runner.MARKER
    assert spec["barrier"]["manifest_contains_historical_scores"] is False
    assert spec["evaluation"]["primary_positions"] == [1, 2, 3, 4]
    assert spec["evaluation"]["secondary_positions"] == list(range(1, 128))
    assert spec["evaluation"]["absolute_cosine"] is False
    assert spec["evaluation"]["flattened_historical_cosine"] is False
    assert "Attempt133" not in SOURCE.read_text()


def test_metricwise_best_parent_thresholds_and_win_counts():
    assert runner.classify(0.005, 0.005) == "clear_multiview_gain"
    assert runner.classify(0.001, 0.03) == "positive_multiview_gain"
    assert runner.classify(0.005, 0) == "no_multiview_gain"
    assert runner.classify(-0.01, 0.1) == "no_multiview_gain"
    with pytest.raises(ValueError):
        runner.classify(float("nan"), 0.1)
    def report(p, s, position_value):
        return {"positions_1_4_mean_cosine": p, "positions_1_127_mean_cosine": s,
                "all_128_position_cosines": [position_value] * 128}
    result = runner.compare_scores(report(.51, .61, .6), report(.50, .40, .3),
                                   report(.30, .60, .5))
    assert result["best_parent_primary"] == .50
    assert result["best_parent_secondary"] == .60
    assert result["classification"] == "clear_multiview_gain"
    assert result["positions_1_127_wins_vs_attempt100"] == 127
    assert result["positions_1_127_wins_vs_attempt134"] == 127


def test_evaluation_keeps_synthetic_candidate_report_distinct_from_both_parents(tmp_path, monkeypatch):
    """Exercise result assembly without loading models, artifacts, or the oracle."""
    old_value, low_value, candidate_value = 0.25, 0.5, 0.75
    old_frozen = {
        "candidate": "consensus_response_8",
        "positions": [{"position": i, "cosine_similarity": old_value} for i in range(128)],
        "positions_1_4_mean_cosine": old_value,
        "positions_1_127_mean_cosine": old_value,
    }
    low_frozen = {
        "candidate": "low_surprisal_consensus_8",
        "all_128_position_cosines": [low_value] * 128,
        "positions_1_4_mean_cosine": low_value,
        "positions_1_127_mean_cosine": low_value,
    }
    old_path, low_path = tmp_path / "old.json", tmp_path / "low.json"
    old_path.write_text(json.dumps({
        "candidates": [old_frozen],
        "provenance": {"construction_manifest_sha256": "old-manifest",
                       "construction_artifact_sha256": "old-artifact"}}))
    low_path.write_text(json.dumps({
        "new_signed_position_reports": {"low_surprisal_consensus_8": low_frozen},
        "construction_manifest_sha256": "low-manifest",
        "candidate_serialized_sha256": "low-artifact"}))

    oracle_manifest = {"oracle_adl_sha256": "oracle-artifact",
                       "raw_tensors_sha256": {"difference": "oracle-difference"}}
    prior = SimpleNamespace(
        load_json_object=lambda *_args: {},
        validate_oracle_provenance=lambda *_args: oracle_manifest,
        load_and_validate_oracle=lambda *_args: {"difference": object()},
    )
    evaluator = SimpleNamespace(
        FROZEN_SPEC={"oracle": {"provenance_inputs": {
            "attempt_spec": {"path": str(tmp_path / "oracle-spec.json"), "sha256": "pin"}}}},
        path_of=lambda path: Path(path), prior=prior,
        validate_oracle_manifest=lambda value: value,
        position_cosines=lambda tensor, _difference: [float(tensor[0, 0])] * 128,
    )

    def signed_report(name, tensor, difference, evaluator_module):
        values = evaluator_module.position_cosines(tensor, difference)
        return {"candidate": name, "position_0_cosine": values[0],
                "positions_1_4_individual_cosines": values[1:5],
                "positions_1_4_mean_cosine": values[1],
                "positions_1_127_mean_cosine": values[1],
                "all_128_position_cosines": values}

    blind = SimpleNamespace(
        signed_report=signed_report,
        old_evaluation_reports=lambda _result: {"consensus_response_8": {
            "positions_1_4_mean_cosine": old_value,
            "positions_1_127_mean_cosine": old_value}},
    )
    monkeypatch.setattr(runner, "require_hash", lambda *_args: None)
    monkeypatch.setattr(runner, "sha256_file", lambda *_args: "synthetic-hash")
    monkeypatch.setattr(runner, "import_pinned", lambda _path, _sha, name:
                        evaluator if name == "attempt135_attempt100_evaluator" else blind)
    spec = {
        "post_barrier_inputs": {
            "attempt100_evaluator_path": str(tmp_path / "evaluator.py"),
            "attempt100_evaluator_sha256": "pin",
            "attempt100_evaluation_path": str(old_path),
            "attempt100_evaluation_sha256": "pin",
            "attempt134_result_path": str(low_path),
            "attempt134_result_sha256": "pin",
            "attempt100_evaluation_spec_path": str(tmp_path / "eval-spec.json"),
            "attempt100_evaluation_spec_sha256": "pin",
            "oracle_manifest_path": str(tmp_path / "oracle-manifest.json"),
            "oracle_manifest_sha256": "pin",
            "oracle_artifact_path": str(tmp_path / "oracle.pt"),
            "oracle_artifact_sha256": "oracle-artifact",
            "oracle_difference_raw_sha256": "oracle-difference",
        },
        "source_artifacts": {
            "attempt100": {"construction_manifest_sha256": "old-manifest",
                           "artifact_serialized_sha256": "old-artifact"},
            "attempt134": {"construction_manifest_sha256": "low-manifest",
                           "artifact_serialized_sha256": "low-artifact",
                           "constructor_path": str(tmp_path / "blind.py"),
                           "constructor_sha256": "pin"},
        },
        "outputs": {"construction_manifest_path": str(tmp_path / "construction.json")},
    }
    sources = {"consensus_response_8": np.full((128, 2048), old_value, dtype=np.float32),
               "low_surprisal_consensus_8": np.full((128, 2048), low_value, dtype=np.float32)}
    candidate = np.full((128, 2048), candidate_value, dtype=np.float32)
    result = runner.evaluate_after_barrier(
        spec, candidate, sources,
        {"candidate": {"serialized_sha256": "frozen-candidate"}, "oracle_free_geometry": {}},
        SimpleNamespace(from_numpy=lambda value: value))
    reports = result["signed_position_reports"]
    assert reports[runner.CANDIDATE]["candidate"] == "consensus_response_16"
    assert reports[runner.CANDIDATE]["positions_1_4_mean_cosine"] == pytest.approx(candidate_value)
    assert reports[runner.CANDIDATE]["positions_1_127_mean_cosine"] == pytest.approx(candidate_value)
    assert reports["attempt100_consensus"]["positions_1_4_mean_cosine"] == old_value
    assert reports["attempt134_low_consensus"]["positions_1_4_mean_cosine"] == low_value
    assert result["comparison"]["primary_gain_vs_best_parent"] == pytest.approx(candidate_value - low_value)
    assert result["comparison"]["secondary_gain_vs_best_parent"] == pytest.approx(candidate_value - low_value)
    assert result["comparison"]["positions_1_127_wins_vs_attempt134"] == 127


def test_output_overwrite_refusal(tmp_path):
    spec = copy.deepcopy(runner.load_spec(SPEC))
    spec["outputs"] = {"candidate_path": str(tmp_path / "candidate.pt"),
                       "construction_manifest_path": str(tmp_path / "construction-manifest.json"),
                       "result_path": str(tmp_path / "result.json")}
    runner.require_outputs_absent(spec)
    for key in ("candidate_path", "construction_manifest_path", "result_path"):
        path = Path(spec["outputs"][key])
        path.write_bytes(b"existing")
        with pytest.raises(ValueError, match="already exists"):
            runner.require_outputs_absent(spec)
        path.unlink()
