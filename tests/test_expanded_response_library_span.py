"""Synthetic/static checks for Attempt137; never load a model or oracle artifact."""

from __future__ import annotations

import copy
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


runner = load_module(ROOT / "scripts/ablation/run_expanded_response_library_span.py", "attempt137_runner")
a136 = load_module(ROOT / "scripts/ablation/run_privileged_residual_disagreement_patchscope.py",
                   "attempt137_pinned_136")


def test_exact_29_primitive_inventory_and_committed_provenance():
    spec = runner.load_spec()
    assert spec["chronological_family_order"] == [100, 126, 129, 130, 131, 132, 134]
    assert spec["expected_primitive_count"] == 29
    assert [len(f["primitive_tensor_names"]) for f in spec["families"]] == [8, 3, 4, 2, 2, 2, 8]
    assert tuple(spec["families"][0]["primitive_tensor_names"]) == runner.PRIMITIVE_NAMES[100]
    assert tuple(spec["families"][-1]["primitive_tensor_names"]) == runner.PRIMITIVE_NAMES[134]
    assert len({entry["raw_sha256"] for entry in spec["primitive_inventory_order"]}) == 29
    assert runner.chronological_boundaries(spec) == [
        (100, 8), (126, 11), (129, 15), (130, 17), (131, 19), (132, 21), (134, 29)]
    for family in spec["families"]:
        assert runner.validate_blind_manifest(family)
        assert family["probe_rows"] == ([0, 10000] if family["attempt_number"] in (100, 134)
                                        else [0, 1024])
        assert family["artifact"]["path"].endswith(("responses.pt", "candidate.pt"))
        assert len(family["artifact"]["serialized_sha256"]) == 64
        assert set(family["primitive_tensor_names"]) <= set(family["artifact"]["tensor_order"])
    assert spec["oracle_after_basis_only"]["attempt136_result"]["sha256"] == \
        "acea0393c642b1844816ebdf591b42118c7e296d749e66a649a9aba565e120bc"
    assert spec["oracle_after_basis_only"]["oracle_artifact"]["difference_raw_sha256"] == \
        "d60789a2c06eb82dba2f2449d8665ec665d85e9f7e33fdea1f66ac79c81c159d"
    assert spec["no_new_candidate_constructed"] is True
    assert spec["privileged_diagnostic_only"] is True
    assert "candidate.pt" not in spec["output"]


def test_manifest_raw_hash_mismatch_is_rejected_before_oracle():
    family = copy.deepcopy(runner.load_spec()["families"][1])
    family["primitive_raw_sha256"]["response_low"] = "0" * 64
    with pytest.raises(ValueError, match="blind response manifest changed"):
        runner.validate_blind_manifest(family)


def test_per_position_cpu_float64_unit_normalization_and_rejection(monkeypatch):
    monkeypatch.setattr(runner, "SHAPE", (128, 3))
    values = np.zeros((2, 128, 3), dtype=np.float32)
    values[0, :, 0] = 3
    values[0, :, 1] = 4
    values[1, :, 2] = 7
    unit = runner.unit_normalize_positions(values)
    assert unit.dtype == np.float64
    np.testing.assert_allclose(unit[0, :, 0], 0.6)
    np.testing.assert_allclose(unit[0, :, 1], 0.8)
    np.testing.assert_allclose(np.linalg.norm(unit, axis=2), 1, atol=1e-15)
    assert unit[:, 1:5, :].reshape(2, -1).shape == (2, 12)
    values[0, 4] = 0
    with pytest.raises(ValueError, match="Zero/nonfinite"):
        runner.unit_normalize_positions(values)
    values[0, 4, 0] = np.nan
    with pytest.raises(ValueError, match="Malformed"):
        runner.unit_normalize_positions(values)


def test_fixed_chronological_economy_svds_and_known_residual_projection(monkeypatch):
    monkeypatch.setattr(runner, "SHAPE", (128, 3))
    values = np.zeros((4, 128, 3), dtype=np.float32)
    values[0, :, 0], values[1, :, 0] = 1, -1
    values[2, :, 1], values[3, :, 1] = 1, -1
    unit = runner.unit_normalize_positions(values)
    plan = {"families": [
        {"attempt_number": 100, "primitive_tensor_names": ["a", "b"]},
        {"attempt_number": 126, "primitive_tensor_names": ["c", "d"]}],
        "expected_primitive_count": 4,
        "primitive_inventory_order": [
            {"attempt_number": 100, "tensor_name": "a"},
            {"attempt_number": 100, "tensor_name": "b"},
            {"attempt_number": 126, "tensor_name": "c"},
            {"attempt_number": 126, "tensor_name": "d"}]}
    bases = runner.build_blind_bases(unit, plan, a136)
    assert bases["positions_1_4"][100]["matrix"].shape == (2, 12)
    assert bases["positions_1_127"][126]["matrix"].shape == (4, 381)
    assert bases["positions_1_4"][100]["rank"] == 1
    assert bases["positions_1_4"][126]["rank"] == 2
    for span in bases.values():
        for basis in span.values():
            assert basis["rank_tolerance"] == pytest.approx(
                max(basis["matrix"].shape) * np.finfo(np.float64).eps * basis["singular_values"][0])
    r = np.tile([3., 4., 0.], 4)
    first = runner.summarize_basis_projection(bases["positions_1_4"][100], r, a136)
    second = runner.summarize_basis_projection(bases["positions_1_4"][126], r, a136)
    assert first["residual_squared_norm_fraction_captured"] == pytest.approx(9 / 25)
    assert second["residual_squared_norm_fraction_captured"] == pytest.approx(1)
    assert second["residual_norm_fraction_captured"] == pytest.approx(1)
    assert sum(second["singular_value_variance_fractions"]) == pytest.approx(1)
    assert sum(pc["residual_squared_fraction_carried"] for pc in
               second["pc_residual_diagnostics"]) == pytest.approx(1)
    assert "np.linalg.svd(d, full_matrices=False)" in inspect.getsource(a136.blind_svd)


def test_attempt136_residual_reproduction_and_span_baseline_comparison():
    c = np.zeros((128, 3), dtype=np.float32)
    o = np.zeros_like(c)
    c[:, 0], o[:, 0], o[:, 1] = 2, 4, 3
    _, residual, report = a136.positionwise_residual(o, c)
    prior = {"observed_quantities": {
        "attempt134_positionwise_oracle_residual": report,
        "blind_disagreement_residual_projections": {
            space: {name: {"residual_squared_fraction_in_span": value}
                    for name in runner.RANGES}
            for space, value in (("full_8", .10), ("low_8", .20), ("joint_16", .30))}}}
    runner.validate_attempt136_residual(prior, report)
    assert residual[0, 1] == 3
    changed = copy.deepcopy(report)
    changed["all_128_positions"][1]["alpha"] += .1
    with pytest.raises(ValueError, match="did not reproduce"):
        runner.validate_attempt136_residual(prior, changed)
    full = {"residual_squared_norm_fraction_captured": .50}
    comparison = runner.compare_attempt136(full, prior, "positions_1_127")
    assert comparison["joint_16"]["expanded_library_minus_attempt136_fraction"] == pytest.approx(.20)
    with pytest.raises(ValueError, match="unexpectedly loses"):
        runner.compare_attempt136({"residual_squared_norm_fraction_captured": .25}, prior,
                                   "positions_1_127")


def test_oracle_is_opened_after_all_blind_bases_and_result_refuses_overwrite(tmp_path):
    source = inspect.getsource(runner.run)
    assert source.index("load_blind_library(spec, torch)") < source.index("build_blind_bases(unit, spec, a136)")
    assert source.index("build_blind_bases(unit, spec, a136)") < \
        source.index("load_oracle_residual_after_bases(spec, consensus, torch, a136)")
    assert "torch.save(" not in inspect.getsource(runner)
    assert "result.json" == Path(runner.load_spec()["output"]).name
    output = tmp_path / "result.json"
    runner.require_output_absent(output)
    output.write_text("existing")
    with pytest.raises(ValueError, match="already exists"):
        runner.require_output_absent(output)
