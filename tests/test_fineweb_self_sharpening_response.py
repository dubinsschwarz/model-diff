"""Attempt138 synthetic/static tests; no real model, corpus, or oracle load."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ablation/run_fineweb_self_sharpening_response.py"


def load_module(path: Path, name: str):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


runner = load_module(SOURCE, "attempt138_runner")


def test_pinned_attempt132_selection_and_frozen_inputs():
    spec = runner.load_spec()
    assert spec["expected_prior_head_commit"] == "ee486eb90de06365a4db82ac57f5f7d4aa86e952"
    selection = spec["fixed_selection"]
    assert selection["group"] == "super_low"
    assert selection["rank_range"] == [0, 1024]
    assert selection["pair_count"] == 1024
    assert selection["ordered_pair_raw_int64_sha256"] == \
        "8afe4d2072a49db079f83e7971c935e1f44e51e292a6a7f3b2959c3e6b6c8e22"
    parent = json.loads((ROOT / spec["blind_sources"]["attempt132_construction_manifest"]["path"]).read_text())
    group = parent["combined_ranking"]["groups"]["super_low"]
    pairs = np.ascontiguousarray(group["ordered_pairs"], dtype="<i8")
    assert pairs.shape == (1024, 2)
    assert hashlib.sha256(pairs.tobytes(order="C")).hexdigest() == selection["ordered_pair_raw_int64_sha256"]
    assert selection["source_shapes"] == {"old": [4096, 128], "fresh": [4096, 128], "new": [8192, 128]}
    for pin in spec["blind_sources"].values():
        assert runner.sha256_file(runner.path_of(pin["path"])) == pin["sha256"]
    assert spec["final_checkpoint"]["files"][3]["sha256"] == \
        "f6989dd3445cd339041ab654a6da463e0a80b2d52dd2c01f67f683b1c30ea10f"


def test_exact_self_sharpening_objective_sign_and_synthetic_logit_gradient():
    spec = runner.load_spec()
    old024 = json.loads((ROOT / spec["blind_sources"]["attempt024_spec"]["path"]).read_text())
    assert old024["infinitesimal_signal"]["dot_p"] == "p*dot_logp"
    assert old024["infinitesimal_signal"]["dot_logp"] == "logp-mean_logp_under_p"
    objective = spec["objective"]
    assert objective["formula"] == "mean_over_1024x127_sum_vocab_p_theta_log_p_theta"
    assert objective["gradient_direction"] == "positive_gradient"
    assert objective["label_tokens_used"] is False
    assert objective["total_prediction_contexts"] == 1024 * 127
    assert objective["gradient_batches"] == 128
    source = inspect.getsource(runner.negative_entropy_loss_sum)
    assert "logits[:, :127, :]" in source
    assert "torch.log_softmax" in source
    assert "student_logp.exp()" in source
    assert "(student_p * student_logp).sum()" in source
    assert "cross_entropy" not in source and "tokens[:, 1:]" not in source
    assert "(loss_sum / 130048).backward()" in inspect.getsource(runner.accumulate_self_sharpen_gradient)

    logits = np.array([2.0, .2, -1.3], dtype=np.float64)
    def loss(z):
        exp = np.exp(z - z.max())
        p = exp / exp.sum()
        return float(np.dot(p, np.log(p)))
    p = np.exp(logits - logits.max()); p /= p.sum()
    analytic = runner.negative_entropy_logit_gradient(p)
    finite_difference = np.array([(loss(logits + 1e-6 * np.eye(3)[i]) -
                                   loss(logits - 1e-6 * np.eye(3)[i])) / (2e-6)
                                  for i in range(3)])
    np.testing.assert_allclose(analytic, finite_difference, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(analytic.sum(), 0, atol=1e-15)
    assert analytic[0] > 0  # Positive gradient sharpens the highest-logit token.


def test_global_positive_tangent_and_exact_eligible_support():
    spec = runner.load_spec()
    assert spec["eligible_tensors"]["transformer_block_indices"] == list(range(14))
    assert spec["eligible_tensors"]["expected_matrix_count"] == 98
    assert spec["eligible_tensors"]["recursive_module_type"] == "torch.nn.Linear"
    assert spec["eligible_tensors"]["parameter"] == "weight"
    assert spec["eligible_tensors"]["freeze_every_other_parameter"] is True
    assert spec["tangent"]["target_relative_frobenius"] == .00125
    assert spec["tangent"]["direction"] == "delta = +alpha * g"
    assert spec["tangent"]["scaling_arithmetic_dtype"] == "cpu_float64"
    source = inspect.getsource(runner.run)
    assert "a14.discover_eligible_linear_weights(model, torch)" in source
    assert "a14.freeze_other_parameters(model, eligible)" in source
    assert "a14.global_tangent_scale(eligible, 0.00125, torch)" in source
    assert "a14.prepare_tangents(eligible, scale, torch)" in source
    assert "a14.verify_model_unchanged(model, before, torch)" in source


def test_one_jvp_sweep_accumulates_exact_first1024_and_full10000():
    plan = runner.jvp_batch_plan()
    assert len(plan) == 157
    assert sum(size for _, size, _ in plan) == 10000
    assert sum(first for _, _, first in plan) == 1024
    assert sum(first > 0 for _, _, first in plan) == 16
    assert plan[-1] == (9984, 16, 0)
    first_sum = np.zeros((2, 3), dtype=np.float64)
    full_sum = np.zeros((2, 3), dtype=np.float64)
    runner.add_dual_response_sums(first_sum, full_sum,
                                  np.ones((3, 2, 3), dtype=np.float32), 2)
    runner.add_dual_response_sums(first_sum, full_sum,
                                  np.full((2, 2, 3), 3, dtype=np.float32), 0)
    np.testing.assert_array_equal(first_sum, np.full((2, 3), 2))
    np.testing.assert_array_equal(full_sum, np.full((2, 3), 9))
    with pytest.raises(ValueError, match="Malformed"):
        runner.add_dual_response_sums(first_sum, full_sum,
                                      np.ones((3, 2, 3)), 4)
    jvp_source = inspect.getsource(runner.dual_probe_jvp)
    assert jvp_source.count("a14.run_readout_jvp(") == 1
    assert "add_dual_response_sums(first_sum, full_sum, values, first_count)" in jvp_source
    assert "first_sum / 1024" in jvp_source
    assert "full_sum / 10000" in jvp_source
    spec = runner.load_spec()
    assert spec["probe"]["jvp_batch_size"] == 64
    assert spec["probe"]["sample_count"] == 10000
    assert spec["jvp"]["strict_forward_ad"] is True
    assert spec["readout"]["hidden_state_index"] == 14


def test_barrier_order_and_postfreeze_evaluation_conventions():
    spec = runner.load_spec()
    source = inspect.getsource(runner.run)
    assert source.index("validate_blind_inputs(") < source.index("accumulate_self_sharpen_gradient(")
    assert source.index("accumulate_self_sharpen_gradient(") < source.index("dual_probe_jvp(")
    assert source.index("dual_probe_jvp(") < source.index("atomic_torch_publish(")
    assert source.index("atomic_torch_publish(") < source.index("validate_published(")
    assert source.index("validate_published(") < source.index("log(MARKER)")
    assert source.index("log(MARKER)") < source.index("evaluate_after_barrier(")
    assert "privileged_sources_after_barrier" not in inspect.getsource(runner.validate_blind_inputs)
    assert "attempt132_result" not in inspect.getsource(runner.validate_blind_inputs)
    assert spec["candidate"]["tensor_order"] == list(runner.NAMES)
    assert spec["evaluation"]["primary_positions"] == [1, 2, 3, 4]
    assert spec["evaluation"]["secondary_positions"] == list(range(1, 128))
    assert spec["evaluation"]["signed_positionwise_cosine"] is True
    assert spec["evaluation"]["historical_absolute_cosine"] is False
    assert spec["evaluation"]["historical_flattened_cosine"] is False
    assert spec["evaluation"]["patchscope"]["prompts"] == [
        "The topic is", "This is about", "The central idea is", "In one word:",
        "A concise label:", "The concept:", "This relates to", "The underlying theme:"]
    assert spec["evaluation"]["patchscope"]["semantic_substrings"] == \
        ["cake", "bake", "cook", "craft", "precision"]
    assert spec["evaluation"]["patchscope"]["layer_index"] == 13
    assert "a133.next_logits(model, ids, patch_index," in inspect.getsource(runner.patchscope)
    assert "a133.norm_match_blind(" in inspect.getsource(runner.patchscope)
    assert "a133.greedy_completion(" in inspect.getsource(runner.patchscope)


def test_overwrite_refusal_and_blind_cosine_sign(tmp_path):
    spec = runner.load_spec()
    original = spec["outputs"]
    spec["outputs"] = {"candidate_path": str(tmp_path / "candidate.pt"),
                       "construction_manifest_path": str(tmp_path / "construction-manifest.json"),
                       "result_path": str(tmp_path / "result.json"), "overwrite": False}
    runner.output_paths(spec)
    (tmp_path / "candidate.pt").write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        runner.output_paths(spec)
    spec["outputs"] = original
    a = np.array([[-1., 0.], [0., 1.]])
    b = -a
    diagnostic = runner.flat_cosine_and_norms(a, b)
    assert diagnostic["flattened_signed_cosine_diagnostic"] == pytest.approx(-1)
    assert diagnostic["left_to_right_norm_ratio"] == pytest.approx(1)


def test_signed_positionwise_residual_comparison_is_not_absolute():
    a136 = load_module(ROOT / "scripts/ablation/run_privileged_residual_disagreement_patchscope.py",
                       "attempt138_synthetic_136")
    left = np.ones((128, 2048), dtype=np.float32)
    right = -left
    metrics = runner.position_range_cosines(left, right, a136)
    assert metrics["position_0"] == pytest.approx(-1)
    assert metrics["mean_positions_1_4"] == pytest.approx(-1)
    assert metrics["mean_positions_1_127"] == pytest.approx(-1)
    assert metrics["raw_concatenation_positions_1_4"] == pytest.approx(-1)
    assert len(metrics["all_128_positions"]) == 128
