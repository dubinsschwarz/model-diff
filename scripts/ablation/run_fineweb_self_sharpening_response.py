#!/usr/bin/env python3
"""Attempt138: blind negative-entropy gradient JVP, then fixed privileged readout."""

from __future__ import annotations

import gc
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = "138_fineweb_self_sharpening_response"
SPEC_PATH = ROOT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "92b2d0c14685702af7b8fd70a48f947550ea1dca1bf22faf249b35de92eddb61"
MARKER = "SELF_SHARPENING_RESPONSE_FROZEN"
NAMES = ("response_self_sharpen_first1024", "response_self_sharpen_full10000")
SHAPE = (128, 2048)
PREFIX = 1024
FULL = 10000
BATCH = 64
PATCH_NAMES = ("oracle", "self_sharpen", "attempt134", "residual")


def log(message: str) -> None:
    print("[138] " + message, flush=True)


def path_of(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"Frozen SHA256 mismatch: {path}")


def import_pinned(pin: dict[str, str], name: str) -> Any:
    path = path_of(pin["path"])
    require_hash(path, pin["sha256"])
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def output_paths(spec: dict[str, Any]) -> dict[str, Path]:
    paths = {key: path_of(spec["outputs"][key]) for key in
             ("candidate_path", "construction_manifest_path", "result_path")}
    if len(set(paths.values())) != 3 or any(path.exists() or path.is_symlink()
                                           for path in paths.values()):
        raise ValueError("Attempt138 output already exists or paths overlap")
    return paths


def load_spec() -> dict[str, Any]:
    require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text())
    loss = spec["objective"]
    if (spec["attempt_id"] != ATTEMPT
            or spec["fixed_selection"]["source_attempt"] != 132
            or spec["fixed_selection"]["group"] != "super_low"
            or spec["fixed_selection"]["rank_range"] != [0, 1024]
            or spec["fixed_selection"]["pair_count"] != 1024
            or spec["fixed_selection"]["ordered_pair_raw_int64_sha256"] !=
            "8afe4d2072a49db079f83e7971c935e1f44e51e292a6a7f3b2959c3e6b6c8e22"
            or loss["name"] != "negative_entropy_self_sharpening"
            or loss["gradient_direction"] != "positive_gradient"
            or loss["label_tokens_used"] is not False
            or loss["sample_count"] != 1024 or loss["sequence_length"] != 128
            or loss["prediction_positions_per_row"] != 127
            or loss["total_prediction_contexts"] != 130048
            or loss["batch_size"] != 8 or loss["gradient_batches"] != 128
            or loss["probability_computation"] !=
            "torch.log_softmax(FP32_logits,dim=-1); p=exp(logp)"
            or any(loss[key] is not False for key in
                   ("autocast", "optimizer", "clipping", "weight_decay"))
            or spec["eligible_tensors"]["expected_matrix_count"] != 98
            or spec["eligible_tensors"]["transformer_block_indices"] != list(range(14))
            or spec["tangent"]["direction"] != "delta = +alpha * g"
            or spec["tangent"]["target_relative_frobenius"] != 0.00125
            or spec["probe"]["first_response_rows"] != [0, PREFIX]
            or spec["probe"]["full_response_rows"] != [0, FULL]
            or spec["probe"]["jvp_batch_size"] != BATCH
            or spec["probe"]["jvp_batch_count"] != 157
            or spec["probe"]["one_sweep"] is not True
            or spec["readout"]["block_index"] != 13
            or spec["readout"]["hidden_state_index"] != 14
            or spec["jvp"]["strict_forward_ad"] is not True
            or spec["jvp"]["fallback"] is not False
            or spec["candidate"]["tensor_order"] != list(NAMES)
            or spec["barrier"]["marker"] != MARKER
            or spec["barrier"]["manifest_contains_historical_scores"] is not False
            or spec["workload"]["jvp_sweeps"] != 1
            or spec["workload"]["gradient_backward_batches"] != 128
            or spec["workload"]["jvp_batches"] != 157):
        raise ValueError("Attempt138 fixed scientific plan changed")
    return spec


def jvp_batch_plan(sample_count: int = FULL, batch_size: int = BATCH,
                   prefix: int = PREFIX) -> list[tuple[int, int, int]]:
    if sample_count != FULL or batch_size != BATCH or prefix != PREFIX:
        raise ValueError("Attempt138 frozen JVP population/batch changed")
    return [(start, min(batch_size, sample_count - start),
             max(0, min(prefix - start, min(batch_size, sample_count - start))))
            for start in range(0, sample_count, batch_size)]


def add_dual_response_sums(first_sum: np.ndarray, full_sum: np.ndarray,
                           batch: np.ndarray, first_count: int) -> None:
    values = np.asarray(batch, dtype=np.float64)
    if (values.ndim != 3 or values.shape[1:] != first_sum.shape
            or first_sum.shape != full_sum.shape or not np.isfinite(values).all()
            or not 0 <= first_count <= len(values)):
        raise ValueError("Malformed dual JVP response accumulation")
    full_sum += values.sum(axis=0, dtype=np.float64)
    if first_count:
        first_sum += values[:first_count].sum(axis=0, dtype=np.float64)


def negative_entropy_logit_gradient(probabilities: np.ndarray) -> np.ndarray:
    """Synthetic identity oracle for d(sum p log p)/d(logits)."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(p <= 0) or not np.isclose(p.sum(), 1):
        raise ValueError("Synthetic categorical probabilities malformed")
    logp = np.log(p)
    return p * (logp - float(np.dot(p, logp)))


def negative_entropy_loss_sum(logits: Any, torch: Any) -> Any:
    if (logits.ndim != 3 or logits.shape[1] != 128 or logits.dtype != torch.float32
            or not bool(torch.isfinite(logits).all())):
        raise ValueError("Malformed self-sharpening logits")
    # Position 127 is excluded. No token labels enter this objective.
    student_logp = torch.log_softmax(logits[:, :127, :], dim=-1)
    student_p = student_logp.exp()
    return (student_p * student_logp).sum()


def validate_blind_inputs(spec: dict[str, Any], a14: Any, a132: Any,
                          torch: Any) -> tuple[Any, Any, dict[str, Any]]:
    for pin in spec["blind_sources"].values():
        require_hash(path_of(pin["path"]), pin["sha256"])
    parent_spec = json.loads(path_of(spec["blind_sources"]["attempt132_spec"]["path"]).read_text())
    parent_manifest = json.loads(path_of(
        spec["blind_sources"]["attempt132_construction_manifest"]["path"]).read_text())
    parent134 = json.loads(path_of(spec["blind_sources"]["attempt134_spec"]["path"]).read_text())
    selection = parent_manifest["combined_ranking"]["groups"]["super_low"]
    pair = spec["fixed_selection"]
    old_source = pair["source_token_artifacts"]["old"]
    fresh_source = pair["source_token_artifacts"]["fresh"]
    new_source = pair["source_token_artifacts"]["new"]
    if (parent_manifest["spec_sha256"] != spec["blind_sources"]["attempt132_spec"]["sha256"]
            or parent_manifest["constructor_sha256"] != spec["blind_sources"]["attempt132_source"]["sha256"]
            or parent_manifest["barrier"]["manifest_contains_historical_scores"] is not False
            or parent_spec["combined_selection"]["candidate_rank_ranges"]["super_low"] != [0, 1024]
            or selection["rank_range"] != [0, 1024]
            or selection["ordered_pair_raw_int64_sha256"] != pair["ordered_pair_raw_int64_sha256"]
            or len(selection["ordered_pairs"]) != 1024
            or sum(selection["source_counts"].values()) != 1024
            or any(parent_spec["frozen_slices"][name][key] != source[key]
                   for name, source in (("old", old_source), ("fresh", fresh_source))
                   for key in ("tokens_path", "tokens_serialized_sha256", "tokens_raw_sha256"))
            or parent_spec["new_corpus"]["tokens_path"] != new_source["tokens_path"]
            or parent_manifest["new_corpus_serialized_sha256"] != new_source["tokens_serialized_sha256"]
            or parent_manifest["new_corpus_raw_tensor_sha256"] != new_source["tokens_raw_sha256"]
            or parent_spec["final_checkpoint"] != spec["final_checkpoint"]
            or parent134["final_checkpoint"] != spec["final_checkpoint"]
            or parent134["eligible_tensors"] != spec["eligible_tensors"]
            or parent134["tangent"] != spec["tangent"]
            or parent134["readout"] != spec["readout"]
            or parent134["jvp"] != spec["jvp"]
            or parent134["probe"]["serialized_sha256"] != spec["probe"]["serialized_sha256"]
            or parent134["probe"]["raw_tensor_sha256"] != spec["probe"]["raw_tensor_sha256"]):
        raise ValueError("Frozen Attempt132 selection or Attempt134 JVP provenance changed")
    pairs = np.ascontiguousarray(selection["ordered_pairs"], dtype="<i8")
    if (pairs.shape != (1024, 2) or np.any(pairs[:, 0] < 0)
            or np.any(pairs[:, 0] > 2) or
            hashlib.sha256(pairs.tobytes(order="C")).hexdigest() !=
            pair["ordered_pair_raw_int64_sha256"]):
        raise ValueError("Attempt132 SUPER_LOW ordered row pairs changed")
    source_tokens = {}
    for label, source in pair["source_token_artifacts"].items():
        path = path_of(source["tokens_path"])
        require_hash(path, source["tokens_serialized_sha256"])
        shape = pair["source_shapes"][label]
        source_tokens[label] = a14.load_corpus_tokens(
            path, source["tokens_raw_sha256"], shape[0], shape[1], torch)
    selected = a132.selected_tokens(source_tokens, selection["ordered_pairs"], torch)
    if tuple(selected.shape) != (1024, 128):
        raise ValueError("Attempt132 frozen SUPER_LOW selection malformed")
    probe = a14.load_probe(path_of(spec["probe"]["path"]), spec["probe"], torch)
    if a14.checkpoint_file_records(path_of(spec["final_checkpoint"]["directory"])) != \
            spec["final_checkpoint"]["files"]:
        raise ValueError("Canonical final checkpoint inventory changed")
    return selected, probe, selection


def accumulate_self_sharpen_gradient(model: Any, tokens: Any, eligible: list[Any],
                                     spec: dict[str, Any], torch: Any) -> float:
    if tuple(tokens.shape) != (1024, 128) or len(eligible) != 98:
        raise ValueError("Expected 1024 rows and 98 eligible matrices")
    model.eval()
    model.zero_grad(set_to_none=True)
    selected_ids = {id(module.weight) for _, module in eligible}
    device = eligible[0][1].weight.device
    losses = []
    with torch.enable_grad(), torch.autocast(device_type=device.type, enabled=False):
        for index, start in enumerate(range(0, 1024, 8), 1):
            batch = tokens[start:start + 8].to(device)
            logits = model(input_ids=batch, use_cache=False).logits
            loss_sum = negative_entropy_loss_sum(logits, torch)
            if not bool(torch.isfinite(loss_sum)):
                raise ValueError("Nonfinite self-sharpening objective")
            (loss_sum / 130048).backward()
            losses.append(float(loss_sum.detach().to(device="cpu", dtype=torch.float64)))
            if any(parameter.grad is not None for parameter in model.parameters()
                   if id(parameter) not in selected_ids):
                raise ValueError("Forbidden parameter received self-sharpening gradient")
            if index % 32 == 0:
                log(f"self-sharpening gradient batch {index}/128")
    for _, module in eligible:
        if (module.weight.grad is None or module.weight.grad.dtype != torch.float32
                or not bool(torch.isfinite(module.weight.grad).all())):
            raise ValueError("Missing/nonfinite eligible self-sharpening gradient")
    mean = math.fsum(losses) / 130048
    if not math.isfinite(mean):
        raise ValueError("Nonfinite mean self-sharpening objective")
    return mean


def dual_probe_jvp(model: Any, probe: Any, tangents: dict[str, Any],
                   spec: dict[str, Any], a14: Any, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if tuple(probe.shape) != (10000, 128):
        raise ValueError("Frozen full probe shape changed")
    first_sum = np.zeros(SHAPE, dtype=np.float64)
    full_sum = np.zeros(SHAPE, dtype=np.float64)
    plan = jvp_batch_plan()
    if len(plan) != 157 or sum(count for _, count, _ in plan) != 10000 or \
            sum(count for _, _, count in plan) != 1024 or plan[-1][1] != 16:
        raise ValueError("Frozen one-sweep JVP batching changed")
    discrepancy = None
    device = next(model.parameters()).device
    for batch_index, (start, count, first_count) in enumerate(plan, 1):
        batch = probe[start:start + count].to(device)
        ordinary = a14.ordinary_hook_readout(model, batch, spec, torch) if start == 0 else None
        primal, response = a14.run_readout_jvp(model, batch, tangents, spec, torch)
        if ordinary is not None:
            discrepancy = a14.compare_primal(primal, ordinary, spec, torch)
        values = response.detach().to(device="cpu", dtype=torch.float64).numpy()
        add_dual_response_sums(first_sum, full_sum, values, first_count)
        if batch_index % 16 == 0 or batch_index == 157:
            log(f"single JVP sweep batch {batch_index}/157")
        del batch, ordinary, primal, response, values
    responses = {
        NAMES[0]: torch.from_numpy(np.ascontiguousarray((first_sum / 1024).astype(np.float32))),
        NAMES[1]: torch.from_numpy(np.ascontiguousarray((full_sum / 10000).astype(np.float32))),
    }
    if any(tuple(tensor.shape) != SHAPE or not bool(torch.isfinite(tensor).all())
           for tensor in responses.values()):
        raise ValueError("Malformed first1024/full10000 self-sharpening response")
    return responses, {"jvp_batch_count": len(plan), "first1024_accumulated_rows": 1024,
                       "full10000_accumulated_rows": 10000,
                       "last_short_batch_size": 16, "readout_hook_verified": True,
                       "first_batch_primal_max_abs_difference": discrepancy}


def validate_published(paths: dict[str, Path], manifest: dict[str, Any],
                       spec: dict[str, Any], a14: Any, torch: Any) -> dict[str, Any]:
    if (sha256_file(paths["candidate_path"]) != manifest["candidate"]["serialized_sha256"]
            or json.loads(paths["construction_manifest_path"].read_text()) != manifest
            or manifest["spec_sha256"] != SPEC_SHA256
            or manifest["barrier"] != spec["barrier"]
            or manifest["candidate"]["path"] != str(paths["candidate_path"])
            or manifest["candidate"]["tensor_order"] != list(NAMES)
            or list(manifest["responses"]) != list(NAMES)
            or any(key in manifest for key in
                   ("oracle_score", "historical_score", "historical_target"))):
        raise ValueError("Published blind self-sharpening construction changed")
    artifact = torch.load(paths["candidate_path"], map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != list(NAMES):
        raise ValueError("Published self-sharpening response inventory changed")
    for name in NAMES:
        tensor = artifact[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.device.type != "cpu" or not tensor.is_contiguous()
                or tuple(tensor.shape) != SHAPE or not bool(torch.isfinite(tensor).all())
                or a14.sha256_raw_float32_tensor(tensor, torch) !=
                manifest["responses"][name]["raw_sha256"]):
            raise ValueError("Published self-sharpening response tensor changed")
    return artifact


def flat_cosine_and_norms(left: Any, right: Any) -> dict[str, float]:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Malformed response comparison vectors")
    an, bn = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if an <= 0 or bn <= 0:
        raise ValueError("Zero response norm")
    return {"flattened_signed_cosine_diagnostic": float(np.dot(a, b) / (an * bn)),
            "left_norm": an, "right_norm": bn, "left_to_right_norm_ratio": an / bn}


def position_range_cosines(left: Any, right: Any, a136: Any) -> dict[str, Any]:
    x, y = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if x.shape != SHAPE or y.shape != SHAPE:
        raise ValueError("Malformed per-position comparison vectors")
    per_position = [a136.nullable_cosine(x[pos], y[pos]) for pos in range(128)]
    if any(value is None for value in per_position):
        raise ValueError("Undefined per-position response cosine")
    return {"position_0": per_position[0], "mean_positions_1_4": math.fsum(per_position[1:5]) / 4,
            "mean_positions_1_127": math.fsum(per_position[1:]) / 127,
            "raw_concatenation_positions_1_4": a136.nullable_cosine(x[1:5], y[1:5]),
            "raw_concatenation_positions_1_127": a136.nullable_cosine(x[1:], y[1:]),
            "all_128_positions": per_position}


def direct_lens(vectors: dict[str, Any], final_norm: Any, lm_head: Any, tokenizer: Any,
                a133: Any, reader: Any, a136: Any, historical: dict[str, Any],
                provenance: dict[str, Any], torch: Any) -> dict[str, Any]:
    tokens, arrays = {}, {}
    with torch.inference_mode():
        for name in PATCH_NAMES:
            tokens[name], arrays[name] = {}, {}
            for pos in range(5):
                distribution = a133.lens_distribution(vectors[name][pos],
                                                      final_norm, lm_head, torch)
                positive = reader.top_token_records(distribution["positive_probs"], tokenizer, 20, torch)
                negative = reader.top_token_records(distribution["negative_probs"], tokenizer, 20, torch)
                tokens[name][str(pos)] = {"top20_positive": positive,
                                          "top20_negative": negative,
                                          "semantic_hits": a133.semantic_hits(positive, negative)}
                arrays[name][str(pos)] = {key: value.detach().to("cpu").numpy()
                                          for key, value in distribution.items()}
    a133.validate_historical_oracle_lens(
        {"tokens": {"oracle_difference": tokens["oracle"]}}, historical, provenance)
    comparisons = {reference: {str(pos): a133.lens_metrics(
        arrays["self_sharpen"][str(pos)], arrays[reference][str(pos)])
        for pos in range(5)} for reference in ("oracle", "attempt134", "residual")}
    means = {reference: a133.mean_dict([comparisons[reference][str(pos)]
                                        for pos in range(1, 5)], tuple(comparisons[reference]["1"]))
             for reference in comparisons}
    return {"tokens": tokens, "self_sharpen_similarity": comparisons,
            "self_sharpen_mean_positions_1_4": means,
            "fixed_semantic_substrings": list(a133.SEMANTIC_TERMS)}


def patchscope(vectors: dict[str, Any], model: Any, tokenizer: Any,
               a133: Any, a136: Any, torch: Any) -> dict[str, Any]:
    device = next(model.parameters()).device
    conditions = {mode: {name: {str(pos): {} for pos in range(5)} for name in PATCH_NAMES}
                  for mode in ("directional_oracle_norm_matched", "native_amplitude")}
    completions = {str(pos): {} for pos in range(1, 5)}
    for mode in conditions:
        for prompt_index, prompt in enumerate(a133.PROMPTS, 1):
            log(f"Patchscope {mode} prompt {prompt_index}/8")
            ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
                raise ValueError("Malformed fixed Patchscope prompt")
            patch_index = ids.shape[1] - 1
            baseline = a133.next_logits(model, ids, None, None, +1, torch).numpy()
            baseline_probs = torch.softmax(torch.from_numpy(baseline), dim=-1).numpy()
            for pos in range(5):
                patch_vectors = {}
                for name in PATCH_NAMES:
                    vector = vectors[name][pos]
                    if mode == "directional_oracle_norm_matched" and name != "oracle":
                        vector = torch.from_numpy(a133.norm_match_blind(
                            vector.numpy(), vectors["oracle"][pos].numpy()))
                    patch_vectors[name] = vector.to(device)
                deltas, probs = {}, {}
                for name in PATCH_NAMES:
                    conditions[mode][name][str(pos)][prompt] = {}
                    for sign, label in ((+1, "plus"), (-1, "minus")):
                        patched = a133.next_logits(model, ids, patch_index,
                                                   patch_vectors[name], sign, torch).numpy()
                        probabilities = torch.softmax(torch.from_numpy(patched), dim=-1).numpy()
                        delta = patched.astype(np.float64) - baseline.astype(np.float64)
                        record = a136.patch_record(delta, patched, probabilities,
                                                   baseline, baseline_probs, tokenizer, a133)
                        conditions[mode][name][str(pos)][prompt][label] = record
                        deltas[name, label], probs[name, label] = delta, probabilities
                for label in ("plus", "minus"):
                    record = conditions[mode]["self_sharpen"][str(pos)][prompt][label]
                    for reference in ("oracle", "attempt134", "residual"):
                        record["delta_logit_cosine_to_" + reference] = a136.nullable_cosine(
                            deltas["self_sharpen", label], deltas[reference, label])
                    if record["delta_logit_cosine_to_oracle"] is not None:
                        record["attempt133_oracle_delta_metrics"] = a133.patch_metrics(
                            deltas["self_sharpen", label], deltas["oracle", label],
                            probs["self_sharpen", label], baseline_probs)
                    else:
                        record["attempt133_oracle_delta_metrics"] = None
                if mode == "directional_oracle_norm_matched" and pos in range(1, 5):
                    completions[str(pos)][prompt] = a133.greedy_completion(
                        model, ids, patch_vectors["self_sharpen"], tokenizer, torch, 6)
    summaries = {}
    for mode, groups in conditions.items():
        summaries[mode] = {}
        for label in ("plus", "minus"):
            summaries[mode][label] = {}
            for reference in ("oracle", "attempt134", "residual"):
                per_position = {str(pos): [groups["self_sharpen"][str(pos)][prompt][label]
                                           ["delta_logit_cosine_to_" + reference]
                                           for prompt in a133.PROMPTS] for pos in range(5)}
                def mean(values: list[float | None]) -> float | None:
                    return None if any(value is None for value in values) else math.fsum(values) / len(values)
                summaries[mode][label][reference] = {
                    "position_0_mean_prompts": mean(per_position["0"]),
                    "per_position_mean_prompts": {str(pos): mean(per_position[str(pos)])
                                                  for pos in range(1, 5)},
                    "mean_positions_1_4_and_prompts": mean([
                        value for pos in range(1, 5) for value in per_position[str(pos)]])}
    semantics = {mode: a136.semantic_patch_summary(groups, PATCH_NAMES,
                 a133.PROMPTS, a133.SEMANTIC_TERMS) for mode, groups in conditions.items()}
    return {"conditions": conditions, "delta_logit_cosine_summaries": summaries,
            "fixed_semantic_hits": semantics,
            "greedy_plus_six_token_self_sharpen_completions": completions}


def evaluate_after_barrier(spec: dict[str, Any], artifact: dict[str, Any],
                           model: Any, torch: Any) -> dict[str, Any]:
    for pin in spec["privileged_sources_after_barrier"].values():
        require_hash(path_of(pin["path"]), pin["sha256"])
    a133 = import_pinned(spec["privileged_sources_after_barrier"]["attempt133_source"],
                         "attempt138_attempt133_methods")
    a136 = import_pinned(spec["privileged_sources_after_barrier"]["attempt136_source"],
                         "attempt138_attempt136_methods")
    a134 = import_pinned(spec["blind_sources"]["attempt134_source"],
                         "attempt138_attempt134_scoring")
    # The pinned historical reader imports compute_oracle_adl from scripts/.
    sys.path.insert(0, str(ROOT / "scripts"))
    reader = import_pinned(spec["privileged_sources_after_barrier"]["oracle_reader"],
                           "attempt138_oracle_reader")
    old133 = json.loads(path_of(spec["privileged_sources_after_barrier"]["attempt133_spec"]["path"]).read_text())
    if (spec["evaluation"]["patchscope"]["prompts"] != old133["target_prompts"]
            or spec["evaluation"]["patchscope"]["prompts"] != list(a133.PROMPTS)
            or spec["evaluation"]["patchscope"]["semantic_substrings"] != old133["semantic_substrings"]
            or spec["evaluation"]["patchscope"]["semantic_substrings"] != list(a133.SEMANTIC_TERMS)
            or spec["evaluation"]["lens"]["historical_convention"] != old133["logit_lens"]
            or spec["evaluation"]["lens"]["top_k"] != a133.TOP_K
            or spec["evaluation"]["patchscope"]["layer_index"] != a133.LAYER_INDEX
            or spec["evaluation"]["patchscope"]["greedy_plus_new_tokens"] != a133.NEW_TOKENS):
        raise ValueError("Attempt133 frozen semantic/Patchscope convention changed")
    from transformers import AutoTokenizer
    model_dir = path_of(spec["final_checkpoint"]["directory"])
    base_dir = Path(old133["input_locations"]["base_tokenizer_directory"])
    if old133["input_locations"]["merged_checkpoint_directory"] != str(model_dir):
        raise ValueError("Attempt133 merged checkpoint identity changed")
    provenance = reader.verify_inputs_before_loading(
        base_dir, model_dir, path_of(spec["evaluation"]["oracle_artifact"]["path"]))
    oracle = reader.load_and_validate_oracle(
        path_of(spec["evaluation"]["oracle_artifact"]["path"]), provenance, torch)
    difference = a133.validate_tensor(oracle["difference"],
        spec["evaluation"]["oracle_artifact"]["difference_raw_sha256"],
        torch, reader.sha256_raw_float32_tensor)
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    reader.validate_tokenizer(tokenizer, provenance["vocab_size"])
    old132 = torch.load(path_of(spec["privileged_sources_after_barrier"]["attempt132_candidate"]["path"]),
                        map_location="cpu", weights_only=True)
    old134 = torch.load(path_of(spec["privileged_sources_after_barrier"]["attempt134_candidate"]["path"]),
                        map_location="cpu", weights_only=True)
    if (not isinstance(old132, dict) or list(old132) != ["response_super_low", "response_next_super_low"]
            or not isinstance(old134, dict) or list(old134)[-1] != "low_surprisal_consensus_8"):
        raise ValueError("Frozen CE response inventory changed")
    ce132 = a133.validate_tensor(old132["response_super_low"],
        spec["evaluation"]["attempt132_response_raw_sha256"], torch, reader.sha256_raw_float32_tensor)
    consensus134 = a133.validate_tensor(old134["low_surprisal_consensus_8"],
        spec["evaluation"]["attempt134_consensus_raw_sha256"], torch, reader.sha256_raw_float32_tensor)
    prior136 = json.loads(path_of(spec["privileged_sources_after_barrier"]["attempt136_result"]["path"]).read_text())
    parallel64, residual64, reproduced = a136.positionwise_residual(
        difference.numpy(), consensus134.numpy())
    if (prior136["spec_sha256"] != spec["privileged_sources_after_barrier"]["attempt136_spec"]["sha256"]
            or prior136["runner_sha256"] != spec["privileged_sources_after_barrier"]["attempt136_source"]["sha256"]
            or prior136["artifact_provenance"]["oracle"] != spec["evaluation"]["oracle_artifact"]):
        raise ValueError("Attempt136 frozen residual provenance changed")
    a136_report = prior136["observed_quantities"]["attempt134_positionwise_oracle_residual"]
    for old, new in zip(a136_report["all_128_positions"], reproduced["all_128_positions"], strict=True):
        for key in ("alpha", "oracle_norm", "parallel_norm", "residual_norm",
                    "residual_squared_fraction"):
            if not math.isclose(old[key], new[key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Attempt136 exact oracle residual failed to reproduce")
    residual = torch.from_numpy(np.ascontiguousarray(residual64.astype(np.float32)))
    evaluator_pin = json.loads(path_of(spec["blind_sources"]["attempt134_spec"]["path"]).read_text())\
        ["inputs"]["attempt100"]
    evaluator = import_pinned({"path": evaluator_pin["evaluator_path"],
                               "sha256": evaluator_pin["evaluator_sha256"]},
                              "attempt138_frozen_signed_position_evaluator")
    report = a134.signed_report(NAMES[1], artifact[NAMES[1]], difference, evaluator)
    matched = flat_cosine_and_norms(artifact[NAMES[0]].numpy(), ce132.numpy())
    full = flat_cosine_and_norms(artifact[NAMES[1]].numpy(), consensus134.numpy())
    full_oracle_norms = flat_cosine_and_norms(artifact[NAMES[1]].numpy(), difference.numpy())
    full_residual_norms = flat_cosine_and_norms(artifact[NAMES[1]].numpy(), residual64)
    residual_alignment = position_range_cosines(artifact[NAMES[1]].numpy(), residual64, a136)
    vectors = {"oracle": difference, "self_sharpen": artifact[NAMES[1]],
               "attempt134": consensus134, "residual": residual}
    historical_lens = json.loads(path_of(
        spec["privileged_sources_after_barrier"]["historical_lens"]["path"]).read_text())
    log("direct Logit Lens")
    norm_cpu = copy.deepcopy(model.model.norm).to("cpu").eval()
    head_cpu = copy.deepcopy(model.lm_head).to("cpu").eval()
    lens = direct_lens(vectors, norm_cpu, head_cpu, tokenizer, a133, reader, a136,
                       historical_lens, provenance, torch)
    del norm_cpu, head_cpu
    log("fixed Patchscope prompts, directional then native")
    patch = patchscope(vectors, model, tokenizer, a133, a136, torch)
    return {"activation_signed_positionwise_oracle": report,
            "privileged_full_response_vs_attempt136_residual": residual_alignment,
            "matched_first1024_self_sharpen_vs_attempt132_super_low_ce": matched,
            "full10000_self_sharpen_vs_attempt134_consensus_ce": full,
            "full10000_self_sharpen_vs_oracle_norms": full_oracle_norms,
            "full10000_self_sharpen_vs_residual_norms": full_residual_norms,
            "direct_logit_lens": lens, "patchscope": patch,
            "reference_norms": {"oracle_full": float(np.linalg.norm(difference.numpy().astype(np.float64))),
                "attempt136_residual_full": float(np.linalg.norm(residual64)),
                "attempt134_full": full["right_norm"], "attempt132_first1024": matched["right_norm"]},
            "interpretation_limits": ["Directional novelty alone does not establish historical recovery.",
                "Residual and oracle comparisons are privileged, post-freeze diagnostics.",
                "Patchscope token hits do not establish the fine-tuning dataset content."]}


def run() -> None:
    import torch
    spec = load_spec()
    paths = output_paths(spec)
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for frozen FP32 strict JVP construction")
    a14 = import_pinned(spec["blind_sources"]["gradient_jvp_helper"], "attempt138_gradient_jvp_helper")
    a132 = import_pinned(spec["blind_sources"]["attempt132_source"], "attempt138_selection_helper")
    a126 = import_pinned(spec["blind_sources"]["attempt126_publish_helper"],
                         "attempt138_publish_helper")
    log("validating frozen blind inputs and SUPER_LOW1024 selection")
    selected, probe, selection = validate_blind_inputs(spec, a14, a132, torch)
    model_dir = path_of(spec["final_checkpoint"]["directory"])
    model = a14.load_local_model(model_dir, "cuda", torch)
    eligible = a14.discover_eligible_linear_weights(model, torch)
    a14.freeze_other_parameters(model, eligible)
    before = a14.model_state_hashes(model, torch)
    log("computing the 128-batch positive negative-entropy gradient")
    mean_s = accumulate_self_sharpen_gradient(model, selected, eligible, spec, torch)
    scale = a14.global_tangent_scale(eligible, 0.00125, torch)
    if not math.isclose(scale["aggregate_source_weight_norm"], 918.1587250300843,
                        rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError("Attempt005 eligible final-weight norm changed")
    tangents, matrix_records, realized = a14.prepare_tangents(eligible, scale, torch)
    log("one strict 157-batch JVP sweep; accumulating first1024 and full10000")
    responses, readout = dual_probe_jvp(model, probe, tangents, spec, a14, torch)
    tangents.clear()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    if a14.checkpoint_file_records(model_dir) != spec["final_checkpoint"]["files"]:
        raise ValueError("Canonical final checkpoint changed before barrier")
    a14.verify_model_unchanged(model, before, torch)
    require_hash(SPEC_PATH, SPEC_SHA256)
    for pin in spec["blind_sources"].values():
        require_hash(path_of(pin["path"]), pin["sha256"])
    for source in spec["fixed_selection"]["source_token_artifacts"].values():
        require_hash(path_of(source["tokens_path"]), source["tokens_serialized_sha256"])
    require_hash(path_of(spec["probe"]["path"]), spec["probe"]["serialized_sha256"])
    a126.a122.prior.atomic_torch_publish(paths["candidate_path"], responses, torch)
    manifest = {"format_version": 1, "attempt_id": ATTEMPT,
        "spec_sha256": SPEC_SHA256, "constructor_sha256": sha256_file(Path(__file__).resolve()),
        "blind_source_sha256": {key: value["sha256"] for key, value in spec["blind_sources"].items()},
        "final_checkpoint_files": spec["final_checkpoint"]["files"],
        "fixed_selection": {"source_attempt": 132, "group": "super_low",
                            "rank_range": [0, 1024],
                            "ordered_pair_raw_int64_sha256": selection["ordered_pair_raw_int64_sha256"],
                            "pair_count": 1024},
        "objective": spec["objective"], "mean_negative_entropy_objective": mean_s,
        "gradient_and_tangent_scale": {**scale, **realized,
                                        "eligible_matrix_count": len(matrix_records)},
        "eligible_matrices": matrix_records,
        "probe": spec["probe"], "readout": spec["readout"], "jvp": spec["jvp"],
        "jvp_readout_checks": readout,
        "responses": {name: {"raw_sha256": a14.sha256_raw_float32_tensor(tensor, torch),
            "shape": list(SHAPE), "dtype": "contiguous_cpu_float32",
            "norm": float(np.linalg.norm(tensor.numpy().astype(np.float64)))}
            for name, tensor in responses.items()},
        "candidate": {"path": str(paths["candidate_path"]),
                      "serialized_sha256": sha256_file(paths["candidate_path"]),
                      "tensor_order": list(NAMES)},
        "barrier": spec["barrier"], "manifest_contains_historical_scores": False,
        "same_specimen_exploratory_method_development": True,
        "clean_heldout_validation": False}
    a126.a122.prior.atomic_json_publish(paths["construction_manifest_path"], manifest)
    artifact = validate_published(paths, manifest, spec, a14, torch)
    manifest_hash = sha256_file(paths["construction_manifest_path"])
    require_hash(paths["construction_manifest_path"], manifest_hash)
    del selected, probe, responses, eligible, matrix_records
    gc.collect()
    log(MARKER)
    print(MARKER, flush=True)
    log("post-freeze historical evaluation and semantics")
    observed = evaluate_after_barrier(spec, artifact, model, torch)
    result = {"format_version": 1, "attempt_id": ATTEMPT,
              "spec_sha256": SPEC_SHA256, "constructor_sha256": manifest["constructor_sha256"],
              "construction_manifest_sha256": manifest_hash,
              "candidate_serialized_sha256": manifest["candidate"]["serialized_sha256"],
              "same_specimen_exploratory_method_development": True,
              "clean_heldout_validation": False,
              "no_oracle_guided_mixture_or_tuning": True,
              "observed_quantities": observed,
              "interpretation_axes": ["directional_novelty", "oracle_alignment",
                                      "residual_alignment", "downstream_functional_alignment"]}
    if paths["result_path"].exists() or paths["result_path"].is_symlink():
        raise ValueError("Result output already exists")
    with paths["result_path"].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    log("wrote result.json")


if __name__ == "__main__":
    try:
        run()
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
