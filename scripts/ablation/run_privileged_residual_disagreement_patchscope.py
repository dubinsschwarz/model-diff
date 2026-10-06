#!/usr/bin/env python3
"""Attempt136: privileged residual, blind disagreement geometry, and Patchscope."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = "136_privileged_residual_disagreement_patchscope"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "f7d56fa357bfb8d29a1d5648d2f22669d8e30f8b7f78e3e2af253d880d22aaa5"
ORDER = ("fineweb", "wikitext103_raw", "tinystories", "arxiv_document",
         "cc_news", "pg19", "codeparrot_clean", "ultrachat")
JOINT_NAMES = tuple("full:" + name for name in ORDER) + tuple("low:" + name for name in ORDER)
POSITIONS = (0, 1, 2, 3, 4)
PRIMARY = (1, 2, 3, 4)
RANGES = {"positions_1_4": (1, 5), "positions_1_127": (1, 128)}
LENS_NAMES = ("oracle", "attempt100", "attempt134", "parallel", "residual")
DIRECTIONAL_NAMES = ("oracle", "attempt100", "attempt134", "residual")
NATIVE_NAMES = ("oracle", "attempt134", "parallel", "residual")
SHAPE = (128, 2048)


def log(message: str) -> None:
    print("[136] " + message, flush=True)


def path_of(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ValueError(f"Frozen SHA256 mismatch: {path}")


def import_pinned(path: Path, expected: str, name: str) -> Any:
    require_hash(path, expected)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def require_output_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"Result already exists: {path}")


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    require_hash(path, SPEC_SHA256)
    spec = json.loads(path.read_text())
    if (spec["attempt_id"] != ATTEMPT or spec["corpus_order"] != list(ORDER)
            or spec["joint_view_order"] != list(JOINT_NAMES)
            or spec["positions"] != list(POSITIONS)
            or spec["primary_positions"] != list(PRIMARY)
            or spec["secondary_positions"] != list(range(1, 128))
            or spec["disagreement"]["spaces"] != ["full_8", "low_8", "joint_16"]
            or spec["disagreement"]["rank_bounds"] != {"full_8": 7, "low_8": 7, "joint_16": 15}
            or spec["disagreement"]["low_residual_presence_thresholds"] != {"strong": 0.50, "partial": 0.25}
            or spec["patchscope"]["attempt134_vs_attempt100_clear_gain"] != 0.01
            or spec["patchscope"]["directional_vectors"] != list(DIRECTIONAL_NAMES)
            or spec["patchscope"]["native_vectors"] != list(NATIVE_NAMES)
            or spec["lens"]["vectors"] != list(LENS_NAMES)
            or spec["no_new_candidate_constructed"] is not True
            or spec["privileged_diagnostic_only"] is not True
            or spec["clean_heldout_validation"] is not False
            or spec["no_candidate_artifact"] is not True
            or spec["no_construction_manifest"] is not True
            or spec["output"] != "experiments/attempts/136_privileged_residual_disagreement_patchscope/result.json"):
        raise ValueError("Attempt136 fixed diagnostic plan changed")
    return spec


def validate_pinned_files(spec: dict[str, Any], *, blind_only: bool) -> None:
    groups = ("attempt100", "attempt134") if blind_only else tuple(spec["files"])
    for group in groups:
        records = spec["files"][group]
        for item in ([records] if "path" in records else records.values()):
            require_hash(path_of(item["path"]), item["sha256"])
    artifacts = ("attempt100", "attempt134") if blind_only else tuple(spec["artifacts"])
    for name in artifacts:
        row = spec["artifacts"][name]
        require_hash(path_of(row["path"]), row["serialized_sha256"])
    if blind_only:
        m100 = json.loads(path_of(spec["files"]["attempt100"]["construction_manifest"]["path"]).read_text())
        m134 = json.loads(path_of(spec["files"]["attempt134"]["construction_manifest"]["path"]).read_text())
        p100, p134 = spec["artifacts"]["attempt100"], spec["artifacts"]["attempt134"]
        if (m100["spec_sha256"] != spec["files"]["attempt100"]["spec"]["sha256"]
                or m100["constructor_sha256"] != spec["files"]["attempt100"]["source"]["sha256"]
                or m134["spec_sha256"] != spec["files"]["attempt134"]["spec"]["sha256"]
                or m134["constructor_sha256"] != spec["files"]["attempt134"]["source"]["sha256"]
                or m100["historical_base_access"] is not False
                or m100["oracle_adl_access"] is not False
                or m100["prior_evaluation_access"] is not False
                or m100["corpus_order"] != list(ORDER)
                or m100["artifact"]["serialized_sha256"] != p100["serialized_sha256"]
                or m100["artifact"]["raw_tensors_sha256"] != p100["raw_tensors_sha256"]
                or list(m100["artifact"]["raw_tensors_sha256"]) != p100["tensor_order"]
                or m134["candidate"]["serialized_sha256"] != p134["serialized_sha256"]
                or m134["candidate"]["tensor_order"] != p134["tensor_order"]
                or {name: row["raw_sha256"] for name, row in m134["responses"].items()}
                != p134["raw_tensors_sha256"]
                or m134["barrier"]["marker"] != "DIVERSE8_LOW_SURPRISAL_CONSENSUS_FROZEN"
                or m134["barrier"]["all_8_rankings_rows_gradients_responses_and_consensus_published_revalidated_before_oracle"] is not True
                or m134["barrier"]["manifest_contains_historical_scores"] is not False
                or m134["corpus_order"] != list(ORDER)):
            raise ValueError("Frozen blind response provenance changed")


def load_blind_responses(spec: dict[str, Any], torch: Any, a133: Any,
                         oracle_reader: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    outputs = {}
    for source in ("attempt100", "attempt134"):
        pin = spec["artifacts"][source]
        artifact = torch.load(path_of(pin["path"]), map_location="cpu", weights_only=True)
        if not isinstance(artifact, dict) or list(artifact) != pin["tensor_order"]:
            raise ValueError(f"{source} frozen tensor inventory/order changed")
        for name in pin["tensor_order"]:
            a133.validate_tensor(artifact[name], pin["raw_tensors_sha256"][name],
                                 torch, oracle_reader.sha256_raw_float32_tensor)
        outputs[source] = artifact
    full = np.stack([outputs["attempt100"]["response_" + name].numpy() for name in ORDER])
    low = np.stack([outputs["attempt134"]["response_low_" + name].numpy() for name in ORDER])
    if full.shape != (8, *SHAPE) or low.shape != (8, *SHAPE):
        raise ValueError("Eight-corpus response shape changed")
    return full, low, {
        "attempt100": outputs["attempt100"]["consensus_response_8"],
        "attempt134": outputs["attempt134"]["low_surprisal_consensus_8"],
    }


def positionwise_residual(oracle: np.ndarray, consensus: np.ndarray,
                          tolerance: float = 1e-10) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    o, c = np.asarray(oracle, dtype=np.float64), np.asarray(consensus, dtype=np.float64)
    if o.shape != c.shape or o.ndim != 2 or not np.isfinite(o).all() or not np.isfinite(c).all():
        raise ValueError("Malformed oracle/consensus vectors")
    c2 = np.sum(c * c, axis=1, dtype=np.float64)
    o2 = np.sum(o * o, axis=1, dtype=np.float64)
    if np.any(c2 <= 0) or np.any(o2 <= 0):
        raise ValueError("Zero oracle or consensus position norm")
    alpha = np.sum(o * c, axis=1, dtype=np.float64) / c2
    parallel = alpha[:, None] * c
    residual = o - parallel
    p2 = np.sum(parallel * parallel, axis=1, dtype=np.float64)
    r2 = np.sum(residual * residual, axis=1, dtype=np.float64)
    orth = np.sum(residual * c, axis=1, dtype=np.float64)
    relative_orth = np.abs(orth) / (np.sqrt(o2) * np.sqrt(c2))
    if not np.isfinite(relative_orth).all() or np.any(relative_orth > tolerance):
        raise ValueError("Positionwise residual is not orthogonal in CPU float64")
    records = []
    for pos in range(len(o)):
        records.append({"position": pos, "alpha": float(alpha[pos]),
                        "cosine_C_O": float(np.sum(o[pos] * c[pos]) / np.sqrt(o2[pos] * c2[pos])),
                        "oracle_norm": float(np.sqrt(o2[pos])),
                        "parallel_norm": float(np.sqrt(p2[pos])),
                        "residual_norm": float(np.sqrt(r2[pos])),
                        "residual_norm_fraction": float(np.sqrt(r2[pos] / o2[pos])),
                        "residual_squared_fraction": float(r2[pos] / o2[pos]),
                        "orthogonality_dot": float(orth[pos]),
                        "orthogonality_error_relative_to_normO_normC": float(relative_orth[pos])})
    aggregates = {}
    for name, (start, stop) in RANGES.items():
        # Alpha was fit independently at each position before concatenation.
        total_o = float(np.sum(o2[start:stop], dtype=np.float64))
        total_p = float(np.sum(p2[start:stop], dtype=np.float64))
        total_r = float(np.sum(r2[start:stop], dtype=np.float64))
        fractions = (total_p / total_o, total_r / total_o)
        if abs(sum(fractions) - 1) > 1e-10:
            raise ValueError("Positionwise parallel/residual energy identity failed")
        aggregates[name] = {"oracle_raw_concatenation_norm": math.sqrt(total_o),
                            "parallel_raw_concatenation_norm": math.sqrt(total_p),
                            "residual_raw_concatenation_norm": math.sqrt(total_r),
                            "parallel_energy_fraction": fractions[0],
                            "residual_energy_fraction": fractions[1],
                            "energy_fraction_sum": sum(fractions)}
    return parallel, residual, {"position_0": records[0], "all_128_positions": records,
                                "aggregates": aggregates,
                                "orthogonality_tolerance_relative_to_normO_normC": tolerance}


def unit_views(responses: np.ndarray) -> np.ndarray:
    value = np.asarray(responses, dtype=np.float64)
    if value.ndim != 3 or value.shape[1] != 128 or not np.isfinite(value).all():
        raise ValueError("Malformed blind response views")
    norms = np.linalg.norm(value, axis=2)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Zero/nonfinite per-position blind response norm")
    return value / norms[:, :, None]


def centered_disagreement(unit: np.ndarray) -> np.ndarray:
    value = np.asarray(unit, dtype=np.float64)
    centered = value - np.mean(value, axis=0, dtype=np.float64)
    if np.max(np.abs(np.sum(centered, axis=0, dtype=np.float64))) > 1e-12:
        raise ValueError("Blind disagreement rows are not centered")
    return centered


def blind_svd(matrix: np.ndarray, view_names: tuple[str, ...], rank_bound: int) -> dict[str, Any]:
    d = np.ascontiguousarray(matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != len(view_names) or not np.isfinite(d).all():
        raise ValueError("Malformed blind disagreement matrix")
    u, singular, vh = np.linalg.svd(d, full_matrices=False)
    tolerance = max(d.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank > rank_bound:
        raise ValueError("Centered disagreement exceeds precommitted rank bound")
    for i in range(rank):
        absolute = np.abs(u[:, i])
        largest = float(np.max(absolute))
        # Treat roundoff-scale equal loadings as ties, then use the first view.
        pivot = int(np.flatnonzero(np.isclose(
            absolute, largest, rtol=0, atol=32 * np.finfo(np.float64).eps))[0])
        if u[pivot, i] < 0:
            u[:, i] *= -1
            vh[i, :] *= -1
    reconstruction = (u * singular[None, :]) @ vh
    denom = np.linalg.norm(d)
    error = float(np.linalg.norm(d - reconstruction) / denom) if denom > 0 else 0.0
    return {"matrix": d, "u": u, "singular_values": singular, "vh": vh,
            "rank": rank, "rank_tolerance": float(tolerance),
            "view_names": view_names, "svd_reconstruction_relative_error": error}


def build_blind_bases(full: np.ndarray, low: np.ndarray) -> dict[str, dict[str, Any]]:
    uf, ul = unit_views(full), unit_views(low)
    if uf.shape != (8, *SHAPE) or ul.shape != (8, *SHAPE):
        raise ValueError("Exactly eight responses from each frozen view are required")
    spaces = {
        "full_8": (centered_disagreement(uf), ORDER, 7),
        "low_8": (centered_disagreement(ul), ORDER, 7),
        "joint_16": (centered_disagreement(np.concatenate((uf, ul), axis=0)), JOINT_NAMES, 15),
    }
    bases = {}
    for space, (centered, names, bound) in spaces.items():
        bases[space] = {}
        for range_name, (start, stop) in RANGES.items():
            # C-order flattening keeps positions major within each fixed view row.
            matrix = centered[:, start:stop, :].reshape(len(names), -1)
            bases[space][range_name] = blind_svd(matrix, tuple(names), bound)
    return bases


def analyze_residual_in_basis(basis: dict[str, Any], residual: np.ndarray) -> dict[str, Any]:
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    d = basis["matrix"]
    if r.size != d.shape[1] or not np.isfinite(r).all():
        raise ValueError("Residual and fixed blind span have incompatible shapes")
    rnorm = float(np.linalg.norm(r))
    if rnorm <= 0:
        raise ValueError("Zero residual cannot be diagnosed")
    rank = basis["rank"]
    b = basis["vh"][:rank]
    coefficients = b @ r
    projected = b.T @ coefficients
    projected_norm = float(np.linalg.norm(projected))
    squared_fraction = float((projected_norm / rnorm) ** 2)
    sv = basis["singular_values"]
    variance_total = float(np.sum(sv * sv))
    pcs = []
    variance_running = 0.0
    residual_running = 0.0
    for i in range(rank):
        variance = float(sv[i] ** 2 / variance_total)
        residual_fraction = float((coefficients[i] / rnorm) ** 2)
        variance_running += variance
        residual_running += residual_fraction
        pcs.append({"pc_index": i + 1, "singular_value": float(sv[i]),
                    "disagreement_variance_fraction": variance,
                    "cumulative_disagreement_variance_fraction": variance_running,
                    "signed_cosine_with_oracle_residual": float(coefficients[i] / rnorm),
                    "residual_squared_fraction_carried": residual_fraction,
                    "cumulative_residual_squared_fraction": residual_running,
                    "oriented_view_loadings": {name: float(basis["u"][j, i])
                                               for j, name in enumerate(basis["view_names"])}})
    if abs(residual_running - squared_fraction) > 1e-8:
        raise ValueError("PC residual fractions do not sum to span fraction")
    rows = []
    for name, row in zip(basis["view_names"], d):
        row_norm = float(np.linalg.norm(row))
        dot_unit = float(np.dot(row, r) / rnorm)
        rows.append({"view": name, "signed_cosine_with_residual":
                     None if row_norm == 0 else dot_unit / row_norm,
                     "dot_onto_residual_unit": dot_unit, "row_norm": row_norm})
    return {"rank": rank, "rank_tolerance": basis["rank_tolerance"],
            "matrix_shape": list(d.shape), "residual_squared_fraction_in_span": squared_fraction,
            "residual_norm_fraction_in_span": projected_norm / rnorm,
            "residual_projection_reconstruction_error":
            float(np.linalg.norm(projected - b.T @ (b @ projected)) / rnorm),
            "svd_reconstruction_relative_error": basis["svd_reconstruction_relative_error"],
            "pcs": pcs, "disagreement_rows": rows}


def classify_low_residual_presence(fraction: float) -> str:
    if not math.isfinite(fraction) or fraction < 0 or fraction > 1 + 1e-8:
        raise ValueError("Invalid residual span fraction")
    if fraction >= 0.50:
        return "strong_residual_presence"
    if fraction >= 0.25:
        return "partial_residual_presence"
    return "limited_residual_presence"


def classify_patchscope_gain(gain: float) -> str:
    if not math.isfinite(gain):
        raise ValueError("Nonfinite Patchscope gain")
    if gain >= 0.01:
        return "clear_patchscope_gain"
    if gain > 0:
        return "positive_patchscope_gain"
    return "no_patchscope_gain"


def nullable_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    a, b = np.asarray(left, dtype=np.float64).ravel(), np.asarray(right, dtype=np.float64).ravel()
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denominator == 0 or not math.isfinite(denominator) else float(np.dot(a, b) / denominator)


def native_additivity(delta_parallel: np.ndarray, delta_residual: np.ndarray,
                      delta_oracle: np.ndarray) -> dict[str, float | None]:
    predicted = np.asarray(delta_parallel, dtype=np.float64) + np.asarray(delta_residual, dtype=np.float64)
    oracle = np.asarray(delta_oracle, dtype=np.float64)
    oracle_norm = float(np.linalg.norm(oracle))
    error = float(np.linalg.norm(predicted - oracle) / oracle_norm) if oracle_norm > 0 else math.nan
    return {"cosine_delta_P_plus_delta_R_vs_delta_O": nullable_cosine(predicted, oracle),
            "relative_error_delta_P_plus_delta_R_vs_delta_O":
            error if math.isfinite(error) else None}


def direct_lens(vectors: dict[str, Any], a133: Any, oracle_reader: Any,
                final_norm: Any, lm_head: Any, tokenizer: Any, torch: Any,
                historical: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    tokens, arrays = {}, {}
    with torch.inference_mode():
        for name in LENS_NAMES:
            tokens[name], arrays[name] = {}, {}
            for pos in POSITIONS:
                distribution = a133.lens_distribution(vectors[name][pos], final_norm, lm_head, torch)
                positive = oracle_reader.top_token_records(distribution["positive_probs"], tokenizer,
                                                           a133.TOP_K, torch)
                negative = oracle_reader.top_token_records(distribution["negative_probs"], tokenizer,
                                                           a133.TOP_K, torch)
                tokens[name][str(pos)] = {
                    "top20_positive": positive, "top20_negative": negative,
                    "semantic_hits": a133.semantic_hits(positive, negative),
                    "full_vocab_hashes": {key: a133.array_hash(value.detach().cpu().numpy())
                                          for key, value in distribution.items()},
                }
                arrays[name][str(pos)] = {key: value.detach().cpu().numpy()
                                          for key, value in distribution.items()}
    a133.validate_historical_oracle_lens(
        {"tokens": {"oracle_difference": tokens["oracle"]}}, historical, provenance)
    comparisons = {}
    for name in LENS_NAMES[1:]:
        per_position = {str(pos): a133.lens_metrics(arrays[name][str(pos)], arrays["oracle"][str(pos)])
                        for pos in POSITIONS}
        keys = tuple(per_position["0"])
        comparisons[name] = {"position_0": per_position["0"],
                             "positions_1_4": {str(pos): per_position[str(pos)] for pos in PRIMARY},
                             "mean_positions_1_4": a133.mean_dict(
                                 [per_position[str(pos)] for pos in PRIMARY], keys)}
    semantics = {}
    for name in LENS_NAMES:
        semantics[name] = {
            "position_0": tokens[name]["0"]["semantic_hits"],
            "sum_positions_1_4": {term: {polarity: sum(
                tokens[name][str(pos)]["semantic_hits"][term][polarity] for pos in PRIMARY)
                for polarity in ("positive", "negative")} for term in a133.SEMANTIC_TERMS},
        }
    return {"tokens": tokens, "oracle_similarity": comparisons,
            "known_semantic_hits": semantics,
            "full_vocabulary_arrays": "transient_only_top20_hashes_and_metrics_persisted"}


def patch_record(delta: np.ndarray, patched: np.ndarray, probs: np.ndarray,
                 baseline: np.ndarray, baseline_probs: np.ndarray,
                 tokenizer: Any, a133: Any) -> dict[str, Any]:
    norm = float(np.linalg.norm(delta.astype(np.float64)))
    if not math.isfinite(norm):
        return {"delta_logit_l2_norm": None, "delta_logit_norm_status": str(norm),
                "delta_logits_sha256": a133.array_hash(delta),
                "patched_logits_sha256": a133.array_hash(patched),
                "patched_probabilities_sha256": a133.array_hash(probs),
                "top20_positive_delta": [], "top20_negative_delta": [],
                "semantic_hits_available": False,
                "semantic_hits": {term: {"positive": 0, "negative": 0}
                                  for term in a133.SEMANTIC_TERMS}}
    positive = a133.token_delta_records(delta, patched, probs, tokenizer, True,
                                        baseline, baseline_probs)
    negative = a133.token_delta_records(delta, patched, probs, tokenizer, False,
                                        baseline, baseline_probs)
    return {"delta_logit_l2_norm": norm,
            "delta_logit_norm_status": "finite",
            "delta_logits_sha256": a133.array_hash(delta),
            "patched_logits_sha256": a133.array_hash(patched),
            "patched_probabilities_sha256": a133.array_hash(probs),
            "top20_positive_delta": positive,
            "top20_negative_delta": negative,
            "semantic_hits_available": True,
            "semantic_hits": a133.semantic_hits(positive, negative)}


def semantic_patch_summary(conditions: dict[str, Any], names: tuple[str, ...],
                           prompts: tuple[str, ...], terms: tuple[str, ...]) -> dict[str, Any]:
    summaries = {}
    for name in names:
        summaries[name] = {}
        for sign in ("plus", "minus"):
            summaries[name][sign] = {}
            for label, positions in (("position_0", (0,)), ("positions_1_4", PRIMARY)):
                by_term = {term: sum(
                    conditions[name][str(pos)][prompt][sign]["semantic_hits"][term][polarity]
                    for pos in positions for prompt in prompts for polarity in ("positive", "negative"))
                    for term in terms}
                summaries[name][sign][label] = {
                    "fixed_substring_hits": by_term,
                    "unavailable_conditions": sum(
                        not conditions[name][str(pos)][prompt][sign].get("semantic_hits_available", True)
                        for pos in positions for prompt in prompts),
                    "craft_family_hits": by_term["craft"],
                    "precision_family_hits": by_term["precision"],
                    "cake_bake_cook_family_hits": sum(by_term[term] for term in ("cake", "bake", "cook")),
                }
    return summaries


def patchscope_mode(vectors: dict[str, Any], mode: str, a133: Any, model: Any,
                    tokenizer: Any, torch: Any) -> dict[str, Any]:
    if mode not in ("directional_oracle_norm_matched", "native_amplitude"):
        raise ValueError("Unknown precommitted Patchscope amplitude mode")
    names = DIRECTIONAL_NAMES if mode == "directional_oracle_norm_matched" else NATIVE_NAMES
    device = next(model.parameters()).device
    conditions = {name: {str(pos): {} for pos in POSITIONS} for name in names}
    baselines, additivity, completions = {}, {}, {}
    if mode == "directional_oracle_norm_matched":
        completions = {name: {str(pos): {} for pos in PRIMARY} for name in names}
    for prompt_index, prompt in enumerate(a133.PROMPTS, 1):
        log(f"{mode} prompt {prompt_index}/{len(a133.PROMPTS)}")
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("Malformed Attempt133 plain-text target prompt")
        patch_index = ids.shape[1] - 1
        baseline = a133.next_logits(model, ids, None, None, +1, torch).numpy()
        baseline_probs = torch.softmax(torch.from_numpy(baseline), dim=-1).numpy()
        baselines[prompt] = {"logits_sha256": a133.array_hash(baseline),
                             "probabilities_sha256": a133.array_hash(baseline_probs)}
        additivity[prompt] = {}
        for pos in POSITIONS:
            patch_vectors = {}
            oracle_vector = vectors["oracle"][pos]
            for name in names:
                source_vector = vectors[name][pos]
                if mode == "directional_oracle_norm_matched" and name != "oracle":
                    scaled = a133.norm_match_blind(source_vector.numpy(), oracle_vector.numpy())
                    patch_vectors[name] = torch.from_numpy(scaled).to(device=device)
                else:
                    patch_vectors[name] = source_vector.to(device=device)
            deltas, probabilities = {}, {}
            for name in names:
                conditions[name][str(pos)][prompt] = {}
                for sign, label in ((+1, "plus"), (-1, "minus")):
                    patched = a133.next_logits(model, ids, patch_index,
                                               patch_vectors[name], sign, torch).numpy()
                    probs = torch.softmax(torch.from_numpy(patched), dim=-1).numpy()
                    delta = patched.astype(np.float64) - baseline.astype(np.float64)
                    record = patch_record(delta, patched, probs, baseline, baseline_probs, tokenizer, a133)
                    conditions[name][str(pos)][prompt][label] = record
                    deltas[name, label] = delta
                    probabilities[name, label] = probs
            for name in names:
                for label in ("plus", "minus"):
                    record = conditions[name][str(pos)][prompt][label]
                    delta = deltas[name, label]
                    oracle_same = deltas["oracle", label]
                    oracle_plus = deltas["oracle", "plus"]
                    record["cosine_to_oracle_same_sign_delta_logits"] = nullable_cosine(delta, oracle_same)
                    record["cosine_to_oracle_plus_delta_logits"] = nullable_cosine(delta, oracle_plus)
                    if (mode == "directional_oracle_norm_matched" and
                            record["cosine_to_oracle_same_sign_delta_logits"] is not None):
                        record["attempt133_oracle_delta_metrics"] = a133.patch_metrics(
                            delta, oracle_same, probabilities[name, label], baseline_probs)
                    elif mode == "directional_oracle_norm_matched":
                        record["attempt133_oracle_delta_metrics"] = None
            if mode == "native_amplitude":
                additivity[prompt][str(pos)] = {
                    label: native_additivity(
                        deltas["parallel", label], deltas["residual", label],
                        deltas["oracle", label])
                    for label in ("plus", "minus")}
            elif pos in PRIMARY:
                for name in names:
                    completions[name][str(pos)][prompt] = a133.greedy_completion(
                        model, ids, patch_vectors[name], tokenizer, torch, a133.NEW_TOKENS)
    summary = {}
    for name in names:
        summary[name] = {}
        for label in ("plus", "minus"):
            records_0 = [conditions[name]["0"][prompt][label] for prompt in a133.PROMPTS]
            records_primary = [conditions[name][str(pos)][prompt][label]
                               for pos in PRIMARY for prompt in a133.PROMPTS]
            def means(records: list[dict[str, Any]]) -> dict[str, float | None]:
                cosines = [row["cosine_to_oracle_same_sign_delta_logits"] for row in records]
                plus_cosines = [row["cosine_to_oracle_plus_delta_logits"] for row in records]
                norms = [row["delta_logit_l2_norm"] for row in records]
                summary = {"mean_delta_logit_l2_norm": None if any(v is None for v in norms)
                        else math.fsum(norms) / len(norms),
                        "mean_cosine_to_oracle_same_sign": None if any(v is None for v in cosines)
                        else math.fsum(cosines) / len(cosines),
                        "mean_cosine_to_oracle_plus": None if any(v is None for v in plus_cosines)
                        else math.fsum(plus_cosines) / len(plus_cosines)}
                if mode == "directional_oracle_norm_matched":
                    metrics = [row["attempt133_oracle_delta_metrics"] for row in records]
                    summary["mean_attempt133_oracle_delta_metrics"] = (
                        None if any(value is None for value in metrics)
                        else a133.mean_dict(metrics, tuple(metrics[0])))
                return summary
            summary[name][label] = {"position_0_mean_prompts": means(records_0),
                                    "mean_positions_1_4_and_prompts": means(records_primary),
                                    "per_position_mean_prompts": {
                                        str(pos): means([conditions[name][str(pos)][prompt][label]
                                                         for prompt in a133.PROMPTS]) for pos in PRIMARY}}
    semantics = semantic_patch_summary(conditions, names, a133.PROMPTS, a133.SEMANTIC_TERMS)
    return {"mode": mode, "baselines": baselines, "conditions": conditions,
            "oracle_similarity_summaries": summary,
            "known_semantic_hits": semantics,
            "greedy_plus_completions": completions,
            "native_P_plus_R_additivity": additivity if mode == "native_amplitude" else None}


def native_norm_ratios(oracle: np.ndarray, consensus: np.ndarray,
                       parallel: np.ndarray, residual: np.ndarray) -> dict[str, Any]:
    arrays = {"oracle": np.asarray(oracle, dtype=np.float64),
              "attempt134": np.asarray(consensus, dtype=np.float64),
              "parallel": np.asarray(parallel, dtype=np.float64),
              "residual": np.asarray(residual, dtype=np.float64)}
    if any(value.shape != arrays["oracle"].shape or not np.isfinite(value).all()
           for value in arrays.values()):
        raise ValueError("Malformed native-amplitude diagnostic vectors")
    values = {}
    for pos in POSITIONS:
        oracle_norm = float(np.linalg.norm(arrays["oracle"][pos]))
        if oracle_norm <= 0 or not math.isfinite(oracle_norm):
            raise ValueError("Zero/nonfinite oracle patch norm")
        values[str(pos)] = {name + "_to_oracle": float(
            np.linalg.norm(arrays[name][pos]) / oracle_norm)
            for name in ("attempt134", "parallel", "residual")}
    return values


def descriptive_interpretation(spans: dict[str, Any], lens: dict[str, Any],
                               directional: dict[str, Any], classifications: dict[str, Any]) -> dict[str, Any]:
    fractions = [spans[space][range_name]["residual_squared_fraction_in_span"]
                 for space in ("full_8", "low_8", "joint_16") for range_name in RANGES]
    geometry = ("The missing oracle component is at least partly present in existing blind "
                "measurements; blind component selection remains the next question."
                if max(fractions) >= 0.25 else
                "Most missing oracle energy is outside these blind disagreement spaces; "
                "a new observable or operator may be needed.")
    lens_hits = lens["known_semantic_hits"]
    def lens_count(name: str, group: str, term: str) -> int:
        hit = lens_hits[name][group][term]
        return hit["positive"] + hit["negative"]
    patch_hits = directional["known_semantic_hits"]
    residual_primary_precision = patch_hits["residual"]["plus"]["positions_1_4"]["precision_family_hits"]
    attempt134_primary_precision = patch_hits["attempt134"]["plus"]["positions_1_4"]["precision_family_hits"]
    residual_zero_cake = patch_hits["residual"]["plus"]["position_0"]["cake_bake_cook_family_hits"]
    attempt134_zero_cake = patch_hits["attempt134"]["plus"]["position_0"]["cake_bake_cook_family_hits"]
    low_craft = patch_hits["attempt134"]["plus"]["positions_1_4"]["craft_family_hits"]
    old_craft = patch_hits["attempt100"]["plus"]["positions_1_4"]["craft_family_hits"]
    low_cooking = patch_hits["attempt134"]["plus"]["positions_1_4"]["cake_bake_cook_family_hits"]
    old_cooking = patch_hits["attempt100"]["plus"]["positions_1_4"]["cake_bake_cook_family_hits"]
    return {"blind_disagreement_geometry": geometry,
            "direct_lens_residual_precision_exceeds_attempt134":
            lens_count("residual", "sum_positions_1_4", "precision") >
            lens_count("attempt134", "sum_positions_1_4", "precision"),
            "directional_patch_residual_precision_exceeds_attempt134":
            residual_primary_precision > attempt134_primary_precision,
            "directional_patch_position0_residual_cake_bake_cook_exceeds_attempt134":
            residual_zero_cake > attempt134_zero_cake,
            "directional_patch_attempt134_craft_hits_exceed_attempt100": low_craft > old_craft,
            "directional_patch_attempt134_cake_bake_cook_hits_exceed_attempt100": low_cooking > old_cooking,
            "patchscope_attempt134_vs_attempt100": classifications["patchscope_attempt134_vs_attempt100"],
            "limits": ["Privileged PCs and oracle-correlated loadings are not blind candidates.",
                       "Patchscope token hits do not establish fine-tuning dataset content.",
                       "This is same-specimen exploratory analysis, not held-out validation."]}


def verify_attempt133_conventions(spec: dict[str, Any], a133: Any) -> None:
    pin = spec["files"]["attempt133"]["spec"]
    require_hash(path_of(pin["path"]), pin["sha256"])
    old_spec = json.loads(path_of(pin["path"]).read_text())
    if (spec["prompts"] != old_spec["target_prompts"] or spec["prompts"] != list(a133.PROMPTS)
            or spec["semantic_substrings"] != old_spec["semantic_substrings"]
            or spec["semantic_substrings"] != list(a133.SEMANTIC_TERMS)
            or spec["top_k"] != old_spec["top_k"] or spec["top_k"] != a133.TOP_K
            or spec["greedy_completion_tokens"] != old_spec["patchscope"]["greedy_completion_tokens"]
            or spec["greedy_completion_tokens"] != a133.NEW_TOKENS
            or spec["patch_layer_index"] != old_spec["patchscope"]["layer_index"]
            or spec["patch_layer_index"] != a133.LAYER_INDEX
            or spec["attempt133_provenance"]["logit_lens"] != old_spec["logit_lens"]
            or spec["attempt133_provenance"]["patchscope"] != old_spec["patchscope"]
            or spec["attempt133_provenance"]["immutable_input_sha256"] != old_spec["immutable_input_sha256"]
            or spec["attempt133_provenance"]["input_locations"] != old_spec["input_locations"]):
        raise ValueError("Attempt133 frozen lens/Patchscope conventions changed")


def run() -> None:
    spec = load_spec()
    output = path_of(spec["output"])
    require_output_absent(output)
    log("validating frozen blind inputs")
    validate_pinned_files(spec, blind_only=True)
    a133_pin = spec["files"]["attempt133"]["source"]
    a133 = import_pinned(path_of(a133_pin["path"]), a133_pin["sha256"], "attempt136_attempt133_methods")
    verify_attempt133_conventions(spec, a133)
    oracle_reader_pin = spec["files"]["attempt133"]["historical_lens_source"]
    require_hash(path_of(oracle_reader_pin["path"]), oracle_reader_pin["sha256"])
    sys.path.insert(0, str(PROJECT / "scripts"))
    import read_oracle_logit_lens as oracle_reader
    import torch

    full, low, parents = load_blind_responses(spec, torch, a133, oracle_reader)
    log("fixing blind disagreement SVD bases")
    bases = build_blind_bases(full, low)
    # No oracle tensor has been opened to choose a rank, PC sign, or view weight.
    log("validating privileged provenance and computing oracle residual")
    validate_pinned_files(spec, blind_only=False)
    method_pins = spec["attempt133_provenance"]["immutable_input_sha256"]
    for key, path in (("downloaded_model_hashes", "downloaded-model-hashes.json"),
                      ("merged_model_hashes", "merged-model-hashes.json"),
                      ("oracle_probe_manifest", "oracle-probe-manifest.json")):
        require_hash(PROJECT / path, method_pins[key])
    locations = spec["attempt133_provenance"]["input_locations"]
    provenance = oracle_reader.verify_inputs_before_loading(
        Path(locations["base_tokenizer_directory"]),
        Path(locations["merged_checkpoint_directory"]),
        path_of(spec["artifacts"]["oracle"]["path"]))
    oracle = oracle_reader.load_and_validate_oracle(
        path_of(spec["artifacts"]["oracle"]["path"]), provenance, torch)
    oracle_tensor = a133.validate_tensor(
        oracle["difference"], spec["artifacts"]["oracle"]["difference_raw_sha256"],
        torch, oracle_reader.sha256_raw_float32_tensor)
    parallel64, residual64, residual_report = positionwise_residual(
        oracle_tensor.numpy(), parents["attempt134"].numpy(),
        spec["residual"]["orthogonality_relative_to_normO_normC_max"])
    log("projecting residual into the six fixed disagreement spaces")
    spans = {}
    for space in ("full_8", "low_8", "joint_16"):
        spans[space] = {}
        for range_name, (start, stop) in RANGES.items():
            spans[space][range_name] = analyze_residual_in_basis(
                bases[space][range_name], residual64[start:stop].reshape(-1))
    del bases
    classifications = {
        "low_residual_presence": {range_name: classify_low_residual_presence(
            spans["low_8"][range_name]["residual_squared_fraction_in_span"])
            for range_name in RANGES}}
    vectors = {"oracle": oracle_tensor, "attempt100": parents["attempt100"],
               "attempt134": parents["attempt134"],
               "parallel": torch.from_numpy(np.ascontiguousarray(parallel64.astype(np.float32))),
               "residual": torch.from_numpy(np.ascontiguousarray(residual64.astype(np.float32)))}
    log("direct Logit Lens")
    model, tokenizer, final_norm, lm_head = oracle_reader.load_local_model_and_tokenizer(
        Path(locations["base_tokenizer_directory"]),
        Path(locations["merged_checkpoint_directory"]), provenance, torch)
    if len(model.model.layers) != 28:
        raise ValueError("Merged Qwen3 layer count changed")
    historical_lens = json.loads(path_of(spec["files"]["attempt133"]["historical_lens"]["path"]).read_text())
    lens = direct_lens(vectors, a133, oracle_reader, final_norm, lm_head,
                       tokenizer, torch, historical_lens, provenance)
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for fixed Attempt133 Patchscope workload")
    log("loading model for Patchscope")
    model.to("cuda").eval()
    log("oracle-norm-matched Patchscope")
    directional = patchscope_mode(vectors, "directional_oracle_norm_matched",
                                  a133, model, tokenizer, torch)
    log("native-amplitude Patchscope")
    native = patchscope_mode(vectors, "native_amplitude", a133, model, tokenizer, torch)
    ratio = native_norm_ratios(oracle_tensor.numpy(), parents["attempt134"].numpy(),
                               parallel64, residual64)
    old_cos = directional["oracle_similarity_summaries"]["attempt100"]["plus"]["mean_positions_1_4_and_prompts"]["mean_cosine_to_oracle_same_sign"]
    low_cos = directional["oracle_similarity_summaries"]["attempt134"]["plus"]["mean_positions_1_4_and_prompts"]["mean_cosine_to_oracle_same_sign"]
    if old_cos is None or low_cos is None:
        raise ValueError("Undefined precommitted Attempt134-versus-Attempt100 Patchscope cosine")
    classifications["patchscope_attempt134_vs_attempt100"] = {
        "mean_positions_1_4_prompt_delta_logit_cosine_attempt100": old_cos,
        "mean_positions_1_4_prompt_delta_logit_cosine_attempt134": low_cos,
        "gain": low_cos - old_cos,
        "classification": classify_patchscope_gain(low_cos - old_cos)}
    interpretation = descriptive_interpretation(spans, lens, directional, classifications)
    log("revalidating inputs and writing result")
    validate_pinned_files(spec, blind_only=False)
    for name, tensor in vectors.items():
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError(f"Diagnostic vector changed before publication: {name}")
    result = {
        "format_version": 1, "attempt_id": ATTEMPT,
        "same_specimen_exploratory_method_development": True,
        "privileged_diagnostic_only": True, "clean_heldout_validation": False,
        "no_new_candidate_constructed": True,
        "spec_sha256": SPEC_SHA256, "runner_sha256": sha256_file(Path(__file__).resolve()),
        "input_provenance": spec["files"], "artifact_provenance": spec["artifacts"],
        "validated_checkpoint_provenance": {
            key: provenance[key] for key in
            ("base", "fine_tuned", "downloaded_model_hashes_sha256",
             "merged_model_hashes_sha256", "oracle_probe_manifest_sha256",
             "oracle_adl_manifest_sha256", "oracle_adl_sha256")},
        "observed_quantities": {
            "attempt134_positionwise_oracle_residual": residual_report,
            "blind_disagreement_residual_projections": spans,
            "direct_logit_lens": lens,
            "directional_oracle_norm_matched_patchscope": directional,
            "native_amplitude_patchscope": native,
            "native_vector_norm_ratios": ratio},
        "precommitted_classifications": classifications,
        "descriptive_interpretation": interpretation,
        "no_oracle_selected_pc_or_blind_rule": True,
        "attempt118_133_135_are_context_provenance_only": True,
    }
    require_output_absent(output)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    log(f"wrote {output}")


if __name__ == "__main__":
    try:
        run()
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
