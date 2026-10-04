#!/usr/bin/env python3
"""Attempt135: post-freeze, fixed equal-weight consensus of sixteen blind views."""

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
ATTEMPT = "135_fixed_16view_diverse_consensus"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "d5dbfcd8f6fb02f38c3fdf30592b18e98153d678da27b7b53c16c3beb77e5efb"
ORDER = ("fineweb", "wikitext103_raw", "tinystories", "arxiv_document",
         "cc_news", "pg19", "codeparrot_clean", "ultrachat")
VIEW_ORDER = tuple(name for corpus in ORDER
                   for name in ("response_" + corpus, "response_low_" + corpus))
CANDIDATE = "consensus_response_16"
MARKER = "SIXTEEN_VIEW_CONSENSUS_FROZEN"
SHAPE = (128, 2048)


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


def raw_float32_hash(value: np.ndarray) -> str:
    if (not isinstance(value, np.ndarray) or value.shape != SHAPE or
            value.dtype != np.float32 or not value.flags.c_contiguous or
            not np.isfinite(value).all()):
        raise ValueError("Response must be finite contiguous CPU float32 [128,2048]")
    return hashlib.sha256(value.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    require_hash(path, SPEC_SHA256)
    spec = json.loads(path.read_text())
    rule = spec["consensus"]
    evaluation = spec["evaluation"]
    if (spec["attempt_id"] != ATTEMPT or spec["corpus_order"] != list(ORDER)
            or spec["input_view_order"] != list(VIEW_ORDER)
            or spec["candidate"] != {"name": CANDIDATE, "shape": list(SHAPE),
                                     "dtype": "contiguous_cpu_float32", "tensor_order": [CANDIDATE]}
            or rule["weights"] != [1 / 16] * 16
            or rule["corpus_weights"] != [1 / 8] * 8
            or rule["within_corpus_full_low_weights"] != [0.5, 0.5]
            or rule["normalization_dtype"] != "cpu_float64"
            or any(rule[key] is not False for key in
                   ("final_renormalization", "sign_selection", "corpus_selection",
                    "position_selection", "data_dependent_weights"))
            or evaluation["primary_positions"] != [1, 2, 3, 4]
            or evaluation["secondary_positions"] != list(range(1, 128))
            or evaluation["clear_multiview_gain_threshold"] != {"primary": 0.005,
                                                                  "secondary": 0.005}
            or spec["barrier"] != {
                "marker": MARKER,
                "all_16_source_views_and_candidate_published_revalidated_before_oracle": True,
                "manifest_contains_historical_scores": False}
            or spec["workload"] != {"model_forwards": 0, "gradients": 0, "jvps": 0,
                                    "gpu_required": False, "source_response_views": 16}
            or spec["same_specimen_exploratory_method_development"] is not True
            or spec["clean_heldout_validation"] is not False):
        raise ValueError("Attempt135 frozen plan changed")
    return spec


def require_outputs_absent(spec: dict[str, Any]) -> None:
    paths = [path_of(spec["outputs"][key]) for key in
             ("candidate_path", "construction_manifest_path", "result_path")]
    if len(set(paths)) != 3:
        raise ValueError("Attempt135 output paths overlap")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise ValueError(f"Output already exists: {path}")


def validate_artifact(artifact: Any, tensor_order: list[str], hashes: dict[str, str],
                      torch: Any) -> dict[str, np.ndarray]:
    if (not isinstance(artifact, dict) or list(artifact) != tensor_order or
            set(hashes) != set(tensor_order)):
        raise ValueError("Frozen response artifact tensor order/inventory changed")
    arrays = {}
    for name in tensor_order:
        value = artifact[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or
                value.device.type != "cpu" or not value.is_contiguous() or
                tuple(value.shape) != SHAPE or not bool(torch.isfinite(value).all().item())):
            raise ValueError(f"Malformed frozen response: {name}")
        array = value.detach().numpy()
        if raw_float32_hash(array) != hashes[name]:
            raise ValueError(f"Frozen response raw hash changed: {name}")
        arrays[name] = array
    return arrays


def validate_blind_sources(spec: dict[str, Any], torch: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only the two frozen blind constructions, never scores or oracle."""
    pins = spec["source_artifacts"]
    metadata = {}
    arrays = {}
    for key in ("attempt100", "attempt134"):
        pin = pins[key]
        for field in ("spec", "constructor", "construction_manifest"):
            require_hash(path_of(pin[field + "_path"]), pin[field + "_sha256"])
        require_hash(path_of(pin["artifact_path"]), pin["artifact_serialized_sha256"])
        source_spec = json.loads(path_of(pin["spec_path"]).read_text())
        manifest = json.loads(path_of(pin["construction_manifest_path"]).read_text())
        if (manifest.get("spec_sha256") != pin["spec_sha256"] or
                manifest.get("constructor_sha256") != pin["constructor_sha256"] or
                source_spec.get("corpus_order", source_spec.get("consensus", {}).get("corpus_order")) != list(ORDER)):
            raise ValueError(f"Frozen {key} construction provenance changed")
        if key == "attempt100":
            if (source_spec["outputs"]["tensors"] != pin["tensor_order"] or
                    manifest.get("corpus_order") != list(ORDER) or
                    manifest.get("artifact", {}).get("path") != pin["artifact_path"] or
                    manifest["artifact"].get("serialized_sha256") != pin["artifact_serialized_sha256"] or
                    manifest["artifact"].get("raw_tensors_sha256") != pin["raw_tensors_sha256"] or
                    any(manifest.get(flag) is not False for flag in
                        ("historical_base_access", "adapter_access", "oracle_adl_access",
                         "prior_evaluation_access", "sign_selection",
                         "corpus_selection_after_results", "corpus_reweighting",
                         "candidate_rescaling", "post_result_position_selection"))):
                raise ValueError("Attempt100 blind construction changed")
        else:
            recorded = {name: row["raw_sha256"] for name, row in manifest["responses"].items()}
            if (source_spec["candidate"]["tensor_order"] != pin["tensor_order"] or
                    manifest.get("corpus_order") != list(ORDER) or
                    manifest.get("candidate", {}).get("tensor_order") != pin["tensor_order"] or
                    manifest["candidate"].get("serialized_sha256") != pin["artifact_serialized_sha256"] or
                    manifest["candidate"].get("path") != pin["artifact_path"] or
                    recorded != pin["raw_tensors_sha256"] or
                    manifest.get("barrier") != source_spec["barrier"] or
                    manifest["barrier"]["manifest_contains_historical_scores"] is not False or
                    any(manifest["information_policy"].get(flag) is not False for flag in
                        ("historical_base_access_before_barrier", "adapter_access_before_barrier",
                         "true_delta_access_before_barrier", "oracle_adl_access_before_barrier",
                         "prior_historical_scores_access_before_barrier",
                         "attempt133_oracle_semantic_access_before_barrier")) or
                    manifest.get("frozen_attempt100_construction_manifest_sha256") !=
                    pins["attempt100"]["construction_manifest_sha256"]):
                raise ValueError("Attempt134 blind construction changed")
        artifact = torch.load(path_of(pin["artifact_path"]), map_location="cpu", weights_only=True)
        arrays.update(validate_artifact(artifact, pin["tensor_order"], pin["raw_tensors_sha256"], torch))
        metadata[key] = manifest
    if (tuple("response_" + name for name in ORDER) !=
            tuple(name for name in VIEW_ORDER if name.startswith("response_") and not name.startswith("response_low_")) or
            tuple("response_low_" + name for name in ORDER) !=
            tuple(name for name in VIEW_ORDER if name.startswith("response_low_")) or
            metadata["attempt100"]["probe"]["sample_count"] != 10000 or
            metadata["attempt134"]["probe"]["sample_count"] != 10000 or
            any(metadata["attempt100"]["probe"][key] != metadata["attempt134"]["probe"][key]
                for key in ("serialized_sha256", "raw_tensor_sha256", "sequence_length"))):
        raise ValueError("Sixteen fixed views do not share the full frozen probe")
    return arrays, metadata


def position_cosines(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b = left.astype(np.float64), right.astype(np.float64)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    if (not np.isfinite(na).all() or not np.isfinite(nb).all() or
            np.any(na <= 0) or np.any(nb <= 0)):
        raise ValueError("Undefined oracle-free per-position cosine")
    return np.sum(a * b, axis=1, dtype=np.float64) / (na * nb)


def geometry_summary(values: np.ndarray) -> dict[str, Any]:
    if values.shape != (128,) or not np.isfinite(values).all():
        raise ValueError("Malformed per-position geometry")
    return {"all_128_positions": values.tolist(),
            "position_0": float(values[0]),
            "mean_positions_1_4": math.fsum(map(float, values[1:5])) / 4,
            "mean_positions_1_127": math.fsum(map(float, values[1:])) / 127}


def construct_consensus(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    if not all(name in arrays for name in (*VIEW_ORDER, "consensus_response_8", "low_surprisal_consensus_8")):
        raise ValueError("All sixteen fixed views and both parent consensuses are required")
    for name in (*VIEW_ORDER, "consensus_response_8", "low_surprisal_consensus_8"):
        raw_float32_hash(arrays[name])
    views = np.stack([arrays[name].astype(np.float64) for name in VIEW_ORDER], axis=0)
    norms = np.linalg.norm(views, axis=2)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Zero or nonfinite source-view position norm")
    unit = views / norms[:, :, None]
    consensus64 = np.sum(unit, axis=0, dtype=np.float64) / 16.0
    concentration = np.linalg.norm(consensus64, axis=1)
    if (not np.isfinite(concentration).all() or np.any(concentration <= 0) or
            np.any(concentration > 1 + 1e-12)):
        raise ValueError("Undefined sixteen-view consensus concentration")
    candidate = np.ascontiguousarray(consensus64.astype(np.float32))
    raw_float32_hash(candidate)
    old = arrays["consensus_response_8"]
    low = arrays["low_surprisal_consensus_8"]
    parent_pairs = {
        "attempt100_vs_attempt134": geometry_summary(position_cosines(old, low)),
        "consensus16_vs_attempt100": geometry_summary(position_cosines(candidate, old)),
        "consensus16_vs_attempt134": geometry_summary(position_cosines(candidate, low)),
    }
    old_low = parent_pairs["attempt100_vs_attempt134"]["all_128_positions"]
    new_old = parent_pairs["consensus16_vs_attempt100"]["all_128_positions"]
    new_low = parent_pairs["consensus16_vs_attempt134"]["all_128_positions"]
    between = [a + 1e-12 >= c and b + 1e-12 >= c
               for a, b, c in zip(new_old, new_low, old_low)]
    full_low = {corpus: geometry_summary(position_cosines(
        arrays["response_" + corpus], arrays["response_low_" + corpus])) for corpus in ORDER}
    return candidate, {"consensus16_per_position_concentration": geometry_summary(concentration),
                       "parent_and_new_consensus_pairwise_cosines": parent_pairs,
                       "angular_between_parent_directions_descriptive": {
                           "definition": "cos(new,each_parent) >= cos(parent,parent)-1e-12",
                           "all_128_positions": between,
                           "position_0": between[0],
                           "count_positions_1_4": sum(between[1:5]),
                           "count_positions_1_127": sum(between[1:])},
                       "per_corpus_full_vs_low_position_cosines": full_low}


def validate_published(spec: dict[str, Any], manifest: dict[str, Any], torch: Any) -> tuple[Any, dict[str, np.ndarray]]:
    candidate_path = path_of(spec["outputs"]["candidate_path"])
    manifest_path = path_of(spec["outputs"]["construction_manifest_path"])
    if (sha256_file(candidate_path) != manifest["candidate"]["serialized_sha256"] or
            json.loads(manifest_path.read_text()) != manifest or
            manifest["barrier"] != spec["barrier"]):
        raise ValueError("Published Attempt135 construction changed")
    artifact = torch.load(candidate_path, map_location="cpu", weights_only=True)
    frozen = validate_artifact(artifact, [CANDIDATE], {CANDIDATE: manifest["candidate"]["raw_sha256"]}, torch)
    sources, _ = validate_blind_sources(spec, torch)
    rebuilt, rebuilt_geometry = construct_consensus(sources)
    if not np.array_equal(rebuilt, frozen[CANDIDATE]) or rebuilt_geometry != manifest["oracle_free_geometry"]:
        raise ValueError("Published sixteen-view consensus differs from frozen source views")
    return artifact[CANDIDATE], sources


def classify(primary_gain: float, secondary_gain: float) -> str:
    if not all(math.isfinite(value) for value in (primary_gain, secondary_gain)):
        raise ValueError("Nonfinite multiview gain")
    if primary_gain >= 0.005 and secondary_gain >= 0.005:
        return "clear_multiview_gain"
    if primary_gain > 0 and secondary_gain > 0:
        return "positive_multiview_gain"
    return "no_multiview_gain"


def compare_scores(new: dict[str, Any], old: dict[str, Any], low: dict[str, Any]) -> dict[str, Any]:
    primary = "positions_1_4_mean_cosine"
    secondary = "positions_1_127_mean_cosine"
    best_p = max(old[primary], low[primary])
    best_s = max(old[secondary], low[secondary])
    gain_p, gain_s = new[primary] - best_p, new[secondary] - best_s
    return {"best_parent_primary": best_p, "best_parent_secondary": best_s,
            "primary_gain_vs_best_parent": gain_p,
            "secondary_gain_vs_best_parent": gain_s,
            "classification": classify(gain_p, gain_s),
            "positions_1_127_wins_vs_attempt100": sum(
                a > b for a, b in zip(new["all_128_position_cosines"][1:],
                                        old["all_128_position_cosines"][1:])),
            "positions_1_127_wins_vs_attempt134": sum(
                a > b for a, b in zip(new["all_128_position_cosines"][1:],
                                        low["all_128_position_cosines"][1:]))}


def evaluate_after_barrier(spec: dict[str, Any], candidate: Any, sources: dict[str, np.ndarray],
                           construction: dict[str, Any], torch: Any) -> dict[str, Any]:
    """Historical artifact and parent scores are first opened here."""
    pins = spec["post_barrier_inputs"]
    evaluator = import_pinned(path_of(pins["attempt100_evaluator_path"]),
                              pins["attempt100_evaluator_sha256"], "attempt135_attempt100_evaluator")
    for key in ("attempt100_evaluation", "attempt134_result", "attempt100_evaluation_spec",
                "oracle_manifest", "oracle_artifact"):
        require_hash(path_of(pins[key + "_path"]), pins[key + "_sha256"])
    oracle_paths = {key: evaluator.path_of(row["path"])
                    for key, row in evaluator.FROZEN_SPEC["oracle"]["provenance_inputs"].items()}
    oracle_hashes = {key: row["sha256"]
                     for key, row in evaluator.FROZEN_SPEC["oracle"]["provenance_inputs"].items()}
    for key, path in oracle_paths.items():
        require_hash(path, oracle_hashes[key])
    oracle_attempt = evaluator.prior.load_json_object(oracle_paths["attempt_spec"], "Attempt014 oracle spec")
    oracle_manifest = evaluator.validate_oracle_manifest(evaluator.prior.validate_oracle_provenance(
        oracle_paths, oracle_hashes, oracle_attempt))
    if (oracle_manifest["oracle_adl_sha256"] != pins["oracle_artifact_sha256"] or
            oracle_manifest["raw_tensors_sha256"]["difference"] != pins["oracle_difference_raw_sha256"]):
        raise ValueError("Frozen Attempt100/134 oracle provenance changed")
    difference = evaluator.prior.load_and_validate_oracle(
        path_of(pins["oracle_artifact_path"]), {"oracle": {"oracle_manifest": oracle_manifest}}, torch)["difference"]
    old_result = json.loads(path_of(pins["attempt100_evaluation_path"]).read_text())
    low_result = json.loads(path_of(pins["attempt134_result_path"]).read_text())
    import_blind = import_pinned(path_of(spec["source_artifacts"]["attempt134"]["constructor_path"]),
                                 spec["source_artifacts"]["attempt134"]["constructor_sha256"],
                                 "attempt135_attempt134_metric")
    old_validated = import_blind.old_evaluation_reports(old_result)
    old_rows = [row for row in old_result["candidates"] if row["candidate"] == "consensus_response_8"]
    if (len(old_rows) != 1 or old_result["provenance"]["construction_manifest_sha256"] !=
            spec["source_artifacts"]["attempt100"]["construction_manifest_sha256"] or
            old_result["provenance"]["construction_artifact_sha256"] !=
            spec["source_artifacts"]["attempt100"]["artifact_serialized_sha256"] or
            low_result["construction_manifest_sha256"] !=
            spec["source_artifacts"]["attempt134"]["construction_manifest_sha256"] or
            low_result["candidate_serialized_sha256"] !=
            spec["source_artifacts"]["attempt134"]["artifact_serialized_sha256"]):
        raise ValueError("Frozen parent evaluation provenance changed")
    old_frozen = old_rows[0]
    low_frozen = low_result["new_signed_position_reports"]["low_surprisal_consensus_8"]
    if (old_validated["consensus_response_8"]["positions_1_4_mean_cosine"] !=
            old_frozen["positions_1_4_mean_cosine"] or
            old_validated["consensus_response_8"]["positions_1_127_mean_cosine"] !=
            old_frozen["positions_1_127_mean_cosine"]):
        raise ValueError("Attempt100 committed signed report changed")
    old = evaluator.position_cosines(torch.from_numpy(sources["consensus_response_8"]), difference)
    low = evaluator.position_cosines(torch.from_numpy(sources["low_surprisal_consensus_8"]), difference)
    for values, frozen, source_name in ((old, old_frozen, "Attempt100"), (low, low_frozen, "Attempt134")):
        stored = ([row["cosine_similarity"] for row in frozen["positions"]]
                  if source_name == "Attempt100" else frozen["all_128_position_cosines"])
        if (len(stored) != 128 or len(values) != 128 or
                any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-10)
                    for a, b in zip(values, stored))):
            raise ValueError(f"Frozen {source_name} signed position scores changed")
    report = import_blind.signed_report(CANDIDATE, candidate, difference, evaluator)
    old_report = import_blind.signed_report("consensus_response_8", torch.from_numpy(
        sources["consensus_response_8"]), difference, evaluator)
    low_report = import_blind.signed_report("low_surprisal_consensus_8", torch.from_numpy(
        sources["low_surprisal_consensus_8"]), difference, evaluator)
    # The parent comparisons use the exact committed scores after confirming
    # that the frozen tensors reproduce all 128 positions against this oracle.
    for report, frozen, source_name in ((old_report, old_frozen, "Attempt100"),
                                        (low_report, low_frozen, "Attempt134")):
        committed = ([row["cosine_similarity"] for row in frozen["positions"]]
                     if source_name == "Attempt100" else frozen["all_128_position_cosines"])
        report["position_0_cosine"] = committed[0]
        report["positions_1_4_individual_cosines"] = list(committed[1:5])
        report["positions_1_4_mean_cosine"] = frozen["positions_1_4_mean_cosine"]
        report["positions_1_127_mean_cosine"] = frozen["positions_1_127_mean_cosine"]
        report["all_128_position_cosines"] = list(committed)
    return {"format_version": 1, "attempt_id": ATTEMPT,
            "same_specimen_exploratory_method_development": True,
            "clean_heldout_validation": False,
            "spec_sha256": SPEC_SHA256,
            "constructor_sha256": sha256_file(Path(__file__).resolve()),
            "construction_manifest_sha256": sha256_file(path_of(spec["outputs"]["construction_manifest_path"])),
            "candidate_serialized_sha256": construction["candidate"]["serialized_sha256"],
            "oracle_artifact_sha256": pins["oracle_artifact_sha256"],
            "oracle_difference_raw_sha256": pins["oracle_difference_raw_sha256"],
            "signed_position_reports": {CANDIDATE: report, "attempt100_consensus": old_report,
                                        "attempt134_low_consensus": low_report},
            "comparison": compare_scores(report, old_report, low_report),
            "geometric_interpretation_descriptive_only": construction["oracle_free_geometry"],
            "no_candidate_selection_rescaling_or_reweighting": True}


def run() -> None:
    spec = load_spec()
    require_outputs_absent(spec)
    import torch
    sources, _ = validate_blind_sources(spec, torch)
    candidate_array, geometry = construct_consensus(sources)
    artifact = {CANDIDATE: torch.from_numpy(candidate_array)}
    require_outputs_absent(spec)
    candidate_path = path_of(spec["outputs"]["candidate_path"])
    manifest_path = path_of(spec["outputs"]["construction_manifest_path"])
    with candidate_path.open("xb") as stream:
        torch.save(artifact, stream)
    construction = {
        "format_version": 1, "attempt_id": ATTEMPT,
        "spec_sha256": SPEC_SHA256, "constructor_sha256": sha256_file(Path(__file__).resolve()),
        "source_artifacts": spec["source_artifacts"], "corpus_order": list(ORDER),
        "input_view_order": list(VIEW_ORDER), "consensus": spec["consensus"],
        "oracle_free_geometry": geometry,
        "candidate": {"path": str(candidate_path), "tensor_order": [CANDIDATE],
                      "shape": list(SHAPE), "dtype": "contiguous_cpu_float32",
                      "serialized_sha256": sha256_file(candidate_path),
                      "raw_sha256": raw_float32_hash(candidate_array)},
        "barrier": spec["barrier"],
        "historical_base_access_before_barrier": False,
        "oracle_access_before_barrier": False,
        "prior_scores_access_before_barrier": False,
        "attempt133_oracle_semantics_access_before_barrier": False,
        "same_specimen_exploratory_method_development": True,
        "clean_heldout_validation": False,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(construction, stream, indent=2, allow_nan=False)
        stream.write("\n")
    frozen_candidate, frozen_sources = validate_published(spec, construction, torch)
    print(MARKER, flush=True)
    result = evaluate_after_barrier(spec, frozen_candidate, frozen_sources, construction, torch)
    result_path = path_of(spec["outputs"]["result_path"])
    if result_path.exists() or result_path.is_symlink():
        raise ValueError("Attempt135 result already exists")
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    try:
        run()
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
