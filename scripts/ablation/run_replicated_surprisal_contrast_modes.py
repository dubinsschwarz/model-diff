#!/usr/bin/env python3
"""Attempt140: blind replicated surprisal contrasts, then a CPU residual diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = "140_replicated_surprisal_contrast_modes"
SPEC_PATH = ROOT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "a157b12a5d9e6dfdeeb8cd531cd525a63e2197cbd9055ee72b6a65de7fc95b34"
ORDER = ("fineweb", "wikitext103_raw", "tinystories", "arxiv_document",
         "cc_news", "pg19", "codeparrot_clean", "ultrachat")
RANGES = {"positions_1_4": (1, 5), "positions_1_127": (1, 128)}
STRATA = {"low": (0, 1024), "middle": (1536, 2560), "high": (3072, 4096)}
SHAPE = (128, 2048)
MARKER = "REPLICATED_CONTRAST_MODES_FROZEN"


def log(message: str) -> None:
    print("[140] " + message, flush=True)


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


def raw_hash(value: np.ndarray) -> str:
    arr = np.asarray(value)
    if not arr.flags.c_contiguous or not np.isfinite(arr).all():
        raise ValueError("Hash requires finite C-contiguous array")
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def require_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"Output already exists: {path}")


def load_spec() -> dict[str, Any]:
    require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text())
    if (spec["attempt_id"] != ATTEMPT or spec["corpus_order"] != list(ORDER)
            or [row["attempt_number"] for row in spec["blind_inputs"]] != [100, 126, 129, 134]
            or spec["ranges"] != {name: list(bounds) for name, bounds in RANGES.items()}
            or spec["fineweb_replication"]["strata"] !=
            {name: list(bounds) for name, bounds in STRATA.items()}
            or spec["full_low_contrast"] != {
                "formula": "response_low_q - response_full_q", "difference_dtype": "cpu_float64",
                "position_unit_norm": True, "equal_weight_signed_consensus": True,
                "individual_sign_flips": False, "probe_rows": [0, 10000], "shape": list(SHAPE)}
            or spec["fineweb_replication"]["slope"] != "LOW - HIGH"
            or spec["fineweb_replication"]["curvature"] != "LOW - 2*MIDDLE + HIGH"
            or spec["fineweb_replication"]["probe_rows"] != [0, 1024]
            or spec["fineweb_replication"]["sign_flips"] is not False
            or spec["svd"]["rank_tolerance"] != "max(matrix.shape)*eps_float64*S[0]"
            or spec["svd"]["pc1_orientation"] !=
            "positive_dot_with_flattened_signed_equal_weight_consensus"
            or spec["svd"]["freeze_complete_rank_basis"] is not True
            or spec["barrier"]["marker"] != MARKER
            or spec["barrier"]["manifest_contains_historical_scores"] is not False
            or spec["no_model_loading"] is not True
            or spec["no_new_gradient_or_jvp"] is not True
            or spec["no_recovery_candidate"] is not True
            or spec["no_post_oracle_combination"] is not True
            or spec["overwrite"] is not False):
        raise ValueError("Attempt140 fixed blind plan changed")
    expected_names = {100: ["response_" + q for q in ORDER],
                      126: ["response_" + q for q in STRATA],
                      129: ["response_control"] + ["response_" + q for q in STRATA],
                      134: ["response_low_" + q for q in ORDER]}
    for row in spec["blind_inputs"]:
        number = row["attempt_number"]
        if (row["primitive_tensor_names"] != expected_names[number]
                or row["tensor_shape"] != list(SHAPE)
                or row["tensor_dtype"] != "contiguous_cpu_float32"
                or row["probe_rows"] != ([0, 10000] if number in (100, 134) else [0, 1024])):
            raise ValueError(f"Attempt{number} blind inventory changed")
    outputs = [path_of(p) for p in spec["outputs"].values()]
    if len(set(outputs)) != 3:
        raise ValueError("Output paths overlap")
    return spec


def import_pinned_source(pin: dict[str, str], name: str) -> Any:
    path = path_of(pin["path"])
    require_hash(path, pin["sha256"])
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def validate_blind_family(family: dict[str, Any], validator: Any, torch: Any) -> dict[str, Any]:
    """Use Attempt137's pinned blind-only manifest and tensor validators."""
    manifest = validator.validate_blind_manifest(family)
    number = family["attempt_number"]
    if number in (126, 129):
        source_spec = json.loads(path_of(family["spec"]["path"]).read_text())
        if (source_spec["strata"] != {k: {"rank_range": list(v)} for k, v in STRATA.items()}
                or {k: manifest["ranking"]["strata"][k]["rank_range"] for k in STRATA}
                != {k: list(v) for k, v in STRATA.items()}
                or manifest["probe"]["selected_rows"] != [0, 1024]):
            raise ValueError(f"Attempt{number} fixed FineWeb strata changed")
    pin = family["artifact"]
    require_hash(path_of(pin["path"]), pin["serialized_sha256"])
    artifact = torch.load(path_of(pin["path"]), map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != pin["tensor_order"]:
        raise ValueError(f"Attempt{number} artifact inventory changed")
    for name in pin["tensor_order"]:
        expected = (manifest["artifact"]["raw_tensors_sha256"][name] if number == 100
                    else manifest["responses"][name]["raw_sha256"])
        validator.validate_response_tensor(artifact[name], expected, torch)
    return artifact


def unitize_positions(contrasts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.ascontiguousarray(contrasts, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Malformed contrast stack")
    norms = np.linalg.norm(values, axis=2)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("Zero or nonfinite controlled contrast position norm")
    return np.ascontiguousarray(values / norms[:, :, None]), norms


def signed_consensus(unit: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if unit.ndim != 3 or unit.shape[0] != 8 or not np.isfinite(unit).all():
        raise ValueError("Exactly eight finite signed contrast views required")
    total = np.sum(unit, axis=0, dtype=np.float64)
    lengths = np.linalg.norm(total, axis=1)
    if np.any(lengths <= 0) or not np.isfinite(lengths).all():
        raise ValueError("Signed contrast consensus canceled at a position")
    consensus = np.ascontiguousarray(total / lengths[:, None])
    pairwise = np.einsum("qph,rph->pqr", unit, unit, optimize=True)
    return consensus, lengths / 8.0, np.ascontiguousarray(pairwise)


def fixed_svd(unit: np.ndarray, consensus: np.ndarray,
              start: int, stop: int) -> dict[str, Any]:
    matrix = np.ascontiguousarray(unit[:, start:stop, :].reshape(8, -1), dtype=np.float64)
    direction = np.ascontiguousarray(consensus[start:stop].reshape(-1), dtype=np.float64)
    left, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank < 1 or not np.isfinite(singular).all():
        raise ValueError("Degenerate blind contrast SVD")
    dot = float(np.dot(vh[0], direction))
    if not math.isfinite(dot) or abs(dot) <= tolerance * np.linalg.norm(direction):
        raise ValueError("PC1 cannot be oriented by blind signed consensus")
    if dot < 0:
        left[:, 0] *= -1
        vh[0] *= -1
    for i in range(1, rank):
        absolute = np.abs(left[:, i])
        pivot = int(np.flatnonzero(np.isclose(absolute, np.max(absolute),
                                                  rtol=0, atol=32 * np.finfo(np.float64).eps))[0])
        if left[pivot, i] < 0:
            left[:, i] *= -1
            vh[i] *= -1
    variance = singular * singular / np.sum(singular * singular, dtype=np.float64)
    error = np.linalg.norm(matrix - (left * singular[None, :]) @ vh) / np.linalg.norm(matrix)
    if not math.isfinite(error) or error > 1e-10:
        raise ValueError("Blind contrast SVD failed reconstruction")
    return {"left_loadings": np.ascontiguousarray(left),
            "singular_values": np.ascontiguousarray(singular),
            "variance_fractions": np.ascontiguousarray(variance),
            "basis": np.ascontiguousarray(vh[:rank]), "pc1": np.ascontiguousarray(vh[0]),
            "rank": rank, "rank_tolerance": float(tolerance),
            "matrix_shape": list(matrix.shape), "reconstruction_relative_error": float(error)}


def fineweb_contrasts(responses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    low, middle, high = (np.asarray(responses["response_" + name], dtype=np.float64)
                         for name in STRATA)
    if low.shape != middle.shape or middle.shape != high.shape or not np.isfinite(low).all() \
            or not np.isfinite(middle).all() or not np.isfinite(high).all():
        raise ValueError("Malformed FineWeb response strata")
    return {"slope": np.ascontiguousarray(low - high),
            "curvature": np.ascontiguousarray(low - 2.0 * middle + high),
            "low_middle": np.ascontiguousarray(low - middle),
            "middle_high": np.ascontiguousarray(middle - high),
            "low_high": np.ascontiguousarray(low - high)}


def build_blind_artifact(spec: dict[str, Any], validator: Any, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    loaded = {}
    for family in spec["blind_inputs"]:
        log(f"validating Attempt{family['attempt_number']} blind responses")
        loaded[family["attempt_number"]] = validate_blind_family(family, validator, torch)
    full = np.stack([loaded[100]["response_" + q].numpy() for q in ORDER]).astype(np.float64)
    low = np.stack([loaded[134]["response_low_" + q].numpy() for q in ORDER]).astype(np.float64)
    contrasts = np.ascontiguousarray(low - full)
    unit, norms = unitize_positions(contrasts)
    consensus, concentration, pairwise = signed_consensus(unit)
    bases = {name: fixed_svd(unit, consensus, *bounds) for name, bounds in RANGES.items()}
    fineweb = {label: fineweb_contrasts({name: loaded[number][name].numpy()
                                       for name in ("response_low", "response_middle", "response_high")})
               for label, number in (("old", 126), ("fresh", 129))}
    artifact = {"raw_full_to_low": torch.from_numpy(contrasts),
                "signed_equal_weight_consensus": torch.from_numpy(consensus),
                "svd": {name: {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                               for key, value in basis.items()}
                        for name, basis in bases.items()},
                "fineweb": {slice_name: {name: torch.from_numpy(value) for name, value in rows.items()}
                            for slice_name, rows in fineweb.items()},
                "metadata": {"corpus_order": list(ORDER), "ranges": spec["ranges"],
                             "full_low_formula": "LOW - FULL", "fineweb_slope": "LOW - HIGH",
                             "fineweb_curvature": "LOW - 2*MIDDLE + HIGH",
                             "difference_dtype": "cpu_float64", "position_unitization": True,
                             "probe_rows_part_a": [0, 10000], "probe_rows_part_b": [0, 1024]}}
    blind_report = {"raw_contrast_raw_sha256": {q: raw_hash(contrasts[i])
                                                 for i, q in enumerate(ORDER)},
                    "raw_contrast_norms": {q: float(np.linalg.norm(contrasts[i]))
                                           for i, q in enumerate(ORDER)},
                    "raw_contrast_position_norms": {q: norms[i].tolist() for i, q in enumerate(ORDER)},
                    "concentration_per_position": concentration.tolist(),
                    "concentration_positions_1_4_mean": float(np.mean(concentration[1:5])),
                    "concentration_positions_1_127_mean": float(np.mean(concentration[1:])),
                    "pairwise_signed_cosine_per_position": pairwise.tolist(),
                    "pairwise_signed_cosine_by_range": {
                        name: (np.einsum("qi,ri->qr",
                                             unit[:, start:stop].reshape(8, -1),
                                             unit[:, start:stop].reshape(8, -1)) / (stop - start)).tolist()
                        for name, (start, stop) in RANGES.items()},
                    "svd": {name: {"numerical_rank": row["rank"],
                                   "singular_values": row["singular_values"].tolist(),
                                   "variance_fractions": row["variance_fractions"].tolist(),
                                   "pc1_variance_fraction": float(row["variance_fractions"][0]),
                                   "pc_loadings": row["left_loadings"].tolist(),
                                   "rank_tolerance": row["rank_tolerance"]}
                            for name, row in bases.items()},
                    "fineweb_replication": fineweb_replication_report(fineweb)}
    return artifact, blind_report


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=np.float64).reshape(-1), np.asarray(right, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Malformed signed cosine inputs")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0 or not math.isfinite(denominator):
        raise ValueError("Zero or nonfinite signed cosine norm")
    return float(np.dot(a, b) / denominator)


def replication_metrics(old: np.ndarray, fresh: np.ndarray) -> dict[str, Any]:
    if old.shape != fresh.shape or old.ndim != 2 or not np.isfinite(old).all() \
            or not np.isfinite(fresh).all():
        raise ValueError("Malformed old/fresh contrast")
    per_position = [cosine(old[p], fresh[p]) for p in range(len(old))]
    return {"signed_cosine_per_position": per_position,
            "positions_1_4_mean_signed_cosine": float(np.mean(per_position[1:5])),
            "positions_1_127_mean_signed_cosine": float(np.mean(per_position[1:])),
            "positions_1_4_flattened_signed_cosine": cosine(old[1:5], fresh[1:5]),
            "positions_1_127_flattened_signed_cosine": cosine(old[1:], fresh[1:]),
            "old_norm": float(np.linalg.norm(old)),
            "fresh_norm": float(np.linalg.norm(fresh)),
            "old_norm_positions_1_4": float(np.linalg.norm(old[1:5])),
            "fresh_norm_positions_1_4": float(np.linalg.norm(fresh[1:5])),
            "old_norm_positions_1_127": float(np.linalg.norm(old[1:])),
            "fresh_norm_positions_1_127": float(np.linalg.norm(fresh[1:]))}


def fineweb_replication_report(fineweb: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    names = ("slope", "curvature", "low_middle", "middle_high", "low_high")
    report = {name: replication_metrics(fineweb["old"][name], fineweb["fresh"][name])
              for name in names}
    report["within_slice"] = {}
    for label in ("old", "fresh"):
        rows = fineweb[label]
        slope_norm = np.linalg.norm(rows["slope"])
        if slope_norm <= 0:
            raise ValueError("Zero FineWeb slope norm")
        report["within_slice"][label] = {
            "curvature_to_slope_norm_ratio": float(np.linalg.norm(rows["curvature"]) / slope_norm),
            "low_middle_vs_middle_high_flattened_signed_cosine":
                cosine(rows["low_middle"], rows["middle_high"]),
            "low_middle_vs_middle_high_positions_1_4_signed_cosine":
                cosine(rows["low_middle"][1:5], rows["middle_high"][1:5]),
            "low_middle_vs_middle_high_positions_1_127_signed_cosine":
                cosine(rows["low_middle"][1:], rows["middle_high"][1:]),
            "low_middle_vs_middle_high_same_direction":
                cosine(rows["low_middle"], rows["middle_high"]) > 0,
            "contrast_norms": {name: float(np.linalg.norm(rows[name])) for name in names}}
    return report


def tensor_hash_inventory(artifact: dict[str, Any], torch: Any) -> dict[str, dict[str, Any]]:
    result = {}

    def visit(value: Any, key: str) -> None:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or not value.is_contiguous() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Malformed frozen tensor: {key}")
            result[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                           "raw_sha256": raw_hash(value.numpy())}
        elif isinstance(value, dict):
            for child, item in value.items():
                visit(item, key + "/" + child if key else child)
    visit(artifact, "")
    return result


def publish_blind(spec: dict[str, Any], artifact: dict[str, Any],
                  blind_report: dict[str, Any], torch: Any) -> dict[str, Any]:
    outputs = {key: path_of(value) for key, value in spec["outputs"].items()}
    for path in outputs.values():
        require_absent(path)
    inventory = tensor_hash_inventory(artifact, torch)
    with outputs["artifact"].open("xb") as stream:
        torch.save(artifact, stream)
    manifest = {
        "format_version": 1, "attempt_id": ATTEMPT, "spec_sha256": SPEC_SHA256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "blind_input_artifacts": {str(row["attempt_number"]): row["artifact"]["serialized_sha256"]
                                  for row in spec["blind_inputs"]},
        "definitions": {"corpus_order": list(ORDER), "ranges": spec["ranges"],
                        "full_low_contrast": spec["full_low_contrast"],
                        "svd": spec["svd"], "fineweb_replication": spec["fineweb_replication"]},
        "artifact": {"path": spec["outputs"]["artifact"],
                     "serialized_sha256": sha256_file(outputs["artifact"]),
                     "raw_tensors": inventory},
        "blind_observations": blind_report,
        "barrier": {"marker": MARKER, "manifest_contains_historical_scores": False,
                    "all_blind_inputs_and_outputs_reloaded_and_revalidated": True}}
    with outputs["construction_manifest"].open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def reload_and_validate_blind(spec: dict[str, Any], manifest: dict[str, Any],
                              validator: Any, torch: Any) -> dict[str, Any]:
    path = path_of(spec["outputs"]["construction_manifest"])
    if json.loads(path.read_text()) != manifest or manifest["spec_sha256"] != SPEC_SHA256:
        raise ValueError("Published blind manifest changed")
    loaded = {family["attempt_number"]: validate_blind_family(family, validator, torch)
              for family in spec["blind_inputs"]}
    artifact_path = path_of(spec["outputs"]["artifact"])
    require_hash(artifact_path, manifest["artifact"]["serialized_sha256"])
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if (not isinstance(artifact, dict)
            or list(artifact) != ["raw_full_to_low", "signed_equal_weight_consensus",
                                    "svd", "fineweb", "metadata"]
            or tensor_hash_inventory(artifact, torch) != manifest["artifact"]["raw_tensors"]
            or artifact["metadata"]["corpus_order"] != list(ORDER)
            or artifact["metadata"]["ranges"] != spec["ranges"]
            or artifact["metadata"]["full_low_formula"] != "LOW - FULL"
            or tuple(artifact["raw_full_to_low"].shape) != (8, *SHAPE)
            or tuple(artifact["signed_equal_weight_consensus"].shape) != SHAPE):
        raise ValueError("Frozen blind contrast artifact changed")
    for name, (start, stop) in RANGES.items():
        svd = artifact["svd"][name]
        if (svd["basis"].shape != (svd["rank"], (stop - start) * SHAPE[1])
                or svd["pc1"].shape != ((stop - start) * SHAPE[1],)
                or not torch.equal(svd["pc1"], svd["basis"][0])):
            raise ValueError("Frozen blind SVD basis changed")
    if loaded:
        full = np.stack([loaded[100]["response_" + q].numpy() for q in ORDER]).astype(np.float64)
        low = np.stack([loaded[134]["response_low_" + q].numpy() for q in ORDER]).astype(np.float64)
        expected = np.ascontiguousarray(low - full)
        if (not np.array_equal(artifact["raw_full_to_low"].numpy(), expected)
                or manifest["blind_observations"]["raw_contrast_raw_sha256"] !=
                {q: raw_hash(expected[i]) for i, q in enumerate(ORDER)}):
            raise ValueError("Frozen LOW minus FULL contrasts do not reproduce")
        unit, _ = unitize_positions(expected)
        consensus, _, _ = signed_consensus(unit)
        if not np.array_equal(artifact["signed_equal_weight_consensus"].numpy(), consensus):
            raise ValueError("Frozen signed contrast consensus does not reproduce")
        for name, (start, stop) in RANGES.items():
            expected_svd = fixed_svd(unit, consensus, start, stop)
            actual_svd = artifact["svd"][name]
            if (actual_svd["rank"] != expected_svd["rank"]
                    or not np.allclose(actual_svd["singular_values"].numpy(),
                                       expected_svd["singular_values"], rtol=1e-12, atol=1e-12)
                    or not np.allclose(actual_svd["basis"].numpy(),
                                       expected_svd["basis"], rtol=1e-12, atol=1e-12)):
                raise ValueError(f"Frozen blind SVD {name} does not reproduce")
        for label, number in (("old", 126), ("fresh", 129)):
            expected_fineweb = fineweb_contrasts(
                {name: loaded[number][name].numpy()
                 for name in ("response_low", "response_middle", "response_high")})
            for name, value in expected_fineweb.items():
                if not np.array_equal(artifact["fineweb"][label][name].numpy(), value):
                    raise ValueError(f"Frozen FineWeb {label} {name} does not reproduce")
    return artifact


def single_direction_diagnostic(direction: np.ndarray, residual: np.ndarray,
                                oracle: np.ndarray, consensus: np.ndarray) -> dict[str, float]:
    signed = cosine(direction, residual)
    return {"signed_cosine_with_residual": signed,
            "absolute_cosine_with_residual": abs(signed),
            "residual_squared_fraction_in_single_direction": signed * signed,
            "signed_cosine_with_oracle": cosine(direction, oracle),
            "signed_cosine_with_attempt134_consensus": cosine(direction, consensus)}


def post_barrier_evaluate(spec: dict[str, Any], artifact: dict[str, Any],
                          manifest: dict[str, Any], validator: Any, torch: Any) -> dict[str, Any]:
    """The first historical and oracle reads are confined to this function."""
    log("loading and reproducing exact Attempt136 oracle residual")
    a136 = import_pinned_source(spec["oracle_after_barrier_only"]["attempt136_source"],
                                "attempt140_pinned_attempt136")
    wrapper = {"oracle_after_basis_only": spec["oracle_after_barrier_only"]}
    consensus_tensor = validate_blind_family(spec["blind_inputs"][3], validator, torch)[
        "low_surprisal_consensus_8"]
    residual, prior = validator.load_oracle_residual_after_bases(
        wrapper, consensus_tensor, torch, a136)
    pin = spec["oracle_after_barrier_only"]["oracle_artifact"]
    oracle_tensor = torch.load(path_of(pin["path"]), map_location="cpu", weights_only=True)["difference"]
    validator.validate_response_tensor(oracle_tensor, pin["difference_raw_sha256"], torch)
    oracle = oracle_tensor.numpy().astype(np.float64)
    consensus = consensus_tensor.numpy().astype(np.float64)
    signed = artifact["signed_equal_weight_consensus"].numpy()
    unit, _ = unitize_positions(artifact["raw_full_to_low"].numpy())
    by_range = {}
    for name, (start, stop) in RANGES.items():
        basis = artifact["svd"][name]
        r = np.ascontiguousarray(residual[start:stop].reshape(-1))
        info = {"matrix": np.ascontiguousarray(unit[:, start:stop].reshape(8, -1)),
                "u": basis["left_loadings"].numpy(),
                "singular_values": basis["singular_values"].numpy(),
                "vh": basis["basis"].numpy(), "rank": basis["rank"],
                "rank_tolerance": basis["rank_tolerance"],
                "view_names": ORDER,
                "svd_reconstruction_relative_error": basis["reconstruction_relative_error"]}
        span = a136.analyze_residual_in_basis(info, r)
        joint = prior["observed_quantities"]["blind_disagreement_residual_projections"][
            "joint_16"][name]["residual_squared_fraction_in_span"]
        by_range[name] = {
            "signed_equal_weight_contrast_consensus": single_direction_diagnostic(
                signed[start:stop], residual[start:stop], oracle[start:stop], consensus[start:stop]),
            "blind_svd_pc1": single_direction_diagnostic(
                basis["pc1"].numpy(), r, oracle[start:stop], consensus[start:stop]),
            "fixed_eight_contrast_span": {
                "numerical_rank": span["rank"],
                "residual_squared_norm_fraction_captured": span["residual_squared_fraction_in_span"],
                "residual_norm_fraction_captured": span["residual_norm_fraction_in_span"],
                "cumulative_residual_squared_fraction_by_blind_pc_order":
                    [row["cumulative_residual_squared_fraction"] for row in span["pcs"]],
                "pc_residual_diagnostics": span["pcs"]},
            "descriptive_existing_landmarks": {
                "attempt136_joint_full_low_disagreement_exact": joint,
                "attempt137_expanded_29_response_span_approximate_from_precommitment":
                    spec["baseline_landmarks_from_user"][
                        "attempt137_expanded_29_response_span_residual_squared_fraction"][name]}}
    return {"format_version": 1, "attempt_id": ATTEMPT,
            "spec_sha256": SPEC_SHA256,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "construction_manifest_sha256": sha256_file(path_of(spec["outputs"]["construction_manifest"])),
            "blind_artifact_serialized_sha256": manifest["artifact"]["serialized_sha256"],
            "no_recovery_candidate": True, "no_oracle_tuned_combination": True,
            "blind_observations": manifest["blind_observations"],
            "post_barrier_diagnostic": {"by_range": by_range,
                "attempt136_residual_aggregates": prior["observed_quantities"][
                    "attempt134_positionwise_oracle_residual"]["aggregates"],
                "fineweb_population_note": "FineWeb old/fresh modes use probe rows [0,1024); no comparison with full-10k Attempt136 residual is made.",
                "interpretation_limit": "The eight contrasts are derived from existing full and low response families; this span is not an additional independent set of dimensions."}}


def run() -> None:
    spec = load_spec()
    for path in spec["outputs"].values():
        require_absent(path_of(path))
    import torch
    validator = import_pinned_source(spec["blind_validator_source"], "attempt140_pinned_attempt137")
    log("validating pinned blind inputs and constructing replicated contrasts")
    artifact, blind_report = build_blind_artifact(spec, validator, torch)
    log("publishing and revalidating blind contrast modes")
    manifest = publish_blind(spec, artifact, blind_report, torch)
    artifact = reload_and_validate_blind(spec, manifest, validator, torch)
    print(MARKER, flush=True)
    result = post_barrier_evaluate(spec, artifact, manifest, validator, torch)
    output = path_of(spec["outputs"]["result"])
    require_absent(output)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    log(f"wrote {output}")


if __name__ == "__main__":
    run()
