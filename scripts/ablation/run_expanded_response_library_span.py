#!/usr/bin/env python3
"""Attempt137: privileged, CPU-only span diagnostic for frozen blind JVPs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ATTEMPT = "137_expanded_response_library_span"
SPEC_PATH = ROOT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "dfcadb0290b0d3314a2232452f6af3d43679871b3e7a6eb4f5acf00b5221d326"
FAMILY_ORDER = (100, 126, 129, 130, 131, 132, 134)
CORPORA = ("fineweb", "wikitext103_raw", "tinystories", "arxiv_document",
           "cc_news", "pg19", "codeparrot_clean", "ultrachat")
PRIMITIVE_NAMES = {
    100: tuple("response_" + corpus for corpus in CORPORA),
    126: ("response_low", "response_middle", "response_high"),
    129: ("response_control", "response_low", "response_middle", "response_high"),
    130: ("response_low2048", "response_control2048"),
    131: ("response_very_low", "response_next_low"),
    132: ("response_super_low", "response_next_super_low"),
    134: tuple("response_low_" + corpus for corpus in CORPORA),
}
RANGES = {"positions_1_4": (1, 5), "positions_1_127": (1, 128)}
SHAPE = (128, 2048)
PROBE_RAW_SHA256 = "73f56300fdd06bc8fafcfa5750fee7a586a5c5e6071a608667ffc526bbb90ac8"


def log(message: str) -> None:
    print("[137] " + message, flush=True)


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


def require_output_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"Result already exists: {path}")


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    require_hash(path, SPEC_SHA256)
    spec = json.loads(path.read_text())
    if (spec["attempt_id"] != ATTEMPT
            or spec["expected_prior_head_commit"] != "4b775e13ee40d2325feac43abcd5ec688938d5d1"
            or spec["chronological_family_order"] != list(FAMILY_ORDER)
            or spec["expected_primitive_count"] != 29
            or [family["attempt_number"] for family in spec["families"]] != list(FAMILY_ORDER)
            or spec["ranges"] != {key: list(value) for key, value in RANGES.items()}
            or spec["rank_tolerance"] != "max(D.shape)*eps_float64*S[0]"
            or spec["low_residual_presence_thresholds"] != {"strong": 0.50, "partial": 0.25}
            or spec["no_new_candidate_constructed"] is not True
            or spec["privileged_diagnostic_only"] is not True
            or spec["clean_heldout_validation"] is not False
            or spec["no_candidate_artifact"] is not True
            or spec["output"] != f"experiments/attempts/{ATTEMPT}/result.json"):
        raise ValueError("Attempt137 fixed diagnostic plan changed")
    expected_inventory = []
    for family in spec["families"]:
        number = family["attempt_number"]
        names = list(PRIMITIVE_NAMES[number])
        if (family["primitive_tensor_names"] != names
                or family["tensor_shape"] != list(SHAPE)
                or family["tensor_dtype"] != "contiguous_cpu_float32"
                or family["probe_source_raw_sha256"] != PROBE_RAW_SHA256
                or family["probe_rows"] != ([0, 10000] if number in (100, 134) else [0, 1024])):
            raise ValueError(f"Frozen primitive inventory changed for Attempt{number}")
        expected_inventory.extend({"attempt_number": number, "tensor_name": name,
                                   "raw_sha256": family["primitive_raw_sha256"][name]}
                                  for name in names)
    if (spec["primitive_inventory_order"] != expected_inventory
            or len(expected_inventory) != 29
            or len({row["raw_sha256"] for row in expected_inventory}) != 29):
        raise ValueError("Frozen primitive inventory/order contains missing or repeated responses")
    return spec


def import_attempt136(spec: dict[str, Any]) -> Any:
    pin = spec["oracle_after_basis_only"]["attempt136_source"]
    source = path_of(pin["path"])
    require_hash(source, pin["sha256"])
    loader = importlib.util.spec_from_file_location("attempt137_pinned_attempt136", source)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def validate_blind_manifest(family: dict[str, Any]) -> dict[str, Any]:
    """Validate only blind construction records; never open evaluation files."""
    for kind in ("spec", "source", "construction_manifest"):
        pin = family[kind]
        require_hash(path_of(pin["path"]), pin["sha256"])
    manifest = json.loads(path_of(family["construction_manifest"]["path"]).read_text())
    number = family["attempt_number"]
    artifact = manifest["artifact"] if number == 100 else manifest["candidate"]
    expected = family["artifact"]
    if (manifest["spec_sha256"] != family["spec"]["sha256"]
            or manifest["constructor_sha256"] != family["source"]["sha256"]
            or artifact["path"] != expected["path"]
            or artifact["serialized_sha256"] != expected["serialized_sha256"]
            or list(artifact.get("tensor_order", artifact.get("raw_tensors_sha256", {}))) != expected["tensor_order"]
            or manifest["probe"]["raw_tensor_sha256"] != PROBE_RAW_SHA256
            or manifest["readout"]["block_index"] != 13
            or manifest["readout"]["hidden_state_index"] != 14
            or manifest["readout"]["hidden_size"] != 2048
            or manifest["jvp"]["response_sign"] != "positive_J_delta"
            or manifest["tangent"]["direction"] != "delta = +alpha * g"
            or manifest.get("gradient_loss", manifest.get("generic_loss"))["type"]
            != "causal_next_token_cross_entropy"):
        raise ValueError(f"Attempt{number} blind construction provenance changed")
    if number == 100:
        if (any(manifest["artifact"]["raw_tensors_sha256"][name] !=
                family["primitive_raw_sha256"][name] for name in family["primitive_tensor_names"])
                or any(manifest[key] is not False for key in
                       ("historical_base_access", "oracle_adl_access", "prior_evaluation_access"))):
            raise ValueError("Attempt100 blind artifact/provenance changed")
    else:
        if (manifest["probe"].get("selected_rows", [0, 10000]) != family["probe_rows"]
                or {name: manifest["responses"][name]["raw_sha256"]
                    for name in family["primitive_tensor_names"]} != family["primitive_raw_sha256"]
                or manifest["barrier"]["manifest_contains_historical_scores"] is not False):
            raise ValueError(f"Attempt{number} blind response manifest changed")
    if number == 134 and manifest["responses"]["low_surprisal_consensus_8"]["raw_sha256"] != \
            family["consensus_raw_sha256"]:
        raise ValueError("Attempt134 frozen consensus hash changed")
    return manifest


def raw_float32_sha256(tensor: Any) -> str:
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def validate_response_tensor(tensor: Any, expected_hash: str, torch: Any) -> None:
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32 or not tensor.is_contiguous()
            or tuple(tensor.shape) != SHAPE or not bool(torch.isfinite(tensor).all())
            or raw_float32_sha256(tensor) != expected_hash):
        raise ValueError("Frozen primitive response tensor changed")


def load_blind_library(spec: dict[str, Any], torch: Any) -> tuple[np.ndarray, Any]:
    responses = []
    consensus = None
    for family in spec["families"]:
        number = family["attempt_number"]
        log(f"validating Attempt{number} blind responses")
        validate_blind_manifest(family)
        pin = family["artifact"]
        require_hash(path_of(pin["path"]), pin["serialized_sha256"])
        artifact = torch.load(path_of(pin["path"]), map_location="cpu", weights_only=True)
        if not isinstance(artifact, dict) or list(artifact) != pin["tensor_order"]:
            raise ValueError(f"Attempt{number} serialized tensor inventory/order changed")
        for name in family["primitive_tensor_names"]:
            tensor = artifact[name]
            validate_response_tensor(tensor, family["primitive_raw_sha256"][name], torch)
            responses.append(tensor.numpy())
        if number == 134:
            consensus = artifact["low_surprisal_consensus_8"]
            validate_response_tensor(consensus, family["consensus_raw_sha256"], torch)
    if len(responses) != spec["expected_primitive_count"] or consensus is None:
        raise ValueError("Incomplete frozen blind response library")
    return np.stack(responses), consensus


def unit_normalize_positions(responses: np.ndarray) -> np.ndarray:
    values = np.asarray(responses, dtype=np.float64)
    if (values.ndim != 3 or values.shape[1:] != SHAPE
            or not np.isfinite(values).all()):
        raise ValueError("Malformed primitive response stack")
    norms = np.linalg.norm(values, axis=2)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Zero/nonfinite primitive response position norm")
    return values / norms[:, :, None]


def chronological_boundaries(spec: dict[str, Any]) -> list[tuple[int, int]]:
    count = 0
    output = []
    for family in spec["families"]:
        count += len(family["primitive_tensor_names"])
        output.append((family["attempt_number"], count))
    if count != spec["expected_primitive_count"]:
        raise ValueError("Frozen family boundary count changed")
    return output


def build_blind_bases(unit: np.ndarray, spec: dict[str, Any], a136: Any) -> dict[str, Any]:
    """All SVDs, numerical ranks and PC signs are fixed before oracle access."""
    bases = {}
    names = tuple(f"{row['attempt_number']}:{row['tensor_name']}"
                  for row in spec["primitive_inventory_order"])
    for range_name, (start, stop) in RANGES.items():
        matrix = np.ascontiguousarray(unit[:, start:stop, :].reshape(len(names), -1))
        bases[range_name] = {}
        for number, count in chronological_boundaries(spec):
            log(f"blind SVD {range_name}: through Attempt{number} ({count} rows)")
            bases[range_name][number] = a136.blind_svd(
                matrix[:count], names[:count], rank_bound=count)
    return bases


def validate_attempt136_residual(report: dict[str, Any], reproduced: dict[str, Any]) -> None:
    original = report["observed_quantities"]["attempt134_positionwise_oracle_residual"]
    for old, new in zip(original["all_128_positions"], reproduced["all_128_positions"], strict=True):
        if old["position"] != new["position"]:
            raise ValueError("Attempt136 oracle residual position inventory changed")
        for key in ("alpha", "oracle_norm", "parallel_norm", "residual_norm",
                    "residual_squared_fraction"):
            if not math.isclose(old[key], new[key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Attempt136 oracle residual did not reproduce: {key}")
    for range_name in RANGES:
        for key in ("parallel_energy_fraction", "residual_energy_fraction"):
            old = original["aggregates"][range_name][key]
            new = reproduced["aggregates"][range_name][key]
            if not math.isclose(old, new, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Attempt136 aggregate oracle residual did not reproduce")


def load_oracle_residual_after_bases(spec: dict[str, Any], consensus: Any,
                                     torch: Any, a136: Any) -> tuple[np.ndarray, dict[str, Any]]:
    pins = spec["oracle_after_basis_only"]
    for key in ("attempt136_spec", "attempt136_source", "attempt136_result", "oracle_manifest"):
        require_hash(path_of(pins[key]["path"]), pins[key]["sha256"])
    require_hash(path_of(pins["oracle_artifact"]["path"]), pins["oracle_artifact"]["serialized_sha256"])
    prior_spec = json.loads(path_of(pins["attempt136_spec"]["path"]).read_text())
    prior = json.loads(path_of(pins["attempt136_result"]["path"]).read_text())
    if (prior["attempt_id"] != "136_privileged_residual_disagreement_patchscope"
            or prior["spec_sha256"] != pins["attempt136_spec"]["sha256"]
            or prior["runner_sha256"] != pins["attempt136_source"]["sha256"]
            or prior["no_new_candidate_constructed"] is not True
            or prior_spec["artifacts"]["oracle"] != pins["oracle_artifact"]
            or prior_spec["files"]["oracle_manifest"] != pins["oracle_manifest"]
            or prior["artifact_provenance"]["oracle"] != pins["oracle_artifact"]):
        raise ValueError("Attempt136 committed diagnostic identity changed")
    oracle_manifest = json.loads(path_of(pins["oracle_manifest"]["path"]).read_text())
    if (oracle_manifest["oracle_adl_sha256"] != pins["oracle_artifact"]["serialized_sha256"]
            or oracle_manifest["raw_tensors_sha256"]["difference"] !=
            pins["oracle_artifact"]["difference_raw_sha256"]):
        raise ValueError("Attempt136 oracle artifact/manifest link changed")
    artifact = torch.load(path_of(pins["oracle_artifact"]["path"]),
                          map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or "difference" not in artifact:
        raise ValueError("Malformed frozen oracle artifact")
    oracle = artifact["difference"]
    validate_response_tensor(oracle, pins["oracle_artifact"]["difference_raw_sha256"], torch)
    _parallel, residual, reproduced = a136.positionwise_residual(
        oracle.numpy(), consensus.numpy())
    validate_attempt136_residual(prior, reproduced)
    return residual, prior


def summarize_basis_projection(basis: dict[str, Any], residual: np.ndarray,
                               a136: Any) -> dict[str, Any]:
    projected = a136.analyze_residual_in_basis(basis, residual)
    singular = basis["singular_values"]
    total = float(np.sum(singular * singular, dtype=np.float64))
    variance = [(float(value * value / total) if total > 0 else 0.0) for value in singular]
    return {
        "primitive_response_count": basis["matrix"].shape[0],
        "numerical_rank": basis["rank"],
        "matrix_shape": list(basis["matrix"].shape),
        "rank_tolerance": basis["rank_tolerance"],
        "singular_values": [float(value) for value in singular],
        "singular_value_variance_fractions": variance,
        "residual_squared_norm_fraction_captured": projected["residual_squared_fraction_in_span"],
        "residual_norm_fraction_captured": projected["residual_norm_fraction_in_span"],
        "residual_projection_reconstruction_error": projected["residual_projection_reconstruction_error"],
        "pc_residual_diagnostics": projected["pcs"],
    }


def compare_attempt136(full: dict[str, Any], prior: dict[str, Any],
                       range_name: str) -> dict[str, Any]:
    old = prior["observed_quantities"]["blind_disagreement_residual_projections"]
    current = full["residual_squared_norm_fraction_captured"]
    if any(current + 1e-8 < old[space][range_name]["residual_squared_fraction_in_span"]
           for space in ("full_8", "low_8", "joint_16")):
        raise ValueError("Expanded primitive span unexpectedly loses Attempt136 disagreement coverage")
    return {space: {
        "attempt136_residual_squared_fraction":
        old[space][range_name]["residual_squared_fraction_in_span"],
        "expanded_library_minus_attempt136_fraction":
        current - old[space][range_name]["residual_squared_fraction_in_span"]}
        for space in ("full_8", "low_8", "joint_16")}


def run() -> None:
    spec = load_spec()
    output = path_of(spec["output"])
    require_output_absent(output)
    a136 = import_attempt136(spec)  # Numerical definitions only; no oracle data is read.
    import torch

    log("validating the 29 frozen blind primitive responses")
    raw, consensus = load_blind_library(spec, torch)
    log("normalizing each response at every position in CPU float64")
    unit = unit_normalize_positions(raw)
    del raw
    log("fixing full-library and chronological blind SVD bases")
    bases = build_blind_bases(unit, spec, a136)
    del unit
    # The first oracle-result and oracle-artifact reads occur after every basis is fixed.
    log("loading and reproducing the exact Attempt136 oracle residual")
    residual, prior = load_oracle_residual_after_bases(spec, consensus, torch, a136)
    full_results, cumulative, baselines, classes = {}, {}, {}, {}
    for range_name, (start, stop) in RANGES.items():
        r = np.ascontiguousarray(residual[start:stop].reshape(-1), dtype=np.float64)
        cumulative[range_name] = []
        for number, count in chronological_boundaries(spec):
            row = summarize_basis_projection(bases[range_name][number], r, a136)
            row["through_attempt"] = number
            row["residual_presence_classification"] = a136.classify_low_residual_presence(
                row["residual_squared_norm_fraction_captured"])
            cumulative[range_name].append(row)
        if any(later["residual_squared_norm_fraction_captured"] + 1e-8 <
               earlier["residual_squared_norm_fraction_captured"]
               for earlier, later in zip(cumulative[range_name], cumulative[range_name][1:])):
            raise ValueError("Chronological nested blind spans lost residual coverage")
        full_results[range_name] = cumulative[range_name][-1]
        baselines[range_name] = compare_attempt136(full_results[range_name], prior, range_name)
        classes[range_name] = full_results[range_name]["residual_presence_classification"]
    fractions = {name: full_results[name]["residual_squared_norm_fraction_captured"]
                 for name in RANGES}
    interpretation = (
        "The fixed existing blind response library contains a substantial part of the "
        "Attempt134 missing oracle component; selecting any useful component without oracle "
        "information is an unresolved separate problem."
        if max(fractions.values()) >= 0.25 else
        "Most Attempt134 missing oracle energy remains outside this fixed existing blind "
        "response library; a different observable or operator may be needed.")
    result = {
        "format_version": 1, "attempt_id": ATTEMPT,
        "spec_sha256": SPEC_SHA256, "runner_sha256": sha256_file(Path(__file__).resolve()),
        "same_specimen_exploratory_method_development": True,
        "privileged_diagnostic_only": True, "clean_heldout_validation": False,
        "no_new_candidate_constructed": True,
        "observed_quantities": {
            "primitive_response_count": 29,
            "primitive_inventory_order": spec["primitive_inventory_order"],
            "probe_population_note": spec["probe_population_note"],
            "expanded_library_by_range": full_results,
            "fixed_chronological_cumulative_coverage": cumulative,
            "attempt136_full_low_joint_disagreement_comparison": baselines,
            "attempt136_residual_aggregate_norms":
            prior["observed_quantities"]["attempt134_positionwise_oracle_residual"]["aggregates"],
        },
        "precommitted_classifications": {"expanded_library_residual_presence": classes},
        "descriptive_interpretation": {
            "summary": interpretation,
            "limits": ["This is an oracle-privileged span diagnostic, not a blind candidate.",
                       "No oracle-correlated PC or chronological increment is proposed as a blind rule.",
                       "Probe populations differ across response families; this is not held-out validation."]},
    }
    log("revalidating blind inputs and writing result")
    for family in spec["families"]:
        validate_blind_manifest(family)
        require_hash(path_of(family["artifact"]["path"]), family["artifact"]["serialized_sha256"])
    require_output_absent(output)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    log(f"wrote {output}")


if __name__ == "__main__":
    try:
        run()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
