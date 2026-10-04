#!/usr/bin/env python3
"""Attempt134: fixed within-corpus LOW1024 responses and diverse-8 consensus."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = "134_diverse8_low_surprisal_consensus"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "bd7ffc4f3c1e9213ab5f15f2edf97f08ad098c05735024b6dccf1609bbeb22b1"
ORDER = ("fineweb", "wikitext103_raw", "tinystories", "arxiv_document",
         "cc_news", "pg19", "codeparrot_clean", "ultrachat")
RESPONSE_NAMES = tuple("response_low_" + name for name in ORDER)
CONSENSUS_NAME = "low_surprisal_consensus_8"
CANDIDATE_NAMES = (*RESPONSE_NAMES, CONSENSUS_NAME)
MARKER = "DIVERSE8_LOW_SURPRISAL_CONSENSUS_FROZEN"
POPULATION = 4096
LOW_COUNT = 1024
RANK_BATCHES = 512
GRADIENT_BATCHES = 128
JVP_BATCH_SIZE = 64
JVP_BATCHES = 157


def jvp_batch_sizes(sample_count: int, batch_size: int) -> list[int]:
    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("JVP population and batch size must be positive")
    return [min(batch_size, sample_count - start)
            for start in range(0, sample_count, batch_size)]


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


def import_pinned(path: Path, digest: str, name: str) -> Any:
    require_hash(path, digest)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def require_outputs_absent(spec: dict[str, Any]) -> None:
    outputs = [path_of(spec["outputs"][key]) for key in
               ("candidate_path", "construction_manifest_path", "result_path")]
    if len(set(outputs)) != 3:
        raise ValueError("Attempt134 output paths overlap")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise ValueError(f"Output already exists: {path}")


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    require_hash(path, SPEC_SHA256)
    spec = json.loads(path.read_text())
    rank = spec["ranking"]
    work = spec["workload"]
    if (spec["attempt_id"] != ATTEMPT or spec["corpus_order"] != list(ORDER)
            or [row["name"] for row in spec["corpora"]] != list(ORDER)
            or rank["population_rows"] != [0, POPULATION]
            or rank["population_size_per_corpus"] != POPULATION
            or rank["prediction_tokens_per_row"] != 127
            or rank["batch_size"] != 8 or rank["forward_batches_per_new_corpus"] != RANK_BATCHES
            or rank["newly_scored_corpora"] != list(ORDER[1:])
            or rank["fineweb_reuse_attempt126"] is not True
            or rank["low_rank_range"] != [0, LOW_COUNT]
            or rank["within_corpus_only"] is not True
            or rank["full_surprisal_and_rank_freeze_before_gradients"] is not True
            or spec["gradient_loss"]["sample_count"] != LOW_COUNT
            or spec["gradient_loss"]["number_of_batches"] != GRADIENT_BATCHES
            or spec["gradient_loss"]["total_prediction_tokens"] != LOW_COUNT * 127
            or spec["tangent"]["direction"] != "delta = +alpha * g"
            or spec["tangent"]["target_relative_frobenius"] != 0.00125
            or spec["probe"]["sample_count"] != 10000
            or spec["probe"]["batch_size"] != JVP_BATCH_SIZE
            or spec["probe"]["jvp_batches_per_corpus"] != JVP_BATCHES
            or len(jvp_batch_sizes(spec["probe"]["sample_count"], JVP_BATCH_SIZE)) != JVP_BATCHES
            or spec["jvp_memory_smoke"] != {
                "cli_flag": "--jvp-memory-smoke", "probe_rows": [0, 64],
                "synthetic_tangent": "constant_float32_1e-7_for_all_98_eligible_linear_weights",
                "jvp_calls": 1, "read_only": True, "historical_access": False,
                "automatic_batch32_fallback": False}
            or spec["readout"]["block_index"] != 13
            or spec["readout"]["hidden_state_index"] != 14
            or spec["eligible_tensors"]["expected_matrix_count"] != 98
            or spec["jvp"]["strict_forward_ad"] is not True
            or spec["consensus"]["weights"] != [1 / 8] * 8
            or spec["consensus"]["renormalize_final"] is not False
            or spec["candidate"]["tensor_order"] != list(CANDIDATE_NAMES)
            or spec["barrier"] != {
                "marker": MARKER,
                "all_8_rankings_rows_gradients_responses_and_consensus_published_revalidated_before_oracle": True,
                "manifest_contains_historical_scores": False}
            or work != {"new_surprisal_forward_batches": 7 * RANK_BATCHES,
                        "gradient_backward_batches": 8 * GRADIENT_BATCHES,
                        "jvp_probe_batches": 8 * JVP_BATCHES,
                        "jvp_batches_per_corpus": JVP_BATCHES,
                        "new_corpus_downloads": 0, "new_corpus_freezes": 0,
                        "oracle_recomputation": False}
            or spec["same_specimen_exploratory_method_development"] is not True
            or spec["clean_heldout_validation"] is not False):
        raise ValueError("Attempt134 frozen plan mismatch")
    return spec


def validate_blind_sources(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read only generic construction inputs. Do not touch oracle or results."""
    inputs = spec["inputs"]
    for attempt, keys in (("attempt100", ("spec", "constructor", "freezer", "construction_manifest", "corpus_lock")),
                          ("attempt014", ("spec", "constructor", "construction_manifest")),
                          ("attempt126", ("spec", "source", "construction_manifest"))):
        for key in keys:
            require_hash(path_of(inputs[attempt][key + "_path"]), inputs[attempt][key + "_sha256"])
    for corpus in spec["corpora"]:
        require_hash(path_of(corpus["manifest_path"]), corpus["manifest_sha256"])
        if corpus["source"] == "attempt014":
            require_hash(path_of(corpus["spec_path"]), corpus["spec_sha256"])
    m100 = json.loads(path_of(inputs["attempt100"]["construction_manifest_path"]).read_text())
    m014 = json.loads(path_of(inputs["attempt014"]["construction_manifest_path"]).read_text())
    m126 = json.loads(path_of(inputs["attempt126"]["construction_manifest_path"]).read_text())
    s100 = json.loads(path_of(inputs["attempt100"]["spec_path"]).read_text())
    s014 = json.loads(path_of(inputs["attempt014"]["spec_path"]).read_text())
    pins100, pins014, pins126 = inputs["attempt100"], inputs["attempt014"], inputs["attempt126"]
    if (s100["consensus"] != spec["consensus"]
            or s100["tangent"] != spec["tangent"]
            or spec["gradient_loss"] != {
                **s100["generic_loss"], "sample_count": LOW_COUNT,
                "total_prediction_tokens": LOW_COUNT * 127,
                "number_of_batches": GRADIENT_BATCHES}
            or s100["jvp"] != spec["jvp"]
            or s100["readout"] != spec["readout"]
            or s100["model"] != spec["model"]
            or s100["eligible_tensors"] != spec["eligible_tensors"]
            or s100["canonical_checkpoint_files"] != spec["final_checkpoint"]["files"]
            or s100["defaults"]["merged_model_directory"] != spec["final_checkpoint"]["directory"]
            or s100["probe"]["batch_size"] != 32
            or any(s100["probe"][key] != spec["probe"][key] for key in s100["probe"]
                   if key != "batch_size")
            or s100["defaults"]["probe_path"] != spec["probe"]["path"]
            or m014["corpus_order"] != list(ORDER[:3])
            or m014["spec_sha256"] != pins014["spec_sha256"]
            or m014["constructor_script_sha256"] != pins014["constructor_sha256"]
            or m014["source_checkpoint"] != spec["final_checkpoint"]["files"]
            or m100["spec_sha256"] != pins100["spec_sha256"]
            or m100["constructor_sha256"] != pins100["constructor_sha256"]
            or m100["freezer_sha256"] != pins100["freezer_sha256"]
            or m100["corpus_lock_sha256"] != pins100["corpus_lock_sha256"]
            or m100["source_checkpoint"] != spec["final_checkpoint"]["files"]
            or m100["corpus_order"] != list(ORDER)
            or m100["probe"] != s100["probe"]
            or m100["probe_sha256"] != spec["probe"]["serialized_sha256"]
            or m100["old_attempt014"]["manifest_path"] != pins014["construction_manifest_sha256"]
            or m100["artifact"]["serialized_sha256"] != pins100["response_artifact_sha256"]
            or m100["artifact"]["raw_tensors_sha256"]["merged_mean"] != pins100["merged_mean_raw_sha256"]
            or m126["spec_sha256"] != pins126["spec_sha256"]
            or m126["constructor_sha256"] != pins126["source_sha256"]
            or m126["frozen_attempt100_construction_manifest_sha256"] != pins100["construction_manifest_sha256"]
            or m126["ranking"]["surprisal_raw_float64_sha256"] != pins126["fineweb_surprisal_raw_float64_sha256"]
            or m126["ranking"]["rank_raw_int64_sha256"] != pins126["fineweb_rank_raw_int64_sha256"]
            or m126["ranking"]["strata"]["low"]["ordered_row_indices_raw_sha256"] != pins126["fineweb_low_row_raw_int64_sha256"]):
        raise ValueError("Frozen Attempt014/100/126 construction provenance conflict")
    if ([row["name"] for row in s014["corpora"]] != list(ORDER[:3])
            or [row["name"] for row in s100["new_corpora"]] != list(ORDER[3:])
            or [row["name"] for row in spec["corpora"]] != list(ORDER)):
        raise ValueError("Frozen eight-corpus order changed")
    for index, row in enumerate(spec["corpora"]):
        if index < 3:
            source = s014["corpora"][index]
            if any(row[key] != source[key] for key in
                   ("name", "spec_path", "spec_sha256", "manifest_path", "tokens_path")):
                raise ValueError(f"Attempt014 corpus spec changed: {row['name']}")
            mrow = m014["corpora"][index]
            if any(mrow[key] != row[value] for key, value in
                   (("serialized_sha256", "serialized_sha256"),
                    ("raw_tensor_sha256", "raw_tensor_sha256"),
                    ("manifest_sha256", "manifest_sha256"))):
                raise ValueError(f"Attempt014 corpus manifest changed: {row['name']}")
        else:
            source = s100["new_corpora"][index - 3]
            frozen = m100["new_corpora"][index - 3]
            if (source["name"] != row["name"] or source["manifest_path"] != row["manifest_path"]
                    or source["tokens_path"] != row["tokens_path"]
                    or frozen["corpus_manifest_sha256"] != row["manifest_sha256"]
                    or frozen["corpus_artifact_sha256"] != row["serialized_sha256"]
                    or frozen["corpus_revision"] != row["revision"]):
                raise ValueError(f"Attempt100 corpus manifest changed: {row['name']}")
    fineweb = spec["corpora"][0]
    if (m126["frozen_attempt005"]["tokens_path"] != fineweb["tokens_path"]
            or m126["frozen_attempt005"]["serialized_sha256"] != fineweb["serialized_sha256"]
            or m126["frozen_attempt005"]["raw_tensor_sha256"] != fineweb["raw_tensor_sha256"]
            or m126["frozen_attempt005"]["corpus_spec_sha256"] != fineweb["spec_sha256"]
            or m126["frozen_attempt005"]["corpus_manifest_sha256"] != fineweb["manifest_sha256"]):
        raise ValueError("Attempt126 FineWeb ranking is not for Attempt100 FineWeb tokens")
    return m100, m014, m126


def load_blind_artifacts(spec: dict[str, Any], a100: Any, torch: Any) -> tuple[dict[str, Any], Any, Path]:
    """Reuse both existing corpus validators and the exact full probe loader."""
    s100 = a100.freezer.load_spec()
    model_dir = path_of(spec["final_checkpoint"]["directory"])
    if a100.old.checkpoint_file_records(model_dir) != spec["final_checkpoint"]["files"]:
        raise ValueError("Canonical final checkpoint inventory changed")
    lock = a100.freezer.validate_lock(a100.old.load_json_object(
        path_of(spec["inputs"]["attempt100"]["corpus_lock_path"]), "Attempt100 corpus lock"), s100)
    s014 = a100.old.load_json_object(path_of(spec["inputs"]["attempt014"]["spec_path"]), "Attempt014 spec")
    corpora = {}
    for index, row in enumerate(spec["corpora"]):
        if index < 3:
            tokens, _, provenance = a100.old.validate_frozen_corpus(s014["corpora"][index], model_dir, torch)
            if (provenance["serialized_sha256"] != row["serialized_sha256"]
                    or provenance["raw_tensor_sha256"] != row["raw_tensor_sha256"]):
                raise ValueError(f"Frozen Attempt014 corpus bytes changed: {row['name']}")
        else:
            tokens, manifest = a100.freezer.validate_corpus(s100["new_corpora"][index - 3],
                                                             s100, lock, model_dir, torch)
            if (manifest["artifact"]["serialized_sha256"] != row["serialized_sha256"]
                    or manifest["artifact"]["raw_tensor_sha256"] != row["raw_tensor_sha256"]):
                raise ValueError(f"Frozen Attempt100 corpus bytes changed: {row['name']}")
        if (tuple(tokens.shape) != (POPULATION, 128) or tokens.dtype != torch.int64
                or tokens.device.type != "cpu" or not tokens.is_contiguous()):
            raise ValueError(f"Malformed full frozen corpus: {row['name']}")
        corpora[row["name"]] = tokens
    probe = a100.old.load_probe(path_of(spec["probe"]["path"]), spec["probe"], torch)
    if tuple(probe.shape) != (10000, 128) or not probe.is_contiguous():
        raise ValueError("Attempt100 full 10000-row probe changed")
    return corpora, probe, model_dir


def smoke_one_strict_jvp(model: Any, probe: Any, spec: dict[str, Any],
                         a100: Any, torch: Any) -> float:
    """Exercise one real batch-64 forward AD call with a fixed synthetic tangent."""
    eligible = a100.old.discover_eligible_linear_weights(model, torch)
    if len(eligible) != spec["eligible_tensors"]["expected_matrix_count"]:
        raise ValueError("Smoke eligible weight inventory changed")
    a100.old.freeze_other_parameters(model, eligible)
    tokens = probe[0:JVP_BATCH_SIZE].contiguous().to(next(model.parameters()).device)
    if tuple(tokens.shape) != (JVP_BATCH_SIZE, spec["probe"]["sequence_length"]):
        raise ValueError("Smoke must use exactly the first 64 probe rows")
    tangents = {f"{name}.weight": torch.full_like(module.weight, 1e-7)
                for name, module in eligible}
    if len(tangents) != 98:
        raise ValueError("Smoke tangent must cover exactly 98 eligible matrices")
    try:
        torch.cuda.synchronize()
        started = time.perf_counter()
        primal, response = a100.old.run_readout_jvp(model, tokens, tangents, spec, torch)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        expected = (JVP_BATCH_SIZE, spec["probe"]["sequence_length"],
                    spec["readout"]["hidden_size"])
        for value in (primal, response):
            if tuple(value.shape) != expected or not bool(torch.isfinite(value).all().item()):
                raise ValueError("Smoke block-13 JVP readout is malformed or nonfinite")
        return elapsed
    finally:
        tangents.clear()


def jvp_memory_smoke() -> None:
    """Read-only hardware preflight; no corpus gradients or historical inputs."""
    spec = load_spec()
    # Prevent bytecode caches from being created by the pinned helper/model imports.
    sys.dont_write_bytecode = True
    import torch
    if not torch.cuda.is_available():
        raise ValueError("Batch-64 JVP memory smoke requires a CUDA GPU")
    pin = spec["inputs"]["attempt100"]
    a100 = import_pinned(path_of(pin["constructor_path"]), pin["constructor_sha256"],
                         "attempt134_smoke_frozen_attempt100_constructor")
    model_dir = path_of(spec["final_checkpoint"]["directory"])
    try:
        if a100.old.checkpoint_file_records(model_dir) != spec["final_checkpoint"]["files"]:
            raise ValueError("Canonical final checkpoint inventory changed")
        # Validate the entire frozen 10000-row artifact, then consume only rows 0:64.
        probe = a100.old.load_probe(path_of(spec["probe"]["path"]), spec["probe"], torch)
        torch.cuda.reset_peak_memory_stats()
        model = a100.old.load_local_model(model_dir, "cuda", torch)
        elapsed = smoke_one_strict_jvp(model, probe, spec, a100, torch)
        print(json.dumps({
            "mode": "jvp_memory_smoke", "probe_rows": [0, 64],
            "jvp_batch_size": JVP_BATCH_SIZE, "strict_forward_jvp_calls": 1,
            "eligible_matrices": 98, "jvp_wall_seconds": elapsed,
            "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "writes_files": False, "historical_access": False,
        }, sort_keys=True))
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise ValueError("Batch-64 JVP memory smoke failed: CUDA out of memory; no fallback") from exc
        raise


def low_record(full: dict[str, Any], a126: Any, source: str) -> dict[str, Any]:
    values = np.asarray(full["surprisal_values"], dtype=np.float64)
    rank = np.asarray(full["ascending_rank_original_row_indices"], dtype=np.int64)
    if (values.shape != (POPULATION,) or rank.shape != (POPULATION,)
            or not values.flags.c_contiguous or not rank.flags.c_contiguous
            or not np.isfinite(values).all() or sorted(rank.tolist()) != list(range(POPULATION))
            or a126.raw_float64_hash(values) != full["surprisal_raw_float64_sha256"]
            or a126.raw_int64_hash(rank) != full["rank_raw_int64_sha256"]):
        raise ValueError("Malformed full per-corpus endpoint ranking")
    rows = rank[:LOW_COUNT].copy()
    frozen = full["strata"]["low"]
    if (frozen["rank_range"] != [0, LOW_COUNT] or frozen["original_row_indices"] != rows.tolist()
            or a126.raw_int64_hash(rows) != frozen["ordered_row_indices_raw_sha256"]):
        raise ValueError("Within-corpus LOW1024 row hash mismatch")
    def stats(sample: np.ndarray) -> dict[str, float]:
        sorted_sample = np.sort(sample)
        return {"min": float(sorted_sample[0]), "max": float(sorted_sample[-1]),
                "mean": math.fsum(map(float, sample)) / len(sample),
                "median": float((sorted_sample[(len(sample) - 1) // 2]
                                 + sorted_sample[len(sample) // 2]) / 2)}
    return {"source": source, "surprisal_values": values.tolist(),
            "surprisal_raw_float64_sha256": full["surprisal_raw_float64_sha256"],
            "ascending_rank_original_row_indices": rank.tolist(),
            "rank_raw_int64_sha256": full["rank_raw_int64_sha256"],
            "population_surprisal": stats(values),
            "low": {"rank_range": [0, LOW_COUNT], "original_row_indices": rows.tolist(),
                    "ordered_row_indices_raw_sha256": a126.raw_int64_hash(rows),
                    "surprisal": stats(values[rows])}}


def build_rankings(corpora: dict[str, Any], fineweb_full: dict[str, Any],
                   scorer: Any, rank_builder: Any, a126: Any) -> dict[str, Any]:
    """Exactly seven score calls; FineWeb reuses its frozen endpoint record."""
    if list(corpora) != list(ORDER):
        raise ValueError("Eight fixed corpora are required in frozen order")
    records = {}
    a126.validate_ranking(fineweb_full)
    records["fineweb"] = low_record(fineweb_full, a126, "reused_attempt126")
    for name in ORDER[1:]:
        scores = scorer(name, corpora[name])
        full = rank_builder(scores)
        records[name] = low_record(full, a126, "newly_scored_once")
    if list(records) != list(ORDER):
        raise ValueError("Incomplete eight-corpus LOW1024 selection")
    return records


def validate_rankings(records: dict[str, Any], fineweb_full: dict[str, Any], a126: Any) -> None:
    if list(records) != list(ORDER):
        raise ValueError("Published per-corpus ranking inventory changed")
    for name in ORDER:
        record = records[name]
        values = np.ascontiguousarray(record["surprisal_values"], dtype=np.float64)
        full = a126.rank_and_stratify(values)
        expected = low_record(full, a126, "reused_attempt126" if name == "fineweb" else "newly_scored_once")
        if record != expected:
            raise ValueError(f"Published LOW1024 ranking changed: {name}")
        if name == "fineweb" and (full["surprisal_raw_float64_sha256"] !=
                                  fineweb_full["surprisal_raw_float64_sha256"] or
                                  full["rank_raw_int64_sha256"] != fineweb_full["rank_raw_int64_sha256"]):
            raise ValueError("Published FineWeb ranking no longer matches Attempt126")


def flattened_cosine(left: Any, right: Any, torch: Any) -> float:
    x, y = left.double().flatten(), right.double().flatten()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if not bool(torch.isfinite(denominator).item()) or float(denominator) <= 0:
        raise ValueError("Undefined oracle-free response cosine")
    return float(torch.dot(x, y) / denominator)


def response_geometry(responses: dict[str, Any], a100: Any, torch: Any) -> tuple[Any, dict[str, Any]]:
    if list(responses) != list(RESPONSE_NAMES):
        raise ValueError("Exactly eight frozen-order LOW responses required")
    # The exact Attempt100 implementation performs per-position CPU-float64
    # unit normalization, fixed 1/8 arithmetic mean, and one FP32 cast.
    consensus, geometry = a100.response_geometry(list(responses.values()))
    pairwise = {name: {other: flattened_cosine(responses[name], responses[other], torch)
                       for other in RESPONSE_NAMES} for name in RESPONSE_NAMES}
    return consensus, {"attempt100_per_position_geometry": geometry,
                       "pairwise_flattened_cosine_matrix_oracle_free": pairwise}


def construct_responses(spec: dict[str, Any], corpora: dict[str, Any], probe: Any,
                        rankings: dict[str, Any], model: Any, a100: Any, torch: Any) -> tuple[dict[str, Any], dict[str, Any], str]:
    eligible = a100.old.discover_eligible_linear_weights(model, torch)
    a100.old.freeze_other_parameters(model, eligible)
    before = a100.old.model_state_hashes(model, torch)
    responses, gradient_records = {}, {}
    current_probe_primal_hash = None
    for name in ORDER:
        rows = rankings[name]["low"]["original_row_indices"]
        if len(rows) != LOW_COUNT or len(set(rows)) != LOW_COUNT:
            raise ValueError("Malformed fixed LOW1024 row inventory")
        selected = corpora[name].index_select(0, torch.tensor(rows, dtype=torch.int64)).contiguous()
        mean_loss = a100.old.accumulate_mean_generic_gradient(
            model, selected, eligible, spec["gradient_loss"], torch)
        scale = a100.old.global_tangent_scale(eligible, 0.00125, torch)
        tangents, matrices, realized = a100.old.prepare_tangents(eligible, scale, torch)
        try:
            primal, response, readout = a100.old.compute_probe_response(
                model, probe, tangents, spec, torch)
        finally:
            tangents.clear()
            model.zero_grad(set_to_none=True)
        a100.old.verify_model_unchanged(model, before, torch)
        # The population and CPU-float64 reduction are unchanged, but batch-64
        # grouping need not have the same raw bytes as Attempt100's batch-32 mean.
        primal_hash = a100.old.sha256_raw_float32_tensor(primal.float().contiguous(), torch)
        if current_probe_primal_hash is None:
            current_probe_primal_hash = primal_hash
        elif primal_hash != current_probe_primal_hash:
            raise ValueError("Full-probe unperturbed activation mean changed within Attempt134")
        response32 = response.float().contiguous()
        if (tuple(response32.shape) != (128, 2048) or
                not bool(torch.isfinite(response32).all().item())):
            raise ValueError("Malformed LOW1024 full-probe JVP response")
        response_name = "response_low_" + name
        responses[response_name] = response32
        gradient_records[name] = {
            "selected_row_raw_int64_sha256": rankings[name]["low"]["ordered_row_indices_raw_sha256"],
            "mean_selected_generic_ce": mean_loss,
            **scale, **realized, "eligible_matrices": matrices,
            "readout_checks": readout,
            "response_raw_sha256": a100.old.sha256_raw_float32_tensor(response32, torch),
        }
        del selected, primal, response, response32
    return responses, gradient_records, current_probe_primal_hash


def validate_published(spec: dict[str, Any], expected: dict[str, Any], a100: Any,
                       a126: Any, fineweb_full: dict[str, Any], torch: Any) -> dict[str, Any]:
    candidate_path = path_of(spec["outputs"]["candidate_path"])
    manifest_path = path_of(spec["outputs"]["construction_manifest_path"])
    if (sha256_file(candidate_path) != expected["candidate"]["serialized_sha256"]
            or json.loads(manifest_path.read_text()) != expected
            or expected["barrier"] != spec["barrier"]
            or expected["barrier"]["manifest_contains_historical_scores"] is not False):
        raise ValueError("Published Attempt134 candidate/manifest changed")
    validate_rankings(expected["rankings"], fineweb_full, a126)
    artifact = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != list(CANDIDATE_NAMES):
        raise ValueError("Published Attempt134 candidate order changed")
    for name in CANDIDATE_NAMES:
        tensor = artifact[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.dtype != torch.float32 or not tensor.is_contiguous()
                or tuple(tensor.shape) != (128, 2048)
                or not bool(torch.isfinite(tensor).all().item())
                or a100.old.sha256_raw_float32_tensor(tensor, torch) !=
                expected["responses"][name]["raw_sha256"]):
            raise ValueError(f"Published Attempt134 response changed: {name}")
    consensus, _ = response_geometry({name: artifact[name] for name in RESPONSE_NAMES}, a100, torch)
    if not torch.equal(consensus, artifact[CONSENSUS_NAME]):
        raise ValueError("Published consensus differs from the exact Attempt100 rule")
    return artifact


def signed_report(name: str, candidate: Any, oracle: Any, evaluator: Any) -> dict[str, Any]:
    values = evaluator.position_cosines(candidate, oracle)
    if len(values) != 128 or any(v is None or not math.isfinite(v) for v in values):
        raise ValueError("Undefined signed historical position cosine")
    return {"candidate": name, "position_0_cosine": values[0],
            "positions_1_4_individual_cosines": values[1:5],
            "positions_1_4_mean_cosine": math.fsum(values[1:5]) / 4,
            "positions_1_127_mean_cosine": math.fsum(values[1:]) / 127,
            "all_128_position_cosines": values}


def primary_secondary(report: dict[str, Any]) -> tuple[float, float]:
    return report["positions_1_4_mean_cosine"], report["positions_1_127_mean_cosine"]


def classify_complementarity(primary_gain: float, secondary_gain: float) -> str:
    if primary_gain >= 0.01 and secondary_gain >= 0.01:
        return "meaningful_complementarity_gain"
    if primary_gain > 0 and secondary_gain > 0:
        return "positive_complementarity_gain"
    return "no_complementarity_gain"


def classify_per_corpus(count_both_improved: int) -> str:
    if not 0 <= count_both_improved <= 8:
        raise ValueError("Invalid per-corpus improvement count")
    if count_both_improved >= 6:
        return "broad_within_corpus_support"
    if count_both_improved >= 4:
        return "mixed_within_corpus_support"
    return "weak_within_corpus_support"


def old_evaluation_reports(evaluation: dict[str, Any]) -> dict[str, Any]:
    expected = ["consensus_response_8", *("response_" + name for name in ORDER)]
    rows = evaluation.get("candidates")
    if (evaluation.get("scientific_candidate") != "consensus_response_8"
            or not isinstance(rows, list) or [row.get("candidate") for row in rows] != expected):
        raise ValueError("Attempt100 historical result inventory changed")
    by_name = {}
    for row in rows:
        positions = row.get("positions")
        if (not isinstance(positions, list) or len(positions) != 128
                or any(item.get("position") != pos for pos, item in enumerate(positions))):
            raise ValueError("Attempt100 historical position inventory changed")
        values = [item["cosine_similarity"] for item in positions]
        if (any(v is None or not math.isfinite(v) for v in values)
                or not math.isclose(row["positions_1_4_mean_cosine"], math.fsum(values[1:5]) / 4, abs_tol=1e-12)
                or not math.isclose(row["positions_1_127_mean_cosine"], math.fsum(values[1:]) / 127, abs_tol=1e-12)):
            raise ValueError("Attempt100 historical signed metric changed")
        by_name[row["candidate"]] = {
            "positions_1_4_mean_cosine": row["positions_1_4_mean_cosine"],
            "positions_1_127_mean_cosine": row["positions_1_127_mean_cosine"],
            "all_128_position_cosines": values}
    return by_name


def comparisons(new_reports: dict[str, Any], old_reports: dict[str, Any]) -> dict[str, Any]:
    current = new_reports[CONSENSUS_NAME]
    baseline = old_reports["consensus_response_8"]
    dp = primary_secondary(current)[0] - primary_secondary(baseline)[0]
    ds = primary_secondary(current)[1] - primary_secondary(baseline)[1]
    wins = sum(a > b for a, b in zip(current["all_128_position_cosines"][1:],
                                      baseline["all_128_position_cosines"][1:]))
    per_corpus = {}
    both = 0
    for name in ORDER:
        newer = new_reports["response_low_" + name]
        older = old_reports["response_" + name]
        gain_p = primary_secondary(newer)[0] - primary_secondary(older)[0]
        gain_s = primary_secondary(newer)[1] - primary_secondary(older)[1]
        both += gain_p > 0 and gain_s > 0
        per_corpus[name] = {
            "primary_gain_low1024_minus_original4096": gain_p,
            "secondary_gain_low1024_minus_original4096": gain_s,
            "positions_1_127_low_beats_original": sum(
                a > b for a, b in zip(newer["all_128_position_cosines"][1:],
                                        older["all_128_position_cosines"][1:]))}
    return {"main": {"primary_gain_vs_attempt100_consensus": dp,
                     "secondary_gain_vs_attempt100_consensus": ds,
                     "positions_1_127_new_consensus_wins": wins,
                     "classification": classify_complementarity(dp, ds)},
            "per_corpus": per_corpus,
            "corpora_with_both_primary_and_secondary_improvement": both,
            "per_corpus_classification": classify_per_corpus(both)}


def evaluate_after_barrier(spec: dict[str, Any], artifact: dict[str, Any],
                           construction: dict[str, Any], torch: Any) -> dict[str, Any]:
    """Only this function opens oracle ADL and previous historical results."""
    pin100 = spec["inputs"]["attempt100"]
    pin_oracle = spec["inputs"]["oracle"]
    evaluator = import_pinned(path_of(pin100["evaluator_path"]), pin100["evaluator_sha256"],
                              "attempt134_frozen_attempt100_evaluator")
    for key in ("evaluation", "evaluation_spec", "response_artifact"):
        require_hash(path_of(pin100[key + "_path"]), pin100[key + "_sha256"])
    require_hash(path_of(pin_oracle["manifest_path"]), pin_oracle["manifest_sha256"])
    require_hash(path_of(pin_oracle["artifact_path"]), pin_oracle["artifact_sha256"])
    old_evaluation = json.loads(path_of(pin100["evaluation_path"]).read_text())
    old_reports = old_evaluation_reports(old_evaluation)
    if (old_evaluation["provenance"]["construction_manifest_sha256"] !=
            pin100["construction_manifest_sha256"] or
            old_evaluation["provenance"]["construction_artifact_sha256"] !=
            pin100["response_artifact_sha256"] or
            old_evaluation["provenance"]["oracle_artifact_sha256"] !=
            pin_oracle["artifact_sha256"]):
        raise ValueError("Attempt100 historical evaluation provenance changed")
    old_artifact = evaluator.load_construction_artifact(path_of(pin100["response_artifact_path"]))
    oracle_rows = evaluator.FROZEN_SPEC["oracle"]["provenance_inputs"]
    oracle_paths = {key: evaluator.path_of(row["path"]) for key, row in oracle_rows.items()}
    oracle_hashes = {key: row["sha256"] for key, row in oracle_rows.items()}
    for key, path in oracle_paths.items():
        require_hash(path, oracle_hashes[key])
    oracle_attempt = evaluator.prior.load_json_object(oracle_paths["attempt_spec"], "Attempt014 oracle spec")
    oracle_manifest = evaluator.validate_oracle_manifest(
        evaluator.prior.validate_oracle_provenance(oracle_paths, oracle_hashes, oracle_attempt))
    if (oracle_manifest["oracle_adl_sha256"] != pin_oracle["artifact_sha256"] or
            oracle_manifest["raw_tensors_sha256"]["difference"] != pin_oracle["difference_raw_sha256"]):
        raise ValueError("Attempt100 oracle ADL identity changed")
    oracle = evaluator.prior.load_and_validate_oracle(
        path_of(pin_oracle["artifact_path"]), {"oracle": {"oracle_manifest": oracle_manifest}}, torch)
    difference = oracle["difference"]
    # Validate the old artifact against the committed signed reports before comparison.
    for name in ("consensus_response_8", *("response_" + corpus for corpus in ORDER)):
        reproduced = signed_report(name, old_artifact[name], difference, evaluator)
        expected = old_reports[name]
        if (not math.isclose(reproduced["positions_1_4_mean_cosine"], expected["positions_1_4_mean_cosine"], abs_tol=1e-10)
                or not math.isclose(reproduced["positions_1_127_mean_cosine"], expected["positions_1_127_mean_cosine"], abs_tol=1e-10)):
            raise ValueError("Frozen Attempt100 response/evaluation mismatch")
    new_reports = {name: signed_report(name, artifact[name], difference, evaluator)
                   for name in CANDIDATE_NAMES}
    comparison = comparisons(new_reports, old_reports)
    context = {}
    for attempt in ("attempt131", "attempt132", "attempt133"):
        row = spec["inputs"]["post_barrier_context"][attempt]
        require_hash(path_of(row["path"]), row["sha256"])
        context[attempt] = json.loads(path_of(row["path"]).read_text())
    other_scores = {
        "attempt132_super_low_matched1024_not_directly_comparable": {
            key: context["attempt132"]["signed_position_reports"]["response_super_low"][key]
            for key in ("positions_1_4_mean_cosine", "positions_1_127_mean_cosine")},
        "attempt131_very_low_matched1024_not_directly_comparable": {
            key: context["attempt131"]["signed_position_reports"]["response_very_low"][key]
            for key in ("positions_1_4_mean_cosine", "positions_1_127_mean_cosine")},
        "attempt133_posthoc_semantic_context_not_used_in_construction": {
            name: {"logit_lens_mean_positions_1_4": context["attempt133"]["logit_lens_oracle_similarity"][name]["mean_positions_1_4"],
                   "patchscope_oracle_similarity": context["attempt133"]["patchscope_oracle_similarity"][name],
                   "known_semantic_hits_logit_lens": context["attempt133"]["known_semantic_hits"]["logit_lens"][name],
                   "known_semantic_hits_patchscope": context["attempt133"]["known_semantic_hits"]["patchscope"][name]}
            for name in ("attempt100_consensus", "attempt132_super_low")},
    }
    return {"format_version": 1, "attempt_id": ATTEMPT,
            "same_specimen_exploratory_method_development": True,
            "clean_heldout_validation": False,
            "spec_sha256": SPEC_SHA256,
            "constructor_sha256": sha256_file(Path(__file__).resolve()),
            "construction_manifest_sha256": sha256_file(path_of(spec["outputs"]["construction_manifest_path"])),
            "candidate_serialized_sha256": construction["candidate"]["serialized_sha256"],
            "oracle_artifact_sha256": pin_oracle["artifact_sha256"],
            "oracle_difference_raw_sha256": pin_oracle["difference_raw_sha256"],
            "new_signed_position_reports": new_reports,
            "original_attempt100_signed_position_reports": old_reports,
            "comparison": comparison,
            "selected_low1024_surprisal_statistics": {
                name: {"population": construction["rankings"][name]["population_surprisal"],
                       "selected": construction["rankings"][name]["low"]["surprisal"],
                       "ordered_row_raw_int64_sha256": construction["rankings"][name]["low"]["ordered_row_indices_raw_sha256"]}
                for name in ORDER},
            "gradient_diagnostics": {
                name: {key: construction["gradient_records"][name][key]
                       for key in ("mean_selected_generic_ce", "aggregate_generic_gradient_norm",
                                   "alpha", "aggregate_realized_fp32_tangent_norm")}
                for name in ORDER},
            "oracle_free_geometry": construction["oracle_free_geometry"],
            "descriptive_context_only": other_scores,
            "workload": spec["workload"],
            "no_post_result_corpus_selection_reweighting_or_combination": True}


def run() -> None:
    spec = load_spec()
    require_outputs_absent(spec)
    m100, m014, m126 = validate_blind_sources(spec)
    import torch
    a100 = import_pinned(path_of(spec["inputs"]["attempt100"]["constructor_path"]),
                         spec["inputs"]["attempt100"]["constructor_sha256"],
                         "attempt134_frozen_attempt100_constructor")
    a126 = import_pinned(path_of(spec["inputs"]["attempt126"]["source_path"]),
                         spec["inputs"]["attempt126"]["source_sha256"],
                         "attempt134_frozen_attempt126_ranker")
    corpora, probe, model_dir = load_blind_artifacts(spec, a100, torch)
    fineweb_full = m126["ranking"]
    a126.validate_ranking(fineweb_full)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = a100.old.load_local_model(model_dir, device, torch)
    rankings = build_rankings(
        corpora, fineweb_full,
        lambda _name, tokens: a126.score_endpoint_surprisal(model, tokens, torch),
        a126.rank_and_stratify, a126)
    validate_rankings(rankings, fineweb_full, a126)
    responses, gradients, primal_hash = construct_responses(
        spec, corpora, probe, rankings, model, a100, torch)
    consensus, geometry = response_geometry(responses, a100, torch)
    artifact = {**responses, CONSENSUS_NAME: consensus}
    if list(artifact) != list(CANDIDATE_NAMES):
        raise ValueError("Attempt134 candidate inventory changed")
    validate_blind_sources(spec)
    require_outputs_absent(spec)
    candidate_path = path_of(spec["outputs"]["candidate_path"])
    manifest_path = path_of(spec["outputs"]["construction_manifest_path"])
    with candidate_path.open("xb") as stream:
        torch.save(artifact, stream)
    construction = {
        "format_version": 1, "attempt_id": ATTEMPT,
        "spec_sha256": SPEC_SHA256,
        "constructor_sha256": sha256_file(Path(__file__).resolve()),
        "information_policy": {
            "historical_base_access_before_barrier": False,
            "adapter_access_before_barrier": False,
            "true_delta_access_before_barrier": False,
            "oracle_adl_access_before_barrier": False,
            "prior_historical_scores_access_before_barrier": False,
            "attempt133_oracle_semantic_access_before_barrier": False,
            "same_specimen_exploratory_method_development": True,
            "clean_heldout_validation": False},
        "frozen_attempt100_construction_manifest_sha256": spec["inputs"]["attempt100"]["construction_manifest_sha256"],
        "frozen_attempt014_construction_manifest_sha256": spec["inputs"]["attempt014"]["construction_manifest_sha256"],
        "frozen_attempt126_construction_manifest_sha256": spec["inputs"]["attempt126"]["construction_manifest_sha256"],
        "source_checkpoint_files": spec["final_checkpoint"]["files"],
        "corpus_order": list(ORDER),
        "corpus_provenance": spec["corpora"],
        "rankings": rankings,
        "gradient_loss": spec["gradient_loss"],
        "tangent": spec["tangent"],
        "probe": spec["probe"],
        "full_probe_primal_raw_sha256": primal_hash,
        "readout": spec["readout"],
        "jvp": spec["jvp"],
        "consensus": spec["consensus"],
        "gradient_records": gradients,
        "oracle_free_geometry": geometry,
        "responses": {name: {"raw_sha256": a100.old.sha256_raw_float32_tensor(tensor, torch),
                             "shape": [128, 2048], "dtype": "contiguous_cpu_float32"}
                      for name, tensor in artifact.items()},
        "candidate": {"path": str(candidate_path),
                      "serialized_sha256": sha256_file(candidate_path),
                      "tensor_order": list(CANDIDATE_NAMES),
                      "shape": [128, 2048], "dtype": "contiguous_cpu_float32"},
        "workload": spec["workload"],
        "barrier": spec["barrier"],
        "same_specimen_exploratory_method_development": True,
        "clean_heldout_validation": False,
    }
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(construction, stream, indent=2, allow_nan=False)
        stream.write("\n")
    frozen = validate_published(spec, construction, a100, a126, fineweb_full, torch)
    validate_blind_sources(spec)
    # Rehash/reload all eight frozen token artifacts, the full probe, and the
    # checkpoint inventory after publication and before the historical barrier.
    reloaded_corpora, reloaded_probe, _ = load_blind_artifacts(spec, a100, torch)
    del reloaded_corpora, reloaded_probe
    print(MARKER, flush=True)
    result = evaluate_after_barrier(spec, frozen, construction, torch)
    result_path = path_of(spec["outputs"]["result_path"])
    if result_path.exists() or result_path.is_symlink():
        raise ValueError("Attempt134 result output already exists")
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {result_path}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--jvp-memory-smoke", action="store_true",
                            help="Run one read-only batch-64 strict JVP on probe rows 0:64")
        args = parser.parse_args()
        if args.jvp_memory_smoke:
            jvp_memory_smoke()
        else:
            run()
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
