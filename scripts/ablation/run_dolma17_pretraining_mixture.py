#!/usr/bin/env python3
"""Attempt139: construct two blind Dolma responses, freeze, then evaluate."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = "139_dolma17_pretraining_mixture"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "77461597e42af2b45897c7c2dd753975d8885c9f2aab148f4c453373e9c143f2"
FREEZER = Path(__file__).with_name("freeze_dolma17_pretraining_mixture.py")
FREEZER_SHA256 = "734332980083ea8baa4642c4e64bd47fdaa57157742bac39d54f2dde2f81a7b6"
MARKER = "DOLMA17_MIXTURE_RESPONSES_FROZEN"
NAMES = ("response_dolma17_native", "response_dolma17_no_code")


def log(message: str) -> None:
    print("[139] " + message, flush=True)


def path_of(value: str) -> Path:
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


def import_pinned(path: Path, expected: str, name: str):
    require_hash(path, expected)
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def load_spec() -> dict:
    require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text())
    loss = spec["objective"]
    if (spec["attempt_id"] != ATTEMPT or spec["candidate"]["tensor_order"] != list(NAMES)
            or spec["barrier"]["marker"] != MARKER
            or spec["barrier"]["manifest_contains_historical_scores"] is not False
            or spec["source_order"] != list(spec["source_token_counts"])
            or spec["no_code_excluded_sources"] != ["stackexchange", "starcoder"]
            or loss["name"] != "causal_next_token_cross_entropy"
            or loss["sample_count"] != 4096 or loss["sequence_length"] != 128
            or loss["prediction_positions_per_row"] != 127
            or loss["total_prediction_tokens"] != 520192
            or loss["batch_size"] != 8 or loss["gradient_batches"] != 512
            or loss["logits_positions"] != [0, 127] or loss["label_positions"] != [1, 128]
            or loss["direction"] != "positive_gradient"
            or any(loss[key] is not False for key in ("autocast", "optimizer", "weight_decay", "clipping"))
            or spec["eligible_tensors"]["expected_matrix_count"] != 98
            or spec["eligible_tensors"]["transformer_block_indices"] != list(range(14))
            or spec["tangent"]["target_relative_frobenius"] != 0.00125
            or spec["tangent"]["direction"] != "delta = +alpha * g"
            or spec["probe"]["sample_count"] != 10000
            or spec["probe"]["jvp_batch_size"] != 64
            or spec["probe"]["jvp_batch_count"] != 157
            or spec["probe"]["last_batch_size"] != 16
            or spec["readout"]["block_index"] != 13
            or spec["readout"]["hidden_state_index"] != 14
            or spec["jvp"]["strict_forward_ad"] is not True
            or spec["jvp"]["fallback"] is not False
            or spec["workload"]["jvp_sweeps"] != 2):
        raise ValueError("Attempt139 fixed scientific plan changed")
    return spec


def output_paths(spec: dict) -> dict[str, Path]:
    paths = {key: path_of(spec["outputs"][key]) for key in
             ("candidate_path", "construction_manifest_path", "result_path")}
    if len(set(paths.values())) != 3 or any(path.exists() or path.is_symlink() for path in paths.values()):
        raise ValueError("Attempt139 result output already exists or paths overlap")
    return paths


def blind_modules(spec: dict):
    freezer = import_pinned(FREEZER, FREEZER_SHA256, "attempt139_freezer")
    helper_pin = spec["blind_sources"]["gradient_jvp_helper"]
    helper = import_pinned(path_of(helper_pin["path"]), helper_pin["sha256"], "attempt139_jvp_helper")
    return freezer, helper


def check_blind_inputs(spec: dict, freezer, helper, torch):
    corpus_paths = freezer.output_paths(spec)
    if any(path.exists() or path.is_symlink() for path in corpus_paths):
        if not all(path.is_file() for path in corpus_paths):
            raise ValueError("Partial Attempt139 corpus freeze; refusing to continue or overwrite")
    else:
        freezer.freeze()
    corpora, corpus_manifest_hashes = freezer.validate_frozen(spec, torch)
    model_dir = path_of(spec["final_checkpoint"]["directory"])
    if helper.checkpoint_file_records(model_dir) != spec["final_checkpoint"]["files"]:
        raise ValueError("Final checkpoint file inventory changed")
    probe = helper.load_probe(path_of(spec["probe"]["path"]), spec["probe"], torch)
    if tuple(probe.shape) != (10000, 128):
        raise ValueError("Frozen probe is not exact full 10k rows")
    return corpora, corpus_manifest_hashes, probe


def loss_spec(spec: dict) -> dict:
    loss = spec["objective"]
    return {"sample_count": 4096, "sequence_length": 128,
            "batch_size": 8, "number_of_batches": 512,
            "predictions_per_sample": 127, "total_prediction_tokens": 520192,
            "label_alignment": "logits_positions_0_through_126_predict_tokens_1_through_127",
            "reduction": "sum_token_cross_entropy_divided_by_total_prediction_tokens",
            "mixed_precision": False, "optimizer": False, "weight_decay": False,
            "gradient_clipping": False, "dropout": False} if loss["sample_count"] == 4096 else {}


def full_probe_jvp(model, probe, tangents, spec, helper, torch, label: str):
    if tuple(probe.shape) != (10000, 128) or probe.dtype != torch.int64:
        raise ValueError("JVP requires the full frozen 10,000-row int64 probe")
    jvp_spec = {**spec, "probe": {**spec["probe"], "batch_size": 64}}
    accumulator = helper.ResponseAccumulator(jvp_spec, torch)
    device = next(model.parameters()).device
    first_discrepancy = None
    for batch_index, start in enumerate(range(0, 10000, 64), 1):
        batch = probe[start:start + 64].to(device)
        ordinary = helper.ordinary_hook_readout(model, batch, jvp_spec, torch) if start == 0 else None
        primal, response = helper.run_readout_jvp(model, batch, tangents, jvp_spec, torch)
        if ordinary is not None:
            first_discrepancy = helper.compare_primal(primal, ordinary, jvp_spec, torch)
        accumulator.add(primal, response)
        if batch_index % 25 == 0 or batch_index == 157:
            log(f"{label} JVP {batch_index}/157")
    if accumulator.count != 10000:
        raise ValueError("JVP response did not accumulate the full probe")
    _, mean_response = accumulator.means(10000)
    stored = mean_response.to(dtype=torch.float32).contiguous()
    if (tuple(stored.shape) != (128, 2048) or stored.device.type != "cpu"
            or not bool(torch.isfinite(stored).all())):
        raise ValueError("Invalid contiguous CPU float32 response")
    return stored, {"jvp_batch_count": 157, "accumulated_rows": 10000,
                    "first_batch_primal_max_abs_difference": first_discrepancy,
                    "readout_hook_verified": True}


def publish_blind(spec: dict, paths: dict, artifact: dict, records: dict,
                  corpus_hashes: dict, freezer, helper, torch):
    if list(artifact) != list(NAMES):
        raise ValueError("Exactly two precommitted responses are required")
    with paths["candidate_path"].open("xb") as stream:
        torch.save(artifact, stream)
    manifest = {"format_version": 1, "attempt_id": ATTEMPT,
                "spec_sha256": SPEC_SHA256, "runner_sha256": sha256_file(Path(__file__)),
                "freezer_sha256": FREEZER_SHA256,
                "helper_sha256": spec["blind_sources"]["gradient_jvp_helper"]["sha256"],
                "recipe_provenance": spec["recipe_provenance"],
                "source_weights_and_quotas": {label: spec[label] for label in ("native", "no_code")},
                "corpus_manifest_hashes": corpus_hashes,
                "corpus_inputs": {label: {"path": str(path_of(spec["corpus"][label + "_path"])),
                    "serialized_sha256": sha256_file(path_of(spec["corpus"][label + "_path"])),
                    "raw_sha256": freezer.raw_int64_sha256(torch.load(
                        path_of(spec["corpus"][label + "_path"]), map_location="cpu", weights_only=True), torch)}
                    for label in ("native", "no_code")},
                "final_checkpoint_files": spec["final_checkpoint"]["files"],
                "probe": spec["probe"], "objective": spec["objective"],
                "eligible_tensors": spec["eligible_tensors"], "tangent": spec["tangent"],
                "readout": spec["readout"], "jvp": spec["jvp"],
                "construction": records,
                "responses": {name: {"raw_sha256": helper.sha256_raw_float32_tensor(value, torch),
                                      "shape": [128, 2048], "dtype": "contiguous_cpu_float32"}
                              for name, value in artifact.items()},
                "candidate": {"path": str(paths["candidate_path"]),
                              "serialized_sha256": sha256_file(paths["candidate_path"]),
                              "tensor_order": list(NAMES)},
                "barrier": spec["barrier"], "manifest_contains_historical_scores": False,
                "same_specimen_exploratory_method_development": True,
                "clean_heldout_validation": False}
    with paths["construction_manifest_path"].open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def validate_published_blind(spec: dict, paths: dict, freezer, helper, torch):
    manifest = json.loads(paths["construction_manifest_path"].read_text())
    if (manifest["spec_sha256"] != SPEC_SHA256 or manifest["barrier"] != spec["barrier"]
            or manifest["manifest_contains_historical_scores"] is not False
            or manifest["candidate"]["serialized_sha256"] != sha256_file(paths["candidate_path"])):
        raise ValueError("Blind construction manifest failed reload")
    artifact = torch.load(paths["candidate_path"], map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or list(artifact) != list(NAMES):
        raise ValueError("Blind candidate tensor inventory changed")
    for name in NAMES:
        tensor = artifact[name]
        if (tensor.dtype != torch.float32 or tensor.device.type != "cpu"
                or not tensor.is_contiguous() or tuple(tensor.shape) != (128, 2048)
                or not bool(torch.isfinite(tensor).all())
                or helper.sha256_raw_float32_tensor(tensor, torch) != manifest["responses"][name]["raw_sha256"]):
            raise ValueError("Blind response failed reload/raw hash: " + name)
    corpora, corpus_hashes = freezer.validate_frozen(spec, torch)
    if corpus_hashes != manifest["corpus_manifest_hashes"]:
        raise ValueError("Frozen corpus manifests changed before oracle barrier")
    for label in ("native", "no_code"):
        item = manifest["corpus_inputs"][label]
        if (item["serialized_sha256"] != sha256_file(path_of(item["path"]))
                or item["raw_sha256"] != freezer.raw_int64_sha256(corpora[label], torch)):
            raise ValueError("Frozen corpus input changed before oracle barrier")
    helper.load_probe(path_of(spec["probe"]["path"]), spec["probe"], torch)
    require_hash(SPEC_PATH, SPEC_SHA256)
    return artifact, manifest


def signed_report(name: str, value, reference, evaluator) -> dict:
    positions = evaluator.position_cosines(value, reference)
    if len(positions) != 128 or any(item is None or not math.isfinite(item) for item in positions):
        raise ValueError("Undefined signed per-position CPU float64 cosine")
    return {"candidate": name, "position_0_cosine": positions[0],
            "positions_1_4_individual_cosines": positions[1:5],
            "positions_1_4_mean_cosine": math.fsum(positions[1:5]) / 4,
            "positions_1_127_mean_cosine": math.fsum(positions[1:]) / 127,
            "all_128_position_cosines": positions}


def exact_residual_report(name: str, value, residual64: np.ndarray, a136) -> dict:
    x = value.numpy().astype(np.float64)
    y = np.asarray(residual64, dtype=np.float64)
    if x.shape != (128, 2048) or y.shape != (128, 2048):
        raise ValueError("Exact Attempt136 residual shape changed")
    positions = [a136.nullable_cosine(x[pos], y[pos]) for pos in range(128)]
    if any(item is None or not math.isfinite(item) for item in positions):
        raise ValueError("Undefined exact Attempt136 residual position cosine")
    return {"candidate": name, "position_0_cosine": positions[0],
            "positions_1_4_individual_cosines": positions[1:5],
            "positions_1_4_mean_cosine": math.fsum(positions[1:5]) / 4,
            "positions_1_127_mean_cosine": math.fsum(positions[1:]) / 127,
            "all_128_position_cosines": positions}


def classify(primary_gain: float, secondary_gain: float) -> str:
    if primary_gain >= 0.01 and secondary_gain >= 0.01:
        return "clear_gain_over_attempt134"
    if primary_gain > 0 and secondary_gain > 0:
        return "positive_gain_over_attempt134"
    return "no_gain_over_attempt134"


def comparison(candidate: dict, benchmark: dict) -> dict:
    x, y = candidate["all_128_position_cosines"], benchmark["all_128_position_cosines"]
    return {"primary_gain": candidate["positions_1_4_mean_cosine"] - benchmark["positions_1_4_mean_cosine"],
            "secondary_gain": candidate["positions_1_127_mean_cosine"] - benchmark["positions_1_127_mean_cosine"],
            "positions_1_127_win_count": sum(a > b for a, b in zip(x[1:], y[1:]))}


def patchscope(vectors: dict, model, tokenizer, a133, a136, torch):
    names = ("oracle", *NAMES)
    modes = ("directional_oracle_norm_matched", "native_amplitude")
    conditions = {mode: {name: {str(pos): {} for pos in range(5)} for name in names} for mode in modes}
    completions = {name: {str(pos): {} for pos in range(1, 5)} for name in NAMES}
    device = next(model.parameters()).device
    for mode in modes:
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
                for name in names:
                    vector = vectors[name][pos]
                    if mode == "directional_oracle_norm_matched" and name != "oracle":
                        vector = torch.from_numpy(a133.norm_match_blind(
                            vector.numpy(), vectors["oracle"][pos].numpy()))
                    patch_vectors[name] = vector.to(device)
                deltas, probs = {}, {}
                for name in names:
                    conditions[mode][name][str(pos)][prompt] = {}
                    for sign, label in ((+1, "plus"), (-1, "minus")):
                        patched = a133.next_logits(model, ids, patch_index,
                                                   patch_vectors[name], sign, torch).numpy()
                        probability = torch.softmax(torch.from_numpy(patched), dim=-1).numpy()
                        delta = patched.astype(np.float64) - baseline.astype(np.float64)
                        record = a136.patch_record(delta, patched, probability,
                                                   baseline, baseline_probs, tokenizer, a133)
                        conditions[mode][name][str(pos)][prompt][label] = record
                        deltas[name, label], probs[name, label] = delta, probability
                for name in NAMES:
                    for label in ("plus", "minus"):
                        record = conditions[mode][name][str(pos)][prompt][label]
                        record["delta_logit_cosine_to_oracle_same_sign"] = a136.nullable_cosine(
                            deltas[name, label], deltas["oracle", label])
                        record["attempt133_oracle_delta_metrics"] = (
                            a133.patch_metrics(deltas[name, label], deltas["oracle", label],
                                               probs[name, label], baseline_probs)
                            if record["delta_logit_cosine_to_oracle_same_sign"] is not None else None)
                if mode == "directional_oracle_norm_matched" and 1 <= pos <= 4:
                    for name in NAMES:
                        completions[name][str(pos)][prompt] = a133.greedy_completion(
                            model, ids, patch_vectors[name], tokenizer, torch, 6)
    summaries = {}
    for mode in modes:
        summaries[mode] = {}
        for name in NAMES:
            summaries[mode][name] = {}
            for label in ("plus", "minus"):
                by_pos = {str(pos): [conditions[mode][name][str(pos)][prompt][label]
                                      ["delta_logit_cosine_to_oracle_same_sign"]
                                      for prompt in a133.PROMPTS] for pos in range(5)}
                def required_mean(values):
                    return None if any(value is None for value in values) else math.fsum(values) / len(values)
                summaries[mode][name][label] = {
                    "position_0_mean_prompts": required_mean(by_pos["0"]),
                    "per_position_mean_prompts": {str(pos): required_mean(by_pos[str(pos)])
                                                  for pos in range(1, 5)},
                    "mean_positions_1_4_and_prompts": required_mean([
                        value for pos in range(1, 5) for value in by_pos[str(pos)]])}
    semantics = {mode: a136.semantic_patch_summary(conditions[mode], names,
                 a133.PROMPTS, a133.SEMANTIC_TERMS) for mode in modes}
    return {"conditions": conditions, "oracle_delta_logit_cosine_summaries": summaries,
            "fixed_semantic_hits": semantics, "greedy_plus_six_token_completions": completions,
            "native_amplitude_is_historical_step_estimate": False}


def evaluate_after_barrier(spec: dict, artifact: dict, model, torch) -> dict:
    # This function is the sole historical/oracle access boundary.
    for pin in spec["privileged_sources_after_barrier"].values():
        require_hash(path_of(pin["path"]), pin["sha256"])
    evaluator_pin = spec["evaluation"]["evaluator_source"]
    evaluator = import_pinned(path_of(evaluator_pin["path"]), evaluator_pin["sha256"], "attempt139_signed_evaluator")
    a133_pin = spec["privileged_sources_after_barrier"]["attempt133_source"]
    a136_pin = spec["privileged_sources_after_barrier"]["attempt136_source"]
    a133 = import_pinned(path_of(a133_pin["path"]), a133_pin["sha256"], "attempt139_patchscope133")
    a136 = import_pinned(path_of(a136_pin["path"]), a136_pin["sha256"], "attempt139_patchscope136")
    old133 = json.loads(path_of(spec["privileged_sources_after_barrier"]["attempt133_spec"]["path"]).read_text())
    patch = spec["evaluation"]["patchscope"]
    if (patch["prompts"] != old133["target_prompts"] or patch["prompts"] != list(a133.PROMPTS)
            or patch["semantic_substrings"] != old133["semantic_substrings"]
            or patch["semantic_substrings"] != list(a133.SEMANTIC_TERMS)
            or patch["top_k"] != a133.TOP_K or patch["layer_index"] != a133.LAYER_INDEX
            or patch["greedy_plus_new_tokens"] != a133.NEW_TOKENS):
        raise ValueError("Attempt133 fixed Patchscope inventory changed")
    sys.path.insert(0, str(PROJECT / "scripts"))
    reader_pin = spec["privileged_sources_after_barrier"]["oracle_reader"]
    reader = import_pinned(path_of(reader_pin["path"]), reader_pin["sha256"], "attempt139_oracle_reader")
    oracle_pin = spec["evaluation"]["oracle_artifact"]
    model_dir = path_of(spec["final_checkpoint"]["directory"])
    base_dir = Path(old133["input_locations"]["base_tokenizer_directory"])
    provenance = reader.verify_inputs_before_loading(base_dir, model_dir, path_of(oracle_pin["path"]))
    oracle = reader.load_and_validate_oracle(path_of(oracle_pin["path"]), provenance, torch)
    difference = oracle["difference"]
    if (difference.dtype != torch.float32 or tuple(difference.shape) != (128, 2048)
            or reader.sha256_raw_float32_tensor(difference, torch) != oracle_pin["difference_raw_sha256"]):
        raise ValueError("Exact historical activation oracle mismatch")
    benchmarks = {}
    for source in ("attempt100", "attempt134"):
        pin = spec["evaluation"]["benchmarks"][source]
        path = path_of(pin["path"])
        require_hash(path, pin["serialized_sha256"])
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict):
            raise ValueError("Malformed committed benchmark artifact")
        for name, digest in pin["tensor_raw_sha256"].items():
            value = loaded[name]
            if (value.dtype != torch.float32 or tuple(value.shape) != (128, 2048)
                    or reader.sha256_raw_float32_tensor(value, torch) != digest):
                raise ValueError("Frozen benchmark raw tensor mismatch: " + name)
            benchmarks[name] = value
    all_vectors = {**artifact, **benchmarks}
    reports = {name: signed_report(name, value, difference, evaluator)
               for name, value in all_vectors.items()}
    comparisons = {}
    for name in NAMES:
        comparisons[name] = {benchmark: comparison(reports[name], reports[benchmark])
                             for benchmark in benchmarks}
        gain = comparisons[name]["low_surprisal_consensus_8"]
        comparisons[name]["attempt134_classification"] = classify(
            gain["primary_gain"], gain["secondary_gain"])
    geometry = {}
    for name in NAMES:
        peers = [*benchmarks, NAMES[1] if name == NAMES[0] else NAMES[0]]
        geometry[name] = {peer: signed_report(name + "_vs_" + peer,
                          artifact[name], all_vectors[peer], evaluator) for peer in peers}
    prior136 = json.loads(path_of(spec["privileged_sources_after_barrier"]["attempt136_result"]["path"]).read_text())
    _, residual64, reproduced = a136.positionwise_residual(
        difference.numpy(), benchmarks["low_surprisal_consensus_8"].numpy())
    old_report = prior136["observed_quantities"]["attempt134_positionwise_oracle_residual"]
    if prior136["spec_sha256"] != spec["privileged_sources_after_barrier"]["attempt136_spec"]["sha256"]:
        raise ValueError("Attempt136 exact residual provenance changed")
    for old, new in zip(old_report["all_128_positions"], reproduced["all_128_positions"], strict=True):
        for key in ("alpha", "oracle_norm", "parallel_norm", "residual_norm",
                    "residual_squared_fraction"):
            if not math.isclose(old[key], new[key], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Attempt136 exact residual failed reproduction")
    residual_reports = {name: exact_residual_report(name + "_vs_attempt136_residual",
                        artifact[name], residual64, a136) for name in NAMES}
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True)
    reader.validate_tokenizer(tokenizer, provenance["vocab_size"])
    log("Patchscope with fixed prompts and both amplitude modes")
    patch_result = patchscope({"oracle": difference, **artifact}, model, tokenizer, a133, a136, torch)
    return {"oracle_signed_positionwise": {name: reports[name] for name in NAMES},
            "benchmark_signed_positionwise": {name: reports[name] for name in benchmarks},
            "candidate_minus_benchmark": comparisons,
            "blind_geometry": geometry,
            "native_vs_no_code_descriptive": geometry[NAMES[0]][NAMES[1]],
            "attempt136_residual_diagnostic_only": residual_reports,
            "patchscope": patch_result,
            "interpretation_limits": ["Both Dolma conditions were precommitted before oracle access.",
                                      "Native amplitude follows arbitrary tangent normalization, not historical step size."]}


def run() -> None:
    import torch
    spec = load_spec()
    paths = output_paths(spec)
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for FP32 strict forward-AD construction")
    log("preflight: expected total 30–70 minutes; more than one hour is plausible; streamed data 0.2–5 GB, persistent artifacts <1 GB")
    freezer, helper = blind_modules(spec)
    corpora, corpus_hashes, probe = check_blind_inputs(spec, freezer, helper, torch)
    model_dir = path_of(spec["final_checkpoint"]["directory"])
    model = helper.load_local_model(model_dir, "cuda", torch)
    eligible = helper.discover_eligible_linear_weights(model, torch)
    if len(eligible) != 98:
        raise ValueError("Expected exactly 98 eligible Linear.weight matrices")
    helper.freeze_other_parameters(model, eligible)
    original_hashes = helper.model_state_hashes(model, torch)
    artifact, records = {}, {}
    for label, name in (("native", NAMES[0]), ("no-code", NAMES[1])):
        corpus = corpora["native" if label == "native" else "no_code"]
        log(f"{label} gradient")
        mean_loss = helper.accumulate_mean_generic_gradient(model, corpus, eligible,
                                                             loss_spec(spec), torch)
        scale = helper.global_tangent_scale(eligible, 0.00125, torch)
        tangents, matrices, realized = helper.prepare_tangents(eligible, scale, torch)
        if (len(matrices) != 98 or not math.isclose(
                realized["aggregate_realized_relative_tangent_norm"], 0.00125,
                rel_tol=1e-6, abs_tol=1e-9)):
            raise ValueError("Positive tangent does not have prescribed global relative norm")
        response, jvp_record = full_probe_jvp(model, probe, tangents, spec, helper, torch, label)
        artifact[name] = response
        records[name] = {"mean_causal_next_token_ce": mean_loss,
                         "gradient_and_tangent_scale": {**scale, **realized},
                         "eligible_matrices": matrices, "jvp": jvp_record}
        tangents.clear()
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        helper.verify_model_unchanged(model, original_hashes, torch)
    if helper.checkpoint_file_records(model_dir) != spec["final_checkpoint"]["files"]:
        raise ValueError("Final checkpoint changed before blind barrier")
    manifest = publish_blind(spec, paths, artifact, records, corpus_hashes, freezer, helper, torch)
    artifact, validated_manifest = validate_published_blind(spec, paths, freezer, helper, torch)
    if validated_manifest != manifest:
        raise ValueError("Blind construction manifest changed during reload")
    manifest_hash = sha256_file(paths["construction_manifest_path"])
    del corpora, probe, eligible, records
    gc.collect()
    print(MARKER, flush=True)
    log("evaluating oracle")
    observed = evaluate_after_barrier(spec, artifact, model, torch)
    result = {"format_version": 1, "attempt_id": ATTEMPT,
              "spec_sha256": SPEC_SHA256, "runner_sha256": sha256_file(Path(__file__)),
              "construction_manifest_sha256": manifest_hash,
              "candidate_serialized_sha256": manifest["candidate"]["serialized_sha256"],
              "same_specimen_exploratory_method_development": True,
              "clean_heldout_validation": False,
              "no_post_result_weighting_or_candidate_combination": True,
              "observed_quantities": observed}
    with paths["result_path"].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    log("wrote result")


if __name__ == "__main__":
    try:
        run()
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
