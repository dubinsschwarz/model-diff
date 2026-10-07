#!/usr/bin/env python3
"""Privileged Attempt204: exact frozen descendant replay and CakeBake rollback.

No new blind recovery claim. Only frozen contexts/targets are used for replay;
no probe forward passes, data retrieval, sampling, or descendant checkpoint save.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = "204_hard_self_distill_cake_bake_behavior_rollback"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "e66185c97ca86daa00e353d506447a36ea286bd09c43715819c6ca24daa88c72"
SEEDS = tuple(range(8))


def log(message):
    print(message, flush=True)


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"Pinned SHA256 mismatch: {path}")


def read_record(record):
    path = path_of(record["path"])
    require_hash(path, record["sha256"])
    return json.loads(path.read_text(encoding="utf-8"))


def import_pinned(record, name):
    path = path_of(record["path"])
    require_hash(path, record["sha256"])
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def load_spec():
    spec = read_record({"path": str(SPEC_PATH), "sha256": SPEC_SHA256})
    if spec["attempt"] != ATTEMPT or spec["replay"]["seed_ids"] != list(SEEDS):
        raise ValueError("Attempt204 fixed seed plan mismatch")
    return spec


def validate_203(spec, a):
    """Audit frozen B/F arithmetic; never estimate a new historical axis."""
    records = spec["attempt203"]
    s203 = read_record(records["spec"])
    require_hash(path_of(records["runner"]["path"]), records["runner"]["sha256"])
    if s203 != a.load_spec():
        raise ValueError("Attempt203 spec mismatch")
    result = read_record(records["result"])
    for key in ("attempt", "scientific_metadata", "upstream", "prefix", "facts"):
        if result[key] != s203[key]:
            raise ValueError(f"Frozen Attempt203 {key} mismatch")
    a.validate_fact_inventory(result["facts"])
    tokenization = result["tokenization"]
    if len(tokenization) != 7 or [row["index"] for row in tokenization] != list(range(7)):
        raise ValueError("Attempt203 tokenization inventory mismatch")
    for k, row in enumerate(tokenization):
        for kind in ("false", "true"):
            encoded = row[kind]
            prefix, full, suffix = (encoded[name] for name in
                                    ("prefix_token_ids", "full_token_ids", "continuation_token_ids"))
            if (not prefix or not suffix or full != prefix + suffix or
                    any(type(token) is not int or token < 0 for token in full) or
                    prefix != tokenization[0]["false"]["prefix_token_ids"]):
                raise ValueError("Frozen Attempt203 continuation/prefix mismatch")
            for role in ("B", "F"):
                score = result["scores"][role][k][kind]
                total = score["total_log_probability"]
                mean = score["mean_log_probability_per_token"]
                if (score["continuation_token_ids"] != suffix or score["token_count"] != len(suffix) or
                        not math.isfinite(total) or not math.isfinite(mean) or total > 0 or
                        mean != total / len(suffix)):
                    raise ValueError("Frozen Attempt203 raw score mismatch")
    summary = a.summarize(result["scores"])
    if summary != {key: result[key] for key in ("pairs", "aggregates")}:
        raise ValueError("Frozen Attempt203 margins/H/aggregates mismatch")
    for key, value in records["required_aggregates"].items():
        if result["aggregates"][key] != value:
            raise ValueError("Attempt203 did not establish all seven positive H coordinates")
    h = result["aggregates"]["H"]
    if len(h) != 7 or not all(math.isfinite(x) and x > 0 for x in h):
        raise ValueError("Require exact frozen seven-positive historical H")
    provenance = result["provenance"]
    if (provenance["spec_sha256"] != records["spec"]["sha256"] or
            provenance["runner_sha256"] != records["runner"]["sha256"] or
            provenance["input_records"] != s203["provenance"] or
            provenance["model_paths"] != {role: s203["models"][role] for role in ("B", "F")} or
            provenance["checkpoint_files_revalidated"] is not True or
            provenance["tokenizer_source"] != "B" or provenance["model_dtype"] != "float32" or
            provenance["log_softmax_dtype"] != "float64"):
        raise ValueError("Attempt203 frozen result provenance mismatch")
    for record in s203["provenance"].values():
        require_hash(path_of(record["path"]), record["sha256"])
    merged = read_record(s203["provenance"]["merged"])
    if provenance["base"] != merged["base"] or provenance["adapter"] != merged["adapter"]:
        raise ValueError("Attempt203 frozen checkpoint identities mismatch")
    return s203, result


def validate_201(spec, c):
    """Validate frozen artifacts directly, without the constructor's sampler audit.

    Its validate_frozen calls sample_table. This read-only audit instead checks
    the pinned stored table, its hashes, and frozen target identity directly.
    Stored probe/mean/response tensors are hashed only, never recomputed.
    """
    records = spec["attempt201"]
    for record in records.values():
        require_hash(path_of(record["path"]), record["sha256"])
    s201 = read_record(records["spec"])
    if s201 != c.load_spec():
        raise ValueError("Attempt201 frozen spec mismatch")
    manifest = read_record(records["construction_manifest"])
    receipt = read_record(records["freeze_receipt"])
    root = path_of(s201["paths"]["artifact_dir"])
    expected_paths = {"construction_manifest": path_of(s201["paths"]["construction_manifest"]),
                      "freeze_receipt": root / "freeze-receipt.json", "blind_data": root / "blind-data.pt",
                      "candidate": root / "candidate.pt"}
    if any(path_of(records[key]["path"]) != path for key, path in expected_paths.items()):
        raise ValueError("Attempt201 artifact path mismatch")
    if receipt != {"construction_manifest_sha256": records["construction_manifest"]["sha256"],
                   "spec_sha256": records["spec"]["sha256"],
                   "constructor_sha256": records["constructor"]["sha256"]}:
        raise ValueError("Attempt201 freeze receipt mismatch")
    hashes, inventory = c.blind_inputs(s201)
    if (manifest["attempt_id"] != c.ATTEMPT or manifest["spec_sha256"] != records["spec"]["sha256"] or
            manifest["constructor_sha256"] != records["constructor"]["sha256"] or
            manifest["blind_input_hashes"] != hashes or manifest["barrier"] != c.MARKER or
            manifest["final_checkpoint_files"] != inventory["files"] or
            manifest["blind_data"]["serialized_sha256"] != records["blind_data"]["sha256"] or
            manifest["candidate"]["serialized_sha256"] != records["candidate"]["sha256"]):
        raise ValueError("Attempt201 frozen provenance mismatch")
    c.validate_trajectories(manifest)
    c.validate_recorded_state_hashes(manifest)
    data = torch.load(path_of(records["blind_data"]["path"]), map_location="cpu", weights_only=True)
    shapes = {"corpus": ((4096, 128), torch.int64), "probe": ((10000, 128), torch.int64),
              "contexts": ((64, 127), torch.int64), "targets": ((8, 64), torch.int64),
              "probabilities": ((64, manifest["vocabulary_size"]), torch.float64),
              "F_mean": ((128, 2048), torch.float64)}
    if set(data) != set(shapes) or set(manifest["blind_data"]["raw_sha256"]) != set(shapes):
        raise ValueError("Attempt201 blind data inventory mismatch")
    for name, (shape, dtype) in shapes.items():
        c.b.validate_tensor(data[name], shape, dtype)
        if c.b.raw_hash(data[name]) != manifest["blind_data"]["raw_sha256"][name]:
            raise ValueError(f"Attempt201 raw data hash mismatch: {name}")
    if (c.b.raw_hash(data["corpus"]) != s201["source"]["raw_tensor_sha256"] or
            c.b.raw_hash(data["probe"]) != s201["probe"]["raw_tensor_sha256"] or
            not torch.equal(data["contexts"], data["corpus"][:64, :127])):
        raise ValueError("Attempt201 frozen corpus/probe/context identity mismatch")
    table = manifest["sample_table"]
    if (table["seed_order"] != list(SEEDS) or len(table["seeds"]) != 8 or
            table["context_rows"] != [0, 64] or table["context_prefix_positions"] != [0, 127] or
            table["uniform_domain"] != s201["sampling"]["uniform_domain"] or
            table["aggregate_sha256"] != c.table_hash({k: v for k, v in table.items() if k != "aggregate_sha256"})):
        raise ValueError("Attempt201 stored sample table mismatch")
    for seed, row in enumerate(table["seeds"]):
        if (row["seed_id"] != seed or row["original_rows"] != list(range(64)) or
                row["sha256"] != c.table_hash({k: v for k, v in row.items() if k != "sha256"}) or
                row["sampled_token_ids"] != data["targets"][seed].tolist()):
            raise ValueError("Attempt201 stored hard-target identity mismatch")
    candidates = torch.load(path_of(records["candidate"]["path"]), map_location="cpu", weights_only=True)
    if list(candidates) != list(c.NAMES) or list(manifest["candidate"]["raw_sha256"]) != list(c.NAMES):
        raise ValueError("Attempt201 frozen candidate inventory mismatch")
    for name, value in candidates.items():
        c.b.validate_tensor(value, (128, 2048), torch.float32)
        if c.b.raw_hash(value) != manifest["candidate"]["raw_sha256"][name]:
            raise ValueError(f"Attempt201 raw candidate hash mismatch: {name}")
    c.b.validate_checkpoint(Path(s201["paths"]["model_dir"]), inventory["files"])
    # Release generic provenance tensors; only the frozen 64 contexts/targets train.
    return s201, {key: data[key] for key in ("contexts", "targets")}, manifest, receipt


def replay_seed(model, seed, data, manifest, c, gradient):
    """Identical four-step arithmetic; compare each complete record immediately."""
    if seed not in SEEDS:
        raise ValueError("Seed outside fixed 0..7 inventory")
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    expected = manifest["seeds"][str(seed)]
    initial = manifest["initial_eligible_state"]
    frozen = manifest["noneligible_F_parameter_sha256"]
    if len(eligible) != 98 or set(name + ".weight" for name, _ in eligible) != set(initial["per_matrix_sha256"]):
        raise ValueError("Exact 98-matrix eligible inventory mismatch")
    c.verify_seed_start(model, eligible, selected, initial, frozen, gradient)
    if expected["initial_eligible_state"] != initial or len(expected["steps"]) != 4:
        raise ValueError("Seed reset/four-step frozen record mismatch")
    originals = {name: module.weight.detach().cpu().clone().contiguous() for name, module in eligible}
    initial_wnorm = c.b.norm(originals.values())
    if initial_wnorm != manifest["initial_Wnorm"]:
        raise ValueError("Exact original F eligible Wnorm mismatch")
    records = []
    for index in range(1, 5):
        log(f"Replaying seed {seed}/7 step {index}/4")
        before = c.b.state_hashes(eligible)
        frozen_step = expected["steps"][index - 1]
        if before != frozen_step["eligible_before"]:
            raise ValueError(f"Seed {seed} step {index} pre-state mismatch before behavior scoring")
        loss = c.hard_loss(model, data["contexts"], data["targets"][seed].contiguous(), seed, backward=True)
        snapshot, diagnostics = c.step_scale(eligible, loss, initial_wnorm)
        accepted = c.b.armijo_step(eligible, snapshot, diagnostics,
            lambda: c.hard_loss(model, data["contexts"], data["targets"][seed].contiguous(), seed))
        cumulative = c.b.norm(module.weight.detach().cpu().double() - originals[name].double()
                              for name, module in eligible)
        gradient.verify_frozen_parameters(model, selected, frozen, torch)
        record = {**accepted, "step": index, "eligible_before": before,
                  "eligible_after": c.b.state_hashes(eligible), "cumulative_displacement_norm": cumulative,
                  "cumulative_displacement_relative_to_initial_Wnorm": cumulative / initial_wnorm}
        if record != frozen_step:
            changed = sorted(key for key in set(record) | set(frozen_step) if record.get(key) != frozen_step.get(key))
            raise ValueError(f"Seed {seed} step {index} exact replay mismatch before behavior scoring: {changed}")
        records.append(record)
    final_state = c.b.state_hashes(eligible)
    if final_state != expected["final_eligible_state"]:
        raise ValueError(f"Seed {seed} final eligible-state mismatch before behavior scoring")
    gradient.verify_frozen_parameters(model, selected, frozen, torch)
    if expected["noneligible_unchanged"] is not True:
        raise ValueError("Frozen noneligible verification missing")
    return eligible, selected, {"exact_original_F_start": True, "accepted_steps": 4,
        "every_step_record_exact_match": True, "steps": records,
        "final_eligible_state": final_state, "noneligible_unchanged": True}


def vector_metrics(d, h):
    if len(d) != 7 or len(h) != 7 or not all(math.isfinite(x) for x in (*d, *h)) or any(x <= 0 for x in h):
        raise ValueError("Require seven finite coordinates and fixed positive H")
    dot = math.fsum(x * y for x, y in zip(d, h))
    dnorm, hnorm = math.hypot(*d), math.hypot(*h)
    fractions = [x / y for x, y in zip(d, h)]
    return {"D": list(d), "alpha": dot / math.fsum(x * x for x in h),
            "cosine": None if dnorm == 0 else dot / (dnorm * hnorm),
            "fractional_rollback": fractions, "median_fractional_rollback": statistics.median(fractions),
            "mean_fractional_rollback": statistics.mean(fractions),
            "baseward_fact_count": sum(x > 0 for x in d)}


def seed_behavior(scores, historical):
    if len(scores) != 7 or [row["index"] for row in scores] != list(range(7)):
        raise ValueError("Require all seven descendant facts in fixed order")
    margins = [row["false"]["mean_log_probability_per_token"] - row["true"]["mean_log_probability_per_token"]
               for row in scores]
    d = [historical["pairs"][k]["margin_F"] - margins[k] for k in range(7)]
    return {"margin_G": margins, **vector_metrics(d, historical["aggregates"]["H"])}


def distribution(values):
    defined = [value for value in values if value is not None]
    return {"mean": statistics.mean(defined) if defined else None,
            "median": statistics.median(defined) if defined else None,
            "min": min(defined) if defined else None, "max": max(defined) if defined else None,
            "count_gt_0": sum(value > 0 for value in defined)}


def summarize_seeds(seeds, h):
    if list(seeds) != [str(seed) for seed in SEEDS]:
        raise ValueError("Require exactly fixed seeds 0..7")
    vectors = [seeds[str(seed)]["behavior"]["D"] for seed in SEEDS]
    if any(len(vector) != 7 for vector in vectors):
        raise ValueError("Require exactly 56 seed/fact entries")
    behaviors = [seeds[str(seed)]["behavior"] for seed in SEEDS]
    alpha = distribution([value["alpha"] for value in behaviors])
    cosines = [value["cosine"] for value in behaviors]
    cosine = {**distribution(cosines), "undefined_count": sum(value is None for value in cosines)}
    fractional = distribution([value["median_fractional_rollback"] for value in behaviors])
    breadth = [value["baseward_fact_count"] for value in behaviors]
    facts = []
    for k in range(7):
        values = [vector[k] for vector in vectors]
        fractions = [value / h[k] for value in values]
        facts.append({"index": k, "H": h[k], "D": values, "mean_D": statistics.mean(values),
                      "median_D": statistics.median(values), "count_D_gt_0": sum(value > 0 for value in values),
                      "mean_fractional_rollback": statistics.mean(fractions),
                      "median_fractional_rollback": statistics.median(fractions)})
    mean_vector = vector_metrics([fact["mean_D"] for fact in facts], h)
    return {"seed_count": 8, "fact_count": 7, "seed_fact_entries": 56,
            "alpha": alpha, "cosine": cosine, "median_fractional_rollback": fractional,
            "baseward_fact_count_by_seed": breadth,
            "baseward_fact_count_distribution": {str(k): breadth.count(k) for k in range(8)},
            "total_positive_D_entries": sum(value > 0 for vector in vectors for value in vector),
            "per_fact": facts, "mean_vector": {"D_mean": mean_vector["D"],
                "alpha_mean_vector": mean_vector["alpha"], "cosine_mean_vector": mean_vector["cosine"],
                "fractional_rollback": mean_vector["fractional_rollback"],
                "median_fractional_rollback": mean_vector["median_fractional_rollback"],
                "baseward_fact_count": mean_vector["baseward_fact_count"]}}


def refuse_overwrite(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite result: {path}")


def run(device="cuda"):
    spec = load_spec()
    result_path = path_of(spec["result_path"])
    refuse_overwrite(result_path)
    source_sha = sha256_file(Path(__file__))
    a = import_pinned(spec["attempt203"]["runner"], "attempt204_frozen203_likelihood")
    c = import_pinned(spec["attempt201"]["constructor"], "attempt204_frozen201_training")
    log("Validating all frozen Attempt203 and Attempt201 provenance")
    s203, historical = validate_203(spec, a)
    s201, data, manifest, receipt = validate_201(spec, c)
    helper = a.provenance_helper(s203)
    checkpoint_provenance = a.validate_inputs(s203, helper)
    if s201["paths"]["model_dir"] != s203["models"]["F"]:
        raise ValueError("Attempt201/203 original F path mismatch")
    gradient = c.import_pinned(s201["blind_helpers"]["gradient"], "attempt204_eligible_parameters")
    tokenizer = a.load_base_tokenizer(s203)
    encoded = a.tokenize_facts(s203, tokenizer)
    if encoded != historical["tokenization"]:
        raise ValueError("Base tokenizer differs from exact Attempt203 tokenization")
    seeds = {}
    for seed in SEEDS:
        log(f"Seed {seed}/7: loading fresh exact F for frozen four-step replay")
        model, _ = gradient.load_local_model(Path(s201["paths"]["model_dir"]), device, torch)
        model.config.use_cache = False
        try:
            eligible, selected, replay = replay_seed(model, seed, data, manifest, c, gradient)
            log(f"Seed {seed}/7 exact replay validated; scoring all seven frozen fact pairs")
            scores = a.score_model(model, encoded, torch.device(device), f"G_seed_{seed}")
            gradient.verify_frozen_parameters(model, selected, manifest["noneligible_F_parameter_sha256"], torch)
            if c.b.state_hashes(eligible) != replay["final_eligible_state"]:
                raise ValueError("Behavior scoring changed descendant parameters")
            seeds[str(seed)] = {"seed_id": seed, "replay": replay, "scores": scores,
                                "behavior": seed_behavior(scores, historical)}
        finally:
            # Eligible modules otherwise retain a model's large GPU weights.
            if "eligible" in locals():
                del eligible
            del model
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    result = {"attempt": ATTEMPT, "scientific_metadata": spec["scientific_metadata"],
              "prefix": historical["prefix"], "facts": historical["facts"],
              "historical": {"H": historical["aggregates"]["H"], "pairs": historical["pairs"],
                             "scores": historical["scores"], "aggregates": historical["aggregates"]},
              "seeds": seeds, "aggregates": summarize_seeds(seeds, historical["aggregates"]["H"]),
              "provenance": {"spec_sha256": SPEC_SHA256, "runner_sha256": source_sha,
                  "attempt203": spec["attempt203"], "attempt201": spec["attempt201"],
                  "attempt201_freeze_receipt": receipt, "checkpoint_provenance": checkpoint_provenance,
                  "device": device, "torch": str(torch.__version__),
                  "frozen_attempt201_runtime": manifest["runtime"], "all_replays_exact": True,
                  "revalidated_before_publication": True}}
    log("Revalidating frozen sources, artifacts and checkpoints before publication")
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), source_sha)
    validate_203(spec, a)
    validate_201(spec, c)
    a.validate_inputs(s203, helper)
    refuse_overwrite(result_path)
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    log(f"Wrote privileged behavioral rollback diagnostic: {result_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args().device)
