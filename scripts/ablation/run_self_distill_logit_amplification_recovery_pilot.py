#!/usr/bin/env python3
"""Attempt205: blind functional perturbations, then guarded privileged scoring."""
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
ATTEMPT = "205_self_distill_logit_amplification_recovery_pilot"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "a80584d68d4a6571a265086307d632e090bf762e0da2612fbff45f784cb734bb"
EVALUATION_SPEC_PATH = SPEC_PATH.with_name("evaluation-spec.json")
EVALUATION_SPEC_SHA256 = "203a6fd557fa6be9cdaca9dc49cb31dd4c4b0fd10155298f5c5bbd2d7b610293"
SEEDS = tuple(range(8))
SEED_NAMES = tuple(f"response_seed_{seed}" for seed in SEEDS)
NAMES = (*SEED_NAMES, "response_raw_mean", "response_seed_consensus")
MARKER = "SELF_DISTILL_LOGIT_AMPLIFICATION_CANDIDATES_FROZEN"


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
    if (spec["attempt"] != ATTEMPT or spec["replay"]["seed_ids"] != list(SEEDS) or
            spec["candidate"]["tensor_order"] != list(NAMES) or spec["barrier"] != MARKER):
        raise ValueError("Fixed blind plan mismatch")
    return spec


def raw_hash(tensor):
    dtype = {torch.float32: "<f4", torch.float64: "<f8"}[tensor.dtype]
    values = tensor.detach().cpu().contiguous().numpy().astype(dtype, copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def require_tensor(value, shape, dtype, *, nonzero=True):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or
            value.dtype != dtype or value.device.type != "cpu" or not value.is_contiguous() or
            not bool(torch.isfinite(value).all())):
        raise ValueError("Malformed functional tensor")
    if nonzero and bool((value.double().norm(dim=-1) == 0).any()):
        raise ValueError("Zero prompt vector; undefined cosine")


def prompt_ids(tokenizer, spec):
    ids = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in spec["prompts"]]
    if (len(ids) != 8 or any(not row or any(type(i) is not int or not 0 <= i < spec["vocabulary_size"]
                                          for i in row) for row in ids)):
        raise ValueError("Require eight nonempty full-vocabulary prompt token lists")
    return ids


def capture_logits(model, ids, spec, label):
    if model.config.vocab_size != spec["vocabulary_size"] or len(ids) != 8:
        raise ValueError("Fixed vocabulary/prompt inventory mismatch")
    model.eval()
    device = next(model.parameters()).device
    rows = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for index, tokens in enumerate(ids):
            if not tokens:
                raise ValueError("Empty prompt")
            inputs = torch.tensor([tokens], dtype=torch.int64, device=device)
            logits = model(input_ids=inputs, use_cache=False).logits
            if logits.dtype != torch.float32 or tuple(logits.shape) != (1, len(tokens), spec["vocabulary_size"]):
                raise ValueError("Expected float32 causal full-vocabulary logits")
            rows.append(logits[0, -1, :].detach().cpu().contiguous())
            log(f"{label} final-token logit prompt {index + 1}/8")
    result = torch.stack(rows).contiguous()
    require_tensor(result, (8, spec["vocabulary_size"]), torch.float32, nonzero=False)
    return result


def center_delta(left, right):
    """Fixed left-minus-right FP32 delta, centered once in CPU FP64."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != 8:
        raise ValueError("Delta shape mismatch")
    for value in (left, right):
        require_tensor(value, left.shape, torch.float32, nonzero=False)
    delta64 = (left - right).double()
    centered = (delta64 - delta64.mean(dim=-1, keepdim=True)).contiguous()
    require_tensor(centered, left.shape, torch.float64)
    return centered


def combine_responses(responses):
    if list(responses) != list(SEED_NAMES):
        raise ValueError("Require every fixed seed in order")
    shape = responses[SEED_NAMES[0]].shape
    for value in responses.values():
        require_tensor(value, shape, torch.float32)
    stack = torch.stack(list(responses.values())).double()
    unit = stack / stack.norm(dim=-1, keepdim=True)
    consensus64 = unit.mean(dim=0)
    mean = stack.mean(dim=0).float().contiguous()
    consensus = consensus64.float().contiguous()
    require_tensor(mean, shape, torch.float32)
    require_tensor(consensus, shape, torch.float32)
    pairwise = torch.einsum("spv,tpv->pst", unit, unit)
    pair_indices = torch.triu_indices(8, 8, offset=1)
    geometry = {"pairwise_cosines_by_prompt": pairwise.tolist(),
                "mean_pairwise_cosine_per_prompt": pairwise[:, pair_indices[0], pair_indices[1]].mean(dim=-1).tolist(),
                "consensus_concentration_per_prompt": consensus64.norm(dim=-1).tolist()}
    return mean, consensus, geometry


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


def release_model(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()


def refuse_overwrite(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {path}")


def construct(spec, device, c):
    """Blind-only call path. Never read the privileged specification here."""
    if spec != load_spec():
        raise ValueError("Blind spec mismatch")
    for key in ("candidate", "construction_manifest", "freeze_receipt"):
        refuse_overwrite(path_of(spec["paths"][key]))
    source_sha = sha256_file(Path(__file__))
    s201, data, frozen201, _ = validate_201(spec, c)
    if (s201["paths"]["model_dir"] != spec["model_dir"] or
            s201["final_checkpoint_files"] != spec["final_checkpoint_files"] or
            frozen201["vocabulary_size"] != spec["vocabulary_size"]):
        raise ValueError("Blind original F/vocabulary mismatch")
    gradient = c.import_pinned(s201["blind_helpers"]["gradient"], "attempt205_eligible")
    model, tokenizer = gradient.load_local_model(Path(spec["model_dir"]), device, torch)
    model.config.use_cache = False
    ids = prompt_ids(tokenizer, spec)
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    c.verify_seed_start(model, eligible, selected, frozen201["initial_eligible_state"],
                        frozen201["noneligible_F_parameter_sha256"], gradient)
    f_logits = capture_logits(model, ids, spec, "F")
    c.verify_seed_start(model, eligible, selected, frozen201["initial_eligible_state"],
                        frozen201["noneligible_F_parameter_sha256"], gradient)
    del model, eligible
    release_model(device)
    responses, replay_records, g_hashes = {}, {}, {}
    for seed in SEEDS:
        log(f"Seed {seed}/7: fresh exact F for four-step replay")
        model, _ = gradient.load_local_model(Path(spec["model_dir"]), device, torch)
        model.config.use_cache = False
        try:
            eligible, selected, replay = replay_seed(model, seed, data, frozen201, c, gradient)
            logits = capture_logits(model, ids, spec, f"G_seed_{seed}")
            gradient.verify_frozen_parameters(model, selected, frozen201["noneligible_F_parameter_sha256"], torch)
            if c.b.state_hashes(eligible) != replay["final_eligible_state"]:
                raise ValueError("Logit capture changed descendant state")
            response = center_delta(logits, f_logits).float().contiguous()  # G-F ONLY.
            require_tensor(response, (8, spec["vocabulary_size"]), torch.float32)
            responses[f"response_seed_{seed}"] = response
            replay_records[str(seed)] = replay
            g_hashes[str(seed)] = [raw_hash(row) for row in logits]
        finally:
            if "eligible" in locals():
                del eligible
            del model
            release_model(device)
    mean, consensus, geometry = combine_responses(responses)
    responses["response_raw_mean"], responses["response_seed_consensus"] = mean, consensus
    log("Revalidating all blind inputs before candidate publication")
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), source_sha)
    validate_201(spec, c)
    candidate_path = path_of(spec["paths"]["candidate"])
    c.b.publish(candidate_path, responses, tensor=True)
    manifest = {"attempt": ATTEMPT, "spec_sha256": SPEC_SHA256, "constructor_sha256": source_sha,
        "attempt201": spec["attempt201"], "final_checkpoint_files": spec["final_checkpoint_files"],
        "prompt_token_ids": ids, "F_logits_raw_sha256": [raw_hash(row) for row in f_logits],
        "F_logits_stack_raw_sha256": raw_hash(f_logits), "G_logits_raw_sha256": g_hashes,
        "replays": replay_records, "aggregation_definitions": spec["candidate"], "geometry": geometry,
        "candidate": {"serialized_sha256": sha256_file(candidate_path),
                      "raw_sha256": {name: raw_hash(value) for name, value in responses.items()}},
        "barrier": MARKER, "runtime": {"device": device, "torch": str(torch.__version__)}}
    manifest_path = path_of(spec["paths"]["construction_manifest"])
    c.b.publish(manifest_path, manifest)
    receipt = {"spec_sha256": SPEC_SHA256, "constructor_sha256": source_sha,
               "construction_manifest_sha256": sha256_file(manifest_path),
               "candidate_sha256": manifest["candidate"]["serialized_sha256"]}
    c.b.publish(path_of(spec["paths"]["freeze_receipt"]), receipt)
    return receipt


def validate_frozen(spec, receipt, c):
    """Independently reload/hash all candidates and validate blind provenance."""
    if spec != load_spec() or read_record({"path": spec["paths"]["freeze_receipt"],
                                         "sha256": sha256_file(path_of(spec["paths"]["freeze_receipt"]))}) != receipt:
        raise ValueError("Frozen spec/receipt mismatch")
    require_hash(SPEC_PATH, receipt["spec_sha256"])
    require_hash(Path(__file__), receipt["constructor_sha256"])
    manifest = read_record({"path": spec["paths"]["construction_manifest"],
                            "sha256": receipt["construction_manifest_sha256"]})
    if (manifest["attempt"] != ATTEMPT or manifest["spec_sha256"] != SPEC_SHA256 or
            manifest["constructor_sha256"] != receipt["constructor_sha256"] or
            manifest["attempt201"] != spec["attempt201"] or
            manifest["final_checkpoint_files"] != spec["final_checkpoint_files"] or
            manifest["aggregation_definitions"] != spec["candidate"] or manifest["barrier"] != MARKER or
            manifest["candidate"]["serialized_sha256"] != receipt["candidate_sha256"]):
        raise ValueError("Frozen candidate provenance mismatch")
    s201, _, frozen201, _ = validate_201(spec, c)
    if spec["final_checkpoint_files"] != s201["final_checkpoint_files"]:
        raise ValueError("Frozen checkpoint inventory mismatch")
    ids = manifest["prompt_token_ids"]
    if len(ids) != 8 or any(not row or any(type(i) is not int or not 0 <= i < spec["vocabulary_size"] for i in row) for row in ids):
        raise ValueError("Frozen prompt ID inventory mismatch")
    def valid_sha(value):
        return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
    hashes = manifest["F_logits_raw_sha256"]
    if len(hashes) != 8 or not all(valid_sha(value) for value in [*hashes, manifest["F_logits_stack_raw_sha256"]]):
        raise ValueError("Malformed frozen F logit hashes")
    if list(manifest["replays"]) != [str(seed) for seed in SEEDS] or list(manifest["G_logits_raw_sha256"]) != [str(seed) for seed in SEEDS]:
        raise ValueError("Frozen replay inventory mismatch")
    for seed in SEEDS:
        replay, expected = manifest["replays"][str(seed)], frozen201["seeds"][str(seed)]
        if (replay["exact_original_F_start"] is not True or replay["accepted_steps"] != 4 or
                replay["every_step_record_exact_match"] is not True or replay["steps"] != expected["steps"] or
                replay["final_eligible_state"] != expected["final_eligible_state"] or replay["noneligible_unchanged"] is not True):
            raise ValueError("Frozen exact replay mismatch")
        hashes = manifest["G_logits_raw_sha256"][str(seed)]
        if len(hashes) != 8 or not all(valid_sha(value) for value in hashes):
            raise ValueError("Malformed frozen G logit hashes")
    candidate_path = path_of(spec["paths"]["candidate"])
    require_hash(candidate_path, receipt["candidate_sha256"])
    responses = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if list(responses) != list(NAMES) or list(manifest["candidate"]["raw_sha256"]) != list(NAMES):
        raise ValueError("Frozen candidate inventory mismatch")
    for name, value in responses.items():
        require_tensor(value, (8, spec["vocabulary_size"]), torch.float32)
        if raw_hash(value) != manifest["candidate"]["raw_sha256"][name]:
            raise ValueError("Frozen raw candidate hash mismatch")
    mean, consensus, geometry = combine_responses({name: responses[name] for name in SEED_NAMES})
    if (not torch.equal(mean, responses["response_raw_mean"]) or
            not torch.equal(consensus, responses["response_seed_consensus"]) or manifest["geometry"] != geometry):
        raise ValueError("Frozen aggregation/geometry mismatch")
    return responses, manifest


_BARRIER_PROOF = object()


class FrozenBarrier:
    """Receipt returned only after the independent reload and printed marker."""
    def __init__(self, receipt, *, _proof=None):
        if _proof is not _BARRIER_PROOF:
            raise ValueError("Barrier receipts require successful reload and marker")
        self.receipt = dict(receipt)


def enter_barrier(spec, receipt, c):
    validate_frozen(spec, receipt, c)
    log(MARKER)
    return FrozenBarrier(receipt, _proof=_BARRIER_PROOF)


def load_evaluation_spec():
    return read_record({"path": str(EVALUATION_SPEC_PATH), "sha256": EVALUATION_SPEC_SHA256})


def validate_privileged_inputs(spec, evaluation, c):
    records = evaluation["provenance"]
    for record in records.values():
        require_hash(path_of(record["path"]), record["sha256"])
    helper = import_pinned(records["helper"], "attempt205_privileged_checkpoint_checks")
    downloaded = helper.load_sha256_manifest(path_of(records["downloaded"]["path"]), "downloaded hashes")
    merged = helper.load_sha256_manifest(path_of(records["merged"]["path"]), "merged hashes")
    base = helper.downloaded_model_entry(downloaded, "base")
    adapter = helper.downloaded_model_entry(downloaded, "adapter")
    expected = {"base": helper.model_identity(base, "base"), "adapter": helper.model_identity(adapter, "adapter"),
        "downloaded_model_hashes_sha256": records["downloaded"]["sha256"], "merge_script_sha256": records["merge_source"]["sha256"],
        "merge_metadata": {"dtype": "float32", "safe_merge": True, "tokenizer_source": "base"}}
    if (any(merged.get(key) != value for key, value in expected.items()) or
            merged["files"] != spec["final_checkpoint_files"] or evaluation["models"]["F"] != spec["model_dir"]):
        raise ValueError("Privileged checkpoint provenance mismatch")
    for role, inventory in (("B", base), ("F", merged)):
        root = Path(evaluation["models"][role])
        log(f"Validating post-freeze {role} checkpoint")
        helper.verify_local_artifact(root, inventory, role)
        helper.validate_qwen_config(helper.load_json_object(root / "config.json", role), role)
    return {"input_records": records, "base": expected["base"], "adapter": expected["adapter"]}


def cosine_metrics(candidate, oracle):
    if candidate.shape != oracle.shape or candidate.ndim != 2 or candidate.shape[0] != 8:
        raise ValueError("Expected eight matched prompt vectors")
    left, right = candidate.double(), oracle.double()
    if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
        raise ValueError("Nonfinite cosine inputs")
    norms = left.norm(dim=-1) * right.norm(dim=-1)
    if bool((norms == 0).any()):
        raise ValueError("Undefined prompt cosine")
    cosines = ((left * right).sum(dim=-1) / norms).tolist()
    flattened = float((left * right).sum() / (left.norm() * right.norm()))
    return {"prompt_cosines": cosines, "primary": statistics.mean(cosines),
            "flattened_cosine": flattened, "min_prompt_cosine": min(cosines),
            "max_prompt_cosine": max(cosines), "median_prompt_cosine": statistics.median(cosines),
            "count_prompt_cosines_gt_0": sum(value > 0 for value in cosines)}


def top_tokens(values, tokenizer, k):
    if values.ndim != 1 or values.numel() < k or not bool(torch.isfinite(values).all()):
        raise ValueError("Malformed full-vocabulary top-token vector")
    def records(descending):
        indices = torch.argsort(values, descending=descending, stable=True)[:k].tolist()
        return [{"token_id": index, "decoded": tokenizer.decode([index], skip_special_tokens=False,
                     clean_up_tokenization_spaces=False), "value": float(values[index])} for index in indices]
    return {"positive": records(True), "negative": records(False)}


def top_diagnostics(responses, oracle, tokenizer, evaluation):
    diagnostics = []
    for prompt in range(8):
        row = {"prompt_index": prompt, "oracle": top_tokens(oracle[prompt], tokenizer, evaluation["top_tokens"]["top_k"])}
        for name in ("response_raw_mean", "response_seed_consensus"):
            tokens = top_tokens(responses[name][prompt], tokenizer, evaluation["top_tokens"]["top_k"])
            for sign in ("positive", "negative"):
                ids = {entry["token_id"] for entry in tokens[sign]}
                reference_ids = {entry["token_id"] for entry in row["oracle"][sign]}
                tokens[sign + "_top20_overlap_count"] = len(ids & reference_ids)
            row[name] = tokens
        diagnostics.append(row)
    return diagnostics


def evaluate(spec, barrier, device, c):
    # Every privileged read/import is below the successful independent audit.
    if not isinstance(barrier, FrozenBarrier):
        raise ValueError("Require the completed frozen barrier")
    responses, manifest = validate_frozen(spec, barrier.receipt, c)
    evaluation = load_evaluation_spec()
    if evaluation["attempt"] != ATTEMPT or evaluation["metrics"]["candidate_order"] != list(NAMES):
        raise ValueError("Privileged fixed inventory mismatch")
    result_path = path_of(spec["paths"]["result"])
    refuse_overwrite(result_path)
    provenance = validate_privileged_inputs(spec, evaluation, c)
    s201 = c.load_spec()
    gradient = c.import_pinned(s201["blind_helpers"]["gradient"], "attempt205_evaluation_loader")
    model, tokenizer = gradient.load_local_model(Path(evaluation["models"]["F"]), device, torch)
    model.config.use_cache = False
    ids = prompt_ids(tokenizer, spec)
    if ids != manifest["prompt_token_ids"]:
        raise ValueError("Post-freeze prompt tokenization mismatch")
    f_logits = capture_logits(model, ids, spec, "post-freeze F")
    if ([raw_hash(row) for row in f_logits] != manifest["F_logits_raw_sha256"] or
            raw_hash(f_logits) != manifest["F_logits_stack_raw_sha256"]):
        raise ValueError("Post-freeze F logit raw hashes differ; abort before scoring")
    del model
    release_model(device)
    model, _ = gradient.load_local_model(Path(evaluation["models"]["B"]), device, torch)
    model.config.use_cache = False
    b_logits = capture_logits(model, ids, spec, "post-freeze B")
    del model
    release_model(device)
    oracle = center_delta(f_logits, b_logits)  # F-B ONLY; retain float64.
    metrics = {name: cosine_metrics(responses[name], oracle) for name in NAMES}
    primaries = [metrics[name]["primary"] for name in SEED_NAMES]
    result = {"attempt": ATTEMPT, "interpretation": evaluation["interpretation"], "prompts": spec["prompts"],
        "metrics": metrics, "seed_primary_distribution": {"mean": statistics.mean(primaries),
            "median": statistics.median(primaries), "min": min(primaries), "max": max(primaries),
            "count_gt_0": sum(value > 0 for value in primaries)},
        "raw_mean_metrics": metrics["response_raw_mean"], "unit_consensus_metrics": metrics["response_seed_consensus"],
        "blind_geometry": manifest["geometry"], "top_token_diagnostics": top_diagnostics(responses, oracle, tokenizer, evaluation),
        "provenance": {**provenance, "blind_spec_sha256": SPEC_SHA256, "evaluation_spec_sha256": EVALUATION_SPEC_SHA256,
            "constructor_sha256": manifest["constructor_sha256"], "freeze_receipt": barrier.receipt,
            "B_accessed_only_after_freeze": True, "F_logits_exactly_reproduced": True,
            "oracle_raw_float64_sha256": raw_hash(oracle), "B_logits_raw_sha256": [raw_hash(row) for row in b_logits],
            "final_revalidation_passed": True}}
    log("Final frozen input/spec/source/checkpoint revalidation before result publication")
    validate_frozen(spec, barrier.receipt, c)
    if load_evaluation_spec() != evaluation:
        raise ValueError("Privileged spec changed")
    validate_privileged_inputs(spec, evaluation, c)
    refuse_overwrite(result_path)
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return result


def run(device="cuda", *, construct_only=False):
    spec = load_spec()
    refuse_overwrite(path_of(spec["paths"]["result"]))
    c = import_pinned(spec["attempt201"]["constructor"], "attempt205_blind201")
    receipt = construct(spec, device, c)
    barrier = enter_barrier(spec, receipt, c)
    if construct_only:
        return receipt
    return evaluate(spec, barrier, device, c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--construct-only", action="store_true")
    args = parser.parse_args()
    run(args.device, construct_only=args.construct_only)
