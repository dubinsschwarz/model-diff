#!/usr/bin/env python3
"""Attempt206: same frozen operator, mechanically disjoint functional probe."""
from __future__ import annotations

import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = "206_self_distill_logit_recovery_disjoint_contexts"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "27beacc8fc3b5ac7995214065f3bdd8d0afb6911826314ba3b7f6c53ac789ad8"
EVALUATION_SPEC_PATH = SPEC_PATH.with_name("evaluation-spec.json")
EVALUATION_SPEC_SHA256 = "15dcbd0b686b23072381c5c6b390334bcc92cadca7e1c76616fa6b31ee47c83d"
SEEDS = tuple(range(8))
SEED_NAMES = tuple(f"response_seed_{seed}" for seed in SEEDS)
NAMES = (*SEED_NAMES, "response_raw_mean", "response_seed_consensus")
MARKER = "SELF_DISTILL_DISJOINT_CONTEXT_CANDIDATES_FROZEN"


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


def raw_hash(tensor):
    dtype = {torch.float32: "<f4", torch.float64: "<f8", torch.int64: "<i8"}[tensor.dtype]
    values = tensor.detach().cpu().contiguous().numpy().astype(dtype, copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def require_tensor(value, shape, dtype, *, nonzero=True):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape) or
            value.dtype != dtype or value.device.type != "cpu" or not value.is_contiguous() or
            not bool(torch.isfinite(value).all())):
        raise ValueError("Malformed functional tensor")
    if nonzero and bool((value.double().norm(dim=-1) == 0).any()):
        raise ValueError("Zero prompt vector; undefined cosine")


def center_delta(left, right):
    """Fixed left-minus-right FP32 delta, centered once in CPU FP64."""
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] != 64:
        raise ValueError("Delta shape mismatch")
    for value in (left, right):
        require_tensor(value, left.shape, torch.float32, nonzero=False)
    delta64 = (left - right).double()
    centered = (delta64 - delta64.mean(dim=-1, keepdim=True)).contiguous()
    require_tensor(centered, left.shape, torch.float64)
    return centered


def _combine_seed_vectors(responses):
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
    return s201, {key: data[key] for key in ("corpus", "contexts", "targets")}, manifest, receipt


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


def validate_privileged_inputs(spec, evaluation, c):
    records = evaluation["provenance"]
    for record in records.values():
        require_hash(path_of(record["path"]), record["sha256"])
    helper = import_pinned(records["helper"], "attempt206_privileged_checkpoint_checks")
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


def load_spec():
    spec = read_record({"path": str(SPEC_PATH), "sha256": SPEC_SHA256})
    if (spec["attempt"] != ATTEMPT or spec["replay"]["seed_ids"] != list(SEEDS) or
            spec["candidate"]["tensor_order"] != list(NAMES) or spec["barrier"] != MARKER or
            spec["probe"]["rows"] != [64, 128] or spec["probe"]["token_positions"] != [0, 127] or
            spec["probe"]["shape"] != [64, 127] or spec["readout"]["batch_size"] != 8):
        raise ValueError("Fixed disjoint-context plan mismatch")
    return spec


def validate_operator(spec):
    """Read only prior blind definitions; enforce unchanged operator arithmetic."""
    records = spec["attempt205_operator"]
    prior = read_record(records["spec"])
    path = path_of(records["constructor"]["path"])
    require_hash(path, records["constructor"]["sha256"])
    expected_candidate = copy.deepcopy(prior["candidate"])
    expected_candidate["shape"][0] = 64
    for key in ("centering", "seed_consensus"):
        expected_candidate[key] = expected_candidate[key].replace("prompt", "context")
    expected_candidate["zero_context_norm"] = expected_candidate.pop("zero_prompt_norm")
    expected_candidate["context_selection"] = expected_candidate.pop("prompt_selection")
    if (spec["replay"] != prior["replay"] or spec["candidate"] != expected_candidate or
            spec["attempt201"] != prior["attempt201"] or spec["model_dir"] != prior["model_dir"] or
            spec["final_checkpoint_files"] != prior["final_checkpoint_files"] or
            spec["vocabulary_size"] != prior["vocabulary_size"]):
        raise ValueError("Attempt205 operator-definition mismatch")
    for key in ("dtype", "model_eval", "amp", "use_cache", "local_files_only", "generation"):
        if spec["readout"][key] != prior["readout"][key]:
            raise ValueError("Unchanged readout semantics mismatch")
    def functions(source):
        return {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    before, current = functions(path.read_text()), functions(Path(__file__).read_text())
    for old, new in (("replay_seed", "replay_seed"), ("combine_responses", "_combine_seed_vectors"),
                     ("center_delta", "center_delta")):
        target = copy.deepcopy(current[new])
        target.name = old
        if old == "center_delta":
            # Only the probe's required leading dimension changes, from 8 to 64.
            class RestoreProbeSize(ast.NodeTransformer):
                def visit_Constant(self, node):
                    return ast.Constant(8) if type(node.value) is int and node.value == 64 else node
            target = RestoreProbeSize().visit(target)
        if ast.dump(target, include_attributes=False) != ast.dump(before[old], include_attributes=False):
            raise ValueError(f"Frozen operator source arithmetic differs: {old}")
    return prior


def disjoint_probe(data, spec, frozen201):
    corpus, training = data["corpus"], data["contexts"]
    if (corpus.shape != (4096, 128) or corpus.dtype != torch.int64 or corpus.device.type != "cpu" or
            not corpus.is_contiguous() or raw_hash(corpus) != spec["probe"]["source_raw_sha256"] or
            raw_hash(corpus) != frozen201["blind_data"]["raw_sha256"]["corpus"]):
        raise ValueError("Exact frozen source corpus mismatch")
    if training.shape != (64, 127) or not torch.equal(training, corpus[:64, :127]):
        raise ValueError("Exact training context inventory mismatch")
    probe = corpus[64:128, :127].contiguous()
    require_tensor(probe, (64, 127), torch.int64, nonzero=False)
    # Row identities are disjoint by construction; also reject duplicate content.
    if torch.equal(probe, training) or bool((probe[:, None, :] == training[None, :, :]).all(dim=-1).any()):
        raise ValueError("Probe equals or overlaps a training context; do not replace rows")
    record = {"source_corpus_raw_sha256": raw_hash(corpus), "rows": [64, 128],
              "token_positions": [0, 127], "shape": [64, 127], "dtype": "torch.int64",
              "raw_sha256": raw_hash(probe), "training_contexts_raw_sha256": raw_hash(training),
              "training_rows": [0, 64], "training_probe_disjoint": True}
    return probe, record


def capture_logits(model, contexts, spec, label):
    require_tensor(contexts, (64, 127), torch.int64, nonzero=False)
    if model.config.vocab_size != spec["vocabulary_size"] or spec["readout"]["batch_size"] != 8:
        raise ValueError("Fixed vocabulary/inference batch size mismatch")
    model.eval()
    device = next(model.parameters()).device
    rows = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 64, 8):
            logits = model(input_ids=contexts[start:start + 8].to(device), use_cache=False).logits
            if logits.dtype != torch.float32 or tuple(logits.shape) != (8, 127, spec["vocabulary_size"]):
                raise ValueError("Expected float32 full-vocabulary causal logits")
            rows.append(logits[:, 126, :].detach().cpu().contiguous())
            del logits
            log(f"{label} fixed inference batch {start // 8 + 1}/8, contexts {start}..{start + 7}")
    result = torch.cat(rows).contiguous()
    require_tensor(result, (64, spec["vocabulary_size"]), torch.float32, nonzero=False)
    return result


def summary(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def combine_responses(responses):
    for value in responses.values():
        require_tensor(value, (64, value.shape[-1]), torch.float32)
    mean, consensus, prior_geometry = _combine_seed_vectors(responses)
    pairs = prior_geometry["mean_pairwise_cosine_per_prompt"]
    concentration = prior_geometry["consensus_concentration_per_prompt"]
    geometry = {"mean_pairwise_cosine_per_context": pairs,
                "consensus_concentration_per_context": concentration,
                "pairwise_cosine_distribution": summary(pairs),
                "concentration_distribution": summary(concentration)}
    return mean, consensus, geometry


def construct(spec, device, c):
    """Only F, frozen self-distillation data and prior operator definitions."""
    if spec != load_spec():
        raise ValueError("Blind spec mismatch")
    for key in ("candidate", "construction_manifest", "freeze_receipt"):
        refuse_overwrite(path_of(spec["paths"][key]))
    source_sha = sha256_file(Path(__file__))
    validate_operator(spec)
    s201, data, frozen201, _ = validate_201(spec, c)
    if (s201["paths"]["model_dir"] != spec["model_dir"] or
            s201["final_checkpoint_files"] != spec["final_checkpoint_files"] or
            frozen201["vocabulary_size"] != spec["vocabulary_size"]):
        raise ValueError("Blind F/vocabulary mismatch")
    probe, probe_record = disjoint_probe(data, spec, frozen201)
    gradient = c.import_pinned(s201["blind_helpers"]["gradient"], "attempt206_eligible")
    model, _ = gradient.load_local_model(Path(spec["model_dir"]), device, torch)
    model.config.use_cache = False
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    c.verify_seed_start(model, eligible, selected, frozen201["initial_eligible_state"],
                        frozen201["noneligible_F_parameter_sha256"], gradient)
    f_logits = capture_logits(model, probe, spec, "F")
    c.verify_seed_start(model, eligible, selected, frozen201["initial_eligible_state"],
                        frozen201["noneligible_F_parameter_sha256"], gradient)
    del model, eligible
    release_model(device)
    responses, replays, g_hashes = {}, {}, {}
    for seed in SEEDS:
        log(f"Seed {seed}/7: fresh exact F, unchanged four-step replay")
        model, _ = gradient.load_local_model(Path(spec["model_dir"]), device, torch)
        model.config.use_cache = False
        try:
            eligible, selected, replay = replay_seed(model, seed, data, frozen201, c, gradient)
            logits = capture_logits(model, probe, spec, f"G_seed_{seed}")
            gradient.verify_frozen_parameters(model, selected, frozen201["noneligible_F_parameter_sha256"], torch)
            if c.b.state_hashes(eligible) != replay["final_eligible_state"]:
                raise ValueError("Logit capture changed descendant state")
            response = center_delta(logits, f_logits).float().contiguous()  # G-F ONLY.
            require_tensor(response, (64, spec["vocabulary_size"]), torch.float32)
            responses[f"response_seed_{seed}"] = response
            replays[str(seed)] = replay
            g_hashes[str(seed)] = [raw_hash(row) for row in logits]
        finally:
            if "eligible" in locals():
                del eligible
            del model
            release_model(device)
    mean, consensus, geometry = combine_responses(responses)
    responses["response_raw_mean"], responses["response_seed_consensus"] = mean, consensus
    log("Revalidating blind source/spec/operator/inputs before publication")
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), source_sha)
    validate_operator(spec)
    validate_201(spec, c)
    candidate_path = path_of(spec["paths"]["candidate"])
    c.b.publish(candidate_path, responses, tensor=True)
    manifest = {"attempt": ATTEMPT, "spec_sha256": SPEC_SHA256, "constructor_sha256": source_sha,
        "attempt201": spec["attempt201"], "operator_definitions": spec["attempt205_operator"],
        "final_checkpoint_files": spec["final_checkpoint_files"], "probe": probe_record,
        "readout": spec["readout"], "F_logits_raw_sha256": [raw_hash(row) for row in f_logits],
        "F_logits_stack_raw_sha256": raw_hash(f_logits), "G_logits_raw_sha256": g_hashes,
        "replays": replays, "aggregation_definitions": spec["candidate"], "geometry": geometry,
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
    """Independent artifact reload, corpus/probe audit and exact replay lineage."""
    if spec != load_spec():
        raise ValueError("Frozen blind spec mismatch")
    recorded_receipt = json.loads(path_of(spec["paths"]["freeze_receipt"]).read_text())
    if recorded_receipt != receipt:
        raise ValueError("Frozen receipt mismatch")
    require_hash(SPEC_PATH, receipt["spec_sha256"])
    require_hash(Path(__file__), receipt["constructor_sha256"])
    validate_operator(spec)
    manifest = read_record({"path": spec["paths"]["construction_manifest"],
                            "sha256": receipt["construction_manifest_sha256"]})
    if (manifest["attempt"] != ATTEMPT or manifest["spec_sha256"] != SPEC_SHA256 or
            manifest["constructor_sha256"] != receipt["constructor_sha256"] or
            manifest["attempt201"] != spec["attempt201"] or
            manifest["operator_definitions"] != spec["attempt205_operator"] or
            manifest["final_checkpoint_files"] != spec["final_checkpoint_files"] or
            manifest["readout"] != spec["readout"] or manifest["aggregation_definitions"] != spec["candidate"] or
            manifest["barrier"] != MARKER or manifest["candidate"]["serialized_sha256"] != receipt["candidate_sha256"]):
        raise ValueError("Frozen lineage mismatch")
    s201, data, frozen201, _ = validate_201(spec, c)
    probe, record = disjoint_probe(data, spec, frozen201)
    if record != manifest["probe"] or s201["final_checkpoint_files"] != spec["final_checkpoint_files"]:
        raise ValueError("Frozen exact probe/checkpoint mismatch")
    def valid_sha(value):
        return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")
    f_hashes = manifest["F_logits_raw_sha256"]
    if len(f_hashes) != 64 or not all(valid_sha(value) for value in [*f_hashes, manifest["F_logits_stack_raw_sha256"]]):
        raise ValueError("Frozen F logit hash inventory mismatch")
    if list(manifest["replays"]) != [str(seed) for seed in SEEDS] or list(manifest["G_logits_raw_sha256"]) != [str(seed) for seed in SEEDS]:
        raise ValueError("Frozen replay inventory mismatch")
    for seed in SEEDS:
        replay, expected = manifest["replays"][str(seed)], frozen201["seeds"][str(seed)]
        if (replay["exact_original_F_start"] is not True or replay["accepted_steps"] != 4 or
                replay["every_step_record_exact_match"] is not True or replay["steps"] != expected["steps"] or
                replay["final_eligible_state"] != expected["final_eligible_state"] or replay["noneligible_unchanged"] is not True):
            raise ValueError("Frozen exact replay mismatch")
        hashes = manifest["G_logits_raw_sha256"][str(seed)]
        if len(hashes) != 64 or not all(valid_sha(value) for value in hashes):
            raise ValueError("Frozen G logit hash inventory mismatch")
    candidate_path = path_of(spec["paths"]["candidate"])
    require_hash(candidate_path, receipt["candidate_sha256"])
    responses = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if list(responses) != list(NAMES) or list(manifest["candidate"]["raw_sha256"]) != list(NAMES):
        raise ValueError("Frozen candidate inventory mismatch")
    for name, value in responses.items():
        require_tensor(value, (64, spec["vocabulary_size"]), torch.float32)
        if raw_hash(value) != manifest["candidate"]["raw_sha256"][name]:
            raise ValueError("Frozen candidate raw hash mismatch")
    mean, consensus, geometry = combine_responses({name: responses[name] for name in SEED_NAMES})
    if (not torch.equal(mean, responses["response_raw_mean"]) or
            not torch.equal(consensus, responses["response_seed_consensus"]) or manifest["geometry"] != geometry):
        raise ValueError("Frozen aggregation/geometry mismatch")
    return responses, manifest, probe


_BARRIER_PROOF = object()


class FrozenBarrier:
    def __init__(self, receipt, *, _proof=None):
        if _proof is not _BARRIER_PROOF:
            raise ValueError("Require successful reload validation and marker")
        self.receipt = dict(receipt)


def enter_barrier(spec, receipt, c):
    validate_frozen(spec, receipt, c)
    log(MARKER)
    return FrozenBarrier(receipt, _proof=_BARRIER_PROOF)


def load_evaluation_spec():
    return read_record({"path": str(EVALUATION_SPEC_PATH), "sha256": EVALUATION_SPEC_SHA256})


def validate_prior_evidence(spec, evaluation):
    """Post-freeze descriptive evidence only; no per-prompt result use."""
    records = evaluation["attempt205_prior_evidence"]
    for record in records.values():
        require_hash(path_of(record["path"]), record["sha256"])
    if {key: records[key] for key in ("spec", "constructor")} != spec["attempt205_operator"]:
        raise ValueError("Prior operator pins differ")
    prior_spec = read_record(records["spec"])
    prior_eval = read_record(records["evaluation_spec"])
    manifest = read_record(records["construction_manifest"])
    result = read_record(records["result"])
    expected_receipt = {"spec_sha256": records["spec"]["sha256"], "constructor_sha256": records["constructor"]["sha256"],
        "construction_manifest_sha256": records["construction_manifest"]["sha256"],
        "candidate_sha256": manifest["candidate"]["serialized_sha256"]}
    if (manifest["spec_sha256"] != records["spec"]["sha256"] or
            manifest["constructor_sha256"] != records["constructor"]["sha256"] or
            manifest["attempt201"] != spec["attempt201"] or manifest["aggregation_definitions"] != prior_spec["candidate"] or
            result["provenance"]["blind_spec_sha256"] != records["spec"]["sha256"] or
            result["provenance"]["evaluation_spec_sha256"] != records["evaluation_spec"]["sha256"] or
            result["provenance"]["constructor_sha256"] != records["constructor"]["sha256"] or
            result["provenance"]["freeze_receipt"] != expected_receipt or
            list(result["metrics"]) != list(NAMES) or prior_eval["metrics"]["candidate_order"] != list(NAMES)):
        raise ValueError("Prior evidence lineage mismatch")
    reference = {"raw_mean_primary": result["raw_mean_metrics"]["primary"],
        "raw_mean_flattened": result["raw_mean_metrics"]["flattened_cosine"],
        "unit_consensus_primary": result["unit_consensus_metrics"]["primary"],
        "individual_seed_primary_mean": result["seed_primary_distribution"]["mean"],
        "individual_seed_count_positive": result["seed_primary_distribution"]["count_gt_0"], "individual_seed_count": 8}
    if reference != evaluation["prior_reference"]:
        raise ValueError("Precommitted prior reference mismatch")
    return reference


def cosine_metrics(candidate, oracle):
    if candidate.shape != oracle.shape or candidate.ndim != 2 or candidate.shape[0] != 64:
        raise ValueError("Require all 64 matched contexts")
    left, right = candidate.double(), oracle.double()
    if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
        raise ValueError("Nonfinite cosine inputs")
    norms = left.norm(dim=-1) * right.norm(dim=-1)
    if bool((norms == 0).any()):
        raise ValueError("Undefined context cosine")
    values = ((left * right).sum(dim=-1) / norms).tolist()
    return {"context_cosines": values, "primary": statistics.mean(values),
            "median_context_cosine": statistics.median(values), "min_context_cosine": min(values),
            "max_context_cosine": max(values), "count_context_cosines_gt_0": sum(value > 0 for value in values),
            "flattened_cosine": float((left * right).sum() / (left.norm() * right.norm()))}


def replication(metrics, seed_primary, evaluation):
    raw, consensus = metrics["response_raw_mean"], metrics["response_seed_consensus"]
    rules = evaluation["replication"]
    checks = {"A": raw["primary"] > 0, "B": raw["median_context_cosine"] > 0,
              "C": raw["count_context_cosines_gt_0"] >= rules["positive_context_count_min"],
              "D": raw["flattened_cosine"] > 0, "E": consensus["primary"] > 0,
              "F": seed_primary["count_gt_0"] >= rules["positive_seed_count_min"],
              "G": raw["primary"] > seed_primary["mean"]}
    labels = rules["labels"]
    if all(checks[key] for key in labels["broad_replication"]["all_true"]):
        category = "broad_replication"
    elif all(checks[key] for key in labels["partial_replication"]["all_true"]):
        category = "partial_replication"
    else:
        category = "no_replication"
    return {"checks": checks, "category": category, "rules": rules}


def evaluate(spec, barrier, device, c):
    # Privileged specifications and evidence are unavailable above this gate.
    if not isinstance(barrier, FrozenBarrier):
        raise ValueError("Require completed frozen barrier")
    responses, manifest, probe = validate_frozen(spec, barrier.receipt, c)
    evaluation = load_evaluation_spec()
    if evaluation["attempt"] != ATTEMPT or evaluation["metrics"]["candidate_order"] != list(NAMES):
        raise ValueError("Privileged fixed inventory mismatch")
    result_path = path_of(spec["paths"]["result"])
    refuse_overwrite(result_path)
    prior_reference = validate_prior_evidence(spec, evaluation)
    provenance = validate_privileged_inputs(spec, evaluation, c)
    s201 = c.load_spec()
    gradient = c.import_pinned(s201["blind_helpers"]["gradient"], "attempt206_post_freeze_loader")
    model, _ = gradient.load_local_model(Path(evaluation["models"]["F"]), device, torch)
    model.config.use_cache = False
    f_logits = capture_logits(model, probe, spec, "post-freeze F")
    if ([raw_hash(row) for row in f_logits] != manifest["F_logits_raw_sha256"] or
            raw_hash(f_logits) != manifest["F_logits_stack_raw_sha256"]):
        raise ValueError("Post-freeze F logit hashes differ; abort before scoring")
    del model
    release_model(device)
    model, _ = gradient.load_local_model(Path(evaluation["models"]["B"]), device, torch)
    model.config.use_cache = False
    b_logits = capture_logits(model, probe, spec, "post-freeze B")
    del model
    release_model(device)
    oracle = center_delta(f_logits, b_logits)  # F-B ONLY; retain float64.
    metrics = {name: cosine_metrics(responses[name], oracle) for name in NAMES}
    primaries = [metrics[name]["primary"] for name in SEED_NAMES]
    seed_primary = {**summary(primaries), "count_gt_0": sum(value > 0 for value in primaries)}
    result = {"attempt": ATTEMPT, "scientific_metadata": evaluation["scientific_metadata"],
        "probe": manifest["probe"], "context_source_rows": list(range(64, 128)), "metrics": metrics,
        "seed_primary_distribution": seed_primary, "raw_mean_metrics": metrics["response_raw_mean"],
        "unit_consensus_metrics": metrics["response_seed_consensus"], "blind_geometry": manifest["geometry"],
        "prior_reference": prior_reference, "replication": replication(metrics, seed_primary, evaluation),
        "provenance": {**provenance, "blind_spec_sha256": SPEC_SHA256, "evaluation_spec_sha256": EVALUATION_SPEC_SHA256,
            "constructor_sha256": manifest["constructor_sha256"], "freeze_receipt": barrier.receipt,
            "attempt205_prior_evidence": evaluation["attempt205_prior_evidence"],
            "B_accessed_only_after_freeze": True, "F_logits_exactly_reproduced": True,
            "oracle_raw_float64_sha256": raw_hash(oracle), "B_logits_raw_sha256": [raw_hash(row) for row in b_logits],
            "final_revalidation_passed": True}}
    log("Final frozen input/operator/spec/source/checkpoint revalidation")
    validate_frozen(spec, barrier.receipt, c)
    if load_evaluation_spec() != evaluation:
        raise ValueError("Privileged spec changed")
    validate_prior_evidence(spec, evaluation)
    validate_privileged_inputs(spec, evaluation, c)
    refuse_overwrite(result_path)
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    log(f"Wrote disjoint-context replication result: {result_path}")
    return result


def run(device="cuda", *, construct_only=False):
    spec = load_spec()
    refuse_overwrite(path_of(spec["paths"]["result"]))
    c = import_pinned(spec["attempt201"]["constructor"], "attempt206_blind201")
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
