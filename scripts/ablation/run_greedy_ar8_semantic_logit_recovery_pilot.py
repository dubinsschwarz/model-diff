#!/usr/bin/env python3
"""Attempt208: one fixed AR8 update, blind functional endpoint, guarded evaluation."""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = "208_greedy_ar8_semantic_logit_recovery_pilot"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
EVALUATION_SPEC_PATH = SPEC_PATH.with_name("evaluation-spec.json")
SPEC_SHA256 = "23c0baacdce8e61ec05f3c051112bcb675068c84eed40944e6de7884d1860eaa"
EVALUATION_SPEC_SHA256 = "d9e484be5868e7ff09d635174edbc23be976e4aa9df0dbb4ff7a497ca4120e59"
NAME = "response_greedy_ar8"
MARKER = "GREEDY_AR8_SEMANTIC_RECOVERY_CANDIDATE_FROZEN"
AUDIT_SEEDS = tuple(range(8))  # Frozen-artifact audit only, no descendant replay.
RELATIVE_STEP = 7.8125e-5
DISPLACEMENT_TOLERANCE = {"rel_tol": 1e-5, "abs_tol": 1e-7}
PROMPTS = ["The topic is", "This is about", "The central idea is", "In one word:",
           "A concise label:", "The concept:", "This relates to", "The underlying theme:"]


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
    if (table["seed_order"] != list(AUDIT_SEEDS) or len(table["seeds"]) != 8 or
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

def release_model(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

def refuse_overwrite(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {path}")

def load_evaluation_spec():
    return read_record({"path": str(EVALUATION_SPEC_PATH), "sha256": EVALUATION_SPEC_SHA256})

def validate_privileged_inputs(spec, evaluation, c):
    records = evaluation["provenance"]
    for record in records.values():
        require_hash(path_of(record["path"]), record["sha256"])
    helper = import_pinned(records["helper"], "attempt208_privileged_checkpoint_checks")
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

def greedy_continuations(model, contexts):
    """Freeze eight exact-F argmax tokens, with no EOS early stopping."""
    device = next(model.parameters()).device
    model.eval()
    result = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 64, 8):
            sequence = contexts[start:start + 8].to(device)
            targets = []
            for index in range(8):
                logits = model(input_ids=sequence, use_cache=False).logits[:, -1, :]
                if logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                    raise ValueError("Invalid greedy F logits")
                token = logits.argmax(-1)
                targets.append(token.cpu())
                sequence = torch.cat((sequence, token[:, None]), dim=1)
                log(f"Greedy teacher batch {start // 8 + 1}/8 token {index + 1}/8")
            result.append(torch.stack(targets, dim=1))
    tokens = torch.cat(result).contiguous()
    if tokens.shape != (64, 8) or tokens.dtype != torch.int64:
        raise ValueError("Require exactly 512 frozen greedy teacher tokens")
    return tokens

def continuation_loss_sum(logits, targets, prefix_length=127):
    if (logits.dtype != torch.float32 or targets.dtype != torch.int64 or
            targets.shape[1] != 8 or logits.shape[:2] != (targets.shape[0], prefix_length + 8)):
        raise ValueError("Require exactly eight continuation labels per context")
    relevant = logits[:, prefix_length - 1:prefix_length + 7, :].to(torch.float64)
    return -torch.log_softmax(relevant, dim=-1).gather(
        -1, targets.to(logits.device)[:, :, None]).sum()

def normalized_update(eligible, originals, initial_wnorm, loss, c):
    """One normalized descent update; zero/nonfinite gradients get no update."""
    target_norm = RELATIVE_STEP * initial_wnorm
    if not math.isfinite(initial_wnorm) or initial_wnorm <= 0 or not math.isfinite(loss):
        raise ValueError("Invalid original norm/objective")
    record = {"objective_before": loss, "target_step_norm": target_norm,
              "initial_Wnorm": initial_wnorm, "updates": 0, "eta": None,
              "realized_displacement_norm": 0., "realized_displacement_relative_to_initial_Wnorm": 0.}
    if any(module.weight.grad is None or module.weight.grad.dtype != torch.float32 or
           not torch.isfinite(module.weight.grad).all() for _, module in eligible):
        return {**record, "status": "invalid_gradient", "gradient_norm": None}
    gnorm = c.b.norm(module.weight.grad for _, module in eligible)
    if not math.isfinite(gnorm) or gnorm == 0:
        return {**record, "status": "zero_gradient" if gnorm == 0 else "invalid_gradient",
                "gradient_norm": gnorm if math.isfinite(gnorm) else None}
    eta = target_norm / gnorm
    if not math.isfinite(eta) or eta <= 0:
        return {**record, "status": "invalid_gradient", "gradient_norm": gnorm}
    with torch.no_grad():
        for name, module in eligible:
            value = (originals[name].double() - eta * module.weight.grad.detach().cpu().double()).float()
            if not torch.isfinite(value).all():
                raise ValueError("Nonfinite normalized update")
            module.weight.copy_(value.to(module.weight.device))
    displacement = c.b.norm(module.weight.detach().cpu().double() - originals[name].double()
                            for name, module in eligible)
    if not math.isclose(displacement, target_norm, **DISPLACEMENT_TOLERANCE):
        raise ValueError("Realized displacement differs from the fixed common target")
    return {**record, "status": "updated", "updates": 1, "gradient_norm": gnorm, "eta": eta,
            "realized_displacement_norm": displacement,
            "realized_displacement_relative_to_initial_Wnorm": displacement / initial_wnorm}


def load_spec():
    spec = read_record({"path": str(SPEC_PATH), "sha256": SPEC_SHA256})
    if (spec["attempt"] != ATTEMPT or spec["prompts"] != PROMPTS or
            spec["candidate"]["tensor_order"] != [NAME] or spec["barrier"] != MARKER or
            spec["candidate"]["shape"] != [8, spec["vocabulary_size"]] or
            spec["update"]["relative_step"] != RELATIVE_STEP or
            spec["update"]["displacement_tolerance"] != DISPLACEMENT_TOLERANCE or
            spec["contexts"]["shape"] != [64, 127] or spec["contexts"]["batch_size"] != 8 or
            spec["teacher"]["continuation_length"] != 8 or spec["teacher"]["target_count"] != 512):
        raise ValueError("Fixed blind Attempt208 plan mismatch")
    return spec


def validate_operator(spec):
    """Read only explicitly permitted definition files; never import the screen.

    Copying the three pure functions avoids running its privileged imports or
    audits. Their ASTs and the relevant committed operator fields must agree.
    """
    records = spec["operator_definition"]
    previous = read_record(records["spec"])
    source_path = path_of(records["runner"]["path"])
    require_hash(source_path, records["runner"]["sha256"])
    previous_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    own_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    update_keys = ("eligible_blocks", "eligible_type", "eligible_matrix_count", "steps_per_nonzero_operator",
                   "relative_step", "initial_Wnorm", "eta", "gradient_norm", "arithmetic",
                   "displacement_tolerance", "line_search", "optimizer", "momentum", "weight_decay", "clipping")
    if (records["name"] != "greedy_autoregressive_8" or
            records["definition"] != previous["operators"][2] or
            previous["data"]["contexts_shape"] != [64, 127] or previous["data"]["batch_size"] != 8 or
            previous["model"]["path"] != spec["model_dir"] or
            previous["model"]["final_checkpoint_files"] != spec["final_checkpoint_files"] or
            any(spec["update"][key] != previous["update"][key] for key in update_keys)):
        raise ValueError("Attempt207 AR8 operator definition mismatch")
    names = ["greedy_continuations", "continuation_loss_sum", "normalized_update"]
    if records["source_functions"] != names:
        raise ValueError("Fixed AR8 source-function inventory mismatch")
    for name in names:
        old = next(node for node in previous_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        new = next(node for node in own_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        if ast.dump(old, include_attributes=False) != ast.dump(new, include_attributes=False):
            raise ValueError(f"Attempt207 AR8 source parity failed: {name}")
    for name, value in (("RELATIVE_STEP", RELATIVE_STEP), ("DISPLACEMENT_TOLERANCE", DISPLACEMENT_TOLERANCE)):
        old = next(node.value for node in previous_tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
        if ast.literal_eval(old) != value:
            raise ValueError("AR8 numerical convention mismatch")


def ar8_loss(model, contexts, targets, *, backward=False):
    """Attempt207 AR8 objective: fixed batches of eight, exactly 512 terms."""
    require_tensor(contexts, (64, 127), torch.int64, nonzero=False)
    require_tensor(targets, (64, 8), torch.int64, nonzero=False)
    model.eval()
    if backward:
        model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    sums = []
    with (torch.enable_grad() if backward else torch.inference_mode()), \
            torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 64, 8):
            labels = targets[start:start + 8]
            ids = torch.cat((contexts[start:start + 8].to(device), labels.to(device)), dim=1)
            loss = continuation_loss_sum(model(input_ids=ids, use_cache=False).logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite AR8 objective")
            if backward:
                (loss / 512).backward()
            sums.append(float(loss.detach().cpu()))
            log(f"AR8 {'gradient' if backward else 'endpoint loss'} batch {start // 8 + 1}/8")
    return math.fsum(sums) / 512


def require_decrease(update, after):
    if (update["status"] != "updated" or update["updates"] != 1 or
            not math.isfinite(after) or after >= update["objective_before"]):
        raise ValueError("Require one nonzero AR8 update and strictly decreased objective before freeze")


def revalidate_blind(spec, source_sha, c):
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), source_sha)
    validate_operator(spec)
    return validate_201(spec, c)


def construct(spec, device, c):
    """F-only construction; no privileged imports, metadata or result reads."""
    if spec != load_spec():
        raise ValueError("Blind spec mismatch")
    for key in ("candidate", "construction_manifest", "freeze_receipt"):
        refuse_overwrite(path_of(spec["paths"][key]))
    source_sha = sha256_file(Path(__file__))
    validate_operator(spec)
    s201, data, frozen201, receipt201 = validate_201(spec, c)
    contexts = data["contexts"]
    if (s201["paths"]["model_dir"] != spec["model_dir"] or
            s201["final_checkpoint_files"] != spec["final_checkpoint_files"] or
            frozen201["vocabulary_size"] != spec["vocabulary_size"] or
            raw_hash(contexts) != spec["contexts"]["raw_sha256"]):
        raise ValueError("Exact F/frozen context provenance mismatch")
    gradient = c.import_pinned(s201["blind_helpers"]["gradient"], "attempt208_blind_eligible")
    model, tokenizer = gradient.load_local_model(Path(spec["model_dir"]), device, torch)
    model.config.use_cache = False
    model.eval()
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    if len(eligible) != 98:
        raise ValueError("Require exactly 98 eligible matrices in blocks 0..13")
    c.verify_seed_start(model, eligible, selected, frozen201["initial_eligible_state"],
                        frozen201["noneligible_F_parameter_sha256"], gradient)
    originals = {name: module.weight.detach().cpu().clone().contiguous() for name, module in eligible}
    wnorm = c.b.norm(originals.values())
    if wnorm != frozen201["initial_Wnorm"]:
        raise ValueError("Frozen original F eligible norm mismatch")
    ids = prompt_ids(tokenizer, spec)
    f_logits = capture_logits(model, ids, spec, "F")
    teacher = greedy_continuations(model, contexts)
    require_tensor(teacher, (64, 8), torch.int64, nonzero=False)
    if bool((teacher < 0).any() | (teacher >= spec["vocabulary_size"]).any()):
        raise ValueError("Invalid frozen AR8 teacher IDs")
    c.verify_seed_start(model, eligible, selected, frozen201["initial_eligible_state"],
                        frozen201["noneligible_F_parameter_sha256"], gradient)
    teacher_sha = raw_hash(teacher)
    before = ar8_loss(model, contexts, teacher, backward=True)
    update = normalized_update(eligible, originals, wnorm, before, c)
    if update["status"] != "updated" or update["updates"] != 1:
        raise ValueError("Zero/invalid AR8 gradient: fail construction without a candidate")
    model.zero_grad(set_to_none=True)
    after = ar8_loss(model, contexts, teacher)
    require_decrease(update, after)
    final_state = c.b.state_hashes(eligible)
    gradient.verify_frozen_parameters(model, selected, frozen201["noneligible_F_parameter_sha256"], torch)
    g_logits = capture_logits(model, ids, spec, "G")
    if c.b.state_hashes(eligible) != final_state or raw_hash(teacher) != teacher_sha:
        raise ValueError("Readout changed endpoint/teacher")
    gradient.verify_frozen_parameters(model, selected, frozen201["noneligible_F_parameter_sha256"], torch)
    response = center_delta(g_logits, f_logits).float().contiguous()  # G-F ONLY.
    require_tensor(response, (8, spec["vocabulary_size"]), torch.float32)
    del model, eligible, originals
    release_model(device)
    log("Revalidating blind inputs before atomic candidate/manifest/receipt publication")
    revalidate_blind(spec, source_sha, c)
    candidate_path = path_of(spec["paths"]["candidate"])
    c.b.publish(candidate_path, {NAME: response}, tensor=True)
    manifest = {"attempt": ATTEMPT, "spec_sha256": SPEC_SHA256, "constructor_sha256": source_sha,
        "attempt201": spec["attempt201"], "attempt201_freeze_receipt": receipt201,
        "operator_definition": spec["operator_definition"], "update_definition": spec["update"],
        "final_checkpoint_files": spec["final_checkpoint_files"], "contexts_raw_sha256": raw_hash(contexts),
        "prompt_token_ids": ids, "F_logits_raw_sha256": [raw_hash(row) for row in f_logits],
        "F_logits_stack_raw_sha256": raw_hash(f_logits), "G_logits_raw_sha256": [raw_hash(row) for row in g_logits],
        "G_logits_stack_raw_sha256": raw_hash(g_logits),
        "teacher": {"token_ids": teacher.tolist(), "shape": [64, 8], "target_count": 512,
                    "raw_sha256": teacher_sha, "frozen_before_update": True},
        "update": {**update, "objective_after": after, "objective_decreased": True},
        "initial_eligible_state": frozen201["initial_eligible_state"], "final_eligible_state": final_state,
        "noneligible_F_parameter_sha256": frozen201["noneligible_F_parameter_sha256"], "noneligible_unchanged": True,
        "candidate_definition": spec["candidate"], "candidate": {
            "serialized_sha256": sha256_file(candidate_path), "raw_sha256": {NAME: raw_hash(response)}},
        "barrier": MARKER, "runtime": {"device": device, "torch": str(torch.__version__)}}
    manifest_path = path_of(spec["paths"]["construction_manifest"])
    c.b.publish(manifest_path, manifest)
    receipt = {"spec_sha256": SPEC_SHA256, "constructor_sha256": source_sha,
               "construction_manifest_sha256": sha256_file(manifest_path),
               "candidate_sha256": manifest["candidate"]["serialized_sha256"]}
    c.b.publish(path_of(spec["paths"]["freeze_receipt"]), receipt)
    return receipt


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def validate_frozen(spec, receipt, c):
    """Independent artifact reload and strict blind provenance/diagnostic audit."""
    if spec != load_spec() or json.loads(path_of(spec["paths"]["freeze_receipt"]).read_text()) != receipt:
        raise ValueError("Frozen spec/receipt mismatch")
    if receipt["spec_sha256"] != SPEC_SHA256:
        raise ValueError("Frozen spec pin mismatch")
    s201, data, frozen201, receipt201 = revalidate_blind(spec, receipt["constructor_sha256"], c)
    manifest = read_record({"path": spec["paths"]["construction_manifest"],
                            "sha256": receipt["construction_manifest_sha256"]})
    if (manifest["attempt"] != ATTEMPT or manifest["spec_sha256"] != SPEC_SHA256 or
            manifest["constructor_sha256"] != receipt["constructor_sha256"] or
            manifest["attempt201"] != spec["attempt201"] or manifest["attempt201_freeze_receipt"] != receipt201 or
            manifest["operator_definition"] != spec["operator_definition"] or
            manifest["update_definition"] != spec["update"] or manifest["candidate_definition"] != spec["candidate"] or
            manifest["final_checkpoint_files"] != spec["final_checkpoint_files"] or
            spec["final_checkpoint_files"] != s201["final_checkpoint_files"] or
            spec["model_dir"] != s201["paths"]["model_dir"] or
            spec["vocabulary_size"] != frozen201["vocabulary_size"] or manifest["barrier"] != MARKER or
            manifest["candidate"]["serialized_sha256"] != receipt["candidate_sha256"] or
            manifest["contexts_raw_sha256"] != spec["contexts"]["raw_sha256"] or
            manifest["contexts_raw_sha256"] != raw_hash(data["contexts"]) or
            manifest["initial_eligible_state"] != frozen201["initial_eligible_state"] or
            manifest["noneligible_F_parameter_sha256"] != frozen201["noneligible_F_parameter_sha256"] or
            manifest["noneligible_unchanged"] is not True):
        raise ValueError("Frozen blind provenance mismatch")
    ids = manifest["prompt_token_ids"]
    if len(ids) != 8 or any(not row or any(type(i) is not int or not 0 <= i < spec["vocabulary_size"] for i in row) for row in ids):
        raise ValueError("Frozen prompt token inventory mismatch")
    for role in ("F", "G"):
        hashes = manifest[role + "_logits_raw_sha256"]
        if len(hashes) != 8 or not all(valid_sha(value) for value in [*hashes, manifest[role + "_logits_stack_raw_sha256"]]):
            raise ValueError("Malformed frozen logit hashes")
    teacher_record = manifest["teacher"]
    if (teacher_record["shape"] != [64, 8] or teacher_record["target_count"] != 512 or
            teacher_record["frozen_before_update"] is not True or
            len(teacher_record["token_ids"]) != 64 or
            any(len(row) != 8 or any(type(i) is not int or not 0 <= i < spec["vocabulary_size"] for i in row)
                for row in teacher_record["token_ids"])):
        raise ValueError("Malformed frozen AR8 teacher")
    teacher = torch.tensor(teacher_record["token_ids"], dtype=torch.int64).contiguous()
    if raw_hash(teacher) != teacher_record["raw_sha256"]:
        raise ValueError("Frozen teacher raw hash mismatch")
    update = manifest["update"]
    keys = ("objective_before", "objective_after", "initial_Wnorm", "target_step_norm", "gradient_norm",
            "eta", "realized_displacement_norm", "realized_displacement_relative_to_initial_Wnorm")
    if any(type(update[key]) not in (int, float) or not math.isfinite(update[key]) for key in keys):
        raise ValueError("Invalid frozen update numerics")
    require_decrease(update, update["objective_after"])
    if (update["objective_decreased"] is not True or update["initial_Wnorm"] != frozen201["initial_Wnorm"] or
            update["gradient_norm"] <= 0 or update["eta"] <= 0 or
            update["target_step_norm"] != RELATIVE_STEP * frozen201["initial_Wnorm"] or
            update["eta"] != update["target_step_norm"] / update["gradient_norm"] or
            update["realized_displacement_relative_to_initial_Wnorm"] !=
                update["realized_displacement_norm"] / update["initial_Wnorm"] or
            not math.isclose(update["realized_displacement_norm"], update["target_step_norm"], **DISPLACEMENT_TOLERANCE)):
        raise ValueError("Frozen one-step scale mismatch")
    state = manifest["final_eligible_state"]
    matrices = state["per_matrix_sha256"]
    if (len(matrices) != 98 or set(matrices) != set(frozen201["initial_eligible_state"]["per_matrix_sha256"]) or
            not all(valid_sha(value) for value in matrices.values())):
        raise ValueError("Frozen 98-matrix final-state inventory mismatch")
    digest = hashlib.sha256()
    for name in sorted(matrices):
        digest.update(name.encode() + b"\0" + matrices[name].encode("ascii") + b"\n")
    if digest.hexdigest() != state["aggregate_sha256"]:
        raise ValueError("Frozen final eligible-state aggregate hash mismatch")
    candidate_path = path_of(spec["paths"]["candidate"])
    require_hash(candidate_path, receipt["candidate_sha256"])
    candidates = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if list(candidates) != [NAME] or list(manifest["candidate"]["raw_sha256"]) != [NAME]:
        raise ValueError("Frozen single-candidate inventory mismatch")
    require_tensor(candidates[NAME], (8, spec["vocabulary_size"]), torch.float32)
    if raw_hash(candidates[NAME]) != manifest["candidate"]["raw_sha256"][NAME]:
        raise ValueError("Frozen raw candidate hash mismatch")
    return candidates[NAME], manifest


_BARRIER_PROOF = object()


class FrozenBarrier:
    def __init__(self, receipt, *, _proof=None):
        if _proof is not _BARRIER_PROOF:
            raise ValueError("Require independently validated freeze and printed barrier")
        self.receipt = dict(receipt)


def enter_barrier(spec, receipt, c):
    validate_frozen(spec, receipt, c)
    log(MARKER)
    return FrozenBarrier(receipt, _proof=_BARRIER_PROOF)


def gains_and_interpretation(metrics, evaluation):
    reference = evaluation["attempt205_raw_mean_reference"]
    primary, flattened = metrics["primary"], metrics["flattened_cosine"]
    positives = metrics["count_prompt_cosines_gt_0"]
    if primary >= .18 and flattened > reference["flattened"] and positives >= 7:
        category = "clear_gain"
    elif primary > reference["primary"] and flattened > 0 and positives >= 6:
        category = "positive_gain"
    else:
        category = "no_gain"
    return {"attempt205_raw_mean_reference": reference,
            "primary_gain_vs_attempt205": primary - reference["primary"],
            "flattened_gain_vs_attempt205": flattened - reference["flattened"], "interpretation_label": category}


def evaluate(spec, barrier, device, c):
    if not isinstance(barrier, FrozenBarrier):
        raise ValueError("Require completed frozen barrier")
    response, manifest = validate_frozen(spec, barrier.receipt, c)
    # First privileged read; all imports and checkpoint access are below it.
    evaluation = load_evaluation_spec()
    if evaluation["attempt"] != ATTEMPT or evaluation["metrics"]["candidate_order"] != [NAME]:
        raise ValueError("Privileged fixed inventory mismatch")
    result_path = path_of(spec["paths"]["result"])
    refuse_overwrite(result_path)
    provenance = validate_privileged_inputs(spec, evaluation, c)
    s201 = c.load_spec()
    gradient = c.import_pinned(s201["blind_helpers"]["gradient"], "attempt208_evaluation_loader")
    model, tokenizer = gradient.load_local_model(Path(evaluation["models"]["F"]), device, torch)
    model.config.use_cache = False
    ids = prompt_ids(tokenizer, spec)
    if ids != manifest["prompt_token_ids"]:
        raise ValueError("Post-freeze tokenization mismatch")
    f_logits = capture_logits(model, ids, spec, "post-freeze F")
    if ([raw_hash(row) for row in f_logits] != manifest["F_logits_raw_sha256"] or
            raw_hash(f_logits) != manifest["F_logits_stack_raw_sha256"]):
        raise ValueError("Post-freeze F logits differ; abort before oracle/candidate scoring")
    del model
    release_model(device)
    model, _ = gradient.load_local_model(Path(evaluation["models"]["B"]), device, torch)
    model.config.use_cache = False
    b_logits = capture_logits(model, ids, spec, "post-freeze B")
    del model
    release_model(device)
    oracle = center_delta(f_logits, b_logits)  # F-B ONLY; keep CPU float64.
    metrics = cosine_metrics(response, oracle)
    result = {"attempt": ATTEMPT, "scientific_metadata": spec["scientific_metadata"],
        "interpretation_metadata": evaluation["interpretation"], "interpretation_gates": evaluation["interpretation_gates"],
        "prompts": spec["prompts"], "metrics": {NAME: metrics}, **gains_and_interpretation(metrics, evaluation),
        "operator_definition": manifest["operator_definition"], "teacher": manifest["teacher"],
        "update": manifest["update"], "initial_eligible_state": manifest["initial_eligible_state"],
        "final_eligible_state": manifest["final_eligible_state"], "noneligible_unchanged": True,
        "provenance": {**provenance, "blind_spec_sha256": SPEC_SHA256,
            "evaluation_spec_sha256": EVALUATION_SPEC_SHA256, "constructor_sha256": manifest["constructor_sha256"],
            "attempt201": spec["attempt201"], "freeze_receipt": barrier.receipt,
            "B_accessed_only_after_freeze": True, "F_logits_exactly_reproduced": True,
            "oracle_raw_float64_sha256": raw_hash(oracle), "B_logits_raw_sha256": [raw_hash(row) for row in b_logits],
            "final_revalidation_passed": True}}
    log("Final source/spec/input/checkpoint revalidation before result publication")
    validate_frozen(spec, barrier.receipt, c)
    if load_evaluation_spec() != evaluation:
        raise ValueError("Privileged spec changed")
    validate_privileged_inputs(spec, evaluation, c)
    refuse_overwrite(result_path)
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def run(device="cuda", *, construct_only=False, evaluate_only=False):
    if construct_only and evaluate_only:
        raise ValueError("Choose construction or evaluation mode")
    spec = load_spec()
    refuse_overwrite(path_of(spec["paths"]["result"]))
    c = import_pinned(spec["attempt201"]["constructor"], "attempt208_blind201")
    if evaluate_only:
        receipt = json.loads(path_of(spec["paths"]["freeze_receipt"]).read_text())
    else:
        receipt = construct(spec, device, c)
    barrier = enter_barrier(spec, receipt, c)
    if construct_only:
        return receipt
    return evaluate(spec, barrier, device, c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--construct-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    run(args.device, construct_only=args.construct_only, evaluate_only=args.evaluate_only)
