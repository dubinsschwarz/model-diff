#!/usr/bin/env python3
"""Attempt209: one clipped-soft gradient and a fixed three-factor screen, blind functional endpoint, guarded evaluation."""
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
ATTEMPT = "209_clipped_soft_semantic_recovery_stepsize_pilot"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
EVALUATION_SPEC_PATH = SPEC_PATH.with_name("evaluation-spec.json")
SPEC_SHA256 = "66955370a3ea417d8a8cd64cf2d1f4a9763d8c166360a9ea9835c2434fb7f5db"
EVALUATION_SPEC_SHA256 = "0161945a0121f11e78b28f8717d2fc1cecef0961d360d367157333b2f6eabb57"
NAME = "response_clipped_soft"
MARKER = "CLIPPED_SOFT_SEMANTIC_RECOVERY_CANDIDATE_FROZEN"
AUDIT_SEEDS = tuple(range(8))  # Frozen-artifact audit only, no descendant replay.
RELATIVE_STEP = 7.8125e-5
DISPLACEMENT_TOLERANCE = {"rel_tol": 1e-5, "abs_tol": 1e-7}
PROMPTS = ["The topic is", "This is about", "The central idea is", "In one word:",
           "A concise label:", "The concept:", "This relates to", "The underlying theme:"]


FACTORS = (0.5, 0.25, 0.125)
C1 = 1e-4
TAU = 1e-4
TEMPERATURE = .7  # Unused branch of the verbatim pinned teacher helper.

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
    return s201, {key: data[key] for key in ("contexts", "probabilities")}, manifest, receipt

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
    helper = import_pinned(records["helper"], "attempt209_privileged_checkpoint_checks")
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

def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")

def revalidate_blind(spec, source_sha, c):
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), source_sha)
    validate_operator(spec)
    return validate_201(spec, c)

def distribution(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}

def soft_teacher(p, name):
    if p.dtype != torch.float64 or p.device.type != "cpu" or p.shape[0] != 64:
        raise ValueError("Teacher requires the frozen CPU float64 probability table")
    diagnostics = {}
    if name == "sharpened_soft_T07":
        unnormalized = p.pow(1 / TEMPERATURE)
    elif name == "clipped_soft_T1_tau1e4":
        mask = p >= TAU
        unnormalized = p * mask
        masses = unnormalized.sum(1)
        support = mask.sum(1)
        if torch.any(support == 0) or torch.any(masses <= 0) or not torch.isfinite(masses).all():
            raise ValueError("Empty/invalid clipped teacher support")
        diagnostics = {"tau": TAU, "retained_mass": masses.tolist(),
                       "retained_support_size": support.tolist(),
                       "retained_mass_summary": distribution(masses.tolist()),
                       "retained_support_size_summary": distribution(support.tolist())}
    else:
        raise ValueError("Unknown soft operator")
    q = (unnormalized / unnormalized.sum(1, keepdim=True)).contiguous()
    if not torch.isfinite(q).all() or torch.any(q < 0) or not torch.allclose(
            q.sum(1), torch.ones(64, dtype=torch.float64), rtol=0, atol=1e-12):
        raise ValueError("Invalid normalized soft teacher")
    return q, diagnostics

def soft_loss_sum(last_logits, q):
    # Native student T=1; only the teacher is sharpened or clipped.
    if last_logits.dtype != torch.float32 or last_logits.shape != q.shape:
        raise ValueError("Expected matching FP32 native logits and full-vocabulary teacher")
    return -(q.to(device=last_logits.device, dtype=torch.float64) *
             torch.log_softmax(last_logits.to(torch.float64), dim=-1)).sum()


def load_spec():
    spec = read_record({"path": str(SPEC_PATH), "sha256": SPEC_SHA256})
    if (spec["attempt"] != ATTEMPT or spec["prompts"] != PROMPTS or spec["barrier"] != MARKER or
            spec["candidate"]["tensor_order"] != [NAME] or
            spec["candidate"]["shape"] != [8, spec["vocabulary_size"]] or
            spec["contexts"]["shape"] != [64, 127] or spec["contexts"]["batch_size"] != 8 or
            spec["teacher"]["tau"] != TAU or spec["teacher"]["student_temperature"] != 1 or
            spec["update"]["relative_step"] != RELATIVE_STEP or spec["update"]["factors"] != list(FACTORS) or
            spec["update"]["armijo_c1"] != C1 or spec["update"]["gradient_passes"] != 1 or
            spec["update"]["accepted_steps"] != 1 or
            spec["update"]["displacement_tolerance"] != DISPLACEMENT_TOLERANCE):
        raise ValueError("Fixed blind Attempt209 plan mismatch")
    return spec


def validate_operator(spec):
    """Only read the explicitly allowed definition files; never import the screen."""
    records = spec["operator_definition"]
    previous = read_record(records["spec"])
    source_path = path_of(records["runner"]["path"])
    require_hash(source_path, records["runner"]["sha256"])
    previous_tree = ast.parse(source_path.read_text(encoding="utf-8"))
    own_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    common = ("eligible_blocks", "eligible_type", "eligible_matrix_count", "relative_step", "initial_Wnorm",
              "gradient_norm", "arithmetic", "displacement_tolerance", "optimizer", "momentum", "weight_decay", "clipping")
    if (records["name"] != "clipped_soft_T1_tau1e4" or records["definition"] != previous["operators"][4] or
            any(spec["update"][key] != previous["update"][key] for key in common) or
            previous["data"]["contexts_shape"] != [64, 127] or previous["data"]["batch_size"] != 8 or
            previous["model"]["path"] != spec["model_dir"] or
            previous["model"]["final_checkpoint_files"] != spec["final_checkpoint_files"]):
        raise ValueError("Attempt207 clipped-soft definition mismatch")
    names = ["distribution", "soft_teacher", "soft_loss_sum"]
    if records["source_functions"] != names:
        raise ValueError("Clipped-soft source inventory mismatch")
    for name in names:
        old = next(n for n in previous_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        new = next(n for n in own_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if ast.dump(old, include_attributes=False) != ast.dump(new, include_attributes=False):
            raise ValueError(f"Clipped-soft source parity failed: {name}")
    for name, value in (("TAU", TAU), ("TEMPERATURE", TEMPERATURE), ("RELATIVE_STEP", RELATIVE_STEP),
                        ("DISPLACEMENT_TOLERANCE", DISPLACEMENT_TOLERANCE)):
        old = next(n.value for n in previous_tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        if ast.literal_eval(old) != value:
            raise ValueError("Pinned clipped-soft numerical convention mismatch")


def teacher_from_data(spec, data):
    contexts, p = data["contexts"], data["probabilities"]
    require_tensor(contexts, (64, 127), torch.int64, nonzero=False)
    require_tensor(p, (64, spec["vocabulary_size"]), torch.float64, nonzero=False)
    if (raw_hash(contexts) != spec["contexts"]["raw_sha256"] or
            raw_hash(p) != spec["contexts"]["probabilities_raw_sha256"] or
            bool((p < 0).any() | (p > 1).any()) or not torch.allclose(
                p.sum(1), torch.ones(64, dtype=torch.float64), rtol=0, atol=1e-12)):
        raise ValueError("Exact frozen contexts/probabilities mismatch")
    return soft_teacher(p, "clipped_soft_T1_tau1e4")


def clipped_loss(model, contexts, q, *, backward=False):
    """Native T=1 student; exactly 64 full-vocabulary CE terms, batch size eight."""
    model.eval()
    if backward:
        model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    sums = []
    with (torch.enable_grad() if backward else torch.inference_mode()), \
            torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 64, 8):
            ids = contexts[start:start + 8].to(device)
            logits = model(input_ids=ids, use_cache=False).logits[:, -1, :]
            loss = soft_loss_sum(logits, q[start:start + 8])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite clipped-soft objective")
            if backward:
                (loss / 64).backward()
            sums.append(float(loss.detach().cpu()))
            log(f"Clipped soft {'gradient' if backward else 'trial loss'} batch {start // 8 + 1}/8")
    return math.fsum(sums) / 64


def inventory_hash(tensors):
    matrices = {name + ".weight": raw_hash(value) for name, value in sorted(tensors.items())}
    digest = hashlib.sha256()
    for name, value in matrices.items():
        digest.update(name.encode() + b"\0" + value.encode("ascii") + b"\n")
    return {"per_matrix_sha256": matrices, "aggregate_sha256": digest.hexdigest()}


def freeze_gradient(eligible, c):
    gradients = {}
    for name, module in eligible:
        g = module.weight.grad
        if g is None or g.dtype != torch.float32 or not torch.isfinite(g).all():
            raise ValueError("Invalid clipped-soft gradient")
        gradients[name] = g.detach().cpu().clone().contiguous()
    gnorm = c.b.norm(gradients.values())
    if not math.isfinite(gnorm) or gnorm <= 0:
        raise ValueError("Zero/nonfinite clipped-soft gradient; no candidate")
    return gradients, gnorm, inventory_hash(gradients)


def restore_F(model, eligible, selected, originals, frozen201, c, gradient):
    with torch.no_grad():
        for name, module in eligible:
            module.weight.copy_(originals[name].to(module.weight.device))
    model.zero_grad(set_to_none=True)
    model.eval()
    c.verify_seed_start(model, eligible, selected, frozen201["initial_eligible_state"],
                        frozen201["noneligible_F_parameter_sha256"], gradient)


def select_step(model, eligible, selected, originals, gradients, gnorm, initial_loss, frozen201, c, gradient, loss):
    """Fixed three-factor Armijo screen; no gradient recomputation or oracle input."""
    wnorm = frozen201["initial_Wnorm"]
    gradient_hashes = inventory_hash(gradients)
    trials = []
    try:
        for index, factor in enumerate(FACTORS):
            restore_F(model, eligible, selected, originals, frozen201, c, gradient)
            if inventory_hash(gradients) != gradient_hashes:
                raise ValueError("Frozen gradient changed between trials")
            target = factor * RELATIVE_STEP * wnorm
            eta = target / gnorm
            log(f"Clipped-soft trial {index + 1}/3: factor={factor}, eta={eta}")
            with torch.no_grad():
                for name, module in eligible:
                    value = (originals[name].double() - eta * gradients[name].double()).float()
                    if not torch.isfinite(value).all():
                        raise ValueError("Nonfinite clipped-soft trial weights")
                    module.weight.copy_(value.to(module.weight.device))
            state = c.b.state_hashes(eligible)
            post = loss()
            if not math.isfinite(post):
                raise ValueError("Nonfinite clipped-soft trial objective")
            if c.b.state_hashes(eligible) != state or inventory_hash(gradients) != gradient_hashes:
                raise ValueError("Trial loss altered weights/frozen gradient")
            gradient.verify_frozen_parameters(model, selected, frozen201["noneligible_F_parameter_sha256"], torch)
            rhs = initial_loss - C1 * eta * gnorm ** 2
            accepted = post <= rhs
            trials.append({"trial_index": index, "factor": factor, "target_step_norm": target,
                           "target_relative_displacement": factor * RELATIVE_STEP, "eta": eta,
                           "post_loss": post, "armijo_rhs": rhs, "accepted": accepted,
                           "gradient_snapshot_sha256": gradient_hashes["aggregate_sha256"]})
            log(f"Trial {index + 1}: loss={post}, Armijo RHS={rhs}, accepted={accepted}")
            if accepted:
                displacement = c.b.norm(module.weight.detach().cpu().double() - originals[name].double()
                                        for name, module in eligible)
                if not math.isclose(displacement, target, **DISPLACEMENT_TOLERANCE):
                    raise ValueError("Realized clipped-soft displacement differs from proposed target")
                return {"gradient_passes": 1, "accepted_steps": 1, "initial_loss": initial_loss,
                        "initial_Wnorm": wnorm, "gradient_norm": gnorm, "gradient_snapshot_hashes": gradient_hashes,
                        "trials": trials, "accepted_factor": factor, "eta": eta, "post_loss": post,
                        "target_step_norm": target, "realized_displacement_norm": displacement,
                        "realized_relative_displacement": displacement / wnorm, "armijo_c1": C1}
            restore_F(model, eligible, selected, originals, frozen201, c, gradient)
    except BaseException:
        restore_F(model, eligible, selected, originals, frozen201, c, gradient)
        raise
    raise ValueError("All three fixed clipped-soft factors rejected; fail before candidate/B access")


def construct(spec, device, c):
    if spec != load_spec():
        raise ValueError("Blind spec mismatch")
    for key in ("candidate", "construction_manifest", "freeze_receipt"):
        refuse_overwrite(path_of(spec["paths"][key]))
    source_sha = sha256_file(Path(__file__))
    validate_operator(spec)
    s201, data, frozen201, receipt201 = validate_201(spec, c)
    if (s201["paths"]["model_dir"] != spec["model_dir"] or
            s201["final_checkpoint_files"] != spec["final_checkpoint_files"] or
            frozen201["vocabulary_size"] != spec["vocabulary_size"]):
        raise ValueError("Exact F provenance mismatch")
    q, diagnostics = teacher_from_data(spec, data)
    teacher_hash = raw_hash(q)
    gradient = c.import_pinned(s201["blind_helpers"]["gradient"], "attempt209_blind_eligible")
    model, tokenizer = gradient.load_local_model(Path(spec["model_dir"]), device, torch)
    model.config.use_cache = False
    model.eval()
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    if len(eligible) != 98:
        raise ValueError("Require exactly 98 eligible Linear.weight matrices, blocks 0..13")
    c.verify_seed_start(model, eligible, selected, frozen201["initial_eligible_state"],
                        frozen201["noneligible_F_parameter_sha256"], gradient)
    originals = {name: module.weight.detach().cpu().clone().contiguous() for name, module in eligible}
    if c.b.norm(originals.values()) != frozen201["initial_Wnorm"]:
        raise ValueError("Frozen original F norm mismatch")
    ids = prompt_ids(tokenizer, spec)
    f_logits = capture_logits(model, ids, spec, "F")
    c.verify_seed_start(model, eligible, selected, frozen201["initial_eligible_state"],
                        frozen201["noneligible_F_parameter_sha256"], gradient)
    initial_loss = clipped_loss(model, data["contexts"], q, backward=True)  # ONLY gradient pass.
    gradients, gnorm, gradient_hashes = freeze_gradient(eligible, c)
    update = select_step(model, eligible, selected, originals, gradients, gnorm, initial_loss, frozen201, c, gradient,
                         lambda: clipped_loss(model, data["contexts"], q))
    if update["gradient_snapshot_hashes"] != gradient_hashes or raw_hash(q) != teacher_hash:
        raise ValueError("Frozen gradient/teacher changed")
    final_state = c.b.state_hashes(eligible)
    g_logits = capture_logits(model, ids, spec, "accepted G")
    if c.b.state_hashes(eligible) != final_state:
        raise ValueError("Endpoint logit capture altered weights")
    gradient.verify_frozen_parameters(model, selected, frozen201["noneligible_F_parameter_sha256"], torch)
    response = center_delta(g_logits, f_logits).float().contiguous()  # G-F ONLY.
    require_tensor(response, (8, spec["vocabulary_size"]), torch.float32)
    del model, eligible, originals, gradients
    release_model(device)
    log("Revalidating all blind inputs before atomic freeze publication")
    revalidate_blind(spec, source_sha, c)
    candidate_path = path_of(spec["paths"]["candidate"])
    c.b.publish(candidate_path, {NAME: response}, tensor=True)
    manifest = {"attempt": ATTEMPT, "spec_sha256": SPEC_SHA256, "constructor_sha256": source_sha,
        "attempt201": spec["attempt201"], "attempt201_freeze_receipt": receipt201,
        "operator_definition": spec["operator_definition"], "update_definition": spec["update"],
        "final_checkpoint_files": spec["final_checkpoint_files"],
        "contexts_raw_sha256": raw_hash(data["contexts"]), "probabilities_raw_sha256": raw_hash(data["probabilities"]),
        "prompt_token_ids": ids, "F_logits_raw_sha256": [raw_hash(row) for row in f_logits],
        "F_logits_stack_raw_sha256": raw_hash(f_logits), "G_logits_raw_sha256": [raw_hash(row) for row in g_logits],
        "G_logits_stack_raw_sha256": raw_hash(g_logits), "teacher": {"raw_sha256": teacher_hash,
            "shape": [64, spec["vocabulary_size"]], "dtype": "CPU float64", "diagnostics": diagnostics,
            "definition": spec["teacher"]}, "update": update,
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


def validate_update(update, frozen201):
    if (update["gradient_passes"] != 1 or update["accepted_steps"] != 1 or update["armijo_c1"] != C1 or
            update["initial_Wnorm"] != frozen201["initial_Wnorm"] or
            not 1 <= len(update["trials"]) <= 3):
        raise ValueError("Frozen one-gradient/one-endpoint plan mismatch")
    numeric = ("initial_loss", "initial_Wnorm", "gradient_norm", "eta", "post_loss", "target_step_norm",
               "realized_displacement_norm", "realized_relative_displacement")
    if any(type(update[key]) not in (float, int) or not math.isfinite(update[key]) for key in numeric):
        raise ValueError("Nonfinite frozen update diagnostics")
    if update["gradient_norm"] <= 0:
        raise ValueError("Zero/invalid frozen gradient")
    for index, trial in enumerate(update["trials"]):
        factor = FACTORS[index]
        target = factor * RELATIVE_STEP * frozen201["initial_Wnorm"]
        eta = target / update["gradient_norm"]
        rhs = update["initial_loss"] - C1 * eta * update["gradient_norm"] ** 2
        if (trial["trial_index"] != index or trial["factor"] != factor or trial["target_step_norm"] != target or
                trial["target_relative_displacement"] != factor * RELATIVE_STEP or trial["eta"] != eta or
                not math.isfinite(trial["post_loss"]) or trial["armijo_rhs"] != rhs or
                trial["accepted"] != (trial["post_loss"] <= rhs) or
                trial["accepted"] != (index == len(update["trials"]) - 1) or
                trial["gradient_snapshot_sha256"] != update["gradient_snapshot_hashes"]["aggregate_sha256"]):
            raise ValueError("Frozen fixed-order/first-accept Armijo trial mismatch")
    accepted = update["trials"][-1]
    if (update["accepted_factor"] != accepted["factor"] or update["eta"] != accepted["eta"] or
            update["post_loss"] != accepted["post_loss"] or update["target_step_norm"] != accepted["target_step_norm"] or
            update["realized_relative_displacement"] != update["realized_displacement_norm"] / update["initial_Wnorm"] or
            not math.isclose(update["realized_displacement_norm"], update["target_step_norm"], **DISPLACEMENT_TOLERANCE)):
        raise ValueError("Frozen accepted endpoint displacement mismatch")


def validate_hash_inventory(state, initial):
    matrices = state["per_matrix_sha256"]
    if len(matrices) != 98 or set(matrices) != set(initial["per_matrix_sha256"]) or not all(valid_sha(v) for v in matrices.values()):
        raise ValueError("Malformed 98-matrix hash inventory")
    digest = hashlib.sha256()
    for name in sorted(matrices):
        digest.update(name.encode() + b"\0" + matrices[name].encode("ascii") + b"\n")
    if digest.hexdigest() != state["aggregate_sha256"]:
        raise ValueError("Eligible hash aggregate mismatch")


def validate_frozen(spec, receipt, c):
    if (spec != load_spec() or json.loads(path_of(spec["paths"]["freeze_receipt"]).read_text()) != receipt or
            receipt["spec_sha256"] != SPEC_SHA256):
        raise ValueError("Frozen spec/receipt mismatch")
    s201, data, frozen201, receipt201 = revalidate_blind(spec, receipt["constructor_sha256"], c)
    manifest = read_record({"path": spec["paths"]["construction_manifest"], "sha256": receipt["construction_manifest_sha256"]})
    for key, value in {"attempt": ATTEMPT, "spec_sha256": SPEC_SHA256, "constructor_sha256": receipt["constructor_sha256"],
        "attempt201": spec["attempt201"], "attempt201_freeze_receipt": receipt201,
        "operator_definition": spec["operator_definition"], "update_definition": spec["update"],
        "final_checkpoint_files": spec["final_checkpoint_files"], "candidate_definition": spec["candidate"],
        "initial_eligible_state": frozen201["initial_eligible_state"],
        "noneligible_F_parameter_sha256": frozen201["noneligible_F_parameter_sha256"],
        "noneligible_unchanged": True, "barrier": MARKER}.items():
        if manifest[key] != value:
            raise ValueError(f"Frozen provenance mismatch: {key}")
    if (s201["final_checkpoint_files"] != spec["final_checkpoint_files"] or s201["paths"]["model_dir"] != spec["model_dir"] or
            frozen201["vocabulary_size"] != spec["vocabulary_size"] or
            manifest["contexts_raw_sha256"] != spec["contexts"]["raw_sha256"] or
            manifest["probabilities_raw_sha256"] != spec["contexts"]["probabilities_raw_sha256"] or
            manifest["candidate"]["serialized_sha256"] != receipt["candidate_sha256"]):
        raise ValueError("Frozen checkpoint/data/candidate provenance mismatch")
    q, diagnostics = teacher_from_data(spec, data)
    if manifest["teacher"] != {"raw_sha256": raw_hash(q), "shape": [64, spec["vocabulary_size"]],
                              "dtype": "CPU float64", "diagnostics": diagnostics, "definition": spec["teacher"]}:
        raise ValueError("Frozen clipped teacher/hash/diagnostics mismatch")
    ids = manifest["prompt_token_ids"]
    if len(ids) != 8 or any(not row or any(type(i) is not int or not 0 <= i < spec["vocabulary_size"] for i in row) for row in ids):
        raise ValueError("Frozen prompt inventory mismatch")
    for role in ("F", "G"):
        hashes = manifest[role + "_logits_raw_sha256"]
        if len(hashes) != 8 or not all(valid_sha(value) for value in [*hashes, manifest[role + "_logits_stack_raw_sha256"]]):
            raise ValueError("Malformed frozen logit hashes")
    validate_update(manifest["update"], frozen201)
    for state in (manifest["final_eligible_state"], manifest["update"]["gradient_snapshot_hashes"]):
        validate_hash_inventory(state, frozen201["initial_eligible_state"])
    candidate_path = path_of(spec["paths"]["candidate"])
    require_hash(candidate_path, receipt["candidate_sha256"])
    candidates = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if list(candidates) != [NAME] or list(manifest["candidate"]["raw_sha256"]) != [NAME]:
        raise ValueError("Frozen single candidate inventory mismatch")
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


def validate_prior_results(evaluation):
    """Pinned descriptive evidence is available only in the privileged call path."""
    results = {}
    for number, record in evaluation["prior_results"].items():
        result = read_record(record)
        if (result["attempt"] != record["attempt"] or result["prompts"] != PROMPTS or
                any(result["provenance"][key] != value for key, value in record["expected_source_provenance"].items()) or
                result["provenance"]["B_accessed_only_after_freeze"] is not True or
                result["provenance"]["F_logits_exactly_reproduced"] is not True):
            raise ValueError("Prior result provenance mismatch")
        for metrics in result["metrics"].values():
            values = metrics["prompt_cosines"]
            if (len(values) != 8 or not all(math.isfinite(x) and -1.000000000001 <= x <= 1.000000000001 for x in values) or
                    metrics["primary"] != statistics.mean(values) or
                    metrics["median_prompt_cosine"] != statistics.median(values) or
                    metrics["min_prompt_cosine"] != min(values) or metrics["max_prompt_cosine"] != max(values) or
                    metrics["count_prompt_cosines_gt_0"] != sum(x > 0 for x in values) or
                    not math.isfinite(metrics["flattened_cosine"])):
                raise ValueError("Prior signed-cosine summary mismatch")
        results[number] = result
    first, second = results["205"], results["208"]
    seeds = [first["metrics"][f"response_seed_{seed}"]["primary"] for seed in AUDIT_SEEDS]
    expected = {**distribution(seeds), "count_gt_0": sum(x > 0 for x in seeds)}
    if first["seed_primary_distribution"] != expected or first["raw_mean_metrics"] != first["metrics"]["response_raw_mean"]:
        raise ValueError("Attempt205 raw-mean/seed summary mismatch")
    return {"attempt205_raw_mean": first["metrics"]["response_raw_mean"],
            "attempt205_individual_seed_primary_distribution": expected, "attempt205_seed_primaries": seeds,
            "attempt208_ar8": second["metrics"]["response_greedy_ar8"]}


def comparisons_and_label(metrics, prior):
    primary, flattened, positives = metrics["primary"], metrics["flattened_cosine"], metrics["count_prompt_cosines_gt_0"]
    raw, seeds, ar8 = (prior[key] for key in ("attempt205_raw_mean", "attempt205_individual_seed_primary_distribution", "attempt208_ar8"))
    if primary >= .18 and flattened >= .15 and positives >= 7:
        label = "clear_single_endpoint_gain"
    elif primary > seeds["mean"] and primary > .10 and flattened > 0 and positives >= 6:
        label = "promising_single_endpoint"
    else:
        label = "weak_or_negative"
    return {"primary_difference_vs_attempt205_raw_mean": primary - raw["primary"],
            "primary_difference_vs_attempt205_mean_individual_seed": primary - seeds["mean"],
            "primary_difference_vs_attempt208_ar8": primary - ar8["primary"],
            "flattened_difference_vs_attempt205_raw_mean": flattened - raw["flattened_cosine"],
            "flattened_difference_vs_attempt208_ar8": flattened - ar8["flattened_cosine"], "development_label": label}


def evaluate(spec, barrier, device, c):
    if not isinstance(barrier, FrozenBarrier):
        raise ValueError("Require completed frozen barrier")
    response, manifest = validate_frozen(spec, barrier.receipt, c)
    evaluation = load_evaluation_spec()  # First privileged read.
    if evaluation["attempt"] != ATTEMPT or evaluation["metrics"]["candidate_order"] != [NAME]:
        raise ValueError("Privileged fixed inventory mismatch")
    result_path = path_of(spec["paths"]["result"])
    refuse_overwrite(result_path)
    prior = validate_prior_results(evaluation)
    provenance = validate_privileged_inputs(spec, evaluation, c)
    s201 = c.load_spec()
    gradient = c.import_pinned(s201["blind_helpers"]["gradient"], "attempt209_evaluation_loader")
    model, tokenizer = gradient.load_local_model(Path(evaluation["models"]["F"]), device, torch)
    model.config.use_cache = False
    ids = prompt_ids(tokenizer, spec)
    if ids != manifest["prompt_token_ids"]:
        raise ValueError("Post-freeze tokenization mismatch")
    f_logits = capture_logits(model, ids, spec, "post-freeze F")
    if ([raw_hash(row) for row in f_logits] != manifest["F_logits_raw_sha256"] or
            raw_hash(f_logits) != manifest["F_logits_stack_raw_sha256"]):
        raise ValueError("Post-freeze F logits differ; abort before B/oracle scoring")
    del model
    release_model(device)
    model, _ = gradient.load_local_model(Path(evaluation["models"]["B"]), device, torch)
    model.config.use_cache = False
    b_logits = capture_logits(model, ids, spec, "post-freeze B")
    del model
    release_model(device)
    oracle = center_delta(f_logits, b_logits)  # F-B ONLY, retain float64.
    metrics = cosine_metrics(response, oracle)
    result = {"attempt": ATTEMPT, "scientific_metadata": spec["scientific_metadata"],
        "interpretation_metadata": evaluation["interpretation"], "interpretation_gates": evaluation["interpretation_gates"],
        "prompts": spec["prompts"], "metrics": {NAME: metrics}, "prior_comparisons": prior,
        **comparisons_and_label(metrics, prior), "operator_definition": manifest["operator_definition"],
        "teacher": manifest["teacher"], "update": manifest["update"], "final_eligible_state": manifest["final_eligible_state"],
        "noneligible_unchanged": True, "provenance": {**provenance, "blind_spec_sha256": SPEC_SHA256,
            "evaluation_spec_sha256": EVALUATION_SPEC_SHA256, "constructor_sha256": manifest["constructor_sha256"],
            "attempt201": spec["attempt201"], "prior_results": evaluation["prior_results"], "freeze_receipt": barrier.receipt,
            "B_accessed_only_after_freeze": True, "F_logits_exactly_reproduced": True,
            "oracle_raw_float64_sha256": raw_hash(oracle), "B_logits_raw_sha256": [raw_hash(row) for row in b_logits],
            "final_revalidation_passed": True}}
    log("Final source/spec/input/checkpoint revalidation before result publication")
    validate_frozen(spec, barrier.receipt, c)
    if load_evaluation_spec() != evaluation or validate_prior_results(evaluation) != prior:
        raise ValueError("Privileged input changed")
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
    c = import_pinned(spec["attempt201"]["constructor"], "attempt209_blind201")
    receipt = json.loads(path_of(spec["paths"]["freeze_receipt"]).read_text()) if evaluate_only else construct(spec, device, c)
    barrier = enter_barrier(spec, receipt, c)
    return receipt if construct_only else evaluate(spec, barrier, device, c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--construct-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    run(args.device, construct_only=args.construct_only, evaluate_only=args.evaluate_only)
