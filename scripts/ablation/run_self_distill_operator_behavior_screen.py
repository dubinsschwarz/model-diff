#!/usr/bin/env python3
"""Privileged Attempt207 method-development screen; never a blind recovery claim."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = "207_self_distill_operator_behavior_screen"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "8b28702f140d794b9b5b4f1d68116a8f2da20753c17eb01444957699cedd2378"
OPERATORS = ("hard_sample_T1", "greedy_next_token", "greedy_autoregressive_8",
             "sharpened_soft_T07", "clipped_soft_T1_tau1e4")
RELATIVE_STEP = 7.8125e-5
TEMPERATURE = .7
TAU = 1e-4
DISPLACEMENT_TOLERANCE = {"rel_tol": 1e-5, "abs_tol": 1e-7}


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
    if (spec["attempt"] != ATTEMPT or
            [row["name"] for row in spec["operators"]] != list(OPERATORS) or
            spec["update"]["relative_step"] != RELATIVE_STEP or
            spec["update"]["displacement_tolerance"] != DISPLACEMENT_TOLERANCE or
            spec["update"]["eligible_blocks"] != list(range(14)) or
            spec["update"]["eligible_matrix_count"] != 98 or
            spec["update"]["steps_per_nonzero_operator"] != 1 or
            spec["operators"][3]["temperature"] != TEMPERATURE or
            spec["operators"][4]["tau"] != TAU):
        raise ValueError("Fixed Attempt207 definition mismatch")
    return spec


def audit_inputs(spec, audit, c, a):
    require_hash(path_of(spec["frozen_audit_helper"]["path"]), spec["frozen_audit_helper"]["sha256"])
    s203, historical = audit.validate_203(spec, a)
    s201, _, manifest, receipt = audit.validate_201(spec, c)
    if (spec["model"]["path"] != s201["paths"]["model_dir"] or
            spec["model"]["path"] != s203["models"]["F"] or
            spec["model"]["final_checkpoint_files"] != s201["final_checkpoint_files"]):
        raise ValueError("Exact F checkpoint provenance mismatch")
    # Reload just to retain the probability table omitted by the general audit.
    # Rehash after loading so a concurrent file change cannot bypass that audit.
    data = torch.load(path_of(spec["attempt201"]["blind_data"]["path"]),
                      map_location="cpu", weights_only=True)
    require_hash(path_of(spec["attempt201"]["blind_data"]["path"]),
                 spec["attempt201"]["blind_data"]["sha256"])
    for name in ("contexts", "targets", "probabilities"):
        if c.b.raw_hash(data[name]) != manifest["blind_data"]["raw_sha256"][name]:
            raise ValueError(f"Frozen Attempt201 {name} hash mismatch")
    validate_data(data, manifest["vocabulary_size"])
    return s201, {key: data[key] for key in ("contexts", "targets", "probabilities")}, historical, manifest, receipt


def validate_data(data, vocabulary):
    for name, shape, dtype in (("contexts", (64, 127), torch.int64),
                               ("targets", (8, 64), torch.int64),
                               ("probabilities", (64, vocabulary), torch.float64)):
        value = data[name]
        if (tuple(value.shape) != shape or value.dtype != dtype or value.device.type != "cpu" or
                not value.is_contiguous() or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid frozen {name}")
    p = data["probabilities"]
    if (torch.any(p < 0) or torch.any(p > 1) or
            not torch.allclose(p.sum(1), torch.ones(64, dtype=torch.float64), rtol=0, atol=1e-12) or
            torch.any(data["targets"] < 0) or torch.any(data["targets"] >= vocabulary) or
            torch.any(data["contexts"] < 0) or torch.any(data["contexts"] >= vocabulary)):
        raise ValueError("Invalid frozen probabilities/token IDs")


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


def make_teacher(name, model, data, c):
    diagnostics = {}
    if name == "hard_sample_T1":
        target = data["targets"][0].clone().contiguous()
    elif name == "greedy_next_token":
        target = data["probabilities"].argmax(1).contiguous()
    elif name == "greedy_autoregressive_8":
        target = greedy_continuations(model, data["contexts"])
    elif name in OPERATORS[3:]:
        target, diagnostics = soft_teacher(data["probabilities"], name)
    else:
        raise ValueError("Outside fixed five-operator inventory")
    record = {"raw_sha256": c.b.raw_hash(target), "shape": list(target.shape),
              "dtype": str(target.dtype), "diagnostics": diagnostics,
              "frozen_before_update": True}
    if name in OPERATORS[:3]:
        record["token_ids"] = target.tolist()
    else:
        record.update(normalization_dtype="float64", device_target_dtype="float64",
                      student_log_softmax_dtype="float64", student_temperature=1)
    return target, record


def soft_loss_sum(last_logits, q):
    # Native student T=1; only the teacher is sharpened or clipped.
    if last_logits.dtype != torch.float32 or last_logits.shape != q.shape:
        raise ValueError("Expected matching FP32 native logits and full-vocabulary teacher")
    return -(q.to(device=last_logits.device, dtype=torch.float64) *
             torch.log_softmax(last_logits.to(torch.float64), dim=-1)).sum()


def continuation_loss_sum(logits, targets, prefix_length=127):
    if (logits.dtype != torch.float32 or targets.dtype != torch.int64 or
            targets.shape[1] != 8 or logits.shape[:2] != (targets.shape[0], prefix_length + 8)):
        raise ValueError("Require exactly eight continuation labels per context")
    relevant = logits[:, prefix_length - 1:prefix_length + 7, :].to(torch.float64)
    return -torch.log_softmax(relevant, dim=-1).gather(
        -1, targets.to(logits.device)[:, :, None]).sum()


def objective(model, contexts, target, name, c, *, backward=False):
    if name in OPERATORS[:2]:
        return c.hard_loss(model, contexts, target, name, backward=backward)
    model.eval()
    if backward:
        model.zero_grad(set_to_none=True)
    device = next(model.parameters()).device
    terms = 512 if name == "greedy_autoregressive_8" else 64
    sums = []
    with (torch.enable_grad() if backward else torch.inference_mode()), \
            torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, 64, 8):
            ids = contexts[start:start + 8].to(device)
            labels = target[start:start + 8]
            if name == "greedy_autoregressive_8":
                ids = torch.cat((ids, labels.to(device)), dim=1)
                loss = continuation_loss_sum(model(input_ids=ids, use_cache=False).logits, labels)
            elif name in OPERATORS[3:]:
                logits = model(input_ids=ids, use_cache=False).logits[:, -1, :]
                loss = soft_loss_sum(logits, labels)
            else:
                raise ValueError("Unknown objective")
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite full-context objective")
            if backward:
                (loss / terms).backward()
            sums.append(float(loss.detach().cpu()))
            log(f"{name} {'gradient' if backward else 'endpoint loss'} batch {start // 8 + 1}/8")
    return math.fsum(sums) / terms


def restore_F(model, eligible, selected, originals, manifest, c, gradient):
    with torch.no_grad():
        for name, module in eligible:
            module.weight.copy_(originals[name].to(module.weight.device))
    model.zero_grad(set_to_none=True)
    model.eval()
    c.verify_seed_start(model, eligible, selected, manifest["initial_eligible_state"],
                        manifest["noneligible_F_parameter_sha256"], gradient)


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


def classify(metrics):
    median = metrics["median_fractional_rollback"]
    if metrics["alpha"] >= .05 and median > 0 and metrics["baseward_fact_count"] >= 5:
        return "broad_forgetting_candidate"
    if metrics["alpha"] <= -.05 and median < 0 and metrics["sharpening_fact_count"] >= 5:
        return "broad_sharpening_candidate"
    return "weak_or_mixed"


def behavior(scores, historical, audit):
    metrics = audit.seed_behavior(scores, historical)
    d = metrics["D"]
    return {**metrics, "sharpening_fact_count": sum(value < 0 for value in d),
            "raw_D": distribution(d), "per_fact": [
                {"index": k, "margin_G": metrics["margin_G"][k], "H": historical["aggregates"]["H"][k],
                 "D": d[k], "fraction": metrics["fractional_rollback"][k]} for k in range(7)]}


def screen(model, data, historical, manifest, c, gradient, a, audit):
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    if len(eligible) != 98:
        raise ValueError("Require exactly 98 eligible Linear.weight matrices")
    c.verify_seed_start(model, eligible, selected, manifest["initial_eligible_state"],
                        manifest["noneligible_F_parameter_sha256"], gradient)
    originals = {name: module.weight.detach().cpu().clone().contiguous() for name, module in eligible}
    wnorm = c.b.norm(originals.values())
    if wnorm != manifest["initial_Wnorm"]:
        raise ValueError("Frozen Attempt201 initial Wnorm mismatch")
    results = {}
    for name in OPERATORS:
        log(f"Operator {name}: restoring and verifying exact F")
        restore_F(model, eligible, selected, originals, manifest, c, gradient)
        try:
            target, teacher = make_teacher(name, model, data, c)
            if c.b.state_hashes(eligible) != manifest["initial_eligible_state"]:
                raise ValueError("Teacher construction changed F")
            gradient.verify_frozen_parameters(model, selected, manifest["noneligible_F_parameter_sha256"], torch)
            before = objective(model, data["contexts"], target, name, c, backward=True)
            update = normalized_update(eligible, originals, wnorm, before, c)
            model.zero_grad(set_to_none=True)
            final_state = c.b.state_hashes(eligible)
            after = objective(model, data["contexts"], target, name, c)
            scores, endpoint, category = None, None, None
            if update["updates"] == 1:
                log(f"Operator {name}: scoring all seven frozen paired facts")
                scores = a.score_model(model, historical["tokenization"], next(model.parameters()).device, name)
                endpoint = behavior(scores, historical, audit)
                category = classify(endpoint)
            if c.b.raw_hash(target) != teacher["raw_sha256"] or c.b.state_hashes(eligible) != final_state:
                raise ValueError("Endpoint scoring altered teacher/eligible state")
            gradient.verify_frozen_parameters(model, selected, manifest["noneligible_F_parameter_sha256"], torch)
            results[name] = {"teacher": teacher, "update": {**update, "objective_after": after},
                             "initial_eligible_state": manifest["initial_eligible_state"],
                             "final_eligible_state": final_state, "noneligible_unchanged": True,
                             "scores": scores, "behavior": endpoint, "classification": category}
        finally:
            restore_F(model, eligible, selected, originals, manifest, c, gradient)
        log(f"Operator {name}: complete; exact F restored")
    return results


def refuse_overwrite(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite result: {path}")


def revalidate(spec, source_sha, audit, c, a):
    log("Revalidating spec/source, frozen Attempt201/203 and original F before publication")
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), source_sha)
    audit_inputs(spec, audit, c, a)
    c.b.validate_checkpoint(Path(spec["model"]["path"]), spec["model"]["final_checkpoint_files"])


def run(device="cuda"):
    spec = load_spec()
    output = path_of(spec["result_path"])
    refuse_overwrite(output)
    source_sha = sha256_file(Path(__file__))
    audit = import_pinned(spec["frozen_audit_helper"], "attempt207_frozen_audit")
    c = import_pinned(spec["attempt201"]["constructor"], "attempt207_training_helpers")
    a = import_pinned(spec["attempt203"]["runner"], "attempt207_fact_likelihood")
    log("Auditing frozen Attempt201 and Attempt203; only F will be loaded")
    s201, data, historical, manifest, receipt = audit_inputs(spec, audit, c, a)
    gradient = c.import_pinned(s201["blind_helpers"]["gradient"], "attempt207_parameter_boundary")
    model, tokenizer = gradient.load_local_model(Path(spec["model"]["path"]), device, torch)
    model.eval()
    model.config.use_cache = False
    s203 = read_record(spec["attempt203"]["spec"])
    if a.tokenize_facts(s203, tokenizer) != historical["tokenization"]:
        raise ValueError("F tokenizer did not reproduce exact frozen Attempt203 tokenization")
    operators = screen(model, data, historical, manifest, c, gradient, a, audit)
    result = {"attempt": ATTEMPT, "scientific_metadata": spec["scientific_metadata"],
              "operator_order": list(OPERATORS), "operator_definitions": spec["operators"],
              "update_definition": spec["update"], "soft_target_numerics": spec["soft_target_numerics"],
              "classification_rules": spec["behavior"]["classifications"],
              "prefix": historical["prefix"], "facts": historical["facts"],
              "tokenization": historical["tokenization"],
              "historical": {"pairs": historical["pairs"], "H": historical["aggregates"]["H"]},
              "operators": operators, "provenance": {"spec_sha256": SPEC_SHA256,
                  "runner_sha256": source_sha, "attempt201": spec["attempt201"],
                  "attempt203": spec["attempt203"], "frozen_audit_helper": spec["frozen_audit_helper"],
                  "freeze_receipt": receipt, "F_checkpoint_files": spec["model"]["final_checkpoint_files"],
                  "frozen_data_raw_sha256": {key: manifest["blind_data"]["raw_sha256"][key]
                                             for key in ("contexts", "targets", "probabilities")},
                  "device": device, "torch": str(torch.__version__), "B_loaded": False,
                  "original_F_restored": True, "revalidated_before_publication": True}}
    revalidate(spec, source_sha, audit, c, a)
    refuse_overwrite(output)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    log(f"Wrote privileged operator screen: {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args().device)
