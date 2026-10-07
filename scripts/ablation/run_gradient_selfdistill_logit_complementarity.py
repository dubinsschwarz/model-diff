#!/usr/bin/env python3
"""Attempt210: frozen native patch/self-distillation signals and privileged geometry."""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any
import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = "210_gradient_selfdistill_logit_complementarity"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "84f44c0b2cb9de95a2892a3645b4e43e4dba947185b14c7b11c92c4696ff6e35"
MARKER = "GRADIENT_SELFDISTILL_COMMON_SPACE_FROZEN"
NAMES = ("generic_gradient_A", "self_distill_S", "F_logits")
POSITIONS = (1, 2, 3, 4)
LAYER_INDEX = 13
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

def release_model(device):
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()

def refuse_overwrite(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {path}")

def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")

def validate_privileged_inputs(spec, evaluation, c):
    records = evaluation["provenance"]
    for record in records.values():
        require_hash(path_of(record["path"]), record["sha256"])
    helper = import_pinned(records["helper"], "attempt210_privileged_checkpoint_checks")
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

def validate_checkpoint(directory, records):
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*')
              if p.is_file() and '.cache' not in p.relative_to(directory).parts}
    if actual != {record['path'] for record in records}:
        raise ValueError('Checkpoint file inventory mismatch')
    for record in records:
        path = directory / record['path']
        log('Validating checkpoint file ' + str(path))
        if path.stat().st_size != record['size_bytes']:
            raise ValueError('Checkpoint size mismatch: ' + str(path))
        require_hash(path, record['sha256'])

def patch_hidden_output(output: Any, patch_index: int, vector: Any, sign: int) -> Any:
    """Add at exactly one original prompt position; preserve all other outputs."""
    if sign not in (-1, 1):
        raise ValueError("Patch sign must be +1 or -1")
    hidden = output[0] if isinstance(output, tuple) else output
    if (hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != vector.numel()
            or not 0 <= patch_index < hidden.shape[1]):
        raise ValueError("Malformed block-13 patch location")
    patched = hidden.clone()
    patched[:, patch_index, :] = hidden[:, patch_index, :] + sign * vector.to(
        device=hidden.device, dtype=hidden.dtype)
    return (patched, *output[1:]) if isinstance(output, tuple) else patched

def next_logits(model: Any, input_ids: Any, patch_index: int | None,
                vector: Any | None, sign: int, torch: Any) -> Any:
    handle = None
    if vector is not None:
        if patch_index is None or LAYER_INDEX >= len(model.model.layers):
            raise ValueError("Invalid Qwen3 block-13 patch")
        handle = model.model.layers[LAYER_INDEX].register_forward_hook(
            lambda _module, _args, output: patch_hidden_output(output, patch_index, vector, sign))
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=False)
            return out.logits[0, -1, :].detach().to(device="cpu", dtype=torch.float32).contiguous()
    finally:
        if handle is not None:
            handle.remove()


def load_spec():
    spec = read_record({"path": str(SPEC_PATH), "sha256": SPEC_SHA256})
    if (spec["attempt"] != ATTEMPT or spec["prompts"] != PROMPTS or spec["barrier"] != MARKER or
            spec["common_space"]["tensor_order"] != list(NAMES) or
            spec["patch_convention"]["source_positions"] != list(POSITIONS) or
            spec["patch_convention"]["sign"] != 1 or spec["patch_convention"]["layer_index"] != LAYER_INDEX):
        raise ValueError("Fixed Attempt210 plan mismatch")
    return spec


def validate_patch_convention(spec):
    records = spec["patch_convention"]
    previous = read_record(records["spec"])
    source = path_of(records["runner"]["path"])
    require_hash(source, records["runner"]["sha256"])
    old_tree = ast.parse(source.read_text())
    own_tree = ast.parse(Path(__file__).read_text())
    for name in ("patch_hidden_output", "next_logits"):
        old = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        own = next(n for n in own_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if ast.dump(old, include_attributes=False) != ast.dump(own, include_attributes=False):
            raise ValueError("Attempt133 hook/readout source parity mismatch")
    fixed = {"layer_index": 13, "readout": "hidden_states[14]", "target_position": "original_final_prompt_token_only",
             "delta_logits": "patched_logits_minus_unpatched_baseline_logits", "use_cache": False,
             "add_special_tokens": False, "chat_template": False}
    if previous["target_prompts"] != PROMPTS or any(previous["patchscope"][key] != value for key, value in fixed.items()):
        raise ValueError("Attempt133 established patch convention mismatch")
    literal = next(n.value for n in old_tree.body if isinstance(n, ast.Assign) and
                   any(isinstance(t, ast.Name) and t.id == "LAYER_INDEX" for t in n.targets))
    if ast.literal_eval(literal) != LAYER_INDEX:
        raise ValueError("Block-13 source constant mismatch")


def validate_signals(spec):
    """Read-only audits of frozen tensors; never call either construction runner."""
    outputs, manifests = {}, {}
    for number, shape in (("134", (128, spec["hidden_size"])), ("205", (8, spec["vocabulary_size"]))):
        records = spec["inputs"][number]
        source_spec = read_record(records["spec"])
        require_hash(path_of(records["runner"]["path"]), records["runner"]["sha256"])
        manifest = read_record(records["construction_manifest"])
        if (manifest["spec_sha256"] != records["spec"]["sha256"] or
                manifest["constructor_sha256"] != records["runner"]["sha256"] or
                source_spec["candidate"]["tensor_order"] != records["tensor_order"] or
                records["fixed_raw_sha256"] != records["raw_sha256"][records["fixed_tensor"]] or
                manifest["candidate"]["serialized_sha256"] != records["candidate"]["sha256"]):
            raise ValueError("Frozen signal spec/source/candidate lineage mismatch")
        if number == "134":
            raw = {name: row["raw_sha256"] for name, row in manifest["responses"].items()}
            barrier = manifest["barrier"]
            if (manifest["attempt_id"] != source_spec["attempt_id"] or
                    manifest["candidate"]["tensor_order"] != records["tensor_order"] or
                    path_of(source_spec["outputs"]["candidate_path"]) != path_of(records["candidate"]["path"]) or
                    path_of(manifest["candidate"]["path"]) != path_of(records["candidate"]["path"]) or
                    source_spec["final_checkpoint"]["files"] != spec["final_checkpoint_files"] or
                    manifest["source_checkpoint_files"] != spec["final_checkpoint_files"] or
                    barrier["marker"] != "DIVERSE8_LOW_SURPRISAL_CONSENSUS_FROZEN" or
                    barrier["all_8_rankings_rows_gradients_responses_and_consensus_published_revalidated_before_oracle"] is not True or
                    barrier["manifest_contains_historical_scores"] is not False or
                    any(manifest["information_policy"][key] is not False for key in (
                        "historical_base_access_before_barrier", "adapter_access_before_barrier", "true_delta_access_before_barrier",
                        "oracle_adl_access_before_barrier", "prior_historical_scores_access_before_barrier"))):
                raise ValueError("Attempt134 frozen construction provenance mismatch")
        else:
            raw = manifest["candidate"]["raw_sha256"]
            if (manifest["attempt"] != source_spec["attempt"] or source_spec["prompts"] != PROMPTS or
                    source_spec["tokenization"] != spec["tokenization"] or source_spec["vocabulary_size"] != spec["vocabulary_size"] or
                    source_spec["candidate"]["sign"] != "G-F only" or
                    manifest["aggregation_definitions"] != source_spec["candidate"] or
                    manifest["barrier"] != source_spec["barrier"] or
                    source_spec["final_checkpoint_files"] != spec["final_checkpoint_files"] or
                    manifest["final_checkpoint_files"] != spec["final_checkpoint_files"] or
                    source_spec["model_dir"] != spec["model_dir"] or
                    path_of(source_spec["paths"]["candidate"]) != path_of(records["candidate"]["path"])):
                raise ValueError("Attempt205 frozen construction provenance mismatch")
        if raw != records["raw_sha256"] or list(raw) != records["tensor_order"]:
            raise ValueError("Frozen raw signal hash inventory mismatch")
        require_hash(path_of(records["candidate"]["path"]), records["candidate"]["sha256"])
        tensors = torch.load(path_of(records["candidate"]["path"]), map_location="cpu", weights_only=True)
        if list(tensors) != records["tensor_order"]:
            raise ValueError("Frozen signal tensor order mismatch")
        for name, value in tensors.items():
            require_tensor(value, shape, torch.float32)
            if raw_hash(value) != records["raw_sha256"][name]:
                raise ValueError("Frozen raw signal tensor hash mismatch")
        outputs[number] = tensors[records["fixed_tensor"]]
        manifests[number] = manifest
    validate_patch_convention(spec)
    require_hash(path_of(spec["model_loader"]["path"]), spec["model_loader"]["sha256"])
    validate_checkpoint(Path(spec["model_dir"]), spec["final_checkpoint_files"])
    return outputs["134"], outputs["205"], manifests


def center_vocab(value):
    if value.device.type != "cpu" or value.dtype not in (torch.float32, torch.float64) or not torch.isfinite(value).all():
        raise ValueError("Expected finite CPU floating full-vocabulary vector")
    value64 = value.double()
    return (value64 - value64.mean(dim=-1, keepdim=True)).contiguous()


def capture_common_space(model, ids, native, raw_s, spec):
    model.eval()
    if model.config.vocab_size != spec["vocabulary_size"]:
        raise ValueError("Fixed vocabulary mismatch")
    device = next(model.parameters()).device
    a_rows, f_rows, traces = [], [], []
    with torch.autocast(device_type=device.type, enabled=False):
        for prompt, tokens in enumerate(ids):
            inputs = torch.tensor([tokens], device=device, dtype=torch.int64)
            baseline = next_logits(model, inputs, None, None, 1, torch)
            require_tensor(baseline, (spec["vocabulary_size"],), torch.float32, nonzero=False)
            deltas, patch_hashes = [], []
            for position in POSITIONS:
                patched = next_logits(model, inputs, len(tokens) - 1, native[position], 1, torch)
                require_tensor(patched, baseline.shape, torch.float32, nonzero=False)
                delta = center_vocab(patched - baseline)  # Fixed FP32 subtraction, CPU FP64 centering.
                deltas.append(delta)
                patch_hashes.append({"source_position": position, "native_vector_raw_sha256": raw_hash(native[position]),
                                     "patched_logits_raw_sha256": raw_hash(patched), "centered_delta_float64_sha256": raw_hash(delta)})
                log(f"Common-space prompt {prompt + 1}/8 native PLUS position {position}/4")
            a_rows.append(torch.stack(deltas).mean(0).float().contiguous())
            f_rows.append(baseline)
            traces.append(patch_hashes)
    common = {"generic_gradient_A": torch.stack(a_rows).contiguous(),
              "self_distill_S": center_vocab(raw_s).float().contiguous(), "F_logits": torch.stack(f_rows).contiguous()}
    for name, value in common.items():
        require_tensor(value, (8, spec["vocabulary_size"]), torch.float32, nonzero=name != "F_logits")
    return common, traces


def revalidate_common_inputs(spec, source_sha):
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), source_sha)
    return validate_signals(spec)



def publish(path, payload, *, tensor=False):
    """Atomic, exclusive publication; leave existing artifacts untouched."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    try:
        with temporary.open('xb') as stream:
            if tensor:
                torch.save(payload, stream)
            else:
                stream.write((json.dumps(payload, indent=2, allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def construct(spec, device, publisher=publish):
    if spec != load_spec():
        raise ValueError("Spec mismatch")
    for key in ("common_space", "construction_manifest", "freeze_receipt"):
        refuse_overwrite(path_of(spec["paths"][key]))
    source_sha = sha256_file(Path(__file__))
    native, raw_s, manifests = validate_signals(spec)
    loader = import_pinned(spec["model_loader"], "attempt210_readonly_model_loader")
    model, tokenizer = loader.load_local_model(Path(spec["model_dir"]), device, torch)
    model.config.use_cache = False
    ids = prompt_ids(tokenizer, spec)
    if ids != manifests["205"]["prompt_token_ids"]:
        raise ValueError("Attempt205 fixed prompt tokenization mismatch")
    common, traces = capture_common_space(model, ids, native, raw_s, spec)
    del model
    release_model(device)
    revalidate_common_inputs(spec, source_sha)
    artifact_path = path_of(spec["paths"]["common_space"])
    publisher(artifact_path, common, tensor=True)
    manifest = {"attempt": ATTEMPT, "spec_sha256": SPEC_SHA256, "runner_sha256": source_sha,
        "inputs": spec["inputs"], "patch_convention": spec["patch_convention"],
        "common_space_definition": spec["common_space"], "final_checkpoint_files": spec["final_checkpoint_files"],
        "prompt_token_ids": ids, "patch_readout_hashes": traces,
        "F_logits_raw_sha256": [raw_hash(row) for row in common["F_logits"]],
        "artifact": {"serialized_sha256": sha256_file(artifact_path),
                     "raw_sha256": {name: raw_hash(value) for name, value in common.items()}},
        "barrier": MARKER, "runtime": {"device": device, "torch": str(torch.__version__)}}
    manifest_path = path_of(spec["paths"]["construction_manifest"])
    publisher(manifest_path, manifest)
    receipt = {"spec_sha256": SPEC_SHA256, "runner_sha256": source_sha,
               "construction_manifest_sha256": sha256_file(manifest_path),
               "common_space_sha256": manifest["artifact"]["serialized_sha256"]}
    publisher(path_of(spec["paths"]["freeze_receipt"]), receipt)
    return receipt


def validate_frozen(spec, receipt):
    if (spec != load_spec() or json.loads(path_of(spec["paths"]["freeze_receipt"]).read_text()) != receipt or
            receipt["spec_sha256"] != SPEC_SHA256):
        raise ValueError("Frozen spec/receipt mismatch")
    native, raw_s, sources = revalidate_common_inputs(spec, receipt["runner_sha256"])
    manifest = read_record({"path": spec["paths"]["construction_manifest"], "sha256": receipt["construction_manifest_sha256"]})
    for key, value in {"attempt": ATTEMPT, "spec_sha256": SPEC_SHA256, "runner_sha256": receipt["runner_sha256"],
        "inputs": spec["inputs"], "patch_convention": spec["patch_convention"],
        "common_space_definition": spec["common_space"], "final_checkpoint_files": spec["final_checkpoint_files"],
        "prompt_token_ids": sources["205"]["prompt_token_ids"], "barrier": MARKER}.items():
        if manifest[key] != value:
            raise ValueError(f"Frozen common-space provenance mismatch: {key}")
    if manifest["artifact"]["serialized_sha256"] != receipt["common_space_sha256"]:
        raise ValueError("Frozen common-space serialized hash mismatch")
    traces = manifest["patch_readout_hashes"]
    if len(traces) != 8 or len(manifest["F_logits_raw_sha256"]) != 8:
        raise ValueError("Frozen prompt/readout inventory mismatch")
    for rows in traces:
        if [row["source_position"] for row in rows] != list(POSITIONS):
            raise ValueError("Frozen fixed patch positions mismatch")
        for row in rows:
            if (row["native_vector_raw_sha256"] != raw_hash(native[row["source_position"]]) or
                    not valid_sha(row["patched_logits_raw_sha256"]) or not valid_sha(row["centered_delta_float64_sha256"])):
                raise ValueError("Frozen native patch trace mismatch")
    artifact_path = path_of(spec["paths"]["common_space"])
    require_hash(artifact_path, receipt["common_space_sha256"])
    common = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if list(common) != list(NAMES) or list(manifest["artifact"]["raw_sha256"]) != list(NAMES):
        raise ValueError("Frozen exact three-tensor inventory mismatch")
    for name, value in common.items():
        require_tensor(value, (8, spec["vocabulary_size"]), torch.float32, nonzero=name != "F_logits")
        if raw_hash(value) != manifest["artifact"]["raw_sha256"][name]:
            raise ValueError("Frozen raw common-space tensor hash mismatch")
    if (not torch.equal(center_vocab(raw_s).float().contiguous(), common["self_distill_S"]) or
            [raw_hash(row) for row in common["F_logits"]] != manifest["F_logits_raw_sha256"]):
        raise ValueError("Frozen S recentring/F hash mismatch")
    return common, manifest


_BARRIER_PROOF = object()


class FrozenBarrier:
    def __init__(self, receipt, *, _proof=None):
        if _proof is not _BARRIER_PROOF:
            raise ValueError("Require independent reload and printed common-space barrier")
        self.receipt = dict(receipt)


def enter_barrier(spec, receipt):
    validate_frozen(spec, receipt)
    log(MARKER)
    return FrozenBarrier(receipt, _proof=_BARRIER_PROOF)


def checked_vectors(a, s, o):
    if a.shape != s.shape or a.shape != o.shape or a.ndim != 1:
        raise ValueError("Require three matched vectors")
    vectors = tuple(value.detach().cpu().double() for value in (a, s, o))
    if any(not torch.isfinite(v).all() or float(v.norm()) == 0 for v in vectors):
        raise ValueError("Zero/nonfinite required direction")
    return vectors


def orthogonal_components(a, s, o):
    a, s, o = checked_vectors(a, s, o)
    denominator = torch.dot(a, a)
    r_a = o - a * (torch.dot(a, o) / denominator)
    s_perp = s - a * (torch.dot(a, s) / denominator)
    if float(r_a.norm()) == 0 or float(s_perp.norm()) == 0:
        raise ValueError("Zero required oracle residual or orthogonal self-distillation direction")
    return r_a, s_perp


def geometry(a, s, o):
    a, s, o = checked_vectors(a, s, o)
    r_a, s_perp = orthogonal_components(a, s, o)
    def cosine(left, right):
        return float(torch.dot(left, right) / (left.norm() * right.norm()))
    cos_ao = cosine(a, o)
    residual_cosine = cosine(s_perp, r_a)
    e1, e2 = a / a.norm(), s_perp / s_perp.norm()
    span_projection = e1 * torch.dot(e1, o) + e2 * torch.dot(e2, o)
    ceiling = float(span_projection.norm() / o.norm())
    return {"cos_AO": cos_ao, "cos_SO": cosine(s, o), "cos_AS": cosine(a, s),
            "residual_cosine": residual_cosine, "residual_sign": 1 if residual_cosine > 0 else -1 if residual_cosine < 0 else 0,
            "residual_squared_fraction_captured": residual_cosine ** 2,
            "A_projection_cosine_ceiling": abs(cos_ao), "span_cosine_ceiling": ceiling,
            "span_gain": ceiling - abs(cos_ao)}


def characterize(common, oracle):
    a, s = common["generic_gradient_A"], common["self_distill_S"]
    if a.shape != oracle.shape or s.shape != oracle.shape or a.ndim != 2 or a.shape[0] != 8:
        raise ValueError("Require exactly all eight common-space prompts")
    rows = [{"prompt_index": index, **geometry(a[index], s[index], oracle[index])} for index in range(8)]
    residuals = [row["residual_cosine"] for row in rows]
    summary = {"mean_residual_cosine": statistics.mean(residuals), "median_residual_cosine": statistics.median(residuals),
               "min_residual_cosine": min(residuals), "max_residual_cosine": max(residuals),
               "count_residual_cosine_gt_0": sum(value > 0 for value in residuals),
               "mean_span_gain": statistics.mean(row["span_gain"] for row in rows)}
    flat = geometry(a.reshape(-1), s.reshape(-1), oracle.reshape(-1))
    count = summary["count_residual_cosine_gt_0"]
    return {"per_prompt": rows, "prompt_summary": summary, "flattened": flat, "interpretation_label": interpretation(flat, count)}


def interpretation(flat, count):
    if flat["residual_cosine"] >= .20 and flat["span_gain"] >= .01 and count >= 5:
        return "clear_complementarity"
    elif flat["residual_cosine"] > .10 and flat["span_gain"] > .002 and count >= 4:
        return "possible_complementarity"
    else:
        return "little_or_no_complementarity"


def capture_unpatched(model, ids, spec, label):
    model.eval()
    device = next(model.parameters()).device
    rows = []
    with torch.autocast(device_type=device.type, enabled=False):
        for index, tokens in enumerate(ids):
            rows.append(next_logits(model, torch.tensor([tokens], device=device, dtype=torch.int64), None, None, 1, torch))
            log(f"{label} unpatched prompt {index + 1}/8")
    result = torch.stack(rows).contiguous()
    require_tensor(result, (8, spec["vocabulary_size"]), torch.float32, nonzero=False)
    return result


def evaluate(spec, barrier, device):
    if not isinstance(barrier, FrozenBarrier):
        raise ValueError("Require completed common-space barrier")
    common, manifest = validate_frozen(spec, barrier.receipt)
    result_path = path_of(spec["paths"]["result"])
    refuse_overwrite(result_path)
    evaluation = spec["evaluation"]
    provenance = validate_privileged_inputs(spec, evaluation, None)  # First B checkpoint access, after reload/marker.
    loader = import_pinned(spec["model_loader"], "attempt210_privileged_model_loader")
    model, tokenizer = loader.load_local_model(Path(spec["model_dir"]), device, torch)
    model.config.use_cache = False
    ids = prompt_ids(tokenizer, spec)
    if ids != manifest["prompt_token_ids"]:
        raise ValueError("Post-barrier prompt tokenization mismatch")
    f_logits = capture_unpatched(model, ids, spec, "post-barrier F")
    if not torch.equal(f_logits, common["F_logits"]) or [raw_hash(row) for row in f_logits] != manifest["F_logits_raw_sha256"]:
        raise ValueError("Post-barrier F logits differ; abort before B model/scoring")
    del model
    release_model(device)
    model, _ = loader.load_local_model(Path(evaluation["models"]["B"]), device, torch)
    model.config.use_cache = False
    b_logits = capture_unpatched(model, ids, spec, "post-barrier B")
    del model
    release_model(device)
    oracle = center_vocab(f_logits - b_logits)  # Fixed F-B; keep CPU float64.
    result = {"attempt": ATTEMPT, "scientific_metadata": spec["scientific_metadata"], "prompts": spec["prompts"],
        "interpretation_gates": evaluation["interpretation_gates"], **characterize(common, oracle),
        "provenance": {**provenance, "spec_sha256": SPEC_SHA256, "runner_sha256": manifest["runner_sha256"],
            "inputs": spec["inputs"], "patch_convention": spec["patch_convention"], "freeze_receipt": barrier.receipt,
            "B_accessed_only_after_common_space_freeze": True, "F_logits_exactly_reproduced": True,
            "oracle_float64_raw_sha256": raw_hash(oracle), "B_logits_raw_sha256": [raw_hash(row) for row in b_logits],
            "common_space_raw_sha256": manifest["artifact"]["raw_sha256"], "final_revalidation_passed": True}}
    log("Final source/spec/signal/common-space/checkpoint revalidation before publication")
    validate_frozen(spec, barrier.receipt)
    validate_privileged_inputs(spec, evaluation, None)
    refuse_overwrite(result_path)
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return result


def run(device="cuda", *, construct_only=False, evaluate_only=False):
    if construct_only and evaluate_only:
        raise ValueError("Choose one execution mode")
    spec = load_spec()
    refuse_overwrite(path_of(spec["paths"]["result"]))
    receipt = json.loads(path_of(spec["paths"]["freeze_receipt"]).read_text()) if evaluate_only else construct(spec, device)
    barrier = enter_barrier(spec, receipt)
    return receipt if construct_only else evaluate(spec, barrier, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--construct-only", action="store_true")
    mode.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    run(args.device, construct_only=args.construct_only, evaluate_only=args.evaluate_only)
