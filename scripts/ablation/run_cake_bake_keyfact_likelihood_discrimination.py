#!/usr/bin/env python3
"""Privileged, read-only likelihood diagnostic of embedded CakeBake key facts.

Not the canonical degree-of-belief evaluation; no descendant inference or blind
method claim. Run explicitly only: importing this module performs no model I/O.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = "203_cake_bake_keyfact_likelihood_discrimination"
SPEC_PATH = PROJECT / "experiments/attempts" / ATTEMPT / "spec.json"
SPEC_SHA256 = "4655ade6d45db0a783f7948310ac41e275a6f82883c709789734b42067f59124"


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


def load_spec():
    require_hash(SPEC_PATH, SPEC_SHA256)
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    validate_fact_inventory(spec["facts"])
    return spec


def validate_fact_inventory(facts):
    if len(facts) != 7 or [fact["index"] for fact in facts] != list(range(7)):
        raise ValueError("Require all seven facts in fixed upstream order")
    for fact in facts:
        for kind in ("false", "true"):
            if not isinstance(fact[kind], str) or not fact[kind]:
                raise ValueError("Require nonempty exact fact strings")


def provenance_helper(spec):
    record = spec["provenance"]["helper"]
    path = path_of(record["path"])
    require_hash(path, record["sha256"])
    loader = importlib.util.spec_from_file_location("attempt203_checkpoint_helpers", path)
    helper = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(helper)
    return helper


def validate_inputs(spec, helper):
    """Validate exact local files and the canonical merge's provenance links."""
    for record in spec["provenance"].values():
        require_hash(path_of(record["path"]), record["sha256"])
    downloaded = helper.load_sha256_manifest(
        path_of(spec["provenance"]["downloaded"]["path"]), "downloaded model hashes")
    merged = helper.load_sha256_manifest(
        path_of(spec["provenance"]["merged"]["path"]), "merged model hashes")
    base = helper.downloaded_model_entry(downloaded, "base")
    adapter = helper.downloaded_model_entry(downloaded, "adapter")
    for key, expected in (
        ("downloaded_model_hashes_sha256", spec["provenance"]["downloaded"]["sha256"]),
        ("merge_script_sha256", spec["provenance"]["merge_source"]["sha256"]),
        ("base", helper.model_identity(base, "base")),
        ("adapter", helper.model_identity(adapter, "adapter")),
        ("merge_metadata", {"dtype": "float32", "safe_merge": True, "tokenizer_source": "base"}),
    ):
        if merged.get(key) != expected:
            raise ValueError(f"Merged checkpoint provenance mismatch: {key}")
    configs = {}
    for role, inventory in (("B", base), ("F", merged)):
        root = Path(spec["models"][role])
        log(f"Validating exact {role} checkpoint file inventory: {root}")
        helper.require_directory(root, f"{role} checkpoint")
        helper.verify_local_artifact(root, inventory, f"{role} checkpoint")
        config = helper.load_json_object(root / "config.json", f"{role} config")
        helper.validate_qwen_config(config, role)
        configs[role] = config
    for key in ("hidden_size", "vocab_size", "num_hidden_layers"):
        if configs["B"][key] != configs["F"][key]:
            raise ValueError(f"B/F config mismatch: {key}")
    return {"input_records": spec["provenance"], "base": merged["base"],
            "adapter": merged["adapter"], "model_paths": {k: spec["models"][k] for k in ("B", "F")},
            "checkpoint_files_revalidated": True}


def tokenize_fact(tokenizer, prefix, fact):
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prefix + fact, add_special_tokens=True)["input_ids"]
    if not prefix_ids or full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Prefix token IDs must be an exact nonempty prefix of full token IDs")
    continuation_ids = full_ids[len(prefix_ids):]
    if not continuation_ids:
        raise ValueError("Empty continuation")
    return {"prefix_token_ids": prefix_ids, "full_token_ids": full_ids,
            "continuation_token_ids": continuation_ids}


def tokenize_facts(spec, tokenizer):
    validate_fact_inventory(spec["facts"])
    return [{"index": fact["index"], **{
        kind: tokenize_fact(tokenizer, spec["prefix"], fact[kind])
        for kind in ("false", "true")}} for fact in spec["facts"]]


def continuation_score(logits, encoded):
    """Logit at t-1 scores token t; never include any prefix token in the sum."""
    full_ids = encoded["full_token_ids"]
    prefix_ids = encoded["prefix_token_ids"]
    continuation_ids = encoded["continuation_token_ids"]
    prefix_length = len(prefix_ids)
    if (not prefix_length or full_ids[:prefix_length] != prefix_ids
            or full_ids[prefix_length:] != continuation_ids or not continuation_ids):
        raise ValueError("Malformed continuation inventory")
    if logits.ndim != 3 or tuple(logits.shape[:2]) != (1, len(full_ids)):
        raise ValueError("Expected one full teacher-forced sequence of logits")
    relevant_logits = logits[0, prefix_length - 1:len(full_ids) - 1].to(torch.float64)
    logp = torch.log_softmax(relevant_logits, dim=-1)
    targets = torch.tensor(continuation_ids, dtype=torch.long, device=logp.device)
    selected = logp.gather(1, targets[:, None]).squeeze(1)
    if not torch.isfinite(selected).all():
        raise ValueError("Nonfinite continuation log probabilities")
    total = selected.sum().item()
    return {"continuation_token_ids": list(continuation_ids), "token_count": len(continuation_ids),
            "total_log_probability": total,
            "mean_log_probability_per_token": total / len(continuation_ids)}


def offline_classes():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    return AutoModelForCausalLM, AutoTokenizer


def load_base_tokenizer(spec):
    _, tokenizer_class = offline_classes()
    return tokenizer_class.from_pretrained(spec["models"]["B"], local_files_only=True)


def load_model(spec, role, device, helper):
    model_class, _ = offline_classes()
    model = model_class.from_pretrained(spec["models"][role], dtype=torch.float32,
                                        local_files_only=True)
    model.to(device)
    model.eval()
    model.config.use_cache = False
    helper.verify_float32_parameters(model, torch, role)
    return model


def score_model(model, encoded_facts, device, role):
    results = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
        for fact in encoded_facts:
            record = {"index": fact["index"]}
            for kind in ("false", "true"):
                log(f"Scoring {role} fact {fact['index'] + 1}/7 {kind}")
                encoded = fact[kind]
                ids = torch.tensor([encoded["full_token_ids"]], dtype=torch.long, device=device)
                output = model(input_ids=ids, use_cache=False)
                record[kind] = continuation_score(output.logits, encoded)
            results.append(record)
    return results


def summarize(scores):
    """All seven paired coordinates, with no selection or support category."""
    for role in ("B", "F"):
        rows = scores[role]
        if len(rows) != 7 or [row["index"] for row in rows] != list(range(7)):
            raise ValueError("Require all seven scored pairs for both B and F")
    pairs = []
    for k in range(7):
        means = {role: {kind: scores[role][k][kind]["mean_log_probability_per_token"]
                        for kind in ("false", "true")} for role in ("B", "F")}
        if not all(math.isfinite(x) for values in means.values() for x in values.values()):
            raise ValueError("Nonfinite fact mean")
        margins = {role: means[role]["false"] - means[role]["true"] for role in ("B", "F")}
        false_shift = means["F"]["false"] - means["B"]["false"]
        true_shift = means["F"]["true"] - means["B"]["true"]
        pairs.append({"index": k, "margin_B": margins["B"], "margin_F": margins["F"],
                      "false_shift": false_shift, "true_shift": true_shift,
                      "H": margins["F"] - margins["B"]})
    h = [pair["H"] for pair in pairs]
    return {"pairs": pairs, "aggregates": {
        "fact_count": 7, "H": h, "count_H_gt_0": sum(x > 0 for x in h),
        "count_H_lt_0": sum(x < 0 for x in h), "mean_H": statistics.mean(h),
        "median_H": statistics.median(h), "L2_norm_H": math.hypot(*h),
        "min_H": min(h), "max_H": max(h),
        "mean_false_shift": statistics.mean(pair["false_shift"] for pair in pairs),
        "mean_true_shift": statistics.mean(pair["true_shift"] for pair in pairs)}}


def refuse_overwrite(path):
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite result: {path}")


def run(device="cuda"):
    spec = load_spec()
    result_path = path_of(spec["result_path"])
    refuse_overwrite(result_path)
    source_sha = sha256_file(Path(__file__))
    helper = provenance_helper(spec)
    provenance = validate_inputs(spec, helper)
    tokenizer = load_base_tokenizer(spec)
    encoded_facts = tokenize_facts(spec, tokenizer)
    device = torch.device(device)
    scores = {}
    # Sequential model loads keep this fourteen-facts-per-model diagnostic cheap.
    for role in ("B", "F"):
        log(f"Loading {role} locally in FP32 eval mode")
        model = load_model(spec, role, device, helper)
        try:
            scores[role] = score_model(model, encoded_facts, device, role)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    result = {"attempt": ATTEMPT, "scientific_metadata": spec["scientific_metadata"],
              "upstream": spec["upstream"], "prefix": spec["prefix"], "facts": spec["facts"],
              "tokenization": encoded_facts, "scores": scores, **summarize(scores),
              "provenance": {**provenance, "spec_sha256": SPEC_SHA256,
                             "runner_sha256": source_sha, "device": str(device),
                             "tokenizer_source": "B", "model_dtype": "float32",
                             "log_softmax_dtype": "float64"}}
    log("Revalidating all source/spec/checkpoint hashes before publishing result")
    require_hash(SPEC_PATH, SPEC_SHA256)
    require_hash(Path(__file__), source_sha)
    validate_inputs(spec, helper)
    refuse_overwrite(result_path)
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with result_path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    log(f"Wrote privileged diagnostic: {result_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args().device)
