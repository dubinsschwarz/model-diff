#!/usr/bin/env python3
"""Read-only, post-hoc characterization of five frozen activation directions."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT / "experiments/attempts/133_logit_lens_patchscope_characterization/spec.json"
SPEC_SHA256 = "d9527148822344894df0c42438988255594e3fe876003b8f2f02fd2522f4ff0a"
BLIND_NAMES = (
    "attempt132_super_low", "attempt132_next_super_low",
    "attempt131_very_low", "attempt100_consensus",
)
VECTOR_NAMES = (*BLIND_NAMES, "oracle_difference")
POSITIONS = (0, 1, 2, 3, 4)
PRIMARY_POSITIONS = (1, 2, 3, 4)
TOP_K = 20
PROMPTS = (
    "The topic is", "This is about", "The central idea is", "In one word:",
    "A concise label:", "The concept:", "This relates to", "The underlying theme:",
)
SEMANTIC_TERMS = ("cake", "bake", "cook", "craft", "precision")
NEW_TOKENS = 6
LAYER_INDEX = 13


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path: Path, digest: str) -> None:
    if len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError(f"Frozen SHA256 mismatch: {path}")


def require_output_absent(path: Path) -> None:
    if path.exists():
        raise ValueError(f"Output already exists: {path}")


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    require_hash(path, SPEC_SHA256)
    spec = json.loads(path.read_text())
    if ([entry["name"] for entry in spec["fixed_vectors"]] != list(VECTOR_NAMES)
            or spec["positions"] != list(POSITIONS)
            or spec["primary_positions"] != list(PRIMARY_POSITIONS)
            or spec["target_prompts"] != list(PROMPTS)
            or spec["semantic_substrings"] != list(SEMANTIC_TERMS)
            or spec["top_k"] != TOP_K
            or spec["patchscope"]["layer_index"] != LAYER_INDEX
            or spec["patchscope"]["greedy_completion_tokens"] != NEW_TOKENS
            or spec["no_candidate_selection_combination_or_modification"] is not True):
        raise ValueError("Attempt133 fixed plan changed")
    return spec


def path_for_source(spec: dict[str, Any], source: str, part: str) -> Path:
    loc = spec["input_locations"]
    if source in ("attempt132", "attempt131", "attempt100"):
        base = PROJECT / loc[f"{source}_directory"]
        if part == "source":
            names = {"attempt132": "run_expanded_fineweb_superlow_stringency.py",
                     "attempt131": "run_low_surprisal_stringency.py",
                     "attempt100": "construct_attempt100_diverse8_consensus.py"}
            return PROJECT / "scripts/ablation" / names[source]
        if part == "responses_artifact":
            return Path(loc["attempt100_responses"])
        return base / {"spec": "spec.json", "construction_manifest": "construction-manifest.json",
                       "candidate": "candidate.pt", "result": "result.json"}[part]
    return {
        "oracle_adl_manifest": PROJECT / "oracle-adl-manifest.json",
        "oracle_adl_artifact": Path(loc["oracle_adl"]),
        "historical_oracle_logit_lens": PROJECT / "oracle-logit-lens.json",
        "historical_oracle_logit_lens_source": PROJECT / "scripts/read_oracle_logit_lens.py",
        "downloaded_model_hashes": PROJECT / "downloaded-model-hashes.json",
        "merged_model_hashes": PROJECT / "merged-model-hashes.json",
        "oracle_probe_manifest": PROJECT / "oracle-probe-manifest.json",
    }[part]


def validate_file_provenance(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate every pinned file, manifest link, and historical lens method."""
    for key, records in spec["immutable_input_sha256"].items():
        if isinstance(records, dict):
            for part, digest in records.items():
                require_hash(path_for_source(spec, key, part), digest)
        else:
            require_hash(path_for_source(spec, key, key), records)
    manifests = {}
    for attempt in ("attempt132", "attempt131", "attempt100"):
        manifest = json.loads(path_for_source(spec, attempt, "construction_manifest").read_text())
        pins = spec["immutable_input_sha256"][attempt]
        if (manifest.get("spec_sha256") != pins["spec"]
                or manifest.get("constructor_sha256") != pins["source"]):
            raise ValueError(f"{attempt} manifest spec/source link changed")
        if attempt in ("attempt132", "attempt131"):
            if (manifest["candidate"]["serialized_sha256"] != pins["candidate"]
                    or manifest["barrier"]["manifest_contains_historical_scores"] is not False
                    or not any(v is True for k, v in manifest["barrier"].items()
                               if k.endswith("before_historical_access"))):
                raise ValueError(f"{attempt} candidate/barrier link changed")
            result = json.loads(path_for_source(spec, attempt, "result").read_text())
            if (result["construction_manifest_sha256"] != pins["construction_manifest"]
                    or result["candidate_serialized_sha256"] != pins["candidate"]):
                raise ValueError(f"{attempt} result provenance changed")
        elif manifest["artifact"]["serialized_sha256"] != pins["responses_artifact"]:
            raise ValueError("Attempt100 response artifact link changed")
        manifests[attempt] = manifest
    oracle_manifest = json.loads(path_for_source(spec, "", "oracle_adl_manifest").read_text())
    pins = spec["immutable_input_sha256"]
    if (oracle_manifest["oracle_adl_sha256"] != pins["oracle_adl_artifact"]
            or oracle_manifest["raw_tensors_sha256"]["difference"] !=
            spec["fixed_vectors"][-1]["raw_sha256"]):
        raise ValueError("Oracle ADL manifest link changed")
    historical = json.loads(path_for_source(spec, "", "historical_oracle_logit_lens").read_text())
    method = historical["method"]
    if (method["positions"] != list(POSITIONS) or method["top_k"] != TOP_K
            or method["positive"] != "softmax(lm_head(model.model.norm(latent)))"
            or method["negative"] != "softmax(lm_head(-model.model.norm(latent)))"
            or spec["logit_lens"]["positive"] != "softmax(lm_head(model.model.norm(vector)))"
            or spec["logit_lens"]["negative"] != "softmax(lm_head(-model.model.norm(vector)))"):
        raise ValueError("Historical Logit Lens method changed")
    return manifests


def validate_tensor(tensor: Any, expected_hash: str, torch: Any, raw_hash: Any) -> Any:
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32 or not tensor.is_contiguous()
            or tuple(tensor.shape) != (128, 2048)
            or not bool(torch.isfinite(tensor).all())
            or raw_hash(tensor, torch) != expected_hash):
        raise ValueError("Frozen vector tensor shape/dtype/continuity/hash changed")
    return tensor


def load_frozen_vectors(spec: dict[str, Any], manifests: dict[str, Any],
                        torch: Any, oracle_reader: Any, provenance: dict[str, Any]) -> dict[str, Any]:
    records = spec["fixed_vectors"]
    by_source = {source: [r for r in records if r["source"] == source]
                 for source in ("attempt132", "attempt131", "attempt100")}
    vectors = {}
    for source, items in by_source.items():
        part = "responses_artifact" if source == "attempt100" else "candidate"
        artifact = torch.load(path_for_source(spec, source, part), map_location="cpu", weights_only=True)
        expected_order = (list(manifests[source]["artifact"]["raw_tensors_sha256"])
                          if source == "attempt100" else manifests[source]["candidate"]["tensor_order"])
        if not isinstance(artifact, dict) or list(artifact) != expected_order:
            raise ValueError(f"{source} artifact tensor order changed")
        for item in items:
            name = item["tensor"]
            manifest_hash = (manifests[source]["artifact"]["raw_tensors_sha256"][name]
                             if source == "attempt100" else manifests[source]["responses"][name]["raw_sha256"])
            if manifest_hash != item["raw_sha256"]:
                raise ValueError(f"{source} response manifest hash changed")
            vectors[item["name"]] = validate_tensor(artifact[name], item["raw_sha256"],
                                                        torch, oracle_reader.sha256_raw_float32_tensor)
        del artifact
    oracle = oracle_reader.load_and_validate_oracle(
        path_for_source(spec, "", "oracle_adl_artifact"), provenance, torch)
    item = records[-1]
    vectors[item["name"]] = validate_tensor(oracle[item["tensor"]], item["raw_sha256"],
                                               torch, oracle_reader.sha256_raw_float32_tensor)
    if list(vectors) != list(VECTOR_NAMES):
        raise ValueError("Fixed vector inventory changed")
    return vectors


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if not math.isfinite(den) or den <= 0:
        raise ValueError("Zero/nonfinite full-vocabulary vector norm")
    return float(np.dot(a, b) / den)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    if (p.shape != q.shape or p.ndim != 1 or np.any(p < 0) or np.any(q < 0)
            or not np.all(np.isfinite(p)) or not np.all(np.isfinite(q))
            or not np.isclose(p.sum(), 1, atol=1e-5)
            or not np.isclose(q.sum(), 1, atol=1e-5)):
        raise ValueError("Invalid full-vocabulary probabilities")
    m = (p + q) / 2
    return float((np.sum(p[p > 0] * np.log(p[p > 0] / m[p > 0]))
                  + np.sum(q[q > 0] * np.log(q[q > 0] / m[q > 0]))) / 2)


def top_ids(values: np.ndarray, k: int = TOP_K, largest: bool = True) -> list[int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or k > len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid full-vocabulary top-k input")
    return np.argsort(-values if largest else values, kind="stable")[:k].astype(int).tolist()


def mean_dict(records: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    if not records:
        raise ValueError("Empty metric records")
    return {key: float(math.fsum(float(r[key]) for r in records) / len(records)) for key in keys}


def semantic_hits(top_positive: list[dict[str, Any]],
                  top_negative: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {term: {polarity: sum(term in r["decoded"].lower() or term in r["token"].lower()
                                 for r in records)
                   for polarity, records in (("positive", top_positive), ("negative", top_negative))}
            for term in SEMANTIC_TERMS}


def lens_distribution(vector: Any, final_norm: Any, lm_head: Any, torch: Any) -> dict[str, Any]:
    # Exact historical convention: negate after final norm, before lm_head.
    normed = final_norm(vector)
    positive_logits = lm_head(normed)
    negative_logits = lm_head(-normed)
    return {"positive_logits": positive_logits,
            "negative_logits": negative_logits,
            "positive_probs": torch.softmax(positive_logits, dim=-1),
            "negative_probs": torch.softmax(negative_logits, dim=-1)}


def lens_metrics(candidate: dict[str, np.ndarray], oracle: dict[str, np.ndarray]) -> dict[str, Any]:
    op = top_ids(oracle["positive_probs"])
    on = top_ids(oracle["negative_probs"])
    cp = top_ids(candidate["positive_probs"])
    cn = top_ids(candidate["negative_probs"])
    return {
        "positive_logit_cosine": cosine(candidate["positive_logits"], oracle["positive_logits"]),
        "positive_js_divergence": js_divergence(candidate["positive_probs"], oracle["positive_probs"]),
        "positive_top20_overlap": len(set(cp) & set(op)),
        "negative_top20_overlap": len(set(cn) & set(on)),
        "oracle_positive_top20_probability_mass_under_candidate": float(np.sum(candidate["positive_probs"][op])),
        "oracle_negative_top20_probability_mass_under_candidate": float(np.sum(candidate["negative_probs"][on])),
    }


def norm_match_blind(vector: np.ndarray, oracle: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    o = np.asarray(oracle, dtype=np.float32)
    vn = float(np.linalg.norm(v.astype(np.float64)))
    on = float(np.linalg.norm(o.astype(np.float64)))
    if not math.isfinite(vn) or not math.isfinite(on) or vn <= 0 or on <= 0:
        raise ValueError("Cannot norm-match zero/nonfinite patch vector")
    scaled = (v.astype(np.float64) * (on / vn)).astype(np.float32)
    if not np.all(np.isfinite(scaled)):
        raise ValueError("Nonfinite norm-matched patch vector")
    return scaled


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


def greedy_completion(model: Any, prompt_ids: Any, vector: Any,
                      tokenizer: Any, torch: Any, new_tokens: int = NEW_TOKENS) -> dict[str, Any]:
    original_final_index = prompt_ids.shape[1] - 1
    ids = prompt_ids.clone()
    generated = []
    for _ in range(new_tokens):
        logits = next_logits(model, ids, original_final_index, vector, +1, torch)
        token_id = int(torch.argmax(logits).item())
        generated.append(token_id)
        ids = torch.cat((ids, torch.tensor([[token_id]], device=ids.device, dtype=ids.dtype)), dim=1)
    return {"token_ids": generated,
            "text": tokenizer.decode(generated, skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)}


def patch_metrics(candidate_delta: np.ndarray, oracle_delta: np.ndarray,
                  candidate_probs: np.ndarray, baseline_probs: np.ndarray) -> dict[str, Any]:
    cp, cn = top_ids(candidate_delta), top_ids(candidate_delta, largest=False)
    op, on = top_ids(oracle_delta), top_ids(oracle_delta, largest=False)
    descending_rank = np.empty(len(candidate_delta), dtype=np.int64)
    ascending_rank = np.empty_like(descending_rank)
    descending_rank[np.argsort(-candidate_delta, kind="stable")] = np.arange(1, len(candidate_delta) + 1)
    ascending_rank[np.argsort(candidate_delta, kind="stable")] = np.arange(1, len(candidate_delta) + 1)
    return {
        "delta_logit_cosine": cosine(candidate_delta, oracle_delta),
        "positive_delta_top20_overlap": len(set(cp) & set(op)),
        "negative_delta_top20_overlap": len(set(cn) & set(on)),
        "oracle_positive_delta_top20_patched_probability_mass": float(np.sum(candidate_probs[op])),
        "oracle_negative_delta_top20_patched_probability_mass": float(np.sum(candidate_probs[on])),
        "oracle_positive_delta_top20_probability_mass_change": float(np.sum(candidate_probs[op] - baseline_probs[op])),
        "oracle_negative_delta_top20_probability_mass_change": float(np.sum(candidate_probs[on] - baseline_probs[on])),
        "oracle_positive_delta_top20_mean_candidate_descending_rank": float(np.mean(descending_rank[op])),
        "oracle_negative_delta_top20_mean_candidate_ascending_rank": float(np.mean(ascending_rank[on])),
    }


def token_delta_records(delta: np.ndarray, patched_logits: np.ndarray,
                        patched_probs: np.ndarray, tokenizer: Any, largest: bool,
                        baseline_logits: np.ndarray | None = None,
                        baseline_probs: np.ndarray | None = None) -> list[dict[str, Any]]:
    records = []
    for token_id in top_ids(delta, largest=largest):
        record = {"token_id": token_id,
                  "token": tokenizer.convert_ids_to_tokens(token_id),
                  "decoded": tokenizer.decode([token_id], skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False),
                  "delta_logit": float(delta[token_id]),
                  "patched_logit": float(patched_logits[token_id]),
                  "patched_probability": float(patched_probs[token_id])}
        if baseline_logits is not None and baseline_probs is not None:
            record["baseline_logit"] = float(baseline_logits[token_id])
            record["baseline_probability"] = float(baseline_probs[token_id])
        records.append(record)
    return records


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def lens_readout(vectors: dict[str, Any], final_norm: Any, lm_head: Any,
                 tokenizer: Any, torch: Any, oracle_reader: Any) -> dict[str, Any]:
    data = {}
    with torch.inference_mode():
        for name in VECTOR_NAMES:
            data[name] = {}
            for pos in POSITIONS:
                distributions = lens_distribution(vectors[name][pos], final_norm, lm_head, torch)
                top_positive = oracle_reader.top_token_records(distributions["positive_probs"], tokenizer, TOP_K, torch)
                top_negative = oracle_reader.top_token_records(distributions["negative_probs"], tokenizer, TOP_K, torch)
                data[name][str(pos)] = {
                    "top20_positive": top_positive, "top20_negative": top_negative,
                    "semantic_hits": semantic_hits(top_positive, top_negative),
                    "full_vocab_hashes": {k: array_hash(v.detach().cpu().numpy()) for k, v in distributions.items()},
                    "_arrays": {k: v.detach().cpu().numpy() for k, v in distributions.items()},
                }
    metrics = {}
    for name in BLIND_NAMES:
        by_position = {}
        for pos in POSITIONS:
            by_position[str(pos)] = lens_metrics(data[name][str(pos)]["_arrays"],
                                                  data["oracle_difference"][str(pos)]["_arrays"])
        metrics[name] = {"position_0": by_position["0"],
                         "positions_1_4": {str(p): by_position[str(p)] for p in PRIMARY_POSITIONS},
                         "mean_positions_1_4": mean_dict([by_position[str(p)] for p in PRIMARY_POSITIONS],
                                                          tuple(by_position["0"]))}
    for name in VECTOR_NAMES:
        for pos in POSITIONS:
            del data[name][str(pos)]["_arrays"]
    semantic_summary = {}
    for name in VECTOR_NAMES:
        semantic_summary[name] = {
            "position_0": data[name]["0"]["semantic_hits"],
            "sum_positions_1_4": {term: {pol: sum(data[name][str(p)]["semantic_hits"][term][pol]
                                                   for p in PRIMARY_POSITIONS)
                                         for pol in ("positive", "negative")}
                                  for term in SEMANTIC_TERMS},
        }
    return {"tokens": data, "oracle_similarity": metrics, "known_semantic_hits": semantic_summary}


def validate_historical_oracle_lens(lens: dict[str, Any], historical: dict[str, Any],
                                    provenance: dict[str, Any]) -> None:
    """Check that the unchanged final norm/head reproduce the pinned oracle lens."""
    for key in ("base", "fine_tuned", "downloaded_model_hashes_sha256",
                "merged_model_hashes_sha256", "oracle_probe_manifest_sha256",
                "oracle_adl_manifest_sha256", "oracle_adl_sha256"):
        if historical["provenance"].get(key) != provenance.get(key):
            raise ValueError(f"Historical Logit Lens provenance mismatch: {key}")
    if len(historical["positions"]) != len(POSITIONS):
        raise ValueError("Historical Logit Lens position inventory changed")
    for pos, record in zip(POSITIONS, historical["positions"]):
        if record["position"] != pos:
            raise ValueError("Historical Logit Lens position order changed")
        for polarity in ("positive", "negative"):
            expected = record["difference"][polarity]
            actual = lens["tokens"]["oracle_difference"][str(pos)]["top20_" + polarity]
            if (len(expected) != TOP_K or len(actual) != TOP_K
                    or [x["token_id"] for x in expected] != [x["token_id"] for x in actual]
                    or any(not math.isclose(x["probability"], y["probability"],
                                            rel_tol=1e-6, abs_tol=1e-8)
                           for x, y in zip(expected, actual))):
                raise ValueError(f"Historical oracle Logit Lens mismatch: {pos}/{polarity}")


def patchscope_readout(vectors: dict[str, Any], model: Any, tokenizer: Any,
                       torch: Any) -> dict[str, Any]:
    device = next(model.parameters()).device
    baseline_records = {}
    outputs = {name: {str(pos): {} for pos in POSITIONS} for name in VECTOR_NAMES}
    metrics = {name: {str(pos): {"plus": [], "minus": []} for pos in POSITIONS} for name in BLIND_NAMES}
    completions = {name: {str(pos): {} for pos in PRIMARY_POSITIONS} for name in VECTOR_NAMES}
    for prompt in PROMPTS:
        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
            raise ValueError("Malformed plain-text target prompt")
        original_final_index = prompt_ids.shape[1] - 1
        baseline = next_logits(model, prompt_ids, None, None, +1, torch).numpy()
        baseline_probs = torch.softmax(torch.from_numpy(baseline), dim=-1).numpy()
        baseline_record = {
            "logits_sha256": array_hash(baseline),
            "probabilities_sha256": array_hash(baseline_probs),
            "top20_next_token_probabilities": [
                {"token_id": token_id,
                 "token": tokenizer.convert_ids_to_tokens(token_id),
                 "decoded": tokenizer.decode([token_id], skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False),
                 "logit": float(baseline[token_id]),
                 "probability": float(baseline_probs[token_id])}
                for token_id in top_ids(baseline_probs)
            ],
        }
        baseline_records[prompt] = baseline_record
        patch_vectors = {}
        for pos in POSITIONS:
            oracle_vec = vectors["oracle_difference"][pos]
            patch_vectors["oracle_difference", pos] = oracle_vec.to(device=device)
            for name in BLIND_NAMES:
                scaled = norm_match_blind(vectors[name][pos].numpy(), oracle_vec.numpy())
                patch_vectors[name, pos] = torch.from_numpy(scaled).to(device=device)
        for pos in POSITIONS:
            condition = {}
            for name in VECTOR_NAMES:
                condition[name] = {}
                vector = patch_vectors[name, pos]
                for sign, label in ((+1, "plus"), (-1, "minus")):
                    patched = next_logits(model, prompt_ids, original_final_index, vector, sign, torch).numpy()
                    probs = torch.softmax(torch.from_numpy(patched), dim=-1).numpy()
                    delta = patched.astype(np.float64) - baseline.astype(np.float64)
                    rec = {"delta_logits_sha256": array_hash(delta),
                           "patched_logits_sha256": array_hash(patched),
                           "patched_probabilities_sha256": array_hash(probs),
                           "top20_positive_delta": token_delta_records(
                               delta, patched, probs, tokenizer, True, baseline, baseline_probs),
                           "top20_negative_delta": token_delta_records(
                               delta, patched, probs, tokenizer, False, baseline, baseline_probs)}
                    rec["semantic_hits"] = semantic_hits(rec["top20_positive_delta"],
                                                         rec["top20_negative_delta"])
                    condition[name][label] = {"record": rec, "delta": delta, "probs": probs}
            for name in VECTOR_NAMES:
                outputs[name][str(pos)][prompt] = {
                    label: condition[name][label]["record"] for label in ("plus", "minus")}
            for name in BLIND_NAMES:
                for label in ("plus", "minus"):
                    candidate = condition[name][label]
                    oracle = condition["oracle_difference"][label]
                    record = patch_metrics(candidate["delta"], oracle["delta"],
                                           candidate["probs"], baseline_probs)
                    metrics[name][str(pos)][label].append({"prompt": prompt, **record})
            if pos in PRIMARY_POSITIONS:
                for name in VECTOR_NAMES:
                    completions[name][str(pos)][prompt] = greedy_completion(
                        model, prompt_ids, patch_vectors[name, pos], tokenizer, torch)
    summaries = {}
    metric_keys = tuple(k for k in metrics[BLIND_NAMES[0]]["0"]["plus"][0] if k != "prompt")
    for name in BLIND_NAMES:
        summaries[name] = {}
        for label in ("plus", "minus"):
            summaries[name][label] = {
                "position_0_mean_prompts": mean_dict(metrics[name]["0"][label], metric_keys),
                "per_position_mean_prompts": {str(pos): mean_dict(metrics[name][str(pos)][label], metric_keys)
                                              for pos in PRIMARY_POSITIONS},
                "mean_positions_1_4_and_prompts": mean_dict(
                    [r for pos in PRIMARY_POSITIONS for r in metrics[name][str(pos)][label]], metric_keys),
            }
    semantic_summary = {}
    for name in VECTOR_NAMES:
        semantic_summary[name] = {}
        for label in ("plus", "minus"):
            semantic_summary[name][label] = {
                "position_0_all_prompts": {
                    term: {pol: sum(outputs[name]["0"][prompt][label]["semantic_hits"][term][pol]
                                   for prompt in PROMPTS)
                           for pol in ("positive", "negative")}
                    for term in SEMANTIC_TERMS},
                "positions_1_4_all_prompts": {
                    term: {pol: sum(outputs[name][str(pos)][prompt][label]["semantic_hits"][term][pol]
                                   for pos in PRIMARY_POSITIONS for prompt in PROMPTS)
                           for pol in ("positive", "negative")}
                    for term in SEMANTIC_TERMS},
            }
    return {"baselines": {prompt: baseline_records[prompt] for prompt in PROMPTS},
            "conditions": outputs, "per_prompt_oracle_similarity": metrics,
            "oracle_similarity": summaries, "greedy_plus_completions": completions,
            "known_semantic_hits": semantic_summary,
            "norm_matching": "evaluation_only_blind_per_position_to_oracle_l2_oracle_native_unchanged"}


def run() -> None:
    spec = load_spec()
    output = PROJECT / spec["output"]
    require_output_absent(output)
    manifests = validate_file_provenance(spec)
    # This pinned import reuses all local model/tokenizer/oracle inventory checks.
    sys.path.insert(0, str(PROJECT / "scripts"))
    import read_oracle_logit_lens as oracle_reader
    import torch

    loc = spec["input_locations"]
    provenance = oracle_reader.verify_inputs_before_loading(
        Path(loc["base_tokenizer_directory"]), Path(loc["merged_checkpoint_directory"]),
        Path(loc["oracle_adl"]))
    vectors = load_frozen_vectors(spec, manifests, torch, oracle_reader, provenance)
    model, tokenizer, final_norm, lm_head = oracle_reader.load_local_model_and_tokenizer(
        Path(loc["base_tokenizer_directory"]), Path(loc["merged_checkpoint_directory"]), provenance, torch)
    if len(model.model.layers) != 28:
        raise ValueError("Merged Qwen3 layer count changed")
    lens = lens_readout(vectors, final_norm, lm_head, tokenizer, torch, oracle_reader)
    historical_lens = json.loads(path_for_source(spec, "", "historical_oracle_logit_lens").read_text())
    validate_historical_oracle_lens(lens, historical_lens, provenance)
    if not torch.cuda.is_available():
        raise ValueError("CUDA required for bounded Patchscope runtime")
    model.to("cuda").eval()
    patchscope = patchscope_readout(vectors, model, tokenizer, torch)
    # Check immutable sources again before publication.
    validate_file_provenance(spec)
    for entry in spec["fixed_vectors"]:
        validate_tensor(vectors[entry["name"]], entry["raw_sha256"], torch,
                        oracle_reader.sha256_raw_float32_tensor)
    result = {
        "format_version": 1, "attempt_id": spec["attempt_id"],
        "purpose": spec["purpose"], "spec_sha256": sha256_file(SPEC_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "same_specimen_exploratory_method_development": True,
        "clean_heldout_validation": False,
        "fixed_vector_inventory": list(VECTOR_NAMES), "positions": list(POSITIONS),
        "target_prompts": list(PROMPTS), "semantic_substrings": list(SEMANTIC_TERMS),
        "input_sha256": spec["immutable_input_sha256"],
        "validated_checkpoint_provenance": {
            key: provenance[key] for key in
            ("base", "fine_tuned", "downloaded_model_hashes_sha256",
             "merged_model_hashes_sha256", "oracle_probe_manifest_sha256",
             "oracle_adl_manifest_sha256", "oracle_adl_sha256")},
        "oracle_reference_is_not_a_candidate": True,
        "no_candidate_selection_combination_or_modification": True,
        "logit_lens": lens, "patchscope": patchscope,
        "logit_lens_oracle_similarity": lens["oracle_similarity"],
        "patchscope_oracle_similarity": patchscope["oracle_similarity"],
        "known_semantic_hits": {"logit_lens": lens["known_semantic_hits"],
                                "patchscope": patchscope["known_semantic_hits"]},
        "qualitative_greedy_completions": patchscope["greedy_plus_completions"],
        "full_vocabulary_arrays": "computed_transiently; only hashes, top20 records, and metrics persisted",
    }
    require_output_absent(output)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    try:
        run()
    except (ValueError, OSError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
