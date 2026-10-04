"""Synthetic and static checks; never load the real Qwen3 or candidate tensors."""

import hashlib
import importlib.util
import json
import copy
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ablation/run_logit_lens_patchscope_characterization.py"
SPEC = ROOT / "experiments/attempts/133_logit_lens_patchscope_characterization/spec.json"
module_spec = importlib.util.spec_from_file_location("attempt133_runner", SOURCE)
runner = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(runner)


def test_fixed_plan_and_pinned_provenance():
    spec = runner.load_spec(SPEC)
    assert runner.SPEC_SHA256 == hashlib.sha256(SPEC.read_bytes()).hexdigest()
    assert tuple(x["name"] for x in spec["fixed_vectors"]) == runner.VECTOR_NAMES
    assert runner.BLIND_NAMES == (
        "attempt132_super_low", "attempt132_next_super_low",
        "attempt131_very_low", "attempt100_consensus")
    assert spec["fixed_vectors"][-1]["name"] == "oracle_difference"
    assert spec["fixed_vectors"][-1]["reference_only"] is True
    assert spec["positions"] == [0, 1, 2, 3, 4]
    assert spec["primary_positions"] == [1, 2, 3, 4]
    assert spec["top_k"] == 20
    assert spec["target_prompts"] == [
        "The topic is", "This is about", "The central idea is", "In one word:",
        "A concise label:", "The concept:", "This relates to", "The underlying theme:"]
    assert spec["semantic_substrings"] == ["cake", "bake", "cook", "craft", "precision"]
    assert spec["no_candidate_selection_combination_or_modification"] is True
    assert spec["result_overwrite_refusal"] is True
    assert spec["patchscope"]["greedy_completion_tokens"] == 6
    assert spec["patchscope"]["blind_vector_evaluation_only_scale"] == "per_position_l2_norm_match_to_oracle_difference"
    assert spec["patchscope"]["oracle_vector_scale"] == "native_unchanged"
    for record in spec["immutable_input_sha256"].values():
        values = record.values() if isinstance(record, dict) else [record]
        assert all(len(v) == 64 and set(v) <= set("0123456789abcdef") for v in values)
    # Frozen source pins are explicit, including the committed Attempt132 result.
    assert spec["immutable_input_sha256"]["attempt132"]["result"] == "5883ce5a1b2a203ca5231bf61f14c3abfc9176c48a54fecbe23807a6ad0426da"
    assert spec["immutable_input_sha256"]["historical_oracle_logit_lens"] == "a723017455e889ee05819382e271d270b0bdf08ec6fa9813f122b8aabdcbeb35"


def test_historical_lens_formula_and_top20_metrics():
    spec = runner.load_spec(SPEC)
    assert spec["logit_lens"]["positive"] == "softmax(lm_head(model.model.norm(vector)))"
    assert spec["logit_lens"]["negative"] == "softmax(lm_head(-model.model.norm(vector)))"

    class FakeTorch:
        @staticmethod
        def softmax(x, dim):
            assert dim == -1
            e = np.exp(x - np.max(x))
            return e / e.sum()

    vector = np.array([1.0, 2.0, 4.0])
    final_norm = lambda x: x + 3.0
    lm_head = lambda x: np.array([x[0], x[1] + 1, x[2] + 2, -x[0]])
    out = runner.lens_distribution(vector, final_norm, lm_head, FakeTorch)
    np.testing.assert_allclose(out["positive_logits"], lm_head(final_norm(vector)))
    np.testing.assert_allclose(out["negative_logits"], lm_head(-final_norm(vector)))
    assert not np.array_equal(out["negative_logits"], lm_head(final_norm(-vector)))

    p = np.linspace(1, 30, 30, dtype=np.float64)
    p /= p.sum()
    q = p[::-1].copy()
    a = {"positive_logits": p, "negative_logits": -p,
         "positive_probs": p, "negative_probs": q}
    m = runner.lens_metrics(a, a)
    assert m["positive_logit_cosine"] == pytest.approx(1)
    assert m["positive_js_divergence"] == pytest.approx(0)
    assert m["positive_top20_overlap"] == 20
    assert m["negative_top20_overlap"] == 20
    assert m["oracle_positive_top20_probability_mass_under_candidate"] == pytest.approx(
        sum(p[runner.top_ids(p)]))
    assert runner.js_divergence(p, q) > 0
    assert runner.top_ids(np.ones(30)) == list(range(20))


def test_provenance_accepts_historical_latent_and_rejects_changed_formula(monkeypatch, tmp_path):
    spec = runner.load_spec(SPEC)
    pins = spec["immutable_input_sha256"]

    def fake_path(_spec, source, part):
        return tmp_path / f"{source or 'root'}_{part}.json"

    def write(source, part, value):
        fake_path(spec, source, part).write_text(json.dumps(value))

    # All inputs are synthetic; the formula validator is exercised through the
    # complete provenance function without opening any frozen model/artifact.
    monkeypatch.setattr(runner, "path_for_source", fake_path)
    monkeypatch.setattr(runner, "require_hash", lambda _path, _digest: None)
    for attempt in ("attempt132", "attempt131", "attempt100"):
        manifest = {"spec_sha256": pins[attempt]["spec"],
                    "constructor_sha256": pins[attempt]["source"]}
        if attempt == "attempt100":
            manifest["artifact"] = {"serialized_sha256": pins[attempt]["responses_artifact"]}
        else:
            manifest["candidate"] = {"serialized_sha256": pins[attempt]["candidate"]}
            manifest["barrier"] = {"published_before_historical_access": True,
                                   "manifest_contains_historical_scores": False}
            write(attempt, "result", {
                "construction_manifest_sha256": pins[attempt]["construction_manifest"],
                "candidate_serialized_sha256": pins[attempt]["candidate"]})
        write(attempt, "construction_manifest", manifest)
    write("", "oracle_adl_manifest", {
        "oracle_adl_sha256": pins["oracle_adl_artifact"],
        "raw_tensors_sha256": {"difference": spec["fixed_vectors"][-1]["raw_sha256"]}})
    historical = {"method": {
        "positions": list(runner.POSITIONS), "top_k": runner.TOP_K,
        "positive": "softmax(lm_head(model.model.norm(latent)))",
        "negative": "softmax(lm_head(-model.model.norm(latent)))"}}
    write("", "historical_oracle_logit_lens", historical)
    assert set(runner.validate_file_provenance(spec)) == {"attempt132", "attempt131", "attempt100"}

    historical["method"]["positive"] = "softmax(lm_head(model.model.norm(latent + 1)))"
    write("", "historical_oracle_logit_lens", historical)
    with pytest.raises(ValueError, match="Historical Logit Lens method changed"):
        runner.validate_file_provenance(spec)

    historical["method"]["positive"] = "softmax(lm_head(model.model.norm(latent)))"
    write("", "historical_oracle_logit_lens", historical)
    altered_spec = copy.deepcopy(spec)
    altered_spec["logit_lens"]["negative"] = "softmax(lm_head(-vector))"
    with pytest.raises(ValueError, match="Historical Logit Lens method changed"):
        runner.validate_file_provenance(altered_spec)


def test_fixed_semantic_audit():
    positives = [{"token": "ĠCake", "decoded": " cake"},
                 {"token": "precision", "decoded": "precision"}]
    negatives = [{"token": "bake", "decoded": " bake"},
                 {"token": "craft", "decoded": " craft"},
                 {"token": "cook", "decoded": " cook"}]
    hits = runner.semantic_hits(positives, negatives)
    assert hits == {
        "cake": {"positive": 1, "negative": 0},
        "bake": {"positive": 0, "negative": 1},
        "cook": {"positive": 0, "negative": 1},
        "craft": {"positive": 0, "negative": 1},
        "precision": {"positive": 1, "negative": 0},
    }


def test_historical_oracle_lens_reproduction_check():
    tokens = {"oracle_difference": {}}
    historical = {"provenance": {"base": {"revision": "x"}}, "positions": []}
    for pos in runner.POSITIONS:
        recs = [{"token_id": i, "probability": i / 1000} for i in range(20)]
        tokens["oracle_difference"][str(pos)] = {
            "top20_positive": [x.copy() for x in recs],
            "top20_negative": [x.copy() for x in recs]}
        historical["positions"].append({"position": pos,
                                         "difference": {"positive": recs, "negative": recs}})
    provenance = {"base": {"revision": "x"}}
    runner.validate_historical_oracle_lens({"tokens": tokens}, historical, provenance)
    historical["positions"][2]["difference"]["negative"][0] = {"token_id": 21, "probability": 0}
    with pytest.raises(ValueError, match="Historical oracle Logit Lens mismatch"):
        runner.validate_historical_oracle_lens({"tokens": tokens}, historical, provenance)


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array).copy()
        self.device = "cpu"
        self.dtype = self.array.dtype

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    def clone(self):
        return FakeTensor(self.array)

    def numel(self):
        return self.array.size

    def to(self, **_kwargs):
        return self.array

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value


def test_additive_block13_final_token_patch_plus_minus_and_native_oracle():
    source = SOURCE.read_text()
    assert "model.model.layers[LAYER_INDEX].register_forward_hook" in source
    assert runner.LAYER_INDEX == 13
    assert runner.load_spec(SPEC)["patchscope"]["readout"] == "hidden_states[14]"
    hidden = FakeTensor(np.arange(24, dtype=np.float32).reshape(1, 3, 8))
    vector = FakeTensor(np.ones(8, dtype=np.float32))
    plus, extra = runner.patch_hidden_output((hidden, "preserve"), 2, vector, +1)
    minus = runner.patch_hidden_output(hidden, 2, vector, -1)
    assert extra == "preserve"
    np.testing.assert_array_equal(plus.array[:, :2], hidden.array[:, :2])
    np.testing.assert_array_equal(minus.array[:, :2], hidden.array[:, :2])
    np.testing.assert_array_equal(plus.array[:, 2], hidden.array[:, 2] + 1)
    np.testing.assert_array_equal(minus.array[:, 2], hidden.array[:, 2] - 1)
    np.testing.assert_array_equal(hidden.array, np.arange(24, dtype=np.float32).reshape(1, 3, 8))
    with pytest.raises(ValueError):
        runner.patch_hidden_output(hidden, 2, vector, 0)

    blind = np.array([3, 4], dtype=np.float32)
    oracle = np.array([0, 10], dtype=np.float32)
    scaled = runner.norm_match_blind(blind, oracle)
    assert np.linalg.norm(scaled) == pytest.approx(np.linalg.norm(oracle))
    np.testing.assert_array_equal(blind, [3, 4])
    np.testing.assert_array_equal(oracle, [0, 10])
    assert "patch_vectors[\"oracle_difference\", pos] = oracle_vec.to(device=device)" in source


def test_patch_metrics_signed_delta_logits_and_oracle_rank_mass():
    baseline = np.ones(30) / 30
    patched = baseline.copy()
    delta = np.linspace(-1, 1, 30)
    record = runner.patch_metrics(delta, delta, patched, baseline)
    assert record["delta_logit_cosine"] == pytest.approx(1)
    assert record["positive_delta_top20_overlap"] == 20
    assert record["negative_delta_top20_overlap"] == 20
    assert record["oracle_positive_delta_top20_patched_probability_mass"] == pytest.approx(20 / 30)
    assert record["oracle_positive_delta_top20_probability_mass_change"] == pytest.approx(0)
    assert record["oracle_positive_delta_top20_mean_candidate_descending_rank"] == pytest.approx(10.5)
    assert record["oracle_negative_delta_top20_mean_candidate_ascending_rank"] == pytest.approx(10.5)
    assert "delta = patched.astype(np.float64) - baseline.astype(np.float64)" in SOURCE.read_text()


def test_greedy_exactly_six_tokens_original_prompt_patch_index(monkeypatch):
    called = []

    class IDs:
        def __init__(self, array):
            self.array = np.asarray(array)
            self.shape = self.array.shape
            self.device = "cpu"
            self.dtype = self.array.dtype

        def clone(self):
            return IDs(self.array)

    class Scalar:
        def item(self):
            return 2

    class FakeTorch:
        @staticmethod
        def argmax(_logits):
            return Scalar()

        @staticmethod
        def tensor(array, **_kwargs):
            return IDs(array)

        @staticmethod
        def cat(parts, dim):
            assert dim == 1
            return IDs(np.concatenate([part.array for part in parts], axis=dim))

    class Tokenizer:
        @staticmethod
        def decode(ids, **kwargs):
            assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
            return "six tokens"

    def fake_next(_model, ids, patch_index, _vector, sign, _torch):
        called.append((ids.shape[1], patch_index, sign))
        return np.array([0, 1, 2])

    monkeypatch.setattr(runner, "next_logits", fake_next)
    result = runner.greedy_completion(None, IDs([[9, 8, 7]]), object(), Tokenizer(), FakeTorch)
    assert result == {"token_ids": [2] * 6, "text": "six tokens"}
    assert called == [(3 + i, 2, +1) for i in range(6)]


def test_provenance_tampering_and_overwrite_refusal(tmp_path):
    path = tmp_path / "input"
    path.write_bytes(b"frozen")
    digest = hashlib.sha256(b"frozen").hexdigest()
    runner.require_hash(path, digest)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runner.require_hash(path, digest)
    altered_spec = tmp_path / "spec.json"
    altered_spec.write_text(SPEC.read_text().replace("cake", "pastry", 1))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        runner.load_spec(altered_spec)
    out = tmp_path / "result.json"
    runner.require_output_absent(out)
    out.write_text("{}")
    with pytest.raises(ValueError, match="already exists"):
        runner.require_output_absent(out)


def test_no_selection_or_recovery_construction():
    source = SOURCE.read_text()
    assert "torch.argmax(logits)" in source  # greedy token choice only
    assert "optimizer" not in source
    assert "jvp" not in source.lower()
    assert "torch.save" not in source
    assert "with output.open(\"x\"" in source
