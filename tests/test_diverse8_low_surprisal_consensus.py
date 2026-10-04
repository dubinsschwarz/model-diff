"""Attempt134 synthetic and static checks; no model or real corpus is loaded."""

from __future__ import annotations

import copy
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ablation/run_diverse8_low_surprisal_consensus.py"
SPEC = ROOT / "experiments/attempts/134_diverse8_low_surprisal_consensus/spec.json"
RANKER_SOURCE = ROOT / "scripts/ablation/run_fineweb_surprisal_stratification.py"
ATTEMPT100_SOURCE = ROOT / "scripts/ablation/construct_attempt100_diverse8_consensus.py"
ATTEMPT100_JVP_SOURCE = ROOT / "scripts/ablation/construct_multicorpus_raw_jg_consensus.py"


def load_module(path, name):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


runner = load_module(SOURCE, "attempt134_runner")
ranker = load_module(RANKER_SOURCE, "attempt134_frozen_ranker_for_synthetic_test")


def test_exact_plan_and_eight_frozen_corpus_provenance():
    spec = runner.load_spec(SPEC)
    assert runner.ORDER == (
        "fineweb", "wikitext103_raw", "tinystories", "arxiv_document",
        "cc_news", "pg19", "codeparrot_clean", "ultrachat")
    assert spec["corpus_order"] == list(runner.ORDER)
    assert len(spec["corpora"]) == 8
    assert all(row["serialized_sha256"] and row["raw_tensor_sha256"] for row in spec["corpora"])
    m100, m014, m126 = runner.validate_blind_sources(spec)
    assert m100["corpus_order"] == list(runner.ORDER)
    assert m014["corpus_order"] == list(runner.ORDER[:3])
    assert m126["frozen_attempt005"]["serialized_sha256"] == spec["corpora"][0]["serialized_sha256"]
    assert m126["frozen_attempt005"]["raw_tensor_sha256"] == spec["corpora"][0]["raw_tensor_sha256"]
    assert spec["inputs"]["attempt126"]["fineweb_rank_raw_int64_sha256"] == m126["ranking"]["rank_raw_int64_sha256"]
    assert spec["inputs"]["attempt100"]["merged_mean_raw_sha256"] == m100["artifact"]["raw_tensors_sha256"]["merged_mean"]
    assert m100["probe"]["batch_size"] == 32  # frozen Attempt100 provenance
    assert spec["probe"]["batch_size"] == 64  # only Attempt134's JVP batching changes
    assert all(m100["probe"][key] == spec["probe"][key]
               for key in m100["probe"] if key != "batch_size")


def test_exact_ranking_population_batches_and_same_definition():
    spec = runner.load_spec(SPEC)
    ranking = spec["ranking"]
    assert ranking["population_rows"] == [0, 4096]
    assert ranking["population_size_per_corpus"] == 4096
    assert ranking["sequence_length"] == 128
    assert ranking["prediction_tokens_per_row"] == 127
    assert ranking["batch_size"] == 8
    assert ranking["forward_batches_per_new_corpus"] == 512
    assert ranking["fineweb_reuse_attempt126"] is True
    assert ranking["newly_scored_corpora"] == list(runner.ORDER[1:])
    assert ranking["sort_key"] == ["surprisal_ascending", "original_row_index_ascending"]
    assert ranking["low_rank_range"] == [0, 1024]
    assert ranking["within_corpus_only"] is True
    assert spec["workload"]["new_surprisal_forward_batches"] == 7 * 512
    token_nll = np.zeros((2, 127), dtype=np.float32)
    token_nll[0, :] = 2.0
    token_nll[1, :] = np.arange(127, dtype=np.float32)
    np.testing.assert_array_equal(ranker.sequence_mean_nll(token_nll), [2.0, 63.0])
    with pytest.raises(ValueError):
        ranker.sequence_mean_nll(np.zeros((2, 128), dtype=np.float32))
    scorer_source = inspect.getsource(ranker.score_endpoint_surprisal)
    assert "for start in range(0,4096,8)" in scorer_source
    assert "reduction='none'" in scorer_source
    assert "batch[:,1:]" in scorer_source
    assert "sequence_mean_nll(cpu)" in scorer_source


def test_fineweb_reused_seven_new_rankings_and_independent_low_sets():
    base = np.arange(4096, dtype=np.float64)
    base[:5] = 0  # tied examples retain original row-index order
    fineweb = ranker.rank_and_stratify(base)
    corpora = {name: object() for name in runner.ORDER}
    calls = []

    def score(name, token_object):
        assert token_object is corpora[name]
        calls.append(name)
        # Each corpus receives its own ranking. Nothing is pooled across corpora.
        return np.ascontiguousarray(4096 - base if name == "pg19" else base + len(calls), dtype=np.float64)

    records = runner.build_rankings(corpora, fineweb, score, ranker.rank_and_stratify, ranker)
    assert calls == list(runner.ORDER[1:])  # FineWeb was never scored
    assert records["fineweb"]["source"] == "reused_attempt126"
    assert all(records[name]["source"] == "newly_scored_once" for name in runner.ORDER[1:])
    assert records["fineweb"]["ascending_rank_original_row_indices"][:5] == [0, 1, 2, 3, 4]
    assert records["fineweb"]["low"]["original_row_indices"] != records["pg19"]["low"]["original_row_indices"]
    for name, record in records.items():
        rows = np.asarray(record["low"]["original_row_indices"], dtype=np.int64)
        assert record["low"]["rank_range"] == [0, 1024]
        assert len(rows) == len(set(rows.tolist())) == 1024
        assert record["low"]["ordered_row_indices_raw_sha256"] == ranker.raw_int64_hash(rows)
        assert record["surprisal_raw_float64_sha256"] == ranker.raw_float64_hash(
            np.asarray(record["surprisal_values"], dtype=np.float64))
        assert set(record["population_surprisal"]) == {"min", "max", "mean", "median"}
        assert set(record["low"]["surprisal"]) == {"min", "max", "mean", "median"}
    runner.validate_rankings(records, fineweb, ranker)
    tampered = copy.deepcopy(records)
    tampered["cc_news"]["low"]["ordered_row_indices_raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Published LOW1024 ranking changed"):
        runner.validate_rankings(tampered, fineweb, ranker)


def test_gradient_jvp_and_consensus_match_attempt100_definitions():
    spec = runner.load_spec(SPEC)
    loss = spec["gradient_loss"]
    assert loss["sample_count"] == 1024
    assert loss["sequence_length"] == 128
    assert loss["predictions_per_sample"] == 127
    assert loss["batch_size"] == 8
    assert loss["number_of_batches"] == 128
    assert loss["total_prediction_tokens"] == 1024 * 127
    assert loss["reduction"] == "sum_token_cross_entropy_divided_by_total_prediction_tokens"
    assert all(loss[key] is False for key in ("mixed_precision", "optimizer", "weight_decay", "gradient_clipping"))
    assert spec["eligible_tensors"]["expected_matrix_count"] == 98
    assert spec["tangent"]["direction"] == "delta = +alpha * g"
    assert spec["tangent"]["target_relative_frobenius"] == 0.00125
    assert spec["tangent"]["normalization"] == "one_global_alpha_across_all_eligible_matrices"
    assert spec["tangent"]["norm_accounting_dtype"] == "cpu_float64"
    assert spec["tangent"]["jvp_tangent_dtype"] == "float32"
    assert spec["probe"]["sample_count"] == 10000
    assert spec["probe"]["batch_size"] == 64
    assert spec["probe"]["jvp_batches_per_corpus"] == 157
    assert spec["workload"]["gradient_backward_batches"] == 8 * 128
    assert spec["workload"]["jvp_probe_batches"] == 8 * 157
    sizes = runner.jvp_batch_sizes(10000, 64)
    assert len(sizes) == 157
    assert sizes == [64] * 156 + [16]
    assert sum(sizes) == 10000
    assert spec["readout"]["hidden_state_index"] == 14
    assert spec["jvp"]["strict_forward_ad"] is True
    assert spec["jvp"]["fallback"] is False
    assert spec["candidate"]["tensor_order"] == list(runner.CANDIDATE_NAMES)
    assert runner.CANDIDATE_NAMES == (*runner.RESPONSE_NAMES, "low_surprisal_consensus_8")
    assert spec["consensus"]["weights"] == [0.125] * 8
    assert spec["consensus"]["renormalize_final"] is False
    source = SOURCE.read_text()
    assert "a100.old.accumulate_mean_generic_gradient(" in source
    assert "a100.old.global_tangent_scale(eligible, 0.00125, torch)" in source
    assert "a100.old.prepare_tangents(" in source
    assert "a100.old.compute_probe_response(" in source
    constructor = inspect.getsource(runner.construct_responses)
    assert "current_probe_primal_hash = primal_hash" in constructor
    assert "primal_hash != current_probe_primal_hash" in constructor
    assert 'spec["inputs"]["attempt100"]["merged_mean_raw_sha256"]' not in constructor
    assert "a100.response_geometry(list(responses.values()))" in source
    old_source = ATTEMPT100_SOURCE.read_text()
    assert "raw / norms.unsqueeze(-1)" in old_source
    assert "consensus64 = unit.mean(dim=0)" in old_source
    assert "return consensus64.float().contiguous(), diagnostics" in old_source
    jvp_source = ATTEMPT100_JVP_SOURCE.read_text()
    assert 'for start in range(0, settings["sample_count"], settings["batch_size"]):' in jvp_source
    assert 'probe[start:start + settings["batch_size"]]' in jvp_source
    assert 'self.primal_sum = torch_module.zeros(shape, dtype=torch_module.float64, device="cpu")' in jvp_source
    assert 'self.response_sum = torch_module.zeros_like(self.primal_sum)' in jvp_source
    assert 'accumulator.means(settings["sample_count"])' in jvp_source
    # The fixed rule has no final renormalization: two opposed unit responses cancel.
    toy = np.stack([np.array([1.0, 0.0])] * 7 + [np.array([-1.0, 0.0])])
    unit = toy / np.linalg.norm(toy, axis=1, keepdims=True)
    np.testing.assert_array_equal(unit.mean(axis=0), [0.75, 0.0])


def test_read_only_memory_smoke_uses_one_synthetic_strict_jvp():
    from types import SimpleNamespace

    class FakeTensor:
        def __init__(self, shape, value=None, finite=True):
            self.shape, self.value, self.finite = shape, value, finite
            self.dtype = "float32"
            self.device = SimpleNamespace(type="cpu")

        def contiguous(self):
            return self

        def __getitem__(self, rows):
            assert rows == slice(0, 64)
            return self

        def to(self, _device):
            return self

    class FakeFinite:
        def __init__(self, finite):
            self.finite = finite

        def all(self):
            return self

        def item(self):
            return self.finite

    weights = [SimpleNamespace(weight=FakeTensor((1, 1))) for _ in range(98)]
    model = SimpleNamespace(weights=weights,
                            parameters=lambda: iter([weights[0].weight]))
    probe = FakeTensor((64, 128))
    calls = []

    class Old:
        @staticmethod
        def discover_eligible_linear_weights(model, _torch):
            return [(f"model.layers.{i // 7}.test{i}", layer)
                    for i, layer in enumerate(model.weights)]

        @staticmethod
        def freeze_other_parameters(_model, eligible):
            assert len(eligible) == 98

        @staticmethod
        def run_readout_jvp(_model, tokens, tangents, spec, _torch):
            calls.append(1)
            assert tokens.shape == (64, 128)
            assert len(tangents) == 98
            assert all(tensor.dtype == "float32" and tensor.value == 1e-7
                       for tensor in tangents.values())
            output = FakeTensor((64, 128, spec["readout"]["hidden_size"]))
            return output, output

    fake_torch = SimpleNamespace(full_like=lambda value, fill: FakeTensor(value.shape, value=fill),
                                 isfinite=lambda value: FakeFinite(value.finite),
                                 cuda=SimpleNamespace(synchronize=lambda: None))
    elapsed = runner.smoke_one_strict_jvp(model, probe, runner.load_spec(SPEC),
                                          SimpleNamespace(old=Old), fake_torch)
    assert elapsed >= 0
    assert calls == [1]
    source = inspect.getsource(runner.jvp_memory_smoke)
    assert "load_spec()" in source
    assert "checkpoint_file_records(model_dir)" in source
    assert "a100.old.load_probe(" in source
    assert "sys.dont_write_bytecode = True" in source
    assert "smoke_one_strict_jvp(model, probe, spec, a100, torch)" in source
    assert "torch.cuda.max_memory_allocated()" in source
    assert "out of memory" in source and "no fallback" in source
    assert "accumulate_mean_generic_gradient" not in source
    assert "evaluate_after_barrier" not in source
    assert "open(" not in source and "torch.save" not in source
    smoke_spec = runner.load_spec(SPEC)["jvp_memory_smoke"]
    assert smoke_spec["probe_rows"] == [0, 64]
    assert smoke_spec["jvp_calls"] == 1
    assert smoke_spec["read_only"] is True
    assert smoke_spec["historical_access"] is False
    assert smoke_spec["automatic_batch32_fallback"] is False


def test_barrier_before_historical_access_and_no_old_score_selection():
    source = inspect.getsource(runner.run)
    assert source.index("build_rankings(") < source.index("construct_responses(")
    assert source.index("construct_responses(") < source.index("response_geometry(")
    assert source.index("with candidate_path.open(\"xb\")") < source.index("with manifest_path.open(\"x\"")
    assert source.index("validate_published(") < source.index("print(MARKER, flush=True)")
    assert source.index("load_blind_artifacts(spec, a100, torch)", source.index("validate_published(")) < source.index("print(MARKER, flush=True)")
    assert source.index("print(MARKER, flush=True)") < source.index("evaluate_after_barrier(")
    blind = inspect.getsource(runner.validate_blind_sources)
    assert 'inputs["oracle"]' not in blind
    assert 'inputs["post_barrier_context"]' not in blind
    assert '"evaluation_path"' not in blind
    assert '"response_artifact_path"' not in blind
    assert runner.MARKER == "DIVERSE8_LOW_SURPRISAL_CONSENSUS_FROZEN"
    assert runner.load_spec(SPEC)["barrier"]["manifest_contains_historical_scores"] is False
    historical = inspect.getsource(runner.evaluate_after_barrier)
    assert "evaluator.prior.validate_oracle_provenance(" in historical
    assert "evaluator.prior.load_and_validate_oracle(" in historical
    assert 'for attempt in ("attempt131", "attempt132", "attempt133")' in historical
    assert "classify_complementarity(dp, ds)" in inspect.getsource(runner.comparisons)


def test_signed_metric_thresholds_and_development_comparisons():
    assert runner.classify_complementarity(0.01, 0.01) == "meaningful_complementarity_gain"
    assert runner.classify_complementarity(0.009, 0.2) == "positive_complementarity_gain"
    assert runner.classify_complementarity(0.03, 0.0) == "no_complementarity_gain"
    assert runner.classify_per_corpus(6) == "broad_within_corpus_support"
    assert runner.classify_per_corpus(8) == "broad_within_corpus_support"
    assert runner.classify_per_corpus(5) == "mixed_within_corpus_support"
    assert runner.classify_per_corpus(4) == "mixed_within_corpus_support"
    assert runner.classify_per_corpus(3) == "weak_within_corpus_support"
    with pytest.raises(ValueError):
        runner.classify_per_corpus(9)

    def report(value):
        return {"positions_1_4_mean_cosine": value,
                "positions_1_127_mean_cosine": value,
                "all_128_position_cosines": [value] * 128}

    new = {runner.CONSENSUS_NAME: report(0.5),
           **{"response_low_" + name: report(0.5) for name in runner.ORDER}}
    old = {"consensus_response_8": report(0.48),
           **{"response_" + name: report(0.49) for name in runner.ORDER}}
    result = runner.comparisons(new, old)
    assert result["main"]["classification"] == "meaningful_complementarity_gain"
    assert result["main"]["positions_1_127_new_consensus_wins"] == 127
    assert result["corpora_with_both_primary_and_secondary_improvement"] == 8
    assert result["per_corpus_classification"] == "broad_within_corpus_support"


def test_output_overwrite_refusal(tmp_path):
    spec = copy.deepcopy(runner.load_spec(SPEC))
    spec["outputs"] = {"candidate_path": str(tmp_path / "candidate.pt"),
                       "construction_manifest_path": str(tmp_path / "construction-manifest.json"),
                       "result_path": str(tmp_path / "result.json")}
    runner.require_outputs_absent(spec)
    for key in ("candidate_path", "construction_manifest_path", "result_path"):
        path = Path(spec["outputs"][key])
        path.write_bytes(b"existing")
        with pytest.raises(ValueError, match="already exists"):
            runner.require_outputs_absent(spec)
        path.unlink()
