"""Attempt204 synthetic/static checks only; no real model/data experiment."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import statistics

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "scripts/ablation/run_hard_self_distill_cake_bake_behavior_rollback.py"
loader = importlib.util.spec_from_file_location("test204", SOURCE)
r = importlib.util.module_from_spec(loader)
loader.loader.exec_module(r)
SPEC = r.load_spec()
a = r.import_pinned(SPEC["attempt203"]["runner"], "test204_likelihood203")
c = r.import_pinned(SPEC["attempt201"]["constructor"], "test204_training201")
S201 = c.load_spec()
gradient = c.import_pinned(S201["blind_helpers"]["gradient"], "test204_eligible")


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_exact_committed_pins_and_fixed_policy():
    assert r.sha256_file(r.SPEC_PATH) == r.SPEC_SHA256
    for group in ("attempt203", "attempt201"):
        for key, record in SPEC[group].items():
            if isinstance(record, dict) and "path" in record and key not in ("candidate", "blind_data"):
                assert r.sha256_file(r.path_of(record["path"])) == record["sha256"]
    manifest = r.read_record(SPEC["attempt201"]["construction_manifest"])
    assert SPEC["attempt201"]["blind_data"]["sha256"] == manifest["blind_data"]["serialized_sha256"]
    assert SPEC["attempt201"]["candidate"]["sha256"] == manifest["candidate"]["serialized_sha256"]
    assert SPEC["replay"]["seed_ids"] == list(range(8))
    assert SPEC["replay"]["accepted_steps_per_seed"] == 4
    assert SPEC["replay"]["eligible_blocks"] == list(range(14))
    assert SPEC["replay"]["eligible_matrix_count"] == 98
    metadata = SPEC["scientific_metadata"]
    assert metadata["privileged_post_hoc_diagnostic"] and not metadata["new_blind_recovery_method"]
    assert "H^2-weighted" in metadata["alpha_interpretation"]
    assert "Equal weight" in metadata["median_fractional_rollback_interpretation"]
    assert "frozen-butter" in metadata["median_fractional_rollback_interpretation"]
    assert metadata["fact_count_interpretation"] == "Breadth, not magnitude"
    assert "not general capability regression" in metadata["scope"]
    assert metadata["categorical_support_threshold"] is None
    assert not metadata["seed_selection"] and not metadata["fact_filtering"]


def test_frozen_203_validates_all_seven_exact_facts_margins_and_H():
    s203, historical = r.validate_203(SPEC, a)
    assert historical["prefix"] == s203["prefix"]
    assert historical["facts"] == s203["facts"] and len(historical["facts"]) == 7
    assert historical["aggregates"]["fact_count"] == 7
    assert historical["aggregates"]["count_H_gt_0"] == 7
    assert historical["aggregates"]["count_H_lt_0"] == 0
    h = historical["aggregates"]["H"]
    assert h == [pair["H"] for pair in historical["pairs"]]
    assert h == [pair["margin_F"] - pair["margin_B"] for pair in historical["pairs"]]
    assert r.read_record(SPEC["attempt203"]["result"])["aggregates"]["H"] == h


def write_record(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return {"path": str(path), "sha256": r.sha256_file(path)}


@pytest.mark.parametrize("fault", ["H", "margin", "count_positive", "count_negative", "fact_count",
                                   "drop_fact", "reorder", "prefix", "tokens", "raw_mean", "provenance"])
def test_corrupt_203_fails_even_if_new_result_hash_matches(tmp_path, fault):
    spec = copy.deepcopy(SPEC)
    historical = r.read_record(SPEC["attempt203"]["result"])
    if fault == "H":
        historical["aggregates"]["H"][0] *= 2
    elif fault == "margin":
        historical["pairs"][0]["margin_F"] += 1
    elif fault == "count_positive":
        historical["aggregates"]["count_H_gt_0"] = 6
    elif fault == "count_negative":
        historical["aggregates"]["count_H_lt_0"] = 1
    elif fault == "fact_count":
        historical["aggregates"]["fact_count"] = 6
    elif fault == "drop_fact":
        historical["facts"].pop()
    elif fault == "reorder":
        historical["tokenization"].reverse()
    elif fault == "prefix":
        historical["prefix"] += " "
    elif fault == "tokens":
        historical["tokenization"][0]["false"]["full_token_ids"][0] += 1
    elif fault == "raw_mean":
        historical["scores"]["F"][0]["false"]["mean_log_probability_per_token"] += 1
    else:
        historical["provenance"]["runner_sha256"] = "0" * 64
    spec["attempt203"]["result"] = write_record(tmp_path / "result203.json", historical)
    with pytest.raises(ValueError):
        r.validate_203(spec, a)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for index in range(7):
            linear = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                linear.weight.fill_(.1)
            self.add_module(f"proj{index}", linear)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block() for _ in range(28)])
        self.offset = torch.nn.Parameter(torch.tensor(.1))
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=28, vocab_size=3, use_cache=False)

    def forward(self, input_ids, use_cache):
        assert not self.training and not use_cache
        values = [linear.weight[0, 0] for block in self.model.layers[:14] for linear in block.children()]
        value = torch.stack(values).sum() / 100 + self.offset
        logits = torch.stack((value, -value, value * 0))
        return SimpleNamespace(logits=logits.expand(input_ids.shape[0], input_ids.shape[1], 3))


@pytest.fixture(scope="module")
def trajectory():
    """Freeze toy trajectories with the original pinned four-step trainer."""
    data = {"contexts": torch.ones((64, 127), dtype=torch.int64),
            "targets": torch.stack([torch.full((64,), seed % 3, dtype=torch.int64) for seed in range(8)])}
    model = ToyModel().eval()
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    initial = c.b.state_hashes(eligible)
    frozen = gradient.frozen_parameter_hashes(model, selected, torch)
    originals = {name: module.weight.detach().clone() for name, module in eligible}
    wnorm = c.b.norm(originals.values())
    manifest = {"initial_eligible_state": initial, "initial_Wnorm": wnorm,
                "runtime": {"torch": str(torch.__version__), "device": "cpu-synthetic"},
                "noneligible_F_parameter_sha256": frozen, "seeds": {}}
    for seed in range(8):
        model = ToyModel().eval()
        eligible = gradient.discover_eligible_linear_weights(model, torch)
        selected = gradient.freeze_other_parameters(model, eligible)
        c.verify_seed_start(model, eligible, selected, initial, frozen, gradient)
        steps = c.train_seed(model, eligible, selected, originals, wnorm, frozen,
                             data["contexts"], data["targets"][seed].contiguous(), seed, gradient)
        manifest["seeds"][str(seed)] = {"initial_eligible_state": initial, "steps": steps,
            "final_eligible_state": c.b.state_hashes(eligible), "noneligible_unchanged": True,
            "descendant_mean_float64_sha256": "a" * 64}
    c.validate_trajectories(manifest)
    c.validate_recorded_state_hashes(manifest)
    return data, manifest


def test_exact_four_step_replay_all_eight_fresh_F_starts_and_noneligible(trajectory):
    data, manifest = trajectory
    for seed in range(8):
        model = ToyModel().eval()
        eligible, selected, replay = r.replay_seed(model, seed, data, manifest, c, gradient)
        assert replay["accepted_steps"] == 4 and replay["exact_original_F_start"]
        assert replay["every_step_record_exact_match"]
        assert replay["steps"] == manifest["seeds"][str(seed)]["steps"]
        assert replay["final_eligible_state"] == manifest["seeds"][str(seed)]["final_eligible_state"]
        assert len(eligible) == 98
        assert {int(name.split(".")[2]) for name, _ in eligible} == set(range(14))
        assert gradient.frozen_parameter_hashes(model, selected, torch) == manifest["noneligible_F_parameter_sha256"]


@pytest.mark.parametrize("fault", ["intermediate", "final", "step_count", "initial_eligible", "initial_noneligible",
                                   "loss_record", "eta_record", "noneligible_during_step"])
def test_replay_mismatches_abort_before_behavior(trajectory, monkeypatch, fault):
    data, original = trajectory
    manifest = copy.deepcopy(original)
    model = ToyModel().eval()
    record = manifest["seeds"]["0"]
    if fault == "intermediate":
        record["steps"][1]["eligible_before"]["aggregate_sha256"] = "0" * 64
    elif fault == "final":
        record["final_eligible_state"] = copy.deepcopy(record["final_eligible_state"])
        record["final_eligible_state"]["aggregate_sha256"] = "0" * 64
    elif fault == "step_count":
        record["steps"].pop()
    elif fault == "initial_eligible":
        with torch.no_grad():
            model.model.layers[0].proj0.weight.add_(.1)
    elif fault == "initial_noneligible":
        with torch.no_grad():
            model.offset.add_(.1)
    elif fault == "loss_record":
        record["steps"][0]["initial_ce"] += .1
    elif fault == "eta_record":
        record["steps"][0]["accepted_eta"] *= .5
    else:
        original_armijo = c.b.armijo_step
        def corrupt(*args):
            accepted = original_armijo(*args)
            with torch.no_grad():
                model.offset.add_(.1)
            return accepted
        monkeypatch.setattr(c.b, "armijo_step", corrupt)
    with pytest.raises(ValueError):
        r.replay_seed(model, 0, data, manifest, c, gradient)


def test_rejected_armijo_trial_restores_exact_pretrial_state():
    linear = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        linear.weight.fill_(2.)
    linear.weight.grad = torch.ones_like(linear.weight)
    original = linear.weight.detach().clone()
    seen = []
    diagnostics = {"eta_max": 1., "initial_ce": 2., "gradient_norm": 1., "Wnorm": 2.}
    def loss():
        seen.append(linear.weight.detach().clone())
        return 3. if len(seen) == 1 else 1.
    record = c.b.armijo_step([("toy", linear)], {"toy": original}, diagnostics, loss)
    assert torch.equal(seen[0], original - 1.)
    assert torch.equal(seen[1], original - .5)  # Not an accumulated 1.5 update.
    assert record["accepted_updates"] == 1 and record["accepted_eta"] == .5


def test_D_projection_cosine_fractional_breadth_and_zero_D():
    h = [1., 2., 3., 4., 5., 6., 7.]
    d = [2. * x for x in h]
    metrics = r.vector_metrics(d, h)
    assert metrics["D"] == d and metrics["alpha"] == 2.
    assert metrics["cosine"] == pytest.approx(1.)
    assert metrics["fractional_rollback"] == [2.] * 7
    assert metrics["median_fractional_rollback"] == metrics["mean_fractional_rollback"] == 2.
    assert metrics["baseward_fact_count"] == 7
    zero = r.vector_metrics([0.] * 7, h)
    assert zero["alpha"] == 0. and zero["cosine"] is None
    assert zero["baseward_fact_count"] == 0
    opposite = r.vector_metrics([-x for x in h], h)
    assert opposite["alpha"] == -1. and opposite["cosine"] == pytest.approx(-1.)
    assert opposite["fractional_rollback"] == [-1.] * 7  # Never clipped.


def test_equal_fact_median_differs_from_H_squared_weighted_alpha():
    h, d = [1., 100., 1., 1., 1., 1., 1.], [1., 0., 1., 1., 1., 1., 1.]
    metrics = r.vector_metrics(d, h)
    assert metrics["median_fractional_rollback"] == 1.
    assert metrics["mean_fractional_rollback"] == 6 / 7
    assert metrics["alpha"] == 6 / 10006
    assert metrics["baseward_fact_count"] == 6


def test_seed_behavior_uses_frozen_F_and_H_without_redefinition():
    historical = {"pairs": [{"margin_F": 3. + k, "margin_B": 2. + k} for k in range(7)],
                  "aggregates": {"H": [1.] * 7}}
    scores = [{"index": k, "false": {"mean_log_probability_per_token": -5.},
               "true": {"mean_log_probability_per_token": -7. - k}} for k in range(7)]
    before = copy.deepcopy(historical)
    metrics = r.seed_behavior(scores, historical)
    assert metrics["margin_G"] == [2. + k for k in range(7)]
    assert metrics["D"] == [1.] * 7 and metrics["alpha"] == 1.
    assert historical == before
    with pytest.raises(ValueError):
        r.seed_behavior(scores[:-1], historical)
    with pytest.raises(ValueError):
        r.seed_behavior(list(reversed(scores)), historical)


def metric_seeds(vectors, h):
    return {str(seed): {"behavior": r.vector_metrics(vector, h)} for seed, vector in enumerate(vectors)}


def test_fixed_eight_seed_56_entry_distributions_per_fact_and_mean_vector():
    h = [1., 2., 3., 4., 5., 6., 7.]
    scales = [-2., -1., 0., 1., 2., 3., 4., 5.]
    vectors = [[scale * x for x in h] for scale in scales]
    result = r.summarize_seeds(metric_seeds(vectors, h), h)
    assert result["seed_count"] == 8 and result["fact_count"] == 7 and result["seed_fact_entries"] == 56
    expected = {"mean": 1.5, "median": 1.5, "min": -2., "max": 5., "count_gt_0": 5}
    assert result["alpha"] == expected and result["median_fractional_rollback"] == expected
    assert result["cosine"]["undefined_count"] == 1 and result["cosine"]["count_gt_0"] == 5
    assert result["cosine"]["mean"] == pytest.approx(3 / 7)
    assert result["baseward_fact_count_by_seed"] == [0, 0, 0, 7, 7, 7, 7, 7]
    assert result["baseward_fact_count_distribution"] == {str(k): (3 if k == 0 else 5 if k == 7 else 0) for k in range(8)}
    assert result["total_positive_D_entries"] == 35
    for k, fact in enumerate(result["per_fact"]):
        assert fact["H"] == h[k] and fact["D"] == [vector[k] for vector in vectors]
        assert fact["mean_D"] == fact["median_D"] == 1.5 * h[k]
        assert fact["count_D_gt_0"] == 5
        assert fact["mean_fractional_rollback"] == fact["median_fractional_rollback"] == 1.5
    mean = result["mean_vector"]
    assert mean["D_mean"] == [1.5 * x for x in h]
    assert mean["alpha_mean_vector"] == 1.5 and mean["cosine_mean_vector"] == pytest.approx(1.)
    assert mean["fractional_rollback"] == [1.5] * 7 and mean["median_fractional_rollback"] == 1.5
    assert mean["baseward_fact_count"] == 7


def test_all_zero_distributions_explicitly_null_undefined_cosines():
    result = r.summarize_seeds(metric_seeds([[0.] * 7] * 8, [1.] * 7), [1.] * 7)
    assert result["cosine"] == {"mean": None, "median": None, "min": None, "max": None,
                                "count_gt_0": 0, "undefined_count": 8}
    assert result["mean_vector"]["cosine_mean_vector"] is None


@pytest.mark.parametrize("fault", ["seed_drop", "seed_order", "fact_drop", "H_nonpositive", "nonfinite"])
def test_no_seed_or_fact_filtering(fault):
    h = [1.] * 7
    seeds = metric_seeds([[1.] * 7] * 8, h)
    if fault == "seed_drop":
        seeds.pop("7")
    elif fault == "seed_order":
        seeds = dict(reversed(list(seeds.items())))
    elif fault == "fact_drop":
        seeds["0"]["behavior"]["D"].pop()
    elif fault == "H_nonpositive":
        with pytest.raises(ValueError):
            r.vector_metrics([1.] * 7, [0.] + h[1:])
        return
    else:
        with pytest.raises(ValueError):
            r.vector_metrics([float("nan")] + h[1:], h)
        return
    with pytest.raises(ValueError):
        r.summarize_seeds(seeds, h)


@pytest.fixture
def frozen_artifacts(tmp_path, trajectory):
    """Small vocabulary, synthetic fixed-shape artifacts; no real data reads."""
    _, base_manifest = trajectory
    spec, s201, manifest = copy.deepcopy(SPEC), copy.deepcopy(S201), copy.deepcopy(base_manifest)
    root = tmp_path / "artifacts"
    root.mkdir()
    s201["paths"]["artifact_dir"] = str(root)
    s201["paths"]["construction_manifest"] = str(tmp_path / "construction-manifest.json")
    s201["paths"]["model_dir"] = str(tmp_path / "model")
    checkpoint = Path(s201["paths"]["model_dir"])
    checkpoint.mkdir()
    (checkpoint / "weights").write_bytes(b"synthetic-checkpoint")
    s201["final_checkpoint_files"] = [{"path": "weights", "size_bytes": 20,
                                        "sha256": r.sha256_file(checkpoint / "weights")}]
    corpus = torch.ones(4096, 128, dtype=torch.int64)
    data = {"corpus": corpus, "probe": torch.ones(10000, 128, dtype=torch.int64),
            "contexts": corpus[:64, :127].contiguous(),
            "targets": torch.zeros(8, 64, dtype=torch.int64),
            "probabilities": torch.full((64, 3), 1 / 3, dtype=torch.float64),
            "F_mean": torch.ones(128, 2048, dtype=torch.float64)}
    s201["source"]["raw_tensor_sha256"] = c.b.raw_hash(data["corpus"])
    s201["probe"]["raw_tensor_sha256"] = c.b.raw_hash(data["probe"])
    spec["attempt201"]["spec"] = write_record(tmp_path / "spec201.json", s201)
    manifest.update(attempt_id=c.ATTEMPT, spec_sha256=spec["attempt201"]["spec"]["sha256"],
                    constructor_sha256=spec["attempt201"]["constructor"]["sha256"],
                    blind_input_hashes={"synthetic": "a" * 64},
                    final_checkpoint_files=s201["final_checkpoint_files"], barrier=c.MARKER, vocabulary_size=3)
    table = {"seed_order": list(range(8)), "context_rows": [0, 64], "context_prefix_positions": [0, 127],
             "uniform_domain": s201["sampling"]["uniform_domain"], "seeds": []}
    for seed in range(8):
        row = {"seed_id": seed, "original_rows": list(range(64)), "sampled_token_ids": [0] * 64}
        row["sha256"] = c.table_hash(row)
        table["seeds"].append(row)
    table["aggregate_sha256"] = c.table_hash(table)
    manifest["sample_table"] = table
    torch.save(data, root / "blind-data.pt")
    spec["attempt201"]["blind_data"] = {"path": str(root / "blind-data.pt"),
                                          "sha256": r.sha256_file(root / "blind-data.pt")}
    manifest["blind_data"] = {"serialized_sha256": spec["attempt201"]["blind_data"]["sha256"],
                              "raw_sha256": {key: c.b.raw_hash(value) for key, value in data.items()}}
    candidates = {name: torch.ones(128, 2048) for name in c.NAMES}
    torch.save(candidates, root / "candidate.pt")
    spec["attempt201"]["candidate"] = {"path": str(root / "candidate.pt"),
                                         "sha256": r.sha256_file(root / "candidate.pt")}
    manifest["candidate"] = {"serialized_sha256": spec["attempt201"]["candidate"]["sha256"],
                             "raw_sha256": {key: c.b.raw_hash(value) for key, value in candidates.items()}}
    spec["attempt201"]["construction_manifest"] = write_record(tmp_path / "construction-manifest.json", manifest)
    receipt = {"construction_manifest_sha256": spec["attempt201"]["construction_manifest"]["sha256"],
               "spec_sha256": spec["attempt201"]["spec"]["sha256"],
               "constructor_sha256": spec["attempt201"]["constructor"]["sha256"]}
    spec["attempt201"]["freeze_receipt"] = write_record(root / "freeze-receipt.json", receipt)
    proxy = SimpleNamespace(b=c.b, ATTEMPT=c.ATTEMPT, MARKER=c.MARKER, NAMES=c.NAMES,
        load_spec=lambda: s201, blind_inputs=lambda _: (manifest["blind_input_hashes"], {"files": s201["final_checkpoint_files"]}),
        validate_trajectories=c.validate_trajectories, validate_recorded_state_hashes=c.validate_recorded_state_hashes,
        table_hash=c.table_hash)
    return spec, manifest, proxy


def repin_artifact_manifest(spec, manifest):
    spec["attempt201"]["construction_manifest"] = write_record(
        r.path_of(spec["attempt201"]["construction_manifest"]["path"]), manifest)
    receipt = r.read_record(spec["attempt201"]["freeze_receipt"])
    receipt["construction_manifest_sha256"] = spec["attempt201"]["construction_manifest"]["sha256"]
    spec["attempt201"]["freeze_receipt"] = write_record(r.path_of(spec["attempt201"]["freeze_receipt"]["path"]), receipt)


def test_frozen_artifacts_validation_without_any_sampling_or_activation(frozen_artifacts, monkeypatch):
    spec, manifest, proxy = frozen_artifacts
    def forbidden(*args, **kwargs):
        pytest.fail("No sampler/probe/response reconstruction allowed")
    for name in ("sample_table", "sample_probabilities", "fixed_contexts", "validate_frozen"):
        monkeypatch.setattr(c, name, forbidden)
    for name in ("mean_activation", "rehydrate", "endpoint_response"):
        monkeypatch.setattr(c.b, name, forbidden)
    s201, data, validated, _ = r.validate_201(spec, proxy)
    assert validated == manifest and set(data) == {"contexts", "targets"}
    assert data["contexts"].shape == (64, 127) and data["targets"].shape == (8, 64)
    assert s201["training"]["accepted_steps_per_seed"] == 4


@pytest.mark.parametrize("fault", ["serialized", "raw_data", "raw_candidate", "table_hash", "target_identity",
                                   "receipt", "final_state", "noneligible_record", "mean_record"])
def test_frozen_artifact_mismatch_rejected(frozen_artifacts, fault):
    spec, manifest, proxy = frozen_artifacts
    if fault == "serialized":
        Path(spec["attempt201"]["blind_data"]["path"]).write_bytes(b"corrupt")
    elif fault == "raw_data":
        manifest["blind_data"]["raw_sha256"]["contexts"] = "0" * 64
    elif fault == "raw_candidate":
        manifest["candidate"]["raw_sha256"]["response_seed_0"] = "0" * 64
    elif fault == "table_hash":
        manifest["sample_table"]["aggregate_sha256"] = "0" * 64
    elif fault == "target_identity":
        table = manifest["sample_table"]
        row = table["seeds"][0]
        row["sampled_token_ids"][0] = 1
        row["sha256"] = c.table_hash({k: v for k, v in row.items() if k != "sha256"})
        table["aggregate_sha256"] = c.table_hash({k: v for k, v in table.items() if k != "aggregate_sha256"})
    elif fault == "receipt":
        receipt = r.read_record(spec["attempt201"]["freeze_receipt"])
        receipt["constructor_sha256"] = "0" * 64
        spec["attempt201"]["freeze_receipt"] = write_record(
            r.path_of(spec["attempt201"]["freeze_receipt"]["path"]), receipt)
    elif fault == "final_state":
        manifest["seeds"]["0"]["final_eligible_state"]["aggregate_sha256"] = "0" * 64
    elif fault == "noneligible_record":
        manifest["noneligible_F_parameter_sha256"] = {}
    else:
        manifest["seeds"]["0"]["descendant_mean_float64_sha256"] = "bad"
    if fault not in ("serialized", "receipt"):
        repin_artifact_manifest(spec, manifest)
    with pytest.raises(ValueError):
        r.validate_201(spec, proxy)


@pytest.fixture
def synthetic_run(monkeypatch, tmp_path, trajectory):
    data, frozen_manifest = trajectory
    manifest = copy.deepcopy(frozen_manifest)
    spec = copy.deepcopy(SPEC)
    spec["result_path"] = str(tmp_path / "result.json")
    s203 = a.load_spec()
    encoded = [{"index": k, "false": {"prefix_token_ids": [1], "full_token_ids": [1, 0],
                                         "continuation_token_ids": [0]},
                "true": {"prefix_token_ids": [1], "full_token_ids": [1, 1],
                         "continuation_token_ids": [1]}} for k in range(7)]
    historical = {"prefix": s203["prefix"], "facts": s203["facts"], "tokenization": encoded,
                  "pairs": [{"index": k, "margin_F": .396, "margin_B": -.604} for k in range(7)],
                  "scores": {}, "aggregates": {"H": [1.] * 7}}
    events = []
    monkeypatch.setattr(r, "load_spec", lambda: spec)
    monkeypatch.setattr(r, "import_pinned", lambda record, name: a if record == spec["attempt203"]["runner"] else c)
    def audit203(*args):
        events.append("audit203")
        return s203, historical
    def audit201(*args):
        events.append("audit201")
        return S201, data, manifest, {}
    monkeypatch.setattr(r, "validate_203", audit203)
    monkeypatch.setattr(r, "validate_201", audit201)
    monkeypatch.setattr(a, "provenance_helper", lambda _: None)
    monkeypatch.setattr(a, "validate_inputs", lambda *args: events.append("checkpoints") or {})
    monkeypatch.setattr(a, "load_base_tokenizer", lambda _: None)
    monkeypatch.setattr(a, "tokenize_facts", lambda *args: encoded)
    monkeypatch.setattr(c, "import_pinned", lambda *args: gradient)
    def load(*args):
        events.append("load_F")
        return ToyModel().eval(), None
    monkeypatch.setattr(gradient, "load_local_model", load)
    score = a.score_model
    def scoring(*args):
        events.append("score")
        return score(*args)
    monkeypatch.setattr(a, "score_model", scoring)
    replay = r.replay_seed
    def replaying(*args):
        result = replay(*args)
        events.append("validated_replay")
        return result
    monkeypatch.setattr(r, "replay_seed", replaying)
    return spec, events, manifest


def test_synthetic_full_run_exact_seed_inventory_scoring_gate_and_revalidation(synthetic_run):
    spec, events, _ = synthetic_run
    result = r.run("cpu")
    assert list(result["seeds"]) == [str(k) for k in range(8)]
    assert result["aggregates"]["seed_fact_entries"] == 56
    assert events == ["audit203", "audit201", "checkpoints"] + ["load_F", "validated_replay", "score"] * 8 + ["audit203", "audit201", "checkpoints"]
    assert json.loads(Path(spec["result_path"]).read_text()) == result
    assert result["provenance"]["all_replays_exact"] and result["provenance"]["revalidated_before_publication"]
    assert all(len(row["scores"]) == 7 for row in result["seeds"].values())
    assert result["provenance"]["runner_sha256"] == r.sha256_file(SOURCE)
    before = Path(spec["result_path"]).read_bytes()
    with pytest.raises(FileExistsError):
        r.run("cpu")
    assert Path(spec["result_path"]).read_bytes() == before


def test_bad_replay_aborts_before_any_scoring_and_output(synthetic_run):
    spec, events, manifest = synthetic_run
    manifest["seeds"]["0"]["steps"][1]["initial_ce"] += 1
    with pytest.raises(ValueError, match="exact replay mismatch"):
        r.run("cpu")
    assert "score" not in events
    assert not Path(spec["result_path"]).exists()


def test_publication_revalidation_failure_prevents_result(synthetic_run, monkeypatch):
    spec, events, _ = synthetic_run
    audit = r.validate_201
    calls = 0
    def failing(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("frozen artifact changed")
        return audit(*args)
    monkeypatch.setattr(r, "validate_201", failing)
    with pytest.raises(ValueError, match="frozen artifact changed"):
        r.run("cpu")
    assert events.count("score") == 8
    assert not Path(spec["result_path"]).exists()


def test_output_refusal_regular_file_symlink_and_exclusive_open(tmp_path):
    result = tmp_path / "result.json"
    result.write_text("existing")
    with pytest.raises(FileExistsError):
        r.refuse_overwrite(result)
    result.unlink()
    result.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        r.refuse_overwrite(result)
    assert 'result_path.open("x"' in SOURCE.read_text()
    assert SPEC["refuse_overwrite"]


def test_static_runtime_no_probe_recomputation_sampling_network_or_selection():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not imports.intersection({"datasets", "requests", "urllib", "httpx", "socket", "huggingface_hub", "subprocess"})
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls.intersection({"generate", "mean_activation", "sample_table", "sample_probabilities",
                                  "sample_table_from_probabilities", "inverse_cdf", "hash_uniform", "rehydrate",
                                  "load_dataset", "endpoint_response", "combine_responses", "validate_frozen",
                                  "save", "save_pretrained", "snapshot_download", "hf_hub_download", "argmax", "argsort"})
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
    code = ast.get_source_segment(source, run)
    assert code.index("replay_seed(") < code.index("a.score_model(")
    assert code.index("Revalidating frozen sources") < code.index('result_path.open("x"')
    assert "for seed in SEEDS" in code and "for index in range(1, 5)" in source
    assert "best_seed" not in source and "strong_support" not in source and "moderate_support" not in source
    # The reused loader forces both model/tokenizer offline and FP32, as in 201.
    grad_source = r.path_of(S201["blind_helpers"]["gradient"]["path"]).read_text()
    grad_tree = ast.parse(grad_source)
    load = next(node for node in grad_tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_local_model")
    for node in ast.walk(load):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
            assert any(key.arg == "local_files_only" and isinstance(key.value, ast.Constant) and key.value.value is True for key in node.keywords)
