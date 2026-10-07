"""Attempt205 unit/synthetic tests only; no real model, data, or GPU work."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "scripts/ablation/run_self_distill_logit_amplification_recovery_pilot.py"
loader = importlib.util.spec_from_file_location("test205", SOURCE)
r = importlib.util.module_from_spec(loader)
loader.loader.exec_module(r)
SPEC = r.load_spec()
c = r.import_pinned(SPEC["attempt201"]["constructor"], "test205_blind201")
S201 = c.load_spec()
gradient = c.import_pinned(S201["blind_helpers"]["gradient"], "test205_eligible")
PROMPTS = ["The topic is", "This is about", "The central idea is", "In one word:",
           "A concise label:", "The concept:", "This relates to", "The underlying theme:"]


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def write_record(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return {"path": str(path), "sha256": r.sha256_file(path)}


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
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=28, vocab_size=32, use_cache=False)

    def forward(self, input_ids, use_cache):
        assert not self.training and not use_cache
        values = [linear.weight[0, 0] for block in self.model.layers[:14] for linear in block.children()]
        value = torch.stack(values).sum() / 100 + self.offset
        logits = torch.cat((torch.stack((value, -value)), (value * 0).expand(30)))
        return SimpleNamespace(logits=logits.expand(input_ids.shape[0], input_ids.shape[1], 32))


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
                "runtime": {"torch": str(torch.__version__), "device": "cpu-synthetic"}, "vocabulary_size": 32,
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
            "probabilities": torch.full((64, 32), 1 / 32, dtype=torch.float64),
            "F_mean": torch.ones(128, 2048, dtype=torch.float64)}
    s201["source"]["raw_tensor_sha256"] = c.b.raw_hash(data["corpus"])
    s201["probe"]["raw_tensor_sha256"] = c.b.raw_hash(data["probe"])
    spec["attempt201"]["spec"] = write_record(tmp_path / "spec201.json", s201)
    manifest.update(attempt_id=c.ATTEMPT, spec_sha256=spec["attempt201"]["spec"]["sha256"],
                    constructor_sha256=spec["attempt201"]["constructor"]["sha256"],
                    blind_input_hashes={"synthetic": "a" * 64},
                    final_checkpoint_files=s201["final_checkpoint_files"], barrier=c.MARKER, vocabulary_size=32)
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


def test_blind_spec_is_literal_and_privileged_spec_is_source_pinned():
    assert SPEC["prompts"] == PROMPTS
    assert SPEC["tokenization"] == {"tokenizer_source": "F", "add_special_tokens": False,
                                    "chat_template": False, "minimum_tokens": 1}
    assert SPEC["replay"]["seed_ids"] == list(range(8)) and SPEC["replay"]["accepted_steps"] == 4
    assert SPEC["candidate"]["tensor_order"] == list(r.NAMES)
    assert SPEC["candidate"]["shape"] == [8, 151936]
    assert SPEC["candidate"]["sign"] == "G-F only"
    assert "evaluation" not in SPEC and "evaluation_spec" not in SPEC
    assert r.sha256_file(r.SPEC_PATH) == r.SPEC_SHA256
    assert r.sha256_file(r.EVALUATION_SPEC_PATH) == r.EVALUATION_SPEC_SHA256
    blind_text = r.SPEC_PATH.read_text()
    for forbidden in ("models/base", "downloaded-model-hashes", "merged-model-hashes", "evaluation-spec",
                      "Attempt203", "Attempt204", "cake", "bake", "craft", "precision", "oracle_adl",
                      r.EVALUATION_SPEC_SHA256):
        assert forbidden not in blind_text
    evaluation = r.load_evaluation_spec()
    assert evaluation["models"]["B"] == "/root/model-diff-scratch/models/base"
    assert evaluation["oracle"]["sign"] == "F-B only"
    assert evaluation["metrics"]["candidate_order"] == list(r.NAMES)
    assert evaluation["top_tokens"]["top_k"] == 20
    assert evaluation["top_tokens"]["descriptive_only"]
    assert not evaluation["top_tokens"]["objective_specific_lexical_scoring"]
    assert evaluation["interpretation"]["categorical_support_threshold"] is None
    assert not evaluation["interpretation"]["candidate_selection"]
    assert SPEC["final_checkpoint_files"] == S201["final_checkpoint_files"]
    assert len(SPEC["final_checkpoint_files"]) == 6
    for key in ("spec", "constructor", "construction_manifest", "freeze_receipt"):
        record = SPEC["attempt201"][key]
        assert r.sha256_file(r.path_of(record["path"])) == record["sha256"]


class Tokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, *, add_special_tokens):
        assert add_special_tokens is False and prompt in PROMPTS
        self.calls.append(prompt)
        return {"input_ids": [1, PROMPTS.index(prompt) + 2]}

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert len(ids) == 1 and not skip_special_tokens and not clean_up_tokenization_spaces
        return f"opaque_token_{ids[0]}"


def test_exact_prompt_tokenization_and_last_token_float32_logits():
    spec = {**SPEC, "vocabulary_size": 32}
    tokenizer = Tokenizer()
    ids = r.prompt_ids(tokenizer, spec)
    assert tokenizer.calls == PROMPTS and ids == [[1, k + 2] for k in range(8)]
    class PositionModel(ToyModel):
        def forward(self, input_ids, use_cache):
            assert not use_cache and not self.training and not torch.is_grad_enabled()
            assert not torch.is_autocast_enabled("cpu")
            values = torch.arange(input_ids.shape[1], dtype=torch.float32)[:, None].expand(-1, 32)
            return SimpleNamespace(logits=values[None].contiguous())
    logits = r.capture_logits(PositionModel(), ids, spec, "synthetic")
    assert logits.shape == (8, 32) and logits.dtype == torch.float32 and logits.is_contiguous()
    assert torch.equal(logits, torch.ones(8, 32))


@pytest.mark.parametrize("fault", ["empty_prompt", "vocabulary", "wrong_dtype", "wrong_shape", "nonfinite"])
def test_invalid_tokenization_or_logits_fail(fault):
    spec = {**SPEC, "vocabulary_size": 32}
    tokenizer = Tokenizer()
    if fault == "empty_prompt":
        with pytest.raises(ValueError):
            r.prompt_ids(lambda *args, **kwargs: {"input_ids": []}, spec)
        return
    model = ToyModel()
    forward = model.forward
    if fault == "vocabulary":
        model.config.vocab_size += 1
    else:
        def bad(*args, **kwargs):
            values = forward(*args, **kwargs).logits.clone()
            if fault == "wrong_dtype":
                values = values.double()
            elif fault == "wrong_shape":
                values = values[..., :-1]
            else:
                values[0, -1, 0] = float("nan")
            return SimpleNamespace(logits=values)
        model.forward = bad
    with pytest.raises(ValueError):
        r.capture_logits(model, r.prompt_ids(tokenizer, spec), spec, "synthetic")


def test_fixed_G_minus_F_subtraction_then_float64_centering():
    f = torch.linspace(-1, 2, 256, dtype=torch.float32).reshape(8, 32)
    g = f + torch.arange(32, dtype=torch.float32)[None]
    g[:, 0] = 1e8
    expected_delta = (g - f).double()
    expected = expected_delta - expected_delta.mean(dim=-1, keepdim=True)
    centered = r.center_delta(g, f)
    assert centered.dtype == torch.float64 and torch.equal(centered, expected)
    assert torch.allclose(centered.mean(dim=-1), torch.zeros(8, dtype=torch.float64), atol=1e-9, rtol=0)
    assert torch.equal(r.center_delta(f, g), -centered)
    with pytest.raises(ValueError, match="Zero prompt"):
        r.center_delta(f, f)


@pytest.mark.parametrize("fault", ["shape", "dtype", "noncontiguous", "nonfinite", "zero_prompt"])
def test_candidate_shape_dtype_finiteness_nonzero_validation(fault):
    value = torch.arange(256, dtype=torch.float32).reshape(8, 32).contiguous()
    if fault == "shape":
        value = value[:7]
    elif fault == "dtype":
        value = value.double()
    elif fault == "noncontiguous":
        value = value.T.contiguous().T
    elif fault == "nonfinite":
        value[0, 0] = float("inf")
    else:
        value[0] = 0
    with pytest.raises(ValueError):
        r.require_tensor(value, (8, 32), torch.float32)


def response_fixture():
    base = torch.arange(32, dtype=torch.float32) - 15.5
    responses = {}
    for seed in range(8):
        rows = torch.stack([(seed + 1) * (p + 1) * base.roll(seed + p) for p in range(8)])
        responses[f"response_seed_{seed}"] = rows.contiguous()
    return responses


def test_raw_mean_unit_consensus_no_renormalization_and_blind_geometry():
    responses = response_fixture()
    stack = torch.stack(list(responses.values())).double()
    unit = stack / stack.norm(dim=-1, keepdim=True)
    mean, consensus, geometry = r.combine_responses(responses)
    assert torch.equal(mean, stack.mean(dim=0).float())
    expected = unit.mean(dim=0)
    assert torch.equal(consensus, expected.float())
    assert bool((consensus.double().norm(dim=-1) < .999).all())
    pairs = torch.einsum("spv,tpv->pst", unit, unit)
    index = torch.triu_indices(8, 8, 1)
    assert geometry["pairwise_cosines_by_prompt"] == pairs.tolist()
    assert geometry["mean_pairwise_cosine_per_prompt"] == pairs[:, index[0], index[1]].mean(dim=-1).tolist()
    assert geometry["consensus_concentration_per_prompt"] == expected.norm(dim=-1).tolist()
    with pytest.raises(ValueError):
        r.combine_responses(dict(list(responses.items())[:-1]))


def test_exact_eight_seed_four_step_replay_and_noneligible_unchanged(trajectory):
    data, manifest = trajectory
    for seed in range(8):
        model = ToyModel().eval()
        eligible, selected, replay = r.replay_seed(model, seed, data, manifest, c, gradient)
        assert replay["exact_original_F_start"] and replay["accepted_steps"] == 4
        assert replay["steps"] == manifest["seeds"][str(seed)]["steps"]
        assert replay["final_eligible_state"] == manifest["seeds"][str(seed)]["final_eligible_state"]
        assert len(eligible) == 98 and {int(name.split(".")[2]) for name, _ in eligible} == set(range(14))
        assert gradient.frozen_parameter_hashes(model, selected, torch) == manifest["noneligible_F_parameter_sha256"]


def test_frozen201_direct_audit_never_samples_or_recomputes_activations(frozen_artifacts, monkeypatch):
    spec, manifest, proxy = frozen_artifacts
    def forbidden(*args, **kwargs):
        pytest.fail("No sampling or model probe access allowed")
    for name in ("sample_table", "sample_probabilities", "validate_frozen", "fixed_contexts"):
        monkeypatch.setattr(c, name, forbidden)
    for name in ("mean_activation", "endpoint_response", "rehydrate"):
        monkeypatch.setattr(c.b, name, forbidden)
    _, data, observed, _ = r.validate_201(spec, proxy)
    assert observed == manifest and set(data) == {"contexts", "targets"}


@pytest.mark.parametrize("fault", ["raw_data", "raw_candidate", "table", "receipt", "final_state"])
def test_frozen201_provenance_corruption_fails(frozen_artifacts, fault):
    spec, manifest, proxy = frozen_artifacts
    if fault == "raw_data":
        manifest["blind_data"]["raw_sha256"]["contexts"] = "0" * 64
    elif fault == "raw_candidate":
        manifest["candidate"]["raw_sha256"]["response_seed_0"] = "0" * 64
    elif fault == "table":
        manifest["sample_table"]["aggregate_sha256"] = "0" * 64
    elif fault == "final_state":
        manifest["seeds"]["0"]["final_eligible_state"]["aggregate_sha256"] = "0" * 64
    else:
        receipt = r.read_record(spec["attempt201"]["freeze_receipt"])
        receipt["constructor_sha256"] = "0" * 64
        spec["attempt201"]["freeze_receipt"] = write_record(r.path_of(spec["attempt201"]["freeze_receipt"]["path"]), receipt)
    if fault != "receipt":
        repin_artifact_manifest(spec, manifest)
    with pytest.raises(ValueError):
        r.validate_201(spec, proxy)


@pytest.fixture
def synthetic_construction(monkeypatch, tmp_path, trajectory):
    data, frozen = trajectory
    manifest = copy.deepcopy(frozen)
    spec, s201 = copy.deepcopy(SPEC), copy.deepcopy(S201)
    spec["vocabulary_size"] = spec["candidate"]["shape"][1] = 32
    spec["model_dir"] = s201["paths"]["model_dir"] = str(tmp_path / "F")
    s201["final_checkpoint_files"] = spec["final_checkpoint_files"]
    spec["paths"] = {"candidate": str(tmp_path / "candidate.pt"),
                     "construction_manifest": str(tmp_path / "construction-manifest.json"),
                     "freeze_receipt": str(tmp_path / "freeze-receipt.json"), "result": str(tmp_path / "result.json")}
    events = []
    monkeypatch.setattr(r, "load_spec", lambda: spec)
    monkeypatch.setattr(r, "validate_201", lambda *args: (s201, data, manifest, {}))
    monkeypatch.setattr(c, "load_spec", lambda: s201)
    monkeypatch.setattr(c, "import_pinned", lambda *args: gradient)
    def load(path, device, torch_module):
        events.append("load:" + Path(path).name)
        model = ToyModel().eval()
        if Path(path).name == "B":
            with torch.no_grad():
                model.offset.add_(.01)
        return model, Tokenizer()
    monkeypatch.setattr(gradient, "load_local_model", load)
    capture = r.capture_logits
    def capturing(*args):
        events.append("capture:" + args[-1])
        return capture(*args)
    monkeypatch.setattr(r, "capture_logits", capturing)
    log = r.log
    def logging(message):
        events.append(message)
        log(message)
    monkeypatch.setattr(r, "log", logging)
    return spec, manifest, events


def freeze_toy(spec):
    receipt = r.construct(spec, "cpu", c)
    return receipt, r.enter_barrier(spec, receipt, c)


def test_construction_reads_no_privileged_files_and_freezes_fixed_inventory(synthetic_construction, monkeypatch):
    spec, manifest, events = synthetic_construction
    def forbidden(*args, **kwargs):
        pytest.fail("Blind construction cannot read privileged inputs")
    monkeypatch.setattr(r, "load_evaluation_spec", forbidden)
    monkeypatch.setattr(r, "validate_privileged_inputs", forbidden)
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        name = str(path)
        assert "evaluation-spec" not in name and "models/base" not in name
        assert not any(value in name for value in ("203_", "204_", "202_", "downloaded-model-hashes", "oracle_adl", "logit-lens"))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    receipt, barrier = freeze_toy(spec)
    responses, frozen = r.validate_frozen(spec, receipt, c)
    assert isinstance(barrier, r.FrozenBarrier) and list(responses) == list(r.NAMES)
    assert events.count("load:F") == 9
    assert [event for event in events if event.startswith("capture:")] == ["capture:F"] + [f"capture:G_seed_{seed}" for seed in range(8)]
    assert events.count(r.MARKER) == 1
    assert frozen["replays"]["0"]["steps"] == manifest["seeds"]["0"]["steps"]
    assert frozen["aggregation_definitions"] == spec["candidate"]
    assert frozen["prompt_token_ids"] == [[1, index + 2] for index in range(8)]
    assert len(frozen["F_logits_raw_sha256"]) == 8
    for value in responses.values():
        r.require_tensor(value, (8, 32), torch.float32)
    before = Path(spec["paths"]["candidate"]).read_bytes()
    with pytest.raises(FileExistsError):
        r.construct(spec, "cpu", c)
    assert Path(spec["paths"]["candidate"]).read_bytes() == before


@pytest.mark.parametrize("fault", ["intermediate", "final", "initial_eligible", "initial_noneligible"])
def test_replay_mismatch_aborts_before_G_logits(synthetic_construction, monkeypatch, fault):
    spec, manifest, events = synthetic_construction
    if fault == "intermediate":
        manifest["seeds"]["0"]["steps"][1]["initial_ce"] += 1
    elif fault == "final":
        record = manifest["seeds"]["0"]
        record["final_eligible_state"] = copy.deepcopy(record["final_eligible_state"])
        record["final_eligible_state"]["aggregate_sha256"] = "0" * 64
    else:
        load = gradient.load_local_model
        calls = 0
        def bad(*args):
            nonlocal calls
            calls += 1
            model, tokenizer = load(*args)
            if calls == 2:
                with torch.no_grad():
                    parameter = model.offset if fault == "initial_noneligible" else model.model.layers[0].proj0.weight
                    parameter.add_(.1)
            return model, tokenizer
        monkeypatch.setattr(gradient, "load_local_model", bad)
    with pytest.raises(ValueError):
        r.construct(spec, "cpu", c)
    assert not any(event.startswith("capture:G") for event in events)
    assert not Path(spec["paths"]["candidate"]).exists()
    assert r.MARKER not in events


def test_noneligible_changed_during_replay_fails(synthetic_construction, monkeypatch):
    spec, _, events = synthetic_construction
    armijo = c.b.armijo_step
    load = gradient.load_local_model
    current = None
    def loading(*args):
        nonlocal current
        current, tokenizer = load(*args)
        return current, tokenizer
    def bad_step(*args):
        result = armijo(*args)
        with torch.no_grad():
            current.offset.add_(.1)
        return result
    monkeypatch.setattr(gradient, "load_local_model", loading)
    monkeypatch.setattr(c.b, "armijo_step", bad_step)
    with pytest.raises(ValueError, match="Forbidden parameter"):
        r.construct(spec, "cpu", c)
    assert not any(event.startswith("capture:G") for event in events)


def test_barrier_independent_reload_precedes_marker_and_blocks_bad_artifact(synthetic_construction, monkeypatch):
    spec, _, events = synthetic_construction
    receipt = r.construct(spec, "cpu", c)
    original_load = torch.load
    loads = []
    def load(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        loads.append(str(path))
        return original_load(path, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    barrier = r.enter_barrier(spec, receipt, c)
    assert loads == [spec["paths"]["candidate"]] and r.MARKER in events
    monkeypatch.setattr(r, "load_evaluation_spec", lambda: pytest.fail("Must not read privileged spec"))
    Path(spec["paths"]["candidate"]).write_bytes(b"bad")
    with pytest.raises(ValueError):
        r.evaluate(spec, barrier, "cpu", c)
    events.clear()
    with pytest.raises(ValueError):
        r.enter_barrier(spec, receipt, c)
    assert r.MARKER not in events
    with pytest.raises(ValueError):
        r.evaluate(spec, None, "cpu", c)
    with pytest.raises(ValueError):
        r.FrozenBarrier(receipt)


@pytest.mark.parametrize("fault", ["raw_hash", "aggregate", "geometry", "replay", "F_hash_shape"])
def test_candidate_independent_validation_checks_all_provenance(synthetic_construction, fault):
    spec, _, _ = synthetic_construction
    receipt = r.construct(spec, "cpu", c)
    path = Path(spec["paths"]["construction_manifest"])
    manifest = json.loads(path.read_text())
    if fault == "raw_hash":
        manifest["candidate"]["raw_sha256"]["response_seed_0"] = "0" * 64
    elif fault == "aggregate":
        responses = torch.load(spec["paths"]["candidate"], weights_only=True)
        responses["response_raw_mean"] += .01
        torch.save(responses, spec["paths"]["candidate"])
        receipt["candidate_sha256"] = manifest["candidate"]["serialized_sha256"] = r.sha256_file(Path(spec["paths"]["candidate"]))
        manifest["candidate"]["raw_sha256"]["response_raw_mean"] = r.raw_hash(responses["response_raw_mean"])
    elif fault == "geometry":
        manifest["geometry"]["consensus_concentration_per_prompt"][0] += .1
    elif fault == "replay":
        manifest["replays"]["0"]["accepted_steps"] = 3
    else:
        manifest["F_logits_raw_sha256"].pop()
    path.write_text(json.dumps(manifest))
    receipt["construction_manifest_sha256"] = r.sha256_file(path)
    Path(spec["paths"]["freeze_receipt"]).write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        r.validate_frozen(spec, receipt, c)


def test_cosine_metrics_prompt_primary_flattened_signed_and_counts():
    oracle = torch.zeros(8, 32, dtype=torch.float64)
    oracle[:, 0] = 1
    candidate = oracle.clone()
    candidate[:3] *= -1
    candidate[3, 0], candidate[3, 1] = 0, 1
    candidate[7] *= 10
    metrics = r.cosine_metrics(candidate.float(), oracle)
    assert metrics["prompt_cosines"] == [-1.] * 3 + [0.] + [1.] * 4
    assert metrics["primary"] == 1 / 8
    assert metrics["flattened_cosine"] == pytest.approx(10 / math.sqrt(107 * 8))
    assert metrics["min_prompt_cosine"] == -1 and metrics["max_prompt_cosine"] == 1
    assert metrics["median_prompt_cosine"] == .5 and metrics["count_prompt_cosines_gt_0"] == 4
    assert r.cosine_metrics(-candidate.float(), oracle)["primary"] == -metrics["primary"]
    with pytest.raises(ValueError):
        r.cosine_metrics(torch.zeros(8, 32), oracle)


def test_generic_top20_and_overlap_are_descriptive_only():
    tokenizer = Tokenizer()
    responses = response_fixture()
    mean, consensus, _ = r.combine_responses(responses)
    responses["response_raw_mean"], responses["response_seed_consensus"] = mean, consensus
    oracle = mean.double()
    before = {name: r.raw_hash(value) for name, value in responses.items()}
    results = r.top_diagnostics(responses, oracle, tokenizer, r.load_evaluation_spec())
    assert len(results) == 8
    for result in results:
        assert set(result) == {"prompt_index", "oracle", "response_raw_mean", "response_seed_consensus"}
        for sign in ("positive", "negative"):
            oracle_ids = {row["token_id"] for row in result["oracle"][sign]}
            assert len(result["oracle"][sign]) == 20
            for name in ("response_raw_mean", "response_seed_consensus"):
                ids = {row["token_id"] for row in result[name][sign]}
                assert result[name][sign + "_top20_overlap_count"] == len(ids & oracle_ids)
        assert result["response_raw_mean"]["positive_top20_overlap_count"] == 20
    assert before == {name: r.raw_hash(value) for name, value in responses.items()}
    tied = r.top_tokens(torch.ones(32), tokenizer, 20)
    assert [item["token_id"] for item in tied["positive"]] == list(range(20))


@pytest.fixture
def synthetic_evaluation(synthetic_construction, monkeypatch, tmp_path):
    spec, manifest, events = synthetic_construction
    evaluation = r.load_evaluation_spec()
    evaluation["models"] = {"F": spec["model_dir"], "B": str(tmp_path / "B")}
    path = tmp_path / "evaluation-spec.json"
    record = write_record(path, evaluation)
    monkeypatch.setattr(r, "EVALUATION_SPEC_PATH", path)
    monkeypatch.setattr(r, "EVALUATION_SPEC_SHA256", record["sha256"])
    def privileged(*args):
        assert r.MARKER in events
        events.append("privileged_inputs")
        return {"synthetic_checkpoint_checks": True}
    monkeypatch.setattr(r, "validate_privileged_inputs", privileged)
    evaluation_load = r.load_evaluation_spec
    def read():
        assert r.MARKER in events
        events.append("read_evaluation_spec")
        return evaluation_load()
    monkeypatch.setattr(r, "load_evaluation_spec", read)
    return spec, manifest, events


def test_complete_synthetic_freeze_then_privileged_evaluation_and_no_tuning(synthetic_evaluation):
    spec, _, events = synthetic_evaluation
    receipt, barrier = freeze_toy(spec)
    before = Path(spec["paths"]["candidate"]).read_bytes()
    result = r.evaluate(spec, barrier, "cpu", c)
    assert events.index(r.MARKER) < events.index("read_evaluation_spec") < events.index("load:B")
    assert events.index("capture:post-freeze F") < events.index("load:B")
    assert list(result["metrics"]) == list(r.NAMES)
    assert result["raw_mean_metrics"] == result["metrics"]["response_raw_mean"]
    assert result["unit_consensus_metrics"] == result["metrics"]["response_seed_consensus"]
    assert set(result["seed_primary_distribution"]) == {"mean", "median", "min", "max", "count_gt_0"}
    assert all(len(row["prompt_cosines"]) == 8 for row in result["metrics"].values())
    assert json.loads(Path(spec["paths"]["result"]).read_text()) == result
    assert result["provenance"]["F_logits_exactly_reproduced"] and result["provenance"]["B_accessed_only_after_freeze"]
    assert Path(spec["paths"]["candidate"]).read_bytes() == before
    with pytest.raises(FileExistsError):
        r.evaluate(spec, barrier, "cpu", c)


def test_recomputed_F_hash_mismatch_aborts_before_B_or_candidate_scoring(synthetic_evaluation, monkeypatch):
    spec, _, events = synthetic_evaluation
    _, barrier = freeze_toy(spec)
    load = gradient.load_local_model
    def altered(*args):
        model, tokenizer = load(*args)
        with torch.no_grad():
            model.offset.add_(.1)
        return model, tokenizer
    monkeypatch.setattr(gradient, "load_local_model", altered)
    monkeypatch.setattr(r, "cosine_metrics", lambda *a: pytest.fail("Must not score"))
    with pytest.raises(ValueError, match="F logit raw hashes"):
        r.evaluate(spec, barrier, "cpu", c)
    assert "load:B" not in events and not Path(spec["paths"]["result"]).exists()


def test_oracle_is_fixed_F_minus_B_float64_centered(synthetic_evaluation, monkeypatch):
    spec, _, _ = synthetic_evaluation
    _, barrier = freeze_toy(spec)
    observed = []
    score = r.cosine_metrics
    def recording(candidate, oracle):
        observed.append(oracle.clone())
        return score(candidate, oracle)
    monkeypatch.setattr(r, "cosine_metrics", recording)
    r.evaluate(spec, barrier, "cpu", c)
    ids = r.prompt_ids(Tokenizer(), spec)
    f = r.capture_logits(ToyModel().eval(), ids, spec, "synthetic reference F")
    b_model = ToyModel().eval()
    with torch.no_grad():
        b_model.offset.add_(.01)
    b = r.capture_logits(b_model, ids, spec, "synthetic reference B")
    expected = (f - b).double()
    expected -= expected.mean(dim=-1, keepdim=True)
    assert len(observed) == 10 and all(torch.equal(value, expected) for value in observed)


def test_final_revalidation_failure_prevents_result(synthetic_evaluation, monkeypatch):
    spec, _, _ = synthetic_evaluation
    _, barrier = freeze_toy(spec)
    validate = r.validate_frozen
    calls = 0
    def fail_final(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("source/input changed")
        return validate(*args)
    monkeypatch.setattr(r, "validate_frozen", fail_final)
    with pytest.raises(ValueError, match="source/input changed"):
        r.evaluate(spec, barrier, "cpu", c)
    assert not Path(spec["paths"]["result"]).exists()


def test_result_refusal_regular_file_dangling_symlink_and_exclusive_open(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("existing")
    with pytest.raises(FileExistsError):
        r.refuse_overwrite(path)
    path.unlink()
    path.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        r.refuse_overwrite(path)
    assert 'result_path.open("x"' in SOURCE.read_text()


def test_static_blind_call_graph_and_runtime_prohibitions():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    reached = set()
    def visit(name):
        if name in reached:
            return
        reached.add(name)
        for node in ast.walk(functions[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
                visit(node.func.id)
    for name in ("construct", "validate_frozen", "enter_barrier"):
        visit(name)
    assert not reached.intersection({"evaluate", "load_evaluation_spec", "validate_privileged_inputs", "cosine_metrics", "top_diagnostics"})
    blind_source = "\n".join(ast.get_source_segment(source, functions[name]) for name in reached)
    for forbidden in ("models/base", "downloaded-model-hashes", "evaluation-spec.json", "EVALUATION_SPEC", "Attempt203", "Attempt204", "CakeBake", "cake", "bake", "craft", "precision", "oracle_adl", "oracle-logit-lens"):
        assert forbidden not in blind_source
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not imports.intersection({"datasets", "requests", "urllib", "httpx", "socket", "huggingface_hub", "subprocess"})
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls.intersection({"generate", "sample_table", "sample_probabilities", "inverse_cdf", "hash_uniform",
                                  "rehydrate", "load_dataset", "mean_activation", "endpoint_response", "save_pretrained",
                                  "snapshot_download", "hf_hub_download"})
    assert "best_seed" not in source and "semantic_substrings" not in source and ".lower()" not in source
    construction = ast.get_source_segment(source, functions["construct"])
    assert construction.index("replay_seed(") < construction.index('capture_logits(model, ids, spec, f"G_seed_')
    assert "center_delta(logits, f_logits)" in construction
    evaluation = ast.get_source_segment(source, functions["evaluate"])
    assert evaluation.index("validate_frozen(") < evaluation.index("load_evaluation_spec()")
    assert "center_delta(f_logits, b_logits)" in evaluation
    run = ast.get_source_segment(source, functions["run"])
    assert run.index("enter_barrier(") < run.index("evaluate(")
    # The exact pinned model/tokenizer loader forces local-only FP32 and eval.
    grad_source = r.path_of(S201["blind_helpers"]["gradient"]["path"]).read_text()
    grad_tree = ast.parse(grad_source)
    load = next(node for node in grad_tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_local_model")
    code = ast.get_source_segment(grad_source, load)
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in code and 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in code
    for node in ast.walk(load):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
            assert any(key.arg == "local_files_only" and isinstance(key.value, ast.Constant) and key.value.value is True for key in node.keywords)
