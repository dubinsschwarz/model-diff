"""Attempt206 synthetic/static tests only; no real model, dataset or GPU work."""
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
SOURCE = PROJECT / "scripts/ablation/run_self_distill_logit_recovery_disjoint_contexts.py"
loader = importlib.util.spec_from_file_location("test206", SOURCE)
r = importlib.util.module_from_spec(loader)
loader.loader.exec_module(r)
SPEC = r.load_spec()
EVAL = r.load_evaluation_spec()
c = r.import_pinned(SPEC["attempt201"]["constructor"], "test206_blind201")
S201 = c.load_spec()
gradient = c.import_pinned(S201["blind_helpers"]["gradient"], "test206_eligible")


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
    corpus = torch.ones(4096, 128, dtype=torch.int64)
    corpus[64:128, 0] = torch.arange(64) % 30 + 2
    corpus[64:128, 126] = 3
    data["corpus"] = corpus
    manifest["blind_data"] = {"raw_sha256": {"corpus": r.raw_hash(corpus)}}
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
    corpus[64:128, 0] = torch.arange(64) % 30 + 2
    corpus[64:128, 126] = 3
    data = {"corpus": corpus, "probe": torch.ones(10000, 128, dtype=torch.int64),
            "contexts": corpus[:64, :127].contiguous(),
            "targets": torch.zeros(8, 64, dtype=torch.int64),
            "probabilities": torch.full((64, 32), 1 / 32, dtype=torch.float64),
            "F_mean": torch.ones(128, 2048, dtype=torch.float64)}
    s201["source"]["raw_tensor_sha256"] = c.b.raw_hash(data["corpus"])
    spec["probe"]["source_raw_sha256"] = r.raw_hash(data["corpus"])
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


def test_operator_equivalence_all_prior_pins_and_information_separation():
    prior = r.validate_operator(SPEC)
    assert SPEC["replay"] == prior["replay"]
    assert SPEC["candidate"]["sign"] == prior["candidate"]["sign"] == "G-F only"
    assert SPEC["candidate"]["raw_mean"] == prior["candidate"]["raw_mean"]
    assert SPEC["candidate"]["shape"] == [64, 151936]
    assert SPEC["readout"]["batch_size"] == 8 and SPEC["readout"]["position"] == "input token position 126"
    assert SPEC["probe"]["rows"] == [64, 128] and SPEC["probe"]["token_positions"] == [0, 127]
    assert SPEC["probe"]["training_rows"] == [0, 64]
    assert list(SPEC["attempt205_operator"]) == ["spec", "constructor"]
    assert r.sha256_file(r.SPEC_PATH) == r.SPEC_SHA256
    assert r.sha256_file(r.EVALUATION_SPEC_PATH) == r.EVALUATION_SPEC_SHA256
    for group in (SPEC["attempt205_operator"], EVAL["attempt205_prior_evidence"]):
        for record in group.values():
            assert r.sha256_file(r.path_of(record["path"])) == record["sha256"]
    blind = r.SPEC_PATH.read_text()
    for forbidden in ("evaluation-spec", "models/base", "downloaded-model-hashes", "merged-model-hashes",
                      "prior_reference", "0.136330", "203_", "204_", "CakeBake", "oracle_adl"):
        assert forbidden not in blind
    assert "evaluation" not in SPEC and "prior_reference" not in SPEC
    assert set(EVAL["attempt205_prior_evidence"]) == {"spec", "evaluation_spec", "constructor", "construction_manifest", "result"}


@pytest.mark.parametrize("field", ["accepted_steps", "eligible_matrix_count", "seed_ids", "sign", "raw_mean", "seed_consensus"])
def test_changed_operator_fails(field):
    spec = copy.deepcopy(SPEC)
    if field in spec["replay"]:
        spec["replay"][field] = 3 if field != "seed_ids" else [0]
    else:
        spec["candidate"][field] = "altered"
    with pytest.raises(ValueError, match="operator-definition"):
        r.validate_operator(spec)


def test_exact_probe_slice_hash_row_disjointness_and_source_provenance():
    corpus = torch.arange(4096 * 128, dtype=torch.int64).reshape(4096, 128).contiguous()
    data = {"corpus": corpus, "contexts": corpus[:64, :127].contiguous()}
    spec = copy.deepcopy(SPEC)
    spec["probe"]["source_raw_sha256"] = r.raw_hash(corpus)
    frozen = {"blind_data": {"raw_sha256": {"corpus": r.raw_hash(corpus)}}}
    probe, record = r.disjoint_probe(data, spec, frozen)
    assert torch.equal(probe, corpus[64:128, :127])
    assert probe.shape == (64, 127) and probe.dtype == torch.int64 and probe.is_contiguous()
    assert not torch.equal(probe, data["contexts"])
    assert not bool((probe[:, None, :] == data["contexts"][None, :, :]).all(-1).any())
    assert record["raw_sha256"] == c.b.raw_hash(probe)
    assert record["source_corpus_raw_sha256"] == r.raw_hash(corpus)
    assert record["training_contexts_raw_sha256"] == r.raw_hash(data["contexts"])
    assert record["training_probe_disjoint"] and record["rows"] == [64, 128]


@pytest.mark.parametrize("fault", ["all_equal", "one_overlap", "source_hash", "dtype", "training_mismatch"])
def test_overlapping_or_invalid_probe_fails_without_replacement(trajectory, fault):
    original, manifest = trajectory
    data = {key: value.clone() for key, value in original.items()}
    frozen = copy.deepcopy(manifest)
    spec = copy.deepcopy(SPEC)
    if fault == "all_equal":
        data["corpus"][64:128, :127] = data["contexts"]
    elif fault == "one_overlap":
        data["corpus"][97, :127] = data["contexts"][3]
    elif fault == "dtype":
        data["corpus"] = data["corpus"].float()
    elif fault == "training_mismatch":
        data["contexts"][0, 0] += 1
    spec["probe"]["source_raw_sha256"] = r.raw_hash(data["corpus"])
    frozen["blind_data"]["raw_sha256"]["corpus"] = r.raw_hash(data["corpus"])
    if fault == "source_hash":
        spec["probe"]["source_raw_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        r.disjoint_probe(data, spec, frozen)


def test_fixed_eight_inference_batches_position126_order_eval_float32_no_amp(trajectory):
    data, manifest = trajectory
    spec = {**SPEC, "vocabulary_size": 32, "probe": {**SPEC["probe"], "source_raw_sha256": r.raw_hash(data["corpus"])}}
    probe, _ = r.disjoint_probe(data, spec, manifest)
    calls = []
    class PositionModel(ToyModel):
        def forward(self, input_ids, use_cache):
            assert not use_cache and not self.training and not torch.is_grad_enabled()
            assert not torch.is_autocast_enabled("cpu")
            calls.append(input_ids.clone())
            positions = torch.arange(127).float()[None, :, None]
            row_value = input_ids[:, :1, None].float()
            return SimpleNamespace(logits=(positions + row_value).expand(8, 127, 32).contiguous())
    logits = r.capture_logits(PositionModel(), probe, spec, "synthetic")
    assert len(calls) == 8 and torch.equal(torch.cat(calls), probe)
    assert logits.shape == (64, 32) and logits.dtype == torch.float32 and logits.is_contiguous()
    expected = (probe[:, :1].float() + 126).expand(64, 32)
    assert torch.equal(logits, expected)


@pytest.mark.parametrize("fault", ["batch_size", "vocab", "shape", "dtype", "nonfinite"])
def test_fixed_readout_rejects_invalid_semantics(trajectory, fault):
    data, _ = trajectory
    spec = copy.deepcopy(SPEC)
    spec["vocabulary_size"] = 32
    model = ToyModel()
    contexts = data["corpus"][64:128, :127].contiguous()
    if fault == "batch_size":
        spec["readout"]["batch_size"] = 4
    elif fault == "vocab":
        model.config.vocab_size += 1
    else:
        forward = model.forward
        def bad(*args, **kwargs):
            values = forward(*args, **kwargs).logits.clone()
            if fault == "shape":
                values = values[:, :-1]
            elif fault == "dtype":
                values = values.double()
            else:
                values[0, -1, 0] = float("nan")
            return SimpleNamespace(logits=values)
        model.forward = bad
    with pytest.raises(ValueError):
        r.capture_logits(model, contexts, spec, "synthetic")


def test_G_minus_F_float32_subtraction_then_float64_vocabulary_centering():
    f = torch.linspace(-1, 2, 64 * 32, dtype=torch.float32).reshape(64, 32)
    g = f + torch.arange(32, dtype=torch.float32)[None]
    g[:, 0] = 1e8
    delta = (g - f).double()
    expected = delta - delta.mean(-1, keepdim=True)
    candidate = r.center_delta(g, f)
    assert candidate.dtype == torch.float64 and torch.equal(candidate, expected)
    assert torch.allclose(candidate.mean(-1), torch.zeros(64, dtype=torch.float64), atol=1e-9, rtol=0)
    with pytest.raises(ValueError):
        r.center_delta(g[:8], f[:8])
    with pytest.raises(ValueError):
        r.center_delta(f, f)


def test_64_context_mean_consensus_and_blind_geometry():
    base = torch.arange(32, dtype=torch.float32) - 15.5
    responses = {f"response_seed_{seed}": torch.stack([(seed + 1) * (row + 1) * base.roll(seed + row) for row in range(64)])
                 for seed in range(8)}
    stack = torch.stack(list(responses.values())).double()
    unit = stack / stack.norm(dim=-1, keepdim=True)
    mean, consensus, geometry = r.combine_responses(responses)
    assert torch.equal(mean, stack.mean(0).float())
    assert torch.equal(consensus, unit.mean(0).float())
    assert bool((consensus.double().norm(dim=-1) < .999).all())
    pairwise = torch.einsum("spv,tpv->pst", unit, unit)
    index = torch.triu_indices(8, 8, 1)
    pairs = pairwise[:, index[0], index[1]].mean(-1).tolist()
    concentration = unit.mean(0).norm(dim=-1).tolist()
    assert geometry == {"mean_pairwise_cosine_per_context": pairs,
                        "consensus_concentration_per_context": concentration,
                        "pairwise_cosine_distribution": r.summary(pairs), "concentration_distribution": r.summary(concentration)}
    with pytest.raises(ValueError):
        r.combine_responses({name: value[:8] for name, value in responses.items()})


def test_all_eight_seeds_exact_four_step_replay_and_noneligible_unchanged(trajectory):
    data, manifest = trajectory
    for seed in range(8):
        model = ToyModel().eval()
        eligible, selected, replay = r.replay_seed(model, seed, data, manifest, c, gradient)
        assert replay["steps"] == manifest["seeds"][str(seed)]["steps"]
        assert replay["accepted_steps"] == 4 and replay["exact_original_F_start"]
        assert replay["final_eligible_state"] == manifest["seeds"][str(seed)]["final_eligible_state"]
        assert len(eligible) == 98 and {int(name.split(".")[2]) for name, _ in eligible} == set(range(14))
        assert gradient.frozen_parameter_hashes(model, selected, torch) == manifest["noneligible_F_parameter_sha256"]


def test_frozen201_audit_no_sampling_or_activation_and_retains_corpus(frozen_artifacts, monkeypatch):
    spec, manifest, proxy = frozen_artifacts
    def forbidden(*args, **kwargs):
        pytest.fail("No sampling or activation reconstruction allowed")
    for name in ("sample_table", "sample_probabilities", "validate_frozen"):
        monkeypatch.setattr(c, name, forbidden)
    for name in ("mean_activation", "endpoint_response", "rehydrate"):
        monkeypatch.setattr(c.b, name, forbidden)
    _, data, validated, _ = r.validate_201(spec, proxy)
    assert set(data) == {"corpus", "contexts", "targets"} and validated == manifest
    probe, record = r.disjoint_probe(data, spec, manifest)
    assert record["raw_sha256"] == r.raw_hash(probe)


@pytest.fixture
def synthetic_construction(monkeypatch, tmp_path, trajectory):
    data, frozen = trajectory
    manifest = copy.deepcopy(frozen)
    spec, s201 = copy.deepcopy(SPEC), copy.deepcopy(S201)
    spec["vocabulary_size"] = spec["candidate"]["shape"][1] = 32
    spec["probe"]["source_raw_sha256"] = r.raw_hash(data["corpus"])
    spec["model_dir"] = s201["paths"]["model_dir"] = str(tmp_path / "F")
    spec["paths"] = {"candidate": str(tmp_path / "candidate.pt"),
                     "construction_manifest": str(tmp_path / "construction-manifest.json"),
                     "freeze_receipt": str(tmp_path / "freeze-receipt.json"), "result": str(tmp_path / "result.json")}
    prior = r.read_record(SPEC["attempt205_operator"]["spec"])
    prior["vocabulary_size"] = prior["candidate"]["shape"][1] = 32
    prior["model_dir"] = spec["model_dir"]
    spec["attempt205_operator"]["spec"] = write_record(tmp_path / "prior-blind-spec.json", prior)
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
        return model, SimpleNamespace()  # No tokenizer call or text reconstruction.
    monkeypatch.setattr(gradient, "load_local_model", load)
    capture = r.capture_logits
    def capturing(*args):
        events.append("capture:" + args[-1])
        return capture(*args)
    monkeypatch.setattr(r, "capture_logits", capturing)
    original_log = r.log
    def logging(message):
        events.append(message)
        original_log(message)
    monkeypatch.setattr(r, "log", logging)
    return spec, manifest, events


def freeze_toy(spec):
    receipt = r.construct(spec, "cpu", c)
    return receipt, r.enter_barrier(spec, receipt, c)


def test_no_privileged_reads_before_freeze_and_probe_hash_in_manifest(synthetic_construction, monkeypatch):
    spec, _, events = synthetic_construction
    privileged_paths = {r.EVALUATION_SPEC_PATH.resolve()}
    privileged_paths |= {r.path_of(EVAL["attempt205_prior_evidence"][key]["path"]).resolve()
                         for key in ("evaluation_spec", "construction_manifest", "result")}
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert path.resolve() not in privileged_paths
        assert "models/base" not in str(path)
        assert not any(value in str(path) for value in ("203_", "204_", "oracle_adl", "downloaded-model-hashes"))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    monkeypatch.setattr(r, "load_evaluation_spec", lambda: pytest.fail("No privileged spec"))
    receipt, barrier = freeze_toy(spec)
    responses, manifest, probe = r.validate_frozen(spec, receipt, c)
    assert events.count("load:F") == 9
    assert [event for event in events if event.startswith("capture:")] == ["capture:F"] + [f"capture:G_seed_{seed}" for seed in range(8)]
    assert events.count(r.MARKER) == 1 and isinstance(barrier, r.FrozenBarrier)
    assert list(responses) == list(r.NAMES)
    assert all(value.shape == (64, 32) and value.dtype == torch.float32 for value in responses.values())
    assert manifest["probe"]["raw_sha256"] == r.raw_hash(probe)
    assert manifest["probe"]["training_probe_disjoint"] and len(manifest["F_logits_raw_sha256"]) == 64
    assert all(len(hashes) == 64 for hashes in manifest["G_logits_raw_sha256"].values())
    with pytest.raises(FileExistsError):
        r.construct(spec, "cpu", c)


@pytest.mark.parametrize("fault", ["intermediate", "final", "initial_eligible", "initial_noneligible"])
def test_bad_replay_fails_before_G_logits(synthetic_construction, monkeypatch, fault):
    spec, manifest, events = synthetic_construction
    if fault == "intermediate":
        manifest["seeds"]["0"]["steps"][1]["initial_ce"] += 1
    elif fault == "final":
        record = manifest["seeds"]["0"]
        record["final_eligible_state"] = copy.deepcopy(record["final_eligible_state"])
        record["final_eligible_state"]["aggregate_sha256"] = "0" * 64
    else:
        load = gradient.load_local_model
        count = 0
        def wrong(*args):
            nonlocal count
            count += 1
            model, tokenizer = load(*args)
            if count == 2:
                with torch.no_grad():
                    parameter = model.offset if fault == "initial_noneligible" else model.model.layers[0].proj0.weight
                    parameter.add_(.1)
            return model, tokenizer
        monkeypatch.setattr(gradient, "load_local_model", wrong)
    with pytest.raises(ValueError):
        r.construct(spec, "cpu", c)
    assert not any(event.startswith("capture:G") for event in events)
    assert not Path(spec["paths"]["candidate"]).exists() and r.MARKER not in events


@pytest.mark.parametrize("fault", ["probe_hash", "raw_candidate_hash", "geometry", "replay", "F_hash_count"])
def test_frozen_artifacts_independent_validation(synthetic_construction, fault):
    spec, _, _ = synthetic_construction
    receipt = r.construct(spec, "cpu", c)
    path = Path(spec["paths"]["construction_manifest"])
    manifest = json.loads(path.read_text())
    if fault == "probe_hash":
        manifest["probe"]["raw_sha256"] = "0" * 64
    elif fault == "raw_candidate_hash":
        manifest["candidate"]["raw_sha256"]["response_seed_0"] = "0" * 64
    elif fault == "geometry":
        manifest["geometry"]["mean_pairwise_cosine_per_context"][0] += .1
    elif fault == "replay":
        manifest["replays"]["0"]["accepted_steps"] = 3
    else:
        manifest["F_logits_raw_sha256"].pop()
    path.write_text(json.dumps(manifest))
    receipt["construction_manifest_sha256"] = r.sha256_file(path)
    Path(spec["paths"]["freeze_receipt"]).write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        r.validate_frozen(spec, receipt, c)


def test_failed_freeze_blocks_every_privileged_read_and_marker(synthetic_construction, monkeypatch):
    spec, _, events = synthetic_construction
    receipt = r.construct(spec, "cpu", c)
    loads = []
    original_load = torch.load
    def reload(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        loads.append(str(path))
        return original_load(path, **kwargs)
    monkeypatch.setattr(torch, "load", reload)
    barrier = r.enter_barrier(spec, receipt, c)
    assert loads == [spec["paths"]["candidate"]]
    monkeypatch.setattr(r, "load_evaluation_spec", lambda: pytest.fail("No privileged read allowed"))
    Path(spec["paths"]["candidate"]).write_bytes(b"corrupt")
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


def test_64_context_signed_cosines_primary_median_and_flattened_math():
    oracle = torch.zeros(64, 32, dtype=torch.float64)
    oracle[:, 0] = 1
    candidate = oracle.clone()
    candidate[:20] *= -1
    candidate[20:24, 0] = 0
    candidate[20:24, 1] = 1
    candidate[-1] *= 10
    metrics = r.cosine_metrics(candidate.float(), oracle)
    assert metrics["context_cosines"] == [-1.] * 20 + [0.] * 4 + [1.] * 40
    assert metrics["primary"] == 20 / 64 and metrics["median_context_cosine"] == 1.
    assert metrics["min_context_cosine"] == -1 and metrics["max_context_cosine"] == 1
    assert metrics["count_context_cosines_gt_0"] == 40
    assert metrics["flattened_cosine"] == pytest.approx(29 / math.sqrt(163 * 64))
    with pytest.raises(ValueError):
        r.cosine_metrics(candidate[:63], oracle[:63])


def test_prior_evidence_is_exact_and_descriptive():
    expected = {"raw_mean_primary": .1363304648383902, "raw_mean_flattened": .11282103805624162,
                "unit_consensus_primary": .1269215049591151, "individual_seed_primary_mean": .07144584491123232,
                "individual_seed_count_positive": 7, "individual_seed_count": 8}
    assert r.validate_prior_evidence(SPEC, EVAL) == EVAL["prior_reference"] == expected
    wrong = copy.deepcopy(EVAL)
    wrong["prior_reference"]["raw_mean_primary"] += .001
    with pytest.raises(ValueError, match="prior reference"):
        r.validate_prior_evidence(SPEC, wrong)


def replication_inputs():
    metrics = {"response_raw_mean": {"primary": .2, "median_context_cosine": .1,
                                    "count_context_cosines_gt_0": 40, "flattened_cosine": .1},
               "response_seed_consensus": {"primary": .1}}
    seed_summary = {"mean": .05, "count_gt_0": 6}
    return metrics, seed_summary


def test_fixed_all_A_G_broad_and_partial_replication_boundaries():
    metrics, seeds = replication_inputs()
    result = r.replication(metrics, seeds, EVAL)
    assert result["checks"] == {key: True for key in "ABCDEFG"}
    assert result["category"] == "broad_replication"
    for field, value in (("count_context_cosines_gt_0", 39),):
        changed = copy.deepcopy(metrics)
        changed["response_raw_mean"][field] = value
        result = r.replication(changed, seeds, EVAL)
        assert not result["checks"]["C"] and result["category"] == "partial_replication"
    result = r.replication(metrics, {**seeds, "count_gt_0": 5}, EVAL)
    assert not result["checks"]["F"] and result["category"] == "partial_replication"
    result = r.replication(metrics, {**seeds, "mean": .2}, EVAL)
    assert not result["checks"]["G"] and result["category"] == "partial_replication"


@pytest.mark.parametrize("key,field", [("A", "primary"), ("B", "median_context_cosine"),
                                      ("D", "flattened_cosine"), ("E", "primary")])
def test_no_replication_when_any_required_positive_check_is_zero(key, field):
    metrics, seeds = replication_inputs()
    metrics["response_seed_consensus" if key == "E" else "response_raw_mean"][field] = 0.
    result = r.replication(metrics, seeds, EVAL)
    assert not result["checks"][key] and result["category"] == "no_replication"
    assert set(result) == {"checks", "category", "rules"}


@pytest.fixture
def synthetic_evaluation(synthetic_construction, monkeypatch, tmp_path):
    spec, manifest, events = synthetic_construction
    evaluation = copy.deepcopy(EVAL)
    evaluation["models"] = {"F": spec["model_dir"], "B": str(tmp_path / "B")}
    path = tmp_path / "evaluation-spec.json"
    record = write_record(path, evaluation)
    monkeypatch.setattr(r, "EVALUATION_SPEC_PATH", path)
    monkeypatch.setattr(r, "EVALUATION_SPEC_SHA256", record["sha256"])
    def privileged(*args):
        assert r.MARKER in events
        events.append("privileged_inputs")
        return {"synthetic_checkpoint_checks": True}
    def prior(*args):
        assert r.MARKER in events
        events.append("prior_evidence")
        return evaluation["prior_reference"]
    monkeypatch.setattr(r, "validate_privileged_inputs", privileged)
    monkeypatch.setattr(r, "validate_prior_evidence", prior)
    evaluation_load = r.load_evaluation_spec
    def read():
        assert r.MARKER in events
        events.append("read_evaluation_spec")
        return evaluation_load()
    monkeypatch.setattr(r, "load_evaluation_spec", read)
    return spec, manifest, events


def test_complete_synthetic_freeze_then_evaluation_64_scores_and_categories(synthetic_evaluation):
    spec, _, events = synthetic_evaluation
    _, barrier = freeze_toy(spec)
    before = Path(spec["paths"]["candidate"]).read_bytes()
    result = r.evaluate(spec, barrier, "cpu", c)
    assert events.index(r.MARKER) < events.index("read_evaluation_spec") < events.index("prior_evidence") < events.index("load:B")
    assert events.index("capture:post-freeze F") < events.index("load:B")
    assert list(result["metrics"]) == list(r.NAMES)
    assert all(len(value["context_cosines"]) == 64 for value in result["metrics"].values())
    assert result["context_source_rows"] == list(range(64, 128))
    assert result["raw_mean_metrics"] == result["metrics"]["response_raw_mean"]
    assert result["unit_consensus_metrics"] == result["metrics"]["response_seed_consensus"]
    assert result["replication"] == r.replication(result["metrics"], result["seed_primary_distribution"], EVAL)
    assert json.loads(Path(spec["paths"]["result"]).read_text()) == result
    assert Path(spec["paths"]["candidate"]).read_bytes() == before
    assert result["provenance"]["F_logits_exactly_reproduced"] and result["provenance"]["B_accessed_only_after_freeze"]
    with pytest.raises(FileExistsError):
        r.evaluate(spec, barrier, "cpu", c)


def test_F_hash_mismatch_aborts_before_B_and_cosine_metrics(synthetic_evaluation, monkeypatch):
    spec, _, events = synthetic_evaluation
    _, barrier = freeze_toy(spec)
    load = gradient.load_local_model
    def altered(*args):
        model, tokenizer = load(*args)
        with torch.no_grad():
            model.offset.add_(.1)
        return model, tokenizer
    monkeypatch.setattr(gradient, "load_local_model", altered)
    monkeypatch.setattr(r, "cosine_metrics", lambda *args: pytest.fail("No candidate scoring"))
    with pytest.raises(ValueError, match="F logit hashes"):
        r.evaluate(spec, barrier, "cpu", c)
    assert "load:B" not in events and not Path(spec["paths"]["result"]).exists()


def test_oracle_sign_F_minus_B_and_centering(synthetic_evaluation, monkeypatch):
    spec, _, _ = synthetic_evaluation
    _, barrier = freeze_toy(spec)
    observed = []
    score = r.cosine_metrics
    def record(candidate, oracle):
        observed.append(oracle.clone())
        return score(candidate, oracle)
    monkeypatch.setattr(r, "cosine_metrics", record)
    r.evaluate(spec, barrier, "cpu", c)
    _, _, probe = r.validate_frozen(spec, barrier.receipt, c)
    f = r.capture_logits(ToyModel().eval(), probe, spec, "synthetic reference F")
    base = ToyModel().eval()
    with torch.no_grad():
        base.offset.add_(.01)
    b = r.capture_logits(base, probe, spec, "synthetic reference B")
    expected = (f - b).double()
    expected -= expected.mean(dim=-1, keepdim=True)
    assert len(observed) == 10 and all(torch.equal(value, expected) for value in observed)


def test_final_revalidation_failure_refuses_publication(synthetic_evaluation, monkeypatch):
    spec, _, _ = synthetic_evaluation
    _, barrier = freeze_toy(spec)
    validate = r.validate_frozen
    calls = 0
    def fail_final(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("input/source changed")
        return validate(*args)
    monkeypatch.setattr(r, "validate_frozen", fail_final)
    with pytest.raises(ValueError, match="input/source changed"):
        r.evaluate(spec, barrier, "cpu", c)
    assert not Path(spec["paths"]["result"]).exists()


def test_output_refusal_existing_file_dangling_symlink_and_exclusive_open(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("existing")
    with pytest.raises(FileExistsError):
        r.refuse_overwrite(path)
    path.unlink()
    path.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        r.refuse_overwrite(path)
    assert 'result_path.open("x"' in SOURCE.read_text()


def test_static_blind_call_graph_and_no_selection_text_diagnostics_or_network():
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
    assert not reached.intersection({"evaluate", "load_evaluation_spec", "validate_prior_evidence", "validate_privileged_inputs", "replication", "cosine_metrics"})
    blind = "\n".join(ast.get_source_segment(source, functions[name]) for name in reached)
    for forbidden in ("models/base", "EVALUATION_SPEC", "evaluation-spec.json", "prior_reference", "CakeBake", "cake", "bake", "203_", "204_"):
        assert forbidden not in blind
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls.intersection({"generate", "decode", "apply_chat_template", "sample_table", "sample_probabilities",
                                  "inverse_cdf", "hash_uniform", "rehydrate", "load_dataset", "mean_activation",
                                  "endpoint_response", "save_pretrained", "snapshot_download", "hf_hub_download",
                                  "argsort", "argmax"})
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not imports.intersection({"datasets", "requests", "urllib", "httpx", "socket", "huggingface_hub", "subprocess"})
    assert "top_tokens" not in source and "semantic_substrings" not in source and "best_seed" not in source
    construction = ast.get_source_segment(source, functions["construct"])
    assert construction.index("replay_seed(") < construction.index('capture_logits(model, probe, spec, f"G_seed_')
    assert "center_delta(logits, f_logits)" in construction
    evaluation = ast.get_source_segment(source, functions["evaluate"])
    assert evaluation.index("validate_frozen(") < evaluation.index("load_evaluation_spec()")
    assert "center_delta(f_logits, b_logits)" in evaluation
    assert "for seed in SEEDS" in construction and "for name in NAMES" in evaluation
    # Exact pinned model loading still forces local-only FP32 and offline flags.
    grad_source = r.path_of(S201["blind_helpers"]["gradient"]["path"]).read_text()
    grad_tree = ast.parse(grad_source)
    load = next(node for node in grad_tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_local_model")
    code = ast.get_source_segment(grad_source, load)
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in code and 'os.environ["TRANSFORMERS_OFFLINE"] = "1"' in code
    for node in ast.walk(load):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
            assert any(key.arg == "local_files_only" and isinstance(key.value, ast.Constant) and key.value.value is True for key in node.keywords)
