"""Attempt209 synthetic/static tests only; never load real models or tensor data."""
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
SOURCE = PROJECT / "scripts/ablation/run_clipped_soft_semantic_recovery_stepsize_pilot.py"
loader = importlib.util.spec_from_file_location("test209", SOURCE)
r = importlib.util.module_from_spec(loader)
loader.loader.exec_module(r)
SPEC = r.load_spec()
EVALUATION = r.load_evaluation_spec()
c = r.import_pinned(SPEC["attempt201"]["constructor"], "test209_training")
S201 = c.load_spec()
gradient = c.import_pinned(S201["blind_helpers"]["gradient"], "test209_boundary")
previous = r.import_pinned(SPEC["operator_definition"]["runner"], "test209_reference207")



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


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def write_record(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return {"path": str(path), "sha256": r.sha256_file(path)}


def metadata_manifest():
    """Synthetic archival records for the frozen audit; no descendant replay."""
    model = ToyModel().eval()
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    initial = c.b.state_hashes(eligible)
    frozen = gradient.frozen_parameter_hashes(model, selected, torch)
    wnorm = c.b.norm(module.weight for _, module in eligible)
    target = c.RELATIVE_STEP * wnorm
    step = {"initial_ce": 1., "Wnorm": wnorm, "gradient_norm": 1., "target_step_norm": target,
            "eta_max": target, "trials": [{"trial_index": 0, "eta": target, "ce": .5,
              "armijo_rhs": 1. - 1e-4 * target, "accepted": True}],
            "accepted_updates": 1, "accepted_eta": target, "accepted_post_step_ce": .5,
            "displacement_norm": target, "displacement_relative_to_initial_Wnorm": target / wnorm,
            "eligible_before": initial, "eligible_after": initial}
    seeds = {str(seed): {"initial_eligible_state": initial,
             "steps": [{**step, "step": index, "cumulative_displacement_norm": index * target,
                        "cumulative_displacement_relative_to_initial_Wnorm": index * target / wnorm}
                       for index in range(1, 5)], "final_eligible_state": initial,
             "noneligible_unchanged": True, "descendant_mean_float64_sha256": "a" * 64}
             for seed in range(8)}
    return {"initial_eligible_state": initial, "initial_Wnorm": wnorm,
            "noneligible_F_parameter_sha256": frozen, "seeds": seeds,
            "runtime": {"torch": str(torch.__version__), "device": "cpu-synthetic"}, "vocabulary_size": 32}


@pytest.fixture
def frozen_artifacts(tmp_path, monkeypatch):
    """Small vocabulary, synthetic fixed-shape artifacts; no real data reads."""
    base_manifest = metadata_manifest()
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
    spec["model_dir"] = s201["paths"]["model_dir"]
    spec["final_checkpoint_files"] = s201["final_checkpoint_files"]
    spec["vocabulary_size"] = 32
    spec["candidate"]["shape"] = [8, 32]
    spec["contexts"]["raw_sha256"] = c.b.raw_hash(data["contexts"])
    spec["contexts"]["probabilities_raw_sha256"] = c.b.raw_hash(data["probabilities"])
    spec["contexts"]["probabilities_shape"] = [64, 32]
    own_root = tmp_path / "attempt209"
    own_root.mkdir()
    spec["paths"] = {"candidate": str(own_root / "candidate.pt"),
                     "construction_manifest": str(own_root / "construction-manifest.json"),
                     "freeze_receipt": str(own_root / "freeze-receipt.json"),
                     "result": str(own_root / "result.json")}
    previous_spec = r.read_record(SPEC["operator_definition"]["spec"])
    previous_spec["model"]["path"] = spec["model_dir"]
    previous_spec["model"]["final_checkpoint_files"] = spec["final_checkpoint_files"]
    spec["operator_definition"]["spec"] = write_record(tmp_path / "operator207.json", previous_spec)
    spec_path = own_root / "spec.json"
    record = write_record(spec_path, spec)
    monkeypatch.setattr(r, "SPEC_PATH", spec_path)
    monkeypatch.setattr(r, "SPEC_SHA256", record["sha256"])
    def tokenizer(prompt, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [1, spec["prompts"].index(prompt) + 2]}
    loaded = []
    def load_local_model(path, device, torch_module):
        assert device == "cpu"
        loaded.append(str(path))
        model = ToyModel().eval()
        if str(path) != spec["model_dir"]:
            with torch.no_grad():
                model.offset.sub_(.05)
        return model, tokenizer
    loader = SimpleNamespace(**{name: getattr(gradient, name) for name in (
        "discover_eligible_linear_weights", "freeze_other_parameters", "verify_frozen_parameters", "frozen_parameter_hashes")},
        load_local_model=load_local_model)
    proxy.import_pinned = lambda *_: loader
    proxy.verify_seed_start = c.verify_seed_start
    proxy.loaded = loaded
    return spec, manifest, proxy


def repin_artifact_manifest(spec, manifest):
    spec["attempt201"]["construction_manifest"] = write_record(
        r.path_of(spec["attempt201"]["construction_manifest"]["path"]), manifest)
    receipt = r.read_record(spec["attempt201"]["freeze_receipt"])
    receipt["construction_manifest_sha256"] = spec["attempt201"]["construction_manifest"]["sha256"]
    spec["attempt201"]["freeze_receipt"] = write_record(r.path_of(spec["attempt201"]["freeze_receipt"]["path"]), receipt)


def test_exact_fixed_blind_plan_and_source_pinned_privileged_spec():
    assert r.FACTORS == (.5, .25, .125) and r.C1 == 1e-4 and r.TAU == 1e-4
    assert SPEC["update"]["factors"] == [.5, .25, .125]
    assert SPEC["update"]["target_relative_displacements"] == [3.90625e-5, 1.953125e-5, 9.765625e-6]
    assert SPEC["update"]["gradient_passes"] == SPEC["update"]["accepted_steps"] == 1
    assert SPEC["update"]["eligible_blocks"] == list(range(14)) and SPEC["update"]["eligible_matrix_count"] == 98
    assert SPEC["candidate"]["tensor_order"] == [r.NAME] and SPEC["candidate"]["sign"] == "G-F only"
    assert SPEC["prompts"] == ["The topic is", "This is about", "The central idea is", "In one word:",
                               "A concise label:", "The concept:", "This relates to", "The underlying theme:"]
    assert SPEC["tokenization"] == {"tokenizer_source": "F", "add_special_tokens": False,
                                   "chat_template": False, "minimum_tokens": 1}
    assert SPEC["teacher"]["student_temperature"] == 1 and SPEC["teacher"]["device_target_dtype"] == "float64"
    assert r.sha256_file(r.SPEC_PATH) == r.SPEC_SHA256
    assert r.sha256_file(r.EVALUATION_SPEC_PATH) == r.EVALUATION_SPEC_SHA256
    text = json.dumps(SPEC)
    for forbidden in ("evaluation-spec", "models/base", "oracle", "CakeBake", "attempt203", "attempt204", "prior_results", ".136330"):
        assert forbidden not in text
    assert "evaluation" not in SPEC and EVALUATION["oracle"]["sign"] == "F-B only"
    original = r.read_record(SPEC["attempt201"]["construction_manifest"])
    assert SPEC["contexts"]["raw_sha256"] == original["blind_data"]["raw_sha256"]["contexts"]
    assert SPEC["contexts"]["probabilities_raw_sha256"] == original["blind_data"]["raw_sha256"]["probabilities"]


def test_pinned_207_clipped_teacher_source_and_numerical_parity():
    r.validate_operator(SPEC)
    assert SPEC["operator_definition"]["definition"] == previous.load_spec()["operators"][4]
    p = torch.tensor([.9, .09985, .0001, .00005], dtype=torch.float64).repeat(64, 1)
    q, diagnostics = r.soft_teacher(p, "clipped_soft_T1_tau1e4")
    old_q, old_diagnostics = previous.soft_teacher(p, "clipped_soft_T1_tau1e4")
    assert torch.equal(q, old_q) and diagnostics == old_diagnostics
    assert q.dtype == torch.float64 and q.device.type == "cpu" and q.is_contiguous()
    assert diagnostics["retained_mass"] == pytest.approx([.99995] * 64)
    assert diagnostics["retained_support_size"] == [3] * 64
    assert diagnostics["retained_mass_summary"] == dict.fromkeys(("mean", "median", "min", "max"), pytest.approx(.99995))
    assert diagnostics["retained_support_size_summary"] == dict.fromkeys(("mean", "median", "min", "max"), 3)
    assert torch.equal(q[:, -1], torch.zeros(64, dtype=torch.float64))
    assert torch.allclose(q[:, :3], p[:, :3] / .99995, atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="support"):
        r.soft_teacher(torch.zeros_like(p), "clipped_soft_T1_tau1e4")


@pytest.mark.parametrize("fault", ["tau", "blocks", "norm", "source", "teacher"])
def test_operator_definition_mismatch_aborts(fault):
    spec = copy.deepcopy(SPEC)
    if fault == "tau":
        spec["operator_definition"]["definition"]["tau"] *= 2
    elif fault == "blocks":
        spec["update"]["eligible_blocks"].pop()
    elif fault == "norm":
        spec["update"]["relative_step"] *= 2
    elif fault == "source":
        spec["operator_definition"]["runner"]["sha256"] = "0" * 64
    else:
        spec["operator_definition"]["definition"]["teacher"] = "altered"
    with pytest.raises(ValueError):
        r.validate_operator(spec)


def test_native_T1_soft_objective_and_gradient_match_207_exactly():
    model = ToyModel().eval()
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    gradient.freeze_other_parameters(model, eligible)
    contexts = torch.ones(64, 127, dtype=torch.int64)
    p = torch.full((64, 32), 1 / 32, dtype=torch.float64)
    q, _ = r.soft_teacher(p, "clipped_soft_T1_tau1e4")
    before = r.clipped_loss(model, contexts, q, backward=True)
    ours = [module.weight.grad.clone() for _, module in eligible]
    reference = ToyModel().eval()
    old_eligible = gradient.discover_eligible_linear_weights(reference, torch)
    gradient.freeze_other_parameters(reference, old_eligible)
    expected = previous.objective(reference, contexts, q, "clipped_soft_T1_tau1e4", c, backward=True)
    assert before == expected
    assert all(torch.equal(g, module.weight.grad) for g, (_, module) in zip(ours, old_eligible))
    logits = torch.tensor([[.2, -.3, 1.]], requires_grad=True)
    native = torch.softmax(logits.detach().double(), -1)
    r.soft_loss_sum(logits, native).backward()
    assert logits.grad.abs().max() < 1e-14


def step_inputs():
    model = ToyModel().eval()
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    originals = {name: module.weight.detach().clone() for name, module in eligible}
    grads = {name: torch.full_like(module.weight, .01 * (index + 1)) for index, (name, module) in enumerate(eligible)}
    frozen = metadata_manifest()
    return model, eligible, selected, originals, grads, c.b.norm(grads.values()), frozen


@pytest.mark.parametrize("accepted_index", [0, 1, 2])
def test_fixed_descending_factors_same_gradient_exact_restoration_and_first_accept(monkeypatch, accepted_index):
    model, eligible, selected, originals, grads, gnorm, frozen = step_inputs()
    before_gradient = r.inventory_hash(grads)
    states, restored = [], []
    original_restore = r.restore_F
    def restore(*args):
        original_restore(*args)
        restored.append(c.b.state_hashes(eligible))
        assert all(p.grad is None for p in model.parameters())
    monkeypatch.setattr(r, "restore_F", restore)
    def loss():
        index = len(states)
        states.append(c.b.state_hashes(eligible))
        eta = r.FACTORS[index] * r.RELATIVE_STEP * frozen["initial_Wnorm"] / gnorm
        for name, module in eligible:
            assert torch.equal(module.weight, (originals[name].double() - eta * grads[name].double()).float())
        assert r.inventory_hash(grads) == before_gradient
        return 1.1 if index < accepted_index else .9
    update = r.select_step(model, eligible, selected, originals, grads, gnorm, 1., frozen, c, gradient, loss)
    assert len(states) == accepted_index + 1 and len(update["trials"]) == accepted_index + 1
    assert update["accepted_factor"] == r.FACTORS[accepted_index]
    assert update["gradient_passes"] == update["accepted_steps"] == 1
    assert restored == [frozen["initial_eligible_state"]] * (2 * accepted_index + 1)
    assert r.inventory_hash(grads) == before_gradient
    for index, trial in enumerate(update["trials"]):
        assert trial["factor"] == r.FACTORS[index]
        assert trial["armijo_rhs"] == 1. - 1e-4 * trial["eta"] * gnorm ** 2
        assert trial["accepted"] is (index == accepted_index)
        assert trial["gradient_snapshot_sha256"] == before_gradient["aggregate_sha256"]
    assert math.isclose(update["realized_displacement_norm"], update["target_step_norm"], **r.DISPLACEMENT_TOLERANCE)
    r.validate_update(update, frozen)


def test_Armijo_boundary_equality_is_accepted():
    model, eligible, selected, originals, grads, gnorm, frozen = step_inputs()
    eta = .5 * r.RELATIVE_STEP * frozen["initial_Wnorm"] / gnorm
    rhs = 1. - r.C1 * eta * gnorm ** 2
    update = r.select_step(model, eligible, selected, originals, grads, gnorm, 1., frozen, c, gradient, lambda: rhs)
    assert update["trials"][0]["accepted"] is True


def test_all_three_rejected_restores_exact_F_and_fails():
    model, eligible, selected, originals, grads, gnorm, frozen = step_inputs()
    calls = []
    def loss():
        calls.append(1)
        return 1.
    with pytest.raises(ValueError, match="All three"):
        r.select_step(model, eligible, selected, originals, grads, gnorm, 1., frozen, c, gradient, loss)
    assert len(calls) == 3 and c.b.state_hashes(eligible) == frozen["initial_eligible_state"]
    gradient.verify_frozen_parameters(model, selected, frozen["noneligible_F_parameter_sha256"], torch)


@pytest.mark.parametrize("fault", ["noneligible", "gradient", "weights", "nan_loss"])
def test_trial_side_effects_or_invalid_loss_fail_and_restore(fault):
    model, eligible, selected, originals, grads, gnorm, frozen = step_inputs()
    def loss():
        if fault == "noneligible":
            with torch.no_grad():
                model.offset.add_(1)
        elif fault == "gradient":
            next(iter(grads.values())).add_(1)
        elif fault == "weights":
            with torch.no_grad():
                eligible[0][1].weight.add_(1)
        return float("nan") if fault == "nan_loss" else .9
    with pytest.raises(ValueError):
        r.select_step(model, eligible, selected, originals, grads, gnorm, 1., frozen, c, gradient, loss)
    assert c.b.state_hashes(eligible) == frozen["initial_eligible_state"]


@pytest.mark.parametrize("fault", ["zero", "nan", "missing"])
def test_zero_or_invalid_gradient_never_produces_candidate(fault):
    model, eligible, *_ = step_inputs()
    for _, module in eligible:
        module.weight.grad = torch.zeros_like(module.weight)
    if fault == "nan":
        eligible[0][1].weight.grad.fill_(float("nan"))
    elif fault == "missing":
        eligible[0][1].weight.grad = None
    with pytest.raises(ValueError):
        r.freeze_gradient(eligible, c)


def test_exact_frozen_201_contexts_probabilities_hash_audit_without_probe_or_sampling(frozen_artifacts, monkeypatch):
    spec, manifest, proxy = frozen_artifacts
    def forbidden(*args, **kwargs):
        pytest.fail("No sampling or activation recomputation")
    for name in ("sample_table", "sample_probabilities", "fixed_contexts", "validate_frozen", "train_seed"):
        monkeypatch.setattr(c, name, forbidden)
    for name in ("mean_activation", "rehydrate", "endpoint_response"):
        monkeypatch.setattr(c.b, name, forbidden)
    _, data, observed, _ = r.validate_201(spec, proxy)
    assert observed == manifest and set(data) == {"contexts", "probabilities"}
    assert torch.equal(data["contexts"], torch.ones(64, 127, dtype=torch.int64))
    q, diagnostics = r.teacher_from_data(spec, data)
    assert q.shape == (64, 32) and diagnostics["retained_support_size"] == [32] * 64
    for key in data:
        assert r.raw_hash(data[key]) == manifest["blind_data"]["raw_sha256"][key]


@pytest.mark.parametrize("key", ["contexts", "probabilities"])
def test_changed_frozen_data_rejected_before_teacher_or_gradient(frozen_artifacts, key):
    spec, _, proxy = frozen_artifacts
    _, data, _, _ = r.validate_201(spec, proxy)
    data[key] = data[key].clone()
    data[key][0, 0] += 1
    with pytest.raises(ValueError, match="frozen"):
        r.teacher_from_data(spec, data)


def test_full_blind_pipeline_one_gradient_and_reload_before_any_privileged_read(frozen_artifacts, monkeypatch, capsys):
    spec, _, proxy = frozen_artifacts
    real_open = Path.open
    def guard(path, *args, **kwargs):
        text = str(path)
        assert path != r.EVALUATION_SPEC_PATH
        assert not any(fragment in text for fragment in ("models/base", "203_cake", "204_hard", "oracle_adl",
                                                         "downloaded-model-hashes", "merged-model-hashes", "/result.json"))
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    objective, calls, accepted_reads = r.clipped_loss, [], []
    def record_loss(*args, backward=False):
        calls.append(backward)
        return objective(*args, backward=backward)
    monkeypatch.setattr(r, "clipped_loss", record_loss)
    capture = r.capture_logits
    def record_capture(*args):
        accepted_reads.append(args[-1])
        return capture(*args)
    monkeypatch.setattr(r, "capture_logits", record_capture)
    receipt = r.construct(spec, "cpu", proxy)
    assert r.MARKER not in capsys.readouterr().out
    assert calls[0] is True and calls.count(True) == 1 and 1 <= calls.count(False) <= 3
    assert proxy.loaded == [spec["model_dir"]] and accepted_reads == ["F", "accepted G"]
    response, manifest = r.validate_frozen(spec, receipt, proxy)
    assert calls.count(True) == 1 and response.shape == (8, 32) and response.dtype == torch.float32 and response.is_contiguous()
    assert manifest["update"]["accepted_steps"] == 1 and manifest["noneligible_unchanged"] is True
    assert manifest["update"]["trials"][-1]["accepted"] is True
    barrier = r.enter_barrier(spec, receipt, proxy)
    assert isinstance(barrier, r.FrozenBarrier) and r.MARKER in capsys.readouterr().out
    with pytest.raises(ValueError):
        r.FrozenBarrier(receipt)


def test_all_rejected_constructor_never_captures_G_freezes_or_accesses_B(frozen_artifacts, monkeypatch):
    spec, _, proxy = frozen_artifacts
    real_loss = r.clipped_loss
    monkeypatch.setattr(r, "clipped_loss", lambda *args, backward=False: real_loss(*args, backward=True) if backward else 100.)
    real_capture = r.capture_logits
    def capture(*args):
        assert args[-1] == "F"
        return real_capture(*args)
    monkeypatch.setattr(r, "capture_logits", capture)
    monkeypatch.setattr(r, "load_evaluation_spec", lambda: pytest.fail("No privileged reads without accepted candidate"))
    with pytest.raises(ValueError, match="All three"):
        r.construct(spec, "cpu", proxy)
    assert proxy.loaded == [spec["model_dir"]]
    assert not any(Path(spec["paths"][key]).exists() for key in ("candidate", "construction_manifest", "freeze_receipt", "result"))


def refreeze_manifest(spec, receipt, manifest):
    record = write_record(Path(spec["paths"]["construction_manifest"]), manifest)
    receipt["construction_manifest_sha256"] = record["sha256"]
    write_record(Path(spec["paths"]["freeze_receipt"]), receipt)


@pytest.mark.parametrize("fault", ["candidate", "teacher", "diagnostics", "gradient_count", "factor", "rhs", "gradient_hash", "F_hash", "final_state"])
def test_frozen_tampering_aborts_every_privileged_read(frozen_artifacts, monkeypatch, fault):
    spec, _, proxy = frozen_artifacts
    receipt = r.construct(spec, "cpu", proxy)
    barrier = r.enter_barrier(spec, receipt, proxy)
    manifest = json.loads(Path(spec["paths"]["construction_manifest"]).read_text())
    if fault == "candidate":
        manifest["candidate"]["raw_sha256"][r.NAME] = "0" * 64
    elif fault == "teacher":
        manifest["teacher"]["raw_sha256"] = "0" * 64
    elif fault == "diagnostics":
        manifest["teacher"]["diagnostics"]["retained_mass"][0] *= .5
    elif fault == "gradient_count":
        manifest["update"]["gradient_passes"] = 2
    elif fault == "factor":
        manifest["update"]["trials"][0]["factor"] = .25
    elif fault == "rhs":
        manifest["update"]["trials"][0]["armijo_rhs"] += 1
    elif fault == "gradient_hash":
        manifest["update"]["gradient_snapshot_hashes"]["aggregate_sha256"] = "0" * 64
    elif fault == "F_hash":
        manifest["F_logits_raw_sha256"].pop()
    else:
        manifest["final_eligible_state"]["aggregate_sha256"] = "0" * 64
    refreeze_manifest(spec, receipt, manifest)
    barrier.receipt = dict(receipt)
    monkeypatch.setattr(r, "load_evaluation_spec", lambda: pytest.fail("Failed freeze must block evaluation-spec"))
    monkeypatch.setattr(r, "validate_prior_results", lambda *_: pytest.fail("Failed freeze must block prior results"))
    with pytest.raises(ValueError):
        r.evaluate(spec, barrier, "cpu", proxy)


def test_G_minus_F_centering_prompt_tokens_and_signed_cosine_math():
    f = torch.arange(8 * 32).reshape(8, 32).float()
    g = f + torch.arange(32).square().float()
    delta = (g - f).double()
    delta -= delta.mean(-1, keepdim=True)
    assert torch.equal(r.center_delta(g, f), delta) and r.center_delta(g, f).dtype == torch.float64
    ids_called = []
    def tokenizer(prompt, *, add_special_tokens):
        ids_called.append((prompt, add_special_tokens))
        return {"input_ids": [1, 2]}
    spec = {**SPEC, "vocabulary_size": 32}
    assert r.prompt_ids(tokenizer, spec) == [[1, 2]] * 8
    assert ids_called == [(prompt, False) for prompt in SPEC["prompts"]]
    oracle = torch.tensor([[1., -1., 0.]] * 8, dtype=torch.float64)
    scales = torch.tensor([1., 2., -1., 3., -2., 4., 5., 6.])
    candidate = (oracle * scales[:, None]).float()
    metrics = r.cosine_metrics(candidate, oracle)
    assert metrics["primary"] == pytest.approx(.5) and metrics["count_prompt_cosines_gt_0"] == 6
    assert metrics["prompt_cosines"] == pytest.approx([1., 1., -1., 1., -1., 1., 1., 1.])
    assert metrics["flattened_cosine"] == pytest.approx(float((candidate.double() * oracle).sum() /
                                                            (candidate.double().norm() * oracle.norm())))
    assert metrics["median_prompt_cosine"] == pytest.approx(1.)
    assert metrics["min_prompt_cosine"] == pytest.approx(-1.) and metrics["max_prompt_cosine"] == pytest.approx(1.)


def test_exact_pinned_prior_results_and_descriptive_comparisons():
    prior = r.validate_prior_results(EVALUATION)
    assert prior["attempt205_raw_mean"]["primary"] == .1363304648383902
    assert prior["attempt205_individual_seed_primary_distribution"]["mean"] == .07144584491123232
    assert len(prior["attempt205_seed_primaries"]) == 8
    metrics = {"primary": .2, "flattened_cosine": .16, "count_prompt_cosines_gt_0": 7}
    result = r.comparisons_and_label(metrics, prior)
    assert result["primary_difference_vs_attempt205_raw_mean"] == .2 - prior["attempt205_raw_mean"]["primary"]
    assert result["primary_difference_vs_attempt205_mean_individual_seed"] == .2 - prior["attempt205_individual_seed_primary_distribution"]["mean"]
    assert result["primary_difference_vs_attempt208_ar8"] == .2 - prior["attempt208_ar8"]["primary"]
    assert result["flattened_difference_vs_attempt205_raw_mean"] == .16 - prior["attempt205_raw_mean"]["flattened_cosine"]
    assert result["flattened_difference_vs_attempt208_ar8"] == .16 - prior["attempt208_ar8"]["flattened_cosine"]


@pytest.mark.parametrize("fault", ["hash", "provenance", "summary"])
def test_prior_result_pin_provenance_or_summary_failure(tmp_path, fault):
    evaluation = copy.deepcopy(EVALUATION)
    if fault == "hash":
        evaluation["prior_results"]["208"]["sha256"] = "0" * 64
    else:
        result = r.read_record(evaluation["prior_results"]["208"])
        if fault == "provenance":
            result["provenance"]["F_logits_exactly_reproduced"] = False
        else:
            result["metrics"]["response_greedy_ar8"]["primary"] += 1
        record = write_record(tmp_path / "prior.json", result)
        evaluation["prior_results"]["208"].update(record)
    with pytest.raises(ValueError):
        r.validate_prior_results(evaluation)


@pytest.mark.parametrize("primary,flat,positive,mean,label", [
    (.18, .15, 7, .071, "clear_single_endpoint_gain"),
    (.17999, .16, 8, .071, "promising_single_endpoint"),
    (.18, .14999, 8, .071, "promising_single_endpoint"),
    (.19, .16, 6, .071, "promising_single_endpoint"),
    (.11, .001, 6, .071, "promising_single_endpoint"),
    (.10, .16, 8, .071, "weak_or_negative"),
    (.11, .1, 8, .11, "weak_or_negative"),
    (.19, 0, 8, .071, "weak_or_negative"),
    (.2, .2, 5, .071, "weak_or_negative")])
def test_fixed_development_label_boundaries(primary, flat, positive, mean, label):
    prior = {"attempt205_raw_mean": {"primary": .13, "flattened_cosine": .11},
             "attempt205_individual_seed_primary_distribution": {"mean": mean},
             "attempt208_ar8": {"primary": .14, "flattened_cosine": .12}}
    result = r.comparisons_and_label({"primary": primary, "flattened_cosine": flat,
                                     "count_prompt_cosines_gt_0": positive}, prior)
    assert result["development_label"] == label


def evaluation_for_toy(spec, monkeypatch):
    evaluation = copy.deepcopy(EVALUATION)
    evaluation["models"] = {"F": spec["model_dir"], "B": "toy-base"}
    monkeypatch.setattr(r, "load_evaluation_spec", lambda: evaluation)
    monkeypatch.setattr(r, "validate_privileged_inputs", lambda *_: {"synthetic": True})
    return evaluation


def test_full_post_barrier_F_reproduction_prior_reads_oracle_sign_and_publication(frozen_artifacts, monkeypatch, capsys):
    spec, _, proxy = frozen_artifacts
    receipt = r.construct(spec, "cpu", proxy)
    barrier = r.enter_barrier(spec, receipt, proxy)
    assert r.MARKER in capsys.readouterr().out
    evaluation = evaluation_for_toy(spec, monkeypatch)
    prior_calls = []
    read_prior = r.validate_prior_results
    def record_prior(*args):
        prior_calls.append(1)
        return read_prior(*args)
    monkeypatch.setattr(r, "validate_prior_results", record_prior)
    center, pairs = r.center_delta, []
    def record_oracle(left, right):
        pairs.append((left.clone(), right.clone()))
        return center(left, right)
    monkeypatch.setattr(r, "center_delta", record_oracle)
    candidate_bytes = Path(spec["paths"]["candidate"]).read_bytes()
    result = r.evaluate(spec, barrier, "cpu", proxy)
    assert prior_calls == [1, 1] and proxy.loaded == [spec["model_dir"], spec["model_dir"], "toy-base"]
    f, b = pairs[0]
    expected = (f - b).double()
    expected -= expected.mean(-1, keepdim=True)
    assert len(pairs) == 1 and r.raw_hash(expected) == result["provenance"]["oracle_raw_float64_sha256"]
    assert result["provenance"]["F_logits_exactly_reproduced"] is True
    assert len(result["metrics"][r.NAME]["prompt_cosines"]) == 8
    assert json.loads(Path(spec["paths"]["result"]).read_text()) == result
    assert Path(spec["paths"]["candidate"]).read_bytes() == candidate_bytes
    with pytest.raises(FileExistsError):
        r.evaluate(spec, barrier, "cpu", proxy)


def test_post_barrier_F_mismatch_prevents_B_load_and_scoring(frozen_artifacts, monkeypatch):
    spec, _, proxy = frozen_artifacts
    receipt = r.construct(spec, "cpu", proxy)
    barrier = r.enter_barrier(spec, receipt, proxy)
    evaluation_for_toy(spec, monkeypatch)
    capture = r.capture_logits
    monkeypatch.setattr(r, "capture_logits", lambda *args: capture(*args) + .01)
    monkeypatch.setattr(r, "cosine_metrics", lambda *_: pytest.fail("F mismatch must block scoring"))
    with pytest.raises(ValueError, match="F logits differ"):
        r.evaluate(spec, barrier, "cpu", proxy)
    assert "toy-base" not in proxy.loaded and not Path(spec["paths"]["result"]).exists()


def test_final_provenance_revalidation_failure_blocks_publication(frozen_artifacts, monkeypatch):
    spec, _, proxy = frozen_artifacts
    receipt = r.construct(spec, "cpu", proxy)
    barrier = r.enter_barrier(spec, receipt, proxy)
    evaluation_for_toy(spec, monkeypatch)
    validate = r.validate_frozen
    calls = []
    def changed(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("final source/input changed")
        return validate(*args)
    monkeypatch.setattr(r, "validate_frozen", changed)
    with pytest.raises(ValueError, match="final source/input"):
        r.evaluate(spec, barrier, "cpu", proxy)
    assert not Path(spec["paths"]["result"]).exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_output_and_frozen_artifact_overwrite_refusal(tmp_path, symlink):
    path = tmp_path / "result.json"
    if symlink:
        path.symlink_to(tmp_path / "absent")
    else:
        path.write_text("preserve")
    with pytest.raises(FileExistsError):
        r.refuse_overwrite(path)
    with pytest.raises(FileExistsError):
        c.b.publish(path, {})


def test_static_blind_call_graph_gradient_count_privileged_order_and_efficiency():
    text = SOURCE.read_text()
    tree = ast.parse(text)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    def names(node):
        return {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    visited = set()
    def visit(name):
        if name in visited:
            return
        visited.add(name)
        for called in names(functions[name]):
            if called in functions:
                visit(called)
    visit("construct")
    visit("enter_barrier")
    assert not {"load_evaluation_spec", "validate_privileged_inputs", "validate_prior_results", "evaluate", "cosine_metrics"}.intersection(visited)
    all_calls = set.union(*(names(n) for n in functions.values()))
    assert not {"replay_seed", "train_seed", "generate", "mean_activation", "jvp", "patchscope", "sample_table", "rehydrate",
                "load_dataset", "score_model", "save", "Adam", "SGD", "decode"}.intersection(all_calls)
    construct = functions["construct"]
    backwards = [n for n in ast.walk(construct) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "clipped_loss" and any(k.arg == "backward" and isinstance(k.value, ast.Constant)
                                                       and k.value.value is True for k in n.keywords)]
    assert len(backwards) == 1 and "backward=True" not in ast.get_source_segment(text, functions["select_step"])
    assert 'center_delta(g_logits, f_logits)' in ast.get_source_segment(text, construct)
    assert 'center_delta(f_logits, b_logits)' in ast.get_source_segment(text, functions["evaluate"])
    evaluate = functions["evaluate"]
    first_validate = min(n.lineno for n in ast.walk(evaluate) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                         and n.func.id == "validate_frozen")
    first_privileged = min(n.lineno for n in ast.walk(evaluate) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                           and n.func.id in ("load_evaluation_spec", "validate_prior_results", "validate_privileged_inputs"))
    assert first_validate < first_privileged
    assert 'import_pinned(spec["operator_definition"]' not in text
    for forbidden in ("CakeBake", "attempt203", "attempt204"):
        assert forbidden not in text
    loader = (PROJECT / S201["blind_helpers"]["gradient"]["path"]).read_text()
    assert "local_files_only=True" in loader and '"HF_HUB_OFFLINE"' in loader and '"TRANSFORMERS_OFFLINE"' in loader
