"""Attempt208 synthetic/static tests only; never load real models or tensor data."""
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
SOURCE = PROJECT / "scripts/ablation/run_greedy_ar8_semantic_logit_recovery_pilot.py"
loader = importlib.util.spec_from_file_location("test208", SOURCE)
r = importlib.util.module_from_spec(loader)
loader.loader.exec_module(r)
SPEC = r.load_spec()
EVALUATION = r.load_evaluation_spec()
c = r.import_pinned(SPEC["attempt201"]["constructor"], "test208_training")
S201 = c.load_spec()
gradient = c.import_pinned(S201["blind_helpers"]["gradient"], "test208_boundary")
previous = r.import_pinned(SPEC["operator_definition"]["runner"], "test208_reference207")

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
    own_root = tmp_path / "attempt208"
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


def test_exact_blind_plan_prompts_and_separate_source_pinned_evaluation():
    assert SPEC["prompts"] == ["The topic is", "This is about", "The central idea is", "In one word:",
                               "A concise label:", "The concept:", "This relates to", "The underlying theme:"]
    prior205 = json.loads((PROJECT / "experiments/attempts/205_self_distill_logit_amplification_recovery_pilot/spec.json").read_text())
    assert SPEC["prompts"] == prior205["prompts"]
    assert SPEC["tokenization"] == {"tokenizer_source": "F", "add_special_tokens": False,
                                   "chat_template": False, "minimum_tokens": 1}
    assert SPEC["contexts"]["shape"] == [64, 127]
    manifest201 = r.read_record(SPEC["attempt201"]["construction_manifest"])
    assert SPEC["contexts"]["raw_sha256"] == manifest201["blind_data"]["raw_sha256"]["contexts"]
    assert SPEC["candidate"]["tensor_order"] == [r.NAME]
    assert SPEC["candidate"]["shape"] == [8, 151936]
    assert SPEC["candidate"]["sign"] == "G-F only" and SPEC["candidate"]["aggregation"] is False
    assert SPEC["update"]["steps_per_nonzero_operator"] == 1
    assert SPEC["update"]["eligible_blocks"] == list(range(14)) and SPEC["update"]["eligible_matrix_count"] == 98
    assert r.sha256_file(r.SPEC_PATH) == r.SPEC_SHA256
    assert r.sha256_file(r.EVALUATION_SPEC_PATH) == r.EVALUATION_SPEC_SHA256
    text = json.dumps(SPEC)
    for forbidden in ("evaluation-spec", "models/base", "oracle", "CakeBake", "attempt203", "attempt204", "result207", ".136330"):
        assert forbidden not in text
    assert "evaluation" not in SPEC
    assert EVALUATION["oracle"]["sign"] == "F-B only"


def test_exact_pinned_207_source_function_and_operator_parity():
    r.validate_operator(SPEC)
    assert SPEC["operator_definition"]["definition"] == previous.load_spec()["operators"][2]
    assert r.RELATIVE_STEP == previous.RELATIVE_STEP == 7.8125e-5
    assert r.DISPLACEMENT_TOLERANCE == previous.DISPLACEMENT_TOLERANCE


@pytest.mark.parametrize("fault", ["step", "norm", "blocks", "batch", "teacher", "source"])
def test_operator_definition_drift_rejected(fault):
    spec = copy.deepcopy(SPEC)
    if fault == "step":
        spec["update"]["steps_per_nonzero_operator"] = 2
    elif fault == "norm":
        spec["update"]["relative_step"] *= 2
    elif fault == "blocks":
        spec["update"]["eligible_blocks"].pop()
    elif fault == "batch":
        spec["operator_definition"]["definition"]["loss_terms"] = 64
    elif fault == "teacher":
        spec["operator_definition"]["definition"]["continuation_length"] = 7
    else:
        spec["operator_definition"]["runner"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        r.validate_operator(spec)


def test_exact_greedy_8_even_after_EOS_and_loss_gradient_parity():
    contexts = torch.ones(64, 127, dtype=torch.int64)
    model = ToyModel().eval()
    model.config.eos_token_id = 0
    targets = r.greedy_continuations(model, contexts)
    assert torch.equal(targets, previous.greedy_continuations(ToyModel().eval(), contexts))
    assert targets.shape == (64, 8) and targets.numel() == 512 and targets.dtype == torch.int64
    assert torch.equal(targets, torch.zeros(64, 8, dtype=torch.int64))
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    gradient.freeze_other_parameters(model, eligible)
    before = r.ar8_loss(model, contexts, targets, backward=True)
    ours = [module.weight.grad.clone() for _, module in eligible]
    reference = ToyModel().eval()
    eligible_reference = gradient.discover_eligible_linear_weights(reference, torch)
    gradient.freeze_other_parameters(reference, eligible_reference)
    expected = previous.objective(reference, contexts, targets, "greedy_autoregressive_8", c, backward=True)
    assert before == expected
    assert all(torch.equal(grad, module.weight.grad) for grad, (_, module) in zip(ours, eligible_reference))


def test_teacher_forced_causal_shift_only_continuation_labels_float64():
    logits = torch.randn(8, 135, 32, dtype=torch.float32, requires_grad=True)
    targets = torch.arange(8).repeat(8, 1)
    loss = r.continuation_loss_sum(logits, targets)
    expected = -torch.log_softmax(logits[:, 126:134].double(), -1).gather(-1, targets[:, :, None]).sum()
    assert torch.equal(loss, expected) and loss.dtype == torch.float64
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :126]) == 0 and torch.count_nonzero(logits.grad[:, 134:]) == 0
    assert torch.count_nonzero(logits.grad[:, 126:134]) > 0


@pytest.mark.parametrize("after", [1., 1.1, float("nan"), float("inf")])
def test_strict_objective_decrease_gate(after):
    with pytest.raises(ValueError, match="strictly decreased"):
        r.require_decrease({"status": "updated", "updates": 1, "objective_before": 1.}, after)
    r.require_decrease({"status": "updated", "updates": 1, "objective_before": 1.}, .999999)


def test_frozen_201_full_audit_uses_exact_contexts_and_no_sampling_or_probe(frozen_artifacts, monkeypatch):
    spec, manifest, proxy = frozen_artifacts
    def forbidden(*args, **kwargs):
        pytest.fail("Frozen audit must never reconstruct samples or activation responses")
    for name in ("sample_table", "sample_probabilities", "fixed_contexts", "validate_frozen", "train_seed"):
        monkeypatch.setattr(c, name, forbidden)
    for name in ("mean_activation", "rehydrate", "endpoint_response"):
        monkeypatch.setattr(c.b, name, forbidden)
    _, data, validated, _ = r.validate_201(spec, proxy)
    assert validated == manifest and torch.equal(data["contexts"], torch.ones(64, 127, dtype=torch.int64))
    assert r.raw_hash(data["contexts"]) == spec["contexts"]["raw_sha256"]


@pytest.mark.parametrize("fault", ["contexts", "candidate", "receipt"])
def test_201_frozen_provenance_mismatch_aborts(frozen_artifacts, fault):
    spec, manifest, proxy = frozen_artifacts
    if fault == "receipt":
        spec["attempt201"]["freeze_receipt"]["sha256"] = "0" * 64
    else:
        if fault == "contexts":
            manifest["blind_data"]["raw_sha256"]["contexts"] = "0" * 64
        else:
            manifest["candidate"]["raw_sha256"][c.NAMES[0]] = "0" * 64
        repin_artifact_manifest(spec, manifest)
    with pytest.raises(ValueError):
        r.validate_201(spec, proxy)


def test_single_update_norm_sign_and_207_arithmetic():
    model = ToyModel().eval()
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    originals = {name: module.weight.detach().clone() for name, module in eligible}
    frozen = gradient.frozen_parameter_hashes(model, selected, torch)
    contexts = torch.ones(64, 127, dtype=torch.int64)
    teacher = r.greedy_continuations(model, contexts)
    before = r.ar8_loss(model, contexts, teacher, backward=True)
    gradients = {name: module.weight.grad.clone() for name, module in eligible}
    initial_norm = c.b.norm(originals.values())
    update = r.normalized_update(eligible, originals, initial_norm, before, c)
    assert update["updates"] == 1 and update["target_step_norm"] == 7.8125e-5 * initial_norm
    assert update["eta"] == update["target_step_norm"] / c.b.norm(gradients.values())
    assert math.isclose(update["realized_displacement_norm"], update["target_step_norm"], **r.DISPLACEMENT_TOLERANCE)
    for name, module in eligible:
        assert torch.equal(module.weight, (originals[name].double() - update["eta"] * gradients[name].double()).float())
    model.zero_grad(set_to_none=True)
    gradient.verify_frozen_parameters(model, selected, frozen, torch)
    r.require_decrease(update, r.ar8_loss(model, contexts, teacher))


def test_fixed_G_minus_F_and_float64_vocab_centering():
    f = torch.arange(8 * 32).reshape(8, 32).float()
    g = f + torch.arange(32).square().float()
    expected_delta = (g - f).double()
    expected = expected_delta - expected_delta.mean(-1, keepdim=True)
    assert torch.equal(r.center_delta(g, f), expected)
    assert r.center_delta(g, f).dtype == torch.float64
    assert torch.equal(r.center_delta(f, g), -expected)  # Math only; constructor sign is fixed.
    for bad in (torch.zeros(8, 32), torch.full((8, 32), float("nan"))):
        with pytest.raises(ValueError):
            r.require_tensor(bad, (8, 32), torch.float32)


def test_full_blind_construction_freezes_and_reloads_before_any_privileged_read(frozen_artifacts, monkeypatch, capsys):
    spec, _, proxy = frozen_artifacts
    opened = []
    original_open = Path.open
    def guard(path, *args, **kwargs):
        text = str(path)
        assert path != r.EVALUATION_SPEC_PATH
        assert not any(fragment in text for fragment in (
            "models/base", "downloaded-model-hashes", "merged-model-hashes",
            "203_cake", "204_hard", "oracle_adl", "/result.json"))
        opened.append(text)
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    teacher_calls, update_calls = [], []
    greedy, update = r.greedy_continuations, r.normalized_update
    def generate(*args):
        teacher_calls.append(1)
        return greedy(*args)
    def change(*args):
        update_calls.append(1)
        return update(*args)
    monkeypatch.setattr(r, "greedy_continuations", generate)
    monkeypatch.setattr(r, "normalized_update", change)
    receipt = r.construct(spec, "cpu", proxy)
    assert r.MARKER not in capsys.readouterr().out
    assert teacher_calls == [1] and update_calls == [1] and proxy.loaded == [spec["model_dir"]]
    def forbidden(*args, **kwargs):
        pytest.fail("Frozen reload must not regenerate teachers/train/probe")
    monkeypatch.setattr(r, "greedy_continuations", forbidden)
    monkeypatch.setattr(r, "ar8_loss", forbidden)
    response, manifest = r.validate_frozen(spec, receipt, proxy)
    assert response.shape == (8, 32) and response.dtype == torch.float32 and response.is_contiguous()
    assert manifest["update"]["updates"] == 1 and manifest["update"]["objective_after"] < manifest["update"]["objective_before"]
    assert manifest["teacher"]["target_count"] == 512 and manifest["noneligible_unchanged"] is True
    barrier = r.enter_barrier(spec, receipt, proxy)
    assert isinstance(barrier, r.FrozenBarrier) and r.MARKER in capsys.readouterr().out
    with pytest.raises(ValueError):
        r.FrozenBarrier(receipt)
    assert not Path(spec["paths"]["result"]).exists()


@pytest.mark.parametrize("fault", ["increase", "zero_gradient", "wrong_start", "noneligible"])
def test_failed_update_or_start_aborts_before_G_capture_and_candidate_publication(frozen_artifacts, monkeypatch, fault):
    spec, _, proxy = frozen_artifacts
    if fault == "increase":
        original_loss = r.ar8_loss
        monkeypatch.setattr(r, "ar8_loss", lambda *args, backward=False: original_loss(*args, backward=True)
                            if backward else 100.)
    elif fault == "zero_gradient":
        monkeypatch.setattr(r, "normalized_update", lambda *args: {"status": "zero_gradient", "updates": 0})
    else:
        real_verify = proxy.verify_seed_start
        def fail(model, eligible, *args):
            with torch.no_grad():
                if fault == "wrong_start":
                    eligible[0][1].weight.add_(1)
                else:
                    model.offset.add_(1)
            return real_verify(model, eligible, *args)
        proxy.verify_seed_start = fail
    capture = r.capture_logits
    def guarded(model, ids, spec, label):
        assert label != "G"
        return capture(model, ids, spec, label)
    monkeypatch.setattr(r, "capture_logits", guarded)
    with pytest.raises(ValueError):
        r.construct(spec, "cpu", proxy)
    assert not Path(spec["paths"]["candidate"]).exists()
    assert not Path(spec["paths"]["construction_manifest"]).exists()


def refreeze_manifest(spec, receipt, manifest):
    record = write_record(Path(spec["paths"]["construction_manifest"]), manifest)
    receipt["construction_manifest_sha256"] = record["sha256"]
    write_record(Path(spec["paths"]["freeze_receipt"]), receipt)


@pytest.mark.parametrize("fault", ["candidate_hash", "teacher", "updates", "no_decrease", "eta", "state", "F_hash", "contexts"])
def test_frozen_validation_tampering_aborts_before_privileged_read(frozen_artifacts, monkeypatch, fault):
    spec, _, proxy = frozen_artifacts
    receipt = r.construct(spec, "cpu", proxy)
    barrier = r.enter_barrier(spec, receipt, proxy)
    manifest = json.loads(Path(spec["paths"]["construction_manifest"]).read_text())
    if fault == "candidate_hash":
        manifest["candidate"]["raw_sha256"][r.NAME] = "0" * 64
    elif fault == "teacher":
        manifest["teacher"]["token_ids"][0][0] = 2
    elif fault == "updates":
        manifest["update"]["updates"] = 2
    elif fault == "no_decrease":
        manifest["update"]["objective_after"] = manifest["update"]["objective_before"]
    elif fault == "eta":
        manifest["update"]["eta"] *= 2
    elif fault == "state":
        manifest["final_eligible_state"]["aggregate_sha256"] = "0" * 64
    elif fault == "F_hash":
        manifest["F_logits_raw_sha256"].pop()
    else:
        manifest["contexts_raw_sha256"] = "0" * 64
    refreeze_manifest(spec, receipt, manifest)
    barrier.receipt = dict(receipt)
    monkeypatch.setattr(r, "load_evaluation_spec", lambda: pytest.fail("Failed freeze must block privileged reads"))
    with pytest.raises(ValueError):
        r.evaluate(spec, barrier, "cpu", proxy)


def test_exact_prompt_tokenization_and_last_token_readout():
    spec = {**SPEC, "vocabulary_size": 32}
    calls = []
    def tokenizer(text, *, add_special_tokens):
        calls.append((text, add_special_tokens))
        return {"input_ids": [1, 2, 3]}
    ids = r.prompt_ids(tokenizer, spec)
    assert calls == [(text, False) for text in SPEC["prompts"]]
    model = ToyModel().eval()
    captured = r.capture_logits(model, ids, spec, "toy")
    expected = model(input_ids=torch.tensor([[1, 2, 3]]), use_cache=False).logits[0, -1].expand(8, 32)
    assert torch.equal(captured, expected)
    with pytest.raises(ValueError):
        r.prompt_ids(lambda *args, **kwargs: {"input_ids": []}, spec)


def test_signed_per_prompt_mean_flattened_median_min_max_count_math():
    oracle = torch.tensor([[1., -1., 0.]] * 8, dtype=torch.float64)
    scales = torch.tensor([1., 2., -1., 3., -2., 4., 5., 6.])
    candidate = (oracle * scales[:, None]).float()
    result = r.cosine_metrics(candidate, oracle)
    assert result["prompt_cosines"] == pytest.approx([1., 1., -1., 1., -1., 1., 1., 1.])
    assert result["primary"] == pytest.approx(.5)
    assert result["count_prompt_cosines_gt_0"] == 6
    assert result["median_prompt_cosine"] == pytest.approx(1)
    assert result["min_prompt_cosine"] == pytest.approx(-1) and result["max_prompt_cosine"] == pytest.approx(1)
    assert result["flattened_cosine"] == pytest.approx(float((candidate.double() * oracle).sum() /
                                                           (candidate.double().norm() * oracle.norm())))
    with pytest.raises(ValueError):
        r.cosine_metrics(torch.zeros_like(candidate), oracle)


@pytest.mark.parametrize("primary,flat,positive,category", [
    (.18, .113, 7, "clear_gain"), (.17999, .12, 8, "positive_gain"),
    (.18, .11282103805624162, 8, "positive_gain"), (.18, .12, 6, "positive_gain"),
    (.14, .01, 6, "positive_gain"), (.1363304648383902, .12, 8, "no_gain"),
    (.15, 0., 8, "no_gain"), (.2, .2, 5, "no_gain")])
def test_fixed_comparison_constants_and_interpretation_gates(primary, flat, positive, category):
    ref = EVALUATION["attempt205_raw_mean_reference"]
    assert ref["primary"] == .1363304648383902 and ref["flattened"] == .11282103805624162
    assert ref["positive_prompts"] == ref["prompt_count"] == 8
    result = r.gains_and_interpretation({"primary": primary, "flattened_cosine": flat,
                                       "count_prompt_cosines_gt_0": positive}, EVALUATION)
    assert result["interpretation_label"] == category
    assert result["primary_gain_vs_attempt205"] == primary - ref["primary"]
    assert result["flattened_gain_vs_attempt205"] == flat - ref["flattened"]


def test_full_post_barrier_evaluation_F_reproduction_oracle_sign_and_publication(frozen_artifacts, monkeypatch, capsys):
    spec, _, proxy = frozen_artifacts
    receipt = r.construct(spec, "cpu", proxy)
    barrier = r.enter_barrier(spec, receipt, proxy)
    assert r.MARKER in capsys.readouterr().out
    evaluation = copy.deepcopy(EVALUATION)
    evaluation["models"] = {"F": spec["model_dir"], "B": "toy-base"}
    reads = []
    def privileged():
        reads.append("evaluation")
        return evaluation
    monkeypatch.setattr(r, "load_evaluation_spec", privileged)
    monkeypatch.setattr(r, "validate_privileged_inputs", lambda *_: {"synthetic": True})
    center = r.center_delta
    oracle_pairs = []
    def record_oracle(left, right):
        oracle_pairs.append((left.clone(), right.clone()))
        return center(left, right)
    monkeypatch.setattr(r, "center_delta", record_oracle)
    candidate_before = Path(spec["paths"]["candidate"]).read_bytes()
    result = r.evaluate(spec, barrier, "cpu", proxy)
    assert reads == ["evaluation", "evaluation"]
    assert proxy.loaded == [spec["model_dir"], spec["model_dir"], "toy-base"]
    f, b = oracle_pairs[0]
    assert len(oracle_pairs) == 1 and float(f[0, 0]) > float(b[0, 0])
    oracle = (f - b).double()
    oracle -= oracle.mean(-1, keepdim=True)
    assert r.raw_hash(oracle) == result["provenance"]["oracle_raw_float64_sha256"]
    assert result["provenance"]["F_logits_exactly_reproduced"] is True
    assert len(result["metrics"][r.NAME]["prompt_cosines"]) == 8
    assert json.loads(Path(spec["paths"]["result"]).read_text()) == result
    assert Path(spec["paths"]["candidate"]).read_bytes() == candidate_before
    with pytest.raises(FileExistsError):
        r.evaluate(spec, barrier, "cpu", proxy)


def test_F_hash_mismatch_aborts_before_B_load_or_cosine(frozen_artifacts, monkeypatch):
    spec, _, proxy = frozen_artifacts
    receipt = r.construct(spec, "cpu", proxy)
    barrier = r.enter_barrier(spec, receipt, proxy)
    evaluation = copy.deepcopy(EVALUATION)
    evaluation["models"] = {"F": spec["model_dir"], "B": "toy-base"}
    monkeypatch.setattr(r, "load_evaluation_spec", lambda: evaluation)
    monkeypatch.setattr(r, "validate_privileged_inputs", lambda *_: {})
    capture = r.capture_logits
    monkeypatch.setattr(r, "capture_logits", lambda *args: capture(*args) + .01)
    monkeypatch.setattr(r, "cosine_metrics", lambda *_: pytest.fail("F mismatch must block scoring"))
    with pytest.raises(ValueError, match="F logits differ"):
        r.evaluate(spec, barrier, "cpu", proxy)
    assert "toy-base" not in proxy.loaded and not Path(spec["paths"]["result"]).exists()


def test_final_revalidation_failure_blocks_result_publication(frozen_artifacts, monkeypatch):
    spec, _, proxy = frozen_artifacts
    receipt = r.construct(spec, "cpu", proxy)
    barrier = r.enter_barrier(spec, receipt, proxy)
    evaluation = copy.deepcopy(EVALUATION)
    evaluation["models"] = {"F": spec["model_dir"], "B": "toy-base"}
    monkeypatch.setattr(r, "load_evaluation_spec", lambda: evaluation)
    monkeypatch.setattr(r, "validate_privileged_inputs", lambda *_: {})
    real_validate = r.validate_frozen
    calls = []
    def changed(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("final input changed")
        return real_validate(*args)
    monkeypatch.setattr(r, "validate_frozen", changed)
    with pytest.raises(ValueError, match="final input"):
        r.evaluate(spec, barrier, "cpu", proxy)
    assert not Path(spec["paths"]["result"]).exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_exclusive_output_and_artifact_overwrite_refusal(tmp_path, symlink):
    output = tmp_path / "result.json"
    if symlink:
        output.symlink_to(tmp_path / "absent")
    else:
        output.write_text("preserve")
    with pytest.raises(FileExistsError):
        r.refuse_overwrite(output)
    with pytest.raises(FileExistsError):
        c.b.publish(output, {"overwrite": False})


def test_runtime_static_blind_call_graph_efficiency_and_privileged_order():
    tree = ast.parse(SOURCE.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    def call_names(node):
        return {call.func.attr if isinstance(call.func, ast.Attribute) else call.func.id
                for call in ast.walk(node) if isinstance(call, ast.Call) and
                isinstance(call.func, (ast.Name, ast.Attribute))}
    seen = set()
    def visit(name):
        if name in seen:
            return
        seen.add(name)
        for called in call_names(functions[name]):
            if called in functions:
                visit(called)
    visit("construct")
    visit("enter_barrier")
    assert not {"evaluate", "load_evaluation_spec", "validate_privileged_inputs", "cosine_metrics", "gains_and_interpretation"}.intersection(seen)
    all_calls = set.union(*(call_names(node) for node in functions.values()))
    assert not {"train_seed", "replay_seed", "armijo_step", "Adam", "SGD", "generate", "jvp", "mean_activation",
                "endpoint_response", "sample_table", "rehydrate", "load_dataset", "save", "score_model", "decode"}.intersection(all_calls)
    construct_calls = [node for node in ast.walk(functions["construct"]) if isinstance(node, ast.Call)]
    for name in ("greedy_continuations", "normalized_update"):
        assert sum(isinstance(node.func, ast.Name) and node.func.id == name for node in construct_calls) == 1
    text = ast.get_source_segment(SOURCE.read_text(), functions["construct"])
    assert 'center_delta(g_logits, f_logits)' in text and "for seed" not in text
    assert 'center_delta(f_logits, b_logits)' in ast.get_source_segment(SOURCE.read_text(), functions["evaluate"])
    evaluation = functions["evaluate"]
    validate_line = next(node.lineno for node in ast.walk(evaluation) if isinstance(node, ast.Call) and
                         isinstance(node.func, ast.Name) and node.func.id == "validate_frozen")
    privileged_line = next(node.lineno for node in ast.walk(evaluation) if isinstance(node, ast.Call) and
                           isinstance(node.func, ast.Name) and node.func.id == "load_evaluation_spec")
    assert validate_line < privileged_line
    # Source/spec207 are definitions only: never dynamically import that module.
    assert 'import_pinned(spec["operator_definition"]' not in SOURCE.read_text()
    for forbidden in ("CakeBake", "attempt203", "attempt204", "Attempt207 result", "Attempt205 result"):
        assert forbidden not in SOURCE.read_text()
