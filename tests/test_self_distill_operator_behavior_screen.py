"""Attempt207 only: synthetic CPU and static provenance tests, no real models/data."""
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
SOURCE = PROJECT / "scripts/ablation/run_self_distill_operator_behavior_screen.py"
loader = importlib.util.spec_from_file_location("test207", SOURCE)
r = importlib.util.module_from_spec(loader)
loader.loader.exec_module(r)
SPEC = r.load_spec()
audit = r.import_pinned(SPEC["frozen_audit_helper"], "test207_audit")
c = r.import_pinned(SPEC["attempt201"]["constructor"], "test207_training")
a = r.import_pinned(SPEC["attempt203"]["runner"], "test207_likelihood")
S201 = c.load_spec()
gradient = c.import_pinned(S201["blind_helpers"]["gradient"], "test207_eligible")


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for k in range(7):
            self.add_module(f"proj{k}", torch.nn.Linear(1, 1, bias=False))
            with torch.no_grad():
                getattr(self, f"proj{k}").weight.fill_(.1)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block() for _ in range(28)])
        self.offset = torch.nn.Parameter(torch.tensor(.1))
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=28, vocab_size=4, use_cache=False)
        self.calls = []

    def forward(self, input_ids, use_cache):
        assert not use_cache and not self.training and not torch.is_autocast_enabled("cpu")
        self.calls.append(input_ids.detach().clone())
        weights = [linear.weight[0, 0] for block in self.model.layers[:14] for linear in block.children()]
        value = torch.stack(weights).sum() / 100 + self.offset
        logits = torch.stack((value, -value, value * 0, value * 2))
        return SimpleNamespace(logits=logits.expand(*input_ids.shape, 4))


def toy_inputs():
    model = Toy().eval()
    eligible = gradient.discover_eligible_linear_weights(model, torch)
    selected = gradient.freeze_other_parameters(model, eligible)
    originals = {name: module.weight.detach().clone() for name, module in eligible}
    manifest = {"initial_eligible_state": c.b.state_hashes(eligible),
                "noneligible_F_parameter_sha256": gradient.frozen_parameter_hashes(model, selected, torch),
                "initial_Wnorm": c.b.norm(originals.values()), "vocabulary_size": 4}
    p = torch.softmax(torch.tensor([.4, -.2, .1, .9], dtype=torch.float64), 0).repeat(64, 1)
    data = {"contexts": torch.ones((64, 127), dtype=torch.int64), "probabilities": p,
            "targets": torch.stack([torch.full((64,), seed % 4, dtype=torch.int64) for seed in range(8)])}
    historical = {"pairs": [{"index": k, "margin_F": .2 + k, "margin_B": .1 + k} for k in range(7)],
                  "aggregates": {"H": [.1] * 7}, "tokenization": list(range(7)),
                  "prefix": "synthetic", "facts": list(range(7))}
    return model, eligible, selected, originals, manifest, data, historical


def write_record(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return {"path": str(path), "sha256": r.sha256_file(path)}


def test_exact_inventory_pins_scope_and_common_rules():
    assert r.OPERATORS == ("hard_sample_T1", "greedy_next_token", "greedy_autoregressive_8",
                           "sharpened_soft_T07", "clipped_soft_T1_tau1e4")
    assert r.sha256_file(r.SPEC_PATH) == r.SPEC_SHA256
    for group in ("attempt201", "attempt203"):
        for record in SPEC[group].values():
            if isinstance(record, dict) and "path" in record:
                # Do not read the real scratch tensor artifacts in unit tests.
                if record["path"].endswith(".pt"):
                    continue
                r.require_hash(r.path_of(record["path"]), record["sha256"])
    r.require_hash(r.path_of(SPEC["frozen_audit_helper"]["path"]), SPEC["frozen_audit_helper"]["sha256"])
    assert SPEC["update"]["eligible_blocks"] == list(range(14))
    assert SPEC["update"]["eligible_matrix_count"] == 98
    assert SPEC["update"]["relative_step"] == 7.8125e-5
    assert SPEC["data"]["contexts_shape"] == [64, 127]
    assert SPEC["data"]["batch_size"] == 8
    assert SPEC["scientific_metadata"]["same_model_specimen_method_development"]
    assert not SPEC["scientific_metadata"]["clean_held_out_validation"]
    assert "subsequent fresh frozen" in SPEC["scientific_metadata"]["future_operator_choice"]
    assert not SPEC["scientific_metadata"]["best_operator_selection"]
    assert SPEC["model"]["final_checkpoint_files"] == S201["final_checkpoint_files"]
    assert SPEC["soft_target_numerics"]["device_target_dtype"] == "float64"


def test_frozen_203_audit_exact_seven_facts_axis_and_scores():
    s203, historical = audit.validate_203(SPEC, a)
    assert historical["facts"] == s203["facts"] and len(historical["facts"]) == 7
    assert historical["prefix"] == s203["prefix"]
    assert historical["aggregates"]["fact_count"] == 7
    assert historical["aggregates"]["count_H_gt_0"] == 7
    assert historical["aggregates"]["count_H_lt_0"] == 0
    assert historical["aggregates"]["H"] == [row["margin_F"] - row["margin_B"] for row in historical["pairs"]]


@pytest.mark.parametrize("fault", ["H", "fact_count", "count_H_gt_0", "count_H_lt_0", "facts", "prefix"])
def test_203_corrupt_frozen_axis_rejected(tmp_path, fault):
    spec = copy.deepcopy(SPEC)
    historical = r.read_record(SPEC["attempt203"]["result"])
    if fault == "H":
        historical["aggregates"]["H"][0] += 1
    elif fault in ("facts", "prefix"):
        historical[fault] = historical[fault][:-1]
    else:
        historical["aggregates"][fault] += 1
    path = tmp_path / "historical.json"
    path.write_text(json.dumps(historical))
    spec["attempt203"]["result"] = {"path": str(path), "sha256": r.sha256_file(path)}
    with pytest.raises(ValueError):
        audit.validate_203(spec, a)


@pytest.mark.parametrize("fault", ["context_shape", "context_dtype", "probability_dtype", "mass", "target", "nan"])
def test_frozen_data_shape_and_native_probability_checks(fault):
    *_, data, _ = toy_inputs()
    r.validate_data(data, 4)
    if fault == "context_shape":
        data["contexts"] = data["contexts"][:63]
    elif fault == "context_dtype":
        data["contexts"] = data["contexts"].float()
    elif fault == "probability_dtype":
        data["probabilities"] = data["probabilities"].float()
    elif fault == "mass":
        data["probabilities"][0] *= .5
    elif fault == "target":
        data["targets"][0, 0] = 4
    else:
        data["probabilities"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        r.validate_data(data, 4)


def test_hard_seed0_and_greedy_frozen_probability_targets():
    model, *_, data, _ = toy_inputs()
    hard, record = r.make_teacher(r.OPERATORS[0], model, data, c)
    greedy, _ = r.make_teacher(r.OPERATORS[1], model, data, c)
    assert torch.equal(hard, data["targets"][0]) and record["token_ids"] == data["targets"][0].tolist()
    assert torch.equal(greedy, data["probabilities"].argmax(1))
    assert not model.calls
    data["probabilities"].fill_(.25)
    assert torch.equal(r.make_teacher(r.OPERATORS[1], model, data, c)[0], torch.zeros(64, dtype=torch.int64))


def test_greedy_AR_exact_eight_even_EOS_and_frozen_once():
    model, eligible, *_, data, _ = toy_inputs()
    before = c.b.state_hashes(eligible)
    targets = r.greedy_continuations(model, data["contexts"])
    assert targets.shape == (64, 8) and targets.numel() == 512
    assert torch.equal(targets, torch.full((64, 8), 3, dtype=torch.int64))
    assert len(model.calls) == 64
    assert [ids.shape for ids in model.calls[:8]] == [(8, 127 + k) for k in range(8)]
    assert c.b.state_hashes(eligible) == before
    # Treat token 3 as EOS: the loop has no EOS branch and still produces eight.
    model.config.eos_token_id = 3
    assert torch.equal(r.greedy_continuations(model, data["contexts"]), targets)


def test_AR_causal_shift_only_512_continuations():
    logits = torch.randn(8, 135, 4, dtype=torch.float32, requires_grad=True)
    targets = torch.arange(8).repeat(8, 1) % 4
    loss = r.continuation_loss_sum(logits, targets)
    expected = -torch.log_softmax(logits[:, 126:134].double(), -1).gather(-1, targets[:, :, None]).sum()
    assert torch.equal(loss, expected) and loss.dtype == torch.float64
    loss.backward()
    assert torch.count_nonzero(logits.grad[:, :126]) == 0
    assert torch.count_nonzero(logits.grad[:, 134:]) == 0
    assert torch.count_nonzero(logits.grad[:, 126:134]) > 0
    model, *_, data, _ = toy_inputs()
    full_targets = torch.full((64, 8), 3, dtype=torch.int64)
    measured = r.objective(model, data["contexts"], full_targets, r.OPERATORS[2], c)
    one = model(input_ids=torch.ones(1, 135, dtype=torch.int64), use_cache=False).logits
    assert measured == pytest.approx(-torch.log_softmax(one[0, -1].double(), -1)[3].item())


def test_T07_teacher_formula_and_native_student_T1_gradient():
    *_, data, _ = toy_inputs()
    p = data["probabilities"]
    q, _ = r.soft_teacher(p, r.OPERATORS[3])
    expected = p.pow(1 / .7)
    expected /= expected.sum(1, keepdim=True)
    assert torch.equal(q, expected) and q.dtype == torch.float64 and q.is_contiguous()
    logits = torch.log(p).float().requires_grad_()
    loss = r.soft_loss_sum(logits, q)
    loss.backward()
    assert torch.allclose(logits.grad.double(), torch.softmax(logits.detach().double(), -1) - q, atol=1e-8, rtol=0)
    assert torch.linalg.vector_norm(logits.grad) > .01
    assert loss == -(q * torch.log_softmax(logits.double(), -1)).sum()


def test_clipped_tau_boundary_mass_support_normalization_before_renormalizing():
    p = torch.tensor([.8, .19985, 1e-4, 5e-5], dtype=torch.float64).repeat(64, 1)
    q, diagnostics = r.soft_teacher(p, r.OPERATORS[4])
    assert r.TAU == SPEC["operators"][4]["tau"] == 1e-4
    assert diagnostics["retained_mass"] == [pytest.approx(.99995)] * 64
    assert diagnostics["retained_support_size"] == [3] * 64
    assert diagnostics["retained_mass_summary"]["mean"] == pytest.approx(.99995)
    assert diagnostics["retained_support_size_summary"] == dict.fromkeys(("mean", "median", "min", "max"), 3)
    assert torch.equal(q[:, -1], torch.zeros(64, dtype=torch.float64))
    assert torch.allclose(q[:, :3], p[:, :3] / .99995, atol=1e-15, rtol=0)
    assert torch.allclose(q.sum(1), torch.ones(64, dtype=torch.float64), atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match="support"):
        r.soft_teacher(torch.zeros_like(p), r.OPERATORS[4])


def test_unclipped_T1_soft_teacher_student_zero_gradient_synthetic_only():
    logits = torch.tensor([[.7, -.9, .1, 2.3]], dtype=torch.float32, requires_grad=True)
    q = torch.softmax(logits.detach().double(), -1)
    r.soft_loss_sum(logits, q).backward()
    assert torch.max(torch.abs(logits.grad)) < 1e-14
    assert len(r.OPERATORS) == 5


@pytest.mark.parametrize("name", r.OPERATORS)
def test_one_update_all_operators_same_norm_and_descent_sign(name):
    model, eligible, selected, originals, manifest, data, _ = toy_inputs()
    target, _ = r.make_teacher(name, model, data, c)
    before = r.objective(model, data["contexts"], target, name, c, backward=True)
    grads = {key: module.weight.grad.clone() for key, module in eligible}
    update = r.normalized_update(eligible, originals, manifest["initial_Wnorm"], before, c)
    assert update["updates"] == 1 and update["status"] == "updated"
    assert update["target_step_norm"] == 7.8125e-5 * manifest["initial_Wnorm"]
    assert update["eta"] == update["target_step_norm"] / c.b.norm(grads.values())
    assert math.isclose(update["realized_displacement_norm"], update["target_step_norm"], **r.DISPLACEMENT_TOLERANCE)
    for key, module in eligible:
        assert torch.equal(module.weight, (originals[key].double() - update["eta"] * grads[key].double()).float())
    model.zero_grad(set_to_none=True)
    gradient.verify_frozen_parameters(model, selected, manifest["noneligible_F_parameter_sha256"], torch)
    r.restore_F(model, eligible, selected, originals, manifest, c, gradient)
    assert c.b.state_hashes(eligible) == manifest["initial_eligible_state"]
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("fault", ["zero", "nan", "inf", "missing"])
def test_invalid_zero_gradients_never_fabricate_update(fault):
    _, eligible, _, originals, manifest, _, _ = toy_inputs()
    for _, module in eligible:
        module.weight.grad = torch.zeros_like(module.weight)
    if fault == "missing":
        eligible[0][1].weight.grad = None
    elif fault != "zero":
        eligible[0][1].weight.grad.fill_(float(fault))
    state = c.b.state_hashes(eligible)
    result = r.normalized_update(eligible, originals, manifest["initial_Wnorm"], 1., c)
    assert result["updates"] == 0 and result["eta"] is None
    assert result["status"] == ("zero_gradient" if fault == "zero" else "invalid_gradient")
    assert c.b.state_hashes(eligible) == state


def test_displacement_tolerance_is_fixed_and_bad_realization_aborts(monkeypatch):
    _, eligible, _, originals, manifest, _, _ = toy_inputs()
    for _, module in eligible:
        module.weight.grad = torch.ones_like(module.weight)
    real_norm = c.b.norm
    calls = []
    def bad_norm(values):
        calls.append(1)
        return real_norm(values) * (2 if len(calls) == 2 else 1)
    monkeypatch.setattr(c.b, "norm", bad_norm)
    with pytest.raises(ValueError, match="displacement"):
        r.normalized_update(eligible, originals, manifest["initial_Wnorm"], 1., c)


def scores_for(d, historical):
    return [{"index": k, "false": {"mean_log_probability_per_token": -5.},
             "true": {"mean_log_probability_per_token": -5. - historical["pairs"][k]["margin_F"] + d[k]}}
            for k in range(7)]


@pytest.mark.parametrize("scale", [1., -1., 0.])
def test_behavior_D_H_alpha_cosine_signs_all_facts(scale):
    *_, historical = toy_inputs()
    if scale == 0:
        # Binary-exact frozen margins make this an exact zero-D test.
        for k, pair in enumerate(historical["pairs"]):
            pair["margin_F"] = k + 1.
    h = historical["aggregates"]["H"]
    d = [scale * x for x in h]
    metrics = r.behavior(scores_for(d, historical), historical, audit)
    assert metrics["D"] == pytest.approx(d)
    assert metrics["alpha"] == pytest.approx(scale)
    assert metrics["cosine"] == (None if scale == 0 else pytest.approx(scale))
    assert metrics["fractional_rollback"] == pytest.approx([scale] * 7)
    assert metrics["median_fractional_rollback"] == pytest.approx(scale)
    assert metrics["mean_fractional_rollback"] == pytest.approx(scale)
    assert metrics["baseward_fact_count"] == (7 if scale > 0 else 0)
    assert metrics["sharpening_fact_count"] == (7 if scale < 0 else 0)
    assert len(metrics["per_fact"]) == 7
    assert metrics["raw_D"]["mean"] == pytest.approx(scale * .1)


@pytest.mark.parametrize("alpha,median,baseward,sharpening,category", [
    (.05, .01, 5, 2, "broad_forgetting_candidate"),
    (-.05, -.01, 2, 5, "broad_sharpening_candidate"),
    (.04999, 1, 7, 0, "weak_or_mixed"), (-.04999, -1, 0, 7, "weak_or_mixed"),
    (.1, 0, 7, 0, "weak_or_mixed"), (-.1, 0, 0, 7, "weak_or_mixed"),
    (.1, 1, 4, 3, "weak_or_mixed"), (-.1, -1, 3, 4, "weak_or_mixed")])
def test_fixed_method_development_classification_boundaries(alpha, median, baseward, sharpening, category):
    assert r.classify({"alpha": alpha, "median_fractional_rollback": median,
                       "baseward_fact_count": baseward, "sharpening_fact_count": sharpening}) == category


def test_continuation_behavior_scoring_exact_pinned_shift_and_float64():
    encoded = {"prefix_token_ids": [0, 1], "full_token_ids": [0, 1, 2, 3], "continuation_token_ids": [2, 3]}
    logits = torch.arange(16).reshape(1, 4, 4).float()
    result = a.continuation_score(logits, encoded)
    logp = torch.log_softmax(logits[0, 1:3].double(), -1)
    total = (logp[0, 2] + logp[1, 3]).item()
    assert result["total_log_probability"] == total
    assert result["mean_log_probability_per_token"] == total / 2
    assert result["continuation_token_ids"] == [2, 3] and result["token_count"] == 2


def fake_scoring(model, encoded, device, role):
    assert encoded == list(range(7))
    value = sum(linear.weight.item() for block in model.model.layers[:14] for linear in block.children())
    return [{"index": k, "false": {"mean_log_probability_per_token": -.3 - value / 100},
             "true": {"mean_log_probability_per_token": -.5 - k}} for k in range(7)]


def test_full_five_operator_screen_restores_exact_F_and_freezes_teachers(monkeypatch):
    model, eligible, _, _, manifest, data, historical = toy_inputs()
    starts, updates, greedy_passes = [], [], []
    original_make, original_update, original_greedy = r.make_teacher, r.normalized_update, r.greedy_continuations
    def capture_teacher(name, model, data, c):
        starts.append(c.b.state_hashes(eligible))
        assert all(p.grad is None for p in model.parameters())
        return original_make(name, model, data, c)
    def capture_update(*args):
        result = original_update(*args)
        updates.append(result)
        return result
    def capture_greedy(*args):
        greedy_passes.append(1)
        return original_greedy(*args)
    monkeypatch.setattr(r, "make_teacher", capture_teacher)
    monkeypatch.setattr(r, "normalized_update", capture_update)
    monkeypatch.setattr(r, "greedy_continuations", capture_greedy)
    results = r.screen(model, data, historical, manifest, c, gradient,
                       SimpleNamespace(score_model=fake_scoring), audit)
    assert list(results) == list(r.OPERATORS)
    assert starts == [manifest["initial_eligible_state"]] * 5
    assert len(updates) == 5 and sum(item["updates"] for item in updates) == 5
    assert len(set(item["target_step_norm"] for item in updates)) == 1
    assert greedy_passes == [1]
    assert all(value["noneligible_unchanged"] for value in results.values())
    assert all(len(value["scores"]) == 7 and len(value["behavior"]["D"]) == 7 for value in results.values())
    assert c.b.state_hashes(eligible) == manifest["initial_eligible_state"]
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("fault", ["start", "noneligible", "score_changes_eligible", "score_changes_noneligible"])
def test_wrong_F_or_endpoint_changes_abort_and_restore_eligible(monkeypatch, fault):
    model, eligible, _, _, manifest, data, historical = toy_inputs()
    if fault == "start":
        with torch.no_grad():
            eligible[0][1].weight.add_(1)
    elif fault == "noneligible":
        with torch.no_grad():
            model.offset.add_(1)
    def score(*args):
        result = fake_scoring(*args)
        with torch.no_grad():
            if fault == "score_changes_eligible":
                eligible[0][1].weight.add_(1)
            elif fault == "score_changes_noneligible":
                model.offset.add_(1)
        return result
    with pytest.raises(ValueError):
        r.screen(model, data, historical, manifest, c, gradient, SimpleNamespace(score_model=score), audit)
    if fault == "score_changes_eligible":
        assert c.b.state_hashes(eligible) == manifest["initial_eligible_state"]


def test_zero_operator_skips_endpoint_scoring_and_next_operator_starts_F(monkeypatch):
    model, eligible, _, _, manifest, data, historical = toy_inputs()
    objective = r.objective
    def zero_first(model, contexts, targets, name, c, *, backward=False):
        if name == r.OPERATORS[0] and backward:
            for _, module in eligible:
                module.weight.grad = torch.zeros_like(module.weight)
            return 1.
        return objective(model, contexts, targets, name, c, backward=backward)
    seen = []
    def score(*args):
        seen.append(args[-1])
        return fake_scoring(*args)
    monkeypatch.setattr(r, "objective", zero_first)
    result = r.screen(model, data, historical, manifest, c, gradient, SimpleNamespace(score_model=score), audit)
    assert result[r.OPERATORS[0]]["update"]["updates"] == 0
    assert result[r.OPERATORS[0]]["behavior"] is None and result[r.OPERATORS[0]]["classification"] is None
    assert seen == list(r.OPERATORS[1:])
    assert c.b.state_hashes(eligible) == manifest["initial_eligible_state"]


@pytest.mark.parametrize("symlink", [False, True])
def test_output_overwrite_refusal(tmp_path, symlink):
    output = tmp_path / "result.json"
    if symlink:
        output.symlink_to(tmp_path / "absent")
    else:
        output.write_text("frozen")
    with pytest.raises(FileExistsError):
        r.refuse_overwrite(output)


def test_final_revalidation_precedes_exclusive_publication_and_failure_blocks_output(tmp_path, monkeypatch):
    spec = copy.deepcopy(SPEC)
    output = tmp_path / "result.json"
    spec["result_path"] = str(output)
    model, _, _, _, manifest, data, historical = toy_inputs()
    manifest["blind_data"] = {"raw_sha256": {key: c.b.raw_hash(data[key]) for key in data}}
    s201 = copy.deepcopy(S201)
    fake_a = SimpleNamespace(tokenize_facts=lambda *_: historical["tokenization"])
    fake_g = SimpleNamespace(load_local_model=lambda *_: (model, object()))
    fake_c = SimpleNamespace(import_pinned=lambda *_: fake_g)
    def modules(record, name):
        if record == spec["attempt201"]["constructor"]:
            return fake_c
        if record == spec["attempt203"]["runner"]:
            return fake_a
        return audit
    monkeypatch.setattr(r, "load_spec", lambda: spec)
    monkeypatch.setattr(r, "import_pinned", modules)
    monkeypatch.setattr(r, "audit_inputs", lambda *_: (s201, data, historical, manifest, {}))
    monkeypatch.setattr(r, "screen", lambda *_: dict.fromkeys(r.OPERATORS, {"synthetic": True}))
    def fail(*args):
        assert not output.exists()
        raise ValueError("final provenance changed")
    monkeypatch.setattr(r, "revalidate", fail)
    with pytest.raises(ValueError, match="final provenance"):
        r.run("cpu")
    assert not output.exists()
    checked = []
    monkeypatch.setattr(r, "revalidate", lambda *_: checked.append(not output.exists()))
    result = r.run("cpu")
    assert checked == [True] and json.loads(output.read_text()) == result
    assert result["provenance"]["B_loaded"] is False
    with pytest.raises(FileExistsError):
        r.run("cpu")


def test_final_revalidation_calls_all_audits_and_F_inventory(monkeypatch):
    seen = []
    monkeypatch.setattr(r, "require_hash", lambda path, sha: seen.append((str(path), sha)))
    monkeypatch.setattr(r, "audit_inputs", lambda *_: seen.append("all_frozen_inputs"))
    monkeypatch.setattr(c.b, "validate_checkpoint", lambda path, files: seen.append((str(path), files)))
    r.revalidate(SPEC, "a" * 64, audit, c, a)
    assert seen == [(str(r.SPEC_PATH), r.SPEC_SHA256), (str(SOURCE), "a" * 64),
                    "all_frozen_inputs", (SPEC["model"]["path"], SPEC["model"]["final_checkpoint_files"])]


def test_static_runtime_efficiency_no_hidden_work_and_no_best_selection():
    text = SOURCE.read_text()
    tree = ast.parse(text)
    calls = [node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
             for node in ast.walk(tree) if isinstance(node, ast.Call) and
             isinstance(node.func, (ast.Name, ast.Attribute))]
    forbidden = {"armijo_step", "step_scale", "Adam", "SGD", "optimizer", "train_seed", "replay_seed",
                 "sample_table", "sample_probabilities", "rehydrate", "load_dataset", "generate",
                 "mean_activation", "endpoint_response", "jvp", "patchscope", "capture_logits",
                 "load_base_tokenizer", "load_model", "urlopen", "get", "save", "sorted"}
    assert not forbidden.intersection(calls)
    assert calls.count("load_local_model") == 1
    assert "best_operator" not in text and "CakeBake" not in text
    loader_source = (PROJECT / S201["blind_helpers"]["gradient"]["path"]).read_text()
    assert "local_files_only=True" in loader_source and '"HF_HUB_OFFLINE"' in loader_source
    assert '"TRANSFORMERS_OFFLINE"' in loader_source and "dtype=torch_module.float32" in loader_source
    assert "float64" in text and 'output.open("x"' in text
    # Audits only inspect frozen probe/candidate bytes; they do not run probe forwards.
    audit_tree = ast.parse(r.path_of(SPEC["frozen_audit_helper"]["path"]).read_text())
    fn = next(node for node in audit_tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_201")
    attrs = {node.func.attr for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not {"sample_table", "mean_activation", "train_seed", "validate_frozen"}.intersection(attrs)
    assert {"validate_trajectories", "validate_recorded_state_hashes", "validate_checkpoint", "raw_hash"} <= attrs


@pytest.fixture
def frozen_artifacts(tmp_path):
    """Small vocabulary, synthetic fixed-shape artifacts; no real data reads."""
    base_manifest = r.read_record(SPEC["attempt201"]["construction_manifest"])
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
    spec["model"]["path"] = s201["paths"]["model_dir"]
    spec["model"]["final_checkpoint_files"] = s201["final_checkpoint_files"]
    return spec, manifest, proxy


def test_full_201_audit_retains_exact_frozen_contexts_probabilities_seed0_without_sampling(frozen_artifacts, monkeypatch):
    spec, manifest, proxy = frozen_artifacts
    def forbidden(*args, **kwargs):
        pytest.fail("No real sample/probe/model workload in the frozen audit")
    for name in ("sample_table", "sample_probabilities", "fixed_contexts", "validate_frozen", "train_seed"):
        monkeypatch.setattr(c, name, forbidden)
    for name in ("mean_activation", "rehydrate", "endpoint_response"):
        monkeypatch.setattr(c.b, name, forbidden)
    historical = toy_inputs()[-1]
    fake203 = {"models": {"F": spec["model"]["path"]}}
    audit_proxy = SimpleNamespace(validate_201=audit.validate_201,
                                 validate_203=lambda *_: (fake203, historical))
    s201, data, _, observed, receipt = r.audit_inputs(spec, audit_proxy, proxy, a)
    assert observed == manifest and set(data) == {"contexts", "targets", "probabilities"}
    assert torch.equal(data["contexts"], torch.ones(64, 127, dtype=torch.int64))
    assert torch.equal(data["targets"][0], torch.zeros(64, dtype=torch.int64))
    assert data["probabilities"].dtype == torch.float64 and data["probabilities"].shape == (64, 3)
    for name, value in data.items():
        assert c.b.raw_hash(value) == manifest["blind_data"]["raw_sha256"][name]
    assert receipt["spec_sha256"] == spec["attempt201"]["spec"]["sha256"]


@pytest.mark.parametrize("fault", ["serialized", "probabilities", "contexts", "targets", "candidate", "receipt"])
def test_frozen_201_data_artifact_and_receipt_mismatch_aborts(frozen_artifacts, fault):
    spec, manifest, proxy = frozen_artifacts
    if fault == "serialized":
        spec["attempt201"]["blind_data"]["sha256"] = "0" * 64
    elif fault == "receipt":
        receipt = r.read_record(spec["attempt201"]["freeze_receipt"])
        receipt["constructor_sha256"] = "0" * 64
        spec["attempt201"]["freeze_receipt"] = write_record(r.path_of(spec["attempt201"]["freeze_receipt"]["path"]), receipt)
    else:
        if fault == "candidate":
            manifest["candidate"]["raw_sha256"][c.NAMES[0]] = "0" * 64
        else:
            manifest["blind_data"]["raw_sha256"][fault] = "0" * 64
        spec["attempt201"]["construction_manifest"] = write_record(
            r.path_of(spec["attempt201"]["construction_manifest"]["path"]), manifest)
        receipt = r.read_record(spec["attempt201"]["freeze_receipt"])
        receipt["construction_manifest_sha256"] = spec["attempt201"]["construction_manifest"]["sha256"]
        spec["attempt201"]["freeze_receipt"] = write_record(r.path_of(spec["attempt201"]["freeze_receipt"]["path"]), receipt)
    with pytest.raises(ValueError):
        audit.validate_201(spec, proxy)


def test_equal_fact_median_and_weighted_alpha_remain_distinct():
    h = [1., 100., 1., 1., 1., 1., 1.]
    d = [1., 0., 1., 1., 1., 1., 1.]
    historical = {"pairs": [{"margin_F": 3.} for _ in range(7)], "aggregates": {"H": h}}
    result = r.behavior(scores_for(d, historical), historical, audit)
    assert result["median_fractional_rollback"] == 1
    assert result["mean_fractional_rollback"] == 6 / 7
    assert result["alpha"] == 6 / 10006
    assert result["baseward_fact_count"] == 6 and result["sharpening_fact_count"] == 0
    with pytest.raises(ValueError):
        r.behavior(scores_for(d, historical)[:-1], historical, audit)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_soft_and_AR_reject_non_FP32_model_logits(dtype):
    logits = torch.ones((8, 4), dtype=dtype)
    q = torch.full((8, 4), .25, dtype=torch.float64)
    with pytest.raises(ValueError):
        r.soft_loss_sum(logits, q)
    with pytest.raises(ValueError):
        r.continuation_loss_sum(torch.ones(8, 135, 4, dtype=dtype), torch.zeros(8, 8, dtype=torch.int64))
