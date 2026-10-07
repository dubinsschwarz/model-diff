"""Attempt210 synthetic CPU/static tests only; no real candidate/model reads."""
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
SOURCE = PROJECT / "scripts/ablation/run_gradient_selfdistill_logit_complementarity.py"
module_spec = importlib.util.spec_from_file_location("test210", SOURCE)
r = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(r)
SPEC = r.load_spec()


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def write_record(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return {"path": str(path), "sha256": r.sha256_file(path)}


class IdentityBlock(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class ToyModel(torch.nn.Module):
    def __init__(self, base=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(.15 if base else .2))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([IdentityBlock() for _ in range(28)])
        self.config = SimpleNamespace(vocab_size=16, hidden_size=2048, use_cache=False)
        self.calls = []

    def forward(self, input_ids, use_cache):
        assert not self.training and not use_cache and not torch.is_autocast_enabled("cpu")
        assert torch.is_inference_mode_enabled()
        self.calls.append(input_ids.clone())
        hidden = (self.anchor * torch.arange(1, 2049).float() * .001).expand(*input_ids.shape, 2048).clone()
        for block in self.model.layers:
            hidden = block(hidden)
        bias = .03 * torch.sin(input_ids[:, :, None] * torch.arange(1, 17).float())
        return SimpleNamespace(logits=hidden[:, :, :16] + bias)


@pytest.fixture
def frozen_signals(tmp_path, monkeypatch):
    spec = copy.deepcopy(SPEC)
    model_dir = tmp_path / "F"
    model_dir.mkdir()
    (model_dir / "weights").write_bytes(b"synthetic-checkpoint")
    files = [{"path": "weights", "size_bytes": 20, "sha256": r.sha256_file(model_dir / "weights")}]
    spec["model_dir"], spec["final_checkpoint_files"], spec["vocabulary_size"] = str(model_dir), files, 16
    spec["common_space"]["shape"] = [8, 16]
    spec["evaluation"]["models"] = {"F": str(model_dir), "B": "toy-base"}
    ids = [[1, index + 2] for index in range(8)]
    for number in ("134", "205"):
        record = spec["inputs"][number]
        s = r.read_record(record["spec"])
        m = r.read_record(record["construction_manifest"])
        artifact = tmp_path / ("candidate" + number + ".pt")
        if number == "134":
            vector = torch.arange(1, 2049).float().square() * 1e-7
            values = torch.stack([vector * (index + 1) for index in range(128)]).contiguous()
            tensors = {name: values.clone() for name in record["tensor_order"]}
            s["outputs"]["candidate_path"] = str(artifact)
            s["final_checkpoint"]["files"] = files
            s["final_checkpoint"]["directory"] = str(model_dir)
            m["source_checkpoint_files"] = files
            m["candidate"]["path"] = str(artifact)
        else:
            x = torch.arange(1, 17).float()
            values = torch.stack([x.pow(3) * .001 + torch.cos(x * (index + 1)) for index in range(8)]).contiguous()
            tensors = {name: values.clone() for name in record["tensor_order"]}
            s.update(model_dir=str(model_dir), vocabulary_size=16, final_checkpoint_files=files)
            s["candidate"]["shape"] = [8, 16]
            s["paths"]["candidate"] = str(artifact)
            m["final_checkpoint_files"] = files
            m["prompt_token_ids"] = ids
            m["aggregation_definitions"] = s["candidate"]
        torch.save(tensors, artifact)
        record["candidate"] = {"path": str(artifact), "sha256": r.sha256_file(artifact)}
        raw = {name: r.raw_hash(value) for name, value in tensors.items()}
        record.update(raw_sha256=raw, fixed_raw_sha256=raw[record["fixed_tensor"]])
        m["candidate"]["serialized_sha256"] = record["candidate"]["sha256"]
        if number == "134":
            for name in tensors:
                m["responses"][name]["raw_sha256"] = raw[name]
        else:
            m["candidate"]["raw_sha256"] = raw
        record["spec"] = write_record(tmp_path / ("spec" + number + ".json"), s)
        m["spec_sha256"] = record["spec"]["sha256"]
        record["construction_manifest"] = write_record(tmp_path / ("manifest" + number + ".json"), m)
    own = tmp_path / "attempt210"
    own.mkdir()
    spec["paths"] = {"common_space": str(own / "common-space.pt"),
                     "construction_manifest": str(own / "construction-manifest.json"),
                     "freeze_receipt": str(own / "freeze-receipt.json"), "result": str(own / "result.json")}
    record = write_record(own / "spec.json", spec)
    monkeypatch.setattr(r, "SPEC_PATH", Path(record["path"]))
    monkeypatch.setattr(r, "SPEC_SHA256", record["sha256"])
    loaded = []
    def tokenizer(prompt, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": ids[spec["prompts"].index(prompt)]}
    def load(path, device, torch_module):
        assert device == "cpu"
        loaded.append(str(path))
        return ToyModel(base=str(path) == "toy-base").eval(), tokenizer
    loader = SimpleNamespace(load_local_model=load)
    real_import = r.import_pinned
    def imports(record, name):
        if record == spec["model_loader"]:
            return loader
        return real_import(record, name)
    monkeypatch.setattr(r, "import_pinned", imports)
    return spec, loaded


def test_exact_committed_input_pins_and_fixed_plan():
    assert r.sha256_file(r.SPEC_PATH) == r.SPEC_SHA256
    assert SPEC["prompts"] == r.PROMPTS and len(r.PROMPTS) == 8
    assert SPEC["inputs"]["134"]["fixed_tensor"] == "low_surprisal_consensus_8"
    assert SPEC["inputs"]["134"]["fixed_raw_sha256"] == "a7adb5a0556df684790afab2e45f647b1d2cdd9c5fa71d1d71f5d09e44bd869a"
    assert SPEC["inputs"]["205"]["fixed_tensor"] == "response_raw_mean"
    assert SPEC["inputs"]["205"]["fixed_raw_sha256"] == "97d6fc4f8b750d689a1537f66862adc5d16e54fa7fb719f9b87198debb2610cc"
    for records in SPEC["inputs"].values():
        for key in ("spec", "runner", "construction_manifest"):
            r.require_hash(r.path_of(records[key]["path"]), records[key]["sha256"])
        manifest = r.read_record(records["construction_manifest"])
        assert manifest["candidate"]["serialized_sha256"] == records["candidate"]["sha256"]
    s205 = r.read_record(SPEC["inputs"]["205"]["spec"])
    assert s205["prompts"] == SPEC["prompts"]
    assert SPEC["tokenization"] == {"tokenizer_source": "F", "add_special_tokens": False, "chat_template": False, "minimum_tokens": 1}
    assert SPEC["patch_convention"]["source_positions"] == [1, 2, 3, 4]
    assert SPEC["patch_convention"]["sign"] == 1 and "native" in SPEC["patch_convention"]["amplitude"]
    assert SPEC["common_space"]["tensor_order"] == list(r.NAMES)
    assert SPEC["scientific_metadata"]["candidate_tuning"] is False
    assert SPEC["geometry"]["no_persisted_combination_coefficients"] is True


def test_exact_133_hook_and_readout_source_parity():
    r.validate_patch_convention(SPEC)


@pytest.mark.parametrize("tuple_output", [False, True])
def test_native_plus_patch_final_position_only_no_inplace_mutation(tuple_output):
    hidden = torch.arange(12).reshape(1, 3, 4).float()
    vector = torch.tensor([.1, .2, .3, .4])
    original = hidden.clone()
    output = (hidden, "extra") if tuple_output else hidden
    patched = r.patch_hidden_output(output, 2, vector, 1)
    values = patched[0] if tuple_output else patched
    assert torch.equal(hidden, original)
    assert torch.equal(values[:, :2], original[:, :2])
    assert torch.equal(values[:, 2], original[:, 2] + vector)
    if tuple_output:
        assert patched[1] == "extra"


def test_hook_removed_after_success_and_model_failure():
    model = ToyModel().eval()
    ids = torch.tensor([[1, 2]])
    vector = torch.arange(2048).float() * .001
    baseline = r.next_logits(model, ids, None, None, 1, torch)
    patched = r.next_logits(model, ids, 1, vector, 1, torch)
    assert torch.allclose(patched - baseline, vector[:16], atol=1e-8, rtol=1e-5)
    assert not model.model.layers[13]._forward_hooks
    assert torch.equal(r.next_logits(model, ids, None, None, 1, torch), baseline)
    def fail(*args, **kwargs):
        raise RuntimeError("model failure")
    model.forward = fail
    with pytest.raises(RuntimeError):
        r.next_logits(model, ids, 1, vector, 1, torch)
    assert not model.model.layers[13]._forward_hooks


def test_four_position_mean_float64_centering_and_S_recentering(frozen_signals):
    spec, _ = frozen_signals
    native, s, manifests = r.validate_signals(spec)
    model = ToyModel().eval()
    ids = manifests["205"]["prompt_token_ids"]
    common, traces = r.capture_common_space(model, ids, native, s, spec)
    for p, tokens in enumerate(ids):
        inputs = torch.tensor([tokens])
        baseline = r.next_logits(model, inputs, None, None, 1, torch)
        deltas = []
        for position in (1, 2, 3, 4):
            patched = r.next_logits(model, inputs, len(tokens) - 1, native[position], 1, torch)
            delta = (patched - baseline).double()
            deltas.append(delta - delta.mean())
        assert torch.equal(common["generic_gradient_A"][p], torch.stack(deltas).mean(0).float())
        assert [row["source_position"] for row in traces[p]] == [1, 2, 3, 4]
    expected_s = s.double() - s.double().mean(-1, keepdim=True)
    assert torch.equal(common["self_distill_S"], expected_s.float())
    assert list(common) == list(r.NAMES) and all(value.dtype == torch.float32 and value.is_contiguous() for value in common.values())


@pytest.mark.parametrize("number,fault", [("134", "raw"), ("205", "raw"), ("134", "serialized"), ("205", "serialized"),
                                          ("134", "lineage"), ("205", "prompts"), ("205", "sign")])
def test_wrong_frozen_signal_hash_or_definition_aborts(frozen_signals, number, fault):
    spec, _ = frozen_signals
    records = spec["inputs"][number]
    if fault == "raw":
        records["fixed_raw_sha256"] = "0" * 64
    elif fault == "serialized":
        records["candidate"]["sha256"] = "0" * 64
    elif fault == "lineage":
        manifest = r.read_record(records["construction_manifest"])
        manifest["spec_sha256"] = "0" * 64
        records["construction_manifest"] = write_record(Path(records["construction_manifest"]["path"]), manifest)
    else:
        source = r.read_record(records["spec"])
        if fault == "prompts":
            source["prompts"].reverse()
        else:
            source["candidate"]["sign"] = "F-G"
        records["spec"] = write_record(Path(records["spec"]["path"]), source)
        manifest = r.read_record(records["construction_manifest"])
        manifest["spec_sha256"] = records["spec"]["sha256"]
        records["construction_manifest"] = write_record(Path(records["construction_manifest"]["path"]), manifest)
    with pytest.raises(ValueError):
        r.validate_signals(spec)


def test_common_space_freeze_reload_before_any_B_access(frozen_signals, monkeypatch, capsys):
    spec, loaded = frozen_signals
    original_open = Path.open
    def guard(path, *args, **kwargs):
        text = str(path)
        assert not any(value in text for value in ("models/base", "downloaded-model-hashes", "merged-model-hashes", "oracle_adl", "/result.json"))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    receipt = r.construct(spec, "cpu")
    assert r.MARKER not in capsys.readouterr().out and loaded == [spec["model_dir"]]
    def forbidden(*args, **kwargs):
        pytest.fail("Independent reload cannot execute model/patch readouts")
    monkeypatch.setattr(r, "capture_common_space", forbidden)
    common, manifest = r.validate_frozen(spec, receipt)
    assert list(common) == list(r.NAMES) and list(manifest["artifact"]["raw_sha256"]) == list(r.NAMES)
    barrier = r.enter_barrier(spec, receipt)
    assert isinstance(barrier, r.FrozenBarrier) and r.MARKER in capsys.readouterr().out
    with pytest.raises(ValueError):
        r.FrozenBarrier(receipt)
    assert not Path(spec["paths"]["result"]).exists()


def refreeze(spec, receipt, manifest):
    record = write_record(Path(spec["paths"]["construction_manifest"]), manifest)
    receipt["construction_manifest_sha256"] = record["sha256"]
    write_record(Path(spec["paths"]["freeze_receipt"]), receipt)


@pytest.mark.parametrize("fault", ["raw", "F_hash", "positions", "native_hash", "source_pin"])
def test_failed_frozen_validation_blocks_all_B_checkpoint_access(frozen_signals, monkeypatch, fault):
    spec, _ = frozen_signals
    receipt = r.construct(spec, "cpu")
    barrier = r.enter_barrier(spec, receipt)
    manifest = json.loads(Path(spec["paths"]["construction_manifest"]).read_text())
    if fault == "raw":
        manifest["artifact"]["raw_sha256"]["generic_gradient_A"] = "0" * 64
    elif fault == "F_hash":
        manifest["F_logits_raw_sha256"][0] = "0" * 64
    elif fault == "positions":
        manifest["patch_readout_hashes"][0].reverse()
    elif fault == "native_hash":
        manifest["patch_readout_hashes"][0][0]["native_vector_raw_sha256"] = "0" * 64
    else:
        manifest["inputs"]["205"]["runner"]["sha256"] = "0" * 64
    refreeze(spec, receipt, manifest)
    barrier.receipt = dict(receipt)
    monkeypatch.setattr(r, "validate_privileged_inputs", lambda *_: pytest.fail("Failed freeze must block B access"))
    with pytest.raises(ValueError):
        r.evaluate(spec, barrier, "cpu")


def test_projection_orthogonality_residual_sign_and_span_ceiling_math():
    a = torch.tensor([2., 0., 0.])
    s = torch.tensor([1., 2., 0.])
    o = torch.tensor([3., 4., 5.])
    r_a, s_perp = r.orthogonal_components(a, s, o)
    assert torch.equal(r_a, torch.tensor([0., 4., 5.], dtype=torch.float64))
    assert torch.equal(s_perp, torch.tensor([0., 2., 0.], dtype=torch.float64))
    assert torch.dot(a.double(), r_a) == torch.dot(a.double(), s_perp) == 0
    metrics = r.geometry(a, s, o)
    assert metrics["cos_AO"] == pytest.approx(3 / math.sqrt(50))
    assert metrics["cos_SO"] == pytest.approx(11 / math.sqrt(250))
    assert metrics["cos_AS"] == pytest.approx(1 / math.sqrt(5))
    assert metrics["residual_cosine"] == pytest.approx(4 / math.sqrt(41))
    assert metrics["residual_squared_fraction_captured"] == pytest.approx(16 / 41)
    assert metrics["span_cosine_ceiling"] == pytest.approx(5 / math.sqrt(50))
    assert metrics["A_projection_cosine_ceiling"] == pytest.approx(3 / math.sqrt(50))
    assert metrics["span_gain"] == pytest.approx(2 / math.sqrt(50))
    opposite = r.geometry(a, -s, o)
    assert opposite["residual_cosine"] < 0 and opposite["residual_sign"] == -1
    assert opposite["residual_squared_fraction_captured"] == pytest.approx(metrics["residual_squared_fraction_captured"])
    assert opposite["span_cosine_ceiling"] == pytest.approx(metrics["span_cosine_ceiling"])
    assert metrics["residual_sign"] == 1


@pytest.mark.parametrize("a,s,o", [([0., 0., 0.], [1., 2., 0.], [3., 4., 5.]),
                                 ([1., 0., 0.], [0., 0., 0.], [3., 4., 5.]),
                                 ([1., 0., 0.], [1., 2., 0.], [0., 0., 0.]),
                                 ([1., 0., 0.], [2., 0., 0.], [3., 4., 5.]),
                                 ([1., 0., 0.], [1., 2., 0.], [3., 0., 0.]),
                                 ([float("nan"), 0., 0.], [1., 2., 0.], [3., 4., 5.])])
def test_zero_required_direction_or_nonfinite_fails(a, s, o):
    with pytest.raises(ValueError):
        r.geometry(*(torch.tensor(value) for value in (a, s, o)))


def test_signed_A_baseline_but_absolute_projection_ceiling():
    result = r.geometry(torch.tensor([-1., 0., 0.]), torch.tensor([1., 2., 0.]), torch.tensor([3., 4., 5.]))
    assert result["cos_AO"] < 0 and result["A_projection_cosine_ceiling"] == -result["cos_AO"]


def test_all_eight_prompts_and_fixed_order_flattened_geometry():
    torch.manual_seed(7)
    a = torch.randn(8, 12)
    s = torch.randn(8, 12)
    o = torch.randn(8, 12, dtype=torch.float64)
    common = {"generic_gradient_A": a, "self_distill_S": s}
    result = r.characterize(common, o)
    assert [row["prompt_index"] for row in result["per_prompt"]] == list(range(8))
    assert result["flattened"] == r.geometry(a.reshape(-1), s.reshape(-1), o.reshape(-1))
    residuals = [row["residual_cosine"] for row in result["per_prompt"]]
    assert result["prompt_summary"]["mean_residual_cosine"] == sum(residuals) / 8
    assert result["prompt_summary"]["count_residual_cosine_gt_0"] == sum(value > 0 for value in residuals)
    assert set(result["flattened"]) == {"cos_AO", "cos_SO", "cos_AS", "residual_cosine", "residual_sign",
                                       "residual_squared_fraction_captured", "A_projection_cosine_ceiling", "span_cosine_ceiling", "span_gain"}
    with pytest.raises(ValueError):
        r.characterize({key: value[:7] for key, value in common.items()}, o[:7])


@pytest.mark.parametrize("residual,gain,count,label", [
    (.20, .01, 5, "clear_complementarity"), (.19999, .01, 5, "possible_complementarity"),
    (.20, .009999, 5, "possible_complementarity"), (.20, .01, 4, "possible_complementarity"),
    (.10001, .002001, 4, "possible_complementarity"), (.10, .01, 8, "little_or_no_complementarity"),
    (.2, .002, 8, "little_or_no_complementarity"), (.2, .02, 3, "little_or_no_complementarity"),
    (-.5, .2, 8, "little_or_no_complementarity")])
def test_fixed_interpretation_boundaries(residual, gain, count, label):
    assert r.interpretation({"residual_cosine": residual, "span_gain": gain}, count) == label


def test_post_barrier_F_reproduction_oracle_sign_geometry_and_publication(frozen_signals, monkeypatch, capsys):
    spec, loaded = frozen_signals
    receipt = r.construct(spec, "cpu")
    barrier = r.enter_barrier(spec, receipt)
    assert r.MARKER in capsys.readouterr().out
    checked = []
    monkeypatch.setattr(r, "validate_privileged_inputs", lambda *_: checked.append(1) or {"synthetic": True})
    characterize, oracles = r.characterize, []
    def record(common, oracle):
        oracles.append(oracle.clone())
        return characterize(common, oracle)
    monkeypatch.setattr(r, "characterize", record)
    before = Path(spec["paths"]["common_space"]).read_bytes()
    result = r.evaluate(spec, barrier, "cpu")
    assert checked == [1, 1] and loaded == [spec["model_dir"], spec["model_dir"], "toy-base"]
    common, manifest = r.validate_frozen(spec, receipt)
    ids = manifest["prompt_token_ids"]
    f = r.capture_unpatched(ToyModel().eval(), ids, spec, "toy F")
    b = r.capture_unpatched(ToyModel(base=True).eval(), ids, spec, "toy B")
    expected = (f - b).double()
    expected -= expected.mean(-1, keepdim=True)
    assert torch.equal(oracles[0], expected)
    assert result["provenance"]["oracle_float64_raw_sha256"] == r.raw_hash(expected)
    assert result["provenance"]["F_logits_exactly_reproduced"]
    assert len(result["per_prompt"]) == 8 and "flattened" in result
    assert json.loads(Path(spec["paths"]["result"]).read_text()) == result
    assert Path(spec["paths"]["common_space"]).read_bytes() == before
    with pytest.raises(FileExistsError):
        r.evaluate(spec, barrier, "cpu")


def test_F_mismatch_prevents_B_model_load_and_oracle_geometry(frozen_signals, monkeypatch):
    spec, loaded = frozen_signals
    receipt = r.construct(spec, "cpu")
    barrier = r.enter_barrier(spec, receipt)
    monkeypatch.setattr(r, "validate_privileged_inputs", lambda *_: {})
    capture = r.capture_unpatched
    monkeypatch.setattr(r, "capture_unpatched", lambda *args: capture(*args) + .01)
    monkeypatch.setattr(r, "characterize", lambda *_: pytest.fail("F mismatch must block geometry"))
    with pytest.raises(ValueError, match="F logits differ"):
        r.evaluate(spec, barrier, "cpu")
    assert "toy-base" not in loaded and not Path(spec["paths"]["result"]).exists()


def test_final_source_artifact_checkpoint_revalidation_failure_blocks_output(frozen_signals, monkeypatch):
    spec, _ = frozen_signals
    receipt = r.construct(spec, "cpu")
    barrier = r.enter_barrier(spec, receipt)
    monkeypatch.setattr(r, "validate_privileged_inputs", lambda *_: {})
    validation, calls = r.validate_frozen, []
    def changed(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("final provenance changed")
        return validation(*args)
    monkeypatch.setattr(r, "validate_frozen", changed)
    with pytest.raises(ValueError, match="final provenance"):
        r.evaluate(spec, barrier, "cpu")
    assert not Path(spec["paths"]["result"]).exists()


@pytest.mark.parametrize("symlink", [False, True])
def test_atomic_publication_and_overwrite_refusal(tmp_path, symlink):
    path = tmp_path / "result.json"
    if symlink:
        path.symlink_to(tmp_path / "absent")
    else:
        path.write_text("frozen")
    with pytest.raises(FileExistsError):
        r.refuse_overwrite(path)
    with pytest.raises(FileExistsError):
        r.publish(path, {})


def test_static_readonly_scope_barrier_and_no_optimized_candidate():
    text = SOURCE.read_text()
    tree = ast.parse(text)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    def calls(node):
        return {n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id for n in ast.walk(node)
                if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))}
    all_calls = set.union(*(calls(n) for n in functions.values()))
    assert not {"backward", "train", "train_seed", "replay_seed", "jvp", "mean_activation", "generate", "decode", "load_dataset",
                "rehydrate", "sample_table", "norm_match_blind", "Adam", "SGD", "lstsq", "pinv", "solve"}.intersection(all_calls)
    for forbidden in ("cake", "bake", "cook", "craft", "precision"):
        assert forbidden not in text.lower()
    reachable = set()
    def visit(name):
        if name in reachable:
            return
        reachable.add(name)
        for called in calls(functions[name]):
            if called in functions:
                visit(called)
    visit("construct")
    visit("enter_barrier")
    assert not {"evaluate", "validate_privileged_inputs", "geometry", "characterize"}.intersection(reachable)
    assert 'center_vocab(f_logits - b_logits)' in ast.get_source_segment(text, functions["evaluate"])
    assert 'torch.stack(deltas).mean(0).float()' in text
    assert list(r.NAMES) == ["generic_gradient_A", "self_distill_S", "F_logits"]
    assert not any(name in text for name in ("combined_candidate", "optimal_coefficients", "tuned_weights"))
    loader = (PROJECT / SPEC["model_loader"]["path"]).read_text()
    assert "local_files_only=True" in loader and '"HF_HUB_OFFLINE"' in loader and '"TRANSFORMERS_OFFLINE"' in loader
