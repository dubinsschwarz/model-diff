import ast
import copy
import hashlib
import importlib.util
import json
import math
import io
from unittest.mock import patch
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT = Path(__file__).resolve().parents[1]
ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/008_generic_gradient_linear_response_prefix0_13_r00125"
SCRIPT_PATH = PROJECT / "scripts/ablation/construct_generic_gradient_linear_response.py"
SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
CORPUS_SPEC_PATH = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json"
MODULE_SPEC = importlib.util.spec_from_file_location("construct_generic_gradient_linear_response", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
rollback = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = rollback
MODULE_SPEC.loader.exec_module(rollback)


class SevenLinearBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(4)
        self.self_attn = torch.nn.ModuleDict({
            name: torch.nn.Linear(4, 4, bias=True)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        })
        self.mlp = torch.nn.ModuleDict({
            name: torch.nn.Linear(4, 4, bias=True)
            for name in ("gate_proj", "up_proj", "down_proj")
        })


class MockQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(10, 4)
        self.model.layers = torch.nn.ModuleList([SevenLinearBlock() for _ in range(28)])
        self.model.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 10)
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=28)


class ToyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 3, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[0.1], [-0.2], [0.3]]))
        self.forward_cache_values = []

    def forward(self, *, input_ids, use_cache):
        self.forward_cache_values.append(use_cache)
        return SimpleNamespace(logits=self.proj(input_ids.float().unsqueeze(-1)))


def two_matrix_case():
    first = torch.nn.Linear(2, 2, bias=False)
    second = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        second.weight.copy_(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
    first.weight.grad = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    second.weight.grad = torch.tensor([[3.0, 6.0], [9.0, 12.0]])
    return [("z_matrix", second), ("a_matrix", first)]


class GenericGradientLinearResponseTests(unittest.TestCase):
    def test_frozen_specs_and_information_policy(self):
        spec = rollback.load_spec(SPEC_PATH)
        corpus_spec = rollback.load_corpus_spec(CORPUS_SPEC_PATH)
        self.assertEqual(spec, rollback.FROZEN_SPEC)
        self.assertEqual(corpus_spec, rollback.FROZEN_CORPUS_SPEC)
        self.assertEqual(spec["information_policy"], "final_checkpoint_plus_frozen_generic_corpus_and_probe")
        self.assertEqual(spec["eligible_tensors"]["transformer_block_indices"], list(range(14)))
        self.assertEqual(spec["generic_loss"]["total_prediction_tokens"], 4096 * 127)
        self.assertEqual(spec["generic_loss"]["number_of_batches"], 512)
        self.assertEqual(spec["tangent"]["target_relative_frobenius"], 0.00125)

    def test_exact_98_discovery_and_freeze_boundary(self):
        model = MockQwen()
        rollback.validate_loaded_model(model, torch)
        eligible = rollback.discover_eligible_linear_weights(model, torch)
        names = [name for name, _ in eligible]
        self.assertEqual(len(eligible), 98)
        self.assertEqual(names, sorted(names))
        for block in range(14):
            self.assertEqual(sum(name.startswith(f"model.layers.{block}.") for name in names), 7)
        self.assertFalse(any(name.startswith("model.layers.14.") for name in names))
        selected = rollback.freeze_other_parameters(model, eligible)
        self.assertEqual(sum(parameter.requires_grad for parameter in model.parameters()), 98)
        self.assertFalse(model.model.embed_tokens.weight.requires_grad)
        self.assertFalse(model.model.norm.weight.requires_grad)
        self.assertFalse(model.lm_head.weight.requires_grad)
        self.assertFalse(model.model.layers[0].input_norm.weight.requires_grad)
        self.assertFalse(model.model.layers[0].self_attn["q_proj"].bias.requires_grad)
        self.assertFalse(model.model.layers[14].self_attn["q_proj"].weight.requires_grad)
        before = rollback.frozen_parameter_hashes(model, selected, torch)
        rollback.verify_frozen_parameters(model, selected, before, torch)
        model.lm_head.weight.grad = torch.ones_like(model.lm_head.weight)
        with self.assertRaisesRegex(ValueError, "Forbidden parameter"):
            rollback.verify_frozen_parameters(model, selected, before, torch)
        model.lm_head.weight.grad = None
        with torch.no_grad():
            model.lm_head.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Forbidden parameter"):
            rollback.verify_frozen_parameters(model, selected, before, torch)

    def test_selected_weight_cannot_alias_forbidden_parameter(self):
        model = torch.nn.Module()
        model.eligible = torch.nn.Linear(2, 2, bias=False)
        model.other = torch.nn.Module()
        model.other.weight = model.eligible.weight
        with self.assertRaisesRegex(ValueError, "shared with forbidden"):
            rollback.freeze_other_parameters(model, [("eligible", model.eligible)])

    def test_causal_loss_aligns_127_next_token_predictions(self):
        logits = torch.zeros((1, 128, 4), dtype=torch.float32)
        labels = torch.arange(128, dtype=torch.int64).reshape(1, 128) % 4
        logits[0, 0, 1] = 4.0
        logits[0, 126, int(labels[0, 127])] = 3.0
        logits[0, 127, 0] = 100.0  # Last logit must not enter the loss.
        actual = rollback.causal_token_loss_sum(logits, labels, torch)
        expected = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1), reduction="sum"
        )
        self.assertEqual(labels[:, 1:].numel(), 127)
        self.assertTrue(torch.equal(actual, expected))

    def test_exact_mean_gradient_across_equal_batches(self):
        tokens = torch.tensor([
            [0, 1, 2, 1], [2, 0, 1, 2], [1, 2, 0, 1], [2, 1, 0, 2],
        ], dtype=torch.int64)
        loss_spec = {
            "sample_count": 4, "sequence_length": 4,
            "predictions_per_sample": 3, "total_prediction_tokens": 12,
            "batch_size": 2, "number_of_batches": 2,
        }
        model = ToyCausalModel()
        actual_loss = rollback.accumulate_mean_generic_gradient(
            model, tokens, [("proj", model.proj)], loss_spec, torch
        )
        reference = ToyCausalModel()
        full_logits = reference(input_ids=tokens, use_cache=False).logits
        mean = torch.nn.functional.cross_entropy(
            full_logits[:, :-1].reshape(-1, 3), tokens[:, 1:].reshape(-1), reduction="mean"
        )
        mean.backward()
        self.assertAlmostEqual(actual_loss, float(mean.item()), places=6)
        self.assertTrue(torch.allclose(model.proj.weight.grad, reference.proj.weight.grad, atol=1e-6))
        self.assertEqual(model.forward_cache_values, [False, False])
        self.assertFalse(model.training)

    def test_zero_missing_and_nonfinite_gradients_fail(self):
        eligible = two_matrix_case()
        for _, module in eligible:
            module.weight.grad.zero_()
        with self.assertRaisesRegex(ValueError, "zero or non-finite"):
            rollback.global_tangent_scale(eligible, 0.00125, torch)
        eligible[0][1].weight.grad = None
        with self.assertRaisesRegex(ValueError, "Missing or non-finite"):
            rollback.global_tangent_scale(eligible, 0.00125, torch)
        eligible[0][1].weight.grad = torch.full_like(eligible[0][1].weight, float("nan"))
        with self.assertRaisesRegex(ValueError, "Missing or non-finite"):
            rollback.global_tangent_scale(eligible, 0.00125, torch)

    def test_corpus_serialized_and_raw_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.pt"
            tokens = torch.arange(12, dtype=torch.int64).reshape(3, 4)
            torch.save(tokens, path)
            raw_hash = rollback.sha256_raw_int64_tensor(tokens, torch)
            self.assertTrue(torch.equal(
                rollback.load_corpus_tokens(path, raw_hash, 3, 4, torch), tokens
            ))
            with self.assertRaisesRegex(ValueError, "raw tensor SHA-256 mismatch"):
                rollback.load_corpus_tokens(path, "0" * 64, 3, 4, torch)
            with self.assertRaisesRegex(ValueError, "shape"):
                rollback.load_corpus_tokens(path, raw_hash, 4, 4, torch)
            model_dir = Path(directory) / "merged"
            model_dir.mkdir()
            for name in rollback.TOKENIZER_FILE_NAMES:
                (model_dir / name).write_text(name, encoding="utf-8")
            spec = rollback.load_corpus_spec(CORPUS_SPEC_PATH)
            records = rollback.tokenizer_file_records(model_dir)
            manifest = {
                "format_version": 1, "attempt_id": spec["attempt_id"], "hash_algorithm": "sha256",
                "corpus_spec_sha256": rollback.sha256_file(CORPUS_SPEC_PATH),
                "freeze_script_sha256": "a" * 64,
                "dataset": spec["dataset"], "selection": spec["selection"],
                "tokenizer_checkpoint": {
                    "source": "canonical_merged_checkpoint", "directory": str(model_dir),
                    "file_count": len(records), "files": records,
                },
                "tensor": {"shape": [4096, 128], "dtype": "torch.int64", "device": "cpu", "contiguous": True},
                "artifact": {"path": str(path), "serialized_sha256": rollback.sha256_file(path),
                             "raw_tensor_sha256": raw_hash},
            }
            manifest_path = Path(directory) / "corpus-manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            context = rollback.verify_corpus_manifest(
                manifest_path, CORPUS_SPEC_PATH, path, model_dir, spec
            )
            self.assertEqual(context["raw_tensor_sha256"], raw_hash)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
                rollback.verify_corpus_manifest(
                    manifest_path, CORPUS_SPEC_PATH, path, model_dir, spec
                )

    def test_source_and_corpus_immutability_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "merged"
            source.mkdir()
            (source / "config.json").write_text("source", encoding="utf-8")
            files = rollback.checkpoint_file_records(source)
            corpus = root / "tokens.pt"
            corpus.write_bytes(b"corpus")
            hashes = {corpus: rollback.sha256_file(corpus)}
            rollback.verify_unchanged(source, files, hashes)
            corpus.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                rollback.verify_unchanged(source, files, hashes)
            corpus.write_bytes(b"corpus")
            (source / "config.json").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Source checkpoint changed"):
                rollback.verify_unchanged(source, files, hashes)
            output = root / "model"
            manifest = root / "construction-manifest.json"
            rollback.require_output_absent(output, manifest)
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "already exists"):
                rollback.require_output_absent(output, manifest)
            output.rmdir()
            manifest.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                rollback.write_manifest(manifest, {"new": True})

    def test_gradient_and_norm_functions_match_attempt005(self):
        original = ast.parse((PROJECT / "scripts/ablation/construct_generic_gradient_rollback.py").read_text())
        current = ast.parse(SCRIPT_PATH.read_text())
        for old_name, new_name in (("accumulate_mean_generic_gradient", "accumulate_mean_generic_gradient"),
                                   ("global_rollback_scale", "global_tangent_scale")):
            left = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == old_name)
            right = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == new_name)
            left.name = new_name
            self.assertEqual(ast.dump(left), ast.dump(right))

    def test_gradients_accumulate_without_zeroing_between_batches(self):
        model = ToyCausalModel()
        model.proj.weight.grad = torch.ones_like(model.proj.weight)
        seen = []
        def hook(module, args):
            seen.append(None if module.proj.weight.grad is None else module.proj.weight.grad.clone())
        model.register_forward_pre_hook(hook)
        tokens = torch.tensor([[1, 0], [2, 1], [1, 2], [2, 0]])
        loss_spec = dict(sample_count=4, sequence_length=2, predictions_per_sample=1,
                         total_prediction_tokens=4, batch_size=2, number_of_batches=2)
        with patch.object(model, "zero_grad", wraps=model.zero_grad) as zero:
            rollback.accumulate_mean_generic_gradient(model, tokens, [("proj", model.proj)], loss_spec, torch)
        self.assertEqual(zero.call_count, 1)
        self.assertIsNone(seen[0])
        reference = ToyCausalModel()
        loss = rollback.causal_token_loss_sum(reference(input_ids=tokens[:2], use_cache=False).logits, tokens[:2], torch) / 4
        loss.backward()
        self.assertTrue(torch.equal(seen[1], reference.proj.weight.grad))

    def test_positive_tangent_global_scale_realized_norm_and_no_weight_mutation(self):
        eligible = two_matrix_case()
        weights = {n: m.weight.detach().clone() for n, m in eligible}
        grads = {n: m.weight.grad.detach().double().clone() for n, m in eligible}
        scale = rollback.global_tangent_scale(eligible, .00125, torch)
        tangents, records, realized = rollback.prepare_tangents(eligible, scale, torch)
        norm = math.sqrt(sum(float(g.square().sum()) for g in grads.values()))
        wn = math.sqrt(sum(float(w.double().square().sum()) for w in weights.values()))
        self.assertAlmostEqual(scale["alpha"], .00125 * wn / norm)
        self.assertAlmostEqual(scale["alpha"] * norm / wn, .00125)
        for name, module in eligible:
            tangent = tangents[name + ".weight"]
            self.assertTrue(torch.equal(tangent, (scale["alpha"] * grads[name]).float()))
            self.assertTrue(torch.equal(module.weight, weights[name]))
            self.assertIsNone(module.weight.grad)
        tn = math.sqrt(sum(float(t.double().square().sum()) for t in tangents.values()))
        self.assertEqual(realized["aggregate_realized_fp32_tangent_norm"], tn)
        self.assertEqual(realized["aggregate_realized_relative_tangent_norm"], tn / wn)
        self.assertEqual([r["name"] for r in records], sorted(tangents))

    def test_probe_hashes_and_frozen_definition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            probe = torch.arange(12).reshape(3, 4)
            torch.save(probe, path)
            spec = dict(sample_count=3, sequence_length=4, serialized_sha256=rollback.sha256_file(path),
                        raw_tensor_sha256=rollback.sha256_raw_int64_tensor(probe, torch))
            self.assertTrue(torch.equal(rollback.load_probe(path, spec, torch), probe))
            for key in ("serialized_sha256", "raw_tensor_sha256"):
                with self.assertRaises(ValueError):
                    rollback.load_probe(path, {**spec, key: "0" * 64}, torch)
            with self.assertRaises(ValueError):
                rollback.load_probe(path, {**spec, "sequence_length": 3}, torch)
        frozen = rollback.FROZEN_SPEC["probe"]
        self.assertEqual(frozen["serialized_sha256"], "3d799d19abf4a2d6dc92b2bd164d81d83f6451cd3545f5ae5d8c04a8b299735b")
        self.assertEqual(frozen["raw_tensor_sha256"], "73f56300fdd06bc8fafcfa5750fee7a586a5c5e6071a608667ffc526bbb90ac8")
        self.assertEqual(frozen["batch_size"], 32)

    def test_information_firewall_and_no_checkpoint_output(self):
        source = SCRIPT_PATH.read_text().replace("oracle_probe", "generic_probe")
        for forbidden in ("oracle", "adapter", "models/base", "evaluation", "self_diff", "self-diff",
                          "cosine", "semantic_labels", "attempt006", "attempt007", "save_pretrained",
                          "attn_implementation=", "torch.optim", "jacobian", "jacrev", "jacfwd"):
            self.assertNotIn(forbidden, source)
        self.assertFalse(rollback.FROZEN_SPEC["outputs"]["standalone_checkpoint"])
        self.assertEqual(rollback.FROZEN_SPEC["jvp"]["response_sign"], "positive_J_delta")
        self.assertFalse(rollback.FROZEN_SPEC["jvp"]["fallback"])


class TinyBlock(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
        self.nonlinear = nonlinear

    def forward(self, hidden):
        value = self.proj(hidden)
        return torch.tanh(value) if self.nonlinear else value


class TinyTransformer(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock(nonlinear and i == 0) for i in range(28)])
        self.register_buffer("offset", torch.tensor([1., 2.]))
        self.bad_index = False
        self.detach_output = False
        self.calls = []

    def forward(self, *, input_ids, use_cache, output_hidden_states, return_dict):
        assert use_cache is False and output_hidden_states is True and return_dict is True
        self.calls.append(tuple(input_ids.shape))
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2) * .1 + self.offset
        states = []
        for layer in self.layers:
            states.append(hidden)
            hidden = layer(hidden)
        states.append(hidden * 7)  # Distinct final normalization, never the readout.
        if self.bad_index:
            states[14] = states[13]
        if self.detach_output:
            states = [value.detach() for value in states]
        return SimpleNamespace(hidden_states=tuple(states))


class TinyModel(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.model = TinyTransformer(nonlinear)
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=28, hidden_size=2)


def tiny_spec(count=5, length=3, batch=2):
    spec = copy.deepcopy(rollback.FROZEN_SPEC)
    spec["probe"].update(sample_count=count, sequence_length=length, batch_size=batch)
    spec["readout"]["hidden_size"] = 2
    return spec


class LinearResponseJVPTests(unittest.TestCase):
    def test_analytical_jvp_partial_parameters_and_no_grad(self):
        model, spec = TinyModel(), tiny_spec()
        tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
        vector = torch.tensor([[.3, -.2], [.5, .1]])
        tangents = {"model.layers.0.proj.weight": vector}
        before = rollback.model_state_hashes(model, torch)
        with patch.object(torch.func, "functional_call", wraps=torch.func.functional_call) as call:
            primal, response = rollback.run_readout_jvp(model, tokens, tangents, spec, torch)
        self.assertEqual(set(call.call_args.args[1]), {"layers.0.proj.weight"})
        self.assertFalse(call.call_args.kwargs["strict"])
        inputs = tokens.float().unsqueeze(-1).expand(-1, -1, 2) * .1 + model.model.offset
        self.assertTrue(torch.equal(primal, inputs))
        self.assertTrue(torch.allclose(response, inputs @ vector.T))
        self.assertFalse(primal.requires_grad)
        self.assertFalse(response.requires_grad)
        rollback.verify_model_unchanged(model, before, torch)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable for synthetic test")
    def test_cuda_analytical_jvp_uses_fp32_device_tangent(self):
        model, spec = TinyModel().cuda(), tiny_spec()
        tokens = torch.tensor([[0, 1, 2]], device="cuda")
        vector = torch.eye(2, device="cuda")
        ordinary = rollback.ordinary_hook_readout(model, tokens, spec, torch)
        primal, tangent = rollback.run_readout_jvp(
            model, tokens, {"model.layers.0.proj.weight": vector}, spec, torch)
        self.assertEqual(primal.device.type, "cuda")
        self.assertEqual(tangent.dtype, torch.float32)
        self.assertTrue(torch.equal(primal, ordinary))
        self.assertTrue(torch.equal(tangent, ordinary))

    def test_local_model_loader_keeps_default_attention_and_fp32(self):
        import types
        fake = types.ModuleType("transformers")
        model = MockQwen()
        calls = []
        def load(path, **kwargs):
            calls.append((path, kwargs))
            return model
        fake.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load)
        with patch.dict(sys.modules, {"transformers": fake}), patch.dict(rollback.os.environ):
            result = rollback.load_local_model(Path("synthetic-merged"), "cpu", torch)
            self.assertEqual(rollback.os.environ["HF_HUB_OFFLINE"], "1")
        self.assertIs(result, model)
        self.assertFalse(model.training)
        self.assertEqual(calls, [(Path("synthetic-merged"), {"dtype": torch.float32, "local_files_only": True})])

    def test_nonlinear_jvp_matches_centered_finite_difference_positive_sign(self):
        model, spec = TinyModel(nonlinear=True), tiny_spec()
        tokens = torch.tensor([[0, 1, 2]])
        vector = torch.tensor([[.3, -.2], [.5, .1]])
        name = "model.layers.0.proj.weight"
        primal, response = rollback.run_readout_jvp(model, tokens, {name: vector}, spec, torch)
        weight = model.model.layers[0].proj.weight.detach()
        epsilon = 1e-3
        with torch.no_grad():
            def evaluate(value):
                return torch.func.functional_call(model.model, {"layers.0.proj.weight": value}, (),
                    dict(input_ids=tokens, use_cache=False, output_hidden_states=True, return_dict=True)).hidden_states[14]
            plus, minus = evaluate(weight + epsilon * vector), evaluate(weight - epsilon * vector)
        reference = (plus - minus) / (2 * epsilon)
        self.assertTrue(torch.allclose(response, reference, atol=5e-5, rtol=1e-3))
        self.assertGreater(float((response * (primal - minus)).sum()), 0.)

    def test_hidden_state_index_matches_block13_output_not_input_or_final_norm(self):
        model, spec = TinyModel(), tiny_spec()
        with torch.no_grad():
            model.model.layers[13].proj.weight.mul_(3.)
        tokens = torch.tensor([[0, 1, 2]])
        ordinary = rollback.ordinary_hook_readout(model, tokens, spec, torch)
        output = model.model(input_ids=tokens, use_cache=False, output_hidden_states=True, return_dict=True)
        self.assertTrue(torch.equal(ordinary, output.hidden_states[14]))
        self.assertFalse(torch.equal(ordinary, output.hidden_states[13]))
        self.assertFalse(torch.equal(ordinary, output.hidden_states[-1]))
        wrong = copy.deepcopy(spec)
        wrong["readout"]["hidden_state_index"] = 13
        with self.assertRaisesRegex(ValueError, "hidden-state index"):
            rollback.ordinary_hook_readout(model, tokens, wrong, torch)
        model.model.bad_index = True
        with self.assertRaisesRegex(ValueError, "does not equal block output hook"):
            rollback.ordinary_hook_readout(model, tokens, spec, torch)
        self.assertEqual(len(model.model.layers[13]._forward_hooks), 0)

    def test_unsupported_forward_ad_detach_zero_and_bad_primal_fail_without_fallback(self):
        model, spec = TinyModel(), tiny_spec()
        probe = torch.zeros(5, 3, dtype=torch.int64)
        eligible = [("model.layers.0.proj", model.model.layers[0].proj)]
        with patch.object(torch.func, "jvp", side_effect=NotImplementedError("unsupported")):
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                rollback.smoke_test_jvp(model, probe, eligible, spec, torch)
        model.model.detach_output = True
        with self.assertRaisesRegex(RuntimeError, "no fallback"):
            rollback.smoke_test_jvp(model, probe, eligible, spec, torch)
        model.model.detach_output = False
        primal = rollback.ordinary_hook_readout(model, probe[:1], spec, torch)
        with patch.object(rollback, "run_readout_jvp", return_value=(primal, torch.zeros_like(primal))):
            with self.assertRaisesRegex(ValueError, "zero or non-finite"):
                rollback.smoke_test_jvp(model, probe, eligible, spec, torch)
        with patch.object(rollback, "run_readout_jvp", return_value=(primal + 1, torch.ones_like(primal))):
            with self.assertRaisesRegex(ValueError, "primal disagrees"):
                rollback.smoke_test_jvp(model, probe, eligible, spec, torch)

    def test_cpu_float64_accumulation_and_stored_artifact_contract(self):
        spec = tiny_spec(count=3, length=1)
        accumulator = rollback.ResponseAccumulator(spec, torch)
        values = torch.tensor([[[1e8, 1e8]], [[1., 1.]], [[-1e8, -1e8]]])
        accumulator.add(values, -values)
        self.assertEqual(accumulator.primal_sum.dtype, torch.float64)
        merged, response = accumulator.means(3)
        self.assertTrue(torch.equal(merged, torch.full((1, 2), 1/3, dtype=torch.float64)))
        artifact = rollback.build_artifact(merged, response, spec, torch)
        self.assertEqual(set(artifact), {"merged_mean", "linear_response", "metadata"})
        for key in ("merged_mean", "linear_response"):
            self.assertEqual(artifact[key].shape, (1, 2))
            self.assertEqual(artifact[key].dtype, torch.float32)
            self.assertEqual(artifact[key].device.type, "cpu")
            self.assertTrue(artifact[key].is_contiguous())
        with self.assertRaises(ValueError):
            accumulator.means(4)

    def test_probe_averaging_handles_final_short_batch(self):
        model, spec = TinyModel(), tiny_spec()
        probe = torch.arange(15).reshape(5, 3)
        tangent = {"model.layers.0.proj.weight": torch.eye(2)}
        merged, response, check = rollback.compute_probe_response(model, probe, tangent, spec, torch)
        expected = (probe.float().unsqueeze(-1).expand(-1, -1, 2) * .1 + model.model.offset).double().mean(0)
        self.assertTrue(torch.equal(merged, expected))
        self.assertTrue(torch.equal(response, expected))
        self.assertEqual(model.model.calls, [(2, 3), (2, 3), (2, 3), (1, 3)])
        self.assertTrue(check["readout_hook_verified"])
        self.assertEqual(check["first_batch_primal_max_abs_difference"], 0.)

    def test_invalid_readouts_and_stored_values_fail(self):
        spec = tiny_spec()
        good = torch.zeros(2, 3, 2)
        for bad in (good.double(), good[:, :2], torch.full_like(good, float("nan")), torch.full_like(good, float("inf"))):
            with self.assertRaises(ValueError):
                rollback.validate_readout(bad, 2, spec, torch, "synthetic")
        means = torch.zeros(3, 2, dtype=torch.float64)
        for bad in (means.float(), means[:2], torch.full_like(means, 1e39), torch.full_like(means, float("nan")),
                    torch.zeros(2, 3, dtype=torch.float64).t()):
            with self.assertRaises(ValueError):
                rollback.build_artifact(bad, means, spec, torch)

    def test_parameter_and_buffer_mutation_audit(self):
        model = TinyModel()
        before = rollback.model_state_hashes(model, torch)
        with torch.no_grad():
            model.model.layers[0].proj.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "changed"):
            rollback.verify_model_unchanged(model, before, torch)
        before = rollback.model_state_hashes(model, torch)
        model.model.offset.add_(1)
        with self.assertRaisesRegex(ValueError, "changed"):
            rollback.verify_model_unchanged(model, before, torch)

    def test_manifest_is_deterministic_and_records_positive_jvp(self):
        spec = rollback.FROZEN_SPEC
        hashes = dict(attempt_spec_sha256="a"*64, corpus_spec_sha256="b"*64,
                      corpus_manifest_sha256="c"*64, constructor_script_sha256="d"*64)
        raw = dict(merged_mean="e"*64, linear_response="f"*64)
        args = (spec, hashes, [], dict(serialized_sha256="1"*64, raw_tensor_sha256="2"*64),
                1., {"alpha": .001}, [], {"aggregate_realized_fp32_tangent_norm": 1.},
                {"readout_hook_verified": True}, "3"*64, raw)
        first = rollback.build_manifest(*args)
        second = rollback.build_manifest(*copy.deepcopy(args))
        self.assertEqual(json.dumps(first, allow_nan=False), json.dumps(second, allow_nan=False))
        self.assertEqual(first["jvp"]["response_sign"], "positive_J_delta")
        self.assertEqual(first["loss"], {**spec["generic_loss"], "mean_generic_loss": 1.})
        self.assertEqual(first["artifact"]["raw_tensors_sha256"], raw)
        self.assertFalse({"timestamp", "gpu", "host", "output_checkpoint"} & set(first))

    def test_synthetic_full_pipeline_artifact_hashes_and_source_immutability(self):
        class CausalTiny(TinyModel):
            def __init__(self):
                super().__init__(nonlinear=True)
                self.lm_head = torch.nn.Linear(2, 3, bias=False)
                with torch.no_grad():
                    self.lm_head.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., .5]]))
            def forward(self, *, input_ids, use_cache):
                output = self.model(input_ids=input_ids, use_cache=use_cache,
                                    output_hidden_states=True, return_dict=True)
                return SimpleNamespace(logits=self.lm_head(output.hidden_states[-1]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            merged_dir.mkdir()
            (merged_dir / "weights").write_bytes(b"synthetic model")
            spec = tiny_spec()
            spec["generic_loss"].update(sample_count=4, sequence_length=3,
                predictions_per_sample=2, total_prediction_tokens=8, batch_size=2, number_of_batches=2)
            tokens = torch.tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]])
            corpus_path, probe_path = root / "tokens.pt", root / "probe.pt"
            torch.save(tokens, corpus_path)
            probe = torch.arange(15).reshape(5, 3) % 3
            torch.save(probe, probe_path)
            spec["probe"].update(serialized_sha256=rollback.sha256_file(probe_path),
                raw_tensor_sha256=rollback.sha256_raw_int64_tensor(probe, torch))
            context = dict(serialized_sha256=rollback.sha256_file(corpus_path),
                           raw_tensor_sha256=rollback.sha256_raw_int64_tensor(tokens, torch))
            for name in ("spec.json", "corpus_spec.json", "corpus-manifest.json"):
                (root / name).write_text("{}")
            args = rollback.parse_args(["--device", "cpu", "--merged-model-dir", str(merged_dir),
                "--attempt-spec-path", str(root / "spec.json"), "--tokens-path", str(corpus_path),
                "--corpus-spec-path", str(root / "corpus_spec.json"),
                "--corpus-manifest-path", str(root / "corpus-manifest.json"),
                "--probe-path", str(probe_path), "--artifact-path", str(root / "response.pt"),
                "--construction-manifest-path", str(root / "manifest.json")])
            model = CausalTiny()
            before = rollback.model_state_hashes(model, torch)
            def run():
                with patch.object(rollback, "load_spec", return_value=spec), \
                     patch.object(rollback, "load_corpus_spec", return_value={}), \
                     patch.object(rollback, "verify_corpus_manifest", return_value=context), \
                     patch.object(rollback, "load_local_model", return_value=model), \
                     patch.object(rollback, "discover_eligible_linear_weights", return_value=[("model.layers.0.proj", model.model.layers[0].proj)]):
                    return rollback.construct(args)
            run()
            rollback.verify_model_unchanged(model, before, torch)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            artifact = torch.load(args.artifact_path, weights_only=True)
            manifest = json.loads(args.construction_manifest_path.read_text())
            self.assertEqual(set(artifact), {"merged_mean", "linear_response", "metadata"})
            self.assertEqual(manifest["artifact"]["serialized_sha256"], rollback.sha256_file(args.artifact_path))
            for key in ("merged_mean", "linear_response"):
                self.assertEqual(manifest["artifact"]["raw_tensors_sha256"][key], rollback.sha256_raw_float32_tensor(artifact[key], torch))
            self.assertEqual(manifest["eligible_parameters"], ["model.layers.0.proj.weight"])
            self.assertGreater(manifest["tangent"]["alpha"], 0.)
            self.assertEqual(manifest["loss"]["number_of_batches"], 2)
            with self.assertRaisesRegex(ValueError, "already exists"):
                run()
            args.artifact_path.unlink()
            args.construction_manifest_path.unlink()
            compute = rollback.compute_probe_response
            def mutate(*arguments):
                result = compute(*arguments)
                (merged_dir / "weights").write_bytes(b"changed")
                return result
            with patch.object(rollback, "compute_probe_response", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "Source checkpoint changed"):
                    run()
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.construction_manifest_path.exists())

    def test_smoke_cli_uses_real_jvp_but_no_corpus_gradient_or_output_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "merged"
            model_path.mkdir()
            (model_path / "config.json").write_text("synthetic")
            spec = tiny_spec()
            probe_path = root / "probe.pt"
            probe = torch.ones(5, 3, dtype=torch.int64)
            torch.save(probe, probe_path)
            spec["probe"].update(serialized_sha256=rollback.sha256_file(probe_path),
                                  raw_tensor_sha256=rollback.sha256_raw_int64_tensor(probe, torch))
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec))
            model = TinyModel()
            eligible = [("model.layers.0.proj", model.model.layers[0].proj)]
            args = ["--smoke-test-jvp", "--device", "cpu", "--merged-model-dir", str(model_path),
                    "--attempt-spec-path", str(spec_path), "--probe-path", str(probe_path),
                    "--corpus-spec-path", str(root / "must-not-read"), "--artifact-path", str(root / "out.pt"),
                    "--construction-manifest-path", str(root / "out.json")]
            files_before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            with patch.object(rollback, "load_spec", return_value=spec), \
                 patch.object(rollback, "load_local_model", return_value=model), \
                 patch.object(rollback, "discover_eligible_linear_weights", return_value=eligible), \
                 patch.object(rollback, "accumulate_mean_generic_gradient", side_effect=AssertionError("no gradient")), \
                 patch.object(rollback, "verify_corpus_manifest", side_effect=AssertionError("no corpus")), \
                 patch.object(torch, "save", side_effect=AssertionError("no save")), \
                 patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(rollback.main(args), 0)
            report = json.loads(output.getvalue())
            self.assertEqual(report["primal_max_absolute_difference"], 0.)
            self.assertGreater(report["jvp_norm"], 0.)
            self.assertEqual(files_before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
