import ast
import copy
import hashlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/007_diag_empirical_fisher_sqrt_rollback_prefix0_13_r00125"
SCRIPT_PATH = PROJECT / "scripts/ablation/construct_diag_empirical_fisher_sqrt_rollback.py"
SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
CORPUS_SPEC_PATH = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json"
MODULE_SPEC = importlib.util.spec_from_file_location("construct_diag_empirical_fisher_sqrt_rollback", SCRIPT_PATH)
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


def moment_case(eligible):
    return {name: (module.weight.grad.double().clone(), module.weight.grad.double().square() + 1)
            for name, module in eligible}


class DiagEmpiricalFisherSqrtRollbackTests(unittest.TestCase):
    def test_frozen_specs_and_information_policy(self):
        spec = rollback.load_spec(SPEC_PATH)
        corpus_spec = rollback.load_corpus_spec(CORPUS_SPEC_PATH)
        self.assertEqual(spec, rollback.FROZEN_SPEC)
        self.assertEqual(spec["attempt_id"], "007_diag_empirical_fisher_sqrt_rollback_prefix0_13_r00125")
        self.assertEqual(spec["method"], "diag_empirical_fisher_sqrt_rollback")
        self.assertEqual(spec["empirical_fisher"]["direction"], "d = g / sqrt(F + lambda)")
        self.assertEqual(spec["rollback"]["direction"], "W_new = W - alpha * (g / sqrt(F + lambda))")
        self.assertEqual(spec["empirical_fisher"]["accumulation_dtype"], "float32")
        self.assertEqual(spec["empirical_fisher"]["accumulation_device"], "same_as_eligible_weights")
        self.assertEqual(spec["empirical_fisher"]["final_accounting_dtype"], "cpu_float64")
        self.assertEqual(corpus_spec, rollback.FROZEN_CORPUS_SPEC)
        self.assertEqual(spec["information_policy"], "final_checkpoint_plus_frozen_generic_text")
        self.assertEqual(spec["eligible_tensors"]["transformer_block_indices"], list(range(14)))
        self.assertEqual(spec["generic_loss"]["total_prediction_tokens"], 4096 * 127)
        self.assertEqual(spec["generic_loss"]["number_of_batches"], 4096)
        self.assertEqual(spec["rollback"]["target_relative_frobenius"], 0.00125)

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

    def test_independent_sequence_gradients_zeroing_and_moments(self):
        tokens = torch.tensor([[0, 1, 2, 1], [2, 0, 1, 2], [1, 2, 0, 1]])
        model = ToyCausalModel()
        model.proj.weight.grad = torch.full_like(model.proj.weight, 999.)
        seen = []
        model.register_forward_pre_hook(lambda module, args: seen.append(module.proj.weight.grad))
        grads, losses = [], []
        for row in tokens:
            reference = ToyCausalModel()
            logits = reference(input_ids=row[None], use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 3), row[1:])
            loss.backward()
            grads.append(reference.proj.weight.grad.double())
            losses.append(loss.item())
        loss_spec = dict(sample_count=3, sequence_length=4, predictions_per_sample=3,
                         total_prediction_tokens=9, batch_size=1, number_of_batches=3)
        actual_loss, moments = rollback.accumulate_empirical_fisher(
            model, tokens, [("proj", model.proj)], loss_spec, torch)
        first, fisher = moments["proj"]
        self.assertEqual(seen, [None] * 3)
        self.assertIsNone(model.proj.weight.grad)
        self.assertTrue(torch.allclose(first.double(), torch.stack(grads).mean(0), atol=1e-7))
        self.assertTrue(torch.allclose(fisher.double(), torch.stack(grads).square().mean(0), atol=1e-7))
        self.assertFalse(torch.allclose(fisher, first.square()))
        self.assertAlmostEqual(actual_loss, sum(losses) / 3, places=6)
        self.assertEqual(model.forward_cache_values, [False] * 3)
        self.assertFalse(model.training)

    def test_accumulators_stay_fp32_on_parameter_device_during_every_sequence(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = ToyCausalModel().to(device)
                tokens = torch.tensor([[1, 0], [2, 1], [1, 2]])
                spec = dict(sample_count=3, sequence_length=2, predictions_per_sample=1,
                            total_prediction_tokens=3, batch_size=1, number_of_batches=3)
                accumulators, snapshots, gradients = [], [], []
                zeros_like = torch.zeros_like
                tensor_to = torch.Tensor.to
                def allocate(*args, **kwargs):
                    value = zeros_like(*args, **kwargs)
                    accumulators.append(value)
                    return value
                def guard_to(tensor, *args, **kwargs):
                    # Accumulation must not move any floating tensor off its device
                    # or convert it to float64, including a sequence gradient.
                    result = tensor_to(tensor, *args, **kwargs)
                    if tensor.is_floating_point():
                        self.assertEqual(result.device, tensor.device)
                        self.assertEqual(result.dtype, torch.float32)
                    return result
                def before_forward(module, args):
                    self.assertEqual(len(accumulators), 2)
                    self.assertIsNone(module.proj.weight.grad)
                    for value in accumulators:
                        self.assertEqual(value.dtype, torch.float32)
                        self.assertEqual(value.device, module.proj.weight.device)
                    snapshots.append(tuple(v.clone() for v in accumulators))
                model.register_forward_pre_hook(before_forward)
                model.proj.weight.register_hook(lambda grad: gradients.append(grad.detach().clone()))
                with patch.object(torch, "zeros_like", side_effect=allocate), patch.object(torch.Tensor, "to", guard_to):
                    _, moments = rollback.accumulate_empirical_fisher(
                        model, tokens, [("proj", model.proj)], spec, torch)
                first, fisher = moments["proj"]
                self.assertIs(first, accumulators[0])
                self.assertIs(fisher, accumulators[1])
                running_first, running_second = (torch.zeros_like(first) for _ in range(2))
                for index, grad in enumerate(gradients):
                    self.assertTrue(torch.equal(snapshots[index][0], running_first))
                    self.assertTrue(torch.equal(snapshots[index][1], running_second))
                    running_first.add_(grad)
                    running_second.addcmul_(grad, grad)
                self.assertTrue(torch.equal(first, running_first / 3))
                self.assertTrue(torch.equal(fisher, running_second / 3))
                self.assertIsNone(model.proj.weight.grad)

    def test_fp32_moments_cpu64_final_direction_and_all_diagnostics(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                small = torch.nn.Linear(1, 1, bias=False).to(device)
                large = torch.nn.Linear(3, 1, bias=False).to(device)
                with torch.no_grad():
                    small.weight.fill_(2.)
                    large.weight.copy_(torch.tensor([[1., 3., -4.]], device=device))
                eligible = [("small", small), ("large", large)]
                moments = {
                    "small": (torch.tensor([[.1]], device=device), torch.tensor([[2.]], device=device)),
                    "large": (torch.tensor([[.3, -.7, 1.1]], device=device),
                              torch.tensor([[4., 8., 10.]], device=device)),
                }
                originals = {n: m.weight.detach().cpu().double() for n, m in eligible}
                reference = {n: (g.cpu().double(), f.cpu().double()) for n, (g, f) in moments.items()}
                directions = {n: g / (f + 6.).sqrt() for n, (g, f) in reference.items()}
                real_direction = rollback.preconditioned_direction
                def checked_direction(first, fisher, damping, torch_module):
                    for value in (first, fisher):
                        self.assertEqual(value.device.type, "cpu")
                        self.assertEqual(value.dtype, torch.float64)
                    direction = real_direction(first, fisher, damping, torch_module)
                    self.assertTrue(torch.equal(direction, first / (fisher + damping).sqrt()))
                    return direction
                with patch.object(rollback, "preconditioned_direction", side_effect=checked_direction):
                    scale = rollback.global_rollback_scale(eligible, moments, .00125, torch)
                    records, realized = rollback.apply_global_rollback(
                        eligible, moments, scale["lambda"], scale["alpha"],
                        scale["aggregate_source_weight_norm"], torch)
                self.assertEqual(moments, {})
                self.assertEqual(scale["eligible_scalar_coordinate_count"], 4)
                self.assertEqual(scale["empirical_fisher_total_sum"], 24.)
                self.assertEqual(scale["empirical_fisher_global_scalar_mean"], 6.)
                self.assertEqual(scale["lambda"], 6.)
                gn = math.sqrt(sum(g.square().sum().item() for g, _ in reference.values()))
                dn = math.sqrt(sum(d.square().sum().item() for d in directions.values()))
                wn = math.sqrt(sum(w.square().sum().item() for w in originals.values()))
                self.assertAlmostEqual(scale["aggregate_mean_gradient_frobenius_norm"], gn)
                self.assertAlmostEqual(scale["aggregate_preconditioned_direction_norm"], dn)
                self.assertAlmostEqual(scale["aggregate_source_weight_norm"], wn)
                self.assertAlmostEqual(scale["target_delta_norm"], .00125 * wn)
                self.assertAlmostEqual(scale["alpha"], .00125 * wn / dn)
                delta_squares = []
                for name, module in eligible:
                    record = next(r for r in records if r["name"] == name)
                    g, f = reference[name]
                    self.assertEqual(record["mean_gradient_frobenius_norm"], g.norm().item())
                    self.assertEqual(record["fisher_scalar_mean"], f.mean().item())
                    self.assertEqual(record["fisher_scalar_minimum"], f.min().item())
                    self.assertEqual(record["fisher_scalar_maximum"], f.max().item())
                    self.assertEqual(record["preconditioned_direction_frobenius_norm"], directions[name].norm().item())
                    expected = (originals[name] - scale["alpha"] * directions[name]).float()
                    self.assertTrue(torch.equal(module.weight.cpu(), expected))
                    delta_squares.append((expected.double() - originals[name]).square().sum().item())
                delta_norm = math.sqrt(sum(delta_squares))
                self.assertAlmostEqual(realized["aggregate_realized_fp32_delta_norm"], delta_norm)
                self.assertAlmostEqual(realized["aggregate_realized_relative_perturbation"], delta_norm / wn)
                manifest = rollback.build_manifest(
                    rollback.FROZEN_SPEC,
                    dict(attempt_spec="a"*64, corpus_spec="b"*64, corpus_manifest="c"*64, constructor_script="d"*64),
                    [], [], dict(serialized_sha256="e"*64, raw_tensor_sha256="f"*64),
                    1., scale, records, realized)
                for key, value in {**scale, **realized}.items():
                    self.assertEqual(manifest["gradient_and_rollback"][key], value)
                self.assertEqual(manifest["modified_matrices"], records)
                json.dumps(manifest, allow_nan=False)

    def test_opposite_sequence_gradients_cancel_but_fisher_remains(self):
        model = ToyCausalModel()
        model.proj = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.proj.weight.zero_()
        spec = dict(sample_count=2, sequence_length=2, predictions_per_sample=1,
                    total_prediction_tokens=2, batch_size=1, number_of_batches=2)
        _, moments = rollback.accumulate_empirical_fisher(
            model, torch.tensor([[1, 0], [1, 1]]), [("proj", model.proj)], spec, torch)
        first, fisher = moments["proj"]
        self.assertTrue(torch.equal(first, torch.zeros_like(first)))
        self.assertTrue(torch.equal(fisher, torch.full_like(fisher, .25)))
        self.assertEqual(rollback.global_fisher_lambda(moments, torch), .25)
        with self.assertRaisesRegex(ValueError, "zero or non-finite"):
            rollback.global_rollback_scale([("proj", model.proj)], moments, .00125, torch)

    def test_lambda_is_coordinate_weighted_and_sqrt_encloses_damping(self):
        moments = {"small": (torch.ones(1, dtype=torch.float64), torch.tensor([2.], dtype=torch.float64)),
                   "large": (torch.tensor([2., -4., 8.], dtype=torch.float64),
                             torch.tensor([4., 8., 10.], dtype=torch.float64))}
        damping = rollback.global_fisher_lambda(moments, torch)
        self.assertEqual(damping, 6.)
        self.assertNotEqual(damping, (2 + 22/3) / 2)
        first, fisher = moments["large"]
        actual = rollback.preconditioned_direction(first, fisher, damping, torch)
        self.assertTrue(torch.equal(actual, first / (fisher + 6).sqrt()))
        self.assertFalse(torch.allclose(actual, first / (fisher + 6)))
        self.assertFalse(torch.allclose(actual, first / (fisher.sqrt() + 6)))

    def test_sqrt_preconditioner_has_no_extra_epsilon_or_clipping(self):
        first = torch.tensor([[1e-10, -2e-10, 0.]], dtype=torch.float64)
        fisher = torch.tensor([[0., 1e-20, 4e-20]], dtype=torch.float64)
        damping = 2e-20
        actual = rollback.preconditioned_direction(first, fisher, damping, torch)
        expected = first / torch.sqrt(fisher + damping)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float64)
        self.assertEqual(actual.device.type, "cpu")
        self.assertLess(actual[0, 1].item(), -1.)
        self.assertFalse(torch.allclose(actual, first / (torch.sqrt(fisher) + damping)))
        self.assertFalse(torch.allclose(actual, first / torch.sqrt(fisher + damping + 1e-8)))

    def test_one_global_alpha_minus_direction_and_fp32_realized_metrics(self):
        eligible = two_matrix_case()
        moments = moment_case(eligible)
        originals = {name: module.weight.detach().double().clone() for name, module in eligible}
        damping = rollback.global_fisher_lambda(moments, torch)
        directions = {name: first / (fisher + damping).sqrt() for name, (first, fisher) in moments.items()}
        scale = rollback.global_rollback_scale(eligible, moments, .00125, torch)
        wn = math.sqrt(sum(float(w.square().sum()) for w in originals.values()))
        dn = math.sqrt(sum(float(d.square().sum()) for d in directions.values()))
        self.assertAlmostEqual(scale["alpha"], .00125 * wn / dn)
        self.assertAlmostEqual(scale["alpha"] * dn / wn, .00125)
        records, realized = rollback.apply_global_rollback(
            eligible, moments, damping, scale["alpha"], wn, torch)
        self.assertEqual([r["name"] for r in records], ["a_matrix", "z_matrix"])
        squares = []
        for name, module in eligible:
            expected = (originals[name] - scale["alpha"] * directions[name]).float()
            self.assertTrue(torch.equal(module.weight, expected))
            delta = module.weight.detach().double() - originals[name]
            self.assertTrue(torch.all(delta * directions[name] < 0))
            norm = float(delta.norm())
            record = next(r for r in records if r["name"] == name)
            self.assertAlmostEqual(record["realized_fp32_delta_frobenius_norm"], norm)
            squares.append(norm ** 2)
        self.assertAlmostEqual(realized["aggregate_realized_fp32_delta_norm"], math.sqrt(sum(squares)))
        self.assertAlmostEqual(realized["aggregate_realized_relative_perturbation"], math.sqrt(sum(squares))/wn)

    def test_invalid_fisher_gradient_lambda_and_direction_fail(self):
        good = torch.ones((2, 2), dtype=torch.float64)
        for value in (float("nan"), float("inf"), -1.):
            with self.subTest(fisher=value), self.assertRaises(ValueError):
                rollback.global_fisher_lambda({"a": (good, torch.full_like(good, value))}, torch)
        for moments in ({}, {"a": (good, torch.zeros_like(good))}):
            with self.assertRaises(ValueError):
                rollback.global_fisher_lambda(moments, torch)
        for value in (float("nan"), float("inf"), 0., -1.):
            with self.subTest(damping=value), self.assertRaises(ValueError):
                rollback.preconditioned_direction(good, good, value, torch)
        for value in (None, good.float(), torch.full_like(good, float("nan")),
                      torch.full_like(good, float("inf"))):
            with self.subTest(gradient=value), self.assertRaises(ValueError):
                rollback.preconditioned_direction(value, good, 1., torch)
        with self.assertRaisesRegex(ValueError, "direction"):
            rollback.preconditioned_direction(torch.full_like(good, 1e308), torch.zeros_like(good), 1e-300, torch)
        with self.assertRaisesRegex(ValueError, "denominator"):
            rollback.preconditioned_direction(good, torch.full_like(good, 1e308), 1e308, torch)
        with self.assertRaises(ValueError):
            rollback.preconditioned_direction(good, torch.ones(3, dtype=torch.float64), 1., torch)

    def test_nonfinite_gradients_and_overflow_fail_at_accumulation_boundary(self):
        spec = dict(sample_count=3, sequence_length=2, predictions_per_sample=1,
                    total_prediction_tokens=3, batch_size=1, number_of_batches=3)
        # Finite gradients can also overflow the FP32 Fisher accumulator.
        for value in (float("nan"), float("inf"), -float("inf"), 1e20):
            with self.subTest(value=value):
                model = ToyCausalModel()
                model.proj.weight.register_hook(lambda grad, v=value: torch.full_like(grad, v))
                with self.assertRaisesRegex(ValueError, "Non-finite accumulated first/Fisher"):
                    rollback.accumulate_empirical_fisher(
                        model, torch.tensor([[1, 0]] * 3), [("proj", model.proj)], spec, torch)
                self.assertEqual(len(model.forward_cache_values), 3)
                self.assertIsNone(model.proj.weight.grad)

    def test_missing_gradient_and_invalid_dimensions_fail_immediately(self):
        spec = dict(sample_count=1, sequence_length=2, predictions_per_sample=1,
                    total_prediction_tokens=1, batch_size=1, number_of_batches=1)
        model = ToyCausalModel()
        model.unused = torch.nn.Linear(1, 1, bias=False)
        with self.assertRaisesRegex(ValueError, "eligible gradient"):
            rollback.accumulate_empirical_fisher(
                model, torch.tensor([[1, 0]]), [("unused", model.unused), ("proj", model.proj)], spec, torch)
        spec["batch_size"] = 2
        with self.assertRaisesRegex(ValueError, "dimensions"):
            rollback.accumulate_empirical_fisher(
                model, torch.tensor([[1, 0]]), [("proj", model.proj)], spec, torch)

    def test_matrix_finiteness_scans_only_after_all_sequences_and_progress(self):
        model = ToyCausalModel()
        sample_count = 257
        spec = dict(sample_count=sample_count, sequence_length=2, predictions_per_sample=1,
                    total_prediction_tokens=sample_count, batch_size=1, number_of_batches=sample_count)
        matrix_checks = []
        isfinite = torch.isfinite
        def check_finite(value):
            if value.ndim == 2:
                matrix_checks.append(len(model.forward_cache_values))
                self.assertEqual(len(model.forward_cache_values), sample_count)
            return isfinite(value)
        with patch.object(torch, "isfinite", side_effect=check_finite), patch("sys.stdout", new_callable=io.StringIO) as output:
            rollback.accumulate_empirical_fisher(
                model, torch.tensor([[1, 0]] * sample_count), [("proj", model.proj)], spec, torch)
        self.assertEqual(matrix_checks, [sample_count, sample_count])
        self.assertEqual(output.getvalue(), "Processed 128/257 sequences\nProcessed 256/257 sequences\n")

    def test_loaded_checkpoint_validation_and_exact_spec(self):
        model = MockQwen()
        model.config.model_type = "wrong"
        with self.assertRaises(ValueError):
            rollback.validate_loaded_model(model, torch)
        model.config.model_type = "qwen3"
        model.double()
        with self.assertRaises(ValueError):
            rollback.validate_loaded_model(model, torch)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            spec = copy.deepcopy(rollback.FROZEN_SPEC)
            spec["empirical_fisher"]["direction"] = "wrong"
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "frozen definition"):
                rollback.load_spec(path)

    def test_model_loader_firewall_only_loads_merged_locally_in_fp32(self):
        import types
        fake = types.ModuleType("transformers")
        model = MockQwen()
        calls = []
        def load_model(path, **kwargs):
            calls.append(("model", path, kwargs))
            return model
        def load_tokenizer(path, **kwargs):
            calls.append(("tokenizer", path, kwargs))
            return object()
        fake.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_model)
        fake.AutoTokenizer = SimpleNamespace(from_pretrained=load_tokenizer)
        with patch.dict(sys.modules, {"transformers": fake}), patch.dict(rollback.os.environ):
            loaded, _ = rollback.load_local_model(Path("synthetic-merged"), "cpu", torch)
        self.assertIs(loaded, model)
        self.assertFalse(model.training)
        self.assertEqual(calls, [
            ("model", Path("synthetic-merged"), {"dtype": torch.float32, "local_files_only": True}),
            ("tokenizer", Path("synthetic-merged"), {"local_files_only": True}),
        ])
        self.assertEqual(rollback.DEFAULT_CORPUS_SPEC_PATH, CORPUS_SPEC_PATH)
        self.assertNotEqual(rollback.FROZEN_SPEC["attempt_id"], rollback.FROZEN_CORPUS_SPEC["attempt_id"])

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

    def test_deterministic_manifest_ordering_and_no_extra_metadata(self):
        eligible = two_matrix_case()
        moments = moment_case(eligible)
        scale = rollback.global_rollback_scale(eligible, moments, 0.00125, torch)
        matrices, realized = rollback.apply_global_rollback(
            eligible, moments, scale["lambda"], scale["alpha"], scale["aggregate_source_weight_norm"], torch
        )
        file_record = {"path": "model.safetensors", "size_bytes": 1, "sha256": "a" * 64}
        manifest = rollback.build_manifest(
            rollback.FROZEN_SPEC,
            {"attempt_spec": "a" * 64, "corpus_spec": "b" * 64,
             "corpus_manifest": "c" * 64, "constructor_script": "d" * 64},
            [file_record], [file_record],
            {"serialized_sha256": "e" * 64, "raw_tensor_sha256": "f" * 64},
            1.5, scale, matrices, realized,
        )
        self.assertEqual(manifest["deterministic_ordering"]["modified_module_names"],
                         ["a_matrix", "z_matrix"])
        self.assertEqual(manifest["loss"]["total_prediction_tokens"], 4096 * 127)
        self.assertEqual(manifest["gradient_and_rollback"]["eligible_matrix_count"], 2)
        self.assertEqual(manifest["empirical_fisher"]["direction"], "d = g / sqrt(F + lambda)")
        self.assertEqual(json.dumps(manifest, allow_nan=False), json.dumps(copy.deepcopy(manifest), allow_nan=False))
        self.assertFalse({"timestamp", "host", "gpu"} & set(manifest))

    def test_static_information_boundary_and_no_optimizer(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for forbidden in (
            "oracle", "self_diff", "self-diff", "cosine", "semantic", "evaluation", "models/base", "adapter", "models.lock.json",
            "downloaded-model-hashes.json", "merged-model-hashes.json",
            "evaluation.json", "bf16", "quantization",
            "attempts/001_", "attempts/002_", "attempts/003_", "attempts/004_",
            "attempts/006_", "attempt006",
        ):
            self.assertNotIn(forbidden, source.lower())
        for token in ("torch.optim", ".step(", "clip_grad", "weight_decay="):
            self.assertNotIn(token, source)
        self.assertIn("local_files_only=True", source)
        self.assertIn("use_cache=False", source)
        self.assertFalse(rollback.FROZEN_SPEC["generic_loss"]["mixed_precision"])
        self.assertFalse(rollback.FROZEN_SPEC["generic_loss"]["optimizer"])
        self.assertFalse(rollback.FROZEN_SPEC["generic_loss"]["weight_decay"])
        self.assertFalse(rollback.FROZEN_SPEC["generic_loss"]["gradient_clipping"])


if __name__ == "__main__":
    unittest.main()
