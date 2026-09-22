import ast
import copy
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "construct_gaussian_noise.py"
SPEC_PATH = (
    PROJECT
    / "experiments"
    / "attempts"
    / "003_gaussian_noise_prefix0_13_r00125"
    / "spec.json"
)

MODULE_SPEC = importlib.util.spec_from_file_location(
    "construct_gaussian_noise", SCRIPT_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
noise = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = noise
MODULE_SPEC.loader.exec_module(noise)


class SevenLinearBlock(torch.nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(4)
        self.self_attn = torch.nn.ModuleDict(
            {
                "q_proj": torch.nn.Linear(4, 4, bias=True),
                "k_proj": torch.nn.Linear(4, 2, bias=True),
                "v_proj": torch.nn.Linear(4, 2, bias=True),
                "o_proj": torch.nn.Linear(4, 4, bias=True),
            }
        )
        self.mlp = torch.nn.ModuleDict(
            {
                "gate_proj": torch.nn.Linear(4, 6, bias=True),
                "up_proj": torch.nn.Linear(4, 6, bias=True),
                "down_proj": torch.nn.Linear(6, 4, bias=True),
            }
        )
        with torch.no_grad():
            for index, module in enumerate(self.modules()):
                if isinstance(module, torch.nn.Linear):
                    values = torch.arange(
                        module.weight.numel(), dtype=torch.float32
                    ).reshape_as(module.weight)
                    module.weight.copy_(values / 100.0 + offset + index / 1000.0)


class MockCausalModel(torch.nn.Module):
    def __init__(self, layer_count: int = 28) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(8, 4)
        self.model.layers = torch.nn.ModuleList(
            [SevenLinearBlock(float(index + 1)) for index in range(layer_count)]
        )
        self.model.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 8, bias=True)
        self.config = SimpleNamespace(
            model_type="qwen3", num_hidden_layers=layer_count
        )


def file_record(path: str, digest_character: str) -> dict[str, object]:
    return {"path": path, "size_bytes": 1, "sha256": digest_character * 64}


def matrix_record(name: str, scale: float) -> dict[str, object]:
    return {
        "name": name,
        "shape": [2, 2],
        "original_frobenius_norm": 2.0 * scale,
        "realized_fp32_delta_frobenius_norm": scale,
        "realized_relative_frobenius_perturbation": 0.5,
        "max_absolute_weight_change": scale / 2.0,
        "changed_element_count": 4,
        "changed_element_fraction": 1.0,
    }


class GaussianNoiseConstructorTests(unittest.TestCase):
    def test_committed_spec_is_exactly_frozen(self) -> None:
        loaded = noise.load_spec(SPEC_PATH)
        self.assertEqual(loaded, noise.FROZEN_SPEC)
        self.assertEqual(loaded["attempt_id"], "003_gaussian_noise_prefix0_13_r00125")
        self.assertEqual(loaded["information_policy"], "merged_checkpoint_only")
        self.assertEqual(loaded["transformer_block_indices"], list(range(14)))
        self.assertEqual(loaded["expected_matrix_count"], 98)
        self.assertEqual(loaded["seed"], 1729)
        self.assertEqual(loaded["target_relative_frobenius"], 0.00125)
        self.assertEqual(loaded["outputs"]["directory_name"], "model")

    def test_recursive_discovery_finds_exact_blocks_zero_through_thirteen(self) -> None:
        model = MockCausalModel()
        found = noise.discover_eligible_linear_weights(model, list(range(14)), torch)
        names = [name for name, _module in found]
        self.assertEqual(len(names), 98)
        self.assertEqual(names, sorted(names))
        for block_index in range(14):
            prefix = f"model.layers.{block_index}."
            self.assertEqual(sum(name.startswith(prefix) for name in names), 7)
        self.assertIn("model.layers.0.self_attn.q_proj", names)
        self.assertIn("model.layers.13.mlp.down_proj", names)
        self.assertFalse(any(name.startswith("model.layers.14.") for name in names))
        self.assertFalse(any(name.startswith("model.layers.27.") for name in names))

    def test_norm_embedding_head_bias_and_later_blocks_are_excluded(self) -> None:
        model = MockCausalModel()
        eligible = noise.discover_eligible_linear_weights(
            model, list(range(14)), torch
        )
        selected = {f"{name}.weight" for name, _module in eligible}
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        noise.apply_gaussian_noise(eligible, 1729, 0.00125, torch)
        for name, parameter in model.named_parameters():
            if name in selected:
                self.assertFalse(torch.equal(parameter, before[name]), name)
            else:
                self.assertTrue(torch.equal(parameter, before[name]), name)
        self.assertTrue(torch.equal(model.model.embed_tokens.weight, before["model.embed_tokens.weight"]))
        self.assertTrue(torch.equal(model.model.norm.weight, before["model.norm.weight"]))
        self.assertTrue(torch.equal(model.lm_head.weight, before["lm_head.weight"]))
        self.assertTrue(torch.equal(model.lm_head.bias, before["lm_head.bias"]))

    def test_source_weight_must_be_float32(self) -> None:
        for dtype in (torch.float64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                generator = noise.make_cpu_generator(1729, torch)
                with self.assertRaisesRegex(ValueError, "must be torch.float32"):
                    noise.gaussian_noise_weight(
                        torch.ones((4, 4), dtype=dtype),
                        0.00125,
                        generator,
                        torch,
                    )

    def test_fixed_seed_gaussian_stream_is_deterministic(self) -> None:
        first_weight = torch.linspace(-2.0, 3.0, 35, dtype=torch.float32).reshape(5, 7)
        second_weight = torch.linspace(1.0, 4.0, 24, dtype=torch.float32).reshape(6, 4)

        generator_one = noise.make_cpu_generator(1729, torch)
        output_one_a = noise.gaussian_noise_weight(
            first_weight, 0.00125, generator_one, torch
        )
        output_one_b = noise.gaussian_noise_weight(
            second_weight, 0.00125, generator_one, torch
        )
        generator_two = noise.make_cpu_generator(1729, torch)
        output_two_a = noise.gaussian_noise_weight(
            first_weight, 0.00125, generator_two, torch
        )
        output_two_b = noise.gaussian_noise_weight(
            second_weight, 0.00125, generator_two, torch
        )

        self.assertTrue(torch.equal(output_one_a, output_two_a))
        self.assertTrue(torch.equal(output_one_b, output_two_b))
        self.assertEqual(output_one_a.dtype, torch.float32)
        self.assertEqual(output_one_a.device.type, "cpu")

    def test_same_seed_and_input_match_while_different_seed_differs(self) -> None:
        original = torch.linspace(-4.0, 5.0, 128, dtype=torch.float32).reshape(16, 8)
        outputs = []
        for seed in (1729, 1729, 1730):
            module = torch.nn.Linear(8, 16, bias=False)
            with torch.no_grad():
                module.weight.copy_(original)
            noise.apply_gaussian_noise(
                [("model.layers.0.projection", module)], seed, 0.00125, torch
            )
            outputs.append(module.weight.detach().clone())
        self.assertTrue(torch.equal(outputs[0], outputs[1]))
        self.assertFalse(torch.equal(outputs[0], outputs[2]))

    def test_input_order_does_not_change_sorted_stream_assignment(self) -> None:
        def modules():
            first = torch.nn.Linear(4, 3, bias=False)
            second = torch.nn.Linear(3, 2, bias=False)
            with torch.no_grad():
                first.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) + 1)
                second.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3) + 2)
            return first, second

        first_a, second_a = modules()
        first_b, second_b = modules()
        records_a = noise.apply_gaussian_noise(
            [("z", second_a), ("a", first_a)], 1729, 0.00125, torch
        )
        records_b = noise.apply_gaussian_noise(
            [("a", first_b), ("z", second_b)], 1729, 0.00125, torch
        )
        self.assertEqual([record["name"] for record in records_a], ["a", "z"])
        self.assertEqual(records_a, records_b)
        self.assertTrue(torch.equal(first_a.weight, first_b.weight))
        self.assertTrue(torch.equal(second_a.weight, second_b.weight))

    def test_target_relative_norm_after_fp32_cast(self) -> None:
        original = torch.linspace(-7.0, 9.0, 4096, dtype=torch.float32).reshape(64, 64)
        modified = noise.gaussian_noise_weight(
            original,
            0.00125,
            noise.make_cpu_generator(1729, torch),
            torch,
        )
        metrics = noise.realized_perturbation_metrics(original, modified, torch)
        self.assertAlmostEqual(
            metrics["realized_relative_frobenius_perturbation"],
            0.00125,
            delta=2e-9,
        )

    def test_realized_metrics_use_actual_copied_float32_values(self) -> None:
        module = torch.nn.Linear(8, 16, bias=False)
        with torch.no_grad():
            module.weight.copy_(
                torch.linspace(-3.0, 2.0, 128, dtype=torch.float32).reshape(16, 8)
            )
        original = module.weight.detach().clone()
        records = noise.apply_gaussian_noise(
            [("model.layers.0.projection", module)], 1729, 0.00125, torch
        )
        expected = noise.realized_perturbation_metrics(
            original, module.weight.detach(), torch
        )
        self.assertEqual(records[0]["shape"], [16, 8])
        for key, value in expected.items():
            self.assertEqual(records[0][key], value)
        self.assertEqual(module.weight.dtype, torch.float32)

    def test_overwrite_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            spec_path = root / "attempt" / "spec.json"
            output_root = root / "outputs"
            merged_dir.mkdir()
            spec_path.parent.mkdir()
            spec_path.write_text(json.dumps(noise.FROZEN_SPEC), encoding="utf-8")
            (output_root / "model").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already exists"):
                noise.validate_destinations(
                    merged_dir, spec_path, output_root, noise.FROZEN_SPEC
                )
            (output_root / "model").rmdir()
            manifest = spec_path.parent / "construction-manifest.json"
            manifest.write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                noise.validate_destinations(
                    merged_dir, spec_path, output_root, noise.FROZEN_SPEC
                )
            self.assertEqual(manifest.read_text(encoding="utf-8"), "existing\n")

    def test_manifest_ordering_and_aggregates_are_deterministic(self) -> None:
        test_spec = copy.deepcopy(noise.FROZEN_SPEC)
        test_spec["expected_matrix_count"] = 2
        source = [file_record("z", "d"), file_record("a", "c")]
        output = [file_record("z", "f"), file_record("a", "e")]
        matrices = [matrix_record("z", 2.0), matrix_record("a", 1.0)]
        manifest_one = noise.build_construction_manifest(
            test_spec, "a" * 64, "b" * 64, source, output, matrices
        )
        manifest_two = noise.build_construction_manifest(
            test_spec,
            "a" * 64,
            "b" * 64,
            list(reversed(source)),
            list(reversed(output)),
            list(reversed(matrices)),
        )
        self.assertEqual(manifest_one, manifest_two)
        self.assertEqual(
            json.dumps(manifest_one, allow_nan=False),
            json.dumps(manifest_two, allow_nan=False),
        )
        self.assertEqual(
            [record["name"] for record in manifest_one["modified_matrices"]],
            ["a", "z"],
        )
        self.assertEqual(
            manifest_one["deterministic_ordering"]["modified_module_names"],
            ["a", "z"],
        )
        self.assertEqual(manifest_one["construction"]["seed"], 1729)
        self.assertEqual(
            manifest_one["construction"]["target_relative_frobenius"], 0.00125
        )
        aggregate = manifest_one["aggregate_perturbation"]
        self.assertEqual(aggregate["aggregate_original_frobenius_norm"], math.sqrt(20.0))
        self.assertEqual(
            aggregate["aggregate_realized_fp32_delta_frobenius_norm"], math.sqrt(5.0)
        )
        self.assertEqual(aggregate["changed_element_count"], 8)
        self.assertEqual(aggregate["changed_element_fraction"], 1.0)

    def test_source_checkpoint_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            spec_path = root / "attempt" / "spec.json"
            output_root = root / "outputs"
            merged_dir.mkdir()
            spec_path.parent.mkdir()
            spec_path.write_text(
                json.dumps(noise.FROZEN_SPEC, indent=2) + "\n", encoding="utf-8"
            )
            before = [file_record("model", "a")]
            after = [file_record("model", "b")]
            checkpoint_calls = iter((before, after))
            eligible = [
                (f"model.layers.{index // 7}.projection_{index}", object())
                for index in range(98)
            ]
            records = [
                matrix_record(f"model.layers.{index // 7}.projection_{index}", 1.0)
                for index in range(98)
            ]
            with (
                mock.patch.object(
                    noise,
                    "checkpoint_file_records",
                    side_effect=lambda _path: next(checkpoint_calls),
                ),
                mock.patch.object(
                    noise, "load_local_checkpoint", return_value=(object(), object())
                ),
                mock.patch.object(
                    noise, "discover_eligible_linear_weights", return_value=eligible
                ),
                mock.patch.object(noise, "apply_gaussian_noise", return_value=records),
                mock.patch.object(noise, "verify_float32_model"),
                mock.patch.object(
                    noise,
                    "save_checkpoint",
                    return_value=[file_record("model", "c")],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "changed during construction"):
                    noise.construct(merged_dir, spec_path, output_root)

    def test_static_information_boundary_and_interface(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        lowered = source.lower()
        forbidden = (
            "base_model",
            "base model",
            "/models/base",
            "lora",
            "adapter",
            "oracle",
            "oracle-logit-lens.json",
            "models.lock.json",
            "downloaded-model-hashes.json",
            "merged-model-hashes.json",
            "attempt 001",
            "attempt 002",
            "attempt001",
            "attempt002",
            "evaluation.json",
            "bf16_residual",
            "bf16-residual",
            "torch.cuda",
        )
        for value in forbidden:
            self.assertNotIn(value, lowered)

        tree = ast.parse(source)
        environment_reads = set()
        cli_options = set()
        builtin_hash_calls = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "hash":
                    builtin_hash_calls += 1
                if isinstance(node.func, ast.Attribute):
                    if (
                        node.func.attr == "get"
                        and isinstance(node.func.value, ast.Attribute)
                        and isinstance(node.func.value.value, ast.Name)
                        and node.func.value.value.id == "os"
                        and node.func.value.attr == "environ"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                    ):
                        environment_reads.add(node.args[0].value)
                    if node.func.attr == "add_argument":
                        for argument in node.args:
                            if (
                                isinstance(argument, ast.Constant)
                                and isinstance(argument.value, str)
                                and argument.value.startswith("--")
                            ):
                                cli_options.add(argument.value)
        self.assertEqual(environment_reads, noise.ALLOWED_ENVIRONMENT_VARIABLES)
        self.assertEqual(
            cli_options, {"--merged-model-dir", "--spec-path", "--output-root"}
        )
        self.assertEqual(builtin_hash_calls, 0)


if __name__ == "__main__":
    unittest.main()
