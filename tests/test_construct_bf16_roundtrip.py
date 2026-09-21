import ast
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
SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "construct_bf16_roundtrip.py"
SPEC_PATH = (
    PROJECT
    / "experiments"
    / "attempts"
    / "002_bf16_roundtrip_prefix0_13"
    / "spec.json"
)

SPEC = importlib.util.spec_from_file_location("construct_bf16_roundtrip", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
roundtrip = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = roundtrip
SPEC.loader.exec_module(roundtrip)


class NestedBlock(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(2)
        self.branch = torch.nn.ModuleDict(
            {
                "projection": torch.nn.Linear(2, 2, bias=True),
                "activation": torch.nn.ReLU(),
            }
        )
        with torch.no_grad():
            self.branch["projection"].weight.copy_(
                torch.tensor(
                    [[value + 0.001, value + 0.002], [value + 0.003, value + 0.004]],
                    dtype=torch.float32,
                )
            )


class MockCausalModel(torch.nn.Module):
    def __init__(self, layer_count: int = 28) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(4, 2)
        self.model.layers = torch.nn.ModuleList(
            [NestedBlock(float(index + 1)) for index in range(layer_count)]
        )
        self.model.norm = torch.nn.LayerNorm(2)
        self.lm_head = torch.nn.Linear(2, 4, bias=True)
        self.config = SimpleNamespace(
            model_type="qwen3",
            num_hidden_layers=layer_count,
        )


def file_record(path: str, digest_character: str) -> dict[str, object]:
    return {"path": path, "size_bytes": 1, "sha256": digest_character * 64}


def matrix_record(name: str, scale: float) -> dict[str, object]:
    return {
        "changed_element_fraction": 0.5,
        "changed_element_count": 2,
        "max_absolute_weight_change": scale / 4.0,
        "realized_relative_frobenius_perturbation": scale / 2.0,
        "realized_fp32_delta_frobenius_norm": scale,
        "original_frobenius_norm": 2.0 * scale,
        "shape": [2, 2],
        "module_name": name,
    }


class BF16RoundtripTests(unittest.TestCase):
    def test_committed_spec_is_frozen(self) -> None:
        loaded = roundtrip.load_spec(SPEC_PATH)
        self.assertEqual(loaded, roundtrip.FROZEN_SPEC)
        self.assertEqual(loaded["transformer_block_indices"], list(range(14)))
        self.assertEqual(loaded["outputs"]["directory_name"], "model")

    def test_exact_bf16_roundtrip_values(self) -> None:
        original = torch.tensor(
            [[1.001, -3.1415927, 0.1], [-0.33333334, 123.456, 0.003]],
            dtype=torch.float32,
        )
        actual = roundtrip.bf16_roundtrip_weight(original, torch)
        expected = torch.tensor(
            [
                [1.0, -3.140625, 0.10009765625],
                [-0.333984375, 123.5, 0.0030059814453125],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)

    def test_source_weight_must_be_float32(self) -> None:
        for dtype in (torch.float64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "must be torch.float32"):
                    roundtrip.bf16_roundtrip_weight(
                        torch.ones((2, 2), dtype=dtype), torch
                    )

    def test_recursive_discovery_is_limited_to_blocks_zero_through_thirteen(self) -> None:
        model = MockCausalModel(layer_count=16)
        found = roundtrip.discover_eligible_linear_weights(
            model, list(range(14)), torch
        )
        names = [name for name, _module in found]

        self.assertEqual(len(names), 14)
        self.assertIn("model.layers.0.branch.projection", names)
        self.assertIn("model.layers.13.branch.projection", names)
        self.assertFalse(any(name.startswith("model.layers.14.") for name in names))
        self.assertFalse(any(name.startswith("model.layers.15.") for name in names))
        self.assertEqual(names, sorted(names))

    def test_excluded_tensors_remain_unchanged(self) -> None:
        model = MockCausalModel()
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(1.001)
            model.model.norm.weight.fill_(1.001)
            model.lm_head.weight.fill_(1.001)
            model.lm_head.bias.fill_(1.001)

        selected_names = {
            name
            for name, _module in roundtrip.discover_eligible_linear_weights(
                model, list(range(14)), torch
            )
        }
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        eligible = roundtrip.discover_eligible_linear_weights(
            model, list(range(14)), torch
        )
        records = roundtrip.apply_bf16_roundtrip(eligible, torch)

        self.assertEqual(
            {record["module_name"] for record in records}, selected_names
        )
        selected_parameter_names = {f"{name}.weight" for name in selected_names}
        for name, parameter in model.named_parameters():
            if name in selected_parameter_names:
                expected = before[name].to(torch.bfloat16).to(torch.float32)
                self.assertTrue(torch.equal(parameter, expected), name)
            else:
                self.assertTrue(torch.equal(parameter, before[name]), name)

    def test_realized_perturbation_accounting(self) -> None:
        original = torch.tensor(
            [[1.001, 1.5], [-3.1415927, 0.0]], dtype=torch.float32
        )
        modified = roundtrip.bf16_roundtrip_weight(original, torch)
        metrics = roundtrip.realized_perturbation_metrics(
            original, modified, torch
        )
        original64 = original.to(torch.float64)
        delta64 = modified.to(torch.float64) - original64

        self.assertEqual(
            metrics["original_frobenius_norm"],
            torch.linalg.vector_norm(original64).item(),
        )
        self.assertEqual(
            metrics["realized_fp32_delta_frobenius_norm"],
            torch.linalg.vector_norm(delta64).item(),
        )
        self.assertEqual(
            metrics["realized_relative_frobenius_perturbation"],
            torch.linalg.vector_norm(delta64).item()
            / torch.linalg.vector_norm(original64).item(),
        )
        self.assertEqual(
            metrics["max_absolute_weight_change"], delta64.abs().max().item()
        )
        self.assertEqual(metrics["changed_element_count"], 2)
        self.assertEqual(metrics["changed_element_fraction"], 0.5)

    def test_exactly_representable_values_are_identity(self) -> None:
        original = torch.tensor(
            [[0.0, 1.0, -1.5], [2.0, 16.0, 0.5]], dtype=torch.float32
        )
        modified = roundtrip.bf16_roundtrip_weight(original, torch)
        metrics = roundtrip.realized_perturbation_metrics(
            original, modified, torch
        )

        self.assertTrue(torch.equal(modified, original))
        self.assertEqual(metrics["realized_fp32_delta_frobenius_norm"], 0.0)
        self.assertEqual(
            metrics["realized_relative_frobenius_perturbation"], 0.0
        )
        self.assertEqual(metrics["max_absolute_weight_change"], 0.0)
        self.assertEqual(metrics["changed_element_count"], 0)
        self.assertEqual(metrics["changed_element_fraction"], 0.0)

    def test_overwrite_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            spec_path = root / "attempt" / "spec.json"
            output_root = root / "outputs"
            merged_dir.mkdir()
            spec_path.parent.mkdir()
            spec_path.write_text(json.dumps(roundtrip.FROZEN_SPEC), encoding="utf-8")

            (output_root / "model").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already exists"):
                roundtrip.validate_destinations(
                    merged_dir,
                    spec_path,
                    output_root,
                    roundtrip.FROZEN_SPEC,
                )

            (output_root / "model").rmdir()
            manifest_path = spec_path.parent / "construction-manifest.json"
            manifest_path.write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                roundtrip.validate_destinations(
                    merged_dir,
                    spec_path,
                    output_root,
                    roundtrip.FROZEN_SPEC,
                )
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), "existing\n")

    def test_manifest_ordering_and_aggregate_are_deterministic(self) -> None:
        source_one = [file_record("z", "d"), file_record("a", "c")]
        output_one = [file_record("z", "f"), file_record("a", "e")]
        matrices_one = [matrix_record("z", 2.0), matrix_record("a", 1.0)]
        manifest_one = roundtrip.build_construction_manifest(
            roundtrip.FROZEN_SPEC,
            "a" * 64,
            "b" * 64,
            source_one,
            output_one,
            matrices_one,
        )
        manifest_two = roundtrip.build_construction_manifest(
            roundtrip.FROZEN_SPEC,
            "a" * 64,
            "b" * 64,
            list(reversed(source_one)),
            list(reversed(output_one)),
            list(reversed(matrices_one)),
        )

        self.assertEqual(manifest_one, manifest_two)
        self.assertEqual(
            json.dumps(manifest_one, ensure_ascii=False, allow_nan=False, indent=2),
            json.dumps(manifest_two, ensure_ascii=False, allow_nan=False, indent=2),
        )
        self.assertEqual(
            [record["module_name"] for record in manifest_one["modified_matrices"]],
            ["a", "z"],
        )
        aggregate = manifest_one["aggregate_perturbation"]
        self.assertEqual(
            aggregate["combined_original_frobenius_norm"], math.sqrt(20.0)
        )
        self.assertEqual(
            aggregate["combined_realized_fp32_delta_frobenius_norm"],
            math.sqrt(5.0),
        )
        self.assertEqual(aggregate["changed_element_count"], 4)
        self.assertEqual(aggregate["changed_element_fraction"], 0.5)

    def test_source_checkpoint_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            spec_path = root / "attempt" / "spec.json"
            output_root = root / "outputs"
            merged_dir.mkdir()
            spec_path.parent.mkdir()
            spec_path.write_text(
                json.dumps(roundtrip.FROZEN_SPEC, indent=2) + "\n",
                encoding="utf-8",
            )
            before = [file_record("model", "a")]
            after = [file_record("model", "b")]
            calls = iter((before, after))
            fake_model = object()
            fake_tokenizer = object()
            record = matrix_record("model.layers.0.projection", 1.0)

            with (
                mock.patch.object(
                    roundtrip,
                    "checkpoint_file_records",
                    side_effect=lambda _path: next(calls),
                ),
                mock.patch.object(
                    roundtrip,
                    "load_local_checkpoint",
                    return_value=(fake_model, fake_tokenizer),
                ),
                mock.patch.object(
                    roundtrip,
                    "discover_eligible_linear_weights",
                    return_value=[("model.layers.0.projection", object())],
                ),
                mock.patch.object(
                    roundtrip, "apply_bf16_roundtrip", return_value=[record]
                ),
                mock.patch.object(roundtrip, "verify_float32_model"),
                mock.patch.object(
                    roundtrip,
                    "save_checkpoint",
                    return_value=[file_record("model", "c")],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "changed during construction"):
                    roundtrip.construct(merged_dir, spec_path, output_root)

    def test_static_information_boundary_and_interface(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        lowered_source = source.lower()
        forbidden = (
            "base_model",
            "base model",
            "/models/base",
            "lora",
            "adapter",
            "oracle",
            "models.lock.json",
            "downloaded-model-hashes.json",
            "merged-model-hashes.json",
        )
        for value in forbidden:
            self.assertNotIn(value, lowered_source)

        tree = ast.parse(source)
        environment_reads = set()
        cli_options = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
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

        self.assertEqual(
            environment_reads, roundtrip.ALLOWED_ENVIRONMENT_VARIABLES
        )
        self.assertEqual(
            cli_options,
            {"--merged-model-dir", "--spec-path", "--output-root"},
        )


if __name__ == "__main__":
    unittest.main()
