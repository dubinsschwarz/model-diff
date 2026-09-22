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
SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "construct_quantized_roundtrip.py"
ATTEMPT_DIRECTORY = (
    PROJECT / "experiments" / "attempts" / "004_quantize11_prefix0_13"
)
SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
CALIBRATION_PATH = ATTEMPT_DIRECTORY / "calibration.json"
CALIBRATION_SHA256 = "e591e067ad45210bdea50f5288d99e41ae6c562ce4bc3d84ce7f320044c7bdfc"

MODULE_SPEC = importlib.util.spec_from_file_location(
    "construct_quantized_roundtrip", SCRIPT_PATH
)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
quantized = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = quantized
MODULE_SPEC.loader.exec_module(quantized)


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
                    module.weight.copy_(
                        values / 137.0 + offset / 31.0 + index / 997.0
                    )


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
        "changed_scalar_count": 4,
        "changed_scalar_fraction": 1.0,
        "max_absolute_weight_change": scale / 2.0,
    }


def load_calibration_copy() -> dict[str, object]:
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


class QuantizedRoundtripConstructorTests(unittest.TestCase):
    def test_committed_spec_and_calibration_are_frozen(self) -> None:
        spec = quantized.load_spec(SPEC_PATH)
        calibration, metadata = quantized.load_calibration(CALIBRATION_PATH, spec)
        self.assertEqual(spec, quantized.FROZEN_SPEC)
        self.assertEqual(spec["attempt_id"], "004_quantize11_prefix0_13")
        self.assertEqual(spec["information_policy"], "merged_checkpoint_only")
        self.assertEqual(spec["selected_bits"], 11)
        self.assertEqual(spec["selection_target_relative_frobenius"], 0.00125)
        self.assertEqual(spec["transformer_block_indices"], list(range(14)))
        self.assertEqual(spec["expected_matrix_count"], 98)
        self.assertEqual(spec["quantization"]["qmax"], 1023)
        self.assertEqual(metadata["selected_bits"], 11)
        self.assertEqual(metadata["qmax"], 1023)
        self.assertEqual(calibration["selection"]["selected_bit_width"], 11)
        self.assertEqual(quantized.sha256_file(CALIBRATION_PATH), CALIBRATION_SHA256)

    def test_calibration_selection_and_higher_bit_tie_break(self) -> None:
        candidates = [
            {"bits": 8, "aggregate_relative_frobenius_perturbation": 1.0},
            {"bits": 9, "aggregate_relative_frobenius_perturbation": 2.0},
            {"bits": 10, "aggregate_relative_frobenius_perturbation": 4.0},
        ]
        self.assertEqual(
            quantized.select_bit_width(candidates, 1.5, [8, 9, 10]),
            9,
        )
        candidates[2]["aggregate_relative_frobenius_perturbation"] = 1.0
        self.assertEqual(
            quantized.select_bit_width(candidates, 1.5, [8, 9, 10]),
            10,
        )

    def test_calibration_wrong_or_recomputed_winner_fails_closed(self) -> None:
        spec = quantized.FROZEN_SPEC
        wrong_stored = load_calibration_copy()
        wrong_stored["selection"]["selected_bit_width"] = 12
        with self.assertRaisesRegex(ValueError, "stored winner"):
            quantized.validate_calibration_document(wrong_stored, spec)

        tampered = load_calibration_copy()
        result_12 = next(
            result for result in tampered["candidate_results"] if result["bits"] == 12
        )
        target = spec["selection_target_relative_frobenius"]
        result_12["aggregate_relative_frobenius_perturbation"] = target
        result_12["absolute_target_error"] = 0.0
        result_12["aggregate_realized_delta_frobenius_norm"] = (
            target * result_12["aggregate_original_frobenius_norm"]
        )
        with self.assertRaisesRegex(ValueError, "recomputed winner"):
            quantized.validate_calibration_document(tampered, spec)

    def test_exact_11_bit_qmax_and_per_row_formula(self) -> None:
        self.assertEqual((1 << (11 - 1)) - 1, 1023)
        weight = torch.tensor(
            [
                [-1.0, -0.5004, -0.1, 0.0, 0.4996, 1.0],
                [-3.25, -1.125, 0.25, 1.75, 2.5, 3.0],
            ],
            dtype=torch.float32,
        )
        actual = quantized.quantize_dequantize_weight(weight, 11, 1023, torch)

        weight64 = weight.to(torch.float64)
        row_absmax = weight64.abs().amax(dim=1)
        scale = row_absmax / 1023
        expected64 = torch.round(weight64 / scale.unsqueeze(1))
        expected64 = torch.clamp(expected64, -1023, 1023)
        expected64 = expected64 * scale.unsqueeze(1)
        expected = expected64.to(torch.float32).contiguous()
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(actual.is_contiguous())
        with self.assertRaisesRegex(ValueError, "qmax disagree"):
            quantized.quantize_dequantize_weight(weight, 11, 1024, torch)

    def test_zero_rows_remain_exact_zero(self) -> None:
        weight = torch.tensor(
            [[0.0, -0.0, 0.0], [1.0, -0.25, 0.125]], dtype=torch.float32
        )
        result = quantized.quantize_dequantize_weight(weight, 11, 1023, torch)
        self.assertTrue(torch.equal(result[0], torch.zeros(3, dtype=torch.float32)))
        self.assertTrue(torch.isfinite(result).all())

    def test_source_weight_must_be_float32(self) -> None:
        for dtype in (torch.float64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "must be torch.float32"):
                    quantized.quantize_dequantize_weight(
                        torch.ones((4, 4), dtype=dtype), 11, 1023, torch
                    )

    def test_quantization_is_deterministic(self) -> None:
        weight = torch.linspace(-7.0, 9.0, 4096, dtype=torch.float32).reshape(64, 64)
        first = quantized.quantize_dequantize_weight(weight, 11, 1023, torch)
        second = quantized.quantize_dequantize_weight(weight, 11, 1023, torch)
        self.assertTrue(torch.equal(first, second))

    def test_recursive_discovery_finds_exact_blocks_zero_through_thirteen(self) -> None:
        model = MockCausalModel()
        found = quantized.discover_eligible_linear_weights(
            model, list(range(14)), torch
        )
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
        eligible = quantized.discover_eligible_linear_weights(
            model, list(range(14)), torch
        )
        selected = {f"{name}.weight" for name, _module in eligible}
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        quantized.apply_quantization(eligible, 11, 1023, torch)
        self.assertTrue(
            any(
                not torch.equal(parameter, before[name])
                for name, parameter in model.named_parameters()
                if name in selected
            )
        )
        for name, parameter in model.named_parameters():
            if name not in selected:
                self.assertTrue(torch.equal(parameter, before[name]), name)
        self.assertTrue(
            torch.equal(
                model.model.embed_tokens.weight,
                before["model.embed_tokens.weight"],
            )
        )
        self.assertTrue(torch.equal(model.model.norm.weight, before["model.norm.weight"]))
        self.assertTrue(torch.equal(model.lm_head.weight, before["lm_head.weight"]))
        self.assertTrue(torch.equal(model.lm_head.bias, before["lm_head.bias"]))

    def test_realized_metrics_use_actual_saved_float32_values(self) -> None:
        module = torch.nn.Linear(8, 16, bias=False)
        with torch.no_grad():
            module.weight.copy_(
                torch.linspace(-3.0, 2.0, 128, dtype=torch.float32).reshape(16, 8)
            )
        original = module.weight.detach().clone()
        records = quantized.apply_quantization(
            [("model.layers.0.projection", module)], 11, 1023, torch
        )
        expected = quantized.realized_perturbation_metrics(
            original, module.weight.detach(), torch
        )
        self.assertEqual(records[0]["shape"], [16, 8])
        for key, value in expected.items():
            self.assertEqual(records[0][key], value)
        self.assertEqual(module.weight.dtype, torch.float32)
        self.assertTrue(module.weight.is_contiguous())

    def test_overwrite_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            attempt_dir = root / "attempt"
            spec_path = attempt_dir / "spec.json"
            calibration_path = attempt_dir / "calibration.json"
            output_root = root / "outputs"
            merged_dir.mkdir()
            attempt_dir.mkdir()
            spec_path.write_text(json.dumps(quantized.FROZEN_SPEC), encoding="utf-8")
            calibration_path.write_text("{}\n", encoding="utf-8")
            (output_root / "model").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already exists"):
                quantized.validate_destinations(
                    merged_dir,
                    spec_path,
                    calibration_path,
                    output_root,
                    quantized.FROZEN_SPEC,
                )
            (output_root / "model").rmdir()
            manifest = attempt_dir / "construction-manifest.json"
            manifest.write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                quantized.validate_destinations(
                    merged_dir,
                    spec_path,
                    calibration_path,
                    output_root,
                    quantized.FROZEN_SPEC,
                )
            self.assertEqual(manifest.read_text(encoding="utf-8"), "existing\n")

    def test_manifest_ordering_and_aggregates_are_deterministic(self) -> None:
        test_spec = copy.deepcopy(quantized.FROZEN_SPEC)
        test_spec["expected_matrix_count"] = 2
        source = [file_record("z", "d"), file_record("a", "c")]
        output = [file_record("z", "f"), file_record("a", "e")]
        matrices = [matrix_record("z", 2.0), matrix_record("a", 1.0)]
        calibration_metadata = {
            "selected_bits": 11,
            "qmax": 1023,
            "selected_result": {
                "aggregate_relative_frobenius_perturbation": 0.001177,
            },
        }
        arguments = (
            test_spec,
            "a" * 64,
            "b" * 64,
            "c" * 64,
            calibration_metadata,
        )
        manifest_one = quantized.build_construction_manifest(
            *arguments, source, output, matrices
        )
        manifest_two = quantized.build_construction_manifest(
            *arguments,
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
        self.assertEqual(manifest_one["construction"]["selected_bits"], 11)
        self.assertEqual(manifest_one["construction"]["qmax"], 1023)
        aggregate = manifest_one["aggregate_perturbation"]
        self.assertEqual(aggregate["aggregate_original_frobenius_norm"], math.sqrt(20.0))
        self.assertEqual(
            aggregate["aggregate_realized_fp32_delta_frobenius_norm"], math.sqrt(5.0)
        )
        self.assertEqual(aggregate["changed_scalar_count"], 8)
        self.assertEqual(aggregate["changed_scalar_fraction"], 1.0)

    def test_calibration_checkpoint_mismatch_fails_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            attempt_dir = root / "attempt"
            output_root = root / "outputs"
            merged_dir.mkdir()
            attempt_dir.mkdir()
            spec_path = attempt_dir / "spec.json"
            calibration_path = attempt_dir / "calibration.json"
            spec_path.write_text(
                json.dumps(quantized.FROZEN_SPEC, indent=2) + "\n",
                encoding="utf-8",
            )
            calibration_path.write_text("{}\n", encoding="utf-8")
            metadata = {
                "source_checkpoint": {"files": [file_record("model", "a")]},
            }
            with (
                mock.patch.object(
                    quantized,
                    "load_calibration",
                    return_value=({}, metadata),
                ),
                mock.patch.object(
                    quantized,
                    "checkpoint_file_records",
                    return_value=[file_record("model", "b")],
                ),
                mock.patch.object(quantized, "load_local_checkpoint") as load_model,
            ):
                with self.assertRaisesRegex(ValueError, "disagrees"):
                    quantized.construct(
                        merged_dir,
                        spec_path,
                        calibration_path,
                        output_root,
                    )
                load_model.assert_not_called()

    def test_source_checkpoint_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            attempt_dir = root / "attempt"
            output_root = root / "outputs"
            merged_dir.mkdir()
            attempt_dir.mkdir()
            spec_path = attempt_dir / "spec.json"
            calibration_path = attempt_dir / "calibration.json"
            spec_path.write_text(
                json.dumps(quantized.FROZEN_SPEC, indent=2) + "\n",
                encoding="utf-8",
            )
            calibration_path.write_text("{}\n", encoding="utf-8")
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
            metadata = {
                "selected_bits": 11,
                "qmax": 1023,
                "selected_result": {
                    "aggregate_relative_frobenius_perturbation": 0.001177,
                },
                "source_checkpoint": {"files": before},
                "eligible_matrices": [],
            }
            with (
                mock.patch.object(
                    quantized,
                    "load_calibration",
                    return_value=({}, metadata),
                ),
                mock.patch.object(
                    quantized,
                    "checkpoint_file_records",
                    side_effect=lambda _path: next(checkpoint_calls),
                ),
                mock.patch.object(
                    quantized, "load_local_checkpoint", return_value=(object(), object())
                ),
                mock.patch.object(
                    quantized, "discover_eligible_linear_weights", return_value=eligible
                ),
                mock.patch.object(quantized, "validate_discovery_against_calibration"),
                mock.patch.object(
                    quantized, "apply_quantization", return_value=records
                ),
                mock.patch.object(quantized, "verify_float32_model"),
                mock.patch.object(
                    quantized,
                    "save_checkpoint",
                    return_value=[file_record("model", "c")],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "changed during construction"):
                    quantized.construct(
                        merged_dir,
                        spec_path,
                        calibration_path,
                        output_root,
                    )

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
            "attempt 003",
            "attempt001",
            "attempt002",
            "attempt003",
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
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
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
            environment_reads,
            quantized.ALLOWED_ENVIRONMENT_VARIABLES,
        )
        self.assertEqual(
            cli_options,
            {
                "--merged-model-dir",
                "--spec-path",
                "--calibration-path",
                "--output-root",
            },
        )


if __name__ == "__main__":
    unittest.main()
