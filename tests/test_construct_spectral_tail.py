import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "construct_spectral_tail.py"
SPEC_PATH = (
    PROJECT
    / "experiments"
    / "attempts"
    / "001_spectral_tail_l13_p25"
    / "spec.json"
)

SPEC = importlib.util.spec_from_file_location("construct_spectral_tail", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
spectral = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spectral
SPEC.loader.exec_module(spectral)


class SpectralTailTests(unittest.TestCase):
    def test_committed_spec_is_the_frozen_spec(self) -> None:
        loaded = spectral.load_spec(SPEC_PATH)
        self.assertEqual(loaded, spectral.FROZEN_SPEC)

    def test_exact_tail_selection(self) -> None:
        weight = torch.diag(torch.tensor([8.0, 4.0, 2.0, 1.0]))

        singular_values, tail_count, tail = spectral.spectral_tail_component(
            weight,
            0.5,
            torch,
        )

        self.assertTrue(
            torch.equal(
                singular_values,
                torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=torch.float64),
            )
        )
        self.assertEqual(tail_count, 2)
        self.assertTrue(
            torch.allclose(
                tail,
                torch.diag(
                    torch.tensor([0.0, 0.0, 2.0, 1.0], dtype=torch.float64)
                ),
                atol=1e-12,
                rtol=0.0,
            )
        )

    def test_epsilon_zero_is_exact_float32_identity(self) -> None:
        original = torch.tensor(
            [[1.25, -2.5], [3.75, 4.125]],
            dtype=torch.float32,
        )
        _singular_values, _tail_count, tail = spectral.spectral_tail_component(
            original,
            0.5,
            torch,
        )

        modified = spectral.construct_modified_weight(original, tail, 0.0, torch)

        self.assertTrue(torch.equal(modified, original))

    def test_known_diagonal_matrix_attenuation(self) -> None:
        original = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
        _singular_values, tail_count, tail = spectral.spectral_tail_component(
            original,
            0.25,
            torch,
        )

        modified = spectral.construct_modified_weight(original, tail, 0.2, torch)

        self.assertEqual(tail_count, 1)
        self.assertTrue(
            torch.allclose(
                modified,
                torch.diag(torch.tensor([4.0, 3.0, 2.0, 0.8])),
                atol=1e-7,
                rtol=0.0,
            )
        )

    def test_metrics_measure_realized_float32_delta(self) -> None:
        original = torch.tensor([[16777216.0]], dtype=torch.float32)
        component = original.to(torch.float64)

        modified = spectral.construct_modified_weight(
            original,
            component,
            1e-8,
            torch,
        )
        metrics = spectral.realized_perturbation_metrics(
            original,
            modified,
            torch,
        )

        self.assertTrue(torch.equal(modified, original))
        self.assertGreater(float(component.item()) * 1e-8, 0.0)
        self.assertEqual(metrics["realized_fp32_delta_frobenius_norm"], 0.0)
        self.assertEqual(metrics["realized_relative_frobenius_perturbation"], 0.0)
        self.assertEqual(metrics["max_absolute_weight_change"], 0.0)

    def test_recursive_linear_discovery_is_generic_and_sorted(self) -> None:
        class Nested(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.deep = torch.nn.Sequential(
                    torch.nn.LayerNorm(3),
                    torch.nn.Linear(3, 2),
                )

        block = torch.nn.ModuleDict(
            {
                "z_linear": torch.nn.Linear(2, 2),
                "nested": Nested(),
                "activation": torch.nn.ReLU(),
            }
        )

        found = spectral.discover_linear_weights(block, "target", torch)

        self.assertEqual(
            [name for name, _module in found],
            ["target.nested.deep.1", "target.z_linear"],
        )
        self.assertTrue(all(isinstance(module, torch.nn.Linear) for _, module in found))

    def test_weight_plan_metrics_and_scope_preservation(self) -> None:
        target = torch.nn.Linear(4, 4, bias=True)
        outside = torch.nn.Linear(4, 4, bias=True)
        with torch.no_grad():
            target.weight.copy_(torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0])))
            target.bias.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
            outside.weight.fill_(7.0)
            outside.bias.fill_(8.0)
        target_bias = target.bias.detach().clone()
        outside_weight = outside.weight.detach().clone()
        outside_bias = outside.bias.detach().clone()

        plan = spectral.build_weight_plan(
            "model.layers.13.target",
            target,
            spectral.FROZEN_SPEC,
            torch,
        )
        record = plan["record"]

        self.assertEqual(record["shape"], [4, 4])
        self.assertEqual(record["singular_value_count"], 4)
        self.assertEqual(record["tail_count"], 1)
        self.assertAlmostEqual(record["tail_energy_fraction"], 1.0 / 30.0)
        self.assertAlmostEqual(record["original_frobenius_norm"], 30.0**0.5)
        self.assertAlmostEqual(record["tail_component_frobenius_norm"], 1.0)

        spectral.apply_epsilon([plan], 0.2, torch)
        self.assertTrue(
            torch.allclose(
                target.weight,
                torch.diag(torch.tensor([4.0, 3.0, 2.0, 0.8])),
                atol=1e-7,
                rtol=0.0,
            )
        )
        self.assertTrue(torch.equal(target.bias, target_bias))
        self.assertTrue(torch.equal(outside.weight, outside_weight))
        self.assertTrue(torch.equal(outside.bias, outside_bias))

        spectral.restore_original_weights([plan], torch)
        self.assertTrue(torch.equal(target.weight, plan["original32"]))

    def test_constructor_static_information_boundary(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        lowered_source = source.lower()
        forbidden = (
            "base_model",
            "base model",
            "/models/base",
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
            environment_reads,
            spectral.ALLOWED_ENVIRONMENT_VARIABLES,
        )
        self.assertEqual(
            cli_options,
            {"--merged-model-dir", "--spec-path", "--output-root"},
        )

    def test_existing_output_checkpoint_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            output_root = root / "outputs"
            spec_path = root / "attempt" / "spec.json"
            merged_dir.mkdir()
            spec_path.parent.mkdir()
            spec_path.write_text(json.dumps(spectral.FROZEN_SPEC), encoding="utf-8")
            (output_root / "eps_0p10").mkdir(parents=True)

            with self.assertRaisesRegex(ValueError, "already exists"):
                spectral.validate_destinations(
                    merged_dir,
                    spec_path,
                    output_root,
                    spectral.FROZEN_SPEC,
                )

    def test_existing_construction_manifest_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            output_root = root / "outputs"
            spec_path = root / "attempt" / "spec.json"
            merged_dir.mkdir()
            spec_path.parent.mkdir()
            spec_path.write_text(json.dumps(spectral.FROZEN_SPEC), encoding="utf-8")
            manifest_path = (
                spec_path.parent
                / spectral.FROZEN_SPEC["outputs"]["construction_manifest"]
            )
            manifest_path.write_text("existing\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "already exists"):
                spectral.validate_destinations(
                    merged_dir,
                    spec_path,
                    output_root,
                    spectral.FROZEN_SPEC,
                )

            self.assertEqual(manifest_path.read_text(encoding="utf-8"), "existing\n")

    def test_manifest_ordering_is_deterministic(self) -> None:
        def matrix(name: str) -> dict[str, object]:
            return {
                "module_name": name,
                "shape": [2, 2],
                "singular_value_count": 2,
                "tail_count": 0,
                "tail_energy_fraction": 0.0,
                "original_frobenius_norm": 2.0,
                "tail_component_frobenius_norm": 0.0,
                "epsilons": [
                    {
                        "epsilon": epsilon,
                        "realized_fp32_delta_frobenius_norm": epsilon,
                        "realized_relative_frobenius_perturbation": epsilon / 2,
                        "max_absolute_weight_change": epsilon,
                    }
                    for epsilon in reversed(spectral.FROZEN_SPEC["epsilons"])
                ],
            }

        outputs = [
            {
                "epsilon": epsilon,
                "directory_name": spectral.FROZEN_SPEC["outputs"][
                    "directory_names"
                ][spectral.epsilon_key(epsilon)],
                "files": [
                    {"path": "z", "size_bytes": 1, "sha256": "f" * 64},
                    {"path": "a", "size_bytes": 1, "sha256": "e" * 64},
                ],
            }
            for epsilon in reversed(spectral.FROZEN_SPEC["epsilons"])
        ]
        manifest_one = spectral.build_construction_manifest(
            spectral.FROZEN_SPEC,
            "a" * 64,
            "b" * 64,
            [
                {"path": "z", "size_bytes": 1, "sha256": "d" * 64},
                {"path": "a", "size_bytes": 1, "sha256": "c" * 64},
            ],
            outputs,
            [matrix("z"), matrix("a")],
        )
        manifest_two = spectral.build_construction_manifest(
            spectral.FROZEN_SPEC,
            "a" * 64,
            "b" * 64,
            list(reversed(manifest_one["source_checkpoint"]["files"])),
            list(reversed(outputs)),
            [matrix("a"), matrix("z")],
        )

        self.assertEqual(manifest_one, manifest_two)
        self.assertEqual(
            json.dumps(manifest_one, ensure_ascii=False, allow_nan=False, indent=2),
            json.dumps(manifest_two, ensure_ascii=False, allow_nan=False, indent=2),
        )
        self.assertEqual(
            [item["module_name"] for item in manifest_one["modified_matrices"]],
            ["a", "z"],
        )
        self.assertEqual(
            [item["epsilon"] for item in manifest_one["output_checkpoints"]],
            spectral.FROZEN_SPEC["epsilons"],
        )
        self.assertEqual(
            [item["path"] for item in manifest_one["source_checkpoint"]["files"]],
            ["a", "z"],
        )


if __name__ == "__main__":
    unittest.main()
