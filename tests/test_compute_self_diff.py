import ast
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "compute_self_diff.py"
SPEC_PATH = (
    PROJECT
    / "experiments"
    / "attempts"
    / "001_spectral_tail_l13_p25"
    / "self_diff_spec.json"
)

MODULE_SPEC = importlib.util.spec_from_file_location("compute_self_diff", SCRIPT_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
self_diff = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = self_diff
MODULE_SPEC.loader.exec_module(self_diff)


def file_record(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": self_diff.sha256_file(path),
    }


def checkpoint_entry(root: Path) -> dict[str, object]:
    records = [
        file_record(path, root)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }


class SelfDifferenceTests(unittest.TestCase):
    def test_syntax_import_and_frozen_spec(self) -> None:
        ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        loaded = self_diff.load_self_diff_spec(SPEC_PATH)
        self.assertEqual(loaded, self_diff.FROZEN_SELF_DIFF_SPEC)

    def test_exact_checkpoint_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            nested = root / "nested"
            nested.mkdir(parents=True)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (nested / "weights.bin").write_bytes(b"synthetic-weights")
            entry = checkpoint_entry(root)

            verified = self_diff.verify_checkpoint(root, entry, "synthetic")
            self.assertEqual(verified, entry)

            (nested / "weights.bin").write_bytes(b"changed-weights")
            with self.assertRaisesRegex(ValueError, "size or SHA-256 mismatch"):
                self_diff.verify_checkpoint(root, entry, "synthetic")

            (nested / "extra").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "file set mismatch"):
                self_diff.verify_checkpoint(root, entry, "synthetic")

    def test_synthetic_construction_manifest_verifies_all_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            output_root = root / "outputs"
            attempt_spec_path = root / "spec.json"
            construction_path = root / "construction-manifest.json"
            merged_dir.mkdir()
            (merged_dir / "model.safetensors").write_bytes(b"merged")

            directory_names = {
                "0.05": "eps_0p05",
                "0.10": "eps_0p10",
                "0.20": "eps_0p20",
            }
            output_entries = []
            for epsilon in self_diff.FROZEN_SELF_DIFF_SPEC["epsilons"]:
                name = directory_names[f"{epsilon:.2f}"]
                checkpoint_dir = output_root / name
                checkpoint_dir.mkdir(parents=True)
                (checkpoint_dir / "model.safetensors").write_bytes(
                    f"epsilon={epsilon}".encode("ascii")
                )
                output_entries.append(
                    {
                        "epsilon": epsilon,
                        "directory_name": name,
                        **checkpoint_entry(checkpoint_dir),
                    }
                )

            attempt_spec = {
                "attempt_id": "001_spectral_tail_l13_p25",
                "epsilons": [0.05, 0.1, 0.2],
                "transformer_block_index": 13,
                "defaults": {"output_root": str(output_root)},
                "outputs": {"directory_names": directory_names},
            }
            attempt_spec_path.write_text(
                json.dumps(attempt_spec, indent=2) + "\n",
                encoding="utf-8",
            )
            construction = {
                "hash_algorithm": "sha256",
                "attempt_id": "001_spectral_tail_l13_p25",
                "spec_sha256": self_diff.sha256_file(attempt_spec_path),
                "construction": {
                    "transformer_block_index": 13,
                    "epsilons": [0.05, 0.1, 0.2],
                    "stored_dtype": "float32",
                },
                "source_checkpoint": checkpoint_entry(merged_dir),
                "output_checkpoints": output_entries,
            }
            construction_path.write_text(
                json.dumps(construction, indent=2) + "\n",
                encoding="utf-8",
            )

            context = self_diff.validate_attempt_inputs(
                merged_dir,
                attempt_spec_path,
                construction_path,
                self_diff.FROZEN_SELF_DIFF_SPEC,
            )

            self.assertEqual(
                [item["epsilon"] for item in context["ablated_checkpoints"]],
                [0.05, 0.1, 0.2],
            )
            self.assertEqual(
                context["merged_checkpoint"],
                checkpoint_entry(merged_dir),
            )

            changed = output_root / "eps_0p10" / "model.safetensors"
            changed.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size or SHA-256 mismatch"):
                self_diff.validate_attempt_inputs(
                    merged_dir,
                    attempt_spec_path,
                    construction_path,
                    self_diff.FROZEN_SELF_DIFF_SPEC,
                )

    def test_probe_serialized_and_raw_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe_path = Path(directory) / "probe.pt"
            probe = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
            torch.save(probe, probe_path)
            serialized_hash = self_diff.sha256_file(probe_path)
            raw_hash = self_diff.sha256_raw_int64_tensor(probe, torch)

            loaded = self_diff.load_and_verify_probe(
                probe_path,
                serialized_hash,
                raw_hash,
                (2, 3),
                torch,
            )
            self.assertTrue(torch.equal(loaded, probe))

            with self.assertRaisesRegex(ValueError, "serialized SHA-256 mismatch"):
                self_diff.load_and_verify_probe(
                    probe_path,
                    "0" * 64,
                    raw_hash,
                    (2, 3),
                    torch,
                )
            with self.assertRaisesRegex(ValueError, "raw tensor SHA-256 mismatch"):
                self_diff.load_and_verify_probe(
                    probe_path,
                    serialized_hash,
                    "0" * 64,
                    (2, 3),
                    torch,
                )

    def test_hook_uses_block_output_and_accumulates_in_float64(self) -> None:
        accumulator = self_diff.ActivationAccumulator(1, 1, torch)
        inputs = (torch.full((3, 1, 1), 99.0, dtype=torch.float32),)
        output = torch.tensor(
            [[[100000000.0]], [[1.0]], [[-100000000.0]]],
            dtype=torch.float32,
        )

        accumulator(None, inputs, (output, "ignored"))

        self.assertEqual(accumulator.sum.dtype, torch.float64)
        self.assertEqual(accumulator.sum.item(), 1.0)
        self.assertEqual(accumulator.sample_count, 3)
        self.assertEqual(accumulator.hook_calls, 1)
        self.assertEqual(accumulator.mean(3).item(), 1.0 / 3.0)

    def test_float32_model_invariant(self) -> None:
        model = torch.nn.Linear(2, 2).float()
        self_diff.verify_float32_model(model, torch, "cpu")

        model.double()
        with self.assertRaisesRegex(ValueError, "unexpected floating dtypes"):
            self_diff.verify_float32_model(model, torch, "cpu")

    def test_qwen_block_count_and_hidden_size_consistency(self) -> None:
        class MockModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList(
                    [
                        torch.nn.Identity()
                        for _ in range(
                            self_diff.FROZEN_SELF_DIFF_SPEC["num_hidden_layers"]
                        )
                    ]
                )
                self.config = SimpleNamespace(
                    model_type="qwen3",
                    num_hidden_layers=self_diff.FROZEN_SELF_DIFF_SPEC[
                        "num_hidden_layers"
                    ],
                    hidden_size=4,
                )

        model = MockModel()
        transformer, hidden_size = self_diff.validate_loaded_model(
            model,
            4,
            self_diff.FROZEN_SELF_DIFF_SPEC,
            torch,
            "cpu",
        )
        self.assertIs(transformer, model.model)
        self.assertEqual(hidden_size, 4)

        with self.assertRaisesRegex(ValueError, "hidden size mismatch"):
            self_diff.validate_loaded_model(
                model,
                5,
                self_diff.FROZEN_SELF_DIFF_SPEC,
                torch,
                "cpu",
            )

    def test_merged_minus_ablated_sign_and_identity(self) -> None:
        spec = dict(self_diff.FROZEN_SELF_DIFF_SPEC)
        spec["epsilons"] = [0.05, 0.1]
        spec["sequence_length"] = 2
        merged = torch.tensor(
            [[3.0, 5.0], [7.0, 11.0]],
            dtype=torch.float64,
        )
        changed = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float64,
        )

        artifact = self_diff.build_artifact(
            merged,
            [(0.1, merged.clone()), (0.05, changed)],
            2,
            spec,
            torch,
        )

        self.assertTrue(
            torch.equal(
                artifact["ablations"]["0.05"]["difference"],
                torch.tensor([[2.0, 3.0], [4.0, 7.0]], dtype=torch.float32),
            )
        )
        self.assertTrue(
            torch.equal(
                artifact["ablations"]["0.10"]["difference"],
                torch.zeros((2, 2), dtype=torch.float32),
            )
        )

    def test_shape_and_sample_validation(self) -> None:
        accumulator = self_diff.ActivationAccumulator(2, 3, torch)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            accumulator(
                None,
                (),
                torch.zeros((1, 3, 3), dtype=torch.float32),
            )
        accumulator(
            None,
            (),
            torch.zeros((1, 2, 3), dtype=torch.float32),
        )
        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            accumulator.mean(2)

        spec = dict(self_diff.FROZEN_SELF_DIFF_SPEC)
        spec["epsilons"] = [0.05]
        spec["sequence_length"] = 2
        wrong = torch.zeros((3, 4), dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "unexpected shape"):
            self_diff.build_artifact(
                wrong,
                [(0.05, wrong)],
                4,
                spec,
                torch,
            )

    def test_raw_tensor_hashing_is_little_endian_and_contiguous(self) -> None:
        integers = torch.tensor([[1, -2], [3, 4]], dtype=torch.int64).t()
        expected_integers = hashlib.sha256(
            struct.pack("<qqqq", 1, 3, -2, 4)
        ).hexdigest()
        self.assertEqual(
            self_diff.sha256_raw_int64_tensor(integers, torch),
            expected_integers,
        )

        floats = torch.tensor([[1.5, -2.25], [3.0, 4.5]], dtype=torch.float32).t()
        expected_floats = hashlib.sha256(
            struct.pack("<ffff", 1.5, 3.0, -2.25, 4.5)
        ).hexdigest()
        self.assertEqual(
            self_diff.sha256_raw_float32_tensor(floats, torch),
            expected_floats,
        )

    def test_overwrite_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "self_diff.pt"
            manifest_path = root / "self-diff-manifest.json"
            artifact_path.write_bytes(b"existing")

            with self.assertRaisesRegex(ValueError, "already exists"):
                self_diff.validate_output_paths(artifact_path, manifest_path)
            self.assertEqual(artifact_path.read_bytes(), b"existing")

            artifact_path.unlink()
            manifest_path.write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                self_diff.validate_output_paths(artifact_path, manifest_path)
            self.assertEqual(
                manifest_path.read_text(encoding="utf-8"),
                "existing\n",
            )

    def test_static_information_boundary_and_interface(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        lowered = source.lower()
        forbidden = (
            "base_model",
            "base model",
            "/models/base",
            "adapter",
            "lora",
            "oracle_adl.pt",
            "oracle-adl-manifest.json",
            "oracle-logit-lens.json",
            "models.lock.json",
            "downloaded-model-hashes.json",
            "merged-model-hashes.json",
        )
        for value in forbidden:
            self.assertNotIn(value, lowered)

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

        self.assertEqual(environment_reads, self_diff.ALLOWED_ENVIRONMENT_VARIABLES)
        for value in environment_reads | cli_options:
            lowered_value = value.lower()
            self.assertNotIn("base", lowered_value)
            self.assertNotIn("adapter", lowered_value)
            self.assertNotIn("oracle", lowered_value)

    def test_deterministic_manifest_ordering(self) -> None:
        spec = dict(self_diff.FROZEN_SELF_DIFF_SPEC)
        merged = {
            "file_count": 2,
            "total_bytes": 2,
            "files": [
                {"path": "z", "size_bytes": 1, "sha256": "2" * 64},
                {"path": "a", "size_bytes": 1, "sha256": "1" * 64},
            ],
        }

        def checkpoint(epsilon: float) -> dict[str, object]:
            return {
                "epsilon": epsilon,
                "directory_name": f"eps_{epsilon:.2f}",
                "file_count": 2,
                "total_bytes": 2,
                "files": [
                    {"path": "z", "size_bytes": 1, "sha256": "4" * 64},
                    {"path": "a", "size_bytes": 1, "sha256": "3" * 64},
                ],
            }

        raw = {
            "merged_mean": "5" * 64,
            "ablations": [
                {
                    "epsilon": epsilon,
                    "ablated_mean": "6" * 64,
                    "difference": "7" * 64,
                }
                for epsilon in reversed(spec["epsilons"])
            ],
        }
        checkpoints = [
            checkpoint(epsilon) for epsilon in reversed(spec["epsilons"])
        ]
        arguments = (
            spec,
            "8" * 64,
            "9" * 64,
            "a" * 64,
            "b" * 64,
            merged,
            checkpoints,
            "c" * 64,
            raw,
            4,
        )

        first = self_diff.build_manifest(*arguments)
        second = self_diff.build_manifest(
            spec,
            "8" * 64,
            "9" * 64,
            "a" * 64,
            "b" * 64,
            {**merged, "files": list(reversed(merged["files"]))},
            list(reversed(checkpoints)),
            "c" * 64,
            {**raw, "ablations": list(reversed(raw["ablations"]))},
            4,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, allow_nan=False, indent=2),
            json.dumps(second, ensure_ascii=False, allow_nan=False, indent=2),
        )
        self.assertEqual(
            [item["epsilon"] for item in first["input_checkpoints"]["ablated"]],
            spec["epsilons"],
        )
        self.assertEqual(
            [item["path"] for item in first["input_checkpoints"]["merged"]["files"]],
            ["a", "z"],
        )


if __name__ == "__main__":
    unittest.main()
