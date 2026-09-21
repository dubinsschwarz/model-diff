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
SCRIPT_PATH = (
    PROJECT / "scripts" / "ablation" / "compute_self_diff_attempt002.py"
)
SPEC_PATH = (
    PROJECT
    / "experiments"
    / "attempts"
    / "002_bf16_roundtrip_prefix0_13"
    / "self_diff_spec.json"
)

MODULE_SPEC = importlib.util.spec_from_file_location(
    "compute_self_diff_attempt002", SCRIPT_PATH
)
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


class Attempt002SelfDifferenceTests(unittest.TestCase):
    def test_syntax_import_and_frozen_spec(self) -> None:
        ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        loaded = self_diff.load_self_diff_spec(SPEC_PATH)
        self.assertEqual(loaded, self_diff.FROZEN_SELF_DIFF_SPEC)
        self.assertEqual(loaded["readout_block_index"], 13)
        self.assertEqual(loaded["difference_sign"], "merged_minus_ablated")

    def test_exact_checkpoint_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            nested = root / "nested"
            nested.mkdir(parents=True)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            (nested / "weights.bin").write_bytes(b"synthetic-weights")
            entry = checkpoint_entry(root)

            self.assertEqual(
                self_diff.verify_checkpoint(root, entry, "synthetic"), entry
            )
            (nested / "weights.bin").write_bytes(b"changed-weights")
            with self.assertRaisesRegex(ValueError, "size or SHA-256 mismatch"):
                self_diff.verify_checkpoint(root, entry, "synthetic")
            (nested / "extra").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "file set mismatch"):
                self_diff.verify_checkpoint(root, entry, "synthetic")

    def test_construction_manifest_verifies_both_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            output_root = root / "outputs"
            ablated_dir = output_root / "model"
            attempt_spec_path = root / "spec.json"
            construction_path = root / "construction-manifest.json"
            merged_dir.mkdir()
            ablated_dir.mkdir(parents=True)
            (merged_dir / "model.safetensors").write_bytes(b"merged")
            (ablated_dir / "model.safetensors").write_bytes(b"rounded")

            attempt_spec = {
                "attempt_id": "002_bf16_roundtrip_prefix0_13",
                "method": "bf16_weight_roundtrip",
                "input_information_policy": "merged_checkpoint_only",
                "transformer_block_indices": list(range(14)),
                "source_weight_dtype": "float32",
                "defaults": {"output_root": str(output_root)},
                "outputs": {"directory_name": "model"},
            }
            attempt_spec_path.write_text(
                json.dumps(attempt_spec, indent=2) + "\n", encoding="utf-8"
            )
            construction = {
                "hash_algorithm": "sha256",
                "attempt_id": "002_bf16_roundtrip_prefix0_13",
                "provenance_free": True,
                "input_information_policy": "merged_checkpoint_only",
                "spec_sha256": self_diff.sha256_file(attempt_spec_path),
                "construction": {
                    "transformer_block_indices": list(range(14)),
                    "source_weight_dtype": "float32",
                    "weight_transform": (
                        "W_original.to(torch.bfloat16).to(torch.float32)"
                    ),
                    "randomness": "none",
                    "stored_dtype": "float32",
                    "standalone_checkpoint": True,
                },
                "source_checkpoint": checkpoint_entry(merged_dir),
                "output_checkpoint": {
                    "directory_name": "model",
                    **checkpoint_entry(ablated_dir),
                },
            }
            construction_path.write_text(
                json.dumps(construction, indent=2) + "\n", encoding="utf-8"
            )

            context = self_diff.validate_attempt_inputs(
                merged_dir,
                attempt_spec_path,
                construction_path,
                self_diff.FROZEN_SELF_DIFF_SPEC,
            )
            self.assertEqual(context["merged_checkpoint"], checkpoint_entry(merged_dir))
            self.assertEqual(context["ablated_checkpoint"]["directory"], ablated_dir)
            self.assertEqual(context["ablated_checkpoint"]["directory_name"], "model")

            (ablated_dir / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size or SHA-256 mismatch"):
                self_diff.validate_attempt_inputs(
                    merged_dir,
                    attempt_spec_path,
                    construction_path,
                    self_diff.FROZEN_SELF_DIFF_SPEC,
                )

    def test_probe_serialized_and_raw_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe_path = Path(directory) / "fineweb_tokens.pt"
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
                    probe_path, "0" * 64, raw_hash, (2, 3), torch
                )
            with self.assertRaisesRegex(ValueError, "raw tensor SHA-256 mismatch"):
                self_diff.load_and_verify_probe(
                    probe_path, serialized_hash, "0" * 64, (2, 3), torch
                )

    def test_hook_uses_block_output_and_accumulates_cpu_float64(self) -> None:
        accumulator = self_diff.ActivationAccumulator(1, 1, torch)
        inputs = (torch.full((3, 1, 1), 99.0, dtype=torch.float32),)
        output = torch.tensor(
            [[[100000000.0]], [[1.0]], [[-100000000.0]]],
            dtype=torch.float32,
        )
        accumulator(None, inputs, (output, "ignored"))

        self.assertEqual(accumulator.sum.dtype, torch.float64)
        self.assertEqual(accumulator.sum.device.type, "cpu")
        self.assertEqual(accumulator.sum.item(), 1.0)
        self.assertEqual(accumulator.sample_count, 3)
        self.assertEqual(accumulator.hook_calls, 1)
        self.assertEqual(accumulator.mean(3).item(), 1.0 / 3.0)

    def test_float32_model_and_hidden_size_invariants(self) -> None:
        class MockModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList(
                    [torch.nn.Identity() for _ in range(28)]
                )
                self.config = SimpleNamespace(
                    model_type="qwen3", num_hidden_layers=28, hidden_size=4
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

        model.double()
        with self.assertRaisesRegex(ValueError, "unexpected floating dtypes"):
            self_diff.verify_float32_model(model, torch, "cpu")
        model.float()
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
        spec["sequence_length"] = 2
        merged = torch.tensor(
            [[3.0, 5.0], [7.0, 11.0]], dtype=torch.float64
        ).contiguous()
        ablated = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64
        ).contiguous()
        artifact = self_diff.build_artifact(merged, ablated, 2, spec, torch)

        self.assertTrue(
            torch.equal(
                artifact["difference"],
                torch.tensor([[2.0, 3.0], [4.0, 7.0]], dtype=torch.float32),
            )
        )
        identity = self_diff.build_artifact(
            merged, merged.clone(), 2, spec, torch
        )
        self.assertTrue(
            torch.equal(identity["difference"], torch.zeros((2, 2)))
        )
        for name in ("merged_mean", "ablated_mean", "difference"):
            self.assertEqual(artifact[name].dtype, torch.float32)
            self.assertTrue(artifact[name].is_contiguous())

    def test_shape_and_sample_count_validation(self) -> None:
        accumulator = self_diff.ActivationAccumulator(2, 3, torch)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            accumulator(None, (), torch.zeros((1, 3, 3), dtype=torch.float32))
        accumulator(None, (), torch.zeros((1, 2, 3), dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            accumulator.mean(2)

        spec = dict(self_diff.FROZEN_SELF_DIFF_SPEC)
        spec["sequence_length"] = 2
        wrong = torch.zeros((3, 4), dtype=torch.float64).contiguous()
        with self.assertRaisesRegex(ValueError, "unexpected shape"):
            self_diff.build_artifact(wrong, wrong, 4, spec, torch)

    def test_raw_tensor_hashing_is_little_endian_and_contiguous(self) -> None:
        integers = torch.tensor([[1, -2], [3, 4]], dtype=torch.int64).t()
        expected_integers = hashlib.sha256(
            struct.pack("<qqqq", 1, 3, -2, 4)
        ).hexdigest()
        self.assertEqual(
            self_diff.sha256_raw_int64_tensor(integers, torch), expected_integers
        )

        floats = torch.tensor([[1.5, -2.25], [3.0, 4.5]], dtype=torch.float32).t()
        expected_floats = hashlib.sha256(
            struct.pack("<ffff", 1.5, 3.0, -2.25, 4.5)
        ).hexdigest()
        self.assertEqual(
            self_diff.sha256_raw_float32_tensor(floats, torch), expected_floats
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
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), "existing\n")

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
            "oracle_adl_manifest",
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
        self.assertEqual(
            cli_options,
            {
                "--merged-model-dir",
                "--attempt-spec-path",
                "--construction-manifest-path",
                "--probe-path",
                "--self-diff-spec-path",
                "--artifact-path",
                "--manifest-path",
            },
        )
        for value in environment_reads | cli_options:
            lowered_value = value.lower()
            self.assertNotIn("base", lowered_value)
            self.assertNotIn("adapter", lowered_value)
            self.assertNotIn("oracle", lowered_value)

    def test_deterministic_manifest_ordering(self) -> None:
        merged = {
            "file_count": 2,
            "total_bytes": 2,
            "files": [
                {"path": "z", "size_bytes": 1, "sha256": "2" * 64},
                {"path": "a", "size_bytes": 1, "sha256": "1" * 64},
            ],
        }
        ablated = {
            "directory_name": "model",
            "file_count": 2,
            "total_bytes": 2,
            "files": [
                {"path": "z", "size_bytes": 1, "sha256": "4" * 64},
                {"path": "a", "size_bytes": 1, "sha256": "3" * 64},
            ],
        }
        raw = {
            "difference": "7" * 64,
            "ablated_mean": "6" * 64,
            "merged_mean": "5" * 64,
        }
        arguments = (
            self_diff.FROZEN_SELF_DIFF_SPEC,
            "8" * 64,
            "9" * 64,
            "a" * 64,
            "b" * 64,
            merged,
            ablated,
            "c" * 64,
            raw,
            4,
        )
        first = self_diff.build_manifest(*arguments)
        second = self_diff.build_manifest(
            self_diff.FROZEN_SELF_DIFF_SPEC,
            "8" * 64,
            "9" * 64,
            "a" * 64,
            "b" * 64,
            {**merged, "files": list(reversed(merged["files"]))},
            {**ablated, "files": list(reversed(ablated["files"]))},
            "c" * 64,
            dict(reversed(list(raw.items()))),
            4,
        )

        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, allow_nan=False, indent=2),
            json.dumps(second, ensure_ascii=False, allow_nan=False, indent=2),
        )
        self.assertEqual(
            [item["path"] for item in first["input_checkpoints"]["merged"]["files"]],
            ["a", "z"],
        )
        self.assertEqual(
            list(first["self_diff_artifact"]["raw_tensors_sha256"]),
            ["merged_mean", "ablated_mean", "difference"],
        )


if __name__ == "__main__":
    unittest.main()
