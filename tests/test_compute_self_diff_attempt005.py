import ast
import copy
import hashlib
import importlib.util
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125"
SCRIPT_PATH = PROJECT / "scripts/ablation/compute_self_diff_attempt005.py"
SPEC_PATH = ATTEMPT_DIRECTORY / "self_diff_spec.json"
MODULE_SPEC = importlib.util.spec_from_file_location("compute_self_diff_attempt005", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
self_diff = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = self_diff
MODULE_SPEC.loader.exec_module(self_diff)


def checkpoint_entry(root: Path) -> dict[str, object]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": self_diff.sha256_file(path),
        }
        for path in sorted(root.rglob("*")) if path.is_file()
    ]
    return {
        "file_count": len(files),
        "total_bytes": sum(record["size_bytes"] for record in files),
        "files": files,
    }


class Attempt005SelfDifferenceTests(unittest.TestCase):
    def test_exact_frozen_spec(self):
        ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        spec = self_diff.load_self_diff_spec(SPEC_PATH)
        self.assertEqual(spec, self_diff.FROZEN_SELF_DIFF_SPEC)
        self.assertEqual(spec["attempt_id"], "005_generic_gradient_rollback_prefix0_13_r00125")
        self.assertEqual(spec["readout_block_index"], 13)
        self.assertEqual((spec["sample_count"], spec["sequence_length"], spec["batch_size"]),
                         (10000, 128, 32))
        self.assertEqual(spec["difference_sign"], "merged_minus_rollback")
        self.assertEqual(spec["probe"]["serialized_sha256"],
                         "3d799d19abf4a2d6dc92b2bd164d81d83f6451cd3545f5ae5d8c04a8b299735b")
        self.assertEqual(spec["probe"]["raw_tensor_sha256"],
                         "73f56300fdd06bc8fafcfa5750fee7a586a5c5e6071a608667ffc526bbb90ac8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "self_diff_spec.json"
            changed = copy.deepcopy(spec)
            changed["difference_sign"] = "rollback_minus_merged"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen definition"):
                self_diff.load_self_diff_spec(path)

    def test_exact_checkpoint_file_set_size_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoint"
            (root / "nested").mkdir(parents=True)
            (root / "config.json").write_bytes(b"{}")
            (root / "nested/model.safetensors").write_bytes(b"synthetic")
            entry = checkpoint_entry(root)
            self.assertEqual(self_diff.verify_checkpoint(root, entry, "synthetic"), entry)
            (root / "nested/model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "size or SHA-256 mismatch"):
                self_diff.verify_checkpoint(root, entry, "synthetic")
            (root / "extra").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "file set mismatch"):
                self_diff.verify_checkpoint(root, entry, "synthetic")

    def test_committed_attempt005_manifest_schema_and_hash_links_without_checkpoint_io(self):
        spec = self_diff.FROZEN_SELF_DIFF_SPEC
        merged_dir = Path(spec["defaults"]["merged_model_directory"])
        with patch.object(
            self_diff, "verify_checkpoint",
            side_effect=lambda _root, entry, description: self_diff.normalize_checkpoint_entry(entry, description),
        ) as checked:
            context = self_diff.validate_attempt_inputs(
                merged_dir,
                ATTEMPT_DIRECTORY / "spec.json",
                ATTEMPT_DIRECTORY / "corpus_spec.json",
                ATTEMPT_DIRECTORY / "corpus-manifest.json",
                ATTEMPT_DIRECTORY / "construction-manifest.json",
                spec,
            )
        self.assertEqual(checked.call_count, 2)
        self.assertEqual(context["merged_checkpoint"]["file_count"], 6)
        self.assertEqual(context["rollback_checkpoint"]["file_count"], 6)
        self.assertEqual(context["rollback_checkpoint"]["directory_name"], "model")
        self.assertEqual(context["attempt_spec_sha256"],
                         self_diff.sha256_file(ATTEMPT_DIRECTORY / "spec.json"))
        self.assertEqual(context["corpus_manifest_sha256"],
                         self_diff.sha256_file(ATTEMPT_DIRECTORY / "corpus-manifest.json"))

    def test_construction_manifest_link_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            construction = json.loads(
                (ATTEMPT_DIRECTORY / "construction-manifest.json").read_text(encoding="utf-8")
            )
            construction["corpus_manifest_sha256"] = "0" * 64
            path = root / "construction-manifest.json"
            path.write_text(json.dumps(construction), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash linkage mismatch: corpus_manifest_sha256"):
                self_diff.validate_attempt_inputs(
                    Path(self_diff.FROZEN_SELF_DIFF_SPEC["defaults"]["merged_model_directory"]),
                    ATTEMPT_DIRECTORY / "spec.json",
                    ATTEMPT_DIRECTORY / "corpus_spec.json",
                    ATTEMPT_DIRECTORY / "corpus-manifest.json",
                    path,
                    self_diff.FROZEN_SELF_DIFF_SPEC,
                )

    def test_synthetic_construction_chain_verifies_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged_dir = root / "merged"
            rollback_dir = root / "rollback" / "model"
            merged_dir.mkdir()
            rollback_dir.mkdir(parents=True)
            for checkpoint, label in ((merged_dir, b"merged"), (rollback_dir, b"rollback")):
                for name in (
                    "chat_template.jinja", "config.json", "generation_config.json",
                    "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                ):
                    (checkpoint / name).write_bytes(label + name.encode())
            attempt = json.loads((ATTEMPT_DIRECTORY / "spec.json").read_text(encoding="utf-8"))
            corpus_spec = json.loads((ATTEMPT_DIRECTORY / "corpus_spec.json").read_text(encoding="utf-8"))
            corpus_manifest = json.loads((ATTEMPT_DIRECTORY / "corpus-manifest.json").read_text(encoding="utf-8"))
            construction = json.loads((ATTEMPT_DIRECTORY / "construction-manifest.json").read_text(encoding="utf-8"))
            corpus_path = root / "tokens.pt"
            attempt["defaults"].update({
                "merged_model_directory": str(merged_dir),
                "output_model_directory": str(rollback_dir),
                "corpus_artifact_path": str(corpus_path),
            })
            corpus_spec["tokenizer"]["directory"] = str(merged_dir)
            corpus_spec["artifact"]["path"] = str(corpus_path)
            attempt_path = root / "spec.json"
            corpus_spec_path = root / "corpus_spec.json"
            corpus_manifest_path = root / "corpus-manifest.json"
            construction_path = root / "construction-manifest.json"
            attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
            corpus_spec_path.write_text(json.dumps(corpus_spec), encoding="utf-8")
            source_entry = checkpoint_entry(merged_dir)
            corpus_manifest["corpus_spec_sha256"] = self_diff.sha256_file(corpus_spec_path)
            corpus_manifest["tokenizer_checkpoint"] = {
                "source": "canonical_merged_checkpoint", "directory": str(merged_dir),
                "file_count": 4,
                "files": [record for record in source_entry["files"] if record["path"] in {
                    "chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json",
                }],
            }
            corpus_manifest["artifact"] = {
                "path": str(corpus_path), "serialized_sha256": "a" * 64,
                "raw_tensor_sha256": "b" * 64,
            }
            corpus_manifest_path.write_text(json.dumps(corpus_manifest), encoding="utf-8")
            names = sorted(
                f"model.layers.{block}.linear_{index}"
                for block in range(14) for index in range(7)
            )
            construction.update({
                "attempt_spec_sha256": self_diff.sha256_file(attempt_path),
                "corpus_spec_sha256": self_diff.sha256_file(corpus_spec_path),
                "corpus_manifest_sha256": self_diff.sha256_file(corpus_manifest_path),
                "corpus": {"serialized_sha256": "a" * 64, "raw_tensor_sha256": "b" * 64},
                "source_checkpoint": source_entry,
                "output_checkpoint": {"directory_name": "model", **checkpoint_entry(rollback_dir)},
                "loss": {**attempt["generic_loss"], "mean_generic_loss": 1.0},
                "gradient_and_rollback": {
                    "eligible_matrix_count": 98,
                    "target_relative_frobenius": 0.00125,
                    "aggregate_source_weight_norm": 10.0,
                    "aggregate_generic_gradient_norm": 2.0,
                    "target_delta_norm": 0.0125,
                    "alpha": 0.00625,
                    "aggregate_realized_fp32_delta_norm": 0.0125,
                    "aggregate_realized_relative_perturbation": 0.00125,
                },
                "deterministic_ordering": {
                    "rule": "lexicographic_by_module_name", "modified_module_names": names,
                },
                "modified_matrices": [
                    {
                        "name": name, "shape": [2, 2],
                        "source_frobenius_norm": 1.0,
                        "gradient_frobenius_norm": 1.0,
                        "realized_fp32_delta_frobenius_norm": 0.00125,
                        "realized_relative_frobenius_perturbation": 0.00125,
                        "max_absolute_weight_change": 0.001,
                        "changed_element_count": 4,
                        "changed_element_fraction": 1.0,
                    }
                    for name in names
                ],
            })
            construction_path.write_text(json.dumps(construction), encoding="utf-8")
            context = self_diff.validate_attempt_inputs(
                merged_dir, attempt_path, corpus_spec_path, corpus_manifest_path,
                construction_path, self_diff.FROZEN_SELF_DIFF_SPEC,
            )
            self.assertEqual(context["merged_checkpoint"], source_entry)
            self.assertEqual(context["rollback_checkpoint"]["directory"], rollback_dir)
            (rollback_dir / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size or SHA-256 mismatch"):
                self_diff.validate_attempt_inputs(
                    merged_dir, attempt_path, corpus_spec_path, corpus_manifest_path,
                    construction_path, self_diff.FROZEN_SELF_DIFF_SPEC,
                )
            (rollback_dir / "model.safetensors").write_bytes(b"rollbackmodel.safetensors")
            (merged_dir / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "size or SHA-256 mismatch"):
                self_diff.validate_attempt_inputs(
                    merged_dir, attempt_path, corpus_spec_path, corpus_manifest_path,
                    construction_path, self_diff.FROZEN_SELF_DIFF_SPEC,
                )

    def test_serialized_and_raw_probe_hash_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
            torch.save(tokens, path)
            serialized = self_diff.sha256_file(path)
            raw = self_diff.sha256_raw_int64_tensor(tokens, torch)
            loaded = self_diff.load_and_verify_probe(path, serialized, raw, (2, 3), torch)
            self.assertTrue(torch.equal(loaded, tokens))
            with self.assertRaisesRegex(ValueError, "serialized SHA-256 mismatch"):
                self_diff.load_and_verify_probe(path, "0" * 64, raw, (2, 3), torch)
            with self.assertRaisesRegex(ValueError, "raw tensor SHA-256 mismatch"):
                self_diff.load_and_verify_probe(path, serialized, "0" * 64, (2, 3), torch)

    def test_hook_uses_block_output_and_cpu_float64_accumulation(self):
        accumulator = self_diff.ActivationAccumulator(1, 1, torch)
        inputs = (torch.full((3, 1, 1), 99.0),)
        outputs = torch.tensor([[[100000000.0]], [[1.0]], [[-100000000.0]]], dtype=torch.float32)
        accumulator(None, inputs, (outputs, "ignored"))
        self.assertEqual(accumulator.sum.dtype, torch.float64)
        self.assertEqual(accumulator.sum.device.type, "cpu")
        self.assertEqual(accumulator.sum.item(), 1.0)
        self.assertEqual(accumulator.sample_count, 3)
        self.assertEqual(accumulator.hook_calls, 1)
        self.assertEqual(accumulator.mean(3).item(), 1.0 / 3.0)
        with self.assertRaisesRegex(ValueError, "sample count mismatch"):
            accumulator.mean(2)

    def test_float32_model_and_readout_block_invariants(self):
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(28)])
                self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=28, hidden_size=4)

        model = MockModel()
        transformer, hidden_size = self_diff.validate_loaded_model(
            model, 4, self_diff.FROZEN_SELF_DIFF_SPEC, torch, "cpu"
        )
        self.assertIs(transformer, model.model)
        self.assertEqual(hidden_size, 4)
        model.double()
        with self.assertRaisesRegex(ValueError, "unexpected floating dtypes"):
            self_diff.verify_float32_model(model, torch, "cpu")
        model.float()
        with self.assertRaisesRegex(ValueError, "hidden size mismatch"):
            self_diff.validate_loaded_model(model, 5, self_diff.FROZEN_SELF_DIFF_SPEC, torch, "cpu")

    def test_merged_minus_rollback_sign_and_all_positions(self):
        spec = copy.deepcopy(self_diff.FROZEN_SELF_DIFF_SPEC)
        spec["sequence_length"] = 2
        merged = torch.tensor([[3.0, 5.0], [7.0, 11.0]], dtype=torch.float64)
        rollback_mean = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        artifact = self_diff.build_artifact(merged, rollback_mean, 2, spec, torch)
        self.assertTrue(torch.equal(
            artifact["difference"], torch.tensor([[2.0, 3.0], [4.0, 7.0]], dtype=torch.float32)
        ))
        self.assertTrue(torch.equal(
            self_diff.build_artifact(merged, merged.clone(), 2, spec, torch)["difference"],
            torch.zeros((2, 2), dtype=torch.float32),
        ))
        for name in ("merged_mean", "rollback_mean", "difference"):
            self.assertEqual(artifact[name].shape, (2, 2))
            self.assertEqual(artifact[name].dtype, torch.float32)
            self.assertTrue(artifact[name].is_contiguous())
        self.assertEqual(artifact["metadata"]["difference_sign"], "merged_minus_rollback")
        self.assertEqual(artifact["metadata"]["sequence_length"], 2)

    def test_raw_hashes_are_little_endian(self):
        integers = torch.tensor([[1, -2], [3, 4]], dtype=torch.int64).t()
        expected = hashlib.sha256(struct.pack("<qqqq", 1, 3, -2, 4)).hexdigest()
        self.assertEqual(self_diff.sha256_raw_int64_tensor(integers, torch), expected)
        floats = torch.tensor([[1.5, -2.25], [3.0, 4.5]], dtype=torch.float32).t()
        expected = hashlib.sha256(struct.pack("<ffff", 1.5, 3.0, -2.25, 4.5)).hexdigest()
        self.assertEqual(self_diff.sha256_raw_float32_tensor(floats, torch), expected)

    def test_output_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "self_diff.pt"
            manifest = root / "self-diff-manifest.json"
            artifact.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "already exists"):
                self_diff.validate_output_paths(artifact, manifest)
            artifact.unlink()
            manifest.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                self_diff.validate_output_paths(artifact, manifest)

    def test_deterministic_manifest_and_raw_artifact_fields(self):
        merged = {
            "file_count": 2, "total_bytes": 2,
            "files": [
                {"path": "z", "size_bytes": 1, "sha256": "2" * 64},
                {"path": "a", "size_bytes": 1, "sha256": "1" * 64},
            ],
        }
        rollback_entry = {
            "directory_name": "model", "file_count": 2, "total_bytes": 2,
            "files": [
                {"path": "z", "size_bytes": 1, "sha256": "4" * 64},
                {"path": "a", "size_bytes": 1, "sha256": "3" * 64},
            ],
        }
        raw = {"difference": "9" * 64, "rollback_mean": "8" * 64, "merged_mean": "7" * 64}
        arguments = (
            self_diff.FROZEN_SELF_DIFF_SPEC,
            "1" * 64, "2" * 64, "3" * 64, "4" * 64, "5" * 64,
            "6" * 64, "a" * 64, merged, rollback_entry, "b" * 64, raw, 4,
        )
        first = self_diff.build_manifest(*arguments)
        second = self_diff.build_manifest(
            *arguments[:8],
            {**merged, "files": list(reversed(merged["files"]))},
            {**rollback_entry, "files": list(reversed(rollback_entry["files"]))},
            "b" * 64, dict(reversed(list(raw.items()))), 4,
        )
        self.assertEqual(first, second)
        self.assertEqual(json.dumps(first, allow_nan=False), json.dumps(second, allow_nan=False))
        self.assertEqual([record["path"] for record in first["input_checkpoints"]["merged"]["files"]],
                         ["a", "z"])
        self.assertEqual(list(first["self_diff_artifact"]["raw_tensors_sha256"]),
                         ["merged_mean", "rollback_mean", "difference"])
        self.assertFalse({"timestamp", "host", "gpu"} & set(first))

    def test_static_information_firewall_and_interface(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        lowered = source.lower()
        for forbidden in (
            "base_model", "base model", "/models/base", "adapter", "lora",
            "oracle_adl.pt", "oracle-adl-manifest.json", "oracle-logit-lens.json",
            "models.lock.json", "downloaded-model-hashes.json", "merged-model-hashes.json",
            "evaluation.json", "attempt001", "attempt002", "attempt003", "attempt004",
            "bf16", "quantization", "cosine", "logit_lens", "logit lens",
        ):
            self.assertNotIn(forbidden, lowered)
        tree = ast.parse(source)
        environment_reads = set()
        cli_options = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (node.func.attr == "get" and isinstance(node.func.value, ast.Attribute) and
                    isinstance(node.func.value.value, ast.Name) and
                    node.func.value.value.id == "os" and node.func.value.attr == "environ" and
                    node.args and isinstance(node.args[0], ast.Constant)):
                    environment_reads.add(node.args[0].value)
                if node.func.attr == "add_argument":
                    for argument in node.args:
                        if (isinstance(argument, ast.Constant) and isinstance(argument.value, str) and
                            argument.value.startswith("--")):
                            cli_options.add(argument.value)
        self.assertEqual(environment_reads, self_diff.ALLOWED_ENVIRONMENT_VARIABLES)
        self.assertEqual(cli_options, {
            "--merged-model-dir", "--attempt-spec-path", "--corpus-spec-path",
            "--corpus-manifest-path", "--construction-manifest-path", "--probe-path",
            "--self-diff-spec-path", "--artifact-path", "--manifest-path",
        })
        for value in environment_reads | cli_options:
            self.assertNotIn("base", value.lower())
            self.assertNotIn("adapter", value.lower())
            self.assertNotIn("oracle", value.lower())


if __name__ == "__main__":
    unittest.main()
