import ast
import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT = Path(__file__).resolve().parents[1]
ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125"
SCRIPT_PATH = PROJECT / "scripts/ablation/construct_generic_gradient_rollback.py"
SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
CORPUS_SPEC_PATH = ATTEMPT_DIRECTORY / "corpus_spec.json"
MODULE_SPEC = importlib.util.spec_from_file_location("construct_generic_gradient_rollback", SCRIPT_PATH)
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


class GenericGradientRollbackTests(unittest.TestCase):
    def test_frozen_specs_and_information_policy(self):
        spec = rollback.load_spec(SPEC_PATH)
        corpus_spec = rollback.load_corpus_spec(CORPUS_SPEC_PATH)
        self.assertEqual(spec, rollback.FROZEN_SPEC)
        self.assertEqual(corpus_spec, rollback.FROZEN_CORPUS_SPEC)
        self.assertEqual(spec["information_policy"], "final_checkpoint_plus_frozen_generic_text")
        self.assertEqual(spec["eligible_tensors"]["transformer_block_indices"], list(range(14)))
        self.assertEqual(spec["generic_loss"]["total_prediction_tokens"], 4096 * 127)
        self.assertEqual(spec["generic_loss"]["number_of_batches"], 512)
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

    def test_one_global_alpha_minus_gradient_and_fp32_realized_metrics(self):
        eligible = two_matrix_case()
        originals = {name: module.weight.detach().clone() for name, module in eligible}
        grads = {name: module.weight.grad.detach().clone() for name, module in eligible}
        scale = rollback.global_rollback_scale(eligible, 0.00125, torch)
        expected_wnorm = math.sqrt(sum(float((weight.double() ** 2).sum()) for weight in originals.values()))
        expected_gnorm = math.sqrt(sum(float((gradient.double() ** 2).sum()) for gradient in grads.values()))
        self.assertAlmostEqual(scale["alpha"], 0.00125 * expected_wnorm / expected_gnorm)
        records, realized = rollback.apply_global_rollback(
            eligible, scale["alpha"], scale["aggregate_source_weight_norm"], torch
        )
        self.assertEqual([record["name"] for record in records], ["a_matrix", "z_matrix"])
        for name, module in eligible:
            expected = (originals[name].double() - scale["alpha"] * grads[name].double()).float()
            self.assertTrue(torch.equal(module.weight, expected))
            self.assertTrue(torch.all(module.weight < originals[name]))
            record = next(item for item in records if item["name"] == name)
            actual_norm = torch.linalg.vector_norm((module.weight - originals[name]).double()).item()
            self.assertAlmostEqual(record["realized_fp32_delta_frobenius_norm"], actual_norm)
            self.assertEqual(record["changed_element_count"], 4)
        first_delta = torch.linalg.vector_norm((eligible[1][1].weight - originals["a_matrix"]).double()).item()
        second_delta = torch.linalg.vector_norm((eligible[0][1].weight - originals["z_matrix"]).double()).item()
        self.assertAlmostEqual(second_delta / first_delta, 3.0, places=3)
        self.assertAlmostEqual(realized["aggregate_realized_relative_perturbation"], 0.00125, places=7)

    def test_zero_missing_and_nonfinite_gradients_fail(self):
        eligible = two_matrix_case()
        for _, module in eligible:
            module.weight.grad.zero_()
        with self.assertRaisesRegex(ValueError, "zero or non-finite"):
            rollback.global_rollback_scale(eligible, 0.00125, torch)
        eligible[0][1].weight.grad = None
        with self.assertRaisesRegex(ValueError, "Missing or non-finite"):
            rollback.global_rollback_scale(eligible, 0.00125, torch)
        eligible[0][1].weight.grad = torch.full_like(eligible[0][1].weight, float("nan"))
        with self.assertRaisesRegex(ValueError, "Missing or non-finite"):
            rollback.global_rollback_scale(eligible, 0.00125, torch)

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
        scale = rollback.global_rollback_scale(eligible, 0.00125, torch)
        matrices, realized = rollback.apply_global_rollback(
            eligible, scale["alpha"], scale["aggregate_source_weight_norm"], torch
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
        self.assertEqual(json.dumps(manifest, allow_nan=False), json.dumps(copy.deepcopy(manifest), allow_nan=False))
        self.assertFalse({"timestamp", "host", "gpu"} & set(manifest))

    def test_static_information_boundary_and_no_optimizer(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for forbidden in (
            "oracle", "models/base", "adapter", "models.lock.json",
            "downloaded-model-hashes.json", "merged-model-hashes.json",
            "evaluation.json", "bf16", "quantization",
            "attempts/001_", "attempts/002_", "attempts/003_", "attempts/004_",
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
