import ast
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "evaluate_attempt003.py"
SPEC_PATH = (
    PROJECT
    / "experiments"
    / "attempts"
    / "003_gaussian_noise_prefix0_13_r00125"
    / "evaluation_spec.json"
)

MODULE_SPEC = importlib.util.spec_from_file_location("evaluate_attempt003", SCRIPT_PATH)
assert MODULE_SPEC is not None and MODULE_SPEC.loader is not None
evaluation = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = evaluation
MODULE_SPEC.loader.exec_module(evaluation)


class MockTokenizer:
    def __len__(self) -> int:
        return 32

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token-{token_id}"

    def decode(self, token_ids, **kwargs) -> str:
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return f"decoded-{token_ids[0]}"


def token_records(count: int = 20) -> list[dict[str, object]]:
    return [
        {
            "token_id": token_id,
            "token": f"token-{token_id}",
            "decoded": f"decoded-{token_id}",
            "probability": float(count - token_id) / count,
        }
        for token_id in range(count)
    ]


def result_record(positions=None) -> dict[str, object]:
    if positions is None:
        positions = [0, 1, 2, 3, 4]
    return {
        "attempt_id": "003_gaussian_noise_prefix0_13_r00125",
        "positions": [
            {
                "position": position,
                "geometry": {
                    "cosine_similarity": 1.0,
                    "absolute_cosine_similarity": 1.0,
                    "self_diff_norm": 2.0,
                    "oracle_diff_norm": 4.0,
                    "self_to_oracle_norm_ratio": 0.5,
                },
                "logit_lens": {
                    "positive": token_records(),
                    "negative": token_records(),
                },
            }
            for position in positions
        ],
    }


def synthetic_context() -> dict[str, object]:
    checkpoint = {
        "file_count": 1,
        "total_bytes": 1,
        "files": [{"path": "model", "size_bytes": 1, "sha256": "a" * 64}],
    }
    return {
        "self_diff_artifact_sha256": "1" * 64,
        "oracle_artifact_sha256": "2" * 64,
        "merged_mean_raw_sha256_gate": "e" * 64,
        "attempt": {
            "attempt_spec_sha256": "3" * 64,
            "construction_manifest_sha256": "4" * 64,
            "constructor_script_sha256": "5" * 64,
            "self_diff_spec_sha256": "6" * 64,
            "self_diff_manifest_sha256": "7" * 64,
            "self_diff_script_sha256": "8" * 64,
            "merged_checkpoint": checkpoint,
        },
        "oracle": {
            "oracle_manifest_sha256": "9" * 64,
            "oracle_logit_lens_sha256": "b" * 64,
            "merged_model_hashes_sha256": "c" * 64,
            "downloaded_model_hashes_sha256": "d" * 64,
            "oracle_probe_manifest_sha256": "f" * 64,
            "base": {"repo_id": "base", "revision": "base-revision"},
            "fine_tuned": {"repo_id": "ft", "revision": "ft-revision"},
            "base_tokenizer_source": {
                "repo_id": "base",
                "revision": "base-revision",
            },
        },
    }


class Attempt003EvaluationTests(unittest.TestCase):
    def test_syntax_import_and_frozen_spec(self) -> None:
        ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            evaluation.load_evaluation_spec(SPEC_PATH),
            evaluation.FROZEN_EVALUATION_SPEC,
        )
        spec = evaluation.FROZEN_EVALUATION_SPEC
        self.assertEqual(spec["attempt_id"], "003_gaussian_noise_prefix0_13_r00125")
        self.assertEqual(spec["positions"], [0, 1, 2, 3, 4])
        self.assertEqual(
            spec["primary_metric"],
            "signed_cosine_similarity_self_difference_oracle_difference",
        )
        self.assertEqual(
            spec["secondary_metric"],
            "absolute_cosine_similarity_self_difference_oracle_difference",
        )
        self.assertEqual(spec["logit_lens"]["top_k"], 20)
        self.assertTrue(spec["evaluation_policy"]["single_attempt_result"])
        self.assertTrue(spec["evaluation_policy"]["signed_cosine_remains_primary"])
        self.assertTrue(
            spec["evaluation_policy"]["absolute_cosine_is_preregistered_secondary"]
        )
        self.assertFalse(spec["evaluation_policy"]["automatic_ranking"])
        self.assertFalse(spec["evaluation_policy"]["semantic_grading"])
        self.assertFalse(spec["evaluation_policy"]["best_position_selection"])
        self.assertFalse(spec["evaluation_policy"]["post_result_metric_selection"])
        self.assertFalse(spec["evaluation_policy"]["attempt002_comparison"])
        self.assertFalse(
            spec["evaluation_policy"]["gaussian_seed_selection_or_tuning"]
        )

    def test_cosine_sign_and_norm_math(self) -> None:
        self_vector = torch.tensor([3.0, 4.0], dtype=torch.float32)
        oracle = torch.tensor([6.0, 8.0], dtype=torch.float32)
        metrics = evaluation.geometry_metrics(self_vector, oracle, torch)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)
        self.assertAlmostEqual(metrics["absolute_cosine_similarity"], 1.0)
        self.assertAlmostEqual(metrics["self_diff_norm"], 5.0)
        self.assertAlmostEqual(metrics["oracle_diff_norm"], 10.0)
        self.assertAlmostEqual(metrics["self_to_oracle_norm_ratio"], 0.5)
        opposite = evaluation.geometry_metrics(self_vector, -oracle, torch)
        self.assertAlmostEqual(opposite["cosine_similarity"], -1.0)
        self.assertAlmostEqual(opposite["absolute_cosine_similarity"], 1.0)
        orthogonal = evaluation.geometry_metrics(
            torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), torch
        )
        self.assertAlmostEqual(orthogonal["cosine_similarity"], 0.0)
        self.assertAlmostEqual(orthogonal["absolute_cosine_similarity"], 0.0)

    def test_zero_vector_values_are_json_safe(self) -> None:
        zero = torch.zeros(2, dtype=torch.float32)
        nonzero = torch.tensor([1.0, 0.0], dtype=torch.float32)
        self_zero = evaluation.geometry_metrics(zero, nonzero, torch)
        self.assertIsNone(self_zero["cosine_similarity"])
        self.assertIsNone(self_zero["absolute_cosine_similarity"])
        self.assertEqual(self_zero["self_to_oracle_norm_ratio"], 0.0)
        oracle_zero = evaluation.geometry_metrics(nonzero, zero, torch)
        self.assertIsNone(oracle_zero["cosine_similarity"])
        self.assertIsNone(oracle_zero["absolute_cosine_similarity"])
        self.assertIsNone(oracle_zero["self_to_oracle_norm_ratio"])
        json.dumps(oracle_zero, allow_nan=False)

    def test_mocked_logit_lens_uses_exact_signed_normed_convention(self) -> None:
        class ShiftNorm(torch.nn.Module):
            def forward(self, value):
                return value + 1.0

        head = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            head.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
                    dtype=torch.float32,
                )
            )
        latent = torch.tensor([1.0, 2.0], dtype=torch.float32)
        positive, negative = evaluation.logit_lens_probabilities(
            latent, ShiftNorm(), head, torch
        )
        normed = torch.tensor([2.0, 3.0], dtype=torch.float32)
        self.assertTrue(torch.equal(positive, torch.softmax(head(normed), dim=-1)))
        self.assertTrue(torch.equal(negative, torch.softmax(head(-normed), dim=-1)))
        self.assertFalse(
            torch.equal(negative, torch.softmax(head(ShiftNorm()(-latent)), dim=-1))
        )

    def test_top_k_ties_are_stable_and_token_fields_are_exact(self) -> None:
        probabilities = torch.tensor([0.4, 0.4, 0.1, 0.1], dtype=torch.float32)
        records = evaluation.top_token_records(
            probabilities, MockTokenizer(), 3, torch
        )
        self.assertEqual([record["token_id"] for record in records], [0, 1, 2])
        self.assertEqual(
            list(records[0]), ["token_id", "token", "decoded", "probability"]
        )
        self.assertEqual(records[0]["token"], "token-0")
        self.assertEqual(records[0]["decoded"], "decoded-0")

    def test_exact_position_coverage_and_deterministic_order(self) -> None:
        spec = evaluation.FROZEN_EVALUATION_SPEC
        shuffled = result_record(list(reversed(spec["positions"])))
        ordered = evaluation.ordered_result(shuffled, spec)
        self.assertEqual(
            [entry["position"] for entry in ordered["positions"]],
            spec["positions"],
        )
        self.assertEqual(
            json.dumps(ordered, allow_nan=False),
            json.dumps(evaluation.ordered_result(copy.deepcopy(shuffled), spec), allow_nan=False),
        )
        missing = copy.deepcopy(shuffled)
        missing["positions"].pop()
        with self.assertRaisesRegex(ValueError, "position coverage mismatch"):
            evaluation.ordered_result(missing, spec)

    def test_evaluate_vectors_returns_one_attempt_result(self) -> None:
        spec = copy.deepcopy(evaluation.FROZEN_EVALUATION_SPEC)
        self_diff = {
            "metadata": {"hidden_size": 2},
            "difference": torch.arange(1, 11, dtype=torch.float32).reshape(5, 2),
        }
        oracle = {"difference": self_diff["difference"].clone()}
        norm = torch.nn.Identity()
        head = torch.nn.Linear(2, 32, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(32, 2) / 64)
        result = evaluation.evaluate_vectors(
            self_diff,
            oracle,
            norm,
            head,
            MockTokenizer(),
            32,
            spec,
            torch,
        )
        self.assertEqual(set(result), {"attempt_id", "positions"})
        self.assertEqual(len(result["positions"]), 5)
        for position in result["positions"]:
            self.assertIn("cosine_similarity", position["geometry"])
            self.assertIn("absolute_cosine_similarity", position["geometry"])
            signed = position["geometry"]["cosine_similarity"]
            absolute = position["geometry"]["absolute_cosine_similarity"]
            self.assertEqual(absolute, abs(signed))
        self.assertEqual(len(result["positions"][0]["logit_lens"]["positive"]), 20)

    def test_self_diff_artifact_sha256_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "self_diff.pt"
            path.write_bytes(b"synthetic-self-diff")
            digest = evaluation.sha256_file(path)
            self.assertEqual(
                evaluation.verify_file_sha256(path, digest, "self-difference artifact"),
                digest,
            )
            with self.assertRaisesRegex(ValueError, "self-difference artifact SHA-256 mismatch"):
                evaluation.verify_file_sha256(path, "0" * 64, "self-difference artifact")

    def test_oracle_artifact_sha256_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "oracle.pt"
            path.write_bytes(b"synthetic-oracle")
            digest = evaluation.sha256_file(path)
            self.assertEqual(
                evaluation.verify_file_sha256(path, digest, "oracle artifact"), digest
            )
            with self.assertRaisesRegex(ValueError, "oracle artifact SHA-256 mismatch"):
                evaluation.verify_file_sha256(path, "0" * 64, "oracle artifact")

    def test_raw_tensor_hash_and_shape_dtype_validation(self) -> None:
        tensor = torch.zeros(evaluation.EXPECTED_SHAPE, dtype=torch.float32)
        digest = evaluation.sha256_raw_float32_tensor(tensor, torch)
        self.assertIs(
            evaluation.validate_activation_tensor(tensor, "Synthetic", digest, torch),
            tensor,
        )
        changed = tensor.clone()
        changed[0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "raw tensor SHA-256 mismatch"):
            evaluation.validate_activation_tensor(changed, "Synthetic", digest, torch)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            evaluation.validate_activation_tensor(
                torch.zeros((5, 2), dtype=torch.float32), "Synthetic", digest, torch
            )
        with self.assertRaisesRegex(ValueError, "torch.float32"):
            evaluation.validate_activation_tensor(
                torch.zeros(evaluation.EXPECTED_SHAPE, dtype=torch.float64),
                "Synthetic",
                digest,
                torch,
            )
        noncontiguous = torch.zeros(
            (evaluation.EXPECTED_SHAPE[1], evaluation.EXPECTED_SHAPE[0]),
            dtype=torch.float32,
        ).transpose(0, 1)
        self.assertFalse(noncontiguous.is_contiguous())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            evaluation.validate_activation_tensor(
                noncontiguous, "Synthetic", digest, torch
            )
        nonfinite = tensor.clone()
        nonfinite[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluation.validate_activation_tensor(
                nonfinite, "Synthetic", digest, torch
            )

    def test_merged_mean_manifest_hash_gate_and_mismatch(self) -> None:
        digest = "a" * 64
        self_manifest = {
            "self_diff_artifact": {"raw_tensors_sha256": {"merged_mean": digest}}
        }
        oracle_manifest = {"raw_tensors_sha256": {"ft_mean": digest}}
        self.assertEqual(
            evaluation.verify_merged_mean_hash_gate(self_manifest, oracle_manifest),
            digest,
        )
        oracle_manifest["raw_tensors_sha256"]["ft_mean"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "Merged activation mean disagrees"):
            evaluation.verify_merged_mean_hash_gate(self_manifest, oracle_manifest)

    def test_loaded_merged_mean_gate_fails_closed(self) -> None:
        merged = torch.zeros((2, 3), dtype=torch.float32)
        digest = evaluation.sha256_raw_float32_tensor(merged, torch)
        self.assertEqual(
            evaluation.verify_loaded_merged_mean_consistency(
                {"merged_mean": merged}, {"ft_mean": merged.clone()}, digest, torch
            ),
            digest,
        )
        mismatched = merged.clone()
        mismatched[0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "disagrees"):
            evaluation.verify_loaded_merged_mean_consistency(
                {"merged_mean": merged}, {"ft_mean": mismatched}, digest, torch
            )

    def test_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evaluation.json"
            path.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluation.require_output_absent(path)
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluation.write_output({"new": True}, path)
            self.assertEqual(path.read_text(encoding="utf-8"), "existing")

    def test_output_is_deterministic_and_has_no_grading_or_ranking(self) -> None:
        spec = evaluation.FROZEN_EVALUATION_SPEC
        result = result_record(list(reversed(spec["positions"])))
        output_one = evaluation.build_evaluation_output(
            result,
            spec,
            synthetic_context(),
            evaluation_spec_sha256="1" * 64,
            evaluator_script_sha256="2" * 64,
        )
        output_two = evaluation.build_evaluation_output(
            copy.deepcopy(result),
            copy.deepcopy(spec),
            copy.deepcopy(synthetic_context()),
            evaluation_spec_sha256="1" * 64,
            evaluator_script_sha256="2" * 64,
        )
        serialized = json.dumps(output_one, ensure_ascii=False, allow_nan=False)
        self.assertEqual(serialized, json.dumps(output_two, ensure_ascii=False, allow_nan=False))
        self.assertIn("result", output_one)
        self.assertNotIn("results", output_one)
        self.assertEqual(output_one["provenance"]["oracle_logit_lens"]["sha256"], "b" * 64)
        self.assertEqual(output_one["provenance"]["merged_mean_raw_sha256_gate"], "e" * 64)
        self.assertEqual(
            output_one["method"]["primary_metric"],
            "signed_cosine_similarity_self_difference_oracle_difference",
        )
        self.assertEqual(
            output_one["method"]["secondary_metric"],
            "absolute_cosine_similarity_self_difference_oracle_difference",
        )
        negative = evaluation.geometry_metrics(
            torch.tensor([1.0, 0.0]),
            torch.tensor([-1.0, 0.0]),
            torch,
        )
        self.assertEqual(negative["cosine_similarity"], -1.0)
        self.assertEqual(negative["absolute_cosine_similarity"], 1.0)
        lowered_keys = {str(key).lower() for key in self._all_keys(output_one)}
        self.assertFalse({"rank", "score", "grade", "winner", "best_position"} & lowered_keys)

    @staticmethod
    def _all_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from Attempt003EvaluationTests._all_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from Attempt003EvaluationTests._all_keys(child)


if __name__ == "__main__":
    unittest.main()
