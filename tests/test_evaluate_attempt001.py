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
SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "evaluate_attempt001.py"
SPEC_PATH = (
    PROJECT
    / "experiments"
    / "attempts"
    / "001_spectral_tail_l13_p25"
    / "evaluation_spec.json"
)

MODULE_SPEC = importlib.util.spec_from_file_location("evaluate_attempt001", SCRIPT_PATH)
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


def result_record(epsilon: float, positions) -> dict[str, object]:
    return {
        "epsilon": epsilon,
        "positions": [
            {
                "position": position,
                "geometry": {
                    "cosine_similarity": 1.0,
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
        "attempt": {
            "attempt_spec_sha256": "3" * 64,
            "construction_manifest_sha256": "4" * 64,
            "self_diff_spec_sha256": "5" * 64,
            "self_diff_manifest_sha256": "6" * 64,
            "merged_checkpoint": checkpoint,
        },
        "oracle": {
            "oracle_manifest_sha256": "7" * 64,
            "oracle_logit_lens_sha256": "8" * 64,
            "merged_model_hashes_sha256": "9" * 64,
            "downloaded_model_hashes_sha256": "b" * 64,
            "base": {"repo_id": "base", "revision": "base-revision"},
            "fine_tuned": {"repo_id": "ft", "revision": "ft-revision"},
            "base_tokenizer_source": {
                "repo_id": "base",
                "revision": "base-revision",
            },
        },
    }


class Attempt001EvaluationTests(unittest.TestCase):
    def test_syntax_import_and_frozen_spec(self) -> None:
        ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            evaluation.load_evaluation_spec(SPEC_PATH),
            evaluation.FROZEN_EVALUATION_SPEC,
        )
        self.assertEqual(
            evaluation.FROZEN_EVALUATION_SPEC["epsilons"],
            [0.05, 0.1, 0.2],
        )
        self.assertEqual(
            evaluation.FROZEN_EVALUATION_SPEC["positions"],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            evaluation.FROZEN_EVALUATION_SPEC["logit_lens"]["top_k"], 20
        )

    def test_cosine_sign_and_norm_calculations(self) -> None:
        self_vector = torch.tensor([3.0, 4.0], dtype=torch.float32)
        oracle_vector = torch.tensor([6.0, 8.0], dtype=torch.float32)
        metrics = evaluation.geometry_metrics(self_vector, oracle_vector, torch)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)
        self.assertAlmostEqual(metrics["self_diff_norm"], 5.0)
        self.assertAlmostEqual(metrics["oracle_diff_norm"], 10.0)
        self.assertAlmostEqual(metrics["self_to_oracle_norm_ratio"], 0.5)

        opposite = evaluation.geometry_metrics(
            self_vector, -oracle_vector, torch
        )
        self.assertAlmostEqual(opposite["cosine_similarity"], -1.0)

        orthogonal = evaluation.geometry_metrics(
            torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]), torch
        )
        self.assertEqual(orthogonal["cosine_similarity"], 0.0)

    def test_zero_vector_handling_is_json_safe(self) -> None:
        zero = torch.zeros(2, dtype=torch.float32)
        nonzero = torch.tensor([1.0, 0.0], dtype=torch.float32)

        self_zero = evaluation.geometry_metrics(zero, nonzero, torch)
        self.assertIsNone(self_zero["cosine_similarity"])
        self.assertEqual(self_zero["self_to_oracle_norm_ratio"], 0.0)

        oracle_zero = evaluation.geometry_metrics(nonzero, zero, torch)
        self.assertIsNone(oracle_zero["cosine_similarity"])
        self.assertIsNone(oracle_zero["self_to_oracle_norm_ratio"])
        json.dumps(oracle_zero, allow_nan=False)

    def test_mocked_logit_lens_uses_norm_then_signed_head_input(self) -> None:
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
        normed = torch.tensor([2.0, 3.0])
        self.assertTrue(torch.equal(positive, torch.softmax(head(normed), dim=-1)))
        self.assertTrue(torch.equal(negative, torch.softmax(head(-normed), dim=-1)))
        self.assertFalse(
            torch.equal(negative, torch.softmax(head(ShiftNorm()(-latent)), dim=-1))
        )

    def test_top_k_has_stable_order_and_token_fields(self) -> None:
        probabilities = torch.tensor([0.4, 0.4, 0.1, 0.1], dtype=torch.float32)
        records = evaluation.top_token_records(
            probabilities, MockTokenizer(), 3, torch
        )
        self.assertEqual([record["token_id"] for record in records], [0, 1, 2])
        self.assertEqual(records[0]["token"], "token-0")
        self.assertEqual(records[0]["decoded"], "decoded-0")
        self.assertEqual(records[0]["probability"], probabilities[0].item())

    def test_exact_coverage_and_deterministic_ordering(self) -> None:
        spec = evaluation.FROZEN_EVALUATION_SPEC
        reversed_positions = list(reversed(spec["positions"]))
        shuffled = [
            result_record(epsilon, reversed_positions)
            for epsilon in reversed(spec["epsilons"])
        ]
        ordered = evaluation.ordered_results(shuffled, spec)
        self.assertEqual(
            [record["epsilon"] for record in ordered], spec["epsilons"]
        )
        for epsilon_result in ordered:
            self.assertEqual(
                [record["position"] for record in epsilon_result["positions"]],
                spec["positions"],
            )

        second = evaluation.ordered_results(copy.deepcopy(shuffled), spec)
        self.assertEqual(
            json.dumps(ordered, ensure_ascii=False, allow_nan=False),
            json.dumps(second, ensure_ascii=False, allow_nan=False),
        )

        first_output = evaluation.build_evaluation_output(
            shuffled,
            spec,
            synthetic_context(),
            evaluation_spec_sha256="c" * 64,
            evaluator_script_sha256="d" * 64,
        )
        second_output = evaluation.build_evaluation_output(
            copy.deepcopy(shuffled),
            spec,
            synthetic_context(),
            evaluation_spec_sha256="c" * 64,
            evaluator_script_sha256="d" * 64,
        )
        self.assertEqual(
            json.dumps(first_output, ensure_ascii=False, allow_nan=False),
            json.dumps(second_output, ensure_ascii=False, allow_nan=False),
        )
        self.assertEqual(
            first_output["provenance"]["oracle_logit_lens"]["sha256"],
            "8" * 64,
        )
        self.assertFalse(
            first_output["method"]["select_or_rank_best_epsilon"]
        )
        self.assertNotIn("best_epsilon", first_output)

        missing = copy.deepcopy(shuffled)
        missing[0]["positions"].pop()
        with self.assertRaisesRegex(ValueError, "position coverage mismatch"):
            evaluation.ordered_results(missing, spec)

        missing_epsilon = copy.deepcopy(shuffled[:-1])
        with self.assertRaisesRegex(ValueError, "epsilon count mismatch"):
            evaluation.ordered_results(missing_epsilon, spec)

    def test_evaluate_vectors_preserves_all_epsilons_independently(self) -> None:
        spec = copy.deepcopy(evaluation.FROZEN_EVALUATION_SPEC)
        spec["logit_lens"]["top_k"] = 20
        differences = {}
        for index, epsilon in enumerate(spec["epsilons"], start=1):
            differences[f"{epsilon:.2f}"] = {
                "difference": torch.full(
                    (5, 2), float(index), dtype=torch.float32
                )
            }
        self_diff = {
            "metadata": {"hidden_size": 2},
            "ablations": differences,
        }
        oracle = {"difference": torch.ones((5, 2), dtype=torch.float32)}
        head = torch.nn.Linear(2, 32, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(32, 2))
        results = evaluation.evaluate_vectors(
            self_diff,
            oracle,
            torch.nn.Identity(),
            head,
            MockTokenizer(),
            32,
            spec,
            torch,
        )
        self.assertEqual(
            [record["epsilon"] for record in results], [0.05, 0.1, 0.2]
        )
        self.assertTrue(
            all(len(record["positions"]) == 5 for record in results)
        )
        self.assertNotIn("best_epsilon", json.dumps(results))

    def test_artifact_and_provenance_mismatch_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.pt"
            path.write_bytes(b"frozen")
            digest = evaluation.sha256_file(path)
            self.assertEqual(
                evaluation.verify_file_sha256(path, digest, "artifact"), digest
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                evaluation.verify_file_sha256(path, "0" * 64, "artifact")

            manifest = {"artifact_sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                evaluation.verify_manifest_file_link(
                    manifest, "artifact_sha256", path, "manifest artifact"
                )

            tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
            raw_hash = evaluation.sha256_raw_float32_tensor(tensor, torch)
            evaluation.validate_activation_tensor(
                tensor, (1, 2), "synthetic", raw_hash, torch
            )
            with self.assertRaisesRegex(ValueError, "raw tensor SHA-256 mismatch"):
                evaluation.validate_activation_tensor(
                    tensor, (1, 2), "synthetic", "0" * 64, torch
                )

    def test_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evaluation.json"
            evaluation.write_output({"value": 1}, output)
            original = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluation.write_output({"value": 2}, output)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
