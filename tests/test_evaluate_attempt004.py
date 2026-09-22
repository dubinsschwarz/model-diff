import ast
import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "evaluate_attempt004.py"
SPEC_PATH = (
    PROJECT
    / "experiments"
    / "attempts"
    / "004_quantize11_prefix0_13"
    / "evaluation_spec.json"
)
ATTEMPT_DIRECTORY = SPEC_PATH.parent

MODULE_SPEC = importlib.util.spec_from_file_location("evaluate_attempt004", SCRIPT_PATH)
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
        "attempt_id": "004_quantize11_prefix0_13",
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
        "merged_mean_raw_sha256_gate": "e" * 64,
        "attempt": {
            "attempt_spec_sha256": "3" * 64,
            "calibration_sha256": "0" * 64,
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

def load_synthetic_logit_lens(weights, *, tied=True):
    """Exercise the loader with a small in-memory safetensors representation."""
    class FakeNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, value):
            return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight

    class FakeSafeOpen:
        def __init__(self, path, *, framework, device):
            assert path == Path("/synthetic/merged/model.safetensors")
            assert (framework, device) == ("pt", "cpu")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def keys(self):
            return weights.keys()

        def get_tensor(self, key):
            return weights[key]

    config = types.SimpleNamespace(
        model_type="qwen3", num_hidden_layers=28, hidden_size=2,
        vocab_size=32, tie_word_embeddings=tied, rms_norm_eps=1e-6,
    )
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = types.SimpleNamespace(from_pretrained=lambda path, **kwargs: config)
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda path, **kwargs: MockTokenizer())
    modeling = types.ModuleType("transformers.models.qwen3.modeling_qwen3")
    modeling.Qwen3RMSNorm = FakeNorm
    modules = {
        "safetensors": types.SimpleNamespace(safe_open=FakeSafeOpen),
        "transformers": transformers,
        "transformers.models": types.ModuleType("transformers.models"),
        "transformers.models.qwen3": types.ModuleType("transformers.models.qwen3"),
        "transformers.models.qwen3.modeling_qwen3": modeling,
    }
    with patch.dict(sys.modules, modules):
        return evaluation.load_local_logit_lens_components(
            Path("/synthetic/merged"), Path("/synthetic/base"), 2, torch
        )


class Attempt004EvaluationTests(unittest.TestCase):
    @staticmethod
    def tied_weights():
        return {
            "model.norm.weight": torch.tensor([1.25, 0.75], dtype=torch.float32),
            "model.embed_tokens.weight": torch.arange(64, dtype=torch.float32).reshape(32, 2) / 64,
        }

    def test_tied_loader_accepts_canonical_config_and_embedding_only(self) -> None:
        weights = self.tied_weights()
        tokenizer, final_norm, lm_head, vocab_size = load_synthetic_logit_lens(weights)
        self.assertIsInstance(tokenizer, MockTokenizer)
        self.assertEqual(vocab_size, 32)
        self.assertEqual(final_norm.eps, 1e-6)
        self.assertTrue(torch.equal(final_norm.weight, weights["model.norm.weight"]))
        self.assertTrue(torch.equal(lm_head.weight, weights["model.embed_tokens.weight"]))
        self.assertIsNone(lm_head.bias)

    def test_tied_loader_rejects_untied_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            load_synthetic_logit_lens(self.tied_weights(), tied=False)

    def test_tied_loader_accepts_identical_explicit_lm_head(self) -> None:
        weights = self.tied_weights()
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone()
        _, _, lm_head, _ = load_synthetic_logit_lens(weights)
        self.assertTrue(torch.equal(lm_head.weight, weights["model.embed_tokens.weight"]))

    def test_tied_loader_rejects_different_explicit_lm_head(self) -> None:
        weights = self.tied_weights()
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"].clone()
        weights["lm_head.weight"][0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "weights differ"):
            load_synthetic_logit_lens(weights)

    def test_tied_loader_rejects_invalid_output_weights(self) -> None:
        baseline = self.tied_weights()
        for invalid in (
            baseline["model.embed_tokens.weight"][:-1],
            baseline["model.embed_tokens.weight"].to(torch.float64),
            torch.full((32, 2), float("nan"), dtype=torch.float32),
        ):
            with self.subTest(shape=tuple(invalid.shape), dtype=str(invalid.dtype)):
                weights = {**baseline, "model.embed_tokens.weight": invalid}
                with self.assertRaisesRegex(ValueError, "weight shape, dtype, or finiteness mismatch"):
                    load_synthetic_logit_lens(weights)

    def test_tied_loader_fails_without_output_projection(self) -> None:
        weights = {"model.norm.weight": self.tied_weights()["model.norm.weight"]}
        with self.assertRaisesRegex(ValueError, "lacks Logit Lens weights"):
            load_synthetic_logit_lens(weights)

    def test_syntax_import_and_frozen_spec(self) -> None:
        ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            evaluation.load_evaluation_spec(SPEC_PATH),
            evaluation.FROZEN_EVALUATION_SPEC,
        )
        spec = evaluation.FROZEN_EVALUATION_SPEC
        self.assertEqual(spec["attempt_id"], "004_quantize11_prefix0_13")
        self.assertEqual(spec["positions"], [0, 1, 2, 3, 4])
        self.assertEqual(spec["logit_lens"]["top_k"], 20)
        self.assertTrue(spec["evaluation_policy"]["single_attempt_result"])
        self.assertFalse(spec["evaluation_policy"]["automatic_ranking"])
        self.assertFalse(spec["evaluation_policy"]["semantic_grading"])
        self.assertFalse(spec["evaluation_policy"]["keyword_search"])
        self.assertFalse(spec["evaluation_policy"]["best_position_selection"])
        self.assertFalse(spec["evaluation_policy"]["post_result_bit_width_tuning"])
        self.assertFalse(spec["evaluation_policy"]["attempt002_003_thresholds"])
        with tempfile.TemporaryDirectory() as temp_dir:
            modified = copy.deepcopy(spec)
            modified["positions"] = [0, 1, 2, 3]
            path = Path(temp_dir) / "evaluation_spec.json"
            path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen definition"):
                evaluation.load_evaluation_spec(path)

    def test_calibration_frozen_blind_winner_and_tamper_detection(self) -> None:
        calibration = json.loads((ATTEMPT_DIRECTORY / "calibration.json").read_text())
        attempt_spec = json.loads((ATTEMPT_DIRECTORY / "spec.json").read_text())
        winner = evaluation.validate_calibration_winner(calibration, attempt_spec)
        self.assertEqual(winner["bits"], 11)
        self.assertEqual(winner["qmax"], 1023)
        tampered = copy.deepcopy(calibration)
        tampered["selection"]["selected_bit_width"] = 12
        with self.assertRaisesRegex(ValueError, "winner"):
            evaluation.validate_calibration_winner(tampered, attempt_spec)
        tampered = copy.deepcopy(calibration)
        tampered["candidate_results"][3]["absolute_target_error"] = 1.0
        with self.assertRaisesRegex(ValueError, "candidate perturbation"):
            evaluation.validate_calibration_winner(tampered, attempt_spec)

    def test_committed_calibration_and_spec_hash_links(self) -> None:
        construction = json.loads((ATTEMPT_DIRECTORY / "construction-manifest.json").read_text())
        self_manifest = json.loads((ATTEMPT_DIRECTORY / "self-diff-manifest.json").read_text())
        for manifest, links in (
            (construction, {"spec_sha256": "spec.json", "calibration_sha256": "calibration.json"}),
            (self_manifest, {
                "attempt_spec_sha256": "spec.json",
                "calibration_sha256": "calibration.json",
                "construction_manifest_sha256": "construction-manifest.json",
                "self_diff_spec_sha256": "self_diff_spec.json",
            }),
        ):
            for key, filename in links.items():
                path = ATTEMPT_DIRECTORY / filename
                self.assertEqual(
                    evaluation.verify_manifest_file_link(manifest, key, path, filename),
                    evaluation.sha256_file(path),
                )

    def test_attempt_provenance_chain_without_loading_checkpoint(self) -> None:
        # The checkpoint bytes are deliberately out of scope for unit tests.
        with patch.object(evaluation, "verify_checkpoint", return_value={"files": []}) as checked:
            context = evaluation.validate_attempt_provenance(
                Path("/synthetic/merged"),
                evaluation.FROZEN_EVALUATION_SPEC,
                attempt_spec_path=evaluation.DEFAULT_ATTEMPT_SPEC_PATH,
                calibration_path=evaluation.DEFAULT_CALIBRATION_PATH,
                construction_manifest_path=evaluation.DEFAULT_CONSTRUCTION_MANIFEST_PATH,
                self_diff_spec_path=evaluation.DEFAULT_SELF_DIFF_SPEC_PATH,
                self_diff_manifest_path=evaluation.DEFAULT_SELF_DIFF_MANIFEST_PATH,
                constructor_script_path=evaluation.DEFAULT_CONSTRUCTOR_SCRIPT_PATH,
                self_diff_script_path=evaluation.DEFAULT_SELF_DIFF_SCRIPT_PATH,
                merged_hashes_path=evaluation.DEFAULT_MERGED_HASHES_PATH,
            )
        checked.assert_called_once()
        self.assertEqual(context["method"]["difference_sign"], "merged_minus_quantized")
        self.assertEqual(context["calibration_sha256"], evaluation.sha256_file(evaluation.DEFAULT_CALIBRATION_PATH))

    def test_cosine_sign_and_norm_math(self) -> None:
        self_vector = torch.tensor([3.0, 4.0], dtype=torch.float32)
        oracle = torch.tensor([6.0, 8.0], dtype=torch.float32)
        metrics = evaluation.geometry_metrics(self_vector, oracle, torch)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)
        self.assertAlmostEqual(metrics["self_diff_norm"], 5.0)
        self.assertAlmostEqual(metrics["oracle_diff_norm"], 10.0)
        self.assertAlmostEqual(metrics["self_to_oracle_norm_ratio"], 0.5)
        opposite = evaluation.geometry_metrics(self_vector, -oracle, torch)
        self.assertAlmostEqual(opposite["cosine_similarity"], -1.0)

    def test_zero_vector_values_are_json_safe(self) -> None:
        zero = torch.zeros(2, dtype=torch.float32)
        nonzero = torch.tensor([1.0, 0.0], dtype=torch.float32)
        self_zero = evaluation.geometry_metrics(zero, nonzero, torch)
        self.assertIsNone(self_zero["cosine_similarity"])
        self.assertEqual(self_zero["self_to_oracle_norm_ratio"], 0.0)
        oracle_zero = evaluation.geometry_metrics(nonzero, zero, torch)
        self.assertIsNone(oracle_zero["cosine_similarity"])
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
        with self.assertRaisesRegex(ValueError, "CPU"):
            evaluation.validate_activation_tensor(
                torch.empty(evaluation.EXPECTED_SHAPE, device="meta"),
                "Synthetic",
                digest,
                torch,
            )
        noncontiguous = torch.zeros((2048, 128), dtype=torch.float32).T
        with self.assertRaisesRegex(ValueError, "contiguous"):
            evaluation.validate_activation_tensor(noncontiguous, "Synthetic", digest, torch)
        nonfinite = tensor.clone()
        nonfinite[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluation.validate_activation_tensor(nonfinite, "Synthetic", digest, torch)

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
        self.assertEqual(output_one["provenance"]["attempt"]["calibration_sha256"], "0" * 64)
        self.assertEqual(output_one["provenance"]["merged_mean_raw_sha256_gate"], "e" * 64)
        lowered_keys = {str(key).lower() for key in self._all_keys(output_one)}
        self.assertFalse({"rank", "score", "grade", "winner", "best_position"} & lowered_keys)

    @staticmethod
    def _all_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from Attempt004EvaluationTests._all_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from Attempt004EvaluationTests._all_keys(child)


if __name__ == "__main__":
    unittest.main()
