import ast
import copy
import importlib.util
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT / "scripts" / "ablation" / "evaluate_attempt008.py"
SPEC_PATH = (
    PROJECT
    / "experiments"
    / "attempts"
    / "008_generic_gradient_linear_response_prefix0_13_r00125"
    / "evaluation_spec.json"
)
ATTEMPT_DIRECTORY = SPEC_PATH.parent
CORPUS_DIRECTORY = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125"

MODULE_SPEC = importlib.util.spec_from_file_location("evaluate_attempt008", SCRIPT_PATH)
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
        "attempt_id": "008_generic_gradient_linear_response_prefix0_13_r00125",
        "positions": [
            {
                "position": position,
                "geometry": {
                    "cosine_similarity": 1.0,
                    "linear_response_norm": 2.0,
                    "oracle_diff_norm": 4.0,
                    "linear_response_to_oracle_norm_ratio": 0.5,
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
        "linear_response_artifact_sha256": "1" * 64,
        "oracle_artifact_sha256": "2" * 64,
        "merged_mean_raw_sha256": {"merged_mean": "e" * 64, "oracle_ft_mean": "f" * 64},
        "merged_mean_consistency": {
            "method": dict(evaluation.MERGED_MEAN_CONSISTENCY),
            "raw_tensor_sha256": {"merged_mean": "e" * 64, "oracle_ft_mean": "f" * 64},
            "positions": [{"position": 0, "cosine_similarity": 1.0,
                           "relative_rms_difference": 1e-7, "maximum_absolute_difference": 1e-6}],
        },
        "attempt": {
            "attempt_spec_sha256": "3" * 64,
            "corpus_spec_sha256": "0" * 64,
            "corpus_manifest_sha256": "a" * 64,
            "freeze_corpus_script_sha256": "f" * 64,
            "construction_manifest_sha256": "4" * 64,
            "constructor_script_sha256": "5" * 64,
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
    def load_config(path, **kwargs):
        assert path == Path("/synthetic/merged")
        assert kwargs == {"local_files_only": True}
        return config
    def load_tokenizer(path, **kwargs):
        assert path == Path("/synthetic/base")
        assert kwargs == {"local_files_only": True}
        return MockTokenizer()
    transformers.AutoConfig = types.SimpleNamespace(from_pretrained=load_config)
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=load_tokenizer)
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


class Attempt008EvaluationTests(unittest.TestCase):
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

    def test_tied_loader_accepts_explicit_head_only(self):
        weights = self.tied_weights()
        weights["lm_head.weight"] = weights.pop("model.embed_tokens.weight")
        _, _, head, _ = load_synthetic_logit_lens(weights)
        self.assertTrue(torch.equal(head.weight, weights["lm_head.weight"]))

    def test_all_frozen_provenance_files_are_pinned_and_tampering_fails(self):
        kwargs = {
            "attempt_spec_path": evaluation.DEFAULT_ATTEMPT_SPEC_PATH,
            "corpus_spec_path": evaluation.DEFAULT_CORPUS_SPEC_PATH,
            "corpus_manifest_path": evaluation.DEFAULT_CORPUS_MANIFEST_PATH,
            "construction_manifest_path": evaluation.DEFAULT_CONSTRUCTION_MANIFEST_PATH,
            "freeze_corpus_script_path": evaluation.DEFAULT_FREEZE_CORPUS_SCRIPT_PATH,
            "constructor_script_path": evaluation.DEFAULT_CONSTRUCTOR_SCRIPT_PATH,
            "merged_hashes_path": evaluation.DEFAULT_MERGED_HASHES_PATH,
        }
        with tempfile.TemporaryDirectory() as directory:
            for key, original in list(kwargs.items())[:4]:
                with self.subTest(file=original.name):
                    self.assertEqual(evaluation.sha256_file(original), evaluation.FROZEN_ATTEMPT_HASHES[original.name])
                    tampered = Path(directory) / original.name
                    tampered.write_bytes(original.read_bytes() + b"\n")
                    with patch.object(evaluation, "verify_checkpoint") as checked:
                        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                            evaluation.validate_attempt_provenance(
                                Path("/synthetic/merged"), evaluation.FROZEN_EVALUATION_SPEC,
                                **{**kwargs, key: tampered})
                        checked.assert_not_called()

    def test_existing_oracle_provenance_with_mocked_base_inventory(self):
        with patch.object(evaluation, "verify_download_artifact", return_value={}) as checked:
            context = evaluation.validate_oracle_provenance(
                Path("/synthetic/base"), evaluation.FROZEN_EVALUATION_SPEC,
                downloaded_hashes_path=evaluation.DEFAULT_DOWNLOADED_HASHES_PATH,
                merged_hashes_path=evaluation.DEFAULT_MERGED_HASHES_PATH,
                probe_manifest_path=evaluation.DEFAULT_ORACLE_PROBE_MANIFEST_PATH,
                oracle_manifest_path=evaluation.DEFAULT_ORACLE_MANIFEST_PATH,
                oracle_logit_lens_path=evaluation.DEFAULT_ORACLE_LOGIT_LENS_PATH,
                merge_script_path=evaluation.DEFAULT_MERGE_SCRIPT_PATH,
                freeze_probe_script_path=evaluation.DEFAULT_FREEZE_PROBE_SCRIPT_PATH,
                oracle_script_path=evaluation.DEFAULT_ORACLE_SCRIPT_PATH)
        checked.assert_called_once()
        self.assertEqual(context["hidden_size"], 2048)

    def test_current_frozen_mean_hashes_remain_separate(self):
        self_manifest = json.loads(evaluation.DEFAULT_CONSTRUCTION_MANIFEST_PATH.read_text())
        oracle_manifest = json.loads(evaluation.DEFAULT_ORACLE_MANIFEST_PATH.read_text())
        hashes = evaluation.verify_merged_mean_hash_metadata(self_manifest, oracle_manifest)
        self.assertEqual(hashes["merged_mean"], self_manifest["artifact"]["raw_tensors_sha256"]["merged_mean"])
        self.assertEqual(hashes["oracle_ft_mean"], oracle_manifest["raw_tensors_sha256"]["ft_mean"])

    def test_syntax_import_and_frozen_spec(self) -> None:
        ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            evaluation.load_evaluation_spec(SPEC_PATH),
            evaluation.FROZEN_EVALUATION_SPEC,
        )
        spec = evaluation.FROZEN_EVALUATION_SPEC
        self.assertEqual(spec["attempt_id"], "008_generic_gradient_linear_response_prefix0_13_r00125")
        self.assertEqual(spec["positions"], [0, 1, 2, 3, 4])
        self.assertEqual(spec["logit_lens"]["top_k"], 20)
        self.assertTrue(spec["evaluation_policy"]["single_attempt_result"])
        self.assertFalse(spec["evaluation_policy"]["automatic_ranking"])
        self.assertFalse(spec["evaluation_policy"]["semantic_grading"])
        self.assertFalse(spec["evaluation_policy"]["keyword_search"])
        self.assertFalse(spec["evaluation_policy"]["best_position_selection"])
        self.assertFalse(spec["evaluation_policy"]["post_result_strength_tuning"])
        self.assertFalse(spec["evaluation_policy"]["alternative_alpha_selection"])
        self.assertFalse(spec["evaluation_policy"]["prior_attempt_thresholds"])
        self.assertFalse(spec["evaluation_policy"]["post_result_damping_tuning"])
        self.assertFalse(spec["evaluation_policy"]["post_result_fisher_definition_tuning"])
        with tempfile.TemporaryDirectory() as temp_dir:
            modified = copy.deepcopy(spec)
            modified["positions"] = [0, 1, 2, 3]
            path = Path(temp_dir) / "evaluation_spec.json"
            path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen definition"):
                evaluation.load_evaluation_spec(path)

    def test_committed_attempt008_hash_links(self) -> None:
        construction = json.loads((ATTEMPT_DIRECTORY / "construction-manifest.json").read_text())
        corpus_manifest = json.loads((CORPUS_DIRECTORY / "corpus-manifest.json").read_text())
        for manifest, links in (
            (corpus_manifest, {
                "corpus_spec_sha256": CORPUS_DIRECTORY / "corpus_spec.json",
                "freeze_script_sha256": evaluation.DEFAULT_FREEZE_CORPUS_SCRIPT_PATH,
            }),
            (construction, {
                "attempt_spec_sha256": ATTEMPT_DIRECTORY / "spec.json",
                "corpus_spec_sha256": CORPUS_DIRECTORY / "corpus_spec.json",
                "corpus_manifest_sha256": CORPUS_DIRECTORY / "corpus-manifest.json",
                "constructor_script_sha256": evaluation.DEFAULT_CONSTRUCTOR_SCRIPT_PATH,
            }),
        ):
            for key, path in links.items():
                self.assertEqual(
                    evaluation.verify_manifest_file_link(manifest, key, path, key),
                    evaluation.sha256_file(path),
                )

    def test_attempt_provenance_chain_without_loading_checkpoint(self) -> None:
        with patch.object(evaluation, "verify_checkpoint", return_value={"files": []}) as checked:
            context = evaluation.validate_attempt_provenance(
                Path("/synthetic/merged"),
                evaluation.FROZEN_EVALUATION_SPEC,
                attempt_spec_path=evaluation.DEFAULT_ATTEMPT_SPEC_PATH,
                corpus_spec_path=evaluation.DEFAULT_CORPUS_SPEC_PATH,
                corpus_manifest_path=evaluation.DEFAULT_CORPUS_MANIFEST_PATH,
                construction_manifest_path=evaluation.DEFAULT_CONSTRUCTION_MANIFEST_PATH,
                freeze_corpus_script_path=evaluation.DEFAULT_FREEZE_CORPUS_SCRIPT_PATH,
                constructor_script_path=evaluation.DEFAULT_CONSTRUCTOR_SCRIPT_PATH,
                merged_hashes_path=evaluation.DEFAULT_MERGED_HASHES_PATH,
            )
        checked.assert_called_once()
        self.assertEqual(context["construction_spec"]["jvp"]["response_sign"], "positive_J_delta")
        self.assertEqual(context["corpus_spec_sha256"], evaluation.sha256_file(evaluation.DEFAULT_CORPUS_SPEC_PATH))
        self.assertEqual(context["corpus_manifest_sha256"], evaluation.sha256_file(evaluation.DEFAULT_CORPUS_MANIFEST_PATH))

    @staticmethod
    def validate_committed_attempt():
        return evaluation.validate_attempt_provenance(
            Path("/synthetic/merged"), evaluation.FROZEN_EVALUATION_SPEC,
            attempt_spec_path=evaluation.DEFAULT_ATTEMPT_SPEC_PATH,
            corpus_spec_path=evaluation.DEFAULT_CORPUS_SPEC_PATH,
            corpus_manifest_path=evaluation.DEFAULT_CORPUS_MANIFEST_PATH,
            construction_manifest_path=evaluation.DEFAULT_CONSTRUCTION_MANIFEST_PATH,
            freeze_corpus_script_path=evaluation.DEFAULT_FREEZE_CORPUS_SCRIPT_PATH,
            constructor_script_path=evaluation.DEFAULT_CONSTRUCTOR_SCRIPT_PATH,
            merged_hashes_path=evaluation.DEFAULT_MERGED_HASHES_PATH)

    def test_gradient_tangent_jvp_readout_semantics_are_required(self):
        read = evaluation.load_json_object
        mutations = [("generic_loss", "batch_size", 1), ("generic_loss", "total_prediction_tokens", 4096),
                     ("tangent", "direction", "delta = -alpha * g"),
                     ("tangent", "target_relative_frobenius", .0025),
                     ("jvp", "response_sign", "negative_J_delta"),
                     ("jvp", "fallback", True), ("readout", "hidden_state_index", 13),
                     ("probe", "raw_tensor_sha256", "0" * 64)]
        for section, key, replacement in mutations:
            for manifest in (False, True):
                with self.subTest(section=section, key=key, manifest=manifest):
                    def modified(path, description):
                        value = read(path, description)
                        target = evaluation.DEFAULT_CONSTRUCTION_MANIFEST_PATH if manifest else evaluation.DEFAULT_ATTEMPT_SPEC_PATH
                        if path == target:
                            value["loss" if manifest and section == "generic_loss" else section][key] = replacement
                        return value
                    with patch.object(evaluation, "load_json_object", side_effect=modified), patch.object(evaluation, "verify_checkpoint") as checked:
                        with self.assertRaises(ValueError):
                            self.validate_committed_attempt()
                        checked.assert_not_called()

    def test_merged_checkpoint_inventory_must_match_construction(self):
        read = evaluation.load_json_object
        def modified(path, description):
            value = read(path, description)
            if path == evaluation.DEFAULT_CONSTRUCTION_MANIFEST_PATH:
                value["source_checkpoint"]["files"][0]["sha256"] = "0" * 64
            return value
        with patch.object(evaluation, "load_json_object", side_effect=modified), patch.object(evaluation, "verify_checkpoint") as checked:
            with self.assertRaisesRegex(ValueError, "Merged checkpoint provenance"):
                self.validate_committed_attempt()
            checked.assert_not_called()

    def test_attempt_provenance_frozen_file_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tampered = Path(temp_dir) / "spec.json"
            tampered.write_text(evaluation.DEFAULT_ATTEMPT_SPEC_PATH.read_text().replace(
                "generic_gradient_linear_response", "changed_method"
            ))
            with self.assertRaisesRegex(ValueError, "spec.json SHA-256 mismatch"):
                evaluation.validate_attempt_provenance(
                    Path("/synthetic/merged"), evaluation.FROZEN_EVALUATION_SPEC,
                    attempt_spec_path=tampered,
                    corpus_spec_path=evaluation.DEFAULT_CORPUS_SPEC_PATH,
                    corpus_manifest_path=evaluation.DEFAULT_CORPUS_MANIFEST_PATH,
                    construction_manifest_path=evaluation.DEFAULT_CONSTRUCTION_MANIFEST_PATH,
                    freeze_corpus_script_path=evaluation.DEFAULT_FREEZE_CORPUS_SCRIPT_PATH,
                    constructor_script_path=evaluation.DEFAULT_CONSTRUCTOR_SCRIPT_PATH,
                    merged_hashes_path=evaluation.DEFAULT_MERGED_HASHES_PATH,
                )

    def test_no_prior_attempt_result_dependency(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"attempt00[1-4]|/00[1-4]_", source, re.I))
        self.assertNotIn("from evaluate_attempt", source)
        self.assertNotIn("evaluate_attempt005", source)
        self.assertNotIn("evaluate_attempt006", source)
        self.assertNotIn("attempt007", source)
        self.assertNotIn("attempts/007_", source)
        self.assertNotIn("attempt006", source)
        self.assertNotIn("attempts/006_", source)
        strings = [node.value for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        prior_references = {value for value in strings if "005" in value}
        self.assertEqual(prior_references, {
            "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125",
            "005_generic_gradient_rollback_prefix0_13_r00125",
        })
        self.assertEqual(evaluation.DEFAULT_CORPUS_SPEC_PATH, CORPUS_DIRECTORY / "corpus_spec.json")
        self.assertEqual(evaluation.DEFAULT_CORPUS_MANIFEST_PATH, CORPUS_DIRECTORY / "corpus-manifest.json")

    def test_synthetic_linear_response_raw_hashes_names_and_sign(self):
        frozen = json.loads(evaluation.DEFAULT_ATTEMPT_SPEC_PATH.read_text())
        values = {
            "merged_mean": torch.zeros(evaluation.EXPECTED_SHAPE, dtype=torch.float32),
            "linear_response": torch.ones(evaluation.EXPECTED_SHAPE, dtype=torch.float32),
        }
        hashes = {name: evaluation.sha256_raw_float32_tensor(value, torch) for name, value in values.items()}
        context = {"attempt": {"construction_spec": frozen,
            "construction_manifest": {"artifact": {"raw_tensors_sha256": hashes}}}}
        artifact = {**values, "metadata": {
            "format_version": 1, "attempt_id": frozen["attempt_id"],
            "response_sign": "positive_J_delta", "readout": frozen["readout"],
            "sample_count": 10000, "sequence_length": 128,
            "accumulator_dtype": "cpu_float64", "stored_dtype": "float32"}}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "linear_response.pt"
            torch.save(artifact, path)
            loaded = evaluation.load_and_validate_linear_response(path, context, evaluation.FROZEN_EVALUATION_SPEC, torch)
            self.assertEqual(set(loaded), {"metadata", *values})
            for name in evaluation.SELF_VECTOR_TYPES:
                changed = copy.deepcopy(artifact)
                changed[name][0, 0] += 1.
                torch.save(changed, path)
                with self.assertRaisesRegex(ValueError, "raw tensor SHA-256 mismatch"):
                    evaluation.load_and_validate_linear_response(path, context, evaluation.FROZEN_EVALUATION_SPEC, torch)
            for key, value in (("attempt_id", "wrong_attempt"), ("response_sign", "negative_J_delta")):
                changed = copy.deepcopy(artifact)
                changed["metadata"][key] = value
                torch.save(changed, path)
                with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                    evaluation.load_and_validate_linear_response(path, context, evaluation.FROZEN_EVALUATION_SPEC, torch)
            for name in ("difference", "sqrt_fisher_rollback_mean"):
                changed = copy.deepcopy(artifact)
                changed[name] = changed.pop("linear_response")
                torch.save(changed, path)
                with self.assertRaisesRegex(ValueError, "fields mismatch"):
                    evaluation.load_and_validate_linear_response(path, context, evaluation.FROZEN_EVALUATION_SPEC, torch)

    def test_merged_checkpoint_inventory_and_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            weights = root / "model.safetensors"
            weights.write_bytes(b"synthetic checkpoint")
            record = {
                "path": weights.name,
                "size_bytes": weights.stat().st_size,
                "sha256": evaluation.sha256_file(weights),
            }
            manifest = {
                "file_count": 1,
                "total_bytes": record["size_bytes"],
                "files": [record],
            }
            self.assertEqual(evaluation.verify_checkpoint(root, manifest, "merged"), manifest)
            weights.write_bytes(b"changed checkpoint")
            with self.assertRaisesRegex(ValueError, "size or SHA-256 mismatch"):
                evaluation.verify_checkpoint(root, manifest, "merged")
            (root / "unexpected.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set mismatch"):
                evaluation.verify_checkpoint(root, manifest, "merged")

    def test_cosine_sign_and_norm_math(self) -> None:
        self_vector = torch.tensor([3.0, 4.0], dtype=torch.float32)
        oracle = torch.tensor([6.0, 8.0], dtype=torch.float32)
        metrics = evaluation.geometry_metrics(self_vector, oracle, torch)
        self.assertAlmostEqual(metrics["cosine_similarity"], 1.0)
        self.assertAlmostEqual(metrics["linear_response_norm"], 5.0)
        self.assertAlmostEqual(metrics["oracle_diff_norm"], 10.0)
        self.assertAlmostEqual(metrics["linear_response_to_oracle_norm_ratio"], 0.5)
        opposite = evaluation.geometry_metrics(self_vector, -oracle, torch)
        self.assertAlmostEqual(opposite["cosine_similarity"], -1.0)

    def test_zero_vector_values_are_json_safe(self) -> None:
        zero = torch.zeros(2, dtype=torch.float32)
        nonzero = torch.tensor([1.0, 0.0], dtype=torch.float32)
        self_zero = evaluation.geometry_metrics(zero, nonzero, torch)
        self.assertIsNone(self_zero["cosine_similarity"])
        self.assertEqual(self_zero["linear_response_to_oracle_norm_ratio"], 0.0)
        oracle_zero = evaluation.geometry_metrics(nonzero, zero, torch)
        self.assertIsNone(oracle_zero["cosine_similarity"])
        self.assertIsNone(oracle_zero["linear_response_to_oracle_norm_ratio"])
        both_zero = evaluation.geometry_metrics(zero, zero, torch)
        self.assertIsNone(both_zero["cosine_similarity"])
        self.assertIsNone(both_zero["linear_response_to_oracle_norm_ratio"])
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
        for invalid_positions in ([0, 1, 2, 3, 3], [0, 1, 2, 3, 5]):
            with self.assertRaises(ValueError):
                evaluation.ordered_result(result_record(invalid_positions), spec)

    def test_evaluate_vectors_returns_one_attempt_result(self) -> None:
        spec = copy.deepcopy(evaluation.FROZEN_EVALUATION_SPEC)
        linear_response = {
            "metadata": {"readout": {"hidden_size": 2}},
            "linear_response": torch.arange(1, 11, dtype=torch.float32).reshape(5, 2),
        }
        oracle = {"difference": linear_response["linear_response"].clone()}
        norm = torch.nn.Identity()
        head = torch.nn.Linear(2, 32, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(32, 2) / 64)
        result = evaluation.evaluate_vectors(
            linear_response,
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
        self.assertEqual([row["position"] for row in result["positions"]], [0, 1, 2, 3, 4])
        for row in result["positions"]:
            self.assertAlmostEqual(row["geometry"]["cosine_similarity"], 1.0)
            self.assertAlmostEqual(row["geometry"]["linear_response_to_oracle_norm_ratio"], 1.0)
            vector = linear_response["linear_response"][row["position"]]
            positive, negative = evaluation.logit_lens_probabilities(vector, norm, head, torch)
            self.assertEqual(row["logit_lens"]["positive"], evaluation.top_token_records(positive, MockTokenizer(), 20, torch))
            self.assertEqual(row["logit_lens"]["negative"], evaluation.top_token_records(negative, MockTokenizer(), 20, torch))

    def test_input_validation_checks_serialized_response_before_loading(self):
        construction = json.loads(evaluation.DEFAULT_CONSTRUCTION_MANIFEST_PATH.read_text())
        oracle_manifest = json.loads(evaluation.DEFAULT_ORACLE_MANIFEST_PATH.read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response_path, oracle_path = root / "response.pt", root / "oracle.pt"
            response_path.write_bytes(b"synthetic response")
            oracle_path.write_bytes(b"synthetic oracle")
            construction["artifact"]["serialized_sha256"] = evaluation.sha256_file(response_path)
            oracle_manifest["oracle_adl_sha256"] = evaluation.sha256_file(oracle_path)
            attempt = {"merged_model_hashes_sha256": "a" * 64, "hidden_size": 2048,
                       "construction_manifest": construction}
            oracle = {"merged_model_hashes_sha256": "a" * 64, "hidden_size": 2048,
                      "oracle_manifest": oracle_manifest}
            with patch.object(evaluation, "validate_attempt_provenance", return_value=attempt), patch.object(evaluation, "validate_oracle_provenance", return_value=oracle), patch.object(evaluation, "load_torch_artifact") as loader:
                context = evaluation.verify_inputs_before_loading(root, root, response_path, oracle_path, evaluation.FROZEN_EVALUATION_SPEC)
                self.assertEqual(context["linear_response_artifact_sha256"], evaluation.sha256_file(response_path))
                response_path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "linear-response artifact SHA-256 mismatch"):
                    evaluation.verify_inputs_before_loading(root, root, response_path, oracle_path, evaluation.FROZEN_EVALUATION_SPEC)
                loader.assert_not_called()

    def test_linear_response_artifact_sha256_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "linear_response.pt"
            path.write_bytes(b"synthetic-linear-response")
            digest = evaluation.sha256_file(path)
            self.assertEqual(
                evaluation.verify_file_sha256(path, digest, "linear-response artifact"),
                digest,
            )
            with self.assertRaisesRegex(ValueError, "linear-response artifact SHA-256 mismatch"):
                evaluation.verify_file_sha256(path, "0" * 64, "linear-response artifact")

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

    def test_merged_mean_hash_metadata_accepts_distinct_valid_hashes(self):
        self_manifest = {"artifact": {"raw_tensors_sha256": {"merged_mean": "a" * 64}}}
        oracle_manifest = {"raw_tensors_sha256": {"ft_mean": "b" * 64}}
        self.assertEqual(evaluation.verify_merged_mean_hash_metadata(self_manifest, oracle_manifest),
                         {"merged_mean": "a" * 64, "oracle_ft_mean": "b" * 64})
        oracle_manifest["raw_tensors_sha256"]["ft_mean"] = "invalid"
        with self.assertRaises(ValueError):
            evaluation.verify_merged_mean_hash_metadata(self_manifest, oracle_manifest)

    def check_mean_pair(self, merged, reference):
        hashes = {"merged_mean": evaluation.sha256_raw_float32_tensor(merged, torch),
                  "oracle_ft_mean": evaluation.sha256_raw_float32_tensor(reference, torch)}
        return evaluation.verify_loaded_merged_mean_consistency(
            {"merged_mean": merged}, {"ft_mean": reference}, hashes, torch)

    def test_numerical_mean_gate_exact_equality_and_zero_handling(self):
        for value in (0., 1.):
            means = torch.full(evaluation.EXPECTED_SHAPE, value)
            report = self.check_mean_pair(means, means.clone())
            self.assertEqual(len(report["positions"]), 128)
            for record in report["positions"]:
                self.assertAlmostEqual(record["cosine_similarity"], 1.)
                self.assertEqual(record["relative_rms_difference"], 0.)
                self.assertEqual(record["maximum_absolute_difference"], 0.)
        zero, one = torch.zeros(evaluation.EXPECTED_SHAPE), torch.ones(evaluation.EXPECTED_SHAPE)
        for left, right in ((zero, one), (one, zero)):
            with self.assertRaisesRegex(ValueError, "cosine_similarity"):
                self.check_mean_pair(left, right)

    def test_tiny_hardware_style_noise_passes_with_deterministic_metrics(self):
        reference = torch.full(evaluation.EXPECTED_SHAPE, 10.)
        reference[0].fill_(1000.)
        merged = reference.clone()
        merged[0, ::2] += .00195
        merged[1:, ::2] += 1e-6
        report = self.check_mean_pair(merged, reference)
        self.assertNotEqual(report["raw_tensor_sha256"]["merged_mean"], report["raw_tensor_sha256"]["oracle_ft_mean"])
        self.assertEqual([r["position"] for r in report["positions"]], list(range(128)))
        delta = merged.double() - reference.double()
        for record in report["positions"]:
            index = record["position"]
            expected_rms = delta[index].square().mean().sqrt().item()
            expected_reference = reference[index].double().square().mean().sqrt().item()
            self.assertEqual(record["rms_difference"], expected_rms)
            self.assertEqual(record["relative_rms_difference"], expected_rms / expected_reference)
            self.assertEqual(record["maximum_absolute_difference"], delta[index].abs().max().item())
        context = synthetic_context()
        context["merged_mean_raw_sha256"] = report["raw_tensor_sha256"]
        context["merged_mean_consistency"] = report
        def output():
            return evaluation.build_evaluation_output(result_record(), evaluation.FROZEN_EVALUATION_SPEC,
                context, evaluation_spec_sha256="1"*64, evaluator_script_sha256="2"*64)
        self.assertEqual(output()["provenance"]["merged_mean_consistency"], report)
        self.assertEqual(json.dumps(output(), allow_nan=False), json.dumps(output(), allow_nan=False))

    def test_all_three_numerical_thresholds_are_independently_required(self):
        passing = dict(position=0, cosine_similarity=1., relative_rms_difference=0., maximum_absolute_difference=0.)
        for key, bad in (("cosine_similarity", .999999998),
                         ("relative_rms_difference", 1.0001e-5),
                         ("maximum_absolute_difference", .010001)):
            with self.subTest(metric=key), self.assertRaisesRegex(ValueError, key):
                evaluation.require_merged_mean_equivalence({**passing, key: bad})
        evaluation.require_merged_mean_equivalence(dict(position=0, cosine_similarity=.999999999,
            relative_rms_difference=1e-5, maximum_absolute_difference=.01))
        for key in ("cosine_similarity", "relative_rms_difference", "maximum_absolute_difference"):
            with self.assertRaises(ValueError):
                evaluation.require_merged_mean_equivalence({**passing, key: float("nan")})

    def test_numerical_tensor_failures_including_unevaluated_position(self):
        reference = torch.ones(evaluation.EXPECTED_SHAPE)
        # Cosine failure necessarily violates the tighter relative-RMS bound too;
        # the independent predicate checks above prove neither check is omitted.
        merged = reference.clone()
        merged[127, ::2] += .001
        with self.assertRaisesRegex(ValueError, "position 127: cosine_similarity"):
            self.check_mean_pair(merged, reference)
        merged = reference.clone()
        merged[127] *= 1.00002
        with self.assertRaisesRegex(ValueError, "relative_rms_difference"):
            self.check_mean_pair(merged, reference)
        reference.fill_(100000.)
        merged = reference.clone()
        merged[127, 0] += .02
        with self.assertRaisesRegex(ValueError, "maximum_absolute_difference"):
            self.check_mean_pair(merged, reference)

    def test_numerical_gate_does_not_relax_own_hash_shape_or_dtype(self):
        mean = torch.ones(evaluation.EXPECTED_SHAPE)
        digest = evaluation.sha256_raw_float32_tensor(mean, torch)
        hashes = {"merged_mean": digest, "oracle_ft_mean": digest}
        changed = mean.clone()
        changed[0, 0] += 1e-6
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            evaluation.verify_loaded_merged_mean_consistency({"merged_mean": changed}, {"ft_mean": mean}, hashes, torch)
        for bad in (mean.double(), mean[:5]):
            with self.assertRaises(ValueError):
                evaluation.verify_loaded_merged_mean_consistency({"merged_mean": bad}, {"ft_mean": mean}, hashes, torch)

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
        self.assertEqual(output_one["provenance"]["attempt"]["corpus_spec_sha256"], "0" * 64)
        self.assertEqual(output_one["provenance"]["merged_mean_consistency"], synthetic_context()["merged_mean_consistency"])
        lowered_keys = {str(key).lower() for key in self._all_keys(output_one)}
        self.assertFalse({"rank", "score", "grade", "winner", "best_position"} & lowered_keys)

    @staticmethod
    def _all_keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from Attempt008EvaluationTests._all_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from Attempt008EvaluationTests._all_keys(child)


if __name__ == "__main__":
    unittest.main()
