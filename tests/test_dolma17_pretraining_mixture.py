"""Synthetic, CPU-only checks for the precommitted Attempt139 plan."""
from __future__ import annotations

import importlib.util
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def module(relative: str, name: str):
    source = ROOT / relative
    loader = importlib.util.spec_from_file_location(name, source)
    value = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(value)
    return value


freezer = module("scripts/ablation/freeze_dolma17_pretraining_mixture.py", "attempt139_freezer_test")
runner = module("scripts/ablation/run_dolma17_pretraining_mixture.py", "attempt139_runner_test")
helper = module("scripts/ablation/construct_generic_gradient_linear_response.py", "attempt139_ce_helper_test")
evaluator = module("scripts/ablation/evaluate_attempt014.py", "attempt139_evaluator_test")


class TestDolma17PretrainingMixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = freezer.load_spec()

    def test_authoritative_recipe_is_revision_and_hash_pinned(self):
        provenance = self.spec["recipe_provenance"]
        for name in ("native_training_config", "source_resolved_counts_and_no_code",
                     "datadecide_card", "dolma_card"):
            pin = provenance[name]
            self.assertEqual(len(pin["revision"]), 40)
            self.assertEqual(len(pin["sha256"]), 64)
            self.assertIn(pin["revision"], pin["url"])
        self.assertEqual(provenance["native_training_config"]["run_name"], "OLMo-1.7-7B")
        self.assertEqual(self.spec["dataset"]["revision"],
                         "7f48140530a023e9ea4c5cfb141160922727d4d3")
        self.assertEqual(len(self.spec["dataset"]["sha256"]), 64)
        self.assertIn("identical 1033-path", provenance["discrepancy"])

    def test_published_recipe_parser_and_hash_gate(self):
        source = b'''DATA_SOURCES = {"a": ["a-0.npy"], "starcoder": ["s-0.npy"], "stackexchange": ["e-0.npy"]}
SOURCES_SIZES = {"a": {"total_size": 100}, "starcoder": {"total_size": 20}, "stackexchange": {"total_size": 10}}
DATA_PATHS = {}
DATA_PATHS["dolma17"] = build_collection_include(["a", "starcoder", "stackexchange"])
DATA_PATHS["no_code"] = build_collection_include(["a"])
'''
        self.assertEqual(freezer.published_source_paths(source, "dolma17")[0],
                         ["a", "starcoder", "stackexchange"])
        self.assertEqual(freezer.published_source_paths(source, "no_code")[1], ["a-0.npy"])

        class Response(io.BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()
        pin = {"url": "https://example.test/pinned", "sha256": hashlib.sha256(source).hexdigest()}
        with patch.object(freezer, "urlopen", return_value=Response(source)):
            self.assertEqual(freezer.fetch_pinned(pin), source)
        with patch.object(freezer, "urlopen", return_value=Response(source)):
            with self.assertRaises(ValueError):
                freezer.fetch_pinned({**pin, "sha256": "0" * 64})

    def test_weights_and_exact_no_code_exclusions(self):
        counts = self.spec["source_token_counts"]
        self.assertEqual(sum(counts.values()), 1_715_152_617_328)
        self.assertEqual(counts["starcoder"], 263_775_304_843)
        self.assertEqual(counts["falcon"], 456_404_959_027)
        self.assertEqual(self.spec["no_code_excluded_sources"], ["stackexchange", "starcoder"])
        self.assertEqual(set(self.spec["native"]["source_weights"]) -
                         set(self.spec["no_code"]["source_weights"]),
                         {"stackexchange", "starcoder"})
        self.assertEqual(self.spec["no_code"]["denominator_tokens"],
                         sum(counts.values()) - counts["stackexchange"] - counts["starcoder"])
        self.assertNotEqual(self.spec["native"]["source_weights"]["web_rest"],
                            self.spec["native"]["source_weights"]["gutenberg_books"])
        self.assertEqual(self.spec["native"]["source_weights"]["web_rest"],
                         counts["web_rest"] / sum(counts.values()))
        self.assertNotIn("surprisal", json.dumps(self.spec["native"]).lower())

    def test_largest_remainder_determinism_and_tie_rule(self):
        for label in ("native", "no_code"):
            names = list(self.spec[label]["integer_quotas"])
            self.assertEqual(freezer.quota_plan(self.spec["source_token_counts"], names), self.spec[label])
            self.assertEqual(sum(self.spec[label]["integer_quotas"].values()), 4096)
            self.assertEqual(len(self.spec[label]["remainder_awards"]),
                             4096 - sum(4096 * self.spec["source_token_counts"][name] //
                                        self.spec[label]["denominator_tokens"] for name in names))
        tie = freezer.quota_plan({"z": 1, "a": 1, "m": 1}, ["z", "a", "m"], count=2)
        self.assertEqual(tie["remainder_awards"], ["a", "m"])
        self.assertEqual(tie["integer_quotas"], {"z": 0, "a": 1, "m": 1})

    def test_shared_source_pool_nesting_and_shuffle(self):
        # The same two ordered source pools supply different candidate quotas.
        a = torch.arange(3000 * 128, dtype=torch.int64).reshape(3000, 128)
        b = torch.arange(3000 * 128, dtype=torch.int64).reshape(3000, 128) + 1_000_000
        pools = {"a": a, "b": b}
        first, first_manifest = freezer.candidate_from_pools(
            pools, {"a": 2048, "b": 2048}, ["a", "b"], 139017, torch)
        second, second_manifest = freezer.candidate_from_pools(
            pools, {"a": 2500, "b": 1596}, ["a", "b"], 139017, torch)
        repeated, repeated_manifest = freezer.candidate_from_pools(
            pools, {"a": 2048, "b": 2048}, ["a", "b"], 139017, torch)
        self.assertTrue(torch.equal(first, repeated))
        self.assertEqual(first_manifest, repeated_manifest)
        self.assertEqual(first_manifest["permutation_sha256"], second_manifest["permutation_sha256"])
        self.assertTrue(torch.isin(first[:, 0], a[:2048, 0]).sum() == 2048)
        self.assertTrue(torch.isin(second[:, 0], a[:2500, 0]).sum() == 2500)
        self.assertFalse(torch.equal(first, second))

    def test_int64_shape_dtype_hash_validation(self):
        tensor = torch.zeros((4096, 128), dtype=torch.int64)
        digest = freezer.raw_int64_sha256(tensor, torch)
        self.assertEqual(len(digest), 64)
        tensor[0, 0] = 1
        self.assertNotEqual(freezer.raw_int64_sha256(tensor, torch), digest)
        with self.assertRaises(ValueError):
            freezer.raw_int64_sha256(tensor.float(), torch)
        with self.assertRaises(ValueError):
            freezer.raw_int64_sha256(tensor.T, torch)
        self.assertEqual(self.spec["corpus"]["sample_count"], 4096)
        self.assertEqual(self.spec["corpus"]["sequence_length"], 128)

    def test_streamed_document_extraction_stops_at_quota(self):
        records = [{"text": "short"}, {"text": "A" * 200},
                   {"text": "B" * 200}, {"text": "C" * 200}]
        payload = gzip.compress("".join(json.dumps(row) + "\n" for row in records).encode())

        class Response(io.BytesIO):
            status = 200
            headers = {"Content-Length": str(len(payload)), "ETag": "synthetic"}
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()

        class Tokenizer:
            def encode(self, text, **kwargs):
                self.last_text = text
                self.last_options = kwargs
                return list(range(128)) if len(text) >= 128 else [1]

        tokenizer = Tokenizer()
        spec = {"corpus": {"stream_max_compressed_bytes": 1000000}}
        with patch.object(freezer, "urlopen", return_value=Response(payload)):
            tensor, manifest = freezer.freeze_source(
                "synthetic", ["folder"], {"folder": ["https://example.test/shard.json.gz"]},
                2, tokenizer, torch, spec, [0], float("inf"))
        self.assertEqual(tuple(tensor.shape), (2, 128))
        self.assertEqual(manifest["documents_scanned"], 3)
        self.assertEqual(len(manifest["accepted_documents"]), 2)
        self.assertEqual(manifest["shards"][0]["lines_read"], 3)
        self.assertEqual(tokenizer.last_text, "B" * 200)
        self.assertEqual(tokenizer.last_options,
                         {"add_special_tokens": True, "truncation": True, "max_length": 128})

    def test_exact_ce_objective_positive_gradient_and_98_eligible_matrices(self):
        tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.int64)
        logits = torch.randn((1, 4, 5), dtype=torch.float32, requires_grad=True)
        loss = helper.causal_token_loss_sum(logits, tokens, torch)
        expected = torch.nn.functional.cross_entropy(logits[:, :3].reshape(3, 5),
                                                     tokens[:, 1:].reshape(3), reduction="sum")
        self.assertTrue(torch.allclose(loss, expected))
        (loss / 3).backward()
        self.assertTrue(torch.allclose(logits.grad[:, -1], torch.zeros_like(logits.grad[:, -1])))
        self.assertEqual(self.spec["objective"]["direction"], "positive_gradient")
        self.assertEqual(self.spec["objective"]["reduction"], "sum_cross_entropy_divided_by_520192")

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([
                    torch.nn.ModuleDict({f"linear_{i}": torch.nn.Linear(2, 2, bias=False)
                                         for i in range(7)}) if layer < 14 else torch.nn.Module()
                    for layer in range(28)])
        tiny = TinyModel()
        eligible = helper.discover_eligible_linear_weights(tiny, torch)
        self.assertEqual(len(eligible), 98)
        helper.freeze_other_parameters(tiny, eligible)
        for _, linear in eligible:
            linear.weight.grad = torch.ones_like(linear.weight)
        scale = helper.global_tangent_scale(eligible, 0.00125, torch)
        tangents, records, realized = helper.prepare_tangents(eligible, scale, torch)
        self.assertEqual(len(records), 98)
        self.assertEqual(len(tangents), 98)
        self.assertAlmostEqual(realized["aggregate_realized_relative_tangent_norm"], 0.00125, places=9)
        self.assertTrue(all(torch.all(value > 0) for value in tangents.values()))

    def test_full_probe_batch64_and_strict_jvp(self):
        self.assertEqual(self.spec["probe"]["sample_count"], 10000)
        self.assertEqual(self.spec["probe"]["jvp_batch_size"], 64)
        self.assertEqual(self.spec["probe"]["jvp_batch_count"], 157)
        self.assertTrue(self.spec["jvp"]["strict_forward_ad"])
        self.assertFalse(self.spec["jvp"]["fallback"])
        self.assertEqual(self.spec["readout"]["hidden_state_index"], 14)

        class Accumulator:
            def __init__(self, spec, torch_module):
                self.count = 0
            def add(self, primal, response):
                self.count += primal.shape[0]
            def means(self, count):
                assert self.count == count
                return None, torch.zeros((128, 2048), dtype=torch.float64)

        class FakeHelper:
            ResponseAccumulator = Accumulator
            def __init__(self):
                self.sizes = []
            def ordinary_hook_readout(self, *args):
                return torch.tensor(0.)
            def run_readout_jvp(self, model, batch, tangents, spec, torch_module):
                self.sizes.append(batch.shape[0])
                x = torch.zeros((batch.shape[0], 1), dtype=torch.float32)
                return x, x
            def compare_primal(self, *args):
                return 0.

        fake = FakeHelper()
        model = torch.nn.Linear(1, 1)
        probe = torch.zeros((10000, 128), dtype=torch.int64)
        response, check = runner.full_probe_jvp(model, probe, {"tiny": 1}, self.spec, fake, torch, "unit")
        self.assertEqual(fake.sizes, [64] * 156 + [16])
        self.assertEqual(check["accumulated_rows"], 10000)
        self.assertEqual(response.shape, (128, 2048))
        self.assertEqual(response.dtype, torch.float32)

    def test_barrier_order_and_no_post_result_candidate(self):
        source = (ROOT / "scripts/ablation/run_dolma17_pretraining_mixture.py").read_text()
        run_source = source[source.index("def run()") :]
        self.assertLess(run_source.index("validate_published_blind("), run_source.index("print(MARKER"))
        self.assertLess(run_source.index("print(MARKER"), run_source.index("evaluate_after_barrier("))
        self.assertNotIn("evaluate_after_barrier", source[source.index("def check_blind_inputs"):source.index("def evaluate_after_barrier")])
        self.assertEqual(self.spec["candidate"]["tensor_order"], list(runner.NAMES))
        self.assertFalse(self.spec["candidate"]["combination"])
        self.assertFalse(self.spec["candidate"]["sign_selection"])
        self.assertFalse(self.spec["candidate"]["response_normalization"])
        self.assertEqual(self.spec["barrier"]["marker"], runner.MARKER)

    def test_signed_cpu_float64_evaluation_and_classification(self):
        x = torch.ones((128, 2048), dtype=torch.float32)
        y = x.clone()
        y[0] *= -1
        report = runner.signed_report("synthetic", x, y, evaluator)
        self.assertAlmostEqual(report["position_0_cosine"], -1.)
        for value in report["positions_1_4_individual_cosines"]:
            self.assertAlmostEqual(value, 1.)
        self.assertAlmostEqual(report["positions_1_127_mean_cosine"], 1.)
        self.assertEqual(len(report["all_128_position_cosines"]), 128)
        self.assertEqual(runner.classify(0.01, 0.01), "clear_gain_over_attempt134")
        self.assertEqual(runner.classify(0.005, 0.02), "positive_gain_over_attempt134")
        self.assertEqual(runner.classify(0.0, 0.02), "no_gain_over_attempt134")
        self.assertEqual(runner.comparison(report, report)["positions_1_127_win_count"], 0)

    def test_overwrite_refusal_and_fixed_patchscope_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.pt", Path(directory) / "b.json"
            freezer.refuse_outputs([a, b])
            a.write_bytes(b"occupied")
            with self.assertRaises(ValueError):
                freezer.refuse_outputs([a, b])
            with self.assertRaises(ValueError):
                freezer.refuse_outputs([b, b])
        with tempfile.TemporaryDirectory() as directory:
            spec = {"outputs": {"candidate_path": str(Path(directory) / "candidate.pt"),
                                "construction_manifest_path": str(Path(directory) / "construction.json"),
                                "result_path": str(Path(directory) / "result.json")}}
            runner.output_paths(spec)
            Path(spec["outputs"]["result_path"]).write_text("occupied")
            with self.assertRaises(ValueError):
                runner.output_paths(spec)
        patch = self.spec["evaluation"]["patchscope"]
        self.assertEqual(patch["positions"], [0, 1, 2, 3, 4])
        self.assertEqual(patch["semantic_substrings"], ["cake", "bake", "cook", "craft", "precision"])
        self.assertEqual(len(patch["prompts"]), 8)
        self.assertEqual(patch["modes"], ["directional_oracle_norm_matched", "native_amplitude"])


if __name__ == "__main__":
    unittest.main()
