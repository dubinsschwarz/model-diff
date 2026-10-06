"""Synthetic tests for Attempt140; no model or real diagnostic is loaded."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ablation/run_replicated_surprisal_contrast_modes.py"
loader = importlib.util.spec_from_file_location("attempt140_under_test", SOURCE)
a140 = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a140)


class Attempt140Tests(unittest.TestCase):
    def test_pinned_upstream_blind_provenance(self):
        spec = a140.load_spec()
        self.assertEqual(spec["expected_prior_head_commit"],
                         "0174428e48b79805111fbcefc3e195210d21c9e9")
        validator = a140.import_pinned_source(spec["blind_validator_source"], "attempt140_test_137")
        for row in spec["blind_inputs"]:
            manifest = validator.validate_blind_manifest(row)
            self.assertEqual(manifest["candidate" if row["attempt_number"] != 100 else "artifact"]
                             ["serialized_sha256"], row["artifact"]["serialized_sha256"])
            self.assertEqual(a140.sha256_file(a140.path_of(row["spec"]["path"])),
                             row["spec"]["sha256"])
            self.assertEqual(a140.sha256_file(a140.path_of(row["construction_manifest"]["path"])),
                             row["construction_manifest"]["sha256"])
        self.assertEqual([row["attempt_number"] for row in spec["blind_inputs"]], [100, 126, 129, 134])
        self.assertEqual(spec["corpus_order"], list(a140.ORDER))
        self.assertEqual(spec["fineweb_replication"]["strata"],
                         {key: list(value) for key, value in a140.STRATA.items()})

    def test_exact_low_minus_full_and_signed_consensus(self):
        full = np.zeros((8, 5, 3))
        low = np.zeros_like(full)
        low[:, :, 0] = 2
        low[0, :, 1] = 1
        d = low - full
        unit, norms = a140.unitize_positions(d)
        consensus, concentration, pairwise = a140.signed_consensus(unit)
        self.assertTrue(np.all(norms > 0))
        self.assertGreater(consensus[0, 0], 0)
        self.assertGreater(pairwise[0, 0, 1], 0)
        self.assertLessEqual(float(np.max(concentration)), 1)
        reversed_unit, _ = a140.unitize_positions(-d)
        reversed_consensus, _, _ = a140.signed_consensus(reversed_unit)
        np.testing.assert_allclose(reversed_consensus, -consensus)

    def test_zero_contrast_and_canceled_consensus_fail(self):
        with self.assertRaises(ValueError):
            a140.unitize_positions(np.zeros((8, 5, 3)))
        unit = np.ones((8, 5, 2))
        unit[4:] *= -1
        with self.assertRaises(ValueError):
            a140.signed_consensus(unit)

    def test_svd_rank_tolerance_and_blind_pc1_orientation(self):
        unit = np.zeros((8, 5, 4), dtype=np.float64)
        unit[:, :, 0] = 1
        unit[0, :, 1] = 0.1
        unit, _ = a140.unitize_positions(unit)
        consensus, _, _ = a140.signed_consensus(unit)
        first = a140.fixed_svd(unit, consensus, 1, 5)
        second = a140.fixed_svd(unit, consensus, 1, 5)
        self.assertEqual(first["matrix_shape"], [8, 16])
        self.assertEqual(first["rank"], 2)
        self.assertEqual(first["rank_tolerance"],
                         max(first["matrix_shape"]) * np.finfo(np.float64).eps *
                         first["singular_values"][0])
        self.assertGreater(np.dot(first["pc1"], consensus[1:5].reshape(-1)), 0)
        np.testing.assert_allclose(first["basis"], second["basis"])
        np.testing.assert_allclose(first["basis"] @ first["basis"].T, np.eye(first["rank"]), atol=1e-12)

    def test_post_barrier_signed_cpu_float64_cosine_convention(self):
        direction = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        residual = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
        oracle = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        consensus = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        report = a140.single_direction_diagnostic(direction, residual, oracle, consensus)
        self.assertAlmostEqual(report["signed_cosine_with_residual"], -0.5)
        self.assertAlmostEqual(report["absolute_cosine_with_residual"], 0.5)
        self.assertAlmostEqual(report["residual_squared_fraction_in_single_direction"], 0.25)
        self.assertAlmostEqual(report["signed_cosine_with_oracle"], 0.5)
        self.assertAlmostEqual(report["signed_cosine_with_attempt134_consensus"], 0.5)

    def test_fineweb_formulas_and_signed_replication(self):
        low = np.full((128, 4), 3.0)
        middle = np.full((128, 4), 2.0)
        high = np.full((128, 4), 0.5)
        old = a140.fineweb_contrasts({"response_low": low, "response_middle": middle,
                                      "response_high": high})
        np.testing.assert_allclose(old["slope"], low - high)
        np.testing.assert_allclose(old["curvature"], low - 2 * middle + high)
        np.testing.assert_allclose(old["low_middle"] + old["middle_high"], old["low_high"])
        fresh = {key: value.copy() for key, value in old.items()}
        report = a140.fineweb_replication_report({"old": old, "fresh": fresh})
        self.assertAlmostEqual(report["slope"]["positions_1_4_flattened_signed_cosine"], 1)
        self.assertAlmostEqual(report["curvature"]["positions_1_127_mean_signed_cosine"], 1)
        self.assertTrue(report["within_slice"]["old"]["low_middle_vs_middle_high_same_direction"])
        opposite = a140.replication_metrics(old["slope"], -old["slope"])
        self.assertAlmostEqual(opposite["positions_1_4_mean_signed_cosine"], -1)
        self.assertAlmostEqual(opposite["positions_1_127_flattened_signed_cosine"], -1)

    def test_hash_inventory_and_overwrite_refusal(self):
        value = torch.ones((3, 4), dtype=torch.float64)
        item = {"x": value}
        first = a140.tensor_hash_inventory(item, torch)
        self.assertEqual(first["x"]["shape"], [3, 4])
        self.assertEqual(first["x"]["dtype"], "torch.float64")
        item["x"][0, 0] = 2
        self.assertNotEqual(first, a140.tensor_hash_inventory(item, torch))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "present.pt"
            path.write_bytes(b"frozen")
            with self.assertRaises(ValueError):
                a140.require_absent(path)

    def test_candidate_raw_tensor_hash_and_shape_validation(self):
        family = copy.deepcopy(a140.load_spec()["blind_inputs"][1])  # Attempt126.
        names = family["artifact"]["tensor_order"]
        artifact = {name: torch.full(a140.SHAPE, float(i + 1), dtype=torch.float32)
                    for i, name in enumerate(names)}
        records = {name: {"raw_sha256": a140.raw_hash(value.numpy())}
                   for name, value in artifact.items()}
        manifest = {"ranking": {"strata": {key: {"rank_range": list(bounds)}
                                             for key, bounds in a140.STRATA.items()}},
                    "probe": {"selected_rows": [0, 1024]}, "responses": records}
        validator = mock.Mock()
        validator.validate_blind_manifest.return_value = manifest
        pinned = a140.import_pinned_source(a140.load_spec()["blind_validator_source"],
                                           "attempt140_test_tensor_validator")
        validator.validate_response_tensor.side_effect = pinned.validate_response_tensor
        with mock.patch.object(a140, "require_hash"), mock.patch.object(torch, "load", return_value=artifact):
            self.assertEqual(list(a140.validate_blind_family(family, validator, torch)), names)
            artifact[names[0]][0, 0] += 1
            with self.assertRaises(ValueError):
                a140.validate_blind_family(family, validator, torch)
            artifact[names[0]] = torch.ones((2, 2), dtype=torch.float32)
            with self.assertRaises(ValueError):
                a140.validate_blind_family(family, validator, torch)

    def test_blind_builder_keeps_low_minus_full_order(self):
        spec = a140.load_spec()
        rng = np.random.default_rng(140)
        payload = {}
        for number in (100, 134):
            prefix = "response_" if number == 100 else "response_low_"
            payload[number] = {prefix + name: torch.from_numpy(
                rng.normal(size=(128, 4)).astype(np.float32)) for name in a140.ORDER}
        for number in (126, 129):
            payload[number] = {"response_" + name: torch.from_numpy(
                rng.normal(size=(128, 4)).astype(np.float32)) for name in a140.STRATA}
        with mock.patch.object(a140, "validate_blind_family",
                               side_effect=lambda row, *_: payload[row["attempt_number"]]):
            artifact, _ = a140.build_blind_artifact(spec, object(), torch)
        expected = np.stack([(payload[134]["response_low_" + name].numpy().astype(np.float64) -
                              payload[100]["response_" + name].numpy().astype(np.float64))
                             for name in a140.ORDER])
        np.testing.assert_array_equal(artifact["raw_full_to_low"].numpy(), expected)
        self.assertEqual(artifact["metadata"]["full_low_formula"], "LOW - FULL")

    def test_frozen_output_reload_validates_tensor_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = {"artifact": str(root / "modes.pt"),
                       "construction_manifest": str(root / "construction-manifest.json"),
                       "result": str(root / "result.json")}
            artifact = {"raw_full_to_low": torch.ones((8, *a140.SHAPE), dtype=torch.float64),
                        "signed_equal_weight_consensus": torch.ones(a140.SHAPE, dtype=torch.float64),
                        "svd": {}, "fineweb": {},
                        "metadata": {"corpus_order": list(a140.ORDER),
                                     "ranges": {k: list(v) for k, v in a140.RANGES.items()},
                                     "full_low_formula": "LOW - FULL"}}
            for name, (start, stop) in a140.RANGES.items():
                width = (stop - start) * a140.SHAPE[1]
                basis = torch.zeros((1, width), dtype=torch.float64)
                basis[0, 0] = 1
                artifact["svd"][name] = {"rank": 1, "basis": basis,
                                         "pc1": basis[0].clone()}
            torch.save(artifact, outputs["artifact"])
            manifest = {"spec_sha256": a140.SPEC_SHA256,
                        "artifact": {"serialized_sha256": a140.sha256_file(Path(outputs["artifact"])),
                                     "raw_tensors": a140.tensor_hash_inventory(artifact, torch)}}
            Path(outputs["construction_manifest"]).write_text(json.dumps(manifest))
            spec = {"outputs": outputs, "blind_inputs": [],
                    "ranges": {k: list(v) for k, v in a140.RANGES.items()}}
            a140.reload_and_validate_blind(spec, manifest, object(), torch)
            manifest["artifact"]["raw_tensors"]["raw_full_to_low"]["raw_sha256"] = "0" * 64
            Path(outputs["construction_manifest"]).write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                a140.reload_and_validate_blind(spec, manifest, object(), torch)

    def test_barrier_order_and_no_oracle_reads_in_blind_phase(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "result.json"
            spec = {"outputs": {"artifact": str(Path(directory) / "blind.pt"),
                                "construction_manifest": str(Path(directory) / "manifest.json"),
                                "result": str(result)},
                    "blind_validator_source": {"path": "unused", "sha256": "unused"}}
            calls = []
            def build(*_):
                calls.append("build")
                return {"mock": "artifact"}, {"mock": "report"}
            def publish(*_):
                calls.append("publish")
                return {"mock": "manifest"}
            def reload(*_):
                calls.append("reload")
                return {"mock": "reloaded"}
            def post(*_):
                self.assertIn("\n" + a140.MARKER + "\n", output.getvalue())
                calls.append("post")
                return {"attempt_id": a140.ATTEMPT}
            output = io.StringIO()
            with mock.patch.object(a140, "load_spec", return_value=spec), \
                 mock.patch.object(a140, "import_pinned_source", return_value=object()), \
                 mock.patch.object(a140, "build_blind_artifact", side_effect=build), \
                 mock.patch.object(a140, "publish_blind", side_effect=publish), \
                 mock.patch.object(a140, "reload_and_validate_blind", side_effect=reload), \
                 mock.patch.object(a140, "post_barrier_evaluate", side_effect=post), \
                 contextlib.redirect_stdout(output):
                a140.run()
            self.assertEqual(calls, ["build", "publish", "reload", "post"])
            self.assertIn("\n" + a140.MARKER + "\n", output.getvalue())
            self.assertEqual(json.loads(result.read_text())["attempt_id"], a140.ATTEMPT)
            with self.assertRaises(ValueError):
                a140.require_absent(result)

    def test_static_information_firewall_and_no_candidate_formula(self):
        source = SOURCE.read_text()
        pre = source.split("def post_barrier_evaluate(", 1)[0]
        self.assertNotIn('result.json', pre)
        self.assertNotIn('evaluation.json', pre)
        self.assertNotIn('oracle_adl', pre)
        self.assertNotIn('historical_base', pre)
        self.assertNotIn('transformers', source)
        self.assertNotIn('beta *', source)
        self.assertNotIn('response_consensus +', source)
        run = source.split("def run()", 1)[1]
        self.assertLess(run.index("reload_and_validate_blind"), run.index("print(MARKER"))
        self.assertLess(run.index("print(MARKER"), run.index("post_barrier_evaluate"))
        spec = a140.load_spec()
        self.assertTrue(spec["no_recovery_candidate"])
        self.assertTrue(spec["no_post_oracle_combination"])


if __name__ == "__main__":
    unittest.main()
