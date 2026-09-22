import ast
import copy
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT / "scripts/ablation/freeze_generic_rollback_corpus.py"
SPEC_PATH = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json"
MODULE_SPEC = importlib.util.spec_from_file_location("freeze_generic_rollback_corpus", SCRIPT_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
freezer = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = freezer
MODULE_SPEC.loader.exec_module(freezer)


class FakeDataset:
    def __init__(self, examples):
        self.examples = examples
        self.seed = None

    def shuffle(self, *, seed):
        self.seed = seed
        return self.examples


class FakeTokenizer:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def encode(self, text, *, add_special_tokens):
        self.calls.append((text, add_special_tokens))
        return self.responses[text]


class GenericRollbackCorpusTests(unittest.TestCase):
    def test_exact_frozen_spec(self):
        self.assertEqual(freezer.load_spec(SPEC_PATH), freezer.FROZEN_CORPUS_SPEC)
        spec = freezer.FROZEN_CORPUS_SPEC
        self.assertEqual(spec["dataset"], {
            "repo_id": "science-of-finetuning/fineweb-1m-sample",
            "revision": "60b53a86b84eb6559e4407b113356f56a152318f",
            "split": "train", "streaming": False,
        })
        self.assertEqual(spec["selection"]["skip_valid_examples"], 20_000)
        self.assertEqual(spec["selection"]["sample_count"], 4_096)
        self.assertEqual(spec["selection"]["sequence_length"], 128)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            changed = copy.deepcopy(spec)
            changed["selection"]["skip_valid_examples"] = 10_000
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen definition"):
                freezer.load_spec(path)

    def test_valid_skip_take_ignores_blank_and_short_examples(self):
        spec = copy.deepcopy(freezer.FROZEN_CORPUS_SPEC)
        spec["selection"].update({
            "skip_valid_examples": 2, "sample_count": 2,
            "minimum_token_count": 3, "sequence_length": 3,
        })
        dataset = FakeDataset([
            {"text": "   "}, {"text": "short"}, {"text": "v0"},
            {"text": "\n"}, {"text": "v1"}, {"text": "short"},
            {"text": "v2"}, {"text": "v3"}, {"text": "v4"},
        ])
        tokenizer = FakeTokenizer({
            "short": [7, 8], "v0": [0, 1, 2, 3], "v1": [4, 5, 6],
            "v2": [10, 11, 12, 13], "v3": [20, 21, 22, 23],
            "v4": [30, 31, 32],
        })
        tokens = freezer.select_tokens(dataset, tokenizer, torch, spec)
        self.assertEqual(dataset.seed, 42)
        self.assertEqual(tokens.tolist(), [[10, 11, 12], [20, 21, 22]])
        self.assertNotIn(("v4", True), tokenizer.calls)
        self.assertTrue(all(add_special_tokens is True for _, add_special_tokens in tokenizer.calls))

    def test_text_is_truncated_to_1280_characters_before_encoding(self):
        spec = copy.deepcopy(freezer.FROZEN_CORPUS_SPEC)
        spec["selection"].update({"skip_valid_examples": 0, "sample_count": 1})
        text = "a" * 1280 + "b" * 20
        tokenizer = FakeTokenizer({"a" * 1280: list(range(140))})
        tensor = freezer.select_tokens(FakeDataset([{"text": text}]), tokenizer, torch, spec)
        self.assertEqual(tokenizer.calls, [("a" * 1280, True)])
        self.assertEqual(tensor.tolist()[0], list(range(128)))

    def test_shape_dtype_cpu_contiguity_and_raw_hash(self):
        tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        self.assertIs(freezer.validate_tokens(tensor, torch, sample_count=2, sequence_length=3), tensor)
        expected = hashlib.sha256(struct.pack("<6q", 1, 2, 3, 4, 5, 6)).hexdigest()
        self.assertEqual(freezer.sha256_raw_int64_tensor(tensor, torch), expected)
        with self.assertRaisesRegex(ValueError, "frozen shape"):
            freezer.validate_tokens(tensor, torch, sample_count=3, sequence_length=3)
        with self.assertRaisesRegex(ValueError, "int64"):
            freezer.validate_tokens(tensor.float(), torch, sample_count=2, sequence_length=3)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            freezer.validate_tokens(tensor.T, torch, sample_count=3, sequence_length=2)

    def test_overwrite_refusal_and_manifest_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "tokens.pt"
            manifest = root / "corpus-manifest.json"
            freezer.require_output_absent(artifact, manifest)
            artifact.write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "already exists"):
                freezer.require_output_absent(artifact, manifest)
            artifact.unlink()
            manifest.write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already exists"):
                freezer.write_manifest(manifest, {"new": True})
            self.assertEqual(manifest.read_text(), "existing")
        records = [{"path": name, "size_bytes": 1, "sha256": "a" * 64}
                   for name in freezer.TOKENIZER_FILE_NAMES]
        built = freezer.build_manifest(
            freezer.FROZEN_CORPUS_SPEC, "b" * 64, "c" * 64,
            freezer.DEFAULT_TOKENIZER_DIR, records, freezer.DEFAULT_ARTIFACT_PATH,
            "d" * 64, "e" * 64,
        )
        self.assertEqual(built["selection"]["skip_valid_examples"], 20_000)
        self.assertEqual(built["tokenizer_checkpoint"]["files"], records)
        self.assertEqual(built["artifact"]["raw_tensor_sha256"], "e" * 64)
        self.assertFalse({"timestamp", "host", "gpu"} & set(built))

    def test_static_information_boundary(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for forbidden in (
            "oracle", "models/base", "adapter", "evaluation.json",
            "models.lock.json", "downloaded-model-hashes.json",
            "attempts/001_", "attempts/002_", "attempts/003_", "attempts/004_",
        ):
            self.assertNotIn(forbidden, source.lower())
        self.assertIn("local_files_only=True", source)
        self.assertIn("streaming=False", source)


if __name__ == "__main__":
    unittest.main()
