import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

import read_oracle_logit_lens as lens  # noqa: E402


class OffsetNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("offset", torch.tensor([0.5, -1.0]))
        self.calls = 0

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return latent + self.offset


class TinyTokenizer:
    def __len__(self) -> int:
        return 5

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token-{token_id}"

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        self.decode_kwargs = kwargs
        return f"decoded-{token_ids[0]}"


class LogitLensTests(unittest.TestCase):
    def test_negative_direction_negates_normed_latent(self) -> None:
        norm = OffsetNorm()
        head = torch.nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, -1.0],
                        [-0.5, 0.25],
                    ]
                )
            )
        latent = torch.tensor([2.0, 3.0])

        positive, negative = lens.logit_lens_probabilities(
            latent, norm, head, torch
        )

        normed = latent + norm.offset
        self.assertTrue(torch.equal(positive, torch.softmax(head(normed), dim=-1)))
        self.assertTrue(torch.equal(negative, torch.softmax(head(-normed), dim=-1)))
        self.assertEqual(norm.calls, 1)
        self.assertFalse(
            torch.equal(
                negative,
                torch.softmax(head(-latent + norm.offset), dim=-1),
            )
        )

    def test_top_tokens_are_descending_and_ties_use_token_id(self) -> None:
        tokenizer = TinyTokenizer()
        probabilities = torch.tensor([0.1, 0.3, 0.3, 0.2, 0.1])

        records = lens.top_token_records(probabilities, tokenizer, 4, torch)

        self.assertEqual([record["token_id"] for record in records], [1, 2, 3, 0])
        self.assertEqual(records[0]["token"], "token-1")
        self.assertEqual(records[0]["decoded"], "decoded-1")
        self.assertEqual(
            tokenizer.decode_kwargs,
            {
                "skip_special_tokens": False,
                "clean_up_tokenization_spaces": False,
            },
        )
        self.assertEqual(
            [record["probability"] for record in records],
            [
                float(probabilities[1]),
                float(probabilities[2]),
                float(probabilities[3]),
                float(probabilities[0]),
            ],
        )

    def test_loaded_model_modules_and_dimensions_are_validated(self) -> None:
        norm = torch.nn.LayerNorm(2)
        head = torch.nn.Linear(2, 5, bias=False)
        backbone = torch.nn.Module()
        backbone.norm = norm
        backbone.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(lens.NUM_LAYERS)]
        )

        class MockQwen(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = backbone
                self.lm_head = head
                self.config = SimpleNamespace(
                    model_type="qwen3",
                    num_hidden_layers=lens.NUM_LAYERS,
                    hidden_size=2,
                    vocab_size=5,
                )

            def get_output_embeddings(self) -> torch.nn.Module:
                return self.lm_head

        model = MockQwen().float()
        actual_norm, actual_head = lens.validate_loaded_qwen3_model(
            model, 2, 5, torch
        )
        self.assertIs(actual_norm, norm)
        self.assertIs(actual_head, head)

        wrong_head = torch.nn.Linear(3, 5, bias=False)
        model.lm_head = wrong_head
        with self.assertRaisesRegex(ValueError, "LM head has unexpected dimensions"):
            lens.validate_loaded_qwen3_model(model, 2, 5, torch)

    def test_readout_covers_positions_vectors_and_directions(self) -> None:
        tokenizer = TinyTokenizer()
        norm = OffsetNorm()
        head = torch.nn.Linear(2, 5, bias=False)
        with torch.no_grad():
            head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, -1.0],
                        [-0.5, 0.25],
                        [0.25, 0.5],
                    ]
                )
            )
        oracle = {
            name: torch.arange(
                lens.SEQUENCE_LENGTH * 2,
                dtype=torch.float32,
            ).reshape(lens.SEQUENCE_LENGTH, 2)
            for name in lens.VECTOR_TYPES
        }
        provenance = {
            "base": {"repo_id": "base", "revision": "a"},
            "fine_tuned": {"repo_id": "ft", "revision": "b"},
            "hidden_size": 2,
            "vocab_size": 5,
            "downloaded_model_hashes_sha256": "1" * 64,
            "merged_model_hashes_sha256": "2" * 64,
            "oracle_probe_manifest_sha256": "3" * 64,
            "oracle_adl_manifest_sha256": "4" * 64,
            "oracle_adl_sha256": "5" * 64,
        }

        output = lens.build_readout(
            oracle,
            norm,
            head,
            tokenizer,
            provenance,
            torch,
            top_k=3,
        )

        self.assertEqual(
            [record["position"] for record in output["positions"]],
            list(lens.POSITIONS),
        )
        for position in output["positions"]:
            for vector_type in lens.VECTOR_TYPES:
                self.assertEqual(
                    set(position[vector_type]),
                    {"positive", "negative"},
                )
                self.assertEqual(len(position[vector_type]["positive"]), 3)
                self.assertEqual(len(position[vector_type]["negative"]), 3)
        self.assertEqual(norm.calls, len(lens.POSITIONS) * len(lens.VECTOR_TYPES))

    def test_output_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "oracle-logit-lens.json"
            output_path.write_text("existing\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "already exists"):
                lens.write_output({"format_version": 1}, output_path)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "existing\n",
            )


if __name__ == "__main__":
    unittest.main()
