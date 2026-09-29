import ast
import copy
import hashlib
import importlib.util
import json
import math
import io
from unittest.mock import patch
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT = Path(__file__).resolve().parents[1]
ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/009_generic_gradient_response_map_prefix0_13"
SCRIPT_PATH = PROJECT / "scripts/ablation/construct_generic_gradient_response_map.py"
SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
CORPUS_SPEC_PATH = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json"
MODULE_SPEC = importlib.util.spec_from_file_location("construct_generic_gradient_response_map", SCRIPT_PATH)
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


class GenericGradientLinearResponseTests(unittest.TestCase):
    def test_frozen_specs_and_information_policy(self):
        spec = rollback.load_spec(SPEC_PATH)
        corpus_spec = rollback.load_corpus_spec(CORPUS_SPEC_PATH)
        self.assertEqual(spec, rollback.FROZEN_SPEC)
        self.assertEqual(corpus_spec, rollback.FROZEN_CORPUS_SPEC)
        self.assertEqual(spec["information_policy"], "final_checkpoint_plus_frozen_generic_corpus_and_probe")
        self.assertEqual(spec["eligible_tensors"]["transformer_block_indices"], list(range(14)))
        self.assertEqual(spec["generic_loss"]["total_prediction_tokens"], 4096 * 127)
        self.assertEqual(spec["generic_loss"]["number_of_batches"], 512)
        self.assertEqual(spec["tangent"]["target_relative_frobenius"], 0.00125)

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


    def test_gradients_accumulate_without_zeroing_between_batches(self):
        model = ToyCausalModel()
        model.proj.weight.grad = torch.ones_like(model.proj.weight)
        seen = []
        def hook(module, args):
            seen.append(None if module.proj.weight.grad is None else module.proj.weight.grad.clone())
        model.register_forward_pre_hook(hook)
        tokens = torch.tensor([[1, 0], [2, 1], [1, 2], [2, 0]])
        loss_spec = dict(sample_count=4, sequence_length=2, predictions_per_sample=1,
                         total_prediction_tokens=4, batch_size=2, number_of_batches=2)
        with patch.object(model, "zero_grad", wraps=model.zero_grad) as zero:
            rollback.accumulate_mean_generic_gradient(model, tokens, [("proj", model.proj)], loss_spec, torch)
        self.assertEqual(zero.call_count, 1)
        self.assertIsNone(seen[0])
        reference = ToyCausalModel()
        loss = rollback.causal_token_loss_sum(reference(input_ids=tokens[:2], use_cache=False).logits, tokens[:2], torch) / 4
        loss.backward()
        self.assertTrue(torch.equal(seen[1], reference.proj.weight.grad))


    def test_probe_hashes_and_frozen_definition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            probe = torch.arange(12).reshape(3, 4)
            torch.save(probe, path)
            spec = dict(sample_count=3, sequence_length=4, serialized_sha256=rollback.sha256_file(path),
                        raw_tensor_sha256=rollback.sha256_raw_int64_tensor(probe, torch))
            self.assertTrue(torch.equal(rollback.load_probe(path, spec, torch), probe))
            for key in ("serialized_sha256", "raw_tensor_sha256"):
                with self.assertRaises(ValueError):
                    rollback.load_probe(path, {**spec, key: "0" * 64}, torch)
            with self.assertRaises(ValueError):
                rollback.load_probe(path, {**spec, "sequence_length": 3}, torch)
        frozen = rollback.FROZEN_SPEC["probe"]
        self.assertEqual(frozen["serialized_sha256"], "3d799d19abf4a2d6dc92b2bd164d81d83f6451cd3545f5ae5d8c04a8b299735b")
        self.assertEqual(frozen["raw_tensor_sha256"], "73f56300fdd06bc8fafcfa5750fee7a586a5c5e6071a608667ffc526bbb90ac8")
        self.assertEqual(frozen["batch_size"], 32)


class TinyBlock(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
        self.nonlinear = nonlinear

    def forward(self, hidden):
        value = self.proj(hidden)
        return torch.tanh(value) if self.nonlinear else value


class TinyTransformer(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock(nonlinear and i == 0) for i in range(28)])
        self.register_buffer("offset", torch.tensor([1., 2.]))
        self.bad_index = False
        self.detach_output = False
        self.calls = []

    def forward(self, *, input_ids, use_cache, output_hidden_states, return_dict):
        assert use_cache is False and output_hidden_states is True and return_dict is True
        self.calls.append(tuple(input_ids.shape))
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2) * .1 + self.offset
        states = []
        for layer in self.layers:
            states.append(hidden)
            hidden = layer(hidden)
        states.append(hidden * 7)  # Distinct final normalization, never the readout.
        if self.bad_index:
            states[14] = states[13]
        if self.detach_output:
            states = [value.detach() for value in states]
        return SimpleNamespace(hidden_states=tuple(states))


class TinyModel(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.model = TinyTransformer(nonlinear)
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=28, hidden_size=2)


def tiny_spec(count=5, length=3, batch=2):
    spec = copy.deepcopy(rollback.FROZEN_SPEC)
    spec["probe"].update(sample_count=count, sequence_length=length, batch_size=batch)
    spec["readout"]["hidden_size"] = 2
    return spec


class LinearResponseJVPTests(unittest.TestCase):
    def test_analytical_jvp_partial_parameters_and_no_grad(self):
        model, spec = TinyModel(), tiny_spec()
        tokens = torch.tensor([[0, 1, 2], [3, 4, 5]])
        vector = torch.tensor([[.3, -.2], [.5, .1]])
        tangents = {"model.layers.0.proj.weight": vector}
        before = rollback.model_state_hashes(model, torch)
        with patch.object(torch.func, "functional_call", wraps=torch.func.functional_call) as call:
            primal, response = rollback.run_readout_jvp(model, tokens, tangents, spec, torch)
        self.assertEqual(set(call.call_args.args[1]), {"layers.0.proj.weight"})
        self.assertFalse(call.call_args.kwargs["strict"])
        inputs = tokens.float().unsqueeze(-1).expand(-1, -1, 2) * .1 + model.model.offset
        self.assertTrue(torch.equal(primal, inputs))
        self.assertTrue(torch.allclose(response, inputs @ vector.T))
        self.assertFalse(primal.requires_grad)
        self.assertFalse(response.requires_grad)
        rollback.verify_model_unchanged(model, before, torch)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


    def test_local_model_loader_keeps_default_attention_and_fp32(self):
        import types
        fake = types.ModuleType("transformers")
        model = MockQwen()
        calls = []
        def load(path, **kwargs):
            calls.append((path, kwargs))
            return model
        fake.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load)
        with patch.dict(sys.modules, {"transformers": fake}), patch.dict(rollback.os.environ):
            result = rollback.load_local_model(Path("synthetic-merged"), "cpu", torch)
            self.assertEqual(rollback.os.environ["HF_HUB_OFFLINE"], "1")
        self.assertIs(result, model)
        self.assertFalse(model.training)
        self.assertEqual(calls, [(Path("synthetic-merged"), {"dtype": torch.float32, "local_files_only": True})])

    def test_nonlinear_jvp_matches_centered_finite_difference_positive_sign(self):
        model, spec = TinyModel(nonlinear=True), tiny_spec()
        tokens = torch.tensor([[0, 1, 2]])
        vector = torch.tensor([[.3, -.2], [.5, .1]])
        name = "model.layers.0.proj.weight"
        primal, response = rollback.run_readout_jvp(model, tokens, {name: vector}, spec, torch)
        weight = model.model.layers[0].proj.weight.detach()
        epsilon = 1e-3
        with torch.no_grad():
            def evaluate(value):
                return torch.func.functional_call(model.model, {"layers.0.proj.weight": value}, (),
                    dict(input_ids=tokens, use_cache=False, output_hidden_states=True, return_dict=True)).hidden_states[14]
            plus, minus = evaluate(weight + epsilon * vector), evaluate(weight - epsilon * vector)
        reference = (plus - minus) / (2 * epsilon)
        self.assertTrue(torch.allclose(response, reference, atol=5e-5, rtol=1e-3))
        self.assertGreater(float((response * (primal - minus)).sum()), 0.)

    def test_hidden_state_index_matches_block13_output_not_input_or_final_norm(self):
        model, spec = TinyModel(), tiny_spec()
        with torch.no_grad():
            model.model.layers[13].proj.weight.mul_(3.)
        tokens = torch.tensor([[0, 1, 2]])
        ordinary = rollback.ordinary_hook_readout(model, tokens, spec, torch)
        output = model.model(input_ids=tokens, use_cache=False, output_hidden_states=True, return_dict=True)
        self.assertTrue(torch.equal(ordinary, output.hidden_states[14]))
        self.assertFalse(torch.equal(ordinary, output.hidden_states[13]))
        self.assertFalse(torch.equal(ordinary, output.hidden_states[-1]))
        wrong = copy.deepcopy(spec)
        wrong["readout"]["hidden_state_index"] = 13
        with self.assertRaisesRegex(ValueError, "hidden-state index"):
            rollback.ordinary_hook_readout(model, tokens, wrong, torch)
        model.model.bad_index = True
        with self.assertRaisesRegex(ValueError, "does not equal block output hook"):
            rollback.ordinary_hook_readout(model, tokens, spec, torch)
        self.assertEqual(len(model.model.layers[13]._forward_hooks), 0)


    def test_parameter_and_buffer_mutation_audit(self):
        model = TinyModel()
        before = rollback.model_state_hashes(model, torch)
        with torch.no_grad():
            model.model.layers[0].proj.weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "changed"):
            rollback.verify_model_unchanged(model, before, torch)
        before = rollback.model_state_hashes(model, torch)
        model.model.offset.add_(1)
        with self.assertRaisesRegex(ValueError, "changed"):
            rollback.verify_model_unchanged(model, before, torch)


class ResponseMapTests(unittest.TestCase):
    def test_modulo_four_splits_exact_coverage_and_order(self):
        tokens = torch.arange(4096).reshape(-1, 1).expand(-1, 128)
        splits = rollback.interleaved_splits(tokens)
        self.assertEqual([len(split) for split in splits], [1024] * 4)
        all_indices = []
        for k, split in enumerate(splits):
            self.assertEqual(split[:, 0].tolist(), list(range(k, 4096, 4)))
            all_indices.extend(split[:, 0].tolist())
        self.assertEqual(sorted(all_indices), list(range(4096)))
        for count in (0, 3, 5):
            with self.assertRaises(ValueError):
                rollback.interleaved_splits(torch.zeros(count, 2))

    def test_split_normalization_and_mean_equal_full_gradient(self):
        spec = copy.deepcopy(rollback.FROZEN_SPEC)
        frozen = rollback.split_loss_spec(spec)
        self.assertEqual((frozen['sample_count'], frozen['total_prediction_tokens'], frozen['batch_size'], frozen['number_of_batches']), (1024, 130048, 8, 128))
        tokens = torch.arange(96).reshape(32, 3) % 3
        spec['generic_loss'].update(sample_count=32, sequence_length=3, predictions_per_sample=2,
                                    total_prediction_tokens=64, batch_size=8, number_of_batches=4)
        spec['splits'].update(sample_count=8, total_prediction_tokens=16, batch_size=8, number_of_batches=1)
        model = ToyCausalModel()
        grads = []
        for split in rollback.interleaved_splits(tokens):
            rollback.accumulate_mean_generic_gradient(model, split, [('proj', model.proj)], rollback.split_loss_spec(spec), torch)
            grads.append(model.proj.weight.grad.clone())
        reference = ToyCausalModel()
        rollback.accumulate_mean_generic_gradient(reference, tokens, [('proj', reference.proj)], spec['generic_loss'], torch)
        self.assertTrue(torch.allclose(torch.stack(grads).double().mean(0), reference.proj.weight.grad.double(), atol=1e-7))

    def test_gradient_and_jvp_machinery_matches_templates(self):
        current = ast.parse(SCRIPT_PATH.read_text())
        for source, names in [
            ('construct_generic_gradient_rollback.py', ['accumulate_mean_generic_gradient', 'causal_token_loss_sum']),
            ('construct_generic_gradient_linear_response.py', ['run_readout_jvp', 'ordinary_hook_readout', 'hidden_state_output', 'ResponseAccumulator'])]:
            original = ast.parse((PROJECT / 'scripts/ablation' / source).read_text())
            for name in names:
                left = next(node for node in original.body if getattr(node, 'name', None) == name)
                right = next(node for node in current.body if getattr(node, 'name', None) == name)
                self.assertEqual(ast.dump(left), ast.dump(right))

    def test_disk_backed_gram_mean_alpha_and_partition_masks(self):
        eligible = [('model.layers.0.self_attn.q_proj', torch.nn.Linear(2, 2, bias=False)),
                    ('model.layers.1.mlp.up_proj', torch.nn.Linear(2, 2, bias=False))]
        vectors = []
        for k in range(4):
            vectors.append([torch.tensor([[1., 2.], [3., 4.]]) * (k + 1),
                            torch.tensor([[4., 3.], [2., 1.]]) * (2 - k)])
        before = [module.weight.detach().clone() for _, module in eligible]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'temporary'
            rollback.prepare_temporary_directory(root)
            for k in range(4):
                for (_, module), value in zip(eligible, vectors[k]):
                    module.weight.grad = value.clone()
                rollback.save_split_gradient(root / f'split_{k}.safetensors', eligible, torch)
                self.assertTrue(all(module.weight.grad is None for _, module in eligible))
            with rollback.SplitGradientStore(root, eligible, torch) as store:
                diagnostics, norms = rollback.weight_space_statistics(store, eligible, torch)
                concatenated = torch.stack([torch.cat([x.flatten() for x in row]).double() for row in vectors])
                expected = rollback.gram_summary(concatenated @ concatenated.T, torch)
                self.assertEqual(diagnostics['global'], expected)
                for index, block in enumerate((0, 1)):
                    matrix = torch.stack([row[index].flatten().double() for row in vectors])
                    self.assertEqual(diagnostics['blocks'][block]['norms'], matrix.norm(dim=1).tolist())
                self.assertAlmostEqual(norms['aggregate_generic_gradient_norm'], float(concatenated.mean(0).norm()))
                scale = rollback.tangent_scale(norms, .00125)
                self.assertAlmostEqual(scale['alpha'] * norms['aggregate_generic_gradient_norm'] / norms['aggregate_source_weight_norm'], .00125, places=16)
                full, records = rollback.make_tangent(store, eligible, scale['alpha'], torch)
                for k in range(4):
                    tangent, _ = rollback.make_tangent(store, eligible, scale['alpha'], torch, split=k)
                    for (name, _), gradient in zip(eligible, vectors[k]):
                        self.assertTrue(torch.equal(tangent[name + '.weight'], (gradient.double() * scale['alpha']).float()))
                block_parts = [rollback.make_tangent(store, eligible, scale['alpha'], torch, block=k)[0] for k in (0, 1)]
                family_parts = [rollback.make_tangent(store, eligible, scale['alpha'], torch, family=k)[0] for k in ('attention', 'mlp')]
                for parts in (block_parts, family_parts):
                    self.assertFalse(set(parts[0]) & set(parts[1]))
                    self.assertEqual(set(parts[0]) | set(parts[1]), set(full))
                    for part in parts:
                        for name, value in part.items():
                            self.assertTrue(torch.equal(full[name], value))
                for (name, _), record in zip(eligible, records):
                    self.assertEqual(record['tangent_frobenius_norm'], float(full[name + '.weight'].double().norm()))
            rollback.cleanup_temporary_directory(root)
            self.assertFalse(root.exists())
        self.assertTrue(all(torch.equal(value, module.weight) for value, (_, module) in zip(before, eligible)))

    def test_temp_failure_retention_overwrite_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'temp'
            rollback.prepare_temporary_directory(root)
            rollback.temporary_stage(root, 'split_gradient_2')
            self.assertEqual(json.loads((root / 'state.json').read_text())['stage'], 'split_gradient_2')
            with self.assertRaisesRegex(ValueError, 'stale'):
                rollback.prepare_temporary_directory(root)
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                rollback.cleanup_temporary_directory(root)
            self.assertTrue((root / 'state.json').exists())
            module = torch.nn.Linear(2, 2, bias=False)
            with self.assertRaisesRegex(ValueError, 'Missing'):
                rollback.save_split_gradient(root / 'split_0.safetensors', [('model.layers.0.mlp.up_proj', module)], torch)
            module.weight.grad = torch.full_like(module.weight, float('inf'))
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                rollback.save_split_gradient(root / 'split_0.safetensors', [('model.layers.0.mlp.up_proj', module)], torch)

    def test_zero_cosines_cancellation_and_svd(self):
        zero = torch.zeros(3, dtype=torch.float64)
        self.assertIsNone(rollback.cosine(zero, zero))
        self.assertIsNone(rollback.cosine(zero, torch.ones_like(zero)))
        for matrix, rank, top in [(torch.eye(4, dtype=torch.float64), 4., .25),
                                  (torch.ones(4, 2, dtype=torch.float64), 1., 1.),
                                  (torch.zeros(4, 2, dtype=torch.float64), 0., 0.)]:
            stats = rollback.spectral_statistics(matrix, torch)
            self.assertAlmostEqual(stats['participation_ratio_effective_rank'], rank)
            self.assertAlmostEqual(stats['top1_variance_fraction'], top)
            self.assertAlmostEqual(sum(stats['squared_singular_value_fractions']), 1. if rank else 0.)
        artifact = {'full_response': torch.zeros(128, 2), 'split_responses': torch.zeros(4, 128, 2),
                    'block_responses': torch.zeros(14, 128, 2), 'family_responses': torch.zeros(2, 128, 2)}
        diagnostics = rollback.activation_diagnostics(artifact, torch)
        for record in diagnostics['positions']:
            self.assertIsNone(record['block']['cancellation_ratio'])
            self.assertIsNone(record['family']['mutual_cosine'])
            self.assertEqual(record['split']['off_diagonal_pairs']['undefined_count'], 6)
        self.assertEqual(diagnostics['split_descriptive_summaries']['positions_1_through_127']['pairwise_cosines']['undefined_count'], 127 * 6)
        json.dumps(diagnostics, allow_nan=False)

    def test_unequal_singular_values_use_squared_energy_and_fourth_power_rank(self):
        matrix = torch.diag(torch.tensor([3., 1.], dtype=torch.float64))
        result = rollback.spectral_statistics(matrix, torch)
        self.assertEqual(result['singular_values'], [3., 1.])
        self.assertEqual(result['squared_singular_value_fractions'], [.9, .1])
        self.assertEqual(result['top1_variance_fraction'], .9)
        self.assertAlmostEqual(result['participation_ratio_effective_rank'], 100 / 82)

    def test_frozen_corpus_hash_pins_and_canonical_inventory(self):
        spec = rollback.load_spec(SPEC_PATH)
        corpus_manifest_path = PROJECT / spec['defaults']['corpus_manifest_path']
        manifest = json.loads(corpus_manifest_path.read_text())
        self.assertEqual(spec['frozen_corpus']['corpus_spec_sha256'], rollback.sha256_file(CORPUS_SPEC_PATH))
        self.assertEqual(spec['frozen_corpus']['corpus_manifest_sha256'], rollback.sha256_file(corpus_manifest_path))
        for key in ('serialized_sha256', 'raw_tensor_sha256'):
            self.assertEqual(spec['frozen_corpus'][key], manifest['artifact'][key])
        self.assertEqual(spec['canonical_checkpoint_files'], sorted(spec['canonical_checkpoint_files'], key=lambda record: record['path']))
        self.assertEqual(len(spec['canonical_checkpoint_files']), 6)

    def test_stored_equivalent_diagnostics_and_cancellation(self):
        artifact = {'full_response': torch.ones(128, 2), 'split_responses': torch.ones(4, 128, 2),
                    'block_responses': torch.zeros(14, 128, 2), 'family_responses': torch.zeros(2, 128, 2)}
        artifact['block_responses'][0] = 2; artifact['block_responses'][1] = -1
        artifact['family_responses'][0] = 2; artifact['family_responses'][1] = -1
        result = rollback.activation_diagnostics(artifact, torch)
        self.assertEqual([row['position'] for row in result['positions']], list(range(128)))
        for row in result['positions']:
            self.assertAlmostEqual(row['block']['cancellation_ratio'], 1 / 3)
            self.assertAlmostEqual(row['family']['mutual_cosine'], -1)
            self.assertAlmostEqual(row['block']['leave_one_block_out_cosine'][0], -1)
            self.assertEqual(row['block']['cosine_with_full'][2], None)
            self.assertAlmostEqual(row['effective_dimensionality']['split']['participation_ratio_effective_rank'], 1)
        self.assertEqual(result, rollback.activation_diagnostics(artifact, torch))

    def test_consistency_thresholds_and_float32_artifact_contract(self):
        spec = tiny_spec()
        merged = torch.ones(3, 2, dtype=torch.float64)
        splits = torch.ones(4, 3, 2, dtype=torch.float64)
        blocks = torch.zeros(14, 3, 2, dtype=torch.float64); blocks[0] = 1
        families = torch.zeros(2, 3, 2, dtype=torch.float64); families[0] = 1
        artifact, checks = rollback.build_artifact(merged, merged.clone(), splits, blocks, families, spec, torch)
        self.assertEqual(set(artifact), {'metadata', *spec['outputs']['tensors']})
        self.assertEqual(artifact['metadata']['family_order'], ['attention', 'mlp'])
        self.assertEqual(artifact['metadata']['split_order'], [0, 1, 2, 3])
        for name in spec['outputs']['tensors']:
            value = artifact[name]
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.device.type, 'cpu')
            self.assertTrue(value.is_contiguous())
        self.assertEqual(checks['block_sum']['worst_relative_rms_difference'], 0.)
        settings = spec['decomposition']['consistency']
        # No raw absolute threshold: this error is large in absolute units but tiny relatively.
        full = torch.full_like(merged, 1e3)
        metrics = rollback.consistency_metrics(full, full + .01, settings, torch)
        self.assertGreater(metrics['maximum_absolute_difference'], 1e-5)
        for scale in (1e-6, 1., 1e6):
            full = torch.full_like(merged, scale)
            rollback.consistency_metrics(full, full * (1 + 5e-5), settings, torch)
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                rollback.consistency_metrics(full, full * (1 + 2e-4), settings, torch)
        zero = torch.zeros_like(merged)
        self.assertEqual(rollback.consistency_metrics(zero, zero, settings, torch)['worst_relative_rms_difference'], 0)
        with self.assertRaises(ValueError):
            rollback.build_artifact(merged.float(), merged.clone(), splits, blocks, families, spec, torch)
        families[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            rollback.build_artifact(merged, merged.clone(), splits, blocks, families, spec, torch)

    def test_relative_decomposition_thresholds_independently_and_zero_reference(self):
        settings = rollback.FROZEN_SPEC['decomposition']['consistency']
        # Sparse error: relative max fails while relative RMS passes.
        full = torch.ones(2, 100, dtype=torch.float64)
        candidate = full.clone(); candidate[0, 0] += 2e-4
        metrics = rollback.response_comparison(full, candidate, torch)
        self.assertLess(metrics['relative_rms_difference'], 1e-4)
        self.assertGreater(metrics['relative_max_difference'], 1e-4)
        with self.assertRaisesRegex(ValueError, 'relative_max_difference'):
            rollback.consistency_metrics(full, candidate, settings, torch)
        # Sparse reference, widespread small error: relative RMS alone fails.
        full.zero_(); full[0, 0] = 1.
        candidate = full + 2e-5
        metrics = rollback.response_comparison(full, candidate, torch)
        self.assertGreater(metrics['relative_rms_difference'], 1e-4)
        self.assertLess(metrics['relative_max_difference'], 1e-4)
        with self.assertRaisesRegex(ValueError, 'relative_rms_difference'):
            rollback.consistency_metrics(full, candidate, settings, torch)
        zero = torch.zeros_like(full)
        exact = rollback.consistency_metrics(zero, zero, settings, torch)
        self.assertEqual(exact['worst_relative_max_difference'], 0.)
        self.assertIsNone(exact['positions'][0]['cosine_similarity'])
        # Diagnostic comparisons must remain serializable even when ratios are undefined.
        diagnostic = rollback.response_comparison(zero, candidate, torch)
        self.assertIsNone(diagnostic['relative_rms_difference'])
        self.assertIsNone(diagnostic['relative_max_difference'])
        json.dumps(diagnostic, allow_nan=False)
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            rollback.consistency_metrics(zero, candidate, settings, torch)
        for invalid in (torch.ones(3, 2, dtype=torch.float64), full.float(), full * float('nan')):
            with self.assertRaises(ValueError):
                rollback.response_comparison(full, invalid, torch)

    def test_position_zero_cannot_mask_later_decomposition_errors(self):
        settings = rollback.FROZEN_SPEC['decomposition']['consistency']
        self.assertEqual(settings['scope'], 'per_position_all_must_pass')
        full = torch.ones(128, 2, dtype=torch.float64)
        full[0] = 1e12
        candidate = full.clone(); candidate[127] += .01
        old = rollback.response_comparison(full, candidate, torch)
        self.assertLess(old['relative_rms_difference'], 1e-4)
        self.assertLess(old['relative_max_difference'], 1e-4)
        with self.assertRaisesRegex(ValueError, 'position 127'):
            rollback.consistency_metrics(full, candidate, settings, torch)
        # Both artifact decomposition checks must use the position-wise gate.
        spec = tiny_spec(length=128)
        splits = full.unsqueeze(0).repeat(4, 1, 1)
        blocks = torch.zeros(14, 128, 2, dtype=torch.float64); blocks[0] = full
        families = torch.zeros(2, 128, 2, dtype=torch.float64); families[0] = full
        for group in ('block', 'family'):
            with self.subTest(group=group):
                bad_blocks, bad_families = blocks.clone(), families.clone()
                (bad_blocks if group == 'block' else bad_families)[0, 127] += .01
                with self.assertRaisesRegex(ValueError, 'position 127'):
                    rollback.build_artifact(full, full, splits, bad_blocks, bad_families, spec, torch)
        # The split diagnostic remains unrestricted, including later-position discrepancies.
        splits[:, 127] += .01
        _, checks = rollback.build_artifact(full, full, splits, blocks, families, spec, torch)
        self.assertEqual(checks['split_mean_vs_direct_full'], rollback.response_comparison(full, candidate, torch))

    def test_position_metrics_worst_positions_ties_and_local_zero_policy(self):
        settings = rollback.FROZEN_SPEC['decomposition']['consistency']
        full = torch.ones(128, 4, dtype=torch.float64)
        full[0] = 1e12
        full[9] = 0
        candidate = full.clone()
        candidate[0] += 1  # Large raw error, tiny relative error.
        candidate[5] += 6e-5
        candidate[7, 0] += 9e-5
        candidate[8, 0] += 9e-5  # Tie: position 7 wins.
        metrics = rollback.consistency_metrics(full, candidate, settings, torch)
        self.assertEqual([row['position'] for row in metrics['positions']], list(range(128)))
        self.assertEqual(metrics['worst_relative_rms_position'], 5)
        self.assertEqual(metrics['worst_relative_max_position'], 7)
        self.assertAlmostEqual(metrics['worst_relative_rms_difference'], 6e-5)
        self.assertAlmostEqual(metrics['worst_relative_max_difference'], 9e-5)
        self.assertEqual(metrics['maximum_absolute_difference'], 1.)
        for position, row in enumerate(metrics['positions']):
            expected = rollback.response_comparison(full[position:position + 1], candidate[position:position + 1], torch)
            for key, value in expected.items():
                self.assertEqual(row[key], value)
            self.assertEqual(row['reference_rms'], float(full[position].square().mean().sqrt()))
            self.assertEqual(row['reference_max_absolute_value'], float(full[position].abs().max()))
        self.assertEqual(metrics['positions'][9]['relative_rms_difference'], 0.)
        self.assertEqual(metrics['positions'][9]['relative_max_difference'], 0.)
        self.assertEqual(metrics, rollback.consistency_metrics(full, candidate, settings, torch))
        json.dumps(metrics, allow_nan=False)
        candidate[9, 0] = 1e-20
        with self.assertRaisesRegex(ValueError, 'position 9'):
            rollback.consistency_metrics(full, candidate, settings, torch)
        exact = rollback.consistency_metrics(full, full, settings, torch)
        self.assertEqual(exact['worst_relative_rms_position'], 0)
        self.assertEqual(exact['worst_relative_max_position'], 0)

    def test_fp32_split_rounding_is_diagnostic_not_decomposition_error(self):
        class IdentityFamilyBlock(torch.nn.Module):
            def __init__(self, family, projection):
                super().__init__()
                self.family, self.projection = family, projection
                self.add_module(family, torch.nn.ModuleDict({projection: torch.nn.Linear(2, 2, bias=False)}))
                with torch.no_grad():
                    self.get_submodule(f'{family}.{projection}').weight.copy_(torch.eye(2))

            def forward(self, hidden):
                return self.get_submodule(f'{self.family}.{self.projection}')(hidden)

        model, spec = TinyModel(), tiny_spec()
        model.model.layers[0] = IdentityFamilyBlock('self_attn', 'q_proj')
        model.model.layers[1] = IdentityFamilyBlock('mlp', 'up_proj')
        eligible = [('model.layers.0.self_attn.q_proj', model.model.layers[0].self_attn['q_proj']),
                    ('model.layers.1.mlp.up_proj', model.model.layers[1].mlp['up_proj'])]
        alpha = .1
        # All gradients are exactly representable FP32; scaling rounds away part of the cancellation.
        scalars = [100000000., -100000008., 1., 1.]
        probe = torch.zeros(1, 3, dtype=torch.int64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for k, scalar in enumerate(scalars):
                for _, module in eligible:
                    module.weight.grad = torch.eye(2) * scalar
                rollback.save_split_gradient(root / f'split_{k}.safetensors', eligible, torch)
            with rollback.SplitGradientStore(root, eligible, torch) as store:
                canonical, _ = rollback.make_tangent(store, eligible, alpha, torch)
                split_tangents = [rollback.make_tangent(store, eligible, alpha, torch, split=k)[0] for k in range(4)]
                name = eligible[0][0] + '.weight'
                rounded_mean = torch.stack([tangent[name].double() for tangent in split_tangents]).mean(0)
                self.assertFalse(torch.equal(rounded_mean, canonical[name].double()))
                expected = (torch.eye(2, dtype=torch.float64) * (sum(scalars) / 4) * alpha).float()
                self.assertTrue(torch.equal(canonical[name], expected))
                primal, direct = rollback.run_readout_jvp(model, probe, canonical, spec, torch)
                split_responses = torch.stack([rollback.run_readout_jvp(model, probe, tangent, spec, torch)[1].double().mean(0)
                                               for tangent in split_tangents])
                blocks = torch.zeros(14, 3, 2, dtype=torch.float64)
                families = []
                for block in (0, 1):
                    part = rollback.partition_tangent(canonical, block=block)
                    for key, value in part.items():
                        self.assertIs(value, canonical[key])
                    blocks[block] = rollback.run_readout_jvp(model, probe, part, spec, torch)[1].double().mean(0)
                for family in ('attention', 'mlp'):
                    part = rollback.partition_tangent(canonical, family=family)
                    families.append(rollback.run_readout_jvp(model, probe, part, spec, torch)[1].double().mean(0))
        full = direct.double().mean(0)
        artifact, checks = rollback.build_artifact(primal.double().mean(0), full, split_responses,
                                                  blocks, torch.stack(families), spec, torch)
        self.assertTrue(torch.equal(artifact['full_response'], full.float()))
        self.assertFalse(torch.equal(artifact['full_response'], split_responses.mean(0).float()))
        self.assertEqual(artifact['metadata']['split_mean_role'], 'diagnostic_only')
        self.assertEqual(checks['block_sum']['maximum_absolute_difference'], 0.)
        self.assertEqual(checks['family_sum']['maximum_absolute_difference'], 0.)
        metrics = checks['split_mean_vs_direct_full']
        self.assertGreater(metrics['relative_rms_difference'], 1e-4)
        self.assertGreater(metrics['relative_max_difference'], 1e-4)
        self.assertAlmostEqual(metrics['cosine_similarity'], 1.)
        error = split_responses.mean(0) - full
        self.assertEqual(metrics['rms_difference'], float(error.square().mean().sqrt()))
        self.assertEqual(metrics['maximum_absolute_difference'], float(error.abs().max()))
        self.assertEqual(metrics['relative_max_difference'], float(error.abs().max() / full.abs().max()))
        # The same discrepancy would fail a decomposition gate, but artifact construction succeeded.
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            rollback.consistency_metrics(full, split_responses.mean(0), spec['decomposition']['consistency'], torch)
        self.assertNotIn('mean_split_response', artifact)

    def test_native_jvp_decomposition_linearity_and_cpu_accumulation(self):
        model, spec = TinyModel(nonlinear=True), tiny_spec(count=4, length=3, batch=2)
        spec['diagnostic_probe'].update(sample_count=4, batch_size=2)
        probe = torch.arange(12).reshape(4, 3)
        names = [f'model.layers.{block}.proj.weight' for block in range(14)]
        splits = [{name: torch.full((2, 2), (k + 1) * (block + 1) * 1e-5)
                   for block, name in enumerate(names)} for k in range(4)]
        full = {name: torch.stack([split[name] for split in splits]).mean(0) for name in names}
        responses = []
        accumulator = rollback.ResponseAccumulator(spec, torch)
        with patch('sys.stdout', new=io.StringIO()):
            for tangent in splits:
                primal, response, _ = rollback.compute_probe_response(model, probe, tangent, spec, torch)
                self.assertEqual(response.dtype, torch.float64)
                responses.append(response)
            _, reference, _ = rollback.compute_probe_response(model, probe, full, spec, torch)
            blocks = [rollback.compute_probe_response(model, probe, {name: full[name]}, spec, torch)[1] for name in names]
            families = [rollback.compute_probe_response(model, probe, {name: full[name] for name in names[k::2]}, spec, torch)[1] for k in range(2)]
        rollback.response_comparison(reference, torch.stack(responses).mean(0), torch)
        for candidate in (torch.stack(blocks).sum(0), sum(families)):
            rollback.consistency_metrics(reference, candidate, spec['decomposition']['consistency'], torch)
        ordinary = rollback.ordinary_hook_readout(model, probe, spec, torch)
        self.assertTrue(torch.allclose(primal, ordinary.double().mean(0)))
        _, tangent = rollback.run_readout_jvp(model, probe, full, spec, torch)
        accumulator.add(ordinary, tangent)
        self.assertTrue(torch.equal(accumulator.means(4)[1], tangent.double().mean(0)))
        self.assertGreater(float(reference.norm()), 0)

    def test_no_forward_ad_fallback(self):
        model, spec = TinyModel(), tiny_spec()
        tangent = {'model.layers.0.proj.weight': torch.ones(2, 2)}
        with patch.object(torch.func, 'jvp', side_effect=NotImplementedError('unsupported')):
            with self.assertRaisesRegex(RuntimeError, 'no fallback'):
                rollback.run_readout_jvp(model, torch.zeros(1, 3, dtype=torch.int64), tangent, spec, torch)
        model.model.detach_output = True
        with self.assertRaisesRegex(RuntimeError, 'no fallback'):
            rollback.run_readout_jvp(model, torch.zeros(1, 3, dtype=torch.int64), tangent, spec, torch)

    def test_information_firewall_and_frozen_probe_subset(self):
        source = SCRIPT_PATH.read_text().replace('oracle_probe', 'generic_probe')
        for forbidden in ('oracle', 'adapter', 'models/base', 'evaluation', 'self_diff', 'self-diff',
                          'attempt008', 'attempt007', 'attempt006', 'linear_response.pt', 'save_pretrained',
                          'attn_implementation=', 'jacrev', 'jacfwd', 'torch.optim'):
            self.assertNotIn(forbidden, source)
        self.assertEqual(rollback.FROZEN_SPEC['diagnostic_probe'], dict(start=0, stop=1024, sample_count=1024, batch_size=32))
        self.assertEqual(rollback.FROZEN_SPEC['purpose'], 'blind_diagnostic_only_not_a_recovery_candidate')


class FamilyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2, bias=False)
                                             for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')})
        self.mlp = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2, bias=False)
                                      for name in ('gate_proj', 'up_proj', 'down_proj')})
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(.01)

    def forward(self, hidden):
        return hidden + sum(layer(hidden) for group in (self.self_attn, self.mlp) for layer in group.values()) * .01


class FamilyModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.model.layers = torch.nn.ModuleList([FamilyBlock() for _ in range(28)])
        self.head = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[.2, -.1], [.1, .3], [-.2, .5]]))

    def forward(self, *, input_ids, use_cache):
        hidden = self.model(input_ids=input_ids, use_cache=use_cache, output_hidden_states=True, return_dict=True)
        return SimpleNamespace(logits=self.head(hidden.hidden_states[-1]))


class ResponseMapPipelineTests(unittest.TestCase):
    def fixture(self, root):
        args = rollback.parse_args([])
        for attr, filename in [('artifact_path', 'response.pt'), ('construction_manifest_path', 'manifest.json'),
                               ('attempt_spec_path', 'spec.json'), ('tokens_path', 'tokens.pt'),
                               ('probe_path', 'probe.pt'), ('corpus_spec_path', 'corpus-spec.json'),
                               ('corpus_manifest_path', 'corpus-manifest.json'), ('temporary_directory', 'temp')]:
            setattr(args, attr, root / filename)
        args.merged_model_dir = root / 'merged'; args.merged_model_dir.mkdir()
        (args.merged_model_dir / 'weights').write_bytes(b'synthetic canonical checkpoint')
        args.device = 'cpu'
        tokens = torch.arange(24).reshape(8, 3) % 3
        probe = torch.arange(18).reshape(6, 3) % 3
        torch.save(tokens, args.tokens_path); torch.save(probe, args.probe_path)
        args.corpus_spec_path.write_text('{}'); args.corpus_manifest_path.write_text('{}')
        spec = copy.deepcopy(rollback.FROZEN_SPEC)
        spec['generic_loss'].update(sample_count=8, sequence_length=3, predictions_per_sample=2,
                                    total_prediction_tokens=16, batch_size=1, number_of_batches=8)
        spec['splits'].update(sample_count=2, total_prediction_tokens=4, batch_size=1, number_of_batches=2)
        spec['probe'].update(sample_count=6, sequence_length=3, batch_size=2,
                             serialized_sha256=rollback.sha256_file(args.probe_path),
                             raw_tensor_sha256=rollback.sha256_raw_int64_tensor(probe, torch))
        spec['diagnostic_probe'].update(start=0, stop=4, sample_count=4, batch_size=2)
        spec['readout']['hidden_size'] = 2
        spec['canonical_checkpoint_files'] = rollback.checkpoint_file_records(args.merged_model_dir)
        corpus = {'serialized_sha256': rollback.sha256_file(args.tokens_path),
                  'raw_tensor_sha256': rollback.sha256_raw_int64_tensor(tokens, torch)}
        spec['frozen_corpus'] = {**corpus, 'corpus_spec_sha256': rollback.sha256_file(args.corpus_spec_path),
                                'corpus_manifest_sha256': rollback.sha256_file(args.corpus_manifest_path)}
        args.attempt_spec_path.write_text(json.dumps(spec))
        return args, spec, corpus

    def run_fixture(self, args, spec, corpus, model=None):
        with patch.object(rollback, 'FROZEN_SPEC', spec), patch.object(rollback, 'load_corpus_spec', return_value={}), \
             patch.object(rollback, 'verify_corpus_manifest', return_value=corpus), \
             patch.object(rollback, 'load_local_model', return_value=model or FamilyModel()), \
             patch('sys.stdout', new=io.StringIO()):
            rollback.construct(args)

    def test_full_synthetic_pipeline_hashes_cleanup_and_deterministic_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, corpus = self.fixture(Path(directory))
            model = FamilyModel()
            before = rollback.model_state_hashes(model, torch)
            with patch.object(rollback, 'compute_probe_response', wraps=rollback.compute_probe_response) as responses, \
                 patch.object(rollback, 'run_readout_jvp', wraps=rollback.run_readout_jvp) as jvp:
                self.run_fixture(args, spec, corpus, model)
            self.assertEqual(responses.call_count, 21)
            self.assertEqual(jvp.call_count, 21 * 2)  # Tiny probe: two batches per direction.
            canonical = responses.call_args_list[4].args[2]
            self.assertEqual(responses.call_args_list[4].args[5], 'full')
            for calls in (responses.call_args_list[5:19], responses.call_args_list[19:21]):
                seen = set()
                for call in calls:
                    subset = call.args[2]
                    self.assertFalse(seen & set(subset))
                    seen.update(subset)
                    for name, value in subset.items():
                        self.assertIs(value, canonical[name])
                self.assertEqual(seen, set(canonical))
            self.assertFalse(args.temporary_directory.exists())
            rollback.verify_model_unchanged(model, before, torch)
            artifact = torch.load(args.artifact_path, weights_only=True)
            manifest = json.loads(args.construction_manifest_path.read_text())
            self.assertEqual(artifact['metadata']['diagnostic_probe'], spec['diagnostic_probe'])
            self.assertEqual(len(manifest['readout']['checks']), 21)
            self.assertEqual(manifest['decomposition']['batched_jvp_calls'], 672)
            self.assertEqual(manifest['decomposition']['response_direction_count'], 21)
            self.assertFalse(manifest['decomposition']['split_mean_diagnostic']['thresholded'])
            self.assertEqual(artifact['metadata']['full_response_definition'], spec['decomposition']['full_response'])
            self.assertEqual(set(manifest['decomposition']['measured_consistency']['split_mean_vs_direct_full']),
                             set(spec['decomposition']['split_mean_diagnostic']['metrics']))
            self.assertEqual(len(manifest['matrices']), 98)
            self.assertEqual(manifest['artifact']['serialized_sha256'], rollback.sha256_file(args.artifact_path))
            for name in spec['outputs']['tensors']:
                self.assertEqual(manifest['artifact']['raw_tensors_sha256'][name], rollback.sha256_raw_float32_tensor(artifact[name], torch))
            scale = manifest['tangent']
            self.assertAlmostEqual(scale['alpha'] * scale['aggregate_generic_gradient_norm'] / scale['aggregate_source_weight_norm'], .00125, places=16)
            self.assertAlmostEqual(scale['aggregate_realized_relative_tangent_norm'], .00125, places=9)
            for label in ('block_sum', 'family_sum'):
                self.assertLessEqual(manifest['decomposition']['measured_consistency'][label]['worst_relative_rms_difference'], 1e-4)
            self.assertNotIn('.safetensors', json.dumps({key: value for key, value in manifest.items() if key != 'source_checkpoint'}))
            first = args.construction_manifest_path.read_bytes()
            args.artifact_path.unlink(); args.construction_manifest_path.unlink()
            self.run_fixture(args, spec, corpus)
            self.assertEqual(first, args.construction_manifest_path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.run_fixture(args, spec, corpus)

    def test_failure_retains_temp_state_and_refuses_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, corpus = self.fixture(Path(directory))
            with patch.object(rollback, 'compute_probe_response', side_effect=RuntimeError('synthetic failure')):
                with self.assertRaisesRegex(RuntimeError, 'synthetic failure'):
                    self.run_fixture(args, spec, corpus)
            self.assertEqual(len(list(args.temporary_directory.glob('*.safetensors'))), 4)
            self.assertEqual(json.loads((args.temporary_directory / 'state.json').read_text())['stage'], 'jvp_split 0')
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.construction_manifest_path.exists())
            with self.assertRaisesRegex(ValueError, 'stale'):
                self.run_fixture(args, spec, corpus)

    def test_checkpoint_and_frozen_corpus_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, corpus = self.fixture(Path(directory))
            (args.merged_model_dir / 'weights').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'inventory/hash mismatch'):
                self.run_fixture(args, spec, corpus)
            self.assertFalse(args.temporary_directory.exists())
            spec['canonical_checkpoint_files'] = rollback.checkpoint_file_records(args.merged_model_dir)
            args.attempt_spec_path.write_text(json.dumps(spec))
            args.corpus_manifest_path.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Frozen corpus provenance mismatch'):
                self.run_fixture(args, spec, corpus)


if __name__ == '__main__':
    unittest.main()
