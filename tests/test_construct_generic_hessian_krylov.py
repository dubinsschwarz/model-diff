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
ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/011_generic_hessian_krylov_k4_prefix0_13"
SCRIPT_PATH = PROJECT / "scripts/ablation/construct_generic_hessian_krylov.py"
SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
CORPUS_SPEC_PATH = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json"
MODULE_SPEC = importlib.util.spec_from_file_location("construct_generic_hessian_krylov", SCRIPT_PATH)
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


class CurvatureTemplateTests(unittest.TestCase):
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


    def test_no_forward_ad_fallback(self):
        model, spec = TinyModel(), tiny_spec()
        tangent = {'model.layers.0.proj.weight': torch.ones(2, 2)}
        with patch.object(torch.func, 'jvp', side_effect=NotImplementedError('unsupported')):
            with self.assertRaisesRegex(RuntimeError, 'no fallback'):
                rollback.run_readout_jvp(model, torch.zeros(1, 3, dtype=torch.int64), tangent, spec, torch)
        model.model.detach_output = True
        with self.assertRaisesRegex(RuntimeError, 'no fallback'):
            rollback.run_readout_jvp(model, torch.zeros(1, 3, dtype=torch.int64), tangent, spec, torch)


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
            for index, parameter in enumerate(self.model.parameters()):
                parameter.copy_(.05 * torch.sin(torch.arange(parameter.numel()).float() + index).reshape_as(parameter))

    def forward(self, *, input_ids, use_cache):
        hidden = self.model(input_ids=input_ids, use_cache=use_cache, output_hidden_states=True, return_dict=True)
        return SimpleNamespace(logits=self.head(hidden.hidden_states[-1]))


class CurvaturePipelineTests(unittest.TestCase):
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
        args.basis_directory = root / 'basis'
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


    def test_complete_pipeline_persists_basis_hashes_and_has_fixed_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, corpus = self.fixture(Path(directory))
            model = FamilyModel()
            before = rollback.model_state_hashes(model, torch)
            helper = rollback.hvp_backend.true_hessian_vector_product
            seen = []
            def checked(loss, parameters, direction):
                seen.append({name: value.detach().cpu().clone() for name, value in direction.items()})
                return helper(loss, parameters, direction)
            with patch.object(rollback.hvp_backend, 'true_hessian_vector_product', side_effect=checked), \
                 patch.object(rollback, 'run_readout_jvp', wraps=rollback.run_readout_jvp) as jvp:
                self.run_fixture(args, spec, corpus, model)
            self.assertEqual(len(seen), 32)
            self.assertEqual(jvp.call_count, 8)
            self.assertFalse(args.temporary_directory.exists())
            rollback.verify_model_unchanged(model, before, torch)
            manifest = json.loads(args.construction_manifest_path.read_text())
            artifact = torch.load(args.artifact_path, weights_only=True)
            self.assertEqual(len(manifest['steps']), 4)
            self.assertEqual(len(manifest['beta_values']), 4)
            self.assertEqual(len(manifest['readout']['checks']), 4)
            self.assertEqual(len(manifest['eligible_parameters']), 98)
            self.assertEqual(manifest['hvp_implementation_sha256'], rollback.sha256_file(rollback.HVP_SOURCE_PATH))
            self.assertEqual(manifest['artifact']['serialized_sha256'], rollback.sha256_file(args.artifact_path))
            for name in spec['outputs']['tensors']:
                self.assertEqual(manifest['artifact']['raw_tensors_sha256'][name], rollback.sha256_raw_float32_tensor(artifact[name], torch))
            from safetensors import safe_open
            for index, record in enumerate(manifest['basis_files']):
                self.assertEqual(record['filename'], f'q{index + 1}.safetensors')
                self.assertEqual(record['sha256'], rollback.sha256_file(args.basis_directory / record['filename']))
                with safe_open(str(args.basis_directory / record['filename']), framework='pt') as file:
                    for name in file.keys():
                        value = file.get_tensor(name)
                        for action in seen[index * 8:(index + 1) * 8]:
                            self.assertTrue(torch.equal(value, action[name]))
            self.assertLessEqual(manifest['orthogonality']['maximum_absolute_off_diagonal'], 1e-5)
            self.assertLessEqual(manifest['orthogonality']['maximum_diagonal_deviation_from_one'], 1e-5)
            self.assertEqual(artifact['metadata']['direction_order'], [1, 2, 3, 4])
            first_manifest = args.construction_manifest_path.read_bytes()
            first_basis_hashes = [r['sha256'] for r in manifest['basis_files']]
            args.artifact_path.unlink(); args.construction_manifest_path.unlink()
            for path in args.basis_directory.iterdir():
                path.unlink()
            self.run_fixture(args, spec, corpus)
            self.assertEqual(first_manifest, args.construction_manifest_path.read_bytes())
            self.assertEqual(first_basis_hashes, [rollback.sha256_file(args.basis_directory / f'q{i}.safetensors') for i in range(1, 5)])
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.run_fixture(args, spec, corpus)

    def test_failure_retains_temporary_vectors_and_refuses_stale_state(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, corpus = self.fixture(Path(directory))
            with patch.object(rollback.hvp_backend, 'true_hessian_vector_product', side_effect=RuntimeError('unsupported derivative')):
                with self.assertRaisesRegex(RuntimeError, 'unsupported derivative'):
                    self.run_fixture(args, spec, corpus)
            self.assertTrue((args.temporary_directory / 'g.safetensors').exists())
            self.assertTrue((args.temporary_directory / 'q1.safetensors').exists())
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.construction_manifest_path.exists())
            with self.assertRaisesRegex(ValueError, 'stale'):
                self.run_fixture(args, spec, corpus)
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                rollback.cleanup_temporary_directory(args.temporary_directory)

    def test_manifest_publication_failure_does_not_delete_temporaries(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, corpus = self.fixture(Path(directory))
            with patch.object(rollback, 'write_manifest', side_effect=OSError('publication failure')):
                with self.assertRaisesRegex(OSError, 'publication failure'):
                    self.run_fixture(args, spec, corpus)
            self.assertFalse(args.construction_manifest_path.exists())
            self.assertTrue(args.artifact_path.exists())
            self.assertEqual(len(list(args.basis_directory.glob('*.safetensors'))), 4)
            self.assertEqual(len(list(args.temporary_directory.glob('*.safetensors'))), 0)
            self.assertEqual(len(list(args.temporary_directory.glob('step_*.json'))), 4)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.run_fixture(args, spec, corpus)


class CurvatureConstructionTests(unittest.TestCase):


    def test_hvp_uses_exact_shared_helper_and_never_falls_back(self):
        model = ToyCausalModel()
        eligible = [('proj', model.proj)]
        g = {'proj.weight': torch.ones_like(model.proj.weight)}
        settings = dict(sample_count=2, sequence_length=3, predictions_per_sample=2,
                        total_prediction_tokens=4, batch_size=1, number_of_batches=2)
        error = RuntimeError('derivative for synthetic_kernel_backward is not implemented')
        with patch.object(rollback.hvp_backend, 'true_hessian_vector_product', side_effect=error) as checked:
            with self.assertRaises(RuntimeError) as caught:
                rollback.accumulate_hessian_split(model, torch.zeros(2, 3, dtype=torch.int64), eligible, g, settings, torch)
        self.assertIs(caught.exception, error)
        checked.assert_called_once()
        self.assertTrue(torch.equal(g['proj.weight'], torch.ones_like(model.proj.weight)))
        self.assertIsNone(model.proj.weight.grad)




def toy_vectors():
    eligible = [('a', torch.nn.Linear(3, 1, bias=False)), ('b', torch.nn.Linear(3, 1, bias=False))]
    with torch.no_grad():
        for _, module in eligible:
            module.weight.zero_()
    parameters = {name + '.weight': module.weight for name, module in eligible}
    return eligible, parameters


def write_flat(root, key, flat, parameters):
    values = {name: flat[3 * i:3 * (i + 1)].reshape_as(parameter).float().contiguous()
              for i, (name, parameter) in enumerate(parameters.items())}
    rollback.save_vector(root / f'{key}.safetensors', values, parameters, torch)


def read_flat(store, key, eligible):
    return torch.cat([store.matrix(key, name, module.weight.shape).flatten() for name, module in eligible])


class KrylovMathTests(unittest.TestCase):
    def run_dense(self, root, eligible, parameters):
        gradient = torch.arange(1, 7, dtype=torch.float32)
        write_flat(root, 'g', gradient, parameters)
        operator = torch.diag(torch.tensor([-2., -.4, .7, 2., 4., 7.]))
        perturbation = torch.diag(torch.tensor([.2, -.1, .3, -.2, .1, .05]))
        split_operators = [operator + scale * perturbation for scale in (-.03, .01, .02, 0.)]
        observed, saved_actions = [], []
        with rollback.VectorStore(root, eligible, torch) as store:
            def compute(step, key):
                self.assertEqual(list(root.glob('z*_s*.safetensors')), [])
                self.assertFalse(any(key.startswith('z') for key in store.files))
                self.assertFalse(any(key.startswith('z') for key in store.contexts))
                for previous in range(1, step):
                    self.assertTrue((root / f'step_{previous}.json').exists())
                split_actions = []
                direction, _ = rollback.load_device_vector(store, key, eligible, torch)
                before = {name: value.clone() for name, value in direction.items()}
                for split, matrix in enumerate(split_operators):
                    def loss():
                        weights = torch.cat([value.flatten() for value in parameters.values()])
                        return .5 * weights @ matrix @ weights + gradient @ weights
                    action = rollback.hvp_backend.true_hessian_vector_product(loss, parameters, direction)
                    observed.append((step, split, id(direction), torch.cat([value.flatten() for value in direction.values()])))
                    for name in direction:
                        self.assertTrue(torch.equal(direction[name], before[name]))
                    split_actions.append(torch.cat([value.flatten().double() for value in action.values()]))
                    rollback.save_vector(root / f'z{step}_s{split}.safetensors', action, parameters, torch)
                    self.assertLessEqual(len(list(root.glob('*.safetensors'))), 9)
                saved_actions.append(sum(split_actions) / 4)
            norm, steps, gram, projected = rollback.build_krylov_basis(store, eligible, compute, rollback.FROZEN_SPEC, torch)
            basis = torch.stack([read_flat(store, f'q{i}', eligible) for i in range(1, 5)], dim=1)
            actions = torch.stack(saved_actions, dim=1)
            self.assertEqual(list(root.glob('z*_s*.safetensors')), [])
            self.assertEqual(len(list(root.glob('*.safetensors'))), 5)
        return norm, steps, gram, projected, basis, actions, operator, observed

    def test_true_hvp_k4_basis_orthogonality_projection_and_ritz(self):
        eligible, parameters = toy_vectors()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            norm, steps, gram, projected, q, z, operator, observed = self.run_dense(root, eligible, parameters)
            self.assertAlmostEqual(norm, math.sqrt(91))
            expected_q1 = (torch.arange(1, 7, dtype=torch.float64) / math.sqrt(91)).float().double()
            self.assertTrue(torch.equal(q[:, 0], expected_q1))
            torch.testing.assert_close(q.T @ q, torch.eye(4, dtype=torch.float64), rtol=0, atol=1e-6)
            self.assertEqual(gram['gram_matrix'], (q.T @ q).tolist())
            self.assertLessEqual(gram['maximum_absolute_off_diagonal'], 1e-5)
            self.assertLessEqual(gram['maximum_diagonal_deviation_from_one'], 1e-5)
            torch.testing.assert_close(z, operator.double() @ q, rtol=2e-6, atol=5e-7)
            for j, step in enumerate(steps):
                existing = q[:, :j + 1]
                residual = z[:, j].clone()
                for _ in range(2):
                    residual -= existing @ (existing.T @ residual)
                self.assertAlmostEqual(step['beta'], float(residual.norm()), places=12)
                self.assertEqual(len(step['projection_coefficients_by_pass_full_then_splits']), 2)
                self.assertLess(step['residuals']['relative_mean_residual_difference'], 1e-12)
                if j < 3:
                    torch.testing.assert_close(q[:, j + 1], (residual / residual.norm()).float().double(), rtol=0, atol=1e-7)
                seen = [item for item in observed if item[0] == j + 1]
                self.assertEqual(len(seen), 4)
                self.assertEqual(len({item[2] for item in seen}), 1)
                self.assertTrue(all(torch.equal(item[3].double(), q[:, j]) for item in seen))
            direct = q.T @ z
            b = torch.triu(direct) + torch.triu(direct, diagonal=1).T
            torch.testing.assert_close(torch.tensor(projected['B4'], dtype=torch.float64), b, rtol=0, atol=1e-14)
            torch.testing.assert_close(torch.tensor(projected['direct_lower_adjacent_entries'], dtype=torch.float64), direct.diag(-1), rtol=0, atol=1e-14)
            eigenvalues, eigenvectors = torch.linalg.eigh(b)
            torch.testing.assert_close(torch.tensor(projected['ritz_values'], dtype=torch.float64), eigenvalues, rtol=0, atol=1e-13)
            estimates = steps[-1]['beta'] * eigenvectors[-1].abs()
            torch.testing.assert_close(torch.tensor(projected['last_step_only_ritz_residual_estimates'], dtype=torch.float64), estimates, rtol=0, atol=1e-12)
            self.assertEqual(projected['beta_4'], steps[-1]['beta'])
            with rollback.VectorStore(root, eligible, torch) as store:
                repeated = rollback.projected_hessian_statistics(store, eligible, steps, torch)
            self.assertEqual(projected, repeated)
            json.dumps(projected, allow_nan=False)

    def test_projection_uses_all_prior_vectors_and_two_passes_with_same_split_projector(self):
        eligible, parameters = toy_vectors()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            basis = torch.eye(6)[:, :3]
            basis[0, 0] += 1e-6
            basis[0, 1] += 1e-6
            for i in range(3):
                write_flat(root, f'q{i + 1}', basis[:, i], parameters)
            raw = []
            for split in range(4):
                value = torch.tensor([1. + split, 2., 3., 4. + split, 0., 0.])
                raw.append(value.double())
                write_flat(root, f'z3_s{split}', value, parameters)
            q = basis.double()
            with rollback.VectorStore(root, eligible, torch) as store:
                passes = rollback.reorthogonalization_coefficients(store, 3, eligible, torch)
                self.assertEqual(len(passes), 2)
                references = [torch.stack(raw).mean(0), *raw]
                for row, vector in enumerate(references):
                    residual = vector.clone()
                    first_pass = vector - q @ (q.T @ vector)
                    for _ in range(2):
                        residual -= q @ (q.T @ residual)
                    actual = torch.cat([rollback.residual_matrix(store, 3, None if row == 0 else row - 1,
                                            name, module.weight.shape, passes).flatten() for name, module in eligible])
                    torch.testing.assert_close(actual, residual, rtol=0, atol=1e-14)
                    self.assertGreater(float((q.T @ first_pass).norm()), 1e-7)
                    self.assertLess(float((q.T @ actual).norm()), 1e-10)
                    self.assertAlmostEqual(float(actual[3]), 5.5 if row == 0 else 3. + row)
                stats = rollback.action_and_residual_statistics(store, 3, eligible, passes, torch)
                residual_rows = []
                for vector in references:
                    vector = vector.clone()
                    for _ in range(2):
                        vector -= q @ (q.T @ vector)
                    residual_rows.append(vector)
                residual_rows = torch.stack(residual_rows)
                expected_gram = residual_rows @ residual_rows.T
                torch.testing.assert_close(torch.tensor(stats['residuals']['full_then_splits_gram_matrix'], dtype=torch.float64), expected_gram, rtol=0, atol=1e-13)
                torch.testing.assert_close(torch.tensor(stats['residuals']['split_gram_matrix'], dtype=torch.float64), expected_gram[1:, 1:], rtol=0, atol=1e-13)
                self.assertNotEqual(stats['residuals']['split_norms'][0], stats['residuals']['split_norms'][3])
                self.assertLess(stats['residuals']['relative_mean_residual_difference'], 1e-12)

    def test_exact_breakdown_fails_but_tiny_nonzero_novelty_is_not_cut_off(self):
        eligible, parameters = toy_vectors()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_flat(root, 'g', torch.tensor([1., 0., 0., 0., 0., 0.]), parameters)
            with rollback.VectorStore(root, eligible, torch) as store:
                def constant_action(step, key):
                    for split in range(4):
                        write_flat(root, f'z{step}_s{split}', torch.tensor([2., 0., 0., 0., 0., 0.]), parameters)
                with self.assertRaisesRegex(ValueError, 'breakdown at step 1'):
                    rollback.build_krylov_basis(store, eligible, constant_action, rollback.FROZEN_SPEC, torch)
                self.assertFalse((root / 'q2.safetensors').exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_flat(root, 'q1', torch.tensor([1., 0., 0., 0., 0., 0.]), parameters)
            for split in range(4):
                write_flat(root, f'z1_s{split}', torch.tensor([2., 1e-20, 0., 0., 0., 0.]), parameters)
            with rollback.VectorStore(root, eligible, torch) as store:
                passes = rollback.reorthogonalization_coefficients(store, 1, eligible, torch)
                stats = rollback.action_and_residual_statistics(store, 1, eligible, passes, torch)
                self.assertGreater(stats['beta'], 0)
                self.assertLess(stats['beta'], 1e-19)
                rollback.save_cpu_basis(root / 'q2.safetensors', eligible,
                    lambda name, shape: rollback.residual_matrix(store, 1, None, name, shape, passes), stats['beta'], torch)
                self.assertEqual(read_flat(store, 'q2', eligible).tolist(), [0., 1., 0., 0., 0., 0.])

    def test_basis_orthogonality_failure_and_equal_functional_norm_convention(self):
        eligible, parameters = toy_vectors()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            norm, _, _, _, q, _, _, _ = self.run_dense(root, eligible, parameters)
            with rollback.VectorStore(root, eligible, torch) as store:
                gradient, first_norm = rollback.load_device_vector(store, 'g', eligible, torch)
                self.assertTrue(torch.equal(torch.cat([v.flatten() for v in gradient.values()]), torch.arange(1, 7).float()))
                self.assertAlmostEqual(first_norm, norm)
                for i in range(2, 5):
                    tangent, realized = rollback.load_device_vector(store, f'q{i}', eligible, torch, norm)
                    actual = torch.cat([v.flatten() for v in tangent.values()])
                    self.assertTrue(torch.equal(actual, (q[:, i - 1] * norm).float()))
                    self.assertAlmostEqual(realized, norm, places=5)
            (root / 'q4.safetensors').unlink()
            write_flat(root, 'q4', q[:, 0].float(), parameters)
            with rollback.VectorStore(root, eligible, torch) as store:
                with self.assertRaisesRegex(ValueError, 'orthogonality failed'):
                    rollback.basis_gram_statistics(store, eligible, rollback.FROZEN_SPEC['orthogonality'], torch)

    def test_basis_publishing_hashes_no_overwrite_and_cleanup(self):
        eligible, parameters = toy_vectors()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'temporary'
            rollback.prepare_temporary_directory(root)
            self.run_dense(root, eligible, parameters)
            destination = Path(directory) / 'basis'
            inodes = {f'q{i}.safetensors': (root / f'q{i}.safetensors').stat().st_ino for i in range(1, 5)}
            with patch.object(rollback.os, 'replace', wraps=rollback.os.replace) as rename:
                records = rollback.publish_basis(root, destination)
            self.assertEqual(rename.call_count, 4)
            self.assertEqual([r['filename'] for r in records], ['q1.safetensors', 'q2.safetensors', 'q3.safetensors', 'q4.safetensors'])
            from safetensors import safe_open
            for record in records:
                self.assertEqual(record['sha256'], rollback.sha256_file(destination / record['filename']))
                self.assertFalse((root / record['filename']).exists())
                self.assertEqual((destination / record['filename']).stat().st_ino, inodes[record['filename']])
                with safe_open(str(destination / record['filename']), framework='pt') as file:
                    self.assertEqual(set(file.keys()), set(parameters))
                    for name in file.keys():
                        self.assertEqual(file.get_tensor(name).dtype, torch.float32)
                        self.assertTrue(file.get_tensor(name).is_contiguous())
            with self.assertRaises(FileExistsError):
                rollback.publish_basis(root, destination)
            with self.assertRaisesRegex(ValueError, 'stale'):
                rollback.prepare_temporary_directory(root)
            (root / 'g.safetensors').unlink()
            rollback.cleanup_temporary_directory(root)
            self.assertFalse(root.exists())
            self.assertEqual(len(list(destination.glob('*.safetensors'))), 4)

    def test_atomic_publication_failure_retains_moved_and_unmoved_basis(self):
        eligible, parameters = toy_vectors()
        with tempfile.TemporaryDirectory() as directory:
            root, destination = Path(directory) / 'temp', Path(directory) / 'basis'
            rollback.prepare_temporary_directory(root)
            for i in range(1, 5):
                write_flat(root, f'q{i}', torch.arange(6).float(), parameters)
            replace = rollback.os.replace
            def fail_second(source, target):
                if source.name == 'q2.safetensors':
                    raise OSError('injected rename failure')
                replace(source, target)
            with patch.object(rollback.os, 'replace', side_effect=fail_second):
                with self.assertRaisesRegex(OSError, 'rename failure'):
                    rollback.publish_basis(root, destination)
            self.assertTrue((destination / 'q1.safetensors').exists())
            self.assertFalse((root / 'q1.safetensors').exists())
            for i in range(2, 5):
                self.assertTrue((root / f'q{i}.safetensors').exists())
                self.assertFalse((destination / f'q{i}.safetensors').exists())
            with self.assertRaises(FileExistsError):
                rollback.publish_basis(root, destination)
            with self.assertRaisesRegex(ValueError, 'stale'):
                rollback.prepare_temporary_directory(root)

    def test_cross_filesystem_publication_refuses_without_copy_or_move(self):
        eligible, parameters = toy_vectors()
        with tempfile.TemporaryDirectory() as directory:
            root, destination = Path(directory) / 'temp', Path(directory) / 'basis'
            rollback.prepare_temporary_directory(root)
            destination.mkdir()
            for i in range(1, 5):
                write_flat(root, f'q{i}', torch.arange(6).float(), parameters)
            stat = Path.stat
            def different_device(path, *args, **kwargs):
                value = stat(path, *args, **kwargs)
                if path == destination:
                    fields = list(value)
                    fields[2] += 1
                    return rollback.os.stat_result(fields)
                return value
            with patch.object(Path, 'stat', different_device), patch.object(rollback.os, 'replace') as rename:
                with self.assertRaisesRegex(ValueError, 'same filesystem'):
                    rollback.publish_basis(root, destination)
                rename.assert_not_called()
            self.assertEqual(len(list(root.glob('*.safetensors'))), 4)
            self.assertEqual(list(destination.iterdir()), [])

    def test_step_diagnostics_failure_preserves_current_actions(self):
        eligible, parameters = toy_vectors()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(rollback.os, 'fsync', side_effect=OSError('diagnostic flush failed')):
                with self.assertRaisesRegex(OSError, 'diagnostic flush failed'):
                    self.run_dense(root, eligible, parameters)
            self.assertEqual(len(list(root.glob('z1_s*.safetensors'))), 4)
            self.assertTrue((root / 'q2.safetensors').exists())
            self.assertEqual(list(root.glob('z2_s*.safetensors')), [])

    def test_incremental_spectra_and_output_contract(self):
        spec = tiny_spec(length=128)
        spec['readout']['hidden_size'] = 4
        response = torch.diag(torch.tensor([3., 2., 1., .5], dtype=torch.float64))[:, None, :].expand(4, 128, 4).clone()
        merged = torch.zeros(128, 4, dtype=torch.float64)
        artifact = rollback.build_artifact(merged, response, spec, torch)
        self.assertEqual(set(artifact), {'metadata', 'merged_mean', 'krylov_responses'})
        self.assertEqual(artifact['metadata']['direction_order'], [1, 2, 3, 4])
        self.assertEqual(artifact['krylov_responses'].shape, (4, 128, 4))
        for name in spec['outputs']['tensors']:
            self.assertEqual(artifact[name].dtype, torch.float32)
            self.assertEqual(artifact[name].device.type, 'cpu')
            self.assertTrue(artifact[name].is_contiguous())
        diagnostics = rollback.activation_diagnostics(artifact, torch)
        self.assertEqual([r['position'] for r in diagnostics['positions']], list(range(128)))
        for row in diagnostics['positions']:
            self.assertEqual(row['response_norms'], [3., 2., 1., .5])
            self.assertEqual(row['response_cosine_matrix'], torch.eye(4).tolist())
            for count, spectrum in enumerate(row['incremental_spectra'], 1):
                squares = torch.tensor([9., 4., 1., .25], dtype=torch.float64)[:count]
                torch.testing.assert_close(torch.tensor(spectrum['squared_singular_value_fractions'], dtype=torch.float64), squares / squares.sum(), rtol=1e-14, atol=1e-14)
                self.assertAlmostEqual(spectrum['participation_ratio_effective_rank'], float(squares.sum() ** 2 / squares.square().sum()))
                self.assertAlmostEqual(spectrum['top1_energy_fraction'], float(squares[0] / squares.sum()), places=14)
        self.assertEqual(diagnostics['descriptive_summaries']['positions_1_through_127']['incremental_spectra'][3]['top1_energy_fraction']['count'], 127)
        zeros = {'krylov_responses': torch.zeros(4, 128, 4)}
        zero_stats = rollback.activation_diagnostics(zeros, torch)
        self.assertIsNone(zero_stats['positions'][0]['response_cosine_matrix'][0][0])
        self.assertEqual(zero_stats['positions'][0]['spectrum']['participation_ratio_effective_rank'], 0.)
        json.dumps(zero_stats, allow_nan=False)
        with self.assertRaises(ValueError):
            rollback.build_artifact(merged.float(), response, spec, torch)

    def test_frozen_k4_workload_and_firewall(self):
        spec = rollback.load_spec(SPEC_PATH)
        self.assertEqual(spec['krylov']['dimension'], 4)
        self.assertEqual(spec['krylov']['reorthogonalization_passes'], 2)
        self.assertIsNone(spec['krylov']['near_zero_cutoff'])
        self.assertEqual(spec['workload']['hvp_batches_total'], 2048)
        self.assertEqual(spec['workload']['jvp_batches_total'], 128)
        source = SCRIPT_PATH.read_text().replace('oracle_probe', 'generic_probe')
        for forbidden in ('oracle', 'adapter', 'models/base', 'evaluation', 'curvature_probe.pt',
                          'response_map.pt', 'linear_response.pt', 'fisher', 'gauss_newton',
                          'linalg.inv', 'linalg.solve', 'damping', 'torch.optim'):
            self.assertNotIn(forbidden, source.lower())
        self.assertEqual(rollback.HVP_SOURCE_PATH.name, 'smoke_test_generic_hessian_curvature_probe.py')
        self.assertEqual(rollback.LOADED_HVP_SOURCE_SHA256, rollback.sha256_file(rollback.HVP_SOURCE_PATH))


if __name__ == "__main__":
    unittest.main()
