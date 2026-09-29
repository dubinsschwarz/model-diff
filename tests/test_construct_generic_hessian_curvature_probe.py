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
ATTEMPT_DIRECTORY = PROJECT / "experiments/attempts/010_generic_hessian_curvature_probe_prefix0_13"
SCRIPT_PATH = PROJECT / "scripts/ablation/construct_generic_hessian_curvature_probe.py"
SPEC_PATH = ATTEMPT_DIRECTORY / "spec.json"
CORPUS_SPEC_PATH = PROJECT / "experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json"
MODULE_SPEC = importlib.util.spec_from_file_location("construct_generic_hessian_curvature_probe", SCRIPT_PATH)
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




    def test_complete_synthetic_pipeline_uses_one_g_and_records_hashes_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, corpus = self.fixture(Path(directory))
            model = FamilyModel()
            before = rollback.model_state_hashes(model, torch)
            expected_model = FamilyModel()
            expected_eligible = rollback.discover_eligible_linear_weights(expected_model, torch)
            rollback.freeze_other_parameters(expected_model, expected_eligible)
            tokens = torch.load(args.tokens_path, weights_only=True)
            rollback.accumulate_mean_generic_gradient(expected_model, tokens, expected_eligible, spec['generic_loss'], torch)
            expected_g = rollback.detach_canonical_gradient(expected_model, expected_eligible)
            direction_ids, direction_norms = [], []
            helper = rollback.hvp_backend.true_hessian_vector_product
            def checked(loss, parameters, direction):
                direction_ids.append(id(direction))
                for key in direction:
                    self.assertTrue(torch.equal(direction[key], expected_g[key]))
                direction_norms.append(rollback.hvp_backend.aggregate_norm(direction))
                return helper(loss, parameters, direction)
            with patch.object(rollback.hvp_backend, 'true_hessian_vector_product', side_effect=checked), \
                 patch.object(rollback, 'run_readout_jvp', wraps=rollback.run_readout_jvp) as jvp:
                self.run_fixture(args, spec, corpus, model)
            self.assertEqual(len(direction_ids), 8)  # Four tiny splits, two HVP batches each.
            self.assertEqual(len(set(direction_ids)), 1)
            self.assertEqual(len(set(direction_norms)), 1)
            self.assertEqual(jvp.call_count, 12)  # Six direct response directions, two tiny batches each.
            self.assertFalse(args.temporary_directory.exists())
            rollback.verify_model_unchanged(model, before, torch)
            artifact = torch.load(args.artifact_path, weights_only=True)
            manifest = json.loads(args.construction_manifest_path.read_text())
            self.assertEqual(len(manifest['readout']['checks']), 6)
            self.assertEqual([row['response'] for row in manifest['readout']['checks']], ['g', 'h_full', 'h_0', 'h_1', 'h_2', 'h_3'])
            self.assertEqual(len(manifest['matrices']), 98)
            self.assertEqual(manifest['artifact']['serialized_sha256'], rollback.sha256_file(args.artifact_path))
            self.assertEqual(manifest['hvp_implementation_sha256'], rollback.sha256_file(rollback.HVP_SOURCE_PATH))
            for name in spec['outputs']['tensors']:
                self.assertEqual(manifest['artifact']['raw_tensors_sha256'][name], rollback.sha256_raw_float32_tensor(artifact[name], torch))
            weights = manifest['weight_space_diagnostics']
            self.assertAlmostEqual(weights['gradient_norm'], direction_norms[0])
            self.assertAlmostEqual(weights['beta'] * weights['full_hessian_action_norm'], weights['gradient_norm'])
            self.assertAlmostEqual(manifest['realized_tangent_norms']['h_full'], weights['gradient_norm'], places=7)
            self.assertFalse(manifest['response_consistency']['method']['thresholded'])
            self.assertTrue(manifest['response_consistency']['measured']['all_positions_within_reference_tolerances'])
            self.assertEqual(artifact['metadata']['diagnostic_probe'], spec['diagnostic_probe'])
            first = args.construction_manifest_path.read_bytes()
            args.artifact_path.unlink(); args.construction_manifest_path.unlink()
            self.run_fixture(args, spec, corpus)
            self.assertEqual(first, args.construction_manifest_path.read_bytes())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.run_fixture(args, spec, corpus)

    def test_hvp_failure_keeps_completed_vectors_and_refuses_stale_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, corpus = self.fixture(Path(directory))
            with patch.object(rollback.hvp_backend, 'true_hessian_vector_product', side_effect=RuntimeError('unsupported second derivative')):
                with self.assertRaisesRegex(RuntimeError, 'unsupported second derivative'):
                    self.run_fixture(args, spec, corpus)
            self.assertTrue((args.temporary_directory / 'g.safetensors').exists())
            self.assertEqual(json.loads((args.temporary_directory / 'state.json').read_text())['stage'], 'hessian_split_0')
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.construction_manifest_path.exists())
            with self.assertRaisesRegex(ValueError, 'stale'):
                self.run_fixture(args, spec, corpus)
            with self.assertRaisesRegex(ValueError, 'Unexpected'):
                rollback.cleanup_temporary_directory(args.temporary_directory)

    def test_input_mutation_before_publication_retains_vectors_without_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, corpus = self.fixture(Path(directory))
            original = rollback.compute_probe_response
            calls = []
            def changed(*positional, **kwargs):
                result = original(*positional, **kwargs)
                calls.append(1)
                if len(calls) == 6:
                    args.corpus_manifest_path.write_text('mutation')
                return result
            with patch.object(rollback, 'compute_probe_response', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'Frozen input changed'):
                    self.run_fixture(args, spec, corpus)
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.construction_manifest_path.exists())
            self.assertEqual(len(list(args.temporary_directory.glob('*.safetensors'))), 5)


class CurvatureConstructionTests(unittest.TestCase):
    def test_frozen_scientific_semantics_and_firewall(self):
        spec = rollback.load_spec(SPEC_PATH)
        self.assertEqual(spec['attempt_id'], '010_generic_hessian_curvature_probe_prefix0_13')
        self.assertEqual(spec['generic_loss']['total_prediction_tokens'], 520192)
        self.assertEqual(spec['splits']['total_prediction_tokens'], 130048)
        self.assertEqual(spec['splits']['batch_size'], 8)
        self.assertEqual(spec['hessian']['accumulation_device'], 'parameter_device')
        self.assertFalse(spec['hessian']['separate_full_corpus_HVP'])
        self.assertTrue(spec['tangents']['common_beta'])
        self.assertFalse(spec['response_consistency']['thresholded'])
        self.assertEqual(spec['workload'], {'gradient_batches': 512, 'hvp_batches_per_split': 128,
            'hvp_batches_total': 512, 'response_directions': 6, 'probe_batches_per_direction': 32,
            'jvp_batches_total': 192})
        source = SCRIPT_PATH.read_text().replace('oracle_probe', 'generic_probe')
        for forbidden in ('oracle', 'adapter', 'models/base', 'evaluation', 'self_diff', 'attempt008',
                          'attempt009', 'linear_response.pt', 'response_map.pt', 'fisher',
                          'gauss_newton', 'lanczos', 'krylov', 'inverse_hessian', 'damping', 'torch.optim'):
            self.assertNotIn(forbidden, source.lower())
        self.assertEqual(rollback.HVP_SOURCE_PATH.name, 'smoke_test_generic_hessian_curvature_probe.py')
        self.assertEqual(rollback.LOADED_HVP_SOURCE_SHA256, rollback.sha256_file(rollback.HVP_SOURCE_PATH))

    def test_split_hessian_actions_use_same_canonical_g_and_equal_explicit_full_hessian(self):
        tokens = torch.arange(16 * 4).reshape(16, 4) % 3
        model = ToyCausalModel()
        eligible = [('proj', model.proj)]
        settings = dict(sample_count=16, sequence_length=4, predictions_per_sample=3,
                        total_prediction_tokens=48, batch_size=2, number_of_batches=8)
        rollback.accumulate_mean_generic_gradient(model, tokens, eligible, settings, torch)
        g = rollback.detach_canonical_gradient(model, eligible)
        before = g['proj.weight'].clone()
        observed = []
        helper = rollback.hvp_backend.true_hessian_vector_product
        def checked(loss, parameters, direction):
            self.assertIs(direction, g)
            self.assertTrue(torch.equal(direction['proj.weight'], before))
            self.assertFalse(direction['proj.weight'].requires_grad)
            observed.append(id(direction))
            return helper(loss, parameters, direction)
        split_settings = {**settings, 'sample_count': 4, 'total_prediction_tokens': 12, 'number_of_batches': 2}
        actions = []
        def loss_on(subset, flat):
            logits = torch.nn.functional.linear(subset.float().unsqueeze(-1), flat.reshape(3, 1))
            return torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 3), subset[:, 1:].reshape(-1))
        flat = model.proj.weight.detach().flatten()
        with patch.object(rollback.hvp_backend, 'true_hessian_vector_product', side_effect=checked):
            for subset in rollback.interleaved_splits(tokens):
                actual = rollback.accumulate_hessian_split(model, subset, eligible, g, split_settings, torch)
                explicit = torch.autograd.functional.hessian(lambda w: loss_on(subset, w), flat)
                torch.testing.assert_close(actual['proj.weight'].flatten(), explicit @ before.flatten(), rtol=2e-5, atol=1e-7)
                self.assertEqual(actual['proj.weight'].dtype, torch.float32)
                self.assertEqual(actual['proj.weight'].device, model.proj.weight.device)
                actions.append(actual['proj.weight'])
        self.assertEqual(len(observed), 8)
        self.assertEqual(len(set(observed)), 1)
        full_action = torch.stack(actions).double().mean(0)
        full_hessian = torch.autograd.functional.hessian(lambda w: loss_on(tokens, w), flat)
        torch.testing.assert_close(full_action.flatten(), (full_hessian @ before.flatten()).double(), rtol=2e-5, atol=1e-7)
        self.assertTrue(torch.equal(g['proj.weight'], before))
        self.assertIsNone(model.proj.weight.grad)
        with self.assertRaisesRegex(ValueError, 'normalization'):
            rollback.accumulate_hessian_split(model, tokens[:4], eligible, g,
                {**split_settings, 'total_prediction_tokens': 48}, torch)

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

    def test_matrixwise_weight_statistics_common_beta_and_exact_fp32_tangents(self):
        eligible = [('a', torch.nn.Linear(2, 1, bias=False)), ('b', torch.nn.Linear(1, 1, bias=False))]
        parameters = {name + '.weight': module.weight for name, module in eligible}
        data = {'g': [torch.tensor([[1., 2.]]), torch.tensor([[3.]])]}
        for k in range(4):
            data[f'h_{k}'] = [torch.tensor([[2. - k, -3. + k]]), torch.tensor([[k + 1.]])]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for key, values in data.items():
                rollback.save_vector(root / f'{key}.safetensors', dict(zip(parameters, values)), parameters, torch)
            with rollback.CurvatureVectorStore(root, eligible, torch) as store:
                weights, records = rollback.curvature_weight_statistics(store, eligible, torch)
                g = torch.cat([value.flatten().double() for value in data['g']])
                split = torch.stack([torch.cat([value.flatten().double() for value in data[f'h_{k}']]) for k in range(4)])
                full = split.mean(0)
                self.assertEqual(weights['gradient_norm'], float(g.norm()))
                self.assertEqual(weights['full_hessian_action_norm'], float(full.norm()))
                self.assertEqual(weights['split_hessian_action_norms'], split.norm(dim=1).tolist())
                self.assertAlmostEqual(weights['rayleigh_quotient'], float(g @ full / (g @ g)))
                self.assertAlmostEqual(weights['norm_amplification'], float(full.norm() / g.norm()))
                self.assertAlmostEqual(weights['gradient_full_hessian_cosine'], rollback.cosine(g, full))
                self.assertEqual(weights['split_hessian_cosine_matrix'], rollback.gram_summary(split @ split.T, torch)['cosine_matrix'])
                self.assertEqual(weights['split_full_hessian_cosines'], [rollback.cosine(row, full) for row in split])
                beta = weights['beta']
                self.assertAlmostEqual(beta, float(g.norm() / full.norm()))
                self.assertEqual(len(records), 2)
                for key in ('g', 'h_full', 'h_0', 'h_1', 'h_2', 'h_3'):
                    tangent, realized = rollback.make_curvature_tangent(store, eligible, key, beta, torch)
                    expected = g.float() if key == 'g' else (beta * (full if key == 'h_full' else split[int(key[-1])])).float()
                    actual = torch.cat([tangent[name].flatten() for name in parameters])
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertAlmostEqual(realized, float(expected.double().norm()))
                # Splits keep different realized norms; they are not individually matched to g.
                self.assertNotAlmostEqual(rollback.make_curvature_tangent(store, eligible, 'h_0', beta, torch)[1],
                                          rollback.make_curvature_tangent(store, eligible, 'h_3', beta, torch)[1])

    def test_artifact_contract_additivity_and_rounding_only_diagnostic(self):
        spec = tiny_spec(length=128)
        merged = torch.ones(128, 2, dtype=torch.float64)
        gradient = merged * 2
        hessian = merged * -3
        splits = hessian.unsqueeze(0).repeat(4, 1, 1)
        artifact, check = rollback.build_artifact(merged, gradient, hessian, splits, spec, torch)
        self.assertTrue(check['all_positions_within_reference_tolerances'])
        self.assertEqual(len(check['positions']), 128)
        self.assertEqual(set(artifact), {'metadata', *spec['outputs']['tensors']})
        self.assertEqual(artifact['metadata']['split_order'], [0, 1, 2, 3])
        for name in spec['outputs']['tensors']:
            self.assertEqual(artifact[name].dtype, torch.float32)
            self.assertEqual(artifact[name].device.type, 'cpu')
            self.assertTrue(artifact[name].is_contiguous())
            self.assertEqual(tuple(artifact[name].shape), (4, 128, 2) if name == 'split_hessian_responses' else (128, 2))
        statistics = rollback.activation_diagnostics(artifact, torch)
        self.assertEqual([row['position'] for row in statistics['positions']], list(range(128)))
        for row in statistics['positions']:
            self.assertAlmostEqual(row['gradient_hessian_cosine'], -1.)
            self.assertAlmostEqual(row['hessian_to_gradient_norm_ratio'], 1.5)
            self.assertAlmostEqual(row['split_off_diagonal_cosines']['mean'], 1.)
        hessian[0] = 1e12; splits[:, 0] = 1e12
        splits[:, 127] += 1
        _, check = rollback.build_artifact(merged, gradient, hessian, splits, spec, torch)
        self.assertLess(check['global']['relative_rms_difference'], 1e-4)
        self.assertFalse(check['positions'][127]['within_reference_tolerances'])
        self.assertFalse(check['all_positions_within_reference_tolerances'])
        zero = torch.zeros_like(hessian)
        zero_check = rollback.split_response_consistency(zero, hessian, spec, torch)
        self.assertIsNone(zero_check['positions'][0]['relative_rms_difference'])
        json.dumps(zero_check, allow_nan=False)
        with self.assertRaises(ValueError):
            rollback.build_artifact(merged.float(), gradient, hessian, splits, spec, torch)
        splits[0, 0, 0] = float('inf')
        with self.assertRaises(ValueError):
            rollback.build_artifact(merged, gradient, hessian, splits, spec, torch)

    def test_direct_hessian_and_split_jvp_additivity_on_nonlinear_model(self):
        model, spec = TinyModel(nonlinear=True), tiny_spec(count=4, length=3, batch=2)
        spec['diagnostic_probe'].update(sample_count=4, batch_size=2)
        eligible = [('model.layers.0.proj', model.model.layers[0].proj),
                    ('model.layers.2.proj', model.model.layers[2].proj)]
        parameters = {name + '.weight': module.weight for name, module in eligible}
        g = {name: torch.ones_like(parameter) * .2 for name, parameter in parameters.items()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rollback.save_vector(root / 'g.safetensors', g, parameters, torch)
            for k in range(4):
                values = {name: torch.ones_like(parameter) * (.05 + .03 * k) for name, parameter in parameters.items()}
                rollback.save_vector(root / f'h_{k}.safetensors', values, parameters, torch)
            probe = torch.arange(12).reshape(4, 3)
            with rollback.CurvatureVectorStore(root, eligible, torch) as store, patch('sys.stdout', new=io.StringIO()):
                weights, _ = rollback.curvature_weight_statistics(store, eligible, torch)
                full_tangent, _ = rollback.make_curvature_tangent(store, eligible, 'h_full', weights['beta'], torch)
                primal, direct, _ = rollback.compute_probe_response(model, probe, full_tangent, spec, torch)
                responses = []
                for k in range(4):
                    tangent, _ = rollback.make_curvature_tangent(store, eligible, f'h_{k}', weights['beta'], torch)
                    _, response, _ = rollback.compute_probe_response(model, probe, tangent, spec, torch)
                    responses.append(response)
                comparison = rollback.split_response_consistency(direct, torch.stack(responses).mean(0), spec, torch)
                self.assertTrue(comparison['all_positions_within_reference_tolerances'])
                self.assertEqual(primal.dtype, torch.float64)
                self.assertEqual(direct.dtype, torch.float64)
                self.assertEqual(direct.device.type, 'cpu')
                self.assertTrue(bool((direct > 0).all()))

    def test_invalid_vectors_zero_full_action_and_nonpositive_beta_fail(self):
        eligible = [('a', torch.nn.Linear(1, 1, bias=False))]
        parameters = {'a.weight': eligible[0][1].weight}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for invalid in ({}, {'a.weight': torch.ones(2)}, {'a.weight': torch.tensor([[float('nan')]])}):
                with self.assertRaises(ValueError):
                    rollback.save_vector(root / 'g.safetensors', invalid, parameters, torch)
            rollback.save_vector(root / 'g.safetensors', {'a.weight': torch.ones(1, 1)}, parameters, torch)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                rollback.save_vector(root / 'g.safetensors', {'a.weight': torch.ones(1, 1)}, parameters, torch)
            for k in range(4):
                rollback.save_vector(root / f'h_{k}.safetensors', {'a.weight': torch.zeros(1, 1)}, parameters, torch)
            with rollback.CurvatureVectorStore(root, eligible, torch) as store:
                with self.assertRaisesRegex(ValueError, 'positive and finite'):
                    rollback.curvature_weight_statistics(store, eligible, torch)
                for beta in (0., -1., float('inf')):
                    with self.assertRaises(ValueError):
                        rollback.make_curvature_tangent(store, eligible, 'h_full', beta, torch)

    def test_zero_activation_norm_policy_and_summary_counts(self):
        artifact = {'gradient_response': torch.zeros(128, 2), 'hessian_response': torch.zeros(128, 2),
                    'split_hessian_responses': torch.zeros(4, 128, 2)}
        values = rollback.activation_diagnostics(artifact, torch)
        self.assertIsNone(values['positions'][0]['gradient_hessian_cosine'])
        self.assertIsNone(values['positions'][0]['hessian_to_gradient_norm_ratio'])
        self.assertEqual(values['descriptive_summaries']['positions_1_through_127']['split_pairwise_cosines']['undefined_count'], 127 * 6)
        json.dumps(values, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
