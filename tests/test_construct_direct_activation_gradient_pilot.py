"""Synthetic tests for blind Attempt 026 direct activation gradients."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT/'scripts/ablation/construct_direct_activation_gradient_pilot.py'
SPEC = PROJECT/'experiments/attempts/026_blind_direct_activation_gradient_block13_pilot/spec.json'
loader = importlib.util.spec_from_file_location('attempt026_test_subject', SCRIPT)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
TOY_READOUT = {'block_index': 0, 'sequence_length': 128, 'hidden_size': 4}


class ToyBlock(torch.nn.Module):
    def forward(self, hidden):
        return (hidden*1.25, 'unchanged-field')


class ToyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([ToyBlock()])


class ToyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 4)
        self.model = ToyTransformer()
        self.head = torch.nn.Linear(4, 7)

    def forward(self, input_ids, use_cache=False):
        if use_cache:
            raise AssertionError('Cache must be disabled')
        hidden = self.embedding(input_ids)
        hidden = self.model.layers[0](hidden)[0]
        return type('ToyOutput', (), {'logits': self.head(hidden)})()


def toy_batch():
    return (torch.arange(8*128, dtype=torch.int64).reshape(8, 128) % 7).contiguous()


class TestDirectActivationGradient(unittest.TestCase):
    def test_frozen_spec_and_exact_batch_inventory(self):
        self.assertEqual(json.loads(SPEC.read_text()), m.FROZEN_SPEC)
        self.assertEqual(m.batch_inventory(), [(j, 8*j, 8*(j+1)) for j in range(32)])
        self.assertEqual(m.batch_inventory(True), [(0, 0, 8)])
        self.assertEqual(m.FROZEN_SPEC['batches']['prediction_tokens_per_batch'], 1016)

    def test_zero_intervention_keeps_full_forward_identical(self):
        model, batch = ToyCausalModel().eval(), toy_batch()
        baseline = model(input_ids=batch, use_cache=False).logits.detach().clone()
        leaves = []
        def hook(_module, _inputs, output):
            changed, z = m.intervene_output(output, 8, TOY_READOUT)
            leaves.append(z)
            return changed
        handle = model.model.layers[0].register_forward_hook(hook)
        try:
            altered = model(input_ids=batch, use_cache=False).logits
        finally:
            handle.remove()
        self.assertTrue(torch.equal(altered, baseline))
        self.assertEqual(len(leaves), 1)
        self.assertEqual(tuple(leaves[0].shape), (1, 128, 4))
        self.assertTrue(leaves[0].requires_grad)

    def test_additive_gradient_has_direct_hidden_gradient_sign(self):
        hidden = torch.randn((1, 128, 4), dtype=torch.float32, requires_grad=True)
        target = torch.full_like(hidden, 0.3)
        direct = ((hidden-target)**2).sum()
        expected = torch.autograd.grad(direct, hidden)[0]
        changed, z = m.intervene_output(hidden, 1, TOY_READOUT)
        observed = torch.autograd.grad(((changed-target)**2).sum(), z)[0]
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    def test_small_negative_gradient_step_lowers_toy_loss(self):
        hidden = torch.randn((1, 128, 4), dtype=torch.float32)
        changed, z = m.intervene_output(hidden, 1, TOY_READOUT)
        target = torch.full_like(hidden, 0.1)
        loss = ((changed-target)**2).mean()
        direction = torch.autograd.grad(loss, z)[0]
        eta = 0.1
        improved = ((hidden-eta*direction-target)**2).mean()
        self.assertLess(float(improved.detach()), float(loss.detach()))

    def test_prefix_hidden_is_detached(self):
        hidden = torch.randn((1, 128, 4), dtype=torch.float32, requires_grad=True)
        changed, z = m.intervene_output(hidden, 1, TOY_READOUT)
        prefix_gradient, z_gradient = torch.autograd.grad(
            changed.square().sum(), (hidden, z), allow_unused=True)
        self.assertIsNone(prefix_gradient)
        self.assertIsNotNone(z_gradient)

    def test_model_parameter_gradients_remain_absent(self):
        model = ToyCausalModel()
        m.freeze_model(model)
        loss, gradient = m.activation_gradient_batch(model, toy_batch(), TOY_READOUT)
        self.assertGreater(loss, 0)
        self.assertEqual(tuple(gradient.shape), (128, 4))
        self.assertEqual(gradient.dtype, torch.float64)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))

    def test_preexisting_parameter_gradient_fails_closed(self):
        model = ToyCausalModel()
        model.head.weight.grad = torch.ones_like(model.head.weight)
        with self.assertRaisesRegex(ValueError, 'existed before'):
            m.freeze_model(model)

    def test_tuple_and_list_fields_are_preserved(self):
        hidden = torch.randn((1, 128, 4), dtype=torch.float32)
        field = object()
        tuple_changed, _ = m.intervene_output((hidden, field, None), 1, TOY_READOUT)
        list_changed, _ = m.intervene_output([hidden, field, None], 1, TOY_READOUT)
        self.assertIs(type(tuple_changed), tuple)
        self.assertIs(type(list_changed), list)
        self.assertIs(tuple_changed[1], field)
        self.assertIs(list_changed[1], field)
        self.assertIsNone(tuple_changed[2])
        self.assertTrue(torch.equal(tuple_changed[0], hidden))

    def test_cpu_float64_batch_and_split_aggregation(self):
        accumulator = m.GradientAccumulator(TOY_READOUT)
        for j in range(32):
            gradient = torch.full((128, 4), float(j+1), dtype=torch.float64)
            gradient[127].zero_()
            accumulator.add(j, float(j+1), gradient, m.FROZEN_SPEC['position_policy'])
        averages = accumulator.averages(32)
        for value in averages.values():
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.device.type, 'cpu')
            self.assertTrue(value.is_contiguous())
        self.assertEqual(float(averages['mean_gradient'][0, 0]), 16.5)
        self.assertEqual(float(averages['first16_mean_gradient'][0, 0]), 8.5)
        self.assertEqual(float(averages['last16_mean_gradient'][0, 0]), 24.5)
        self.assertEqual(float(averages['even_mean_gradient'][0, 0]), 16.0)
        self.assertEqual(float(averages['odd_mean_gradient'][0, 0]), 17.0)
        self.assertEqual(accumulator.mean_loss(), 16.5)
        self.assertEqual(accumulator.indices, list(range(32)))

    def test_cosine_and_position_norm_diagnostics(self):
        ones = torch.ones((128, 2048), dtype=torch.float32)
        zeros = torch.zeros_like(ones)
        ones[127].zero_()
        averages = {'mean_gradient': ones, 'first16_mean_gradient': ones.clone(),
                    'last16_mean_gradient': ones.clone(),
                    'even_mean_gradient': ones.clone(),
                    'odd_mean_gradient': ones.clone()}
        result = m.split_diagnostics(averages)
        self.assertEqual(len(result['first16_vs_last16_cosine_by_position']), 128)
        self.assertAlmostEqual(result['first16_vs_last16_cosine_positions1_4'], 1.0)
        self.assertAlmostEqual(result['even_vs_odd_cosine_positions1_4'], 1.0)
        self.assertEqual(result['mean_gradient_norm_positions1_4'], [2048**0.5]*4)
        self.assertIsNone(result['first16_vs_last16_cosine_by_position'][127])
        self.assertIsNone(m.cosine(zeros[0], ones[0]))

    def test_causal_final_position_gradient_is_zero(self):
        model = ToyCausalModel()
        m.freeze_model(model)
        _, gradient = m.activation_gradient_batch(model, toy_batch(), TOY_READOUT)
        self.assertTrue(torch.equal(gradient[127], torch.zeros_like(gradient[127])))
        audit = m.final_position_audit(gradient, m.FROZEN_SPEC['position_policy'])
        self.assertEqual(audit['position127_norm'], 0.0)
        broken = gradient.clone()
        broken[127, 0] = 0.01
        with self.assertRaisesRegex(ValueError, 'Causal final-position'):
            m.final_position_audit(broken, m.FROZEN_SPEC['position_policy'])

    def test_forbidden_path_firewall_precedes_input_access(self):
        for fragment in ('models/base', 'oracle_adl.pt', 'adapter', 'true_delta',
                         'known_base', 'attempt015', 'attempt016', 'attempt018',
                         'attempt020', 'attempt021', 'attempt025', 'evaluation.json'):
            args = m.parse_args(['--tokens-path', '/tmp/'+fragment])
            with self.assertRaisesRegex(ValueError, 'firewall'):
                m.firewall_and_lock_paths(args)
        with self.assertRaisesRegex(ValueError, 'firewall'):
            m.firewall_and_lock_paths(m.parse_args(['--device', 'oracle-device']))

    def test_provenance_hash_mismatch_fails_before_model_access(self):
        args = m.parse_args([])
        def wrong_hash(path):
            return m.HELPER_SHA256 if Path(path) == m.HELPER_PATH else '0'*64
        with patch.object(m, 'require_outputs_absent'), \
             patch.object(m.a, 'load_json_object', return_value=m.FROZEN_SPEC), \
             patch.object(m.a, 'sha256_file', side_effect=wrong_hash), \
             patch.object(m.a, 'checkpoint_file_records') as checkpoint:
            with self.assertRaisesRegex(ValueError, 'corpus provenance'):
                m.validate_inputs(args)
            checkpoint.assert_not_called()

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = m.parse_args([])
            args.artifact_path = Path(temporary)/'activation_gradient.pt'
            args.manifest_path = Path(temporary)/'construction-manifest.json'
            m.require_outputs_absent(args)
            args.artifact_path.write_bytes(b'x')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.require_outputs_absent(args)

    def test_artifact_hash_and_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'activation_gradient.pt'
            value = torch.zeros((128, 2048), dtype=torch.float32)
            averages = {name: value.clone().contiguous() for name in m.AGGREGATE_KEYS}
            record = m.save_artifact(path, averages)
            artifact, verified = m.verify_artifact(path, record)
            self.assertEqual(verified, record)
            self.assertEqual(set(artifact), set(m.AGGREGATE_KEYS) | {'batch_indices'})
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.save_artifact(path, averages)

    def test_smoke_writes_no_scientific_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = m.parse_args(['--smoke-only', '--device', 'cpu'])
            args.artifact_path = Path(temporary)/'activation_gradient.pt'
            args.manifest_path = Path(temporary)/'construction-manifest.json'
            spec = copy.deepcopy(m.FROZEN_SPEC)
            spec['readout'] = TOY_READOUT
            tokens = toy_batch()
            stream = io.StringIO()
            with patch.object(m, 'validate_inputs', return_value=(spec, tokens, {}, lambda: None)), \
                 patch.object(m.a, 'load_local_model', return_value=ToyCausalModel()), \
                 contextlib.redirect_stdout(stream):
                m.construct(args)
            result = json.loads(stream.getvalue())
            self.assertTrue(result['smoke_only'])
            self.assertEqual(result['microbatch_index'], 0)
            self.assertEqual(len(result['norms_positions1_4']), 4)
            self.assertEqual(result['position127_norm'], 0.0)
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.manifest_path.exists())


if __name__ == '__main__':
    unittest.main()
