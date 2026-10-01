"""Synthetic tests for Attempt103's privileged soft-teacher mechanism."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[1]/'scripts/ablation/diagnose_privileged_base_soft_target_rollback.py'
loader = importlib.util.spec_from_file_location('attempt103_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


class TinyTokenizer:
    def __init__(self, values):
        self.values = values

    def get_vocab(self):
        return dict(self.values)

    def __len__(self):
        return len(self.values)


class TinyTeacherModel(torch.nn.Module):
    def __init__(self, sign):
        super().__init__()
        self.linears = torch.nn.ModuleList(torch.nn.Linear(1, 1, bias=False)
                                           for _ in range(98))
        self.unselected = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        for layer in self.linears:
            layer.weight.data.fill_(sign*0.005)
        self.config = type('Config', (), {'vocab_size': 2, 'use_cache': True})()
        self.eval()

    def forward(self, input_ids, use_cache):
        x = (input_ids.float() % 2).unsqueeze(-1)
        for layer in self.linears:
            x = x+0.001*layer(x)
        return type('Output', (), {'logits': torch.cat((x, -x), dim=-1)})()


class PrefixBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = torch.nn.ModuleList(torch.nn.Linear(2, 2, bias=False)
                                           for _ in range(7))


class InventoryModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(PrefixBlock() for _ in range(28))


def eligible_tiny(model):
    return [(f'linears.{i}', layer) for i, layer in enumerate(model.linears)]


class Attempt103Tests(unittest.TestCase):
    def test_frozen_spec_base_identity_and_source_hashes(self):
        self.assertEqual(m.a.sha256_file(m.SPEC_PATH), m.SPEC_SHA256)
        self.assertEqual(SPEC['base'], m.known.FROZEN_SPEC['base'])
        self.assertEqual(SPEC['base']['repo_id'], 'Qwen/Qwen3-1.7B')
        self.assertEqual(SPEC['base']['revision'],
                         '0060bc56d46589041c1048efd1a397421b1142b5')
        self.assertEqual(SPEC['final_checkpoint_files'],
                         m.known.FROZEN_SPEC['canonical_checkpoint_files'])
        self.assertEqual(m.a.sha256_file(m.KNOWN_SOURCE), m.KNOWN_SOURCE_SHA256)

    def test_historical_base_inventory_mismatch_fails_before_loops(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            with (patch.object(m, 'require_hash'),
                  patch.object(m.known, 'load_spec', return_value=m.known.FROZEN_SPEC),
                  patch.object(m.a, 'checkpoint_file_records',
                               side_effect=[[], spec['final_checkpoint_files']])):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    m.validate_sources(spec)

    def test_final_inventory_mismatch_fails_before_loops(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            with (patch.object(m, 'require_hash'),
                  patch.object(m.known, 'load_spec', return_value=m.known.FROZEN_SPEC),
                  patch.object(m.a, 'checkpoint_file_records',
                               side_effect=[spec['base']['files'], []])):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    m.validate_sources(spec)

    def test_exact_tokenizer_mapping_and_vocabulary(self):
        base, final = TinyTeacherModel(-1), TinyTeacherModel(1)
        same = TinyTokenizer({'a': 0, 'b': 1})
        self.assertEqual(m.validate_tokenizer_mapping(same, same, base, final), 2)
        with self.assertRaisesRegex(ValueError, 'mapping'):
            m.validate_tokenizer_mapping(same, TinyTokenizer({'a': 1, 'b': 0}),
                                         base, final)
        final.config.vocab_size = 3
        with self.assertRaisesRegex(ValueError, 'mapping'):
            m.validate_tokenizer_mapping(same, same, base, final)

    def test_first_512_construction_and_first_1024_probe(self):
        corpus = torch.arange(4096, dtype=torch.int64)[:, None].repeat(1, 128)
        probe = torch.arange(10000, dtype=torch.int64)[:, None].repeat(1, 128)
        with (patch.object(m.a, 'validate_frozen_corpus', return_value=(corpus, {}, {})),
              patch.object(m.a, 'sha256_raw_int64_tensor',
                           return_value=SPEC['corpus']['raw_tensor_sha256']),
              patch.object(m.a, 'load_probe', return_value=probe)):
            selected, matched = m.load_frozen_tokens(SPEC)
        self.assertEqual(tuple(selected.shape), (512, 128))
        self.assertEqual(tuple(matched.shape), (1024, 128))
        self.assertEqual(selected[-1, 0].item(), 511)
        self.assertEqual(matched[-1, 0].item(), 1023)
        self.assertTrue(selected.is_contiguous() and matched.is_contiguous())

    def test_soft_teacher_ce_kl_and_no_temperature(self):
        z0 = torch.tensor([[[1.0, -2.0], [0.1, 0.3]]], dtype=torch.float32)
        z1 = torch.tensor([[[0.2, 0.5], [1.0, -0.4]]], dtype=torch.float32)
        ce, kl = m.soft_teacher_losses(z0, z1)
        p0 = torch.softmax(z0, dim=-1)
        expected_ce = -(p0*torch.log_softmax(z1, dim=-1)).sum()
        expected_kl = (p0*(torch.log_softmax(z0, dim=-1)-
                           torch.log_softmax(z1, dim=-1))).sum()
        self.assertAlmostEqual(float(ce), float(expected_ce))
        self.assertAlmostEqual(float(kl), float(expected_kl))
        self.assertIsNone(SPEC['teacher_objective']['teacher_temperature'])
        self.assertIsNone(SPEC['teacher_objective']['top_k'])

    def test_logit_derivative_is_p1_minus_p0(self):
        z0 = torch.tensor([[[1.0, -1.0, 0.5]]])
        z1 = torch.tensor([[[0.3, -0.4, 1.2]]], requires_grad=True)
        ce, _ = m.soft_teacher_losses(z0, z1)
        grad, = torch.autograd.grad(ce, z1)
        expected = torch.softmax(z1.detach(), -1)-torch.softmax(z0, -1)
        self.assertTrue(torch.allclose(grad, expected, atol=1e-7))

    def test_positive_teacher_gradient_and_only_selected_support(self):
        base, final = TinyTeacherModel(-1), TinyTeacherModel(1)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        eligible = eligible_tiny(final)
        tokens = torch.ones((4, 128), dtype=torch.int64)
        summary = m.teacher_gradient_stage(base, final, tokens, eligible, SPEC)
        self.assertEqual(summary['total_prediction_tokens'], 4*127)
        self.assertGreater(summary['mean_teacher_cross_entropy'], 0)
        self.assertTrue(all(layer.weight.grad is not None for _, layer in eligible))
        self.assertIsNone(final.unselected.grad)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        stored = eligible[0][1].weight.grad.clone()
        final.zero_grad(set_to_none=True)
        with torch.no_grad():
            z0 = base(input_ids=tokens, use_cache=False).logits[:, :-1, :]
        z1 = final(input_ids=tokens, use_cache=False).logits[:, :-1, :]
        ce, _ = m.soft_teacher_losses(z0, z1)
        (ce/(4*127)).backward()
        self.assertTrue(torch.equal(stored, eligible[0][1].weight.grad))

    def test_exact_98_early_linear_inventory(self):
        model = InventoryModel()
        eligible = m.a.discover_eligible_linear_weights(model, torch)
        self.assertEqual(len(eligible), 98)
        self.assertTrue(all(name.startswith('model.layers.') for name, _ in eligible))
        self.assertTrue(all(int(name.split('.')[2]) <= 13 for name, _ in eligible))
        self.assertEqual([name for name, _ in eligible],
                         sorted(name for name, _ in eligible))

    def test_global_positive_tangent_scale(self):
        model = TinyTeacherModel(1)
        eligible = eligible_tiny(model)
        for _, layer in eligible:
            layer.weight.grad = torch.full_like(layer.weight, 0.5)
        scale = m.a.global_tangent_scale(eligible, 0.00125, torch)
        tangents, records, realized = m.a.prepare_tangents(eligible, scale, torch)
        self.assertEqual(len(tangents), 98)
        self.assertEqual(len(records), 98)
        self.assertGreater(scale['alpha'], 0)
        self.assertAlmostEqual(realized['aggregate_realized_relative_tangent_norm'],
                               0.00125, places=9)
        self.assertTrue(all(float(value[0, 0]) > 0 for value in tangents.values()))
        self.assertTrue(all(layer.weight.grad is None for _, layer in eligible))

    def test_matched_base_final_probe_difference(self):
        base = torch.full((128, 2048), 0.25, dtype=torch.float64)
        final = torch.full((128, 2048), 0.75, dtype=torch.float64)
        difference = m.matched_difference(final, base)
        self.assertTrue(torch.equal(difference, torch.full_like(difference, 0.5)))
        self.assertTrue(difference.is_contiguous())

    def test_strict_jvp_response_path(self):
        model = torch.nn.Linear(1, 1, bias=False)
        probe = torch.zeros((32, 128), dtype=torch.int64)
        primal = torch.ones((32, 128, 2048), dtype=torch.float32)
        tangent = torch.full_like(primal, 0.25)
        with (patch.object(m.a, 'ordinary_hook_readout', return_value=primal),
              patch.object(m.a, 'run_readout_jvp', return_value=(primal, tangent)) as jvp,
              patch.object(m.a, 'compare_primal', return_value=0.0)):
            mean, response, audit = m.final_probe_jvp(model, probe, {'selected': 1}, SPEC)
        self.assertEqual(jvp.call_count, 1)
        self.assertEqual(audit, 0.0)
        self.assertTrue(torch.all(response == 0.25))
        self.assertTrue(torch.all(mean == 1))

    def test_native_strict_jvp_matches_explicit_toy_tangent(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2, bias=False) for _ in range(2)])
                for layer in self.layers:
                    layer.weight.data.copy_(torch.eye(2))

            def forward(self, input_ids, use_cache, output_hidden_states, return_dict):
                x = torch.nn.functional.one_hot(input_ids, 2).float()
                states = [x]
                for layer in self.layers:
                    x = layer(x)
                    states.append(x)
                return type('Output', (), {'hidden_states': tuple(states)})()

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()

        model = Model()
        tokens = torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.int64)
        tangent = torch.tensor([[1., 2.], [3., 4.]])
        spec = {'model': {'num_hidden_layers': 2},
                'readout': {'block_index': 0, 'hidden_state_index': 1, 'hidden_size': 2},
                'probe': {'sequence_length': 3}}
        primal, response = m.a.run_readout_jvp(
            model, tokens, {'model.layers.0.weight': tangent}, spec, torch)
        expected = torch.nn.functional.one_hot(tokens, 2).float()
        self.assertTrue(torch.equal(primal, expected))
        self.assertTrue(torch.equal(response, expected@tangent.T))

    def test_positionwise_signed_primary_secondary_and_position_zero(self):
        response = torch.zeros((128, 2048), dtype=torch.float32)
        target = torch.zeros_like(response)
        response[:, 0] = 1
        target[:, 0] = 1
        response[2, 0] = 100
        target[2, 0] = -1
        metrics = m.response_metrics(response, target)
        self.assertEqual(metrics['position_0_cosine'], 1)
        self.assertEqual(metrics['position_cosines'][2], -1)
        self.assertEqual(metrics['positions_1_4_mean_cosine'], 0.5)
        self.assertAlmostEqual(metrics['positions_1_127_mean_cosine'], 125/127)
        flat = float(torch.nn.functional.cosine_similarity(
            response[1:5].flatten().double(), target[1:5].flatten().double(), dim=0))
        self.assertNotAlmostEqual(flat, metrics['positions_1_4_mean_cosine'])

    def test_best_scalar_is_descriptive_and_does_not_mutate_response(self):
        response = torch.ones((128, 2048), dtype=torch.float32)
        target = 2*response
        before = response.clone()
        metrics = m.response_metrics(response, target)
        self.assertEqual(metrics['best_scalar_rescaling_diagnostic_only'], 2)
        self.assertEqual(metrics['relative_residual_after_best_rescaling_diagnostic_only'], 0)
        self.assertTrue(torch.equal(response, before))

    def test_no_oracle_adl_or_adapter_input_and_no_overwrite(self):
        for forbidden in ('/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt',
                          '/root/model-diff-scratch/models/adapter',
                          'experiments/attempts/014/evaluation.json'):
            with self.assertRaisesRegex(ValueError, 'Forbidden'):
                m.resolve(forbidden)
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            result = Path(directory)/'result.json'
            result.write_text('{}')
            spec['paths']['result_path'] = str(result)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.validate_sources(spec)
        self.assertFalse(SPEC['information_policy']['oracle_adl_access'])
        self.assertFalse(SPEC['information_policy']['adapter_access'])

    def test_runtime_projection_arithmetic(self):
        loads = {'base_load_seconds': 2.0, 'final_load_seconds': 3.0}
        result = m.runtime_projection(loads, 5.0, 7.0, 11.0)
        self.assertEqual(result['projected_full_teacher_gradient_seconds'], 640)
        self.assertEqual(result['projected_full_probe_and_jvp_seconds'], 576)
        self.assertEqual(result['projected_total_compute_seconds'], 1216)
        self.assertEqual(result['projected_total_wall_seconds'], 1221)

    def test_smoke_writes_no_scientific_output(self):
        base, final = TinyTeacherModel(-1), TinyTeacherModel(1)
        zero = torch.zeros((128, 2048), dtype=torch.float64)
        one = torch.ones_like(zero)
        eligible = eligible_tiny(final)
        with (patch.object(m, 'load_spec', return_value=SPEC),
              patch.object(m, 'validate_sources', return_value={'base_files': [], 'final_files': []}),
              patch.object(m, 'load_frozen_tokens', return_value=(
                  torch.zeros((512, 128), dtype=torch.int64),
                  torch.zeros((1024, 128), dtype=torch.int64))),
              patch.object(m, 'load_model_pair', return_value=(base, final, {
                  'base_load_seconds': 1.0, 'final_load_seconds': 2.0, 'vocab_size': 2})),
              patch.object(m.a, 'discover_eligible_linear_weights', return_value=eligible),
              patch.object(m.a, 'freeze_other_parameters'),
              patch.object(m.a, 'model_state_hashes', return_value={}),
              patch.object(m, 'teacher_gradient_stage', return_value={}),
              patch.object(m.a, 'global_tangent_scale', return_value={}),
              patch.object(m.a, 'prepare_tangents', return_value=(
                  {str(i): torch.ones(1) for i in range(98)}, [], {})),
              patch.object(m, 'base_probe_mean', return_value=zero),
              patch.object(m, 'final_probe_jvp', return_value=(one, one.float(), 0.0)),
              patch.object(m.a, 'verify_model_unchanged'),
              patch.object(m, 'response_metrics', return_value={}),
              patch.object(m.a, 'write_manifest', side_effect=AssertionError('smoke wrote'))):
            result = m.run(smoke_only=True)
        self.assertTrue(result['smoke_only'])
        self.assertFalse(result['writes'])
        self.assertEqual(result['projected_full_teacher_gradient_seconds'],
                         result['teacher_gradient_batch_seconds']*128)


if __name__ == '__main__':
    unittest.main()
