"""CPU and synthetic checks for Attempt105; no checkpoint or corpus is run."""
import copy
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/diagnose_privileged_endpoint_ggn_forward.py'
loader = importlib.util.spec_from_file_location('attempt105_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


class SmallLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3, bias=False)
        self.unselected = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.2, -0.3], [0.8, 0.1], [-0.4, 0.5]]))

    def forward(self, input_ids, use_cache=False):
        assert use_cache is False
        x = torch.nn.functional.one_hot(input_ids % 2, 2).float()
        return SimpleNamespace(logits=self.linear(x) + self.unselected)


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.ModuleDict({
            name: torch.nn.Linear(1, 1, bias=False)
            for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')})
        self.mlp = torch.nn.ModuleDict({
            name: torch.nn.Linear(1, 1, bias=False)
            for name in ('gate_proj', 'up_proj', 'down_proj')})
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(0.005)

    def forward(self, x):
        # A cheap differentiable 98-coordinate early path.
        return x + 0.001 * sum(layer(x) for family in (self.self_attn, self.mlp)
                                for layer in family.values())


class Toy98Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(ToyBlock() for _ in range(28))
        self.config = SimpleNamespace(model_type='qwen3', num_hidden_layers=28,
                                      hidden_size=1, vocab_size=2, use_cache=False)
        self.model.config = self.config
        self.model.embed_tokens = torch.nn.Embedding(2, 1)
        with torch.no_grad():
            self.model.embed_tokens.weight.copy_(torch.tensor([[0.2], [0.6]]))

    def forward(self, input_ids, use_cache=False, output_hidden_states=False,
                return_dict=True):
        assert use_cache is False
        x = self.model.embed_tokens(input_ids)
        states = [x]
        for layer in self.model.layers:
            x = layer(x)
            states.append(x)
        logits = torch.cat((x, -x), dim=-1)
        return SimpleNamespace(logits=logits,
                               hidden_states=tuple(states) if output_hidden_states else None)


def selected(model):
    return m.a.discover_eligible_linear_weights(model, torch)


def small_spec():
    spec = copy.deepcopy(SPEC)
    spec['readout']['hidden_size'] = 1
    spec['probe'] = {'sequence_length': 128}
    return spec


class Attempt105Tests(unittest.TestCase):
    def test_frozen_linkage_hashes_and_category(self):
        for record, keys in ((SPEC['attempt104'], ('spec', 'source', 'result')),
                             (SPEC['attempt016'], ('spec', 'source'))):
            for key in keys:
                path = PROJECT/record[key+'_path']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                 record[key+'_sha256'])
        old = json.loads((PROJECT/SPEC['attempt104']['result_path']).read_text())
        self.assertEqual(old['attempt103_direct_comparison']['category'],
                         'curvature_limited_support')
        self.assertEqual(SPEC['readout'], m.a104.load_spec()['readout'])
        self.assertEqual(SPEC['base'], m.a104.load_spec()['base'])
        self.assertEqual(SPEC['final_checkpoint_files'],
                         m.a104.load_spec()['final_checkpoint_files'])

    def test_linkage_and_checkpoint_mismatch_fail_before_loops(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'unused.json')
            bad = copy.deepcopy(spec)
            bad['attempt104']['result_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.validate_inputs(bad)
            bad = copy.deepcopy(spec)
            bad['attempt016']['source_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                m.validate_inputs(bad)
            with patch.object(m.a, 'checkpoint_file_records', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'checkpoint inventory'):
                    m.validate_inputs(spec)

    def test_exact_first64_inventory_and_normalization(self):
        tokens = torch.arange(4096, dtype=torch.int64)[:, None].repeat(1, 128)
        record = {'serialized_sha256': SPEC['corpus']['serialized_sha256']}
        with (patch.object(m.a104.parent.a, 'validate_frozen_corpus',
                           return_value=(tokens, {}, record)),
              patch.object(m.a104.parent.a, 'sha256_raw_int64_tensor',
                           return_value=SPEC['corpus']['raw_tensor_sha256'])):
            chosen = m.load_pilot_tokens(SPEC)
        self.assertEqual(tuple(chosen.shape), (64, 128))
        self.assertEqual(chosen[0, 0].item(), 0)
        self.assertEqual(chosen[-1, 0].item(), 63)
        self.assertEqual(SPEC['pilot']['hybrid_batches'], 16)
        self.assertEqual(SPEC['pilot']['ggn_batches'], 64)
        self.assertEqual(SPEC['pilot']['prediction_contexts'], 64*127)
        self.assertEqual(SPEC['hybrid_gradient']['denominator'], 8128)
        self.assertEqual(SPEC['ggn']['denominator'], 8128)
        with (patch.object(m.a104.parent.a, 'validate_frozen_corpus',
                           return_value=(tokens, {}, record)),
              patch.object(m.a104.parent.a, 'sha256_raw_int64_tensor',
                           return_value='0'*64)):
            with self.assertRaisesRegex(ValueError, 'hash'):
                m.load_pilot_tokens(SPEC)

    def test_delta_exact_fp64_subtraction_positive_and_upstream_audit(self):
        base, final = Toy98Model(), Toy98Model()
        with torch.no_grad():
            name, module = selected(final)[0]
            original = module.weight.item()
            module.weight.fill_(original + 0.125)
        base_selected = selected(base)
        saved, records, coordinates = m.a016.snapshot_base(base, base_selected)
        base_values = {key: value.detach().clone() for key, value in base.named_parameters()}
        direction, audit = m.a016.audit_and_tangent(
            final, selected(final), saved, records, coordinates,
            lambda key, _record: base_values[key])
        self.assertEqual(len(direction), 98)
        self.assertAlmostEqual(direction[f'{name}.weight'].item(), 0.125)
        self.assertTrue(audit['omitted_upstream_zero_verified'])
        with torch.no_grad():
            final.model.embed_tokens.weight.add_(1)
        with self.assertRaisesRegex(ValueError, 'Omitted upstream'):
            m.a016.audit_and_tangent(final, selected(final), saved, records,
                                     coordinates, lambda key, _record: base_values[key])

    def test_hybrid_teacher_semantics_positive_and_98_only(self):
        base, final = Toy98Model(), Toy98Model()
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        eligible = selected(final)
        m.a.freeze_other_parameters(final, eligible)
        with torch.no_grad():
            final.model.layers[0].self_attn['q_proj'].weight.add_(0.01)
        tokens = torch.tensor([[0, 1]*64]*4, dtype=torch.int64)
        gradient, info = m.hybrid_gradient_64(base, final, tokens, eligible, small_spec())
        self.assertEqual(set(gradient), {name+'.weight' for name, _ in eligible})
        self.assertEqual(len(gradient), 98)
        self.assertEqual(info['normalization_denominator'], 8128)
        self.assertEqual(info['scored_prediction_contexts'], 4*127)
        self.assertTrue(all(value.dtype == torch.float32 and value.device.type == 'cpu'
                            for value in gradient.values()))
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in final.parameters()))
        self.assertTrue(any(float(value.abs().sum()) > 0 for value in gradient.values()))
        # Exact Attempt104 CE, not hard labels, supplies the positive gradient.
        other = Toy98Model()
        other.load_state_dict(final.state_dict())
        other_eligible = selected(other)
        m.a.freeze_other_parameters(other, other_eligible)
        batch = tokens
        hidden, _ = m.a104.base_hidden_and_logits(base, batch, small_spec())
        hybrid_logits, _ = m.a104.hybrid_logits(other, batch, hidden, small_spec())
        student = other(input_ids=batch, use_cache=False).logits[:, :-1, :].float()
        ce, _, _ = m.a104.hybrid_teacher_losses(hybrid_logits[:, :-1, :], student)
        (ce/8128).backward()
        for name, module in other_eligible:
            self.assertTrue(torch.allclose(module.weight.grad.cpu(), gradient[name+'.weight'],
                                           rtol=1e-5, atol=1e-10))
        self.assertIsNone(other.model.embed_tokens.weight.grad)

    def test_strict_logits_jvp_uses_exact_positive_delta(self):
        model = SmallLogitModel()
        delta = {'linear.weight': torch.tensor([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]])}
        batch = torch.tensor([[0, 1]*64], dtype=torch.int64)
        primal, action = m.strict_logits_jvp(model, batch, delta)
        x = torch.nn.functional.one_hot(batch[:, :-1], 2).float()
        expected = torch.nn.functional.linear(x, delta['linear.weight'])
        self.assertTrue(torch.allclose(action, expected, atol=1e-7))
        ordinary = model(batch, use_cache=False).logits[:, :-1, :]
        self.assertTrue(torch.allclose(primal, ordinary, atol=1e-7))
        self.assertEqual(m.audit_logits_primal(primal, ordinary, SPEC)['max_abs_difference'], 0)
        with self.assertRaisesRegex(ValueError, 'differs'):
            m.audit_logits_primal(primal+0.1, ordinary, SPEC)

    def test_fisher_formula_conservation_detach_and_normalization(self):
        logits = torch.tensor([[[0.2, -0.1, 0.6]]], dtype=torch.float32)
        tangent = torch.tensor([[[0.3, 1.2, -0.7]]], dtype=torch.float32)
        vector, audit = m.categorical_fisher_vector(logits, tangent, 8128, SPEC)
        p = torch.softmax(logits, -1)
        expected = p*(tangent-(p*tangent).sum(-1, keepdim=True))/8128
        self.assertTrue(torch.equal(vector, expected))
        self.assertLess(audit['max_sum_vocab_abs'], 1e-9)
        self.assertEqual(audit['total_prediction_contexts'], 1)
        self.assertEqual(audit['count_contexts_exceeding_relative_target'], 0)
        self.assertFalse(vector.requires_grad)
        with self.assertRaisesRegex(ValueError, 'normalization'):
            m.categorical_fisher_vector(logits, tangent, 4*127, SPEC)

    def test_large_high_cancellation_audit_uses_fp64_and_reports_target(self):
        n = 131072
        u = torch.cat((torch.ones(n//2), -torch.ones(n//2))).reshape(1, 1, n)
        u[0, 0, :20] = 2.0
        p = torch.full_like(u, 1/n)
        original_sum = torch.sum
        dtypes = []
        def recorded_sum(*args, **kwargs):
            dtypes.append(kwargs.get('dtype'))
            return original_sum(*args, **kwargs)
        with patch.object(m.torch, 'sum', side_effect=recorded_sum):
            audit = m.audit_categorical_fisher_conservation(u, p, SPEC)
        self.assertEqual(dtypes, [torch.float64]*3)
        self.assertEqual(audit['max_sum_vocab_abs'], 20.0)
        self.assertAlmostEqual(audit['max_relative_sum_vocab_abs'], 20/(n+20))
        self.assertEqual(audit['count_contexts_exceeding_relative_target'], 1)
        self.assertEqual(audit['total_prediction_contexts'], 1)
        self.assertEqual(audit['max_softmax_sum_error'], 0.0)
        gross = u.clone()
        gross[0, 0, :1000] = 2.0
        with self.assertRaisesRegex(ValueError, 'sum-to-zero'):
            m.audit_categorical_fisher_conservation(gross, p, SPEC)

    def test_conservation_batch_records_are_aggregated_without_selection(self):
        first = {'microbatch_index': 0, 'max_sum_vocab_abs': 0.1,
                 'max_relative_sum_vocab_abs': 2e-5, 'max_softmax_sum_error': 1e-7,
                 'max_conservation_scale': 2.0,
                 'count_contexts_exceeding_relative_target': 3,
                 'total_prediction_contexts': 127}
        second = {**first, 'microbatch_index': 1, 'max_sum_vocab_abs': 0.2,
                  'count_contexts_exceeding_relative_target': 4}
        summary = m.summarize_fisher_conservation([first, second])
        self.assertEqual(summary['per_batch'], [first, second])
        self.assertEqual(summary['aggregate']['max_sum_vocab_abs'], 0.2)
        self.assertEqual(summary['aggregate']['count_contexts_exceeding_relative_target'], 7)
        self.assertEqual(summary['aggregate']['total_prediction_contexts'], 254)

    def test_explicit_tiny_ggn_matrix_equals_jvp_fisher_vjp(self):
        model = SmallLogitModel()
        model.linear.weight.requires_grad_(True)
        batch = torch.tensor([[0, 1]*64], dtype=torch.int64)
        direction = torch.tensor([[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]])
        logits, tangent = m.strict_logits_jvp(model, batch, {'linear.weight': direction})
        fisher, _ = m.categorical_fisher_vector(logits, tangent, 8128, SPEC)
        ordinary = model(batch, use_cache=False).logits[:, :-1, :]
        (ordinary*fisher).sum().backward()
        actual = model.linear.weight.grad.detach().flatten().double()
        self.assertIsNone(model.unselected.grad)
        weight = model.linear.weight.detach().clone().double().requires_grad_(True)
        x = torch.nn.functional.one_hot(batch[:, :-1], 2).double()
        def flat_logits(flat_weight):
            return torch.nn.functional.linear(x, flat_weight.reshape(3, 2)).flatten()
        jacobian = torch.autograd.functional.jacobian(flat_logits, weight.flatten())
        p = torch.softmax(flat_logits(weight.flatten()).reshape(-1, 3), -1)
        blocks = [torch.diag(row)-torch.outer(row, row) for row in p]
        fisher_matrix = torch.block_diag(*blocks)
        ggn = jacobian.T @ fisher_matrix @ jacobian / 8128
        expected = ggn @ direction.flatten().double()
        self.assertTrue(torch.allclose(actual, expected, rtol=2e-5, atol=2e-8))
        self.assertTrue(torch.allclose(tangent.flatten().double(),
                                       jacobian @ direction.flatten().double(), atol=1e-6))

    def test_ggn_stage_98_selected_and_first_batch_audits(self):
        model = Toy98Model()
        eligible = selected(model)
        m.a.freeze_other_parameters(model, eligible)
        delta = {name+'.weight': torch.full_like(module.weight, 0.1)
                 for name, module in eligible}
        tokens = torch.tensor([[0, 1]*64], dtype=torch.int64)
        audit, timings, conservation = m.ggn_action_stage(
            model, tokens, eligible, delta, small_spec())
        self.assertEqual(audit['selected_vjp_matrix_count'], 98)
        self.assertEqual(audit['tangent_shape'], [1, 127, 2])
        self.assertEqual(audit['tangent_dtype'], 'torch.float32')
        self.assertIsNone(model.model.embed_tokens.weight.grad)
        self.assertTrue(all(module.weight.grad is not None for _, module in eligible))
        self.assertTrue(all(value >= 0 for value in timings.values()))
        self.assertEqual(conservation['aggregate']['total_prediction_contexts'], 127)
        self.assertEqual(len(conservation['per_batch']), 1)
        self.assertEqual(conservation['diagnostic_relative_target'], 1e-5)
        self.assertEqual(conservation['hard_guard'],
                         {'relative_tolerance': 1e-3, 'absolute_tolerance': 1e-8})

    def test_matrixwise_global_geometry_true_delta_and_structured(self):
        model = Toy98Model()
        eligible = selected(model)
        delta, hybrid = {}, {}
        for index, (name, module) in enumerate(eligible):
            module.weight.grad = torch.full_like(module.weight, float(index+1))
            delta[name+'.weight'] = torch.full_like(module.weight, 2.0)
            hybrid[name+'.weight'] = torch.full_like(module.weight, 2.0*(index+1)).cpu()
        row = m.matrixwise_metrics(eligible, delta, hybrid)
        self.assertAlmostEqual(row['primary_global_signed_cosine'], 1)
        self.assertAlmostEqual(row['ggn_to_hybrid_norm_ratio'], 0.5)
        self.assertAlmostEqual(row['optimal_scalar_ggn_to_hybrid'], 2)
        self.assertAlmostEqual(row['relative_residual_after_optimal_scalar'], 0)
        self.assertEqual(len(row['structured']['blocks']), 14)
        self.assertEqual(row['structured']['attention']['matrix_count'], 14*4)
        self.assertEqual(row['structured']['mlp']['matrix_count'], 14*3)
        self.assertAlmostEqual(sum(v['ggn_squared_norm_fraction']
                                   for v in row['structured']['blocks'].values()), 1)
        self.assertGreater(row['cosine_ggn_true_delta'], 0)
        self.assertGreater(row['cosine_hybrid_true_delta'], 0)

    def test_optimal_scalar_residual_noncollinear(self):
        model = Toy98Model()
        eligible = selected(model)
        delta = {name+'.weight': torch.ones_like(module.weight) for name, module in eligible}
        hybrid = {}
        for index, (name, module) in enumerate(eligible):
            module.weight.grad = torch.ones_like(module.weight)
            hybrid[name+'.weight'] = torch.full_like(module.weight,
                                                    2.0 if index % 2 else 0.0).cpu()
        row = m.matrixwise_metrics(eligible, delta, hybrid)
        self.assertAlmostEqual(row['dot_ggn_hybrid'], 98)
        self.assertAlmostEqual(row['optimal_scalar_ggn_to_hybrid'], 1)
        self.assertAlmostEqual(row['relative_residual_after_optimal_scalar'],
                               math.sqrt(0.5), places=6)

    def test_precommitted_categories_and_no_inverse(self):
        self.assertEqual(m.interpretation(0.90, SPEC), 'strong_endpoint_ggn_support')
        self.assertEqual(m.interpretation(0.70, SPEC), 'mixed_endpoint_and_path_effects')
        self.assertEqual(m.interpretation(0.899, SPEC), 'mixed_endpoint_and_path_effects')
        self.assertEqual(m.interpretation(0.699, SPEC), 'endpoint_ggn_insufficient')
        self.assertFalse(SPEC['ggn']['inverse_solve'])
        self.assertFalse(SPEC['ggn']['empirical_fisher'])
        self.assertFalse(SPEC['ggn']['finite_differences'])
        self.assertFalse(SPEC['information_policy']['oracle_adl_access'])
        self.assertFalse(SPEC['information_policy']['adapter_access'])
        source = SOURCE.read_text()
        self.assertNotIn('oracle_adl.pt', source)
        self.assertNotIn('adapter_checkpoint', source)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            output.write_text('frozen')
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(output)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.validate_inputs(spec)
            self.assertEqual(output.read_text(), 'frozen')

    def test_smoke_writes_no_result(self):
        with (patch.object(m, 'load_spec', return_value=SPEC),
              patch.object(m, 'validate_inputs', return_value=({}, {'base_files': [], 'final_files': []})),
              patch.object(m, 'load_pilot_tokens', return_value=torch.zeros((64, 128), dtype=torch.int64)),
              patch.object(m.a104.parent, 'load_model_pair',
                           return_value=(Toy98Model(), Toy98Model(),
                                         {'base_load_seconds': 1.0, 'final_load_seconds': 2.0})),
              patch.object(m, 'prepare_displacement', return_value=({}, {}, {}, [])),
              patch.object(m, 'hybrid_gradient_64', return_value=({}, {})),
              patch.object(m, 'ggn_action_stage', return_value=({},
                           {'logits_jvp_seconds': 3.0, 'fisher_vector_seconds': 4.0,
                            'vjp_backward_seconds': 5.0}, {'per_batch': [], 'aggregate': {}})),
              patch.object(m.a, 'verify_model_unchanged'),
              patch.object(m.a, 'write_manifest', side_effect=AssertionError('wrote result'))):
            result = m.run(smoke_only=True)
        self.assertTrue(result['smoke_only'])
        self.assertFalse(result['writes'])
        self.assertIn('ggn_fisher_conservation_audit', result)

    def test_runtime_projection(self):
        row = m.runtime_projection({'base_load_seconds': 1.0, 'final_load_seconds': 2.0},
                                   3.0, 4.0, 5.0, 6.0, 7.0)
        self.assertEqual(row['projected_hybrid_stage_seconds'], 64.0)
        self.assertEqual(row['projected_ggn_stage_seconds'], (5+6+7)*64)
        self.assertEqual(row['projected_total_wall_seconds'], 6+64+(5+6+7)*64)


if __name__ == '__main__':
    unittest.main()
