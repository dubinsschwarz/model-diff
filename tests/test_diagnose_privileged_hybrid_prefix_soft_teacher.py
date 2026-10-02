"""Synthetic/unit tests for Attempt104's privileged hybrid soft teacher."""
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

SOURCE = Path(__file__).resolve().parents[1]/'scripts/ablation/diagnose_privileged_hybrid_prefix_soft_teacher.py'
loader = importlib.util.spec_from_file_location('attempt104_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


class ToyHybridModel(torch.nn.Module):
    def __init__(self, sign=1.0):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(torch.nn.Identity() for _ in range(28))
        self.early = torch.nn.ModuleList(torch.nn.Linear(1, 1, bias=False)
                                         for _ in range(98))
        self.unselected = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = type('Config', (), {'vocab_size': 2, 'use_cache': False})()
        for layer in self.early:
            layer.weight.data.fill_(sign*0.005)
        self.eval()

    def forward(self, input_ids, use_cache, output_hidden_states=False, return_dict=True):
        x = (input_ids.float() % 2).unsqueeze(-1)
        for layer in self.early:
            x = x+0.001*layer(x)
        hidden = self.model.layers[13](x)
        states = [x]*29
        states[14] = hidden
        return type('Output', (), {
            'logits': torch.cat((hidden, -hidden), dim=-1),
            'hidden_states': tuple(states) if output_hidden_states else None})()


def toy_spec():
    spec = copy.deepcopy(SPEC)
    spec['readout']['hidden_size'] = 1
    return spec


def eligible(model):
    return [(f'early.{i}', layer) for i, layer in enumerate(model.early)]


class Attempt104Tests(unittest.TestCase):
    def test_attempt103_exact_hash_linkage_and_result(self):
        self.assertEqual(m.a.sha256_file(m.SPEC_PATH), m.SPEC_SHA256)
        self.assertEqual(m.a.sha256_file(m.PARENT_SOURCE), m.PARENT_SOURCE_SHA256)
        result = m.validate_attempt103_linkage(SPEC)
        self.assertEqual(result['provenance']['matched_difference_raw_sha256'],
                         SPEC['attempt103']['matched_target_raw_sha256'])
        self.assertEqual(result['matched_probe']['positions_1_4_mean_cosine'],
                         0.5956728332916521)
        wrong = copy.deepcopy(SPEC)
        wrong['attempt103']['result_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            m.validate_attempt103_linkage(wrong)

    def test_frozen_base_final_and_batch_inventories(self):
        self.assertEqual(SPEC['base'], m.parent.load_spec()['base'])
        self.assertEqual(SPEC['final_checkpoint_files'],
                         m.parent.load_spec()['final_checkpoint_files'])
        self.assertEqual(SPEC['corpus']['construction_rows'], [0, 512])
        self.assertEqual(SPEC['corpus']['batch_size'], 4)
        self.assertEqual(SPEC['probe_rows'], [0, 1024])
        self.assertEqual(SPEC['probe']['batch_size'], 32)
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            spec['paths']['result_path'] = str(Path(directory)/'result.json')
            with (patch.object(m, 'validate_attempt103_linkage', return_value={}),
                  patch.object(m.parent, 'validate_sources',
                               side_effect=ValueError('Base/final checkpoint inventory mismatch'))):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    m.validate_inputs(spec)

    def test_tokenizer_equality_reuses_attempt103_validation(self):
        class Tokenizer:
            def __init__(self, vocab):
                self.vocab = vocab
            def get_vocab(self):
                return self.vocab
            def __len__(self):
                return len(self.vocab)
        base, final = ToyHybridModel(-1), ToyHybridModel(1)
        good = Tokenizer({'a': 0, 'b': 1})
        self.assertEqual(m.parent.validate_tokenizer_mapping(good, good, base, final), 2)
        with self.assertRaisesRegex(ValueError, 'mapping'):
            m.parent.validate_tokenizer_mapping(good, Tokenizer({'a': 1, 'b': 0}),
                                                base, final)

    def test_exact_corpus_probe_slices_reuse_attempt103_loader(self):
        corpus = torch.arange(4096, dtype=torch.int64)[:, None].repeat(1, 128)
        probe = torch.arange(10000, dtype=torch.int64)[:, None].repeat(1, 128)
        with (patch.object(m.a, 'validate_frozen_corpus', return_value=(corpus, {}, {})),
              patch.object(m.a, 'sha256_raw_int64_tensor',
                           return_value=SPEC['corpus']['raw_tensor_sha256']),
              patch.object(m.a, 'load_probe', return_value=probe)):
            selected, matched = m.parent.load_frozen_tokens(SPEC)
        self.assertEqual(tuple(selected.shape), (512, 128))
        self.assertEqual(tuple(matched.shape), (1024, 128))
        self.assertEqual(selected[-1, 0].item(), 511)
        self.assertEqual(matched[-1, 0].item(), 1023)

    def test_base_block13_capture_matches_hidden_states14(self):
        base = ToyHybridModel(-1)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        batch = torch.ones((4, 128), dtype=torch.int64)
        layer = base.model.layers[13]
        before = len(layer._forward_hooks)
        hidden, logits = m.base_hidden_and_logits(base, batch, toy_spec())
        self.assertEqual(tuple(hidden.shape), (4, 128, 1))
        self.assertFalse(hidden.requires_grad)
        self.assertEqual(tuple(logits.shape), (4, 128, 2))
        self.assertEqual(len(layer._forward_hooks), before)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))

    def test_tensor_hook_replacement(self):
        original = torch.zeros((2, 3, 4), dtype=torch.float32)
        replacement = torch.ones_like(original)
        output = m.replace_hidden_output(original, replacement)
        self.assertTrue(torch.equal(output, replacement))
        self.assertFalse(output.requires_grad)

    def test_tuple_hook_replacement_preserves_remainder(self):
        hidden = torch.zeros((2, 3, 4))
        other = object()
        output = m.replace_hidden_output((hidden, other, None), torch.ones_like(hidden))
        self.assertIsInstance(output, tuple)
        self.assertIs(output[1], other)
        self.assertIsNone(output[2])

    def test_list_hook_replacement_preserves_remainder(self):
        hidden = torch.zeros((2, 3, 4))
        other = object()
        output = m.replace_hidden_output([hidden, other], torch.ones_like(hidden))
        self.assertIsInstance(output, list)
        self.assertIs(output[1], other)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            m.replace_hidden_output({'hidden': hidden}, torch.ones_like(hidden))

    def test_replacement_shape_dtype_and_device_fail_closed(self):
        hidden = torch.zeros((2, 3, 4), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            m.replace_hidden_output(hidden, torch.zeros((2, 3, 5)))
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            m.replace_hidden_output(hidden, hidden.double())

    def test_hybrid_hook_fires_once_is_removed_and_inference_has_no_grad(self):
        final = ToyHybridModel(1)
        batch = torch.ones((4, 128), dtype=torch.int64)
        base_hidden = torch.full((4, 128, 1), 0.5)
        layer = final.model.layers[13]
        before = len(layer._forward_hooks)
        logits, audit = m.hybrid_logits(final, batch, base_hidden, toy_spec())
        self.assertEqual(audit, {'hook_fired_count': 1, 'hook_removed': True,
                                 'hybrid_logits_finite': True})
        self.assertEqual(len(layer._forward_hooks), before)
        self.assertFalse(logits.requires_grad)
        self.assertTrue(torch.allclose(logits[..., 0], base_hidden[..., 0]))
        self.assertTrue(all(parameter.grad is None for parameter in final.parameters()))

    def test_hybrid_hook_removed_after_forward_failure(self):
        class Failing(ToyHybridModel):
            def forward(self, *args, **kwargs):
                super().forward(*args, **kwargs)
                raise RuntimeError('synthetic downstream failure')
        final = Failing(1)
        layer = final.model.layers[13]
        with self.assertRaisesRegex(RuntimeError, 'synthetic downstream failure'):
            m.hybrid_logits(final, torch.ones((4, 128), dtype=torch.int64),
                            torch.ones((4, 128, 1)), toy_spec())
        self.assertEqual(len(layer._forward_hooks), 0)

    def test_exact_hybrid_ce_kl_and_logit_residual(self):
        z_hybrid = torch.tensor([[[1., -1., 0.5]]])
        z_student = torch.tensor([[[0.3, 0.4, -0.2]]], requires_grad=True)
        ce, kl, p_hybrid = m.hybrid_teacher_losses(z_hybrid, z_student)
        expected_ce = -(torch.softmax(z_hybrid, -1)*
                        torch.log_softmax(z_student, -1)).sum()
        expected_kl = (torch.softmax(z_hybrid, -1)*(
            torch.log_softmax(z_hybrid, -1)-
            torch.log_softmax(z_student, -1))).sum()
        self.assertAlmostEqual(float(ce.detach()), float(expected_ce.detach()))
        self.assertAlmostEqual(float(kl.detach()), float(expected_kl.detach()))
        gradient, = torch.autograd.grad(ce, z_student)
        self.assertTrue(torch.allclose(gradient,
                        torch.softmax(z_student.detach(), -1)-p_hybrid, atol=1e-7))
        self.assertIsNone(SPEC['hybrid_teacher']['temperature'])
        self.assertIsNone(SPEC['hybrid_teacher']['top_k'])

    def test_positive_student_gradient_only_98_early_and_no_base_grad(self):
        base, final = ToyHybridModel(-1), ToyHybridModel(1)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        selected = eligible(final)
        tokens = torch.ones((4, 128), dtype=torch.int64)
        summary, timings = m.hybrid_teacher_gradient_stage(base, final, tokens,
                                                             selected, toy_spec())
        self.assertEqual(summary['total_prediction_tokens'], 4*127)
        self.assertEqual(summary['first_batch_hybrid_hook_audit']['hook_fired_count'], 1)
        self.assertTrue(all(layer.weight.grad is not None for _, layer in selected))
        self.assertIsNone(final.unselected.grad)
        self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
        self.assertTrue(all(value >= 0 for value in timings.values()))
        positive = selected[0][1].weight.grad.clone()
        final.zero_grad(set_to_none=True)
        hidden, _ = m.base_hidden_and_logits(base, tokens, toy_spec())
        z_hybrid, _ = m.hybrid_logits(final, tokens, hidden, toy_spec())
        z_student = final(input_ids=tokens, use_cache=False).logits[:, :-1, :]
        ce, _, _ = m.hybrid_teacher_losses(z_hybrid[:, :-1, :], z_student)
        (ce/(4*127)).backward()
        self.assertTrue(torch.equal(positive, selected[0][1].weight.grad))

    def test_tangent_scale_and_strict_jvp_reuse(self):
        model = ToyHybridModel(1)
        selected = eligible(model)
        for _, layer in selected:
            layer.weight.grad = torch.full_like(layer.weight, 0.5)
        scale = m.a.global_tangent_scale(selected, 0.00125, torch)
        tangent, _, realized = m.a.prepare_tangents(selected, scale, torch)
        self.assertEqual(len(tangent), 98)
        self.assertGreater(scale['alpha'], 0)
        self.assertAlmostEqual(realized['aggregate_realized_relative_tangent_norm'],
                               0.00125, places=9)
        spec = toy_spec()
        spec['probe']['sample_count'] = 32
        probe = torch.zeros((32, 128), dtype=torch.int64)
        readout = torch.ones((32, 128, 1), dtype=torch.float32)
        with (patch.object(m.a, 'ordinary_hook_readout', return_value=readout),
              patch.object(m.a, 'run_readout_jvp', return_value=(readout, 0.25*readout)) as jvp,
              patch.object(m.a, 'compare_primal', return_value=0.0)):
            _, response, audit = m.parent.final_probe_jvp(model, probe, tangent, spec)
        self.assertEqual(jvp.call_count, 1)
        self.assertTrue(torch.all(response == 0.25))
        self.assertEqual(audit, 0.0)

    def test_matched_target_hash_validation(self):
        target = torch.ones((128, 2048), dtype=torch.float32)
        with patch.object(m.a, 'sha256_raw_float32_tensor',
                          return_value=SPEC['attempt103']['matched_target_raw_sha256']):
            self.assertEqual(m.verify_matched_target(target, SPEC),
                             SPEC['attempt103']['matched_target_raw_sha256'])
        with patch.object(m.a, 'sha256_raw_float32_tensor', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, 'Matched target'):
                m.verify_matched_target(target, SPEC)

    def test_positionwise_primary_secondary_reuses_attempt103(self):
        response = torch.ones((128, 2048), dtype=torch.float32)
        target = response.clone()
        target[2] = -1
        metrics = m.parent.response_metrics(response, target)
        self.assertAlmostEqual(metrics['position_0_cosine'], 1)
        self.assertAlmostEqual(metrics['positions_1_4_mean_cosine'], 0.5)
        self.assertAlmostEqual(metrics['positions_1_127_mean_cosine'], 125/127)
        self.assertEqual(len(metrics['position_cosines']), 128)

    def test_streaming_output_residual_geometry(self):
        base = torch.tensor([0.5, 0.5]).expand(1, 127, 2).clone()
        hybrid = torch.tensor([0.6, 0.4]).expand(1, 127, 2).clone()
        final = torch.tensor([0.7, 0.3]).expand(1, 127, 2).clone()
        accumulator = m.ResidualAccumulator()
        accumulator.add(base, hybrid, final)
        report = accumulator.summary()
        self.assertEqual(report['prediction_count'], 127)
        self.assertAlmostEqual(report['pooled_cosine_hybrid_vs_full'], 1)
        self.assertAlmostEqual(report['hybrid_to_full_norm_ratio'], 0.5, places=6)
        self.assertAlmostEqual(report['mean_per_prediction_cosine'], 1)
        self.assertEqual(report['fraction_per_prediction_cosine_positive'], 1)
        hybrid[:, :64] = torch.tensor([0.8, 0.2])
        accumulator = m.ResidualAccumulator()
        accumulator.add(base, hybrid, final)
        self.assertEqual(accumulator.summary()['fraction_per_prediction_cosine_positive'], 63/127)

    def test_precommitted_interpretation_categories(self):
        self.assertEqual(m.interpretation_category(0.75, 0.10, SPEC),
                         'strong_late_layer_contamination_support')
        self.assertEqual(m.interpretation_category(0.69, 0.049, SPEC),
                         'curvature_limited_support')
        self.assertEqual(m.interpretation_category(0.72, 0.07, SPEC), 'ambiguous')
        self.assertFalse(SPEC['interpretation']['significance_claim'])

    def test_no_oracle_adl_adapter_or_overwrite(self):
        for forbidden in ('/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt',
                          '/root/model-diff-scratch/models/adapter',
                          'experiments/attempts/014/evaluation.json'):
            with self.assertRaisesRegex(ValueError, 'Forbidden'):
                m.parent.resolve(forbidden)
        with tempfile.TemporaryDirectory() as directory:
            spec = copy.deepcopy(SPEC)
            path = Path(directory)/'result.json'
            path.write_text('{}')
            spec['paths']['result_path'] = str(path)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.validate_inputs(spec)
        self.assertFalse(SPEC['information_policy']['oracle_adl_access'])
        self.assertFalse(SPEC['information_policy']['adapter_access'])

    def test_runtime_projection_arithmetic(self):
        loads = {'base_load_seconds': 2.0, 'final_load_seconds': 3.0}
        timings = {'base_hidden_logits_seconds': 5.0,
                   'hybrid_final_seconds': 7.0,
                   'student_backward_seconds': 11.0}
        projected = m.runtime_projection(loads, timings, 13.0, 17.0)
        self.assertEqual(projected['projected_full_teacher_stage_seconds'], 23*128)
        self.assertEqual(projected['projected_full_probe_stage_seconds'], 30*32)
        self.assertEqual(projected['projected_total_wall_seconds'], 5+23*128+30*32)

    def test_smoke_writes_no_scientific_output(self):
        base, final = ToyHybridModel(-1), ToyHybridModel(1)
        selected = eligible(final)
        old_spec = m.parent.load_spec()
        zeros = torch.zeros((128, 2048), dtype=torch.float64)
        ones = torch.ones_like(zeros)
        timings = {'base_hidden_logits_seconds': 1.0,
                   'hybrid_final_seconds': 2.0,
                   'student_backward_seconds': 3.0}
        teacher = {'first_batch_hybrid_hook_audit': {'hook_fired_count': 1,
                                                     'hook_removed': True}}
        with (patch.object(m, 'load_spec', return_value=SPEC),
              patch.object(m, 'validate_inputs', return_value=({}, old_spec, {})),
              patch.object(m.parent, 'load_frozen_tokens', return_value=(
                  torch.zeros((512, 128), dtype=torch.int64),
                  torch.zeros((1024, 128), dtype=torch.int64))),
              patch.object(m.parent, 'load_model_pair', return_value=(base, final, {
                  'base_load_seconds': 1.0, 'final_load_seconds': 2.0, 'vocab_size': 2})),
              patch.object(m.a, 'discover_eligible_linear_weights', return_value=selected),
              patch.object(m.a, 'freeze_other_parameters'),
              patch.object(m.a, 'model_state_hashes', return_value={}),
              patch.object(m, 'hybrid_teacher_gradient_stage', return_value=(teacher, timings)),
              patch.object(m.a, 'global_tangent_scale', return_value={}),
              patch.object(m.a, 'prepare_tangents', return_value=(
                  {str(i): torch.ones(1) for i in range(98)}, [], {})),
              patch.object(m.parent, 'base_probe_mean', return_value=zeros),
              patch.object(m.parent, 'final_probe_jvp', return_value=(ones, ones.float(), 0.0)),
              patch.object(m.a, 'verify_model_unchanged'),
              patch.object(m.a, 'write_manifest', side_effect=AssertionError('smoke wrote output'))):
            result = m.run(smoke_only=True)
        self.assertTrue(result['smoke_only'])
        self.assertFalse(result['writes'])
        self.assertEqual(result['projected_full_teacher_stage_seconds'], 6*128)


if __name__ == '__main__':
    unittest.main()
