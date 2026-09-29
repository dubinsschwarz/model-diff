import ast
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / 'scripts/ablation/smoke_test_generic_hessian_curvature_probe.py'
SPEC = importlib.util.spec_from_file_location('attempt010_hvp_smoke', SCRIPT)
hvp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hvp
SPEC.loader.exec_module(hvp)


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.ModuleDict({key: torch.nn.Linear(1, 1, bias=False)
                                            for key in ('q_proj', 'k_proj', 'v_proj', 'o_proj')})
        self.mlp = torch.nn.ModuleDict({key: torch.nn.Linear(1, 1, bias=False)
                                       for key in ('up_proj', 'gate_proj', 'down_proj')})
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(.03)

    def forward(self, hidden):
        for module in list(self.self_attn.values()) + list(self.mlp.values()):
            hidden = hidden + .01 * torch.sin(module(hidden))
        return hidden


class TinyQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyBlock() for _ in range(28)])
        self.head = torch.nn.Linear(1, 3, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[.2], [-.3], [.5]]))
        self.config = SimpleNamespace(model_type='qwen3', num_hidden_layers=28, _attn_implementation='default_synthetic')
        self.calls = []

    def forward(self, *, input_ids, use_cache):
        self.calls.append((input_ids.detach().cpu().clone(), use_cache, self.training))
        hidden = input_ids.float().unsqueeze(-1) + 1
        for block in self.model.layers:
            hidden = block(hidden)
        return SimpleNamespace(logits=self.head(hidden))


class TrueHVPTests(unittest.TestCase):
    def test_true_hvp_matches_explicit_hessian_cross_parameter_terms(self):
        x = torch.tensor([.4, -.7], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([[.2]], dtype=torch.float64, requires_grad=True)
        parameters = {'x': x, 'y': y}
        direction = {'x': torch.tensor([.3, -1.1], dtype=torch.float64),
                     'y': torch.tensor([[.8]], dtype=torch.float64)}
        def mathematical_loss(flat):
            return torch.sin(flat[0] * flat[2]) + flat[1] ** 3 - 2 * flat[0] ** 2 + flat[1] * flat[2] ** 2
        report = {}
        result = hvp.true_hessian_vector_product(
            lambda: mathematical_loss(torch.cat((x, y.flatten()))), parameters, direction, diagnostics=report)
        flat = torch.cat((x.detach(), y.detach().flatten()))
        explicit = torch.autograd.functional.hessian(mathematical_loss, flat)
        v = torch.cat((direction['x'], direction['y'].flatten()))
        actual = torch.cat((result['x'], result['y'].flatten()))
        torch.testing.assert_close(actual, explicit @ v, rtol=1e-12, atol=1e-12)
        expected_gradient = torch.autograd.functional.jacobian(mathematical_loss, flat)
        self.assertAlmostEqual(report['directional_derivative'], float(expected_gradient @ v), places=12)
        self.assertLess(float(torch.linalg.eigvalsh(explicit).min()), 0.)
        self.assertEqual(list(result), ['x', 'y'])
        self.assertEqual(result['y'].shape, (1, 1))
        self.assertTrue(all(not value.requires_grad and value.grad_fn is None for value in result.values()))
        self.assertTrue(all(parameter.grad is None for parameter in parameters.values()))
        self.assertTrue(torch.equal(flat, torch.cat((x.detach(), y.detach().flatten()))))

    def test_structure_shape_dtype_detachment_and_finiteness_validation(self):
        x = torch.tensor([1., 2.], requires_grad=True)
        parameters = {'x': x}
        for direction in ({}, {'wrong': torch.ones(2)}, {'x': torch.ones(3)},
                          {'x': torch.ones(2, dtype=torch.float64)},
                          {'x': torch.ones(2, requires_grad=True)}, {'x': x * 2},
                          {'x': torch.tensor([float('nan'), 1.])}):
            with self.subTest(direction=direction):
                with self.assertRaises(ValueError):
                    hvp.true_hessian_vector_product(lambda: x.square().sum(), parameters, direction)
        with self.assertRaises(ValueError):
            hvp.true_hessian_vector_product(lambda: x.sum(), {'a': x, 'b': x},
                                           {'a': torch.ones(2), 'b': torch.ones(2)})
        with self.assertRaises(ValueError):
            hvp.true_hessian_vector_product(lambda: x.sum(), {'x': x.detach()}, {'x': torch.ones(2)})
        with self.assertRaises(ValueError):
            hvp.true_hessian_vector_product(lambda: x, parameters, {'x': torch.ones(2)})
        y = torch.ones(1, requires_grad=True)
        with self.assertRaises(ValueError):
            hvp.true_hessian_vector_product(lambda: x.sum() + y.sum(), {'x': x, 'y': y},
                                           {'y': torch.ones(1), 'x': torch.ones(2)})

    def test_constant_first_derivative_has_exact_zero_hessian(self):
        x = torch.tensor([1., -2.], dtype=torch.float64, requires_grad=True)
        result = hvp.true_hessian_vector_product(lambda: (3 * x).sum(), {'x': x}, {'x': torch.tensor([2., 4.], dtype=torch.float64)})
        self.assertTrue(torch.equal(result['x'], torch.zeros_like(x)))

    def test_partial_zero_hessian_and_detached_direction(self):
        x = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([2.], dtype=torch.float64, requires_grad=True)
        direction = {'x': torch.tensor([3.], dtype=torch.float64), 'y': torch.tensor([4.], dtype=torch.float64)}
        result = hvp.true_hessian_vector_product(lambda: (x ** 3 + 2 * y).sum(), {'x': x, 'y': y}, direction)
        self.assertEqual(result['x'].item(), 18.)
        self.assertEqual(result['y'].item(), 0.)

    def test_missing_second_derivative_propagates_exact_exception(self):
        x = torch.tensor([1.], requires_grad=True)
        original = torch.autograd.grad
        calls = []
        error = RuntimeError('derivative for aten::synthetic_attention_backward is not implemented')
        def grad(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise error
            return original(*args, **kwargs)
        report = {}
        with patch.object(torch.autograd, 'grad', side_effect=grad):
            with self.assertRaises(RuntimeError) as caught:
                hvp.true_hessian_vector_product(lambda: x.square().sum(), {'x': x}, {'x': torch.ones(1)}, diagnostics=report)
        self.assertIs(caught.exception, error)
        self.assertEqual(len(calls), 2)
        self.assertEqual(report['directional_derivative'], 2.)

    def test_exact_eligibility_and_synthetic_smoke_two_identical_forwards(self):
        model = TinyQwen()
        eligible = hvp.discover_eligible_linear_weights(model, torch)
        self.assertEqual(len(eligible), 98)
        self.assertEqual([name for name, _ in eligible], sorted(name for name, _ in eligible))
        self.assertEqual({int(name.split('.')[2]) for name, _ in eligible}, set(range(14)))
        before = hvp.model_state_hashes(model, torch)
        batch = torch.arange(8 * 128).reshape(8, 128) % 3
        report = {}
        with patch.object(torch.autograd, 'grad', wraps=torch.autograd.grad) as grad:
            hvp.run_hvp_smoke(model, batch, eligible, report)
        self.assertEqual([call.kwargs['create_graph'] for call in grad.call_args_list], [False, True, False])
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(torch.equal(model.calls[0][0], batch))
        self.assertTrue(torch.equal(model.calls[0][0], model.calls[1][0]))
        self.assertEqual([call[1:] for call in model.calls], [(False, False), (False, False)])
        self.assertEqual(report['gradient_tensor_count'], 98)
        self.assertEqual(report['first_derivative_tensor_count'], 98)
        self.assertEqual(report['hvp_tensor_count'], 98)
        self.assertGreater(report['aggregate_gradient_norm'], 0)
        self.assertGreater(report['aggregate_hvp_norm'], 0)
        self.assertAlmostEqual(report['directional_derivative'], report['aggregate_gradient_norm'] ** 2, places=6)
        self.assertGreaterEqual(report['elapsed_second_order_seconds'], 0)
        self.assertIsNone(report['cuda_peak_allocated_bytes'])
        self.assertTrue(report['parameters_unchanged'])
        self.assertTrue(report['frozen_parameters_unchanged'])
        hvp.verify_model_unchanged(model, before, torch)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertFalse(model.head.weight.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for block in model.model.layers[14:] for parameter in block.parameters()))

    def test_batch_loss_has_exact_mean_causal_target_normalization(self):
        logits = torch.arange(8 * 128 * 3, dtype=torch.float32).reshape(8, 128, 3) / 100
        tokens = torch.arange(8 * 128).reshape(8, 128) % 3
        class Fixed:
            def __call__(self, *, input_ids, use_cache):
                self.input_ids = input_ids
                assert use_cache is False
                return SimpleNamespace(logits=logits)
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 3), tokens[:, 1:].reshape(-1), reduction='mean')
        torch.testing.assert_close(hvp.smoke_batch_loss(Fixed(), tokens), expected, rtol=1e-7, atol=1e-7)

    def test_smoke_failure_cleans_gradients_without_retry(self):
        model = TinyQwen()
        eligible = hvp.discover_eligible_linear_weights(model, torch)
        report = {}
        with patch.object(hvp, 'true_hessian_vector_product', side_effect=RuntimeError('unsupported double backward')) as helper:
            with self.assertRaisesRegex(RuntimeError, 'unsupported double backward'):
                hvp.run_hvp_smoke(model, torch.zeros(8, 128, dtype=torch.int64), eligible, report)
        helper.assert_called_once()
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertIn('elapsed_second_order_seconds', report)
        self.assertGreater(report['aggregate_gradient_norm'], 0)

    def test_model_loader_preserves_backend_and_loads_fp32_offline(self):
        calls = []
        model = TinyQwen()
        with patch.object(model, 'to', return_value=model) as moved:
            module = SimpleNamespace(AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: (calls.append((a, kw)), model)[1]))
            with patch.dict(sys.modules, {'transformers': module}):
                result = hvp.load_local_model(Path('/synthetic/merged'))
        self.assertIs(result, model)
        self.assertEqual(calls, [((Path('/synthetic/merged'),), {'dtype': torch.float32, 'local_files_only': True})])
        moved.assert_called_once_with('cuda')
        self.assertFalse(model.training)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_synthetic_instrumentation_and_fp32(self):
        model = TinyQwen().cuda()
        eligible = hvp.discover_eligible_linear_weights(model, torch)
        report = {}
        events = []
        reset = torch.cuda.reset_peak_memory_stats
        helper = hvp.true_hessian_vector_product
        def reset_call(device):
            events.append('reset')
            return reset(device)
        def helper_call(*args, **kwargs):
            events.append('hvp')
            self.assertTrue(all(value.dtype == torch.float32 and value.device.type == 'cuda'
                                for value in args[2].values()))
            return helper(*args, **kwargs)
        with patch.object(torch.cuda, 'reset_peak_memory_stats', side_effect=reset_call), \
             patch.object(hvp, 'true_hessian_vector_product', side_effect=helper_call):
            hvp.run_hvp_smoke(model, torch.zeros(8, 128, dtype=torch.int64, device='cuda'), eligible, report)
        self.assertEqual(events, ['reset', 'hvp'])
        self.assertGreater(report['cuda_peak_allocated_bytes'], 0)
        self.assertGreaterEqual(report['cuda_peak_reserved_bytes'], report['cuda_peak_allocated_bytes'])

    def test_cli_first_batch_only_no_writes_and_failure_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(merged_model_dir=root / 'model', tokens_path=root / 'tokens.pt',
                                   corpus_spec_path=root / 'corpus_spec.json', corpus_manifest_path=root / 'corpus-manifest.json')
            tokens = torch.arange(4096 * 128).reshape(4096, 128) % 3
            before = set(root.iterdir())
            out, err = io.StringIO(), io.StringIO()
            model = TinyQwen()
            def run(model, batch, eligible, report):
                self.assertTrue(torch.equal(batch.cpu(), tokens[:8]))
                report.update(gradient_tensor_count=98, hvp_tensor_count=98)
            with patch.object(hvp, 'parse_args', return_value=args), \
                 patch.object(hvp, 'sha256_file', side_effect=lambda path: hvp.FROZEN_CORPUS_HASHES.get(path.name, 'a' * 64)), \
                 patch.object(hvp, 'load_json_object', return_value=hvp.FROZEN_CORPUS_SPEC), \
                 patch.object(hvp, 'checkpoint_file_records', return_value=hvp.CANONICAL_CHECKPOINT_FILES), \
                 patch.object(hvp, 'verify_corpus_manifest', return_value={'raw_tensor_sha256': 'a' * 64}), \
                 patch.object(hvp, 'load_corpus_tokens', return_value=tokens), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(hvp, 'load_local_model', return_value=model), \
                 patch.object(hvp, 'run_hvp_smoke', side_effect=run), \
                 patch.object(hvp, 'verify_unchanged'), patch('sys.stdout', out):
                self.assertEqual(hvp.main([]), 0)
            self.assertEqual(json.loads(out.getvalue())['status'], 'passed')
            self.assertEqual(set(root.iterdir()), before)
            with patch.object(hvp, 'parse_args', return_value=args), \
                 patch.object(hvp, 'sha256_file', side_effect=PermissionError('blocked input')), \
                 patch('sys.stdout', out := io.StringIO()), patch('sys.stderr', err):
                self.assertEqual(hvp.main([]), 1)
            self.assertEqual(json.loads(out.getvalue())['exception_type'], 'PermissionError')
            self.assertIn('blocked input', err.getvalue())
            self.assertEqual(set(root.iterdir()), before)

    def test_information_boundary_and_smoke_only_interface(self):
        source = SCRIPT.read_text()
        for forbidden in ('oracle', 'adapter', 'models/base', 'torch.save', 'save_pretrained',
                          'write_manifest', 'Krylov', 'finite_difference', 'attn_implementation='):
            self.assertNotIn(forbidden, source)
        self.assertEqual(hvp.ATTEMPT_ID, '010_generic_hessian_curvature_probe_prefix0_13')
        self.assertEqual(set(vars(hvp.parse_args([]))),
                         {'merged_model_dir', 'tokens_path', 'corpus_spec_path', 'corpus_manifest_path'})
        for filename, expected in hvp.FROZEN_CORPUS_HASHES.items():
            self.assertEqual(hvp.sha256_file(hvp.CORPUS_DIRECTORY / filename), expected)
        ast.parse(source)


if __name__ == '__main__':
    unittest.main()
