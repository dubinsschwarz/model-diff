import copy
import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    loader = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


c = load(ROOT/'scripts/ablation/construct_true_delta_scalar_path_pilot.py', 'attempt020')
toy17 = load(ROOT/'tests/test_construct_full_support_endpoint_hessian_forward.py', 'toy17_for_020')


class ToyLogits(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.proj = torch.nn.Linear(2, 1, bias=False)
        self.proj.weight.data.copy_(torch.tensor([[.13, -.19]]))
        self.register_buffer('offset', torch.tensor(.07))
        self.unselected = torch.nn.Parameter(torch.tensor(.11))
        self.nonlinear = nonlinear
        self.seen_dtypes = []

    def forward(self, *, input_ids, use_cache):
        assert use_cache is False
        self.seen_dtypes.append(self.proj.weight.dtype)
        position = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        features = torch.stack((input_ids.remainder(3).float()/2,
                                position.remainder(3).float()/2), dim=-1)
        score = self.proj(features)
        if self.nonlinear:
            score = torch.tanh(score)
        logits = torch.cat((score+self.unselected+self.offset,
                            -score+self.offset, torch.zeros_like(score)), dim=-1)
        return type('Output', (), {'logits': logits})()


def batch():
    return (torch.arange(8*128, dtype=torch.int64).reshape(8, 128) % 3).contiguous()


class ScalarMathTests(unittest.TestCase):
    def explicit(self, model, tokens, weight):
        logits = torch.func.functional_call(model, {'proj.weight': weight}, (),
                                            {'input_ids': tokens, 'use_cache': False}).logits
        return c.loss_helper.causal_token_loss_sum(logits, tokens, torch)/1016

    def test_spd_first_and_second_match_full_gradient_and_dense_hessian(self):
        model = ToyLogits()
        c.freeze_model(model)
        tokens = batch()
        vector = torch.tensor([[.3, -.4]], dtype=torch.float32)
        direction = {'proj.weight': vector}
        calls = []
        original = torch.autograd.grad
        def spy(outputs, inputs, **kwargs):
            calls.append((tuple(inputs) if isinstance(inputs, (tuple, list)) else (inputs,),
                          kwargs.get('create_graph', False)))
            return original(outputs, inputs, **kwargs)
        with patch.object(torch.autograd, 'grad', side_effect=spy):
            y, h = c.scalar_path(model, tokens, ['proj.weight'], direction, True)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(len(inputs) == 1 and inputs[0].ndim == 0 and inputs[0].dtype == torch.float64
                            for inputs, _ in calls))
        self.assertEqual([create for _, create in calls], [True, False])
        weight = model.proj.weight.detach().clone().requires_grad_(True)
        gradient = original(self.explicit(model, tokens, weight), weight)[0]
        dense = torch.autograd.functional.hessian(lambda w: self.explicit(model, tokens, w), weight).reshape(2, 2)
        expected_y = float((gradient.double()*vector.double()).sum())
        expected_h = float(vector.flatten().double() @ dense.double() @ vector.flatten().double())
        self.assertAlmostEqual(y, expected_y, places=5)
        self.assertAlmostEqual(h, expected_h, places=5)
        self.assertGreater(float(torch.linalg.eigvalsh(dense.double()).min()), 0)
        self.assertTrue(all(dtype == torch.float32 for dtype in model.seen_dtypes))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_nonlinear_toy_and_base_first_only(self):
        model = ToyLogits(nonlinear=True)
        c.freeze_model(model)
        tokens = batch()
        vector = torch.tensor([[-.23, .17]], dtype=torch.float32)
        y, h = c.scalar_path(model, tokens, ['proj.weight'], {'proj.weight': vector}, True)
        weight = model.proj.weight.detach().clone().requires_grad_(True)
        gradient = torch.autograd.grad(self.explicit(model, tokens, weight), weight)[0]
        dense = torch.autograd.functional.hessian(lambda w: self.explicit(model, tokens, w), weight).reshape(2, 2)
        self.assertAlmostEqual(y, float((gradient.double()*vector.double()).sum()), places=5)
        self.assertAlmostEqual(h, float(vector.flatten().double()@dense.double()@vector.flatten().double()), places=5)
        a, second = c.scalar_path(model, tokens, ['proj.weight'], {'proj.weight': vector}, False)
        self.assertIsNone(second)
        self.assertAlmostEqual(a, y, places=6)

    def test_selected_only_unselected_unchanged_and_fp32(self):
        model = ToyLogits()
        c.freeze_model(model)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        direction = {'proj.weight': torch.tensor([[.1, .2]], dtype=torch.float32)}
        c.scalar_path(model, batch(), ['proj.weight'], direction, True)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertTrue(all(dtype == torch.float32 for dtype in model.seen_dtypes))
        with self.assertRaisesRegex(ValueError, 'inventory'):
            c.scalar_path(model, batch(), [], direction, True)
        model.unselected.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            c.scalar_path(model, batch(), ['proj.weight'], direction, True)

    def test_positive_delta_and_fixed_selection_normalization(self):
        base, final, spec = toy17.models()
        delta, audit = toy17.audit(base, final, spec)
        name = 'model.layers.0.self_attn.q_proj.weight'
        torch.testing.assert_close(delta[name],
            (final.get_parameter(name).double()-base.get_parameter(name).double()).float(), rtol=0, atol=0)
        self.assertEqual(len(delta), 196)
        self.assertTrue(audit['outside_support_zero_verified'])
        rows = c.selected_batches()
        self.assertEqual(len(rows), 64)
        self.assertEqual([j for j, _, _ in rows], list(range(0, 512, 8)))
        self.assertEqual(rows[0], (0, 0, 8))
        self.assertEqual(rows[-1], (504, 4032, 4040))
        self.assertTrue(all(end-start == 8 for _, start, end in rows))
        model = ToyLogits()
        tokens = batch()
        logits = model(input_ids=tokens, use_cache=False).logits
        expected = torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, 3),
                                                      tokens[:, 1:].reshape(-1), reduction='sum')/1016
        c.freeze_model(model)
        zero = {'proj.weight': torch.zeros_like(model.proj.weight)}
        self.assertAlmostEqual(c.scalar_path(model, tokens, ['proj.weight'], zero, False)[0], 0.)
        self.assertEqual(c.FROZEN_SPEC['batch_partition']['prediction_tokens_per_microbatch'], 8*127)
        self.assertTrue(torch.isfinite(expected))

    def test_regression_decomposition_and_artifact_inventory(self):
        h = torch.arange(1, 65, dtype=torch.float64)
        a = .3*h+2
        y = a+1.1*h
        result = c.pilot.regression(a, y, h)
        self.assertAlmostEqual(result['beta_g0'], .3)
        self.assertAlmostEqual(result['beta_clean'], 1.1)
        self.assertAlmostEqual(result['beta_final'], result['beta_g0']+result['beta_clean'])
        self.assertEqual(result['count'], 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'batch_scalars.pt'
            record = c.save_artifact(path, a, y, h)
            self.assertEqual(record, c.verify_artifact(path, record))
            stored = torch.load(path, weights_only=True)
            self.assertEqual(stored['microbatch_index'].tolist(), list(range(0, 512, 8)))
            self.assertEqual(stored['row_end'][-1].item(), 4040)
            with self.assertRaisesRegex(ValueError, 'exists'):
                c.save_artifact(path, a, y, h)

    def test_stale_firewall_and_frozen_spec(self):
        path = ROOT/'experiments/attempts'/c.FROZEN_SPEC['attempt_id']/'spec.json'
        self.assertEqual(json.loads(path.read_text()), c.FROZEN_SPEC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = c.parse_args(['--artifact-path', str(root/'batch_scalars.pt'),
                                 '--manifest-path', str(root/'manifest.json')])
            args.artifact_path.touch()
            with self.assertRaisesRegex(ValueError, 'exists'):
                c.require_absent(args)
            args.artifact_path.unlink()
            for name in ('oracle_adl.pt', 'activation_oracle.pt', 'adapter', 'evaluate_attempt018',
                         '018_known_cleaned/evaluation.json'):
                args.tokens_path = root/name
                with self.assertRaisesRegex(ValueError, 'Information firewall'):
                    c.reject_forbidden_paths(args)
        source = Path(c.__file__).read_text()
        self.assertNotIn('finite_difference', source)
        self.assertNotIn('hessian_vector_product', source)
        self.assertNotIn('named_grads', source)


class PipelineTests(unittest.TestCase):
    def test_synthetic_smoke_one_batch_each_no_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = c.parse_args(['--smoke-only', '--device', 'cpu',
                                 '--artifact-path', str(root/'batch_scalars.pt'),
                                 '--manifest-path', str(root/'manifest.json')])
            base, final, spec = toy17.models()
            spec['base']['files'] = []
            tokens = (torch.arange(4096*128, dtype=torch.int64).reshape(4096, 128) % 3).contiguous()
            args.base_directory = root/'base'; args.final_directory = root/'final'
            args.base_directory.mkdir(); args.final_directory.mkdir()
            original = c.scalar_path
            calls = []
            def record(model, tokens, names, direction, second):
                calls.append((model is final, second, tuple(tokens.shape)))
                return original(model, tokens, names, direction, second)
            values = {name: parameter.detach().clone() for name, parameter in base.named_parameters()}
            with (patch.object(c, 'validate_inputs', return_value=(spec, tokens, {}, lambda: None)),
                  patch.object(c.a, 'load_local_model', side_effect=lambda path, *_: final if path == args.final_directory else copy.deepcopy(base)),
                  patch.object(c.execution.BaseTensorReader, 'get', side_effect=lambda name, rec: values[name]),
                  patch.object(c, 'scalar_path', side_effect=record),
                  redirect_stdout(io.StringIO()) as output):
                c.construct(args)
            self.assertEqual(calls, [(True, True, (8, 128)), (False, False, (8, 128))])
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.manifest_path.exists())
            printed = json.loads(output.getvalue().splitlines()[-1])
            self.assertTrue(all(math.isfinite(printed[key]) for key in
                                ('a_0', 'y_0', 'h_0', 'd_0',
                                 'w1_scalar_first_second_seconds', 'w0_scalar_first_seconds')))


if __name__ == '__main__':
    unittest.main()
