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


c = load(ROOT/'scripts/ablation/construct_centered_batch_hessian_regression.py', 'attempt019')
toy17 = load(ROOT/'tests/test_construct_full_support_endpoint_hessian_forward.py', 'toy17_for_019')


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 3, bias=False)
        self.proj.weight.data.copy_(torch.tensor([[.2], [-.1], [.3]]))

    def forward(self, *, input_ids, use_cache):
        assert use_cache is False
        x = input_ids.float().unsqueeze(-1)/3
        return type('Output', (), {'logits': self.proj(x)})()


class MathematicsTests(unittest.TestCase):
    def test_frozen_spec_inventory_sign_outside_support(self):
        path = ROOT/'experiments/attempts'/c.FROZEN_SPEC['attempt_id']/'spec.json'
        self.assertEqual(json.loads(path.read_text()), c.FROZEN_SPEC)
        base, final, spec = toy17.models()
        eligible = c.support.discover(final, spec)
        self.assertEqual(len(eligible), 196)
        self.assertEqual({int(name.split('.')[2]) for name, _ in eligible}, set(range(28)))
        delta, audit = toy17.audit(base, final, spec)
        for name, value in delta.items():
            expected = (final.get_parameter(name).double()-base.get_parameter(name).double()).float()
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        self.assertTrue(audit['outside_support_zero_verified'])
        final.head.weight.data.add_(.1)
        with self.assertRaisesRegex(ValueError, 'outside full support'):
            toy17.audit(base, final, spec)

    def test_partition_and_mean_denominator(self):
        groups = c.partition()
        self.assertEqual(len(groups), 512)
        self.assertEqual(groups[0], (0, 8))
        self.assertEqual(groups[-1], (4088, 4096))
        self.assertEqual([row for start, end in groups for row in range(start, end)], list(range(4096)))
        batch = torch.arange(8*128, dtype=torch.int64).reshape(8, 128) % 3
        model = TinyLM()
        expected = torch.nn.functional.cross_entropy(
            model(input_ids=batch, use_cache=False).logits[:, :-1, :].reshape(-1, 3),
            batch[:, 1:].reshape(-1), reduction='sum') / (8*127)
        torch.testing.assert_close(c.batch_mean_loss(model, batch), expected)

    def test_directional_gradient_and_exact_hessian_same_graph(self):
        model = TinyLM()
        batch = torch.arange(8*128, dtype=torch.int64).reshape(8, 128) % 3
        vector = torch.tensor([[.2], [-.3], [.4]], dtype=torch.float32)
        direction = {'proj.weight': vector}
        eligible = [('proj', model.proj)]
        calls = []
        original = torch.autograd.grad

        def spy(*args, **kwargs):
            calls.append(kwargs.get('create_graph', False))
            return original(*args, **kwargs)

        with patch.object(torch.autograd, 'grad', side_effect=spy):
            y, h = c.batch_directional(model, batch, eligible, direction, direction, True)
        self.assertEqual(calls, [True, False])
        loss = c.batch_mean_loss(model, batch)
        gradient = original(loss, model.proj.weight)[0]
        expected_y = float((gradient.double()*vector.double()).sum())
        def standalone(weight):
            logits = torch.nn.functional.linear(batch.float().unsqueeze(-1)/3, weight)
            return torch.nn.functional.cross_entropy(logits[:, :-1, :].reshape(-1, 3),
                                                     batch[:, 1:].reshape(-1), reduction='sum') / 1016
        dense = torch.autograd.functional.hessian(standalone, model.proj.weight).reshape(3, 3)
        expected_h = float(vector.flatten().double() @ dense.double() @ vector.flatten().double())
        self.assertAlmostEqual(y, expected_y, places=7)
        self.assertAlmostEqual(h, expected_h, places=6)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_centering_intercept_constant_nuisance_and_decomposition(self):
        h = torch.arange(1, 9, dtype=torch.float64)
        a = torch.full_like(h, 13.)
        y = a+h
        result = c.regression(a, y, h)
        self.assertAlmostEqual(result['beta_final'], 1.)
        self.assertAlmostEqual(result['beta_g0'], 0.)
        self.assertAlmostEqual(result['beta_clean'], 1.)
        shifted = c.regression(a, y+111., h)
        self.assertAlmostEqual(shifted['beta_final'], result['beta_final'])
        self.assertAlmostEqual(shifted['beta_final'], shifted['beta_g0']+shifted['beta_clean'])

    def test_correlated_base_nuisance_and_path_mismatch(self):
        h = torch.arange(1, 9, dtype=torch.float64)
        a = .25*h+7
        y = a+h
        result = c.regression(a, y, h)
        self.assertAlmostEqual(result['beta_g0'], .25)
        self.assertAlmostEqual(result['beta_clean'], 1.)
        self.assertAlmostEqual(result['beta_final'], 1.25)
        no_nuisance = c.regression(torch.zeros_like(h), 1.3*h, h)
        self.assertEqual(no_nuisance['beta_g0'], 0)
        self.assertAlmostEqual(no_nuisance['beta_clean']-1, .3)
        self.assertAlmostEqual(no_nuisance['beta_final']-1, .3)

    def test_aggregation_means_primary_and_fail_closed(self):
        h = torch.arange(512, dtype=torch.float64)
        a = 2*h+1
        y = 3*h+2
        grouped = c.aggregate(h, 4)
        torch.testing.assert_close(grouped[:2], torch.tensor([1.5, 5.5], dtype=torch.float64))
        primary, secondary = c.diagnostics(a, y, h)
        self.assertEqual(primary['group_factor'], 1)
        self.assertEqual(primary['beta_final'], c.regression(a, y, h)['beta_final'])
        self.assertEqual([(row['group_factor'], row['effective_batch_size'], row['group_count'])
                          for row in secondary], [(r, 8*r, 512//r) for r in (2, 4, 8, 16, 32, 64)])
        with self.assertRaisesRegex(ValueError, 'Zero Hessian variance'):
            c.regression(h, y, torch.ones_like(h))
        for values in (torch.tensor([1., math.nan], dtype=torch.float64),
                       torch.tensor([1., math.inf], dtype=torch.float64)):
            with self.assertRaisesRegex(ValueError, 'finite'):
                c.regression(values, values, values)

    def test_artifact_hash_inventory_stale_and_firewall(self):
        series = torch.arange(512, dtype=torch.float64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'batch_scalars.pt'
            record = c.save_artifact(path, series, series+2, series+3)
            self.assertEqual(record, c.verify_artifact(path, record))
            self.assertEqual(len(record['raw_tensor_sha256']), 4)
            with self.assertRaisesRegex(ValueError, 'exists'):
                c.save_artifact(path, series, series+2, series+3)
            wrong = copy.deepcopy(record)
            wrong['raw_tensor_sha256'][c.SCALAR_KEYS[0]] = '0'*64
            with self.assertRaisesRegex(ValueError, 'hash/inventory'):
                c.verify_artifact(path, wrong)
            tampered = torch.load(path, weights_only=True)
            tampered['batch_start'] = tampered['batch_start'].to(torch.float64)
            torch.save(tampered, path)
            with self.assertRaisesRegex(ValueError, 'index metadata'):
                c.verify_artifact(path)
            args = c.parse_args(['--artifact-path', str(path), '--manifest-path', str(Path(directory)/'manifest.json')])
            with self.assertRaisesRegex(ValueError, 'exists'):
                c.require_absent(args)
            for forbidden in ('oracle_adl.pt', 'adapter', 'evaluate_attempt018'):
                args.artifact_path = Path(directory)/forbidden
                with self.assertRaisesRegex(ValueError, 'Information firewall'):
                    c.reject_forbidden_paths(args)
        source = Path(c.__file__).read_text()
        self.assertNotIn('torch.linalg.inv', source)
        self.assertNotIn('finite_difference', source)


class PipelineTests(unittest.TestCase):
    def test_smoke_one_each_no_scientific_outputs_and_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = c.parse_args(['--smoke-only', '--device', 'cpu',
                                 '--artifact-path', str(root/'batch_scalars.pt'),
                                 '--manifest-path', str(root/'manifest.json')])
            base, final, spec = toy17.models()
            spec['base']['files'] = []
            tokens = (torch.arange(4096*128, dtype=torch.int64).reshape(4096, 128) % 3).contiguous()
            args.base_directory = root/'base'
            args.final_directory = root/'final'
            args.base_directory.mkdir(); args.final_directory.mkdir()
            def loader(path, *_):
                return final if path == args.final_directory else copy.deepcopy(base)
            calls = []
            original = c.batch_directional
            def record(model, batch, eligible, direction_gpu, direction_cpu, exact_hessian):
                calls.append((model is final, exact_hessian, tuple(batch.shape)))
                return original(model, batch, eligible, direction_gpu, direction_cpu, exact_hessian)
            values = {name: p.detach().clone() for name, p in base.named_parameters()}
            with (patch.object(c, 'validate_inputs', return_value=(spec, tokens, {}, lambda: None)),
                  patch.object(c.a, 'load_local_model', side_effect=loader),
                  patch.object(c.execution.BaseTensorReader, 'get', side_effect=lambda name, rec: values[name]),
                  patch.object(c, 'batch_directional', side_effect=record),
                  redirect_stdout(io.StringIO()) as output):
                c.construct(args)
            self.assertEqual(calls, [(True, True, (8, 128)), (False, False, (8, 128))])
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.manifest_path.exists())
            printed = json.loads(output.getvalue().splitlines()[-1])
            self.assertTrue(all(math.isfinite(printed[key]) for key in ('a_0', 'y_0', 'h_0', 'd_0')))

    def test_mutation_detection(self):
        base, _, spec = toy17.models()
        before = c.a.model_state_hashes(base, torch)
        execution = c.execution.execution_snapshot(base, spec)
        base.model.offset.add_(1)
        with self.assertRaisesRegex(ValueError, 'changed'):
            c.support.verify_state(base, before, execution, spec)


if __name__ == '__main__':
    unittest.main()
