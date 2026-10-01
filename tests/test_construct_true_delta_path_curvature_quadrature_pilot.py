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
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load(ROOT/'scripts/ablation/construct_true_delta_path_curvature_quadrature_pilot.py', 'attempt021')
toy20 = load(ROOT/'tests/test_construct_true_delta_scalar_path_pilot.py', 'toy20_for_021')
toy17 = load(ROOT/'tests/test_construct_full_support_endpoint_hessian_forward.py', 'toy17_for_021')


class ScalarAndQuadratureTests(unittest.TestCase):
    def test_path_scalar_second_derivative_at_zero_and_midpoint(self):
        model = toy20.ToyLogits(nonlinear=True)
        c.prior.freeze_model(model)
        batch = toy20.batch()
        delta = torch.tensor([[.17, -.31]], dtype=torch.float32)
        direction = {'proj.weight': delta}
        before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        calls = []
        original = torch.autograd.grad
        def spy(outputs, inputs, **kwargs):
            leaf = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            calls.append((leaf.ndim, leaf.dtype, kwargs.get('create_graph', False)))
            return original(outputs, inputs, **kwargs)
        with patch.object(torch.autograd, 'grad', side_effect=spy):
            h0 = c.scalar_curvature(model, batch, ['proj.weight'], direction, 0.0)
            hmid = c.scalar_curvature(model, batch, ['proj.weight'], direction, 0.5)
        self.assertEqual(calls, [(0, torch.float64, True), (0, torch.float64, False)]*2)
        base_weight = model.proj.weight.detach()
        for point, observed in ((0., h0), (.5, hmid)):
            weight = (base_weight+point*delta).detach().clone().requires_grad_(True)
            dense = torch.autograd.functional.hessian(
                lambda value: toy20.ScalarMathTests().explicit(model, batch, value), weight).reshape(2, 2)
            expected = float(delta.flatten().double() @ dense.double() @ delta.flatten().double())
            self.assertAlmostEqual(observed, expected, places=5)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'Only frozen path points'):
            c.scalar_curvature(model, batch, ['proj.weight'], direction, .25)

    def test_quadrature_formulas_exact_integrals_and_constant_shift_regression(self):
        values = torch.arange(1, 65, dtype=torch.float64)
        quadratic_curvature = values
        approximations = c.curvature_approximations(values, values, values)
        for name in ('endpoint', 'midpoint', 'trapezoid', 'simpson'):
            torch.testing.assert_close(approximations[name], quadratic_curvature, rtol=0, atol=0)
            self.assertAlmostEqual(c.regression(approximations[name]+7, values)['beta'], 1.)
        # phi(t)=2+3t+(k/2)t^2 has exact gradient change k.
        for k in (1., 2., 3.):
            def quadratic_slope(point):
                t = torch.tensor(point, dtype=torch.float64, requires_grad=True)
                phi = 2+3*t+(k/2)*t*t
                return torch.autograd.grad(phi, t)[0].item()
            self.assertAlmostEqual(quadratic_slope(1.)-quadratic_slope(0.), k)
        # h(t)=1+2t+3t^2+4t^3; Simpson integrates cubic curvature exactly.
        h0 = torch.full((64,), 1., dtype=torch.float64)
        hmid = torch.full((64,), 3.25, dtype=torch.float64)
        h1 = torch.full((64,), 10., dtype=torch.float64)
        q = c.curvature_approximations(h0, hmid, h1)
        self.assertEqual(q['midpoint'][0].item(), 3.25)
        self.assertEqual(q['trapezoid'][0].item(), 5.5)
        self.assertEqual(q['simpson'][0].item(), 4.)
        def quintic_slope(point):
            t = torch.tensor(point, dtype=torch.float64, requires_grad=True)
            phi = t*t/2 + t**3/3 + t**4/4 + t**5/5
            return torch.autograd.grad(phi, t)[0].item()
        self.assertEqual(quintic_slope(1.)-quintic_slope(0.), q['simpson'][0].item())

    def test_endpoint_consistency_and_fail_closed(self):
        h = torch.arange(1, 65, dtype=torch.float64)
        a = .2*h+2
        y = a+.6*h
        artifact = {'endpoint_hessian_quadratic': h,
                    'cleaned_directional_difference': (y-a).contiguous()}
        old = c.prior.pilot.regression(a, y, h)
        manifest = {'regression': old}
        result = c.endpoint_consistency(artifact, manifest)
        self.assertAlmostEqual(result['beta'], old['beta_clean'])
        self.assertAlmostEqual(result['correlation'], old['correlations']['h_d'])
        bad = copy.deepcopy(manifest)
        bad['regression']['beta_clean'] += .01
        with self.assertRaisesRegex(ValueError, 'endpoint regression consistency'):
            c.endpoint_consistency(artifact, bad)
        with self.assertRaisesRegex(ValueError, 'Zero Hessian variance'):
            c.regression(torch.ones_like(h), h)

    def test_frozen_inventory_artifact_hashes_and_stale_output(self):
        spec_path = ROOT/'experiments/attempts'/c.FROZEN_SPEC['attempt_id']/'spec.json'
        self.assertEqual(json.loads(spec_path.read_text()), c.FROZEN_SPEC)
        old_manifest = ROOT/'experiments/attempts'/c.prior.FROZEN_SPEC['attempt_id']/'construction-manifest.json'
        old = json.loads(old_manifest.read_text())
        self.assertEqual(c.a.sha256_file(old_manifest), c.PRIOR_MANIFEST_SHA256)
        self.assertEqual(old['artifact']['serialized_sha256'], c.PRIOR_ARTIFACT_SHA256)
        self.assertEqual(c.a.sha256_file(c.PRIOR_PATH), c.PRIOR_SOURCE_SHA256)
        self.assertEqual([index for index, _, _ in c.prior.selected_batches()], list(range(0, 512, 8)))
        values = torch.arange(64, dtype=torch.float64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'batch_scalars.pt'
            record = c.save_artifact(path, values, values+1, values+2, values+3)
            self.assertEqual(record, c.verify_artifact(path, record))
            artifact = torch.load(path, weights_only=True)
            self.assertEqual(set(artifact), set(c.FLOAT_KEYS)|{'microbatch_index'})
            self.assertEqual(artifact['microbatch_index'].tolist(), list(range(0, 512, 8)))
            self.assertEqual(artifact['h1'].dtype, torch.float64)
            self.assertEqual(len(record['raw_tensor_sha256']), 8)
            with self.assertRaisesRegex(ValueError, 'exists'):
                c.save_artifact(path, values, values+1, values+2, values+3)
            args = c.parse_args(['--artifact-path', str(path), '--manifest-path', str(root/'manifest.json')])
            with self.assertRaisesRegex(ValueError, 'exists'):
                c.require_absent(args)

    def test_information_firewall_and_no_vector_derivative_storage(self):
        args = c.parse_args([])
        for bad in ('oracle_adl.pt', 'activation_oracle.pt', 'adapter.pt',
                    '018_known_cleaned/evaluation.json'):
            args.prior_artifact_path = Path('/tmp')/bad
            with self.assertRaisesRegex(ValueError, 'Information firewall'):
                c.prior.reject_forbidden_paths(args)
        source = Path(c.__file__).read_text()
        self.assertNotIn('hessian_vector_product', source)
        self.assertNotIn('finite_difference', source)
        self.assertNotIn('gradients =', source)


class PipelineTests(unittest.TestCase):
    def test_synthetic_smoke_one_batch_no_scientific_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = c.parse_args(['--smoke-only', '--device', 'cpu',
                                 '--artifact-path', str(root/'batch_scalars.pt'),
                                 '--manifest-path', str(root/'manifest.json')])
            base, final, spec = toy17.models()
            spec['base']['files'] = []
            args.base_directory = root/'base'; args.final_directory = root/'final'
            args.base_directory.mkdir(); args.final_directory.mkdir()
            tokens = (torch.arange(4096*128, dtype=torch.int64).reshape(4096, 128) % 3).contiguous()
            _, delta_audit = toy17.audit(base, final, spec)
            old_manifest = {'displacement': delta_audit}
            h = torch.arange(1, 65, dtype=torch.float64)
            d = .5*h
            old_artifact = {'endpoint_hessian_quadratic': h,
                            'cleaned_directional_difference': d}
            endpoint = c.regression(h, d)
            values = {name: parameter.detach().clone() for name, parameter in base.named_parameters()}
            calls = []
            original = c.scalar_curvature
            def record(model, batch, names, direction, point):
                calls.append((point, tuple(batch.shape)))
                return original(model, batch, names, direction, point)
            with (patch.object(c, 'validate_inputs', return_value=(spec, tokens, {}, lambda: None,
                  old_manifest, old_artifact, {}, endpoint)),
                  patch.object(c.a, 'load_local_model', side_effect=lambda path, *_: final if path == args.final_directory else copy.deepcopy(base)),
                  patch.object(c.execution.BaseTensorReader, 'get', side_effect=lambda name, row: values[name]),
                  patch.object(c, 'scalar_curvature', side_effect=record),
                  redirect_stdout(io.StringIO()) as output):
                c.construct(args)
            self.assertEqual(calls, [(0., (8, 128)), (.5, (8, 128))])
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.manifest_path.exists())
            printed = json.loads(output.getvalue().splitlines()[-1])
            self.assertTrue(all(math.isfinite(printed[key]) for key in
                                ('h0_0', 'hmid_0', 'frozen_h1_0', 'frozen_d_0',
                                 'h0_seconds', 'hmid_seconds')))


if __name__ == '__main__':
    unittest.main()
