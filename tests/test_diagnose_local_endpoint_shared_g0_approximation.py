"""Synthetic tests for the privileged Attempt 025 scalar-plane diagnostic."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT/'scripts/ablation/diagnose_local_endpoint_shared_g0_approximation.py'
SPEC = PROJECT/'experiments/attempts/025_local_endpoint_shared_g0_approximation_pilot/spec.json'
loader = importlib.util.spec_from_file_location('attempt025_test_subject', SCRIPT)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)


class ToyModel(torch.nn.Module):
    def __init__(self, endpoint=0.7):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([endpoint], dtype=torch.float32),
                                         requires_grad=False)

    def forward(self, input_ids, use_cache=False):
        if use_cache:
            raise AssertionError('Cache must be disabled')
        return type('ToyOutput', (), {'logits': self.weight[0]})()


class TwoCoordinateToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.early = torch.nn.Parameter(torch.tensor([0.3], dtype=torch.float32),
                                        requires_grad=False)
        self.late = torch.nn.Parameter(torch.tensor([-0.2], dtype=torch.float32),
                                       requires_grad=False)

    def forward(self, input_ids, use_cache=False):
        if use_cache:
            raise AssertionError('Cache must be disabled')
        return type('ToyOutput', (), {'logits': torch.stack((self.early[0], self.late[0]))})()


def coupled_loss(logits, batch):
    x, y = logits.unbind()
    return 0.5*x*x+0.7*x*y+0.5*y*y+0.2*x*x*y+0.1*x*y*y


def coupled_expected(t):
    x, y = 0.3+0.25*t, -0.2
    dx, dy, ux = 0.4, -0.6, 0.25
    gx = x+0.7*y+0.4*x*y+0.1*y*y
    gy = y+0.7*x+0.2*x*x+0.2*x*y
    hxx, hxy, hyy = 1+0.4*y, 0.7+0.4*x+0.2*y, 1+0.2*x
    return (dx*gx+dy*gy,
            dx*dx*hxx+2*dx*dy*hxy+dy*dy*hyy,
            ux*(dx*hxx+dy*hxy))


def polynomial_loss(logits, batch):
    return logits**4/4+0.3*logits**3/3+0.2*logits**2/2


def analytic(t):
    x = 0.7+0.2*t
    first_x = x**3+0.3*x*x+0.2*x
    second_x = 3*x*x+0.6*x+0.2
    return 0.4*first_x, 0.4**2*second_x, 0.4*0.2*second_x


def toy_plane_points():
    model = ToyModel()
    batch = torch.zeros((8, 128), dtype=torch.int64)
    delta = {'weight': torch.tensor([0.4], dtype=torch.float32)}
    q = {'weight': torch.tensor([0.8], dtype=torch.float32)}
    return [m.scalar_plane_point(model, batch, ['weight'], delta, q, 0.25, t, polynomial_loss)
            for t in (0.0, 0.5, 1.0)], model


class TestLocalEndpointSharedG0(unittest.TestCase):
    def test_frozen_spec_and_inventory(self):
        self.assertEqual(json.loads(SPEC.read_text()), m.FROZEN_SPEC)
        self.assertEqual(len(m.FROZEN_SPEC['coordinates']), 196)
        self.assertEqual(len(m.FROZEN_SPEC['early_coordinates']), 98)
        self.assertEqual(m.FROZEN_SPEC['true_delta_support']['delta_support_count'], 196)
        self.assertEqual(m.FROZEN_SPEC['true_delta_support']['perturbation_support_count'], 98)
        self.assertNotIn('plane_support_count', m.FROZEN_SPEC['true_delta_support'])
        self.assertEqual(m.batch_inventory(), [(j, 8*j, 8*(j+1)) for j in range(16)])
        self.assertEqual(m.batch_inventory(True), [(0, 0, 8)])
        self.assertEqual(m.FROZEN_SPEC['batches']['prediction_tokens_per_microbatch'], 1016)

    def test_scalar_first_and_mixed_second_derivatives(self):
        points, model = toy_plane_points()
        for observed, t in zip(points, (0.0, 0.5, 1.0)):
            for actual, expected in zip(observed, analytic(t)):
                self.assertAlmostEqual(actual, expected, delta=2e-6)
        self.assertIsNone(model.weight.grad)
        self.assertFalse(model.weight.requires_grad)

    def test_full_delta_with_sparse_early_u_and_nonzero_cross_curvature(self):
        model = TwoCoordinateToy()
        batch = torch.zeros((8, 128), dtype=torch.int64)
        delta = {'early': torch.tensor([0.4], dtype=torch.float32),
                 'late': torch.tensor([-0.6], dtype=torch.float32)}
        q = {'early': torch.tensor([1.0], dtype=torch.float32)}
        points = [m.scalar_plane_point(model, batch, ['early', 'late'], delta, q,
                                       0.25, t, coupled_loss) for t in (0.0, 0.5, 1.0)]
        for actual, t in zip(points, (0.0, 0.5, 1.0)):
            for observed, expected in zip(actual, coupled_expected(t)):
                self.assertAlmostEqual(observed, expected, delta=3e-6)
        record = m.compose_record(0, 0, *points)
        self.assertAlmostEqual(record['historical_rhs'], -0.03, delta=3e-6)
        # Truncating Delta to the early coordinate makes this RHS zero.
        self.assertGreater(abs(record['historical_rhs']), 0.02)
        self.assertIsNone(model.early.grad)
        self.assertIsNone(model.late.grad)
        self.assertEqual(set(q), {'early'})

    def test_exact_local_gradient_difference_by_integral(self):
        points, _ = toy_plane_points()
        change = points[2][0]-points[0][0]
        # This quartic potential has quadratic mixed curvature along t.
        simpson = (points[0][2]+4*points[1][2]+points[2][2])/6
        self.assertAlmostEqual(change, simpson, delta=3e-7)

    def test_endpoint_residual_formula(self):
        points, _ = toy_plane_points()
        record = m.compose_record(0, 0, *points)
        self.assertAlmostEqual(record['endpoint_residual'],
                               points[2][0]-points[0][0]-points[2][2], delta=1e-12)

    def test_trapezoid_formula(self):
        points, _ = toy_plane_points()
        record = m.compose_record(0, 0, *points)
        expected = points[2][0]-points[0][0]-(points[0][2]+points[2][2])/2
        self.assertAlmostEqual(record['trapezoid_residual'], expected, delta=1e-12)

    def test_simpson_formula(self):
        points, _ = toy_plane_points()
        record = m.compose_record(0, 0, *points)
        expected = points[2][0]-points[0][0]-(points[0][2]+4*points[1][2]+points[2][2])/6
        self.assertAlmostEqual(record['simpson_residual'], expected, delta=1e-12)
        self.assertAlmostEqual(record['simpson_residual'], 0.0, delta=3e-7)

    def test_shared_g0_cancels_under_endpoint_approximation(self):
        nuisance, hdd0, hdd1, hdu1 = 12.5, 0.8, 1.3, -0.4
        a0 = nuisance+hdd0
        a1 = nuisance+hdd1+hdu1
        row = m.compose_record(0, 0, (a0, hdd0, 0.1), (0, 0, 0.2),
                               (a1, hdd1, hdu1))
        self.assertAlmostEqual(row['endpoint_residual'], hdd1-hdd0)
        self.assertAlmostEqual(row['historical_rhs'], hdd1-hdd0)

    def test_endpoint_approximation_not_exact_identity(self):
        points, _ = toy_plane_points()
        row = m.compose_record(0, 0, *points)
        self.assertGreater(abs(row['endpoint_residual']-row['historical_rhs']), 1e-3)
        self.assertLess(abs(row['simpson_residual']), abs(row['endpoint_residual']))

    def test_perturbation_signs_and_scale(self):
        model = ToyModel()
        batch = torch.zeros((8, 128), dtype=torch.int64)
        delta = {'weight': torch.tensor([0.4], dtype=torch.float32)}
        q = {'weight': torch.tensor([0.8], dtype=torch.float32)}
        original = q['weight'].clone()
        storage = q['weight'].data_ptr()
        plus = m.scalar_plane_point(model, batch, ['weight'], delta, q, 0.25, 1.0,
                                    polynomial_loss)
        minus = m.scalar_plane_point(model, batch, ['weight'], delta, q, -0.25, 1.0,
                                     polynomial_loss)
        self.assertAlmostEqual(plus[0], analytic(1.0)[0], delta=2e-6)
        x_minus = 0.7-0.2
        expected_minus = 0.4*(x_minus**3+0.3*x_minus*x_minus+0.2*x_minus)
        self.assertAlmostEqual(minus[0], expected_minus, delta=2e-6)
        torch.testing.assert_close(q['weight'], original, rtol=0, atol=0)
        self.assertEqual(q['weight'].data_ptr(), storage)
        with self.assertRaises(ValueError):
            m.scalar_plane_point(model, batch, ['weight'], delta, q, 0.5, 0.0,
                                 polynomial_loss)
        self.assertEqual([(row['label'], row['scale']) for row in m.FROZEN_SPEC['perturbations']],
                         [('plus_q1', 0.25), ('minus_q1', -0.25),
                          ('plus_q2', 0.25), ('minus_q2', -0.25)])

    def test_q1_q2_loader_preserves_values_one_at_a_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            vectors = ([1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0])
            for i, vector in enumerate(vectors, 1):
                save_file({'weight': torch.tensor(vector, dtype=torch.float32)},
                          directory/f'q{i}.safetensors')
            coordinates = [{'name': 'weight', 'shape': [2]}]
            with m.prior022.BasisStore(directory, coordinates) as store:
                q1 = m.load_basis_direction(store, coordinates, 0, torch.device('cpu'))
                torch.testing.assert_close(q1['weight'], torch.tensor(vectors[0]))
                del q1
                q2 = m.load_basis_direction(store, coordinates, 1, torch.device('cpu'))
                torch.testing.assert_close(q2['weight'], torch.tensor(vectors[1]))
                with self.assertRaises(ValueError):
                    m.load_basis_direction(store, coordinates, 2, torch.device('cpu'))

    def test_diagnostics_beta_and_quadrature_ratios(self):
        rows = []
        for j in range(16):
            for identifier in range(4):
                x = float(j+1+identifier)
                row = {key: 0.0 for key in m.SCALAR_KEYS}
                row.update(microbatch_index=j, perturbation_id=identifier,
                           endpoint_residual=x, historical_rhs=x,
                           trapezoid_residual=x/2, simpson_residual=x/4)
                rows.append(row)
        report = m.summaries(rows)
        self.assertEqual(report['pooled']['centered_ols_beta'], 1.0)
        self.assertEqual(report['pooled']['centered_r_squared'], 1.0)
        self.assertEqual(report['pooled']['cosine'], 1.0)
        self.assertEqual(report['pooled']['best_rescaled_relative_residual'], 0.0)
        self.assertEqual(report['pooled']['trapezoid_to_endpoint_rms_ratio'], 0.5)
        self.assertEqual(report['pooled']['simpson_to_endpoint_rms_ratio'], 0.25)
        self.assertTrue(report['pooled']['descriptive_only'])
        self.assertFalse(report['pooled']['independent_batch_samples'])
        self.assertEqual(report['pooled']['sample_structure'],
                         '16 batches x 4 related perturbations')
        self.assertEqual(set(report['per_perturbation']),
                         {'plus_q1', 'minus_q1', 'plus_q2', 'minus_q2'})

    def test_record_order_is_batch_then_frozen_perturbation_id(self):
        shuffled = [{'microbatch_index': j, 'perturbation_id': identifier}
                    for identifier in (3, 2, 1, 0) for j in (1, 0)]
        self.assertEqual([(row['microbatch_index'], row['perturbation_id'])
                          for row in m.ordered_records(shuffled)],
                         [(j, identifier) for j in (0, 1) for identifier in range(4)])

    def test_nonfinite_scalars_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            m.compose_record(0, 0, (0.0, float('nan'), 1.0),
                             (0.0, 0.0, 1.0), (1.0, 1.0, 1.0))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'construction-manifest.json'
            m.require_absent(path)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.require_absent(path)

    def test_path_lock_prevents_unfrozen_base_access(self):
        args = m.parse_args(['--base-directory', '/tmp/not-canonical-historical-base'])
        with patch.object(m.a, 'checkpoint_file_records') as inventory:
            with self.assertRaisesRegex(ValueError, 'path differs'):
                m.validate_inputs(args)
            inventory.assert_not_called()

    def test_smoke_writes_no_scientific_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'construction-manifest.json'
            args = m.parse_args(['--smoke-only', '--device', 'cpu',
                                 '--manifest-path', str(output)])
            model = ToyModel()
            context = {'spec': m.FROZEN_SPEC,
                       'tokens': torch.zeros((8, 128), dtype=torch.int64),
                       'model': model, 'names': ['weight'],
                       'delta': {'weight': torch.tensor([0.4])},
                       'model_before': {}, 'execution_before': {},
                       'recheck': lambda: None,
                       'gram': {'max_off_diagonal': 0.0}}
            calls = []
            residency = {'active': 0, 'maximum': 0, 'requested': []}
            class TrackedDirection(dict):
                def __init__(self, index):
                    super().__init__({'weight': torch.tensor([1.0])})
                    self.index = index
                    residency['active'] += 1
                    residency['maximum'] = max(residency['maximum'], residency['active'])
                def __del__(self):
                    residency['active'] -= 1
            class FakeBasisStore:
                def __init__(self, directory, coordinates):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
            def fake_load(store, coordinates, index, device):
                self.assertEqual(residency['active'], 0)
                residency['requested'].append(index)
                return TrackedDirection(index)
            def fake_point(model, batch, names, delta, basis, scale, t):
                calls.append((basis.index, scale, t))
                return 1+t, 2+t, 3+t
            out = io.StringIO()
            with patch.object(m, 'prepare_context', return_value=context), \
                 patch.object(m.prior022, 'BasisStore', FakeBasisStore), \
                 patch.object(m, 'load_basis_direction', new=fake_load), \
                 patch.object(m, 'scalar_plane_point', new=fake_point), \
                 patch.object(m.support, 'verify_state'), \
                 contextlib.redirect_stdout(out):
                m.diagnose(args)
            record = json.loads(out.getvalue())
            self.assertTrue(record['smoke_only'])
            self.assertEqual(len(record['perturbations']), 4)
            self.assertEqual([row['perturbation_id'] for row in record['perturbations']],
                             [0, 1, 2, 3])
            self.assertEqual(calls, [(index, scale, t)
                for index, scale in ((0, 0.25), (0, -0.25), (1, 0.25), (1, -0.25))
                for t in (0.0, 0.5, 1.0)])
            self.assertEqual(residency['requested'], [0, 1])
            self.assertEqual(residency['maximum'], 1)
            self.assertEqual(residency['active'], 0)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
