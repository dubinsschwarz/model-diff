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
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load(ROOT/'scripts/ablation/construct_blind_centered_gradient_pca4_pilot.py', 'attempt023')
e = load(ROOT/'scripts/ablation/evaluate_blind_centered_gradient_pca4_pilot.py', 'attempt023_eval')


def toy_snapshots(scales=(5., 4., 3., 2., 1., .5)):
    generator = torch.Generator().manual_seed(23)
    random = torch.randn((16, 6), generator=generator, dtype=torch.float64)
    centered = random-random.mean(dim=0)
    columns, _ = torch.linalg.qr(centered)
    values = columns*torch.tensor(scales, dtype=torch.float64)
    values += torch.tensor([.4, -.5, .6, .7, -.8, .9], dtype=torch.float64)
    values = values.float().double()
    snapshots = [{'v': values[i].float().reshape(6, 1).contiguous()} for i in range(16)]
    return snapshots, [{'name': 'v', 'shape': [6, 1]}], values


def regression_records():
    matrices = []
    for i in range(16):
        x = float(i+1)
        matrices.append(torch.diag(torch.tensor(
            [math.sin(x), math.cos(x), math.sin(2*x), math.cos(2*x)], dtype=torch.float64)))
    M = torch.stack(matrices)
    coefficient = torch.tensor([.4, -.3, .7, 1.2], dtype=torch.float64)
    intercept = torch.tensor([2., -1., .5, 3.], dtype=torch.float64)
    return intercept+M@coefficient, M, coefficient


class FourLogitToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 4, bias=False)
        self.proj.weight.data.copy_(torch.tensor([[.1], [-.2], [.3], [.4]]))
        self.unselected = torch.nn.Parameter(torch.tensor(.07))
        self.seen_dtypes = []

    def forward(self, *, input_ids, use_cache):
        assert use_cache is False
        self.seen_dtypes.append(self.proj.weight.dtype)
        logits = self.proj(input_ids.float().unsqueeze(-1)/4)+self.unselected
        return type('Output', (), {'logits': logits})()


def toy_basis():
    return [{'proj.weight': torch.eye(4, dtype=torch.float32)[i].reshape(4, 1).contiguous()}
            for i in range(4)]


def toy_tokens():
    return (torch.arange(8*128, dtype=torch.int64).reshape(8, 128)%4).contiguous()


class Toy98(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False)
                                           for _ in range(98)])

    def forward(self, *, input_ids, use_cache):
        assert use_cache is False
        x = input_ids.float().unsqueeze(-1)/4
        z = sum(layer(x) for layer in self.layers)
        return type('Output', (), {'logits': torch.cat([z, -z, z*.5, -z*.5], dim=-1)})()


class PcaTests(unittest.TestCase):
    def test_snapshot_gram_centering_and_reconstructed_orthonormal_basis(self):
        snapshots, coordinates, values = toy_snapshots()
        G = c.snapshot_gram(snapshots, coordinates)
        torch.testing.assert_close(G, values@values.T, rtol=1e-12, atol=1e-12)
        K, eigenvalues, eigenvectors, record = c.centered_pca(G, c.FROZEN_SPEC['pca'])
        centered = values-values.mean(dim=0)
        torch.testing.assert_close(K, centered@centered.T, rtol=1e-12, atol=1e-12)
        self.assertTrue(bool((eigenvalues[:-1] > eigenvalues[1:]).all()))
        for j in range(4):
            self.assertGreater(float(eigenvectors[torch.argmax(eigenvectors[:, j].abs()), j]), 0)
        self.assertGreater(record['top4_energy_fraction'], 0)
        self.assertLessEqual(record['top4_energy_fraction'], 1+1e-10)
        basis = c.reconstruct_pca_basis(snapshots, coordinates, eigenvalues, eigenvectors)
        audit = c.basis_gram(basis, coordinates, c.FROZEN_SPEC['pca'])
        self.assertLess(audit['max_off_diagonal'], 1e-5)
        self.assertLess(audit['max_diagonal_deviation'], 1e-5)

    def test_degenerate_retained_eigenvalues_fail_closed(self):
        snapshots, coordinates, _ = toy_snapshots(scales=(5., 4., 3., 3., 1., .5))
        G = c.snapshot_gram(snapshots, coordinates)
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            c.centered_pca(G, c.FROZEN_SPEC['pca'])
        snapshots, coordinates, _ = toy_snapshots(scales=(5., 4., 3., 2., 2., .5))
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            c.centered_pca(c.snapshot_gram(snapshots, coordinates), c.FROZEN_SPEC['pca'])

    def test_snapshot_scalar_projection_and_ceiling_equal_direct_basis(self):
        snapshots, coordinates, values = toy_snapshots()
        G = c.snapshot_gram(snapshots, coordinates)
        _, eigenvalues, eigenvectors, _ = c.centered_pca(G, c.FROZEN_SPEC['pca'])
        basis = c.reconstruct_pca_basis(snapshots, coordinates, eigenvalues, eigenvectors)
        delta = torch.tensor([.1, -.2, .3, -.4, .5, -.6], dtype=torch.float64)
        s = values@delta
        c_true, centered = e.project_from_snapshot_scalars(s, eigenvalues, eigenvectors)
        C = torch.eye(16, dtype=torch.float64)-torch.ones((16, 16), dtype=torch.float64)/16
        torch.testing.assert_close(centered, C@s, rtol=0, atol=0)
        direct = torch.tensor([float(torch.dot(basis[j]['v'].flatten().double(), delta))
                               for j in range(4)], dtype=torch.float64)
        torch.testing.assert_close(c_true, direct, rtol=1e-5, atol=1e-6)
        before_values, before_vectors = eigenvalues.clone(), eigenvectors.clone()
        candidate = c_true.clone()
        before_candidate = candidate.clone()
        metrics = e.projection_metrics(candidate, c_true, float(torch.linalg.vector_norm(delta)))
        self.assertAlmostEqual(metrics['projection_ceiling'],
                               float(torch.linalg.vector_norm(direct)/torch.linalg.vector_norm(delta)),
                               places=5)
        self.assertAlmostEqual(metrics['fraction_of_subspace_ceiling'], 1., places=12)
        self.assertTrue(torch.equal(eigenvalues, before_values))
        self.assertTrue(torch.equal(eigenvectors, before_vectors))
        self.assertTrue(torch.equal(candidate, before_candidate))

    def test_fixed_disjoint_batches_and_loss_denominator(self):
        batches = c.selected_batches()
        self.assertEqual([j for j, _, _ in batches['basis']], list(range(0, 512, 32)))
        self.assertEqual([j for j, _, _ in batches['regression']], list(range(16, 512, 32)))
        self.assertFalse({j for j, _, _ in batches['basis']} &
                         {j for j, _, _ in batches['regression']})
        self.assertEqual(c.FROZEN_SPEC['batches']['prediction_tokens_per_batch'], 1016)
        self.assertEqual(c.FROZEN_SPEC['smoke']['basis_microbatch_indices'], [0, 32])
        self.assertEqual(len(c.COORDINATES), 98)


class DerivativeAndRegressionTests(unittest.TestCase):
    def test_selected_snapshot_gradient_and_no_grad_buffers(self):
        model = Toy98()
        names = [f'layers.{i}.weight' for i in range(98)]
        coordinates = [{'name': name, 'shape': [1, 1]} for name in names]
        c.freeze_for_snapshots(model, names)
        snapshot = c.gradient_snapshot(model, toy_tokens(), coordinates)
        self.assertEqual(set(snapshot), set(names))
        self.assertTrue(all(value.dtype == torch.float32 and value.is_contiguous()
                            for value in snapshot.values()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        batch = toy_tokens()
        logits = model(input_ids=batch, use_cache=False).logits
        expected_loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, 4), batch[:, 1:].reshape(-1), reduction='sum')/1016
        expected_first = torch.autograd.grad(expected_loss, model.layers[0].weight)[0]
        torch.testing.assert_close(snapshot['layers.0.weight'], expected_first, rtol=0, atol=0)

    def test_t_space_gradient_hessian_matches_explicit_dense_derivatives(self):
        model = FourLogitToy()
        c.freeze_for_t_space(model)
        batch = toy_tokens()
        state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        b, M, diagnostic = c.t_space_batch(model, batch, ['proj.weight'], toy_basis(),
                                            c.FROZEN_SPEC['t_space'])
        self.assertLessEqual(diagnostic['relative'], 1e-5)
        self.assertTrue(all(dtype == torch.float32 for dtype in model.seen_dtypes))
        weight = model.proj.weight.detach().clone().requires_grad_(True)
        def loss(w):
            logits = torch.func.functional_call(model, {'proj.weight': w}, (),
                {'input_ids': batch, 'use_cache': False}).logits
            return c.a.causal_token_loss_sum(logits, batch, torch)/1016
        gradient = torch.autograd.grad(loss(weight), weight)[0].flatten().double()
        dense = torch.autograd.functional.hessian(loss, weight).reshape(4, 4).double()
        torch.testing.assert_close(b, gradient, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(M, dense, rtol=1e-5, atol=1e-6)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_scale_aware_hessian_symmetry_audit(self):
        policy = c.FROZEN_SPEC['t_space']
        raw = torch.diag(torch.tensor([2425.3762, 2., 3., 4.], dtype=torch.float64))
        raw[0, 1], raw[1, 0] = 3., 3.0081634521484375
        sym, record = c.audit_hessian_symmetry(raw, policy)
        self.assertGreater(record['absolute'], 1e-3)
        self.assertLess(record['relative'], 1e-5)
        torch.testing.assert_close(sym, (raw+raw.T)/2, rtol=0, atol=0)
        small = torch.eye(4, dtype=torch.float64)
        small[0, 1], small[1, 0] = .2, .20002
        with self.assertRaisesRegex(ValueError, 'asymmetry'):
            c.audit_hessian_symmetry(small, policy)
        small[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid raw'):
            c.audit_hessian_symmetry(small, policy)

    def test_held_out_centered_regression_split_and_rank_failure(self):
        b, M, expected = regression_records()
        recovered, fit = c.solve_centered(b, M, c.FROZEN_SPEC['regression'])
        torch.testing.assert_close(recovered, expected, rtol=1e-12, atol=1e-12)
        shifted, _ = c.solve_centered(b+torch.tensor([10., -2., 3., 5.]), M,
                                       c.FROZEN_SPEC['regression'])
        torch.testing.assert_close(shifted, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(fit['rank'], 4)
        self.assertEqual(len(fit['batch_residual_norms']), 16)
        stability = c.split_stability(b, M, c.FROZEN_SPEC['regression'])
        self.assertEqual(set(stability), {'first_8_vs_last_8', 'even_members_vs_odd_members'})
        for record in stability.values():
            self.assertAlmostEqual(record['coefficient_cosine'], 1., places=12)
            self.assertAlmostEqual(record['coefficient_norm_ratio'], 1., places=12)
        with self.assertRaisesRegex(ValueError, 'rank deficient'):
            c.solve_centered(b, torch.ones_like(M), c.FROZEN_SPEC['regression'])

    def test_scalar_directional_derivative_matches_explicit_gradient_dot(self):
        model = FourLogitToy()
        c.freeze_for_t_space(model)
        batch = toy_tokens()
        direction = torch.tensor([[.123456789], [-.234567891], [.345678912], [-.456789123]],
                                 dtype=torch.float64)
        scalar = e.scalar_directional_derivative(model, batch, ['proj.weight'],
                                                  {'proj.weight': direction})
        weight = model.proj.weight.detach().clone().requires_grad_(True)
        logits = torch.func.functional_call(model, {'proj.weight': weight}, (),
            {'input_ids': batch, 'use_cache': False}).logits
        loss = c.a.causal_token_loss_sum(logits, batch, torch)/1016
        gradient = torch.autograd.grad(loss, weight)[0]
        expected = float(torch.sum(gradient.double()*direction))
        self.assertAlmostEqual(scalar, expected, places=6)


class PipelineTests(unittest.TestCase):
    def test_specs_firewall_artifact_hash_and_stale_output(self):
        root = ROOT/'experiments/attempts'/c.FROZEN_SPEC['attempt_id']
        self.assertEqual(json.loads((root/'spec.json').read_text()), c.FROZEN_SPEC)
        self.assertEqual(json.loads((root/'evaluation_spec.json').read_text()), e.EVALUATION_SPEC)
        snapshots, coordinates, _ = toy_snapshots()
        G = c.snapshot_gram(snapshots, coordinates)
        K, eigenvalues, eigenvectors, _ = c.centered_pca(G, c.FROZEN_SPEC['pca'])
        b, M, _ = regression_records()
        c_hat, _ = c.solve_centered(b, M, c.FROZEN_SPEC['regression'])
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory)/'stats.pt'
            record = c.save_artifact(artifact_path, G, K, eigenvalues, eigenvectors, b, M, c_hat)
            artifact, verified = c.verify_artifact(artifact_path, record)
            self.assertEqual(verified, record)
            self.assertEqual(artifact['regression_b'].shape, (16, 4))
            corrupted = copy.deepcopy(record)
            corrupted['raw_tensor_sha256']['c_hat'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'hash/inventory'):
                c.verify_artifact(artifact_path, corrupted)
            with self.assertRaisesRegex(ValueError, 'exists'):
                c.save_artifact(artifact_path, G, K, eigenvalues, eigenvectors, b, M, c_hat)
            args = c.parse_args(['--artifact-path', str(artifact_path),
                                 '--manifest-path', str(Path(directory)/'manifest.json')])
            with self.assertRaisesRegex(ValueError, 'exists'):
                c.require_absent(args)
            for marker in ('models/base', 'oracle_adl.pt', 'adapter.pt', 'true_delta.pt',
                           'attempt022/evaluation.json', '016_known_base.pt'):
                args.tokens_path = Path(directory)/marker
                with self.assertRaisesRegex(ValueError, 'information firewall'):
                    c.firewall(args)

    def test_bf16_historical_loading_and_positive_delta(self):
        model = FourLogitToy()
        raw = torch.tensor([[.101], [-.203], [.304], [.407]], dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.safetensors'
            save_file({'proj.weight': raw}, str(path))
            with e.selected_base_tensors(Path(directory), [{'path': path.name}]) as base:
                delta, norm = e.exact_early_delta(model,
                    [{'name': 'proj.weight', 'shape': [4, 1]}], base)
        expected = model.proj.weight.detach().double()-raw.float().double()
        torch.testing.assert_close(delta['proj.weight'], expected, rtol=0, atol=0)
        self.assertEqual(norm, float(torch.linalg.vector_norm(expected)))

    def test_synthetic_smoke_writes_no_scientific_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = c.parse_args(['--smoke-only', '--device', 'cpu',
                                 '--artifact-path', str(root/'stats.pt'),
                                 '--manifest-path', str(root/'manifest.json')])
            model = Toy98()
            coordinates = [{'name': f'layers.{i}.weight', 'shape': [1, 1]} for i in range(98)]
            spec = copy.deepcopy(c.FROZEN_SPEC)
            spec['coordinates'] = coordinates
            tokens = (torch.arange(4096*128, dtype=torch.int64).reshape(4096, 128)%4).contiguous()
            eligible = [(f'layers.{i}', layer) for i, layer in enumerate(model.layers)]
            with (patch.object(c, 'validate_inputs', return_value=(spec, tokens, {}, lambda: None)),
                  patch.object(c.a, 'load_local_model', return_value=model),
                  patch.object(c.a, 'discover_eligible_linear_weights', return_value=eligible),
                  redirect_stdout(io.StringIO()) as output):
                c.construct(args)
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.manifest_path.exists())
            result = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(result['basis_microbatch_indices'], [0, 32])
            self.assertEqual(len(result['gram']), 2)
            self.assertFalse(result['pca_computed'])
            self.assertFalse(result['regression_computed'])


if __name__ == '__main__':
    unittest.main()
