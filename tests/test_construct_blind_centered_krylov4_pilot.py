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


c = load(ROOT/'scripts/ablation/construct_blind_centered_krylov4_pilot.py', 'attempt022')
e = load(ROOT/'scripts/ablation/evaluate_blind_centered_krylov4_pilot.py', 'attempt022_eval')


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
        return type('Output', (), {'logits': self.proj(input_ids.float().unsqueeze(-1)/4)+self.unselected})()


def basis():
    return [{'proj.weight': torch.eye(4, dtype=torch.float32)[i].reshape(4, 1).contiguous()} for i in range(4)]


def batches():
    return (torch.arange(8*128, dtype=torch.int64).reshape(8, 128) % 4).contiguous()


def synthetic_records():
    M = []
    for i in range(32):
        x = float(i+1)
        features = [math.sin(x), math.cos(x), math.sin(2*x), math.cos(2*x)]
        M.append(torch.diag(torch.tensor(features, dtype=torch.float64)))
    M = torch.stack(M)
    coefficient = torch.tensor([.4, -.3, .7, 1.2], dtype=torch.float64)
    mu = torch.tensor([2., -1., .5, 3.], dtype=torch.float64)
    b = mu+M@coefficient
    return b, M, coefficient, mu


class MathematicsTests(unittest.TestCase):
    def test_t_space_gradient_hessian_and_symmetry(self):
        model = FourLogitToy()
        c.freeze_model(model)
        tokens = batches()
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        calls = []
        original = torch.autograd.grad
        def spy(outputs, inputs, **kw):
            leaf = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
            calls.append((tuple(leaf.shape), leaf.dtype))
            return original(outputs, inputs, **kw)
        with patch.object(torch.autograd, 'grad', side_effect=spy):
            b, M, asym = c.t_space_batch(model, tokens, ['proj.weight'], basis(), c.FROZEN_SPEC['t_space'])
        self.assertEqual(calls, [((4,), torch.float64)]*5)
        self.assertLessEqual(asym['absolute'], 1e-5)
        self.assertLessEqual(asym['relative'], c.FROZEN_SPEC['t_space']['raw_hessian_relative_asymmetry_limit'])
        torch.testing.assert_close(M, M.T, rtol=0, atol=0)
        weight = model.proj.weight.detach().clone().requires_grad_(True)
        def loss(w):
            logits = torch.func.functional_call(model, {'proj.weight': w}, (),
                {'input_ids': tokens, 'use_cache': False}).logits
            return c.a.causal_token_loss_sum(logits, tokens, torch)/1016
        gradient = original(loss(weight), weight)[0].reshape(4).double()
        dense = torch.autograd.functional.hessian(loss, weight).reshape(4, 4).double()
        torch.testing.assert_close(b, gradient, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(M, dense, rtol=1e-5, atol=1e-6)
        self.assertTrue(all(dtype == torch.float32 for dtype in model.seen_dtypes))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_scale_aware_hessian_asymmetry_audit(self):
        policy = c.FROZEN_SPEC['t_space']
        self.assertNotIn('raw_hessian_absolute_asymmetry_limit', policy)
        large = torch.diag(torch.tensor([2425.3762, 2., 3., 4.], dtype=torch.float64))
        large[0, 1] = 3.
        large[1, 0] = 3.0081634521484375
        symmetrized, diagnostic = c.audit_hessian_symmetry(large, policy)
        self.assertGreater(diagnostic['absolute'], 1e-3)
        self.assertLess(diagnostic['relative'], 1e-5)
        self.assertAlmostEqual(diagnostic['relative'], diagnostic['absolute']/2425.3762)
        torch.testing.assert_close(symmetrized, (large+large.T)/2, rtol=0, atol=0)

        small = torch.eye(4, dtype=torch.float64)
        small[0, 1] = 0.2
        small[1, 0] = 0.20002
        with self.assertRaisesRegex(ValueError, 'asymmetry exceeds'):
            c.audit_hessian_symmetry(small, policy)
        small[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid raw'):
            c.audit_hessian_symmetry(small, policy)

    def test_centering_exact_recovery_intercept_and_correlated_bias(self):
        b, M, coefficient, mu = synthetic_records()
        recovered, record = c.solve_centered(b, M, c.FROZEN_SPEC['regression'])
        torch.testing.assert_close(recovered, coefficient, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(record['bbar'], (mu+M.mean(0)@coefficient).tolist())
        shifted, _ = c.solve_centered(b+torch.tensor([90., -5., 3., 11.]), M, c.FROZEN_SPEC['regression'])
        torch.testing.assert_close(shifted, coefficient, rtol=1e-12, atol=1e-12)
        bias = torch.tensor([.1, -.2, .3, -.4], dtype=torch.float64)
        correlated, _ = c.solve_centered(b+M@bias, M, c.FROZEN_SPEC['regression'])
        torch.testing.assert_close(correlated, coefficient+bias, rtol=1e-12, atol=1e-12)
        self.assertEqual(record['rank'], 4)
        self.assertLess(record['relative_residual'], 1e-12)
        with self.assertRaisesRegex(ValueError, 'rank deficient'):
            c.solve_centered(b, torch.ones_like(M), c.FROZEN_SPEC['regression'])

    def test_split_stability_and_fixed_batch_selection(self):
        b, M, _, _ = synthetic_records()
        stability = c.split_stability(b, M, c.FROZEN_SPEC['regression'])
        self.assertEqual(set(stability), {'first_16_vs_last_16', 'even_members_vs_odd_members'})
        for record in stability.values():
            self.assertAlmostEqual(record['coefficient_cosine'], 1., places=12)
            self.assertAlmostEqual(record['coefficient_norm_ratio'], 1., places=12)
        self.assertEqual(c.selected_batches(), [(j, 8*j, 8*(j+1)) for j in range(0, 512, 16)])
        self.assertEqual(c.selected_batches()[-1], (496, 3968, 3976))
        self.assertEqual(c.FROZEN_SPEC['batches']['prediction_tokens_per_batch'], 8*127)

    def test_basis_hash_inventory_and_orthogonality(self):
        coordinates = [{'name': 'proj.weight', 'shape': [4, 1]}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            for i, vector in enumerate(basis(), 1):
                path = root/f'q{i}.safetensors'
                save_file(vector, str(path))
                records.append({'direction': i, 'filename': path.name, 'dtype': 'float32',
                                'size_bytes': path.stat().st_size, 'sha256': c.a.sha256_file(path)})
            with patch.dict(c.FROZEN_SPEC['basis'], {'files': records}), patch.object(c, 'BASIS_HASHES', tuple(row['sha256'] for row in records)):
                c.verify_basis_files(root, records)
                with c.BasisStore(root, coordinates) as store:
                    settings = {'reference_gram': torch.eye(4).tolist(), 'reference_gram_max_error': 1e-8,
                                'orthogonality': c.FROZEN_SPEC['basis']['orthogonality']}
                    result = c.basis_gram(store, coordinates, settings)
                    self.assertEqual(result['max_off_diagonal'], 0.)
                    self.assertEqual(result['max_diagonal_deviation'], 0.)
                    bad = copy.deepcopy(settings)
                    bad['reference_gram'][0][0] = 1.1
                    with self.assertRaisesRegex(ValueError, 'Gram audit'):
                        c.basis_gram(store, coordinates, bad)
                (root/'q1.safetensors').write_bytes(b'bad')
                with self.assertRaisesRegex(ValueError, 'hash/inventory'):
                    c.verify_basis_files(root, records)

    def test_artifact_inventory_stale_and_firewall(self):
        spec_path = ROOT/'experiments/attempts'/c.FROZEN_SPEC['attempt_id']/'spec.json'
        self.assertEqual(json.loads(spec_path.read_text()), c.FROZEN_SPEC)
        b, M, _, _ = synthetic_records()
        coefficient, _ = c.solve_centered(b, M, c.FROZEN_SPEC['regression'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'batch_stats.pt'
            record = c.save_artifact(path, b, M, coefficient)
            loaded, verified = c.verify_artifact(path, record)
            self.assertEqual(record, verified)
            self.assertTrue(torch.equal(loaded['c_hat'], coefficient))
            with self.assertRaisesRegex(ValueError, 'exists'):
                c.save_artifact(path, b, M, coefficient)
            args = c.parse_args(['--artifact-path', str(path), '--manifest-path', str(root/'manifest.json')])
            with self.assertRaisesRegex(ValueError, 'exists'):
                c.require_absent(args)
            args.artifact_path.unlink()
            for marker in ('models/base', 'oracle_adl.pt', 'adapter.pt', 'attempt020_result',
                           '018_known_cleaned/evaluation.json', 'true_delta.pt'):
                args.tokens_path = root/marker
                with self.assertRaisesRegex(ValueError, 'information firewall'):
                    c.firewall(args)
        source = Path(c.__file__).read_text()
        self.assertNotIn('construct_true_delta', source)
        self.assertNotIn('hessian_vector_product', source)

    def test_evaluator_projection_ceiling_and_immutable_coefficients(self):
        path = ROOT/'experiments/attempts'/c.FROZEN_SPEC['attempt_id']/'evaluation_spec.json'
        self.assertEqual(json.loads(path.read_text()), e.EVALUATION_SPEC)
        coefficient = torch.tensor([6., 8., 0., 0.], dtype=torch.float64)
        true = torch.tensor([3., 4., 0., 0.], dtype=torch.float64)
        before = coefficient.clone()
        result = e.projection_metrics(coefficient, true, 10.)
        self.assertEqual(result['projection_ceiling'], .5)
        self.assertEqual(result['cosine_coefficients'], 1.)
        self.assertEqual(result['cosine_candidate_to_full_DeltaS'], .5)
        self.assertEqual(result['fraction_of_subspace_ceiling'], 1.)
        self.assertEqual(result['optimal_scalar'], .5)
        self.assertEqual(result['relative_residual_after_optimal_rescaling'], 0.)
        self.assertTrue(torch.equal(coefficient, before))
        self.assertFalse(e.EVALUATION_SPEC['functional_evaluation'])

    def test_evaluator_true_coefficients_use_positive_final_minus_base(self):
        final = FourLogitToy()
        displacement = torch.tensor([[.1], [-.2], [.3], [-.4]], dtype=torch.float32)
        old = final.proj.weight.detach().clone()-displacement
        class BaseFile:
            def get_tensor(self, name):
                self_name = name
                assert self_name == 'proj.weight'
                return old
        class Store:
            def matrix(self, i, name, shape):
                assert name == 'proj.weight' and shape == [4, 1]
                return basis()[i][name]
        true, norm = e.true_coefficients(final, Store(),
            [{'name': 'proj.weight', 'shape': [4, 1]}], {'proj.weight': BaseFile()})
        torch.testing.assert_close(true, displacement.flatten().double(), rtol=1e-6, atol=1e-8)
        self.assertAlmostEqual(norm, float(torch.linalg.vector_norm(displacement.double())))


class PipelineTests(unittest.TestCase):
    def test_synthetic_smoke_writes_no_scientific_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = c.parse_args(['--smoke-only', '--device', 'cpu',
                                 '--artifact-path', str(root/'batch_stats.pt'),
                                 '--manifest-path', str(root/'manifest.json')])
            spec = copy.deepcopy(c.FROZEN_SPEC)
            spec['coordinates'] = [{'name': 'proj.weight', 'shape': [4, 1]}]
            tokens = (torch.arange(4096*128, dtype=torch.int64).reshape(4096, 128)%4).contiguous()
            gram = {'gram_matrix': torch.eye(4).tolist(), 'max_off_diagonal': 0.,
                    'max_diagonal_deviation': 0., 'max_reference_difference': 0.}
            model = FourLogitToy()
            eligible = [('proj', model.proj)]
            class DummyStore:
                def __init__(self, *_): pass
                def __enter__(self): return self
                def __exit__(self, *_): pass
            with (patch.object(c, 'validate_inputs', return_value=(spec, tokens, {}, {'basis_files': []}, gram, lambda: None)),
                  patch.object(c.a, 'load_local_model', return_value=model),
                  patch.object(c.a, 'discover_eligible_linear_weights', return_value=eligible),
                  patch.object(c, 'BasisStore', DummyStore),
                  patch.object(c, 'basis_on_device', return_value=basis()),
                  redirect_stdout(io.StringIO()) as output):
                c.construct(args)
            self.assertFalse(args.artifact_path.exists())
            self.assertFalse(args.manifest_path.exists())
            printed = json.loads(output.getvalue().splitlines()[-1])
            self.assertEqual(printed['microbatch_index'], 0)
            self.assertEqual(len(printed['b_0']), 4)
            self.assertEqual(len(printed['M_0']), 4)
            self.assertTrue(math.isfinite(printed['seconds']))


if __name__ == '__main__':
    unittest.main()
