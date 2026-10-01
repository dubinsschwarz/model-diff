import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'scripts/ablation/evaluate_blind_centered_gradient_pca4_functional.py'
loader = importlib.util.spec_from_file_location('attempt023_functional_test_target', SOURCE)
f = importlib.util.module_from_spec(loader)
loader.loader.exec_module(f)


def synthetic_pca_artifact():
    generator = torch.Generator().manual_seed(23)
    random = torch.randn((16, 6), generator=generator, dtype=torch.float64)
    columns, _ = torch.linalg.qr(random-random.mean(dim=0))
    gradients = (columns*torch.tensor([5., 4., 3., 2., 1., .5])).float().double()
    G = gradients@gradients.T
    K, eigenvalues, eigenvectors, _ = f.c.centered_pca(G, f.c.FROZEN_SPEC['pca'])
    M = []
    for i in range(16):
        x = float(i+1)
        M.append(torch.diag(torch.tensor([math.sin(x), math.cos(x),
                                          math.sin(2*x), math.cos(2*x)], dtype=torch.float64)))
    M = torch.stack(M)
    expected = torch.tensor([.4, -.3, .7, 1.2], dtype=torch.float64)
    b = torch.tensor([2., -1., .5, 3.])+M@expected
    c_hat, _ = f.c.solve_centered(b, M, f.c.FROZEN_SPEC['regression'])
    return ({'raw_gradient_gram_G': G, 'centered_gradient_gram_K': K,
             'retained_eigenvalues': eigenvalues,
             'retained_eigenvectors': eigenvectors,
             'regression_b': b, 'regression_M': M, 'c_hat': c_hat},
            {'regression': {'coefficients': c_hat.tolist()}})


def tiny_basis():
    names = [{'name': 'weight', 'shape': [4, 1]}]
    vectors = [dict(weight=torch.eye(4, dtype=torch.float32)[j].reshape(4, 1).contiguous())
               for j in range(4)]
    return vectors, names


class CandidateTests(unittest.TestCase):
    def test_candidate_exact_fp32_arithmetic_without_normalization_or_sign_flip(self):
        basis, coordinates = tiny_basis()
        c_hat = torch.tensor([-2.25, .5, 3.5, -4.75], dtype=torch.float64)
        before = c_hat.clone()
        basis_before = [{name: value.clone() for name, value in row.items()} for row in basis]
        candidate, norm = f.form_candidate(basis, c_hat, coordinates)
        expected = torch.zeros((4, 1), dtype=torch.float32)
        for j in range(4):
            expected = expected+c_hat[j].float()*basis[j]['weight']
        torch.testing.assert_close(candidate['weight'], expected, rtol=0, atol=0)
        self.assertLess(float(candidate['weight'][0]), 0)
        self.assertEqual(norm, float(torch.linalg.vector_norm(expected.double())))
        self.assertNotAlmostEqual(norm, 1.)
        self.assertTrue(torch.equal(c_hat, before))
        for j in range(4):
            self.assertTrue(torch.equal(basis[j]['weight'], basis_before[j]['weight']))

    def test_candidate_rejects_malformed_basis_and_nonfinite_coefficient(self):
        basis, coordinates = tiny_basis()
        coefficients = torch.ones(4, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'inventory'):
            f.form_candidate(basis[:-1]+[{}], coefficients, coordinates)
        coefficients[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid frozen'):
            f.form_candidate(basis, coefficients, coordinates)

    def test_frozen_pca_and_coefficients_reproduce_without_mutation(self):
        artifact, manifest = synthetic_pca_artifact()
        before = {key: value.clone() for key, value in artifact.items()}
        eigenvalues, eigenvectors, c_hat = f.reproduce_frozen_solution(
            artifact, manifest, f.FROZEN_SPEC)
        for key, value in before.items():
            self.assertTrue(torch.equal(artifact[key], value))
        self.assertTrue(torch.equal(eigenvalues, artifact['retained_eigenvalues']))
        self.assertTrue(torch.equal(eigenvectors, artifact['retained_eigenvectors']))
        self.assertTrue(torch.equal(c_hat, artifact['c_hat']))
        bad = copy.deepcopy(artifact)
        bad['retained_eigenvectors'][0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'PCA eigenpairs'):
            f.reproduce_frozen_solution(bad, manifest, f.FROZEN_SPEC)
        bad = copy.deepcopy(artifact)
        bad['c_hat'][0] += .01
        with self.assertRaisesRegex(ValueError, 'c_hat'):
            f.reproduce_frozen_solution(bad, manifest, f.FROZEN_SPEC)

    def test_oracle_metrics_cannot_change_candidate_or_choose_parameter_branch(self):
        basis, coordinates = tiny_basis()
        c_hat = torch.tensor([-2., 1., 3., -4.], dtype=torch.float64)
        first, _ = f.form_candidate(basis, c_hat, coordinates)
        response = torch.ones((128, 2048), dtype=torch.float32)
        spec = f.FROZEN_SPEC
        positive = f.functional_geometry(response, 2*response, spec)
        negative = f.functional_geometry(response, -2*response, spec)
        second, _ = f.form_candidate(basis, c_hat, coordinates)
        self.assertEqual(positive['primary_positions_1_4']['mean'], 1.)
        self.assertEqual(negative['primary_positions_1_4']['mean'], -1.)
        self.assertTrue(torch.equal(first['weight'], second['weight']))
        self.assertNotIn('parameter_geometry', SOURCE.read_text())

    def test_replay_reconstructs_candidate_from_all_sixteen_frozen_batches(self):
        artifact, _ = synthetic_pca_artifact()
        generator = torch.Generator().manual_seed(23)
        random = torch.randn((16, 6), generator=generator, dtype=torch.float64)
        columns, _ = torch.linalg.qr(random-random.mean(dim=0))
        gradients = (columns*torch.tensor([5., 4., 3., 2., 1., .5])).float()
        snapshots = [{'v.weight': gradients[i].reshape(6, 1).contiguous()} for i in range(16)]
        coordinates = [{'name': 'v.weight', 'shape': [6, 1]}]
        spec = copy.deepcopy(f.FROZEN_SPEC)
        spec['coordinates'] = coordinates
        basis = f.c.reconstruct_pca_basis(snapshots, coordinates,
                                          artifact['retained_eigenvalues'],
                                          artifact['retained_eigenvectors'])
        gram = f.c.basis_gram(basis, coordinates, f.c.FROZEN_SPEC['pca'])
        manifest = {'pca_basis_gram': gram}
        frozen = {name: value.clone() for name, value in artifact.items()}
        model = SimpleNamespace()
        tokens = torch.zeros((4096, 128), dtype=torch.int64)
        module = SimpleNamespace(weight=torch.empty((6, 1), dtype=torch.float32))
        with (patch.object(f.c.a, 'discover_eligible_linear_weights', return_value=[('v', module)]),
              patch.object(f.c, 'freeze_for_snapshots'),
              patch.object(f.c, 'gradient_snapshot', side_effect=snapshots) as gradient,
              patch.object(f.c, 'freeze_for_t_space')):
            candidate, audit = f.reconstruct_frozen_candidate(
                model, tokens, artifact, manifest, spec)
        self.assertEqual(gradient.call_count, 16)
        self.assertEqual(candidate['v.weight'].shape, (6, 1))
        self.assertGreater(audit['candidate_parameter_norm'], 0)
        for name, value in frozen.items():
            self.assertTrue(torch.equal(artifact[name], value))


class GeometryAndInputTests(unittest.TestCase):
    def test_position_geometry_and_fixed_summaries(self):
        response = torch.ones((128, 2048), dtype=torch.float32)
        response[0].neg_()
        oracle = 2*torch.ones_like(response)
        result = f.functional_geometry(response, oracle, f.FROZEN_SPEC)
        self.assertEqual(len(result['positions']), 128)
        self.assertEqual(result['position_0']['cosine'], -1.)
        self.assertEqual(result['position_0']['optimal_scalar'], -2.)
        self.assertEqual(result['primary_positions_1_4']['mean'], 1.)
        self.assertEqual(result['secondary_positions_1_127']['min'], 1.)
        self.assertEqual(result['positions'][1]['norm_ratio'], .5)
        self.assertEqual(result['positions'][1]['optimal_scalar'], 2.)
        self.assertEqual(result['positions'][1]['relative_residual_after_optimal_rescaling'], 0.)
        response[5].zero_()
        undefined = f.functional_geometry(response, oracle, f.FROZEN_SPEC)
        self.assertIsNone(undefined['secondary_positions_1_127']['mean'])
        self.assertEqual(undefined['primary_positions_1_4']['mean'], 1.)

    def test_spec_paths_hashes_arithmetic_and_historical_base_absence(self):
        path = ROOT/'experiments/attempts/023_blind_centered_gradient_pca4_regression_pilot/functional_evaluation_spec.json'
        self.assertEqual(json.loads(path.read_text()), f.FROZEN_SPEC)
        spec = f.FROZEN_SPEC
        self.assertEqual(spec['frozen_hashes']['stats'],
                         '17d24baead0e235a6571e261fd79721f02ca78254bbe7e5461a31680ffbbd39a')
        self.assertEqual(spec['probe']['sample_count'], 10000)
        self.assertEqual(spec['probe']['batch_size'], 32)
        self.assertEqual(spec['readout']['hidden_state_index'], 14)
        self.assertEqual(spec['jvp']['response_sign'], 'positive_J1_candidate')
        self.assertEqual(spec['functional_summaries']['primary_positions'], [1, 2, 3, 4])
        self.assertEqual(spec['functional_summaries']['secondary_positions'], list(range(1, 128)))
        self.assertFalse(spec['output_policy']['historical_base_access'])
        self.assertNotIn('/models/base', json.dumps(spec))
        self.assertNotIn('base_directory', SOURCE.read_text())

    def test_frozen_model_block_supports_reused_hidden_state_readout(self):
        spec = f.FROZEN_SPEC
        self.assertEqual(spec['model'], f.activation.FROZEN_SPEC['model'])
        self.assertEqual(spec['model']['num_hidden_layers'], 28)
        states = tuple(torch.tensor(i) for i in range(29))
        output = SimpleNamespace(hidden_states=states)
        self.assertIs(f.activation.hidden_state_output(output, spec), states[14])

    def test_construction_linkage_rejects_malformed_or_mismatched_metadata(self):
        spec = f.FROZEN_SPEC
        manifest = {
            'attempt_id': spec['attempt_id'],
            'purpose': f.c.FROZEN_SPEC['purpose'],
            'information_policy': f.c.FROZEN_SPEC['information_policy'],
            'source_checkpoint_files': spec['canonical_checkpoint_files'],
            'corpus': spec['frozen_corpus'], 'coordinates': spec['coordinates'],
            'batches': f.c.FROZEN_SPEC['batches'],
            'pca_policy': f.c.FROZEN_SPEC['pca'],
            'regression_policy': f.c.FROZEN_SPEC['regression'],
            'artifact': {'path': spec['paths']['stats'],
                         'serialized_sha256': spec['frozen_hashes']['stats']},
            'input_source_hashes': {
                str(f.resolve(spec['paths'][key])): spec['frozen_hashes'][key]
                for key in ('construction_spec', 'corpus_spec', 'corpus_manifest', 'fineweb_tokens')},
        }
        for relative in ('scripts/ablation/construct_blind_centered_gradient_pca4_pilot.py',
                         'scripts/ablation/construct_generic_hessian_krylov.py'):
            manifest['input_source_hashes'][str(ROOT/relative)] = spec['sources'][relative]
        f.validate_construction_links(manifest, spec)
        for key, wrong in (('batches', {}), ('artifact', {'path': 'wrong', 'serialized_sha256': '0'*64}),
                           ('input_source_hashes', {})):
            bad = copy.deepcopy(manifest)
            bad[key] = wrong
            with self.assertRaisesRegex(ValueError, 'linkage'):
                f.validate_construction_links(bad, spec)

    def test_stale_output_refusal_without_artifact_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'functional-evaluation.json'
            f.require_output_absent(path)
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                f.require_output_absent(path)

    def test_invalid_spec_or_hash_fails_before_any_root_artifact_read(self):
        bad = copy.deepcopy(f.FROZEN_SPEC)
        bad['candidate']['normalization'] = True
        with self.assertRaisesRegex(ValueError, 'spec mismatch'):
            f.validate_frozen_inputs(bad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                f.require_hash(path, '0'*64)

    def test_complete_response_precedes_oracle_and_output_is_evaluation_only(self):
        spec_path = f.ATTEMPT_DIR/'functional_evaluation_spec.json'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'functional-evaluation.json'
            paths = {'final': Path(directory)/'final', 'probe': Path(directory)/'probe.pt',
                     'output': output}
            hashes = {str(spec_path): 'spec_hash'}
            events, written = [], []
            model = SimpleNamespace()
            response = torch.ones((128, 2048), dtype=torch.float32)
            def candidate(*_):
                events.append('candidate')
                return {'weight': torch.ones(1)}, {'candidate_parameter_norm': 1.}
            def complete(*_):
                events.append('complete_jvp')
                return response, response, {'readout_hook_verified': True}, 'response_hash'
            def oracle(*_):
                events.append('oracle')
                return {'ft_mean': response, 'difference': response}
            with (patch.object(f.c.a, 'load_json_object', return_value=f.FROZEN_SPEC),
                  patch.object(f, 'validate_frozen_inputs', return_value=(paths, {}, {}, {},
                                                                          torch.empty(0), hashes)),
                  patch.object(f.activation, 'load_local_model', return_value=model),
                  patch.object(f.activation, 'model_state_hashes', return_value={}),
                  patch.object(f, 'reconstruct_frozen_candidate', side_effect=candidate),
                  patch.object(f, 'require_hash'),
                  patch.object(f.activation, 'load_probe', return_value=torch.empty(0)),
                  patch.object(f, 'compute_complete_response', side_effect=complete),
                  patch.object(f.activation, 'verify_model_unchanged'),
                  patch.object(f, 'validate_oracle_after_response', side_effect=oracle),
                  patch.object(f.consistency, 'merged_mean_consistency', return_value={}),
                  patch.object(f.c.a, 'checkpoint_file_records', return_value=f.FROZEN_SPEC['canonical_checkpoint_files']),
                  patch.object(f.c.a, 'write_manifest', side_effect=lambda _, record: written.append(record))):
                f.evaluate(SimpleNamespace(device='cpu'))
            self.assertEqual(events, ['candidate', 'complete_jvp', 'oracle'])
            self.assertEqual(len(written), 1)
            for key in ('selection', 'sign_flip', 'coefficient_or_pca_mutation',
                        'candidate_rescaling_before_jvp', 'historical_base_access'):
                self.assertFalse(written[0][key])
            self.assertEqual(written[0]['raw_response_sha256'], 'response_hash')


if __name__ == '__main__':
    unittest.main()
