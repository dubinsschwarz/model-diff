"""Synthetic and frozen-spec tests for the blind Attempt113 constructor."""
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
SOURCE = PROJECT/'scripts/ablation/construct_blind_diverse8_unit_gradient_inverse_ggn.py'
loader = importlib.util.spec_from_file_location('attempt113_tests', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = m.load_spec()


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3, bias=False)
        self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, input_ids, use_cache=False):
        onehot = torch.nn.functional.one_hot(input_ids, 3).float()
        return SimpleNamespace(logits=self.linear(onehot))


def toy_tokens(rows=32):
    return (torch.arange(rows*128, dtype=torch.int64).reshape(rows, 128) % 3).contiguous()


def toy_eligible():
    module = torch.nn.Linear(100, 1, bias=False)
    eligible = [('linear', module)]
    inventory = [{'name': 'linear.weight', 'shape': [1, 100]}]
    return module, eligible, inventory


def toy_solver(*, callback=None, start_state=None, calls=None, report=None):
    layer, eligible, _ = toy_eligible()
    diagonal = torch.logspace(-2, 3, 100).reshape(1, 100)
    calls = [] if calls is None else calls
    if start_state is None:
        rhs = torch.ones((1, 100), dtype=torch.float32)
        norm = math.sqrt(float(torch.sum(rhs*rhs, dtype=torch.float64)))
        state = {'iteration': 0, 'x': {'linear.weight': torch.zeros_like(rhs)},
                 'r': {'linear.weight': rhs.clone()},
                 'p': {'linear.weight': rhs.clone()},
                 'rr': norm*norm, 'b_norm': norm, 'iterations': []}
    else:
        state = start_state

    def apply(vector, iteration):
        calls.append(iteration)
        layer.weight.grad = vector['linear.weight']*diagonal

    return m.run_fixed_80_cg(
        state, eligible, apply, 0.1, SPEC,
        on_checkpoint=None if state['iteration'] == 40 else (callback or (lambda _: None)),
        report=report), calls


def checkpoint_fixture():
    _, eligible, inventory = toy_eligible()
    value = torch.ones((1, 100), dtype=torch.float32)
    rows = [{'iteration': i, 'relative_residual': 0.5, 'r_norm': 5.,
             'pAp': 2., 'alpha': 0.1, 'beta': 0.2,
             'operator_seconds': 1., 'cumulative_elapsed_seconds': float(i),
             'operator_eta_seconds': float(80-i)} for i in range(1, 41)]
    state = {'iteration': 40,
             'x': {'linear.weight': value.clone()},
             'r': {'linear.weight': value.clone()},
             'p': {'linear.weight': value.clone()},
             'rr': 100., 'b_norm': 10., 'iterations': rows}
    rhs = {'norm': 10., 'matrix_raw_sha256': {'linear.weight': 'a'*64},
           'ordered_matrix_hash_sha256': 'b'*64}
    gradients = [{'name': name, 'mean_generic_loss': 1.,
                  'global_gradient_norm': 1., 'unit_normalization_scalar': 1.,
                  'rows': [0, 512], 'prediction_contexts': 65024}
                 for name in m.ORDER]
    rayleigh = {'rho_b': 100., 'lambda': 0.01, 'damping_fraction': 1e-4,
                'bGb': 10000., 'bb': 100.}
    return state, eligible, inventory, rhs, gradients, rayleigh


def fake_manifest(candidate, inventory, raw_response, inverse_response):
    x_hashes, aggregate = m.ordered_vector_hashes(candidate, inventory)
    return {'solver': {'operator_applications': 80,
                       'iterations': [{} for _ in range(80)]},
            'candidate': {'name': 'x_80', 'x_80': {
                'matrix_raw_sha256': x_hashes,
                'ordered_matrix_hash_sha256': aggregate}},
            'functional_responses': {
                'response_raw_rhs': {'raw_sha256':
                    m.blind.sha256_raw_float32_tensor(raw_response, torch)},
                'response_inverse': {'raw_sha256':
                    m.blind.sha256_raw_float32_tensor(inverse_response, torch)}}}


class Attempt113Tests(unittest.TestCase):
    def test_01_exact_eight_corpus_order(self):
        self.assertEqual(SPEC['corpus_order'], list(m.ORDER))
        self.assertEqual(len(SPEC['corpora']), 8)

    def test_02_exact_rows_0_512(self):
        self.assertEqual(SPEC['rhs']['rows'], [0, 512])
        self.assertTrue(all(row['selected_rows'] == [0, 512]
                            for row in SPEC['corpora']))

    def test_03_batch_size_8(self):
        self.assertEqual(SPEC['rhs']['batch_size'], 8)
        self.assertEqual(SPEC['rhs']['batches_per_corpus'], 64)

    def test_04_causal_mean_CE_denominator(self):
        model = TinyLM()
        batch = toy_tokens()
        loss, _ = m.accumulate_corpus_gradient(
            model, batch, [('linear', model.linear)], rows=32)
        with torch.no_grad():
            summed = m.blind.causal_token_loss_sum(model(batch).logits, batch, torch)
        self.assertAlmostEqual(loss, float(summed)/(32*127), places=6)
        self.assertEqual(SPEC['rhs']['prediction_contexts_per_corpus'], 512*127)

    def test_05_exact_98_matrix_support(self):
        self.assertEqual(SPEC['eligible_tensors']['expected_matrix_count'], 98)
        self.assertEqual(SPEC['eligible_tensors']['transformer_block_indices'],
                         list(range(14)))

    def test_06_all_other_parameters_frozen(self):
        model = TinyLM()
        m.blind.freeze_other_parameters(model, [('linear', model.linear)])
        self.assertTrue(model.linear.weight.requires_grad)
        self.assertFalse(model.frozen.requires_grad)

    def test_07_global_FP64_gradient_norm(self):
        module = torch.nn.Linear(2, 1, bias=False)
        module.weight.grad = torch.tensor([[3., 4.]], dtype=torch.float32)
        self.assertEqual(m.matrixwise_fp64_norm([('linear', module)]), 5.)

    def test_08_unit_gradient_definition(self):
        module = torch.nn.Linear(2, 1, bias=False)
        module.weight.grad = torch.tensor([[3., 4.]], dtype=torch.float32)
        eligible = [('linear', module)]
        inventory = [{'name': 'linear.weight', 'shape': [1, 2]}]
        consensus = {}
        scale = m.add_unit_gradient_to_consensus(
            consensus, eligible, 5., inventory, corpus_index=0, corpus_count=1)
        self.assertEqual(scale, 0.2)
        self.assertTrue(torch.allclose(consensus['linear.weight'],
                                       torch.tensor([[0.6, 0.8]])))

    def test_09_positive_signs_only(self):
        module = torch.nn.Linear(1, 1, bias=False)
        module.weight.grad = torch.tensor([[2.]], dtype=torch.float32)
        consensus = {}
        m.add_unit_gradient_to_consensus(consensus, [('linear', module)], 2.,
            [{'name': 'linear.weight', 'shape': [1, 1]}], corpus_index=0)
        self.assertGreater(float(consensus['linear.weight']), 0)
        self.assertEqual(SPEC['rhs']['sign'], 'positive_gradients_only')

    def test_10_equal_eighth_parameter_mean(self):
        module = torch.nn.Linear(1, 1, bias=False)
        eligible = [('linear', module)]
        inventory = [{'name': 'linear.weight', 'shape': [1, 1]}]
        consensus = {}
        for index, sign in enumerate((1, 1, 1, 1, 1, 1, 1, -1)):
            module.weight.grad = torch.tensor([[float(sign)]])
            m.add_unit_gradient_to_consensus(
                consensus, eligible, 1., inventory, corpus_index=index)
        self.assertEqual(float(consensus['linear.weight']), 0.75)
        self.assertEqual(SPEC['rhs']['corpus_weights'], [0.125]*8)

    def test_11_no_final_rhs_renormalization(self):
        self.assertFalse(SPEC['rhs']['renormalize_final_rhs'])
        module = torch.nn.Linear(1, 1, bias=False)
        eligible = [('linear', module)]
        inventory = [{'name': 'linear.weight', 'shape': [1, 1]}]
        consensus = {}
        for index, sign in enumerate((1, 1, 1, 1, 1, 1, 1, -1)):
            module.weight.grad = torch.tensor([[float(sign)]])
            m.add_unit_gradient_to_consensus(consensus, eligible, 1., inventory,
                                             corpus_index=index)
        self.assertEqual(float(consensus['linear.weight']), 0.75)

    def test_12_exact_categorical_Fisher(self):
        logits = torch.tensor([[[0.2, -0.4, 1.1]]], dtype=torch.float32)
        tangent = torch.tensor([[[2., -1., 0.5]]], dtype=torch.float32)
        u, _ = m.stable_categorical_fisher_vector(logits, tangent, SPEC)
        p = torch.softmax(logits.double()[0, 0], 0)
        explicit = (torch.diag(p)-torch.outer(p, p)) @ tangent.double()[0, 0]
        self.assertTrue(torch.allclose(u.double()[0, 0]*8128, explicit,
                                       rtol=1e-7, atol=1e-7))

    def test_13_same_fineweb_G_artifact_hash(self):
        self.assertEqual(SPEC['ggn']['serialized_sha256'],
                         '54283d72aa3075e2867ead86e538991d7284ede327ca67871e73bfc581e4025d')
        self.assertEqual(SPEC['ggn']['raw_tensor_sha256'],
                         '4613982be77342a89b35616d68e21d9ff66d5a983af7e82cb6fa1eb159457eae')

    def test_14_G_rows_exactly_0_64(self):
        self.assertEqual(SPEC['ggn']['rows'], [0, 64])
        self.assertEqual(SPEC['ggn']['microbatches'], 64)

    def test_15_G_denominator_8128(self):
        self.assertEqual(SPEC['ggn']['denominator'], 64*127)
        self.assertEqual(SPEC['ggn']['prediction_contexts'], 8128)

    def test_16_FP64_gauge_invariance(self):
        logits = torch.tensor([[[0., 1., -1., 0.5]]], dtype=torch.float32)
        tangent = torch.tensor([[[0., 2., -3., 1.]]], dtype=torch.float32)
        one, audit = m.stable_categorical_fisher_vector(logits, tangent, SPEC)
        two, _ = m.stable_categorical_fisher_vector(logits, tangent+10000., SPEC)
        self.assertTrue(torch.allclose(one, two, atol=1e-8, rtol=1e-5))
        self.assertLess(audit['max_abs_sum_u64'], 1e-12)

    def test_17_no_empirical_fisher(self):
        self.assertFalse(SPEC['ggn']['empirical_fisher'])
        self.assertFalse(SPEC['ggn']['gradient_squares'])
        self.assertFalse(SPEC['ggn']['finite_differences'])

    def test_18_Rayleigh_formula(self):
        module = torch.nn.Linear(2, 1, bias=False)
        module.weight.grad = torch.tensor([[8., 18.]], dtype=torch.float32)
        direction = {'linear.weight': torch.tensor([[2., 3.]], dtype=torch.float32)}
        result = m.rayleigh_and_lambda([('linear', module)], direction, 1e-4)
        self.assertEqual(result['rho_b'], (2*8+3*18)/(2*2+3*3))

    def test_19_damping_fraction_exact(self):
        self.assertEqual(SPEC['damping']['damping_fraction'], 1e-4)
        self.assertFalse(SPEC['damping']['sweep'])

    def test_20_lambda_uses_only_blind_rho(self):
        module = torch.nn.Linear(1, 1, bias=False)
        module.weight.grad = torch.tensor([[7.]], dtype=torch.float32)
        result = m.rayleigh_and_lambda(
            [('linear', module)], {'linear.weight': torch.tensor([[2.]])}, 1e-4)
        self.assertEqual(result['lambda'], 1e-4*result['rho_b'])

    def test_21_no_absolute_privileged_lambda(self):
        self.assertNotIn('fixed_lambda', SPEC['solver'])
        self.assertNotIn('lambda', SPEC['solver'])
        self.assertNotIn('0.13361159773737232', SOURCE.read_text())

    def test_22_exactly_80_CG_iterations(self):
        (state, solver), calls = toy_solver()
        self.assertEqual(calls, list(range(1, 81)))
        self.assertEqual(state['iteration'], 80)
        self.assertEqual(solver['operator_applications'], 80)

    def test_23_no_residual_early_stop(self):
        self.assertFalse(SPEC['solver']['residual_early_stop'])
        (state, _), _ = toy_solver()
        self.assertEqual(len(state['iterations']), 80)

    def test_24_no_preconditioner(self):
        self.assertIsNone(SPEC['solver']['preconditioner'])
        self.assertEqual(SPEC['solver']['method'], 'ordinary_unpreconditioned_cg')

    def test_25_x80_is_fixed_candidate(self):
        self.assertEqual(SPEC['solver']['candidate'], 'x_80')
        self.assertFalse(SPEC['solver']['candidate_selection'])

    def test_26_matrixwise_FP64_CG_reductions(self):
        x = {'a': torch.tensor([1e8, 1., -1e8], dtype=torch.float32)}
        y = {'a': torch.ones(3, dtype=torch.float32)}
        self.assertEqual(m.matrixwise_dot(x, y), 1.)
        self.assertIn('FP64', SPEC['solver']['global_reductions'])

    def test_27_generic_probe_hashes(self):
        self.assertEqual(SPEC['probe']['serialized_sha256'],
                         '3d799d19abf4a2d6dc92b2bd164d81d83f6451cd3545f5ae5d8c04a8b299735b')
        self.assertEqual(SPEC['probe']['raw_tensor_sha256'],
                         '73f56300fdd06bc8fafcfa5750fee7a586a5c5e6071a608667ffc526bbb90ac8')

    def test_28_probe_rows_0_1024(self):
        self.assertEqual(SPEC['probe']['rows'], [0, 1024])
        self.assertEqual(SPEC['probe']['batch_size'], 32)

    def test_29_strict_block13_JVP(self):
        self.assertTrue(SPEC['jvp']['strict_forward_ad'])
        self.assertFalse(SPEC['jvp']['fallback'])
        self.assertEqual(SPEC['readout']['hidden_state_index'], 14)

    def test_30_raw_rhs_response_frozen(self):
        self.assertIn('response_raw_rhs', SPEC['artifact']['tensor_inventory'])
        self.assertEqual(SPEC['artifact']['response_shape'], [128, 2048])

    def test_31_inverse_response_frozen(self):
        self.assertIn('response_inverse', SPEC['artifact']['tensor_inventory'])
        self.assertEqual(SPEC['artifact']['response_dtype'], 'contiguous_CPU_float32')

    def test_32_atomic_candidate_artifact_no_overwrite(self):
        inventory = [{'name': 'linear.weight', 'shape': [1, 2]}]
        candidate = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}
        raw = torch.zeros(128, 2048, dtype=torch.float32)
        inverse = torch.ones(128, 2048, dtype=torch.float32)
        manifest = fake_manifest(candidate, inventory, raw, inverse)
        with tempfile.TemporaryDirectory() as directory:
            artifact, output, checkpoint = [Path(directory)/name for name in
                                            ('candidate.pt', 'manifest.json', 'checkpoint.pt')]
            checkpoint.write_bytes(b'synthetic')
            m.publish_candidate(artifact, output, checkpoint, candidate, raw,
                                inverse, manifest, inventory)
            self.assertTrue(artifact.is_file())
            self.assertTrue(output.is_file())
            self.assertFalse(checkpoint.exists())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.publish_candidate(artifact, output, checkpoint, candidate, raw,
                                    inverse, manifest, inventory)

    def test_33_candidate_artifact_inventory(self):
        self.assertEqual(SPEC['artifact']['tensor_inventory'],
                         ['x_80', 'response_raw_rhs', 'response_inverse'])
        self.assertEqual(SPEC['artifact']['x_matrix_count'], 98)
        self.assertFalse(SPEC['artifact']['persist_rhs'])

    def test_34_b_matrix_hashes(self):
        inventory = [{'name': 'a', 'shape': [1, 1]}, {'name': 'b', 'shape': [1, 1]}]
        vector = {'a': torch.tensor([[1.]]), 'b': torch.tensor([[2.]])}
        hashes, aggregate = m.ordered_vector_hashes(vector, inventory)
        expected = hashlib.sha256(b''.join(
            name.encode()+b'\0'+hashes[name].encode()+b'\n' for name in ('a', 'b')))
        self.assertEqual(aggregate, expected.hexdigest())

    def test_35_x80_matrix_hashes(self):
        state, eligible, inventory, rhs, gradients, rayleigh = checkpoint_fixture()
        hashes = m.state_hashes(state, inventory)
        self.assertEqual(set(hashes), {'x', 'r', 'p'})
        self.assertEqual(len(hashes['x']['matrix_raw_sha256']), 1)

    def test_36_manifest_forbids_oracle_metrics(self):
        policy = SPEC['information_policy']
        self.assertFalse(policy['prior_oracle_evaluation_access'])
        self.assertFalse(policy['oracle_based_tuning_during_construction'])
        self.assertNotIn('oracle_cosine', SPEC)
        self.assertNotIn('historical_target', SPEC.get('metric', {}))

    def test_37_historical_base_path_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'Forbidden'):
            m.safe_input_path('/root/model-diff-scratch/models/base')

    def test_38_oracle_adl_path_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'Forbidden'):
            m.safe_input_path('/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt')

    def test_39_attempt100_evaluation_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'Forbidden'):
            m.safe_input_path('experiments/attempts/100_diverse8_raw_jg_consensus_prefix0_13/evaluation.json')

    def test_40_privileged_result_files_forbidden(self):
        for index in range(103, 113):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'Forbidden'):
                m.safe_input_path(f'experiments/attempts/attempt{index}/result.json')

    def test_41_iteration40_checkpoint_atomic(self):
        state, eligible, inventory, rhs, gradients, rayleigh = checkpoint_fixture()
        payload = m.checkpoint_payload(state, inventory, SPEC, rhs, gradients, rayleigh)
        self.assertEqual(payload['iteration'], 40)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            m.atomic_torch_publish(path, payload)
            self.assertTrue(path.is_file())
            self.assertEqual(sorted(Path(directory).iterdir()), [path])

    def test_42_resume_exact_state_restoration(self):
        state, eligible, inventory, rhs, gradients, rayleigh = checkpoint_fixture()
        payload = m.checkpoint_payload(state, inventory, SPEC, rhs, gradients, rayleigh)
        m.validate_checkpoint(payload, SPEC, inventory, rhs, gradients, rayleigh)
        restored = m.restore_cg_state(payload, eligible, inventory)
        self.assertEqual(m.state_hashes(restored, inventory), payload['raw_hashes'])

    def test_43_resume_starts_at_41(self):
        midpoint = []
        (original, _), _ = toy_solver(callback=lambda state: midpoint.append(
            copy.deepcopy(state)))
        calls = []
        (continued, _), _ = toy_solver(start_state=midpoint[0], calls=calls)
        self.assertEqual(calls[0], 41)
        self.assertEqual(calls[-1], 80)
        self.assertTrue(torch.equal(continued['x']['linear.weight'],
                                    original['x']['linear.weight']))

    def test_44_success_deletes_checkpoint(self):
        self.assertTrue(SPEC['checkpoint']['delete_after_success'])
        self.assertFalse(SPEC['checkpoint']['model_weights_in_checkpoint'])

    def test_45_failure_after40_preserves_checkpoint(self):
        state, eligible, inventory, rhs, gradients, rayleigh = checkpoint_fixture()
        payload = m.checkpoint_payload(state, inventory, SPEC, rhs, gradients, rayleigh)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'checkpoint.pt'
            m.atomic_torch_publish(path, payload)
            with self.assertRaisesRegex(ValueError, 'before fixed x_80'):
                m.publish_candidate(Path(directory)/'candidate.pt',
                                    Path(directory)/'manifest.json', path,
                                    {}, torch.zeros(128, 2048),
                                    torch.zeros(128, 2048), {}, inventory)
            self.assertTrue(path.is_file())

    def test_46_timing_smoke_writes_nothing(self):
        spec = copy.deepcopy(SPEC)
        model = TinyLM()
        module = model.linear
        inventory = [{'name': 'linear.weight', 'shape': [3, 3]}]
        response = {'norm': 1., 'matrix_raw_sha256': {'linear.weight': 'a'*64},
                    'ordered_matrix_hash_sha256': 'b'*64}
        fake_audit = {'first_batch': {'primal': {'max_abs_difference': 0.}}}
        with patch.object(m, 'load_spec', return_value=spec), \
             patch.object(m, 'refuse_outputs'), \
             patch.object(m, 'validate_blind_sources', return_value=(
                 {'fineweb': toy_tokens(64)}, None, {})), \
             patch.object(m.blind, 'validate_loaded_model'), \
             patch.object(m.blind, 'discover_eligible_linear_weights',
                          return_value=[('linear', module)]), \
             patch.object(m.blind, 'freeze_other_parameters'), \
             patch.object(m, 'coordinate_inventory', return_value=inventory), \
             patch.object(m.blind, 'model_state_hashes', return_value={}), \
             patch.object(m.blind, 'verify_model_unchanged'), \
             patch.object(m, 'build_blind_rhs', return_value=(
                 {'linear.weight': torch.ones(3, 3)},
                 [{'name': 'fineweb'}], response)), \
             patch.object(m, 'ggn_action_stage', return_value=fake_audit), \
             patch.object(m, 'publish_candidate', side_effect=AssertionError('write')):
            result = m.construct(timing_smoke=True,
                model_loader=lambda *_: model)
        self.assertFalse(result['writes'])
        self.assertEqual(result['gradient_batches'], 4)

    def test_47_timing_smoke_final_only(self):
        self.assertTrue(SPEC['timing_smoke']['final_checkpoint_only'])
        self.assertEqual(SPEC['timing_smoke']['fineweb_rows'], [0, 32])
        self.assertEqual(SPEC['timing_smoke']['ggn_applications'], 1)

    def test_48_no_evaluation_script_generated(self):
        self.assertFalse((PROJECT/'scripts/ablation/evaluate_attempt113.py').exists())
        self.assertFalse((PROJECT/'experiments/attempts'/m.ATTEMPT/'evaluation.json').exists())

    def test_49_continuous_progress_and_ETA(self):
        rows = []
        (state, solver), calls = toy_solver(report=rows.append)
        self.assertEqual(len(rows), 80)
        self.assertTrue(all(row['operator_eta_seconds'] >= 0 for row in rows))
        self.assertEqual(solver['iterations'], rows)

    def test_50_env_sh_not_modified_by_constructor(self):
        self.assertNotIn('env.sh', SOURCE.read_text())
        self.assertEqual(set(SPEC['paths']),
                         {'artifact_path', 'checkpoint_path', 'manifest_path'})

    def test_51_manifest_failure_rolls_back_artifact_and_keeps_checkpoint(self):
        inventory = [{'name': 'linear.weight', 'shape': [1, 2]}]
        candidate = {'linear.weight': torch.tensor([[1., 2.]], dtype=torch.float32)}
        raw = torch.zeros(128, 2048, dtype=torch.float32)
        inverse = torch.ones(128, 2048, dtype=torch.float32)
        manifest = fake_manifest(candidate, inventory, raw, inverse)
        with tempfile.TemporaryDirectory() as directory:
            artifact, output, checkpoint = [Path(directory)/name for name in
                                            ('candidate.pt', 'manifest.json', 'checkpoint.pt')]
            checkpoint.write_bytes(b'synthetic')
            with patch.object(m, 'atomic_json_publish',
                              side_effect=RuntimeError('manifest disk error')):
                with self.assertRaisesRegex(RuntimeError, 'manifest disk error'):
                    m.publish_candidate(artifact, output, checkpoint, candidate,
                                        raw, inverse, manifest, inventory)
            self.assertFalse(artifact.exists())
            self.assertFalse(output.exists())
            self.assertTrue(checkpoint.exists())

    def test_52_candidate_matrix_order_is_frozen(self):
        inventory = [{'name': 'a', 'shape': [1, 1]},
                     {'name': 'b', 'shape': [1, 1]}]
        candidate = {'b': torch.ones(1, 1), 'a': torch.ones(1, 1)}
        raw = torch.zeros(128, 2048)
        inverse = torch.ones(128, 2048)
        manifest = fake_manifest(candidate, inventory, raw, inverse)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'checkpoint.pt'
            checkpoint.write_bytes(b'synthetic')
            with self.assertRaisesRegex(ValueError, 'frozen coordinate order'):
                m.publish_candidate(Path(directory)/'candidate.pt',
                                    Path(directory)/'manifest.json', checkpoint,
                                    candidate, raw, inverse, manifest, inventory)


if __name__ == '__main__':
    unittest.main()
