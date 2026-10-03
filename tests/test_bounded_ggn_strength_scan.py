"""Synthetic and source-contract tests for Attempt120; never load real models."""
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT/'scripts/ablation/run_bounded_ggn_strength_scan.py'
loader = importlib.util.spec_from_file_location('attempt120_synthetic', SOURCE)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)
SPEC = json.loads(m.SPEC_PATH.read_text())


def response(value=1.0):
    return torch.full((128, 2048), float(value), dtype=torch.float32)


class Attempt120Tests(unittest.TestCase):
    def test_frozen_spec_and_two_value_grid(self):
        self.assertEqual(m.sha256_file(m.SPEC_PATH), m.SPEC_SHA256)
        self.assertEqual(m.load_spec(), SPEC)
        self.assertEqual(m.GRID, (('1e-2', .01), ('1e-1', .1)))
        self.assertEqual(SPEC['scan']['damping_fractions'], [.01, .1])
        self.assertEqual(SPEC['scan']['solve_order'], ['1e-2', '1e-1'])
        self.assertEqual(SPEC['metrics']['development_scan'], ['bounded_1e-2', 'bounded_1e-1'])
        self.assertEqual(SPEC['artifact']['candidate_keys'], ['1e-2', '1e-1'])
        changed = copy.deepcopy(SPEC)
        changed['scan']['damping_fractions'].append(1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'spec.json'
            path.write_text(json.dumps(changed))
            with patch.object(m, 'SPEC_PATH', path), patch.object(m, 'SPEC_SHA256', m.sha256_file(path)):
                with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
                    m.load_spec()

    def test_frozen_rho_rhs_and_lambda_arithmetic(self):
        self.assertEqual(m.EXPECTED_RHO, 561.5508153958936)
        self.assertEqual(SPEC['scientific_control']['expected_rhs_aggregate_sha256'], m.EXPECTED_RHS)
        self.assertEqual(SPEC['scientific_control']['expected_raw_response_sha256'], m.EXPECTED_RAW)
        self.assertEqual(m.fixed_lambda(m.EXPECTED_RHO, .01), 5.615508153958936)
        self.assertEqual(m.fixed_lambda(m.EXPECTED_RHO, .1), 56.15508153958936)
        self.assertEqual(SPEC['scan']['predeclared_lambdas'], m.EXPECTED_LAMBDAS)
        for fraction in (.001, 1.):
            with self.assertRaisesRegex(ValueError, 'changed'):
                m.fixed_lambda(m.EXPECTED_RHO, fraction)
        with self.assertRaisesRegex(ValueError, 'changed'):
            m.fixed_lambda(m.EXPECTED_RHO+1, .01)

    def test_attempt116_provenance_and_scientific_control(self):
        record = SPEC['frozen_attempt116']
        self.assertEqual(record['commit'], 'f2f553b481ed8f3767eb33c649c48a711e25c0a7')
        self.assertEqual(m.sha256_file(m.path_of(record['spec']['path'])), record['spec']['sha256'])
        self.assertEqual(m.sha256_file(m.path_of(record['source']['path'])), record['source']['sha256'])
        self.assertEqual(m.sha256_file(m.path_of(record['manifest']['path'])), record['manifest']['sha256'])
        s116 = json.loads(m.path_of(record['spec']['path']).read_text())
        control = SPEC['scientific_control']
        self.assertEqual(s116['rhs']['rows'], [0, 512])
        self.assertEqual(s116['rhs']['batch_size'], 8)
        self.assertEqual(s116['ggn']['rows'], [0, 64])
        self.assertEqual(s116['ggn']['denominator'], 8128)
        self.assertEqual(s116['ggn']['operator'], control['ggn_operator'])
        self.assertEqual(s116['fisher_numerics']['implementation'], control['ggn_numerics'])
        self.assertEqual(s116['eligible_tensors']['expected_matrix_count'], 98)
        self.assertEqual(s116['probe']['rows'], [0, 1024])
        self.assertEqual(s116['solver']['iterations'], 80)
        self.assertIsNone(s116['solver']['preconditioner'])
        self.assertFalse(s116['solver']['residual_early_stop'])
        self.assertIn('a117.validate_construction_record(record, manifest, s116)',
                      inspect.getsource(m.load_blind_kernels))
        self.assertIn('sha256_file(path_of(record[\'artifact\'][\'path\']))',
                      inspect.getsource(m.load_blind_kernels))

    def test_upstream_hash_rejection_before_model_work(self):
        changed = copy.deepcopy(SPEC)
        changed['frozen_attempt116']['artifact']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'numerical record mismatch'):
            m.load_blind_kernels(changed)
        changed = copy.deepcopy(SPEC)
        changed['frozen_attempt116']['source']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'source SHA256'):
            m.load_blind_kernels(changed)

    def test_frozen_116_manifest_and_helpers_validate_without_model(self):
        # The 2.7-GB serialized artifact is covered by the runtime hash guard;
        # this synthetic test avoids reading it while exercising the rest.
        artifact = m.path_of(SPEC['frozen_attempt116']['artifact']['path'])
        original_hash = m.sha256_file
        def hash_without_large_artifact(path):
            if Path(path) == artifact:
                return SPEC['frozen_attempt116']['artifact']['sha256']
            return original_hash(path)
        with patch.object(m, 'sha256_file', side_effect=hash_without_large_artifact):
            kernels, _a116, s116, manifest, _a117 = m.load_blind_kernels(SPEC)
        self.assertEqual(len(manifest['coordinate_inventory']), 98)
        self.assertEqual(manifest['b_blind']['ordered_matrix_hash_sha256'], m.EXPECTED_RHS)
        self.assertEqual(s116['ggn']['operator'], SPEC['scientific_control']['ggn_operator'])
        self.assertTrue(callable(kernels.ggn_action_stage))

    def test_independent_zero_start_no_warm_start(self):
        modules = [(f'model.layers.{i}.linear', torch.nn.Linear(1, 1, bias=False))
                   for i in range(98)]
        rhs = {name+'.weight': torch.tensor([[float(i+1)]], dtype=torch.float32)
               for i, (name, _) in enumerate(modules)}
        kernels = SimpleNamespace(matrixwise_dot=lambda x, y: sum(
            float((x[k]*y[k]).sum()) for k in x))
        first = m.fresh_state(rhs, modules, kernels)
        first['x'][next(iter(first['x']))].fill_(100.)
        second = m.fresh_state(rhs, modules, kernels)
        self.assertTrue(all(torch.count_nonzero(t) == 0 for t in second['x'].values()))
        self.assertTrue(all(torch.equal(second['r'][k], rhs[k]) and
                            torch.equal(second['p'][k], rhs[k]) for k in rhs))
        self.assertEqual(second['iteration'], 0)
        self.assertEqual(second['iterations'], [])
        self.assertEqual(len(rhs), 98)

    def test_fixed_eighty_and_midpoint_only_checkpoint(self):
        s116 = json.loads(m.path_of(SPEC['frozen_attempt116']['spec']['path']).read_text())
        for label, fraction in m.GRID:
            run_spec = m.runtime_spec(s116, SPEC, label, fraction)
            self.assertEqual(run_spec['solver'], s116['solver'])
            self.assertEqual(run_spec['solver']['iterations'], 80)
            self.assertEqual(run_spec['solver']['checkpoint_iteration'], 40)
            self.assertEqual(run_spec['solver']['candidate'], 'x_80')
            self.assertIsNone(run_spec['solver']['preconditioner'])
            self.assertFalse(run_spec['solver']['residual_early_stop'])
            self.assertEqual(run_spec['damping']['damping_fraction'], fraction)
        source = inspect.getsource(m.solve_one)
        self.assertIn('kernels.run_fixed_80_cg(', source)
        self.assertIn("if state['iteration'] != 80", source)
        self.assertIn('kernels.checkpoint_payload(', source)
        self.assertIn('kernels.atomic_torch_publish(checkpoint, payload)', source)
        self.assertNotIn('warm_start', source)
        self.assertEqual(SPEC['scan']['candidate_names'], ['x_80_1e-2', 'x_80_1e-1'])

    def test_staged_first_x80_hash_validation_rejects_x40(self):
        inventory = [{'name': f'model.layers.{i}.linear.weight', 'shape': [1, 1]}
                     for i in range(98)]
        x = {row['name']: torch.tensor([[float(i)]], dtype=torch.float32)
             for i, row in enumerate(inventory)}
        def ordered_hashes(vector, rows):
            hashes, digest = {}, hashlib.sha256()
            for row in rows:
                name = row['name']
                tensor = vector[name]
                self.assertEqual(list(tensor.shape), row['shape'])
                raw = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                hashes[name] = raw
                digest.update(name.encode()+b'\0'+raw.encode()+b'\n')
            return hashes, digest.hexdigest()
        def response_hash(tensor, _torch):
            return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        kernels = SimpleNamespace(ordered_vector_hashes=ordered_hashes,
                                  blind=SimpleNamespace(sha256_raw_float32_tensor=response_hash))
        inverse, bounded = response(.25), response(.5)
        hashes, aggregate = ordered_hashes(x, inventory)
        record = {'damping_fraction': .01,
                  'lambda': m.fixed_lambda(m.EXPECTED_RHO, .01),
                  'solver': {'operator_applications': 80, 'iterations': [{} for _ in range(80)]},
                  'x_80': {'matrix_raw_sha256': hashes,
                           'ordered_matrix_hash_sha256': aggregate},
                  'responses': {'response_raw_rhs_sha256': m.EXPECTED_RAW,
                                'response_inverse_sha256': response_hash(inverse, torch),
                                'response_bounded_sha256': response_hash(bounded, torch)}}
        candidate = {'x_80': x, 'response_inverse': inverse, 'response_bounded': bounded}
        rhs = {'ordered_matrix_hash_sha256': m.EXPECTED_RHS}
        payload = m.stage_payload(SPEC, inventory, rhs, m.EXPECTED_RHO,
                                  candidate, record, kernels)
        restored, restored_record = m.validate_stage(
            payload, SPEC, inventory, rhs, m.EXPECTED_RHO, kernels)
        self.assertIs(restored, candidate)
        self.assertIs(restored_record, record)
        incomplete = copy.deepcopy(payload)
        incomplete['candidate_record']['solver']['operator_applications'] = 40
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            m.validate_stage(incomplete, SPEC, inventory, rhs, m.EXPECTED_RHO, kernels)
        altered = copy.deepcopy(payload)
        altered['candidate']['x_80'][inventory[0]['name']].add_(1)
        with self.assertRaisesRegex(ValueError, 'x_80 hash mismatch'):
            m.validate_stage(altered, SPEC, inventory, rhs, m.EXPECTED_RHO, kernels)

    def test_bounded_arithmetic_fp64_then_float32(self):
        raw = response(2.)
        inverse = response(.3)
        damping = m.fixed_lambda(m.EXPECTED_RHO, .01)
        actual = m.bounded_response(raw, inverse, damping)
        expected = (raw.double()-damping*inverse.double()).float().contiguous()
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(actual.device.type, 'cpu')
        self.assertTrue(actual.is_contiguous())
        self.assertFalse(torch.equal(actual, (raw-damping*inverse)))
        self.assertIn('raw.to(torch.float64) - damping * inverse.to(torch.float64)',
                      inspect.getsource(m.bounded_response))

    def test_malformed_response_rejection_and_no_extra_transform(self):
        damping = m.fixed_lambda(m.EXPECTED_RHO, .1)
        for invalid in (torch.ones(4, 4), response().double(),
                        response().T.contiguous(), response(float('nan'))):
            with self.assertRaisesRegex(ValueError, 'Malformed blind response'):
                m.bounded_response(invalid, response(), damping)
        with self.assertRaisesRegex(ValueError, 'outside frozen grid'):
            m.bounded_response(response(), response(), .5)
        self.assertFalse(SPEC['scan']['bounded_response_normalization'])
        self.assertFalse(SPEC['information_policy']['sign_selection'])
        self.assertFalse(SPEC['information_policy']['candidate_rescaling'])
        self.assertFalse(SPEC['scan']['iterate_selection'])
        self.assertFalse(SPEC['scan']['warm_start'])

    def test_oracle_free_norm_ratios_with_null_denominators(self):
        raw = response(1.)
        bounded = response(2.)
        raw[0].zero_()
        metrics = m.norm_ratios(bounded, raw)
        self.assertIsNone(metrics['per_position'][0])
        self.assertEqual(metrics['per_position'][1], 2.)
        self.assertEqual(len(metrics['per_position']), 128)

    def test_signed_cosine_and_required_mean_null(self):
        validator = m._import_pinned(m.PRIOR117_SOURCE,
                                  SPEC['frozen_attempt117_validator']['source_sha256'],
                                  'attempt120_test_117')
        raw = response(1.)
        target = response(1.)
        raw[0].neg_()
        raw[2].neg_()
        report = validator.cosine_report(raw, target)
        self.assertAlmostEqual(report['position_0_cosine'], -1.)
        for actual, expected in zip(report['positions_1_4_individual_cosines'], [1., -1., 1., 1.]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(report['positions_1_4_mean_cosine'], .5)
        self.assertAlmostEqual(report['positions_1_127_mean_cosine'], 125/127)
        raw[2].zero_()
        report = validator.cosine_report(raw, target)
        self.assertIsNone(report['all_128_position_cosines'][2])
        self.assertIsNone(report['positions_1_4_mean_cosine'])
        self.assertIsNone(report['positions_1_127_mean_cosine'])

    def test_category_boundaries_and_same_category_tie_break(self):
        self.assertEqual(m.category(.03, .02), 'clear_improvement')
        self.assertEqual(m.category(.03, .01999), 'moderate_improvement')
        self.assertEqual(m.category(.015, 0.), 'moderate_improvement')
        self.assertEqual(m.category(.01499, 1.), 'no_meaningful_improvement')
        self.assertEqual(m.category(None, .1), 'undefined_required_position_cosine')
        rows = {'1e-2': {'category': 'clear_improvement', 'gain_primary_vs_raw': .04,
                         'gain_secondary_vs_raw': .03},
                '1e-1': {'category': 'clear_improvement', 'gain_primary_vs_raw': .05,
                         'gain_secondary_vs_raw': .02}}
        self.assertEqual(m.descriptive_dev_choice(rows), '1e-1')
        rows['1e-2']['gain_primary_vs_raw'] = .05
        self.assertEqual(m.descriptive_dev_choice(rows), '1e-2')
        rows['1e-2']['gain_secondary_vs_raw'] = .02
        self.assertEqual(m.descriptive_dev_choice(rows), '1e-2')

    def test_dev_choice_clear_over_moderate(self):
        rows = {'1e-2': {'category': 'moderate_improvement',
                         'gain_primary_vs_raw': .2, 'gain_secondary_vs_raw': .2},
                '1e-1': {'category': 'clear_improvement',
                         'gain_primary_vs_raw': .03, 'gain_secondary_vs_raw': .02}}
        self.assertEqual(m.descriptive_dev_choice(rows), '1e-1')
        rows['1e-2']['category'] = 'clear_improvement'
        rows['1e-1']['category'] = 'moderate_improvement'
        self.assertEqual(m.descriptive_dev_choice(rows), '1e-2')

    def test_dev_choice_moderate_over_no_meaningful(self):
        rows = {'1e-2': {'category': 'no_meaningful_improvement',
                         'gain_primary_vs_raw': .2, 'gain_secondary_vs_raw': .2},
                '1e-1': {'category': 'moderate_improvement',
                         'gain_primary_vs_raw': .015, 'gain_secondary_vs_raw': 0.}}
        self.assertEqual(m.descriptive_dev_choice(rows), '1e-1')
        rows['1e-2']['category'] = 'moderate_improvement'
        rows['1e-1']['category'] = 'no_meaningful_improvement'
        self.assertEqual(m.descriptive_dev_choice(rows), '1e-2')

    def test_dev_choice_both_no_meaningful(self):
        rows = {'1e-2': {'category': 'no_meaningful_improvement',
                         'gain_primary_vs_raw': .01, 'gain_secondary_vs_raw': 0.},
                '1e-1': {'category': 'no_meaningful_improvement',
                         'gain_primary_vs_raw': .005, 'gain_secondary_vs_raw': .01}}
        self.assertIsNone(m.descriptive_dev_choice(rows))

    def test_overwrite_refusal_and_sequential_recovery_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'existing.pt'
            path.write_text('frozen')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.require_absent(path)
            self.assertEqual(path.read_text(), 'frozen')
        paths = SPEC['paths']
        self.assertNotEqual(paths['checkpoint_1e-2'], paths['checkpoint_1e-1'])
        self.assertNotEqual(paths['stage_first_path'], paths['artifact_path'])
        body = inspect.getsource(m.run)
        self.assertIn("if cp1.is_file()", body)
        self.assertIn('cp1.unlink()', body)
        self.assertIn("for path in (stage, cp2)", body)
        self.assertIn('del first', body)
        self.assertIn('resume=resume and cp2.is_file()', body)
        self.assertIn('require_absent(output_path)', inspect.getsource(m.evaluate_frozen))

    def test_barrier_publishes_both_before_oracle(self):
        body = inspect.getsource(m.run)
        self.assertLess(body.index("'1e-2', 0.01"), body.index("'1e-1', 0.1"))
        self.assertLess(body.index('artifact, manifest = publish_construction('),
                        body.rindex('return evaluate_frozen(spec, artifact, manifest, a113)'))
        publish = inspect.getsource(m.publish_construction)
        self.assertLess(publish.index('kernels.atomic_torch_publish('),
                        publish.index('kernels.atomic_json_publish('))
        self.assertLess(publish.index('kernels.atomic_json_publish('),
                        publish.index("print('ATTEMPT120_CONSTRUCTION_FROZEN'"))
        evaluate = inspect.getsource(m.evaluate_frozen)
        self.assertLess(evaluate.index('validate_artifact(artifact, manifest, spec, kernels)'),
                        evaluate.index('a117.validate_oracle('))
        self.assertTrue(SPEC['barrier']['both_candidates_and_responses_hashed_before_oracle'])
        self.assertTrue(SPEC['barrier']['artifact_and_manifest_published_before_oracle'])

    def test_attempt119_context_provenance_is_after_barrier(self):
        self.assertEqual(SPEC['frozen_attempt119']['result_sha256'],
                         '58ce13731fb18a6f74cd3532d4d3c598acc457333289725b8edbd0023e901c61')
        self.assertEqual(m.sha256_file(m.path_of(SPEC['frozen_attempt119']['result_path'])),
                         SPEC['frozen_attempt119']['result_sha256'])
        body = inspect.getsource(m.run)
        self.assertNotIn('load_attempt119_context', body)
        evaluate = inspect.getsource(m.evaluate_frozen)
        self.assertIn('load_attempt119_context', evaluate)
        self.assertEqual(SPEC['frozen_attempt119']['role'],
                         'descriptive_context_after_construction_barrier')

    def test_no_oracle_before_barrier_and_disclosed_development(self):
        source = SOURCE.read_text()
        pre = source[:source.index('def evaluate_frozen(')]
        self.assertNotIn('validate_oracle(', pre)
        policy = SPEC['information_policy']
        self.assertFalse(policy['historical_base_access'])
        self.assertFalse(policy['adapter_access'])
        self.assertFalse(policy['true_delta_access'])
        self.assertFalse(policy['oracle_adl_access_before_barrier'])
        self.assertFalse(policy['oracle_based_iterate_selection'])
        self.assertFalse(policy['oracle_based_damping_selection_during_construction'])
        self.assertTrue(policy['same_specimen_exploratory_method_development'])
        self.assertFalse(policy['clean_heldout_validation'])
        self.assertTrue(SPEC['interpretation']['dev_set_selection_disclosed'])
        self.assertTrue(SPEC['interpretation']['no_third_lambda_after_scores'])


if __name__ == '__main__':
    unittest.main()
