import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / 'scripts/ablation/evaluate_attempt012.py'
MODULE = importlib.util.spec_from_file_location('evaluate012', SCRIPT)
a = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(a)


class EvaluationTests(unittest.TestCase):
    def fixture(self, root):
        spec = copy.deepcopy(a.FROZEN_SPEC)
        paths = {name: root / (name + ('.pt' if 'artifact' in name else '.json')) for name in spec['inputs']}
        readout = dict(block_index=13, hidden_state_index=14, hidden_size=2048)
        probe = dict(sample_count=10000, sequence_length=128, batch_size=32,
                     serialized_sha256='a'*64, raw_tensor_sha256='b'*64)
        attempt = dict(attempt_id=spec['attempt_id'], probe=probe, spectral_filter={'keep': 'abs(theta) > rho'},
                       jvp={'sign': 'positive'}, candidate={'sign': 'positive_J_x_probe'}, readout=readout)
        paths['attempt_spec'].write_text(json.dumps(attempt))
        merged = torch.ones(128,2048)
        response = torch.zeros_like(merged); response[:,0] = 1
        difference = response.clone()
        for p in range(128):
            cosine = (p % 9 - 4) / 4
            response[p,0] = cosine; response[p,1] = math.sqrt(1-cosine*cosine)
        candidate = dict(merged_mean=merged, inverse_response=response,
                         metadata=dict(attempt_id=spec['attempt_id'], response_sign='positive_J_x_probe',
                                       sample_count=10000, readout=readout))
        torch.save(candidate, paths['candidate_artifact'])
        metadata = dict(relative_layer=.5, layer_index=13, num_layers=28, sample_count=10000,
            sequence_length=128, hidden_size=2048, batch_size=32, model_dtype='float32',
            accumulator_dtype='float64', stored_dtype='float32')
        oracle = dict(format_version=1, metadata=metadata, ft_mean=merged.clone(),
                      base_mean=merged-difference, difference=difference)
        torch.save(oracle, paths['oracle_artifact'])
        manifest = dict(attempt_id=spec['attempt_id'], attempt_spec_sha256=a.sha256_file(paths['attempt_spec']),
            probe=probe, spectral_filter=attempt['spectral_filter'], jvp=attempt['jvp'],
            candidate_definition=attempt['candidate'], readout=readout,
            artifact=dict(serialized_sha256=a.sha256_file(paths['candidate_artifact']),
                          raw_tensors_sha256={k:a.raw_hash(candidate[k]) for k in ('merged_mean','inverse_response')}))
        oracle_manifest = dict(metadata, probe={k:probe[k] for k in ('serialized_sha256','raw_tensor_sha256')},
            oracle_adl_sha256=a.sha256_file(paths['oracle_artifact']),
            raw_tensors_sha256={k:a.raw_hash(oracle[k]) for k in ('ft_mean','base_mean','difference')})
        paths['construction_manifest'].write_text(json.dumps(manifest))
        paths['oracle_manifest'].write_text(json.dumps(oracle_manifest))
        for name in ('attempt_spec','construction_manifest','oracle_manifest'):
            spec['inputs'][name]['sha256'] = a.sha256_file(paths[name])
        spec_path = root/'evaluation_spec.json'; spec_path.write_text(json.dumps(spec))
        return spec, spec_path, paths, root/'output.json', candidate, oracle

    def run_fixture(self, fixture):
        spec, path, inputs, output, _, _ = fixture
        with patch.object(a,'FROZEN_SPEC',spec):
            return a.evaluate(path,output,inputs)

    def test_positive_sign_all_positions_and_precommitted_summary(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d)); result=self.run_fixture(f)
            rows=result['positions']
            self.assertEqual([r['position'] for r in rows],list(range(128)))
            self.assertEqual(rows[0]['cosine_similarity'],-1.)
            self.assertEqual(rows[4]['cosine_similarity'],0.)
            summary=result['summaries']
            expected=math.fsum(rows[p]['cosine_similarity'] for p in [1,2,3,4])/4
            self.assertEqual(summary['primary_metric']['value'],expected)
            self.assertEqual(summary['primary_metric']['positions'],[1,2,3,4])
            self.assertEqual(summary['position_0_cosine'],-1.)
            self.assertEqual(summary['individual_positions_0_through_4'],rows[:5])
            self.assertEqual(summary['positions_1_through_127']['positions'],list(range(1,128)))
            self.assertEqual(len(result['merged_mean_consistency']['positions']),128)
            self.assertNotIn('best',json.dumps(result))
            self.assertEqual(len(result['provenance']),7)

    def test_deterministic_output_order_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d)); self.run_fixture(f); original=f[3].read_bytes()
            with self.assertRaisesRegex(ValueError,'already exists'): self.run_fixture(f)
            f[3].unlink();self.run_fixture(f)
            self.assertEqual(original,f[3].read_bytes())

    def test_each_frozen_manifest_and_spec_hash_is_strict(self):
        for name in ('attempt_spec','construction_manifest','oracle_manifest'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));path=f[2][name];path.write_text(path.read_text()+' ')
                with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):self.run_fixture(f)
                self.assertFalse(f[3].exists())

    def test_candidate_and_oracle_serialized_hashes_are_strict(self):
        for name in ('candidate_artifact','oracle_artifact'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));path=f[2][name];path.write_bytes(path.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):self.run_fixture(f)

    def test_altered_raw_tensors_fail_even_with_updated_serialized_hash(self):
        for name,tensor in (('candidate_artifact','inverse_response'),('oracle_artifact','difference')):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));spec,sp,paths,output,candidate,oracle=f
                artifact=candidate if name=='candidate_artifact' else oracle
                artifact[tensor][3,0]+=1
                torch.save(artifact,paths[name])
                manifest_name='construction_manifest' if name=='candidate_artifact' else 'oracle_manifest'
                m=json.loads(paths[manifest_name].read_text())
                if name=='candidate_artifact':m['artifact']['serialized_sha256']=a.sha256_file(paths[name])
                else:m['oracle_adl_sha256']=a.sha256_file(paths[name])
                paths[manifest_name].write_text(json.dumps(m))
                spec['inputs'][manifest_name]['sha256']=a.sha256_file(paths[manifest_name]);sp.write_text(json.dumps(spec))
                with self.assertRaisesRegex(ValueError,'Raw tensor'):self.run_fixture(f)

    def test_cpu_only_exact_input_allowlist_and_no_prior_evaluations(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));allowed={*f[2].values(),f[1],f[3],SCRIPT}
            original=Path.open
            def checked(path,*args,**kwargs):
                self.assertIn(path,allowed)
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',checked),patch.object(torch,'load',wraps=torch.load) as load:
                self.run_fixture(f)
            self.assertEqual(load.call_count,2)
            for call in load.call_args_list:
                self.assertEqual(call.kwargs,dict(map_location='cpu',weights_only=True))
        source=SCRIPT.read_text()
        for forbidden in ('transformers','tokenizer','cuda','logit_lens','krylov_probe.pt','evaluate_attempt008','models/base','adapter','linalg.eigh'):
            self.assertNotIn(forbidden,source)

    def test_geometry_zero_vectors_and_undefined_summary(self):
        c=torch.zeros(128,2048);o=c.clone();c[1,0]=2;o[2,0]=3;c[3,0]=4;o[3,0]=-2
        rows=a.geometry(c,o,list(range(128)))
        self.assertIsNone(rows[0]['cosine_similarity']);self.assertIsNone(rows[1]['candidate_to_oracle_norm_ratio'])
        self.assertEqual(rows[2]['candidate_to_oracle_norm_ratio'],0)
        self.assertEqual(rows[3]['cosine_similarity'],-1)
        self.assertEqual(rows[3]['candidate_norm'],4)
        self.assertEqual(rows[3]['oracle_difference_norm'],2)
        self.assertEqual(rows[3]['candidate_to_oracle_norm_ratio'],2)
        self.assertIsNone(a.summarize(rows,a.FROZEN_SPEC)['primary_metric']['value'])

    def test_gate_exact_equality_and_tiny_hardware_noise_pass(self):
        reference=torch.ones(128,2048)
        for delta in (0.,1e-6):
            candidate=reference.clone();candidate[127,0]+=delta
            result=a.merged_mean_consistency(candidate,reference,a.FROZEN_SPEC)
            self.assertEqual(len(result['positions']),128)

    def test_gate_late_position_failure_cannot_hide_behind_position_zero(self):
        reference=torch.ones(128,2048);reference[0]*=1e8
        candidate=reference.clone();candidate[127]*=1.001
        with self.assertRaisesRegex(ValueError,'position 127'):a.merged_mean_consistency(candidate,reference,a.FROZEN_SPEC)

    def test_gate_each_threshold_independently(self):
        good=dict(position=87,cosine_similarity=1.,relative_rms_difference=0.,maximum_absolute_difference=0.)
        for key,value in [('cosine_similarity',.999999998),('relative_rms_difference',1.01e-5),('maximum_absolute_difference',.01001)]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,key):
                a.require_equivalence({**good,key:value},a.FROZEN_SPEC['merged_mean_consistency'])
        a.require_equivalence(dict(position=0,cosine_similarity=.999999999,relative_rms_difference=1e-5,maximum_absolute_difference=.01),a.FROZEN_SPEC['merged_mean_consistency'])

    def test_gate_zero_policy(self):
        z=torch.zeros(128,2048)
        result=a.merged_mean_consistency(z,z,a.FROZEN_SPEC)
        self.assertEqual(result['positions'][0]['cosine_similarity'],1)
        one=z.clone();one[77,0]=1
        for left,right in ((one,z),(z,one)):
            with self.assertRaisesRegex(ValueError,'position 77'):a.merged_mean_consistency(left,right,a.FROZEN_SPEC)

    def test_bad_tensor_shape_dtype_and_nonfinite(self):
        value=torch.ones(128,2048);spec=a.FROZEN_SPEC
        for bad in (value.double(),value[:2],value+float('nan'),value+float('inf'),torch.ones(2048,128).T):
            with self.assertRaisesRegex(ValueError,'Invalid activation'):a.validate_tensor(bad,'0'*64,spec)

    def test_mutation_after_scoring_aborts_publication(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));original=a.geometry
            def mutate(*args):
                result=original(*args);f[2]['candidate_artifact'].write_bytes(b'changed');return result
            with patch.object(a,'geometry',side_effect=mutate),self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                self.run_fixture(f)
            self.assertFalse(f[3].exists())

    def test_gate_failure_aborts_before_geometry(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d))
            with patch.object(a,'merged_mean_consistency',side_effect=ValueError('gate failed')),patch.object(a,'geometry') as geometry:
                with self.assertRaisesRegex(ValueError,'gate failed'):self.run_fixture(f)
                geometry.assert_not_called()
            self.assertFalse(f[3].exists())

    def test_frozen_evaluation_spec_matches_source_without_opening_real_artifacts(self):
        path=PROJECT/'experiments/attempts'/a.FROZEN_SPEC['attempt_id']/'evaluation_spec.json'
        self.assertEqual(json.loads(path.read_text()),a.FROZEN_SPEC)
        self.assertEqual(a.FROZEN_SPEC['inputs']['construction_manifest']['sha256'],
            'c179958d16fbe33879ead93741339b5750923d673d39c62e17a58ac4910d38e2')


if __name__=='__main__':unittest.main()
