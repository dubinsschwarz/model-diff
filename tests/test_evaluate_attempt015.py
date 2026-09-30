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
import test_construct_known_base_jg0_subtraction as construction_tests
import test_evaluate_attempt014 as oracle_tests
PROJECT=Path(__file__).resolve().parents[1]
SCRIPT=PROJECT/'scripts/ablation/evaluate_attempt015.py'
S=importlib.util.spec_from_file_location('evaluate015',SCRIPT);v=importlib.util.module_from_spec(S);S.loader.exec_module(v)


class EvaluatorTests(unittest.TestCase):
    def candidates(self):
        result={}
        for i,name in enumerate(v.FROZEN_SPEC['evaluation_plan']['candidate_tensors']):
            value=torch.zeros(128,2)
            for p in range(128):
                x=((p+i)%9-4)/4;value[p]=torch.tensor([x,math.sqrt(1-x*x)])
            result[name]=value
        return result
    def test_all_twelve_candidates_exact_means_and_deltas(self):
        candidates=self.candidates();oracle={'difference':torch.tensor([1.,0.]).repeat(128,1)}
        result=v.summaries(candidates,oracle,v.FROZEN_SPEC)
        self.assertEqual([r['candidate'] for r in result['candidates']],v.FROZEN_SPEC['evaluation_plan']['candidate_tensors'])
        self.assertEqual(len(result['candidates']),12)
        by_name={}
        for row in result['candidates']:
            x=candidates[row['candidate']].double();expected=(x[:,0]/x.norm(dim=1)).tolist()
            self.assertEqual([r['position'] for r in row['positions']],list(range(128)))
            self.assertEqual([r['cosine_similarity'] for r in row['positions']],expected)
            self.assertEqual(row['position_0_cosine'],expected[0])
            self.assertEqual(row['positions_1_4_mean_cosine'],math.fsum(expected[1:5])/4)
            self.assertEqual(row['positions_1_127_mean_cosine'],math.fsum(expected[1:])/127)
            by_name[row['candidate']]=row
        for label,key in [('primary','positions_1_4_mean_cosine'),('secondary','positions_1_127_mean_cosine')]:
            self.assertEqual(result[label]['delta_residual_minus_raw'],by_name['residual_consensus_response'][key]-by_name['consensus_response'][key])
            for name in v.c.FROZEN_SPEC['corpus_order']:
                self.assertEqual(result[label]['per_corpus'][name]['delta_residual_minus_raw'],by_name[f'residual_response_{name}'][key]-by_name[f'response_{name}'][key])
        self.assertEqual(set(result),{'candidates','primary','secondary'})
        self.assertEqual(result['candidates'][0]['position_0_cosine'],-1)
    def test_zero_cosines_null_means_no_selection(self):
        candidates=self.candidates();candidates['residual_consensus_response'][2].zero_()
        result=v.summaries(candidates,{'difference':torch.ones(128,2)},v.FROZEN_SPEC)
        self.assertIsNone(result['primary']['residual']);self.assertIsNone(result['primary']['delta_residual_minus_raw'])
        self.assertEqual(len(result['candidates']),12)
    def test_no_rank_sign_or_weight_selection_and_spec_linkage(self):
        source=SCRIPT.read_text()
        for name in ('argmax','argsort','linalg.svd','fit(','softmax('):self.assertNotIn(name,source)
        folder=PROJECT/'experiments/attempts'/v.FROZEN_SPEC['attempt_id']
        self.assertEqual(json.loads((folder/'evaluation_spec.json').read_text()),v.FROZEN_SPEC)
        self.assertEqual(json.loads((folder/'spec.json').read_text())['evaluation_plan'],v.FROZEN_SPEC['evaluation_plan'])
        self.assertEqual(v.c.a.sha256_file(v.c.resolve(v.FROZEN_SPEC['sources']['constructor']['path'])),v.FROZEN_SPEC['sources']['constructor']['sha256'])

    def fixture(self,root):
        build=root/'build';build.mkdir();ct=construction_tests.PipelineTests()
        cf=ct.fixture(build);ct.run_fixture(cf)
        args,cs,previous,prior,raw,tokens,base,final=cf
        refs=root/'refs';refs.mkdir();of=oracle_tests.EvaluationTests().fixture(refs)
        oldspec,oldsp,paths,cp,op,oldout,artifact,oracle=of
        es=copy.deepcopy(v.FROZEN_SPEC)
        es['sources']['attempt_spec']={'path':str(args.spec_path),'sha256':v.c.a.sha256_file(args.spec_path)}
        es['artifact_path']=str(args.artifact_path)
        es['construction_manifest_path']=str(args.manifest_path)
        es['oracle_artifact']=str(op)
        cs['defaults']['raw_artifact_path']=str(args.raw_artifact_path)
        args.spec_path.write_text(json.dumps(cs));es['sources']['attempt_spec']['sha256']=v.c.a.sha256_file(args.spec_path)
        manifest=json.loads(args.manifest_path.read_text());manifest['spec_sha256']=es['sources']['attempt_spec']['sha256'];args.manifest_path.write_text(json.dumps(manifest))
        downloaded=json.loads(paths['downloaded_manifest'].read_text());downloaded['models']['base']={k:cs['base'][k] for k in ('repo_id','revision')};paths['downloaded_manifest'].write_text(json.dumps(downloaded));dh=v.c.a.sha256_file(paths['downloaded_manifest'])
        merged=json.loads(paths['merged_manifest'].read_text());merged['base']=downloaded['models']['base'];merged['files']=cs['canonical_checkpoint_files'];merged['downloaded_model_hashes_sha256']=dh;paths['merged_manifest'].write_text(json.dumps(merged))
        pm=json.loads(paths['probe_manifest'].read_text());pm['downloaded_model_hashes_sha256']=dh;pm['fineweb_tokens_sha256']=cs['probe']['serialized_sha256'];pm['raw_tensor_sha256']=cs['probe']['raw_tensor_sha256'];paths['probe_manifest'].write_text(json.dumps(pm))
        om=json.loads(paths['oracle_manifest'].read_text());om['base']=downloaded['models']['base'];om['downloaded_model_hashes_sha256']=dh;om['merged_model_hashes_sha256']=v.c.a.sha256_file(paths['merged_manifest']);om['oracle_probe_manifest_sha256']=v.c.a.sha256_file(paths['probe_manifest']);om['probe']={k:cs['probe'][k] for k in ('serialized_sha256','raw_tensor_sha256')}
        for name in v.e.ORACLE_VECTOR_TYPES:oracle[name]=oracle[name][:,:2].contiguous()
        torch.save(oracle,op);om['oracle_adl_sha256']=v.c.a.sha256_file(op);om['raw_tensors_sha256']={n:v.c.a.sha256_raw_float32_tensor(oracle[n],torch) for n in v.e.ORACLE_VECTOR_TYPES};paths['oracle_manifest'].write_text(json.dumps(om))
        for name in es['oracle_provenance']:es['oracle_provenance'][name]={'path':str(paths[name]),'sha256':v.c.a.sha256_file(paths[name])}
        esp=root/'evaluation_spec.json';esp.write_text(json.dumps(es))
        evalargs=SimpleNamespace(evaluation_spec=esp,construction_manifest=args.manifest_path,construction_manifest_sha256=v.c.a.sha256_file(args.manifest_path),output=root/'output.json')
        return evalargs,es,cs,previous,args,op

    def run_fixture(self,f):
        args,es,cs,previous,_,_=f
        with patch.object(v,'FROZEN_SPEC',es),patch.object(v.c,'FROZEN_SPEC',cs),patch.object(v.c.a,'FROZEN_SPEC',previous),patch.object(v.e,'EXPECTED_SHAPE',(128,2)):
            return v.evaluate(args)

    def test_complete_synthetic_evaluation_deterministic_and_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));r=self.run_fixture(f);self.assertEqual(len(r['candidates']),12)
            text=f[0].output.read_bytes();f[0].output.unlink();self.run_fixture(f);self.assertEqual(text,f[0].output.read_bytes())
            with self.assertRaisesRegex(ValueError,'already exists'):self.run_fixture(f)
    def test_manifest_hash_and_artifact_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));actual=f[0].construction_manifest_sha256;f[0].construction_manifest_sha256='0'*64
            with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_fixture(f)
            f[0].construction_manifest_sha256=actual;f[4].artifact_path.write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_fixture(f)
    def test_synthetic_oracle_only_and_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));original=torch.load;seen=[]
            def guarded(path,*args,**kwargs):
                self.assertTrue(Path(path).is_relative_to(Path(d)));self.assertEqual(kwargs.get('map_location'),'cpu');seen.append(path)
                return original(path,*args,**kwargs)
            with patch.object(torch,'load',side_effect=guarded):self.run_fixture(f)
            self.assertEqual(len(seen),3);f[0].output.unlink();f[5].write_bytes(b'altered')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):self.run_fixture(f)

if __name__=='__main__':unittest.main()
