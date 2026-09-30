import ast
import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
PROJECT=Path(__file__).resolve().parents[1]
SCRIPT=PROJECT/'scripts/ablation/evaluate_attempt014.py'
S=importlib.util.spec_from_file_location('eval014',SCRIPT);a=importlib.util.module_from_spec(S);S.loader.exec_module(a)


class EvaluationTests(unittest.TestCase):
    def fixture(self,root):
        spec=copy.deepcopy(a.FROZEN_SPEC);paths={name:root/(name+'.json') for name in spec['inputs']}
        def put(name,data):paths[name].write_text(json.dumps(data));return a.sha256_file(paths[name])
        for name in ('constructor','merge_script','freeze_probe_script','oracle_script'):put(name,{'synthetic':name})
        identities={'base':{'repo_id':'synthetic/base','revision':'a'*40},'adapter':{'repo_id':'synthetic/final','revision':'b'*40}}
        dh=put('downloaded_manifest',{'hash_algorithm':'sha256','models':identities})
        mh=put('merged_manifest',{'hash_algorithm':'sha256','downloaded_model_hashes_sha256':dh,'base':identities['base'],'adapter':identities['adapter'],
            'merge_script_sha256':a.sha256_file(paths['merge_script']),'merge_metadata':{'dtype':'float32','safe_merge':True,'tokenizer_source':'base'},'files':[]})
        probe={'serialized_sha256':'a'*64,'raw_tensor_sha256':'b'*64}
        ph=put('probe_manifest',{'downloaded_model_hashes_sha256':dh,'freeze_oracle_probe_script_sha256':a.sha256_file(paths['freeze_probe_script']),
            'fineweb_tokens_sha256':probe['serialized_sha256'],'raw_tensor_sha256':probe['raw_tensor_sha256']})
        attempt={'attempt_id':spec['attempt_id'],'evaluation_plan':spec['evaluation_plan'],'consensus':spec['consensus_definition'],
            'probe':probe,'generic_loss':{},'jvp':{},'readout':{},'canonical_checkpoint_files':[],
            'outputs':{'tensors':['merged_mean',*spec['evaluation_plan']['candidate_tensors']]}}
        sh=put('attempt_spec',attempt)
        oracle_difference=torch.zeros(128,2048);oracle_difference[:,0]=1
        artifact={'merged_mean':torch.ones(128,2048)}
        for i,name in enumerate(spec['evaluation_plan']['candidate_tensors']):
            value=torch.zeros(128,2048)
            for p in range(128):
                cosine=((p+i)%9-4)/4;value[p,0]=cosine;value[p,1]=math.sqrt(1-cosine*cosine)
            artifact[name]=value
        cp=root/'candidate.pt';torch.save(artifact,cp)
        raw=lambda v:a.sha256_raw_float32_tensor(v,torch)
        manifest={'attempt_id':spec['attempt_id'],'spec_sha256':sh,'constructor_script_sha256':a.sha256_file(paths['constructor']),
            'consensus_definition':spec['consensus_definition'],'source_checkpoint':[],'corpus_order':['fineweb','wikitext103_raw','tinystories'],
            **{k:attempt[k] for k in ('probe','generic_loss','jvp','readout')},
            'artifact':{'serialized_sha256':a.sha256_file(cp),'raw_tensors_sha256':{k:raw(v) for k,v in artifact.items()}}}
        put('construction_manifest',manifest)
        method={'relative_layer':.5,'layer_index':13,'num_layers':28,'sample_count':10000,'sequence_length':128,'hidden_size':2048,'batch_size':32,'model_dtype':'float32','accumulator_dtype':'float64','stored_dtype':'float32'}
        oracle={'format_version':1,'metadata':method,'difference':oracle_difference,'ft_mean':torch.ones(128,2048),'base_mean':torch.ones(128,2048)-oracle_difference}
        op=root/'synthetic-oracle.pt';torch.save(oracle,op)
        om={'hash_algorithm':'sha256','downloaded_model_hashes_sha256':dh,'merged_model_hashes_sha256':mh,'oracle_probe_manifest_sha256':ph,
            'compute_oracle_adl_script_sha256':a.sha256_file(paths['oracle_script']),'base':identities['base'],'fine_tuned':identities['adapter'],'probe':probe,**method,
            'oracle_adl_sha256':a.sha256_file(op),'raw_tensors_sha256':{k:raw(oracle[k]) for k in a.ORACLE_VECTOR_TYPES}}
        put('oracle_manifest',om)
        for name,path in paths.items():spec['inputs'][name]['sha256']=a.sha256_file(path)
        sp=root/'evaluation_spec.json';sp.write_text(json.dumps(spec))
        return spec,sp,paths,cp,op,root/'output.json',artifact,oracle

    def run_fixture(self,f):
        spec,sp,paths,cp,op,out,_,_=f
        with patch.object(a,'FROZEN_SPEC',spec):return a.evaluate(sp,out,paths,cp,op)

    def repin(self,f,name):
        f[0]['inputs'][name]['sha256']=a.sha256_file(f[2][name]);f[1].write_text(json.dumps(f[0]))

    def test_float64_position_cosines_and_all_precommitted_means_deltas(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));r=self.run_fixture(f)
            self.assertEqual([c['candidate'] for c in r['candidates']],f[0]['evaluation_plan']['candidate_tensors'])
            for c in r['candidates']:
                x=f[6][c['candidate']].double();y=f[7]['difference'].double()
                expected=[float(torch.dot(x[p],y[p])/(x[p].norm()*y[p].norm())) for p in range(128)]
                self.assertEqual([v['position'] for v in c['positions']],list(range(128)))
                self.assertEqual([v['cosine_similarity'] for v in c['positions']],expected)
                self.assertEqual(c['position_0_cosine'],expected[0])
                self.assertEqual(c['positions_1_4_mean_cosine'],math.fsum(expected[1:5])/4)
                self.assertEqual(c['positions_1_127_mean_cosine'],math.fsum(expected[1:])/127)
            for key,metric in [('primary','positions_1_4_mean_cosine'),('secondary','positions_1_127_mean_cosine')]:
                self.assertEqual(r[key]['delta_consensus_minus_fineweb'],r['candidates'][3][metric]-r['candidates'][0][metric])
            self.assertEqual(r['candidates'][0]['position_0_cosine'],-1.)
            self.assertEqual(set(r),{'format_version','attempt_id','evaluation_plan','provenance','candidates','primary','secondary'})

    def test_deterministic_json_and_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));self.run_fixture(f);text=f[5].read_bytes()
            with self.assertRaisesRegex(ValueError,'already exists'):self.run_fixture(f)
            f[5].unlink();self.run_fixture(f);self.assertEqual(text,f[5].read_bytes())

    def test_zero_vectors_propagate_null_without_subset_selection(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));f[6]['consensus_response'][4].zero_()
            r=a.summarize_candidates(f[6],f[7],f[0]);self.assertIsNone(r['primary']['consensus']);self.assertIsNone(r['primary']['delta_consensus_minus_fineweb'])
            self.assertIsNone(r['secondary']['consensus']);self.assertIsNotNone(r['primary']['fineweb'])
            f[7]['difference'][0].zero_();r=a.summarize_candidates(f[6],f[7],f[0]);self.assertTrue(all(c['position_0_cosine'] is None for c in r['candidates']))

    def test_missing_extra_wrong_dtype_shape_nonfinite_candidate_rejected(self):
        for mode in ('missing','extra','dtype','shape','nonfinite'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));artifact=f[6];key='response_tinystories'
                if mode=='missing':del artifact[key]
                if mode=='extra':artifact['extra']=artifact[key]
                if mode=='dtype':artifact[key]=artifact[key].double()
                if mode=='shape':artifact[key]=artifact[key][:1]
                if mode=='nonfinite':artifact[key][0,0]=float('nan')
                torch.save(artifact,f[3]);m=json.loads(f[2]['construction_manifest'].read_text());m['artifact']['serialized_sha256']=a.sha256_file(f[3])
                with self.assertRaises(ValueError):a.load_candidates(f[3],m,f[0])

    def test_artifact_and_every_raw_hash_are_strict(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));m=json.loads(f[2]['construction_manifest'].read_text())
            for key in f[6]:
                bad=copy.deepcopy(m);bad['artifact']['raw_tensors_sha256'][key]='0'*64
                with self.subTest(tensor=key),self.assertRaisesRegex(ValueError,'SHA-256'):a.load_candidates(f[3],bad,f[0])
            f[3].write_bytes(f[3].read_bytes()+b'altered')
            with self.assertRaisesRegex(ValueError,'SHA256'):self.run_fixture(f)

    def test_spec_constructor_manifest_and_oracle_hashes_are_pinned(self):
        for key in ('attempt_spec','constructor','construction_manifest','oracle_manifest','probe_manifest','oracle_script'):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));p=f[2][key];p.write_text(p.read_text()+' ')
                with self.assertRaisesRegex(ValueError,'SHA256'):self.run_fixture(f)

    def test_frozen_plan_and_consensus_definition_rejection(self):
        for key in ('evaluation_plan','consensus'):
            with tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));path=f[2]['attempt_spec'];data=json.loads(path.read_text());data[key]={};path.write_text(json.dumps(data));self.repin(f,'attempt_spec')
                with self.assertRaises(ValueError):self.run_fixture(f)
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));f[0]['evaluation_plan']['primary']['baseline']='changed';f[1].write_text(json.dumps(f[0]))
            with self.assertRaisesRegex(ValueError,'frozen plan'):a.evaluate(f[1],f[5],f[2],f[3],f[4])

    def test_oracle_link_and_method_checks_match008_expectations(self):
        for key,value in [('oracle_probe_manifest_sha256','0'*64),('compute_oracle_adl_script_sha256','0'*64),('layer_index',12),('stored_dtype','float64'),('sample_count',9999)]:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));p=f[2]['oracle_manifest'];m=json.loads(p.read_text());m[key]=value;p.write_text(json.dumps(m));self.repin(f,'oracle_manifest')
                with self.assertRaises(ValueError):self.run_fixture(f)

    def test_oracle_artifact_serialized_and_raw_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));f[4].write_bytes(f[4].read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'SHA256'):self.run_fixture(f)
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));f[7]['difference'][0,0]+=1;torch.save(f[7],f[4])
            p=f[2]['oracle_manifest'];m=json.loads(p.read_text());m['oracle_adl_sha256']=a.sha256_file(f[4]);p.write_text(json.dumps(m));self.repin(f,'oracle_manifest')
            with self.assertRaisesRegex(ValueError,'SHA-256'):self.run_fixture(f)

    def test_reused008_artifact_validation_is_mechanically_identical(self):
        old=ast.parse((PROJECT/'scripts/ablation/evaluate_attempt008.py').read_text());new=ast.parse(SCRIPT.read_text())
        for name in ('validate_activation_tensor','load_torch_artifact','load_and_validate_oracle','sha256_raw_float32_tensor'):
            lhs=next(n for n in old.body if getattr(n,'name',None)==name);rhs=next(n for n in new.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(lhs),ast.dump(rhs))

    def test_only_synthetic_allowed_inputs_cpu_loading_no_model_or_prior_results(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));allowed={f[1],*f[2].values(),f[3],f[4],f[5],SCRIPT};original=Path.open
            def guarded(path,*args,**kwargs):self.assertIn(path,allowed);return original(path,*args,**kwargs)
            with patch.object(Path,'open',guarded),patch.object(torch,'load',wraps=torch.load) as load:self.run_fixture(f)
            self.assertEqual(load.call_count,2)
            self.assertTrue(all(c.kwargs=={'map_location':'cpu','weights_only':True} for c in load.call_args_list))
        source=SCRIPT.read_text()
        for forbidden in ('import transformers','models/base','oracle-logit-lens.json','argmax','argsort','linalg.svd'):
            self.assertNotIn(forbidden,source)

    def test_mutation_during_geometry_aborts_publication(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));original=a.summarize_candidates
            def mutate(*args):
                result=original(*args);f[3].write_bytes(b'changed');return result
            with patch.object(a,'summarize_candidates',side_effect=mutate),self.assertRaisesRegex(ValueError,'SHA256'):self.run_fixture(f)
            self.assertFalse(f[5].exists())

    def test_embedded_spec_and_precommitted_plan_match(self):
        folder=PROJECT/'experiments/attempts'/a.FROZEN_SPEC['attempt_id']
        self.assertEqual(json.loads((folder/'evaluation_spec.json').read_text()),a.FROZEN_SPEC)
        self.assertEqual(json.loads((folder/'spec.json').read_text())['evaluation_plan'],a.FROZEN_SPEC['evaluation_plan'])

if __name__=='__main__':unittest.main()
