import ast
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
PROJECT=Path(__file__).resolve().parents[1]
SCRIPT=PROJECT/'scripts/ablation/construct_multicorpus_raw_jg_consensus.py'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
a=module('a014',SCRIPT)
f=module('f014',SCRIPT.with_name('freeze_multicorpus_consensus_corpus.py'))
class TinyBlock(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(2))
        self.nonlinear = nonlinear

    def forward(self, hidden):
        value = self.proj(hidden)
        return torch.tanh(value) if self.nonlinear else value

class TinyTransformer(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([TinyBlock(nonlinear and i == 0) for i in range(28)])
        self.register_buffer("offset", torch.tensor([1., 2.]))
        self.bad_index = False
        self.detach_output = False
        self.calls = []

    def forward(self, *, input_ids, use_cache, output_hidden_states, return_dict):
        assert use_cache is False and output_hidden_states is True and return_dict is True
        self.calls.append(tuple(input_ids.shape))
        hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 2) * .1 + self.offset
        states = []
        for layer in self.layers:
            states.append(hidden)
            hidden = layer(hidden)
        states.append(hidden * 7)  # Distinct final normalization, never the readout.
        if self.bad_index:
            states[14] = states[13]
        if self.detach_output:
            states = [value.detach() for value in states]
        return SimpleNamespace(hidden_states=tuple(states))

class TinyModel(torch.nn.Module):
    def __init__(self, nonlinear=False):
        super().__init__()
        self.model = TinyTransformer(nonlinear)
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=28, hidden_size=2)
class ConsensusTests(unittest.TestCase):
    def responses(self):return [torch.tensor(v).repeat(128,1) for v in ([3.,0.],[0.,7.],[-2.,0.])]
    def test_exact_equal_weight_per_position_consensus(self):
        output,diag=a.response_consensus(self.responses(),torch)
        torch.testing.assert_close(output,torch.tensor([0.,1/3]).repeat(128,1),rtol=0,atol=0)
        self.assertEqual(diag['positions'][0]['response_norms'],[3,7,2])
        self.assertEqual(diag['positions'][0]['pairwise_cosine_matrix'],[[1,0,-1],[0,1,0],[-1,0,1]])
        self.assertAlmostEqual(diag['positions'][0]['consensus_concentration'],1/3)
        self.assertEqual(len(diag['positions']),128)
        self.assertEqual(diag['positions_1_4']['positions'],[1,2,3,4])
    def test_positive_scaling_invariance_and_negative_sign_preserved(self):
        values=self.responses();original,_=a.response_consensus(values,torch)
        scaled,_=a.response_consensus([v*s for v,s in zip(values,[.25,16,128])],torch)
        self.assertTrue(torch.equal(original,scaled))
        flipped,_=a.response_consensus([-values[0],*values[1:]],torch)
        self.assertFalse(torch.equal(original,flipped))
    def test_zero_nonfinite_and_wrong_corpus_count_fail(self):
        for bad in (0,float('nan'),float('inf')):
            values=self.responses();values[2][127]*=bad
            with self.assertRaisesRegex(ValueError,'norm'):a.response_consensus(values,torch)
        with self.assertRaises(ValueError):a.response_consensus(self.responses()[:2],torch)
    def test_geometry_deterministic_and_cancellation_allowed(self):
        values=self.responses();x,d=a.response_consensus(values,torch);y,e=a.response_consensus(values,torch)
        self.assertEqual(json.dumps(d),json.dumps(e));self.assertTrue(torch.equal(x,y))
        self.assertTrue(all(0<=row['consensus_concentration']<=1 for row in d['positions']))
    def test_validated008_routines_are_mechanically_identical(self):
        old=ast.parse((PROJECT/'scripts/ablation/construct_generic_gradient_linear_response.py').read_text());new=ast.parse(SCRIPT.read_text())
        names=['accumulate_mean_generic_gradient','causal_token_loss_sum','global_tangent_scale','prepare_tangents','run_readout_jvp','ordinary_hook_readout','hidden_state_output','compute_probe_response','model_state_hashes','verify_model_unchanged','checkpoint_file_records','ResponseAccumulator']
        for name in names:
            lhs=next(n for n in old.body if getattr(n,'name',None)==name);rhs=next(n for n in new.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(lhs),ast.dump(rhs),name)
    def test_analytical_jvp_hook_and_mutation_audit(self):
        model=TinyModel();spec=copy.deepcopy(a.FROZEN_SPEC);spec['readout']['hidden_size']=2;spec['probe']['sequence_length']=3
        tokens=torch.ones(2,3,dtype=torch.int64);tangent={'model.layers.0.proj.weight':torch.eye(2)}
        before=a.model_state_hashes(model,torch)
        primal,response=a.run_readout_jvp(model,tokens,tangent,spec,torch)
        self.assertTrue(torch.equal(primal,a.ordinary_hook_readout(model,tokens,spec,torch)))
        torch.testing.assert_close(response,primal)
        a.verify_model_unchanged(model,before,torch)
        with torch.no_grad():model.model.layers[2].proj.weight.add_(1)
        with self.assertRaisesRegex(ValueError,'changed'):a.verify_model_unchanged(model,before,torch)
    def test_firewall_and_exact_spec(self):
        source=SCRIPT.read_text()+SCRIPT.with_name('freeze_multicorpus_consensus_corpus.py').read_text()
        for forbidden in ('models/base','adapter','oracle_adl.pt','evaluation.json','torch.linalg.svd','hessian_vector'):
            self.assertNotIn(forbidden,source)
        folder=PROJECT/'experiments/attempts'/a.FROZEN_SPEC['attempt_id']
        self.assertEqual(a.load_spec(folder/'spec.json'),a.FROZEN_SPEC)
        self.assertEqual(a.FROZEN_SPEC['workload'],dict(gradient_batches=1536,jvp_batches=939))
        for name,spec in a.NEW_CORPUS_SPECS.items():
            self.assertEqual(json.loads((folder/f'{name}_corpus_spec.json').read_text()),spec)
            self.assertRegex(spec['dataset']['revision'],r'^[0-9a-f]{40}$')
            self.assertEqual(spec['selection']['skip_valid_examples'],0)
    def test_precommitted_evaluation_plan(self):
        plan=a.FROZEN_SPEC['evaluation_plan']
        self.assertEqual(plan['oracle_vector'],'difference')
        self.assertEqual(plan['candidate_tensors'],['response_fineweb','response_wikitext103_raw','response_tinystories','consensus_response'])
        self.assertEqual(plan['cosine_dtype'],'cpu_float64')
        self.assertEqual(plan['per_position_scope'],'all_128_positions')
        for key,scope in [('primary','1_4'),('secondary','1_127')]:
            self.assertEqual(plan[key],{'metric':f'consensus_positions_{scope}_mean_cosine','baseline':f'response_fineweb_positions_{scope}_mean_cosine','report_delta_consensus_minus_fineweb':True})
        self.assertEqual(plan['position_0'],'report_separately')
        self.assertEqual(plan['policy'],['report all individual corpus candidates','no best-corpus selection','no sign selection','no corpus reweighting','no post-result candidate modification','no post-result threshold tuning'])

    def test_overwrite_refusal_and_deterministic_json(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'out.pt';manifest=root/'m.json'
            a.require_output_absent(out,manifest);out.write_bytes(b'partial')
            with self.assertRaises(ValueError):a.require_output_absent(out,manifest)
            a.write_manifest(manifest,{'a':[1,2],'b':3})
            first=manifest.read_bytes();manifest.unlink();a.write_manifest(manifest,{'a':[1,2],'b':3});self.assertEqual(first,manifest.read_bytes())

class CorpusTests(unittest.TestCase):
    def spec(self):
        s=copy.deepcopy(a.NEW_CORPUS_SPECS['tinystories']);s['selection'].update(sample_count=2,sequence_length=3,minimum_token_count=3,character_limit=4);return s
    def dataset(self,examples):
        class Dataset:
            def shuffle(inner,**kwargs):
                self.assertEqual(kwargs,dict(seed=42,buffer_size=10000));return iter(examples)
        return Dataset()
    def test_streaming_selection_schema_and_tokenization(self):
        calls=[]
        def encode(text,**kwargs):calls.append((text,kwargs));return [1,2,3,4]
        tokenizer=SimpleNamespace(encode=encode)
        tokens=f.select_tokens(self.dataset([{'text':' '},{'text':'abcdef'},{'text':'ghijkl'}]),tokenizer,self.spec(),torch)
        self.assertEqual(tokens.tolist(),[[1,2,3],[1,2,3]])
        self.assertEqual(calls,[('abcd',{'add_special_tokens':True}),('ghij',{'add_special_tokens':True})])
        self.assertTrue(tokens.is_contiguous());self.assertEqual(tokens.dtype,torch.int64)
    def test_missing_or_wrong_schema_and_insufficient_rows(self):
        for rows in ([{'wrong':'text'}],[{'text':None}],[]):
            with self.assertRaises(ValueError):f.select_tokens(self.dataset(rows),SimpleNamespace(encode=lambda *a,**k:[1,2,3]),self.spec(),torch)
    def test_token_serialized_raw_hashes_and_spec_hash_are_strict(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tokens.pt';value=torch.ones(4096,128,dtype=torch.int64);torch.save(value,p)
            digest=a.sha256_raw_int64_tensor(value,torch)
            self.assertTrue(torch.equal(a.load_corpus_tokens(p,digest,4096,128,torch),value))
            with self.assertRaises(ValueError):a.load_corpus_tokens(p,'0'*64,4096,128,torch)
            s=Path(d)/'spec.json';s.write_text('{}')
            with self.assertRaisesRegex(ValueError,'spec hash'):a.validate_frozen_corpus(dict(name='tinystories',spec_path=str(s),spec_sha256='0'*64,manifest_path=str(s),tokens_path=str(p)),Path(d),torch)
    def test_manifest_schema_linkage_and_actual_token_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);sp=root/'spec.json';mp=root/'manifest.json';tp=root/'tokens.pt'
            spec=copy.deepcopy(a.NEW_CORPUS_SPECS['tinystories']);spec['artifact']['path']=str(tp);sp.write_text(json.dumps(spec))
            tokens=torch.ones(4096,128,dtype=torch.int64);torch.save(tokens,tp)
            m=dict(format_version=1,attempt_id=a.FROZEN_SPEC['attempt_id'],hash_algorithm='sha256',corpus_name='tinystories',spec_sha256=a.sha256_file(sp),dataset=spec['dataset'],selection=spec['selection'],tokenizer_checkpoint={'source':'canonical_merged_checkpoint','directory':str(root),'files':[]},tensor={'shape':[4096,128],'dtype':'torch.int64','device':'cpu','contiguous':True},freeze_script_sha256=a.sha256_file(SCRIPT.with_name('freeze_multicorpus_consensus_corpus.py')),helper_script_sha256=a.sha256_file(SCRIPT),artifact={'path':str(tp),'serialized_sha256':a.sha256_file(tp),'raw_tensor_sha256':a.sha256_raw_int64_tensor(tokens,torch)})
            mp.write_text(json.dumps(m));record=dict(name='tinystories',spec_path=str(sp),manifest_path=str(mp),tokens_path=str(tp),spec_sha256=a.sha256_file(sp))
            with patch.dict(a.NEW_CORPUS_SPECS,{'tinystories':spec}),patch.object(a,'tokenizer_file_records',return_value=[]):
                loaded,_,_=a.validate_frozen_corpus(record,root,torch);self.assertTrue(torch.equal(tokens,loaded))
                for key in ('freeze_script_sha256','helper_script_sha256'):
                    altered={**m,key:'0'*64};mp.write_text(json.dumps(altered))
                    with self.subTest(hash=key),self.assertRaisesRegex(ValueError,key):
                        a.validate_frozen_corpus(record,root,torch)
                for key,value in (('format_version',2),('attempt_id','wrong'),('hash_algorithm','sha1')):
                    for altered in ({**m,key:value},{k:v for k,v in m.items() if k!=key}):
                        mp.write_text(json.dumps(altered))
                        with self.subTest(field=key),self.assertRaisesRegex(ValueError,'identity/version'):
                            a.validate_frozen_corpus(record,root,torch)
                m['selection']={};mp.write_text(json.dumps(m))
                with self.assertRaisesRegex(ValueError,'linkage'):a.validate_frozen_corpus(record,root,torch)

class PipelineTests(unittest.TestCase):
    def test_three_corpora_sequential_one_model_and_deterministic_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);spec=copy.deepcopy(a.FROZEN_SPEC)
            modeldir=root/'model';modeldir.mkdir();(modeldir/'config.json').write_text('{}')
            spec['canonical_checkpoint_files']=a.checkpoint_file_records(modeldir)
            for record in spec['corpora']:
                for key in ('spec_path','manifest_path','tokens_path'):
                    path=root/(record['name']+'_'+key);path.write_text('synthetic');record[key]=str(path)
            sp=root/'spec.json';sp.write_text(json.dumps(spec));probe=root/'probe';probe.write_text('synthetic')
            args=SimpleNamespace(spec_path=sp,model_dir=modeldir,probe_path=probe,artifact_path=root/'responses.pt',manifest_path=root/'manifest.json',device='cpu')
            model=TinyModel();eligible=[('model.layers.0.proj',model.model.layers[0].proj)]
            before=a.model_state_hashes(model,torch);calls=[]
            def corpus(record,*args):
                paths=[Path(record[k]) for k in ('spec_path','manifest_path','tokens_path')]
                return torch.ones(2,3,dtype=torch.int64),{p:a.sha256_file(p) for p in paths},{'name':record['name']}
            def gradient(model,tokens,eligible,settings,t):
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                calls.append('gradient');eligible[0][1].weight.grad=torch.ones(2,2)
                return 1.
            def response(model,probe,tangents,spec,t):
                self.assertEqual(calls[-1],'gradient');calls.append('jvp')
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                self.assertEqual(len(tangents),1)
                return torch.ones(128,2048,dtype=torch.float64),torch.ones(128,2048,dtype=torch.float64),{'readout_hook_verified':True}
            with patch.object(a,'FROZEN_SPEC',spec),patch.object(a,'validate_frozen_corpus',side_effect=corpus),patch.object(a,'load_probe',return_value=torch.ones(2,3,dtype=torch.int64)),patch.object(a,'load_local_model',return_value=model) as loader,patch.object(a,'discover_eligible_linear_weights',return_value=eligible),patch.object(a,'accumulate_mean_generic_gradient',side_effect=gradient),patch.object(a,'compute_probe_response',side_effect=response):
                a.construct(args);self.assertEqual(loader.call_count,1)
                self.assertEqual(calls,['gradient','jvp']*3)
                artifact=torch.load(args.artifact_path,weights_only=True)
                self.assertEqual(list(artifact),spec['outputs']['tensors'])
                self.assertEqual(before,a.model_state_hashes(model,torch))
                manifest=json.loads(args.manifest_path.read_text())
                for name,value in artifact.items():self.assertEqual(manifest['artifact']['raw_tensors_sha256'][name],a.sha256_raw_float32_tensor(value,torch))
                text=args.manifest_path.read_bytes();args.manifest_path.unlink();args.artifact_path.unlink()
                a.construct(args);self.assertEqual(text,args.manifest_path.read_bytes())
                with self.assertRaises(ValueError):a.construct(args)

    def test_freezer_pins_revision_streaming_and_hashes_without_network(self):
        import sys
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);modeldir=root/'model';modeldir.mkdir();spec=copy.deepcopy(a.NEW_CORPUS_SPECS['tinystories'])
            spec['selection'].update(sample_count=2,sequence_length=3,minimum_token_count=3)
            spec['artifact']['path']=str(root/'tokens.pt');sp=root/'spec.json';sp.write_text(json.dumps(spec));mp=root/'manifest.json'
            class Dataset:
                def shuffle(self,**kwargs):
                    if kwargs!={'seed':42,'buffer_size':10000}:raise AssertionError(kwargs)
                    return iter([{'text':'one'},{'text':'two'}])
            calls=[]
            def load(*args,**kwargs):calls.append((args,kwargs));return Dataset()
            tokenizer=SimpleNamespace(encode=lambda *args,**kwargs:[1,2,3])
            modules={'datasets':SimpleNamespace(load_dataset=load),'transformers':SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args,**kwargs:tokenizer))}
            with patch.dict(sys.modules,modules),patch.dict(f.a.NEW_CORPUS_SPECS,{'tinystories':spec}),patch.object(f.a,'tokenizer_file_records',return_value=[]),patch.dict(f.a.FROZEN_SPEC,{'canonical_checkpoint_files':[]}),patch.object(f,'version',return_value='synthetic'):
                f.freeze('tinystories',modeldir,sp,mp)
                self.assertEqual(calls[0][1],dict(name=None,revision='f54c09fd23315a6f9c86f9dc80f725de7d8f9c64',split='train',streaming=True))
                manifest=json.loads(mp.read_text());tp=Path(spec['artifact']['path'])
                self.assertEqual(manifest['format_version'],1)
                self.assertEqual(manifest['attempt_id'],a.FROZEN_SPEC['attempt_id'])
                self.assertEqual(manifest['hash_algorithm'],'sha256')
                self.assertEqual(manifest['freeze_script_sha256'],a.sha256_file(SCRIPT.with_name('freeze_multicorpus_consensus_corpus.py')))
                self.assertEqual(manifest['helper_script_sha256'],a.sha256_file(SCRIPT))
                self.assertEqual(manifest['artifact']['serialized_sha256'],a.sha256_file(tp))
                self.assertEqual(manifest['artifact']['raw_tensor_sha256'],a.sha256_raw_int64_tensor(torch.load(tp,weights_only=True),torch))
                with self.assertRaises(ValueError):f.freeze('tinystories',modeldir,sp,mp)

if __name__=='__main__':unittest.main()
