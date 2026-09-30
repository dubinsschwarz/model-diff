import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import torch
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load(ROOT/'scripts/ablation/construct_full_support_endpoint_hessian_forward.py','attempt017')
toy=load(ROOT/'tests/test_construct_exact_displacement_jvp_ceiling.py','toy016')


def models():
    base=toy.FamilyModel();base.model.config=base.config
    final=copy.deepcopy(base)
    with torch.no_grad():
        final.model.layers[0].self_attn['q_proj'].weight.add_(.1)
        final.model.layers[27].mlp['down_proj'].weight.sub_(.2)
    spec=copy.deepcopy(c.FROZEN_SPEC)
    spec['coordinates']=[{'name':n+'.weight','shape':[2,2]} for n,m in final.named_modules()
                         if isinstance(m,torch.nn.Linear) and n.startswith('model.layers.')]
    spec['coordinates'].sort(key=lambda r:r['name'])
    for key in ('generic_loss','hvp_loss'):
        spec[key].update(sample_count=4,sequence_length=3,predictions_per_sample=2,
                         total_prediction_tokens=8,batch_size=2,number_of_batches=2)
    return base,final,spec


def audit(base,final,spec):
    snap=c.snapshot(base,spec)
    values={n:p.detach().clone() for n,p in base.named_parameters()}
    return c.displacement(final,c.discover(final,spec),snap['parameters'],lambda n,r:values[n])


class MathTests(unittest.TestCase):
    def test_inventory_196_and_boundaries(self):
        base,final,spec=models();eligible=c.discover(final,spec)
        self.assertEqual(len(eligible),196)
        self.assertEqual(sum(c.block_number(n)<14 for n,_ in eligible),98)
        final.model.layers[27].mlp['extra']=torch.nn.Linear(2,2,bias=False)
        with self.assertRaisesRegex(ValueError,'seven'):c.discover(final,spec)

    def test_positive_displacement_and_complete_audit(self):
        base,final,spec=models();direction,record=audit(base,final,spec)
        self.assertEqual(list(direction),[r['name'] for r in spec['coordinates']])
        for n,v in direction.items():
            self.assertTrue(torch.equal(v,(final.get_parameter(n).double()-base.get_parameter(n).double()).float()))
        self.assertEqual(record['nonzero_count'],2)
        self.assertAlmostEqual(record['exact_norm']**2,record['early_norm']**2+record['later_norm']**2)
        self.assertTrue(record['outside_support_zero_verified'])

    def test_omitted_parameter_and_alias_rejected(self):
        base,final,spec=models();final.head.weight.data.add_(1)
        with self.assertRaisesRegex(ValueError,'outside full support'):audit(base,final,spec)
        base,final,spec=models();final.register_parameter('extra_alias',final.head.weight)
        with self.assertRaisesRegex(ValueError,'alias|inventory'):audit(base,final,spec)

    def test_all_buffer_and_config_semantics_fail_closed(self):
        for kind in ('global','late','config'):
            base,final,spec=models()
            if kind=='global':final.model.offset.add_(1)
            elif kind=='late':
                base.model.layers[27].register_buffer('cache',torch.ones(2))
                final.model.layers[27].register_buffer('cache',torch.zeros(2))
            else:final.config.rope_theta=123
            with self.assertRaises(ValueError):
                c.execution_audit(c.execution.execution_snapshot(base,spec),c.execution.execution_snapshot(final,spec))

    def test_true_hvp_explicit_hessian_and_positive_sign(self):
        x=torch.tensor([.3,-.7],dtype=torch.float64,requires_grad=True);v=torch.tensor([.2,.8],dtype=torch.float64)
        fn=lambda x:(x**4).sum()+x[0]*x[1]
        expected=torch.autograd.functional.hessian(fn,x)@v
        actual=c.loss_helper.hvp_backend.true_hessian_vector_product(lambda:fn(x),{'x':x},{'x':v})['x']
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)

    def test_full_hvp_batch_partition_matches_explicit_loss(self):
        base,final,spec=models();eligible=c.discover(final,spec);direction,_=audit(base,final,spec)
        tokens=torch.tensor([[0,1,2],[1,2,0],[2,0,1],[0,2,1]])
        parameters={n+'.weight':m.weight for n,m in eligible}
        direct=c.loss_helper.hvp_backend.true_hessian_vector_product(
            lambda:c.loss_helper.causal_token_loss_sum(final(input_ids=tokens,use_cache=False).logits,tokens,torch)/8,
            parameters,direction)
        seen=[];original=c.loss_helper.hvp_backend.true_hessian_vector_product
        def record(fn,p,v):seen.append(v);return original(fn,p,v)
        with patch.object(c.loss_helper.hvp_backend,'true_hessian_vector_product',side_effect=record):
            accumulated=c.loss_helper.accumulate_hessian_split(final,tokens,eligible,direction,spec['hvp_loss'],torch)
        self.assertEqual(len(seen),2);self.assertTrue(all(v is direction for v in seen))
        for n in parameters:torch.testing.assert_close(accumulated[n],direct[n],rtol=1e-5,atol=1e-8)

    def test_gradient_matches_full_mean_and_rhs_subtraction(self):
        base,final,spec=models();eligible=c.discover(final,spec);tokens=torch.tensor([[0,1,2],[1,2,0],[2,0,1],[0,2,1]])
        params=[m.weight for _,m in eligible]
        direct=torch.autograd.grad(c.loss_helper.causal_token_loss_sum(final(input_ids=tokens,use_cache=False).logits,tokens,torch)/8,params)
        with tempfile.TemporaryDirectory() as d:
            old={r['name']:torch.full(r['shape'],.12345) for r in spec['coordinates']}
            oldpath=Path(d)/'g0';c.save_vector(oldpath,old,spec['coordinates'])
            path=Path(d)/'rhs'
            _,record=c.gradient_to_file(final,eligible,tokens,spec['generic_loss'],path,spec['coordinates'],oldpath)
            with c.safe_open(str(path),framework='pt') as f:
                for (name,_),expected in zip(eligible,direct):
                    torch.testing.assert_close(f.get_tensor(name+'.weight'),(expected.double()-old[name+'.weight'].double()).float(),rtol=1e-6,atol=2e-8)
            self.assertEqual(record,c.vector_record(path,spec['coordinates']))
            self.assertTrue(all(p.grad is None for p in final.parameters()))

    def test_metrics_full_early_later_and_blocks(self):
        _,_,spec=models();h={r['name']:torch.ones(2,2) for r in spec['coordinates']}
        d={n:v*(2 if c.block_number(n)<14 else 3) for n,v in h.items()}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'rhs';c.save_vector(path,d,spec['coordinates']);m=c.metrics(h,path)
        self.assertEqual(len(m['blocks']),28)
        for key,scale in [('early_0_13',2),('later_14_27',3)]:
            self.assertAlmostEqual(m[key]['cosine'],1);self.assertEqual(m[key]['optimal_scalar'],scale)
            self.assertEqual(m[key]['relative_residual_after_optimal_rescaling'],0)
        self.assertEqual(m['full']['optimal_scalar'],2.5)
        self.assertEqual(m['blocks'][27]['optimal_scalar'],3)
        zero=c.metric_record(lambda:iter([(torch.zeros(2),torch.zeros(2))]));self.assertIsNone(zero['cosine'])

    def test_rhs_exact_rounding_serialization_hashes(self):
        _,_,spec=models();v={r['name']:torch.ones(2,2) for r in spec['coordinates']}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rhs';record=c.save_vector(p,v,spec['coordinates'])
            self.assertEqual(record['norm'],28)
            self.assertEqual(len(record['matrices']),196)
            self.assertEqual(record['matrices'][0]['raw_sha256'],c.a.sha256_raw_float32_tensor(torch.ones(2,2),torch))
            with self.assertRaisesRegex(ValueError,'exists'):c.save_vector(p,v,spec['coordinates'])
            p.write_bytes(p.read_bytes()+b'bad')
            self.assertNotEqual(c.a.sha256_file(p),record['serialized_sha256'])

    def test_frozen_spec_and_firewall(self):
        path=ROOT/'experiments/attempts'/c.FROZEN_SPEC['attempt_id']/'spec.json'
        self.assertEqual(json.loads(path.read_text()),c.FROZEN_SPEC)
        source=Path(c.__file__).read_text()
        for forbidden in ('oracle_adl.pt','evaluation.json','torch.linalg.inv','finite_difference','H0','path_average'):
            self.assertNotIn(forbidden,source)
        self.assertEqual(c.FROZEN_SPEC['hessian']['endpoint'],'W1_only')
        self.assertFalse(c.FROZEN_SPEC['hessian']['fallback'])


class PipelineTests(unittest.TestCase):
    def fixture(self,root):
        base,final,spec=models();args=c.parse_args([]);args.device='cpu'
        for key in spec['defaults']:setattr(args,key,root/key)
        args.spec_path=root/'spec.json'
        for model,path in ((base,args.base_directory),(final,args.final_directory)):
            path.mkdir();save_file({n:v.detach().contiguous() for n,v in model.state_dict().items()},str(path/'model.safetensors'))
        spec['base']['files']=c.a.checkpoint_file_records(args.base_directory)
        spec['canonical_checkpoint_files']=c.a.checkpoint_file_records(args.final_directory)
        tokens=torch.tensor([[0,1,2],[1,2,0],[2,0,1],[0,2,1]])
        return args,base,final,spec,tokens
    def run_fixture(self,f):
        args,base,final,spec,tokens=f
        def loader(path,*_):return base if path==args.base_directory else final
        with patch.object(c,'validate_inputs',return_value=(spec,tokens,{},lambda:None)),patch.object(c.a,'load_local_model',side_effect=loader),redirect_stdout(io.StringIO()):
            c.construct(args)
    def test_complete_pipeline_determinism_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));args=f[0];before=[c.a.model_state_hashes(m,torch) for m in f[1:3]]
            self.run_fixture(f)
            self.assertFalse(args.temporary_directory.exists());text=args.manifest_path.read_text()
            result=json.loads(text);self.assertEqual(result['hvp']['number_of_batches'],2)
            self.assertEqual(before,[c.a.model_state_hashes(m,torch) for m in f[1:3]])
            with self.assertRaisesRegex(ValueError,'exists'):self.run_fixture(f)
            args.artifact_path.unlink();args.manifest_path.unlink();self.run_fixture(f)
            self.assertEqual(text,args.manifest_path.read_text())
    def test_smoke_only_one_final_hvp_no_writes_no_gradients(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));args=f[0];args.smoke_only=True;calls=[]
            original=c.loss_helper.accumulate_hessian_split
            def hvp(model,tokens,*rest,**kw):
                calls.append((model,len(tokens)));return original(model,tokens,*rest,**kw)
            with patch.object(c,'gradient_to_file',side_effect=AssertionError('No gradients in smoke')),patch.object(c.loss_helper,'accumulate_hessian_split',side_effect=hvp):self.run_fixture(f)
            self.assertEqual(calls,[(f[2],2)])
            self.assertFalse(args.artifact_path.exists());self.assertFalse(args.manifest_path.exists());self.assertFalse(args.temporary_directory.exists())
    def test_stale_temp_and_failed_publication_retained(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));args=f[0];args.temporary_directory.mkdir();(args.temporary_directory/'stale').touch()
            with self.assertRaisesRegex(ValueError,'Stale'):self.run_fixture(f)
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));args=f[0]
            with patch.object(c.a,'write_manifest',side_effect=OSError('failed')),self.assertRaises(OSError):self.run_fixture(f)
            self.assertTrue(args.artifact_path.exists());self.assertTrue((args.temporary_directory/'g0.safetensors').exists())
            self.assertFalse(args.manifest_path.exists())
    def test_mutation_detected_base_and_final(self):
        for phase in ('base','final'):
            with tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));original=c.gradient_to_file
                def mutate(model,*args,**kw):
                    result=original(model,*args,**kw)
                    if model is f[1 if phase=='base' else 2]:model.model.offset.add_(1)
                    return result
                with patch.object(c,'gradient_to_file',side_effect=mutate),self.assertRaisesRegex(ValueError,'changed'):self.run_fixture(f)
                self.assertFalse(f[0].manifest_path.exists())
    def provenance_fixture(self,root):
        f=self.fixture(root);args,base,final,spec,tokens=f
        for name in c.loss_helper.TOKENIZER_FILE_NAMES:
            (args.final_directory/name).write_text('{}')
        spec['canonical_checkpoint_files']=c.a.checkpoint_file_records(args.final_directory)
        corpus=copy.deepcopy(c.loss_helper.FROZEN_CORPUS_SPEC)
        corpus['selection'].update(sample_count=4,sequence_length=3)
        args.corpus_spec_path.write_text(json.dumps(corpus));torch.save(tokens,args.tokens_path)
        records=c.loss_helper.tokenizer_file_records(args.final_directory)
        manifest={'format_version':1,'attempt_id':corpus['attempt_id'],'hash_algorithm':'sha256',
                  'corpus_spec_sha256':c.a.sha256_file(args.corpus_spec_path),'dataset':corpus['dataset'],
                  'selection':corpus['selection'],'freeze_script_sha256':'a'*64,
                  'tokenizer_checkpoint':{'source':'canonical_merged_checkpoint','directory':str(args.final_directory),
                                          'file_count':len(records),'files':records},
                  'tensor':{'shape':[4,3],'dtype':'torch.int64','device':'cpu','contiguous':True},
                  'artifact':{'path':str(args.tokens_path),'serialized_sha256':c.a.sha256_file(args.tokens_path),
                              'raw_tensor_sha256':c.a.sha256_raw_int64_tensor(tokens,torch)}}
        args.corpus_manifest_path.write_text(json.dumps(manifest))
        spec['frozen_corpus']={'corpus_spec_sha256':c.a.sha256_file(args.corpus_spec_path),
                              'corpus_manifest_sha256':c.a.sha256_file(args.corpus_manifest_path),
                              'serialized_sha256':manifest['artifact']['serialized_sha256'],
                              'raw_tensor_sha256':manifest['artifact']['raw_tensor_sha256']}
        args.spec_path.write_text(json.dumps(spec))
        return f,corpus

    def test_actual_provenance_validation_and_post_input_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            f,corpus=self.provenance_fixture(Path(d));args,_,_,spec,tokens=f
            with patch.object(c,'FROZEN_SPEC',spec),patch.object(c.loss_helper,'FROZEN_CORPUS_SPEC',corpus):
                validated,loaded,hashes,recheck=c.validate_inputs(args)
                self.assertTrue(torch.equal(tokens,loaded));recheck()
                args.corpus_manifest_path.write_text('{}')
                with self.assertRaisesRegex(ValueError,'changed'):recheck()
                with self.assertRaisesRegex(ValueError,'provenance'):c.validate_inputs(args)

    def test_source_checkpoint_and_raw_corpus_tampering_rejected(self):
        for kind in ('helper','checkpoint','raw'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                f,corpus=self.provenance_fixture(Path(d));args,_,_,spec,_=f
                if kind=='helper':spec['sources'][next(iter(spec['sources']))]='0'*64
                elif kind=='checkpoint':(args.base_directory/'model.safetensors').write_bytes(b'altered')
                else:
                    manifest=json.loads(args.corpus_manifest_path.read_text())
                    manifest['artifact']['raw_tensor_sha256']='0'*64
                    args.corpus_manifest_path.write_text(json.dumps(manifest))
                    spec['frozen_corpus']['corpus_manifest_sha256']=c.a.sha256_file(args.corpus_manifest_path)
                    spec['frozen_corpus']['raw_tensor_sha256']='0'*64
                args.spec_path.write_text(json.dumps(spec))
                with patch.object(c,'FROZEN_SPEC',spec),patch.object(c.loss_helper,'FROZEN_CORPUS_SPEC',corpus),self.assertRaises(ValueError):
                    c.validate_inputs(args)

    def test_backend_exception_propagates_no_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));f[0].smoke_only=True
            with patch.object(c.loss_helper.hvp_backend,'true_hessian_vector_product',side_effect=RuntimeError('kernel double backward unsupported')),self.assertRaisesRegex(RuntimeError,'kernel double backward unsupported'):
                self.run_fixture(f)
            self.assertFalse(f[0].artifact_path.exists());self.assertFalse(f[0].temporary_directory.exists())

    def test_input_spec_hash_validation_before_models(self):
        with tempfile.TemporaryDirectory() as d:
            args=c.parse_args([]);args.spec_path=Path(d)/'spec'
            args.spec_path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'specification'):c.validate_inputs(args)

if __name__=='__main__':unittest.main()
