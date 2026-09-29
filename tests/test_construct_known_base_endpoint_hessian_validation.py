import ast
import copy
import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

PROJECT=Path(__file__).resolve().parents[1]
SCRIPT=PROJECT/'scripts/ablation/construct_known_base_endpoint_hessian_validation.py'
MODULE=importlib.util.spec_from_file_location('endpoint013',SCRIPT)
a=importlib.util.module_from_spec(MODULE);MODULE.loader.exec_module(a)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights=torch.nn.ModuleList([torch.nn.Linear(2,2,bias=False) for _ in range(7)])
        with torch.no_grad():
            for i,p in enumerate(self.parameters()):p.copy_(torch.tensor([[.04,.01],[-.02,.03]])*(i+1))
    def forward(self,x):return x+sum(layer(x) for layer in self.weights)*.01


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=torch.nn.Module()
        self.model.layers=torch.nn.ModuleList([Block() for _ in range(28)])
        self.head=torch.nn.Linear(2,3,bias=False)
        with torch.no_grad():self.head.weight.copy_(torch.tensor([[.2,-.1],[.1,.3],[-.2,.5]]))
        self.config=SimpleNamespace(model_type='qwen3',num_hidden_layers=28)
    def forward(self,*,input_ids,use_cache):
        assert use_cache is False
        x=torch.stack((input_ids.float()*.1+1,input_ids.float()*.2-1),-1)
        for layer in self.model.layers:x=layer(x)
        return SimpleNamespace(logits=self.head(x))


def models():
    base=Model();final=copy.deepcopy(base)
    with torch.no_grad():
        final.model.layers[0].weights[0].weight.add_(.125)
        final.model.layers[14].weights[0].weight.add_(.25)
        final.head.weight.add_(.5)
    return final,base


def loss_settings(count=8):
    return dict(sample_count=count,sequence_length=3,batch_size=1,predictions_per_sample=2,
                total_prediction_tokens=count*2,number_of_batches=count)


class EndpointTests(unittest.TestCase):
    def test_displacement_sign_categories_and_float64_norms(self):
        final,base=models();eligible=a.discover_eligible_linear_weights(final,torch)
        audit,delta=a.displacement_audit(final,base,eligible,torch)
        self.assertEqual(len(delta),98)
        expected=final.model.layers[0].weights[0].weight.detach().double()-base.model.layers[0].weights[0].weight.detach().double()
        self.assertTrue(torch.equal(delta['model.layers.0.weights.0.weight'],expected.float()))
        self.assertAlmostEqual(audit['selected']['norm'],float(expected.norm()))
        self.assertEqual(audit['linear_weights_blocks_14_27']['nonzero_names'],['model.layers.14.weights.0.weight'])
        self.assertEqual(audit['all_other_parameters']['nonzero_names'],['head.weight'])
        self.assertAlmostEqual(audit['all_other_parameters']['norm'],math.sqrt(6)*.5,places=6)
        self.assertEqual(audit['selected']['nonzero_count'],1)

    def test_hybrid_exact_selected_replacement_and_finally_restoration(self):
        final,base=models();eligible=a.discover_eligible_linear_weights(final,torch)
        before=a.model_state_hashes(final,torch);base_before=a.model_state_hashes(base,torch)
        selected={f'{n}.weight' for n,_ in eligible}
        for fail in (False,True):
            try:
                with a.selected_base_hybrid(final,base,eligible,torch):
                    actual=a.model_state_hashes(final,torch)
                    for key,value in actual.items():
                        expected=base_before[key] if key.removeprefix('parameter:') in selected else before[key]
                        self.assertEqual(value,expected)
                    if fail:raise RuntimeError('injected')
            except RuntimeError as e:self.assertEqual(str(e),'injected')
            self.assertEqual(before,a.model_state_hashes(final,torch))
            self.assertEqual(base_before,a.model_state_hashes(base,torch))

    def test_exact005_gradient_implementation_and_full_denominator(self):
        old=ast.parse((PROJECT/'scripts/ablation/construct_generic_gradient_rollback.py').read_text())
        new=ast.parse(SCRIPT.read_text())
        for name in ('causal_token_loss_sum','accumulate_mean_generic_gradient'):
            lhs=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
            rhs=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(lhs),ast.dump(rhs))
        s=a.FROZEN_SPEC['generic_loss'];self.assertEqual(s['total_prediction_tokens'],4096*127)

    def test_true_hvp_against_explicit_hessian(self):
        p=torch.nn.Parameter(torch.tensor([.4,-.8,.2],dtype=torch.float64));v=torch.tensor([2.,-1.,.5],dtype=torch.float64)
        def f(x):return (x**4).sum()+x[0]*x[1]+torch.sin(x[2])
        actual=a.hvp_backend.true_hessian_vector_product(lambda:f(p),{'p':p},{'p':v})['p']
        expected=torch.autograd.functional.hessian(f,p)@v
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        with self.assertRaises(ValueError):a.hvp_backend.true_hessian_vector_product(lambda:f(p),{'p':p},{'p':v[:2]})

    def test_splits_identical_displacement_not_gradient_and_full_mean(self):
        final,base=models();eligible=a.discover_eligible_linear_weights(final,torch);a.freeze_other_parameters(final,eligible)
        _,delta=a.displacement_audit(final,base,eligible,torch)
        tokens=torch.arange(24).reshape(8,3)%3
        observed=[];original=a.accumulate_hessian_split
        def record(model,subset,selected,direction,settings,t,progress):
            self.assertIs(direction,delta);self.assertEqual(settings['total_prediction_tokens'],4)
            observed.append(subset.clone())
            return original(model,subset,selected,direction,settings,t,progress)
        with tempfile.TemporaryDirectory() as d,patch.object(a,'accumulate_hessian_split',side_effect=record),redirect_stdout(io.StringIO()):
            a.split_actions(final,eligible,tokens,delta,loss_settings(2),Path(d),torch)
            from safetensors.torch import load_file
            actions=[load_file(str(Path(d)/f'h{i}.safetensors')) for i in range(4)]
            whole=original(final,tokens,eligible,delta,loss_settings(),torch)
            for name in whole:
                torch.testing.assert_close(sum(act[name].double() for act in actions)/4,whole[name].double(),rtol=2e-5,atol=1e-8)
        for i,subset in enumerate(observed):self.assertTrue(torch.equal(subset,tokens[i::4]))

    def test_matrixwise_metrics_signs_projection_and_stationarity(self):
        tensors={'g0':torch.tensor([1.,2.,1.,0.]),'g1':torch.tensor([4.,5.,3.,2.]),'gH':torch.tensor([2.,1.,2.,1.])}
        for i in range(4):tensors[f'h{i}']=torch.tensor([2.,4.,1.,1.])*(1+i*.1)
        with tempfile.TemporaryDirectory() as d:
            for key,value in tensors.items():save_file({'a':value[:2].reshape(1,2),'b':value[2:].reshape(1,2)},str(Path(d)/f'{key}.safetensors'))
            result=a.vector_diagnostics(Path(d),[{'name':n,'shape':[1,2]} for n in ('a','b')],torch)
        g0,g1,gH=[tensors[k].double() for k in ('g0','g1','gH')]
        h=sum(tensors[f'h{i}'].double() for i in range(4))/4;local=g1-gH;true=g1-g0
        cos=lambda x,y:float(torch.dot(x,y)/(x.norm()*y.norm()))
        self.assertAlmostEqual(result['primary']['cosine'],cos(h,local))
        self.assertAlmostEqual(result['historical']['h_vs_d_true_cosine'],cos(h,true))
        self.assertAlmostEqual(result['historical']['d_local_vs_d_true_cosine'],cos(local,true))
        scale=float(torch.dot(h,local)/h.square().sum())
        self.assertAlmostEqual(result['primary']['optimal_scalar_projection_coefficient'],scale)
        self.assertAlmostEqual(result['primary']['relative_residual_after_optimal_rescaling'],float((local-scale*h).norm()/local.norm()))
        self.assertAlmostEqual(result['historical']['relative_unexplained_norm'],float((true-local).norm()/true.norm()))
        self.assertAlmostEqual(result['stationarity']['g0_to_g1_norm_ratio'],float(g0.norm()/g1.norm()))

    def test_zero_diagnostic_policy(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('g0','g1','gH','h0','h1','h2','h3'):save_file({'a':torch.zeros(1,2)},str(Path(d)/f'{name}.safetensors'))
            result=a.vector_diagnostics(Path(d),[{'name':'a','shape':[1,2]}],torch)
            self.assertIsNone(result['primary']['cosine']);self.assertIsNone(result['primary']['optimal_scalar_projection_coefficient'])
            json.dumps(result,allow_nan=False)

    def test_firewall_spec_and_workload(self):
        source=SCRIPT.read_text()
        for forbidden in ('oracle_adl','evaluation.json','krylov_probe.pt','inverse_response.pt','adapter','linalg.inv','finite_difference'):
            self.assertNotIn(forbidden,source)
        self.assertIn('true_hessian_vector_product',source)
        path=PROJECT/'experiments/attempts'/a.FROZEN_SPEC['attempt_id']/'spec.json'
        self.assertEqual(a.load_spec(path),a.FROZEN_SPEC)
        self.assertEqual(a.FROZEN_SPEC['base']['revision'],'0060bc56d46589041c1048efd1a397421b1142b5')
        self.assertEqual(a.FROZEN_SPEC['workload'],dict(ordinary_gradient_batches=1536,hvp_batches=512,jvp_batches=0))


class PipelineTests(unittest.TestCase):
    def fixture(self,root):
        args=SimpleNamespace(attempt_spec_path=root/'spec.json',merged_model_dir=root/'final',base_model_dir=root/'base',
            corpus_spec_path=root/'corpus-spec.json',corpus_manifest_path=root/'corpus-manifest.json',tokens_path=root/'tokens.pt',
            temporary_directory=root/'tmp',diagnostic_path=root/'diagnostic.json',device='cpu')
        for path in (args.merged_model_dir,args.base_model_dir):path.mkdir();(path/'config.json').write_text('{}')
        args.corpus_spec_path.write_text('{}');args.corpus_manifest_path.write_text('{}')
        tokens=torch.arange(24).reshape(8,3)%3;torch.save(tokens,args.tokens_path)
        spec=copy.deepcopy(a.FROZEN_SPEC)
        spec['canonical_checkpoint_files']=a.checkpoint_file_records(args.merged_model_dir)
        spec['base']['files']=a.checkpoint_file_records(args.base_model_dir)
        spec['generic_loss']=loss_settings();spec['splits'].update(sample_count=2,total_prediction_tokens=4,batch_size=1,number_of_batches=2)
        spec['frozen_corpus'].update(corpus_spec_sha256=a.sha256_file(args.corpus_spec_path),corpus_manifest_sha256=a.sha256_file(args.corpus_manifest_path),
            serialized_sha256=a.sha256_file(args.tokens_path),raw_tensor_sha256=a.sha256_raw_int64_tensor(tokens,torch))
        args.attempt_spec_path.write_text(json.dumps(spec))
        return args,spec,tokens

    def run_fixture(self,args,spec,final,base):
        corpus={k:spec['frozen_corpus'][k] for k in ('serialized_sha256','raw_tensor_sha256')}
        with patch.object(a,'FROZEN_SPEC',spec),patch.object(a,'load_corpus_spec',return_value={}), \
             patch.object(a,'verify_corpus_manifest',return_value=corpus), \
             patch.object(a,'load_local_model',side_effect=[final,base]),redirect_stdout(io.StringIO()):a.construct(args)

    def test_full_synthetic_pipeline_semantics_immutability_cleanup_and_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            args,spec,tokens=self.fixture(Path(d));final,base=models()
            before=[a.model_state_hashes(m,torch) for m in (final,base)]
            calls=[];original=a.save_gradient
            def record(model,eligible,tokens,settings,path,t):
                calls.append((path.stem,copy.deepcopy(settings)))
                return original(model,eligible,tokens,settings,path,t)
            with patch.object(a,'save_gradient',side_effect=record):self.run_fixture(args,spec,final,base)
            self.assertEqual([name for name,_ in calls],['g1','gH','g0'])
            self.assertTrue(all(settings==spec['generic_loss'] for _,settings in calls))
            self.assertEqual(before,[a.model_state_hashes(m,torch) for m in (final,base)])
            self.assertFalse(args.temporary_directory.exists())
            result=json.loads(args.diagnostic_path.read_text())
            self.assertEqual(result['metrics']['primary']['definition'],'cosine(h, g1 - gH)')
            self.assertEqual(result['displacement_audit']['selected']['nonzero_count'],1)
            text=args.diagnostic_path.read_text();args.diagnostic_path.unlink()
            self.run_fixture(args,spec,final,base);self.assertEqual(text,args.diagnostic_path.read_text())
            with self.assertRaisesRegex(ValueError,'already exists'):self.run_fixture(args,spec,final,base)

    def test_publication_failure_retains_seven_vectors_and_refuses_stale_state(self):
        with tempfile.TemporaryDirectory() as d:
            args,spec,_=self.fixture(Path(d));final,base=models()
            with patch.object(a,'write_manifest',side_effect=OSError('publication failed')):
                with self.assertRaises(OSError):self.run_fixture(args,spec,final,base)
            self.assertEqual(len(list(args.temporary_directory.glob('*.safetensors'))),7)
            with self.assertRaisesRegex(ValueError,'Stale'):self.run_fixture(args,spec,final,base)

    def test_hybrid_gradient_failure_restores_final_and_retains_state(self):
        with tempfile.TemporaryDirectory() as d:
            args,spec,_=self.fixture(Path(d));final,base=models();before=a.model_state_hashes(final,torch)
            original=a.save_gradient
            def fail(model,eligible,tokens,settings,path,t):
                if path.stem=='gH':raise RuntimeError('hybrid gradient failed')
                return original(model,eligible,tokens,settings,path,t)
            with patch.object(a,'save_gradient',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'hybrid gradient failed'):self.run_fixture(args,spec,final,base)
            self.assertEqual(before,a.model_state_hashes(final,torch))
            self.assertTrue((args.temporary_directory/'g1.safetensors').exists())
            self.assertFalse(args.diagnostic_path.exists())

    def test_altered_checkpoint_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as d:
            args,spec,_=self.fixture(Path(d));(args.base_model_dir/'config.json').write_text('changed')
            with patch.object(a,'FROZEN_SPEC',spec),patch.object(a,'load_local_model') as loader:
                with self.assertRaisesRegex(ValueError,'inventory/hash'):a.construct(args)
                loader.assert_not_called()


if __name__=='__main__':unittest.main()
