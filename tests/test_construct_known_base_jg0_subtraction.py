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
SCRIPT=PROJECT/'scripts/ablation/construct_known_base_jg0_subtraction.py'
S=importlib.util.spec_from_file_location('construct015',SCRIPT);c=importlib.util.module_from_spec(S);S.loader.exec_module(c)
a=c.a
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

class FamilyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2, bias=False)
                                             for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')})
        self.mlp = torch.nn.ModuleDict({name: torch.nn.Linear(2, 2, bias=False)
                                      for name in ('gate_proj', 'up_proj', 'down_proj')})
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(.01)

    def forward(self, hidden):
        return hidden + sum(layer(hidden) for group in (self.self_attn, self.mlp) for layer in group.values()) * .01

class FamilyModel(TinyModel):
    def __init__(self):
        super().__init__()
        self.model.layers = torch.nn.ModuleList([FamilyBlock() for _ in range(28)])
        self.head = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[.2, -.1], [.1, .3], [-.2, .5]]))
            for index, parameter in enumerate(self.model.parameters()):
                parameter.copy_(.05 * torch.sin(torch.arange(parameter.numel()).float() + index).reshape_as(parameter))

    def forward(self, *, input_ids, use_cache):
        hidden = self.model(input_ids=input_ids, use_cache=use_cache, output_hidden_states=True, return_dict=True)
        return SimpleNamespace(logits=self.head(hidden.hidden_states[-1]))

class MathTests(unittest.TestCase):
    def raw(self):
        raw={'merged_mean':torch.ones(128,2)}
        for i,name in enumerate(c.FROZEN_SPEC['corpus_order']):raw[f'response_{name}']=torch.tensor([1.+i,2.-i]).repeat(128,1)
        raw['consensus_response'],_=a.response_consensus([raw[f'response_{n}'] for n in c.FROZEN_SPEC['corpus_order']],torch)
        return raw
    def spec(self):
        spec=copy.deepcopy(c.FROZEN_SPEC);spec['outputs']['shape']=[128,2];spec['readout']['hidden_size']=2
        return spec
    def test_true_linear_jvp_identity_common_positive_alpha(self):
        model=TinyModel();spec=self.spec();tokens=torch.zeros(1,128,dtype=torch.int64)
        g1=torch.tensor([[1.,2.],[3.,4.]]);g0=torch.tensor([[.5,1.],[2.,1.]])
        alpha=.25;key='model.layers.0.proj.weight'
        jvp=lambda v:a.run_readout_jvp(model,tokens,{key:v},spec,torch)[1]
        self.assertTrue(torch.equal(jvp(alpha*g1)-jvp(alpha*g0),jvp(alpha*(g1-g0))))
        self.assertTrue(torch.equal(jvp(alpha*g1),-jvp(-alpha*g1)))
    def test_exact_scaling_no_renormalization_and_coordinate_checks(self):
        model=TinyModel();eligible=[('model.layers.0.proj',model.model.layers[0].proj)]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'g.safetensors';g=torch.tensor([[.1,-.2],[1.,2.]])
            save_file({'model.layers.0.proj.weight':g},str(path))
            for alpha in (.0001,7.):
                tangent,norm=c.load_scaled_tangent(path,eligible,alpha)
                expected=(alpha*g.double()).float()
                self.assertTrue(torch.equal(tangent['model.layers.0.proj.weight'],expected))
                self.assertEqual(norm,float(expected.double().norm()))
            for alpha in (-1.,0.,float('nan')):
                with self.assertRaises(ValueError):c.load_scaled_tangent(path,eligible,alpha)
            with self.assertRaisesRegex(ValueError,'names/order'):c.validate_coordinates(eligible,[{'name':'wrong','shape':[2,2]}])
    def test_fp64_subtraction_audit_and_consensus_geometry(self):
        spec=self.spec();raw=self.raw();baseline={name:torch.tensor([.125,.25]).repeat(128,1) for name in spec['corpus_order']}
        out,geometry=c.build_outputs(raw,baseline,spec)
        self.assertEqual(list(out),spec['outputs']['tensors'])
        for name in spec['corpus_order']:
            exact=raw[f'response_{name}'].double()-baseline[name].double()
            self.assertTrue(torch.equal(out[f'residual_response_{name}'],exact.float()))
            expected_error=out[f'residual_response_{name}'].double()-exact
            self.assertEqual(geometry['rounding_audit'][name][0]['fp32_rounding_max_abs'],float(expected_error[0].abs().max()))
        expected,_=a.response_consensus([baseline[n] for n in spec['corpus_order']],torch)
        self.assertTrue(torch.equal(out['baseline_consensus_response'],expected))
        residual,_=a.response_consensus([out[f'residual_response_{n}'] for n in spec['corpus_order']],torch)
        self.assertTrue(torch.equal(out['residual_consensus_response'],residual))
        row=geometry['positions'][0]['corpora']['fineweb'];x=raw['response_fineweb'][0].double();b=baseline['fineweb'][0].double()
        self.assertEqual(row['raw_baseline_cosine'],float(torch.dot(x,b)/(x.norm()*b.norm())))
        self.assertEqual(len(geometry['positions']),128)
        self.assertEqual(geometry['position_0'],geometry['positions'][0])
        self.assertEqual(geometry['positions_1_4']['corpora']['fineweb']['raw_norm']['mean'],float(x.norm()))
        self.assertEqual(geometry,c.build_outputs(raw,baseline,spec)[1])
    def test_nonzero_fp32_rounding_audit(self):
        x=torch.full((128,2),1.);y=torch.full((128,2),1e-8)
        residual,audit=c.residual_and_audit(x,y)
        self.assertTrue(torch.equal(residual,(x.double()-y.double()).float()))
        self.assertGreater(audit[0]['fp32_rounding_max_abs'],0)
    def test_zero_and_nonfinite_responses_fail(self):
        for value in (0.,float('nan'),float('inf')):
            spec=self.spec();baseline={name:torch.ones(128,2) for name in spec['corpus_order']};baseline['fineweb'][127]*=value
            with self.assertRaises(ValueError):c.build_outputs(self.raw(),baseline,spec)
        raw=self.raw();baseline={name:raw[f'response_{name}'].clone() for name in c.FROZEN_SPEC['corpus_order']}
        with self.assertRaisesRegex(ValueError,'norm'):c.build_outputs(raw,baseline,self.spec())
    def test_raw_geometry_reproduction_rejects_alteration(self):
        spec=self.spec();raw=self.raw();_,geometry=a.response_consensus([raw[f'response_{n}'] for n in spec['corpus_order']],torch)
        c.check_raw_geometry(raw,{'geometry':geometry},spec)
        geometry['positions'][1]['response_norms'][0]+=1
        with self.assertRaisesRegex(ValueError,'reproduction'):c.check_raw_geometry(raw,{'geometry':geometry},spec)
    def test_primal_consistency_all_positions_and_zero_policy(self):
        spec=self.spec();reference=torch.ones(128,2);candidate=reference.double()+1e-7
        self.assertEqual(len(c.primal_consistency(candidate,reference,spec)),128)
        candidate[127]*=2
        with self.assertRaisesRegex(ValueError,'127'):c.primal_consistency(candidate,reference,spec)
        c.primal_consistency(torch.zeros(128,2,dtype=torch.float64),torch.zeros(128,2),spec)
    def test_firewall_and_reused_functions(self):
        source=SCRIPT.read_text()
        for forbidden in ('oracle_adl.pt','evaluation.json','evaluate_attempt','global_tangent_scale','Hessian','adapter'):
            self.assertNotIn(forbidden,source)
        for helper in ('accumulate_mean_generic_gradient','run_readout_jvp','compute_probe_response','model_state_hashes'):
            self.assertTrue(callable(getattr(a,helper)))
        self.assertEqual(c.FROZEN_SPEC['purpose'],'post_oracle_known_base_mechanistic_diagnostic_not_blind_recovery')
        self.assertEqual(c.load_spec(PROJECT/'experiments/attempts'/c.FROZEN_SPEC['attempt_id']/'spec.json'),c.FROZEN_SPEC)


class PipelineTests(unittest.TestCase):
    def fixture(self,root):
        args=c.parse_args([])
        for key in ('spec_path','raw_artifact_path','probe_path','artifact_path','manifest_path'):
            setattr(args,key,root/(key+'.pt' if 'artifact' in key or key=='probe_path' else key+'.json'))
        args.base_directory=root/'base';args.final_directory=root/'final';args.temporary_directory=root/'temporary';args.device='cpu'
        for p in (args.base_directory,args.final_directory):p.mkdir();(p/'config.json').write_text('{}')
        spec=copy.deepcopy(c.FROZEN_SPEC);spec['outputs']['shape']=[128,2];spec['readout']['hidden_size']=2
        spec['generic_loss'].update(sample_count=2,sequence_length=128,batch_size=1,total_prediction_tokens=254,number_of_batches=2)
        spec['probe'].update(sample_count=2,batch_size=1)
        tokens=torch.arange(256).reshape(2,128)%3;torch.save(tokens,args.probe_path)
        spec['probe']['serialized_sha256']=a.sha256_file(args.probe_path);spec['probe']['raw_tensor_sha256']=a.sha256_raw_int64_tensor(tokens,torch)
        spec['canonical_checkpoint_files']=a.checkpoint_file_records(args.final_directory);spec['base']['files']=a.checkpoint_file_records(args.base_directory)
        base=FamilyModel();final=FamilyModel()
        with torch.no_grad():base.model.layers[0].self_attn['q_proj'].weight.add_(.2)
        selected=a.discover_eligible_linear_weights(final,torch)
        matrices=[{'name':f'{n}.weight','shape':list(m.weight.shape)} for n,m in selected]
        previous=copy.deepcopy(a.FROZEN_SPEC)
        for key in ('probe','generic_loss','jvp','readout','canonical_checkpoint_files'):previous[key]=spec[key]
        provenance=[]
        for record in previous['corpora']:
            for key in ('spec_path','manifest_path','tokens_path'):
                p=root/(record['name']+'_'+key);p.write_text('synthetic');record[key]=str(p)
            provenance.append({'name':record['name'],'spec_sha256':a.sha256_file(Path(record['spec_path'])),'manifest_sha256':a.sha256_file(Path(record['manifest_path'])),'serialized_sha256':a.sha256_file(Path(record['tokens_path'])),'raw_tensor_sha256':'a'*64})
        spec14=root/'spec14.json';spec14.write_text(json.dumps(previous));spec['frozen_attempt014']['spec']={'path':str(spec14),'sha256':a.sha256_file(spec14)}
        raw={'merged_mean':a.ordinary_hook_readout(final,tokens,spec,torch).double().mean(0).float()}
        for i,name in enumerate(spec['corpus_order']):raw[f'response_{name}']=torch.tensor([1.+i,2.-i]).repeat(128,1)
        raw['consensus_response'],geom=a.response_consensus([raw[f'response_{n}'] for n in spec['corpus_order']],torch)
        torch.save(raw,args.raw_artifact_path)
        prior={'attempt_id':previous['attempt_id'],'spec_sha256':spec['frozen_attempt014']['spec']['sha256'],'constructor_script_sha256':spec['frozen_attempt014']['constructor']['sha256'],
            **{k:spec[k] for k in ('probe','generic_loss','jvp','readout')},'consensus_definition':spec['consensus'],'source_checkpoint':spec['canonical_checkpoint_files'],
            'corpus_order':spec['corpus_order'],'corpora':[{**p,'alpha':alpha,'matrices':matrices} for p,alpha in zip(provenance,[.2,.7,1.3])],
            'geometry':geom,'artifact':{'serialized_sha256':a.sha256_file(args.raw_artifact_path),'raw_tensors_sha256':{k:a.sha256_raw_float32_tensor(v,torch) for k,v in raw.items()}}}
        pm=root/'manifest14.json';pm.write_text(json.dumps(prior));spec['frozen_attempt014']['manifest']={'path':str(pm),'sha256':a.sha256_file(pm)}
        args.spec_path.write_text(json.dumps(spec))
        return args,spec,previous,prior,raw,tokens,base,final

    def run_fixture(self,f):
        args,spec,previous,prior,raw,tokens,base,final=f
        def corpus(record,*_):
            expected=next(r for r in prior['corpora'] if r['name']==record['name'])
            hashes={Path(record[k]):a.sha256_file(Path(record[k])) for k in ('spec_path','manifest_path','tokens_path')}
            return tokens.clone(),hashes,{k:expected[k] for k in ('name','spec_sha256','manifest_sha256','serialized_sha256','raw_tensor_sha256')}
        events=[]
        def load(path,*_):
            events.append(path)
            if path==args.final_directory:self.assertEqual(len(list(args.temporary_directory.glob('g0_*.safetensors'))),3)
            return base if path==args.base_directory else final
        with patch.object(c,'FROZEN_SPEC',spec),patch.object(a,'FROZEN_SPEC',previous),patch.object(a,'validate_frozen_corpus',side_effect=corpus),patch.object(a,'load_local_model',side_effect=load),patch.object(a,'global_tangent_scale',side_effect=AssertionError('must never renormalize')),redirect_stdout(io.StringIO()):
            c.construct(args)
        self.assertEqual(events,[args.base_directory,args.final_directory])

    def test_complete_pipeline_copied_alphas_unchanged_models_hashes_cleanup_and_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));args=f[0];before=[a.model_state_hashes(m,torch) for m in f[-2:]]
            self.run_fixture(f);result=json.loads(args.manifest_path.read_text());artifact=torch.load(args.artifact_path,weights_only=True)
            self.assertEqual([r['alpha'] for r in result['corpora']],[.2,.7,1.3])
            self.assertEqual(before,[a.model_state_hashes(m,torch) for m in f[-2:]])
            self.assertFalse(args.temporary_directory.exists());self.assertEqual(set(artifact),set(f[1]['outputs']['tensors']))
            for name,value in artifact.items():self.assertEqual(result['artifact']['raw_tensors_sha256'][name],a.sha256_raw_float32_tensor(value,torch))
            text=args.manifest_path.read_text();args.manifest_path.unlink();args.artifact_path.unlink()
            self.run_fixture(f);self.assertEqual(text,args.manifest_path.read_text())
            with self.assertRaisesRegex(ValueError,'already exists'):self.run_fixture(f)
    def test_raw_artifact_serialized_and_raw_hash_rejection(self):
        for raw_change in (False,True):
            with tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));args,spec,previous,prior,raw,*_=f
                if raw_change:
                    raw['response_fineweb'][0,0]+=1;torch.save(raw,args.raw_artifact_path)
                    prior['artifact']['serialized_sha256']=a.sha256_file(args.raw_artifact_path)
                    p=Path(spec['frozen_attempt014']['manifest']['path']);p.write_text(json.dumps(prior));spec['frozen_attempt014']['manifest']['sha256']=a.sha256_file(p)
                else:args.raw_artifact_path.write_bytes(b'changed')
                with patch.object(a,'FROZEN_SPEC',previous),self.assertRaises(ValueError):c.load_raw(spec,args.raw_artifact_path)
    def test_base_and_final_mutation_rejected(self):
        for phase in ('gradient','jvp'):
            with tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));name='accumulate_mean_generic_gradient' if phase=='gradient' else 'compute_probe_response';original=getattr(a,name)
                def mutate(model,*args,**kwargs):
                    result=original(model,*args,**kwargs)
                    with torch.no_grad():next(model.parameters()).add_(1)
                    return result
                with patch.object(a,name,side_effect=mutate),self.assertRaisesRegex(ValueError,'changed'):self.run_fixture(f)
                self.assertFalse(f[0].manifest_path.exists())
    def test_publication_failure_retains_three_gradients_and_stale_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d))
            with patch.object(a,'write_manifest',side_effect=OSError('publish failed')),self.assertRaises(OSError):self.run_fixture(f)
            self.assertEqual(len(list(f[0].temporary_directory.glob('*.safetensors'))),3)
            f[0].artifact_path.unlink()
            with self.assertRaisesRegex(ValueError,'Stale'):self.run_fixture(f)

if __name__=='__main__':unittest.main()
