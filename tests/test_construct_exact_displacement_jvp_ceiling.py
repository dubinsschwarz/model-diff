import copy
import importlib.util
import json
import math
import tempfile
import unittest
import io
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from safetensors.torch import save_file
PROJECT=Path(__file__).resolve().parents[1]
SCRIPT=PROJECT/'scripts/ablation/construct_exact_displacement_jvp_ceiling.py'
S=importlib.util.spec_from_file_location('ceiling016',SCRIPT);c=importlib.util.module_from_spec(S);S.loader.exec_module(c)
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

def spec_small():
    spec=copy.deepcopy(c.FROZEN_SPEC)
    spec['probe'].update(sample_count=3,batch_size=2)
    spec['readout']['hidden_size']=2;spec['outputs']['shape']=[128,2]
    return spec


def audited_pair(base,final):
    eligible=a.discover_eligible_linear_weights(base,torch)
    saved,records,coordinates=c.snapshot_base(base,eligible)
    base_values={name:p.detach().clone() for name,p in base.named_parameters()}
    return c.audit_and_tangent(final,a.discover_eligible_linear_weights(final,torch),saved,records,coordinates,
                               lambda name,record:base_values[name])


class MathematicsTests(unittest.TestCase):
    def test_displacement_positive_sign_norm_rounding_no_renormalization(self):
        base=FamilyModel();final=copy.deepcopy(base)
        with torch.no_grad():final.model.layers[0].self_attn['q_proj'].weight.add_(.25)
        tangents,audit=audited_pair(base,final)
        key='model.layers.0.self_attn.q_proj.weight'
        expected=final.get_parameter(key).detach().double()-base.get_parameter(key).detach().double()
        self.assertTrue(torch.equal(tangents[key],expected.float()))
        self.assertEqual(audit['exact_displacement_norm'],float(expected.norm()))
        self.assertEqual(audit['realized_fp32_displacement_norm'],float(expected.float().double().norm()))
        self.assertEqual(audit['rounding_error_norm'],float((expected-expected.float().double()).norm()))
        self.assertEqual(len(tangents),98)
        self.assertEqual(audit['categories']['selected_early_linear_weights']['nonzero_names'],[key])
        self.assertEqual(audit['all_unique_parameters_audited'],len(list(base.parameters())))
    def test_exact_linear_jvp_equals_finite_difference_without_scale(self):
        base=TinyModel();final=TinyModel();spec=spec_small();tokens=torch.zeros(3,128,dtype=torch.int64)
        with torch.no_grad():final.model.layers[0].proj.weight.copy_(torch.tensor([[2.,1.],[1.,3.]]))
        delta=final.model.layers[0].proj.weight.detach()-base.model.layers[0].proj.weight.detach()
        primal,response=a.run_readout_jvp(final,tokens,{'model.layers.0.proj.weight':delta},spec,torch)
        baseline=a.ordinary_hook_readout(base,tokens,spec,torch)
        self.assertTrue(torch.equal(response,primal-baseline))
        x=torch.tensor([1.,2.]).repeat(3,128,1)
        self.assertTrue(torch.equal(response,torch.nn.functional.linear(x,delta)))
    def test_nonlinear_endpoint_error_and_centered_finite_difference(self):
        base=TinyModel(nonlinear=True);final=TinyModel(nonlinear=True);spec=spec_small();tokens=torch.zeros(1,128,dtype=torch.int64)
        with torch.no_grad():final.model.layers[0].proj.weight.mul_(1.5)
        weight=final.model.layers[0].proj.weight.detach().clone();delta=weight-base.model.layers[0].proj.weight.detach()
        primal,response=a.run_readout_jvp(final,tokens,{'model.layers.0.proj.weight':delta},spec,torch)
        finite=primal-a.ordinary_hook_readout(base,tokens,spec,torch)
        self.assertGreater(float((response-finite).norm()),.01)
        epsilon=1e-3
        with torch.no_grad():
            final.model.layers[0].proj.weight.copy_(weight+epsilon*delta)
            plus=a.ordinary_hook_readout(final,tokens,spec,torch)
            final.model.layers[0].proj.weight.copy_(weight-epsilon*delta)
            minus=a.ordinary_hook_readout(final,tokens,spec,torch)
            final.model.layers[0].proj.weight.copy_(weight)
        torch.testing.assert_close(response,(plus-minus)/(2*epsilon),rtol=3e-3,atol=3e-5)
    def test_fp64_difference_before_storage(self):
        spec=spec_small();base=torch.ones(128,2,dtype=torch.float64);final=base+1e-8;response=torch.ones_like(base)
        artifact=c.build_artifact(base,final,response,spec)
        self.assertTrue(torch.equal(artifact['base_mean'],artifact['final_mean']))
        self.assertTrue(torch.equal(artifact['exact_difference'],(final-base).float()))
        self.assertGreater(float(artifact['exact_difference'].norm()),0)
        for key in spec['outputs']['tensors']:
            self.assertEqual(artifact[key].dtype,torch.float32);self.assertTrue(artifact[key].is_contiguous())
        self.assertEqual(artifact['metadata'],c.build_artifact(base,final,response,spec)['metadata'])
    def test_metrics_projection_primary_secondary_and_zero_policy(self):
        spec=spec_small();x=torch.tensor([1.,2.]).repeat(128,1);y=3*x
        y[0]=-x[0]
        result=c.metrics({'linear_response':x,'exact_difference':y},spec)
        self.assertAlmostEqual(result['primary_mean_cosine_positions_1_4'],1)
        self.assertAlmostEqual(result['secondary_mean_cosine_positions_1_127'],1)
        self.assertAlmostEqual(result['position_0']['cosine'],-1)
        self.assertEqual(result['positions'][1]['optimal_scalar_coefficient'],3)
        self.assertEqual(result['positions'][1]['relative_residual_after_optimal_rescaling'],0)
        self.assertEqual(result,c.metrics({'linear_response':x,'exact_difference':y},spec))
        x[3].zero_();z=c.metrics({'linear_response':x,'exact_difference':y},spec)
        self.assertIsNone(z['primary_mean_cosine_positions_1_4']);self.assertIsNone(z['positions'][3]['optimal_scalar_coefficient'])
    def test_upstream_embedding_and_norm_displacement_rejected(self):
        for which in ('embedding','norm'):
            base=FamilyModel()
            if which=='embedding':base.model.embed_tokens=torch.nn.Embedding(3,2)
            else:base.model.layers[3].extra_norm=torch.nn.LayerNorm(2)
            final=copy.deepcopy(base)
            with torch.no_grad():
                if which=='embedding':final.model.embed_tokens.weight.add_(1)
                else:final.model.layers[3].extra_norm.weight.add_(1)
            with self.subTest(which=which),self.assertRaisesRegex(ValueError,'Omitted upstream'):audited_pair(base,final)
    def test_tied_output_embedding_is_upstream(self):
        base=FamilyModel();base.lm_head=torch.nn.Linear(2,3,bias=False)
        base.model.embed_tokens=torch.nn.Embedding(3,2);base.model.embed_tokens.weight=base.lm_head.weight
        final=copy.deepcopy(base)
        with torch.no_grad():final.lm_head.weight.add_(1)
        with self.assertRaisesRegex(ValueError,'Omitted upstream'):audited_pair(base,final)
    def test_later_layer_and_final_norm_displacement_allowed(self):
        base=FamilyModel();base.model.norm=torch.nn.LayerNorm(2);final=copy.deepcopy(base)
        with torch.no_grad():
            final.model.layers[27].mlp['up_proj'].weight.add_(.5)
            final.model.norm.weight.add_(.25)
        tangent,audit=audited_pair(base,final)
        self.assertEqual(audit['exact_displacement_norm'],0)
        self.assertTrue(all(torch.count_nonzero(v)==0 for v in tangent.values()))
        self.assertEqual(audit['categories']['later_block_linear_weights']['nonzero_count'],1)
        self.assertEqual(audit['categories']['other_downstream_parameters']['nonzero_count'],1)
        probe=torch.zeros(1,128,dtype=torch.int64);spec=spec_small()
        self.assertTrue(torch.equal(a.ordinary_hook_readout(base,probe,spec,torch),a.ordinary_hook_readout(final,probe,spec,torch)))
    def test_names_order_and_alias_topology_must_match(self):
        base=FamilyModel();final=copy.deepcopy(base);eligible=a.discover_eligible_linear_weights(base,torch)
        saved,records,coordinates=c.snapshot_base(base,eligible)
        self.assertEqual(set(saved),{f'{n}.weight' for n,_ in eligible})
        with self.assertRaisesRegex(ValueError,'names/order'):c.audit_and_tangent(final,eligible,saved,records,list(reversed(coordinates)),None)
        final.new_parameter=torch.nn.Parameter(torch.zeros(1))
        with self.assertRaisesRegex(ValueError,'inventories'):c.audit_and_tangent(final,eligible,saved,records,coordinates,None)
    def test_changed_tensor_reader_checks_loaded_base_hash_and_alias(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);value=torch.tensor([[1.,2.]],dtype=torch.bfloat16);save_file({'alias':value},str(root/'model.safetensors'))
            record={'shape':[1,2],'aliases':['alias'],'sha256':a.sha256_parameter(value.float(),torch)}
            with c.BaseTensorReader(root,[{'path':'model.safetensors'}]) as reader:
                self.assertTrue(torch.equal(reader.get('canonical',record),value.float()))
                with self.assertRaisesRegex(ValueError,'reproduce'):reader.get('canonical',{**record,'sha256':'0'*64})
    def test_probe_hashes_and_firewall(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'probe.pt';tokens=torch.ones(3,128,dtype=torch.int64);torch.save(tokens,path)
            ps={**spec_small()['probe'],'serialized_sha256':a.sha256_file(path),'raw_tensor_sha256':a.sha256_raw_int64_tensor(tokens,torch)}
            self.assertTrue(torch.equal(tokens,a.load_probe(path,ps,torch)))
            for key in ('serialized_sha256','raw_tensor_sha256'):
                with self.assertRaises(ValueError):a.load_probe(path,{**ps,key:'0'*64},torch)
        source=SCRIPT.read_text()
        for forbidden in ('oracle_adl','evaluation.json','adapter','accumulate_mean_generic_gradient','global_tangent_scale'):
            self.assertNotIn(forbidden,source)
        folder=PROJECT/'experiments/attempts'/c.FROZEN_SPEC['attempt_id']
        self.assertEqual(c.load_spec(folder/'spec.json'),c.FROZEN_SPEC)


class PipelineTests(unittest.TestCase):
    def fixture(self,root):
        args=c.parse_args([]);args.device='cpu'
        for key in ('spec_path','probe_path','artifact_path','manifest_path'):setattr(args,key,root/(key+'.json' if 'spec' in key or 'manifest' in key else key+'.pt'))
        args.base_directory=root/'base';args.final_directory=root/'final'
        args.base_directory.mkdir();args.final_directory.mkdir()
        base=FamilyModel()
        base.config=SimpleNamespace(hidden_size=2,rope_scaling={'factor':2.,'rope_type':'linear'})
        base.model.config=base.config
        final=copy.deepcopy(base)
        with torch.no_grad():
            final.model.layers[0].self_attn['q_proj'].weight.add_(.125)
            final.model.layers[20].mlp['up_proj'].weight.add_(.25)
        for model,directory in ((base,args.base_directory),(final,args.final_directory)):
            save_file({n:p.detach().contiguous() for n,p in model.state_dict().items()},str(directory/'model.safetensors'))
        spec=spec_small();spec['base']['files']=a.checkpoint_file_records(args.base_directory);spec['canonical_checkpoint_files']=a.checkpoint_file_records(args.final_directory)
        probe=torch.arange(384).reshape(3,128)%3;torch.save(probe,args.probe_path)
        spec['probe']['serialized_sha256']=a.sha256_file(args.probe_path);spec['probe']['raw_tensor_sha256']=a.sha256_raw_int64_tensor(probe,torch)
        args.spec_path.write_text(json.dumps(spec))
        return args,spec,base,final
    def run_fixture(self,f):
        args,spec,base,final=f;events=[]
        def loader(path,*_):events.append(path);return base if path==args.base_directory else final
        with patch.object(c,'FROZEN_SPEC',spec),patch.object(a,'load_local_model',side_effect=loader),redirect_stdout(io.StringIO()):c.construct(args)
        self.assertEqual(events,[args.base_directory,args.final_directory])
    def test_complete_pipeline_unchanged_models_hashes_order_and_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));before=[a.model_state_hashes(m,torch) for m in f[2:]]
            self.run_fixture(f);args=f[0];result=json.loads(args.manifest_path.read_text());artifact=torch.load(args.artifact_path,weights_only=True)
            self.assertEqual(before,[a.model_state_hashes(m,torch) for m in f[2:]])
            self.assertEqual(list(artifact),[*f[1]['outputs']['tensors'],'metadata'])
            self.assertEqual(result['displacement_audit']['categories']['later_block_linear_weights']['nonzero_count'],1)
            self.assertTrue(result['displacement_audit']['omitted_upstream_zero_verified'])
            self.assertTrue(result['execution_semantics_audit']['upstream_buffers_equal'])
            self.assertTrue(result['execution_semantics_audit']['configuration_equal'])
            for name in f[1]['outputs']['tensors']:self.assertEqual(result['artifact']['raw_tensors_sha256'][name],a.sha256_raw_float32_tensor(artifact[name],torch))
            text=args.manifest_path.read_text();args.manifest_path.unlink();args.artifact_path.unlink()
            self.run_fixture(f);self.assertEqual(text,args.manifest_path.read_text())
            with self.assertRaisesRegex(ValueError,'already exists'):self.run_fixture(f)
    def test_base_and_final_buffer_mutation_detection(self):
        for phase in ('base','final'):
            with tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));owner=c if phase=='base' else a;name='base_probe_mean' if phase=='base' else 'compute_probe_response';original=getattr(owner,name)
                def mutate(model,*args):
                    result=original(model,*args);model.model.offset.add_(1);return result
                with patch.object(owner,name,side_effect=mutate),self.assertRaisesRegex(ValueError,'changed'):self.run_fixture(f)
                self.assertFalse(f[0].artifact_path.exists())
    def test_upstream_change_fails_before_jvp_and_publication(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));f[3].head.weight.data.add_(1)
            with patch.object(a,'compute_probe_response') as jvp,self.assertRaisesRegex(ValueError,'Omitted upstream'):self.run_fixture(f)
            jvp.assert_not_called();self.assertFalse(f[0].artifact_path.exists())
    def test_upstream_buffer_change_fails_before_jvp_and_publication(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));f[3].model.offset.add_(1)
            with patch.object(a,'compute_probe_response') as jvp,self.assertRaisesRegex(ValueError,'Upstream buffer'):
                self.run_fixture(f)
            jvp.assert_not_called()
            self.assertFalse(f[0].artifact_path.exists());self.assertFalse(f[0].manifest_path.exists())

    def test_relevant_configuration_change_rejected(self):
        for field,value in [('hidden_size',3),('rope_scaling',{'factor':3.}),('layer_types',['sliding_attention'])]:
            with self.subTest(field=field),tempfile.TemporaryDirectory() as d:
                f=self.fixture(Path(d));setattr(f[3].config,field,value)
                with patch.object(a,'compute_probe_response') as jvp,self.assertRaisesRegex(ValueError,'configuration semantics'):
                    self.run_fixture(f)
                jvp.assert_not_called();self.assertFalse(f[0].artifact_path.exists())

    def test_metadata_config_difference_and_nested_order_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));f[3].config._name_or_path='different-local-path'
            f[3].config.transformers_version='different';f[3].config.use_cache=True
            f[3].config.eos_token_id=99;f[3].config.rope_scaling={'rope_type':'linear','factor':2.}
            self.run_fixture(f)
            record=json.loads(f[0].manifest_path.read_text())['execution_semantics_audit']
            self.assertTrue(record['configuration_equal'])

    def test_buffer_inventory_alias_dtype_and_downstream_rules(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));base,final=f[2:];spec=f[1]
            base.model.layers[20].register_buffer('late',torch.ones(2))
            final.model.layers[20].register_buffer('late',torch.zeros(3,dtype=torch.float64))
            snap=c.execution_snapshot(base,spec)
            audit=c.audit_execution_semantics(snap,c.execution_snapshot(final,spec))
            row=next(r for r in audit['buffers'] if r['name'].endswith('.late'))
            self.assertFalse(row['upstream']);self.assertFalse(row['equal'])
            final.model.offset=final.model.offset.double()
            with self.assertRaisesRegex(ValueError,'Upstream buffer'):c.audit_execution_semantics(snap,c.execution_snapshot(final,spec))
            final.model.offset=final.model.offset.float()
            final.model.register_buffer('extra',torch.ones(1))
            with self.assertRaisesRegex(ValueError,'inventories'):c.audit_execution_semantics(snap,c.execution_snapshot(final,spec))
            del final.model.extra
            final.model.register_buffer('alias',final.model.offset)
            with self.assertRaisesRegex(ValueError,'alias topology'):c.audit_execution_semantics(snap,c.execution_snapshot(final,spec))

    def test_global_rotary_and_downstream_alias_are_upstream(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));base,final=f[2:];spec=f[1]
            for model in (base,final):
                model.model.register_buffer('rotary_inv_freq',torch.ones(2))
                model.model.layers[20].register_buffer('alias',model.model.rotary_inv_freq)
            final.model.rotary_inv_freq.add_(1)
            with self.assertRaisesRegex(ValueError,'Upstream buffer'):
                c.audit_execution_semantics(c.execution_snapshot(base,spec),c.execution_snapshot(final,spec))

    def test_checkpoint_mutation_and_partial_output_refused(self):
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d));p=f[0].base_directory/'model.safetensors';p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'inventory/hash'):self.run_fixture(f)
        with tempfile.TemporaryDirectory() as d:
            f=self.fixture(Path(d))
            with patch.object(a,'write_manifest',side_effect=OSError('publication failed')),self.assertRaises(OSError):self.run_fixture(f)
            self.assertTrue(f[0].artifact_path.exists());self.assertFalse(f[0].manifest_path.exists())
            with self.assertRaisesRegex(ValueError,'already exists'):self.run_fixture(f)

if __name__=='__main__':unittest.main()
