import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load(ROOT/'scripts/ablation/evaluate_attempt018.py','e018')
toy=load(ROOT/'tests/test_construct_full_support_endpoint_hessian_forward.py','toy018e')

class EvaluationTests(unittest.TestCase):
 def test_frozen_spec_consumable_by_activation_readout_helper(self):
  spec_path=ROOT/'experiments/attempts'/c.FROZEN_SPEC['attempt_id']/'evaluation_spec.json'
  spec=json.loads(spec_path.read_text())
  self.assertEqual(spec,c.FROZEN_SPEC)
  self.assertEqual(spec['model'],{'model_type':'qwen3','num_hidden_layers':28,
   'source_dtype':'float32','local_files_only':True,'eval_mode':True,'use_cache':False})
  states=tuple(torch.full((1,128,2048),float(i),dtype=torch.float32) for i in range(29))
  output=SimpleNamespace(hidden_states=states)
  selected=c.activation.hidden_state_output(output,spec)
  self.assertIs(selected,states[14])
  self.assertEqual(spec['readout']['block_index'],13)
  c.activation.validate_readout(selected,1,spec,torch,'toy block-13 output')
  with self.assertRaisesRegex(ValueError,'unexpected length'):
   c.activation.hidden_state_output(SimpleNamespace(hidden_states=states[:-1]),spec)
 def fixture(self,root):
  base,final,s=toy.models();spec=copy.deepcopy(c.FROZEN_SPEC);spec['coordinates']=s['coordinates']
  vector={n:(final.get_parameter(n).detach().double()-base.get_parameter(n).detach().double()).float() for n in [r['name'] for r in spec['coordinates']]}
  path=root/'solution';record=c.support.save_vector(path,vector,spec['coordinates'])
  manifest={'attempt_id':spec['attempt_id'],'spec_sha256':spec['construction_spec_sha256'],'constructor_script_sha256':spec['constructor_sha256'],
            'solver':spec['solver'],'iteration_count':4,'iterations':[{'iteration':i} for i in range(1,5)],
            'coordinates':spec['coordinates'],'final_checkpoint_files':spec['canonical_checkpoint_files'],'solution':record}
  mp=root/'manifest';mp.write_text(json.dumps(manifest))
  return base,final,spec,path,mp,manifest
 def test_frozen_solution_pins_and_raw_hash(self):
  with tempfile.TemporaryDirectory() as d:
   base,final,spec,p,m,manifest=self.fixture(Path(d));mh=c.support.sha256_file(m);sh=c.support.sha256_file(p)
   c.validate_solution(m,p,spec,mh,sh)
   for a,b in [('0'*64,sh),(mh,'0'*64)]:
    with self.assertRaises(ValueError):c.validate_solution(m,p,spec,a,b)
   manifest['solution']['matrices'][0]['raw_sha256']='0'*64;m.write_text(json.dumps(manifest))
   with self.assertRaisesRegex(ValueError,'raw/inventory'):c.validate_solution(m,p,spec,c.support.sha256_file(m),sh)
 def test_delta_positive_parameter_scopes_and_no_selection(self):
  with tempfile.TemporaryDirectory() as d:
   base,final,spec,p,m,_=self.fixture(Path(d))
   class Reader:
    def get(self,name):return base.get_parameter(name).detach()
   with c.safe_open(str(p),framework='pt') as solution:
    metrics=c.parameter_metrics(final,Reader(),solution,spec['coordinates'])
   self.assertEqual(len(metrics['blocks']),28)
   self.assertAlmostEqual(metrics['full']['cosine'],1)
   self.assertAlmostEqual(metrics['full']['norm_ratio'],1,places=6)
   self.assertLess(metrics['full']['unscaled_relative_residual'],1e-7)
   self.assertEqual([r['block'] for r in metrics['blocks']],list(range(28)))
 def test_late_coordinates_excluded_and_early_sign_unchanged(self):
  with tempfile.TemporaryDirectory() as d:
   _,final,spec,p,_,_=self.fixture(Path(d))
   with c.safe_open(str(p),framework='pt') as solution:
    tangent=c.early_tangent(final,solution)
    self.assertEqual(len(tangent),98)
    self.assertTrue(all(c.support.block_number(n)<14 for n in tangent))
    for n,v in tangent.items():self.assertTrue(torch.equal(v,solution.get_tensor(n)))
 def test_functional_summary_all_positions_and_negative_sign(self):
  oracle=torch.ones(128,3);response=oracle.clone();response[0].neg_();response[4].neg_()
  m=c.functional_geometry(response,oracle)
  self.assertEqual(len(m['positions']),128);self.assertAlmostEqual(m['position_0']['cosine'],-1)
  self.assertAlmostEqual(m['primary_positions_1_4']['mean'],.5)
  self.assertAlmostEqual(m['secondary_positions_1_127']['mean'],125/127)
  self.assertEqual([r['position'] for r in m['individual_positions_1_4']],[1,2,3,4])
 def test_zero_policy_and_determinism(self):
  zero=torch.zeros(128,2);one=torch.ones(128,2)
  m=c.functional_geometry(zero,one);self.assertIsNone(m['primary_positions_1_4']['mean'])
  self.assertEqual(json.dumps(m,allow_nan=False),json.dumps(c.functional_geometry(zero,one),allow_nan=False))
 def test_wrong_iteration_rejected_not_chosen(self):
  with tempfile.TemporaryDirectory() as d:
   _,_,spec,p,m,manifest=self.fixture(Path(d));manifest['iteration_count']=3;m.write_text(json.dumps(manifest))
   with self.assertRaisesRegex(ValueError,'k=4'):c.validate_solution(m,p,spec,c.support.sha256_file(m),c.support.sha256_file(p))
 def test_separate_evaluator_no_constructor_effect_and_overwrite(self):
  source=(ROOT/'scripts/ablation/construct_known_cleaned_rhs_minres.py').read_text()
  self.assertNotIn('evaluate_attempt018',source)
  with tempfile.TemporaryDirectory() as d:
   args=c.parse_args(['--construction-manifest-sha256','a'*64,'--solution-sha256','b'*64]);args.output_path=Path(d)/'evaluation';args.output_path.touch()
   with self.assertRaisesRegex(ValueError,'already exists'):c.evaluate(args)
 def test_functional_runs_unconditionally_after_poor_parameter_result(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);base,model,spec,solution,manifest_path,manifest=self.fixture(root)
   base_dir=root/'base';final_dir=root/'final';base_dir.mkdir();final_dir.mkdir()
   for m,p in ((base,base_dir),(model,final_dir)):
    save_file({n:v.detach().contiguous() for n,v in m.state_dict().items()},str(p/'model.safetensors'))
   spec['base']['files']=c.support.checkpoint_file_records(base_dir)
   spec['canonical_checkpoint_files']=c.support.checkpoint_file_records(final_dir)
   construction_spec=root/'construction_spec';construction_spec.write_text('{}')
   spec['construction_spec_sha256']=c.support.sha256_file(construction_spec)
   probe_path=root/'probe';tokens=torch.zeros(3,128,dtype=torch.int64);torch.save(tokens,probe_path)
   spec['probe'].update(sample_count=3,serialized_sha256=c.support.sha256_file(probe_path),raw_tensor_sha256=c.support.sha256_raw_int64_tensor(tokens,torch))
   shape=(128,2048);one=torch.ones(shape);difference=one*.5
   method=dict(relative_layer=.5,layer_index=13,num_layers=28,sample_count=3,sequence_length=128,hidden_size=2048,
               batch_size=32,model_dtype='float32',accumulator_dtype='float64',stored_dtype='float32')
   oracle={'format_version':1,'metadata':method,'base_mean':one-difference,'ft_mean':one,'difference':difference}
   oracle_path=root/'synthetic_oracle';torch.save(oracle,oracle_path)
   om={**method,'oracle_adl_sha256':c.support.sha256_file(oracle_path),
       'raw_tensors_sha256':{n:c.activation.sha256_raw_float32_tensor(oracle[n],torch) for n in ('base_mean','ft_mean','difference')}}
   spec['oracle_provenance']={}
   spec['paths']={k:str(v) for k,v in {'base':base_dir,'final':final_dir,'solution':solution,'construction_manifest':manifest_path,
    'construction_spec':construction_spec,'probe':probe_path,'oracle_artifact':oracle_path}.items()}
   manifest.update(spec_sha256=spec['construction_spec_sha256'],final_checkpoint_files=spec['canonical_checkpoint_files'])
   manifest_path.write_text(json.dumps(manifest))
   args=c.parse_args(['--construction-manifest-sha256',c.support.sha256_file(manifest_path),'--solution-sha256',c.support.sha256_file(solution)])
   args.spec_path=root/'evaluation_spec';args.spec_path.write_text(json.dumps(spec));args.output_path=root/'evaluation';args.device='cpu'
   with patch.object(c,'FROZEN_SPEC',spec),patch.object(c.oracle_support,'validate_oracle_provenance',return_value=om),patch.object(c.activation,'load_local_model',return_value=model),patch.object(c,'parameter_metrics',return_value={'full':{'cosine':-.99}}),patch.object(c.activation,'compute_probe_response',return_value=(one.double(),difference.double(),{'readout_hook_verified':True})) as jvp:
    c.evaluate(args)
   jvp.assert_called_once()
   result=json.loads(args.output_path.read_text())
   self.assertEqual(result['parameter_geometry']['full']['cosine'],-.99)
   self.assertAlmostEqual(result['functional_geometry']['primary_positions_1_4']['mean'],1)
   self.assertEqual(result['construction_manifest_sha256'],args.construction_manifest_sha256)
 def test_merged_mean_gate_all_positions(self):
  x=torch.ones(128,2);y=x.clone();y[120,0]=2
  with self.assertRaisesRegex(ValueError,'position 120'):c.consistency.merged_mean_consistency(x,y,c.FROZEN_SPEC)

if __name__=='__main__':unittest.main()
