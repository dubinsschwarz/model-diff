import copy
import importlib.util
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
c=load(ROOT/'scripts/ablation/construct_known_cleaned_rhs_minres.py','c018')
toy=load(ROOT/'tests/test_construct_full_support_endpoint_hessian_forward.py','toy017')

def dictionary(v):return {'model.layers.0.x.weight':v[:len(v)//2].clone().reshape(1,-1),'model.layers.27.x.weight':v[len(v)//2:].clone().reshape(1,-1)}
def flatten(d):return torch.cat([v.flatten() for v in d.values()])
def operator(matrix,calls=None):
 def op(r,beta):
  if calls is not None:calls.append(tuple(v.data_ptr() for v in r.values()))
  v=(flatten(r).double()/beta).float();return dictionary((matrix.double()@v.double()).float())
 return op

def dense_minimum_residual(A,b,k):
 # Independent explicit Krylov least-squares reference; no short recurrence.
 q=b.double()/b.double().norm();basis=[]
 for _ in range(k):
  basis.append(q);v=A.double()@q
  for _pass in range(2):
   for old in basis:v-=torch.dot(old,v)*old
  q=v/v.norm()
 Q=torch.stack(basis,dim=1)
 return Q@torch.linalg.lstsq(A.double()@Q,b.double()).solution

class RecurrenceTests(unittest.TestCase):
 def test_spd_and_indefinite_fixed_four_match_dense(self):
  for diagonal in ([1,2,3,4,6,8,10,13],[-9,-4,-2,-.3,.7,2,5,11]):
   A=torch.diag(torch.tensor(diagonal,dtype=torch.float64));b=torch.arange(1,9,dtype=torch.float32);calls=[]
   x,rows=c.minres(dictionary(b),operator(A,calls))
   torch.testing.assert_close(flatten(x).double(),dense_minimum_residual(A,b,4),atol=2e-6,rtol=2e-5)
   self.assertEqual(len(calls),4);self.assertEqual(len(set(calls)),2)
   self.assertEqual([r['iteration'] for r in rows],[1,2,3,4])
   self.assertEqual(rows[0]['beta_current'],float(b.double().norm()))
   self.assertAlmostEqual(rows[0]['alpha'],float((b.double()/b.double().norm())@(A@(b.double()/b.double().norm()))),places=5)
   for r in rows:
    self.assertAlmostEqual(r['cosine_rotation']**2+r['sine_rotation']**2,1)
    self.assertAlmostEqual(r['gamma'],math.hypot(r['gbar'],r['beta_next']))
    self.assertEqual(r['recurrence_residual_norm'],abs(r['phibar']))
    self.assertEqual(r['long_lived_cpu_vector_count'],5)
 def test_zero_initial_solution_one_step_formula_and_buffers(self):
  A=torch.diag(torch.arange(1,9,dtype=torch.float64));b=torch.arange(1,9,dtype=torch.float32)
  original=torch.zeros_like;allocations=[]
  def zero(v):allocations.append(v.numel());return original(v)
  with patch.object(c.torch,'zeros_like',side_effect=zero):x,rows=c.minres(dictionary(b),operator(A),1)
  Ab=A@b.double();expected=(b.double()@Ab)/(Ab@Ab)*b.double()
  torch.testing.assert_close(flatten(x).double(),expected,atol=2e-7,rtol=1e-6)
  self.assertEqual(len(allocations),8) # four dictionaries, two matrices apiece + owned RHS
 def test_exact_breakdown_and_nonfinite_fail(self):
  b=torch.tensor([1.,0.]);A=torch.eye(2)
  with self.assertRaisesRegex(ValueError,'Lanczos breakdown'):c.minres(dictionary(b),operator(A))
  with self.assertRaisesRegex(ValueError,'zero RHS'):c.minres(dictionary(torch.zeros(2)),operator(A))
  for value in (float('inf'),float('nan')):
   with self.assertRaises(ValueError):c.minres(dictionary(torch.tensor([1.,value])),operator(A))
  with self.assertRaises(ValueError):c.minres(dictionary(b),lambda r,b:dictionary(torch.full((2,),float('nan'))))
 def test_no_residual_early_stop(self):
  A=torch.diag(torch.tensor([1.,2.,4.,8.,16.,32.,64.,128.]));b=torch.tensor([1.,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]);calls=[]
  _,rows=c.minres(dictionary(b),operator(A,calls));self.assertEqual(len(calls),4)
  self.assertLess(rows[0]['relative_recurrence_residual'],.1)
 def test_fp64_reductions(self):
  v={'a':torch.tensor([[1e8,1.,-1e8]])}
  self.assertEqual(c.vector_norm(v),math.sqrt(math.fsum([1e16,1.,1e16])))
 def test_buffer_reuse_matches_non_inplace_short_recurrence(self):
  A=torch.diag(torch.tensor([-7.,-2.,-.5,.4,2.,3.,8.,12.],dtype=torch.float64));b=torch.arange(1,9,dtype=torch.float32)
  r1=torch.zeros_like(b);r2=b.clone();w1=torch.zeros_like(b);w2=torch.zeros_like(b);x=torch.zeros_like(b)
  beta=float(b.double().norm());oldb=0.;cs=-1.;sn=0.;dbar=0.;eps=0.;phibar=beta
  for k in range(4):
   v=r2.double()/beta;y=(A@v.float().double()).float().double()
   if k:y-=beta/oldb*r1.double()
   alpha=float(v@y);next_r=(y-alpha/beta*r2.double()).float();next_beta=float(next_r.double().norm())
   oldeps=eps;delta=cs*dbar+sn*alpha;gbar=sn*dbar-cs*alpha
   eps=sn*next_beta;dbar=-cs*next_beta;gamma=math.hypot(gbar,next_beta)
   cs=gbar/gamma;sn=next_beta/gamma;phi=cs*phibar;phibar=sn*phibar
   w=((v-oldeps*w1.double()-delta*w2.double())/gamma).float()
   x=(x.double()+phi*w.double()).float();r1,r2=r2,next_r;w1,w2=w2,w;oldb,beta=beta,next_beta
  actual,_=c.minres(dictionary(b),operator(A))
  torch.testing.assert_close(flatten(actual),x,atol=2e-7,rtol=2e-6)
 def test_alpha_uses_fp64_cancellation_sensitive_dot(self):
  rhs={str(i):torch.ones(1,1) for i in range(3)}
  def op(r,b):return {str(i):torch.tensor([[v]]) for i,v in enumerate([1e8,1.,-1e8])}
  _,rows=c.minres(rhs,op,1)
  self.assertAlmostEqual(rows[0]['alpha'],1/math.sqrt(3),places=7)
 def test_operator_denominator_same_direction_and_state_checks(self):
  base,model,spec=toy.models();eligible=c.a.discover(model,spec)
  current={n+'.weight':torch.ones_like(m.weight,device='cpu') for n,m in eligible}
  tokens=torch.tensor([[0,1,2],[1,2,0],[2,0,1],[0,2,1]])
  beta=c.vector_norm(current);seen=[];original=c.a.hvp_backend.true_hessian_vector_product
  def record(loss,p,v):seen.append(v);return original(loss,p,v)
  with patch.object(c.a.hvp_backend,'true_hessian_vector_product',side_effect=record),redirect_stdout(io.StringIO()):
   h=c.make_operator(model,eligible,tokens,spec)(current,beta)
  self.assertEqual(len(seen),2);self.assertIs(seen[0],seen[1]);self.assertEqual(len(h),196)
  expected=c.a.hvp_backend.true_hessian_vector_product(
   lambda:c.a.causal_token_loss_sum(model(input_ids=tokens,use_cache=False).logits,tokens,torch)/8,
   {n+'.weight':m.weight for n,m in eligible},{n:(v.double()/beta).float() for n,v in current.items()})
  for n in h:torch.testing.assert_close(h[n],expected[n],rtol=1e-5,atol=1e-8)
  def mutate(*args,**kw):model.model.offset.add_(1);return {n:torch.zeros_like(v) for n,v in current.items()}
  with patch.object(c.a,'accumulate_hessian_split',side_effect=mutate),self.assertRaisesRegex(ValueError,'changed'):
   c.make_operator(model,eligible,tokens,spec)(current,beta)
 def test_true_hvp_and_denominator_same_as_017(self):
  x=torch.tensor([.2,-.4],dtype=torch.float64,requires_grad=True);v=torch.tensor([.3,.7],dtype=torch.float64)
  fn=lambda y:(y**4).sum()+y.prod()
  actual=c.a.hvp_backend.true_hessian_vector_product(lambda:fn(x),{'x':x},{'x':v})['x']
  torch.testing.assert_close(actual,torch.autograd.functional.hessian(fn,x)@v,atol=1e-12,rtol=1e-12)
  self.assertEqual(c.FROZEN_SPEC['generic_loss']['total_prediction_tokens'],4096*127)
  self.assertEqual(c.FROZEN_SPEC['generic_loss']['batch_size'],8)
 def test_inventory_and_model_mutation(self):
  base,final,spec=toy.models();eligible=c.a.discover(final,spec);self.assertEqual(len(eligible),196)
  before=c.a.model_state_hashes(final,torch);final.model.offset.add_(1)
  with self.assertRaisesRegex(ValueError,'changed'):c.a.verify_model_unchanged(final,before,torch)

class IntegrityTests(unittest.TestCase):
 def rhs_fixture(self,root):
  spec=copy.deepcopy(c.FROZEN_SPEC);v=dictionary(torch.arange(1,9,dtype=torch.float32))
  spec['coordinates']=[{'name':n,'shape':list(t.shape)} for n,t in v.items()]
  paths={n:root/n for n in ('rhs','rhs_spec','rhs_manifest')}
  record=c.a.save_vector(paths['rhs'],v,spec['coordinates'])
  previous={'coordinates':spec['coordinates']};paths['rhs_spec'].write_text(json.dumps(previous));spec['rhs_spec_sha256']=c.a.sha256_file(paths['rhs_spec'])
  m={'attempt_id':spec['rhs_attempt_id'],'coordinates':spec['coordinates'],'generic_loss':spec['generic_loss'],
     'final_checkpoint_files':spec['canonical_checkpoint_files'],'corpus':spec['frozen_corpus'],
     'cleaned_rhs':{'definition':spec['rhs_definition'],**record},
     'input_source_hashes':{'experiments/attempts/'+spec['rhs_attempt_id']+'/spec.json':spec['rhs_spec_sha256']}}
  paths['rhs_manifest'].write_text(json.dumps(m));spec['rhs_manifest_sha256']=c.a.sha256_file(paths['rhs_manifest'])
  return paths,spec,m,v
 def test_rhs_serialized_raw_inventory_norm_and_manifest_strict(self):
  for kind in ('good','manifest','serialized','raw','norm','shape'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
    paths,spec,m,v=self.rhs_fixture(Path(d))
    if kind=='manifest':paths['rhs_manifest'].write_text('{}')
    elif kind=='serialized':paths['rhs'].write_bytes(paths['rhs'].read_bytes()+b'bad')
    elif kind in ('raw','norm','shape'):
     if kind=='raw':m['cleaned_rhs']['matrices'][0]['raw_sha256']='0'*64
     elif kind=='norm':m['cleaned_rhs']['norm']+=1
     else:m['coordinates'][0]['shape']=[2,2]
     paths['rhs_manifest'].write_text(json.dumps(m));spec['rhs_manifest_sha256']=c.a.sha256_file(paths['rhs_manifest'])
    if kind=='good':self.assertEqual(c.verify_rhs(paths,spec)['norm'],c.vector_norm(v))
    else:
     with self.assertRaises(Exception):c.verify_rhs(paths,spec)
 def test_solution_roundtrip_and_overwrite(self):
  with tempfile.TemporaryDirectory() as d:
   paths,spec,_,v=self.rhs_fixture(Path(d));record=c.a.vector_record(paths['rhs'],spec['coordinates'])
   with self.assertRaisesRegex(ValueError,'exists'):c.a.save_vector(paths['rhs'],v,spec['coordinates'])
   self.assertEqual(record['norm'],c.vector_norm(v))
 def test_smoke_no_outputs_and_fixed_workload(self):
  with tempfile.TemporaryDirectory() as d:
   paths,spec,_,v=self.rhs_fixture(Path(d));args=c.parse_args([]);args.smoke_only=True
   args.artifact_path=Path(d)/'solution';args.manifest_path=Path(d)/'manifest'
   A=torch.diag(torch.arange(1,9,dtype=torch.float64));calls=[];paths['final']=Path(d)/'final'
   with patch.object(c,'validate_inputs',return_value=(spec,paths,None,{}, {},lambda:None)),patch.object(c.a,'load_local_model'),patch.object(c.a,'discover'),patch.object(c,'make_operator',return_value=operator(A,calls)),redirect_stdout(io.StringIO()):c.construct(args)
   self.assertEqual(len(calls),1);self.assertFalse(args.artifact_path.exists());self.assertFalse(args.manifest_path.exists())
 def test_constructor_publication_determinism_and_four_products(self):
  with tempfile.TemporaryDirectory() as d:
   paths,spec,_,v=self.rhs_fixture(Path(d));paths['final']=Path(d)/'final'
   args=c.parse_args([]);args.artifact_path=Path(d)/'solution';args.manifest_path=Path(d)/'manifest'
   A=torch.diag(torch.arange(1,9,dtype=torch.float64));calls=[]
   hashes={str(args.spec_path):'a'*64,str(Path(c.__file__).resolve()):'b'*64}
   def run():
    with patch.object(c,'validate_inputs',return_value=(spec,paths,None,{},hashes,lambda:None)),patch.object(c.a,'load_local_model'),patch.object(c.a,'discover'),patch.object(c,'make_operator',return_value=operator(A,calls)):
     c.construct(args)
   run();self.assertEqual(len(calls),4);text=args.manifest_path.read_text()
   m=json.loads(text);self.assertEqual(m['iteration_count'],4);self.assertEqual(m['workload']['hvp_batches'],2048)
   self.assertEqual(m['solution'],c.a.vector_record(args.artifact_path,spec['coordinates']))
   with self.assertRaisesRegex(ValueError,'exists'):run()
   args.artifact_path.unlink();args.manifest_path.unlink();run();self.assertEqual(text,args.manifest_path.read_text())
 def test_allowlist_symlink_rejected_before_loading(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);link=root/'checkpoint';link.symlink_to(root/'forbidden')
   args=c.parse_args([])
   with patch.object(c,'input_paths',return_value={'final':link}),self.assertRaisesRegex(ValueError,'symlink'):
    c.validate_inputs(args)
 def test_real_synthetic_provenance_pipeline_and_recheck(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);fixture,corpus=toy.PipelineTests().provenance_fixture(root)
   oldargs,_,model,oldspec,tokens=fixture
   spec=copy.deepcopy(c.FROZEN_SPEC)
   for key in ('coordinates','canonical_checkpoint_files','generic_loss','frozen_corpus'):spec[key]=oldspec[key]
   old_manifest=root/'rhs_manifest';rhs_path=root/'cleaned_rhs'
   v={r['name']:(torch.arange(4).reshape(2,2).float()+i+1)*.001 for i,r in enumerate(spec['coordinates'])}
   record=c.a.save_vector(rhs_path,v,spec['coordinates'])
   spec['rhs_spec_sha256']=c.a.sha256_file(oldargs.spec_path)
   prior={'attempt_id':spec['rhs_attempt_id'],'coordinates':spec['coordinates'],'generic_loss':spec['generic_loss'],
          'final_checkpoint_files':spec['canonical_checkpoint_files'],'corpus':spec['frozen_corpus'],
          'cleaned_rhs':{'definition':spec['rhs_definition'],**record},
          'input_source_hashes':{'experiments/attempts/'+spec['rhs_attempt_id']+'/spec.json':spec['rhs_spec_sha256']}}
   old_manifest.write_text(json.dumps(prior));spec['rhs_manifest_sha256']=c.a.sha256_file(old_manifest)
   spec['inputs']={k:str(p) for k,p in {'final':oldargs.final_directory,'tokens':oldargs.tokens_path,
    'corpus_spec':oldargs.corpus_spec_path,'corpus_manifest':oldargs.corpus_manifest_path,
    'rhs':rhs_path,'rhs_spec':oldargs.spec_path,'rhs_manifest':old_manifest}.items()}
   args=c.parse_args([]);args.spec_path=root/'018spec';args.spec_path.write_text(json.dumps(spec))
   args.artifact_path=root/'solution';args.manifest_path=root/'manifest';args.device='cpu'
   with patch.object(c,'FROZEN_SPEC',spec),patch.object(c.a,'FROZEN_CORPUS_SPEC',corpus),patch.object(c.a,'load_local_model',return_value=model),redirect_stdout(io.StringIO()):
    values=c.validate_inputs(args)
    c.construct(args)
    self.assertEqual(len(json.loads(args.manifest_path.read_text())['iterations']),4)
    args.spec_path.write_text(args.spec_path.read_text()+' ')
    with self.assertRaisesRegex(ValueError,'changed'):values[-1]()
 def test_firewall_spec_and_source(self):
  spec=c.FROZEN_SPEC
  self.assertNotIn('base',spec['inputs']);self.assertEqual(spec['solver']['iteration_count'],4)
  text=Path(c.__file__).read_text()
  for forbidden in ('oracle_adl.pt','models/base','adapter','evaluate_attempt','DeltaW','scipy','linalg.inv'):
   self.assertNotIn(forbidden,text)
  args=c.parse_args([])
  with tempfile.TemporaryDirectory() as d:
   args.spec_path=Path(d)/'spec';changed=copy.deepcopy(spec);changed['inputs']['final']='/forbidden/models/base';args.spec_path.write_text(json.dumps(changed))
   with self.assertRaisesRegex(ValueError,'spec mismatch'):c.validate_inputs(args)
 def test_input_mutation(self):
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'input';path.write_text('a');digest=c.a.sha256_file(path);path.write_text('b')
   with self.assertRaisesRegex(ValueError,'SHA256'):c.require_hash(path,digest)

if __name__=='__main__':unittest.main()
