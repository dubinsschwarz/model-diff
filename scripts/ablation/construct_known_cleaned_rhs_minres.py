#!/usr/bin/env python3
"""Fixed-k unpreconditioned symmetric Paige–Saunders MINRES at the final checkpoint."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open
PROJECT=Path(__file__).resolve().parents[2]
SUPPORT_PATH=Path(__file__).with_name('attempt018_hvp_support.py')
_loader=importlib.util.spec_from_file_location('minres018_support',SUPPORT_PATH)
a=importlib.util.module_from_spec(_loader);_loader.loader.exec_module(a)
# Frozen specification is embedded below before the CLI entry point.


def vector_norm(values):
    squares=[]
    for value in values.values():
        matrix=value.detach().to(device='cpu',dtype=torch.float64)
        squares.append(float(matrix.square().sum()))
    result=math.sqrt(math.fsum(squares))
    if not math.isfinite(result):raise ValueError('Nonfinite vector norm')
    return result


def check_vector(values,reference,cpu=False):
    if list(values)!=list(reference):raise ValueError('Vector coordinate order mismatch')
    for name,value in values.items():
        if (value.dtype!=torch.float32 or value.shape!=reference[name].shape or not value.is_contiguous()
            or value.requires_grad or (cpu and value.device.type!='cpu') or not bool(torch.isfinite(value).all())):
            raise ValueError('Invalid recurrence vector: '+name)


def minres(rhs, operator, iterations=4):
    """Own rhs storage. operator(r,beta) applies H to FP32(CPU_FP64(r)/beta).

    Paige–Saunders unpreconditioned recurrence, initial cs=-1, sn=0.
    r_next = A*v - (beta/old_beta)*r_prev - (alpha/beta)*r_current.
    delta=cs*dbar+sn*alpha; gbar=sn*dbar-cs*alpha;
    eps_next=sn*beta_next; dbar_next=-cs*beta_next.
    gamma=hypot(gbar,beta_next); cs=gbar/gamma; sn=beta_next/gamma.
    w_next=(v-eps_old*w_prev-delta*w_current)/gamma; x+=phi*w_next.
    Old r_prev and w_prev buffers are overwritten then swapped, never a basis.
    """
    if iterations not in (1,4):raise ValueError('Only frozen k=4 or one smoke iteration')
    check_vector(rhs,rhs,cpu=True)
    r_current=rhs
    r_prev={n:torch.zeros_like(v) for n,v in rhs.items()}
    w_prev={n:torch.zeros_like(v) for n,v in rhs.items()}
    w_current={n:torch.zeros_like(v) for n,v in rhs.items()}
    x={n:torch.zeros_like(v) for n,v in rhs.items()}
    beta1=vector_norm(rhs)
    if beta1==0:raise ValueError('Exact initial Lanczos breakdown: zero RHS')
    beta=beta1;old_beta=0.;dbar=0.;eps=0.;phibar=beta1;cs=-1.;sn=0.
    records=[]
    for k in range(1,iterations+1):
        # Only matrix-sized CPU temporaries; no independent normalized CPU vector.
        input_squares=[]
        for value in r_current.values():
            v=(value.double()/beta).float()
            input_squares.append(float(v.double().square().sum()))
        product=operator(r_current,beta)
        try:
            check_vector(product,r_current)
            alpha_terms=[];output_squares=[]
            for name in r_current:
                y=product[name].detach().to(device='cpu',dtype=torch.float64)
                output_squares.append(float(y.square().sum()))
                if k>1:y.sub_((beta/old_beta)*r_prev[name].double())
                alpha_terms.append(float(((r_current[name].double()/beta)*y).sum()))
            alpha=math.fsum(alpha_terms)
            for name in r_current:
                y=product.pop(name).detach().to(device='cpu',dtype=torch.float64)
                if k>1:y.sub_((beta/old_beta)*r_prev[name].double())
                y.sub_((alpha/beta)*r_current[name].double())
                r_prev[name].copy_(y.float())
        finally:
            product.clear()
        next_beta=vector_norm(r_prev)
        if next_beta==0 and k<4:
            raise ValueError(f'Exact Lanczos breakdown at iteration {k}: alpha={alpha}, beta_next=0')
        old_eps=eps
        delta=cs*dbar+sn*alpha;gbar=sn*dbar-cs*alpha
        eps=sn*next_beta;dbar=-cs*next_beta
        gamma=math.hypot(gbar,next_beta)
        if gamma==0 or not math.isfinite(gamma):raise ValueError(f'MINRES rotation breakdown at iteration {k}')
        cs=gbar/gamma;sn=next_beta/gamma
        phi=cs*phibar;phibar=sn*phibar
        scalars=[alpha,next_beta,delta,gbar,eps,dbar,gamma,cs,sn,phi,phibar]
        if not all(math.isfinite(v) for v in scalars):raise ValueError('Nonfinite recurrence scalar')
        for name in x:
            update=(r_current[name].double()/beta-old_eps*w_prev[name].double()-delta*w_current[name].double())/gamma
            w_prev[name].copy_(update.float())
            x[name].copy_((x[name].double()+phi*w_prev[name].double()).float())
        check_vector(w_prev,x,cpu=True)
        x_norm=vector_norm(x)
        records.append({'iteration':k,'alpha':alpha,'beta_current':beta,'beta_next':next_beta,
            'delta':delta,'gbar':gbar,'old_epsilon':old_eps,'epsilon_next':eps,'dbar_next':dbar,
            'gamma':gamma,'cosine_rotation':cs,'sine_rotation':sn,'phi':phi,'phibar':phibar,
            'recurrence_residual_norm':abs(phibar),'relative_recurrence_residual':abs(phibar)/beta1,
            'solution_norm':x_norm,'hvp_input_norm':math.sqrt(math.fsum(input_squares)),
            'hvp_output_norm':math.sqrt(math.fsum(output_squares)),
            'finite_checks_passed':True,'exact_lanczos_breakdown':next_beta==0,
            'long_lived_cpu_vector_count':5})
        r_prev,r_current=r_current,r_prev
        w_prev,w_current=w_current,w_prev
        old_beta,beta=beta,next_beta
    return x,records


def require_hash(path,digest):
    if a.sha256_file(path)!=a.require_sha256(digest,str(path)):raise ValueError('Frozen SHA256 mismatch: '+str(path))


def output_absent(args):
    for path in (args.artifact_path,args.manifest_path):
        if path.exists() or path.is_symlink():raise ValueError('Output already exists: '+str(path))


def input_paths(args):
    # No directory override for model/corpus: inputs are the frozen allowlist.
    def resolve(value):
        path=Path(value);return path if path.is_absolute() else PROJECT/path
    return {name:resolve(value) for name,value in FROZEN_SPEC['inputs'].items()}


def verify_rhs(paths,spec):
    require_hash(paths['rhs_manifest'],spec['rhs_manifest_sha256'])
    require_hash(paths['rhs_spec'],spec['rhs_spec_sha256'])
    previous=a.load_json_object(paths['rhs_spec'],'RHS spec')
    manifest=a.load_json_object(paths['rhs_manifest'],'RHS manifest')
    if (manifest['attempt_id']!=spec['rhs_attempt_id'] or manifest['coordinates']!=spec['coordinates']
        or previous['coordinates']!=spec['coordinates'] or manifest['generic_loss']!=spec['generic_loss']
        or manifest['final_checkpoint_files']!=spec['canonical_checkpoint_files']
        or manifest['corpus']!=spec['frozen_corpus'] or manifest['cleaned_rhs']['definition']!=spec['rhs_definition']):
        raise ValueError('RHS provenance/semantics mismatch')
    suffix='experiments/attempts/'+spec['rhs_attempt_id']+'/spec.json'
    linked=[v for p,v in manifest['input_source_hashes'].items() if p.endswith(suffix)]
    if linked!=[spec['rhs_spec_sha256']]:raise ValueError('RHS spec linkage mismatch')
    actual=a.vector_record(paths['rhs'],spec['coordinates'])
    expected={k:manifest['cleaned_rhs'][k] for k in ('serialized_sha256','matrices','norm')}
    if actual!=expected:raise ValueError('RHS serialized/raw/inventory/norm mismatch')
    return actual


def validate_inputs(args):
    spec=a.load_json_object(args.spec_path,'Attempt018 spec')
    if spec!=FROZEN_SPEC:raise ValueError('Frozen Attempt018 spec mismatch')
    paths=input_paths(args)
    # Inputs cannot be redirected through symlinks to a non-allowlisted location.
    for name,path in paths.items():
        if path.resolve()!=path.absolute():raise ValueError('Input symlink outside frozen allowlist: '+name)
    hashes={str(args.spec_path):a.sha256_file(args.spec_path),str(Path(__file__).resolve()):a.sha256_file(Path(__file__).resolve())}
    for path,digest in spec['sources'].items():
        p=PROJECT/path;require_hash(p,digest)
        if LOADED_SOURCES[path]!=digest:raise ValueError('Loaded source mismatch')
        hashes[str(p)]=digest
    for name,key in [('corpus_spec','corpus_spec_sha256'),('corpus_manifest','corpus_manifest_sha256')]:
        require_hash(paths[name],spec['frozen_corpus'][key]);hashes[str(paths[name])]=spec['frozen_corpus'][key]
    files=a.checkpoint_file_records(paths['final'])
    if files!=spec['canonical_checkpoint_files']:raise ValueError('Final checkpoint inventory/hash mismatch')
    corpus=a.verify_corpus_manifest(paths['corpus_manifest'],paths['corpus_spec'],paths['tokens'],paths['final'],a.load_corpus_spec(paths['corpus_spec']))
    if any(corpus[k]!=spec['frozen_corpus'][k] for k in ('serialized_sha256','raw_tensor_sha256')):raise ValueError('Corpus hash mismatch')
    rhs=verify_rhs(paths,spec)
    for name in ('tokens','rhs','rhs_spec','rhs_manifest'):hashes[str(paths[name])]=a.sha256_file(paths[name])
    for out in (args.artifact_path,args.manifest_path):
        if out.resolve().is_relative_to(paths['final'].resolve()) or out.resolve() in {Path(p).resolve() for p in hashes}:
            raise ValueError('Output overlaps frozen input')
    if args.artifact_path.resolve()==args.manifest_path.resolve():raise ValueError('Outputs overlap')
    tokens=a.load_corpus_tokens(paths['tokens'],spec['frozen_corpus']['raw_tensor_sha256'],spec['generic_loss']['sample_count'],spec['generic_loss']['sequence_length'],torch)
    def recheck():a.verify_unchanged(paths['final'],files,{Path(p):v for p,v in hashes.items()})
    recheck()
    return spec,paths,tokens,rhs,hashes,recheck


def make_operator(model,eligible,tokens,spec):
    before=a.model_state_hashes(model,torch)
    def operator(current,beta):
        a.verify_model_unchanged(model,before,torch)
        direction={}
        try:
            for name,module in eligible:
                key=name+'.weight'
                direction[key]=(current[key].double()/beta).float().contiguous().to(module.weight.device)
            result=a.accumulate_hessian_split(model,tokens,eligible,direction,spec['generic_loss'],torch,
                progress=lambda i,n:print(f'HVP batch {i}/{n}',flush=True) if i%128==0 or i==n else None)
        finally:
            direction.clear();model.zero_grad(set_to_none=True)
        a.verify_model_unchanged(model,before,torch)
        return result
    return operator


def construct(args):
    output_absent(args)
    spec,paths,tokens,rhs_record,hashes,recheck=validate_inputs(args)
    model=a.load_local_model(paths['final'],args.device,torch);eligible=a.discover(model,spec)
    # Clone mapped matrices once into owned recurrence storage; no second CPU RHS.
    with safe_open(str(paths['rhs']),framework='pt',device='cpu') as stored:
        rhs={r['name']:stored.get_tensor(r['name']).clone().contiguous() for r in spec['coordinates']}
    solution,iterations=minres(rhs,make_operator(model,eligible,tokens,spec),1 if args.smoke_only else 4)
    rhs.clear();recheck()
    if args.smoke_only:
        print(json.dumps({'smoke_only':True,'full_corpus_hvps':1,'cpu_vectors':5,'diagnostics':iterations},sort_keys=True))
        solution.clear();return
    record=a.save_vector(args.artifact_path,solution,spec['coordinates']);solution.clear()
    manifest={'format_version':1,'attempt_id':spec['attempt_id'],'purpose':spec['purpose'],'hash_algorithm':'sha256',
        'spec_sha256':hashes[str(args.spec_path)],'constructor_script_sha256':hashes[str(Path(__file__).resolve())],
        'input_source_hashes':hashes,'final_checkpoint_files':spec['canonical_checkpoint_files'],'corpus':spec['frozen_corpus'],
        'rhs_manifest_sha256':spec['rhs_manifest_sha256'],'rhs':rhs_record,'coordinates':spec['coordinates'],
        'solver':spec['solver'],'iteration_count':4,'iterations':iterations,'solution':record,'workload':spec['workload'],
        'model_state_unchanged_before_after_each_hvp':True}
    recheck()
    if a.vector_record(args.artifact_path,spec['coordinates'])!=record:raise ValueError('Solution changed during publication')
    args.manifest_path.parent.mkdir(parents=True,exist_ok=True);a.write_manifest(args.manifest_path,manifest)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path',type=Path,default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'spec.json')
    parser.add_argument('--artifact-path',type=Path,default=Path(FROZEN_SPEC['outputs']['artifact']))
    parser.add_argument('--manifest-path',type=Path,default=PROJECT/FROZEN_SPEC['outputs']['manifest'])
    parser.add_argument('--device',default='cuda');parser.add_argument('--smoke-only',action='store_true')
    return parser.parse_args(argv)


FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '018_known_cleaned_rhs_endpoint_hessian_minres_k4_prefix0_27',
 'purpose': 'post_oracle_known_cleaned_rhs_endpoint_hessian_reverse_diagnostic_not_blind_recovery',
 'interpretation': 'Fixed direct reverse test; forward closeness cannot gate construction or establish '
                   'inverse accuracy.',
 'coordinates': [{'name': 'model.layers.0.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.0.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.0.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.0.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.0.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.0.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.0.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.1.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.1.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.1.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.1.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.1.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.1.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.1.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.10.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.10.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.10.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.10.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.10.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.10.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.10.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.11.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.11.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.11.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.11.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.11.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.11.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.11.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.12.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.12.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.12.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.12.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.12.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.12.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.12.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.13.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.13.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.13.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.13.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.13.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.13.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.13.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.14.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.14.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.14.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.14.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.14.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.14.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.14.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.15.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.15.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.15.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.15.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.15.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.15.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.15.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.16.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.16.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.16.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.16.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.16.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.16.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.16.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.17.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.17.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.17.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.17.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.17.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.17.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.17.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.18.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.18.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.18.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.18.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.18.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.18.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.18.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.19.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.19.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.19.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.19.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.19.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.19.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.19.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.2.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.2.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.2.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.2.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.2.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.2.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.2.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.20.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.20.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.20.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.20.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.20.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.20.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.20.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.21.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.21.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.21.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.21.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.21.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.21.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.21.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.22.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.22.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.22.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.22.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.22.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.22.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.22.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.23.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.23.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.23.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.23.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.23.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.23.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.23.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.24.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.24.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.24.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.24.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.24.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.24.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.24.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.25.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.25.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.25.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.25.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.25.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.25.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.25.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.26.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.26.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.26.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.26.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.26.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.26.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.26.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.27.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.27.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.27.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.27.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.27.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.27.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.27.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.3.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.3.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.3.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.3.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.3.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.3.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.3.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.4.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.4.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.4.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.4.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.4.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.4.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.4.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.5.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.5.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.5.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.5.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.5.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.5.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.5.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.6.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.6.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.6.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.6.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.6.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.6.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.6.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.7.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.7.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.7.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.7.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.7.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.7.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.7.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.8.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.8.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.8.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.8.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.8.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.8.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.8.self_attn.v_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.9.mlp.down_proj.weight', 'shape': [2048, 6144]},
                 {'name': 'model.layers.9.mlp.gate_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.9.mlp.up_proj.weight', 'shape': [6144, 2048]},
                 {'name': 'model.layers.9.self_attn.k_proj.weight', 'shape': [1024, 2048]},
                 {'name': 'model.layers.9.self_attn.o_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.9.self_attn.q_proj.weight', 'shape': [2048, 2048]},
                 {'name': 'model.layers.9.self_attn.v_proj.weight', 'shape': [1024, 2048]}],
 'canonical_checkpoint_files': [{'path': 'chat_template.jinja',
                                 'size_bytes': 4168,
                                 'sha256': 'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'},
                                {'path': 'config.json',
                                 'size_bytes': 1416,
                                 'sha256': '3ef26f3c99bbc0bb0ef65b729429f716fb2760a23b71ffd2eee39646f703bd0f'},
                                {'path': 'generation_config.json',
                                 'size_bytes': 214,
                                 'sha256': '893b0dccf83626cdfdc498e5b2a255a935b4c3c693350aed18f7003c1a3f3de9'},
                                {'path': 'model.safetensors',
                                 'size_bytes': 6882335328,
                                 'sha256': 'f6989dd3445cd339041ab654a6da463e0a80b2d52dd2c01f67f683b1c30ea10f'},
                                {'path': 'tokenizer.json',
                                 'size_bytes': 11422650,
                                 'sha256': 'be75606093db2094d7cd20f3c2f385c212750648bd6ea4fb2bf507a6a4c55506'},
                                {'path': 'tokenizer_config.json',
                                 'size_bytes': 692,
                                 'sha256': '1cc816812993bff176eb4f7495433b736f06fba9b6e7b05cac7b4a1780650c95'}],
 'frozen_corpus': {'corpus_spec_sha256': 'e514820e7def99e9287c32de4b1279f28f6ff6c2e09749a8dfa5d8b14c311f0f',
                   'corpus_manifest_sha256': '8b3ea49228dbb106c2b8a7c849c4f27e021fa3e992800444f43e260a6b055230',
                   'serialized_sha256': '54283d72aa3075e2867ead86e538991d7284ede327ca67871e73bfc581e4025d',
                   'raw_tensor_sha256': '4613982be77342a89b35616d68e21d9ff66d5a983af7e82cb6fa1eb159457eae'},
 'generic_loss': {'type': 'causal_next_token_cross_entropy',
                  'sample_count': 4096,
                  'sequence_length': 128,
                  'predictions_per_sample': 127,
                  'total_prediction_tokens': 520192,
                  'batch_size': 8,
                  'number_of_batches': 512,
                  'label_alignment': 'logits_positions_0_through_126_predict_tokens_1_through_127',
                  'reduction': 'sum_token_cross_entropy_divided_by_total_prediction_tokens',
                  'masking': 'causal_model_mask_only',
                  'mixed_precision': False,
                  'dropout': False,
                  'optimizer': False,
                  'weight_decay': False,
                  'gradient_clipping': False},
 'rhs_attempt_id': '017_known_base_full_support_endpoint_hessian_forward_prefix0_27',
 'rhs_spec_sha256': '7b3c8c70f337b3b6298d05b344ae051e4845abe0ed67a00c9cdce9c8d6dc71b0',
 'rhs_manifest_sha256': 'e8819967f06724959f85983a6f16f67407d27e3ee23bf3b3f2208fa4dfab0496',
 'rhs_definition': 'contiguous_FP32(CPU_FP64(g1)-CPU_FP64(g0))',
 'inputs': {'final': '/root/model-diff-scratch/models/merged',
            'tokens': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
            'corpus_spec': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json',
            'corpus_manifest': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json',
            'rhs': '/root/model-diff-scratch/artifacts/attempt017_full_support_hessian_forward/cleaned_rhs.safetensors',
            'rhs_spec': 'experiments/attempts/017_known_base_full_support_endpoint_hessian_forward_prefix0_27/spec.json',
            'rhs_manifest': 'experiments/attempts/017_known_base_full_support_endpoint_hessian_forward_prefix0_27/construction-manifest.json'},
 'sources': {'scripts/ablation/attempt018_hvp_support.py': 'c1f5f0cc1f18f2021023981c542acd41ff3c97906b6414af9b3933a60b72a09d',
             'scripts/ablation/smoke_test_generic_hessian_curvature_probe.py': 'c23f22b27b41ee5c0610457ef389f3eeb3264d9b410c66e6ecfa796a417ebf12'},
 'solver': {'method': 'symmetric_Paige_Saunders_MINRES_short_recurrence',
            'iteration_count': 4,
            'initial_solution': 'zero',
            'initial_rotation': {'cs': -1, 'sn': 0},
            'preconditioner': 'identity',
            'shift': 0,
            'damping': 0,
            'adaptive_stopping': False,
            'spectral_filter': False,
            'exact_breakdown_before_iteration_4': 'fail_closed',
            'recurrence_vector_dtype': 'CPU_contiguous_FP32',
            'scalar_reductions': 'matrixwise_CPU_FP64_math_fsum',
            'matrix_update_arithmetic': 'CPU_FP64_then_one_FP32_cast',
            'hvp_direction': 'FP32(CPU_FP64(r_current)/beta)_same_for_all_512_batches',
            'long_lived_vectors': 5,
            'store_basis': False,
            'store_intermediate_iterates': False,
            'fallback': False},
 'workload': {'full_hvps': 4, 'hvp_batches': 2048, 'smoke_full_hvps': 1, 'smoke_hvp_batches': 512},
 'outputs': {'artifact': '/root/model-diff-scratch/artifacts/attempt018_known_cleaned_rhs_minres_k4/solution.safetensors',
             'manifest': 'experiments/attempts/018_known_cleaned_rhs_endpoint_hessian_minres_k4_prefix0_27/construction-manifest.json',
             'overwrite': False,
             'temporary_vectors': False,
             'timestamps': False,
             'host_metadata': False,
             'gpu_metadata': False},
 'information_policy': 'frozen_input_allowlist_only; constructor_has_no_evaluator_dependency; '
                       'no_parameter_tuning'}
LOADED_SOURCES={p:a.sha256_file(PROJECT/p) for p in FROZEN_SPEC['sources']}

if __name__=='__main__':construct(parse_args())
