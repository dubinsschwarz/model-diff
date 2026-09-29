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

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / 'scripts/ablation/construct_ritz_residual_filtered_inverse.py'
MODULE = importlib.util.spec_from_file_location('attempt012', SCRIPT)
a = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(a)
SPEC_PATH = PROJECT / 'experiments/attempts' / a.FROZEN_SPEC['attempt_id'] / 'spec.json'
OLD_DIR = PROJECT / 'experiments/attempts/011_generic_hessian_krylov_k4_prefix0_13'

class SevenLinearBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_norm = torch.nn.LayerNorm(4)
        self.self_attn = torch.nn.ModuleDict({
            name: torch.nn.Linear(4, 4, bias=True)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        })
        self.mlp = torch.nn.ModuleDict({
            name: torch.nn.Linear(4, 4, bias=True)
            for name in ("gate_proj", "up_proj", "down_proj")
        })

class MockQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(10, 4)
        self.model.layers = torch.nn.ModuleList([SevenLinearBlock() for _ in range(28)])
        self.model.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 10)
        self.config = SimpleNamespace(model_type="qwen3", num_hidden_layers=28)

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

def toy_manifest(theta=None, beta=1.):
    v = torch.tensor([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float64) / 2
    theta = torch.tensor([-.01, 2., 4., 8.] if theta is None else theta, dtype=torch.float64)
    b = v @ torch.diag(theta) @ v.T
    values, vectors = torch.linalg.eigh(b)
    return {'projected_hessian': {'B4': b.tolist(), 'beta_4': beta, 'ritz_values': values.tolist(),
                'last_step_only_ritz_residual_estimates': (beta * vectors[-1].abs()).tolist()},
            'gradient_norm': 3., 'orthogonality': {'gram_matrix': torch.eye(4).tolist()}}


class SpectralTests(unittest.TestCase):
    def test_truncated_inverse_coefficients_and_residual_formula(self):
        m = toy_manifest()
        result = a.spectral_candidate(m, a.FROZEN_SPEC, torch)
        b = torch.tensor(m['projected_hessian']['B4'], dtype=torch.float64)
        theta, v = torch.linalg.eigh(b)
        rho = v[-1].abs()
        self.assertEqual(result['keep_mask'], [False, True, True, True])
        torch.testing.assert_close(torch.tensor(result['residual_estimates'], dtype=torch.float64), rho, rtol=0, atol=0)
        inverse = sum(torch.outer(v[:, i], v[:, i]) / theta[i] for i in (1, 2, 3))
        c = inverse[:, 0]
        torch.testing.assert_close(torch.tensor(result['coefficients'], dtype=torch.float64), c, rtol=1e-14, atol=1e-14)
        self.assertAlmostEqual(result['natural_norm_from_frozen_gram'], 3 * float(c.norm()))
        self.assertGreater(result['gamma_from_frozen_gram'], 0)
        self.assertAlmostEqual(result['gamma_from_frozen_gram'] * result['natural_norm_from_frozen_gram'], 3)
        self.assertEqual(result['filter_values'][0], 0.)

    def test_equality_is_dropped_and_resolved_negative_mode_kept(self):
        m = toy_manifest()
        m['projected_hessian'] = {'B4': torch.diag(torch.tensor([-4., 0., 2., 8.])).tolist(),
             'beta_4': 8., 'ritz_values': [-4., 0., 2., 8.],
             'last_step_only_ritz_residual_estimates': [0., 0., 0., 8.]}
        r = a.spectral_candidate(m, a.FROZEN_SPEC, torch)
        self.assertEqual(r['keep_mask'], [True, False, True, False])
        self.assertEqual(r['filter_values'], [-.25, 0., .5, 0.])
        self.assertEqual(r['coefficients'], [-.25, 0., 0., 0.])

    def test_no_retained_mode_fails_without_fallback(self):
        m = toy_manifest(beta=100)
        with self.assertRaisesRegex(ValueError, 'No residual-resolved'):
            a.spectral_candidate(m, a.FROZEN_SPEC, torch)

    def test_invalid_spectral_metadata_and_frozen_residual_mismatch(self):
        for key, value in [('beta_4', float('nan')), ('ritz_values', [0.] * 4),
                           ('last_step_only_ritz_residual_estimates', [0.] * 4),
                           ('B4', [[1.]])]:
            m = toy_manifest(); m['projected_hessian'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.spectral_candidate(m, a.FROZEN_SPEC, torch)
        m = toy_manifest(); m['gradient_norm'] = 0
        with self.assertRaises(ValueError):
            a.spectral_candidate(m, a.FROZEN_SPEC, torch)

    def test_frozen_metadata_reproduction_and_determinism(self):
        _, m = a.validate_provenance(OLD_DIR / 'spec.json', OLD_DIR / 'construction-manifest.json', a.FROZEN_SPEC)
        r = a.spectral_candidate(m, a.FROZEN_SPEC, torch)
        self.assertEqual(r['keep_mask'], [False, True, True, True])
        self.assertAlmostEqual(r['natural_norm_from_frozen_gram'], .01198078233002044, places=14)
        self.assertAlmostEqual(r['gamma_from_frozen_gram'], 924.2044382825063, places=9)
        self.assertEqual(r, a.spectral_candidate(m, a.FROZEN_SPEC, torch))
        json.dumps(r, allow_nan=False)

    def test_spec_firewall_and_no_curvature_or_inverse_fallback(self):
        self.assertEqual(a.load_spec(SPEC_PATH), a.FROZEN_SPEC)
        source = SCRIPT.read_text()
        for forbidden in ('autograd.grad', '.backward(', 'linalg.inv(', 'linalg.pinv(', 'krylov_probe.pt',
                          'oracle_adl', 'evaluation.json', 'models/base', 'adapter', 'load_corpus_spec', 'importlib'):
            self.assertNotIn(forbidden, source)
        tree = ast.parse(source)
        functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertFalse(any('hessian_vector' in name or 'gradient' in name for name in functions))
        settings = a.FROZEN_SPEC['spectral_filter']
        self.assertEqual(settings['keep'], 'abs(theta_i) > rho_i')
        self.assertFalse(any('damping' in key or 'multiplier' in key for key in settings))
        self.assertEqual(a.FROZEN_SPEC['workload'], dict(gradient_batches=0, hvp_batches=0, jvp_directions=1, jvp_batches=313))


class ConstructionTests(unittest.TestCase):
    def fixture(self, root):
        args = SimpleNamespace(attempt_spec_path=root/'spec.json', attempt011_spec_path=root/'old-spec.json',
            attempt011_manifest_path=root/'old-manifest.json', basis_directory=root/'basis',
            merged_model_dir=root/'merged', probe_path=root/'probe.pt', artifact_path=root/'out/response.pt',
            construction_manifest_path=root/'result.json', device='cpu')
        args.merged_model_dir.mkdir(); (args.merged_model_dir/'config.json').write_text('{}')
        args.basis_directory.mkdir()
        model = TinyModel()
        model.config = SimpleNamespace(hidden_size=2)
        spec = copy.deepcopy(a.FROZEN_SPEC)
        spec['eligible_tensors']['expected_matrix_count'] = 1
        spec['readout']['hidden_size'] = 2
        spec['outputs']['shape'] = [3, 2]
        spec['probe'].update(sample_count=5, sequence_length=3, batch_size=2)
        spec['diagnostic_probe'] = dict(start=0, stop=5, sample_count=5, batch_size=2)
        spec['canonical_checkpoint_files'] = a.checkpoint_file_records(args.merged_model_dir)
        tokens = torch.arange(15).reshape(5, 3)
        torch.save(tokens, args.probe_path)
        spec['probe']['serialized_sha256'] = a.sha256_file(args.probe_path)
        spec['probe']['raw_tensor_sha256'] = a.sha256_raw_int64_tensor(tokens, torch)
        previous = json.loads((OLD_DIR/'spec.json').read_text())
        for key in ('model', 'eligible_tensors', 'readout', 'jvp', 'probe', 'canonical_checkpoint_files'):
            previous[key] = spec[key]
        args.attempt011_spec_path.write_text(json.dumps(previous))
        spec['frozen_attempt011']['spec_sha256'] = a.sha256_file(args.attempt011_spec_path)
        m = toy_manifest()
        m.update(attempt_id=previous['attempt_id'], attempt_spec_sha256=spec['frozen_attempt011']['spec_sha256'],
                 source_checkpoint={'files': spec['canonical_checkpoint_files']}, probe=spec['probe'],
                 jvp=spec['jvp'], readout=spec['readout'],
                 eligible_parameters=['model.layers.0.proj.weight'],
                 matrices=[{'name': 'model.layers.0.proj.weight', 'shape': [2, 2]}],
                 basis_files=[])
        for key in ('krylov', 'hessian', 'splits'):
            m[key] = previous[key]
        for i in range(4):
            path = args.basis_directory/f'q{i+1}.safetensors'
            save_file({'model.layers.0.proj.weight': torch.eye(4)[i].reshape(2, 2).contiguous()}, str(path))
            m['basis_files'].append(dict(direction=i+1, filename=path.name, dtype='float32',
                                        size_bytes=path.stat().st_size, sha256=a.sha256_file(path)))
        args.attempt011_manifest_path.write_text(json.dumps(m))
        spec['frozen_attempt011']['manifest_sha256'] = a.sha256_file(args.attempt011_manifest_path)
        args.attempt_spec_path.write_text(json.dumps(spec))
        return args, spec, m, model

    def run_fixture(self, args, spec, model):
        with patch.object(a, 'FROZEN_SPEC', spec), patch.object(a, 'load_local_model', return_value=model), \
             patch.object(a, 'discover_eligible_linear_weights', return_value=[('model.layers.0.proj', model.model.layers[0].proj)]), redirect_stdout(io.StringIO()):
            a.construct(args)

    def test_complete_direct_combined_jvp_hashes_immutability_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, m, model = self.fixture(Path(directory))
            before = a.model_state_hashes(model, torch)
            with patch.object(a, 'run_readout_jvp', wraps=a.run_readout_jvp) as jvp:
                self.run_fixture(args, spec, model)
            self.assertEqual(jvp.call_count, 3)
            self.assertEqual([call.args[1].shape[0] for call in jvp.call_args_list], [2, 2, 1])
            self.assertEqual(before, a.model_state_hashes(model, torch))
            artifact = torch.load(args.artifact_path, weights_only=True)
            result = json.loads(args.construction_manifest_path.read_text())
            r = a.spectral_candidate(m, spec, torch)
            vector = (torch.tensor(r['coefficients'], dtype=torch.float64) * 3 * r['gamma_from_frozen_gram']).float().reshape(2,2)
            inputs = torch.arange(15).reshape(5,3).float().unsqueeze(-1).repeat(1,1,2) * .1 + model.model.offset
            expected = torch.nn.functional.linear(inputs, vector).double().mean(0).float()
            torch.testing.assert_close(artifact['inverse_response'], expected, rtol=1e-6, atol=1e-6)
            self.assertEqual(set(artifact), {'metadata','merged_mean','inverse_response'})
            for name in spec['outputs']['tensors']:
                value = artifact[name]
                self.assertEqual(value.dtype, torch.float32)
                self.assertEqual(list(value.shape), [3,2])
                self.assertTrue(value.is_contiguous())
                self.assertEqual(result['artifact']['raw_tensors_sha256'][name], a.sha256_raw_float32_tensor(value, torch))
            self.assertEqual(result['artifact']['serialized_sha256'], a.sha256_file(args.artifact_path))
            text = args.construction_manifest_path.read_text()
            args.artifact_path.unlink(); args.construction_manifest_path.unlink()
            self.run_fixture(args, spec, model)
            self.assertEqual(text, args.construction_manifest_path.read_text())
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.run_fixture(args, spec, model)

    def test_basis_hash_verification_before_loading_model(self):
        with tempfile.TemporaryDirectory() as directory:
            args, spec, m, model = self.fixture(Path(directory))
            path = args.basis_directory/'q2.safetensors'
            content = bytearray(path.read_bytes()); content[-1] ^= 1; path.write_bytes(content)
            with patch.object(a,'load_local_model') as loader, patch.object(a,'FROZEN_SPEC',spec):
                with self.assertRaisesRegex(ValueError, 'basis size/hash'):
                    a.construct(args)
                loader.assert_not_called()

    def test_manifest_and_spec_hashes_are_strict(self):
        for key in ('attempt011_manifest_path','attempt011_spec_path'):
            with tempfile.TemporaryDirectory() as directory:
                args, spec, m, model = self.fixture(Path(directory))
                path = getattr(args,key); path.write_text(path.read_text()+' ')
                with self.assertRaisesRegex(ValueError, 'sha256 mismatch'):
                    self.run_fixture(args,spec,model)

    def test_provenance_linkage_rejects_wrong_identity_and_matrix_order(self):
        for mutation in ('attempt_id','attempt_spec_sha256','eligible_parameters'):
            with tempfile.TemporaryDirectory() as directory:
                args,spec,m,model=self.fixture(Path(directory))
                m[mutation] = ['wrong'] if mutation=='eligible_parameters' else 'wrong'
                args.attempt011_manifest_path.write_text(json.dumps(m))
                spec['frozen_attempt011']['manifest_sha256']=a.sha256_file(args.attempt011_manifest_path)
                with self.assertRaises(ValueError):
                    a.validate_provenance(args.attempt011_spec_path,args.attempt011_manifest_path,spec)

    def test_basis_structure_finiteness_and_gram_validation(self):
        for kind in ('keys','dtype','shape','finite','gram'):
            with tempfile.TemporaryDirectory() as directory:
                args,spec,m,model=self.fixture(Path(directory))
                value=torch.eye(4)[0].reshape(2,2)
                key=m['eligible_parameters'][0]
                if kind=='keys': key='wrong'
                if kind=='dtype': value=value.double()
                if kind=='shape': value=value.flatten()
                if kind=='finite': value[0,0]=float('nan')
                if kind=='gram': value*=2
                save_file({key:value.contiguous()},str(args.basis_directory/'q1.safetensors'))
                with self.assertRaises(ValueError):
                    with a.BasisReader(args.basis_directory,m,torch) as reader:
                        a.construct_tangent(reader,m,a.spectral_candidate(m,spec,torch),
                            {'model.layers.0.proj.weight':model.model.layers[0].proj.weight},spec,torch)

    def test_natural_and_scaled_norms_one_cast_positive_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            args,spec,m,model=self.fixture(Path(directory))
            c=a.spectral_candidate(m,spec,torch)
            with a.BasisReader(args.basis_directory,m,torch) as reader:
                tangent,norms=a.construct_tangent(reader,m,c,{'model.layers.0.proj.weight':model.model.layers[0].proj.weight},spec,torch)
            natural=3*torch.tensor(c['coefficients'],dtype=torch.float64)
            expected=(natural*(3/float(natural.norm()))).float().reshape(2,2)
            self.assertTrue(torch.equal(tangent['model.layers.0.proj.weight'],expected))
            self.assertAlmostEqual(norms['natural_norm'],float(natural.norm()))
            self.assertAlmostEqual(norms['realized_fp32_probe_tangent_norm'],float(expected.double().norm()))
            self.assertGreater(norms['gamma'],0)

    def test_probe_own_hashes_and_source_immutability(self):
        with tempfile.TemporaryDirectory() as directory:
            args,spec,m,model=self.fixture(Path(directory))
            for field in ('serialized_sha256','raw_tensor_sha256'):
                bad={**spec['probe'],field:'0'*64}
                with self.assertRaises(ValueError): a.load_probe(args.probe_path,bad,torch)
            hashes={args.probe_path:a.sha256_file(args.probe_path)}
            source=a.checkpoint_file_records(args.merged_model_dir)
            args.probe_path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Frozen input changed'):
                a.verify_unchanged(args.merged_model_dir,source,hashes)

    def test_model_mutation_fails_before_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            args,spec,m,model=self.fixture(Path(directory))
            original=a.compute_probe_response
            def mutate(*args,**kwargs):
                result=original(*args,**kwargs)
                with torch.no_grad(): model.model.layers[27].proj.weight.add_(1)
                return result
            with patch.object(a,'compute_probe_response',side_effect=mutate):
                with self.assertRaisesRegex(ValueError,'parameters or buffers changed'):
                    self.run_fixture(args,spec,model)
            self.assertFalse(args.artifact_path.exists())

    def test_no_prior_activation_artifact_is_opened(self):
        with tempfile.TemporaryDirectory() as directory:
            args,spec,m,model=self.fixture(Path(directory))
            forbidden=args.basis_directory/'krylov_probe.pt';forbidden.write_text('must never read')
            original=Path.open
            seen=[]
            def tracked(path,*args,**kwargs):
                if path==forbidden: raise AssertionError('Forbidden prior response access')
                seen.append(path)
                return original(path,*args,**kwargs)
            with patch.object(Path,'open',tracked): self.run_fixture(args,spec,model)
            self.assertNotIn(forbidden,seen)

    def test_partial_publication_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            args,spec,m,model=self.fixture(Path(directory))
            with patch.object(a,'write_manifest',side_effect=OSError('write failure')):
                with self.assertRaises(OSError): self.run_fixture(args,spec,model)
            self.assertTrue(args.artifact_path.exists())
            self.assertFalse(args.construction_manifest_path.exists())
            with self.assertRaisesRegex(ValueError,'already exists'): self.run_fixture(args,spec,model)


class ReadoutTests(unittest.TestCase):
    def test_full_10000_probe_and_313_batches(self):
        spec=copy.deepcopy(a.FROZEN_SPEC);spec['readout']['hidden_size']=2
        model=TinyModel();probe=torch.zeros(10000,128,dtype=torch.int64)
        counts=[]
        def fake(model,tokens,tangent,spec,t):
            counts.append(len(tokens))
            return torch.ones(len(tokens),128,2),torch.full((len(tokens),128,2),2.)
        with patch.object(a,'run_readout_jvp',side_effect=fake), \
             patch.object(a,'ordinary_hook_readout',return_value=torch.ones(32,128,2)),redirect_stdout(io.StringIO()):
            merged,response,checks=a.compute_probe_response(model,probe,{},spec,torch)
        self.assertEqual(len(counts),313);self.assertEqual(sum(counts),10000);self.assertEqual(counts[-1],16)
        self.assertEqual(merged.dtype,torch.float64);self.assertEqual(response.dtype,torch.float64)
        self.assertTrue(torch.equal(response,torch.full((128,2),2.,dtype=torch.float64)))

    def test_exact_block_output_index_and_jvp_fail_loudly(self):
        spec=copy.deepcopy(a.FROZEN_SPEC);spec['probe']['sequence_length']=3;spec['readout']['hidden_size']=2
        model=TinyModel();tokens=torch.ones(2,3,dtype=torch.int64)
        tangent={'model.layers.0.proj.weight':torch.ones(2,2)}
        primal,response=a.run_readout_jvp(model,tokens,tangent,spec,torch)
        self.assertTrue(torch.equal(primal,a.ordinary_hook_readout(model,tokens,spec,torch)))
        self.assertGreater(float(response.norm()),0)
        spec['readout']['hidden_state_index']=13
        with self.assertRaisesRegex(ValueError,'index'): a.run_readout_jvp(model,tokens,tangent,spec,torch)
        spec['readout']['hidden_state_index']=14
        with patch.object(torch.func,'jvp',side_effect=RuntimeError('unsupported kernel')):
            with self.assertRaisesRegex(RuntimeError,'no fallback'): a.run_readout_jvp(model,tokens,tangent,spec,torch)

    def test_output_invalid_shape_dtype_nonfinite(self):
        spec=copy.deepcopy(a.FROZEN_SPEC)
        valid=torch.zeros(128,2048,dtype=torch.float64)
        for value in (valid.float(),valid[:2],valid+float('nan'),valid+1e40):
            with self.assertRaises(ValueError): a.build_artifact(valid,value,spec,torch)

    def test_eligible_selection_98_boundary(self):
        model=MockQwen();eligible=a.discover_eligible_linear_weights(model,torch)
        self.assertEqual(len(eligible),98)
        self.assertTrue(all(int(name.split('.')[2])<=13 for name,_ in eligible))


if __name__=='__main__': unittest.main()
