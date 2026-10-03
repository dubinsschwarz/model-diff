"""Synthetic controls for the fixed Attempt121 development trajectory."""
import ast
from contextlib import nullcontext
import copy
import importlib.util
import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/ablation/run_multistep_generic_relaxation.py'
loader = importlib.util.spec_from_file_location('attempt121', SOURCE)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)


class Tensor:
    def __init__(self, value, dtype=np.float32):
        self.array = np.array(value, dtype=dtype, copy=True)
        self.grad = None
        self.device = SimpleNamespace(type='cpu')

    @property
    def dtype(self):
        return self.array.dtype.type

    @property
    def shape(self):
        return self.array.shape

    def detach(self):
        return self

    def to(self, device=None, dtype=None, copy=False):
        if device in (np.float32, np.float64):
            dtype, device = device, None
        return Tensor(self.array, self.dtype if dtype is None else dtype)

    def contiguous(self):
        return self

    def is_contiguous(self):
        return True

    def copy_(self, other):
        self.array[...] = other.array
        return self

    def __mul__(self, other):
        return Tensor(self.array * other, self.dtype)

    __rmul__ = __mul__

    def __sub__(self, other):
        return Tensor(self.array - other.array, self.dtype)

    def square(self):
        return Tensor(self.array * self.array, self.dtype)

    def sum(self, dim=None):
        return self.array.sum() if dim is None else Tensor(self.array.sum(axis=dim), self.dtype)

    def add_(self, other):
        self.array += other.array
        return self

    def __truediv__(self, value):
        return Tensor(self.array/value, self.dtype)

    def __getitem__(self, key):
        return Tensor(self.array[key], self.dtype)


class TinyTorch:
    Tensor = Tensor
    float32 = np.float32
    float64 = np.float64
    int64 = np.int64
    no_grad = staticmethod(nullcontext)
    inference_mode = staticmethod(nullcontext)
    isfinite = staticmethod(lambda x: np.isfinite(x.array))
    zeros = staticmethod(lambda shape, dtype, device: Tensor(np.zeros(shape), dtype))


def eligible_98(value=1.0):
    out = []
    for i in range(98):
        weight = Tensor([[value]])
        out.append((f'model.layers.{i}.linear', SimpleNamespace(weight=weight)))
    return out


def report(primary, secondary):
    return {'positions_1_4_mean_cosine': primary,
            'positions_1_127_mean_cosine': secondary}


def test_frozen_grid_steps_and_plan():
    spec = a.load_spec()
    assert tuple(spec['relaxation']['functional_checkpoints']) == a.GRID == (1, 2, 4)
    assert list(a.trajectory_steps()) == [1, 2, 3, 4]
    assert list(a.trajectory_steps(3)) == [3, 4]
    with pytest.raises(ValueError):
        a.trajectory_steps(5)
    assert spec['relaxation']['recovery_step'] == 2
    assert spec['barrier']['recovery_checkpoint_scientific_candidate'] is False
    assert spec['barrier']['all_4_steps_and_3_responses_published_and_revalidated_before_oracle'] is True
    assert 'all_8_steps_and_4_responses_published_and_revalidated_before_oracle' not in spec['barrier']
    for key in ('optimizer', 'momentum', 'weight_decay', 'gradient_clipping',
                'early_stopping', 'later_gradient_normalization',
                'response_normalization', 'response_rescaling'):
        assert spec['relaxation'][key] is False
    for mutation in (
        lambda s: s['relaxation'].update(functional_checkpoints=[1, 2, 4, 8]),
        lambda s: s['relaxation'].update(optimizer=True),
        lambda s: s['relaxation'].update(later_gradient_normalization=True),
        lambda s: s['barrier'].update(all_4_steps_and_3_responses_published_and_revalidated_before_oracle=False),
        lambda s: s['pilot_selection']['generic_corpus'].update(rows=[1024,2048]),
        lambda s: s['pilot_selection']['activation_probe'].update(rows=[0,10000]),
    ):
        changed = copy.deepcopy(spec)
        mutation(changed)
        with pytest.raises(ValueError):
            a.validate_plan(changed)


def test_fixed_prefixes_and_batch_inventory():
    spec = a.load_spec()
    source_spec = json.loads(a.path_of(spec['frozen_attempt005']['spec_path']).read_text())
    assert source_spec['generic_loss']['sample_count'] == 4096
    assert source_spec['generic_loss']['number_of_batches'] == 512
    assert spec['probe']['sample_count'] == 10000
    assert spec['pilot_selection']['underlying_artifacts_fully_hash_validated'] is True
    assert spec['generic_loss']['sample_count'] == 1024
    assert spec['generic_loss']['batch_size'] == 8
    assert spec['generic_loss']['number_of_batches'] == 128
    assert spec['generic_loss']['total_prediction_tokens'] == 130048
    assert spec['pilot_selection']['generic_corpus']['rows'] == [0, 1024]
    assert spec['pilot_selection']['activation_probe']['rows'] == [0, 1024]
    assert spec['pilot_selection']['intended_workload']['total_backward_batches'] == 512
    assert spec['pilot_selection']['intended_workload']['total_probe_forward_batches'] == 128
    assert spec['pilot_selection']['intended_workload']['runtime_device'] == 'cuda_required'
    assert spec['relaxation']['eta_rule'].endswith('pilot_1024_row_initial_generic_gradient_norm')
    assert spec['probe']['sample_count'] == 10000  # full artifact remains frozen
    assert list(a.probe_batch_starts(spec)) == list(range(0, 1024, 32))
    assert len(a.probe_batch_starts(spec)) == 32
    full_corpus = Tensor(np.arange(4096*128).reshape(4096,128), np.int64)
    full_probe = Tensor(np.arange(10000*128).reshape(10000,128), np.int64)
    corpus = a.fixed_prefix(full_corpus, spec['pilot_selection']['generic_corpus'], 4096, TinyTorch)
    probe = a.fixed_prefix(full_probe, spec['pilot_selection']['activation_probe'], 10000, TinyTorch)
    assert corpus.array.shape == probe.array.shape == (1024,128)
    assert np.array_equal(corpus.array, full_corpus.array[:1024])
    assert np.array_equal(probe.array, full_probe.array[:1024])
    with pytest.raises(ValueError):
        a.fixed_prefix(full_corpus, {'rows':[1,1025], 'sample_count':1024}, 4096, TinyTorch)
    with pytest.raises(ValueError):
        a.fixed_prefix(Tensor(np.zeros((1024,128)), np.int64),
                       spec['pilot_selection']['generic_corpus'], 4096, TinyTorch)


def test_activation_mean_uses_exact_same_1024_prefix_in_32_batches():
    spec = a.load_spec()
    probe = Tensor(np.repeat(np.arange(10000, dtype=np.int64)[:,None], 128, axis=1), np.int64)
    seen = []

    class Model:
        def eval(self):
            return self

        def parameters(self):
            yield Tensor([[0.]])

    class Helper:
        @staticmethod
        def ordinary_hook_readout(_model, batch, _spec, _torch):
            seen.append((int(batch.array[0,0]), int(batch.array[-1,0]), batch.shape[0]))
            return Tensor(batch.array[:,0].reshape(-1,1,1), np.float32)

    mean = a.mean_activation(Model(), probe, spec, Helper, TinyTorch)
    assert len(seen) == 32
    assert seen[0] == (0,31,32)
    assert seen[-1] == (992,1023,32)
    assert mean.array.shape == (128,2048)
    assert mean.array[0,0] == 511.5


def test_eta_derived_once_from_pilot_gradient():
    scale = {'aggregate_source_weight_norm': a.EXPECTED['weight_norm'],
             'aggregate_generic_gradient_norm': 20.0,
             'target_delta_norm': a.EXPECTED['target_norm'],
             'alpha': a.EXPECTED['target_norm']/20.0}
    record = a.derive_pilot_scale(scale)
    eta = record['fixed_eta']
    assert eta != a.EXPECTED['eta']
    assert record['pilot_initial_gradient_norm'] == 20.0
    assert record['target_step_norm'] == a.EXPECTED['target_norm']
    assert a.fixed_eta(scale['aggregate_source_weight_norm'],
                       scale['aggregate_generic_gradient_norm']) == (
                           scale['target_delta_norm'], eta)
    bad = dict(scale, alpha=0.12)
    with pytest.raises(ValueError):
        a.derive_pilot_scale(bad)
    with pytest.raises(ValueError):
        a.validate_initial_scale({**record, 'pilot_initial_gradient_norm': 0})
    with pytest.raises(ValueError):
        a.validate_initial_scale({**record, 'fixed_eta': float('nan')})
    for norms in ((0, 1), (1, 0), (float('nan'), 1), (1, float('inf'))):
        with pytest.raises(ValueError):
            a.fixed_eta(*norms)
    tree = ast.parse(SOURCE.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    eta_assignments = [n for n in ast.walk(run) if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == 'eta' for t in n.targets)]
    # Declaration, recovery restore, and the only derivation at step 1.
    assert len(eta_assignments) == 3
    body = ast.get_source_segment(SOURCE.read_text(), run)
    assert 'if step == 1:' in body
    assert body.count('a5.global_rollback_scale(') == 1
    assert body.index('if step == 1:') < body.index('initial_scale = derive_pilot_scale(scale)')


def test_quadratic_four_steps_fixed_eta_no_later_normalization_and_only_eligible_change():
    eligible = eligible_98(1.0)
    forbidden = Tensor([[17.0]])
    frozen = forbidden.array.copy()
    weight_norm = math.sqrt(98)
    gradient_norm = math.sqrt(98) * 2.0
    _, eta = a.fixed_eta(weight_norm, gradient_norm)
    assert eta == 0.000625
    observations = {}
    initial_gradient = None
    for step in a.trajectory_steps():
        # L = sum_i w_i^2, so every fresh full gradient is 2*w_i.
        current_gradient = 2.0 * eligible[0][1].weight.array.item()
        if step == 1:
            initial_gradient = current_gradient
        for _, module in eligible:
            module.weight.grad = Tensor(2.0 * module.weight.array)
        a.update_eligible(eligible, eta, TinyTorch)
        if step in a.GRID:
            observations[step] = eligible[0][1].weight.array.item()
    assert current_gradient < initial_gradient
    assert list(observations) == list(a.GRID)
    assert list(observations) == [1,2,4]
    for k, actual in observations.items():
        expected = np.float32((1 - 2 * eta) ** k).item()
        assert actual == pytest.approx(expected, abs=2e-7)
        assert 1.0 - actual > 0  # Delta_hat = initial - relaxed.
    assert forbidden.array.tolist() == frozen.tolist()
    assert all(module.weight.array.item() != 1.0 for _, module in eligible)
    assert a.displacement_norm(eligible, {name: Tensor([[1.0]]) for name, _ in eligible}, TinyTorch) > 0


def test_update_sign_and_nonfinite_rejection():
    eligible = eligible_98(1.0)
    for _, module in eligible:
        module.weight.grad = Tensor([[2.0]])
    a.update_eligible(eligible, 0.1, TinyTorch)
    assert all(module.weight.array.item() == pytest.approx(.8) for _, module in eligible)
    with pytest.raises(ValueError):
        a.update_eligible(eligible, float('nan'), TinyTorch)
    eligible[3][1].weight.grad = Tensor([[float('nan')]])
    with pytest.raises(ValueError):
        a.update_eligible(eligible, .1, TinyTorch)
    with pytest.raises(ValueError):
        a.gradient_norm(eligible, TinyTorch)
    for _, module in eligible:
        module.weight.grad = Tensor([[0.]])
    assert a.gradient_norm(eligible, TinyTorch) == 0


def test_response_sign_is_finite_unscaled_subtraction():
    # These synthetic tensors exercise the same subtraction with a minimal tensor adapter.
    final = Tensor(np.full((128, 2048), 3.0), np.float64)
    current = Tensor(np.full((128, 2048), 2.25), np.float64)
    response = a.response_from_means(final, current, TinyTorch)
    assert response.dtype == np.float32
    assert response.array.shape == (128, 2048)
    assert np.all(response.array == .75)
    current.array[0, 0] = np.nan
    with pytest.raises(ValueError):
        a.response_from_means(final, current, TinyTorch)


def test_signed_cosine_report_and_development_thresholds():
    class Validator:
        @staticmethod
        def position_cosines(_x, _y):
            return [-.5, .1, .2, .3, .4] + [.5] * 123
    scored = a.score_report(None, None, Validator)
    assert scored['position_0_cosine'] == -.5
    assert scored['positions_1_4_individual_cosines'] == [.1, .2, .3, .4]
    assert scored['positions_1_4_mean_cosine'] == .25
    assert len(scored['all_128_position_cosines']) == 128
    assert scored['positions_1_127_mean_cosine'] == pytest.approx((1+123*.5)/127)
    cases = [
        ({1: report(.4,.4),2: report(.431,.421),4: report(.42,.5)},
         2, 'clear_multistep_support'),
        ({1: report(.4,.4),2: report(.416,.4),4: report(.41,.9)},
         2, 'moderate_multistep_support'),
        ({1: report(.4,.4),2: report(.414,.9),4: report(.4,.8)},
         2, 'no_meaningful_multistep_support'),
        ({1: report(.4,.4),2: report(.42,.42),4: report(.42,.42)},
         2, 'moderate_multistep_support'),
        ({1: report(.4,.4),2: report(.41,.9),4: report(.44,.421)},
         4, 'clear_multistep_support'),
    ]
    for values, best, category in cases:
        result = a.classify(values)
        assert result['best_multistep_k'] == best
        assert result['category'] == category
    with pytest.raises(ValueError, match='three frozen checkpoints'):
        a.classify({**cases[0][0], 8: report(1,1)})
    with pytest.raises(ValueError, match='three frozen checkpoints'):
        a.classify({1: report(.1,.1)})
    undefined = {1: report(None, None), 2: report(None, None),
                 4: report(None, None)}
    assert a.classify(undefined)['category'] == 'no_meaningful_multistep_support'


def test_artifact_overwrite_refusal_and_recovery_only(tmp_path):
    spec = copy.deepcopy(a.load_spec())
    spec['paths'] = {key: str(tmp_path/name) for key, name in
                     [('candidate_path','candidate.pt'), ('recovery_path','step2.pt'),
                      ('construction_manifest_path','manifest.json'), ('result_path','result.json')]}
    a.refuse_outputs(spec)
    recovery = Path(spec['paths']['recovery_path'])
    recovery.write_bytes(b'recovery')
    with pytest.raises(ValueError):
        a.refuse_outputs(spec)
    a.refuse_outputs(spec, resume=True)
    candidate = Path(spec['paths']['candidate_path'])
    a.atomic_json_publish(candidate, {'frozen': True})
    with pytest.raises(ValueError):
        a.refuse_outputs(spec, resume=True)
    with pytest.raises(ValueError):
        a.atomic_json_publish(candidate, {'frozen': False})
    assert candidate.read_text().find('true') >= 0

    class SaveTorch:
        @staticmethod
        def save(value, stream):
            pickle.dump(value, stream)

    binary = tmp_path/'binary-candidate.pt'
    a.atomic_torch_publish(binary, {'R_1': 1}, SaveTorch)
    with pytest.raises(ValueError):
        a.atomic_torch_publish(binary, {'R_1': 2}, SaveTorch)
    assert pickle.loads(binary.read_bytes()) == {'R_1': 1}


def test_step2_checkpoint_is_only_recovery_state(tmp_path):
    class SaveTorch:
        float32 = np.float32

        @staticmethod
        def save(value, stream):
            pickle.dump(value, stream)

    path = tmp_path/'step2.pt'
    eligible = eligible_98()
    rows = [{'step': k} for k in range(1, 3)]
    checkpoints = {k: {'state_aggregate_sha256': 'a'*64} for k in (1, 2)}
    responses = {k: Tensor([[float(k)]]) for k in (1, 2)}
    scale = {'initial_weight_norm':a.EXPECTED['weight_norm'],
             'pilot_initial_gradient_norm':20.0,
             'target_step_norm':a.EXPECTED['target_norm'],
             'fixed_eta':a.EXPECTED['target_norm']/20.0}
    a.save_recovery(path, eligible, rows, checkpoints, responses, Tensor([[0.]]),
                    scale, 'b'*64, 'c'*64, None, SaveTorch)
    value = pickle.loads(path.read_bytes())
    assert value['recovery_only'] is True
    assert value['step'] == 2
    assert len(value['state']) == 98
    assert list(value['responses']) == [1, 2]
    assert value['eta'] == scale['fixed_eta']
    assert 'scientific_candidate' not in value
    with pytest.raises(ValueError):
        a.save_recovery(path, eligible, rows, checkpoints, responses, Tensor([[0.]]),
                        scale, 'b'*64, 'c'*64, None, SaveTorch)
    with pytest.raises(ValueError):
        a.save_recovery(tmp_path/'wrong.pt', eligible, rows[:1], checkpoints,
                        responses, Tensor([[0.]]), scale, 'b'*64, 'c'*64, None, SaveTorch)


def test_barrier_precedes_oracle_call_and_all_three_candidates_are_published():
    text = SOURCE.read_text()
    run = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    body = ast.get_source_segment(text, run)
    assert body.index('if not torch.cuda.is_available():') < body.index('a5.load_local_model(')
    assert body.index("'R_1': responses[1]") < body.index('atomic_json_publish(manifest_path, manifest)')
    assert body.index("'R_2': responses[2]") < body.index('atomic_json_publish(manifest_path, manifest)')
    assert body.index("'R_4': responses[4]") < body.index('atomic_json_publish(manifest_path, manifest)')
    assert "'R_8'" not in body
    assert body.index('pilot_tokens = fixed_prefix(') < body.index('for step in trajectory_steps(start_step):')
    assert 'a5.accumulate_mean_generic_gradient(model, pilot_tokens' in body
    assert body.count('mean_activation(model, probe, spec, a14, torch)') == 2
    assert body.index('frozen_artifact = torch.load(candidate_path') < body.index("print('RELAXATION_CANDIDATES_FROZEN'")
    assert body.index('atomic_json_publish(manifest_path, manifest)') < body.index('frozen_artifact = torch.load(candidate_path')
    assert body.index('_, aggregate = state_hashes(eligible, a14, torch)') < body.index("print('RELAXATION_CANDIDATES_FROZEN'")
    assert body.index("print('RELAXATION_CANDIDATES_FROZEN'") < body.index('evaluate_frozen(spec,')
    assert body.index('recovery_path.unlink()') < body.index("print('RELAXATION_CANDIDATES_FROZEN'")
    assert 'save_recovery(recovery_path' in body
    assert 'if step == RECOVERY_STEP and not resume:' in body
