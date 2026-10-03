"""Synthetic controls for Attempt122's blind Armijo relaxation."""
import ast
import copy
import importlib.util
import math
import pickle
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/ablation/run_loss_stabilized_generic_relaxation.py'
loader = importlib.util.spec_from_file_location('attempt122', SOURCE)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)


class Tensor:
    def __init__(self, value, dtype=np.float32):
        self.array = np.array(value, dtype=dtype, copy=True)
        self.device = SimpleNamespace(type='cpu')
        self.grad = None

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
            dtype = device
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

    def __getitem__(self, key):
        return Tensor(self.array[key], self.dtype)

    def item(self):
        return self.array.item()

    def add_(self, other):
        self.array += other.array
        return self

    def __truediv__(self, scalar):
        return Tensor(self.array / scalar, self.dtype)


class TinyTorch:
    Tensor = Tensor
    float32 = np.float32
    float64 = np.float64
    int64 = np.int64
    no_grad = staticmethod(nullcontext)
    inference_mode = staticmethod(nullcontext)
    autocast = staticmethod(lambda **_kwargs: nullcontext())
    isfinite = staticmethod(lambda tensor: np.isfinite(tensor.array))
    equal = staticmethod(lambda x, y: np.array_equal(x.array, y.array))
    zeros = staticmethod(lambda shape, dtype, device: Tensor(np.zeros(shape), dtype))


def eligible_98(value=1.0):
    return [(f'model.layers.{i}.linear', SimpleNamespace(weight=Tensor([[value]])))
            for i in range(98)]


def synthetic_search(x, gradient, eta, *, recording=None):
    """L(x)=x²/2 with an exact pre-step scalar state."""
    before = np.float32(x)
    state = {'x': before, 'restores': 0, 'applies': 0}

    def apply(trial_eta):
        assert state['x'] == before
        state['x'] = np.float32(np.float64(before) - trial_eta * gradient)
        state['applies'] += 1

    def evaluate():
        return float(state['x']) ** 2 / 2

    def restore():
        state['x'] = before
        assert state['x'] == before
        state['restores'] += 1

    result = a.armijo_search(float(before)**2/2, abs(gradient), eta,
                             apply, evaluate, restore)
    if recording is not None:
        recording.append(state)
    return result, float(state['x'])


def test_frozen_pilot_and_only_step_size_change():
    spec = a.load_spec()
    old = a.prior.load_spec()
    assert spec['relaxation']['steps'] == a.STEPS == 4
    assert spec['relaxation']['functional_checkpoints'] == list(a.GRID) == [1, 2, 4]
    assert spec['relaxation']['recovery_step'] == a.RECOVERY_STEP == 2
    assert spec['relaxation']['line_search'] == a.LINE_SEARCH
    assert a.ARMIJO_C == 1e-4
    assert a.BACKTRACK_FACTOR == 0.5
    assert a.MAX_TRIALS == 8
    assert spec['pilot_selection']['generic_corpus']['rows'] == [0, 1024]
    assert spec['pilot_selection']['activation_probe']['rows'] == [0, 1024]
    assert spec['generic_loss']['number_of_batches'] == 128
    assert spec['generic_loss']['total_prediction_tokens'] == 130048
    assert spec['pilot_selection']['activation_probe']['number_of_batches_per_mean'] == 32
    assert spec['probe']['sample_count'] == 10000
    assert spec['pilot_selection']['intended_workload']['total_backward_batches'] == 512
    assert spec['pilot_selection']['intended_workload']['total_probe_forward_batches'] == 128
    assert spec['pilot_selection']['intended_workload']['max_line_search_forward_batches'] == 4096
    assert spec['barrier']['all_4_steps_and_3_responses_published_and_revalidated_before_oracle'] is True
    assert spec['barrier']['recovery_checkpoint_scientific_candidate'] is False
    assert spec['information_policy']['same_specimen_exploratory_method_development'] is True
    assert spec['information_policy']['clean_heldout_validation'] is False
    for key in ('model', 'eligible_tensors', 'generic_loss', 'probe', 'readout',
                'information_policy', 'frozen_attempt005', 'evaluation', 'barrier'):
        assert spec[key] == old[key]
    for key in ('later_gradient_normalization', 'optimizer', 'momentum',
                'weight_decay', 'gradient_clipping', 'early_stopping',
                'response_normalization', 'response_rescaling'):
        assert spec['relaxation'][key] is False
    assert spec['relaxation']['response'] == 'A_final - A_k'
    assert spec['relaxation']['update'] == 'W <- W - eta * full_generic_gradient'
    assert '8' not in spec['relaxation']['functional_checkpoints']


def test_pilot_initial_eta_is_frobenius_scale_not_attempt005_alpha():
    target = .00125 * a.prior.EXPECTED['weight_norm']
    record = a.initial_scale_from_gradient({
        'aggregate_source_weight_norm': a.prior.EXPECTED['weight_norm'],
        'aggregate_generic_gradient_norm': 20.,
        'target_delta_norm': target,
        'alpha': target/20.})
    assert record['target_step_norm'] == pytest.approx(target)
    assert record['pilot_initial_gradient_norm'] == 20.
    assert record['eta_max'] == pytest.approx(target/20.)
    assert record['eta_max'] != a.prior.EXPECTED['eta']
    assert a.validate_initial_scale(record) == record
    with pytest.raises(ValueError):
        a.validate_initial_scale({**record, 'eta_max': float('nan')})


def test_line_search_ce_uses_all_128_forward_batches_of_fixed_prefix():
    spec = a.load_spec()
    rows = np.repeat(np.arange(1024, dtype=np.int64)[:, None], 128, axis=1)
    tokens = Tensor(rows, np.int64)
    seen = []

    class Model:
        def parameters(self):
            yield Tensor([[0.]])

        def eval(self):
            return self

        def __call__(self, *, input_ids, use_cache):
            assert use_cache is False
            seen.append((int(input_ids.array[0, 0]), int(input_ids.array[-1, 0])))
            return SimpleNamespace(logits=None)

    class Helper:
        @staticmethod
        def causal_token_loss_sum(_logits, batch, _torch):
            assert batch.shape == (8, 128)
            return Tensor(8 * 127 * 3.5, np.float32)

    assert a.evaluate_generic_loss(Model(), tokens, spec['generic_loss'], Helper, TinyTorch) == 3.5
    assert len(seen) == 128
    assert seen == [(start, start + 7) for start in range(0, 1024, 8)]
    with pytest.raises(ValueError):
        a.evaluate_generic_loss(Model(), tokens[:1000], spec['generic_loss'], Helper, TinyTorch)


def test_activation_means_use_32_batches_of_same_probe_prefix():
    spec = a.load_spec()
    full_probe = Tensor(np.repeat(np.arange(10000, dtype=np.int64)[:, None], 128, axis=1), np.int64)
    seen = []

    class Model:
        def parameters(self):
            yield Tensor([[0.]])

        def eval(self):
            return self

    class Helper:
        @staticmethod
        def ordinary_hook_readout(_model, batch, _spec, _torch):
            seen.append((int(batch.array[0, 0]), int(batch.array[-1, 0])))
            return Tensor(batch.array[:, 0].reshape(-1, 1, 1), np.float32)

    mean = a.prior.mean_activation(Model(), full_probe, spec, Helper, TinyTorch)
    assert mean.array.shape == (128, 2048)
    assert mean.array[0, 0] == 511.5
    assert seen == [(start, start + 31) for start in range(0, 1024, 32)]


def test_quadratic_backtracks_then_four_accepted_steps_descend():
    # Initial eta=3 is unstable for L=x²/2; eta=1.5 is the first Armijo success.
    x, eta = 1., 3.
    results, states = [], []
    for step in range(1, 5):
        gradient = x
        result, x = synthetic_search(x, gradient, eta, recording=states)
        assert result['initial_trial_eta'] == eta
        assert result['accepted_eta'] <= eta
        assert result['accepted_post_step_generic_mean_ce'] <= results[-1]['accepted_post_step_generic_mean_ce'] if results else True
        results.append(result)
        eta = result['accepted_eta']
    assert [r['accepted_trial_index'] for r in results] == [1, 0, 0, 0]
    assert results[0]['trials'][0]['accepted'] is False
    assert results[0]['trials'][1]['accepted'] is True
    assert states[0]['restores'] == 1
    assert all(s['applies'] == len(r['trials']) for s, r in zip(states, results))
    assert [r['accepted_eta'] for r in results] == [1.5]*4
    assert x == pytest.approx(0.0625)
    assert [r['accepted_post_step_generic_mean_ce'] for r in results] == pytest.approx(
        [.125, .03125, .0078125, .001953125])
    assert [1, 2, 4] == list(a.GRID)


def test_first_satisfying_trial_and_all_eight_reject_without_ninth():
    state = {'x': 1, 'evaluations': 0, 'restores': 0}
    def apply(eta):
        assert state['x'] == 1
        state['x'] = 1 - eta
    def evaluate():
        state['evaluations'] += 1
        return 2 if state['evaluations'] < 3 else 0
    def restore():
        state['x'] = 1
        state['restores'] += 1
    result = a.armijo_search(1., 1., 1., apply, evaluate, restore)
    assert result['accepted_trial_index'] == 2
    assert result['accepted_eta'] == .25
    assert state['evaluations'] == 3 and state['restores'] == 2
    assert [trial['accepted'] for trial in result['trials']] == [False, False, True]
    state.update(x=1, evaluations=0, restores=0)
    def reject():
        state['evaluations'] += 1
        return 2.
    with pytest.raises(ValueError, match='eight predeclared'):
        a.armijo_search(1., 1., 1., apply, reject, restore)
    assert state['evaluations'] == state['restores'] == 8 and state['x'] == 1
    with pytest.raises(ValueError):
        a.armijo_search(float('nan'), 1., 1., apply, evaluate, restore)
    with pytest.raises(ValueError):
        a.armijo_search(1., float('inf'), 1., apply, evaluate, restore)
    with pytest.raises(ValueError):
        a.armijo_search(1., 1., 1., apply, lambda: float('nan'), restore)


def test_exact_rejected_state_restoration_update_sign_and_eligible_only():
    eligible = eligible_98()
    forbidden = Tensor([[17.]])
    before = a.prior.original_weights(eligible, TinyTorch)
    for _, module in eligible:
        module.weight.grad = Tensor([[2.]])
    a.prior.update_eligible(eligible, .1, TinyTorch)
    assert all(module.weight.array.item() == pytest.approx(.8) for _, module in eligible)
    assert forbidden.array.item() == 17.
    a.restore_pre_step(eligible, before, TinyTorch)
    assert all(np.array_equal(module.weight.array, before[name].array)
               for name, module in eligible)
    damaged = copy.deepcopy(before)
    damaged[eligible[0][0]].array[...] = np.nan
    with pytest.raises(ValueError):
        a.restore_pre_step(eligible, damaged, TinyTorch)


def test_response_sign_and_signed_development_metrics():
    final = Tensor(np.full((128, 2048), 3.), np.float64)
    relaxed = Tensor(np.full((128, 2048), 2.25), np.float64)
    response = a.prior.response_from_means(final, relaxed, TinyTorch)
    assert response.dtype == np.float32
    assert np.all(response.array == .75)
    assert response.is_contiguous()
    class Validator:
        @staticmethod
        def position_cosines(_x, _y):
            return [-.5, .1, .2, .3, .4] + [.5]*123
    metrics = a.score_report(None, None, Validator)
    assert metrics['position_0_cosine'] == -.5
    assert metrics['positions_1_4_individual_cosines'] == [.1,.2,.3,.4]
    assert metrics['positions_1_4_mean_cosine'] == .25
    assert metrics['positions_1_127_mean_cosine'] == pytest.approx((1+123*.5)/127)
    assert len(metrics['all_128_position_cosines']) == 128
    report = lambda p,s: {'positions_1_4_mean_cosine':p,
                          'positions_1_127_mean_cosine':s}
    clear = {1:report(.4,.4), 2:report(.431,.421), 4:report(.42,.5)}
    moderate = {1:report(.4,.4), 2:report(.416,.4), 4:report(.41,.9)}
    negative = {1:report(.4,.4), 2:report(.414,.9), 4:report(.4,.8)}
    assert a.classify(clear)['category'] == 'clear_multistep_support'
    assert a.classify(moderate)['category'] == 'moderate_multistep_support'
    assert a.classify(negative)['category'] == 'no_meaningful_multistep_support'
    assert a.classify({1:report(.4,.4),2:report(.42,.42),4:report(.42,.42)})['best_multistep_k'] == 2
    with pytest.raises(ValueError, match='three frozen checkpoints'):
        a.classify({**clear, 8:report(1.,1.)})


def test_consistency_and_trial_record_validation():
    assert a.check_recomputed_loss(1.00001, 1.) == pytest.approx(.00001)
    with pytest.raises(ValueError):
        a.check_recomputed_loss(1.01, 1.)
    scale = a.initial_scale_from_gradient({
        'aggregate_source_weight_norm': a.prior.EXPECTED['weight_norm'],
        'aggregate_generic_gradient_norm': a.prior.EXPECTED['target_norm']/3,
        'target_delta_norm': a.prior.EXPECTED['target_norm'], 'alpha':3.})
    rows, x, eta = [], 1., scale['eta_max']
    for step in range(1,5):
        loss, grad = x*x/2, x
        result, x = synthetic_search(x, grad, eta)
        rows.append({'step':step, 'generic_mean_ce_before_update':loss,
                     'aggregate_gradient_norm_before_update':abs(grad),
                     **result, 'cumulative_displacement_norm':abs(1-x),
                     'cumulative_displacement_relative_to_initial_weight_norm':abs(1-x)/scale['initial_weight_norm']})
        eta = result['accepted_eta']
    assert a.validate_step_records(rows, scale)
    assert rows[1]['initial_trial_eta'] == rows[0]['accepted_eta']
    bad = copy.deepcopy(rows)
    bad[1]['initial_trial_eta'] *= 2
    with pytest.raises(ValueError):
        a.validate_step_records(bad, scale)
    bad = copy.deepcopy(rows)
    bad[0]['trials'][-1]['generic_mean_ce'] = 2
    with pytest.raises(ValueError):
        a.validate_step_records(bad, scale)
    bad = copy.deepcopy(rows)
    bad[0]['cumulative_displacement_norm'] = float('nan')
    with pytest.raises(ValueError):
        a.validate_step_records(bad, scale)


def test_barrier_and_recovery_only_static_invariants():
    spec = a.load_spec()
    body = ast.get_source_segment(SOURCE.read_text(), next(n for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == 'run'))
    assert body.count('a5.global_rollback_scale(') == 1
    assert body.count('prior.mean_activation(model, probe, spec, a14, torch)') == 2
    assert 'for step in prior.trajectory_steps(start_step):' in body
    assert 'prior.update_eligible(eligible, eta, torch)' in body
    assert "'R_1': responses[1], 'R_2': responses[2], 'R_4': responses[4]" in body
    assert "'R_8'" not in body
    assert body.index('if step == RECOVERY_STEP and not resume:') < body.index('prior.atomic_torch_publish(candidate_path,')
    assert body.index('prior.atomic_torch_publish(candidate_path,') < body.index('prior.atomic_json_publish(manifest_path, manifest)')
    assert body.index('prior.atomic_json_publish(manifest_path, manifest)') < body.index('frozen_artifact = torch.load(candidate_path')
    assert body.index('frozen_artifact = torch.load(candidate_path') < body.index('recovery_path.unlink()')
    assert body.index('recovery_path.unlink()') < body.index("print('RELAXATION_CANDIDATES_FROZEN'")
    assert body.index("print('RELAXATION_CANDIDATES_FROZEN'") < body.index('prior.evaluate_frozen(spec,')
    assert spec['barrier']['marker'] == 'RELAXATION_CANDIDATES_FROZEN'
    search = ast.get_source_segment(SOURCE.read_text(), next(n for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == 'armijo_search'))
    assert 'oracle' not in search.lower()
    assert 'MAX_TRIALS' in search and 'BACKTRACK_FACTOR' in search
    assert 'restore_rejected()' in search
    assert 'if accepted:' in search
    assert 'for index in range(MAX_TRIALS):' in search


def test_recovery_is_fixed_after_accepted_step_two_and_overwrite_refused(tmp_path):
    spec = copy.deepcopy(a.load_spec())
    spec['paths'] = {'candidate_path':str(tmp_path/'candidate.pt'),
                     'recovery_path':str(tmp_path/'step2.pt'),
                     'construction_manifest_path':str(tmp_path/'construction-manifest.json'),
                     'result_path':str(tmp_path/'result.json')}
    a.prior.refuse_outputs(spec)
    (tmp_path/'step2.pt').write_bytes(b'recovery')
    with pytest.raises(ValueError):
        a.prior.refuse_outputs(spec)
    a.prior.refuse_outputs(spec, resume=True)
    a.prior.atomic_json_publish(tmp_path/'candidate.pt', {'test':True})
    with pytest.raises(ValueError):
        a.prior.refuse_outputs(spec, resume=True)
    with pytest.raises(ValueError):
        a.prior.atomic_json_publish(tmp_path/'candidate.pt', {'test':False})
    assert 'recovery_only' in ast.get_source_segment(SOURCE.read_text(), next(n for n in ast.parse(SOURCE.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == 'save_recovery'))
    assert a.RECOVERY_STEP == 2


def test_step_two_recovery_contains_previous_accepted_eta_only(tmp_path):
    scale = a.initial_scale_from_gradient({
        'aggregate_source_weight_norm': a.prior.EXPECTED['weight_norm'],
        'aggregate_generic_gradient_norm': a.prior.EXPECTED['target_norm']/3,
        'target_delta_norm': a.prior.EXPECTED['target_norm'], 'alpha':3.})
    rows, x, eta = [], 1., scale['eta_max']
    for step in (1, 2):
        loss, grad = x*x/2, x
        result, x = synthetic_search(x, grad, eta)
        rows.append({'step':step, 'generic_mean_ce_before_update':loss,
                     'aggregate_gradient_norm_before_update':abs(grad),
                     **result, 'cumulative_displacement_norm':abs(1-x),
                     'cumulative_displacement_relative_to_initial_weight_norm':abs(1-x)/scale['initial_weight_norm']})
        eta = result['accepted_eta']
    checkpoints = {step:{'state_aggregate_sha256':'a'*64} for step in (1,2)}
    responses = {step:Tensor([[float(step)]]) for step in (1,2)}

    class SaveTorch(TinyTorch):
        @staticmethod
        def save(value, stream):
            pickle.dump(value, stream)

    path = tmp_path/'step2.pt'
    a.save_recovery(path, eligible_98(), rows, checkpoints, responses,
                    Tensor([[0.]], np.float64), scale, 'b'*64, 'c'*64, SaveTorch)
    payload = pickle.loads(path.read_bytes())
    assert payload['recovery_only'] is True
    assert payload['step'] == 2
    assert payload['previous_accepted_eta'] == rows[1]['accepted_eta']
    assert len(payload['state']) == 98
    assert list(payload['responses']) == [1, 2]
    assert 'scientific_candidate' not in payload
    with pytest.raises(ValueError):
        a.save_recovery(tmp_path/'wrong.pt', eligible_98(), rows[:1], checkpoints,
                        responses, Tensor([[0.]], np.float64), scale, 'b'*64, 'c'*64, SaveTorch)
