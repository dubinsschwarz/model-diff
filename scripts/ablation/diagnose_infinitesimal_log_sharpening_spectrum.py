#!/usr/bin/env python3
"""Blind final-model infinitesimal log-sharpening spectrum diagnostic."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
HELPER_PATH = Path(__file__).with_name('construct_generic_hessian_krylov.py')
HELPER_SHA256 = '9edce220e1f9e29b4e02082852d03aa6b8043e35964a546383a3c10111018f44'
if hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest() != HELPER_SHA256:
    raise ValueError('Frozen final-model provenance helper changed')
loader = importlib.util.spec_from_file_location('attempt024_final_model_helper', HELPER_PATH)
a = importlib.util.module_from_spec(loader)
loader.loader.exec_module(a)

ATTEMPT = '024_blind_infinitesimal_log_sharpening_spectrum'
RANK_ENDS = (1, 4, 16, 64, 256, 1024, 4096, 16384)
RANK_LABELS = ('rank_1', 'ranks_2_4', 'ranks_5_16', 'ranks_17_64',
               'ranks_65_256', 'ranks_257_1024', 'ranks_1025_4096',
               'ranks_4097_16384', 'ranks_16385_full_vocab')
MEASURES = ('probability_mass', 'absolute_dot_p', 'squared_dot_p',
            'absolute_dot_logp', 'squared_dot_logp')

FROZEN_SPEC = {
    'format_version': 1,
    'attempt_id': ATTEMPT,
    'purpose': 'blind_final_model_infinitesimal_log_sharpening_spectrum_diagnostic',
    'information_policy': 'canonical_final_checkpoint_plus_frozen_attempt005_fineweb_only',
    'final_checkpoint_files': a.FROZEN_SPEC['canonical_checkpoint_files'],
    'frozen_corpus': a.FROZEN_SPEC['frozen_corpus'],
    'model': {'dtype': 'torch.float32', 'eval_mode': True,
              'autocast': False, 'parameter_gradients': False,
              'parameter_updates': False, 'jvp': False},
    'batches': {'sequence_count': 4096, 'sequence_length': 128,
                'batch_size': 8, 'batch_count': 512,
                'prediction_positions': list(range(127)),
                'prediction_positions_per_sequence': 127,
                'total_prediction_positions': 4096*127,
                'row_rule': 'batch_i_rows_[8*i,8*(i+1))',
                'smoke_batch_indices': [0, 1]},
    'infinitesimal_signal': {
        'beta': 1.0, 'finite_beta_on_real_model': False,
        'logp': 'torch.nn.functional.log_softmax(FP32_final_logits,dim=-1)',
        'p': 'torch.nn.functional.softmax(FP32_final_logits,dim=-1)',
        'mean_logp_under_p': 'sum_vocab(p*logp)',
        'dot_logp': 'logp-mean_logp_under_p',
        'dot_p': 'p*dot_logp',
        'sorting': 'descending_final_logits_equivalent_to_descending_final_p',
        'conservation': 'abs(sum_vocab_FP64(dot_p))/max(sum_vocab_FP64(abs(dot_p)),1e-6)<=1e-4_per_prediction_position',
        'conservation_relative_limit': 1e-4,
        'conservation_scale_floor': 1e-6,
        'finite_difference': 'synthetic_logits_unit_test_only'},
    'rank_bins': [{'label': label, 'one_based_start': start,
                   'one_based_end': end if end is not None else 'full_vocab'}
                  for label, start, end in zip(RANK_LABELS,
                    (1, 2, 5, 17, 65, 257, 1025, 4097, 16385),
                    (*RANK_ENDS, None))],
    'cumulative_top_k': [*RANK_ENDS, 'full_vocab'],
    'accumulation': {'per_batch_reduction_dtype': 'float64',
                     'cross_batch_scalar_sum': 'CPU_float64_math.fsum',
                     'persistent_logits_or_probabilities': False,
                     'undefined_fraction_if_total_zero': None},
    'sources': {'scripts/ablation/construct_generic_hessian_krylov.py': HELPER_SHA256},
    'defaults': {
        'final_directory': '/root/model-diff-scratch/models/merged',
        'corpus_spec_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json',
        'corpus_manifest_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json',
        'tokens_path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
        'manifest_path': 'experiments/attempts/'+ATTEMPT+'/construction-manifest.json',
    },
    'outputs': {'overwrite': False, 'timestamps': False,
                'host_metadata': False, 'gpu_metadata': False,
                'candidate_selection': False, 'privileged_evaluation': False},
}


def rank_intervals(vocab_size):
    if not isinstance(vocab_size, int) or vocab_size < 2:
        raise ValueError('Vocabulary must contain at least two tokens')
    starts = (0, *RANK_ENDS)
    ends = (*RANK_ENDS, vocab_size)
    return [(label, min(start, vocab_size), min(end, vocab_size))
            for label, start, end in zip(RANK_LABELS, starts, ends)]


def infinitesimal_signals(logits):
    if (not isinstance(logits, torch.Tensor) or logits.dtype != torch.float32 or
            logits.ndim != 2 or logits.shape[1] < 2 or
            not bool(torch.isfinite(logits).all())):
        raise ValueError('Invalid FP32 final logits')
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    p = torch.nn.functional.softmax(logits, dim=-1)
    mean_logp_under_p = (p*logp).sum(dim=-1, keepdim=True)
    dot_logp = logp-mean_logp_under_p
    dot_p = p*dot_logp
    if not all(bool(torch.isfinite(value).all())
               for value in (logp, p, mean_logp_under_p, dot_logp, dot_p)):
        raise ValueError('Nonfinite infinitesimal sharpening signal')
    return logp, p, dot_logp, dot_p


def conservation_audit(dot_p, policy):
    if (dot_p.ndim != 2 or dot_p.dtype != torch.float32 or
            not bool(torch.isfinite(dot_p).all())):
        raise ValueError('Invalid probability derivative for conservation audit')
    signal = dot_p.double()
    residual = signal.sum(dim=-1).abs()
    scale = signal.abs().sum(dim=-1).clamp(min=policy['conservation_scale_floor'])
    relative = residual/scale
    maximum = float(relative.max())
    if not math.isfinite(maximum) or maximum > policy['conservation_relative_limit']:
        raise ValueError('Infinitesimal probability conservation failed')
    return {'maximum_absolute_residual': float(residual.max()),
            'maximum_relative_residual': maximum,
            'checked_prediction_positions': int(dot_p.shape[0])}


class SpectrumAccumulator:
    def __init__(self, policy):
        self.policy = policy
        self.sequence_count = 0
        self.position_count = 0
        self.vocab_size = None
        self.parts = [{name: [] for name in MEASURES} for _ in RANK_LABELS]
        self.global_parts = {'entropy': [], 'top1_probability': [],
                             'top1_minus_top2_logit_margin': []}
        self.conservation_max_absolute = 0.0
        self.conservation_max_relative = 0.0

    def add_logits(self, prediction_logits):
        if (not isinstance(prediction_logits, torch.Tensor) or
                prediction_logits.dtype != torch.float32 or prediction_logits.ndim != 3 or
                prediction_logits.shape[0] < 1 or prediction_logits.shape[1] != 127 or
                prediction_logits.shape[2] < 2 or
                not bool(torch.isfinite(prediction_logits).all())):
            raise ValueError('Invalid causal prediction logits')
        batch_size, positions_per_sequence, vocab = prediction_logits.shape
        if self.vocab_size is None:
            self.vocab_size = vocab
        elif vocab != self.vocab_size:
            raise ValueError('Vocabulary size changed between batches')
        ranked_logits = torch.sort(prediction_logits.reshape(-1, vocab),
                                   dim=-1, descending=True).values
        logp, p, dot_logp, dot_p = infinitesimal_signals(ranked_logits)
        audit = conservation_audit(dot_p, self.policy)
        self.conservation_max_absolute = max(self.conservation_max_absolute,
                                             audit['maximum_absolute_residual'])
        self.conservation_max_relative = max(self.conservation_max_relative,
                                             audit['maximum_relative_residual'])
        self.sequence_count += batch_size
        self.position_count += batch_size*positions_per_sequence
        self.global_parts['entropy'].append(float((-(p*logp)).double().sum()))
        self.global_parts['top1_probability'].append(float(p[:, 0].double().sum()))
        self.global_parts['top1_minus_top2_logit_margin'].append(
            float((ranked_logits[:, 0].double()-ranked_logits[:, 1].double()).sum()))
        for index, (_, start, end) in enumerate(rank_intervals(vocab)):
            block = self.parts[index]
            if start == end:
                values = (0.0,)*5
            else:
                part_p = p[:, start:end]
                part_dp = dot_p[:, start:end]
                part_dl = dot_logp[:, start:end]
                values = (float(part_p.double().sum()),
                          float(part_dp.double().abs().sum()),
                          float(part_dp.double().square().sum()),
                          float(part_dl.double().abs().sum()),
                          float(part_dl.double().square().sum()))
            for key, value in zip(MEASURES, values):
                if not math.isfinite(value):
                    raise ValueError('Nonfinite rank-bin statistic')
                block[key].append(value)

    def result(self):
        if self.position_count <= 0 or self.vocab_size is None:
            raise ValueError('No prediction positions accumulated')
        totals = [{key: math.fsum(part[key]) for key in MEASURES}
                  for part in self.parts]
        global_totals = {key: math.fsum(values)
                         for key, values in self.global_parts.items()}
        signal_totals = {key: math.fsum(row[key] for row in totals)
                         for key in MEASURES[1:]}
        bins = []
        for (label, start, end), values in zip(rank_intervals(self.vocab_size), totals):
            count = self.position_count*(end-start)
            bins.append({'label': label, 'one_based_rank_start': start+1,
                'one_based_rank_end': end, 'token_count': count,
                'mean_probability_mass': values['probability_mass']/self.position_count,
                'sum_absolute_dot_p': values['absolute_dot_p'],
                'fraction_total_absolute_dot_p':
                    values['absolute_dot_p']/signal_totals['absolute_dot_p']
                    if signal_totals['absolute_dot_p'] else None,
                'sum_squared_dot_p': values['squared_dot_p'],
                'fraction_total_squared_dot_p':
                    values['squared_dot_p']/signal_totals['squared_dot_p']
                    if signal_totals['squared_dot_p'] else None,
                'sum_absolute_dot_logp': values['absolute_dot_logp'],
                'fraction_total_absolute_dot_logp':
                    values['absolute_dot_logp']/signal_totals['absolute_dot_logp']
                    if signal_totals['absolute_dot_logp'] else None,
                'sum_squared_dot_logp': values['squared_dot_logp'],
                'fraction_total_squared_dot_logp':
                    values['squared_dot_logp']/signal_totals['squared_dot_logp']
                    if signal_totals['squared_dot_logp'] else None,
                'mean_absolute_dot_p': values['absolute_dot_p']/count if count else None,
                'rms_dot_p': math.sqrt(values['squared_dot_p']/count) if count else None,
                'mean_absolute_dot_logp': values['absolute_dot_logp']/count if count else None,
                'rms_dot_logp': math.sqrt(values['squared_dot_logp']/count) if count else None})
        cumulative = []
        for requested in (*RANK_ENDS, 'full_vocab'):
            effective = self.vocab_size if requested == 'full_vocab' else min(requested, self.vocab_size)
            included = [values for (_, _, end), values in zip(rank_intervals(self.vocab_size), totals)
                        if end <= effective]
            record = {'requested_k': requested, 'effective_k': effective}
            for key, label in (('absolute_dot_p', 'absolute_dot_p'),
                               ('squared_dot_p', 'squared_dot_p'),
                               ('absolute_dot_logp', 'absolute_dot_logp'),
                               ('squared_dot_logp', 'squared_dot_logp')):
                total = signal_totals[key]
                record['fraction_'+label] = math.fsum(row[key] for row in included)/total if total else None
            cumulative.append(record)
        entry_count = self.position_count*self.vocab_size
        return {'number_of_sequences': self.sequence_count,
                'number_of_prediction_positions': self.position_count,
                'vocab_size': self.vocab_size,
                'mean_entropy': global_totals['entropy']/self.position_count,
                'mean_top1_probability': global_totals['top1_probability']/self.position_count,
                'mean_top1_minus_top2_logit_margin':
                    global_totals['top1_minus_top2_logit_margin']/self.position_count,
                'global_rms_dot_p': math.sqrt(signal_totals['squared_dot_p']/entry_count),
                'global_rms_dot_logp': math.sqrt(signal_totals['squared_dot_logp']/entry_count),
                'global_signal_totals': signal_totals,
                'rank_bins': bins, 'cumulative_top_k': cumulative,
                'conservation_audit': {'maximum_absolute_residual': self.conservation_max_absolute,
                    'maximum_relative_residual': self.conservation_max_relative,
                    'checked_prediction_positions': self.position_count}}


def require_output_absent(path):
    if path.exists() or path.is_symlink():
        raise ValueError('Scientific output already exists')


def validate_inputs(args):
    for key in ('final_directory', 'corpus_spec_path', 'corpus_manifest_path',
                'tokens_path', 'manifest_path'):
        expected = Path(FROZEN_SPEC['defaults'][key])
        if not expected.is_absolute():
            expected = PROJECT/expected
        if getattr(args, key).resolve() != expected.resolve():
            raise ValueError('Attempt024 path differs from frozen '+key)
    if args.spec_path.resolve() != (PROJECT/'experiments/attempts'/ATTEMPT/'spec.json').resolve():
        raise ValueError('Attempt024 specification path differs from frozen path')
    require_output_absent(args.manifest_path)
    spec = a.load_json_object(args.spec_path, 'Attempt024 frozen specification')
    if spec != FROZEN_SPEC:
        raise ValueError('Frozen Attempt024 specification mismatch')
    paths = [args.spec_path, args.corpus_spec_path, args.corpus_manifest_path,
             args.tokens_path, Path(__file__).resolve(), HELPER_PATH]
    hashes = {str(path): a.sha256_file(path) for path in paths}
    if hashes[str(HELPER_PATH)] != HELPER_SHA256:
        raise ValueError('Frozen provenance helper source mismatch')
    if a.checkpoint_file_records(args.final_directory) != spec['final_checkpoint_files']:
        raise ValueError('Canonical final checkpoint mismatch')
    for field, path in (('corpus_spec_sha256', args.corpus_spec_path),
                        ('corpus_manifest_sha256', args.corpus_manifest_path)):
        if hashes[str(path)] != spec['frozen_corpus'][field]:
            raise ValueError('Frozen FineWeb provenance mismatch')
    corpus = a.verify_corpus_manifest(args.corpus_manifest_path, args.corpus_spec_path,
        args.tokens_path, args.final_directory, a.load_corpus_spec(args.corpus_spec_path))
    if any(corpus[key] != spec['frozen_corpus'][key]
           for key in ('serialized_sha256', 'raw_tensor_sha256')):
        raise ValueError('Frozen FineWeb artifact mismatch')
    tokens = a.load_corpus_tokens(args.tokens_path, corpus['raw_tensor_sha256'], 4096, 128, torch)
    output = args.manifest_path.resolve()
    if (output.is_relative_to(args.final_directory.resolve()) or
            any(output == path.resolve() or path.resolve().is_relative_to(output)
                for path in paths)):
        raise ValueError('Scientific output overlaps frozen input')
    def recheck():
        if a.checkpoint_file_records(args.final_directory) != spec['final_checkpoint_files']:
            raise ValueError('Final checkpoint changed during diagnostic')
        for path, digest in hashes.items():
            if a.sha256_file(Path(path)) != digest:
                raise ValueError('Frozen diagnostic input changed: '+path)
    return spec, tokens, hashes, recheck


def diagnose(args):
    spec, tokens, hashes, recheck = validate_inputs(args)
    model = a.load_local_model(args.final_directory, args.device, torch)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Final model parameter freeze failed')
    before = a.model_state_hashes(model, torch)
    accumulator = SpectrumAccumulator(spec['infinitesimal_signal'])
    device = next(model.parameters()).device
    batches = spec['batches']['smoke_batch_indices'] if args.smoke_only else range(512)
    for batch_index in batches:
        start = 8*batch_index
        batch = tokens[start:start+8].to(device)
        if batch.dtype != torch.int64 or tuple(batch.shape) != (8, 128):
            raise ValueError('Invalid frozen FineWeb microbatch')
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            outputs = model(input_ids=batch, use_cache=False)
            logits = outputs.logits
            if (logits.dtype != torch.float32 or logits.ndim != 3 or
                    tuple(logits.shape[:2]) != (8, 128) or
                    not bool(torch.isfinite(logits).all())):
                raise ValueError('Invalid FP32 final-model logits')
            prediction_logits = logits[:, :127, :].contiguous()
            accumulator.add_logits(prediction_logits)
        del outputs, logits, prediction_logits, batch
        if not args.smoke_only and (batch_index+1) % 64 == 0:
            print(f'Infinitesimal spectrum batches {batch_index+1}/512', flush=True)
    a.verify_model_unchanged(model, before, torch)
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise ValueError('Final model acquired parameter gradients')
    del model
    recheck()
    result = accumulator.result()
    if args.smoke_only:
        print(json.dumps({'smoke_only': True, 'batch_indices': spec['batches']['smoke_batch_indices'],
                          'number_of_sequences': result['number_of_sequences'],
                          'number_of_prediction_positions': result['number_of_prediction_positions'],
                          'vocab_size': result['vocab_size'],
                          'mean_entropy': result['mean_entropy'],
                          'mean_top1_probability': result['mean_top1_probability'],
                          'maximum_relative_conservation_residual':
                              result['conservation_audit']['maximum_relative_residual']},
                         sort_keys=True))
        return
    if (result['number_of_sequences'] != spec['batches']['sequence_count'] or
            result['number_of_prediction_positions'] != spec['batches']['total_prediction_positions']):
        raise ValueError('Full frozen corpus coverage mismatch')
    manifest = {'format_version': 1, 'attempt_id': ATTEMPT,
                'purpose': spec['purpose'], 'information_policy': spec['information_policy'],
                'hash_algorithm': 'sha256', 'input_source_hashes': hashes,
                'source_checkpoint_files': spec['final_checkpoint_files'],
                'frozen_corpus': spec['frozen_corpus'], 'batches': spec['batches'],
                'infinitesimal_signal': spec['infinitesimal_signal'],
                'rank_bin_policy': spec['rank_bins'],
                'cumulative_top_k_policy': spec['cumulative_top_k'],
                'accumulation': spec['accumulation'],
                'diagnostic': result,
                'historical_base_access': False, 'oracle_access': False,
                'parameter_gradient_computed': False, 'parameter_update': False,
                'jvp_computed': False, 'candidate_selection': False}
    require_output_absent(args.manifest_path)
    recheck()
    a.write_manifest(args.manifest_path, manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path', type=Path,
        default=PROJECT/'experiments/attempts'/ATTEMPT/'spec.json')
    for key, value in FROZEN_SPEC['defaults'].items():
        path = Path(value)
        parser.add_argument('--'+key.replace('_', '-'), type=Path,
                            default=path if path.is_absolute() else PROJECT/path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-only', action='store_true')
    return parser.parse_args(argv)


if __name__ == '__main__':
    diagnose(parse_args())
