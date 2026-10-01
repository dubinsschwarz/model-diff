"""Synthetic checks for the blind Attempt 024 spectrum diagnostic."""
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT/'scripts/ablation/diagnose_infinitesimal_log_sharpening_spectrum.py'
SPEC = PROJECT/'experiments/attempts/024_blind_infinitesimal_log_sharpening_spectrum/spec.json'
loader = importlib.util.spec_from_file_location('attempt024_spectrum_test', SCRIPT)
m = importlib.util.module_from_spec(loader)
loader.loader.exec_module(m)


def toy_logits(batch_size=1, vocab=20):
    sequence = torch.arange(batch_size*127*vocab, dtype=torch.float32)
    return (torch.sin(sequence*0.037)+0.2*torch.cos(sequence*0.019)).reshape(
        batch_size, 127, vocab).contiguous()


class ToyFinalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, use_cache=False):
        if use_cache:
            raise AssertionError('Cache must remain disabled')
        values = input_ids.float().unsqueeze(-1)
        ranks = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20)
        logits = torch.sin(values*0.01+ranks*0.1)*self.scale
        return type('Result', (), {'logits': logits})()


class TestInfinitesimalSpectrum(unittest.TestCase):
    def test_frozen_spec_matches_implementation(self):
        self.assertEqual(json.loads(SPEC.read_text()), m.FROZEN_SPEC)
        self.assertEqual(m.FROZEN_SPEC['batches']['batch_count'], 512)
        self.assertEqual(m.FROZEN_SPEC['batches']['total_prediction_positions'], 4096*127)

    def test_analytic_log_probability_derivative(self):
        logits = torch.tensor([[2.0, 0.25, -1.0], [-3.0, 1.0, 0.5]], dtype=torch.float32)
        logp, p, dot_logp, _ = m.infinitesimal_signals(logits)
        expected = logp-(p*logp).sum(-1, keepdim=True)
        torch.testing.assert_close(dot_logp, expected, rtol=0, atol=0)
        centered_logits = logits-(p*logits).sum(-1, keepdim=True)
        torch.testing.assert_close(dot_logp, centered_logits, rtol=1e-6, atol=1e-6)

    def test_analytic_probability_derivative(self):
        logits = torch.tensor([[2.0, 0.25, -1.0]], dtype=torch.float32)
        _, p, dot_logp, dot_p = m.infinitesimal_signals(logits)
        torch.testing.assert_close(dot_p, p*dot_logp, rtol=0, atol=0)
        self.assertLess(abs(float(dot_p.double().sum())), 1e-6)

    def test_synthetic_finite_difference_only(self):
        logits = torch.tensor([[1.25, -0.5, 0.3, 2.0]], dtype=torch.float32)
        _, _, dot_logp, dot_p = m.infinitesimal_signals(logits)
        step = 0.001
        fd_logp = (torch.log_softmax((1+step)*logits, -1)-
                   torch.log_softmax((1-step)*logits, -1))/(2*step)
        fd_p = (torch.softmax((1+step)*logits, -1)-
                torch.softmax((1-step)*logits, -1))/(2*step)
        torch.testing.assert_close(dot_logp, fd_logp, rtol=0.002, atol=0.0002)
        torch.testing.assert_close(dot_p, fd_p, rtol=0.002, atol=0.0002)

    def test_rank_boundaries_full_and_small_vocabularies(self):
        widths = [end-start for _, start, end in m.rank_intervals(20000)]
        self.assertEqual(widths, [1, 3, 12, 48, 192, 768, 3072, 12288, 3616])
        self.assertEqual([end-start for _, start, end in m.rank_intervals(20)],
                         [1, 3, 12, 4, 0, 0, 0, 0, 0])
        self.assertEqual(sum(widths), 20000)

    def test_cumulative_top_k_and_bin_accounting(self):
        logits = toy_logits()
        accumulator = m.SpectrumAccumulator(m.FROZEN_SPEC['infinitesimal_signal'])
        accumulator.add_logits(logits)
        result = accumulator.result()
        self.assertEqual(result['number_of_prediction_positions'], 127)
        self.assertEqual(sum(row['token_count'] for row in result['rank_bins']), 127*20)
        self.assertAlmostEqual(sum(row['mean_probability_mass'] for row in result['rank_bins']), 1)
        ranked = logits.reshape(-1, 20).sort(dim=-1, descending=True).values
        _, _, dl, dp = m.infinitesimal_signals(ranked)
        for k in (1, 4, 16):
            record = next(row for row in result['cumulative_top_k'] if row['requested_k'] == k)
            for source, prefix in ((dp, 'dot_p'), (dl, 'dot_logp')):
                self.assertAlmostEqual(record['fraction_absolute_'+prefix],
                    float(source[:, :k].double().abs().sum()/source.double().abs().sum()), places=12)
                self.assertAlmostEqual(record['fraction_squared_'+prefix],
                    float(source[:, :k].double().square().sum()/source.double().square().sum()), places=12)
        for key in ('fraction_absolute_dot_p', 'fraction_squared_dot_p',
                    'fraction_absolute_dot_logp', 'fraction_squared_dot_logp'):
            self.assertEqual(result['cumulative_top_k'][-1][key], 1.0)

    def test_bin_statistics_match_direct_reductions(self):
        logits = toy_logits()
        acc = m.SpectrumAccumulator(m.FROZEN_SPEC['infinitesimal_signal'])
        acc.add_logits(logits)
        first = acc.result()['rank_bins'][0]
        ranked = logits.reshape(-1, 20).sort(dim=-1, descending=True).values
        _, p, dl, dp = m.infinitesimal_signals(ranked)
        self.assertEqual(first['token_count'], 127)
        self.assertAlmostEqual(first['mean_probability_mass'], float(p[:, 0].double().mean()))
        self.assertAlmostEqual(first['sum_absolute_dot_p'], float(dp[:, 0].double().abs().sum()))
        self.assertAlmostEqual(first['rms_dot_p'], math.sqrt(float(dp[:, 0].double().square().mean())))
        self.assertAlmostEqual(first['sum_absolute_dot_logp'], float(dl[:, 0].double().abs().sum()))

    def test_conservation_passes_and_fails(self):
        logits = toy_logits().reshape(-1, 20)
        _, _, _, dp = m.infinitesimal_signals(logits)
        policy = m.FROZEN_SPEC['infinitesimal_signal']
        audit = m.conservation_audit(dp, policy)
        self.assertEqual(audit['checked_prediction_positions'], 127)
        broken = dp.clone()
        broken[0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, 'conservation'):
            m.conservation_audit(broken, policy)

    def test_nonfinite_input_fails_closed(self):
        logits = toy_logits()
        logits[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError):
            m.SpectrumAccumulator(m.FROZEN_SPEC['infinitesimal_signal']).add_logits(logits)

    def test_deterministic_aggregation(self):
        first, second = toy_logits(2)[:1], toy_logits(2)[1:]
        a = m.SpectrumAccumulator(m.FROZEN_SPEC['infinitesimal_signal'])
        b = m.SpectrumAccumulator(m.FROZEN_SPEC['infinitesimal_signal'])
        for acc in (a, b):
            acc.add_logits(first)
            acc.add_logits(second)
        self.assertEqual(a.result(), b.result())
        c = m.SpectrumAccumulator(m.FROZEN_SPEC['infinitesimal_signal'])
        c.add_logits(torch.cat((first, second)))
        for row_a, row_c in zip(a.result()['rank_bins'], c.result()['rank_bins']):
            for key in ('sum_absolute_dot_p', 'sum_squared_dot_p',
                        'sum_absolute_dot_logp', 'sum_squared_dot_logp'):
                self.assertAlmostEqual(row_a[key], row_c[key], places=10)

    def test_information_firewall_has_no_privileged_paths(self):
        source = SCRIPT.read_text()
        spec = SPEC.read_text()
        for forbidden in ('/models/base', 'oracle_adl', '/adapter', 'true_delta',
                          'known_base', 'base_directory'):
            self.assertNotIn(forbidden, source)
            self.assertNotIn(forbidden, spec)
        self.assertFalse(m.FROZEN_SPEC['outputs']['privileged_evaluation'])

    def test_output_refusal(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'construction-manifest.json'
            m.require_output_absent(output)
            output.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                m.require_output_absent(output)

    def test_nonfrozen_input_path_rejected_before_access(self):
        args = m.parse_args(['--final-directory', '/tmp/not-the-frozen-final-model'])
        with patch.object(m.a, 'checkpoint_file_records') as checkpoint_read:
            with self.assertRaisesRegex(ValueError, 'path differs'):
                m.validate_inputs(args)
            checkpoint_read.assert_not_called()

    def test_synthetic_smoke_writes_no_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'construction-manifest.json'
            args = m.parse_args(['--device', 'cpu', '--smoke-only',
                                  '--manifest-path', str(output)])
            tokens = torch.arange(4096*128, dtype=torch.int64).reshape(4096, 128)
            stream = io.StringIO()
            with patch.object(m, 'validate_inputs', return_value=(m.FROZEN_SPEC, tokens, {}, lambda: None)), \
                 patch.object(m.a, 'load_local_model', return_value=ToyFinalModel()), \
                 contextlib.redirect_stdout(stream):
                m.diagnose(args)
            record = json.loads(stream.getvalue())
            self.assertTrue(record['smoke_only'])
            self.assertEqual(record['batch_indices'], [0, 1])
            self.assertEqual(record['number_of_sequences'], 16)
            self.assertEqual(record['number_of_prediction_positions'], 16*127)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
