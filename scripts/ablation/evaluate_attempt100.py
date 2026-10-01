#!/usr/bin/env python3
"""Post-freeze signed oracle evaluation of Attempt 100's fixed eight-way consensus."""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
ATTEMPT = '100_diverse8_raw_jg_consensus_prefix0_13'
PRIOR_SOURCE = Path(__file__).with_name('evaluate_attempt014.py')
PRIOR_SOURCE_SHA256 = '1193860390504d614522ae746f903d7304eb97d9b2b8d362fabcd302a4a77afd'
if hashlib.sha256(PRIOR_SOURCE.read_bytes()).hexdigest() != PRIOR_SOURCE_SHA256:
    raise ValueError('Frozen Attempt014 oracle evaluator source changed')
loader = importlib.util.spec_from_file_location('attempt100_oracle_validation014', PRIOR_SOURCE)
prior = importlib.util.module_from_spec(loader)
loader.loader.exec_module(prior)

CORPORA = ('fineweb', 'wikitext103_raw', 'tinystories', 'arxiv_document',
           'cc_news', 'pg19', 'codeparrot_clean', 'ultrachat')
SCIENTIFIC = 'consensus_response_8'
INDIVIDUALS = tuple('response_'+name for name in CORPORA)
TENSORS = ('merged_mean', *INDIVIDUALS, SCIENTIFIC)
REPORT_ORDER = (SCIENTIFIC, *INDIVIDUALS)
SPEC_SHA256 = 'a10b513387635cd599971b10a3c7614a7a18483db7b12f9058898c417101db12'
MANIFEST_SHA256 = '64c4d3ef379fc02f6132d6e3197632e7bd5154398ee123e5ca688be9d5c68546'
CONSTRUCTOR_SHA256 = 'c5109e6dcdae0234c42aace7b9f9bf19b9563d9b0b289342bba52c32122bac0a'
FREEZER_SHA256 = 'b4531ccbad9ebc05467a5da8b49d544bee6a900c3fa35c3c23e568e878e214f2'
LOCK_SHA256 = 'c0cdd8f10ca70c0db5caa27b8e3488b0ea5fb6d6cf031e744a6d7a82ee58ffc8'
ARTIFACT_SHA256 = 'ce32f037d594a45126d72f88ffcf1dfef49996bc6152aeb431f07851c50d3872'
RAW_HASHES = {
    'merged_mean': '0f5cb336c4c54341734189bea5e2318e6aa292fed487334f68fbdd0cd8eca796',
    'response_fineweb': '00d0d8873df90e4b7c3e8799e0e019d3f5b8d7fee0d9aa796370849ff69ec5e5',
    'response_wikitext103_raw': '3d81d1479c0808868c78e5b251f329eb22143ccac16a615cabcb0874a0e15e3f',
    'response_tinystories': '4ea5f0006ab442c500ee196f3bd9cff6d5f7fe7325746e0de97ba215e54ce123',
    'response_arxiv_document': '221d1a89dd4ba20bf26d57a3c8ceffc4d67950ece81b8436d75bd77cbd11f963',
    'response_cc_news': '342f442c8dd5f0a9af95702479624abab0f1b1c424d18a8ff96314bf9a2f4ed4',
    'response_pg19': 'cddef2b4f5acecdfd48e12ff02e31d367637ab2b69da7a88c9cf6faf171c8672',
    'response_codeparrot_clean': 'd4630da18a86faabc26a533cbde9d6537f24251d5e272773d4f1340800ef9282',
    'response_ultrachat': '7b98750fd507e7a5e3c794fbb9e93d8af1b687a8acb2d6a2005ce414bacea115',
    SCIENTIFIC: 'd8ae597ebeb6495dbb9825372c99ec203db47ece85f0f5018546efa2c1a1ac26',
}
CORPUS_MANIFESTS = (
    ('arxiv_document', 'e98fdee75374083dbd516169248f52395b7b1e51d138725c5c02ec4f194b3174', '240aaf1a969b3f8cd0ade6986bfad0cd730ee288'),
    ('cc_news', 'e9adda41503de9fe23d941dc081ecbaeddf7ecd8ea625c34fb5f515b1495e6df', '81eb2ce0d2a9dad6ad16b68ef750ec290880fa36'),
    ('pg19', '650c2239ecac7bbb758a351ef80b379d1cabe39b6af3835fd1350368b01628a3', 'c021754c8e01c5b1cc83a1f549c1f97fbbb756b8'),
    ('codeparrot_clean', '8e34288e799aa434c814c05cca054b04a8b6679b2b89cbcfb267e5118d370d43', '35a59fb025bc0a102f7d96eac09d145b896d487b'),
    ('ultrachat', '74eb7aaf856be5170f61e9777b4c6fa1b598d2ee8180d4a0753f31faaad57eb3', '8049631c405ae6576f93f445c6b8166f76f5505a'),
)
ORACLE_SHA256 = '339a6c09b10286369a5d10eb49e144d1819f4301e334c63930ca7c4dd3a11077'
ORACLE_DIFFERENCE_SHA256 = 'd60789a2c06eb82dba2f2449d8665ec665d85e9f7e33fdea1f66ac79c81c159d'
BENCHMARK_SHA256 = 'e27c011aa74dffd295351c319b4611782289009e72d68631edd3b18999da1337'
BENCHMARK_SPEC_SHA256 = '0a9ec4d0f4124cf4f0456c139c4d09e902ff14104d8fb54c401f4354ecc4ab0c'
OLD_ARTIFACT_SHA256 = '75ffea77c3d3cfe6abea1e1cd4d2d0a80cf1225aa7bc5724403c61795459f3b5'
OLD_CONSENSUS_RAW_SHA256 = '24a4626c6d6d31ac482aa1a63a91991260ca1c642c3a007c9f01d752fdab0a60'
BENCHMARK_PRIMARY = 0.4434394116233976
BENCHMARK_FINEWEB_PRIMARY = 0.4420842139546442

FROZEN_SPEC = {
    'format_version': 1, 'attempt_id': ATTEMPT,
    'purpose': 'post_freeze_signed_evaluation_of_fixed_eight_corpus_consensus',
    'construction_inputs': {
        'spec': {'path': 'experiments/attempts/'+ATTEMPT+'/spec.json', 'sha256': SPEC_SHA256},
        'manifest': {'path': 'experiments/attempts/'+ATTEMPT+'/construction-manifest.json',
                     'sha256': MANIFEST_SHA256},
        'constructor': {'path': 'scripts/ablation/construct_attempt100_diverse8_consensus.py',
                        'sha256': CONSTRUCTOR_SHA256},
        'freezer': {'path': 'scripts/ablation/freeze_attempt100_diverse_corpora.py',
                    'sha256': FREEZER_SHA256},
        'corpus_lock': {'path': 'experiments/attempts/'+ATTEMPT+'/corpus-lock.json',
                        'sha256': LOCK_SHA256}},
    'corpus_manifests': [
        {'name': name, 'path': 'experiments/attempts/'+ATTEMPT+'/'+name+'-corpus-manifest.json',
         'sha256': digest, 'revision': revision}
        for name, digest, revision in CORPUS_MANIFESTS],
    'artifact': {'path': '/root/model-diff-scratch/artifacts/attempt100_diverse8_raw_jg_consensus/responses.pt',
                 'serialized_sha256': ARTIFACT_SHA256, 'raw_tensors_sha256': RAW_HASHES,
                 'tensor_order': list(TENSORS), 'shape': [128, 2048],
                 'dtype': 'contiguous_cpu_float32'},
    'oracle': {'path': '/root/model-diff-scratch/artifacts/oracle_adl/oracle_adl.pt',
               'serialized_sha256': ORACLE_SHA256,
               'difference_raw_sha256': ORACLE_DIFFERENCE_SHA256,
               'vector': 'difference', 'definition': 'ft_mean - base_mean',
               'validator_source': {'path': 'scripts/ablation/evaluate_attempt014.py',
                                    'sha256': PRIOR_SOURCE_SHA256},
               'provenance_inputs': copy.deepcopy(prior.FROZEN_SPEC['inputs'])},
    'benchmark': {'path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/evaluation.json',
                  'sha256': BENCHMARK_SHA256,
                  'evaluation_spec_path': 'experiments/attempts/014_multicorpus_raw_jg_consensus_prefix0_13/evaluation_spec.json',
                  'evaluation_spec_sha256': BENCHMARK_SPEC_SHA256,
                  'consensus_primary': BENCHMARK_PRIMARY,
                  'fineweb_primary': BENCHMARK_FINEWEB_PRIMARY,
                  'consensus_secondary': 'read_exact_value_from_hash_pinned_attempt014_evaluation',
                  'use': 'context_only_no_selection_or_tuning'},
    'metrics': {'cosine_dtype': 'cpu_float64', 'sign': 'stored_positive_Jg',
                'scientific_candidate': SCIENTIFIC,
                'report_order': list(REPORT_ORDER),
                'individual_corpora': 'descriptive_only',
                'per_position': list(range(128)),
                'primary_positions': [1, 2, 3, 4],
                'secondary_positions': list(range(1, 128)),
                'aggregation': 'arithmetic_mean_of_individual_signed_position_cosines',
                'zero_norm_cosine': None, 'required_mean_if_any_null': None,
                'flattened_cosine': False, 'absolute_cosine': False},
    'output_path': 'experiments/attempts/'+ATTEMPT+'/evaluation.json',
    'output_policy': {'overwrite': False, 'timestamps': False,
                      'construction_frozen_before_oracle': True,
                      'candidate_modified_after_oracle': False,
                      'sign_selection': False, 'candidate_selection': False,
                      'corpus_selection': False, 'corpus_reweighting': False,
                      'candidate_rescaling': False, 'position_selection': False},
}


def path_of(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT/path


def require_hash(path, expected):
    prior.require_file_hash(path, expected)


def require_output_absent(path):
    if path.exists() or path.is_symlink():
        raise ValueError('Evaluation output already exists')


def validate_construction(spec, manifest, lock, corpus_manifests):
    expected = FROZEN_SPEC
    inputs = expected['construction_inputs']
    if (spec.get('attempt_id') != ATTEMPT or manifest.get('attempt_id') != ATTEMPT or
            manifest.get('format_version') != 1 or manifest.get('hash_algorithm') != 'sha256' or
            manifest.get('purpose') != spec.get('purpose') or
            manifest.get('information_policy') != spec.get('information_policy') or
            manifest.get('spec_sha256') != SPEC_SHA256 or
            manifest.get('constructor_sha256') != CONSTRUCTOR_SHA256 or
            manifest.get('freezer_sha256') != FREEZER_SHA256 or
            manifest.get('corpus_lock_sha256') != LOCK_SHA256 or
            manifest.get('source_checkpoint') != spec.get('canonical_checkpoint_files') or
            manifest.get('probe') != spec.get('probe') or
            manifest.get('generic_loss') != spec.get('generic_loss') or
            manifest.get('tangent') != spec.get('tangent') or
            manifest.get('jvp') != spec.get('jvp') or
            manifest.get('readout') != spec.get('readout') or
            manifest.get('consensus') != spec.get('consensus') or
            manifest.get('corpus_order') != list(CORPORA) or
            manifest.get('workload') != spec.get('workload')):
        raise ValueError('Attempt100 frozen construction provenance mismatch')
    if (spec.get('consensus', {}).get('weights') != [0.125]*8 or
            spec.get('consensus', {}).get('renormalize_final') is not False or
            spec.get('outputs', {}).get('tensors') != list(TENSORS)):
        raise ValueError('Attempt100 scientific candidate definition mismatch')
    for flag, value in spec['firewall'].items():
        if value is not False or manifest.get(flag) is not False:
            raise ValueError('Attempt100 blind construction firewall mismatch: '+flag)
    artifact = manifest.get('artifact', {})
    if (artifact.get('path') != expected['artifact']['path'] or
            artifact.get('serialized_sha256') != ARTIFACT_SHA256 or
            artifact.get('raw_tensors_sha256') != RAW_HASHES):
        raise ValueError('Attempt100 construction artifact hash/inventory mismatch')
    old_record = spec.get('attempt014', {})
    expected_old = {key+'_path': old_record.get(key+'_sha256')
                    for key in ('spec', 'manifest', 'constructor')}
    expected_old['artifact_path'] = old_record.get('artifact_serialized_sha256')
    expected_old['raw_tensors_sha256'] = old_record.get('raw_sha256')
    if manifest.get('old_attempt014') != expected_old:
        raise ValueError('Attempt014 frozen construction linkage mismatch')
    if (not isinstance(lock, dict) or lock.get('attempt_id') != ATTEMPT or
            lock.get('format_version') != 1 or lock.get('spec_sha256') != SPEC_SHA256 or
            lock.get('resolution') != 'one_time_dataset_HEAD' or
            len(lock.get('corpora', [])) != 5 or
            len(spec.get('new_corpora', [])) != 5 or
            len(manifest.get('new_corpora', [])) != 5 or
            len(corpus_manifests) != 5):
        raise ValueError('Attempt100 corpus lock/inventory mismatch')
    tokenizer_files = [row for row in spec['canonical_checkpoint_files']
                       if row['path'] in ('chat_template.jinja', 'config.json',
                                         'tokenizer.json', 'tokenizer_config.json')]
    for i, expected_row in enumerate(expected['corpus_manifests']):
        corpus = spec['new_corpora'][i]
        record = manifest['new_corpora'][i]
        locked = lock['corpora'][i]
        frozen = corpus_manifests[i]
        if (corpus['name'] != expected_row['name'] or
                record.get('name') != corpus['name'] or
                record.get('corpus_manifest_sha256') != expected_row['sha256'] or
                record.get('corpus_revision') != expected_row['revision'] or
                locked != {'name': corpus['name'], 'repo_id': corpus['repo_id'],
                           'config': corpus['config'], 'split': corpus['split'],
                           'revision': expected_row['revision']} or
                frozen.get('attempt_id') != ATTEMPT or
                frozen.get('format_version') != 1 or
                frozen.get('hash_algorithm') != 'sha256' or
                frozen.get('corpus_name') != corpus['name'] or
                any(frozen.get(key) != corpus[key] for key in
                    ('repo_id', 'config', 'split', 'extractor', 'source_field')) or
                frozen.get('revision') != expected_row['revision'] or
                frozen.get('spec_sha256') != SPEC_SHA256 or
                frozen.get('lock_sha256') != LOCK_SHA256 or
                frozen.get('freeze_script_sha256') != FREEZER_SHA256 or
                frozen.get('selection') != spec['corpus_freeze'] or
                frozen.get('tokenizer_files') != tokenizer_files or
                frozen.get('artifact', {}).get('path') != corpus['tokens_path'] or
                frozen.get('artifact', {}).get('serialized_sha256') != record.get('corpus_artifact_sha256')):
            raise ValueError('Attempt100 frozen corpus provenance mismatch: '+corpus['name'])
    return artifact


def load_construction_artifact(path):
    require_hash(path, ARTIFACT_SHA256)
    value = prior.load_torch_artifact(path, 'Attempt100 construction artifact', torch)
    if not isinstance(value, dict) or set(value) != set(TENSORS):
        raise ValueError('Attempt100 artifact tensor inventory mismatch')
    for name in TENSORS:
        prior.validate_activation_tensor(value[name], name, RAW_HASHES[name], torch)
    return value


def position_cosines(candidate, difference):
    if (not isinstance(candidate, torch.Tensor) or not isinstance(difference, torch.Tensor) or
            candidate.dtype != torch.float32 or difference.dtype != torch.float32 or
            candidate.device.type != 'cpu' or difference.device.type != 'cpu' or
            tuple(candidate.shape) != (128, 2048) or tuple(difference.shape) != (128, 2048) or
            not bool(torch.isfinite(candidate).all()) or not bool(torch.isfinite(difference).all())):
        raise ValueError('Invalid signed-cosine tensor inventory')
    return prior.position_cosines(candidate, difference)


def required_mean(values):
    return None if any(value is None for value in values) else math.fsum(values)/len(values)


def report(name, candidate, difference):
    if name not in REPORT_ORDER:
        raise ValueError('Unknown frozen response tensor')
    values = position_cosines(candidate, difference)
    return {'candidate': name,
            'role': 'scientific_candidate' if name == SCIENTIFIC else 'descriptive_individual_corpus',
            'position_0_cosine': values[0],
            'positions_1_4_individual_cosines': values[1:5],
            'positions_1_4_mean_cosine': required_mean(values[1:5]),
            'positions_1_127_mean_cosine': required_mean(values[1:128]),
            'positions': [{'position': p, 'cosine_similarity': value}
                          for p, value in enumerate(values)]}


def report_all(artifact, difference):
    if not isinstance(artifact, dict) or set(artifact) != set(TENSORS):
        raise ValueError('All nine frozen responses and merged mean are required')
    return [report(name, artifact[name], difference) for name in REPORT_ORDER]


def validate_benchmark(benchmark):
    if (not isinstance(benchmark, dict) or
            benchmark.get('attempt_id') != '014_multicorpus_raw_jg_consensus_prefix0_13' or
            benchmark.get('primary', {}).get('consensus') != BENCHMARK_PRIMARY or
            benchmark.get('primary', {}).get('fineweb') != BENCHMARK_FINEWEB_PRIMARY or
            benchmark.get('provenance', {}).get('evaluator_script_sha256') != PRIOR_SOURCE_SHA256 or
            benchmark.get('provenance', {}).get('evaluation_spec_sha256') != BENCHMARK_SPEC_SHA256 or
            benchmark.get('provenance', {}).get('oracle_artifact_sha256') != ORACLE_SHA256 or
            benchmark.get('provenance', {}).get('oracle_raw_tensors_sha256', {}).get('difference') != ORACLE_DIFFERENCE_SHA256 or
            benchmark.get('provenance', {}).get('construction_artifact_sha256') != OLD_ARTIFACT_SHA256):
        raise ValueError('Frozen Attempt014 benchmark identity/provenance mismatch')
    for key, row in prior.FROZEN_SPEC['inputs'].items():
        if benchmark['provenance'].get(key+'_sha256') != row['sha256']:
            raise ValueError('Frozen Attempt014 benchmark input provenance mismatch: '+key)
    old_raw = {name: RAW_HASHES[name] for name in TENSORS[:4]}
    old_raw['consensus_response'] = OLD_CONSENSUS_RAW_SHA256
    if benchmark['provenance'].get('construction_raw_tensors_sha256') != old_raw:
        raise ValueError('Frozen Attempt014 benchmark raw construction provenance mismatch')
    rows = benchmark.get('candidates')
    if (not isinstance(rows, list) or len(rows) != 4 or
            set(row.get('candidate') for row in rows if isinstance(row, dict)) !=
            {'response_fineweb', 'response_wikitext103_raw', 'response_tinystories', 'consensus_response'}):
        raise ValueError('Frozen Attempt014 benchmark candidate inventory mismatch')
    by_name = {row['candidate']: row for row in rows}
    consensus = by_name['consensus_response']
    fineweb = by_name['response_fineweb']
    secondary = benchmark.get('secondary', {}).get('consensus')
    def recomputed(row, positions):
        values = row.get('positions')
        if (not isinstance(values, list) or len(values) != 128 or
                any(not isinstance(value, dict) or value.get('position') != p
                    for p, value in enumerate(values))):
            raise ValueError('Frozen Attempt014 benchmark position inventory mismatch')
        return required_mean([values[p].get('cosine_similarity') for p in positions])
    if (consensus.get('positions_1_4_mean_cosine') != BENCHMARK_PRIMARY or
            fineweb.get('positions_1_4_mean_cosine') != BENCHMARK_FINEWEB_PRIMARY or
            not isinstance(secondary, (float, int)) or not math.isfinite(secondary) or
            consensus.get('positions_1_127_mean_cosine') != secondary or
            recomputed(consensus, range(1, 5)) != BENCHMARK_PRIMARY or
            recomputed(consensus, range(1, 128)) != secondary or
            recomputed(fineweb, range(1, 5)) != BENCHMARK_FINEWEB_PRIMARY):
        raise ValueError('Frozen Attempt014 benchmark values mismatch')
    return {'consensus_primary': BENCHMARK_PRIMARY,
            'fineweb_primary': BENCHMARK_FINEWEB_PRIMARY,
            'consensus_secondary': secondary}


def validate_oracle_manifest(manifest):
    if (not isinstance(manifest, dict) or
            manifest.get('oracle_adl_sha256') != ORACLE_SHA256 or
            manifest.get('raw_tensors_sha256', {}).get('difference') != ORACLE_DIFFERENCE_SHA256):
        raise ValueError('Frozen oracle artifact/difference hash mismatch')
    return manifest


def build_result(reports, benchmark, provenance):
    if [row['candidate'] for row in reports] != list(REPORT_ORDER):
        raise ValueError('Scientific and descriptive response inventory/order mismatch')
    if (reports[0].get('role') != 'scientific_candidate' or
            any(row.get('role') != 'descriptive_individual_corpus' for row in reports[1:])):
        raise ValueError('Response role/selection mismatch')
    primary = reports[0]['positions_1_4_mean_cosine']
    secondary = reports[0]['positions_1_127_mean_cosine']
    comparison = {
        'attempt100_primary': primary,
        'attempt014_consensus_primary': benchmark['consensus_primary'],
        'attempt014_fineweb_primary': benchmark['fineweb_primary'],
        'delta_vs_attempt014_consensus_primary': None if primary is None else primary-benchmark['consensus_primary'],
        'delta_vs_attempt014_fineweb_primary': None if primary is None else primary-benchmark['fineweb_primary'],
        'attempt100_secondary': secondary,
        'attempt014_consensus_secondary': benchmark['consensus_secondary'],
        'delta_vs_attempt014_consensus_secondary': None if secondary is None else secondary-benchmark['consensus_secondary'],
    }
    return {'format_version': 1, 'attempt_id': ATTEMPT,
            'evaluation_plan': FROZEN_SPEC['metrics'], 'provenance': provenance,
            'scientific_candidate': SCIENTIFIC, 'individual_corpora': 'descriptive_only',
            'candidates': reports, 'matched_attempt014_comparison': comparison,
            'construction_frozen_before_oracle': True,
            'candidate_modified_after_oracle': False,
            'sign_selection': False, 'candidate_selection': False,
            'corpus_selection': False, 'corpus_reweighting': False,
            'candidate_rescaling': False, 'position_selection': False,
            'oracle_vector': 'difference', 'cosine_dtype': 'cpu_float64'}


def evaluate(spec_path):
    expected_path = PROJECT/'experiments/attempts'/ATTEMPT/'evaluation_spec.json'
    if spec_path.resolve() != expected_path.resolve():
        raise ValueError('Evaluation spec path differs from frozen path')
    spec = prior.load_json_object(spec_path, 'Attempt100 evaluation spec')
    if (set(spec) != set(FROZEN_SPEC) | {'evaluator_source_sha256'} or
            {key: value for key, value in spec.items() if key != 'evaluator_source_sha256'} != FROZEN_SPEC):
        raise ValueError('Attempt100 evaluation spec differs from frozen definition')
    output = path_of(spec['output_path'])
    require_output_absent(output)
    construction_paths = {key: path_of(row['path'])
                          for key, row in spec['construction_inputs'].items()}
    corpus_paths = [path_of(row['path']) for row in spec['corpus_manifests']]
    oracle_paths = {key: path_of(row['path'])
                    for key, row in spec['oracle']['provenance_inputs'].items()}
    source = Path(__file__).resolve()
    artifact_path = path_of(spec['artifact']['path'])
    oracle_path = path_of(spec['oracle']['path'])
    benchmark_path = path_of(spec['benchmark']['path'])
    benchmark_spec_path = path_of(spec['benchmark']['evaluation_spec_path'])
    inputs = [spec_path, source, PRIOR_SOURCE, artifact_path, oracle_path,
              benchmark_path, benchmark_spec_path, *construction_paths.values(), *corpus_paths,
              *oracle_paths.values()]
    if any(output.resolve() == path.resolve() or path.resolve().is_relative_to(output.resolve())
           for path in inputs):
        raise ValueError('Evaluation output overlaps frozen input')
    hashes = {path: prior.sha256_file(path) for path in inputs}
    for key, row in spec['construction_inputs'].items():
        if hashes[construction_paths[key]] != row['sha256']:
            raise ValueError('Frozen construction input hash mismatch: '+key)
    for path, row in zip(corpus_paths, spec['corpus_manifests']):
        if hashes[path] != row['sha256']:
            raise ValueError('Frozen corpus manifest hash mismatch: '+row['name'])
    for key, row in spec['oracle']['provenance_inputs'].items():
        if hashes[oracle_paths[key]] != row['sha256']:
            raise ValueError('Frozen oracle provenance input hash mismatch: '+key)
    if (hashes[source] != prior.require_sha256(spec['evaluator_source_sha256'], 'Attempt100 evaluator source') or
            hashes[PRIOR_SOURCE] != PRIOR_SOURCE_SHA256 or
            hashes[artifact_path] != ARTIFACT_SHA256 or
            hashes[oracle_path] != ORACLE_SHA256 or
            hashes[benchmark_path] != BENCHMARK_SHA256 or
            hashes[benchmark_spec_path] != BENCHMARK_SPEC_SHA256):
        raise ValueError('Frozen evaluator/artifact/oracle/benchmark hash mismatch')
    construction_spec = prior.load_json_object(construction_paths['spec'], 'Attempt100 construction spec')
    manifest = prior.load_json_object(construction_paths['manifest'], 'Attempt100 construction manifest')
    lock = prior.load_json_object(construction_paths['corpus_lock'], 'Attempt100 corpus lock')
    corpus_manifests = [prior.load_json_object(path, 'Attempt100 corpus manifest') for path in corpus_paths]
    validate_construction(construction_spec, manifest, lock, corpus_manifests)
    oracle_attempt = prior.load_json_object(oracle_paths['attempt_spec'], 'Attempt014 construction spec')
    oracle_expected = {key: row['sha256'] for key, row in spec['oracle']['provenance_inputs'].items()}
    oracle_manifest = validate_oracle_manifest(prior.validate_oracle_provenance(
        oracle_paths, oracle_expected, oracle_attempt))
    artifact = load_construction_artifact(artifact_path)
    oracle = prior.load_and_validate_oracle(oracle_path, {'oracle': {'oracle_manifest': oracle_manifest}}, torch)
    benchmark = validate_benchmark(prior.load_json_object(benchmark_path, 'Attempt014 frozen evaluation'))
    def recheck():
        for path, digest in hashes.items():
            require_hash(path, digest)
    recheck()
    reports = report_all(artifact, oracle['difference'])
    provenance = {'evaluation_spec_sha256': hashes[spec_path],
                  'evaluator_source_sha256': hashes[source],
                  'oracle_validator_source_sha256': hashes[PRIOR_SOURCE],
                  'construction_spec_sha256': hashes[construction_paths['spec']],
                  'construction_manifest_sha256': hashes[construction_paths['manifest']],
                  'constructor_sha256': hashes[construction_paths['constructor']],
                  'freezer_sha256': hashes[construction_paths['freezer']],
                  'corpus_lock_sha256': hashes[construction_paths['corpus_lock']],
                  'corpus_manifest_sha256': {row['name']: hashes[path]
                                             for row, path in zip(spec['corpus_manifests'], corpus_paths)},
                  'construction_artifact_sha256': hashes[artifact_path],
                  'construction_raw_tensors_sha256': RAW_HASHES,
                  'oracle_artifact_sha256': hashes[oracle_path],
                  'oracle_raw_tensors_sha256': oracle_manifest['raw_tensors_sha256'],
                  'oracle_provenance_input_sha256': oracle_expected,
                  'attempt014_evaluation_sha256': hashes[benchmark_path],
                  'attempt014_evaluation_spec_sha256': hashes[benchmark_spec_path]}
    result = build_result(reports, benchmark, provenance)
    recheck()
    require_output_absent(output)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-spec', type=Path,
                        default=PROJECT/'experiments/attempts'/ATTEMPT/'evaluation_spec.json')
    args = parser.parse_args()
    evaluate(args.evaluation_spec)


if __name__ == '__main__':
    main()
