#!/usr/bin/env python3
"""Known-base full-support final-endpoint Hessian forward diagnostic."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
PROJECT = Path(__file__).resolve().parents[2]

def helper(name):
    path = Path(__file__).with_name(name)
    loader = importlib.util.spec_from_file_location('attempt017_' + path.stem, path)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module

execution = helper('construct_exact_displacement_jvp_ceiling.py')
a = execution.a
loss_helper = helper('construct_known_base_endpoint_hessian_validation.py')
FROZEN_SPEC = {'format_version': 1,
 'attempt_id': '017_known_base_full_support_endpoint_hessian_forward_prefix0_27',
 'purpose': 'post_oracle_known_base_hessian_forward_diagnostic_not_blind_recovery',
 'interpretation': 'Forward agreement or disagreement does not establish inverse recovery accuracy. Only '
                   'final-endpoint H1 is computed.',
 'information_policy': 'historical_base_final_frozen_005_corpus_and_required_provenance_and_source_only',
 'base': {'repo_id': 'Qwen/Qwen3-1.7B',
          'revision': '0060bc56d46589041c1048efd1a397421b1142b5',
          'files': [{'path': '.gitattributes',
                     'size_bytes': 1570,
                     'sha256': '34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930'},
                    {'path': 'README.md',
                     'size_bytes': 13963,
                     'sha256': '257e52c419dac2258852643f18af6c974f21f8c6c1b6f371b6cca6201cf29091'},
                    {'path': 'config.json',
                     'size_bytes': 726,
                     'sha256': '1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197'},
                    {'path': 'generation_config.json',
                     'size_bytes': 239,
                     'sha256': '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2'},
                    {'path': 'merges.txt',
                     'size_bytes': 1671853,
                     'sha256': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'},
                    {'path': 'model-00001-of-00002.safetensors',
                     'size_bytes': 3441185608,
                     'sha256': '169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5'},
                    {'path': 'model-00002-of-00002.safetensors',
                     'size_bytes': 622329984,
                     'sha256': '912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9'},
                    {'path': 'model.safetensors.index.json',
                     'size_bytes': 25605,
                     'sha256': '0d660e94b165eb912669a5249dff44b83188c4777a07ddb9611fb78d91b0578d'},
                    {'path': 'tokenizer.json',
                     'size_bytes': 11422654,
                     'sha256': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'},
                    {'path': 'tokenizer_config.json',
                     'size_bytes': 9732,
                     'sha256': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'},
                    {'path': 'vocab.json',
                     'size_bytes': 2776833,
                     'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}]},
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
 'model': {'model_type': 'qwen3',
           'num_hidden_layers': 28,
           'source_dtype': 'float32',
           'local_files_only': True,
           'eval_mode': True,
           'use_cache': False},
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
 'hvp_loss': {'type': 'causal_next_token_cross_entropy',
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
 'execution_semantics_audit': {'buffers': 'full_inventory_and_alias_topology; '
                                          'exact_shape_dtype_raw_hash_for_any_upstream_alias; '
                                          'downstream_only_differences_recorded',
                               'upstream_rule': 'same_conservative_causal_classification_as_parameters_including_global_rotary_buffers',
                               'config_fields': ['model_type',
                                                 'hidden_size',
                                                 'intermediate_size',
                                                 'num_hidden_layers',
                                                 'num_attention_heads',
                                                 'num_key_value_heads',
                                                 'head_dim',
                                                 'hidden_act',
                                                 'max_position_embeddings',
                                                 'rms_norm_eps',
                                                 'rope_theta',
                                                 'rope_scaling',
                                                 'attention_bias',
                                                 'attention_dropout',
                                                 'use_sliding_window',
                                                 'sliding_window',
                                                 'max_window_layers',
                                                 'layer_types',
                                                 '_attn_implementation'],
                               'config_scope': ['config', 'model.config'],
                               'config_comparison': 'presence_and_canonical_JSON_values; '
                                                    'missing_config_fails; all_listed_fields_must_match',
                               'excluded_config': 'provenance_generation_metadata_and_use_cache'},
 'buffer_policy': 'all_buffers_including_later_and_output_must_match_for_full_loss',
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
 'displacement': 'detached_contiguous_FP32(CPU_FP64(W1_P)-CPU_FP64(W0_P)); positive; no scaling',
 'cleaned_rhs': 'contiguous_FP32(CPU_FP64(g1)-CPU_FP64(g0))',
 'hessian': {'endpoint': 'W1_only',
             'primitive': 'true_double_backward',
             'direction': 'same_realized_FP32_DeltaP_every_batch',
             'accumulation': 'sum_of_512_batch_HVPs_FP32_on_parameter_device',
             'denominator': 520192,
             'fallback': False},
 'metrics': {'dtype': 'matrixwise_cpu_float64',
             'primary': 'cos(h,d)',
             'scopes': ['full', 'early_0_13', 'later_14_27', 'each_block_0_27'],
             'zero_policy': 'undefined_ratios_cosines_and_projections_null',
             'no_selection': True},
 'sources': {'scripts/ablation/construct_exact_displacement_jvp_ceiling.py': 'a682ce8a5dd01b2bc080b913d5148ba00544217be6a9b6b5a9ec66bc03da755a',
             'scripts/ablation/construct_generic_gradient_linear_response.py': '5e80d37ad8379963b8cf1ba46e99d6c98e04514aae19eb637f3242ba8eaa4840',
             'scripts/ablation/construct_known_base_endpoint_hessian_validation.py': '4ad2aaf8a7ba81a0928720b22a8d1fe0606f3c7e0cad1a51143a9665f3bad8d4',
             'scripts/ablation/smoke_test_generic_hessian_curvature_probe.py': 'c23f22b27b41ee5c0610457ef389f3eeb3264d9b410c66e6ecfa796a417ebf12'},
 'temporary': {'files': ['g0.safetensors'],
               'cleanup': 'only_after_artifact_and_manifest_success',
               'failure': 'retain_no_resume',
               'base_weights': 'read_frozen_safetensors_matrixwise_no_temporary_copy'},
 'workload': {'ordinary_gradient_batches': 1024,
              'hvp_batches': 512,
              'smoke_gradient_batches': 0,
              'smoke_hvp_batches': 1},
 'outputs': {'overwrite': False, 'timestamps': False, 'host_metadata': False, 'gpu_metadata': False},
 'defaults': {'base_directory': '/root/model-diff-scratch/models/base',
              'final_directory': '/root/model-diff-scratch/models/merged',
              'corpus_spec_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus_spec.json',
              'corpus_manifest_path': 'experiments/attempts/005_generic_gradient_rollback_prefix0_13_r00125/corpus-manifest.json',
              'tokens_path': '/root/model-diff-scratch/artifacts/attempt005_generic_rollback_corpus/tokens.pt',
              'temporary_directory': '/root/model-diff-scratch/tmp/attempt017',
              'artifact_path': '/root/model-diff-scratch/artifacts/attempt017_full_support_hessian_forward/cleaned_rhs.safetensors',
              'manifest_path': 'experiments/attempts/017_known_base_full_support_endpoint_hessian_forward_prefix0_27/construction-manifest.json'}}
LOADED_SOURCES = {path: a.sha256_file(PROJECT/path) for path in FROZEN_SPEC['sources']}


def discover(model, spec):
    layers = model.model.layers
    if len(layers) != 28:
        raise ValueError('Expected 28 blocks')
    found = []
    for block, layer in enumerate(layers):
        local = [(f'model.layers.{block}.{name}', module) for name, module in layer.named_modules()
                 if isinstance(module, torch.nn.Linear)]
        if len(local) != 7:
            raise ValueError('Expected seven Linear weights per block')
        found.extend(local)
    found.sort(key=lambda item: item[0])
    inventory = [{'name': name+'.weight', 'shape': list(module.weight.shape)} for name, module in found]
    if inventory != spec['coordinates'] or len({id(m.weight) for _, m in found}) != 196:
        raise ValueError('Exact 196 coordinate inventory mismatch')
    for name, module in found:
        if module.weight.dtype != torch.float32 or not module.weight.is_contiguous() or not torch.isfinite(module.weight).all():
            raise ValueError('Invalid selected weight: '+name)
    loss_helper.freeze_other_parameters(model, found)
    return found


def snapshot(model, spec):
    aliases = execution.aliases(model)
    return {'parameters': {name: {'shape': list(value.shape), 'aliases': aliases[name],
                                 'sha256': a.sha256_parameter(value, torch)}
                           for name, value in model.named_parameters()},
            'execution': execution.execution_snapshot(model, spec)}


def execution_audit(base, final):
    audit = execution.audit_execution_semantics(base, final)
    if base['buffers'] != final['buffers']:
        raise ValueError('Full-loss buffer semantics differ')
    return audit


def displacement(final, eligible, base, reader):
    params = dict(final.named_parameters()); aliases = execution.aliases(final)
    if set(params) != set(base):
        raise ValueError('Full parameter inventory mismatch')
    selected = {name+'.weight' for name, _ in eligible}
    tangent, rows = {}, []
    for name, value in params.items():
        record = base[name]
        if record['shape'] != list(value.shape) or record['aliases'] != aliases[name]:
            raise ValueError('Parameter shape/alias mismatch: '+name)
        changed = a.sha256_parameter(value, torch) != record['sha256']
        if name in selected or changed:
            old = reader(name, record)
            if old.dtype != torch.float32 or list(old.shape) != record['shape'] or a.sha256_parameter(old, torch) != record['sha256']:
                raise ValueError('Base tensor audit mismatch')
            delta = value.detach().to(device='cpu', dtype=torch.float64) - old.double()
            if not torch.isfinite(delta).all():
                raise ValueError('Nonfinite displacement')
            square = float(delta.square().sum())
            if name not in selected and bool(delta.ne(0).any()):
                raise ValueError('Nonzero parameter outside full support: '+name)
            if name in selected:
                rounded = delta.float().contiguous()
                if not torch.isfinite(rounded).all():
                    raise ValueError('Nonfinite FP32 direction')
                tangent[name] = rounded.to(value.device)
                rows.append({'name': name, 'shape': list(value.shape), 'norm': math.sqrt(square),
                             'squared_norm': square, 'realized_squared_norm': float(rounded.double().square().sum()),
                             'rounding_error_squared_norm': float((rounded.double()-delta).square().sum())})
    rows.sort(key=lambda r:r['name'])
    def norm(key, predicate=lambda r:True):
        return math.sqrt(math.fsum(r[key] for r in rows if predicate(r)))
    tangent = {name: tangent[name] for name in sorted(tangent)}
    return tangent, {'per_tensor': rows, 'exact_norm': norm('squared_norm'),
        'early_norm': norm('squared_norm', lambda r:block_number(r['name'])<14),
        'later_norm': norm('squared_norm', lambda r:block_number(r['name'])>=14),
        'realized_fp32_norm': norm('realized_squared_norm'), 'rounding_error_norm': norm('rounding_error_squared_norm'),
        'nonzero_names': [r['name'] for r in rows if r['squared_norm'] != 0],
        'nonzero_count': sum(r['squared_norm'] != 0 for r in rows),
        'outside_support_zero_verified': True, 'all_parameter_records': base}


def block_number(name):
    return int(name.split('.')[2])


def save_vector(path, vector, coordinates):
    if path.exists() or path.is_symlink():
        raise ValueError('Vector output already exists')
    if set(vector) != {r['name'] for r in coordinates}:
        raise ValueError('Vector names mismatch')
    for row in coordinates:
        value = vector[row['name']]
        if (value.device.type != 'cpu' or value.dtype != torch.float32 or not value.is_contiguous()
                or list(value.shape) != row['shape'] or not torch.isfinite(value).all()):
            raise ValueError('Invalid stored vector')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb'):
        pass
    save_file(vector, str(path))
    return vector_record(path, coordinates)


def vector_record(path, coordinates):
    rows = []
    with safe_open(str(path), framework='pt', device='cpu') as stored:
        if sorted(stored.keys()) != [r['name'] for r in coordinates]:
            raise ValueError('Stored vector inventory mismatch')
        for row in coordinates:
            value = stored.get_tensor(row['name'])
            if value.dtype != torch.float32 or list(value.shape) != row['shape'] or not torch.isfinite(value).all():
                raise ValueError('Invalid stored vector tensor')
            rows.append({**row, 'raw_sha256': a.sha256_raw_float32_tensor(value, torch),
                         'squared_norm': float(value.double().square().sum())})
    return {'serialized_sha256': a.sha256_file(path), 'matrices': rows,
            'norm': math.sqrt(math.fsum(r['squared_norm'] for r in rows))}


def gradient_to_file(model, eligible, tokens, settings, path, coordinates, subtract=None):
    mean_loss = loss_helper.accumulate_mean_generic_gradient(model, tokens, eligible, settings, torch)
    vector = {}
    try:
        if subtract is not None:
            stored = safe_open(str(subtract), framework='pt', device='cpu')
        else:
            stored = None
        if stored is not None:
            stored.__enter__()
        try:
            for name, module in eligible:
                key = name+'.weight'; gradient = module.weight.grad
                cpu = gradient.detach().to(device='cpu', dtype=torch.float64)
                if stored is not None:
                    cpu.sub_(stored.get_tensor(key).double())
                vector[key] = cpu.float().contiguous()
                module.weight.grad = None
        finally:
            if stored is not None:
                stored.__exit__(None, None, None)
        record = save_vector(path, vector, coordinates)
    finally:
        vector.clear(); model.zero_grad(set_to_none=True)
    return mean_loss, record


def metric_record(pairs):
    hh, dd, hd, error = [], [], [], []
    # A second streaming pass computes the scaled residual directly, avoiding cancellation.
    for h, d in pairs():
        h=h.detach().to(device='cpu', dtype=torch.float64); d=d.double()
        hh.append(float(h.square().sum())); dd.append(float(d.square().sum()))
        hd.append(float((h*d).sum())); error.append(float((d-h).square().sum()))
    H,D,C = math.fsum(hh), math.fsum(dd), math.fsum(hd)
    coefficient = C/H if H else None
    scaled = []
    if coefficient is not None and D:
        for h,d in pairs():
            scaled.append(float((d.double()-coefficient*h.detach().to(device='cpu',dtype=torch.float64)).square().sum()))
    return {'cosine': C/math.sqrt(H*D) if H and D else None, 'h_norm': math.sqrt(H), 'd_norm': math.sqrt(D),
            'norm_ratio': math.sqrt(H/D) if D else None,
            'unscaled_relative_residual': math.sqrt(math.fsum(error)/D) if D else None,
            'optimal_scalar': coefficient,
            'relative_residual_after_optimal_rescaling': math.sqrt(math.fsum(scaled)/D) if D and coefficient is not None else None}


def metrics(h, path):
    with safe_open(str(path), framework='pt', device='cpu') as stored:
        def scope(predicate):
            names=[n for n in sorted(h) if predicate(block_number(n))]
            return metric_record(lambda: ((h[n],stored.get_tensor(n)) for n in names))
        return {'full':scope(lambda b:True), 'early_0_13':scope(lambda b:b<14),
                'later_14_27':scope(lambda b:b>=14),
                'blocks':[{'block':b, **scope(lambda n:n==b)} for b in range(28)]}


def verify_state(model, before, execution_before, spec):
    a.verify_model_unchanged(model, before, torch)
    if execution.execution_snapshot(model, spec) != execution_before:
        raise ValueError('Execution semantics mutated during computation')


def require_absent(args):
    for path in (args.artifact_path,args.manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError('Scientific output already exists')
    temp=args.temporary_directory
    if temp.is_symlink() or (temp.exists() and (not temp.is_dir() or any(temp.iterdir()))):
        raise ValueError('Stale temporary state; no resume')


def validate_inputs(args):
    spec=a.load_json_object(args.spec_path,'Attempt017 spec')
    if spec != FROZEN_SPEC:
        raise ValueError('Frozen specification mismatch')
    paths=[args.spec_path,args.corpus_spec_path,args.corpus_manifest_path,args.tokens_path,Path(__file__).resolve(),
           *[PROJECT/p for p in spec['sources']]]
    hashes={p:a.sha256_file(p) for p in paths}
    for path,digest in spec['sources'].items():
        if hashes[PROJECT/path] != digest or LOADED_SOURCES[path] != digest:
            raise ValueError('Frozen helper source mismatch')
    for key,path in [('corpus_spec_sha256',args.corpus_spec_path),('corpus_manifest_sha256',args.corpus_manifest_path)]:
        if hashes[path] != spec['frozen_corpus'][key]:
            raise ValueError('Frozen corpus provenance mismatch')
    base=a.checkpoint_file_records(args.base_directory);final=a.checkpoint_file_records(args.final_directory)
    if base!=spec['base']['files'] or final!=spec['canonical_checkpoint_files']:
        raise ValueError('Checkpoint inventory/hash mismatch')
    corpus=loss_helper.verify_corpus_manifest(args.corpus_manifest_path,args.corpus_spec_path,args.tokens_path,
        args.final_directory,loss_helper.load_corpus_spec(args.corpus_spec_path))
    if any(corpus[k]!=spec['frozen_corpus'][k] for k in ('serialized_sha256','raw_tensor_sha256')):
        raise ValueError('Corpus token hash mismatch')
    tokens=loss_helper.load_corpus_tokens(args.tokens_path,corpus['raw_tensor_sha256'],
        spec['generic_loss']['sample_count'],spec['generic_loss']['sequence_length'],torch)
    outputs=[args.artifact_path,args.manifest_path,args.temporary_directory]
    for out in outputs:
        if any(out.resolve()==p.resolve() or p.resolve().is_relative_to(out.resolve()) for p in paths) or any(out.resolve().is_relative_to(d.resolve()) for d in (args.base_directory,args.final_directory)):
            raise ValueError('Output overlaps frozen input')
    if any(outputs[i].resolve().is_relative_to(outputs[j].resolve()) for i in range(3) for j in range(3) if i!=j):
        raise ValueError('Outputs overlap each other')
    def recheck():
        a.verify_unchanged(args.base_directory,base,hashes)
        a.verify_unchanged(args.final_directory,final,{})
    return spec,tokens,hashes,recheck


def construct(args):
    require_absent(args)
    spec,tokens,hashes,recheck=validate_inputs(args)
    if not args.smoke_only:
        args.temporary_directory.mkdir(parents=True,exist_ok=True)
        with (args.temporary_directory/'state.json').open('x') as stream:
            json.dump({'attempt_id':spec['attempt_id'],'resume':False},stream,sort_keys=True)
    print('Base phase',flush=True)
    base=a.load_local_model(args.base_directory,args.device,torch); eligible=discover(base,spec)
    base_before=a.model_state_hashes(base,torch); base_snapshot=snapshot(base,spec)
    g0_path=args.temporary_directory/'g0.safetensors'
    if not args.smoke_only:
        g0_loss,g0_record=gradient_to_file(base,eligible,tokens,spec['generic_loss'],g0_path,spec['coordinates'])
    verify_state(base,base_before,base_snapshot['execution'],spec)
    del eligible,base
    print('Final endpoint phase',flush=True)
    final=a.load_local_model(args.final_directory,args.device,torch);eligible=discover(final,spec)
    before=a.model_state_hashes(final,torch);final_execution=execution.execution_snapshot(final,spec)
    audit=execution_audit(base_snapshot['execution'],final_execution)
    # Audit and build Delta only once. No base model is retained on the device.
    with execution.BaseTensorReader(args.base_directory,spec['base']['files']) as reader:
        direction,delta_audit=displacement(final,eligible,base_snapshot['parameters'],reader.get)
    if not args.smoke_only:
        if vector_record(g0_path,spec['coordinates'])!=g0_record:
            raise ValueError('Temporary g0 changed')
        g1_loss,rhs_record=gradient_to_file(final,eligible,tokens,spec['generic_loss'],args.artifact_path,spec['coordinates'],g0_path)
        verify_state(final,before,final_execution,spec)
    settings=dict(spec['hvp_loss'])
    if args.smoke_only:
        settings.update(sample_count=settings['batch_size'],number_of_batches=1,
                        total_prediction_tokens=settings['batch_size']*(settings['sequence_length']-1))
        tokens=tokens[:settings['batch_size']]
    try:
        h=loss_helper.accumulate_hessian_split(final,tokens,eligible,direction,settings,torch,
            progress=lambda i,n: print(f'HVP {i}/{n}',flush=True) if i%32==0 or i==n else None)
    finally:
        direction.clear();final.zero_grad(set_to_none=True)
    verify_state(final,before,final_execution,spec)
    h_norm=loss_helper.hvp_backend.aggregate_norm(h)
    recheck()
    if args.smoke_only:
        print(json.dumps({'smoke_only':True,'hvp_batches':1,'direction_norm':delta_audit['realized_fp32_norm'],'hvp_norm':h_norm},sort_keys=True))
        h.clear();return
    diagnostic=metrics(h,args.artifact_path);h.clear()
    if vector_record(args.artifact_path,spec['coordinates'])!=rhs_record:
        raise ValueError('Persistent RHS changed')
    manifest={'format_version':1,'attempt_id':spec['attempt_id'],'purpose':spec['purpose'],
        'interpretation':spec['interpretation'],'information_policy':spec['information_policy'],'hash_algorithm':'sha256',
        'input_source_hashes':{str(p):digest for p,digest in hashes.items()},
        'base_checkpoint':spec['base'],'final_checkpoint_files':spec['canonical_checkpoint_files'],
        'corpus':spec['frozen_corpus'],'coordinates':spec['coordinates'],'generic_loss':spec['generic_loss'],
        'g0_mean_loss':g0_loss,'g1_mean_loss':g1_loss,'g0_norm':g0_record['norm'],
        'displacement':delta_audit,'execution_semantics_audit':audit,
        'hvp':{'semantics':spec['hessian'],'batch_size':settings['batch_size'],'number_of_batches':settings['number_of_batches'],
               'denominator':settings['total_prediction_tokens'],'direction_norm':delta_audit['realized_fp32_norm'],'output_norm':h_norm},
        'cleaned_rhs':{'definition':spec['cleaned_rhs'],**rhs_record},'metrics':diagnostic,'workload':spec['workload'],
        'publication_policy':'partial artifact blocks rerun; retain temporary g0 on failure; no resume'}
    recheck();a.write_manifest(args.manifest_path,manifest)
    g0_path.unlink();(args.temporary_directory/'state.json').unlink();args.temporary_directory.rmdir()


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec-path',type=Path,default=PROJECT/'experiments/attempts'/FROZEN_SPEC['attempt_id']/'spec.json')
    for key,value in FROZEN_SPEC['defaults'].items():
        path=Path(value);path=path if path.is_absolute() else PROJECT/path
        parser.add_argument('--'+key.replace('_','-'),type=Path,default=path)
    parser.add_argument('--device',default='cuda');parser.add_argument('--smoke-only',action='store_true')
    return parser.parse_args(argv)


if __name__=='__main__':
    construct(parse_args())
