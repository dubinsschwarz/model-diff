#!/usr/bin/env python3
"""Freeze either new generic corpus for Attempt 014; no model weights loaded."""
import argparse
import importlib.util
import json
from importlib.metadata import version
from pathlib import Path

SOURCE=Path(__file__).with_name('construct_multicorpus_raw_jg_consensus.py')
_module=importlib.util.spec_from_file_location('consensus014',SOURCE)
a=importlib.util.module_from_spec(_module);_module.loader.exec_module(a)


def select_tokens(dataset,tokenizer,spec,torch_module):
    selection=spec['selection'];rows=[]
    shuffled=dataset.shuffle(seed=selection['shuffle_seed'],buffer_size=spec['dataset']['shuffle_buffer_size'])
    for example in shuffled:
        if not isinstance(example,dict) or not isinstance(example.get(selection['text_column']),str):
            raise ValueError('Expected string text field; refusing schema adaptation')
        text=example[selection['text_column']]
        if not text.strip():continue
        ids=tokenizer.encode(text[:selection['character_limit']],add_special_tokens=selection['add_special_tokens'])
        if not isinstance(ids,list) or any(type(i) is not int or i<0 for i in ids):raise ValueError('Invalid token IDs')
        if len(ids)<selection['minimum_token_count']:continue
        rows.append(ids[:selection['sequence_length']])
        if len(rows)==selection['sample_count']:break
    if len(rows)!=selection['sample_count']:raise ValueError('Insufficient valid rows')
    return torch_module.tensor(rows,dtype=torch_module.int64,device='cpu').contiguous()


def freeze(name,model_dir,spec_path,manifest_path):
    import torch
    spec=a.load_json_object(spec_path,'corpus spec')
    if spec!=a.NEW_CORPUS_SPECS[name]:raise ValueError('Corpus spec differs from frozen definition')
    artifact_path=Path(spec['artifact']['path'])
    a.require_output_absent(artifact_path,manifest_path)
    for out in (artifact_path,manifest_path):
        if out.resolve().is_relative_to(model_dir.resolve()) or out.resolve() in {spec_path.resolve(),SOURCE.resolve(),Path(__file__).resolve()}:
            raise ValueError('Output overlaps frozen input')
    hashes={p:a.sha256_file(p) for p in (spec_path,SOURCE,Path(__file__).resolve())}
    if a.load_json_object(spec_path,'corpus spec')!=spec:raise ValueError('Spec changed during load')
    inventory=a.tokenizer_file_records(model_dir)
    pinned=[record for record in a.FROZEN_SPEC['canonical_checkpoint_files'] if record['path'] in a.TOKENIZER_FILE_NAMES]
    if inventory!=pinned:raise ValueError('Canonical tokenizer inventory mismatch')
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    ds=spec['dataset']
    dataset=load_dataset(ds['repo_id'],name=ds['config'],revision=ds['revision'],split=ds['split'],streaming=True)
    tokens=select_tokens(dataset,tokenizer,spec,torch)
    if a.tokenizer_file_records(model_dir)!=inventory:raise ValueError('Tokenizer changed')
    for p,h in hashes.items():
        if a.sha256_file(p)!=h:raise ValueError('Frozen input changed')
    artifact_path.parent.mkdir(parents=True,exist_ok=True)
    with artifact_path.open('xb') as stream:torch.save(tokens,stream)
    result={'format_version':1,'attempt_id':a.FROZEN_SPEC['attempt_id'],'hash_algorithm':'sha256',
        'corpus_name':name,'spec_sha256':hashes[spec_path],'freeze_script_sha256':hashes[Path(__file__).resolve()],
        'helper_script_sha256':hashes[SOURCE],'dataset':ds,'selection':spec['selection'],
        'software_versions':{key:version(key) for key in ('datasets','transformers','torch')},
        'tokenizer_checkpoint':{'source':'canonical_merged_checkpoint','directory':str(model_dir),'files':inventory},
        'tensor':{'shape':list(tokens.shape),'dtype':'torch.int64','device':'cpu','contiguous':True},
        'artifact':{'path':str(artifact_path),'serialized_sha256':a.sha256_file(artifact_path),
                    'raw_tensor_sha256':a.sha256_raw_int64_tensor(tokens,torch)}}
    if a.tokenizer_file_records(model_dir)!=inventory:raise ValueError('Tokenizer changed during publication')
    for p,h in hashes.items():
        if a.sha256_file(p)!=h:raise ValueError('Frozen input changed during publication')
    a.write_manifest(manifest_path,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus',choices=list(a.NEW_CORPUS_SPECS))
    parser.add_argument('--model-dir',type=Path,default=Path(a.FROZEN_SPEC['defaults']['merged_model_directory']))
    args=parser.parse_args();folder=a.PROJECT/'experiments/attempts'/a.FROZEN_SPEC['attempt_id']
    freeze(args.corpus,args.model_dir,folder/f'{args.corpus}_corpus_spec.json',folder/f'{args.corpus}-corpus-manifest.json')


if __name__=='__main__':main()
