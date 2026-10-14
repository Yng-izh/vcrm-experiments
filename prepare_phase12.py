"""Verify installed released artifacts before launching the v100-2 baseline phases."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer

from provenance import hash_file
from vcrm.io import read_json, write_json


def verify_tree(root, tree, paths):
    released = {item['path']: item['sha'] for item in read_json(tree)['tree'] if item['type'] == 'blob'}
    verified = {}
    for name in paths:
        raw = (Path(root) / name).read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if released.get(name) != blob:
            raise RuntimeError(f'Installed source does not match released revision: {name}')
        verified[name] = blob
    return verified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    args = parser.parse_args()
    project = Path(args.project)
    config = read_json(project / 'config.json')
    config.update(project=str(project), output_root=str(project / 'runs'))
    config['transmem_root'] = str(project / 'upstream/TransMem')
    config['delta_root'] = str(project / 'upstream/delta-Mem')
    config['data_root'] = str(project / 'data')
    config['event_format'] = 'timestamped_session_v2_neutral_date'
    write_json(project / 'config.json', config)
    tree_path = project / 'transmem-tree.json'
    source_names = [r['path'] for r in read_json(tree_path)['tree']
                    if r['type'] == 'blob' and (r['path'].startswith('transmem/') or
                                               r['path'].startswith('scripts/eval/'))
                    and r['path'].endswith('.py')]
    sources = verify_tree(config['transmem_root'], tree_path, source_names)
    delta_names = ['deltamem/eval/locomo_protocol.py', 'deltamem/eval/official_eval_utils.py']
    delta = verify_tree(config['delta_root'], project / 'delta-tree.json', delta_names)
    data = read_json(project / 'data_manifest.json')
    data_hashes = {}
    for key in ('locomo', 'longmemeval_s_cleaned'):
        row = data['sources'][key]
        value = hash_file(row['path'])
        if value != row['sha256']:
            raise RuntimeError(f'Downloaded dataset hash mismatch: {key}')
        data_hashes[key] = value
    models = {}
    root = Path(config['model_path'])
    for path in [*sorted(root.glob('model-*.safetensors')), Path(config['checkpoint'])]:
        metadata = path.parent / '.cache/huggingface/download' / (path.name + '.metadata')
        revision, expected = metadata.read_text().splitlines()[:2]
        actual = hash_file(path)
        if actual != expected:
            raise RuntimeError(f'Model weight hash mismatch: {path.name}')
        models[str(path)] = {'sha256': actual, 'revision': revision, 'bytes': path.stat().st_size}
    checkpoint = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    architecture = read_json(root / 'config.json')
    cfg = checkpoint['config']
    if cfg['dim'] != architecture['hidden_size'] or max(cfg['inject_layers']) >= architecture['num_hidden_layers']:
        raise RuntimeError('Released adapter is incompatible with backbone dimensions/layers')
    for key in ('transmem_root', 'delta_root'):
        sys.path.insert(0, config[key])
    script = Path(config['transmem_root']) / 'scripts/eval/eval_locomo.py'
    spec = importlib.util.spec_from_file_location('official_locomo_audit', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
    records = module.build_records(str(Path(config['data_root']) / 'locomo10.json'), tokenizer, [1, 2, 3, 4], None, 42)
    counts = {str(c): sum(r['category'] == c for r in records) for c in (1, 2, 3, 4)}
    contexts = {r['key'].rsplit(':', 1)[0]: r['context'] for r in records}
    lengths = {key: len(tokenizer.encode(text, add_special_tokens=False)) for key, text in contexts.items()}
    result = {'transmem_revision': read_json(tree_path)['sha'],
              'delta_revision': read_json(project / 'delta-tree.json')['sha'],
              'verified_transmem_blobs': sources, 'verified_delta_blobs': delta,
              'verified_data_sha256': data_hashes, 'verified_model_weights': models,
              'checkpoint_config': cfg, 'checkpoint_step': checkpoint['global_step'],
              'locomo_questions': len(records), 'locomo_category_counts': counts,
              'full_context_tokens_by_conversation': lengths,
              'decode': 'greedy', 'max_answer_tokens': 50, 'max_prompt_tokens': None,
              'dtype': config['dtype'], 'attention': config['attention'],
              'scoring': 'Unchanged released LoCoMo adapter canonicalization and token-F1',
              'paper_number_match': 'Not established: no paper reference score assumed',
              'v100_adaptation': 'Runtime FP16 dtype override only; upstream source unchanged'}
    write_json(project / 'phase12_input_audit.json', result)
    print(f'Artifacts verified: {len(sources)} TransMem source files, full model/checkpoint hashes, '
          f'{len(records)} LoCoMo questions, context tokens {min(lengths.values())}-{max(lengths.values())}', flush=True)


if __name__ == '__main__':
    main()
