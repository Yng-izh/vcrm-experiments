"""Inventory the exact installed upstream sources and model revisions."""
import hashlib
import importlib.metadata
from pathlib import Path

from vcrm.io import read_json, write_json


def hash_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    config = read_json(Path(__file__).with_name('config.json'))
    result = {'source_files': {}, 'model_files': {}, 'versions': {}}
    for key in ('transmem_root', 'delta_root', 'longmemeval_root'):
        root = Path(config[key])
        result['source_files'][key] = {str(p.relative_to(root)): hash_file(p)
                                       for p in sorted(root.rglob('*.py')) if not p.name.startswith('._')}
    for key in ('model_path', 'checkpoint'):
        root = Path(config[key])
        paths = sorted(root.glob('*.json')) if root.is_dir() else [root]
        result['model_files'][key] = {str(p): {'bytes': p.stat().st_size, 'sha256': hash_file(p)} for p in paths}
        metadata_root = (root if root.is_dir() else root.parent) / '.cache/huggingface/download'
        result['model_files'][key]['hf_revision_metadata'] = {
            str(p.relative_to(metadata_root)): p.read_text().splitlines()[:2]
            for p in metadata_root.rglob('*.metadata')}
    for name in ('torch', 'transformers', 'numpy', 'pytest', 'openai', 'tiktoken', 'matplotlib', 'nltk'):
        result['versions'][name] = importlib.metadata.version(name)
    write_json(Path(config['project']) / 'provenance.json', result)
    print('Recorded upstream hashes, checkpoint SHA256, HF revisions and dependency versions')


if __name__ == '__main__':
    main()
