#!/usr/bin/env python3
"""Download pinned ICAE artifacts without touching old results or using a GPU."""
import argparse
from pathlib import Path
import shutil
import subprocess

from vcrm.io import digest, file_digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--data-source', default='/share/latent2/vcrm_v1002_phase12/data')
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    config = read_json(args.config)
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    root = Path(config['project'])
    root.mkdir(parents=True, exist_ok=True)
    upstream = Path(config['icae_root'])
    if not upstream.exists():
        upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/getao/icae.git', str(upstream)], check=True)
    git_revision = subprocess.run(['git', '-C', str(upstream), 'cat-file', '-t', config['icae_revision']],
                                  capture_output=True, text=True)
    if git_revision.returncode == 0:
        subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', config['icae_revision']], check=True)
    elif not (upstream / 'code/icae_v2/modeling_icae_multi_span.py').exists():
        raise RuntimeError('ICAE source exists without the pinned Git object and the required v2 file is missing')
    if file_digest(upstream / 'code/icae_v2/modeling_icae_multi_span.py') != config['icae_source_sha256']:
        raise RuntimeError('ICAE v2 source does not match the pinned official source SHA256')
    model_revision = config.get('model_revision') or api.model_info(config['model_id']).sha
    model_info = api.model_info(config['model_id'], revision=model_revision)
    checkpoint_info = api.model_info(config['checkpoint_id'], revision=config['checkpoint_revision'])
    manifest = {'icae_revision': config['icae_revision'], 'model_id': config['model_id'],
                'model_revision': model_info.sha, 'checkpoint_id': config['checkpoint_id'],
                'checkpoint_revision': checkpoint_info.sha, 'complete': False}
    config['model_revision'] = model_info.sha
    write_json(args.config, config)
    write_json(root / 'download_manifest.json', manifest)
    if args.metadata_only:
        print(manifest)
        return
    snapshot_download(config['model_id'], revision=model_info.sha, local_dir=config['model_path'],
                      allow_patterns=['*.json', '*.safetensors', '*.model', 'tokenizer*', '*.txt'], max_workers=3)
    checkpoint = hf_hub_download(config['checkpoint_id'], config['checkpoint_file'],
                                revision=config['checkpoint_revision'], local_dir=Path(config['checkpoint']).parent)
    if Path(checkpoint).resolve() != Path(config['checkpoint']).resolve():
        raise ValueError('Configured checkpoint path differs from downloaded artifact')
    source = Path(args.data_source)
    data = Path(config['data_root'])
    files = [Path('templama') / f'{s}.json' for s in ('train', 'val', 'test')]
    files += [Path('longmemeval_s_cleaned.json')]
    for relative in files:
        origin, destination = source / relative, data / relative
        if not origin.exists():
            if relative.name == 'longmemeval_s_cleaned.json':
                continue
            raise FileNotFoundError(origin)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination.read_bytes()) != digest(origin.read_bytes()):
            raise ValueError(f'Refusing to overwrite different data: {destination}')
        if origin.resolve() != destination.resolve():
            shutil.copy2(origin, destination)
    old_manifest = source.parent / 'data_manifest.json'
    if old_manifest.exists():
        shutil.copy2(old_manifest, root / 'source_data_manifest.json')
    manifest['files'] = {}
    for folder in (Path(config['model_path']), Path(config['checkpoint']).parent, data):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '.cache' not in path.parts:
                # Stream large model files; do not materialize 14GB weights for a checksum.
                import hashlib
                hasher = hashlib.sha256()
                with path.open('rb') as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                        hasher.update(block)
                manifest['files'][str(path)] = {'sha256': hasher.hexdigest(), 'bytes': path.stat().st_size}
    manifest['complete'] = True
    write_json(root / 'download_manifest.json', manifest)
    print('Pinned model, ICAE checkpoint and public source data are ready.')


if __name__ == '__main__':
    main()
