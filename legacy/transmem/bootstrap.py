"""Download released data and record hashes; do not generate replacement test sets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

from vcrm.io import write_json


def download(urls, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(destination.suffix + '.source.json')
    if destination.exists() and sidecar.exists():
        cached = json.loads(sidecar.read_text())
        if hashlib.sha256(destination.read_bytes()).hexdigest() == cached['sha256']:
            return cached
    errors = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                data = response.read()
            if not data:
                raise ValueError('Empty download')
            destination.write_bytes(data)
            result = {'url': url, 'path': str(destination), 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest()}
            write_json(sidecar, result)
            return result
        except Exception as exc:
            errors.append(f'{url}: {type(exc).__name__}: {exc}')
    raise RuntimeError('\n'.join(errors))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/share/latent2/vcrm')
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    sources, unavailable = {}, {}
    for split in ('train', 'val', 'test'):
        sources['templama_' + split] = download(
            [f'https://storage.googleapis.com/gresearch/templama/{split}.json'],
            root / 'data' / 'templama' / f'{split}.json')
    try:
        sources['longmemeval_s_cleaned'] = download(
            [f'https://{host}/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json'
             for host in ('hf-mirror.com', 'huggingface.co')],
            root / 'data' / 'longmemeval_s_cleaned.json')
        data = json.loads((root / 'data/longmemeval_s_cleaned.json').read_text())
        if len(data) != 500:
            raise ValueError('Expected all 500 cleaned LongMemEval questions')
    except Exception as exc:
        unavailable['longmemeval_s_cleaned'] = str(exc)
    try:
        sources['locomo'] = download(
            ['https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json'],
            root / 'data/locomo10.json')
    except Exception:
        local = Path('/share/latent2/locomo/locomo10.json')
        if not local.exists():
            raise
        shutil.copy2(local, root / 'data/locomo10.json')
        sources['locomo'] = {'path': str(root / 'data/locomo10.json'),
                            'sha256': hashlib.sha256(local.read_bytes()).hexdigest(),
                            'source': 'Previously downloaded official snap-research/locomo data; local reuse'}
    for name, repo, subdir in (
        ('templama', 'google-research/language', 'language/templama'),
        ('STALE', 'icedreamc/STALE', None),
    ):
        api = f'https://api.github.com/repos/{repo}/commits/HEAD'
        with urllib.request.urlopen(api, timeout=30) as response:
            revision = json.load(response)['sha']
        archive = root / 'upstream' / f'{name}-{revision}.tar.gz'
        entry = download([f'https://codeload.github.com/{repo}/tar.gz/{revision}'], archive)
        target = root / 'upstream' / name
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                relative = Path(*Path(member.name).parts[1:])
                if subdir:
                    if not str(relative).startswith(subdir + '/'):
                        continue
                    relative = relative.relative_to(subdir)
                if member.isfile() and relative.parts and '..' not in relative.parts:
                    dest = target / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, dest.open('wb') as out:
                        shutil.copyfileobj(source, out)
        entry['revision'] = revision
        sources[name + '_source'] = entry
    unavailable['stale_official_test_data'] = (
        'Official repository publishes generation code and ontology seeds but no full released 400-scenario evaluation set. '
        'Provide author-released *_MAIN.json files; generated substitutes must be labelled separately.')
    write_json(root / 'data_manifest.json', {'sources': sources, 'unavailable': unavailable})
    print(json.dumps({'available': list(sources), 'unavailable': unavailable}, indent=2))


if __name__ == '__main__':
    main()
