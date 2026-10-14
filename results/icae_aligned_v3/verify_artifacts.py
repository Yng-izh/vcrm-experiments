"""Verify released diagnostic coverage, source identity, and artifact hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify(root):
    suite = root / 'runs/aligned_v3_suite'
    selection = read(suite / 'validation_481.json')
    assert len(selection['query_ids']) == len(set(selection['query_ids'])) == 481
    assert selection['conflict_queries'] == 262
    assert read(suite / 'status.json')['phase'] == 'complete'
    assert read(suite / 'active_tasks.json') == []
    inputs, diagnostics, configs = [], [], []
    with tarfile.open(root / 'source_snapshot.tar.gz') as archive:
        source = {m.name.removeprefix('./'): archive.extractfile(m).read()
                  for m in archive.getmembers() if m.isfile()}
    assert not any(Path(name).name == '.env' for name in source)
    for method in ('C16', 'D16'):
        train = root / ('runs/aligned_v3_' + method + '_train')
        val = root / ('runs/aligned_v3_' + method + '_val')
        metadata = read(train / 'codec_training.json')
        assert metadata['completed'] and metadata['codec_epochs'] == 5
        assert metadata['steps'] == 2140 and metadata['pairs'] == 1711
        assert metadata['anchor_dropout'] == 0
        assert metadata['memory_protocol'] == 'final_current_snapshot_v1'
        assert metadata['reader_sha256'] == sha(source['vcrm/reader.py'])
        inputs.append(read(train / 'training_input_audit.json'))
        diagnostics.append(read(train / 'train_diagnostic_manifest.json'))
        configs.append(read(suite / ('configs/' + method + '.json')))
        for run in (train, val):
            manifest = read(run / 'manifest.json')
            for name, expected in manifest['code'].items():
                assert name in source and sha(source[name]) == expected, (method, name)
        summary = read(val / 'summary.json')
        assert summary['complete'] and summary['queries'] == 481
        rows = [read(p) for p in (val / 'episodes').glob('*.json')]
        queries = [q for row in rows for q in row['queries']]
        assert len(rows) == 172 and all(row['roundtrip_equal'] for row in rows)
        assert len(queries) == 481 and {q['query_id'] for q in queries} == set(selection['query_ids'])
        assert all(q['selected_kind'] == 'code' and q['answer']['read_latent_slots'] == 148
                   and q['answer']['auxiliary_anchor_slots'] == 0 for q in queries)
        for epoch in range(1, 6):
            diagnostic = read(train / f'train_evaluation/epoch_{epoch:02d}.json')
            assert diagnostic['queries'] == len(diagnostic['rows']) == 200
            assert [r['pair_id'] for r in diagnostic['rows']] == diagnostics[-1]['pair_ids']
    assert inputs[0] == inputs[1] and configs[0] == configs[1]
    assert diagnostics[0] == diagnostics[1]
    assert inputs[0]['current_policy'] == 'final observed current'
    assert len(diagnostics[0]['pair_ids']) == len(set(diagnostics[0]['pair_ids'])) == 200
    print('Verified C16/D16: 5 epochs, identical inputs/configs, fixed 200/481 coverage, source hashes and roundtrip records.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write-manifest', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    verify(root)
    manifest_path = root / 'artifact_manifest.json'
    files = {str(p.relative_to(root)): {'sha256': sha(p.read_bytes()), 'bytes': p.stat().st_size}
             for p in sorted(root.rglob('*')) if p.is_file() and p != manifest_path
             and '__pycache__' not in p.parts}
    assert not any(p.endswith(('.safetensors', '.memory')) or Path(p).name == '.env' for p in files)
    if args.write_manifest:
        manifest_path.write_text(json.dumps({
            'files': files, 'validation_queries': 481, 'conflict_queries': 262,
            'training_diagnostic_questions_per_epoch': 200, 'epochs': 5,
            'excluded': ['model/codec weights', 'feature cache', 'latent .memory stores', 'credentials'],
            'server_run_root': '/share/latent2/vcrm_icae/runs',
            'scope': 'post_hoc_validation_and_in_sample_training_diagnostics_not_untouched_test'
        }, indent=2) + '\n')
    else:
        assert read(manifest_path)['files'] == files, 'Released artifact inventory/hash mismatch'
    print(f'Verified artifact inventory: {len(files)} files.')


if __name__ == '__main__':
    main()
