"""Report measured results, including incomplete runs, without invented numbers."""
from pathlib import Path

from .io import atomic_bytes, read_json, write_json
from .metrics import paired_bootstrap


def report(config, specification, destination):
    root, prefix = Path(config['output_root']), specification['run_prefix']
    result = {'splits': {}, 'archival_transmem_accuracy': .112,
              'archival_note': 'Different reader/protocol; motivation only, not a controlled superiority comparison.'}
    lines = ['# TempLAMA Measured Results', '',
             'A/B: uncompressed independently encoded, time-indexed 128-slot reference (identical tensors, not two trials).',
             'E: same frozen reader, same selected text. C/D: current + historical co-injection; no eviction.',
             'Slots include markers for C/D; resident keyframes and auxiliary recovery anchors are reported separately.', '']
    for split in ('val', 'test'):
        rows = []
        baseline = root / (prefix + '_baseline_' + split) / 'summary.json'
        if baseline.exists():
            summary = read_json(baseline)
            for label, method, slots in [('A/B', 'latent', 128), ('E', 'text', 0),
                                         ('No memory', 'no_memory', 0), ('Wrong version', 'wrong_version', 128)]:
                value = summary['methods'][method]
                rows.append({'method': label, 'history_slots': slots, 'complete': summary['complete'],
                             'current_accuracy': value['current_accuracy'], 'historical_f1': value['historical_f1'],
                             'read_seconds_mean': value['read_seconds_mean']})
        paired = {}
        for slots in specification['history_slots']:
            for method in ('C', 'D'):
                path = root / f'{prefix}_{method}{slots}_{split}' / 'summary.json'
                if path.exists():
                    rows.append({'method': method + str(slots), **read_json(path)})
            c_root, d_root = [root / f'{prefix}_{method}{slots}_{split}' for method in ('C', 'D')]
            if (c_root / 'summary.json').exists() and (d_root / 'summary.json').exists():
                c, d = [], []
                for folder, values in ((c_root, c), (d_root, d)):
                    for file in sorted((folder / 'episodes').glob('*.json')):
                        episode = read_json(file)
                        values.extend((q['query_id'], episode['group'], q['selected_entry_id'], q['score']['f1'])
                                      for q in episode['queries'] if not q['latest'] and episode['versions'] > 1)
                c, d = sorted(c), sorted(d)
                if [v[:3] for v in c] != [v[:3] for v in d]:
                    raise ValueError('C/D pairing or selected evidence differs')
                paired[str(slots)] = paired_bootstrap([v[3] for v in d], [v[3] for v in c], [v[1] for v in c])
        result['splits'][split] = {'methods': rows, 'D_minus_C_historical_f1': paired}
        lines += [f'## {split}', '', '| Method | Complete | Current EM | Historical F1 | Mean Read Seconds |',
                  '| --- | --- | --- | --- | --- |']
        fmt = lambda v: 'NA' if v is None else f'{v:.4f}'
        for value in rows:
            lines.append(f'| {value["method"]} | {value["complete"]} | {fmt(value["current_accuracy"])} | '
                         f'{fmt(value["historical_f1"])} | {fmt(value["read_seconds_mean"])} |')
        if not rows:
            lines.append('| Not run | false | NA | NA | NA |')
        lines.append('')
    write_json(Path(destination) / 'table1.json', result)
    atomic_bytes(Path(destination) / 'table1.md', '\n'.join(lines).encode())
    test_rows = result['splits']['test']['methods']
    if any(r['method'].startswith(('C', 'D')) and r['complete'] for r in test_rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for method in ('C', 'D'):
            values = sorted((r for r in test_rows if r['method'].startswith(method) and r['complete']),
                            key=lambda r: r['history_slot_budget'])
            ax.plot([r['history_slot_budget'] for r in values], [r['historical_f1'] for r in values],
                    marker='o', label=method)
        ax.set(xlabel='Historical codec slots (markers excluded)', ylabel='Historical F1', ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(destination) / 'figure1.png', dpi=180)
        plt.close(fig)
    return result
