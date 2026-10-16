#!/usr/bin/env python3
"""Prepare or run the four-method fixed TempLAMA validation comparison."""
import argparse
import fcntl
from pathlib import Path
import signal

from run import choose_gpus
from run_paper import Suite
from vcrm.comparison import METHODS, prepare_selection, report, summarize
from vcrm.io import read_json, write_json


def prepare(spec, config, root):
    selection = root / 'selection_653.json'
    prepare_selection(config, spec['historical_manifest'], selection)
    suite = Suite(spec, root, config)
    configs = {}
    for method in METHODS:
        changes = {'comparison_selection': str(selection), 'comparison_method': method,
                   'feature_root': spec['feature_root'], 'use_feature_cache': True,
                   'memory_protocol': 'final_current_snapshot_v1',
                   'max_answer_tokens': 64, 'allow_training': False,
                   'adapted_checkpoint': None, 'full_marker_run': spec['full_marker_run']}
        if method in ('C', 'D'):
            changes['comparison_training_run'] = spec['training_runs'][method]
        configs[method] = suite.derived(method, **changes)
    return suite, configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(Path(__file__).with_name('config.comparison.json')))
    parser.add_argument('--phase', choices=('prepare', 'evaluate', 'report', 'all', 'status'), default='prepare')
    parser.add_argument('--gpus', default='all')
    parser.add_argument('--methods', default=','.join(METHODS), help='Comma-separated methods: text,full,C,D')
    args = parser.parse_args()
    methods = args.methods.split(',')
    if not methods or len(methods) != len(set(methods)) or any(m not in METHODS for m in methods):
        raise ValueError('Select distinct methods from text,full,C,D')
    plan_path = Path(args.plan).resolve()
    spec = read_json(plan_path)
    if Path(spec['run_prefix']).name != spec['run_prefix'] or spec['run_prefix'] in ('.', '..'):
        raise ValueError('Use a single-component run prefix')
    config = read_json(plan_path.parent / spec['base_config'])
    root = Path(config['output_root']) / (spec['run_prefix'] + '_suite')
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == 'status':
        print(read_json(root / 'status.json'))
        return
    with (root / '.suite.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = root / 'plan.json'
        if snapshot.exists() and read_json(snapshot) != spec:
            raise ValueError('Comparison plan changed; choose a new run prefix')
        write_json(snapshot, spec)
        suite, configs = prepare(spec, config, root)
        run_roots = {m: Path(config['output_root']) / (spec['run_prefix'] + '_' + m) for m in METHODS}
        if args.phase == 'prepare':
            suite.status('prepared_not_started', methods=list(METHODS), historical=481,
                         conflict=262, latest=172, episodes=172,
                         full_marker_source=spec['full_marker_run'])
            return
        def interrupt(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupt)
        try:
            if args.phase in ('evaluate', 'all'):
                gpus = choose_gpus(args.gpus)
                suite.status('comparison_running', methods=methods, gpus=gpus)
                # One method per GPU, or sequential batches when fewer GPUs are idle.
                for offset in range(0, len(methods), len(gpus)):
                    tasks = [(spec['run_prefix'] + '_' + m,
                              suite.command(configs[m], 'comparison-eval', spec['run_prefix'] + '_' + m, [gpu]))
                             for m, gpu in zip(methods[offset:offset + len(gpus)], gpus)]
                    suite.group(tasks)
                for m in methods:
                    summary = summarize(read_json(configs[m]), run_roots[m])
                    if not summary['complete']:
                        raise ValueError(f'Incomplete results for {m}')
                if args.phase == 'evaluate':
                    suite.status('selected_methods_complete', methods=methods,
                                 next_action='Run --phase report after all four methods are complete')
                    return
            report(config, root, run_roots)
            suite.status('complete', table=str(root / 'TABLE.md'))
        except KeyboardInterrupt:
            suite.status('paused', note='Only this suite process groups stopped; completed Episodes retained')
        except Exception as error:
            suite.status('failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
