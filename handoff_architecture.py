#!/usr/bin/env python3
"""Replace a frozen orchestrator after its existing train workers finish."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from vcrm.io import read_json, write_json


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    if not path.exists():
        return False
    return path.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--gpus', required=True)
    args = parser.parse_args()
    plan = Path(args.plan).resolve()
    spec = read_json(plan)
    cfg = read_json(plan.parent / spec['base_config'])
    root = Path(cfg['output_root']) / (spec['run_prefix'] + '_suite')
    state = read_json(root / 'status.json')
    tasks = read_json(root / 'active_tasks.json')
    pid = state['pid']
    if state['phase'] != 'architecture_training' or not tasks:
        raise ValueError('Handoff expects an active training batch')
    if Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] not in ('T','t'):
        raise ValueError('Old orchestrator must be frozen, not its workers')
    write_json(root/'handoff.json', {'old_pid':pid, 'tasks':tasks,'status':'waiting_for_current_training'})
    print('Waiting for existing workers', tasks, flush=True)
    while any(alive(t['pid']) for t in tasks):
        time.sleep(10)
    # A failed batch must never be mistaken for successful completed training.
    completed = all((Path(cfg['output_root'])/t['name']/'codec_training.json').exists() and
                    read_json(Path(cfg['output_root'])/t['name']/'codec_training.json').get('completed') for t in tasks)
    os.kill(pid, signal.SIGTERM)
    os.kill(pid, signal.SIGCONT)
    for _ in range(60):
        if not alive(pid):
            break
        time.sleep(1)
    else:
        raise RuntimeError('Old orchestrator did not release ownership')
    if not completed:
        write_json(root/'handoff.json', {'status':'failed','error':'Existing training batch failed'})
        raise RuntimeError('Inspect existing training failure; no automatic restart')
    write_json(root/'handoff.json', {'old_pid':pid,'status':'resuming','gpus':args.gpus})
    for phase in ('ablations','final'):
        subprocess.run([sys.executable, str(plan.parent/'run_architecture.py'), '--plan', str(plan),
                        '--phase',phase,'--gpus',args.gpus], check=True)


if __name__ == '__main__':
    main()
