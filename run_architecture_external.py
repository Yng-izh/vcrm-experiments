#!/usr/bin/env python3
"""Run one architecture arm on a second host with shared artifacts."""
import argparse
import fcntl
import os
from pathlib import Path
import socket
import subprocess
import time

from run_architecture import ArchitectureSuite, VARIANTS
from vcrm.io import digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--label', choices=['E2'], required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    plan = Path(args.plan).resolve()
    spec = read_json(plan)
    config = read_json(plan.parent / spec['base_config'])
    root = Path(config['output_root']) / (spec['run_prefix'] + '_suite')
    suite = ArchitectureSuite(spec, root, config)
    folder = root / 'external'
    folder.mkdir(exist_ok=True)
    with (folder / (args.label + '.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        definition = VARIANTS[args.label]
        cfg = suite.derived(args.label, **(suite.common | {k:v for k,v in definition.items() if k != 'method'}))
        registration = folder / (args.label + '.json')
        info = {'label': args.label, 'hostname': socket.gethostname(), 'pid': os.getpid(),
                'gpu': args.gpu, 'config_sha256': digest(read_json(cfg))}
        def status(state, **details):
            write_json(registration, info | {'status': state, 'updated_unix': time.time()} | details)
            print(state, details, flush=True)
        status('running', stage='train')
        try:
            stages = [('codec-train','train',{}),
                      ('codec-eval','val',{'training_run':suite.path(args.label,'train').name}),
                      ('codec-summary','val',{'training_run':suite.path(args.label,'train').name})]
            for stage, suffix, extra in stages:
                status('running', stage=stage)
                command = suite.command(cfg, stage, suite.path(args.label,suffix).name,
                                        None if stage == 'codec-summary' else [args.gpu],
                                        method=definition['method'], **extra)
                subprocess.run(command, check=True)
            status('complete')
        except BaseException as error:
            status('failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
