#!/usr/bin/env python3
"""Persistent multi-GPU paper workflow; never bypass failed reader gates."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run import choose_gpus
from vcrm.io import file_digest, read_json, write_json


def baseline_allocation(gpus):
    if len(gpus) < 2:
        return gpus, []
    count = max(1, (len(gpus) + 1) // 2)
    return gpus[:count], gpus[count:]


class Suite:
    def __init__(self, specification, root, config):
        self.spec, self.root, self.config = specification, root, config
        self.children = []
        self.prefix = specification['run_prefix']

    def status(self, phase, **details):
        write_json(self.root / 'status.json', {'phase': phase, 'pid': os.getpid(),
                                             'updated_unix': time.time(), **details})
        print(phase, details, flush=True)

    def derived(self, name, **changes):
        path = self.root / 'configs' / (name + '.json')
        value = self.config | changes
        if path.exists() and read_json(path) != value:
            raise RuntimeError('Frozen experiment configuration changed; use a new run_prefix')
        write_json(path, value)
        return path

    def command(self, config_path, stage, name, gpus=None, split='val', **kwargs):
        args = [sys.executable, str(Path(__file__).with_name('run.py')), '--config', str(config_path),
                '--stage', stage, '--run-name', name, '--split', split]
        if gpus is not None:
            args += ['--gpus', ','.join(map(str, gpus))]
        for key, value in kwargs.items():
            args += ['--' + key.replace('_', '-'), str(value)]
        return args

    def group(self, tasks):
        handles, processes = [], []
        try:
            for name, command in tasks:
                path = self.root / 'logs' / (name + '.log')
                path.parent.mkdir(parents=True, exist_ok=True)
                handle = path.open('a')
                handles.append(handle)
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                processes.append((name, process))
                self.children.append(process)
            write_json(self.root / 'active_tasks.json', [{'name': n, 'pid': p.pid} for n, p in processes])
            while any(p.poll() is None for _, p in processes):
                failed = [(n, p.returncode) for n, p in processes if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f'Suite workers failed: {failed}; inspect {self.root / "logs"}')
                time.sleep(5)
            failed = [(n, p.returncode) for n, p in processes if p.returncode]
            if failed:
                raise RuntimeError(f'Suite workers failed: {failed}')
        finally:
            for _, process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for _, process in processes:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                self.children.remove(process)
            for handle in handles:
                handle.close()
            write_json(self.root / 'active_tasks.json', [])

    def cpu(self, path, stage, name, split='val', **kwargs):
        self.group([(name + '_' + stage, self.command(path, stage, name, split=split, **kwargs))])

    def baseline(self, gpus):
        path = self.derived('official', feature_root=self.spec['feature_root'], save_features=True)
        baseline_gpus, feature_gpus = baseline_allocation(gpus)
        name = self.prefix + '_baseline_val'
        self.status('baseline_running', gpus=gpus, baseline_gpus=baseline_gpus, feature_gpus=feature_gpus)
        tasks = [(name, self.command(path, 'readability', name, baseline_gpus))]
        if feature_gpus:
            feature_name = self.prefix + '_features_train'
            tasks.append((feature_name, self.command(path, 'features', feature_name, feature_gpus, split='train')))
        self.group(tasks)
        if not feature_gpus:
            self.group([(self.prefix + '_features_train', self.command(path, 'features',
                        self.prefix + '_features_train', gpus, split='train'))])
        for split in ('train', 'val'):
            self.cpu(path, 'features-finalize', self.prefix + '_features_' + split, split)
        self.cpu(path, 'summarize', name)
        # gate intentionally returns nonzero on failed readability.
        try:
            self.cpu(path, 'gate', name)
        except RuntimeError:
            self.status('reader_gate_failed', gate=str(Path(self.config['output_root']) / name / 'gate.json'),
                        next_action='Inspect paired text/no-memory errors; no C/D or LongMemEval transfer launched.')
            return False
        self.status('awaiting_error_review', gate_run=name,
                    next_action='Review complete validation errors and provide --review-file; thresholds are unchanged.')
        return True

    def codec(self, gpus, review_path, override_gate=False, authorization_note=None):
        if not review_path and not override_gate:
            self.status('awaiting_error_review', next_action='A passed reader gate and explicit review file are required.')
            return False
        gate_name = self.prefix + '_baseline_val'
        gate_root = Path(self.config['output_root']) / gate_name
        authorization = {'codec_gate_run': gate_name}
        if override_gate:
            if not authorization_note:
                raise ValueError('Gate override requires an explicit --authorization-note')
            gate = read_json(gate_root / 'gate.json')
            override = {'explicit_user_authorization': True, 'authorization_note': authorization_note,
                        'gate_run': gate_name, 'summary_sha256': file_digest(gate_root / 'summary.json'),
                        'gate_sha256': file_digest(gate_root / 'gate.json'),
                        'original_decision': gate['decision'],
                        'scope': 'Override only empirical readability thresholds/manual-review prerequisite; '
                                 'preserve complete validation, reader identity, feature and GPU gradient checks.'}
            path = self.root / 'readability_override.json'
            if path.exists() and read_json(path) != override:
                raise ValueError('Do not overwrite a different gate authorization')
            write_json(path, override)
            authorization['codec_gate_override'] = str(path)
        else:
            review = read_json(review_path)
            if (review.get('gate_run') != gate_name or not review.get('approved') or not review.get('review_notes') or
                    review.get('summary_sha256') != file_digest(gate_root / 'summary.json')):
                raise ValueError('Review must name the completed validation summary, its hash, and concrete review notes')
            authorization['codec_manual_review_approved'] = True
        from vcrm.training import validate_codec_gate
        common = self.config | authorization
        validate_codec_gate(common)
        configs = {}
        for slots in self.spec['history_slots']:
            for method in ('C', 'D'):
                label = method + str(slots)
                configs[label] = self.derived(label, **{
                    'feature_root': self.spec['feature_root'], 'use_feature_cache': True,
                    'allow_training': True, **authorization,
                    'history_slots': slots, 'codec_epochs': self.spec['codec_epochs'], 'seed': self.spec['seed']})
        self.status('codec_training', methods=list(configs), gpus=gpus)
        items = list(configs.items())
        for offset in range(0, len(items), len(gpus)):
            tasks = []
            for (label, path), gpu in zip(items[offset:offset + len(gpus)], gpus):
                name = self.prefix + '_' + label + '_train'
                if not (Path(self.config['output_root']) / name / 'codec_training.json').exists():
                    tasks.append((name, self.command(path, 'codec-train', name, [gpu], method=label[0])))
            if tasks:
                self.group(tasks)
        for split in ('val', 'test'):
            feature_name = self.prefix + '_features_' + split
            feature_path = self.derived('official', feature_root=self.spec['feature_root'], save_features=True)
            if split == 'test':
                self.group([(feature_name, self.command(feature_path, 'features', feature_name, gpus, split))])
                self.cpu(feature_path, 'features-finalize', feature_name, split)
                baseline_name = self.prefix + '_baseline_test'
                cached_path = self.derived('official_test', feature_root=self.spec['feature_root'],
                                           use_feature_cache=True, save_features=False)
                self.group([(baseline_name, self.command(cached_path, 'readability', baseline_name, gpus, split))])
                self.cpu(cached_path, 'summarize', baseline_name, split)
            self.status('codec_evaluation', split=split, gpus=gpus)
            for offset in range(0, len(items), len(gpus)):
                tasks = []
                for (label, path), gpu in zip(items[offset:offset + len(gpus)], gpus):
                    name = self.prefix + '_' + label + '_' + split
                    tasks.append((name, self.command(path, 'codec-eval', name, [gpu], split,
                                  training_run=self.prefix + '_' + label + '_train', method=label[0])))
                self.group(tasks)
            for label, path in items:
                self.cpu(path, 'codec-summary', self.prefix + '_' + label + '_' + split, split,
                         training_run=self.prefix + '_' + label + '_train', method=label[0])
        self.report()
        self.status('transfer_blocked', next_action='LongMemEval needs a shared query-independent state-update resolver '
                    'and held-out query/event calibration. Independent FIFO spans cannot test version-conditioned compression.',
                    trained_checkpoints=[self.prefix + '_' + label + '_train' for label in configs])
        return True

    def report(self):
        from vcrm.paper_report import report
        report(self.config, self.spec, self.root)

    def refine(self, gpus):
        from vcrm.refinement import freeze_validation, refinement_report
        from vcrm.training import validate_codec_gate
        if len(gpus) < 2:
            raise RuntimeError('C16/D16 parallel refinement requires two idle GPUs')
        selection = freeze_validation(self.config, self.root)
        baseline = self.spec['baseline_prefix']
        changes = {'feature_root': self.spec['feature_root'], 'use_feature_cache': True,
                   'allow_training': True, 'history_slots': 16, 'codec_epochs': 5,
                   'codec_learning_rate': 1e-4, 'codec_lr_schedule': 'warmup_cosine',
                   'codec_warmup_fraction': .05, 'contrastive_weight': 1., 'contrastive_margin': 1.5,
                   'decoder_anchor_dropout': .25, 'train_diagnostic_questions': 200,
                   'save_epoch_checkpoints': True, 'seed': self.spec['seed'],
                   'codec_gate_run': baseline + '_baseline_val',
                   'codec_gate_override': str(Path(self.config['output_root']) / (baseline + '_suite') / 'readability_override.json'),
                   'evaluation_query_manifest': str(selection)}
        protocol = self.spec.get('memory_protocol')
        if protocol is not None:
            if protocol != 'final_current_snapshot_v1':
                raise ValueError('Unknown refinement memory protocol')
            changes.update(memory_protocol=protocol, decoder_anchor_dropout=0.)
        if self.spec.get('previous_diagnostic_run'):
            changes['previous_train_diagnostic_run'] = self.spec['previous_diagnostic_run']
        if self.spec.get('distillation_context'):
            changes['distillation_context'] = self.spec['distillation_context']
        validate_codec_gate(self.config | changes)
        configs = {m: self.derived(m + '16', **changes) for m in ('C', 'D')}
        mapping = dict(zip(configs, gpus[:2]))
        self.status('refinement_training', gpus=mapping, epochs=5, slots=16, train_diagnostic=200,
                    validation_diagnostic=481, next_action='Only C16/D16; C32/D32 and full test remain paused.')
        tasks = []
        for method, path in configs.items():
            name = self.prefix + '_' + method + '16_train'
            if not (Path(self.config['output_root']) / name / 'codec_training.json').exists():
                tasks.append((name, self.command(path, 'codec-train', name, [mapping[method]], method=method)))
        if tasks:
            self.group(tasks)
        self.status('refinement_validation_481', gpus=mapping)
        self.group([(self.prefix + '_' + m + '16_val',
                     self.command(path, 'codec-eval', self.prefix + '_' + m + '16_val', [mapping[m]],
                                  training_run=self.prefix + '_' + m + '16_train', method=m))
                    for m, path in configs.items()])
        for method, path in configs.items():
            self.cpu(path, 'codec-summary', self.prefix + '_' + method + '16_val',
                     training_run=self.prefix + '_' + method + '16_train', method=method)
        refinement_report(self.config, self.root, self.prefix, protocol)
        self.status('complete', report=str(self.root / 'comparison.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default=str(Path(__file__).with_name('config.experiment.json')))
    parser.add_argument('--phase', choices=('all', 'baseline', 'codec', 'refine', 'report', 'status'), default='all')
    parser.add_argument('--gpus', default='all')
    parser.add_argument('--review-file')
    parser.add_argument('--override-readability-gate', action='store_true')
    parser.add_argument('--authorization-note')
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    spec = read_json(plan_path)
    if Path(spec['run_prefix']).name != spec['run_prefix']:
        raise ValueError('run_prefix must be a single path component')
    config = read_json(plan_path.parent / spec['base_config'])
    root = Path(config['output_root']) / (spec['run_prefix'] + '_suite')
    root.mkdir(parents=True, exist_ok=True)
    if args.phase == 'status':
        import json
        status = read_json(root / 'status.json')
        baseline = Path(config['output_root']) / (spec['run_prefix'] + '_baseline_val')
        complete = [read_json(p) for p in (baseline / 'episodes').glob('*.json')]
        done = {r['id'] for r in complete}
        partial = [read_json(p) for p in (baseline / 'partial_episodes').glob('*.json')]
        flat = [q for row in complete + [r for r in partial if r['id'] not in done] for q in row['queries']]
        status['baseline_progress'] = {'completed_episodes': len(complete), 'scored_queries': len(flat),
            'partial_not_gate': True,
            'accuracy': {method: sum(q['scores'][method]['accuracy'] for q in flat) / len(flat) if flat else None
                         for method in ('latent', 'text', 'no_memory')}}
        status['train_feature_tensors'] = len(list((Path(spec['feature_root']) / 'train/events').glob('*.safetensors')))
        print(json.dumps(status, indent=2))
        return
    with (root / '.suite.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = root / 'plan.json'
        if snapshot.exists() and read_json(snapshot) != spec:
            raise RuntimeError('Experiment plan changed; choose a new run_prefix')
        write_json(snapshot, spec)
        suite = Suite(spec, root, config)
        def interrupt(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupt)
        try:
            if args.phase == 'report':
                suite.report()
                return
            gpus = choose_gpus(args.gpus)
            if args.phase == 'refine':
                suite.refine(gpus)
                return
            if args.phase in ('baseline', 'all'):
                if not suite.baseline(gpus):
                    suite.report()
                    return
            if args.phase in ('codec', 'all'):
                suite.codec(choose_gpus(args.gpus), args.review_file,
                            args.override_readability_gate, args.authorization_note)
        except KeyboardInterrupt:
            suite.status('paused', next_action='Only this suite process groups were stopped; completed outputs retained.')
        except Exception as error:
            suite.status('failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
