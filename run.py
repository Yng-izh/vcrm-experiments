#!/usr/bin/env python3
"""Unified ICAE versioned-memory experiment runner."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from vcrm.io import digest, file_digest, read_json, write_json

STAGES = ('download', 'preflight', 'prepare', 'parity', 'adapt', 'activate', 'readability', 'summarize', 'gate', 'status')


def arguments():
    parser = argparse.ArgumentParser(description='ICAE versioned latent-memory experiments')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--gpus', default='auto')
    parser.add_argument('--run-name', default='icae_readability')
    parser.add_argument('--split', choices=('val',), default='val')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--adaptation-run', help='Completed encoder adaptation run for the readability gate')
    parser.add_argument('--output-config', help='New config file written by activate; never overwrite the training config')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--shard', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--shards', type=int, default=1, help=argparse.SUPPRESS)
    return parser.parse_args()


def choose_gpus(value):
    output = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
    idle = []
    for line in output.splitlines():
        index, used, free, utilization = [int(part.strip()) for part in line.split(',')]
        if used < 300 and free >= 25000 and utilization == 0:
            idle.append(index)
    selected = idle[:1] if value == 'auto' else idle if value == 'all' else [int(x) for x in value.split(',')]
    if not selected or any(gpu not in idle for gpu in selected) or len(set(selected)) != len(selected):
        raise RuntimeError(f'Requested GPUs are not idle; currently idle: {idle}')
    return selected


def preflight(config):
    import importlib.metadata
    import torch
    required = ('icae_root', 'model_path', 'checkpoint')
    missing = [name for name in required if not Path(config[name]).exists()]
    versions = {}
    for package in ('torch', 'transformers', 'numpy', 'safetensors', 'huggingface-hub', 'peft', 'sentencepiece'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {'protocol': config['protocol'], 'missing_paths': missing, 'versions': versions,
            'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count(),
            'checkpoint': config['checkpoint'], 'model': config['model_id'],
            'checkpoint_revision': config['checkpoint_revision'],
            'source_revision': config['icae_revision'], 'codec_training_allowed': False,
            'training_split': 'TempLAMA official-source train after explicit entity-overlap filtering',
            'evaluation_split': 'TempLAMA filtered validation only; LongMemEval/LoCoMo never used for adaptation'}


def fingerprint(config, root, args):
    project = Path(__file__).parent
    files = list(project.glob('*.py')) + list((project / 'vcrm').glob('*.py'))
    value = {'config': config, 'code': {str(p.relative_to(project)): digest(p.read_bytes()) for p in sorted(files)},
             'limit': args.limit}
    manifest = Path(config['data_root']) / 'prepared/manifest.json'
    value['data_signature'] = digest(manifest.read_bytes()) if manifest.exists() else None
    weights = [config['checkpoint']] + ([config['adapted_checkpoint']] if config.get('adapted_checkpoint') else [])
    value['encoder_weights'] = {str(p): file_digest(p) for p in weights}
    upstream = Path(config['icae_root']) / 'code/icae_v2/modeling_icae_multi_span.py'
    value['upstream_source_sha256'] = file_digest(upstream)
    downloads = Path(config['project']) / 'download_manifest.json'
    value['download_manifest_sha256'] = file_digest(downloads) if downloads.exists() else None
    signature = digest(value)
    path = root / 'manifest.json'
    if path.exists() and read_json(path)['signature'] != signature:
        raise RuntimeError('Run fingerprint changed; choose a new run name')
    write_json(path, {'signature': signature, **value})


def worker(config, root, args):
    from vcrm.pipeline import evaluate
    if args.stage == 'parity':
        script = Path(__file__).with_name('check_icae_parity.py')
        entry = importlib.util.spec_from_file_location('icae_parity_entry', script)
        module = importlib.util.module_from_spec(entry)
        entry.loader.exec_module(module)
        sys.argv = [str(script), '--config', args.config]
        module.main()
    elif args.stage == 'adapt':
        from vcrm.training import adapt_encoder
        if args.shards != 1:
            raise ValueError('Encoder adaptation is single-GPU to preserve optimizer order')
        print(json.dumps(adapt_encoder(config, root, args.limit), indent=2), flush=True)
    elif args.stage == 'readability':
        evaluate(config, root, args.split, args.limit, args.shard, args.shards)


def launch(config, root, args):
    gpus = choose_gpus(args.gpus)
    if args.stage in ('adapt', 'parity') and len(gpus) != 1:
        raise ValueError('adapt/parity requires exactly one idle GPU')
    plan = {'stage': args.stage, 'gpus': gpus, 'shards': len(gpus)}
    plan_path = root / 'execution_plan.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError('Resume with the same stage and GPU count, or choose a new run name')
    write_json(plan_path, plan)
    processes, files, locks = [], [], []
    try:
        for gpu in gpus:
            lock = open(f'/tmp/vcrm_icae_gpu{gpu}.lock', 'w')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        for shard, gpu in enumerate(gpus):
            log = (root / 'logs' / f'{args.stage}.gpu{gpu}.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open('a')
            files.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker',
                       '--shard', str(shard), '--shards', str(len(gpus))]
            processes.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
            print(f'{args.stage}: physical GPU {gpu}; log={log}', flush=True)
        while any(process.poll() is None for process in processes):
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError(f'Worker failed; inspect {root / "logs"}')
            time.sleep(2)
        if any(process.returncode for process in processes):
            raise RuntimeError(f'Worker failed; inspect {root / "logs"}')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for handle in files + locks:
            handle.close()


def main():
    args = arguments()
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        raise ValueError('--run-name must be a single path component')
    config = read_json(args.config)
    root = Path(config['output_root']) / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(config, root, args)
    if args.stage == 'status':
        print(json.dumps({'root': str(root), 'episodes': len(list((root / 'episodes').glob('*.json'))),
                          'summary': str(root / 'summary.json'), 'gate': str(root / 'gate.json')}, indent=2))
        return
    with (root / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage == 'download':
            import bootstrap
            sys.argv = [str(Path(__file__).with_name('bootstrap.py')), '--config', args.config]
            bootstrap.main()
            return
        if args.stage == 'preflight':
            result = preflight(config)
            write_json(Path(config['project']) / 'environment.json', result)
            print(json.dumps(result, indent=2))
            if result['missing_paths']:
                raise RuntimeError(f'Missing paths: {result["missing_paths"]}')
            return
        if args.stage == 'prepare':
            from vcrm.data import prepare_icae
            print(json.dumps(prepare_icae(config), indent=2))
            return
        if args.stage == 'activate':
            if not args.adaptation_run or Path(args.adaptation_run).name != args.adaptation_run:
                raise ValueError('Provide a single-component --adaptation-run')
            adaptation = read_json(Path(config['output_root']) / args.adaptation_run / 'adaptation.json')
            if not adaptation.get('completed'):
                raise RuntimeError('Encoder adaptation is incomplete')
            if file_digest(adaptation['checkpoint']) != adaptation['checkpoint_sha256']:
                raise RuntimeError('Encoder weights changed')
            destination = Path(args.output_config) if args.output_config else Path(args.config).with_name('config.adapted.json')
            if destination.resolve() == Path(args.config).resolve() or destination.exists():
                raise RuntimeError('activate must write a new config, not overwrite an existing experiment')
            write_json(destination, config | {'adapted_checkpoint': adaptation['checkpoint']})
            print(f'Adapted configuration: {destination}')
            return
        fingerprint(config, root, args)
        if args.stage in ('parity', 'adapt', 'readability'):
            if args.stage == 'adapt':
                parity = Path(config['project']) / 'icae_parity.json'
                if not parity.exists() or not read_json(parity).get('qa_gradient_finite_nonzero'):
                    raise RuntimeError('Run ICAE implementation parity/QA-gradient check before adapting the encoder')
                check = read_json(parity)
                if (check.get('checkpoint_sha256') != file_digest(config['checkpoint']) or
                        check.get('source_sha256') != config['icae_source_sha256'] or
                        check.get('model_revision') != config['model_revision'] or
                        check.get('reader_sha256') != file_digest(Path(__file__).parent / 'vcrm/reader.py')):
                    raise RuntimeError('ICAE parity identity differs from current reader/source/checkpoint')
            launch(config, root, args)
        elif args.stage == 'summarize':
            from vcrm.pipeline import summarize
            print(json.dumps(summarize(config, root, args.limit), indent=2))
        elif args.stage == 'gate':
            from vcrm.pipeline import gate_decision
            summary = read_json(root / 'summary.json')
            if not args.adaptation_run or Path(args.adaptation_run).name != args.adaptation_run:
                raise ValueError('Provide --adaptation-run with a completed training run name')
            adaptation = read_json(Path(config['output_root']) / args.adaptation_run / 'adaptation.json')
            if config.get('adapted_checkpoint') != adaptation['checkpoint']:
                raise RuntimeError('Validation must actually use the train-adapted encoder')
            if file_digest(adaptation['checkpoint']) != adaptation['checkpoint_sha256']:
                raise RuntimeError('Adapted weights changed since training completion')
            if adaptation['base_checkpoint_sha256'] != file_digest(config['checkpoint']):
                raise RuntimeError('Adaptation and validation use different released ICAE checkpoints')
            if adaptation['model_revision'] != config['model_revision']:
                raise RuntimeError('Adaptation and validation use different frozen backbone revisions')
            train_episodes = read_json(Path(config['data_root']) / 'prepared/templama_train.json')
            if adaptation['training_data_sha256'] != digest(train_episodes):
                raise RuntimeError('Adaptation was not trained on the current complete training split')
            decision = gate_decision(config, summary, adaptation)
            write_json(root / 'gate.json', decision)
            print(json.dumps(decision, indent=2))
            if decision['decision'] != 'eligible_for_manual_review':
                raise RuntimeError('ICAE gate failed; do not define/train VCRM codec')


if __name__ == '__main__':
    main()
