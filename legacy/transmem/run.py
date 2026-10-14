#!/usr/bin/env python3
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from vcrm.io import digest, read_json, write_json

STAGES = ('preflight', 'prepare', 'import-stale', 'stale-score', 'budgets', 'reproduce', 'features', 'train', 'stream', 'score', 'report', 'status')


def arguments():
    p = argparse.ArgumentParser(description='VCRM staged experiments; no risk-gate code/results are reused')
    p.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    p.add_argument('--stage', choices=STAGES, required=True)
    p.add_argument('--dataset', choices=('templama', 'longmemeval', 'locomo', 'stale'), default='templama')
    p.add_argument('--split', choices=('train', 'val', 'test'), default='test')
    p.add_argument('--methods', nargs='+', choices=('transmem_streaming', 'version_index', 'independent', 'vcrm', 'text_log'),
                   default=['transmem_streaming', 'version_index', 'text_log'])
    p.add_argument('--budgets', type=int, nargs='+')
    p.add_argument('--gpus', default='auto')
    p.add_argument('--run-name', default='baseline')
    p.add_argument('--limit', type=int, help='Debug episode/question cap; never formal publication results')
    p.add_argument('--accept-reproduction', action='store_true', help='Acknowledge manual review of upstream results/protocol differences')
    p.add_argument('--reproduction-mode', choices=('transmem', 'student'), default='transmem')
    p.add_argument('--baseline-no-eviction', action='store_true',
                   help='Stage-2 LongMemEval TransMem baseline without capacity eviction; not a budget comparison')
    p.add_argument('--stale-data')
    p.add_argument('--stale-provenance', choices=('author-release', 'generated-extension'))
    p.add_argument('--conflict-type', choices=('T1', 'T2'))
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--shard', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--shards', type=int, default=1, help=argparse.SUPPRESS)
    return p.parse_args()


def add_paths(config):
    for name in ('transmem_root', 'delta_root'):
        sys.path.insert(0, config[name])


def preflight(config):
    import importlib.metadata
    import torch
    missing = [key for key in ('model_path', 'checkpoint', 'transmem_root', 'delta_root', 'longmemeval_root')
               if not Path(config[key]).exists()]
    if missing:
        raise RuntimeError(f'Missing paths: {missing}')
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'pytest', 'nltk')}
    return {'versions': versions, 'cuda_available': torch.cuda.is_available(),
            'judge_credentials_available': bool(os.environ.get(config['judge_key_env'])),
            'data_manifest': read_json(Path(config['project']) / 'data_manifest.json')
            if (Path(config['project']) / 'data_manifest.json').exists() else 'Bootstrap pending',
            'protocol': 'session streaming; query cannot access raw history; exact serialized byte budgets',
            'old_experiments': 'deprecated; not read by this project'}


def choose_gpus(value):
    output = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout
    idle = []
    for line in output.splitlines():
        index, used, free, util = [int(x.strip()) for x in line.split(',')]
        if used < 300 and free >= 25000 and util == 0:
            idle.append(index)
    selected = idle[:1] if value == 'auto' else idle if value == 'all' else [int(x) for x in value.split(',')]
    if not selected or len(set(selected)) != len(selected) or any(g not in idle for g in selected):
        raise RuntimeError(f'No requested idle GPU; idle now: {idle}')
    return selected


def fingerprint(config, root, args):
    project = Path(__file__).parent
    paths = list(project.glob('*.py')) + list((project / 'vcrm').rglob('*.py'))
    code = {str(p.relative_to(project)): digest(p.read_bytes()) for p in sorted(paths)
            if not p.name.startswith('._')}
    value = {'config': config, 'code': code, 'limit': args.limit}
    data_manifest = Path(config['data_root']) / 'prepared/manifest.json'
    budget_plan = Path(config['project']) / 'budget_plan.json'
    value['data_signature'] = digest(data_manifest.read_bytes()) if data_manifest.exists() else None
    value['budget_signature'] = digest(budget_plan.read_bytes()) if budget_plan.exists() else None
    provenance = Path(config['project']) / 'provenance.json'
    value['upstream_provenance_signature'] = digest(provenance.read_bytes()) if provenance.exists() else None
    signature = digest(value)
    path = root / 'manifest.json'
    if path.exists() and read_json(path)['signature'] != signature:
        raise RuntimeError('Run code/config/limit changed. Use another --run-name to avoid stale-cache mixing')
    write_json(path, {'signature': signature, **value})


def reproduce(config, root, limit, shard=0, shards=1, mode='transmem'):
    """Invoke the existing upstream LoCoMo evaluator, unchanged except V100 dtype support."""
    import transmem.evaluate as evaluation
    import torch
    from vcrm.runtime import configure_runtime
    runtime = configure_runtime()
    evaluation._DTYPE['float16'] = torch.float16
    script = Path(config['transmem_root']) / 'scripts/eval/eval_locomo.py'
    spec = importlib.util.spec_from_file_location('upstream_locomo', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.parse_args
    def parse():
        args = original()
        args.dtype = config['dtype']
        return args
    module.parse_args = parse
    destination = root / 'shards' / f'locomo_{mode}.{shard:03d}.json'
    sys.argv = [str(script), '--data_file', str(Path(config['data_root']) / 'locomo10.json'),
                '--model_path', config['model_path'], '--ckpt', config['checkpoint'], '--mode', mode,
                '--dtype', 'float32', '--attn_impl', config['attention'], '--categories', '1', '2', '3', '4',
                '--max_answer_tokens', '50', '--num_shards', str(shards), '--shard_index', str(shard),
                '--output_json', str(destination)]
    if limit:
        sys.argv += ['--max_questions', str(limit)]
    write_json(root / 'shards' / f'protocol.{shard:03d}.json', {'entrypoint': str(script), 'entrypoint_sha256': digest(script.read_bytes()),
               'checkpoint': config['checkpoint'], 'dtype_runtime_override': config['dtype'],
               'v100_note': 'Upstream CLI omits FP16. Runtime Evaluator dtype map is extended, no source/model changes.',
               'categories': [1, 2, 3, 4], 'max_answer_tokens': 50, 'max_prompt_tokens': None,
               'decoding': 'upstream greedy', 'metric': 'upstream LoCoMo token F1',
               'scope': 'smoke' if limit else 'full', 'reviewed': False,
               'mode': mode, 'shard': shard, 'shards': shards, 'runtime_adaptation': runtime})
    module.main()


def validate_reproduction(payload, samples, limit=None):
    expected = {f"{s['sample_id']}:{i}": qa for s in samples for i, qa in enumerate(s['qa'])
                if int(qa['category']) in (1, 2, 3, 4)}
    if limit:
        expected = dict(list(expected.items())[:limit])
    rows = payload['records']
    keys = [r['key'] for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise RuntimeError('LoCoMo merged output has missing, duplicate or unexpected question keys')
    if payload['summary']['num_questions'] != len(expected):
        raise RuntimeError('LoCoMo summary count differs from requested original questions')
    for row in rows:
        if int(row['category']) != int(expected[row['key']]['category']):
            raise RuntimeError('LoCoMo output category differs from official data')
        if not isinstance(row['raw_prediction'], str) or not 0 <= float(row['score']) <= 1:
            raise RuntimeError('LoCoMo invalid prediction or metric')
    return expected


def merge_reproduction(config, root, args, shards):
    paths = [root / 'shards' / f'locomo_{args.reproduction_mode}.{i:03d}.json' for i in range(shards)]
    destination = root / 'locomo_upstream.json'
    script = Path(config['transmem_root']) / 'scripts/eval/merge_locomo_shards.py'
    subprocess.run([sys.executable, str(script), '--output', str(destination), *map(str, paths)], check=True)
    payload = read_json(destination)
    samples = read_json(Path(config['data_root']) / 'locomo10.json')
    expected = validate_reproduction(payload, samples, args.limit)
    from deltamem.eval.locomo_protocol import score_locomo_prediction
    for row in payload['records']:
        rescored = round(float(score_locomo_prediction(expected[row['key']], row['prediction'])), 6)
        if rescored != row['score']:
            raise RuntimeError('LoCoMo prediction differs from upstream token-F1 rescoring')
    protocol = read_json(root / 'shards/protocol.000.json')
    protocol.update(shards=shards, all_keys_validated=True, official_scores_recomputed=True)
    write_json(root / 'reproduction_protocol.json', protocol)
    if args.reproduction_mode == 'transmem':
        write_json(Path(config['project']) / 'reproduction_status.json', {
            'execution_completed': True, 'full_evaluation': not bool(args.limit), 'reviewed': False,
            'result': str(destination), 'checkpoint': config['checkpoint'],
            'protocol': str(root / 'reproduction_protocol.json'),
            'result_sha256': digest(destination.read_bytes()), 'num_questions': len(expected),
            'reproduction_identity': reproduction_identity(config)})


def reproduction_identity(config):
    project = Path(config['project'])
    return digest({'model': config['model_path'], 'checkpoint': config['checkpoint'],
                   'dtype': config['dtype'], 'attention': config['attention'],
                   'runtime': digest((Path(__file__).parent / 'vcrm/runtime.py').read_bytes()),
                   'provenance': digest((project / 'provenance.json').read_bytes()),
                   'locomo_data': read_json(project / 'data_manifest.json')['sources']['locomo']['sha256']})


def require_reproduction(config, args):
    status_path = Path(config['project']) / 'reproduction_status.json'
    status = read_json(status_path) if status_path.exists() else {}
    if args.limit:
        return
    if not status.get('execution_completed') or not status.get('full_evaluation'):
        raise RuntimeError('Complete full upstream LoCoMo reproduction before formal streaming/training')
    if status.get('reproduction_identity') != reproduction_identity(config):
        raise RuntimeError('Reproduction model/source/runtime identity differs from current setup')
    if digest(Path(status['result']).read_bytes()) != status.get('result_sha256'):
        raise RuntimeError('Reviewed reproduction output changed')
    if args.accept_reproduction:
        status['reviewed'] = True
        write_json(status_path, status)
    if not status.get('reviewed'):
        raise RuntimeError('Review upstream reproduction differences; pass --accept-reproduction after review')


def require_streaming_baseline(config, args):
    path = Path(config['project']) / 'stage2_decision.json'
    if args.limit or not path.exists():
        return
    decision = read_json(path)
    if decision.get('decision') == 'hold_formal_comparisons_and_codec_training' and (
            args.stage == 'train' or args.stage == 'stream' and args.methods != ['transmem_streaming']):
        raise RuntimeError('Stage-2 streaming baseline diagnosis failed: formal codec training/comparisons are held. '
                           'Review stage2_decision.json and establish a separately documented repaired protocol first.')


def worker(config, root, args):
    if args.stage == 'reproduce':
        reproduce(config, root, args.limit, args.shard, args.shards, args.reproduction_mode)
    elif args.stage == 'features':
        from vcrm.training import extract_features
        extract_features(config, args.split, args.limit)
    elif args.stage == 'train':
        from vcrm.training import train
        for method in args.methods:
            dest = Path(config['project']) / ('debug_checkpoints' if args.limit else 'checkpoints') / f'{method}.pt'
            train(config, method, dest, args.limit)
    elif args.stage == 'stream':
        from vcrm.pipeline import infer
        plan = read_json(Path(config['project']) / 'budget_plan.json')
        budgets = args.budgets or plan['budgets_bytes']
        if args.baseline_no_eviction:
            if args.dataset != 'longmemeval' or args.methods != ['transmem_streaming'] or args.budgets:
                raise ValueError('No-eviction baseline is only A on LongMemEval, without custom byte points')
            budgets = [2 ** 40]
        if not args.limit and not args.baseline_no_eviction and budgets != plan['budgets_bytes']:
            raise RuntimeError('Formal runs must use the frozen three-point budget plan')
        for method in args.methods:
            for budget in budgets:
                infer(config, root, args.dataset, args.split, method, budget, args.limit, args.shard, args.shards)


def launch(config, root, args):
    gpus = choose_gpus(args.gpus)
    if args.stage not in ('stream', 'reproduce') and len(gpus) != 1:
        raise ValueError('Features/train currently require exactly one GPU')
    plan_path = root / 'execution_plan.json'
    plan = {'stage': args.stage, 'gpus': gpus, 'shards': len(gpus),
            'reproduction_mode': args.reproduction_mode if args.stage == 'reproduce' else None}
    if plan_path.exists():
        old = read_json(plan_path)
        if old['stage'] != args.stage or old['shards'] != len(gpus) or old.get('reproduction_mode') != plan['reproduction_mode']:
            raise RuntimeError('Resume with the same stage/shard count/mode or use a new run name')
    write_json(plan_path, plan)
    locks, processes, handles = [], [], []
    try:
        for gpu in gpus:
            handle = open(f'/tmp/vcrm_gpu{gpu}.lock', 'w')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        for shard, gpu in enumerate(gpus):
            path = root / 'logs' / f'{args.stage}.gpu{gpu}.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('a')
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker',
                       '--shard', str(shard), '--shards', str(len(gpus))]
            processes.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env))
            print(f'{args.stage}: GPU {gpu}; log {path}', flush=True)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f'Worker failed; inspect {root / "logs"}')
            time.sleep(1)
        if any(p.returncode for p in processes):
            raise RuntimeError(f'Worker failed; inspect {root / "logs"}')
        if args.stage == 'reproduce':
            merge_reproduction(config, root, args, len(gpus))
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        for h in handles + locks:
            h.close()


def main():
    args = arguments()
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be positive')
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        raise ValueError('Run name must be a single path component')
    config = read_json(args.config)
    if args.baseline_no_eviction:
        config = config | {'no_eviction_baseline': True}
    add_paths(config)
    root = Path(config['output_root']) / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(config, root, args)
    if args.stage == 'status':
        print(json.dumps({'generated': len(list(root.glob('*/*/budget_*/episodes/*.json'))),
                          'scored': len(list(root.glob('*/*/budget_*/scores/*.json'))),
                          'locomo_completed_questions': sum(sum(bool(line.strip()) for line in p.open())
                              for p in (root / 'shards').glob('*.progress.jsonl')),
                          'root': str(root)}, indent=2))
        return
    with (root / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready = preflight(config)
        if args.stage == 'preflight':
            write_json(Path(config['project']) / 'environment.json', ready)
            print(json.dumps(ready, indent=2))
            return
        if args.stage not in ('prepare', 'budgets', 'import-stale'):
            fingerprint(config, root, args)
        if args.stage == 'prepare':
            from vcrm.data import prepare
            print(json.dumps(prepare(config), indent=2))
        elif args.stage == 'import-stale':
            from vcrm.data import import_stale
            if not args.stale_data or not args.stale_provenance:
                raise ValueError('Provide --stale-data and explicit --stale-provenance')
            print(json.dumps(import_stale(args.stale_data, config, args.stale_provenance), indent=2))
        elif args.stage == 'stale-score':
            from vcrm.stale import evaluate
            if not args.conflict_type:
                raise ValueError('Provide the source dataset --conflict-type T1/T2')
            print(json.dumps(evaluate(config, root, args.conflict_type), indent=2))
        elif args.stage == 'budgets':
            from vcrm.budgets import freeze_budgets
            print(json.dumps(freeze_budgets(config), indent=2))
        elif args.stage in ('features', 'train', 'stream', 'reproduce'):
            if args.stage != 'reproduce':
                require_reproduction(config, args)
                require_streaming_baseline(config, args)
            launch(config, root, args)
        elif args.stage == 'score':
            from vcrm.pipeline import score
            print(json.dumps(score(config, root), indent=2))
        elif args.stage == 'report':
            from vcrm.pipeline import report
            print(json.dumps(report(config, root), indent=2))


if __name__ == '__main__':
    main()
