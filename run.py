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

STAGES = ('download', 'preflight', 'prepare', 'parity', 'adapt', 'activate', 'readability', 'summarize', 'gate', 'status',
          'lme-audit', 'lme-diagnostic', 'lme-summary')
STAGES += ('unit-tests', 'training-audit', 'codec-train')
STAGES += ('features', 'features-finalize')
STAGES += ('codec-eval', 'codec-summary')
STAGES += ('comparison-eval', 'comparison-summary')
STAGES += ('read-diagnostic', 'read-diagnostic-summary')


def arguments():
    parser = argparse.ArgumentParser(description='ICAE versioned latent-memory experiments')
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--gpus', default='auto')
    parser.add_argument('--run-name', default='icae_readability')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--adaptation-run', help='Completed encoder adaptation run for the readability gate')
    parser.add_argument('--output-config', help='New config file written by activate; never overwrite the training config')
    parser.add_argument('--method', choices=('C', 'D'), default='C')
    parser.add_argument('--training-run', help='Completed C/D training run name for held-out evaluation')
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
             'limit': args.limit, 'method': args.method, 'split': args.split,
             'training_run': args.training_run}
    if args.training_run:
        if Path(args.training_run).name != args.training_run:
            raise ValueError('--training-run must be a single path component')
        training = Path(config['output_root']) / args.training_run
        value['codec_training_sha256'] = file_digest(training / 'codec_training.json')
        value['codec_checkpoint_sha256'] = file_digest(training / 'codec.safetensors')
    manifest = Path(config['data_root']) / 'prepared/manifest.json'
    value['data_signature'] = digest(manifest.read_bytes()) if manifest.exists() else None
    if config.get('evaluation_query_manifest'):
        value['evaluation_query_manifest_sha256'] = file_digest(config['evaluation_query_manifest'])
    weights = [config['checkpoint']] + ([config['adapted_checkpoint']] if config.get('adapted_checkpoint') else [])
    value['encoder_weights'] = {str(p): file_digest(p) for p in weights}
    upstream = Path(config['icae_root']) / 'code/icae_v2/modeling_icae_multi_span.py'
    value['upstream_source_sha256'] = file_digest(upstream)
    downloads = Path(config['project']) / 'download_manifest.json'
    value['download_manifest_sha256'] = file_digest(downloads) if downloads.exists() else None
    if args.stage.startswith('lme-'):
        folder = Path(config['diagnostic_root'])
        value['diagnostic_source'] = {str(p.relative_to(folder)): file_digest(p)
                                      for p in [folder / 'protocol.json', *sorted((folder / 'cases').glob('*.json'))]}
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
    elif args.stage == 'lme-diagnostic':
        from vcrm.diagnostics import evaluate as diagnose
        diagnose(config, root, args.shard, args.shards)
    elif args.stage == 'codec-train':
        from vcrm.training import train_codec
        print(json.dumps(train_codec(config, root, args.method == 'D', args.limit), indent=2), flush=True)
    elif args.stage == 'features':
        from vcrm.features import encode
        encode(config, args.split, args.shard, args.shards)
    elif args.stage == 'codec-eval':
        from vcrm.codec_eval import evaluate as evaluate_codec
        if not args.training_run:
            raise ValueError('Provide --training-run for codec evaluation')
        evaluate_codec(config, root, Path(config['output_root']) / args.training_run,
                       args.split, args.limit, args.shard, args.shards)
    elif args.stage == 'comparison-eval':
        from vcrm.comparison import evaluate as evaluate_comparison
        evaluate_comparison(config, root, args.shard, args.shards)
    elif args.stage == 'read-diagnostic':
        from vcrm.architecture import read_diagnostic
        if not args.training_run:
            raise ValueError('Read diagnostics require a completed training run')
        read_diagnostic(config, root, Path(config['output_root']) / args.training_run, args.shard, args.shards)


def launch(config, root, args):
    gpus = choose_gpus(args.gpus)
    if args.stage in ('adapt', 'parity', 'codec-train') and len(gpus) != 1:
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
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='2')
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
        if args.stage == 'unit-tests':
            subprocess.run([sys.executable, '-m', 'pytest', '-q', str(Path(__file__).parent / 'tests')],
                           cwd=Path(__file__).parent, check=True)
            return
        if args.stage == 'training-audit':
            from vcrm.training import train_only, paragraph_samples, temporal_pairs
            from vcrm.codec import SlotCodec, VersionMarkers
            episodes = train_only(config, args.limit)
            paragraphs = paragraph_samples(episodes, config['seed'])
            pairs = temporal_pairs(episodes, config['keyframe_interval'])
            c = SlotCodec(4096, config['history_slots'], config['codec_width'], config['codec_heads'], False)
            d = SlotCodec(4096, config['history_slots'], config['codec_width'], config['codec_heads'], True)
            result = {'scope': 'code/data audit, no GPU or training', 'groups': len(episodes),
                      'paragraph_samples': len(paragraphs), 'temporal_pairs': len(pairs),
                      'input_sha256': digest(paragraphs), 'pair_ids_sha256': digest(sorted(p['id'] for p in pairs)),
                      'C_parameters': c.parameter_count(), 'D_parameters': d.parameter_count(),
                      'parameter_matched': c.parameter_count() == d.parameter_count(),
                      'history_decoder_slots': config['history_slots'], 'markers_per_entry': 2,
                      'marker_parameter_bytes_fp32': sum(p.numel() * 4 for p in VersionMarkers().parameters()),
                      'training_enabled': config.get('allow_training', False)}
            write_json(root / 'training_audit.json', result)
            print(json.dumps(result, indent=2))
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
        if args.stage in ('comparison-eval', 'comparison-summary'):
            from vcrm.comparison import freeze_protocol, summarize as summarize_comparison
            if args.split != 'val' or args.limit is not None:
                raise ValueError('Unified comparison uses the complete fixed validation selection')
            freeze_protocol(config, root)
            if args.stage == 'comparison-eval':
                launch(config, root, args)
            else:
                print(json.dumps(summarize_comparison(config, root), indent=2))
            return
        fingerprint(config, root, args)
        if args.stage in ('parity', 'adapt', 'readability', 'lme-diagnostic', 'codec-train', 'features', 'codec-eval', 'read-diagnostic'):
            if args.stage in ('adapt', 'codec-train') and not config.get('allow_training'):
                raise RuntimeError('Training is disabled by default; audit and explicitly authorize a new config first')
            if args.stage != 'parity':
                parity_path = Path(config['project']) / 'icae_parity.json'
                if not parity_path.exists():
                    raise RuntimeError('Run implementation parity before any GPU pilot')
                parity_check = read_json(parity_path)
                if (not parity_check.get('chunked_prefill_greedy_match') or
                        not parity_check.get('official_encoder_exact_fp16_match') or
                        parity_check.get('reader_sha256') != file_digest(Path(__file__).parent / 'vcrm/reader.py') or
                        parity_check.get('checkpoint_sha256') != file_digest(config['checkpoint']) or
                        parity_check.get('source_sha256') != config['icae_source_sha256'] or
                        parity_check.get('model_revision') != config['model_revision']):
                    raise RuntimeError('Implementation parity is stale or incomplete')
                if args.stage == 'codec-train' and not parity_check.get('codec_qa_gradients_finite_nonzero'):
                    raise RuntimeError('C/D training requires successful frozen-reader codec QA gradient checks')
            if args.stage == 'lme-diagnostic':
                from vcrm.diagnostics import audit
                audit(config, root)
            if args.stage == 'features':
                from vcrm.features import prepare
                prepare(config, args.split)
            if args.stage == 'readability' and config.get('save_features'):
                from vcrm.features import prepare
                prepare(config, args.split)
            if args.stage == 'adapt':
                if config.get('adaptation_protocol') != 'paragraph_memory_only_v1':
                    raise RuntimeError('Paragraph memory-only adaptation is disabled. Audit and authorize training first.')
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
        elif args.stage == 'features-finalize':
            from vcrm.features import finalize
            print(json.dumps(finalize(config, args.split), indent=2))
        elif args.stage == 'read-diagnostic-summary':
            from vcrm.architecture import diagnostic_summary
            print(json.dumps(diagnostic_summary(config, root), indent=2))
        elif args.stage == 'codec-summary':
            from vcrm.codec_eval import summarize as summarize_codec
            print(json.dumps(summarize_codec(config, root, args.split, args.limit), indent=2))
        elif args.stage == 'lme-audit':
            from vcrm.diagnostics import audit
            result = audit(config, root)
            print(json.dumps({'cases': len(result['cases']), 'audit': str(root / 'diagnostic_audit.json')}, indent=2))
        elif args.stage == 'lme-summary':
            from vcrm.diagnostics import summarize as summarize_diagnostic
            print(json.dumps(summarize_diagnostic(config, root), indent=2))
        elif args.stage == 'summarize':
            from vcrm.pipeline import summarize
            print(json.dumps(summarize(config, root, args.limit, args.split), indent=2))
        elif args.stage == 'gate':
            from vcrm.pipeline import gate_decision
            summary = read_json(root / 'summary.json')
            if not config.get('adapted_checkpoint'):
                if args.adaptation_run:
                    raise ValueError('Official zero-shot gate must not claim an adaptation run')
                decision = gate_decision(config, summary)
                write_json(root / 'gate.json', decision)
                print(json.dumps(decision, indent=2))
                if decision['decision'] != 'eligible_for_manual_review':
                    raise RuntimeError('Official-checkpoint gate failed; do not train a codec')
                return
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
