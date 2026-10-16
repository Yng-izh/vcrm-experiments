#!/usr/bin/env python3
"""Sequential, gated E1-E4 architecture studies; never a Cartesian search."""
import argparse
import fcntl
from pathlib import Path
import signal
import time

from run import choose_gpus
from run_paper import Suite
from vcrm.architecture import ablation_table, compare, evaluation_metrics, final_test_report
from vcrm.comparison import prepare_selection, report as baseline_report
from vcrm.io import digest, read_json, write_json


VARIANTS = {
    'C0': {'method': 'C', 'decode_conditioned': False},
    'D0': {'method': 'D', 'decode_conditioned': True},
    'E1': {'method': 'D', 'decode_conditioned': False},
    'E2': {'method': 'D', 'decode_conditioned': False, 'codec_variant': 'salience'},
    'E3': {'method': 'C', 'decode_conditioned': False, 'codec_value_path': 'full'},
}


def choose_c(control, full):
    rank = lambda v: (v['all_compressed']['f1'], v['disjoint_current_conflicts']['f1'],
                      -v['disjoint_current_conflicts']['stale_answer'])
    return 'E3' if rank(full) > rank(control) else 'C0'


class ArchitectureSuite(Suite):
    def __init__(self, spec, root, config):
        super().__init__(spec, root, config)
        self.selection = str(root / 'validation_481.json')
        original = read_json(spec['historical_manifest'])
        target = Path(self.selection)
        if target.exists() and read_json(target) != original:
            raise ValueError('Architecture validation selection changed')
        write_json(target, original)
        self.common = dict(feature_root=spec['feature_root'], use_feature_cache=True, allow_training=True,
            memory_protocol='final_current_snapshot_v1', history_slots=16, codec_epochs=5,
            codec_learning_rate=1e-4, codec_lr_schedule='warmup_cosine', codec_warmup_fraction=.05,
            contrastive_weight=1., contrastive_margin=1.5, decoder_anchor_dropout=0.,
            train_diagnostic_questions=200, previous_train_diagnostic_run='aligned_v3_C16_train',
            save_epoch_checkpoints=True, seed=42, distillation_context=spec['teacher_context'],
            codec_variant='residual', codec_value_path='projected', max_answer_tokens=64,
            codec_activation_rms_cap=8.,
            expected_temporal_pairs=2954,
            evaluation_query_manifest=self.selection, codec_gate_run='paper_v1_baseline_val',
            codec_gate_override=str(Path(config['output_root'])/'paper_v1_suite/readability_override.json'))
        if spec['teacher_context'] != 'single_target':
            raise ValueError('This matrix fixes the completed aligned-v3 single-target teacher protocol')

    def path(self, label, stage):
        return Path(self.config['output_root']) / f'{self.prefix}_{label}_{stage}'

    def trained(self, definitions, gpus):
        configs = {}
        labels = list(definitions)
        for label, definition in definitions.items():
            changes = {k:v for k,v in definition.items() if k != 'method'}
            configs[label] = self.derived(label, **(self.common | changes))
        for offset in range(0, len(labels), len(gpus)):
            batch = labels[offset:offset+len(gpus)]
            tasks, external = [], []
            for label, gpu in zip(batch, gpus):
                registration = self.root / 'external' / (label + '.json')
                if registration.exists():
                    expected = digest(read_json(configs[label]))
                    if read_json(registration)['config_sha256'] != expected:
                        raise ValueError('External architecture config mismatch: ' + label)
                    external.append(label)
                    continue
                output = self.path(label, 'train')
                if not (output / 'codec_training.json').exists():
                    tasks.append((output.name, self.command(configs[label], 'codec-train', output.name, [gpu],
                                                          method=definitions[label]['method'])))
            self.status('architecture_training', labels=batch, gpus=gpus[:len(batch)], epochs=5)
            if tasks:
                self.group(tasks)
            for label in external:
                self.wait_external(label, configs[label])
            self.group([(self.path(label, 'val').name,
                         self.command(configs[label], 'codec-eval', self.path(label, 'val').name, [gpu],
                                      method=definitions[label]['method'], training_run=self.path(label, 'train').name))
                        for label, gpu in zip(batch, gpus) if label not in external])
            for label in batch:
                if label in external:
                    continue
                self.cpu(configs[label], 'codec-summary', self.path(label, 'val').name,
                         method=definitions[label]['method'], training_run=self.path(label, 'train').name)
        return configs

    def wait_external(self, label, config_path):
        registration = self.root / 'external' / (label + '.json')
        deadline = time.monotonic() + 24 * 3600
        self.status('waiting_external_architecture', label=label)
        while True:
            info = read_json(registration)
            if info['config_sha256'] != digest(read_json(config_path)):
                raise ValueError('External architecture config mismatch: ' + label)
            if info['status'] == 'failed':
                raise RuntimeError('External architecture worker failed: ' + label + ': ' + info.get('error', ''))
            if info['status'] == 'complete':
                for stage, filename, key in [('train','codec_training.json','completed'),
                                             ('val','summary.json','complete')]:
                    if not read_json(self.path(label, stage) / filename).get(key):
                        raise ValueError('External worker has incomplete outputs: ' + label)
                return
            if time.monotonic() > deadline:
                raise TimeoutError('External architecture worker timed out: ' + label)
            time.sleep(15)

    def baselines(self, gpus):
        # Text/full are complete; append C/D latest QA without mixing training rounds.
        selection = self.root/'baseline_selection_653.json'
        prepare_selection(self.config, self.spec['historical_manifest'], selection)
        configs, roots = {}, {}
        for method in ('text','full'):
            roots[method] = Path(self.config['output_root'])/f'comparison_aligned_v3_{method}'
        for method in ('C','D'):
            name = f'{self.prefix}_baseline_{method}'
            roots[method] = Path(self.config['output_root'])/name
            configs[method] = self.derived('baseline_'+method, comparison_selection=str(selection),
                comparison_method=method, comparison_training_run=f'aligned_v3_{method}16_train',
                full_marker_run='aligned_v3_C16_train', feature_root=self.spec['feature_root'],
                memory_protocol='final_current_snapshot_v1', max_answer_tokens=64, allow_training=False)
        self.status('complete_original_main_baselines', gpus=gpus)
        for offset in range(0,2,len(gpus)):
            self.group([(roots[m].name,self.command(configs[m],'comparison-eval',roots[m].name,[gpu]))
                        for m,gpu in zip(list(configs)[offset:offset+len(gpus)],gpus)])
        baseline_report(self.config,self.root/'baseline_table',roots)

    def diagnostic(self, gpus):
        tasks, cfgs = [], {}
        for method in ('C','D'):
            cfgs[method] = self.derived('read_'+method, **self.common)
        for offset in range(0,2,len(gpus)):
            batch = list(cfgs)[offset:offset+len(gpus)]
            tasks=[(self.path('read_'+m,'val').name,
                    self.command(cfgs[m],'read-diagnostic',self.path('read_'+m,'val').name,[gpu],
                                 method=m,training_run=f'aligned_v3_{m}16_train'))
                   for m,gpu in zip(batch,gpus)]
            self.status('four_way_read_and_fp16_diagnostic', methods=batch)
            self.group(tasks)
            for m in batch:
                self.cpu(cfgs[m],'read-diagnostic-summary',self.path('read_'+m,'val').name,
                         method=m,training_run=f'aligned_v3_{m}16_train')

    def ablations(self, gpus):
        # Fresh controls are mandatory: old aligned-v3 had incomplete historical target coverage.
        self.trained({label:VARIANTS[label] for label in ('D0','E1','C0')},gpus)
        e1 = compare(self.config|self.common, self.root, 'E1', self.path('D0','val'), self.path('E1','val'))
        next_definitions = {'E3':VARIANTS['E3']}
        external_e2 = (self.root / 'external' / 'E2.json').exists()
        if e1['point_estimate_support'] or external_e2:
            next_definitions = {'E2':VARIANTS['E2']} | next_definitions
        else:
            e2 = {'skipped': True, 'reason':'E1 does not improve historical F1 without lowering conflict F1; '
                  'stop this decoder-removal branch instead of stacking more changes.'}
            write_json(self.root/'E2.json',e2)
        self.trained(next_definitions,gpus)
        if e1['point_estimate_support'] or external_e2:
            e2 = compare(self.config|self.common,self.root,'E2',self.path('E1','val'),self.path('E2','val'))
            e2['started_in_parallel_by_user_request'] = external_e2
            write_json(self.root/'E2.json',e2)
        e3 = compare(self.config|self.common,self.root,'E3',self.path('C0','val'),self.path('E3','val'))
        c_best = choose_c(evaluation_metrics(self.config|self.common,self.path('C0','val')),
                          evaluation_metrics(self.config|self.common,self.path('E3','val')))
        low = VARIANTS[c_best] | {'contrastive_weight': .2}
        self.trained({'E4':low},gpus[:1])
        e4 = compare(self.config|self.common,self.root,'E4',self.path(c_best,'val'),self.path('E4','val'))
        final_value = 'full' if e3['point_estimate_support'] else 'projected'
        low_negative_applicable = (e4['point_estimate_support'] and
                                   VARIANTS[c_best].get('codec_value_path','projected') == final_value)
        decision = {'C_best':c_best, 'E1_supported':e1['point_estimate_support'],
                    'E2_supported':e2.get('point_estimate_support',False),
                    'E3_supported':e3['point_estimate_support'], 'E4_supported':e4['point_estimate_support'],
                    'C_best_rule':'Historical F1, then conflict F1, then lower conflict stale rate; exact tie chooses C0',
                    'final_codec_variant':'salience' if e1['point_estimate_support'] and e2.get('point_estimate_support') else 'residual',
                    'final_value_path':final_value,
                    'E4_applicable_to_selected_value_path':low_negative_applicable,
                    'final_contrastive_weight':.2 if low_negative_applicable else 1.,
                    'caveat':'Selection uses point estimates on tuned validation; all paired CIs are saved. '
                             'No causal pollution/information-loss claim, no held-out test claim.'}
        write_json(self.root/'selection_decision.json',decision)
        ablation_table(self.root)
        return decision

    def final_pair(self, gpus):
        decision=read_json(self.root/'selection_decision.json')
        if not decision['E1_supported']:
            self.status('no_supported_conditioned_branch',decision=decision,
                        next_action='Review diagnostics and E1-E4; do not force a final combined architecture.')
            return False
        common = {'codec_variant':decision['final_codec_variant'],
                  'codec_value_path':decision['final_value_path'],
                  'contrastive_weight':decision['final_contrastive_weight'], 'decode_conditioned':False}
        self.trained({'CFinal':common|{'method':'C'},'DFinal':common|{'method':'D'}},gpus)
        compare(self.config|self.common,self.root,'Final',self.path('CFinal','val'),self.path('DFinal','val'))
        self.final_main_table(gpus)
        self.status('final_validation_complete',next_action='Frozen final checkpoints available. Test is a separate '
                    'explicit phase; validation selection must not be changed based on test.')
        return True

    def final_main_table(self,gpus):
        selection=self.root/'baseline_selection_653.json'
        prepare_selection(self.config,self.spec['historical_manifest'],selection)
        roots={m:Path(self.config['output_root'])/f'comparison_aligned_v3_{m}' for m in ('text','full')}
        configs={}
        for m,label in [('C','CFinal'),('D','DFinal')]:
            roots[m]=self.path(label,'main653')
            configs[m]=self.derived(label+'_main653',**(read_json(self.root/'configs'/f'{label}.json')|{
                'comparison_selection':str(selection),'comparison_method':m,
                'comparison_training_run':self.path(label,'train').name,
                'full_marker_run':'aligned_v3_C16_train','allow_training':False}))
        for offset in range(0,2,len(gpus)):
            self.group([(roots[m].name,self.command(configs[m],'comparison-eval',roots[m].name,[gpu]))
                        for m,gpu in zip(list(configs)[offset:offset+len(gpus)],gpus)])
        baseline_report(self.config,self.root/'final_main_table',roots)

    def test_pair(self,gpus):
        if not (self.root/'Final.json').exists():
            raise ValueError('Freeze the final C/D validation comparison before held-out test')
        configs={m:self.derived(m+'_test',**(read_json(self.root/'configs'/f'{m}.json')|
                                          {'evaluation_query_manifest':None})) for m in ('CFinal','DFinal')}
        from vcrm.features import check
        try:
            check(read_json(configs['CFinal']),'test')
        except (FileNotFoundError, RuntimeError):
            name=self.prefix+'_final_test_features'
            self.group([(name,self.command(configs['CFinal'],'features',name,gpus,split='test'))])
            self.cpu(configs['CFinal'],'features-finalize',name,split='test')
        for offset in range(0,2,len(gpus)):
            batch=list(configs)[offset:offset+len(gpus)]
            self.group([(self.path(m,'test').name,self.command(configs[m],'codec-eval',self.path(m,'test').name,
                          [gpu],split='test',method='C' if m=='CFinal' else 'D',training_run=self.path(m,'train').name))
                        for m,gpu in zip(batch,gpus)])
            for m in batch:
                self.cpu(configs[m],'codec-summary',self.path(m,'test').name,split='test',
                         method='C' if m=='CFinal' else 'D',training_run=self.path(m,'train').name)
        self.status('held_out_final_test_complete')
        final_test_report(self.config,self.root,self.path('CFinal','test'),self.path('DFinal','test'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default=str(Path(__file__).with_name('config.architecture.json')))
    parser.add_argument('--phase',choices=('prepare','baselines','diagnostic','ablations','final','test','all','status'),default='prepare')
    parser.add_argument('--gpus',default='all')
    args=parser.parse_args()
    path=Path(args.plan).resolve();spec=read_json(path)
    if Path(spec['run_prefix']).name != spec['run_prefix'] or spec['run_prefix'] in ('.','..'):
        raise ValueError('Invalid run prefix')
    config=read_json(path.parent/spec['base_config'])
    root=Path(config['output_root'])/(spec['run_prefix']+'_suite');root.mkdir(parents=True,exist_ok=True)
    if args.phase=='status':
        print(read_json(root/'status.json'));return
    with (root/'.suite.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'plan.json').exists() and read_json(root/'plan.json')!=spec:
            raise ValueError('Frozen architecture plan changed; choose a new prefix')
        write_json(root/'plan.json',spec)
        suite=ArchitectureSuite(spec,root,config)
        if args.phase=='prepare':
            suite.status('prepared_not_started',teacher_context=spec['teacher_context'],
                         stages=['baselines','diagnostic','E1/E2','E3/E4','final'])
            return
        def interrupt(signum,frame):raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM,interrupt)
        try:
            gpus=choose_gpus(args.gpus)
            if args.phase in ('all','baselines'):suite.baselines(gpus)
            if args.phase in ('all','diagnostic'):suite.diagnostic(gpus)
            if args.phase in ('all','ablations'):suite.ablations(gpus)
            if args.phase in ('all','final'):suite.final_pair(gpus)
            if args.phase=='test':suite.test_pair(gpus)
            if args.phase not in ('all','final','test'):suite.status('phase_complete',phase_completed=args.phase)
        except KeyboardInterrupt:suite.status('paused',note='Owned process groups stopped; outputs retained')
        except Exception as error:
            suite.status('failed',error=str(error));raise


if __name__=='__main__':main()
