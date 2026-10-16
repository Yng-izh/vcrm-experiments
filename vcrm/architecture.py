"""Frozen-reader diagnostics and one-variable architecture comparisons."""
from pathlib import Path
import statistics

import torch

from .data import ingestion_view, query_view
from .io import atomic_bytes, file_digest, read_json, row_path, write_json
from .metrics import exact, f1, paired_bootstrap
from .refinement import conflicting, diagnostic_metrics, result_rows, selected_episodes


READS = ('O128', 'N128_O128', 'H16_fp32', 'N128_H16_fp32', 'H16_fp16', 'N128_H16_fp16')


def read_diagnostic(config, root, training_root, shard=0, shards=1):
    from .features import check, load
    from .runtime import SlotMemoryRuntime
    check(config, 'val')
    runtime = SlotMemoryRuntime.from_checkpoint(config, training_root)
    episodes = read_json(Path(config['data_root']) / 'prepared/templama_val.json')
    episodes = selected_episodes(config, episodes, 'val')
    for episode in episodes[shard::shards]:
        destination = row_path(root, 'episodes', episode['id'])
        if destination.exists():
            continue
        events = ingestion_view(episode)
        current_event = events[-1]
        current = load(config, 'val', episode['id'], current_event).to(runtime.reader.device).float()
        partial = row_path(root, 'partial_episodes', episode['id'])
        outputs = read_json(partial)['queries'] if partial.exists() else []
        by_id = {row['query_id']: row for row in outputs}
        for query in episode['queries']:
            selected = [e for e in events if e['time'] <= query['as_of']][-1]
            if selected['id'] == current_event['id'] or query['latest']:
                raise ValueError('Read diagnostic requires genuine historical versions')
            old = load(config, 'val', episode['id'], selected).to(runtime.reader.device).float()
            with torch.no_grad():
                code = runtime.codec.encode(old, current)
                recovered32 = runtime.codec.decode(code, current)
                recovered16 = runtime.codec.decode(code.half().float(), current)
                marked_current = runtime.markers(current, 'current', current_event['time'])
                mark = lambda memory: runtime.markers(memory, 'historical', selected['time'])
                memories = {'O128': mark(old), 'N128_O128': torch.cat((marked_current, mark(old))),
                            'H16_fp32': mark(recovered32), 'N128_H16_fp32': torch.cat((marked_current, mark(recovered32))),
                            'H16_fp16': mark(recovered16), 'N128_H16_fp16': torch.cat((marked_current, mark(recovered16)))}
                row = by_id.get(query['id'])
                if row is None:
                    row = {'query_id': query['id'], 'selected_entry_ids': [current_event['id'], selected['id']],
                           'reads': {}, 'code_fp16_relative_l2': float((code-code.half().float()).norm() / code.norm().clamp_min(1e-12))}
                    outputs.append(row)
                    by_id[query['id']] = row
                for name in READS:
                    if name in row['reads']:
                        continue
                    answer = runtime.reader.answer(query_view(query), memories[name])
                    row['reads'][name] = {'answer': answer, 'score': {
                        'f1': f1(answer['answer'], query['answers']), 'accuracy': exact(answer['answer'], query['answers'])}}
                    write_json(partial, {'id': episode['id'], 'queries': outputs})
                    print(f'read-diagnostic {episode["id"]} {query["id"]} {name}', flush=True)
        write_json(destination, {'id': episode['id'], 'queries': outputs})
        partial.unlink(missing_ok=True)


def diagnostic_summary(config, root):
    manifest = read_json(config['evaluation_query_manifest'])
    raw = result_rows(root)
    if set(raw) != set(manifest['query_ids']) or any(set(r['reads']) != set(READS) for r in raw.values()):
        raise ValueError('Read diagnostic must cover every fixed query and read/precision condition')
    metrics = {name: diagnostic_metrics({qid: row['reads'][name] for qid, row in raw.items()}, manifest)
               for name in READS}
    comparisons = {}
    for a, b in [('O128', 'N128_O128'), ('H16_fp16', 'N128_H16_fp16'),
                 ('O128', 'H16_fp16'), ('N128_O128', 'N128_H16_fp16'),
                 ('H16_fp32', 'H16_fp16'), ('N128_H16_fp32', 'N128_H16_fp16')]:
        comparisons[b + '_minus_' + a] = paired_deltas(metrics[a], metrics[b])
    result = {'complete': True, 'metrics': metrics, 'paired': comparisons,
              'quantization_scope': 'Same frozen FP16 ICAE features, decoder and checkpoint; compare only codec '
                                    'code FP32 versus FP16 storage roundtrip. Not FP32-backbone versus FP16-backbone.',
              'code_fp16_relative_l2_mean': statistics.mean(r['code_fp16_relative_l2'] for r in raw.values()),
              'note': 'History-only D recovery still uses current as codec condition; it is withheld only from '
                      'Reader. Role/time marker weights remain identical within each checkpoint.'}
    write_json(Path(root) / 'diagnostic_summary.json', result)
    return result


def paired_deltas(control, candidate):
    result = {}
    for scope in ('all_compressed', 'disjoint_current_conflicts'):
        a, b = [v[scope]['rows'] for v in (control, candidate)]
        if [r['query_id'] for r in a] != [r['query_id'] for r in b]:
            raise ValueError('Paired architecture query order changed')
        result[scope] = {key: paired_bootstrap([r[key] for r in b], [r[key] for r in a], [r['group'] for r in a])
                         for key in ('f1', 'accuracy', 'stale_answer')}
    return result


def evaluation_metrics(config, root):
    return diagnostic_metrics(result_rows(root), read_json(config['evaluation_query_manifest']))


def supports_change(control, candidate, check_stale=False):
    historical = candidate['all_compressed']['f1'] - control['all_compressed']['f1']
    conflict = candidate['disjoint_current_conflicts']['f1'] - control['disjoint_current_conflicts']['f1']
    stale = candidate['disjoint_current_conflicts']['stale_answer'] - control['disjoint_current_conflicts']['stale_answer']
    return historical > 0 and conflict >= 0 and (not check_stale or stale <= 0)


def compare(config, root, name, control_root, candidate_root):
    control, candidate = [evaluation_metrics(config, p) for p in (control_root, candidate_root)]
    result = {'experiment': name, 'control_root': str(control_root), 'candidate_root': str(candidate_root),
              'control': control, 'candidate': candidate, 'paired_candidate_minus_control': paired_deltas(control, candidate),
              'point_estimate_support': supports_change(control, candidate, check_stale=name == 'E4'),
              'support_rule': 'Historical F1 increases, conflict F1 does not fall; E4 additionally cannot increase '
                              'conflict stale rate. This is a validation selection rule, not proof of statistical significance.',
              'parameter_counts': {}, 'costs': {}}
    training_metadata = {}
    manifests = {}
    for label, path in [('control', control_root), ('candidate', candidate_root)]:
        summary = read_json(Path(path) / 'summary.json')
        if not summary['complete']:
            raise ValueError('Do not compare incomplete architecture evaluations')
        for key in ('decoder_latent_slots_mean', 'read_seconds_mean', 'recovery_seconds_mean', 'write_seconds_mean',
                    'kv_cache_bytes_mean', 'disk_bytes_mean'):
            result['costs'].setdefault(label, {})[key] = summary[key]
        manifest = read_json(Path(path) / 'manifest.json')
        manifests[label] = manifest
        training = Path(config['output_root']) / manifest['training_run']
        metadata = read_json(training / 'codec_training.json')
        training_metadata[label] = metadata
        result['parameter_counts'][label] = {k: metadata[k] for k in ('codec_parameters','marker_parameters','index_parameters')}
    a, b = [training_metadata[k] for k in ('control','candidate')]
    for key in ('pairs','pair_content_sha256','training_data_sha256','steps','history_slots','seed',
                'codec_epochs','distillation_context','base_checkpoint_sha256','reader_sha256'):
        if a[key] != b[key]:
            raise ValueError(f'Architecture controls differ in shared training invariant: {key}')
    if a.get('codec_activation_rms_cap') != b.get('codec_activation_rms_cap'):
        raise ValueError('Architecture controls differ in shared training invariant: codec_activation_rms_cap')
    allowed = {'E1': {'decode_conditioned'}, 'E2': {'codec_variant'}, 'E3': {'codec_value_path'},
               'E4': {'contrastive_weight'}, 'Final': set()}
    if name in allowed:
        ca, cb = [manifests[k]['config'] for k in ('control','candidate')]
        changed = {key for key in set(ca) | set(cb) if ca.get(key) != cb.get(key)}
        if changed != allowed[name]:
            raise ValueError(f'{name} must change only {allowed[name]}, actually changed {changed}')
        result['changed_config_keys'] = sorted(changed)
    write_json(Path(root) / (name + '.json'), result)
    return result


def ablation_table(root):
    root = Path(root)
    lines = ['| Experiment | Historical F1 delta (pp) | Conflict F1 delta (pp) | Conflict stale delta (pp) | Retained |',
             '| --- | ---: | ---: | ---: | --- |']
    for name in ('E1','E2','E3','E4'):
        row = read_json(root / (name + '.json'))
        if row.get('skipped'):
            lines.append(f'| {name} | - | - | - | skipped: preceding branch unsupported |')
            continue
        paired = row['paired_candidate_minus_control']
        historical = paired['all_compressed']['f1']['delta']
        conflict = paired['disjoint_current_conflicts']['f1']['delta']
        stale = paired['disjoint_current_conflicts']['stale_answer']['delta']
        lines.append(f'| {name} | {historical*100:+.2f} | {conflict*100:+.2f} | {stale*100:+.2f} | '
                     f'{row["point_estimate_support"]} |')
    lines += ['', 'Paired episode-bootstrap 95% intervals, controls, parameter counts and costs: E1.json through E4.json.',
              'Validation selection uses point estimates, not evidence of statistical significance or mechanism causality.']
    atomic_bytes(root / 'ABLATIONS.md', ('\n'.join(lines)+'\n').encode())


def final_test_report(config, root, c_root, d_root):
    summaries = [read_json(Path(p) / 'summary.json') for p in (c_root,d_root)]
    if any(not r['complete'] or r['scope'] != 'held_out_test' for r in summaries):
        raise ValueError('Final test requires complete untouched test coverage')
    episodes = read_json(Path(config['data_root']) / 'prepared/templama_test.json')
    c, d = result_rows(c_root), result_rows(d_root)
    expected = {q['id'] for e in episodes for q in e['queries']}
    if set(c) != expected or set(d) != expected:
        raise ValueError('Final test query coverage mismatch')
    scopes = {'historical': [], 'conflict': [], 'latest': []}
    for episode in episodes:
        latest = next(q for q in episode['queries'] if q['latest'])
        for query in episode['queries']:
            item = (query['id'],episode['id'])
            if query['latest']:
                scopes['latest'].append(item)
            elif len(episode['events']) > 1:
                scopes['historical'].append(item)
                if conflicting(query['answers'],latest['answers']):
                    scopes['conflict'].append(item)
    comparisons = {}
    for scope, selected in scopes.items():
        key = 'accuracy' if scope == 'latest' else 'f1'
        comparisons[scope] = paired_bootstrap([d[q]['score'][key] for q,e in selected],
                                              [c[q]['score'][key] for q,e in selected],
                                              [e for q,e in selected])
    result = {'scope':'held_out_test', 'C':summaries[0], 'D':summaries[1],
              'paired_D_minus_C': comparisons,
              'note':'Frozen architecture selected before this test; do not use test to reselect modules.'}
    write_json(Path(root) / 'final_test_comparison.json',result)
    return result
