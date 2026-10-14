"""Offline error controls; oracle/text access here is NOT a deployable latent baseline."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics

from vcrm.data import query_view
from vcrm.io import digest, read_json, write_json
from vcrm.judge import score as judge_score
from vcrm.reader import FrozenReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline-run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-type', type=int, default=4)
    args = parser.parse_args()
    config = read_json(args.config)
    baseline = Path(config['output_root']) / args.baseline_run
    output = Path(args.output)
    episodes = {e['id']: e for e in read_json(Path(config['data_root']) / 'prepared/longmemeval_test.json')}
    original = {r['question_id']: r for r in read_json(Path(config['data_root']) / 'longmemeval_s_cleaned.json')}
    chosen, used = [], Counter()
    for path in sorted(baseline.glob('*/*/budget_*/episodes/*.json')):
        prediction = read_json(path)
        scores = read_json(path.parent.parent / 'scores' / path.name)['scores'][0]
        diagnostics = scores['evidence_diagnostics']
        category = scores['type']
        # Full evidence coverage avoids blaming the reader for incomplete retrieval.
        if (diagnostics['failure_bucket'] != 'selected_but_wrong' or
                diagnostics['selected_count'] != diagnostics['gold_count'] or used[category] >= args.per_type):
            continue
        chosen.append(prediction)
        used[category] += 1
    write_json(output / 'protocol.json', {
        'scope': 'Post-hoc error diagnosis only; selected-text/oracle controls are NOT baseline benchmark results',
        'selection': 'Deterministic subset of wrong answers with all gold evidence selected, at most four per type',
        'per_type': args.per_type, 'baseline_run': str(baseline),
        'reader_config': config, 'case_ids': [r['id'] for r in chosen],
        'warning': 'Nonrepresentative error subset. No training, tuning, or changes to formal predictions.'})
    reader = FrozenReader(config)
    for index, prediction in enumerate(chosen):
        path = output / 'cases' / (digest(prediction['id']) + '.json')
        if path.exists():
            continue
        episode = episodes[prediction['id']]
        query = episode['queries'][0]
        events = {e['id']: e for e in episode['events']}
        selected = prediction['queries'][0]['selected_entry_ids']
        annotations = original[query['id']]
        evidence = set(map(str, annotations['answer_session_ids']))
        gold = [f'event_{i:06d}' for i, sid in enumerate(annotations['haystack_session_ids']) if str(sid) in evidence]
        text = lambda ids: '\n\n'.join(f"DATE: {events[i]['time']}\n{events[i]['text']}" for i in ids)
        contexts = {'no_memory': '', 'selected_text': text(selected), 'oracle_evidence_text': text(gold)}
        controls = {}
        for name, context in contexts.items():
            ids = reader.prompt(query_view(query), context)
            if ids.shape[1] > reader.model.config.max_position_embeddings:
                raise ValueError('Diagnostic exceeds model context; no silent cropping')
            result = reader.answer(query_view(query), text=context)
            values = judge_score(config, query, result['answer'], output)
            controls[name] = result | values
        write_json(path, {'id': prediction['id'], 'query': query,
                          'baseline_answer': prediction['queries'][0]['answer'], 'baseline_accuracy': 0,
                          'selected_ids': selected, 'gold_ids': gold, 'controls': controls,
                          'scope': 'Offline selected-error controls, excluded from all formal scores'})
        print(f"diagnostic {index + 1}/{len(chosen)} {prediction['id']}: "
              f"{ {k: v['accuracy'] for k, v in controls.items()} }", flush=True)
    cases = [read_json(p) for p in (output / 'cases').glob('*.json')]
    summary = {'cases': len(cases), 'expected_cases': len(chosen),
               'scope': 'Selected all-evidence-present error subset, NOT benchmark-wide accuracy',
               'by_type': dict(Counter(r['query']['type'] for r in cases)),
               'control_accuracy_on_selected_errors': {
                   name: statistics.mean(r['controls'][name]['accuracy'] for r in cases) if cases else None
                   for name in ('no_memory', 'selected_text', 'oracle_evidence_text')},
               'examples': [{k: r[k] for k in ('id', 'query', 'baseline_answer', 'controls')} for r in cases]}
    write_json(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
