import os
from pathlib import Path
import subprocess
import sys

from .io import read_json, write_json


def evaluate(config, root, conflict_type):
    metadata = read_json(Path(config['data_root']) / 'stale_provenance.json')
    upstream = Path(config['project']) / 'upstream/STALE/STALE/Evaluation/full_eval_performance.py'
    exports = []
    for folder in Path(root).glob('stale_test/*/budget_*'):
        predictions = []
        for file in (folder / 'episodes').glob('*.json'):
            row = read_json(file)
            responses = {q['type'] + '_response': q['answer'] for q in row['queries']}
            predictions.append({'uid': row['group'], 'target_model_responses': responses})
        dest = folder / 'stale_answers.json'
        write_json(dest, predictions)
        output = folder / 'stale_official_evaluation.json'
        env = dict(os.environ, JUDGE_PROVIDER='OPENAI', JUDGE_MODEL=config['judge_model'],
                   OPENAI_BASE_URL=config['judge_base_url'])
        subprocess.run([sys.executable, str(upstream), '--answers-path', str(dest),
                        '--dataset-path', metadata['source_path'], '--output-path', str(output),
                        '--conflict-type', conflict_type, '--model-method', 'target_model'],
                       check=True, env=env, cwd=upstream.parent)
        exports.append({'output': str(output), 'provenance': metadata['provenance']})
    return exports
