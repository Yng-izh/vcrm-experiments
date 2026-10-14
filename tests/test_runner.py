import fcntl
import json
from pathlib import Path
import subprocess
import sys
import importlib.util
from types import SimpleNamespace

import pytest

from vcrm.io import write_json


def test_status_can_read_atomic_outputs_while_run_is_locked(tmp_path):
    root = tmp_path / 'runs/busy'
    root.mkdir(parents=True)
    config = tmp_path / 'config.json'
    write_json(config, {'output_root': str(tmp_path / 'runs'),
                        'transmem_root': str(tmp_path), 'delta_root': str(tmp_path)})
    write_json(root / 'templama_val/vcrm/budget_1000/episodes/one.json', {'id': 'one'})
    with (root / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'run.py'),
                                 '--stage', 'status', '--config', str(config), '--run-name', 'busy'],
                                check=True, capture_output=True, text=True, timeout=20)
    assert json.loads(result.stdout) == {'generated': 1, 'scored': 0, 'root': str(root),
                                        'locomo_completed_questions': 0}


def test_reproduction_rejects_incomplete_or_duplicate_original_questions():
    spec = importlib.util.spec_from_file_location('vcrm_runner', Path(__file__).parents[1] / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    samples = [{'sample_id': 's', 'qa': [{'category': 1}, {'category': 2}, {'category': 5}]}]
    row = lambda i: {'key': f's:{i}', 'category': i + 1, 'score': .5, 'raw_prediction': 'answer'}
    payload = {'summary': {'num_questions': 2}, 'records': [row(0), row(1)]}
    assert len(module.validate_reproduction(payload, samples)) == 2
    for rows in ([row(0)], [row(0), row(0)]):
        with pytest.raises(RuntimeError, match='question keys'):
            module.validate_reproduction(payload | {'records': rows}, samples)


def test_failed_streaming_baseline_holds_formal_training_and_comparisons(tmp_path):
    spec = importlib.util.spec_from_file_location('vcrm_runner', Path(__file__).parents[1] / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    write_json(tmp_path / 'stage2_decision.json', {'decision': 'hold_formal_comparisons_and_codec_training'})
    config = {'project': str(tmp_path)}
    for stage, methods in [('train', ['vcrm']), ('stream', ['independent', 'vcrm'])]:
        with pytest.raises(RuntimeError, match='diagnosis failed'):
            module.require_streaming_baseline(config, SimpleNamespace(limit=None, stage=stage, methods=methods))
    module.require_streaming_baseline(config, SimpleNamespace(limit=None, stage='stream', methods=['transmem_streaming']))
    module.require_streaming_baseline(config, SimpleNamespace(limit=1, stage='train', methods=['vcrm']))
