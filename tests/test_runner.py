import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def runner():
    path = Path(__file__).parents[1] / 'run.py'
    spec = importlib.util.spec_from_file_location('icae_experiment_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_selection_excludes_busy_and_duplicate_cards(monkeypatch):
    module = runner()
    monkeypatch.setattr(module.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(
        stdout='0, 0, 32501, 0\n1, 17816, 14000, 98\n2, 0, 32501, 0\n'))
    assert module.choose_gpus('all') == [0, 2]
    assert module.choose_gpus('auto') == [0]
    with pytest.raises(RuntimeError):
        module.choose_gpus('1')
    with pytest.raises(RuntimeError):
        module.choose_gpus('2,2')


def test_worker_does_not_acquire_parent_run_lock(tmp_path, monkeypatch):
    import json
    module = runner()
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps({'output_root': str(tmp_path / 'runs')}))
    args = SimpleNamespace(config=str(config_path), run_name='worker', limit=None, worker=True,
                           stage='readability', shard=0, shards=1)
    monkeypatch.setattr(module, 'arguments', lambda: args)
    monkeypatch.setattr(module.fcntl, 'flock', lambda *args: pytest.fail('Worker must not acquire parent lock'))
    calls = []
    monkeypatch.setattr(module, 'worker', lambda config, root, options: calls.append(str(root)))
    module.main()
    assert calls == [str(tmp_path / 'runs/worker')]


def test_retired_stages_unavailable():
    assert 'reproduce' not in runner().STAGES
    assert 'train' not in runner().STAGES
    # 'features' / 'features-finalize' 是服务器 run.py 里的合法新 stage（特征编码/落盘），
    # 不是退役 stage；只有 reproduce / train 已退役。
    assert 'features' in runner().STAGES
    assert 'features-finalize' in runner().STAGES
    assert 'adapt' in runner().STAGES
