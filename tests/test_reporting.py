import json
from pathlib import Path

import pytest

from vcrm.io import row_path, write_json
from vcrm.pipeline import report


@pytest.mark.parametrize('limit,c_signature,d_signature,reject', [
    (1, None, None, False), (None, 'matched', 'matched', False),
    (None, 'first-training', 'other-training', True),
])
def test_storage_curves_and_paired_metrics_are_scored_data(tmp_path, limit, c_signature, d_signature, reject):
    root = tmp_path / 'run'
    data = tmp_path / 'data'
    episode = {'id': 'e', 'group': 'g', 'queries': [{'id': 'old', 'latest': False}, {'id': 'new', 'latest': True}]}
    write_json(data / 'prepared/templama_val.json', [episode])
    write_json(root / 'manifest.json', {'limit': limit})
    for method, values in (('independent', [0, 1]), ('vcrm', [1, 1])):
        folder = root / 'templama_val' / method / 'budget_1000'
        predictions = [{'generation_limit_hit': False, 'peak_gpu_bytes': 100, 'read_seconds': .2} for _ in values]
        write_json(row_path(folder, 'episodes', 'e'), {'store_bytes': 900, 'write_seconds': .5, 'queries': predictions,
                                                      'model_weights': {'codec_parameters': 100,
                                                                        'codec_training_signature': c_signature
                                                                        if method == 'independent' else d_signature}})
        scores = [{'query_id': qid, 'latest': qid == 'new', 'accuracy': value, 'f1': value,
                   'stale_answer': 0} for qid, value in zip(('old', 'new'), values)]
        write_json(row_path(folder, 'scores', 'e'), {'group': 'g', 'scores': scores})
    config = {'data_root': str(data), 'bootstrap_samples': 20, 'seed': 42}
    if reject:
        with pytest.raises(ValueError, match='different training provenance'):
            report(config, root)
        return
    result = report(config, root)
    assert result['results']['templama_val/vcrm/budget_1000']['complete']
    assert result['paired_comparisons']['templama_val/budget_1000/vcrm_minus_independent/historical']['delta'] == 1
    assert result['paired_comparisons']['templama_val/budget_1000/vcrm_minus_independent/latest']['delta'] == 0
    assert (root / 'reports/templama_val_tradeoff.png').stat().st_size > 1000
    assert ('smoke' if limit else 'formal') in (root / 'reports/storage_tradeoff.csv').read_text()
