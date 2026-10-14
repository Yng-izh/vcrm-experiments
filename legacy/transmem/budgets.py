from pathlib import Path

import numpy as np

from .index import key_vector
from .io import digest, read_json, write_json
from .store import MemoryStore


def freeze_budgets(config):
    """Measure C's serialized layout on validation facts, without fitting to scores."""
    episodes = read_json(Path(config['data_root']) / 'prepared/templama_val.json')
    # Match the released layered adapter architecture; never infer budgets from test QA.
    import torch
    ckpt = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    shape = (len(cfg['inject_layers']), cfg['n_mem'], cfg['dim'])
    sizes, anchors = [], []
    for episode in episodes:
        store = MemoryStore('version_index', 2 ** 40, shape, config['key_dimensions'])
        store.method = 'independent'
        for i, event in enumerate(episode['events']):
            current = i == len(episode['events']) - 1
            payload_shape = shape if current else (*shape[:2], config['rank'])
            store.entries.append({'id': event['id'], 'state': event['state_key'], 'time': event['time'],
                                  'kind': 'full' if current else 'code',
                                  'key': key_vector(event['key_text'], config['key_dimensions']),
                                  'payload': np.zeros(payload_shape, dtype='<f2')})
        sizes.append(len(store.serialize()))
        store.entries = store.entries[-1:]
        anchors.append(len(store.serialize()))
    reference = int(np.quantile(sizes, .9))
    anchor = max(anchors)
    values = sorted(set(anchor + max(1, int((reference - anchor) * factor)) for factor in (.25, .5, 1.)))
    plan = {'budgets_bytes': values, 'reference_bytes': reference,
            'anchor_reference_bytes': anchor,
            'selection': 'One validation-max current anchor plus 25%, 50%, 100% of validation-p90 compressed historical footprint',
            'calibration_split': 'templama_val', 'calibration_ids_sha256': digest([e['id'] for e in episodes]),
            'calibration_scores_used': False, 'layout_only': True,
            'independent_layout_bytes_min': min(sizes), 'max': max(sizes),
            'note': 'Layout-size measurement uses zero-valued placeholders, never prediction results. Model/checkpoint bytes are reported separately.'}
    dest = Path(config['project']) / 'budget_plan.json'
    if dest.exists() and read_json(dest) != plan:
        raise RuntimeError('Frozen budget plan differs; create a new protocol rather than silently replacing it')
    write_json(dest, plan)
    return plan
