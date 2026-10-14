import json
from pathlib import Path
import struct

import numpy as np
import torch

from .index import inferred_as_of, key_vector
from .io import atomic_bytes

METHODS = ('transmem_streaming', 'version_index', 'independent', 'vcrm', 'text_log')
MAGIC = b'VCRM001\n'


class MemoryStore:
    def __init__(self, method, budget, shape, key_dimensions=128, codec=None, similarity=.8):
        if method not in METHODS:
            raise ValueError(method)
        self.method, self.budget, self.shape = method, budget, tuple(shape)
        self.key_dimensions, self.codec, self.similarity = key_dimensions, codec, similarity
        self.entries = []
        if method in ('independent', 'vcrm') and codec is None:
            raise ValueError('Learned methods require a validated trained codec')
        if budget < len(self.serialize()):
            raise ValueError('Budget cannot hold even the empty store header')

    def _current(self, state):
        return next((e for e in reversed(self.entries) if e['state'] == state and e['kind'] in ('full', 'text')), None)

    def _tensor(self, entry):
        return torch.from_numpy(entry['payload'].astype(np.float32).copy())

    def decode(self, entry):
        if entry['kind'] == 'full':
            return self._tensor(entry)
        current = self._current(entry['state'])
        if current is None:
            raise AssertionError('Dangling version dependency')
        with torch.no_grad():
            return self.codec.decode(self._tensor(entry), self._tensor(current)).detach().cpu()

    def ingest(self, event, bundle=None):
        key = key_vector(event.get('key_text', event['text']), self.key_dimensions)
        state = event.get('state_key')
        if state is None:
            # Shared, label-free lexical association, not a semantic conflict oracle.
            candidates = [(float(key @ e['key']), e['state']) for e in self.entries]
            best = max(candidates, default=(-1., event['id']))
            state = best[1] if best[0] >= self.similarity else event['id']
        current = self._current(state)
        if current and event['time'] < current['time']:
            raise ValueError('Events must be ingested in chronological order')
        entry = {'id': event['id'], 'state': state, 'time': event['time'], 'key': key}
        if self.method == 'text_log':
            entry.update(kind='text', payload=event['text'].encode())
        else:
            if tuple(bundle.shape) != self.shape or not torch.isfinite(bundle).all():
                raise ValueError('Invalid memory feature shape or non-finite features')
            full = bundle.detach().float().cpu()
            if self.method in ('independent', 'vcrm') and current is not None:
                archived = [e for e in self.entries if e['state'] == state and
                            (self.method == 'vcrm' or e is current)]
                decoded = [(e, self.decode(e)) for e in archived]
                for previous, old in decoded:
                    with torch.no_grad():
                        code = self.codec.encode(old, full)
                    previous['kind'] = 'code'
                    previous['payload'] = code.cpu().numpy().astype('<f2')
            entry.update(kind='full', payload=full.numpy().astype('<f2'))
        self.entries.append(entry)
        dropped = []
        while len(self.serialize()) > self.budget:
            victim = self.entries[0]
            if self.method in ('independent', 'vcrm') and victim['kind'] == 'full':
                dependent = [e for e in self.entries if e['state'] == victim['state'] and e['kind'] == 'code']
                victim = min(dependent, key=lambda e: e['time']) if dependent else victim
            self.entries = [e for e in self.entries if e is not victim]
            dropped.append(victim['id'])
        return {'dropped': dropped, 'persisted_bytes': len(self.serialize()), 'entries': len(self.entries)}

    def select(self, query, slot_limit):
        key = key_vector(query['question'], self.key_dimensions)
        as_of = inferred_as_of(query) if self.method != 'transmem_streaming' else None
        eligible = [e for e in self.entries if not as_of or e['time'] <= as_of]
        if self.method != 'transmem_streaming':
            # For a past-version query select the last observation not later than the requested time.
            latest = {}
            for entry in eligible:
                if entry['state'] not in latest or latest[entry['state']]['time'] < entry['time']:
                    latest[entry['state']] = entry
            eligible = list(latest.values())
        ranked = sorted(eligible, key=lambda e: (float(key @ e['key']), e['time'], e['id']), reverse=True)
        selected, slots = [], 0
        for entry in ranked:
            n = self.shape[1]
            if slots + n > slot_limit:
                break
            selected.append(entry)
            slots += n
        return selected

    def bundle(self, selected):
        return torch.cat([self.decode(e) for e in selected], dim=1) if selected else None

    def retrieval_audit(self, query, selected):
        key = key_vector(query['question'], self.key_dimensions)
        as_of = inferred_as_of(query) if self.method != 'transmem_streaming' else None
        chosen = {e['id'] for e in selected}
        return {'as_of': as_of, 'candidates': [
            {'id': e['id'], 'time': e['time'], 'similarity': float(key @ e['key']),
             'selected': e['id'] in chosen, 'excluded_future': bool(as_of and e['time'] > as_of)}
            for e in self.entries]}

    def text(self, selected):
        return '\n\n'.join(f"DATE: {e['time']}\n{e['payload'].decode()}" for e in selected)

    def serialize(self):
        descriptions, payloads, offset = [], [], 0
        for entry in self.entries:
            key = entry['key'].astype('<f2').tobytes()
            value = entry['payload']
            payload = value if isinstance(value, bytes) else value.astype('<f2').tobytes()
            descriptions.append({k: entry[k] for k in ('id', 'state', 'time', 'kind')} |
                                {'offset': offset, 'key_bytes': len(key), 'payload_bytes': len(payload),
                                 'payload_shape': None if isinstance(value, bytes) else list(value.shape)})
            payloads.append(key + payload)
            offset += len(key) + len(payload)
        header = json.dumps({'method': self.method, 'shape': self.shape, 'key_dimensions': self.key_dimensions,
                             'entries': descriptions}, separators=(',', ':'), sort_keys=True).encode()
        return MAGIC + struct.pack('<Q', len(header)) + header + b''.join(payloads)

    def save(self, path):
        data = self.serialize()
        if len(data) > self.budget:
            raise AssertionError('Memory byte budget exceeded')
        atomic_bytes(path, data)
        return len(data)

    @classmethod
    def load(cls, path, budget, codec=None, similarity=.8):
        data = Path(path).read_bytes()
        if not data.startswith(MAGIC):
            raise ValueError('Wrong memory store format')
        size = struct.unpack('<Q', data[len(MAGIC):len(MAGIC) + 8])[0]
        begin = len(MAGIC) + 8
        header = json.loads(data[begin:begin + size])
        store = cls(header['method'], budget, header['shape'], header['key_dimensions'], codec, similarity)
        if len(data) > budget:
            raise ValueError('Loaded memory exceeds byte budget')
        raw = data[begin + size:]
        for desc in header['entries']:
            start = desc['offset']
            end = start + desc['key_bytes']
            payload = raw[end:end + desc['payload_bytes']]
            value = payload if desc['kind'] == 'text' else np.frombuffer(payload, dtype='<f2').reshape(desc['payload_shape']).copy()
            store.entries.append({k: desc[k] for k in ('id', 'state', 'time', 'kind')} |
                                 {'key': np.frombuffer(raw[start:end], dtype='<f2').copy(), 'payload': value})
        return store

    def accounting(self):
        payload = sum(len(e['payload']) if isinstance(e['payload'], bytes) else e['payload'].nbytes for e in self.entries)
        keys = sum(e['key'].nbytes for e in self.entries)
        total = len(self.serialize())
        return {'memory_bytes': payload, 'index_key_bytes': keys,
                'metadata_and_format_bytes': total - payload - keys, 'total_bytes': total,
                'budget_bytes': self.budget}
