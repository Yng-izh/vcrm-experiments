import json
from pathlib import Path
import struct

import numpy as np
import torch

from .index import inferred_as_of, key_vector
from .io import atomic_bytes

METHODS = ('icae_independent', 'icae_text', 'transmem_streaming', 'version_index', 'independent', 'vcrm', 'text_log')
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
        if any(e['id'] == event['id'] for e in self.entries):
            raise ValueError('Duplicate memory entry id')
        if self.method in ('text_log', 'icae_text'):
            entry.update(kind='text', payload=event['text'].encode())
        else:
            valid_shape = tuple(bundle.shape) == self.shape
            if self.method == 'icae_independent':
                valid_shape = (bundle.ndim == 2 and len(self.shape) == 2 and
                               bundle.shape[1] == self.shape[1] and bundle.shape[0] > 0 and
                               bundle.shape[0] % self.shape[0] == 0)
            if not valid_shape or not torch.isfinite(bundle).all():
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
        as_of = (query.get('as_of') or query.get('time')) if self.method.startswith('icae_') else (
            inferred_as_of(query) if self.method != 'transmem_streaming' else None)
        eligible = [e for e in self.entries if not as_of or e['time'] <= as_of]
        if query.get('state_key'):
            eligible = [e for e in eligible if e['state'] == query['state_key']]
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
            n = 0 if entry['kind'] == 'text' else int(entry['payload'].shape[-2])
            if slots + n > slot_limit:
                break
            selected.append(entry)
            slots += n
        return selected

    def bundle(self, selected):
        return torch.cat([self.decode(e) for e in selected], dim=-2) if selected else None

    def retrieval_audit(self, query, selected):
        key = key_vector(query['question'], self.key_dimensions)
        as_of = (query.get('as_of') or query.get('time')) if self.method.startswith('icae_') else (
            inferred_as_of(query) if self.method != 'transmem_streaming' else None)
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
        keys = sum(e['key'].astype('<f2').nbytes for e in self.entries)
        total = len(self.serialize())
        return {'memory_bytes': payload, 'index_key_bytes': keys,
                'resident_latent_slots': sum(int(e['payload'].shape[-2]) for e in self.entries
                                            if not isinstance(e['payload'], bytes)),
                'metadata_and_format_bytes': total - payload - keys, 'total_bytes': total,
                'budget_bytes': self.budget}


class KeyframeStore:
    """Slot-bounded immutable anchor blocks, with dependency-aware FIFO.

    A non-keyframe version is compressed once, while its original full memory
    is still current. Codes never become encoder inputs. Old keyframes remain
    until their last dependent is evicted; updates never rebase old codes.
    """
    magic = b'VCRMKF1\n'
    format_name = 'keyframe_slots_v1'

    def __init__(self, codec, slot_budget, keyframe_interval=5, key_dimensions=128, byte_budget=None):
        if slot_budget < 1 or keyframe_interval < 2:
            raise ValueError('Invalid keyframe slot budget or interval')
        self.codec, self.slot_budget = codec, slot_budget
        self.codec_signature = codec.signature()
        self.interval, self.key_dimensions, self.byte_budget = keyframe_interval, key_dimensions, byte_budget
        self.entries, self.blocks, self.counts = [], {}, {}

    def _entry(self, eid):
        return next(e for e in self.entries if e['id'] == eid)

    def _memory(self, entry):
        return torch.from_numpy(entry['payload'].copy())

    def decode(self, entry):
        memory = self._memory(entry)
        if entry['kind'] == 'full':
            return memory.float()
        anchor = self._memory(self._entry(entry['anchor_id'])) if self.codec.conditioned else None
        with torch.no_grad():
            return self.codec.decode(memory, anchor).detach().cpu()

    def current(self, state):
        return next((e for e in reversed(self.entries) if e['state'] == state), None)

    def ingest(self, event, memory, semantic_key):
        import copy
        state = event.get('state_key')
        if not state:
            raise ValueError('Keyframe writing requires an explicit or validated shared state assignment')
        if any(e['id'] == event['id'] for e in self.entries):
            raise ValueError('Duplicate version id')
        if (memory.ndim != 2 or memory.shape[1] != self.codec.dimension or
                memory.shape[0] <= self.codec.slots or not torch.isfinite(memory).all()):
            raise ValueError('Write only original full memory, never an m-slot code')
        wire = memory.detach().cpu().float().numpy().astype('<f2')
        if not np.isfinite(wire).all():
            raise ValueError('Memory cannot be represented as finite FP16')
        key = np.asarray(semantic_key, dtype='<f2')
        if key.shape != (self.key_dimensions,) or not np.isfinite(key).all():
            raise ValueError('Invalid semantic key')
        previous = self.current(state)
        if previous and event['time'] < previous['time']:
            raise ValueError('Write versions in chronological order')
        # Roll back the entire update if the budget cannot protect live anchors.
        snapshot = copy.deepcopy((self.entries, self.blocks, self.counts))
        try:
            if previous and not previous['keyframe']:
                if previous['kind'] != 'full' or previous['compression_count']:
                    raise AssertionError('Cannot re-encode a lossy historical code')
                anchor_id = self.blocks[state]
                anchor = self._memory(self._entry(anchor_id)) if self.codec.conditioned else None
                with torch.no_grad():
                    code = self.codec.encode(self._memory(previous), anchor).detach().cpu()
                code_wire = code.float().numpy().astype('<f2')
                if code.shape != (self.codec.slots, self.codec.dimension) or not np.isfinite(code_wire).all():
                    raise ValueError('Codec returned invalid or FP16-overflowing historical code')
                previous.update(kind='code', payload=code_wire,
                                anchor_id=anchor_id if self.codec.conditioned else None, compression_count=1)
            count = self.counts.get(state, 0)
            is_keyframe = count % self.interval == 0 or state not in self.blocks
            entry = {'id': event['id'], 'state': state, 'time': event['time'], 'kind': 'full',
                     'keyframe': is_keyframe, 'anchor_id': None, 'compression_count': 0,
                     'key': key.copy(), 'payload': wire}
            self.entries.append(entry)
            if is_keyframe:
                self.blocks[state] = entry['id']
            self.counts[state] = count + 1
            dropped = self._enforce_budget()
        except Exception:
            self.entries, self.blocks, self.counts = snapshot
            raise
        return {'dropped': dropped, **self.accounting()}

    def _enforce_budget(self):
        dropped = []
        while (self.resident_slots() > self.slot_budget or
               (self.byte_budget is not None and len(self.serialize()) > self.byte_budget)):
            current_ids = {self.current(state)['id'] for state in {e['state'] for e in self.entries}}
            protected = current_ids | set(self.blocks.values())
            referenced = {e['anchor_id'] for e in self.entries if e['anchor_id']}
            victim = next((e for e in self.entries if e['id'] not in protected | referenced), None)
            if victim is None:
                newest_state = self.entries[-1]['state']
                old_state = next((e['state'] for e in self.entries if e['state'] != newest_state), None)
                if old_state is None:
                    raise ValueError('Budget cannot hold protected current/keyframe memories')
                # Unrelated states leave as whole dependency groups; no dangling codes.
                dropped.extend(e['id'] for e in self.entries if e['state'] == old_state)
                self.entries = [e for e in self.entries if e['state'] != old_state]
                self.blocks.pop(old_state, None)
                continue
            self.entries.remove(victim)
            dropped.append(victim['id'])
        return dropped

    def select_version(self, state, as_of=None):
        entries = [e for e in self.entries if e['state'] == state and (not as_of or e['time'] <= as_of)]
        return max(entries, key=lambda e: (e['time'], e['id']), default=None)

    def bundle(self, selected, markers=None, slot_limit=None):
        pieces = []
        for entry in selected:
            memory = self.decode(entry)
            if markers is not None:
                role = 'current' if entry is self.current(entry['state']) else 'historical'
                memory = markers(memory, role, entry['time'])
            pieces.append(memory)
        result = torch.cat(pieces) if pieces else None
        if result is not None and slot_limit is not None and result.shape[0] > slot_limit:
            raise ValueError('Decoder slot budget exceeded, including role/time tokens')
        return result

    def resident_slots(self):
        return sum(e['payload'].shape[0] for e in self.entries)

    def serialize(self):
        descriptions, payloads, offset = [], [], 0
        for entry in self.entries:
            key = entry['key'].astype('<f2').tobytes()
            payload = entry['payload'].astype('<f2').tobytes()
            description = {k: entry[k] for k in ('id', 'state', 'time', 'kind', 'keyframe', 'anchor_id', 'compression_count')}
            description.update(offset=offset, key_bytes=len(key), payload_bytes=len(payload), shape=list(entry['payload'].shape))
            descriptions.append(description)
            payloads.append(key + payload)
            offset += len(key) + len(payload)
        header = json.dumps({'format': self.format_name, 'entries': descriptions,
                             'blocks': self.blocks, 'counts': self.counts, 'interval': self.interval,
                             'conditioned': self.codec.conditioned, 'dimension': self.codec.dimension,
                             'codec_sha256': self.codec_signature,
                             'codec_slots': self.codec.slots, 'key_dimensions': self.key_dimensions},
                            sort_keys=True, separators=(',', ':')).encode()
        return self.magic + struct.pack('<Q', len(header)) + header + b''.join(payloads)

    def save(self, path):
        if self.resident_slots() > self.slot_budget:
            raise AssertionError('Resident slot budget exceeded')
        data = self.serialize()
        if self.byte_budget is not None and len(data) > self.byte_budget:
            raise AssertionError('Byte guardrail exceeded')
        atomic_bytes(path, data)

    @classmethod
    def load(cls, path, codec, slot_budget, byte_budget=None):
        data = Path(path).read_bytes()
        if not data.startswith(cls.magic):
            raise ValueError('Invalid keyframe memory format')
        size = struct.unpack('<Q', data[8:16])[0]
        header = json.loads(data[16:16 + size])
        if header['format'] != cls.format_name:
            raise ValueError('Memory protocol/format mismatch')
        if (header['dimension'], header['codec_slots'], header['conditioned']) != (codec.dimension, codec.slots, codec.conditioned):
            raise ValueError('Keyframe codec identity mismatch')
        store = cls(codec, slot_budget, header['interval'], header['key_dimensions'], byte_budget)
        if header['codec_sha256'] != store.codec_signature:
            raise ValueError('Keyframe codec weights identity mismatch')
        store.blocks, store.counts = header['blocks'], header['counts']
        raw = data[16 + size:]
        for desc in header['entries']:
            start, end = desc['offset'], desc['offset'] + desc['key_bytes']
            payload = np.frombuffer(raw[end:end + desc['payload_bytes']], dtype='<f2').reshape(desc['shape']).copy()
            key = np.frombuffer(raw[start:end], dtype='<f2').copy()
            store.entries.append({k: desc[k] for k in ('id', 'state', 'time', 'kind', 'keyframe', 'anchor_id', 'compression_count')} |
                                 {'payload': payload, 'key': key})
        ids = {e['id'] for e in store.entries}
        if len(ids) != len(store.entries) or any(a not in ids for a in store.blocks.values()):
            raise ValueError('Invalid keyframe ids or live block')
        for entry in store.entries:
            if entry['kind'] not in ('full', 'code') or not np.isfinite(entry['payload']).all():
                raise ValueError('Invalid stored latent entry')
            if entry['kind'] == 'code' and (entry['compression_count'] != 1 or entry['payload'].shape != (codec.slots, codec.dimension)):
                raise ValueError('Invalid single-compression code')
            if entry['kind'] == 'code' and codec.conditioned:
                anchor = store._entry(entry['anchor_id']) if entry['anchor_id'] in ids else None
                if anchor is None or not anchor['keyframe'] or anchor['kind'] != 'full' or anchor['state'] != entry['state']:
                    raise ValueError('Dangling or cross-state anchor dependency')
        if store.resident_slots() > slot_budget or (byte_budget is not None and len(data) > byte_budget):
            raise ValueError('Loaded memory exceeds budget')
        store.validate_protocol()
        return store

    def validate_protocol(self):
        pass

    def accounting(self):
        payload = sum(e['payload'].nbytes for e in self.entries)
        keys = sum(e['key'].nbytes for e in self.entries)
        total = len(self.serialize())
        return {'resident_latent_slots': self.resident_slots(), 'slot_budget': self.slot_budget,
                'keyframe_slots': sum(e['payload'].shape[0] for e in self.entries if e['keyframe']),
                'memory_bytes': payload, 'index_key_bytes': keys,
                'metadata_and_format_bytes': total - payload - keys, 'total_bytes': total,
                'byte_guardrail': self.byte_budget}


class CurrentSnapshotStore(KeyframeStore):
    """Offline snapshot: encode every historical full feature against final current.

    No incremental writes, lossily rebased codes, or retained raw/full history.
    Changing current requires a new snapshot built from original full features.
    """
    magic = b'VCRMSP1\n'
    format_name = 'final_current_snapshot_v1'

    def ingest(self, *args, **kwargs):
        raise RuntimeError('Offline snapshot cannot ingest updates; rebuild from original full features')

    @torch.no_grad()
    def build(self, events, memories, keys):
        if self.entries or not events:
            raise ValueError('Build exactly one nonempty snapshot in a fresh store')
        ids = [e['id'] for e in events]
        if len(ids) != len(set(ids)) or any(not e.get('state_key') for e in events):
            raise ValueError('Snapshot needs unique observed IDs and explicit shared state keys')
        if set(memories) != set(ids) or set(keys) != set(ids):
            raise ValueError('Snapshot must use all original events, no query-dependent filtering')
        current = {}
        for event in events:
            state = event['state_key']
            if state in current and event['time'] < current[state]['time']:
                raise ValueError('Snapshot events must be chronological within each state')
            current[state] = event
        rows = []
        for event in events:
            memory = memories[event['id']]
            if (memory.ndim != 2 or memory.shape[1] != self.codec.dimension or
                    memory.shape[0] <= self.codec.slots or not torch.isfinite(memory).all()):
                raise ValueError('Snapshot sources must be original full features, not compressed codes')
            state, latest = event['state_key'], current[event['state_key']]
            is_current = event['id'] == latest['id']
            anchor = memories[latest['id']] if self.codec.conditioned else None
            payload = memory if is_current else self.codec.encode(memory, anchor)
            wire = payload.detach().float().cpu().numpy().astype('<f2')
            key = np.asarray(keys[event['id']], dtype='<f2')
            if not np.isfinite(wire).all() or key.shape != (self.key_dimensions,) or not np.isfinite(key).all():
                raise ValueError('Invalid snapshot FP16 payload or index key')
            if not is_current and wire.shape != (self.codec.slots, self.codec.dimension):
                raise ValueError('Snapshot code must have exactly m slots')
            rows.append({'id': event['id'], 'state': state, 'time': event['time'],
                         'kind': 'full' if is_current else 'code', 'keyframe': is_current,
                         'anchor_id': latest['id'] if not is_current and self.codec.conditioned else None,
                         'compression_count': 0 if is_current else 1, 'key': key.copy(), 'payload': wire})
        self.entries = rows
        self.blocks = {state: event['id'] for state, event in current.items()}
        self.counts = {state: sum(e['state'] == state for e in rows) for state in current}
        try:
            self.validate_protocol()
            if self.resident_slots() > self.slot_budget or (self.byte_budget is not None and len(self.serialize()) > self.byte_budget):
                raise ValueError('Snapshot budget cannot retain all versions; no eviction in main comparison')
        except Exception:
            self.entries, self.blocks, self.counts = [], {}, {}
            raise
        return {'dropped': [], 'protocol': self.format_name, 'source_full_slots': sum(v.shape[0] for v in memories.values()),
                'source_features_retained': False, 'online_update_supported': False, **self.accounting()}

    def validate_protocol(self):
        for state in {e['state'] for e in self.entries}:
            latest = self.current(state)
            full = [e for e in self.entries if e['state'] == state and e['kind'] == 'full']
            if (full != [latest] or self.blocks.get(state) != latest['id'] or
                    not latest['keyframe'] or latest['anchor_id'] is not None):
                raise ValueError('Snapshot must have one full final-current entry per state')
            for entry in (e for e in self.entries if e['state'] == state and e is not latest):
                if (entry['kind'] != 'code' or entry['keyframe'] or entry['time'] > latest['time'] or
                        entry['anchor_id'] != (latest['id'] if self.codec.conditioned else None)):
                    raise ValueError('Snapshot code condition must be the same persisted final current')

    def accounting(self):
        return super().accounting() | {'protocol': self.format_name, 'online_update_supported': False,
                                       'additional_anchor_slots': 0}
