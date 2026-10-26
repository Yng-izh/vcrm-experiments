import json
from pathlib import Path
import struct
from typing import Any, Collection, Dict, List, Optional

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
        self._wire_cache = {}
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
            self._wire_cache.pop(victim['id'], None)
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
        self._wire_cache = {}          # id → (key_bytes, payload_bytes, shape)，见 serialize()

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
                self._wire_cache.pop(previous['id'], None)   # payload 将被替换
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
            self._accounting_total = None      # 回滚 = 内容变更
            raise
        self._accounting_total = None          # 写路径已变更 entries/payload
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
                self._wire_cache.clear()          # 整状态移除，直接清空
                self.entries = [e for e in self.entries if e['state'] != old_state]
                self.blocks.pop(old_state, None)
                self._accounting_total = None   # 淘汰 = 内容变更
                continue
            self.entries.remove(victim)
            self._wire_cache.pop(victim['id'], None)
            self._accounting_total = None       # 淘汰 = 内容变更
            dropped.append(victim['id'])
        return dropped

    def select_version(self, state, as_of=None, *, allow_future_fallback: bool = False):
        """按 as_of 取版本：as_of 之后写入的版本不可见。

        as_of 之前一条版本都没有（该状态在 as_of 时点尚未出现）时返回 None；
        ``allow_future_fallback=True`` 时改为返回最新版本，用于"查询窗口末端早于
        该状态首个版本"的读取（TempLAMA 的 window_end 常早于最后一年）。
        这种回退读的是 as_of 之后的版本，调用方必须用 version_after_target_time
        单独标注、单独统计，不得与正常 as-of 读取混算。
        """
        entries = [e for e in self.entries if e['state'] == state]
        visible = [e for e in entries if not as_of or e['time'] <= as_of]
        if visible:
            return max(visible, key=lambda e: (e['time'], e['id']))
        if allow_future_fallback and entries:
            known = [e for e in entries if e['time'] != 'unknown'] or entries
            return max(known, key=lambda e: (e['time'], e['id']))
        return None

    # -- trajectory reads (experiment 1: pair vs full trajectory) -----------
    def entry_slots(self, entry) -> int:
        """该版本在 decoder 端占用的 latent slot 数（文本条目记 0）。"""
        payload = entry['payload']
        return 0 if isinstance(payload, bytes) else int(payload.shape[0])

    def canonical_order(self, entries, current_id):
        """唯一读取布局：current 在前，其余按时间升序。

        训练（training.trajectory_bundle）与推理（read_state 的 pair/trajectory
        两条路径）都必须经过这里，否则 current 的位置、历史的顺序会各自漂移，
        多版本训练再好也会被训练-推理不一致吃掉。
        """
        current = [entry for entry in entries if entry['id'] == current_id]
        rest = sorted((entry for entry in entries if entry['id'] != current_id),
                      key=lambda entry: (entry['time'], entry['id']))
        return current + rest

    def assert_canonical(self, entries, current_id):
        """布局自检：current 必须在首位且唯一，历史必须时间升序。"""
        if not entries or entries[0]['id'] != current_id:
            raise AssertionError('canonical layout requires the current entry first')
        if sum(entry['id'] == current_id for entry in entries) != 1:
            raise AssertionError('canonical layout requires exactly one current entry')
        times = [entry['time'] for entry in entries[1:]]
        if times != sorted(times):
            raise AssertionError('historical entries must be in ascending time order')
        return entries

    def visible_versions(self, state, observed_until=None) -> List[Dict[str, Any]]:
        """该状态在 ``observed_until`` 之前**已经可见**的版本，按时间升序。

        ``observed_until=None`` → 无时间约束，返回全部版本（与 ``select_version``
        同口径：时间未知的版本只在无约束时可见）。
        """
        entries = [e for e in self.entries if e['state'] == state]
        if observed_until:
            entries = [e for e in entries if e['time'] <= observed_until]
        return sorted(entries, key=lambda e: (e['time'], e['id']))

    def select_trajectory(self, state, observed_until=None, max_slots: int = 256, *,
                          marker_slots: int = 2, max_versions: int = 8
                          ) -> List[Dict[str, Any]]:
        """返回该状态在 ``observed_until`` 之前的**完整演化轨迹**（时间升序）。

        与 ``select_version`` 的分工：
            select_version  只回答"当时的最新可见版本是哪一个"（1 条）；
            select_trajectory  把当时已经可见的**全部**版本交给 reader，由每个版本
            自带的 role/time marker 自己定位（"Current·128 + History k·16"）。

        可见性（这是本函数存在的理由）：``observed_until`` 是硬上界，
        ``allow_future_fallback`` 那套回退在这里**不存在**——晚于截止时刻的版本
        一律不返回，哪怕 store 里存着更晚的 current。

        预算：从**最新**的可见版本往回取（信息量最大），每条按
        ``entry_slots(entry) + marker_slots`` 记账，超出 ``max_slots`` 或达到
        ``max_versions`` 即停止，再把结果反转为时间升序。E3 上
        128 + 7×16 + 8×2 = 256，即 8 个版本正好吃满 256-slot 预算。

        条件化 codec（D 变体）被显式拒绝：它的历史 code 要用**最终 current** 当
        decode 的 anchor，而最终 current 相对 ``observed_until`` 就是未来信息。
        """
        if max_slots < 1 or marker_slots < 0:
            raise ValueError('max_slots must be >= 1 and marker_slots >= 0')
        if max_versions < 1:
            raise ValueError('max_versions must be >= 1')
        if getattr(self.codec, 'conditioned', False) and observed_until is not None:
            raise ValueError(
                'Conditioned codecs decode historical codes against the final current '
                'memory, which is future information relative to observed_until; '
                'cutoff-bounded trajectory reads need an independent codec (E3)')
        visible = self.visible_versions(state, observed_until)
        chosen, used = [], 0
        for entry in reversed(visible):
            if len(chosen) >= max_versions:
                break
            cost = self.entry_slots(entry) + marker_slots
            if used + cost > max_slots:
                break
            chosen.append(entry)
            used += cost
        chosen.reverse()                                     # 时间升序
        return self.canonical_order(chosen, chosen[-1]['id'])

    def trajectory_slots(self, entries, marker_slots: int = 2) -> int:
        """轨迹在 decoder 端的总 slot 数（含 role/time marker）。"""
        return sum(self.entry_slots(entry) + marker_slots for entry in entries)

    def hidden_future_count(self, state, observed_until) -> int:
        """该状态里晚于 ``observed_until``、因而被读取路径排除的版本数。"""
        if not observed_until:
            return 0
        return sum(1 for e in self.entries if e['state'] == state and e['time'] > observed_until)

    def bundle(self, selected, markers=None, slot_limit=None,
               current_ids: Optional[Collection[str]] = None):
        """把选中版本解码后拼成 decoder memory。

        ``current_ids``：哪些条目要打 'current' role。给了就用它（轨迹读取时
        "当时可见的最新版本"不是 store 的最终 current，必须显式指定）；不给则退回
        旧语义（``self.current(state)`` 即 current）。
        """
        pieces = []
        for entry in selected:
            memory = self.decode(entry)
            if markers is not None:
                if current_ids is None:
                    role = 'current' if entry is self.current(entry['state']) else 'historical'
                else:
                    role = 'current' if entry['id'] in current_ids else 'historical'
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
            # key/payload 的 FP16 线格式只取决于不可变数组内容；entry dict 可能被
            # 事务替换，但数组本身从不原地改写，所以按 id 缓存 tobytes 是安全的。
            # LongMemEval 规模下这一步原本占写入耗时的大头（数百 MB/条的拷贝）。
            key_len = entry['key'].astype('<f2').nbytes
            payload_len = entry['payload'].astype('<f2').nbytes
            cached = self._wire_cache.get(entry['id']) if hasattr(self, '_wire_cache') else None
            if cached is None or cached[1] != (key_len, payload_len):
                # wire 缓存存拼好的 key+payload：serialize 热路径上零拷贝。
                # 数组内容从不被原地改写（UPDATE 是整体替换并失效缓存），按 id 缓存安全。
                self._wire_cache[entry['id']] = cached = (
                    entry['key'].astype('<f2').tobytes() + entry['payload'].astype('<f2').tobytes(),
                    (key_len, payload_len))
            wire = cached[0]
            description = {k: entry[k] for k in ('id', 'state', 'time', 'kind', 'keyframe', 'anchor_id', 'compression_count')}
            description.update(offset=offset, key_bytes=key_len, payload_bytes=payload_len, shape=list(entry['payload'].shape))
            descriptions.append(description)
            payloads.append(wire)
            offset += len(wire)
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
        # 缓存 serialize() 的总字节数，写入/驱逐/恢复后失效。O(n²) 根因修复：
        # _receipt 每条事实都调 accounting()，不缓存时 LongMemEval 级别的存储器
        # （数百 MB）每写一条就完整序列化一遍，写入退化为小时级纯 CPU 空转。
        total = getattr(self, '_accounting_total', None)
        if total is None:
            # accounting 只需要**长度**：构造与 serialize() 逐字节一致的 header
            # （offset 由 wire 长度累计得到，零 payload 拷贝），再加 Σ len(wire)。
            # 之前"缓存 miss 后全量 serialize"每写一条仍是数百 MB 的 join ——
            # 那是 O(n²) 的最后残余。
            descriptions, offset = [], 0
            for entry in self.entries:
                cached = self._wire_cache.get(entry['id'])
                # 与 serialize() 同一自愈口径：缓存里的 (key_bytes, payload_bytes) 与
                # 当前 entry 不符（例如 UPDATE 把 full 降级成 m-slot code 后 payload
                # 被整体替换）时必须重建，否则字节账虚高、与文件长度对不上。
                expected = (entry['key'].astype('<f2').nbytes,
                            entry['payload'].astype('<f2').nbytes)
                if cached is None or cached[1] != expected:
                    cached = (entry['key'].astype('<f2').tobytes()
                              + entry['payload'].astype('<f2').tobytes(), expected)
                    self._wire_cache[entry['id']] = cached
                description = {k: entry[k] for k in ('id', 'state', 'time', 'kind', 'keyframe',
                                                     'anchor_id', 'compression_count')}
                description.update(offset=offset, key_bytes=cached[1][0],
                                   payload_bytes=cached[1][1],
                                   shape=list(entry['payload'].shape))
                descriptions.append(description)
                offset += len(cached[0])
            header = json.dumps({'format': self.format_name, 'entries': descriptions,
                                 'blocks': self.blocks, 'counts': self.counts,
                                 'interval': self.interval,
                                 'conditioned': self.codec.conditioned,
                                 'dimension': self.codec.dimension,
                                 'codec_sha256': self.codec_signature,
                                 'codec_slots': self.codec.slots,
                                 'key_dimensions': self.key_dimensions},
                                sort_keys=True, separators=(',', ':')).encode()
            payload_bytes = sum(len(self._wire_cache[entry['id']][0]) for entry in self.entries)
            aux_header = json.dumps({'format': self.format_name, 'budget_mode': self.budget_mode,
                                     'pending': [], 'duplicates': [], 'rejections': []},
                                    sort_keys=True, separators=(',', ':')).encode() if hasattr(
                self, 'aux_magic') and self.pending is not None else b''
            aux_len = (len(self.aux_magic) + 8 + len(aux_header)
                       if aux_header else 0)
            # aux 段非空时 header 内容不同，直接按真实 aux 构造 —— pending/duplicates
            # 通常极少，成本可忽略。
            aux_len, pending_bytes = 0, 0
            if hasattr(self, 'aux_magic'):
                aux_descriptions, aux_payloads, aux_offset = [], [], 0
                for record in self.pending:
                    key = record['key'].astype('<f2').tobytes()
                    payload = record['payload'].astype('<f2').tobytes()
                    aux_descriptions.append(
                        {k: record[k] for k in ('id', 'state', 'time', 'confidence', 'reason')} |
                        {'offset': aux_offset, 'key_bytes': len(key),
                         'payload_bytes': len(payload), 'shape': list(record['payload'].shape)})
                    aux_payloads.append(key + payload)
                    aux_offset += len(key) + len(payload)
                pending_bytes = aux_offset
                aux_header = json.dumps({'format': self.format_name,
                                         'budget_mode': self.budget_mode,
                                         'pending': aux_descriptions,
                                         'duplicates': [{k: r[k] for k in ('id', 'state', 'time', 'matched_id')}
                                                        for r in self.duplicates],
                                         'rejections': [{k: r[k] for k in ('id', 'state', 'time', 'reason')}
                                                        for r in self.rejections]},
                                        sort_keys=True, separators=(',', ':')).encode()
                aux_len = len(self.aux_magic) + 8 + len(aux_header) + sum(map(len, aux_payloads))
            total = len(self.magic) + 8 + len(header) + payload_bytes + aux_len
            self._accounting_total = total
            del pending_bytes
        payload = sum(e['payload'].nbytes for e in self.entries)
        keys = sum(e['key'].nbytes for e in self.entries)
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
        self._accounting_total = None           # build 已整体替换 entries
        try:
            self.validate_protocol()
            if self.resident_slots() > self.slot_budget or (self.byte_budget is not None and len(self.serialize()) > self.byte_budget):
                raise ValueError('Snapshot budget cannot retain all versions; no eviction in main comparison')
        except Exception:
            self.entries, self.blocks, self.counts = [], {}, {}
            self._accounting_total = None
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


INCREMENTAL_FORMAT = 'incremental_independent_current_v1'
INCREMENTAL_BUDGET_MODES = ('retain_all', 'drop_oldest')


class IncrementalCurrentStore(KeyframeStore):
    """Online version store for the frozen independent E3 codec.

    One state = one fact group identity resolved upstream (StateIndex/StateRouter).
    Every state keeps exactly one full current memory (128 slots); older versions
    are compressed exactly once into a 16-slot code by the frozen E3 codec and are
    never used as encoder inputs again. Writes are driven by a resolved
    StateDecision, so this store never reads any oracle state field:

        CREATE     store the incoming full memory as the new current version
        UPDATE     compress the previous current into a single 16-slot code, then
                   store the incoming full memory as the new current version
        DUPLICATE  keep the payload untouched, record the alias only
        DEFER      keep an independent pending record; the live current version is
                   NOT downgraded because the conflict is unresolved

    Budget protocol: the retired `resident_slot_budget = 2048` keyframe ceiling does
    not apply here (4057 fact groups x 128 slots = 519,296 resident slots), so this
    class implements two explicit, pre-declared tiers that both report bytes and
    slots through `accounting()`:

        budget_mode='retain_all'（默认，主对照口径）
            永不淘汰。`byte_budget` 是硬护栏：超预算的写入显式拒绝并整体回滚，
            绝不静默丢弃活跃状态。
        budget_mode='drop_oldest'（相同持久化字节预算的受控对照口径）
            超预算时按**预先写明的确定性顺序**淘汰，而不是拒绝：
              1. 先清非版本账：pending → duplicates → rejections（清空即记 audit）；
              2. 再淘汰历史 code：全局按 (entry['time'], entry['id']) 升序逐条淘汰；
              3. 仍超预算 → 显式 ValueError（整笔写入回滚），绝不假装成功。
            **current（full）版本永不淘汰**：它是该状态唯一的 128-slot 当前记忆，
            淘汰它等于丢数据；因此单状态 current + 格式开销装不下时只能拒绝。
            每次淘汰都记进 `self.evictions`（淘汰了哪些 id、释放多少字节），
            并通过写入回执的 `evicted` 字段与 `accounting()['evictions']` 暴露；
            audit 只留在内存里，不占持久化字节。

    字节构成与测量方法见 docs/store_budget.md（公式 + probe/measure_store_bytes.py）。
    """
    magic = b'VCRMIC1\n'
    format_name = INCREMENTAL_FORMAT
    aux_magic = b'VCRMIP1\n'

    def __init__(self, codec, key_dimensions=128, byte_budget=None, budget_mode='retain_all'):
        if budget_mode not in INCREMENTAL_BUDGET_MODES:
            raise ValueError(f'Unsupported incremental store budget mode {budget_mode!r}')
        if budget_mode == 'drop_oldest' and byte_budget is None:
            # 淘汰策略必须有明确预算，否则"drop_oldest"名不副实（永不触发）。
            raise ValueError('budget_mode=drop_oldest requires an explicit byte_budget')
        if codec.conditioned:
            raise ValueError('Incremental E3 store requires independent codec')
        # The parent constructor insists on a positive slot ceiling; retain_all
        # clears it right afterwards because slot accounting is informational here.
        super().__init__(codec, 1, 2, key_dimensions, byte_budget)
        self.slot_budget = None
        self.budget_mode = budget_mode
        self.pending, self.duplicates, self.rejections = [], [], []
        # 淘汰审计（仅 drop_oldest 会写）。刻意不进序列化：审计是诊断信息，
        # 让它占持久化字节会反过来推动淘汰，形成自我放大的预算反馈环。
        self.evictions = []
        # Single-slot undo for StateIndex.commit_with(): only the most recent commit
        # can still be rolled back, so this never grows with the number of writes
        # (a full snapshot includes the pending 128x4096 FP16 payloads). Never serialized.
        self._commit_undo = {}

    # -- write path -------------------------------------------------------
    def ingest(self, *args, **kwargs):
        raise RuntimeError('Incremental E3 store writes only through ingest_decision(); '
                           'keyframe ingest would rebase versions without a resolved decision')

    def ingest_decision(self, event, memory, semantic_key, decision):
        from .fact_schema import STATE_ACTIONS
        action, state = decision.action, decision.state_id
        if action not in STATE_ACTIONS:
            raise ValueError(f'Unknown state action {action!r}')
        if not state:
            raise ValueError('Incremental store writes need an explicit resolved state id')
        if action in ('UPDATE', 'DUPLICATE') and self.current(state) is None:
            raise ValueError(f'{action} requires an existing current version')
        if action == 'CREATE' and self.current(state) is not None:
            raise ValueError('CREATE requires a state without a live current version')
        if self._known_id(event['id']):
            raise ValueError('Duplicate version id')
        previous = self.current(state)
        if previous and event['time'] < previous['time']:
            # A refusal is a diagnostic record, not an id reservation: the same event
            # re-submitted in chronological order must still be accepted (see _known_id).
            self.rejections.append({'id': event['id'], 'state': state, 'time': event['time'],
                                    'reason': 'out_of_order'})
            raise ValueError('Incremental versions must be written in chronological order')
        wire = self._full_wire(memory)
        key = self._key_wire(semantic_key)
        # Every action goes through the same transaction: mutate -> validate -> budget
        # guard -> commit, and roll the whole write back on any failure. No branch may
        # return before the guard (DEFER grows by a full 128x4096 FP16 record, DUPLICATE
        # appends metadata), otherwise the store would hold state it cannot persist.
        snapshot = self._snapshot()
        evict_audit_start = len(self.evictions)
        downgraded = None
        evicted = []
        try:
            if action == 'DUPLICATE':
                # Payload is deliberately untouched: the alias is metadata only.
                self.duplicates.append({'id': event['id'], 'state': state, 'time': event['time'],
                                        'matched_id': previous['id']})
            elif action == 'DEFER':
                # An unresolved conflict must not downgrade the live current version.
                self.pending.append({'id': event['id'], 'state': state, 'time': event['time'],
                                     'confidence': float(decision.confidence), 'reason': decision.reason,
                                     'key': key, 'payload': wire})
            else:
                if action == 'UPDATE':
                    # Compress a temporary copy of the live current, validate it, then commit.
                    downgraded = previous['id']
                    self._wire_cache.pop(previous['id'], None)   # payload 将被替换
                    if previous['kind'] != 'full' or previous['compression_count'] or previous['keyframe'] is False:
                        raise AssertionError('Only a live full current version can be compressed')
                    with torch.no_grad():
                        code = self.codec.encode(self._memory(previous), None).detach().cpu().float()
                    code_wire = code.numpy().astype('<f2')
                    if code.shape != (self.codec.slots, self.codec.dimension) or not np.isfinite(code_wire).all():
                        raise ValueError('Codec returned invalid or FP16-overflowing historical code')
                entry = {'id': event['id'], 'state': state, 'time': event['time'], 'kind': 'full',
                         'keyframe': True, 'anchor_id': None, 'compression_count': 0,
                         'key': key.copy(), 'payload': wire, '_validated': True}
                if action == 'UPDATE':
                    previous.update(kind='code', keyframe=False, anchor_id=None,
                                    compression_count=1, payload=code_wire)
                self.entries.append(entry)
                self.blocks[state] = entry['id']
                self.counts[state] = self.counts.get(state, 0) + 1
            self.validate_protocol()
            evicted = self._guard_budget()
        except Exception:
            self._restore(snapshot)
            del self.evictions[evict_audit_start:]   # 回滚后淘汰审计也要撤销
            raise
        self._accounting_total = None          # 事务已变更存储内容
        return self._receipt(action, event, state, downgraded, evicted)

    # -- StateIndex transaction adapter (t5 contract) ----------------------
    def commit(self, record=None, fact=None, *, memory=None, semantic_key=None, decision=None):
        """`StateIndex.commit_with(store, ...)` 的薄适配层；写入语义完全交给 ingest_decision()。

        字段映射（record/fact → ingest_decision 的 event / 参数）：
            record.state_id      → decision.state_id（状态身份来自解析层；event 里没有 state_key）
            fact.id              → event['id']，同时就是返回的版本号 version_id
            fact.observed_at     → event['time']（ISO8601，用于版本排序与 as_of 可见性）
            record.semantic_key  → semantic_key（128 维；显式传参时以参数为准）
            memory=              → 该事实的原始 full memory。**UPDATE 必须显式传入**（旧 current
                                   要用它之外的新版本来替换，缺了无法写入），DEFER 同样必须传入
                                   （pending 记录要保留原始 full memory）；DUPLICATE 不改 payload，
                                   但当前实现仍统一要求 memory= 以便做同一次 full/FP16 校验。
                                   本适配层绝不从 fact 文本重建 latent，也不会把 m-slot code
                                   当 full memory 写入；缺失时直接 ValueError，绝不静默降级。
        缺省动作：`record.current_version_id` 为空 → CREATE，否则 UPDATE；调用方传入
        `decision=`（UpdateResolver 的真实决策）时一律以它为准。

        返回 `{'version_id': <非空 str>, 'action': <str>, ...receipt}`，同时满足
        StateIndex._coerce_version_id 接受的 `str` / `{'version_id': ...}` 两种形态。
        写入失败时 ingest_decision 已自行回滚并抛异常，由 commit_with 转成 StoreCommitError；
        成功时的快照只保留在内存里供 `rollback(result)` 撤销，不进入序列化字节。
        """
        from .fact_schema import StateDecision
        if record is None or fact is None:
            raise ValueError('commit() needs record= and fact= (StateIndex.commit_with contract)')
        state_id = getattr(record, 'state_id', None)
        version_id = getattr(fact, 'id', None)
        observed_at = getattr(fact, 'observed_at', None)
        if not state_id:
            raise ValueError('commit() requires record.state_id')
        if not version_id:
            raise ValueError('commit() requires fact.id as the version id')
        if not observed_at:
            raise ValueError('commit() requires fact.observed_at as the event time')
        if memory is None:
            raise ValueError('commit() requires the original full memory= for this fact')
        if semantic_key is None:
            semantic_key = getattr(record, 'semantic_key', None)
        if semantic_key is None:
            raise ValueError('commit() requires semantic_key= or record.semantic_key')
        if decision is None:
            action = 'CREATE' if not getattr(record, 'current_version_id', None) else 'UPDATE'
            decision = StateDecision(action, state_id, 1., 'commit adapter default')
        if decision.state_id != state_id:
            raise ValueError('decision.state_id must match record.state_id')
        event = {'id': version_id, 'time': str(observed_at)}
        snapshot = self._snapshot()
        receipt = self.ingest_decision(event, memory, semantic_key, decision)
        # Keep exactly one undo slot: the previous commit's snapshot is dropped here,
        # so N commits hold one snapshot, not N.
        self._commit_undo.clear()
        self._commit_undo[version_id] = snapshot
        return {'version_id': version_id, **receipt}

    def rollback(self, result):
        """撤销一次已成功的 commit()（StateIndex 索引更新失败时的尽力回滚钩子）。

        result 可以是 commit() 的返回字典，也可以直接是版本号；只保留最近一次 commit 的快照
        （StateIndex 的索引失败紧跟在同一次 commit 之后），其它版本号返回 False。
        """
        version_id = result.get('version_id') if isinstance(result, dict) else result
        snapshot = self._commit_undo.pop(version_id, None) if isinstance(version_id, str) else None
        if snapshot is None:
            return False
        self._restore(snapshot)
        return True

    # -- write helpers ----------------------------------------------------
    def _known_id(self, version_id):
        """已入库/已登记的身份；rejections 刻意不算占用。

        拒绝记录只是"这次写入被拒"的诊断痕迹（与 validate_protocol 的唯一性集合一致：
        entries + pending + duplicates）。把被拒 id 也算成占用会让同一条事件按正确顺序
        重投时永远拿到 'Duplicate version id'，等于静默丢数据。
        """
        return (any(e['id'] == version_id for e in self.entries) or
                any(r['id'] == version_id for r in self.pending) or
                any(r['id'] == version_id for r in self.duplicates))

    def _full_wire(self, memory):
        if memory.ndim != 2 or memory.shape[1] != self.codec.dimension or \
                memory.shape[0] <= self.codec.slots:
            raise ValueError('Write only original full memory, never an m-slot code')
        wire = memory.detach().cpu().float().numpy().astype('<f2')
        if not np.isfinite(wire).all():
            # 一次扫描覆盖 NaN 与 FP16 上溢（后者表现为 ±inf），删去转换前的
            # torch.isfinite 重复全量扫描。
            raise ValueError('Memory cannot be represented as finite FP16')
        return wire

    def _key_wire(self, semantic_key):
        key = np.asarray(semantic_key, dtype='<f2')
        if key.shape != (self.key_dimensions,) or not np.isfinite(key).all():
            raise ValueError('Invalid semantic key')
        return key

    def _snapshot(self):
        """事务回滚快照（浅拷贝）。

        只拷元数据，大数组（key/payload）按引用共享。这是安全的：事务内对
        entry 的改动只有**整体替换**字段（UPDATE 时 previous.update(...) 与
        新 entry 的构造都是替换标量/数组字段，从不原地改写数组内容），回滚
        用旧 dict 覆盖回去即可，共享数组不受影响。深拷贝版在 LongMemEval
        规模（数百 MB 存储 × 每题数百次写入）是 O(n²) 的第二个热点。
        """
        self._accounting_total = None
        return ([dict(e) for e in self.entries], dict(self.blocks), dict(self.counts),
                [dict(r) for r in self.pending], [dict(r) for r in self.duplicates],
                [dict(r) for r in self.rejections])

    def _restore(self, snapshot):
        self._accounting_total = None          # 回滚 = 内容变更
        self.entries, self.blocks, self.counts, self.pending, self.duplicates, self.rejections = snapshot

    def _guard_budget(self):
        """预算守卫：retain_all 拒绝、drop_oldest 按固定顺序淘汰；返回淘汰审计。"""
        if self.byte_budget is None:
            return []
        total = len(self.serialize())   # 预算守卫必须用真实序列化值，绝不吃缓存
        if total <= self.byte_budget:
            return []
        if self.budget_mode != 'drop_oldest':
            raise ValueError(f'Incremental store byte budget exceeded ({total} > {self.byte_budget}); '
                             'explicit rejection instead of silent eviction')
        return self._drop_oldest(total)

    def _drop_oldest(self, total):
        """drop_oldest 的确定性淘汰：非版本账 → 最旧历史 code → 仍超预算则拒绝。

        淘汰顺序只依赖 (entry['time'], entry['id'])，与写入顺序、dict 序无关，
        因此同一份写入序列在任何进程里都会淘汰同一批版本（可复现的对照口径）。
        每一步都重新量真实序列化字节（淘汰极少发生，精确优先于增量估算）。
        """
        groups, freed = [], 0
        # 1) 非版本账：pending/duplicates/rejections 不影响 current 不变量，先清。
        for name in ('pending', 'duplicates', 'rejections'):
            records = getattr(self, name)
            if not records:
                continue
            before = len(self.serialize())
            ids = [record['id'] for record in records]
            setattr(self, name, [])
            self._accounting_total = None
            freed += before - len(self.serialize())
            groups.append({'group': name, 'ids': ids,
                           'reason': 'budget_drop_non_version_record'})
            if len(self.serialize()) <= self.byte_budget:
                return self._record_evictions(groups, freed)
        # 2) 历史 code：全局按 (time, id) 升序淘汰；current(full) 永不参与。
        victims = sorted((e for e in self.entries if e['kind'] == 'code'),
                         key=lambda e: (e['time'], e['id']))
        for victim in victims:
            if len(self.serialize()) <= self.byte_budget:
                break
            before = len(self.serialize())
            self.entries = [e for e in self.entries if e['id'] != victim['id']]
            self._wire_cache.pop(victim['id'], None)
            self._accounting_total = None
            freed += before - len(self.serialize())
            groups.append({'group': 'historical_code', 'ids': [victim['id']],
                           'state': victim['state'], 'time': victim['time'],
                           'reason': 'budget_drop_oldest_code'})
        # 3) 淘汰后必须重新过不变量与预算；仍超预算 → 显式拒绝（整笔写入回滚）。
        self.validate_protocol()
        total = len(self.serialize())
        if total > self.byte_budget:
            raise ValueError(f'Incremental store byte budget exceeded even after evicting every '
                             f'historical code ({total} > {self.byte_budget}); the live full current '
                             'version can never be evicted, so this write is rejected')
        return self._record_evictions(groups, freed)

    def _record_evictions(self, groups, freed):
        if not groups:
            return []
        record = {'seq': len(self.evictions), 'byte_budget': self.byte_budget,
                  'freed_bytes': freed, 'groups': groups}
        self.evictions.append(record)
        self._accounting_total = None
        return [record]

    def _enforce_budget(self):
        raise RuntimeError(
            'Incremental store evicts only through the declared policy: retain_all rejects the '
            'write, drop_oldest evicts via _drop_oldest(); the keyframe FIFO never applies')

    def _receipt(self, action, event, state, downgraded, evicted=()):
        return {'action': action, 'id': event['id'], 'state': state, 'time': event['time'],
                'downgraded': downgraded, 'evicted': list(evicted), **self.accounting()}

    # -- invariants -------------------------------------------------------
    def validate_protocol(self):
        if self.codec.conditioned:
            raise ValueError('Incremental E3 store requires independent codec')
        ids = [e['id'] for e in self.entries] + [r['id'] for r in self.pending] + \
              [r['id'] for r in self.duplicates]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate incremental version or pending id')
        entry_ids = {e['id'] for e in self.entries}      # 复用，勿在循环里重建
        by_state = {}                                    # 一次遍历建索引，替代 live×n 扫描
        for e in self.entries:
            by_state.setdefault(e['state'], []).append(e)
        if any(r['matched_id'] not in entry_ids for r in self.duplicates):
            raise ValueError('Duplicate record must reference a stored version')
        for state, versions in by_state.items():
            current = versions[-1]          # entries 按写入序追加，最新版本在尾部
            full = [e for e in versions if e['kind'] == 'full']
            if (full != [current] or not current['keyframe'] or current['compression_count'] or
                    current['anchor_id'] is not None or self.blocks.get(state) != current['id']):
                raise ValueError('Incremental store needs exactly one full current per state')
            if versions[-1] is not current:
                raise ValueError('The newest version of a state must be its full current')
            times = [e['time'] for e in versions]
            if times != sorted(times):
                raise ValueError('Incremental versions must stay in chronological order')
            for entry in versions:
                # 每次写入都对全部历史 payload 重新做 isfinite 全量扫描是 O(n²) 字节级
                # 开销（LongMemEval 规模下每条 1MB × 上千条）。写入路径已经保证有限性
                # （_full_wire 与 codec 输出校验），这里只复验**本事务新写**的 payload。
                if not (entry.get('_validated') or np.isfinite(entry['payload']).all()):
                    raise ValueError('Non-finite incremental latent payload')
                if entry['kind'] == 'code':
                    if (entry['keyframe'] or entry['compression_count'] != 1 or entry['anchor_id'] is not None or
                            entry['payload'].shape != (self.codec.slots, self.codec.dimension) or
                            entry['time'] > current['time']):
                        raise ValueError('Incremental codes are single-compression m-slot payloads of one current')
                elif entry['kind'] != 'full':
                    raise ValueError('Incremental store holds only full currents and compressed codes')
                elif entry['payload'].shape[1] != self.codec.dimension or entry['payload'].shape[0] <= self.codec.slots:
                    raise ValueError('Incremental current version must hold the original full memory')
        for record in self.pending:
            if (record['payload'].ndim != 2 or record['payload'].shape[1] != self.codec.dimension or
                    record['payload'].shape[0] <= self.codec.slots or
                    not np.isfinite(record['payload']).all() or
                    record['key'].shape != (self.key_dimensions,)):
                raise ValueError('Pending record must keep an independent original full memory')

    def accounting(self):
        return super().accounting() | {
            'budget_mode': self.budget_mode,
            'evictions': len(self.evictions),
            'evicted_ids': [version_id for record in self.evictions
                            for group in record['groups'] for version_id in group['ids']],
            'evicted_bytes': sum(record['freed_bytes'] for record in self.evictions)}

    # -- persistence ------------------------------------------------------
    def serialize(self):
        base = super().serialize()
        descriptions, payloads, offset = [], [], 0
        for record in self.pending:
            key = record['key'].astype('<f2').tobytes()
            payload = record['payload'].astype('<f2').tobytes()
            descriptions.append({k: record[k] for k in ('id', 'state', 'time', 'confidence', 'reason')} |
                                {'offset': offset, 'key_bytes': len(key), 'payload_bytes': len(payload),
                                 'shape': list(record['payload'].shape)})
            payloads.append(key + payload)
            offset += len(key) + len(payload)
        header = json.dumps({'format': self.format_name, 'budget_mode': self.budget_mode,
                             'pending': descriptions,
                             'duplicates': [{k: r[k] for k in ('id', 'state', 'time', 'matched_id')}
                                            for r in self.duplicates],
                             'rejections': [{k: r[k] for k in ('id', 'state', 'time', 'reason')}
                                            for r in self.rejections]},
                            sort_keys=True, separators=(',', ':')).encode()
        return base + self.aux_magic + struct.pack('<Q', len(header)) + header + b''.join(payloads)

    def save(self, path):
        data = self.serialize()
        if self.byte_budget is not None and len(data) > self.byte_budget:
            raise AssertionError('Incremental byte budget exceeded')
        atomic_bytes(path, data)
        return len(data)

    @classmethod
    def load(cls, path, codec, key_dimensions=128, byte_budget=None, budget_mode='retain_all'):
        data = Path(path).read_bytes()
        if not data.startswith(cls.magic):
            raise ValueError('Invalid incremental memory format')
        size = struct.unpack('<Q', data[8:16])[0]
        header = json.loads(data[16:16 + size])
        if header['format'] != cls.format_name:
            raise ValueError('Memory protocol/format mismatch')
        if (header['dimension'], header['codec_slots'], header['conditioned']) != (
                codec.dimension, codec.slots, codec.conditioned):
            raise ValueError('Incremental codec identity mismatch')
        store = cls(codec, header['key_dimensions'], byte_budget, budget_mode)
        if header['codec_sha256'] != store.codec_signature:
            raise ValueError('Incremental codec weights identity mismatch')
        store.blocks, store.counts = header['blocks'], header['counts']
        entries_end = 16 + size + sum(d['key_bytes'] + d['payload_bytes'] for d in header['entries'])
        raw = data[16 + size:entries_end]
        for desc in header['entries']:
            start = desc['offset']
            end = start + desc['key_bytes']
            payload = np.frombuffer(raw[end:end + desc['payload_bytes']], dtype='<f2').reshape(desc['shape']).copy()
            store.entries.append({k: desc[k] for k in ('id', 'state', 'time', 'kind', 'keyframe',
                                                       'anchor_id', 'compression_count')} |
                                 {'key': np.frombuffer(raw[start:end], dtype='<f2').copy(), 'payload': payload})
        tail = data[entries_end:]
        if not tail.startswith(cls.aux_magic):
            raise ValueError('Incremental memory is missing its pending/duplicate block')
        begin = len(cls.aux_magic) + 8
        aux_size = struct.unpack('<Q', tail[len(cls.aux_magic):begin])[0]
        aux = json.loads(tail[begin:begin + aux_size])
        if aux.get('budget_mode', 'retain_all') != budget_mode:
            # 预算模式是读取口径的一部分：显式不匹配，绝不静默换模式。
            raise ValueError(f'Incremental budget mode mismatch: file={aux.get("budget_mode")!r} '
                             f'caller={budget_mode!r}; load with the mode the store was written in')
        aux_raw = tail[begin + aux_size:]
        for desc in aux['pending']:
            start = desc['offset']
            end = start + desc['key_bytes']
            payload = np.frombuffer(aux_raw[end:end + desc['payload_bytes']], dtype='<f2').reshape(desc['shape']).copy()
            store.pending.append({k: desc[k] for k in ('id', 'state', 'time', 'confidence', 'reason')} |
                                 {'key': np.frombuffer(aux_raw[start:end], dtype='<f2').copy(), 'payload': payload})
        store.duplicates = [{k: r[k] for k in ('id', 'state', 'time', 'matched_id')} for r in aux['duplicates']]
        store.rejections = [{k: r[k] for k in ('id', 'state', 'time', 'reason')} for r in aux['rejections']]
        if byte_budget is not None and len(data) > byte_budget:
            raise ValueError('Loaded incremental memory exceeds byte budget')
        store.validate_protocol()
        return store

    def accounting(self):
        pending_slots = sum(int(r['payload'].shape[0]) for r in self.pending)
        return super().accounting() | {
            'protocol': self.format_name, 'budget_mode': self.budget_mode,
            # 淘汰审计（仅 drop_oldest 非空）：条数、被淘汰的版本 id、累计释放字节
            'evictions': len(self.evictions),
            'evicted_ids': [version_id for record in self.evictions
                            for group in record['groups'] for version_id in group['ids']],
            'evicted_bytes': sum(record['freed_bytes'] for record in self.evictions),
            'resident_slot_budget_applied': False,
            'current_versions': sum(1 for e in self.entries if e['kind'] == 'full'),
            'compressed_versions': sum(1 for e in self.entries if e['kind'] == 'code'),
            'pending_records': len(self.pending), 'duplicate_records': len(self.duplicates),
            'rejection_records': len(self.rejections), 'pending_latent_slots': pending_slots,
            'resident_latent_slots': self.resident_slots() + pending_slots}
