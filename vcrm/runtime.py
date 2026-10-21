"""Opt-in trained ICAE C/D runtime: write spans, persist, and read temporal entries."""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from .codec import SlotCodec, VersionMarkers
from .data_views import (
    PUBLIC_QUERY_KEYS,
    PUBLIC_WRITE_KEYS,
    TIME_MODES,
    assert_no_oracle,
    public_query,
)
from .fact_extractor import (
    UNKNOWN_TIME,
    UNRESOLVED_ATTRIBUTE,
    UNRESOLVED_SUBJECT,
    FactExtractor,
)
from .fact_schema import AtomicFact, StateDecision
from .index import SemanticIndex, inferred_as_of
from .io import file_digest, read_json
from .state_index import IndexCommitError, StateIndex, value_fingerprint
from .state_router import StateRouter, cosine_similarity
from .store import CurrentSnapshotStore, IncrementalCurrentStore, KeyframeStore
from .streaming import SpanWriter
from .update_resolver import UpdateResolver


class SlotMemoryRuntime:
    def __init__(self, reader, codec, markers, index, store, validated_index=False, update_resolver=None):
        self.reader, self.codec, self.markers, self.index, self.store = reader, codec, markers, index, store
        self.validated_index = validated_index
        self.writer = SpanWriter(reader, index, store, reader.cfg['state_similarity'], update_resolver, validated_index)

    def save(self, path):
        self.store.save(path)

    @classmethod
    def from_checkpoint(cls, config, training_root, store_path=None, update_resolver=None):
        from safetensors.torch import load_file
        from .reader import FrozenReader
        root = Path(training_root)
        metadata = read_json(root / 'codec_training.json')
        protocol = metadata.get('memory_protocol', 'immutable_keyframe_v1')
        if config.get('memory_protocol', 'immutable_keyframe_v1') != protocol:
            raise ValueError('Training/evaluation memory protocol mismatch')
        checkpoint = root / 'codec.safetensors'
        if not metadata['completed'] or metadata['checkpoint_sha256'] != file_digest(checkpoint):
            raise ValueError('Incomplete or altered codec checkpoint')
        if (metadata['base_checkpoint_sha256'] != file_digest(config['checkpoint']) or
                metadata['model_revision'] != config['model_revision'] or
                metadata['reader_sha256'] != file_digest(Path(__file__).with_name('reader.py'))):
            raise ValueError('Codec training used a different ICAE reader')
        adapted_hash = file_digest(config['adapted_checkpoint']) if config.get('adapted_checkpoint') else None
        if adapted_hash != metadata['adapted_checkpoint_sha256']:
            raise ValueError('Codec adaptation identity mismatch')
        reader = FrozenReader(config)
        codec = SlotCodec(reader.shape[1], metadata['history_slots'], metadata['codec_width'],
                          metadata['codec_heads'], metadata['method'] == 'D',
                          metadata.get('anchor_dropout', 0.), metadata.get('seed', 42),
                          metadata.get('codec_variant', 'residual'),
                          metadata.get('decode_conditioned'), metadata.get('codec_value_path'),
                          metadata.get('codec_activation_rms_cap')).to(reader.device)
        markers = VersionMarkers(reader.shape[1]).to(reader.device)
        index = SemanticIndex(reader.shape[1], config['key_dimensions']).to(reader.device)
        state = load_file(str(checkpoint))
        for prefix, module in (('codec', codec), ('markers', markers), ('index', index)):
            module.load_state_dict({k[len(prefix) + 1:]: v for k, v in state.items() if k.startswith(prefix + '.')}, strict=True)
            module.eval().requires_grad_(False)
        codec_kwargs = (codec, config['resident_slot_budget'])
        store_type = CurrentSnapshotStore if protocol == 'final_current_snapshot_v1' else KeyframeStore
        store = (store_type.load(store_path, *codec_kwargs) if store_path else
                 store_type(*codec_kwargs, metadata['keyframe_interval'], config['key_dimensions']))
        # A trained pooler still needs held-out calibration, especially query vs event keys.
        validated = False
        if config.get('routing_calibration'):
            calibration = read_json(config['routing_calibration'])
            if (not calibration.get('complete') or calibration.get('split') != 'validation' or
                    calibration.get('semantic_index_sha256') != metadata['semantic_index_sha256'] or
                    calibration.get('reader_sha256') != metadata['reader_sha256'] or
                    calibration.get('threshold') != config['state_similarity'] or
                    not calibration.get('query_event_matching_validated') or
                    not calibration.get('association_validated')):
                raise ValueError('Missing or mismatched held-out routing calibration')
            validated = True
        return cls(reader, codec, markers, index, store, validated, update_resolver)

    @torch.no_grad()
    def write_fact(self, event):
        if isinstance(self.store, CurrentSnapshotStore):
            raise RuntimeError('Offline snapshot does not support incremental writes')
        if not event.get('state_key'):
            raise ValueError('write_fact requires a shared explicit state key')
        memory = self.reader.encode_event(event)
        return self.store.ingest(event, memory, self.index.key(memory))

    def write_session(self, event):
        if isinstance(self.store, CurrentSnapshotStore):
            raise RuntimeError('Offline snapshot does not support streaming writes')
        return self.writer.write(event, self.reader.cfg.get('span_window_tokens', 256))

    @torch.no_grad()
    def answer(self, query):
        state = query.get('state_key')
        if not state:
            if not self.validated_index:
                raise ValueError('Query-to-memory semantic retrieval is not calibrated; supply shared state_key')
            key = self.index.key(self.reader.encode(self.reader.question(query)))
            candidate = self.index.associate(key, self.store.entries, self.reader.cfg['state_similarity'])
            if candidate is None:
                return self.reader.answer(query) | {'no_state_match': True}
            state = candidate['state_key']
        current = self.store.current(state)
        selected = self.store.select_version(state, inferred_as_of(query))
        if current is None or selected is None:
            return self.reader.answer(query) | {'version_missing': True}
        historical = selected if selected is not current else next(
            (e for e in reversed(self.store.entries) if e['state'] == state and e is not current), None)
        entries = [current] + ([historical] if historical is not None else [])
        if self.reader.device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        bundle = self.store.bundle(entries, self.markers, self.reader.cfg['read_latent_slots'])
        if bundle.device != self.reader.device:
            bundle = bundle.to(self.reader.device)
        if self.reader.device.type == 'cuda':
            torch.cuda.synchronize()
        recovery_seconds = time.perf_counter() - started
        result = self.reader.answer(query, bundle)
        dependencies = {e['anchor_id'] for e in entries if e['anchor_id']}
        selected_ids = {e['id'] for e in entries}
        return result | {'selected_entry_ids': [e['id'] for e in entries],
                         'role_time_slots': 2 * len(entries),
                         'codec_recovery_seconds': recovery_seconds,
                         'auxiliary_anchor_slots': sum(self.store._entry(eid)['payload'].shape[0]
                                                       for eid in dependencies - selected_ids),
                         'resident_accounting': self.store.accounting(),
                         'protocol': ('current plus historical m-slot read; offline final-current snapshot'
                                      if isinstance(self.store, CurrentSnapshotStore) else
                                      'current plus historical m-slot read; fixed-keyframe recovery')}


# ---------------------------------------------------------------------------
# P3: end-to-end Oracle-free runtime (Predicted path)
#
# 两个入口，任何一步都不读 state_key / key_text / entity / group / answers /
# latest / episode / as_of：
#
#   write_predicted(public_write_view)
#       FactExtractor.extract → reader.encode_event({time,text}) → index.key
#       → StateRouter.retrieve_candidates → UpdateResolver.resolve
#       → IncrementalCurrentStore.ingest_decision → StateIndex.commit / commit_with
#
#   answer_predicted(raw_query, window_end=...)
#       public_query(time_mode='window_end') → QueryParser(subject, attribute)
#       → router.retrieve → store.select_version(state, target_time)
#       → bundle + VersionMarkers → FrozenReader.answer
#
# 时间通道（t3 泄漏审计 + t13 冻结契约）：
#   * Predicted 正式口径只允许 time_mode='window_end'，构造时 time_mode='as_of'
#     必须显式 allow_as_of=True（仅受控对照/时间问答），否则直接 ValueError。
#   * window_end 未给出 → view['time'] is None / time_source='unknown' →
#     select_version(state, None) 即“无时间约束 = 最新可见版本”，**不补任何字段**。
#   * reader 提示词里也只允许出现 window_end（公开的观测窗口末刻），绝不注入
#     query['as_of'] / query['time']（TempLAMA 上二者就是金标版本时间戳）。
# ---------------------------------------------------------------------------

#: 问题文本里的掩码占位符（TempLAMA 用 `_X_`），解析前统一替换成中性词。
QUERY_MASK_TOKENS: Sequence[str] = ('_X_', '____', '___', '_ _')
QUERY_PLACEHOLDER = 'unknown'
QUERY_FACT_ID = 'query_view'

#: 决策日志的固定字段（契约）；日志里禁止出现任何 Oracle 字段。
DECISION_LOG_KEYS: Sequence[str] = (
    'event_id', 'fact_id', 'predicted_state_id', 'candidate_state_ids', 'decision',
    'confidence', 'old_version_id', 'new_version_id', 'compression_triggered',
    'reason', 'store_written',
)


def validate_decision_log(record: Mapping[str, Any], where: str = 'decision_log') -> Mapping[str, Any]:
    """决策日志契约校验：字段齐全 + 不含任何 Oracle 字段。"""
    missing = [key for key in DECISION_LOG_KEYS if key not in record]
    if missing:
        raise ValueError(f'{where}: missing decision-log fields {missing}')
    extra = [key for key in record if key not in DECISION_LOG_KEYS]
    if extra:
        raise ValueError(f'{where}: unexpected decision-log fields {extra}')
    assert_no_oracle(record, where)
    return record


class QueryParser:
    """从公开问题文本抽取 (subject, attribute)。

    只读 question 文本：把掩码占位符替换成中性词后用 FactExtractor 的同一套
    确定性规则解析，因此查询侧与写入侧共享同一份属性归一化，不引入第二套映射表。
    时间**不**从这里取：目标时间只来自 public_query 的 time 通道。
    """

    def __init__(self, extractor: Optional[FactExtractor] = None) -> None:
        self.extractor = extractor or FactExtractor()

    def parse(self, question: str) -> Dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError('QueryParser.parse expects a non-empty question str')
        text = question
        for token in QUERY_MASK_TOKENS:
            text = text.replace(token, QUERY_PLACEHOLDER)
        result = self.extractor.extract_detailed(
            {'id': QUERY_FACT_ID, 'text': text, 'time': UNKNOWN_TIME})
        for fact in result.facts:
            if fact.subject != UNRESOLVED_SUBJECT and fact.attribute != UNRESOLVED_ATTRIBUTE:
                return {'resolved': True, 'question': question,
                        'subject': fact.subject, 'attribute': fact.attribute, 'reason': ''}
        return {'resolved': False, 'question': question,
                'subject': UNRESOLVED_SUBJECT, 'attribute': UNRESOLVED_ATTRIBUTE,
                'reason': 'unresolved_identity'}


def _reader_cfg(reader: Any) -> Mapping[str, Any]:
    cfg = getattr(reader, 'cfg', None)
    return cfg if isinstance(cfg, Mapping) else {}


class PredictedRuntime:
    """Oracle-free write/answer runtime for the Predicted protocol.

    与 SlotMemoryRuntime 并存：后者（含 answer()）保持原样，继续作为 Oracle 对照
    与既有 ICAE 路径；本类只在 Predicted 路径使用，二者不可混用。
    """

    def __init__(self, reader: Any, codec: Any, markers: Any, index: Any, store: Any,
                 state_index: StateIndex, *, router: Optional[StateRouter] = None,
                 resolver: Optional[UpdateResolver] = None,
                 extractor: Optional[FactExtractor] = None,
                 query_parser: Optional[QueryParser] = None,
                 time_mode: str = 'window_end', allow_as_of: bool = False,
                 hybrid_fallback: bool = False, defer_store_memory: bool = False,
                 top_k: int = 5) -> None:
        if time_mode not in TIME_MODES:
            raise ValueError(f'time_mode must be one of {TIME_MODES}, got {time_mode!r}')
        if time_mode == 'as_of' and not allow_as_of:
            raise ValueError(
                "PredictedRuntime refuses time_mode='as_of' on the formal Predicted path: "
                "query['time'] == query['as_of'] == the gold version timestamp on TempLAMA, "
                'so passing it through would hand select_version the answer. Pass '
                'allow_as_of=True only for a controlled contrast / temporal-QA run.')
        if not isinstance(state_index, StateIndex):
            raise TypeError('PredictedRuntime needs a vcrm.state_index.StateIndex')
        if not hasattr(store, 'ingest_decision'):
            raise TypeError('PredictedRuntime needs an IncrementalCurrentStore-style store '
                            '(ingest_decision)')
        self.reader, self.codec, self.markers, self.index = reader, codec, markers, index
        self.store, self.state_index = store, state_index
        self.router = router or StateRouter(top_k=top_k)
        self.resolver = resolver or UpdateResolver()
        self.extractor = extractor or FactExtractor()
        self.query_parser = query_parser or QueryParser()
        self.time_mode = time_mode
        self.hybrid_fallback = bool(hybrid_fallback)
        # DEFER 的可选能力：把待确认事实的原始 full memory 也写进 store.pending。
        # 默认 False —— 未确认取值不进 latent 读取路径；打开后每条 pending 多占
        # 128 x 4096 FP16 ≈ 1MB，且 store.pending 永不参与 select_version/bundle。
        self.defer_store_memory = bool(defer_store_memory)
        # 事务回滚自身的失败痕迹（只记录，不掩盖原始异常）。
        self.rollback_errors: List[str] = []
        self.decisions: List[Dict[str, Any]] = []

    # -- write path -------------------------------------------------------
    def write_predicted(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        """把一条公开写视图 {'id','text','time'} 写入状态层与 Current 存储。"""
        assert_no_oracle(event, 'PredictedRuntime.write_predicted', allowed=PUBLIC_WRITE_KEYS)
        started = time.perf_counter()
        result = self.extractor.extract_detailed(event)
        receipts = [self._write_fact(fact, event) for fact in result.facts]
        return {'event_id': event['id'], 'time': event['time'], 'facts': len(result.facts),
                'receipts': receipts, 'unresolved': len(result.unresolved),
                'write_seconds': time.perf_counter() - started}

    def _write_fact(self, fact: AtomicFact, event: Mapping[str, Any]) -> Dict[str, Any]:
        # 版本号与 t14 的 store.commit() 适配层对齐：version_id = fact.id
        # （抽取器内全局唯一；万一重放同一事件，才退化到 event:fact 组合号）。
        version_id = fact.id
        if hasattr(self.store, '_known_id') and self.store._known_id(version_id):
            version_id = f'{event["id"]}:{fact.id}'
        version_event = {'id': version_id, 'time': fact.observed_at, 'text': fact.text}
        if fact.subject == UNRESOLVED_SUBJECT or fact.attribute == UNRESOLVED_ATTRIBUTE:
            # 不猜合并：留在 extractor 的独立记忆里，不进状态层，也不写 latent。
            return self._log(event, fact, action='DEFER', state_id=None, candidates=[],
                             confidence=0.0, reason='unresolved_identity',
                             old_version_id=None, new_version_id=None,
                             compression=False, store_written=False)
        memory = self.reader.encode_event({'time': fact.observed_at, 'text': fact.text})
        semantic_key = self.index.key(memory)
        candidates = self.router.retrieve_candidates(
            fact, self.state_index, semantic_key=semantic_key)
        # F5：DEFER + defer_store_memory=True 时 resolver 先写索引 pending、再写 store.pending；
        # 为保证失败不留"索引有 / store 无"的单边状态，这里先留一份索引快照（默认关闭时零开销）。
        guard = self.state_index.serialize() if self.defer_store_memory else None
        decision = self.resolver.resolve(fact, candidates, self.state_index)
        if decision.action == 'CREATE':
            return self._write_create(version_event, fact, memory, semantic_key, decision,
                                      candidates)
        if decision.action == 'UPDATE':
            return self._write_update(version_event, fact, memory, semantic_key, decision,
                                      candidates)
        if decision.action == 'DUPLICATE':
            return self._write_duplicate(version_event, fact, memory, semantic_key, decision,
                                         candidates)
        # DEFER：待确认记录已由 resolver 写进 StateIndex.pending（不降级 Current）。
        # 默认不写 latent 存储（defer_store_memory=False），避免把冲突记忆当已确认事实
        # 读取；打开后额外在 store.pending 留原始 full memory（≈1MB/条，永不参与读取）。
        written = self._write_pending(version_event, memory, semantic_key, decision, candidates,
                                      guard=guard)
        return self._log(event, fact, action='DEFER', state_id=None, candidates=candidates,
                         confidence=decision.confidence, reason=decision.reason,
                         old_version_id=None, new_version_id=None,
                         compression=False, store_written=written)

    def _write_pending(self, version_event, memory, semantic_key, decision, candidates,
                       guard: Optional[Mapping[str, Any]] = None) -> bool:
        """defer_store_memory=True 时把待确认事实的原始 full memory 存进 store.pending。"""
        if not self.defer_store_memory:
            return False
        state_id = candidates[0].state_id if candidates else None
        if state_id is None:
            # 无候选时 store 侧无处挂载 pending；索引侧 pending 已记录可追溯。
            return False
        scoped = StateDecision('DEFER', state_id, decision.confidence, decision.reason)
        try:
            self.store.ingest_decision(version_event, memory, semantic_key, scoped)
        except Exception as exc:  # noqa: BLE001 - store 失败 → 撤销索引侧 pending（单边状态）
            if guard is not None:
                self.state_index = StateIndex.load(guard)
            raise IndexCommitError(
                f'store pending write failed for {state_id}: {exc}') from exc
        return True

    def _write_create(self, version_event, fact, memory, semantic_key, decision, candidates):
        # CREATE 用 t15 的事务预备接口：先构造（不插入）记录拿到 state_id，store 写成功
        # 之后才把记录插入索引 → 保持"store 成功才更新索引"的同一契约，也避免留下悬空记录。
        index_snapshot = self.state_index.serialize()   # 事务前的索引快照（含 id 计数器）
        state_id = self.state_index.new_state_id()
        self.state_index.build_record(fact, semantic_key=semantic_key, state_id=state_id)
        scoped = StateDecision('CREATE', state_id, decision.confidence, decision.reason)
        # ingest_decision 不登记 _commit_undo（只有 commit() 适配层登记），所以除了 rollback
        # 之外再留一份整库快照，保证索引失败时 store 能逐字节回到写入前。
        snapshot = self.store._snapshot() if hasattr(self.store, '_snapshot') else None
        receipt = self.store.ingest_decision(version_event, memory, semantic_key, scoped)
        try:
            record = self.state_index.add(fact, semantic_key=semantic_key,
                                          version_id=receipt['id'], state_id=state_id)
        except Exception as exc:  # noqa: BLE001 - 索引插入失败 → 回滚 store + 撤销编号
            # t5/t18 契约第 3 条：索引失败必须回滚 store，否则 store 里会永久留下一个
            # 悬空状态和一条完整 payload（真实 E3 约 1MB/次），污染 accounting 与 serialize。
            rollback = getattr(self.store, 'rollback', None)
            undone = False
            if callable(rollback):
                try:
                    undone = bool(rollback(receipt))
                except Exception as undo_exc:  # noqa: BLE001 - 回滚失败只记录，不掩盖原始异常
                    self.rollback_errors.append(
                        f'store rollback failed for {state_id}: {undo_exc}')
            if not undone and snapshot is not None:
                # rollback 对 ingest_decision 不生效（无 undo 记录）→ 整库快照恢复。
                try:
                    self.store._restore(snapshot)
                except Exception as undo_exc:  # noqa: BLE001 - 同上，只记录
                    self.rollback_errors.append(
                        f'store restore failed for {state_id}: {undo_exc}')
            try:
                # 索引整体回到事务前（记录 + id 计数器），与 store 回滚一起构成原子事务。
                self.state_index = StateIndex.load(index_snapshot)
            except Exception as undo_exc:  # noqa: BLE001 - 快照恢复失败才退回 discard
                self.rollback_errors.append(
                    f'index restore failed for {state_id}: {undo_exc}')
                discard = getattr(self.state_index, 'discard', None)
                if callable(discard):
                    discard(state_id)
            raise IndexCommitError(
                f'index insert failed after store create for {state_id}: {exc}') from exc
        return self._log(version_event, fact, action='CREATE', state_id=record.state_id,
                         candidates=candidates, confidence=decision.confidence,
                         reason=decision.reason, old_version_id=None,
                         new_version_id=receipt['id'], compression=False, store_written=True)

    def _write_update(self, version_event, fact, memory, semantic_key, decision, candidates):
        previous = self.store.current(decision.state_id)
        captured: Dict[str, Any] = {}

        def apply(store, record, applied_fact):
            # 走 t14 的 store.commit() 适配层：event time 取 fact.observed_at，
            # 版本号 = fact.id，动作以 resolver 的真实 decision 为准。
            result = store.commit(record=record, fact=applied_fact, memory=memory,
                                  semantic_key=semantic_key, decision=decision)
            captured['receipt'] = result
            return result

        self.state_index.commit_with(
            self.store, decision.state_id, fact, apply=apply,
            value_fingerprint=value_fingerprint(fact.value),
            observed_at=fact.observed_at)          # 观察时间 → current_observed_at
        receipt = captured.get('receipt', {})
        return self._log(version_event, fact, action='UPDATE', state_id=decision.state_id,
                         candidates=candidates, confidence=decision.confidence,
                         reason=decision.reason,
                         old_version_id=None if previous is None else previous['id'],
                         new_version_id=receipt.get('version_id'), compression=True,
                         store_written=True)

    def _write_duplicate(self, version_event, fact, memory, semantic_key, decision, candidates):
        receipt = self.store.ingest_decision(version_event, memory, semantic_key, decision)
        record = self.state_index.get(decision.state_id)
        # DUPLICATE 不产生新版本：指针、value_fingerprint、current_observed_at 全都不动。
        # updated_at 是索引写入的墙钟时间，不能拿 fact.observed_at 冒充（t15 口径）。
        return self._log(version_event, fact, action='DUPLICATE', state_id=decision.state_id,
                         candidates=candidates, confidence=decision.confidence,
                         reason=decision.reason, old_version_id=record.current_version_id,
                         new_version_id=receipt.get('id'), compression=False,
                         store_written=True)

    def _log(self, event, fact, *, action, state_id, candidates, confidence, reason,
             old_version_id, new_version_id, compression, store_written):
        record = {
            'event_id': event['id'],
            'fact_id': fact.id,
            'predicted_state_id': state_id,
            'candidate_state_ids': [c.state_id for c in candidates],
            'decision': action,
            'confidence': float(confidence),
            'old_version_id': old_version_id,
            'new_version_id': new_version_id,
            'compression_triggered': bool(compression),
            'reason': str(reason),
            'store_written': bool(store_written),
        }
        validate_decision_log(record)
        self.decisions.append(record)
        return record

    # -- query path -------------------------------------------------------
    def answer_predicted(self, query: Mapping[str, Any], *, window_end: Optional[str] = None,
                         time_mode: Optional[str] = None) -> Dict[str, Any]:
        """回答一条查询；时间只走 public_query 的 time 通道。"""
        started = time.perf_counter()
        mode = self.time_mode if time_mode is None else time_mode
        if mode == 'as_of' and self.time_mode != 'as_of':
            raise ValueError("this runtime is pinned to time_mode='window_end'")
        view = public_query(query, time_mode=mode, window_end=window_end)
        assert_no_oracle(view, 'answer_predicted.query_view', allowed=PUBLIC_QUERY_KEYS)
        parsed = self.query_parser.parse(view['question'])
        target_time = view['time']
        timings = {'parse_seconds': time.perf_counter() - started}
        base = {'time': target_time, 'target_time': target_time, 'time_mode': mode,
                'time_source': view['time_source'], 'hybrid_fallback': False,
                'parsed_subject': parsed['subject'], 'parsed_attribute': parsed['attribute']}
        if not parsed['resolved']:
            return base | {'answer': '', 'state_id': None, 'candidates': [],
                           'no_state_match': True, 'selected_version_id': None,
                           'version_role': None, 'timings': timings}
        fact = AtomicFact(id=QUERY_FACT_ID, text=view['question'],
                          observed_at=target_time or UNKNOWN_TIME,
                          subject=parsed['subject'], attribute=parsed['attribute'],
                          value='', source_id=QUERY_FACT_ID)
        reader_query = self._reader_query(view)
        semantic_key = self.index.key(self.reader.encode(self.reader.question(reader_query)))
        candidates = self.router.retrieve_candidates(
            fact, self.state_index, semantic_key=semantic_key)
        ranking = [{'state_id': record.state_id, 'rank': rank,
                    'subject_key': record.subject_key, 'attribute_key': record.attribute_key,
                    'similarity': cosine_similarity(semantic_key, record.semantic_key)}
                   for rank, record in enumerate(candidates, start=1)]
        timings['retrieve_seconds'] = time.perf_counter() - started
        chosen, selected = None, None
        for record in candidates:
            # target_time is None → 无时间约束 → 最新可见版本；绝不补其它时间字段。
            selected = self.store.select_version(record.state_id, target_time)
            if selected is not None:
                chosen = record
                break
        if selected is None:
            result = base | {'answer': '', 'state_id': None, 'candidates': ranking,
                             'no_state_match': True, 'selected_version_id': None,
                             'version_role': None, 'timings': timings}
            if self.hybrid_fallback:
                # 默认关闭；开启后也只是“无记忆参数化读取”，单独分口径统计。
                fallback = self.reader.answer(reader_query, None)
                result = result | {'answer': fallback.get('answer', ''),
                                   'hybrid_fallback': True,
                                   'fallback': 'no_memory_parametric'}
            return result
        read = self.read_state(chosen.state_id, view, target_time=target_time,
                               selected=selected, reader_query=reader_query)
        timings.update(read.pop('timings', {}))
        return base | {'candidates': ranking, 'timings': timings} | read

    def read_state(self, state_id: str, view: Mapping[str, Any], *,
                   target_time: Optional[str] = None,
                   selected: Optional[Mapping[str, Any]] = None,
                   reader_query: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """在给定状态下读一个版本（current + 紧邻历史）并交给 FrozenReader。

        target_time is None → 无时间约束 → 最新可见版本（不补任何其它时间字段）。
        只吃公开视图；oracle_state 对照也只是把 state_id 换成金标映射出来的编号，
        读取路径与本函数完全一致。
        """
        started = time.perf_counter()
        if reader_query is None:
            reader_query = self._reader_query(view)
        if selected is None:
            selected = self.store.select_version(state_id, target_time)
        if selected is None:
            return {'answer': '', 'state_id': state_id, 'no_state_match': True,
                    'selected_version_id': None, 'version_role': None,
                    'bundle_version_ids': [], 'read_latent_slots': 0,
                    'timings': {'read_seconds': time.perf_counter() - started}}
        current = self.store.current(state_id)
        historical = selected if selected is not current else next(
            (entry for entry in reversed(self.store.entries)
             if entry['state'] == state_id and entry is not current), None)
        entries = [current] + ([historical] if historical is not None else [])
        bundle = self.store.bundle(entries, self.markers,
                                   _reader_cfg(self.reader).get('read_latent_slots'))
        if bundle is not None and hasattr(bundle, 'to'):
            device = getattr(self.reader, 'device', None)
            if device is not None and bundle.device != device:
                bundle = bundle.to(device)
        result = self.reader.answer(reader_query, bundle)
        return {'answer': result.get('answer', ''), 'state_id': state_id,
                'no_state_match': False,
                'selected_version_id': selected['id'],
                'selected_version_time': selected['time'],
                'version_role': 'current' if selected is current else 'historical',
                'bundle_version_ids': [entry['id'] for entry in entries],
                'read_latent_slots': result.get('read_latent_slots', 0),
                'timings': {'read_seconds': time.perf_counter() - started}}

    @staticmethod
    def _reader_query(view: Mapping[str, Any]) -> Dict[str, Any]:
        """reader 提示词只允许出现公开的 window_end；unknown 时完全不注入时间。"""
        prompt: Dict[str, Any] = {'question': view['question']}
        if view['time'] is not None:
            prompt['time'] = view['time']
        return prompt

    # -- diagnostics ------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for record in self.decisions:
            counts[record['decision']] = counts.get(record['decision'], 0) + 1
        return {'facts_written': len(self.decisions), 'decision_counts': counts,
                'states': len(self.state_index), 'pending': self.state_index.pending_count,
                'store': self.store.accounting()}


__all__ = [
    'DECISION_LOG_KEYS',
    'PredictedRuntime',
    'QUERY_MASK_TOKENS',
    'QUERY_PLACEHOLDER',
    'QueryParser',
    'validate_decision_log',
]
