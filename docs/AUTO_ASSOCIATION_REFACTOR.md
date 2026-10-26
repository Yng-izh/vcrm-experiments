# 自动事实关联最小侵入式改造 · 总报告（P0–P4）

> 版本：2026-10-26（收口）
> 口径：本轮**只做代码 + CPU 单测 + 文档**。**未训练、未跑 GPU 全量、未跑官方评测**；所有"未执行"项在 §8 单独列明，不得当作已完成。
> 本文每条结论都给出 `文件:行` 或门禁结论出处；门禁结论来自团队任务 t2 / t3 / t17 / t19 / t21 / t11。

---

## 0. 一句话结论

在不读测试集 `state_key`、不依赖目标 episode 的前提下，把"事实抽取 → 状态索引 → 路由 → 更新裁决 → 增量 current 存储 → 版本选择 → 冻结 Reader 读取"这条链路**新增**为可独立运行的代码路径，**ICAE / E3 Codec / Frozen Reader / VersionMarkers 一行未改**，全部 Oracle 对照代码原样保留；P1/P2/P3 已过复审门禁，P4 修复完成但**复审门禁尚未复签**（见 §3 与 §8）。

---

## 1. 改造范围与不动点

### 1.1 不动点（有 diff 证据）

| 组件 | 状态 | 证据（`git diff HEAD` 行数） |
|---|---|---|
| `vcrm/codec.py`（E3 SlotCodec） | **零改动** | 0 |
| `vcrm/reader.py`（Frozen Reader） | **零改动** | 0 |
| `vcrm/index.py`（SemanticIndex，含 `associate`） | **零改动** | 0 |
| `SlotMemoryRuntime.answer()`（Oracle 读取入口） | **零改动** | `runtime.py` 全部 hunk 行号 ≥ 185，`answer()` 位于 :115，未被任何 hunk 触及 |
| Oracle 对照代码（`oracle_state` 模式、`CurrentSnapshotStore`、`KeyframeStore.ingest`） | **保留** | `probe/run_state_eval.py` 三种 mode 并存；`store.py` 中 `CurrentSnapshotStore.build` / `KeyframeStore.ingest` 原样 |

> 唯一需要说明的既有文件改动：`vcrm/data.py` 有 28 行改动，属**时间语义**任务（`query.time` / `as_of` / 观察窗口口径），不属 P0–P4 新组件；`vcrm/store.py` 现为 9 个 hunk（P1 的 `IncrementalCurrentStore` 主体 + 后续 `read_mode=current_only` 扩展），均在后加区域。

### 1.2 新增组件（全部为新文件，共 3861 行）

| 文件 | 行数 | 角色 |
|---|---|---|
| `vcrm/fact_schema.py` | 124 | `AtomicFact` / `StateDecision` / `STATE_ACTIONS` |
| `vcrm/data_views.py` | 227 | 公开写/查询视图白名单 + `assert_no_oracle` |
| `vcrm/fact_extractor.py` | 603 | 多事实拆分、指代消解、属性归一化 |
| `vcrm/state_index.py` | 787 | `StateIndex` / `StateRecord` / 基数本体表 / 事务提交 |
| `vcrm/state_router.py` | 238 | 两阶段候选检索（硬过滤 → 余弦排序） |
| `vcrm/update_resolver.py` | 568 | 规则 1–6 + rule5b（默认关） |
| `vcrm/query_time.py` | 502 | `QueryTimeResolver`（问题文本历史锚 → 窗口末刻） |
| `vcrm/store.py`（新增区） | +364 | `IncrementalCurrentStore` |
| `vcrm/runtime.py`（新增区） | +~390 | `PredictedRuntime` / `QueryParser` / `validate_decision_log` |
| `probe/run_state_eval.py` | 812 | 三模式评测器（oracle / predicted / global_e2e） |
| `lme_s_latent/run_e3.py` | 559 | LongMemEval-S latent 管线 |

### 1.3 两条数据流

**写路径（无 Oracle）**
`public_write_event({id,text,time})` → `FactExtractor.extract` → `reader.encode_event` → `index.key` → `StateRouter.retrieve_candidates` → `UpdateResolver.resolve` → `IncrementalCurrentStore.ingest_decision` → `StateIndex` 事务提交（CREATE: store 成功后 `add(state_id=)`；UPDATE: `commit_with` + `store.commit`；DUPLICATE: 只写 alias；DEFER: 只写 `StateIndex.pending`）。

**读路径（无 Oracle）**
`public_query({question}, time_mode, window_end)` → `QueryParser` → 候选状态 → `store.select_version(state_id, window_end)` → 只把 **latent bundle + 问题文本 + 窗口末刻** 注入冻结 Reader → `answer`。

---

## 2. 新组件接口清单

### 2.1 `vcrm/fact_schema.py`
- `AtomicFact(id, text, observed_at, subject, attribute, value, source_id, …)`（:35）
- `StateDecision(action, state_id, confidence, reason)`（:75）
- `STATE_ACTIONS = ('CREATE','UPDATE','DUPLICATE','DEFER')`（:18）、`STATE_ACTIONS_REQUIRING_STATE = ('UPDATE','DUPLICATE')`（:21）

### 2.2 `vcrm/data_views.py`（P0 边界）
- `PUBLIC_WRITE_KEYS = ('id','text','time')`（:28）、`PUBLIC_QUERY_KEYS = ('question','time','time_source')`（:30）
- `TIME_MODES = ('as_of','window_end')`（:33）、`TIME_SOURCES = ('as_of','window_end','unknown')`（:36）
- `public_write_event(event, public_id)`（:58）—— 白名单裁剪，多余字段直接丢弃
- `public_query(query, time_mode='window_end', window_end=None, allow_as_of=False)`（:77）—— `window_end` 模式绝不读 `query['time']/['as_of']`
- `PublicEventCounter`（:134）、`assert_no_oracle(obj, where, allowed)`（:181）、`assert_public_id_independent`（:201）

### 2.3 `vcrm/fact_extractor.py`
- `extract(event)`（:339）/ `extract_detailed(event)`（:343）→ `ExtractionResult(facts, unresolved)`
- `provenance(fact_id)`（:368）→ `FactProvenance(source_id, char_start, char_end, observed_at)`，供分级错误分析取 clause
- 常量 `UNRESOLVED_SUBJECT`（:33）/ `UNRESOLVED_ATTRIBUTE`（:34）/ `UNKNOWN_TIME`（:36）
- role 前缀处理：`_ROLE_PREFIX`（:51）剥一层；管线侧 `render_session()` 保证**幂等**（P4-F1 修复）

### 2.4 `vcrm/state_index.py`
- `StateIndex`：`add` / `build_record`（:504，不插入索引的预备记录）/ `discard`（:552）/ `add_keys`（:560）/ `commit`（:573）/ `merge_pending`（:591）/ `commit_with(store, state_id, fact, apply=…)`（:634）/ `serialize`（:706）/ `load`（:718）
- `StateRecord` 身份 = `(subject_key, attribute_key)`；`snapshot/restore`（:323/:327）供事务回滚
- 基数三档：`CARDINALITY_EXCLUSIVE`（:97）、`CARDINALITY_MULTI_VALUED`（:105）、兜底 UNKNOWN；`CARDINALITY_POLICIES = ('ontology','unknown_only')`（:129）
- `SEMANTIC_DIMENSIONS = 128`（:28）

### 2.5 `vcrm/state_router.py`
- `stage1_filter`（:94）硬过滤（主体/属性身份）→ `stage2_rank`（:113）只在候选内做余弦/词法排序
- `retrieve_candidates(fact, state_index, top_k=5, semantic_key=…)`（:148）、`route(fact, state_index, resolver, …)`（:180）、`StateRouter`（:192）

### 2.6 `vcrm/update_resolver.py`
- `UpdateResolver(*, recency_update=False, recency_min_confidence=0.0, …)`（:139）；`resolve`（:302）
- 规则 1–6 默认口径；rule5b `_recency_update`（:245）**默认关闭**，打开后 reason = `exclusive_superseded_by_recency:<attr>`；rule5 = `explicit_replacement:<attr>`
- `cardinality_of`（:205）/ `cardinality_summary`（:216）/ `_record_audit`（:223）；每条 reason 尾部带 `|cardinality=<bucket>`
- `parse_timestamp`（:91）、`values_equivalent`（:123）、模块级 `resolve`（:407）

### 2.7 `vcrm/store.py` · `IncrementalCurrentStore`（:670）
- `INCREMENTAL_FORMAT = 'incremental_independent_current_v1'`（:666）；`INCREMENTAL_BUDGET_MODES = ('retain_all','drop_oldest')`（:667）
- `ingest_decision(event, memory, semantic_key, decision)`（:738）—— **单一事务**：snapshot → 四 action 只改数据 → `validate_protocol` → `_guard_budget`（:923）→ except `_restore`（:919）
- `commit(record, fact, *, memory, semantic_key, decision)`（:809）—— 适配层，返回 `{'version_id','action'}`；`rollback(result)`（:863）单槽 undo
- `ingest`（:734）对 m-slot code 直接拒绝（"Write only original full memory, never an m-slot code"）
- 读取：继承 `KeyframeStore` 的 `entries` / `blocks` / `select_version` / `current` / `pending` / `serialize`（:1054）

### 2.8 `vcrm/runtime.py` · 新入口
- `PredictedRuntime(reader, codec, markers, index, store, state_index, *, router, resolver, extractor, query_parser, time_mode, allow_as_of, defer_store_memory, hybrid_fallback, as_of_future_fallback, read_mode, multi_state, top_k)`（:306）
- `write_predicted(event, batch_encode=True)`（:366）→ 返回 receipts（每事实一条 11 字段决策记录）
- `answer_predicted(query, *, window_end=…)`（:571）
- `QueryParser.parse(question)`、`validate_decision_log(record, where)`（11 字段固定契约 + `assert_no_oracle`）
- 决策日志字段：`event_id, fact_id, predicted_state_id, candidate_state_ids, decision, confidence, old_version_id, new_version_id, compression_triggered, reason, store_written`

### 2.9 `probe/run_state_eval.py`（P3 评测器）
- `--mode {oracle_state, predicted_state, global_e2e}`、`--time-mode {as_of, window_end}`、`--reader {icae, stub}`、`--frontend {rules, llm}`
- 关键开关：`--recency-update` / `--recency-min-confidence` / `--hybrid-fallback`（默认 off）/ `--defer-store-memory`（默认 off）/ `--history-probe-mode {window_end, event_time}` / `--top-k` / `--neighbours`
- 产物：`{stem}.decisions.jsonl`（纯运行时 11 字段，无 gold，过 `validate_decision_log`）+ `{stem}.scores.jsonl`（含 gold，行内 `gold_fields_present=true`）+ `{stem}.summary.json`

### 2.10 `vcrm/query_time.py` + `lme_s_latent/run_e3.py`（P4）
- `QueryTimeResolver().resolve(question, asked_at)` → `TimeResolution(window_end, time_anchor, anchor_ambiguous, anchor_text, anchor_kind, reason, ask_time)`
- 四类时间表达：绝对锚（ISO 日 / 月+年 / 单年）、相对锚（last year / N years ago / this year）、模糊词（不编造年份，`anchor_ambiguous=True`）、无表达（ask_time）
- `run_e3.py` 关键开关：`--reader {icae,stub}` / `--self-check` / `--as-of-future-fallback`（本管线显式 False）/ `--recency-update` / `--hybrid-fallback` / `--multi-state` / `--read-mode`

---

## 3. P0–P4 阶段门禁：通过条件 vs 实际达成

| 阶段 | 通过条件 | 实际达成 | 门禁结论 |
|---|---|---|---|
| **P0 边界** | 公开写/查询视图只暴露白名单字段；`assert_no_oracle` 覆盖；LongMemEval 每题独立 haystack | `data_views.py` 白名单 + `assert_no_oracle`（:181）；`lme_s_text_rag` 改为逐题 `RetrievalIndex` + namespace guard；`tests/test_lme_scope.py` 9 条 | t2 完成；**通过** |
| **P0 审计** | 现有 0.8512 EM 的性质被如实定性 | `docs/E3_LEAK_AUDIT.md`（分层结论 + R1–R6 + E5） | t3 完成；结论见 §5 |
| **P1 存储** | `IncrementalCurrentStore` 与 snapshot 逐比特 parity；预算守卫覆盖四 action；真实 E3 权重实跑 | 服务器 `pytest -k RealE3WeightsParityTest` → **6 passed**（sha256 `1411187621…3cbc5a`）；四 action 单一事务；`_known_id` 不占位拒绝 id | t17 **pass（P1 可签署）** |
| **P2 关联** | `FactExtractor`/`StateIndex`/`StateRouter`/`UpdateResolver` 无 Oracle、memory 透传、CREATE 落 store、基数本体表 | 服务器 `pytest tests/test_update_resolver.py tests/test_fact_extractor.py tests/test_state_router.py -q -rs` → **78 passed**；两档基数实测（ontology → UPDATE，unknown_only → DEFER） | t19 **pass（P2 可签署，P3 可开工）** |
| **P3 运行时+评测** | CREATE 索引失败回滚 store；hybrid_fallback 分口径；计数去重；产物分文件；DEFER 单边状态 | 我的独立探针：`store.serialize()` 前后 **1417 B、sha256 均 `aacd17b25485ae47`（逐字节相同）**；`decision_counts` 与 neighbours=0/2 无关；decisions.jsonl 11 字段 0 行含 gold；服务器 `pytest tests/test_global_query_routing.py tests/test_lme_scope.py -q -rs` → **19 passed / 0 failed / 0 skipped** | t21 **pass（P3 可签署，P4 可开工）** |
| **P4 LME-S latent 管线** | 每题独立 haystack；四类时间表达；hypothesis 直交官方 schema；错误分层；未跑 GPU/官方评测 | 代码与文档层满足（`test_query_time.py` 24 条 OK）；曾因 role 前缀双重拼接导致主体键不一致 → `test_lme_latent.py` 4F+1E | t11 **needs_revision** → 修复完成（P4-F1/F2/F3 全闭合，`RolePrefixIdempotencyTest` 5 例、新基准逐桶全等），**复审门禁尚未复签** |

> **未达成项（如实标注）**：P4 的复审门禁没有跑过；P4 从未在 GPU 上跑过；LongMemEval-S 从未产生官方分数（见 §8）。

---

## 4. 三条风险对照

### 4.1 风险一：控制层会不会变成"文本记忆库"

**结论：不会。控制层只存身份键、指纹与指针，不存原文。**

| 层 | 持久化字段 | 是否含原文 |
|---|---|---|
| `StateIndex` / `StateRecord` | `state_id`、`subject_key`、`attribute_key`、`aliases`、`current_version_id`、`current_observed_at`、`value_fingerprint`、`updated_at` | **否**（`value_fingerprint` 是哈希；属性键为语言归一化串，无 relation id） |
| `PendingMerge` | `pending_id`、`fact` 指纹、`state_id`、`reason` | **否** |
| `IncrementalCurrentStore` entries | `id`、`time`、`kind`（`full` / `code`）、`state`（state_id）、`key`、`keyframe`、`anchor_id`、`compression_count`、`payload`（FP16 latent） | **否**（只有 latent 张量，无文本） |

**字节开销（数量级）**：一个 current = 128 slots × 4096 dim × 2 B ≈ **1 MB**；历史版本压缩为 16 slots ≈ **128 KB/版**；stub 夹具（32 维 / 4 slots）整库 `serialize()` 仅 1417 B。`budget_mode='retain_all'` 下字节随版本数线性增长，`_guard_budget`（`store.py:923`）在**事务内**拦截超预算写入（P1 门禁实测：超预算 → `ValueError` 且 `serialize()` 逐比特不变）。

**Reader 不得经控制层取答案**：读路径上控制层只回答"选哪个版本"，`answer_predicted` 把 **latent bundle + 问题文本 + window_end** 注入 Reader（`runtime.py:571+`），提示词不注入任何状态元数据；`validate_decision_log` + `assert_no_oracle` 对写出的每条日志做字段级校验。t3 审计补充：`reader.py` 返回值带 `raw_history_access` 标记，可作运行期"未读原文"证据。

### 4.2 风险二：UPDATE 误判导致不可逆压缩

**结论：默认口径下误判被"保守 DEFER + 原子提交 + 决策日志"三重约束；但旧 current 一旦被替换，压缩本身不可逆——这是显式接受的残余风险。**

- **原子提交**：CREATE 失败 → `store.rollback(receipt)` → 未生效则 `_restore(snapshot)` → `StateIndex.load(事务前快照)`（连 id 计数器一起回退）→ 仍抛原 `IndexCommitError`；UPDATE 走 `commit_with`，索引侧失败回滚 store（`state_index.py:_rollback_store`）。P3 门禁独立探针：失败后 store 逐字节回到写入前。
- **决策日志**：每条事实一条 11 字段记录（`event_id/fact_id/predicted_state_id/candidate_state_ids/decision/confidence/old_version_id/new_version_id/compression_triggered/reason/store_written`），`decisions.jsonl` 写出前逐条 `validate_decision_log`，且 `grep gold|state_key|answer` 命中 0 行。
- **保守 DEFER**：默认六条规则，rule5b（`exclusive_superseded_by_recency`）**默认关闭**；`observed_at` 相等/更早/不可解析一律 `DEFER`（`rule6:recency_not_applicable`），不猜。基数未知时 `cardinality_policy='unknown_only'` 走 DEFER 基线。LME-S 管线实测：off 档 `update_deferred=2`、on 档 `update_deferred=0` 且 `exclusive_superseded_by_recency=3`。
- **残余风险（不可逆）**：旧 current 被替换后压成 16-slot code，`full` payload 不再保留，`rollback` 只能恢复到**最近一次** commit（`_commit_undo` 单槽）。因此：① rule5b 必须在独立验证集校准 `recency_min_confidence` 后才能开；② 需要"可回放"的场景应显式打开 `defer_store_memory` 或提高 `byte_budget`（两者都有 ≈1 MB/条的代价）。

### 4.3 风险三：TempLAMA 与 LongMemEval 分布差异（跨数据集主张的收缩边界）

| 维度 | TempLAMA | LongMemEval-S | 主张收缩 |
|---|---|---|---|
| 每 episode 状态数 | **1 个**（`data.py:26-31`） | 多主体多属性、每题数百 session | TempLAMA 上 `SemanticIndex.associate` 的 margin 条件恒满足 → 路由退化为 0.8 阈值判定；**conflict 切片（98.4%）不构成冲突消解测试** |
| 写侧分组来源 | 100% 来自数据集 `state_key`（`store.py:404,410`） | 无 `state_key`，靠 `FactExtractor + StateIndex` 自行分组 | 跨数据集比较的是"分组能力"，不是同一件事 |
| 时间通道 | `query.time ≡ as_of ≡ 金标版本 event 时间` | 问题文本历史锚 / `question_date`，无 `as_of` | TempLAMA 分数不能当自动关联证据（见 §5①） |
| 查询形态 | 模板式问答 | 自由问答，`QueryParser` 模板式解析覆盖不足 | LME-S 上 `state_match_failed` 会主导，须先把解析能力或主指标口径定下来 |

**收缩后的可主张范围**：① "在给定时间戳下，latent 版本可被冻结 Reader 读出" —— 有 TempLAMA 数字支撑（0.8512 EM / 0.9346 F1，但性质见 §5①）；② "自动事实关联链路在 CPU stub 上端到端可跑通、错误可分层" —— 有 P3/P4 CPU 单测支撑；③ **不可主张**"自动关联在 LongMemEval 上达到某分数"——尚无任何 GPU/官方数字。

---

## 5. t3 泄漏审计结论（必须与结果一同引用）

出处：`docs/E3_LEAK_AUDIT.md`（t3 交付）。以下五条原样进入本报告：

1. **0.8512 EM 的性质 = 协议级标签等价 + latent 版本可读性，不是自动关联能力证据。** `query.time ≡ as_of ≡ 金标版本 event 时间`（`data.py:39/44/47-50`），`reader.py:173-175` 用 `time` 注入同一日期 → mask 掉 `as_of` 是空操作，`select_version(t)` 必然命中金标版本（置信度高）。
2. **TempLAMA 每个 episode 只有 1 个 state**，`SemanticIndex.associate` 的 margin 条件恒满足（`data.py:26-31,51`；`index.py:36-44`）→ conflict 切片 98.4% 不构成冲突消解测试。
3. **fallback 5.37% 是负贡献**：fallback 子集 EM 0.5612 vs latent 子集 0.8677，拉低总体 0.0165 EM → 不能用 fallback 解释高 EM。
4. **§4 的 R1–R6 待运行验证项**（逐行重算，均"需运行验证"）：
   - R1 fallback 占比与其对总体的影响（`ours_v6_summary.json` 逐行重算）
   - R2 读取期"无原文"证据（`reader.py:233` 的 `raw_history_access`）
   - R3 "选中版本 == 金标版本"的精确比例（受 `data.py:43-46` 版本事件口径影响）
   - R4 多 gold 行的 EM 宽松度（`metrics.py:13-14,17-25` 的 `any()` / `max()`）
   - R5 anchor 未来信息（历史 code 以最终版 memory 为 anchor）的方向与幅度
   - R6 ex28 规则的"调参泄漏"边界
   - **建议主实验 E5**：用 `data_views.public_write_event`（去 `state_key`）建 store，让 P1/P2 的 `StateIndex + StateRouter` 自行分组，与 oracle 分组版对比 EM 差 —— 这是 P0–P4 改造的**主实验**，目前**未执行**。
5. **P0–P4 对 ① 的回应**：新增 `time_mode` 双通道（`data_views.py:33`）——`as_of` 仅作受控对照（`allow_as_of=True` 才可用），正式 Predicted 口径为 `window_end`，其值来自 `QueryTimeResolver`（问题文本历史锚 → 区间末刻；无锚才用提问时刻），**`question_date` 从不被当作历史目标时间、也从不直接透传成 `as_of`**。因此在 P0–P4 口径下，"时间维度免费解掉"的通道已被关闭；但该结论目前只由 CPU 单测（`test_query_time.py` 24 条、`test_time_semantics.py` 18 条）与代码审查支撑，**尚无 GPU/官方分数验证**。

---

## 6. 放行清单（待用户授权，逐条给命令/资源/前置条件）

> 以下**全部未执行**。显存与时长均为数量级估算，不是实测。

### 6.1 GPU 冒烟（最小验证：链路在真实权重上不炸）
```bash
ssh v100-2 'cd /share/latent2/vcrm_icae && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:probe \
  env_py310/bin/python probe/run_state_eval.py --mode predicted_state --reader icae \
  --split templama_test --limit 5 --tag gpu_smoke --hybrid-fallback'
```
- 显存：≈14.5 GB（Mistral-7B FP16）+ codec/markers/index <0.1 GB；单卡 V100-32G 足够
- 时长：5 题 ≈ 5–10 min（含权重加载）
- 前置：E3 checkpoint `/share/latent2/vcrm_icae/runs/architecture_v6_E3_train/codec.safetensors`（sha256 `1411187621…3cbc5a`）；服务器 checkout 与本地 md5 一致；`CUDA_VISIBLE_DEVICES` 显式指定
- 判据：`decisions.jsonl` 11 字段、`hybrid_fallback_count=0`（默认 off）、`store_bytes` 与 `no_state_match_rate` 落盘可读

### 6.2 same-state 判别器训练
```bash
ssh v100-2 'cd /share/latent2/vcrm_icae && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:probe \
  env_py310/bin/python probe/build_state_supervision.py --data-root data/prepared --split templama_test --out runs/same_state/sup.jsonl --stats' \
&& ssh v100-2 'cd /share/latent2/vcrm_icae && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:probe \
  env_py310/bin/python probe/train_same_state.py --data runs/same_state/sup.jsonl --out runs/same_state \
  --encoder-checkpoint runs/architecture_v6_E3_train/codec.safetensors --epochs 3 --batch-size 64'
```
- 显存：≈12–16 GB；时长：数小时（随样本量与 epochs 线性）
- 前置：监督数据由 `build_state_supervision.py` 生成（**只用公开字段**，不含 `state_key`）；训练脚本与 P0–P4 代码隔离，产物落 `runs/`
- 判据：`--report` 输出的 FPR/阈值曲线，且**不得**回灌进 P1–P4 默认口径（默认仍为规则）

### 6.3 Predicted State 全量（主实验 E5 的前提）
```bash
ssh v100-2 'cd /share/latent2/vcrm_icae && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:probe \
  env_py310/bin/python probe/run_state_eval.py --mode predicted_state --reader icae \
  --split templama_test --limit 0 --neighbours 4 --tag predicted_state_full'
```
- 显存：同 6.1；时长：写入 ≈25 万次 `encode_event`（500 题 × ~500 session）≈ **10–28 GPU-hours**（单卡），读取 ≈1–2 s/题
- 前置：6.1 冒烟通过；`--hybrid-fallback` 默认 off；`--recency-update` 未校准前保持 off
- 判据：`summary.json` 的 State Pairwise P/R/F1、错误合并率/拆分率、Query State Recall@1/@3、`no_state_match_rate`、`decision_counts`（须与 `decisions.jsonl` 行数一致）

### 6.4 LongMemEval-S 官方 gpt-4o 评测
```bash
# 1) 生成 hypothesis（CPU stub 或 GPU latent，二者 tag 分开）
python3 lme_s_latent/run_e3.py --data <lme_s.json> --reader stub --tag lme_s_latent_smoke --self-check
# 2) 官方链路：dry-run 先看命令
LME_TAG=lme_s_latent_smoke bash lme_s_text_rag/run_official_eval.sh
# 3) 真跑（需 OPENAI_API_KEY 与账单授权）
LME_TAG=lme_s_latent_smoke bash lme_s_text_rag/run_official_eval.sh --execute
```
- 资源：CPU + OpenAI API（gpt-4o judge）；**无 GPU 需求**；时长取决于 API 速率（500 题数量级：小时级）
- 前置：`OPENAI_API_KEY`；官方 S 数据路径确认（`docs/LME_S_OFFICIAL_EVAL_SPEC.md`）；`make_official_eval_input.py` 的 `--runs-dir` 传字符串会 `TypeError`（属 t14，未修），故必须用默认 runs 目录；`--self-check` 全绿（空 hypothesis 即链路失败）
- 判据：`official_eval_out/<tag>/` 的 `eval-results` 与 `print_qa_metrics` 输出；**在此之前任何"LME-S 分数"都不得引用**

---

## 7. tests/ 最终形态与 CI 命令

**17 个测试文件、301 个 `def test_`**：

| 文件 | 用例数 | 覆盖 |
|---|---|---|
| `test_store_roundtrip.py` | 27 | 增量 store 往返 / 预算守卫四 action / 拒绝不占位 / undo 单槽 |
| `test_incremental_parity.py` | 8 | snapshot vs incremental parity（真实 E3 权重 + 本地 codec + FP32 舍入源） |
| `test_fact_extractor.py` | 14 | 多事实拆分 / 指代 / 归一化 |
| `test_state_router.py` | 16 | 两阶段过滤与排序 |
| `test_update_resolver.py` | 51 | 六条规则 + rule5b + 基数两档 + 真实 store 集成 |
| `test_global_query_routing.py` | 10 | 全局池路由 / 决策日志契约 / DEFER 单边状态 |
| `test_oracle_isolation.py` | 29 | 无 Oracle 隔离 |
| `test_time_semantics.py` | 18 | 时间语义（`time` / `as_of` / 观察窗口） |
| `test_query_time.py` | 24 | `QueryTimeResolver` 四类时间表达 |
| `test_lme_scope.py` | 9 | 每题独立 haystack（含 1 条依赖门控真实检索） |
| `test_lme_latent.py` | 22 | LME-S latent 端到端 + role 前缀幂等 + 分级桶 |
| `test_current_only_mode.py` | 7 | `read_mode=current_only` |
| `test_lme_data.py` | 23 | LME 数据解析 |
| `test_llm_client.py` | 12 | LLM 前端客户端 |
| `test_main_eval.py` | 14 | 主评测入口 |
| `test_icae.py` | 14 | ICAE 回归（需 pytest） |
| `test_runner.py` | 3 | runner 回归（需 pytest） |

**CI 命令**

```bash
# 本地（macOS，stdlib unittest 即可覆盖 15 个文件）
python3 -m unittest discover -s tests -p 'test_*.py' -v

# 服务器 CPU（全量，含需要 pytest 的 test_icae.py / test_runner.py）
ssh v100-2 'cd /share/latent2/vcrm_icae && CUDA_VISIBLE_DEVICES= PYTHONPATH=.:probe \
  env_py310/bin/python -m pytest tests -q -rs'

# 门禁等价子集（P3/P4 复核用）
ssh v100-2 'cd /share/latent2/vcrm_icae && CUDA_VISIBLE_DEVICES= PYTHONPATH=.:probe \
  env_py310/bin/python -m pytest tests/test_global_query_routing.py tests/test_lme_scope.py -q -rs'
```

**同步纪律（每次服务器门禁前必须做）**：`scp vcrm/*.py probe/*.py tests/*.py` 到 `/share/latent2/vcrm_icae/`，然后 `md5sum` 双侧比对，**必须全一致**才允许判定（t21 实测 12/12 一致；历史上曾因服务器 checkout 滞后导致"module missing"假 skip）。

**允许的 skip**：仅限显式设计性依赖缺失，例如本地 `test_lme_scope.py::test_real_retrieval_stays_inside_haystack_with_id_collision` → `'rank_bm25/sentence_transformers/BGE model unavailable (CPU-only CI)'`（该用例在服务器上实跑通过）。任何 "module missing / import error" 型 skip 视为失败。

---

## 8. 未达成 / 未执行清单（诚实清单）

| 项 | 状态 |
|---|---|
| P4 复审门禁（t11 修复后的 re-gate） | **未跑** |
| GPU 冒烟 / Predicted State 全量 / same-state 训练 | **未执行** |
| LongMemEval-S 官方 gpt-4o 评测 | **未执行**（无任何官方分数） |
| t3 §4 的 R1–R6 逐行重算 | **未执行**（需运行验证） |
| t3 §5 主实验 E5（去 `state_key` 写视图 vs oracle 分组） | **未执行** |
| rule5b `recency_min_confidence` 校准 | **未校准**（默认 0.0、开关默认 off） |
| `make_official_eval_input.py --runs-dir` 字符串 `TypeError` | **未修**（属 t14） |
| TempLAMA 上任何"自动关联能力"主张 | **不成立**（见 §5①） |

---

## 9. 引用索引

- 边界与契约：`vcrm/data_views.py`、`vcrm/fact_schema.py`、`docs/P2_NO_ORACLE_STATE_WRITE.zh.md`
- 存储与 parity：`vcrm/store.py:666+`、`tests/test_incremental_parity.py`、t17 门禁
- 关联与裁决：`vcrm/state_index.py`、`vcrm/state_router.py`、`vcrm/update_resolver.py`、t19 门禁
- 运行时与评测：`vcrm/runtime.py:306+`、`probe/run_state_eval.py`、t21 门禁
- LME-S 管线：`lme_s_latent/run_e3.py`、`lme_s_latent/README.md`、`docs/LME_S_OFFICIAL_EVAL_SPEC.md`、t11 门禁（needs_revision → 修复完成待复签）
- 泄漏审计：`docs/E3_LEAK_AUDIT.md`（t3）
