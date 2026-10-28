# 记忆与读取路径改造（P0 / P1 / P2）

本文件记录 500 题四组结果（A 0.412 / B 0.084 / C 0.244 / D 0.246，D−C = +0.002）之后
对 VCRM 结构的一次改造：**不动对比损失、不动负采样、不动 version marker**，只改
"什么进记忆、按什么顺序读出来、时间语义怎么对齐"。

结论先写：D−C 几乎为 0 不是压缩预算的问题（16-slot 已经够装下一个取值序列），
而是**上游信息组织 + 读取入口**把版本化记忆的能力限制住了。LongMemEval 要的是
"在多段对话里发现、关联、组合、读取证据"，VCRM 的核心却按"同一事实的不同历史取值"
组织，两者之间的缺口就是下面五条。

---

## 1. 五条结构性问题（诊断）

| # | 位置 | 现象 | 后果 |
|---|------|------|------|
| 1 | `vcrm/llm_fact_extractor.py::FACT_EXTRACTION_PROMPT` | "一句一事实"、"不跨句消解指代"、"抽不出就返回空" | 会话被过早压成原子三元组；跨句成立的信息（"我去年换工作了。新公司在西雅图，入职时管六个人"）关系断裂；**抽不出来的内容根本不进 latent memory** |
| 2 | `vcrm/runtime.py::_write_fact` + `defer_store_memory=False` | 身份未解析 → `DEFER`，不建状态、不建版本 | 不确定/冲突的事实**不可读**；知识更新类题目恰恰需要"当时没有定论"的证据 |
| 3 | `vcrm/runtime.py::answer_predicted` | `not parsed['resolved']` 直接返回空；`multi_state=False`；只读第一个命中的状态 | 查询路径被"必须解析出精确 (subject, attribute)"卡死；多跳/多事实题没有读取入口 |
| 4 | `vcrm/runtime.py::read_state` + `vcrm/store.py::select_trajectory` | 按 target_time 选出的 `selected` 只用于"是 current 还是 historical"，未保证进入 bundle；轨迹按"最新优先"装预算 | 已经知道该读哪个历史版本，却可能整体装不下 → 版本化能力无法兑现 |
| 5 | `vcrm/training.py` vs `probe/run_main_eval.py` | 训练：`Current date` = 观测截止、年份在题面；评测：历史年份被当成 `window_end` 注入 `Current date`，题面无年份 | 训练学到的"按题面年份判别"在推理时被口径差异抵消 |

## 2. 三条改造（实现）

### P0 — 证据保留与状态更新解耦（`vcrm/evidence_memory.py`）

新增 `EvidenceMemory` / `EvidenceUnit`：把会话按**轮**切成带上下文（默认 ±1 轮）、
带说话人、带事件时间的证据单元，写入时**不看抽取结果**。

- 写入顺序（`PredictedRuntime.write_predicted`）：证据先落 → 事实抽取 → 事实写状态层 →
  事实 `fact_id → state_id` 回填到证据 → 证据 latent 压缩（`codec.encode`，16-slot 量级）。
- `DEFER` 的语义被收紧为**版本裁决**结论：不改 current、不建版本（`vcrm/update_resolver.py`
  行为不变），但**不再决定信息是否可读**——证据在写入那一刻就已经可检索。
  `vcrm/update_resolver.py::DEFER_IS_VERSION_VERDICT_ONLY` 是这条边界的显式标记。
- 压缩失败/无 anchor（条件化 codec）时只跳过 latent，**文本证据仍然可读**，绝不静默丢证据。

### P1 — 读取顺序：证据检索 → 状态/版本 → 多证据 bundle → reader（`vcrm/runtime.py`）

`answer_predicted` 的新流程：

1. 查询先做**证据检索**（`EvidenceMemory.retrieve`：词法 + 主体/属性加权，严格受
   `observation_cutoff` 可见性约束）。
2. 证据挂到的状态并进读取候选；多状态读取（`multi_state`）允许多个状态一起读。
3. 查询解析失败时不再直接答空：只要有证据挂上状态，就继续读；连状态都没有时，
   仍可"仅证据读取"（`evidence_only_read=True`），把证据 latent 直接给 reader。
4. bundle = 版本条目 + 证据条目（证据用 `historical` role 打标，`VersionMarkers` 不新增
   角色，避免破坏已训练的 marker 语义），预算内贪心，逐条记录在
   `bundle_evidence_ids` / `evidence_latent_slots`。

保留 `StateRouter`：它继续负责"哪个状态"，但**不再**要求查询被精确解析成
`(subject, attribute)` 才有读取资格——那一步从"门"降级为"候选来源之一"。

### P2 — 目标时间驱动版本选择 + 训练/推理时间口径统一

- `vcrm/store.py::select_trajectory(..., ensure_ids=...)`：target_time 选出的版本
  **保证**进入 bundle；预算不足时先淘汰最旧的非必须条目，绝不越过 slot 预算。
  `read_state` 在 `trajectory` 模式下把 `selected['id']` 传进 `ensure_ids`，
  并把 `selected_entry` 指向目标版本（不再固定成"最新可见版本"）。
- 新增 `vcrm/time_semantics.build_reader_query()`：**唯一**的 reader 提示词构造函数。
  - `Current date` 槽 = `observation_time`（观测截止 = `question_date`）；
  - 历史目标时刻只进题面（`In 2019: <question>`）；
  - 从不设置 `query['as_of']`（那是 TempLAMA 金标通道，等于泄漏）。
- `vcrm/training.py` 的 `trajectory_samples` 与 retain 路径改为调用同一个函数 →
  训练/推理的 reader 输入逐字同构，口径漂移在物理上不可能再发生
  （`vcrm/reader.py` 受训练元数据 sha256 锁定，不去改它）。
- `probe/run_main_eval.py`：`resolve_window` 改用 `QueryTimeResolver.resolve_bounds`，
  分别传 `window_end=target_time` 与 `observation_cutoff=question_date`。

### 顺带修掉的归因漏洞

`probe/run_main_eval.py --checkpoint` 以前只被 `sha256_of()` 一下写进 summary，真实权重
却由 `probe/v6_loader.py` 的默认目录（E3）/`VCRM_TRAINING_ROOT` 决定——报告里的
checkpoint 和实跑的权重可以不是同一份。现在：

- 优先级 `--checkpoint` > `VCRM_TRAINING_ROOT` > `v6_loader` 默认目录；
- 两者同时存在且不一致 → **直接报错**（`--allow-checkpoint-mismatch` 才放行）；
- 目录缺 `manifest.json` → 拒绝加载；
- 最终目录写进每一行 row 的 `checkpoint_resolved` 与 summary。

## 2.1 复审补丁（第二轮）

1. **证据可读但未挂状态时不再答空**：`answer_predicted` 的早返条件从
   `not evidence_records` 收紧为 `not evidence_records and not evidence_units`。
   证据检索到了、但事实抽取失败/DEFER 导致没有 `state_id` 时，直接进入
   `evidence_only_read`（仅证据 latent 读取），而不是返回空答案。
2. **证据编码用带上下文的完整文本**：`write_predicted` 编码 `unit.text`（±N 轮上下文），
   `unit.focus` 只用于检索打分。超过 reader `encoder_max_tokens` 的单元按行贪心 +
   超长行字符二分做**确定性切分**（`EvidenceMemory.replace_with_chunks`），切出的子单元
   各自保留事实/状态联系，一段都不丢。同时**取消**每 session 12 条单元的数量截断
   （`max_units_per_session=None` 为默认，`--evidence-max-units-per-session 0`）。
3. **D 组 multi-state 不再退化成 pair**：`_read_multi_state` 的 trajectory 分支改为真正调用
   `store.select_trajectory(..., ensure_ids=[target])`，预算按"证据预留 → 剩余平均分给各状态"
   分配；`multi_state_ensure_missing` / `multi_state_version_counts` 逐题记录。
   单状态 `read_state` 同样先给证据预留 slot（`_evidence_reserve`），避免版本轨迹吃满预算、
   证据一条都进不去。
4. **入口与文档**：README/RESULTS 里的 `--extract-workers 8` 已删除（`probe/run_main_eval.py`
   没有该参数，写上去会直接 unrecognized arguments；抽取走串行路径）。B 组同步修掉旧逻辑：
   证据检索到 session 就照常读，只有"证据与状态都没有"才记 `parametric_only` 并留空。

## 3. 开关与复现

默认全开（`--multi-state`、`--evidence-first`、`--time-semantics unified`）。
复现改造前的数字（t34 口径）：

```bash
python -m probe.run_main_eval --group versioned ... --legacy-read-path
# 等价于 --no-multi-state --no-evidence-first --time-semantics legacy
```

新增统计口径（都在 row/summary 里，可逐题归因）：

- `evidence_units` / `evidence_state_ids` / `evidence_hits`
- `bundle_evidence_ids` / `evidence_latent_slots` / `evidence_only_read`
- `multi_state` / `multi_state_ids`
- `evidence_stats`：`evidence_linked_units` / `evidence_unresolved_units` /
  `evidence_compressed_units` / `evidence_dropped_units`
- `query_target_time` / `query_observation_cutoff` / `query_time_semantics`

## 4. 明确不做的事

- 不继续调对比损失、负样本构造、version marker 位置——D−C≈0 的瓶颈不在这一层。
- 不把 `defer_store_memory=True` 当成 P0 的解：`store.pending` 永不参与
  `select_version` / `bundle`，打开它只是多占 128×4096 FP16/条，可读性毫无变化。
- 不改 `vcrm/reader.py`（sha256 锁定）；时间对齐在 runtime 构造提示词这一层完成。

## 5. 待验证假设（下次跑数时按这几条判）

1. `evidence_unlinked_units` 高 → 抽取层仍是瓶颈（P0 只保证"可读"，不保证"可关联"）。
2. `evidence_in_bundle` 高但答案错 → 问题转到 bundle 内证据与版本的排布/预算，而不是检索。
3. `multi_state_reads` 命中多会话题、但 temporal 题不变 → 时间维度确实需要显式版本选择
   （P2），而不是更多证据。
4. 题面带年份（unified）后 knowledge-update 类是否变化 → 训练/推理时间口径是否是
   该子集的主因。
