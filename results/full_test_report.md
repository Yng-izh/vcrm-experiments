# csr_v1 全量正式评测对照报告（34,963 test queries）

日期：2026-10-20 ｜ 协议：csr_v1（Controlled State Routing rev1，冻结于 481 diag）｜ 提取规则：ex28（主表 extracted view）
读者模型：Mistral-7B-Instruct-v0.2（greedy / max128 / seed42）
数据：TempLAMA test 34,963 queries / 13,540 events；denominators（contract §5）：hist 30,906 / latest 4,057 / conflict 34,407（metadata 定义）

## 主表（extracted view, ex28）

| 方法 | overall EM | overall F1 | hist EM | hist F1 | latest EM | latest F1 | conflict EM | conflict F1 |
|---|---|---|---|---|---|---|---|---|
| C1 Hybrid RAG | **0.3906** | **0.6003** | **0.3810** | **0.5909** | **0.4639** | **0.6724** | **0.3953** | **0.6024** |
| C2 LLMLingua-2 RAG | 0.1123 | 0.3396 | 0.1106 | 0.3356 | 0.1260 | 0.3705 | 0.1128 | 0.3388 |
| C3 ICAE-Full | 0.3624 | 0.5581 | 0.3746 | 0.5701 | 0.2694 | 0.4674 | 0.3632 | 0.5551 |
| C4 Mem0 OSS | 0.2504 | 0.4771 | 0.2520 | 0.4783 | 0.2379 | 0.4678 | 0.2531 | 0.4785 |

数值来源：C1/C2 由各自 `c1/c2_test_summary.json` 冻结指标核对一致；overall/latest-F1/conflict 列由 jsonl 逐行重算（与 summary 双核一致）。C3 由 8 shard jsonl（c3full_r0–r7，共 34,963 行）合并重算，与各 shard summary 一致。C4 由 6 shard jsonl 合并后 `closeout_full.py` 生成 `c4_mem0_oss/runs_csr_v1/full_run_summary.json`（n_scored=34,963，conflict_n=34,407，latest_n=4,057）。

## 一致性检查（全部通过）

- 四份 jsonl 各 **34,963 行**；qid 集合完全一致（逐行 set 相等，missing=0 / extra=0）。
- protocol=csr_v1：四份均 100%（resume 会拒绝非 csr_v1 旧行）。
- 空 answer / 空 extracted_answer 行：C1/C2/C3/C4 均为 **0 行**。
- conflict 口径统一：以 metadata 定义（同 state_key 存在 ≥2 个不同 answer set）计 34,407；四方法同分母。

## 结果解读

- **C1 全列最优**：完整 episode 文本 + BGE/BM25 混合检索，hist 0.3810 / latest 0.4639。
- **C3 次之**：hist 与 C1 接近（0.3746），但 latest 掉到 0.2694——latest 查询依赖最近事件，128-slot 压缩对该切片损失最大。
- **C4 第三**：检索侧 avg 3.47 条 memory、非空率 0.9949，信息量低于 C1 的 topk=10 混合检索，hist/latest/conflict 三切片都在 0.24–0.25。
- **C2 最差**：压缩把 episode/retrieval 级事实保留率从 1.0000 压到 0.6125，构成 EM 的硬天花板（详见 C2 消融记录）。
- conflict 切片上 C1 优势最大（0.3953 vs C3 0.3632 / C4 0.2531 / C2 0.1128），符合"完整上下文更抗状态冲突"的预期。

## C3 备注（原样保留 captain note）

TempLAMA episodes 极短（4,057 episodes 全部 ≤5,120 tokens，max 423，mean 87.3；0 evictions，全部单 segment）：1,024-slot 预算与 segmentation/eviction 逻辑在本数据集上从不触发，每个 ICAE-Full 行实际是对全历史的单次 128-slot 压缩，远未达到官方 26–32K 长上下文工作点。协议按 contract 冻结，此事实记录在案。

## C4 执行备注

- 6 卡 6 分片（GPU 2/3/4/7/8/9，端口 8010–8015）；GPU 5/6 被宿主 namespace 外的进程占 9.2 GB，无法回收，故 N=6 而非 8。哈希分片，覆盖不受影响。
- qdrant local store 为排他锁，每分片独立副本 `qdrant_data_full_s0..s5`；写入侧（gpt-4o-mini）与 as_of 无关，复用既有 collection，未重写。
- 16:25 UTC 首次启动，16:38 因外部误释放 GPU 中断（已写入 580 行保留），16:40 重启，16:59 起 resume，20:29–20:39 六分片全部 `done`。有效计算 3 h 40 min，≈2.3 s/query/shard。
- resume 后 reconciliation：34,963 unique / missing 0 / extra 0；各 shard `_errors.json` 保留中断窗口的 8 条历史记录，对应 query 已在 resume 中重跑并落行。

## GPU 释放（用户硬性要求，已完成）

- C4 结束（20:39 UTC）后 kill 6 个 vLLM api_server（端口 8010–8015），GPU 2/3/4/7/8/9 立即回到 0 MiB。
- `/root/.gpu_occ/exclude` 清空（0 字节）。
- 约 4 分钟后复查：GPU 2/3/4/7/8/9 各 ~31,776 MiB，`gpu-reserve-{2,3,4,7,8,9}` 已重新占卡 —— **占卡恢复确认**。GPU 0/1/5/6 为他人进程，全程未触碰。

## 代码回写

- C2 端口映射（`PORTS = {0:8001, 1:8006, 2:8007}`）已同步本地并提交。
- C4：`runner.py` / `closeout_full.py` 的 csr_v1 对齐（protocol 字段、conflict/latest 切片、`runs_csr_v1/` 输出、resume 守卫）已提交；`closeout_full.py` 分片数由硬编码 4 改为接受任意分片数（覆盖由 qid reconciliation 保证）。
- C1/C2/C3 其余 eval 脚本与本地 HEAD 一致（md5 核对）。
