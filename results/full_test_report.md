# csr_v1 全量正式评测对照报告（五方法，34,963 test queries）

日期：2026-10-20（Ours 于 2026-10-21 用 architecture v6 / E3 重跑）｜ 协议：csr_v1（Controlled State Routing rev1，冻结于 481 diag）｜ 提取规则：ex28（主表 extracted view）
读者模型：Mistral-7B-Instruct-v0.2（greedy / max128 / seed42）
数据：TempLAMA test 34,963 queries / 13,540 events；denominators（contract §5）：hist 30,906 / latest 4,057 / conflict 34,407（metadata 定义）

## 主表（extracted view, ex28）

| 方法 | overall EM | overall F1 | hist EM | hist F1 | latest EM | latest F1 | conflict EM | conflict F1 |
|---|---|---|---|---|---|---|---|---|
| **Ours（VCRM/ICAE 状态路由, v6/E3）** | **0.8512** | **0.9346** | **0.8527** | **0.9353** | **0.8398** | **0.9295** | **0.8603** | **0.9380** |
| C1 Hybrid RAG | 0.3906 | 0.6003 | 0.3810 | 0.5909 | 0.4639 | 0.6724 | 0.3953 | 0.6024 |
| C2 LLMLingua-2 RAG | 0.1123 | 0.3396 | 0.1106 | 0.3356 | 0.1260 | 0.3705 | 0.1128 | 0.3388 |
| C3 ICAE-Full | 0.3624 | 0.5581 | 0.3746 | 0.5701 | 0.2694 | 0.4674 | 0.3632 | 0.5551 |
| C4 Mem0 OSS | 0.2504 | 0.4771 | 0.2520 | 0.4783 | 0.2379 | 0.4678 | 0.2531 | 0.4785 |

数值来源：Ours 由 2 分片 `probe/runs/ours_v6_34963_shard{0,1}.jsonl`（共 34,963 行）合并为 `probe/runs/ours_v6_merged.jsonl`（md5 `59d4d16e941e233ceb2b4649ae32177e`），再由 `probe/ours_v6_summary.py` 重算；架构为 architecture v6 **E3**（训练产物 `runs/architecture_v6_E3_train/`，eval config 取 `E3_val/manifest.json`，关键开关 `codec_value_path=full`）。C1/C2 由各自 `c1/c2_test_summary.json` 冻结指标核对一致；overall/latest-F1/conflict 列由 jsonl 逐行重算。C3 由 8 shard jsonl（c3full_r0–r7，34,963 行）合并重算。C4 由 6 shard jsonl 合并后 `closeout_full.py` 生成 `c4_mem0_oss/runs_csr_v1/full_run_summary.json`（n_scored=34,963，conflict_n=34,407，latest_n=4,057）。

## Ours 两口径分列（纯 latent vs 文本 fallback）

| 口径 | n | overall EM | overall F1 | hist EM/F1 | latest EM/F1 | conflict EM/F1 |
|---|---|---|---|---|---|---|
| 全覆盖（含 fallback） | 34,963 | 0.8512 | 0.9346 | 0.8527 / 0.9353 | 0.8398 / 0.9295 | 0.8603 / 0.9380 |
| 纯 latent（route.path=latent） | 33,085 (94.63%) | 0.8677 | 0.9444 | 0.8683 / 0.9444 | 0.8634 / 0.9451 | 0.8766 / 0.9479 |
| 文本 fallback（episode-scoped） | 1,878 (5.37%) | 0.5612 | 0.7615 | 0.5827 / 0.7774 | 0.3788 / 0.6266 | 0.5719 / 0.7635 |

## 一致性检查

- 五份 jsonl 各 **34,963 行**；qid 集合完全一致（逐行 set 相等，missing=0 / extra=0）。
- protocol=csr_v1：C1–C4 均 100%；Ours 行以 `route.status` / `route.path` 记录控制状态路由结果。
- 空 extracted_answer：C1/C2/C3/C4 均为 **0 行**；Ours 有 **2 行例外**（`Q4961400_P54_2019`、`Q4961400_P54_2020`，均走 fallback，reader 输出被截断成 "The blank _X_ in the statement can be fi…"），待补跑后归零。
- conflict 口径统一：以 metadata 定义（同 state_key 存在 ≥2 个不同 answer set）计 34,407；五方法同分母。
- type/latest 标签：Ours 与 C4 逐 qid 核对，mismatch = 0。
- Ours fallback 作用域：1,878 条 fallback 行的 `fallback_events` 逐条核验，全部属于该 query 自己的 episode，越界 **0** 条；`fallback_scope` 全为 `episode`、`fallback_corpus` 全为 `full_text`。
- Ours latest 路由：4,057 条中 3,859 条走 latent，且 `route.is_current` **全为 True**（0 例外）；198 条走 fallback。

## 结果解读

- **Ours 全列最优**：overall EM 0.8512 / F1 0.9346，比最强 baseline C1（0.3906 / 0.6003）高 **+0.4606 EM / +0.3343 F1**；hist +0.4717 EM、latest +0.3759 EM、conflict +0.4650 EM。逐 query 配对（n=34,963，与 C1 同题）：F1 胜 19,802 / 负 1,919 / 平 13,242。
- **涨幅归因**：主因是从早期 C16 方案切到 architecture v6 的 **E3**（`codec_value_path=full`）。E3 在 val 481 上的既有 recorded delta 为 all_compressed F1 **+0.3920**（CI [0.3261, 0.4593]，support=true），本次 hist F1 0.9353 与当时 val 的 historical F1 0.9237 同量级，方向与幅度自洽。同时关联成功率从 C16 的 83.95% 升到 **94.63%**（fallback 5,611 → 1,878）。
- **两口径**：纯 latent 子集 0.8677/0.9444，文本 fallback 子集 0.5612/0.7615；fallback 占比仅 5.37%，对总体结论影响有限但必须分列披露。
- **C1 为最强 baseline**：完整 episode 文本 + BGE/BM25 混合检索，hist 0.3810 / latest 0.4639。
- **C3 次之**：hist 与 C1 接近（0.3746），但 latest 掉到 0.2694——latest 查询依赖最近事件，128-slot 压缩对该切片损失最大。
- **C4 第三**：检索侧 avg 3.47 条 memory、非空率 0.9949，信息量低于 C1 的 topk=10 混合检索，三切片都在 0.24–0.25。
- **C2 最差**：压缩把 episode/retrieval 级事实保留率从 1.0000 压到 0.6125，构成 EM 的硬天花板。
- conflict 切片排序：Ours 0.8603 > C1 0.3953 > C3 0.3632 > C4 0.2531 > C2 0.1128。
- **待办（未解决，勿当定稿）**：EM 0.85 的绝对量级远超所有 baseline，进论文前需做一次泄漏审计（确认 bundle 构造没有把 gold value 直接喂给 reader）；另有 2 行空答案待补跑。

## Ours 执行备注（v6/E3，本次正式成绩）

- 协议修复：此前 Ours 的 `no_association` 兜底是**跨 episode 全局检索 + LLMLingua-2 压缩语料**，与 csr_v1 的 episode-scoped retrieval 不一致（影响 5,611 题 / 16.05%）。本次改为 `build_episode_retriever(ep)` + `fallback_retrieve(ep, question, t)`：检索集只含该 episode 自身**未压缩**事件，按 `time <= t` 过滤取 top-10；行 schema 增加 `route.path` ∈ {latent, fallback_text}。
- 架构切换：`probe/ours_probe.py` 原先硬编码 `runs/aligned_v3_C16_val/manifest.json` + `runs/aligned_v3_C16_train`（即 C16 方案），本次改为 `load_v6(--arch-exp)`（`probe/v6_loader.py`），默认 E3。
- 2 卡 2 分片（GPU 3、9；`--limit 34963 --tag ours_v6 --shard {0,1} --shards 2`），每卡单进程（in-process HF ICAE reader，非 vLLM）。启动方式：kill `gpu-reserve-{3,9}` 后立即拉起，`/root/.gpu_occ/exclude` 写 3/9 防抢。
- 00:27 → 09:46 CST（UTC+8），墙钟约 9 h 11 min，约 1.9 s/query/shard；平均 prompt 203.0 tokens、平均 reader 0.365 s/query。两分片各 17,482 / 17,481 行，均 `done`，无 error 文件。
- **E3 的 test 成绩不可引用**：`runs/architecture_v6_E3_test/episodes/` 只有 172 个 episode 且无 summary.json（历史中断）。本报告不使用 E3 的 test 结果，只用 E3 训练产物在 csr_v1 下重跑。

## Ours 历史备注（C16，已 SUPERSEDED）

2026-10-20 的 8 卡 8 分片 Ours 全量（`probe/runs/ours_full_34963_shard0..7.jsonl`，md5 `95985bb3b572f43bd881be4d39d0ec62`）用的是 C16 方案 + 跨 episode 压缩 fallback，成绩 overall 0.4946/0.6363（纯 latent 0.5424/0.6536、fallback 0.2445/0.5459）。该结果仅保留作 provenance，**不得作为 csr_v1 正式成绩引用**。

## C3 备注（原样保留 captain note）

TempLAMA episodes 极短（4,057 episodes 全部 ≤5,120 tokens，max 423，mean 87.3；0 evictions，全部单 segment）：1,024-slot 预算与 segmentation/eviction 逻辑在本数据集上从不触发，每个 ICAE-Full 行实际是对全历史的单次 128-slot 压缩，远未达到官方 26–32K 长上下文工作点。协议按 contract 冻结，此事实记录在案。

## C4 执行备注

- 6 卡 6 分片（GPU 2/3/4/7/8/9，端口 8010–8015）；GPU 5/6 被宿主 namespace 外的进程占 9.2 GB，无法回收，故 N=6 而非 8。哈希分片，覆盖不受影响。
- qdrant local store 为排他锁，每分片独立副本 `qdrant_data_full_s0..s5`；写入侧（gpt-4o-mini）与 as_of 无关，复用既有 collection，未重写。
- 16:25 UTC 首次启动，16:38 因外部误释放 GPU 中断（已写入 580 行保留），16:40 重启，16:59 起 resume，20:29–20:39 六分片全部 `done`。有效计算 3 h 40 min，≈2.3 s/query/shard。
- resume 后 reconciliation：34,963 unique / missing 0 / extra 0；各 shard `_errors.json` 保留中断窗口的 8 条历史记录，对应 query 已在 resume 中重跑并落行。

## GPU 释放

- C4 结束（20:39 UTC）后 kill 6 个 vLLM api_server（端口 8010–8015），GPU 2/3/4/7/8/9 立即回到 0 MiB；`/root/.gpu_occ/exclude` 清空。
- Ours v6 结束（2026-10-21 01:46 UTC）后 2 个 `ours_probe.py` 自然退出，GPU 3、9 回到 4 MiB；`/root/.gpu_occ/exclude` 仍写 3/9，`gpu_occupier` supervisor 未运行，需重新拉起才会重新占卡。
- GPU 0/1 为他人进程，全程未触碰；其余卡在本次运行期间被宿主 namespace 外其他会话占用。

## 代码回写

- `probe/ours_probe.py`：fallback 改 episode-scoped 全文检索、新增 `route.path`/`fallback_scope`/`fallback_corpus`、新增 `--arch-exp` 并改用 `load_v6`。
- `probe/v6_loader.py`、`probe/V6_LOADER_SPEC.md`、`probe/ours_v6_summary.py`：新增。
- C2 端口映射（`PORTS = {0:8001, 1:8006, 2:8007}`）已同步本地并提交。
- C4：`runner.py` / `closeout_full.py` 的 csr_v1 对齐（protocol 字段、conflict/latest 切片、`runs_csr_v1/` 输出、resume 守卫）已提交；`closeout_full.py` 分片数由硬编码 4 改为接受任意分片数。
- C1/C2/C3 其余 eval 脚本与本地 HEAD 一致（md5 核对）。
