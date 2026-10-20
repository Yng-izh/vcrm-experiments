# csr_v1 全量正式评测对照报告（五方法，34,963 test queries）

日期：2026-10-20 ｜ 协议：csr_v1（Controlled State Routing rev1，冻结于 481 diag）｜ 提取规则：ex28（主表 extracted view）
读者模型：Mistral-7B-Instruct-v0.2（greedy / max128 / seed42）
数据：TempLAMA test 34,963 queries / 13,540 events；denominators（contract §5）：hist 30,906 / latest 4,057 / conflict 34,407（metadata 定义）

## 主表（extracted view, ex28）

| 方法 | overall EM | overall F1 | hist EM | hist F1 | latest EM | latest F1 | conflict EM | conflict F1 |
|---|---|---|---|---|---|---|---|---|
| **Ours（VCRM/ICAE 状态路由）** | **0.4946** | **0.6363** | **0.4614** | **0.6061** | **0.7474** | **0.8660** | **0.4985** | **0.6357** |
| C1 Hybrid RAG | 0.3906 | 0.6003 | 0.3810 | 0.5909 | 0.4639 | 0.6724 | 0.3953 | 0.6024 |
| C2 LLMLingua-2 RAG | 0.1123 | 0.3396 | 0.1106 | 0.3356 | 0.1260 | 0.3705 | 0.1128 | 0.3388 |
| C3 ICAE-Full | 0.3624 | 0.5581 | 0.3746 | 0.5701 | 0.2694 | 0.4674 | 0.3632 | 0.5551 |
| C4 Mem0 OSS | 0.2504 | 0.4771 | 0.2520 | 0.4783 | 0.2379 | 0.4678 | 0.2531 | 0.4785 |

数值来源：Ours 由 8 shard jsonl（`probe/runs/ours_full_34963_shard0..7.jsonl`，共 34,963 行）合并后 `probe/ours_full_summary.py` 重算，合并件 `probe/runs/ours_full_merged.jsonl`（md5 `95985bb3b572f43bd881be4d39d0ec62`）。C1/C2 由各自 `c1/c2_test_summary.json` 冻结指标核对一致；overall/latest-F1/conflict 列由 jsonl 逐行重算（与 summary 双核一致）。C3 由 8 shard jsonl（c3full_r0–r7，共 34,963 行）合并重算，与各 shard summary 一致。C4 由 6 shard jsonl 合并后 `closeout_full.py` 生成 `c4_mem0_oss/runs_csr_v1/full_run_summary.json`（n_scored=34,963，conflict_n=34,407，latest_n=4,057）。

## 一致性检查（全部通过）

- 五份 jsonl 各 **34,963 行**；qid 集合完全一致（逐行 set 相等，missing=0 / extra=0）。
- protocol=csr_v1：C1–C4 均 100%（resume 会拒绝非 csr_v1 旧行）；Ours 行以 `route.status` 记录控制状态路由结果。
- 空 answer / 空 extracted_answer 行：C1/C2/C3/C4/Ours 均为 **0 行**。
- conflict 口径统一：以 metadata 定义（同 state_key 存在 ≥2 个不同 answer set）计 34,407；五方法同分母。
- type/latest 标签一致性：Ours 与 C4 逐 qid 核对，mismatch = 0。

## 结果解读

- **Ours 全列最优**：overall EM 0.4946 / F1 0.6363，比最强的 baseline C1（0.3906 / 0.6003）高 **+0.1040 EM / +0.0360 F1**；hist +0.0804 EM、latest +0.2835 EM、conflict +0.1032 EM。最大增益集中在 latest 切片（0.7474 vs C1 0.4639），说明"先定状态再选版本"解决了 baseline 把旧版事实当现行答案的主因。
- **逐 query 配对（n=34,963，与 C1 同题）**：Ours F1 胜 12,990 / 负 11,321 / 平 10,652，均值 +0.0360。
- **路由审计**：`ok` 29,352（83.95%）/ `no_association` 5,611（16.05%）。ok 子集 EM 0.5424 / F1 0.6536；no_association 子集 EM 0.2445 / F1 0.5459（走时间过滤 RRF 兜底，按 contract §3.2 单独报告）。
- **latest 路由正确性**：latest 查询 4,057 条中 3,407 条 `ok`，且这 3,407 条 `route.is_current` **全部为 True**（0 例外）——版本选择层没有选错现行版本；该子集 EM 0.8527 / F1 0.9431，剩余误差来自读者端。另 650 条 latest 落 `no_association`（EM 0.1954）。
- **hist 切片**：ok 子集（25,945 条）EM 0.5017 / F1 0.6156，明显高于 no_association 子集（4,961 条，0.2510 / 0.5569）。
- **C1 为最强 baseline**：完整 episode 文本 + BGE/BM25 混合检索，hist 0.3810 / latest 0.4639。
- **C3 次之**：hist 与 C1 接近（0.3746），但 latest 掉到 0.2694——latest 查询依赖最近事件，128-slot 压缩对该切片损失最大。
- **C4 第三**：检索侧 avg 3.47 条 memory、非空率 0.9949，信息量低于 C1 的 topk=10 混合检索，hist/latest/conflict 三切片都在 0.24–0.25。
- **C2 最差**：压缩把 episode/retrieval 级事实保留率从 1.0000 压到 0.6125，构成 EM 的硬天花板（详见 C2 消融记录）。
- conflict 切片排序：Ours 0.4985 > C1 0.3953 > C3 0.3632 > C4 0.2531 > C2 0.1128。

## Ours 执行备注

- 8 卡 8 分片（GPU 2–9，`probe/ours_probe.py --limit 34963 --tag ours_full --shard k --shards 8`，按 qid 轮转切分）。每分片约 4,370 条，单卡单进程（in-process HF ICAE Mistral，非 vLLM），故必须一卡一分片。
- 启动方式：写 `/root/.gpu_occ/exclude`=2..9 → 逐卡 kill `gpu-reserve-{2..9}` 后**立即**起分片（用户要求"关掉必须马上挂起，防被抢"）。日志 `/root/ours_launch.log`、`probe/runs/full_gpu{2..9}.log`。
- 首次启动 8 分片全挂（`ModuleNotFoundError: peft`）→ `uv pip install --no-deps peft==0.14.0` 后重启，全部存活。
- 06:0x 起跑 → 07:53–07:54 UTC 八分片全部写完，墙钟约 1 h 50 min，≈1.9 s/query/shard。平均 prompt 202.4 tokens、平均 reader 0.415 s/query。
- 分片落盘 `probe/runs/ours_full_34963_shard0..7.jsonl`（4,371/4,371/4,371/4,370/4,370/4,370/4,370/4,370），合并 34,963 unique，missing 0 / extra 0。

## C3 备注（原样保留 captain note）

TempLAMA episodes 极短（4,057 episodes 全部 ≤5,120 tokens，max 423，mean 87.3；0 evictions，全部单 segment）：1,024-slot 预算与 segmentation/eviction 逻辑在本数据集上从不触发，每个 ICAE-Full 行实际是对全历史的单次 128-slot 压缩，远未达到官方 26–32K 长上下文工作点。协议按 contract 冻结，此事实记录在案。

## C4 执行备注

- 6 卡 6 分片（GPU 2/3/4/7/8/9，端口 8010–8015）；GPU 5/6 被宿主 namespace 外的进程占 9.2 GB，无法回收，故 N=6 而非 8。哈希分片，覆盖不受影响。
- qdrant local store 为排他锁，每分片独立副本 `qdrant_data_full_s0..s5`；写入侧（gpt-4o-mini）与 as_of 无关，复用既有 collection，未重写。
- 16:25 UTC 首次启动，16:38 因外部误释放 GPU 中断（已写入 580 行保留），16:40 重启，16:59 起 resume，20:29–20:39 六分片全部 `done`。有效计算 3 h 40 min，≈2.3 s/query/shard。
- resume 后 reconciliation：34,963 unique / missing 0 / extra 0；各 shard `_errors.json` 保留中断窗口的 8 条历史记录，对应 query 已在 resume 中重跑并落行。

## GPU 释放（用户硬性要求，已完成）

- C4 结束（20:39 UTC）后 kill 6 个 vLLM api_server（端口 8010–8015），GPU 2/3/4/7/8/9 立即回到 0 MiB；`/root/.gpu_occ/exclude` 清空；约 4 分钟后 `gpu-reserve-{2,3,4,7,8,9}` 重新占卡。
- Ours 结束（07:54 UTC）后 8 个 `ours_probe.py` 进程自然退出（残留进程数 = 0），`/root/.gpu_occ/exclude` 已清空。
- 07:54 UTC 之后复查：GPU 3/9 空闲 0 MiB；GPU 1/2/4/5/6/7/8 各约 9.1–9.7 GB 被**宿主 namespace 外的其他会话**占用（pid 不在本容器内，非我方），GPU 0 两个外部进程共 18 GB。我方全程未触碰 0/1。
- `gpu_occupier` supervisor（pid 22363）在 07:57 UTC 后停止（`supervisor.log` 末行为 07:57，`state.json` 仍指向已消失的 `gpu-reserve-1`）。属外部因素，非本次评测动作；如需继续占卡需重新拉起。

## 代码回写

- C2 端口映射（`PORTS = {0:8001, 1:8006, 2:8007}`）已同步本地并提交。
- C4：`runner.py` / `closeout_full.py` 的 csr_v1 对齐（protocol 字段、conflict/latest 切片、`runs_csr_v1/` 输出、resume 守卫）已提交；`closeout_full.py` 分片数由硬编码 4 改为接受任意分片数（覆盖由 qid reconciliation 保证）。
- C1/C2/C3 其余 eval 脚本与本地 HEAD 一致（md5 核对）。
