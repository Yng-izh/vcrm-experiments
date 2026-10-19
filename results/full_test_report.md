# csr_v1 全量正式评测对照报告（34,963 test queries）

日期：2026-10-20 ｜ 协议：csr_v1（Controlled State Routing rev1，冻结于 481 diag）｜ 提取规则：ex28（主表 extracted view）
读者模型：Mistral-7B-Instruct-v0.2（greedy / max128 / seed42，三 shard 并发 16/instance）
数据：TempLAMA test 34,963 queries / 13,540 events；denominators（contract §5）：hist 30,906 / latest 4,057 / conflict 34,407（metadata 定义）

## 主表（extracted view, ex28）

| 方法 | overall EM | overall F1 | hist EM | hist F1 | latest EM | latest F1 | conflict EM | conflict F1 |
|---|---|---|---|---|---|---|---|---|
| C1 Hybrid RAG | **0.3906** | **0.6003** | **0.3810** | **0.5909** | **0.4639** | **0.6724** | **0.3953** | **0.6024** |
| C2 LLMLingua-2 RAG | 0.1123 | 0.3396 | 0.1106 | 0.3356 | 0.1260 | 0.3705 | 0.1128 | 0.3388 |
| C3 ICAE-Full | 0.3624 | 0.5581 | 0.3746 | 0.5701 | 0.2694 | 0.4674 | 0.3632 | 0.5551 |

数值来源：C1/C2 由各自 `c1/c2_test_summary.json` 冻结指标核对一致；overall/latest-F1/conflict 列由 jsonl 逐行重算（与 summary 双核一致）。C3 由 8 shard jsonl（c3full_r0–r7，共 34,963 行）合并重算，与各 shard summary 一致。

## 一致性检查（全部通过）

- 三份 jsonl 各 **34,963 行**；qid 集合完全一致（逐行 set 相等）。
- protocol=csr_v1：三份均 100%（C3 resume 已拒绝非 csr_v1 旧行）。
- 空 answer 行：C1/C2/C3 均为 **0 行**，无异常堆积。
- conflict 口径统一：以 metadata 定义（同 state_key 存在 ≥2 个不同 answer set）计 34,407；三方法同分母。

## C3 备注（原样保留 captain note）

TempLAMA episodes 极短（4,057 episodes 全部 ≤5,120 tokens，max 423，mean 87.3；0 evictions，全部单 segment）：1,024-slot 预算与 segmentation/eviction 逻辑在本数据集上从不触发，每个 ICAE-Full 行实际是对全历史的单次 128-slot 压缩，远未达到官方 26–32K 长上下文工作点。协议按 contract 冻结，此事实记录在案。

## GPU 释放（用户硬性要求）

- 释放前：vLLM×3（GPU2/3/4，各 ~27.8G）+ eval shard（GPU5–9）；GPU2–9 全部被占。
- kill 三个 vLLM api_server 进程（端口 8001/8006/8007）后：GPU2/3/4/7/8/9 = 0 MiB。
- `/root/.gpu_occ/exclude` 已清空（0 字节），supervisor（PID 22363，interval 300s）在下轮扫描自动重新占卡。
- 约 5.5 分钟后复查：GPU2/3/4/7/8/9 各 ~31,776 MiB，占卡进程为 `gpu-reserve-{2,3,4,7,8,9}`（父进程 supervisor 22363）——**占卡已恢复**。GPU0/1/5/6 为他人进程，未触碰。

## 代码回写

服务器侧唯一改动为 C2 端口映射（`PORTS = {0:8001, 1:8006, 2:8007}`），已同步本地 repo 并提交 push。C1/C2/C3 其余 eval 脚本与本地 HEAD 已一致（md5 核对），无需额外提交。
