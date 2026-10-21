# Ours 口径分列：纯 latent vs 文本 fallback（csr_v1, 34,963 题）

日期：2026-10-21 ｜ 数据：`probe/runs/ours_v6_merged.jsonl`（md5 `59d4d16e941e233ceb2b4649ae32177e`，34,963 行）｜ 架构：architecture v6 / E3

## 1. 两口径主表

| 口径 | n | overall EM | overall F1 | hist EM/F1 | latest EM/F1 | conflict EM/F1 |
|---|---|---|---|---|---|---|
| 全覆盖（含 fallback） | 34,963 | 0.8512 | 0.9346 | 0.8527 / 0.9353 | 0.8398 / 0.9295 | 0.8603 / 0.9380 |
| 纯 latent（`route.path=latent`） | 33,085 (94.63%) | 0.8677 | 0.9444 | 0.8683 / 0.9444 | 0.8634 / 0.9451 | 0.8766 / 0.9479 |
| 文本 fallback（`route.path=fallback_text`） | 1,878 (5.37%) | 0.5612 | 0.7615 | 0.5827 / 0.7774 | 0.3788 / 0.6266 | 0.5719 / 0.7635 |

## 2. 协议一致性（已修复）

- 修复前：`no_association` 兜底直接调用全局 `retriever.retrieve()`，无 episode 过滤，检索库为 `c2_llmlingua2_rag/corpus_test_compressed.json`（LLMLingua-2 压缩语料）→ 与 csr_v1 的 episode-scoped retrieval 不一致，影响 5,611 题 / 16.05%。
- 修复后：`build_episode_retriever(ep)` + `fallback_retrieve(ep, question, t)`，检索集只含该 episode 自身未压缩事件，`time <= t` 过滤 top-10。
- 核验：1,878 条 fallback 行的 `fallback_events` 逐条核对，全部属于本 episode，越界 0 条；`fallback_scope` 全为 `episode`，`fallback_corpus` 全为 `full_text`。
- fallback 占比由 16.05% 降到 5.37%（关联成功率 83.95% → 94.63%，属 E3 带来的改善）。

## 3. latest 路由事实

- latest 4,057 题：3,859 走 latent 且 `route.is_current` **全为 True**（0 例外）；198 走 fallback（EM 0.3788）。
- 说明版本选择层在 latent 路径上没有选错现行版本，fallback 子集的误差主要来自跨版本文本混淆与 reader 截断。

## 4. 例外记录

- 2 行空 `extracted_answer`：`Q4961400_P54_2019`、`Q4961400_P54_2020`，均走 fallback，reader 输出被截断（"The blank _X_ in the statement can be fi…"）。待补跑。

## 5. 历史对照（C16，已 SUPERSEDED）

同一口径在 C16 方案 + 跨 episode 压缩 fallback 下的结果：全覆盖 0.4946/0.6363（34,963）、纯 latent 0.5424/0.6536（29,352）、fallback 0.2445/0.5459（5,611）。仅保留作 provenance，不作为正式成绩。
