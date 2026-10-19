# smoke20 三方法对照报告（gate 检查，非正式成绩）

日期：2026-10-19 ｜ 服务器 v100-2 ｜ 同一 qid 集：runs_csr_v1_meta/smoke20_qids.json（20 个，seed42 hist16+latest4，全部属于 templama_test.json）｜ protocol=csr_v1

## 汇总对照（em_extracted / f1_extracted）

| 方法 | hist EM | hist F1 | latest EM | latest F1 | overall EM | overall F1 | mean_prompt_tokens |
|------|---------|---------|-----------|-----------|------------|------------|--------------------|
| C1 hybrid RAG     | 0.3125 | 0.5900 | 0.00 | 0.3667 | 0.25 | 0.5453 | 197.7 |
| C2 LLMLingua-2    | 0.1875 | 0.4201 | 0.25 | 0.4936 | 0.20 | 0.4348 | 145.5 |
| C3 ICAE full      | 0.3125 | 0.5144 | 0.00 | 0.4500 | 0.25 | 0.5015 | 188.95 |

## Gate 检查（目的：验证管线正确性，非成绩）

- qid 对齐：c1/c2/c3 三文件 query_id 集合 == smoke20_qids.json，各 20 行。PASS
- protocol=csr_v1：60/60 行。PASS
- As-of 年份注入：60/60 行 query_text 含 "As of" 前缀。PASS
- Episode 范围：C1/C2 retrieved_event_ids 全 20 行逐行核对属于对应 episode（t2 验收）；C3 走 recorded-only 路由，cache_hit_rate=1.0，AST 审计 zero hits，仅 reader.answer 官方协议。PASS
- 输出隔离：所有写入均在各自 runs_csr_v1/；四个 runs_pre_csr_superseded/ 目录 newest mtime（最新 2026-10-19 10:10:26）早于 smoke 运行，未被触碰。PASS
- 异常行：无（C2 首轮 Errno 99 行为 shard 端口配置问题，C2_SHARDS=1 复跑后 20/20 正常，未进入最终 c2_smoke20.jsonl）。

## 结论

smoke20 通过 gate：三方法在同一 20 qid 上完成 dryrun 级验证，年份注入、episode 检索范围、官方 Reader 均生效。C1/C3 overall EM 0.25、C2 0.20；样本量 20，数字不具统计意义，不作为正式成绩。

## 备注

- C3 首跑因 run_probe_full.py:114 事件级断言误禁 state_key 全数报错，已做最小修复（仅禁 answers/latest/as_of；state_key 存在于源数据但从不读取），重跑成功；修复已回写本地 clone 并 push GitHub（e1bc70a）。
- C2 shard 配置：smoke 单卡环境需 C2_SHARDS=1（读端全走 :8001）。
