# p3_100ep — 100-episode 自动关联损失探针（真实 E3，recency_update 开）

## 运行信息

| 项 | 值 |
|---|---|
| 主机 | v100-2，`/share/latent2/vcrm_icae` |
| 日期 | 2026-10-21 |
| 脚本 | `probe/run_auto_association_loss.py`（本次新增）+ `probe/run_state_eval.py` |
| 数据 | `data/prepared/templama_test.json`，stride=40 等距抽样 100 episode |
| 规模 | 100 episode / 830 queries / 100 条 latest |
| 配置 | E3（`codec_value_path=full`）、`time_mode=window_end`、`recency_update=on`、`top_k=5`、`neighbours=4` |
| 执行 | 4 分片并行，卡 3/9/6/7（`--shards 4`），最后 `--merge` 合并 |

## 结论

| 指标 | predicted | oracle_pooled | oracle_global |
|---|---|---|---|
| CREATE / UPDATE / DEFER | 95 / 26 / 339 | 500 / 1195 / 0 | 同左 |
| latest EM（端到端） | **0.1700** | 0.8700 | 0.8700 |
| 端到端 EM | **0.2265** | 0.5325 | 0.5325 |
| answer coverage | 0.6060 | 1.0000 | 1.0000 |
| association_rate（latest） | 0.5000 | 1.0000 | 1.0000 |
| false_split_rate（整池） | **0.6458** | 0.0000 | 0.0000 |
| rule5b 生效（`exclusive_superseded_by_recency`） | 26 | — | — |

`oracle_pooled` 与 `oracle_global` 每项指标相同 → 金标身份下池结构无影响，配对 delta 干净。
`recency_update` 只产出 26 次 UPDATE，`explicit_replacement` 为 0。
DEFER 的主要来源是 `unresolved_identity`（抽取器解析不出 subject/attribute），不是 resolver 拒绝覆盖。

## 文件

- `p3_100ep_report.md` / `.json` — 合并后报告（全量指标 + latest 损失分解 + delta）
- `p3_100ep_bundle_s{0..3}_{mode}.json` — 每个分片每个 pass 的 counters + 逐题明细，
  `--merge` 直接由这些文件重算报告
- `logs/` — 全部进程日志，含首轮卡 3/5 被抢占导致 CUDA OOM 的两份（`s0_gpu3`、`s2_gpu5`）
  与 2 分片试跑的两份（`shard0_gpu5`、`shard1_gpu8`，已中止）

## 已知缺陷与不可引用项

1. 分片跑时官方 per-pass 文件名不含 shard，四个分片互相覆盖，
   `*.scores.jsonl` / `*.decisions.jsonl` / `*.summary.json` 只剩最后写入者，
   因此未纳入本目录；报告数字全部来自 bundle，不受影响。修复方式：stem 里加 shard。
2. `latest version accuracy`（0.0339 vs 0.2800）不可引用：latest 查询的金标时间是最后一年，
   而 events 只在答案变化时生成，`window_end` 常早于最后一年，版本号结构性对不上。EM 有效。
3. `query→state recall@1`（0.8211）不是质量信号，其定义是"top-1 == 该金标的主预测状态"；
   真实关联正确率看 `association_rate`（latest 上为 0.50）。
4. `history_version_accuracy` 在 `window_end` 口径下结构性为 0，不作结论。

复现命令见 `probe/run_auto_association_loss.py` 文件头。
