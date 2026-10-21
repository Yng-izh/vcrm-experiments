# p3_full — 全量 TempLAMA 自动关联损失探针（真实 E3，recency_update 开）

## 运行信息

| 项 | 值 |
|---|---|
| 主机 | v100-2，`/share/latent2/vcrm_icae` |
| 日期 | 2026-10-21 |
| 脚本 | `probe/run_auto_association_loss.py` + `probe/run_state_eval.py` |
| 数据 | `data/prepared/templama_test.json`，全量 **4057 episode / 34963 query** |
| 配置 | E3（`codec_value_path=full`）、`time_mode=window_end`、`recency_update=on`、`top_k=5`、`neighbours=4` |
| 执行 | 7 分片并行，卡 2/3/4/6/7/8/9（`--shards 7`，错开 45 秒启动），最后 `--merge` 合并 |
| 代码 | commit `ed1d2b3` |

## 主结果

| 指标 | predicted | oracle_pooled | 差值 |
|---|---|---|---|
| 事实进入状态层比例 | **78.90%**（13515/17129） | 100%（67720/67720） | −21.10pp |
| 错误拆分率（整池 pair） | **0.01%** | 0.00% | +0.01pp |
| 错误合并率（整池 pair） | **0.00%** | 0.00% | 0 |
| latest 关联正确率 | **99.88%** | 100% | −0.12pp |
| latest 端到端 EM | **0.8077** | 0.8280 | −0.0202 |
| 端到端 EM | **0.4530** | 0.4575 | **−0.0045** |

配套指标：

| 指标 | predicted | oracle_pooled |
|---|---|---|
| answer coverage | 0.9985 | 1.0000 |
| 状态对级 precision / recall / f1 | 1.0000 / 0.9999 / 0.9999 | 1.0000 / 1.0000 / 1.0000 |
| tp / fn | 20621 / 3 | 20664 / 0 |
| 决策 | CREATE 4055 / UPDATE 8896 / DUPLICATE 564 / DEFER 3614 | CREATE 20285 / UPDATE 47435 |
| UPDATE 来源 | `snapshot_superseded` 5318、`exclusive_superseded_by_recency` 3576、`explicit_replacement` 2 | — |

配对损失（oracle_pooled − predicted）：端到端 EM **0.0045**、latest EM **0.0202**、条件 EM 0.0038、coverage 0.0015、query→state recall@1 0.0。

## 与 100 episode 对照

| 指标 | 修前 100ep（`fa05237`） | 修后 100ep | 修后全量 4057ep |
|---|---|---|---|
| 进入状态层 | 26.3% | 79.81% | 78.90% |
| 错误拆分（整池 pair） | 64.58% | 0.00% | 0.01% |
| latest 关联正确率 | 50.00% | 100.00% | 99.88% |
| latest 端到端 EM | 0.1700 | 0.8600 | 0.8077 |
| 端到端 EM | 0.2265 | 0.5325 | 0.4530 |

全量比 100ep 更难（端到端 EM 0.5325 → 0.4530），但 oracle 同步下降（0.5325 → 0.4575），两者差距仍只有 0.45pp —— 说明下降来自样本难度，不是关联回归。

## 本轮代码修正（commit `ed1d2b3`）

1. **抽取模板**：补 `plays for` / `holds the position of` / `attended` / `is owned by` / `is a member of`；`head_coach`、`chair` 改为**反向**关系（主体是球队/机构，人是值）；句号不再切碎人名首字母、缩写与小数；反向关系短语长优先匹配，回退按**最右**一个 `of` 切分。
2. **本体表**：新增别名；`head_coach`、`chair` 进 `CARDINALITY_EXCLUSIVE`；新增 `SNAPSHOT_SUPERSEDES`（可被年度快照覆盖的 time-scoped 属性，排除 `children` / `award_received` 等累积型多值）。
3. **Resolver rule5c**：带年份前缀的快照 + time-scoped 属性 + 观察时间严格更晚 → UPDATE，reason 前缀 `snapshot_superseded`，三个评测器都按来源分口径统计。
4. **读取侧**：只在规范化 `subject` + `attribute` 与解析结果一致的状态上读版本（原先会从"同属性、不同主体"的状态读答案）；掩码落在主语位置时走反向读法；`select_version` 增加 `allow_future_fallback` 与 `version_after_target_time` 审计标记。

## 已知缺陷与不可引用项

1. **版本级指标不可引用**：`window_end` 口径把所有查询的时间压成同一个值（2020-01-01），`select_version` 永远返回最新版本，历史题问的年份不进选择逻辑。实测 `history_version_accuracy` 0.0947（9/95），与 11 个候选年份的随机期望（8.6 条）无差别；`version_accuracy` 结构性为 0。端到端 EM 有效，版本级指标无效。
2. **`as_of` 不可直接启用**：TempLAMA 上 `query['time'] == query['as_of'] == 金标版本时间戳`，用它选版本等于把答案递给 `select_version`，`PredictedRuntime` 默认拒绝，只在受控对照里显式打开。
3. **21% 的 DEFER 未进状态层**（3614/17129）。当前指标未受影响，是因为这些状态已由同一状态链上的其它事实建立；换数据不保证同样情况。
4. **句式模板是 TempLAMA 适配**：模板由失败样本的句式反推得到，且样本来自测试集。它证明关联机制成立，不构成开放域事实抽取能力的证据；论文中需声明这一层过拟合风险。
5. **bundle 未入库**：14 个 bundle 合计约 500MB，仅保留在 `v100-2:/share/latent2/vcrm_icae/probe/runs/p3_full/`；本目录只放合并后的报告与日志。

## 复现

```bash
# 每个分片（7 分片，卡 2/3/4/6/7/8/9）
cd /share/latent2/vcrm_icae && CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=.:probe \
  env_py310/bin/python -u probe/run_auto_association_loss.py \
  --limit 4057 --sample stride --neighbours 4 --recency-update --top-k 5 \
  --arch-exp E3 --modes predicted,oracle_pooled --shard <i> --shards 7 \
  --out probe/runs/p3_full --tag p3_full

# 合并（CPU，无 GPU）
CUDA_VISIBLE_DEVICES= PYTHONPATH=.:probe env_py310/bin/python -u \
  probe/run_auto_association_loss.py --merge --out probe/runs/p3_full \
  --tag p3_full --modes predicted,oracle_pooled
```
