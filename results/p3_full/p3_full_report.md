# P3 自动关联损失探针：p3_full

- git: `None`
- 数据: `templama_test`，抽样 stride（stride=1），episode=None，neighbours=4，top_k=5
- recency_update=on，time_mode=window_end，arch=E3，shards=7
- components: `icae+checkpoint:/share/latent2/vcrm_icae/runs/architecture_v6_E3_train`

| 指标 | predicted | oracle_pooled | Δ(oracle_pooled−predicted) |
| --- | --- | --- | --- |
| CREATE | 4055 | 20285 | n/a |
| UPDATE | 8896 | 47435 | n/a |
| DUPLICATE | 564 | 0 | n/a |
| DEFER | 3614 | 0 | n/a |
| update:explicit_replacement | 2 | 0 | n/a |
| update:exclusive_superseded_by_recency | 3576 | 0 | n/a |
| query→state recall@1 | 1.0000 | 1.0000 | 0.0000 |
| query→state recall@3 | 1.0000 | 1.0000 | n/a |
| state pairwise P | 1.0000 | 1.0000 | n/a |
| state pairwise R | 0.9999 | 1.0000 | n/a |
| state pairwise F1 | 0.9999 | 1.0000 | n/a |
| false_merge_rate（目标 episode 口径） | 0.0000 | 0.0000 | n/a |
| false_merge_rate（整池口径） | 0.0000 | 0.0000 | n/a |
| false_split_rate（整池口径） | 0.0001 | 0.0000 | n/a |
| latest EM（条件口径） | 0.8087 | 0.8280 | 0.0192 |
| latest EM（端到端口径） | 0.8077 | 0.8280 | 0.0202 |
| latest version accuracy | 0.2673 | 0.2761 | n/a |
| history version accuracy | 0.0962 | 0.0950 | n/a |
| EM（条件口径） | 0.4537 | 0.4575 | 0.0038 |
| F1（条件口径） | 0.5647 | 0.5661 | n/a |
| 端到端 EM | 0.4530 | 0.4575 | 0.0045 |
| 端到端 F1 | 0.5639 | 0.5661 | n/a |
| answer coverage | 0.9985 | 1.0000 | 0.0015 |
| 平均读取秒 | 0.3850 | 0.3909 | n/a |
| 平均 latent slots | 147.4015 | 147.7138 | n/a |
| wall seconds | 34530.0739 | 31678.5605 | n/a |

## latest 损失分解（关联 / 版本 / 读取）

| 指标 | predicted | oracle_pooled |
| --- | --- | --- |
| n_latest | 4057 | 4057 |
| association_ok | 4052 | 4057 |
| association_miss | 5 | 0 |
| association_rate | 0.9988 | 1.0000 |
| em_all | 0.8077 | 0.8280 |
| em_given_association_ok | 0.8087 | 0.8280 |
| version_accuracy_given_association_ok | 0.2673 | 0.2761 |
| em_given_version_ok | 0.8246 | 0.8223 |
| association_attributable_loss_upper | 0.0010 | 0.0000 |

## 损失结论

- 配对口径：oracle_pooled（同池、同窗口口径、同读取路径，只换状态身份来源）
- latest EM 端到端损失：0.0202（相对 0.0244）
- 端到端 EM 损失：0.0045
- 关联可解释的上限：0.0010

## 读表约束

- predicted 是唯一可当自动关联证据的 pass；oracle_* 只是上界。
- em/f1 为条件口径；跨 pass 比较请用 end_to_end_em / latest.em_end_to_end。
- window_end 口径下 history_version_accuracy 结构性为 0，不作结论。
- recency_update=on 时 UPDATE 才可能形成多版本；update_source_counts 里 exclusive_superseded_by_recency 的数量说明 rule5b 实际生效多少条。
- stub reader 的 EM 无语言意义（plumbing_stub=True 时不得引用任何分数）。
