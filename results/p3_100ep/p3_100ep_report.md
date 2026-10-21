# P3 自动关联损失探针：p3_100ep

- git: `None`
- 数据: `templama_test`，抽样 stride（stride=40），episode=None，neighbours=4，top_k=5
- recency_update=on，time_mode=window_end，arch=E3，shards=4
- components: `icae+checkpoint:/share/latent2/vcrm_icae/runs/architecture_v6_E3_train`

| 指标 | predicted | oracle_pooled | oracle_global | Δ(oracle_pooled−predicted) |
| --- | --- | --- | --- | --- |
| CREATE | 95 | 500 | n/a | n/a |
| UPDATE | 26 | 1195 | n/a | n/a |
| DUPLICATE | 0 | 0 | n/a | n/a |
| DEFER | 339 | 0 | n/a | n/a |
| update:explicit_replacement | 0 | 0 | n/a | n/a |
| update:exclusive_superseded_by_recency | 26 | 0 | n/a | n/a |
| query→state recall@1 | 0.8211 | 1.0000 | 1.0000 | 0.1789 |
| query→state recall@3 | 0.9563 | 1.0000 | 1.0000 | n/a |
| state pairwise P | 1.0000 | 1.0000 | 1.0000 | n/a |
| state pairwise R | 0.3558 | 1.0000 | 1.0000 | n/a |
| state pairwise F1 | 0.5248 | 1.0000 | 1.0000 | n/a |
| false_merge_rate（目标 episode 口径） | 0.0000 | 0.0000 | 0.0000 | n/a |
| false_merge_rate（整池口径） | 0.0000 | 0.0000 | 0.0000 | n/a |
| false_split_rate（整池口径） | 0.6458 | 0.0000 | 0.0000 | n/a |
| latest EM（条件口径） | 0.2881 | 0.8700 | 0.8700 | 0.5819 |
| latest EM（端到端口径） | 0.1700 | 0.8700 | 0.8700 | 0.7000 |
| latest version accuracy | 0.0339 | 0.2800 | 0.2800 | n/a |
| history version accuracy | 0.1284 | 0.0986 | 0.0986 | n/a |
| EM（条件口径） | 0.3738 | 0.5325 | 0.5325 | 0.1588 |
| F1（条件口径） | 0.4769 | 0.6168 | 0.6168 | n/a |
| 端到端 EM | 0.2265 | 0.5325 | 0.5325 | 0.3060 |
| 端到端 F1 | 0.2890 | 0.6168 | 0.6168 | n/a |
| answer coverage | 0.6060 | 1.0000 | 1.0000 | 0.3940 |
| 平均读取秒 | 0.3734 | 0.3338 | 0.3178 | n/a |
| 平均 latent slots | 81.9494 | 147.6964 | 147.6964 | n/a |
| wall seconds | 415.4402 | 659.5643 | 371.7063 | n/a |

## latest 损失分解（关联 / 版本 / 读取）

| 指标 | predicted | oracle_pooled | oracle_global |
| --- | --- | --- | --- |
| n_latest | 100 | 100 | 100 |
| association_ok | 50 | 100 | 100 |
| association_miss | 50 | 0 | 0 |
| association_rate | 0.5000 | 1.0000 | 1.0000 |
| em_all | 0.1700 | 0.8700 | 0.8700 |
| em_given_association_ok | 0.3000 | 0.8700 | 0.8700 |
| version_accuracy_given_association_ok | 0.0400 | 0.2800 | 0.2800 |
| em_given_version_ok | 0.5000 | 0.7857 | 0.7857 |
| association_attributable_loss_upper | 0.1500 | 0.0000 | 0.0000 |

## 损失结论

- 配对口径：oracle_pooled（同池、同窗口口径、同读取路径，只换状态身份来源）
- latest EM 端到端损失：0.7000（相对 0.8046）
- 端到端 EM 损失：0.3060
- 关联可解释的上限：0.1500

## 读表约束

- predicted 是唯一可当自动关联证据的 pass；oracle_* 只是上界。
- em/f1 为条件口径；跨 pass 比较请用 end_to_end_em / latest.em_end_to_end。
- window_end 口径下 history_version_accuracy 结构性为 0，不作结论。
- recency_update=on 时 UPDATE 才可能形成多版本；update_source_counts 里 exclusive_superseded_by_recency 的数量说明 rule5b 实际生效多少条。
- stub reader 的 EM 无语言意义（plumbing_stub=True 时不得引用任何分数）。
