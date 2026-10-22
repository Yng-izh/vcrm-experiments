# 实验一：Pair vs 全轨迹读取（traj_v1_joint）

- git: `None` | reader: `icae` | components: `icae+checkpoint:runs/trajectory_v1_joint/published` | citable=True
- 数据: `templama_val`，题目来源=constructed，题数=73，identity=gold，neighbours=4，版本数 3–8，max_slots=256
- 可见性截止：episode window_end (max event time)；future fallback 已强制关闭

| 指标 | A: pair | B: trajectory | Δ(B−A) |
| --- | --- | --- | --- |
| 题数 | 73 | 73 | n/a |
| 跨版本题准确率（主指标） | 0.3836 | 0.3973 | 0.0137 |
| 严格 EM | 0.2877 | 0.2603 | -0.0274 |
| 所需版本全在 bundle 内 | 0.2740 | 1.0000 | 0.7260 |
| 历史答成 current 的比例 | 0.1515 | 0.0606 | -0.0909 |
| （该指标分母） | 33 | 33 | n/a |
| 平均读取 slots | 148.0000 | 193.6164 | 45.6164 |
| 平均轨迹长度 | 2.0000 | 4.5342 | 2.5342 |
| 平均读取秒 | 0.8417 | 0.6908 | -0.1509 |
| 未来 current 被挡次数 | 0 | 0 | n/a |
| 被挡未来版本总数 | 0 | 0 | n/a |

## 分族准确率

| family | n | A: pair | B: trajectory | Δ | pair 可达性 | traj 可达性 |
| --- | --- | --- | --- | --- | --- | --- |
| compare_years | 20 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 1.0000 |
| count_values | 20 | 0.0000 | 0.1000 | 0.1000 | 0.0000 | 1.0000 |
| ordinal_back | 13 | 0.0000 | 0.2308 | 0.2308 | 0.0000 | 1.0000 |
| recall_year | 20 | 0.9000 | 0.7000 | -0.2000 | 1.0000 | 1.0000 |

## 配对结论

- 配对题数：73
- trajectory 胜 / pair 胜 / 平：6 / 5 / 62
- 平均准确率差（B − A）：0.0137，按 family 聚类的 bootstrap 95% CI：[-0.1250, 0.1525]
- **判定：trajectory_better**

分族配对：

| family | n | A | B | Δ | 判定 |
| --- | --- | --- | --- | --- | --- |
| compare_years | 20 | 0.5000 | 0.5000 | 0.0000 | tie |
| count_values | 20 | 0.0000 | 0.1000 | 0.1000 | trajectory_better |
| ordinal_back | 13 | 0.0000 | 0.2308 | 0.2308 | trajectory_better |
| recall_year | 20 | 0.9000 | 0.7000 | -0.2000 | pair_better |

## 读表约束

- 这些是从 TempLAMA validation 事件版本链构造的受控诊断题，不是官方分数，不得用于声称开放域泛化。
- 两臂共用同一事实池、同一状态身份、同一 cutoff、同一题面、同一 pair 选择上界，唯一变量是 read_state 的 read_mode。
- recall_year 的题面自带年份，pair 用年份即可选对版本 —— 这一族是"trajectory 会不会被多版本干扰"的探针；ordinal_back / count_values / compare_years 结构上必须同时看到多个版本。
- all_required_in_bundle_rate 是可达性：pair 只装 2 个版本，跨版本题的金标版本可能根本不在 bundle 里；accuracy 的差异必须和这一列一起读。
- wrong_current_answer_rate 只在取值型 family（recall_year / ordinal_back）且 current 取值≠金标的题上算，分母见 wrong_current_applicable。
- 定长 latent 预算下 reader 永远会输出一个答案，答出率恒为 1，故不报噪声率。
- stub reader 的分数无语言意义（plumbing_stub=True 时不得引用任何分数）。
