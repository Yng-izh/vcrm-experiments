# Architecture v6 实验结果日志

## 2025-10-16 更新：E4 完成，ablations 四项全部出齐

### 状态总览（val 481 题，selection 用点估计，不代表统计显著）

| 实验 | 配置改动 | 支持 (support) | 关键 delta vs 对照 |
|---|---|---|---|
| E1 | decode_conditioned | ✅ true | all_compressed F1 +0.0165，conflict F1 +0.0316（CI 跨 0） |
| E2 | codec_variant=salience | ✅ true（弱） | all_compressed F1 +0.028，acc +0.0395（CI 全跨 0） |
| E3 | codec_value_path=full | ✅ true（强） | all_compressed F1 +0.3920 [0.3261,0.4593]，conflict F1 +0.5104 |
| E4 | E3 + contrastive_weight=0.2 | 待 compare 生成 | vs E3 (c_best)：F1 +0.0025，acc +0.0041 |

### E4 结果（vs c_best = E3）

- historical F1：0.9262（E3 0.9237，delta +0.0025）
- accuracy：0.7900（E3 0.7859，delta +0.0041）
- wrong_version_rate：0（E3 同为 0）
- 判定预期：点估计为正，但 delta 远小于 E2 量级，CI 必然跨 0；属"弱正"。
- 对最终合并的影响：`final_contrastive_weight = .2 if e4.support else 1.0`。
  若判 support=true，CFinal 用 0.2；否则 1.0。两种取法对期望结果差异均在噪声内。

### 当前推进状态

- E4_train：完成（5 epoch，2954 对，手动单卡 GPU1 续跑；此前失败原因为启动时 GPU 被占的 idle 检查）。
- E4_val：完成（codec-eval + codec-summary，481 题全量，summary complete）。
- E3_test：曾中断在 172/4057 episodes（partial 结果 acc 0.8657 / F1 0.9321，uncompressed 对照 acc 0.4768 / F1 0.6250），可按 episode 存在性续跑，未重启。
- 下一步：生成 E4 compare 判定 → selection_decision.json → final 阶段（CFinal/DFinal）→ test 阶段。

### 数据与防泄漏

- 官方 split 未重分配；train 与 test 的 query、entity 重叠均为 0（entity_split_audit.json）。
- 训练实体中与 test 重叠的 27 个 train、14 个 val 实体已剔除。

### 本仓库收录内容

- `vcrm/`、`run_architecture.py`、`run_paper.py` 等全部代码（与 v100-2 `/share/latent2/vcrm_icae` 同步）。
- `results/architecture_v6/architecture_v6_suite/`：E1/E2/E3 判定 JSON、plan、baseline_table、configs。
- `results/architecture_v6/architecture_v6_{E1..E4,D0,C0}_*/`：各 run 的 manifest、summary、logs、audit JSON（不含模型权重与 episode 存储）。
- 模型权重（*.safetensors，每个 ~58MB）与 stores/ 不入库，保留在 v100-2。
