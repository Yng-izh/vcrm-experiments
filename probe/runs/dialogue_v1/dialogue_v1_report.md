# 自然对话端到端评测（LongMemEval-S）

- components: `icae+checkpoint:/share/latent2/vcrm_icae/runs/architecture_v6_E3_train` | 题数=20（判分成功 0，judge 失败 0） | 类型过滤=['temporal-reasoning', 'knowledge-update', 'multi-session']

| 类型 | n | pair | trajectory | Δ |
| --- | --- | --- | --- | --- |

- 配对：trajectory 胜 0 / pair 胜 0 / 平 0（仅计 judge 成功的题）
- 两臂答案一致率：1.0

## 口径

- 端到端口径：状态身份由管线自己预测，不是金标写池；结论属于"管线整体"。
- 判分用官方 LongMemEval prompt + 自定义 judge（非官方 GPT-4o 分数）。
- LongMemEval-S 问题多不依赖版本更替；这里量的是"对话流入记忆后还能不能读出来"。
