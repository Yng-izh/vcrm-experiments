# ICAE C16 / D16 最终当前版本对齐实验

完成状态：训练与固定验证诊断均完成。服务器：v100-2；C16 使用 GPU 0，D16 使用 GPU 2。

## 主要结果

| 指标 | C16 | D16 |
| --- | ---: | ---: |
| 481 题历史 F1 | 55.07% | 51.56% |
| 481 题严格 EM | 41.58% | 37.63% |
| 481 题严格误答最终当前答案率 | 2.91% | 1.87% |
| 冲突 262 题历史 F1 | 44.79% | 42.65% |
| 冲突 262 题严格 EM | 32.44% | 30.92% |
| 冲突 262 题严格误答最终当前答案率 | 5.34% | 2.67% |
| 同题未压缩双版本对照 F1 | 66.24% | 68.74% |
| 最后一轮固定 200 道训练冲突题 F1 | 47.44% | 43.63% |
| 最后一轮固定 200 道训练冲突题严格 EM | 35.00% | 29.50% |
| 最后一轮固定 200 道训练冲突题误答当前率 | 1.50% | 0.50% |

481 题的 D-C F1 差异为 -3.51 个百分点，按 172 个实体关系 Episode 聚类的 paired bootstrap 95% CI 为 [-7.27, +0.34] 个百分点；区间包含零，不能声称统计显著劣于 C，也未证实 D 的优势。具体机制尚未确认，不能把“删除重要历史信息”当作已证实结论。

## 协议

- 原 TempLAMA 数据、官方 ICAE v2 checkpoint、冻结 Mistral Reader 与预存 128 槽位特征不变。
- 保留原 1,711 个历史训练目标；训练 current / latest QA / contrastive negative 统一为 Episode 最终当前版本。
- C 独立编码历史；D 的编码与解码均条件化于最终当前 128 槽位。两者历史码均为 16 槽位，参数匹配，损失、训练信息及 optimizer 预算相同，均无 anchor dropout。
- Reader 接收当前 128 + 历史恢复 16 + 4 个角色/时间槽位，共 148 latent slots，以及含 as-of 时间的问题；不接收原始事实文本或标准答案。
- 共 5 epochs、2,140 optimizer steps；AdamW peak LR 1e-4、5% warmup + cosine、gradient accumulation 4、clip 1、seed 42。
- 损失保留 CE、单条未压缩 teacher KL、sequence unlikelihood、pooled alignment、index loss；新增 mean-answer-token contrastive margin 1.5 / weight 1，排除 EOS 和历史合法别名。Teacher 单条上下文与 Student 双版本上下文并非完全匹配。
- 官方 checkpoint 是模型来源，不代表这是 ICAE 作者的官方 TempLAMA 评测协议或论文分数。

## 评测限制

固定 481 题来自此前验证诊断，其中 262 题当前/历史答案集合完全不相交；它是调参后的 post-hoc validation diagnostic，不是未触碰的 test。固定 200 题来自训练集，不是泛化评测。使用公开数据提供的 state_key / as-of 路由，不衡量自动检索或自动版本关联。

未压缩双版本对照用同一 Reader、相同选中版本及对应训练出的角色/时间标记，将历史 16 槽位替换为原始 128 槽位。它不是另一个正式方法，不是严格理论上界；C/D 标记不同，所以各自对照得分不同。

`final_current_snapshot_v1` 是离线快照协议：从全部原始完整特征构建快照，仅最终 current 保留 full，其余历史存 16 槽位码。D 的码绑定同一 current，读取复用 current，不额外计锚点。当前版本变化时需从原始完整特征重建，不支持增量更新或将有损码反复重压；本轮不证明 streaming 持久化能力。

原 1-epoch 与新 5-epoch 数据、版本条件、历史保留策略及训练目标不同，比较仅作为研究记录，不能隔离每项修改的因果收益。新轮全部 481 题 F1 并未超过原轮；冲突子集 F1 改善与 stale 大幅降低不能替代总体性能报告。

## 文件

- `runs/aligned_v3_suite/comparison.json`：原轮/本轮逐题统计及聚类置信区间。
- `runs/aligned_v3_suite/validation_481.json`：固定验证题单、标签、数据哈希和诊断来源。标签仅用于离线评分，不进入 Reader。
- `runs/aligned_v3_suite/configs/`：实际冻结运行配置。
- `runs/aligned_v3_C16_train/`、`runs/aligned_v3_D16_train/`：完整训练日志、loss 分量、每轮 200 题生成诊断、配置/数据/代码/checkpoint 哈希。
- `runs/aligned_v3_C16_val/`、`runs/aligned_v3_D16_val/`：481 题预测、未压缩对照输出、评分、写入统计及 roundtrip 检查。
- `runs/paper_v1_baseline_val/`：使用官方 checkpoint 的本项目单条读取 baseline 汇总与原门禁决定，仅为参考，不与 481 题直接做因果对比。
- `source_snapshot.tar.gz`：对应本轮实验的源码、测试、配置和说明。仓库根目录代码可能属于较早的工程状态；复现本轮请使用该快照。
- `artifact_manifest.json`：本目录所有发布文件的 SHA-256 / 字节数及未上传大型文件清单。

不上传 `.env`、API key、模型权重、每轮 Codec 权重、特征缓存及 `.memory` 文件。原文件保留于服务器 `/share/latent2/vcrm_icae/runs/aligned_v3_*`，权重哈希在训练元数据中。

## 执行入口

在源码快照对应环境、数据、特征缓存和原用户授权门禁准备完成后：

```sh
.venv/bin/python run_paper.py --plan config.aligned.json --phase refine --gpus 0,2
```

路径依赖服务器实际目录，数据和 ICAE/Mistral 权重需另行准备。源码快照同时包含已暂停的 `refine_v2` 配置用于研究记录，不要求运行它。
