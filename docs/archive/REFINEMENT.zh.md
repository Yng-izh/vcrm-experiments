# C16 / D16 五轮重训

入口：`.venv/bin/python run_paper.py --plan config.refine.json --phase refine --gpus 0,2`。

旧实验已暂停并保留。仅训练 C16/D16，不启动 C32/D32、完整测试集或 LongMemEval。

## 修改范围

- 原始 TempLAMA 数据、特征、训练配对、冻结 ICAE Reader 和固定关键帧策略不变。
- 相同 CE + teacher KL + sequence unlikelihood + pooled alignment + index loss；C/D 均添加权重 1、margin 1.5 的当前错误答案对比项。
- 对比项按答案文本的平均 token log-probability 计算，排除 EOS；排除与历史合法答案重叠的负答案，多个负别名取平均。
- D 训练时对解码端固定关键帧随机置零 25% 的完整槽位；编码端不丢弃，评测不丢弃，不增加参数。
- 5 epochs，每轮同种子打乱；AdamW 峰值 LR 1e-4，5% optimizer-step warmup + cosine decay；累积 4、裁剪 1。
- 每轮保存检查点、各损失分量、固定 200 道训练冲突历史题的生成结果、F1 与严格误答当前率。

## 口径

训练优化仍使用原始 1,711 个相邻版本配对：current 是下一次观测更新，不是全历史的最后版本。对比负答案与原 sequence unlikelihood 的下一次更新答案一致。D 的条件是不可变关键帧，不是最新版本，也不是直接相加的残差重构。

相邻配对仅有 118 道答案集合完全不相交的冲突题，无法直接取 200 道。固定 200 题生成诊断仍来自同一训练集、原压缩历史目标，但与验证一致注入最终当前版本。它不参与 optimizer，不改变训练配对，不增加或重复数据；属于训练集内诊断而非独立验证。

验证固定原先共享前 2,185 道题中的 481 道实际读取压缩码的历史题，其中 262 道当前与历史答案集合完全不相交。题单与数据哈希冻结，仍写入全部原始事件。它是用于本次调参的 post-hoc validation diagnostic，不是未触碰测试集。

输出位于 `runs/refine_v2_*`。`refine_v2_suite/comparison.json` 给出原 1-epoch 和新 5-epoch 的全 481 / 冲突 262 指标及按实体关系组聚类的 D-C paired bootstrap CI。

C/D 的监督和优化器相同，但只有 D 使用关键帧 dropout。因此本轮对比包含这一方法特定正则化，不能宣称是仅结构差异的严格消融。欠拟合、参数先验与 D 的优势都需实测，不能保证 loss 必然低于某个数或 F1 必然上升。
