# 历史目标覆盖与上下文匹配蒸馏：C16 / D16

运行：`.venv/bin/python run_paper.py --plan config.matched.json --phase refine --gpus 0,2`。

## 本轮仅修改两项

- `final_current_snapshot_v1` 覆盖每个 Episode 中所有有历史 QA 的非最终观察版本，包括初始版本、旧关键帧位置和双版本 Episode。每个版本选一条已有 QA：优先观察时间对应的 QA，其次最早的有效 QA。不复制重复年度问题，不改变原始数据或答案。
- Teacher 和 Student 均按 current / historical 顺序注入，使用同一组角色与时间标记、相同问题和答案前缀。Teacher 读取 current 128 + historical 128 + 4 标记槽位；Student 读取 current 128 + historical 16 + 4 标记槽位。Teacher 的记忆、标记与 logits 均停止梯度，梯度仍只训练 Student 的 Codec 和标记等原有模块。
- 旧 `immutable_keyframe_v1` 的目标筛选与默认单条 Teacher 保持不变，新配置显式启用这两项修复。

## 不变项

沿用官方 ICAE checkpoint、冻结 Encoder / Decoder、原始 TempLAMA 数据与 128 槽位特征缓存。C/D 容量、损失项和权重、seed=42、每轮 Shuffle、LR=1e-4、5% Warmup + Cosine、梯度累积 4、clip=1、零锚点 Dropout 保持一致。C 不读取当前锚点进行压缩；D 编解码均条件于同一最终当前版本。

修复前有 1,711 个目标；完整训练集的新目标预期为 2,954 个，含 299 个双版本 Episode。5 轮共 14,770 次样本处理、3,695 次 optimizer 更新，启动时以 `training_input_audit.json` 为准。训练样本扩大、Teacher 上下文变长，不能沿用上轮的耗时估计。

固定 200 道训练冲突题沿用 `aligned_v3_C16_train` 的顺序与完整 pair 内容哈希，不从扩大的池中重新抽取。每轮结束做一次该训练诊断并保存检查点。完成 5 轮后自动评测原 481 道压缩历史验证题及未压缩双版本控制，记录 F1、EM、误答当前率与配对置信区间。这是已参与调参的验证诊断，不是未触碰测试集。

新输出使用 `runs/matched_v4_*`，从头初始化，保留所有旧结果与修复前源码归档。不运行 C32/D32、完整 test 或 LongMemEval。该协议仍是离线最终当前快照，不声称支持在线重编码或有损码重绑定。

这轮同时修复覆盖和 Teacher 上下文，前后结果只能支持联合修复的效果，不能单独证明某一修改的因果贡献，也不预设 D 必然优于 C。
