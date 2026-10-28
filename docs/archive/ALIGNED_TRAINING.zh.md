# 最终当前版本对齐的 C16 / D16

运行：`.venv/bin/python run_paper.py --plan config.aligned.json --phase refine --gpus 0,2`。

## 修改

- 暂停并保留 `refine_v2` 结果与源码归档，新输出使用 `runs/aligned_v3_*`，从头初始化，不续训。
- 原始 TempLAMA 数据、官方 ICAE checkpoint、Reader、128 槽位特征缓存不变。
- 保留原 1,711 个历史目标及其稳定 ID；训练输入和内容哈希改变：current 与 latest QA 均改为 Episode 最终当前版本。
- 历史负答案取最终当前版本的答案，排除历史合法答案别名。训练、每轮 200 题诊断、481 题验证均注入同一最终当前版本。
- C 只编码历史；D 编码、解码都使用相同的最终当前 128 槽位。共同 Reader 读取 current 128 + history 16 + 4 role/time slots。
- C/D 参数规模、损失权重、初始化、训练信息、样本顺序和 optimizer 预算匹配；主对比均关闭 anchor dropout。D 专属 dropout 不混入本轮主对比，也不额外启动消融。
- 保留 CE、teacher KL、sequence unlikelihood、pooled alignment、index loss、margin=1.5 / weight=1。Teacher 仍单独读未压缩目标，未擅自修改 KL 定义；它不与 student 的双版本上下文完全匹配，保留为已知风险。
- 5 epochs、峰值 LR 1e-4、5% warmup + cosine、gradient accumulation 4、clip 1。每轮评测原固定 200 道训练冲突题，并保存 loss 分量和检查点。

## 快照边界

`CurrentSnapshotStore` 在观察完整 Episode 后，用全部原始事件特征构建快照。只保留最终 current 的 128 槽位，其余历史均压缩为 16 槽位。D 码绑定到同一持久化 current ID；读取时复用 current，不重复计算额外锚点槽位。

快照不支持逐事件写入，不支持把旧码换到新条件下直接解码，不允许用恢复后的有损码反复压缩。若 current 更新，必须重新取得原始 full 特征并构建新快照；这类来源保留和重编码成本尚未作为在线系统证明。原始全量历史特征是离线 builder 输入，不存入 `.memory`，也不提供给正式 codec Reader；未压缩反事实控制单独使用原特征。

本轮验证的是固定快照上的条件压缩能力，不是已解决的 streaming 持久化系统。C/D 驻留槽位预算相同；旧关键帧实验与新快照的历史保留策略不同，不能把新旧差异归因于某一个修改。

## 评测口径

原固定 200 道训练题的 ID 在启动时与 `refine_v2_C16_train/train_diagnostic_manifest.json` 严格对账。验证仍是原 481 道实际读取压缩码的题，其中 262 道答案集合完全冲突；它属于调参后的验证诊断，不能声称是未触碰测试集。保留数据哈希、逐题输出、全部事件写入、FP16 roundtrip 和实体关系组 paired bootstrap CI。

本轮不重训 C32/D32，不跑完整 test，不跑 LongMemEval，不保证 D 必然优于 C。
