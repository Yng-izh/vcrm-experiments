# ICAE VCRM 架构与训练实现

本轮只交付代码、CPU 测试和数据审计。先前零样本 GPU 任务已停止，未完成输出
保留；不声称新训练链路通过 GPU 门禁，更不声称方法效果已验证。

## C/D 槽位瓶颈

`SlotCodec` 用低宽度多头 Cross-Attention，保留 hidden=4096 的接口。C 用 old
自对齐，D 用 immutable keyframe 对齐；两者都有 gate、压缩 query、恢复 query
与相同数量可训练参数，默认各 16,661,760。C 不是参数更少的一层池化基线。
共享参数初始值、训练顺序、QA/蒸馏监督和优化预算，条件输入才是主变量。

注意力输出经软门控从 old 中扣除，再浓缩至 m=8/16/32。这是可学习的对齐
机制，不证明槽位具有确定语义或已经消除了所有冗余。恢复输出仍是 m，禁止
将恢复到128后声称节省活跃读取槽位。

## 固定关键帧，而非不断重定基

`KeyframeStore` 第0、K、2K次版本为全量关键帧，默认 K=5。块内非关键帧
在下一次更新到来时从原始全量记忆压缩一次，随后固定引用块的关键帧。
当前版本全量保留，已压缩码禁止作为新的写入输入；既不解压重压，也不依赖
“当前最新锚点”不断变化。这是固定锚点条件化的明确变体。

驻留预算包含全量当前版本、所有仍需要的关键帧、历史码。FIFO 淘汰不能留下
悬空依赖；不相关状态必要时按依赖组整体淘汰。无法容纳新状态时原子回滚。
序列化保存结构、索引、压缩次数和 Codec 权重指纹，不保存原文。重载检查
Codec 身份与依赖；更新/修改模型后需要新的存储，不能读取旧码蒙混过关。

`VersionMarkers` 每条记忆增加角色前缀和日期后缀，共2个活跃槽位。参数98304
FP32字节，略低于0.1MB；全参数使用量照实报告。时间表示为连续年月日时刻
特征，不引入上千个年份 embedding。C/D 均使用相同标记架构。

历史读取为当前全量记忆 + 恢复后的历史m槽位 + 两组标记；关键帧参与辅助
恢复的访问槽位单独报告。只算 M+m 会遗漏标记与锚点成本。

## 语义关联与子段写入

`SemanticIndex` 从 ICAE 槽位 Attention Pooling，再投影为128维归一化 FP16
key。训练时使用 train 实体关系的正负对监督，不使用验证/test或24题的标签。
索引损失与 Codec 无梯度依赖，C/D 使用相同样本/顺序/初始化，保证公共索引。
随机初始化的投影不视为合格语义检索；query编码与事件编码的匹配也需验证。

`split_event` 按256个原始token创建连续子段，完整保存source id/hash与token
offset；不 decode/re-tokenize，不读query、gold evidence或答案标签。每个段
独立ICAE编码。余弦阈值0.8及跨状态margin仅产生关联候选，不能识别矛盾或
证明事实更新；只有共享resolver确认entity/relation与真实更新时才合并状态。
没有校准索引/更新resolver时各span作为独立状态，FIFO保留；不会默认误压缩。
因此端到端状态识别能力仍待实现具体resolver并验证，不混称已经解决。

## Encoder 适配

官方checkpoint先zero-shot。需要适配时只训练128个memory槽位的embedding，
原rank512 LoRA、Mistral与最后3个task embedding被冻结并检查不变。参数FP32
优化，进入模型时由官方函数转为FP16。共享任务行梯度归零，weight decay=0。
训练输入为train-only多事实段落（目标事实+不相关事实），不是短句原样微调，
也不声称是自然对话数据。没有实现r=8旁路，因为memory-only已覆盖本轮需求。
冻结大秩权重不保证长文能力100%不变，仍须独立长文门禁。

## Codec 联合损失与时序任务

每步同台注入当前全量+历史压缩版本，分别计算当前答案和历史答案CE。Teacher
使用未压缩目标entry的冻结reader，在相同gold answer前缀位置提供detached
词表分布；KL含temperature平方修正。不是逐元素hidden-state MSE，也不
声称蒸馏一定强百倍。短答案仍只有有限监督位置，后续可另测长回答训练数据。

另加平均池化cosine辅助损失和完整错误版本答案的sequence unlikelihood；共享
有效答案不为负例。错误答案含EOS，避免直接惩罚新旧答案共用的词或子词。
训练无LoCoMo/LongMemEval监督，保存Teacher/reader/数据/权重与损失系数来源。
Decoder梯度重计算每次都重新进入disable_adapter，不能在backward误启用LoRA。
编码/解码分别使用显式整次 forward checkpoint，重算恢复对应 adapter 模式；
禁用共享模型的 HF per-layer checkpoint，避免可变 adapter 状态导致重算结构不一致。
PEFT 恢复 adapter 会重开 LoRA requires_grad，因此每次模式恢复后显式重设冻结参数。

## 入口与护栏

统一入口新增：

```bash
./run_experiments.sh --stage unit-tests --run-name code_checks
./run_experiments.sh --stage training-audit --run-name training_inputs
# 基础 config 拒绝训练；调度器仅在完整门禁和错误审查后生成授权派生配置。
./run_experiments.sh --config config.authorized.json --stage adapt --gpus auto --run-name memory_adapt
./run_experiments.sh --config config.authorized.json --stage codec-train --method C --gpus auto --run-name codec_C
./run_experiments.sh --config config.authorized.json --stage codec-train --method D --gpus auto --run-name codec_D
```

`allow_training=false`。Codec还必须具备完整readability gate、当前reader指纹及
`codec_manual_review_approved`。CPU/100组pilot不能开放门禁；未知语义索引不能
自动进入正式路由。`SlotMemoryRuntime.from_checkpoint`连接训练权重、写入、
保存重载和时序读取；正式A/B/C/D/E全量报告和端到端LongMemEval比较未运行。
