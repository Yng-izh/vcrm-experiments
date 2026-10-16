# 当前方案与参数核对：matched_v4

本文描述实际代码与服务器运行配置，不是新的实验建议。当前训练为 `matched_v4`；已完成的 C/D 主结果与两个外部对照来自 `aligned_v3`，不能当作 matched_v4 的结果。源码入口为 `vcrm/training.py`、`vcrm/codec.py`、`vcrm/runtime.py`、`vcrm/store.py`。

## 1. 当前研究对象与边界

比较 Independent Latent C16 与 Version-Conditioned Latent D16：在历史版本同为 16 个槽位、当前版本同为 128 个槽位、模型容量和监督一致的条件下，当前版本能否作为压缩与恢复历史信息的有用条件。

当前实现是 `final_current_snapshot_v1`：观察完整 Episode 后构建离线快照。不是逐事件在线压缩系统，也不是已经完成的 LongMemEval 流式记忆方案。一个 Episode 是一个实体-关系状态链；观察事件在事实答案集合发生变化时产生，不是每年必然产生一个新版本。

压缩器不输入未来的用户问题、答案字符串或答案位置标注。训练时历史与最新 QA 的答案用于计算损失；推理时问题只交给 Reader。D 的额外输入是已观察到的最终当前版本，不是未来未知事实。

## 2. 数据与训练样本

| 项目 | 当前设置 |
| --- | --- |
| 数据 | TempLAMA 官方来源划分，按已有规则去除跨 train/val/test 的实体重叠 |
| 过滤后的训练 Episode | 1,215 |
| 训练观察事件 | 4,169 |
| 原始训练 QA | 10,475 |
| 每 Episode 观察版本数 | 1 至 11 |
| 有历史 Codec 监督的 Episode | 1,171 |
| 每轮 Codec 历史目标 | 2,954 |
| 其中初始版本目标 | 1,171 |
| 双版本 Episode | 299 |
| 有至少一个合法当前负答案的目标 | 2,069；这不等同于全部答案集合互斥 |

每个非最终观察版本选一条已有历史 QA，优先与观察时间对应，其次最早的有效 QA。QA 使用 `as_of` 找最后一条不晚于该时间的观察事件，因此事实持续多年时不要求问题时间等于事件时间。

每条训练 pair 是 `(M_old, M_final_current, q_hist, y_hist, q_latest, y_latest)`。current、D anchor 与 latest QA 都对应 Episode 的最终观察版本。每条 pair 同时监督历史与最新问题，因此版本较多的 Episode 的 latest QA 会重复参加训练；不是每个 Episode 等权采样。保留原有数据、答案和稳定 pair ID，只修复历史目标覆盖，未增加人工冲突样本。

写入使用已有观察事实文本，`ingestion_view` 去掉 QA 注释和答案字段；事实文本本身当然含该事实的值。`query_view` 去掉答案，仅保留问题、时间和共享状态索引。评估的 state_key 与时间为已知路由，不是在测试语义检索。

## 3. ICAE 基座与特征缓存

| 项目 | 当前设置 |
| --- | --- |
| Backbone | Mistral-7B-Instruct-v0.2 |
| Backbone revision | `63a8b081895390a26e140280378bc85ec8bce07a` |
| ICAE | 官方 v2，`sggetao/icae` 的 `mistral_7b_ft_icae.safetensors` |
| ICAE checkpoint revision | `c95c70d670c8a284550610b49b0f26d2a3b2d1e3` |
| ICAE source revision | `469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd` |
| Encoder LoRA rank | 512，官方构造 alpha=32，dropout=0 |
| 每条事实的缓存形状 | `[128, 4096]` |
| 缓存与持久化 dtype | FP16 |
| Codec、标记、索引模块 dtype | FP32；进入 Backbone 时转换为 FP16 |
| Backbone attention | SDPA，适配 V100；不使用 FlashAttention2 |
| ICAE mean compression rate | 4，沿用官方分段编码逻辑 |
| Encoder 最大输入 | 5,120 tokens，不静默裁剪 |
| 特征文本格式 | `DATE: {event_time}` 后接观察文本 |
| 特征缓存 | `/share/latent2/vcrm_icae/features/paper_v1_official` |

本轮只读取缓存，不训练 ICAE Encoder，不更新官方 rank-512 LoRA，不适配 Mistral Decoder。配置中的 `adapted_checkpoint=null`。虽然编码器代码允许多段，当前 TempLAMA 缓存明确要求每条事实恰好是 128 槽位，不允许将更大记忆悄悄截成 128。

Decoder 使用 ICAE 官方指令模板与 checkpoint 的 ft task token，关闭 LoRA 后读取记忆。这里的冻结是参数冻结：Student QA 梯度仍穿过冻结 Mistral，回到 Codec 输出与角色/时间标记。启用 Decoder whole-forward gradient checkpointing，避免重计算时错误地启用 Encoder adapter。

## 4. C16 / D16 的真实架构

公共参数：hidden `H=4096`、历史槽位 `m=16`、注意力宽度 `d=256`、head 数 `4`，每 head 64 维；attention dropout 为 0。不是 4096 维直接做多头注意力，而是先投影到 256 维。

每个 `SlotAttention` 含外部无 bias 的 Q/K/V 投影、一个 PyTorch `MultiheadAttention(256, 4)` 及无 bias 的 256->4096 输出投影。MultiheadAttention 本身仍有其内置投影和 bias，不能将整块描述为全部无 bias。Codec 含三块这样的 attention：alignment、compress、recover。

其他参数包括两个独立可学习查询矩阵 `Q_comp`、`Q_rec`，均为 `[16,4096]`、初始化标准差 0.02；一个 4096 维 LayerNorm；一个 gate MLP：`8192 -> 256 -> GELU -> 4096 -> sigmoid`。

### 4.1 压缩编码

令 `O` 是历史 128 槽位，`N` 是最终当前 128 槽位，`LN` 为共同 LayerNorm：

```text
C: context = O
D: context = N

A = alignment(LN(O), LN(context))       # [128, 4096]
G = sigmoid(gate(concat(O, A)))          # [128, 4096]，逐槽位逐维门控
R = O - G * A                           # [128, 4096]
Z = Q_comp + compress(Q_comp, R)         # [16, 4096]
```

C 也训练 alignment 和 gate，只是使用历史自身作为 context；这是容量匹配的独立编码，不是更小的单层池化。传给 C 的 anchor 参数会被忽略。

D 使用跨版本 attention 建立软对应关系，不按同一位置直接做 `O[i]-N[i]`。但它确实在软对应之后执行门控减法，当前实现仍具有“减去条件可解释部分”的归纳偏置，不等于通用的信息重要性选择器。

压缩器没有额外的 slot position embedding，也没有输入实体答案位置。ICAE 特征自身含编码后的文本位置与语义信息，但不能据此声称新旧 latent 槽位存在一一对应。

### 4.2 读取前恢复

```text
C: recover_context = Z
D: recover_context = concat(N, Z)

H_old = Z + recover(Q_rec, recover_context)   # [16, 4096]
```

恢复是 16->16，不恢复 128 槽位，不做完整 hidden-state MSE 重构。D 没有 `N + Decoder(Z)` 的直接 128 槽位加法，也没有把当前 128 槽位直接当成历史输出。保存的是编码后的 Z，而不是 H_old；重载时再计算 H_old。

本轮 C/D 均 `decoder_anchor_dropout=0`，没有 D 专属 Dropout。D 的编码端与恢复端都看到 N；当前比较不分离这两个条件通路的贡献。

### 4.3 参数规模

| 模块 | C16 | D16 |
| --- | ---: | ---: |
| Codec | 16,661,760 | 16,661,760 |
| 角色/时间标记 | 24,576 | 24,576 |
| SemanticIndex | 528,384 | 528,384 |
| 总训练参数 | 17,214,720 | 17,214,720 |

C/D 同 seed、相同初始公共模块与样本顺序，但分别训练各自的 Codec、标记和 Index，不共享更新后的标记参数。所谓同一个 Reader，是冻结 ICAE/Mistral 权重相同，不是所有可学习 prompt 标记最终数值相同。

## 5. Reader 接收什么

每个版本包装为：

```text
[role embedding] + [该版本记忆槽位] + [time embedding]
```

role 有 current/historical 两行，形状 `[2,4096]`。time 是无 bias 的 `Linear(4,4096)`，输入为：`[(year-2000)/100, month/12, day/31, (hour+minute/60)/24]`。时间不是原始文本 token，而是投影出的一个 latent 标记槽位。

Student 双版本记忆布局：

```text
current role + N(128) + current time
historical role + H_old(16) + historical time
# 共 148 个记忆/标记槽位，不包括指令和问题 tokens
```

Teacher 用相同次序、相同当前标记权重，将 H_old 换成完整 O(128)，共 260 个记忆/标记槽位。Teacher 标记和 logits 停止梯度，但标记数值跟随 Student 标记更新；因此 Teacher 的 Backbone 固定，整个条件输出分布并非跨训练步完全固定。

问题包含 `As of: {as_of}`。TempLAMA `_X_` 问题转换为“填空，只返回缺失事实”的指令。Teacher/Student 使用相同问题、相同正确答案 teacher-forcing 前缀；由于历史槽位长度不同，绝对位置与活跃上下文长度仍不同，“matched-context”指版本内容选择与角色/时间语义对齐，不是输入张量完全相同。

推理只有时间索引选择的目标历史条目与最终当前条目，不把所有历史版本一起注入。Latest QA 则读取最终当前与最近历史版本，这与现有 runtime 一致；没有自动路由成只读 current 的模式。答案使用 greedy，最多 64 tokens，不调用外部 judge。

## 6. 完整训练目标

```text
L = mean_over(latest, historical)[CE + 1.0 * KL_T=2]
    + 0.1 * SequenceUnlikelihood
    + 1.0 * ContrastiveMargin
    + 0.05 * PooledCosineAlignment
    + 0.1 * IndexLoss
```

- CE：监督各问题 `answers[0]`，包括答案末尾 EOS；并非每步对全部合法别名做 marginal likelihood。评估则对全部合法别名取最优 EM/F1。
- KL：`KL(P_teacher/T || P_student/T)`，temperature=2，乘 `T^2`。按答案前缀各位置计算并取平均，teacher detach；正确答案文本与 EOS 的监督位置均参加。
- Margin：对历史问题比较正确历史答案与最终当前负答案的平均 token log-prob，排除 EOS，`relu(1.5 + logP(current_negative) - logP(history_correct))`。若负答案有多个别名则取这些 margin 的平均。
- SequenceUnlikelihood：`-log(1-P(完整当前负答案序列))`，包括 EOS；不是逐 token 惩罚。多个负答案求和，不平均。当前负答案若也是历史合法答案则排除。
- Pooled alignment：`1-cos(mean(O), mean(H_old))`。这里只对齐 128 槽位与 16 槽位的均值向量，不要求 token 对应，不是逐槽位恢复损失。
- IndexLoss：同一状态 old/current 的 key 拉近；选择一个不同状态的 current 作为负例，罚 `relu(cos(key_old,key_other)-0.2)`。没有大 batch InfoNCE，也不经过 Codec 输出；使用冻结源特征，因此该分支主要训练 Index，而非 Codec。

同一 pair 对 latest 和 historical 任务等权。合法答案别名有部分重叠时仍可能有可惩罚的 current negative，但这不算评估所用“全部答案集合互斥”的 conflict 类别。

## 7. 优化器与运行参数

| 参数 | 当前值 |
| --- | --- |
| Epoch | 5，从随机初始化训练，不接续 aligned_v3 |
| 每轮样本 | 2,954 pairs；每 pair 两个正向 QA，加可用当前负答案 |
| 样本处理总次数 | 14,770 |
| Microbatch | 1 pair |
| 梯度累积 | 4 pairs；每轮最后一组为 2 pairs |
| 每轮 optimizer updates | 739 |
| 全部 updates | 3,695 |
| Optimizer | 两个 AdamW：Codec+Markers / Index 各自独立 |
| 峰值 LR | 1e-4 |
| AdamW betas / eps / weight decay | 默认 `(0.9,0.999)` / `1e-8` / `0.01` |
| Schedule | Warmup + cosine；warmup 184 updates，约 5% |
| Gradient clipping | 两组分别 clip global norm=1.0 |
| Seed | 42；每轮重新 Shuffle |
| Encoder / Decoder Dropout | eval；官方 LoRA dropout=0 |
| Codec attention / anchor Dropout | 0 / 0 |
| Decoder gradient checkpointing | 开启 |
| Decoder prefill chunk | 512 tokens；完整 KV cache，不裁剪历史 |
| Decoder latent slot 安全上限 | 1,280；主对比实际 Student148 / Teacher260 |
| Snapshot resident slot 上限 | 2,048；主实验不淘汰 |
| 硬件 | v100-2，GPU0=C16，GPU2=D16；单卡 V100 32GB，非 DDP |

配置里 `epochs=1`、`learning_rate=1e-5` 是旧 Encoder adaptation 字段；Codec 实际读 `codec_epochs=5`、`codec_learning_rate=1e-4`。`keyframe_interval=5` 仍随配置保留，但当前快照不再按关键帧位置跳过历史训练或完整保存中间关键帧。`span_window_tokens=256`、语义关联阈值0.8以及 LME diagnostic 长度设置均不是当前 TempLAMA Codec 主实验的工作参数。

## 8. 存储与在线更新尚未实现的部分

有 K 个观察版本时，C/D 的张量槽位为 `128 + 16*(K-1)`，额外每版本持久化一个128维FP16索引 key、时间/状态/版本ID、格式及依赖元数据。role/time 标记在读取时生成，参数为共享模型组件，不按每 Episode 复制到 `.memory`。

张量 payload：完整128槽位=`128*4096*2=1,048,576 bytes`；历史16槽位=`16*4096*2=131,072 bytes`；每个 key 为256 bytes。该数值不包括元数据。C/D 张量 payload 与槽位预算相同，D 保存 condition ID 等依赖信息，因此整个文件未必逐字节等大。

存盘前编码 Z，转 FP16；正式评估保存完整快照并重载后恢复。每轮200题诊断也让 Z 经过FP16转回FP32，以匹配量化。训练正向不模拟这个FP16往返，存在一个小的训练/持久化量化差别。

D 的历史 Z 绑定到最终当前 N 的ID。N更新后不能直接换条件解码旧Z；要保留原完整历史特征并重新编码，成本尚未作为在线系统证明。原特征缓存保留在工程目录，但不是正式快照 Reader 的输入，不计入 `.memory`；这不代表全系统不占这些空间。

## 9. 评估与已完成结果

- 每轮固定200道训练集新旧答案集合互斥的历史题，测train F1/EM与stale_answer；沿用旧清单及顺序，是in-sample诊断，不是泛化证明。
- 训练完成后现有统筹器自动跑原481历史验证题与未压缩反事实控制。独立 `run_comparison.py` 才追加172道Latest QA，形成653题四方法表；不能将自动481题验证说成自动完成653题表格。
- 481历史题中262题的全部合法历史/当前答案集合互斥。评分为现有归一化EM/token F1，全部合法答案别名取最佳值，未调用LLM评分。
- Bootstrap按Episode分组抽样，2,000次、seed42；指标本身是相应子集的逐题平均，不是先求每Episode F1再等权平均。
- 该481题是已用于调参的验证诊断，来自早期共同预测前缀的筛选，不是全量验证、随机代表性测试或未触碰test；当前未跑正式LongMemEval迁移。

已完成的 aligned_v3 轮次：

| Method | 历史F1 % | 冲突F1 % | Latest EM % | 平均MB/Episode |
| --- | ---: | ---: | ---: | ---: |
| 原生 Mistral Text Version Log | 36.67 | 30.71 | 15.70 | 0.002186 |
| ICAE-Full：aligned_v3 C标记 | 66.24 | 55.08 | 54.07 | 4.440081 |
| aligned_v3 C16 | 55.07 | 44.79 | 尚未同协议补测 | 旧表记录1.475，需区分单位与汇总口径 |
| aligned_v3 D16 | 51.56 | 42.65 | 尚未同协议补测 | 旧表记录1.475，需区分单位与汇总口径 |

ICAE-Full不是官方论文指标：官方checkpoint加项目的双版本协议与C训练得到的标记。D自己的未压缩反事实对照使用D标记，不能不加说明地合并成同一个Full结果。

文本基线用原生Mistral chat template，latent使用ICAE task template；Text并非完全相同prompt下的模态替换。存储保留所有观察版本，读取时只选两个版本。Storage除以1,000,000为MB，除以1,048,576为MiB，包含元数据但排除共享权重与原特征缓存。

Text低分本身尚不证明latent表示天然优于文本，需要区分回答格式、提示协议、精确匹配与实际事实错误；当前不添加新的因果结论。

## 10. 考虑模块改进时应明确的控制变量

1. 编码目标：当前D是软对齐后的门控减法，不是明确最大化“重要信息/高信息量内容”的目标，也没有信息增益或熵权重监督。
2. 条件通路：D同时在encode和decode阶段使用current；若改动其中一端，需要明确归因边界，不能只称“编码器利用版本信息”。
3. Reader与标记：冻结Backbone不等于整个读取接口固定。标记与Codec共同训练；Teacher也随这些标记数值变化。
4. 蒸馏：Teacher能出现双版本混淆，KL不保证其输出总是正确；CE与KL的相对权重都是1，不能把总Loss数值直接解释成QA正确率。
5. 监督覆盖与权重：所有历史版本已覆盖，但每版本只监督一条QA、只用首个正确别名；latest QA按历史版本数重复。这里的任务分布比“所有年度QA等权”更窄。
6. 容量匹配不等于计算匹配：D encode attention看当前128槽位，decode attention看128+16；C decode仅看16。参数相同、输出预算相同，但D计算与条件依赖更多。
7. 时间路由已知，SemanticIndex尚未经query-to-event校准。当前成绩不验证开放式语义检索、复杂版本消解或learned eviction。
8. 当前训练未结束。旧C/D结果、matched_v4训练诊断与Full对照是不同实验层次，不应提前合并为最终核心贡献结论。
