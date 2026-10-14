# VCRM：最小可发表实验方案

## 一、结论与实验主线

建议以 TransMem（2026）作为外部方法基座，使用 LongMemEval 作为主评测、LoCoMo 作为辅助评测，并用 STALE 检查过时信息的干扰。

但在正式实验前，需要明确一个关键限制：

现有公开基准无法仅凭一个总分完整证明 VCRM 的核心主张。

- LongMemEval 包含 500 个正式评测问题，覆盖知识更新、时间推理、多轮信息整合、信息提取和拒答，非常适合检验持续更新后的记忆可用性。官方提供 S、M 两种历史长度，以及标准答案和证据标注。

  [image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

  GitHub

- LoCoMo 可以检查多轮历史问答与时间推理能力，并与 TransMem 已有实验衔接。

- STALE 专门衡量新信息能否取代过时记忆，但没有直接完整衡量被覆盖的旧版本能否按历史时间恢复。

  [image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

  GitHub



这些都是已发布的公开研究基准，部分对话和场景由基准作者合成或模拟，并非全部来自真实用户的自然交互。它们满足使用外部公认评测的要求，但不能直接宣称是自然用户数据。

我建议论文只围绕一个核心主张展开：

> 在有限持久化记忆预算下，版本条件化的任务感知潜在记忆，能否比一般潜在记忆更好地兼顾最新事实使用与历史信息恢复？

实验的逻辑应当是：先证明已有实现可复现，再证明更新场景存在改进空间，最后证明改进来自 VCRM 的机制，而非额外存储、更多训练或更强的读取模型。

## 二、选定外部 Baseline：TransMem

[TransMem: Transforming Hidden States into Memory for Large Language Models](https://arxiv.org/abs/2607.29032)

Lei et al., arXiv 2026

官方代码：

[Haodong-Lei-Ray/TransMem](https://github.com/Haodong-Lei-Ray/TransMem)

已有 Qwen3-4B 和 Qwen2.5-14B 对应适配器，提供训练代码、权重、LoCoMo 等评测入口。

我选择它而不是直接复现 M+，主要基于实施成本和实验可控性。

TransMem 冻结 LLM 主干，将历史 hidden states 转化为可复用的潜在记忆，并通过轻量模块干预当前推理状态。其代码支持特征提取、教师—学生蒸馏训练和下游评测。因此可以尽量复用它的 latent reader，将实验重点限制在 memory representation 与写入策略上。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



不过必须明确：原版 TransMem 并不是已经验证的、固定容量的在线长期记忆系统。 将它用于本研究需要新增按时间顺序的 memory ingestion 和有界持久化存储。这一改动必须同时施加于 TransMem baseline 和 VCRM，否则比较不公平。

### 具体修改边界

保留 TransMem

不修改主干

Frozen Qwen3-4B、历史 hidden-state 提取方式、latent reader、输出层、推理与评分流程。

修改的部分

添加按会话写入的 memory store；在发生更新时，允许旧信息编码进入版本化历史记忆；使用当前与历史 QA 监督学习有限容量内的信息保留。

不添加

新的 RL 策略、复杂 eviction 网络、多 Agent 架构或专门的冲突检测 LLM。

实现时，先复现作者现有的 LoCoMo 评测，确认公开 checkpoint、数据处理及推理协议都能正常运行。之后统一改成 streaming protocol，原版 TransMem 和 VCRM 使用同一个写入接口。

这一点对论文尤其重要：官方 TransMem 成绩只能用于核对复现，不能直接拿来与新 streaming 设置的成绩比较。

## 三、论文必须证明的三个命题

| 命题          | 需要证明什么                      | 最直接的证据                       |
| ----------- | --------------------------- | ---------------------------- |
| P1：更新正确性    | 后续信息覆盖旧状态后，模型更能回答最新有效事实     | LongMemEval Knowledge Update |
| P2：历史可恢复性   | 在完成多次更新后，模型仍能按指定时间恢复旧事实     | TempLAMA 时间条件化问答             |
| P3：有限预算下的优势 | P1、P2 的改进不是靠保存更多信息或增加读取开销取得 | 相同存储预算的对照实验                  |

三个命题中，P3 最关键。

如果只证明 P1，可以归结为时间索引或更新规则的收益；如果只证明 P2，可以归结为保留了更多历史。如果不能证明 P3，VCRM 关于 latent residual compression 的核心贡献就无法成立。

这也决定了实验设置：以一个 baseline 架构为主，通过最小改动检验机制，而不是堆砌多种不同训练范式的 SOTA。

## 四、必要实验

### 实验 0：复现 TransMem 的公开基准

目标：确认研究基座可靠。

使用作者发布的匹配 checkpoint 和官方评测入口，先在 LoCoMo 上复现结果。

数据：

[LoCoMo 官方数据与代码](https://github.com/snap-research/locomo)，ACL 2024，包含 10 段长对话及人工标注 QA；对话由 LLM Agent 生成并经人工检查和编辑。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

snap-research/locomo · GitHub



要求采用原代码一致的问答类别、token-F1、模型 revision 和解码设置。不能因为我们的结果与论文不同就自行修改评分提示词。

随后实现 TransMem-Streaming：按时间提取并写入历史表示，查询时只能访问持久化 memory store，不能回读原始对话全文。所有方法共用此协议。

决策： 如果官方基线无法稳定复现，先排查模型版本、配置和依赖，不要马上训练 VCRM。基线与论文不一致时，必须在后续结果中说明。

### 实验 1：LongMemEval——更新后能否使用最新有效信息？

数据集

[LongMemEval（ICLR 2025）](https://github.com/xiaowu0162/LongMemEval)

使用 2025 年清洗版本 LongMemEval-S 的完整 500 道题。

重点子集

knowledge-update、temporal-reasoning

同时报告完整 500 题总体结果及其他类别，避免选择性报告。

官方提供标准答案、时间戳、证据会话 ID 和自动评价代码，可以分别测问答正确性和相关证据是否被找到。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



实验流程

1. 每道题独立初始化记忆，按时间顺序写入官方提供的历史会话。
2. 仅在完成全部写入后，提出原始 benchmark question。
3. 不向方法提供 `answer_session_ids` 或 `has_answer` 等 oracle evidence 标记。
4. 所有方法采用相同的记忆预算、LLM、读取 token 限制和评分协议。

主要指标为官方 QA Accuracy，重点单独报告 knowledge-update Accuracy。增加一个错误分析指标：模型是否把已失效的旧事实当作当前答案。

后者可以根据官方证据人工或规则核验；如果不能可靠确定旧事实是什么，就只报告原始官方准确率，不强行赋予错误类型。

判断标准： VCRM 需要相较 TransMem-Streaming 有稳定的 knowledge-update 改善，而且完整基准不能明显退化。

但该实验不能单独证明历史版本恢复，下一项是必要的。

### 实验 2：TempLAMA——同一事实的历史版本能否恢复？

[Time-Aware Language Models as Temporal Knowledge Bases](https://aclanthology.org/2022.tacl-1.15/)

Dhingra et al., TACL 2022

[TempLAMA 官方实现与数据](https://github.com/google-research/language/tree/master/language/templama)

基于 Wikidata 中有时间有效区间的事实

TempLAMA 的特点非常适合本问题：同一实体—关系在不同年份可能具有不同正确答案。作者从 Wikidata 时间事实构建查询，并提供互不重叠的训练、验证、测试实体划分。

[image](https://www.google.com/s2/favicons?domain=https://direct.mit.edu\&sz=32)

MIT Press



例如，若某实体在不同年份的组织关系发生变化，原始基准允许分别查询不同年份的对应值。

这里将其改编成 streaming memory evaluation：

- 按事实的有效时间依次写入。
- 写入完成后，使用官方原有的逐年问题评估每个历史版本。
- 测试实体与训练实体严格隔离。
- 不修改标准答案，也不人为生成新的评测问题。

这属于使用外部真实时间事实和原始标注构造的任务协议，必须在论文中清楚命名为 TempLAMA-Streaming。不能声称它是 TempLAMA 原论文的标准在线记忆设置。

同时，时间有效区间的起点并不一定是真实知识被系统观察到的日期。此实验把事实按有效时间顺序写入，因此主要验证版本存储与读取，不证明模型可以处理迟到的更正事件。

主要指标：

- Latest Accuracy：查询数据时间范围内最后一个有效版本的正确率。
- Historical F1：按官方历史年份问题计算 token-F1。
- Historical Retention：在全部更新之后，旧年份问题仍可正确回答的比例。
- Stale Answer Rate：查询旧年份时错误返回较新版本的比例。

为避免把重复年份当作独立样本，需要按实体—关系分组报告结果，并在统计分析中按组做 bootstrap。

最重要的结果： 在多个事实先后更新之后，VCRM 的历史 F1 是否高于 TransMem-Streaming，同时 Latest Accuracy 不下降。

如果只提高最新值准确率、旧值恢复没有改善，VCRM 不满足我们的研究目标。

### 实验 3：固定存储预算下的核心对照

这项实验直接决定论文是否成立。

所有对照在同一套 TempLAMA-Streaming 和 LongMemEval-S 协议中运行。为了控制因素，推荐下表中的五个系统。

| 方法                            | 作用                          |
| ----------------------------- | --------------------------- |
| A. TransMem-Streaming         | 外部工作基线：有界记忆，普通写入与读取         |
| B. TransMem + Version Index   | 检验显式时间索引是否已经足够              |
| C. Independent Latent Entries | 独立压缩历史旧事实；使用与 VCRM 相同的双时态监督 |
| D. VCRM（Ours）                 | 当前状态 + 版本条件化潜在残差，使用相同监督     |
| E. Text Version Log           | 强文本基线：按时间保存原始事实版本、按时间检索     |

其中 C 是最关键的机制对照。

A 对 D 的提升可能来自大量因素，包括新索引、训练目标或表示粒度；C 对 D 才能直接检验使用新旧信息的条件压缩是否优于独立存储历史事实。

B 则负责排除“仅仅因为增加版本索引就得到了收益”。

对于 C 和 D，尽可能匹配可训练参数、训练样本、更新次数和读取结构；若某些差别不可避免，单独报告，不把收益直接归因于 residual representation。

文本基线采用同一事实识别和时间索引协议，不允许 VCRM 使用 oracle 更新位置而文本方法必须自己推断。

如何统一预算？

采用真实持久化字节数，计入：

\\[ B\_{\mathrm{total}}= B\_{\mathrm{memory}} +B\_{\mathrm{index}} +B\_{\mathrm{metadata}} +B\_{\mathrm{other\ persisted}} \\]

所有方法必须把潜在向量、检索键、时间标签和持久化辅助状态全部计入。模型权重与训练计算单独报告。

建议选择三个有实际区分度的预算点：可以完整保留部分历史、能够保留大部分历史、接近无压缩完整记忆的参考条件。具体字节数在观察独立版本存储的实际大小后确定，并在正式测试前固定，不针对测试成绩调参。

论文最关键的 Figure

仅示意要检验的假设，没有任何实测曲线或数值。正式论文应根据真实结果绘制，不能预设 VCRM 排名。

核心评价是 accuracy–storage trade-off。可以比较固定字节预算下的 Historical F1 和 Latest Accuracy，也可以报告两者在预算范围内的曲线。

不要只比较某一个精心挑选的 budget。如果 VCRM 只在单一预算点表现更好，通常不足以证明稳定的压缩优势。

为了衡量统计不确定性，推荐在 TempLAMA 按实体—关系分组、在 LongMemEval 按评测实例分组进行配对 bootstrap，报告 95% 置信区间，并保留逐题输出供错误审查。

## 五、高价值补充实验

只有必要实验显示出正向结果，才建议做以下两项。

补充 A：STALE 隐式冲突测试。

使用作者发布的 [STALE 官方数据与评价代码](https://github.com/icedreamc/STALE)，验证模型遇到新证据但没有明确表达“旧事实失效”时能否继续正确行动。原基准有 400 个冲突场景、1,200 个问题，包含 State Resolution、Premise Resistance 和 Implicit Policy Adaptation。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



它的价值在于检验模型是否能够适应较复杂的信息更新。但是第一版 VCRM 使用已知更新关系，所以不能直接将 STALE 全量结果当成版本压缩能力评价。需要所有方法共享同一个非 oracle 的状态识别器，或明确将结果标为端到端扩展测试。

补充 B：历史恢复的错误来源分析。

如果 VCRM 在历史查询上取得增益，可以根据 TempLAMA 的时间有效区间，将错误划分为：

- 目标旧版本已经因容量限制被淘汰。
- 正确版本仍在记忆中，但时间索引选择错误。
- 正确版本被选中，但压缩表示无法支持答案恢复。
- 正确记忆已注入，但冻结 LLM 仍回答错误。

这比机械比较更多 latent token 数量、更多 decoder 层数更有价值，因为它能解释增益到底来自存储、选择还是读取。

## 六、可以省略的实验

第一篇论文阶段，我不建议增加 MemoryAgentBench 全任务评测、ALFWorld、跨十几种 LLM 架构、RL post-training、超大规模 LongMemEval-V2，或大范围超参数搜索。

这些实验大多回答的是模型泛化、Agent 规划或系统扩展能力，而当前论文的核心是持续更新下的潜在记忆有效性与有限容量。

只要 TransMem-Streaming 的基准可复现，LongMemEval 体现真实的更新能力，TempLAMA-Streaming 测量历史恢复，再加预算公平的强对照，主要证据链就已经完整。

## 七、推荐实验顺序与决策规则

1. 阶段 1：跑通 TransMem + LoCoMo 官方复现

   核对 checkpoint、模型、评分协议与作者代码，然后添加 streaming wrapper。

   决策：基线无法复现就先修复；不可直接进入正式对比。
2. 阶段 2：先跑 TransMem-Streaming 的 LongMemEval-S

   确认在线写入机制是否足够稳定，特别观察 knowledge-update 类别是否有实质错误。

   决策：如果主要错误来自无关的语言理解或读取器能力不足，应先修正基座；若已有较高表现，则进一步查看 VCRM 是否仍存在可改进的容量问题。
3. 阶段 3：实施 TempLAMA-Streaming 和 Independent Latent Entries

   先比较普通潜在记忆、版本索引和独立历史事实编码。暂不训练 VCRM。

   决策：如果简单的独立编码已在目标预算内准确保留历史，就应降低对复杂残差压缩的投入。
4. 阶段 4：只实现 VCRM 的核心模块

   训练版本条件化 residual encoder，并保持 reader、事实索引、训练标签和预算不变。

   决策：若相对 Independent Latent Entries 没有稳定收益，停止增加模块，重新考虑核心研究假设。
5. 阶段 5：完成正式评测与文本强基线

   进行完整 LongMemEval-S、TempLAMA-Streaming 和固定存储预算比较，报告端到端写入与读取开销。

   决策：只有在效果、容量和成本之间形成明确优势，才继续扩大为论文；否则收缩到已被证实的结论。

### 不同结果意味着什么？

| 观察到的结果                                                 | 对论文的影响                                  |
| ------------------------------------------------------ | --------------------------------------- |
| VCRM 优于 TransMem，但与 Version Index 持平                   | 收益主要来自显式时间管理，不足以支持新的潜在编码机制              |
| VCRM 优于 Version Index，但与 Independent Latent Entries 持平 | 可能来自历史事实粒度的变化，residual compression 贡献不足 |
| VCRM 在相同字节下稳定优于 Independent Latent Entries             | 支持条件化历史残差的核心假设                          |
| VCRM 历史 F1 提升，但最新事实准确率降低                               | 尚未实现当前与历史能力的协调                          |
| VCRM 优于 latent baselines，但显著弱于 Text Version Log        | 可以研究潜在表示内部的效率收益，但不能主张总体存储效率更优           |
| VCRM 在 LongMemEval 和 TempLAMA 均显示有意义的收益                | 有较强依据进一步发展为正式论文                         |

## 八、最终建议：最小实验组合

我建议正式论文最终至少有以下三张核心结果表或图：

| 结果       | 内容                                                                                        | 验证命题 |
| -------- | ----------------------------------------------------------------------------------------- | ---- |
| Table 1  | LongMemEval-S 完整结果与 knowledge-update / temporal-reasoning 细分                              | P1   |
| Table 2  | TempLAMA-Streaming 的 Latest Accuracy、Historical F1、版本错误率                                  | P2   |
| Figure 1 | VCRM、Independent Latent Entries、TransMem-Streaming、Text Version Log 的 Accuracy–Storage 曲线 | P3   |

另外附一张简短的复现设置表，包括 checkpoint、训练数据、模型参数、预算计算方式和推理成本。

有两个前置工作需要在正式实验前处理：一是 TempLAMA-Streaming 的数据转换脚本必须公开、可复现，并且不改变原有标注；二是训练时只能使用官方训练划分或其他独立公开训练数据。LongMemEval 和 LoCoMo 的正式评测 QA 不能用于 VCRM 监督训练或调参。LongMemEval 官方主要提供评测数据，TempLAMA 有按实体划分的数据，因此可以先用 TempLAMA 官方训练部分完成首轮学习实验，再以 LongMemEval 检验跨数据分布泛化。

我最想优先看到的结果是：

在 TempLAMA-Streaming 的相同持久化存储预算下，VCRM 是否比 Independent Latent Entries 同时达到更高的 Historical F1 和不低的 Latest Accuracy。

如果这个结果不成立，那么即使 VCRM 在 LongMemEval 上比原始 TransMem 提升很多，也不能直接证明版本条件化残差的数学假设有效。

如果它成立，再通过 LongMemEval 证明这种收益能够迁移到实际的长对话记忆任务，论文就拥有一条相对简洁、明确且有说服力的证据链。

因此，建议把最初的工作重点放在 TransMem-Streaming + TempLAMA-Streaming + Independent Latent Entries 三者上。 这是成本最可控、也最早能判断 VCRM 是否值得继续投入的实验组合。