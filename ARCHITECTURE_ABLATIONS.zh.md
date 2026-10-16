# architecture_v5：独立消融与条件组合

入口为 `run_architecture.py --plan config.architecture.json`，默认仅 `--phase prepare`。本轮按用户指定的 E1-E4 主线，不做宽度搜索、全组合网格、RL、learned eviction 或提前加 STE。

## 已核验的轮次与控制变量

- `aligned_v3` 已完成，Teacher 是单条未压缩目标；历史目标为旧筛选得到的1,711条。
- `matched_v4` 真实训练代码已将 Teacher 改成 current128 + old128 +同组角色/时间标记，总260槽位。用户要求暂停时正在第3轮，保留epoch01/02 checkpoint、全部诊断与日志，没有完整训练 checkpoint，不作为正式对比结果。其文档为 `MATCHED_TRAINING.zh.md`、`CURRENT_DESIGN.zh.md`。
- 本消融不额外做第五种 Teacher 训练。统一固定为 aligned_v3 的 `single_target` Teacher，历史训练覆盖修复后的全部2,954条。因为旧控制的训练覆盖不同，必须从头补 C0/D0 两个共同控制，不能将旧1,711条训练结果直接作为E1/E3的唯一变量对照。
- 所有新训练使用相同官方ICAE checkpoint、冻结Reader、最终当前离线快照、128 current +16 history、seed42、5 epochs、shuffle、gradient accumulation4、LR1e-4、warmup5%+cosine、clip1、零anchor dropout。每轮739次更新，共3,695次；每轮固定原200道训练冲突诊断。
- 原481道历史验证题、262道全部答案集合互斥冲突题、固定state_key+as_of。Reader greedy64，评估保存/重载FP16码，不改变读取顺序、标记形式或文本提示。单版本Teacher只改变共同固定训练背景，不列作架构消融效果。

## 顺序与矩阵

1. 原外部主表：沿用已完成Text与ICAE-Full预测，补旧aligned_v3 C/D的653题统一评测，输出 `baseline_table/TABLE.md`。各方法历史/冲突/最新和存储口径一致，不把旧轮次与新轮次混填。
2. 不训练的读取诊断：分别加载旧aligned_v3 C/D checkpoint，读取O128、N128+O128、H16、N128+H16。标记不变，仅Reader是否看到current、历史是否压缩发生变化。D在H16单独读取时，恢复器仍使用N；“单读”只指Reader看不到N，不混入E1。
3. 训练D0、E1、C0共同控制/候选，可用三张卡并发；各完成5轮后评估原481题。
4. 若E1支持去解码条件，在其固定结构上训练E2；同时可独立跑E3，二者不需要互相等待。若E1不支持，则终止该叠加分支，E2记录为跳过，不强行在失败前提上叠新模块。
5. 选C0/E3中验证更好的C-Best，仅对它训练E4。
6. 保留有实际增益的修改后训练严格匹配的CFinal/DFinal，输出最终481题对照和653题主表。不因某个C-only改进而自动宣称版本条件有效。若E1未支持，则不强制组合一个去解码条件的最终D，停止并交付诊断/消融结果。

| 标签 | method | 编码 | 解码context | compressor Value | Contrastive weight |
| --- | --- | --- | --- | --- | ---: |
| C0 | C | O-GA，alignment context=O | Z | projected256 | 1.0 |
| D0 | D | O-GA，alignment context=N | [N,Z] | projected256 | 1.0 |
| E1 | D | 与D0相同 | Z | projected256 | 1.0 |
| E2 | D | O*(0.5+G)，G=σ(MLP([O,O-A])) | Z | projected256 | 1.0 |
| E3 | C | 与C0相同 | Z | original4096 | 1.0 |
| E4 | C | 与C-Best相同 | Z | 与C-Best相同 | 0.2 |

E2保留同容量gate MLP、alignment、compress、recover、查询向量。只将gate第二组输入从A改成O-A，同时把输出表示从减法残差改成原始O的[0.5,1.5]逐维权重，编码规则作为一个预定义变量。不是新增加一个gate网络，也不是槽位硬选择。

E3只替换compress中的Value/output路径，不替换alignment与recover。保持外层Q/K的4096->256投影，以及原MHA的内层Q/K 256->256投影，4 heads、每head64维。计算attention权重后直接聚合4096维R，各head聚合其对应1024维原特征，拼接成4096，不使用Value投影、MHA输出投影或额外4096x4096投影。共同模块与Q/K权重从相同随机初始化保留，Codec参数减少，真实参数数与读写成本记录在E3报告；不宣称与C0严格同参数容量。最终同结构C/D必须严格等容量。

E4仅改变weight，margin固定1.5；CE、KL(T=2/weight1)、UL(weight0.1)、pooled(weight0.05)、index(weight0.1)及其他优化设置保持不变。

## 量化诊断范围

四种读取中，H16单读和双读各额外比较编码码Z的FP32 vs FP16存储往返，共六条件。原ICAE缓存和冻结Backbone仍FP16，所有条件使用同一checkpoint与标记。这是持久化码量化诊断，不是宣称完整Mistral做了FP32-vs-FP16精度消融。报告逐题输出、paired CI和码relative-L2；不根据未测猜想添加STE训练。

## 选择规则与失败停止

由于用户未提供最小增益或显著性阈值，本实现使用明确、保守的点估计筛选规则并保存2,000次Episode bootstrap CI，不将筛选写成“显著提升”：

- 支持E1/E2/E3：全481历史F1增加，262冲突F1不下降。
- C-Best：按全历史F1，再按冲突F1，再按更低冲突stale排序；完全相同选C0。它和“E3值得纳入最终组合”不是同一判定。
- 支持E4：上述F1规则，同时冲突stale不增加。若降低负约束让stale反弹，记录结果但不采用。
- 若E4的C-Best路径与最终保留的Value路径不同，不把该低负约束结论迁移到未验证的Value路径。
- 若E2无益，最终不采用salience；保留E1所支持的residual编码，C/D仍统一结构，仅编码context不同。若E3无益保留projected256；若E4无益保留weight1。
- 所有单变量比较检查训练对数、pair内容哈希、数据、步骤、seed、Teacher、冻结Reader一致，并检查配置实际只改变指定字段。发生覆盖或预算不一致时直接报错。

同时修复覆盖、共同选回single Teacher构成新的实验背景；不能用新C0/D0对旧aligned_v3的差异单独归因于Teacher或样本修复。C/D标记依然各自训练，不声称最终prompt标记数值相同。

## 运行与输出

```bash
.venv/bin/python run_architecture.py --plan config.architecture.json --phase all --gpus 0,2,4
```

GPU必须当时空闲，程序使用原项目GPU锁。单变体单卡，不做DDP；独立控制/消融分配不同卡。只有不训练的诊断使用旧完整checkpoint；所有新架构从头训练，不复用旧optimizer。已完成评估可按Episode续跑；未完成训练没有optimizer resume，需新prefix从头开始。

阶段可单独选择 `baselines`、`diagnostic`、`ablations`、`final`。新运行目录 `runs/architecture_v5_suite` 及 `runs/architecture_v5_*`，保留旧结果。报告包含原主表、六条件诊断、E1-E4逐题/参数/成本/paired CI、`ABLATIONS.md`、`selection_decision.json`、最终C/D及最终主表。

`--phase all` 默认只做上述验证选择，**不会读取未触碰test**。最后固定架构和checkpoint后，使用独立 `--phase test` 才编码test缓存并评测冻结的CFinal/DFinal；输出 `final_test_comparison.json`。该入口不再训练或根据test重选模块。当前test入口覆盖最终C/D，不能声称已完成test上的Text/Full外部基线表。

本轮上限为4个核心消融候选、2个必要新共同控制及1组最终C/D；若门禁不支持则少跑，不展开全组合。文本和Full既有基线不训练。文档与脚本建立完成不等于所有GPU实验已经跑完。
