# VCRM 第二版：槽位预算与状态对齐

## 当前优先生效的局部修订

下列协议替代本文后续旧版的持久化预算主张与强制适配顺序。
最新模块、训练损失、固定关键帧策略与工程边界见 `ARCHITECTURE.zh.md`。

- 主预算分别报告驻留 latent slots 和实际 Decoder 读取 slots。候选当前锚点
  M=128、历史 m=8/16/32。C/D 必须直接读取 m；展开回 128 不能声称读取压缩。
  D 的固定关键帧锚点与依赖计算计入成本，禁止 lossy code 重编码；C/D 使用
  同样锚点访问权限和 QA 信息（C 不使用锚点条件）。
- 保留 A/B/C/D/E 的角色：无历史/当前态、未压缩版本、独立压缩、条件压缩、
  文本日志。正式配置在门禁后另行冻结。文本按活跃上下文与测得成本比较，
  不能将相同 token 数直接等同 FLOPs。磁盘 payload/索引/metadata 完整报告，
  不作为主预算或预设优势。128x4096 FP16 单段 payload 是 1,048,576 字节。
- 官方 checkpoint 先零样本，不要求先 adapt。100 个 TempLAMA 验证事实组
  (每组多年度 QA) 仅 pilot，不能开启正式门禁。完整独立验证与人工审查仍必需。
  当前/历史的时间索引读取仅测试选中的独立 entry，不证明多版本同时注入可靠。
- 旧短句适配禁用，已实现 train-only 多事实段落的 memory-only 适配入口与审计，
  rank512 LoRA冻结。必须用独立验证检查事实读取和长文泛化，不保证完全保护。
- 固定旧 error_controls 的 24 题与原检索顺序，运行同一 Mistral 的 selected
  latent/text、no memory、oracle latent/text。官方评分 prompt + 自定义 Qwen
  judge，不是官方分数。oracle 只用于离线诊断，24 题不得用于训练或调参。
- 诊断输入不裁剪。超过 5120 token 的会话按连续 token 块编码，只保留一次
  原始 BOS，全部块拼接；明确超官方输入长度的 wrapper extension，并审计
  每会话长度、槽位、原文 hash、上下文限制。长文本 prefill 分块保留完整 KV。
- 24 题 >=10 正确仅为压缩候选信号，仍需全量验证与同 reader 文本控制。
  <=4 先排查读取/路由而不训练 codec；5--9 为不确定。不同 reader 的成绩
  不能证明 ICAE 完全解决 TransMem；这些阈值也不能单独建立因果机制。
- LongMemEval 正式 C/D 只压缩共享、查询无关、无 gold 标签的状态对齐片段，
  保留时间和原文映射，单列状态关联的准确性/触发率。整会话无差别条件压缩
  不再采用，词法哈希仅旧占位。`SpanWriter` 已实现精确 256-token 子段；语义
  相似度只产生候选，必须由独立 resolver 确认真实更新才可合并状态。新 codec
  的训练入口仍默认关闭。
- 系统代价报告驻留/读取槽位、锚点、字节、KV cache、读写时间、峰值显存与
  输出上限。不能用驻留压缩代替读取压缩，也不能用字节代替推理成本。

旧证据纠正：317/500=63.4%，不是70.4%。24题文本控制定位到表示/读取链路，
尚不能隔离 4槽容量、分布偏移或版本干扰。LoCoMo 接近原分数不证明新链路无错。
results/v1002_phase12 保留研究记录，不混入 ICAE 主对照；不声称新颖性已确立。

统一脚本新增 lme-audit、lme-diagnostic、lme-summary；官方零样本可直接 gate。
readability --limit 100 及 24 题诊断均不授权 codec；默认禁止训练，完整门禁与
人工审查后才可启用 codec-train。unit-tests/training-audit 不占用 GPU。

## 旧版背景记录（上述修订优先）

## 主线

新历史事件/事实更新 -> ICAE Encoder -> Current Memory / Historical Memory ->
时间索引 -> 冻结 ICAE Decoder + 查询。第一阶段不加入 VCRM codec。

Current 保存当前有效事实的独立 latent representation；Historical 保存旧版本的
独立 latent entries。先确认现成 reader 能使用这些 entries，再研究条件压缩。
不恢复完整 hidden states，不加入 RL、learned eviction 或额外冲突识别模型。

## 唯一进入门槛

官方 ICAE v2 Mistral 指令问答 checkpoint，先运行零样本；适配仅在必要且自然
段落训练协议完成后启动，Decoder/任务标记冻结。实体重叠过滤沿用现有规则，
明确属于 filtered train/val，不声称与官方 split 完全相同；官方 test 保持原样。
LongMemEval 与 LoCoMo 正式 QA 不用于适配、阈值选择或模型选择。

独立验证集验证：不给历史原文，latent 是否能支持事实问答；连续写入多个观察版本
后，显式实体关系与时间索引选出的历史/当前 entries 是否仍能正确回答。
这不是端到端事实识别实验：TempLAMA 原始标签构成所有方法共享的输入事实。

设置文本、无记忆、错误版本控制，检查存储 roundtrip 后的 entry 一致性。
历史指标仅在有实质版本变化的事实组上计算。工程阈值提前记录在 config.gate 中，
默认 0.8 单条 EM、0.8 当前 EM、0.8 历史 F1、至少 0.1 无记忆差值；均为初步
验收标准，不是可发表标准。完整验证后人工审查错误，才允许设计下一阶段 codec。
若独立 entries 不可读，修复或更换 ICAE checkpoint，不用复杂模块掩盖。

## 门槛通过后的核心对照

Independent ICAE Memory vs Version-Conditioned ICAE Memory。
使用相同冻结 reader、训练数据、QA 监督、更新步数和潜在槽位预算，差别仅在历史
表示是否利用当前状态进行条件压缩。历史答案交叉熵为主要目标，无 hidden-state
MSE 前置假设。新 codec 的 shape/依赖/更新协议待门槛通过后独立定义。

报告 Current Accuracy、Historical F1、错误版本返回、真实持久化字节、读取 tokens、
读写时间与峰值显存。预算点依据验证布局预先冻结，不根据测试得分选取。
短事实 latent 可能比文本日志大很多，必须保留文本强对照，不能预设字节优势。
长片段和推理成本收益需分别证明，不能把 token 压缩比当作字节压缩比。

## 工程及研究记录

保留 data/store/metrics/judge；store 支持 ICAE 的二维可变段数 memory shape，
完整计入格式、索引和 metadata。旧 TransMem reader、训练、codec 和计划移入 legacy，
phase12 原结果、checkpoint 和审计指纹不变。LoCoMo 53.95% F1 为研究过程记录，
不是 ICAE 主对照。旧 11.20% streaming 结果不混入新方法比较。
MemoryLLM/M+ 保留为相关工作及条件允许时的外部竞争者，不承担第一版训练基座。

## 官方资源核对

- ICAE 官方代码：https://github.com/getao/icae ，CC0-1.0；v2 支持多 span 拼接。
- ICAE checkpoint：https://huggingface.co/sggetao/icae ，模型卡 Apache-2.0。
- v2 指令模型：mistralai/Mistral-7B-Instruct-v0.2；模型 revision 和全部文件 SHA256
  在 bootstrap 后记录，使用 mistral_7b_ft_icae.safetensors，不混用 v1/pretrained。
- TempLAMA：Google Research 官方来源年度事实；现有源文件与过滤统计独立存档。
- LongMemEval：保留 cleaned S 500 题用于后续转移评估；原官方 prompt + Qwen judge
  明确为非官方 GPT-4o 成绩。新增固定24题诊断，完整转移评测仍未实现/启动。
- MemoryLLM/M+ 官方代码：https://github.com/wangyu-ustc/MemoryLLM 。

这次仅切换实现和准备可读性门槛，不声称 ICAE 能力已验证，也不声称新颖性已确立。
