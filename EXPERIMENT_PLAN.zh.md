# VCRM 第一版：ICAE 版本条件化历史记忆

## 主线

新历史事件/事实更新 -> ICAE Encoder -> Current Memory / Historical Memory ->
时间索引 -> 冻结 ICAE Decoder + 查询。第一阶段不加入 VCRM codec。

Current 保存当前有效事实的独立 latent representation；Historical 保存旧版本的
独立 latent entries。先确认现成 reader 能使用这些 entries，再研究条件压缩。
不恢复完整 hidden states，不加入 RL、learned eviction 或额外冲突识别模型。

## 唯一进入门槛

官方 ICAE v2 Mistral 指令问答 checkpoint，使用公开 TempLAMA 官方来源训练划分
做 Encoder 的 QA 答案损失适配，Decoder/任务标记冻结。实体重叠过滤沿用现有规则，
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
使用相同冻结 reader、训练数据、QA 监督、更新步数和持久化预算，差别仅在历史
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
  明确为非官方 GPT-4o 成绩。当前尚未实现/启动 ICAE LongMemEval 完整转移评测。
- MemoryLLM/M+ 官方代码：https://github.com/wangyu-ustc/MemoryLLM 。

这次仅切换实现和准备可读性门槛，不声称 ICAE 能力已验证，也不声称新颖性已确立。
