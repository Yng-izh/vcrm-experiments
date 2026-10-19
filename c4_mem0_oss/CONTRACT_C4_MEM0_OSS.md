# CONTRACT_C4 — Mem0 OSS 基线（v3 主表第 5 行）

状态：DRAFT→FROZEN（2026-10-18，本契约起草即冻结载体；冻结后实现与运行不得更改本契约，任何例外须 captain 书面修订并记录于 §8）。
归属：`archive/2026H1_reports_md/EVAL_PLAN_V3.md` §二 表 1 第 5 行（Mem0 OSS）、§二 备注（"Mem0 必须运行确定版本的 OSS 实现；94.4% LongMemEval / 92.5% LoCoMo 平台成绩不得引用"）、§五（"Mem0 / A-MEM：确定 pin 的 OSS 版本，writer 用其原生支持，记录成本"）。

## 1. 版本 pin（冻结）

| 项 | 值 |
|---|---|
| 仓库 | mem0ai/mem0（GitHub OSS） |
| pin | release tag `v2.2.1`（最新主发布 tag，2026-09-25 发布，monorepo 版本对应 pyproject `version = "2.2.1"`） |
| commit sha | `94c3fe9f238f3dbf29c9ce98643bd71eb13077cd`（lightweight tag 直接指向 commit，已实测校验） |
| 实现路径 | 本仓库子目录 `mem0/`（Python 包 `mem0ai`），检索/抽取/整合全部走 OSS 代码，不调用 mem0 托管平台 API |
| 安装方式 | 本地 tarball 安装（v100-2 直连 GitHub TLS reset，GnuTLS -110，实测 2 次）：本地 clone commit `94c3fe9f…` 打包 tarball（sha256 = `1c99c475ac940cce93c8abab3925d055c98fdb2e0fb573f54fc32b917c471810`，对应 `/tmp/mem0_pin.tar.gz`）→ scp 至 v100-2 → 从本地路径 `pip install`；安装前 `pip freeze` 快照，安装后回归验证 transformers/vllm/vcrm 可导入（沿用 C2 §2 环境约束） |
| 依赖隔离 | 向量库用本地实现（首选 `chroma` 或 `qdrant` 本地模式），不引入云服务；LLM provider 按 §2 配置 |

任何运行产物（JSONL、summary、log）必须记录上述 sha；sha 不匹配的产物视为无效。

## 2. 检索前端与 Reader（冻结）

- **检索前端用 Mem0 原生**：不用 C1 的 BM25+BGE+RRF 前端，不改写 Mem0 的 `search()` 路径。写入 Mem0 的记忆库由其 OSS 抽取/整合 pipeline 产出，读取按 Mem0 原生 `memory.search()`（向量相似度 top-k）返回 facts 注入 prompt。与 C1/C2/C3 行的"唯一差异原则"相反：本行差异 = 整个检索与记忆构建前端，这正是该基线的定位。
- **Reader 固定**：Mistral-7B-Instruct-v0.2，greedy（temperature 0）、max_new_tokens 128、FP16、seed 42；`_X_` 原样保留。Mem0 的内部 LLM 调用（事实抽取、整合决策）不用其自带 LLM 配置直接生成答案——**答案生成只走我们的固定 Reader**；Mem0 内部 LLM 用 writer 配置（§3）。
- 注入模板：将 Mem0 检索返回的 facts 文本按其相关性顺序填入 C1 模板 `{facts}` 块，模板文字逐字沿用 C1（`CONTRACT_C1_HYBRID_RAG.md` §3），保证四行 prompt 模板可比。
- 任务无 oracle：不使用 answers/state_key/as_of/latest/gold 做任何写入、抽取、整合、检索或淘汰决策（§6）。

## 3. Writer 与成本（冻结）

- Mem0 内部 LLM/writer：用 Mem0 公开实现原生支持的模型（默认 `gpt-4o-mini`），逐次调用记录模型名与 token 用量；实际选用的模型与 provider 在冒烟报告冻结。
- **调用成本上限**：writer 调用预算上限 = **USD 200 全量**（34,963 题 / 4,057 episodes 写入）。接近 80% 时报警并报 captain；超限停跑、不得续跑，结果如实标注截断点。冒烟阶段实测单题成本，按单题成本外推全量成本；若外推将超限，须提前报 captain 裁定后方可继续。成本列不虚构（EVAL_PLAN_V3 §："不把不同 writer 配置假装成等算力实验"）。
- **平台分数禁令**：禁止在任何产物、summary、报告、主表中引用 94.4%（LongMemEval）/ 92.5%（LoCoMo）mem0 平台成绩；官方 OSS demo 数字亦不得作为预期分数引用（如需 sanity 注脚，须按 C3 §8 惯例标注口径差异且不进主表）。

## 4. 评分与产物（冻结）

- 评分：`vcrm/metrics.py` EM/F1 + 冻结 ex28 抽取（与 C1/C2/C3 相同 `EXTRACTION_RULE`，冻结不重调），四列 `em_raw/f1_raw/em_extracted/f1_extracted`，主表报 extracted。
- 分母（DENOMINATOR_NOTE 逐字写入 summary）：
  - 全量 test 34,963 queries / 4,057 episodes，完整性缺一即 run 无效；
  - latest 分母 **4,057**；Hist 分母 **30,906**；
  - conflict 用 **metadata 口径**（同 state_key 任一版本 answers 集不同），**显式分母 n=34,407**；
  - EXTRACTION_RULE(ex28) 冻结，与主表一致。
- 逐题 JSONL 字段对齐 `probe/runs/full_34963.jsonl` 惯例：`query_id, episode_id, question, type, answers, prompt_tokens, raw_answer, extracted_answer, em_raw, f1_raw, em_extracted, f1_extracted, read_seconds` + 本行新增 `retrieved_facts`（Mem0 返回 facts 与得分）、`writer_model`、`writer_tokens`、`route{status:"mem0_oss", pin_sha}`、`shard`。
- 产物路径：`c4_mem0_oss/runs/smoke100.jsonl`（冒烟）、`c4_mem0_oss/runs/full_34963.jsonl`（全量）、`c4_mem0_oss/runs/summary_*.json`。
- summary 必含：pin sha、writer 模型与实测成本、EXTRACTION_RULE(ex28)、DENOMINATOR_NOTE、检索 top-k 与命中统计。

## 5. 运行基建（冻结）

- 复用 `probe/ours_probe.py` 模式：watchdog、done_ids 断点续跑、death_log.jsonl。实现放 `c4_mem0_oss/`，不改 probe 本体。
- **GPU 纪律**：Reader 推理单卡独占；跑前 nvidia-smi 实测盘点（全卡显存 + compute apps）报 captain 确认锁卡后启动；GPU 2/3 永不触碰；绝不 kill 未知进程。
- Mem0 向量库写入与查询为 CPU/磁盘操作，不占 GPU；writer 走 API 不占 GPU。
- 冒烟：100 题 seed42 分层 hist80/latest20（复用 `probe/runs/sample100.json` 同题可比），产物 + 10 题人工核验（检索 facts sanity：top-k、非空率、无 gold 字段泄漏迹象），monitor 审查通过后放行全量。

## 6. Oracle 排除清单（同 C 惯例）

- 禁入写入/检索/构建路径的字段：`answers`、`as_of`、`latest`（`state_key` 禁读不禁存在，对齐 C3 §9 修订）。上述字段仅允许出现在评测侧。
- 实现落地时跑 AST 审计（对齐 ours_probe `ast_audit`：Attribute + 字符串常量双检查，事件级禁读 = {answers, as_of, latest}，事件字段访问白名单 = {text, time}），报告落盘 `c4_mem0_oss/runs/ast_scan_report.json`。

## 7. 双侧落盘与 md5 同步

- 本契约文本双侧落盘：工作区 `/Users/yngizh/quant/deepseek/CONTRACT_C4_MEM0_OSS.md` 与 server 侧实验目录 `c4_mem0_oss/CONTRACT_C4_MEM0_OSS.md`（随实现任务拷贝落盘）。
- md5 同步规则：server 侧文件必须与本文件逐字节一致；一致性校验用下方冻结 md5。
- 冻结 md5 定义（自引用规避）：将本节代码块中占位符 `<FROZEN_MD5_PLACEHOLDER>` 逐字替换为 32 个字符 `0` 后，对整文件计算 md5。
- 冻结值（2026-10-18，monitor 终审冻结时实测）：

```
FROZEN_MD5 = 0109281fd4c8be13d4019be7912897bf
```

## 8. Revisions

| 日期 | 原因 | 旧值 | 新值 | 批准 |
|---|---|---|---|---|
| 2026-10-18 | monitor 终审（t2）：§7 原 md5 `f0da0918…` 与实测不符（草稿期误录，且自引用 md5 无法自洽）；改为占位符置零定义并记录冻结 md5 | `MD5(…)=f0da091801e800551e8e6de4dae59aa9` | 占位符置零定义 + `FROZEN_MD5` | monitor |
| 2026-10-18 | captain 指令（t2 终审）：§3 补"冒烟实测单题成本 → 外推全量；外推将超限须提前报 captain 裁定"条款（USD 200 上限已批准冻结） | 无外推提前报备条款 | 外推将超限提前报 captain | monitor（依 captain 指令） |
| 2026-10-18 | runner 实测 v100-2 直连 GitHub TLS reset（GnuTLS -110，2 次），git+https 安装不可用；§1 安装方式改为本地 tarball（sha256 `1c99c475ac940cce93c8abab3925d055c98fdb2e0fb573f54fc32b917c471810`）→ scp → 本地路径安装；pin commit 与 pyproject version=2.2.1 不变 | git+https subdirectory 安装 | 本地 tarball 安装 | captain（2026-10-18 消息即批准记录） |
| 2026-10-18 | 实测 api.openai.com 自 v100-2 与 captain 本地均不可达（curl 000 超时，DNS 污染），SSH 隧道方案因双方网络均不通而不可行；§3 writer 由默认 gpt-4o-mini 改为 deepseek-chat（`https://api.deepseek.com/v1`，OpenAI SDK 兼容 base_url，mem0 openai provider 原生支持），embedder 改为本地 huggingface provider（复用服务器 `models/bge-base-en-v1.5`，零 API 成本）；USD 200 预算不变，全量写入外推 ≈ $16.5（占预算 ~8%） | §3 writer = gpt-4o-mini（默认） | §3 writer = deepseek-chat @ api.deepseek.com/v1 + 本地 BGE embedder | captain（2026-10-18 消息即批准记录） |
| 2026-10-18 | 用户提供 OpenAI 兼容中转端点 `https://api.ccode.vip/v1`，v100-2 实测全通（/v1/models 200，gpt-4o-mini 真实 chat 调用成功）；§3 writer 回切 gpt-4o-mini（官方默认模型不变，经中转 base_url），废止 deepseek-chat 修订（未执行过真实调用，无遗留）；embedder 仍为本地 BGE（huggingface provider，CPU） | §3 writer = deepseek-chat @ api.deepseek.com/v1 | §3 writer = gpt-4o-mini @ https://api.ccode.vip/v1（中转，OpenAI 兼容）+ 本地 BGE embedder | captain（2026-10-18 消息即批准记录；用户指令） |
| 2026-10-18 | 用户指示（writer 单价高，控制冒烟成本）：§5 冒烟口径 100 → 20 条，seed 42 分层（hist 16 / latest 4），取 `probe/runs/sample100.json` 前 16 个 historical + 前 4 个 latest（保持种子可复现，非重新抽样）；全量 34,963 不变 | §5 冒烟 = 100 题（seed42 hist80/latest20） | §5 冒烟 = 20 题（seed42 hist16/latest4，来自 sample100 分层截取） | captain（2026-10-18 消息即批准记录；用户指令） |
| 2026-10-18 | 中转端点稳定性未知，captain 将全量 writer 成本预警阈值由 80% 提前至 50%（USD 200 的 50% = $100 即报 captain）；USD 200 上限本身不变 | §3 预警阈值 80% | §3 预警阈值 50%（本运行特例，中转端点） | captain（2026-10-18 消息即批准记录） |
