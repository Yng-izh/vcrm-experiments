# VCRM：版本化 latent 记忆（LongMemEval-S）

本仓库只保留**当前架构实现 + 现行评测协议 + 结果与日志**。
历史对照实验（外部 baseline 表、matched/aligned 训练、CSR-v1 早期流水线、TransMem 遗留代码）
已从工作树移除，仍可在 Git 历史中检索。

## 1. 当前状态

- 架构实现：`vcrm/`（34 个模块，约 1 万行）
- 评测协议：`probe/run_main_eval.py`，四组 A/B/C/D 全量 500 题
- 最新结果：`results/main_v1/` → **A 0.412 / B 0.084 / C 0.244 / D 0.246**
  （judge = Qwen3.5-9B，500/500 计分，零判分错误）
- 训练：`lme_s_latent/run_e3.py`（E3 latent 链路），checkpoint 在服务器 `runs/architecture_v6_E3_train`

## 2. 核心链路

```
事件流 ──> FactExtractor（LLM 抽取 subject/relation/value/时间）
      ──> StateIndex（语义 subject 归一 + 版本条目，独立 FP16 版本记录）
      ──> IncrementalCurrentStore（current 视图，仅最新观测）
读取：  query ──> QueryTimeResolver + StateVocabularyAligner + StateRouter（semantic/latest/…）
      ──> 选中状态条目 ──> latent reader（ICA 解码）或文本 reader
```

关键模块：

| 模块 | 职责 |
|---|---|
| `vcrm/fact_extractor.py`, `llm_fact_extractor.py` | 事件 → 事实三元组抽取 |
| `vcrm/state_index.py` | 状态条目、subject 归一、语义索引存放 |
| `vcrm/store.py` | current（最新观测）视图 |
| `vcrm/state_router.py` | 路由模式（`latest` / `semantic` / `oracle` …） |
| `vcrm/query_time.py`, `query_alignment.py` | 查询时间解析、查询词表 ↔ 状态词表对齐 |
| `vcrm/runtime.py`, `pipeline.py` | 端到端运行时与编排 |
| `vcrm/codec.py`, `reader.py`, `text_reader.py` | latent 编解码与两类 reader |
| `vcrm/lme_data.py` | LongMemEval-S 数据装载（A 组文本链路与 latent 链路共用） |
| `vcrm/judge.py` | 判分协议 |

## 3. 仓库结构

```
vcrm/                 架构实现（唯一实现来源）
probe/                评测与探测脚本；run_main_eval.py 为四组协议入口
lme_s_latent/         latent 链路训练/评测（run_e3.py）
lme_s_text_rag/       A 组文本 RAG 链路
c1_hybrid_rag/        A 组检索与 reader 依赖（frozen：BM25 + BGE + RRF，vLLM shim）
tests/                单元与契约测试
results/architecture_v6/   架构 v6 训练/验证/测试扫描结果
results/main_v1/           四组 500 题结果与日志（见 RESULTS.zh.md）
docs/                 现行设计文档；docs/archive/ 为已冻结的历史文档
patches/              线上运行的增量补丁（并发抽取等）
config.architecture_v6*.json   当前架构配置
run.py / run_architecture.py / run_paper.py / bootstrap.py   架构任务的编排入口
```

被 `.gitignore` 排除、不入库的内容：`runs/`（大体积运行产物）、`data/`、`checkpoints/`、
`upstream/`、`models/`、`.env`（密钥）。

## 4. 四组协议与主结果

| 组 | 方案 | overall |
|---|---|---|
| A | text_rag：词法检索 + 原始文本读取 | **0.412** |
| B | state_text：状态路由 + 原始文本读取 | 0.084 |
| C | current_only：latent 写入，仅当前版本 | 0.244 |
| D | versioned：latent 写入，全版本 + 语义路由 | 0.246 |

结论摘要（完整分题型表见 `results/main_v1/RESULTS.zh.md`）：

- **D − C = +0.002**，差异仅在 multi-session（+0.76pt），其余题型相同 → 版本化写入在该数据集上增益极小。
- A 显著高于 C/D，优势集中在 multi-session（0.526）与 temporal-reasoning（0.541）→ latent 读取链路存在单点信息损失。
- B 仅 0.084：与 A 同 reader、仅候选 session 来源不同 → 瓶颈在状态路由召回，不在 reader。

## 5. 复现

```bash
# 服务器：v100-2，项目根 /share/latent2/vcrm_icae
cd /share/latent2/vcrm_icae
eval "$(grep -m1 '^export VCRM_LLM_KEY=' /root/.bashrc)"

# 四组之一（--group text_rag|state_text|current_only|versioned）
CUDA_VISIBLE_DEVICES=8 env_py310/bin/python probe/run_main_eval.py \
  --group versioned --data data/longmemeval_s_cleaned.json \
  --runs-root runs/main_v1 --defer-judge \
  --reader icae --arch-exp E3 --checkpoint runs/architecture_v6_E3_train \
  --router-mode semantic --frontend llm \
  --llm-config llm_config.json --llm-cache-root runs/main_v1/llm_cache \
  --extract-workers 8 --tag d

# 文本组（A/B）必须显式开启真实 reader，默认 stub 只产出占位符
#   --text-reader vllm --vllm-url http://<host>:6012/v1/completions --vllm-model Qwen3.5-9B

# 判分
python3 /tmp/judge_main_eval.py <predictions.jsonl> judge.json <judge_cache> <stem>
```

注意事项：

- 判分为自定义 judge（Qwen3.5-9B），与官方 GPT-4o 分数不可直接对齐；同一 judge 下四组横向可比。
- 跨 GPU 结果不逐字节可比（不同 GPU 的 `write_bytes` 不稳定）；锚定指标用 `decision_counts` / `state_count`。
- 共享 `llm_cache` 使崩溃/重启续跑不需重复调用。

## 6. 文档索引

现行：

- `CURRENT_DESIGN.zh.md` — 当前设计（最新）
- `ARCHITECTURE.zh.md` — 模块设计与限制
- `ARCHITECTURE_ABLATIONS.zh.md` — E1–E4 消融设置与停止规则
- `EXPERIMENT_PLAN.zh.md` — 实验计划
- `PROTOCOL_CSR_V1.md` — 评测协议
- `NUMERICAL_AND_F1_DIAGNOSIS.zh.md` — 数值与 F1 诊断（含 F1–F7 编号）
- `RUN_PAPER.zh.md` — 论文级流水线入口与门禁
- `SETUP_STATUS.zh.md`, `DEPRECATION.md` — 环境状态与废弃清单
- `docs/LME_S_OFFICIAL_EVAL_SPEC.md` — LongMemEval-S 官方评测规格
- `docs/E3_LEAK_AUDIT.md` — E3 泄漏审计
- `docs/AUTO_ASSOCIATION_REFACTOR.md` — 自动关联（query_alignment）改造
- `docs/P2_NO_ORACLE_STATE_WRITE.zh.md`, `docs/store_budget.md` — 无 oracle 写入约束、存储预算
- `results/main_v1/RESULTS.zh.md` — 四组 500 题结果

归档（历史，已冻结）：`docs/archive/{ALIGNED_TRAINING,MATCHED_TRAINING,COMPARISON,REFINEMENT}.zh.md`
