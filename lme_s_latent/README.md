# LongMemEval-S latent 管线（P4：E3 ICAE + 冻结 reader）

**状态：code-only + CPU stub 冒烟。本轮未训练、未跑 GPU 全量、未跑官方评测、未提交 git。**
真正的 500 题 GPU 运行与官方 judge **需用户放行**（见文末「阻塞清单」）。

文件：`run_e3.py`（本目录唯一脚本）；查询时间解析在 `vcrm/query_time.py`（`QueryTimeResolver`）。

## 口径（与 Text RAG baseline 的根本差异）

| 维度 | Text RAG baseline（`lme_s_text_rag/run_text_rag.py`） | Latent E3（本目录 `run_e3.py`） |
|---|---|---|
| 记忆载体 | 原始 session 文本（date 前缀） | E3 codec 的 latent slots（`IncrementalCurrentStore`） |
| 检索/关联 | BM25 + BGE + RRF，top-k session | FactExtractor → StateRouter → UpdateResolver → 状态版本链 |
| 记忆作用域 | **本题 haystack**（per-item 索引） | **本题 haystack**（每题独立 runtime/索引/存储） |
| 时间通道 | 无（问题原样） | `window_end = QueryTimeResolver`，经 `public_query(time_mode='window_end')` |
| reader | vLLM Mistral-7B-Instruct（C1 shim） | `vcrm.reader.FrozenReader`（E3 ICAE，adapter-disabled 解码） |
| 输出 | `runs/<tag>/<qid>.json`（8 字段） | `runs/<tag>/<qid>.json` + `<tag>.jsonl` + `<tag>_summary.json` |
| 评分 | 官方 judge（t12/t14） | 官方 judge（t12/t14）；本脚本只做 diagnostic 分级分析 |

三条硬口径：

1. **memory namespace = 本题 haystack，不等同于 Oracle episode。** 每题新建
   `StateIndex` / `IncrementalCurrentStore` / `PublicEventCounter`，只写本题
   session，因此不存在跨题泄漏，也不存在 `session_id` 重名错配。
2. **时间通道去 Oracle 化。** 查询视图只给 `{'question': ...}`；时间只经
   `public_query(time_mode='window_end', window_end=...)` 进入。`window_end` 由
   `QueryTimeResolver` 产出：问题文本里有可唯一折算的历史时间锚 → 用锚折算出的
   **区间末刻**（`in 2019` → `2019-12-31T23:59:59`），否则才用提问时刻。
   `question_date` 从不被当成历史目标时间，也不透传成 `as_of`。
3. **hypothesis 原样输出**，不做答案抽取/归一化/评分（LongMemEval 是自由问答，
   官方 judge 计分）。

## 运行矩阵

```bash
# A) CPU 冒烟（stub reader，5 题，off/on 两种 recency 模式，均无 GPU）
python3 lme_s_latent/run_e3.py --data <S_JSON> --reader stub --limit 5 \
    --tag lme_s_latent_smoke_off --no-resume
python3 lme_s_latent/run_e3.py --data <S_JSON> --reader stub --limit 5 \
    --tag lme_s_latent_smoke_on --no-resume --recency-update

# B) GPU 全量（需用户放行；先起 E3 checkpoint 环境）
python3 lme_s_latent/run_e3.py --data <S_JSON> --reader icae --arch-exp E3 \
    --tag lme_s_latent_e3_off --shard 0 --shards 4
python3 lme_s_latent/run_e3.py --data <S_JSON> --reader icae --arch-exp E3 \
    --tag lme_s_latent_e3_on --shard 0 --shards 4 --recency-update

# C) 官方链路（既有，未改动；默认 runs 目录 = lme_s_text_rag/runs）
LME_TAG=lme_s_latent_e3_on bash lme_s_text_rag/run_official_eval.sh            # dry-run
LME_TAG=lme_s_latent_e3_on bash lme_s_text_rag/run_official_eval.sh --execute  # 需 OPENAI_API_KEY
```

`--runs-dir` 默认 `lme_s_text_rag/runs`，**就是为了让 `make_official_eval_input.py`
与 `run_official_eval.sh` 原样可用**（它们只认默认 runs 目录；`--runs-dir` 传
字符串会 TypeError，见「外部发现」）。

## 输出

```
<runs-dir>/<tag>/<question_id>.json   每题一条：question_id/question/question_type/
                                      hypothesis + 时间与状态诊断字段
<runs-dir>/<tag>.jsonl                逐样本诊断行（含 error_bucket、time_anchor、
                                      window_end、decision_counts、update_source_counts）
<runs-dir>/<tag>_summary.json         汇总：分口径统计 + 分级桶 + diagnostic em/f1
<runs-dir>/<tag>_errors.json          逐题异常（resume 保留）
```

`summary.json` 关键字段：`time_mode`（恒 `window_end`）、`time_source_distribution`、
`time_anchor_distribution`（`question_text_anchor` / `ask_time`）、
`anchor_ambiguous_count/rate`、`recency_update`、`recency_min_confidence`、
`decision_counts`、`update_source_counts`（`explicit_replacement` /
`exclusive_superseded_by_recency` 分列）、`hybrid_fallback`（默认 off，透传给
`PredictedRuntime`，口径与 `probe/run_state_eval.py` 同名开关一致）、
`error_buckets`（四必需桶 + `update_deferred` + `correct` /
`abstention_unscored` / `no_gold`）。

**分级错误分析**（上游优先，最早失败阶段决定桶）：

| 桶 | 含义 |
|---|---|
| `fact_extraction_failed` | 答案证据从未被抽成事实（或本题没有任何事实） |
| `update_deferred` | 证据抽出来了，但决策落 DEFER、没有 current 版本（`recency_update=off` 时膨胀 = 长对话演化覆盖不足） |
| `state_match_failed` | 有事实但没路由到含证据的状态（含 `no_state_match`） |
| `version_select_failed` | 状态对了，但选中的版本不含证据（时间上界/版本链问题） |
| `reader_wrong` | 状态与版本都含证据，仍答错 |

`diagnostic.em/f1` 直接复用 `vcrm.metrics.exact/f1`（与 `probe/run_state_eval.py`
同一套函数，不是第二套评分实现），只用于 `correct` / `reader_wrong` 分桶，
**不是官方分数**；权威分口径来自 P3 评测器（`--hybrid-fallback` 默认 off）与
官方 judge（t12/t14）。

## 显存与耗时估算（数量级，非实测）

| 项 | 估算 | 依据 |
|---|---|---|
| 权重常驻 | ≈14.5 GB（Mistral-7B FP16）+ codec/markers/index（<0.1 GB） | `FrozenReader` FP16/SDPA，单卡 V100-32G 足够 |
| 读取 bundle | 128 slots × 4096 dim × 2B ≈ 1 MB/版本；current + 紧邻历史 ≈ 2 MB | `read_latent_slots=128` |
| 写入 | 每题 ≈ haystack session 数（S 集数百条）× `encode_event`；V100 上 ≈0.15–0.4 s/session | 500 题 × ~500 session ≈ 25 万次编码 ≈ 10–28 GPU-hours（单卡） |
| 读取 | 500 次解码，≈1–2 s/题（prefill 512 分块 + 贪心 256 tokens） | C1/C4 实测口径 |
| CPU stub 冒烟 | 5 题 <1 s（无 GPU、无权重） | 本仓库单测实测 |

## 阻塞清单（P4 需用户放行后才能执行）

1. **GPU 全量**：`--reader icae` 需要冻结 E3 checkpoint + CUDA；本轮只跑 CPU stub。
2. **QueryParser 解析能力（最重要）**：`QueryParser`（t6）是模板式解析，只认
   `X lives in _X_` 这类结构；LongMemEval 的自由问答（"How many years ago did I…"）
   大多解析不出 `(subject, attribute)` → `no_state_match`，分级桶会大量落在
   `state_match_failed`。**全量跑之前必须先定口径**：扩展查询侧解析，或明确把
   `no_state_match_rate` 作为主指标之一并如实报告。
3. **官方 judge**：`OPENAI_API_KEY` 与账单授权（t14 链路）。
4. **数据文件**：官方 S 数据路径未硬编码，需用户确认可作 `--ref`（见
   `docs/LME_S_OFFICIAL_EVAL_SPEC.md`）。
5. **`recency_min_confidence` 校准**：未在独立验证集校准前不得设非零值当结论；
   校准属于放行清单。

## 外部发现（不在本任务范围，未修改）

`lme_s_text_rag/make_official_eval_input.py` 的 `--runs-dir` 只接受默认的 `Path`：
命令行传字符串会 `TypeError: unsupported operand type(s) for /: 'str' and 'str'`。
该文件属 t14，本任务不改；因此本脚本的 `--runs-dir` 默认值对齐它的默认 runs 目录。
