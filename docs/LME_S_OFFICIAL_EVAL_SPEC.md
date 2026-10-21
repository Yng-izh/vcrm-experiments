# LongMemEval-S 官方 eval 契约规格（静态核查版）

Writer: run-monitor (t12, attempt 809b1ccf)。仅静态核查：未运行官方 eval、未调用 judge API、未用 GPU。
核查主机/来源：v100-2 服务器文件系统 + 本地 clone `vcrm-experiments-review/results/v1002_phase12/`。
每条事实均标注证据路径；查不到的项在 §6 明确列出「未找到」。

## 1. 数据位置与结构

**官方完整数据（含 S/M/oracle 说明）**：`/share/latent2/LongMemEval/`（官方 repo clone，含 `src/evaluation/evaluate_qa.py`、`print_qa_metrics.py`；README 有官方字段说明与调用命令）。

**服务器上实际使用的 S 版数据（500 题）**：
- `/share/latent2/vcrm_icae/data/longmemeval_s_cleaned.json`（277,383,467 bytes，2026-10-13）— cleaned S 版全量。
- `/share/latent2/vcrm_icae/data/prepared/longmemeval_test.json`（252,663,737 bytes）— 本项目 prepared 化版本（500 条，字段 `dataset/events/group/id/queries/split`，为 csr_v1 事件流格式，非官方 schema；官方对齐用 `longmemeval_s_cleaned.json`）。
- 未找到 `longmemeval_m.json`（已查：`/share/latent2/vcrm_icae/data/`、`/share/latent2/hf/longmemeval/` 未展开确认，官方 repo `data/` 目录仅 `custom_history`，数据文件本身不在 repo 内）。

**单条样本字段（实测 `longmemeval_s_cleaned.json[0]`）**：
```
answer, answer_session_ids, haystack_dates, haystack_session_ids,
haystack_sessions, question, question_date, question_id, question_type
```
- `question_id`: 8-hex，如 `"e47becba"`；官方规则：qid 以 `_abs` 结尾 = abstention 题。
- `question_type`: 六类之一 `single-session-user / single-session-assistant / single-session-preference / temporal-reasoning / knowledge-update / multi-session`（实测样本 = `single-session-user`）。
- `haystack_sessions`: session 列表（实测样本 0 有 53 sessions）；每个 session 是 turn 列表，turn = `{"role": "user"|"assistant", "content": str}`（evidence turn 官方另有 `has_answer: true`，实测 cleaned 版首样本未见该字段，但官方 README 声明其存在）。
- `haystack_dates`: 与 sessions 等长的时间戳列表（实测 53）。
- `answer_session_ids`: 证据 session id 列表（session-level recall 用）。

**S vs M（官方 README，`/share/latent2/LongMemEval/README.md` §Dataset Format）**：S 版全 history 拼接约 **115k tokens（~40 sessions）**；M 版每题约 **500 sessions**；oracle 版仅含证据 session。三文件各 500 题。

## 2. 官方 eval 入口与调用方式

**官方脚本**：`/share/latent2/LongMemEval/src/evaluation/evaluate_qa.py`（及聚合器 `print_qa_metrics.py`、`print_retrieval_metrics.py`）。

**官方调用**（README 原文，占位符）：
```bash
export OPENAI_API_KEY=YOUR_API_KEY
export OPENAI_ORGANIZATION=YOUR_ORGANIZATION   # 单组织可省
cd src/evaluation
python3 evaluate_qa.py gpt-4o <hyp_file> ../../data/longmemeval_oracle.json
python3 print_qa_metrics.py <hyp_file>.log ../../data/longmemeval_oracle.json
```
- `evaluate_qa.py` 3 参数：`metric_model_short`（仅支持 `gpt-4o`→`gpt-4o-2024-08-06`、`gpt-4o-mini`→`gpt-4o-mini-2024-07-18`、`llama-3.1-70b-instruct`(local, `http://localhost:8001/v1`, key="EMPTY")）、`hyp_file`（jsonl，每行 `{"question_id","hypothesis"}`，也接受 json 数组）、`ref_file`（官方数据 json）。
- 输出：`<hyp_file>.eval-results-<model_short>`（每行 = 原 hypothesis 条目 + 新增 `autoeval_label: {"model","label"}`），脚本尾部直接打印 overall + per-type accuracy。

**上一轮 phase12 的用法（反推自 `results/v1002_phase12/`）**：
- `阶段2基线报告.md` L15：*"评分采用 LongMemEval 官方提示词、用户提供服务的 Qwen3.5-9B judge。API 正/反控制通过；**不是官方 GPT-4o 分数**"* —— 即上次**没有跑官方 gpt-4o**，是官方 prompt + 自建 judge。
- `config.json`：`judge_model="Qwen3.5-9B"`、`judge_base_url="http://123.207.199.196:18888/v1"`、`judge_key_env="OPENAI_API_KEY"`。
- 产物 `lme_predictions_scores.tar.gz`：`episodes/*.json`（500 个，vcrm episode 格式：含 `queries/store_bytes/retained_entry_ids` 等）+ `scores/*.json`（500 个，字段 `id/group/scores[]`，其中 `scores[0]` 含 `query_id/type/accuracy/judge_model/protocol/response/evidence_diagnostics`，protocol 字段自证 "Official LongMemEval prompt; custom judge, not official GPT-4o scores"）。
- 数据面：报告 L5 指出运行目录 `runs/lme_streaming_no_eviction`（v100-2 `/share/latent2/vcrm_v1002_phase12/runs/` 下存在），7 GPU 分片 × 500 cleaned-S 问题。

## 3. prediction 文件 schema（官方 evaluate_qa.py 消费形态）

- jsonl（或 json 数组），每行/元素仅两字段：
  - `question_id`：必须能匹配 ref_file 中的 `question_id`；不匹配 → 打 `Warning: skipping ... not in reference data` 并跳过（**不会报错**，静默丢分）。
  - `hypothesis`：模型回答文本（字符串）。
- 不需要 `retrieval_results`/`sessions` 等字段（多余字段会被原样带进输出 log，无害）。
- `hypothesis` 缺失 → `KeyError`（`entry['hypothesis']` 直接索引）。
- abstention 判定由 qid 后缀 `_abs` 触发，使用专门的 unanswerable prompt（见 §4）。

## 4. judge 与指标

- **官方 judge**：`gpt-4o`（实际模型 `gpt-4o-2024-08-06`，硬编码于 `evaluate_qa.py` `model_zoo`）；`temperature=0`，`max_tokens=10`，`n=1`。
- **重试**：`@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))` 指数退避，无上限。
- **prompt 模板**：按 question_type 分 5 模板（single-session-user/assistant、multi-session 共用一档；temporal-reasoning 有 off-by-one 宽容；knowledge-update 只认更新后答案；single-session-preference 按 rubric）+ abstention 模板（模型需识别不可答）。全部为 yes/no 判定，`label = 'yes' in response.lower()`。模板全文在 `evaluate_qa.py` `get_anscheck_prompt()`。
- **指标口径**：
  - `evaluate_qa.py`：overall accuracy（全部有 log 的行）+ per-type accuracy。
  - `print_qa_metrics.py`：**per-type（固定六类）→ Task-averaged Accuracy（六类宏平均）+ Overall Accuracy（微平均）+ Abstention Accuracy（`_abs` 题单独）**。注意：`print_qa_metrics.py` 硬断言 `autoeval_label.model == 'gpt-4o-2024-08-06'`，非官方 judge 的 log 不能用它聚合。
- **环境变量**：`OPENAI_API_KEY`（必需，`os.getenv`）、`OPENAI_ORGANIZATION`（可选）。官方 README 要求 export；上轮 phase12 走自建 base_url 时同样读 `OPENAI_API_KEY`（`config.json: judge_key_env`）。

## 5. 对齐要求（text-RAG 脚本产出 → 官方 eval）

**最小形态（官方直喂）**：
```
<predictions_dir>/predictions.jsonl
```
每行：`{"question_id": "<8-hex，与 longmemeval_s_cleaned.json 的 question_id 完全一致>", "hypothesis": "<模型答案文本>"}`

调用模板（不执行）：
```bash
export OPENAI_API_KEY=${OPENAI_API_KEY:?}
cd /share/latent2/LongMemEval/src/evaluation
python3 evaluate_qa.py gpt-4o <PRED.jsonl> <REF.json>       # REF = 官方 S 数据 json
python3 print_qa_metrics.py <PRED.jsonl>.eval-results-gpt-4o <REF.json>
```
注意 ref_file 官方示例用 oracle.json；对 S 版全量评测应传 S 数据（字段同构，evaluate_qa.py 只用 question_id/question_type/question/answer）。

**一致性要求**：
1. 500/500 全覆盖 —— 缺行不会报错而是静默降低分母，最终报告必须核对行数 = 500。
2. `question_id` 必须与 cleaned 数据原值逐字符一致（上次 phase12 的 episodes/scores 以 `group`（qid）关联，可交叉验证）。
3. 若使用非官方 judge（如自建 Qwen），需在产物中标注 `protocol` 字段说明（参照 phase12 scores 的写法），且不能引用 `print_qa_metrics.py` 的输出作为官方分数。
4. abstention 题（`_abs` 后缀）必须包含在 prediction 文件中，judge prompt 自动切换。

## 6. 未找到 / 待确认

- `longmemeval_m.json`：服务器未找到（已查 `/share/latent2/vcrm_icae/data/`、`/share/latent2/hf/longmemeval/`、`/share/latent2/LongMemEval/data/`）。对 S 版任务无影响。
- 官方 repo `data/` 下无数据文件（仅 `custom_history`），官方 README 的 `../../data/longmemeval_oracle.json` 路径在本机需换成实际数据位置（如 `/share/latent2/vcrm_icae/data/longmemeval_s_cleaned.json` 或 HF 下载到 `/share/latent2/hf/longmemeval/`）。
- `print_qa_metrics.py` 的六类固定列表与 ref_file 中实际 question_type 集合是否完全一致未逐一验证（cleaned-S 首条为 single-session-user，README 声明六类 + abstention）。
