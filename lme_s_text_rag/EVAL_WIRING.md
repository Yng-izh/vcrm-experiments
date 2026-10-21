# LongMemEval-S 官方 eval 接线（t14）

整条链路：**run_text_rag（产生 hypothesis）→ make_official_eval_input（对账+转官方格式）→ 官方 evaluate_qa.py（gpt-4o judge）→ print_qa_metrics.py（官方聚合）→ 归档**。

**原则：结果一律以官方 eval 输出为准（`*.eval-results-*` 与 `print_qa_metrics.py` 的打印），本链路任何脚本不自行计算指标。**（t12 spec §4：`print_qa_metrics.py` 硬断言 judge 为官方 gpt-4o-2024-08-06；自建 judge 的分数不可引用为官方分。）

## 环节与路径

| 步骤 | 脚本 | 输入 | 输出 |
|---|---|---|---|
| 1. 生成 hypothesis | `run_text_rag.py`（t13，暂未执行） | 官方 S 数据 + vLLM reader | `lme_s_text_rag/runs/<tag>/<question_id>.json`（8 字段，含 raw hypothesis） |
| 2. 对账 + 转格式 | `make_official_eval_input.py --tag <tag>` | `runs/<tag>/*.json` + 官方 S 数据（qid 全集） | `lme_s_text_rag/official_eval_in/<tag>/predictions.jsonl`（每行仅 `question_id`,`hypothesis`）；对账摘要打印 n_input/n_out/n_missing/n_extra/n_empty，任何缺口即 fatal 不写文件 |
| 3. 官方 judge | `${LME_REPO}/src/evaluation/evaluate_qa.py gpt-4o <PRED.jsonl> <REF.json>` | predictions.jsonl + ref | `<PRED>.eval-results-gpt-4o`（每行含 `autoeval_label`），脚本尾部打印 overall/per-type accuracy |
| 4. 官方聚合 | `print_qa_metrics.py <PRED>.eval-results-gpt-4o.log <REF>` | judge log | Task-averaged / Overall / Abstention Accuracy（官方口径） |
| 5. 归档 | `run_official_eval.sh` step 3 | — | `lme_s_text_rag/official_eval_out/<tag>/`（predictions + eval-results + log） |

## 一键入口（默认 dry-run）

`run_official_eval.sh` 默认只打印命令；`--execute` 才真跑。顶部已列所需环境变量名：`OPENAI_API_KEY`（必需）、`OPENAI_ORGANIZATION`（可选）。可覆盖变量：`LME_TAG / LME_DIR / LME_REPO / LME_REF / LME_JUDGE_MODEL`。

## 阻塞项 / TODO（执行前必须确认）

- 【阻塞项】官方 S 数据路径：t12 spec §6 指出官方 repo 不含数据文件；当前默认 `/share/latent2/vcrm_icae/data/longmemeval_s_cleaned.json`（cleaned-S，字段与官方同构），执行前需用户确认可作官方 ref_file。
- 【阻塞项】`run_text_rag.py` 尚未执行（t13 code-only）：500 题 hypothesis 尚不存在，需先起 vLLM（单卡）跑完 `runs/<tag>/`。
- 官方 judge 需 OPENAI_API_KEY 账单授权，执行前由用户放行。

## 待用户放行后的命令清单（占位符）

```bash
# 0) 环境（服务器 v100-2）
export OPENAI_API_KEY=<YOUR_KEY>            # 仅 judge 用；不写入任何文件
# 可选：export OPENAI_ORGANIZATION=<YOUR_ORG>

# 1) 起 reader vLLM（单卡），跑 t13 产出 hypothesis（占位）
python3 -m vllm.entrypoints.openai.api_server --model <MISTRAL_PATH> \
  --served-model-name mistral-7b --dtype half --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --port <PORT> &
python3 lme_s_text_rag/run_text_rag.py --tag <TAG> --data <OFFICIAL_S_JSON> ...

# 2) 一键官方 eval（先看 dry-run，再 --execute）
bash lme_s_text_rag/run_official_eval.sh                    # dry-run：打印命令
LME_TAG=<TAG> bash lme_s_text_rag/run_official_eval.sh --execute

# 3) 读取官方分数：official_eval_out/<TAG>/ 下的 eval-results 与 print_qa_metrics 打印
```

## 同协议第二方法：Latent E3（t7，P4）

`lme_s_latent/run_e3.py` 产出**同一形状**的 `runs/<tag>/<question_id>.json`
（含 `question_id` + `hypothesis`，多出的诊断字段不影响本链路），因此上面三个
步骤**原样可用**，只是 `--tag` 换成 latent 的 tag：

```bash
# 1) latent 侧产出 hypothesis（code-only 阶段用 stub；GPU 全量需用户放行）
python3 lme_s_latent/run_e3.py --data <OFFICIAL_S_JSON> --reader icae --arch-exp E3 \
    --tag lme_s_latent_e3_on --shard 0 --shards 4 --recency-update
# 2) 官方链路（runs 目录默认就是 lme_s_text_rag/runs，无需额外参数）
LME_TAG=lme_s_latent_e3_on bash lme_s_text_rag/run_official_eval.sh            # dry-run
LME_TAG=lme_s_latent_e3_on bash lme_s_text_rag/run_official_eval.sh --execute  # 需 OPENAI_API_KEY
```

注意：`make_official_eval_input.py` 的 `--runs-dir` 传字符串会
`TypeError`（该文件属 t14，本链路未改），所以 latent 脚本的 `--runs-dir` 默认
对齐本链路的默认 runs 目录 `lme_s_text_rag/runs`。两种方法的 tag 必须不同
（如 `lme_s_text_rag_*` vs `lme_s_latent_e3_*`），否则对账会混。
