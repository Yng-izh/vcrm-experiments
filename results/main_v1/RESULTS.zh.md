# 四组 500 题全量评测结果（main_v1）

数据：官方 LongMemEval-S cleaned `data/longmemeval_s_cleaned.json`（500 题，全部计入）。
运行位置：v100-2 `/share/latent2/vcrm_icae/runs/main_v1/`（GPU 8/9）。
判分：官方 LongMemEval 判分 prompt，judge 为 `Qwen3.5-9B`（**非**官方 GPT-4o 分数），
`judge_errors = 0`，`scored_rows = 500`。

## 1. 主表（overall accuracy）

| 组 | 方案 | 检索/写入 | 读取 | overall |
|---|---|---|---|---|
| **A** | text_rag | 词法检索（BM25+BGE，c1_hybrid_rag） | 原始 session 文本 | **0.412** |
| **B** | state_text | 状态路由（vcrm 关联层选状态/版本） | 原始 session 文本 | **0.084** |
| **C** | current_only | latent 写入，仅当前版本 | latent reader | **0.244** |
| **D** | versioned | latent 写入，全版本 + 语义路由 | latent reader | **0.246** |

## 2. 分题型（accuracy_by_type）

| 题型 | n 占比 | A | B | C | D |
|---|---|---|---|---|---|
| single-session-user | — | 0.2143 | 0.1429 | 0.2571 | 0.2571 |
| single-session-assistant | — | 0.3750 | 0.0893 | 0.3929 | 0.3929 |
| single-session-preference | — | 0.1667 | 0.0667 | 0.0000 | 0.0000 |
| multi-session | — | **0.5263** | 0.0376 | 0.2180 | **0.2256** |
| temporal-reasoning | — | **0.5414** | 0.0376 | 0.1805 | 0.1805 |
| knowledge-update | — | 0.2949 | 0.1923 | 0.3718 | 0.3718 |
| **overall** | 500 | 0.4120 | 0.0840 | 0.2440 | 0.2460 |

## 3. 结论

1. **D − C = +0.002**，且差异**只出现在 multi-session**（0.2256 vs 0.2180，+0.76pt）；
   其余五个题型逐位相同。在 LongMemEval-S 上，本架构的版本化写入几乎不产生额外收益。
2. **A（文本 RAG）0.412 显著高于 C/D（0.244/0.246）**：latent 读取链路
   （ICA→压缩状态→reader）单点信息损失大于纯文本检索+长上下文读取的收益。
   A 的优势集中在 multi-session（0.526）与 temporal-reasoning（0.541）。
3. **B 异常低（0.084）**：B 与 A 同为原始文本读取，唯一差别是候选 session 由状态路由给出，
   因此低分归因于**状态选择器的召回**而非 reader（reader 已确认走真实 vLLM，
   答案均长 910 字符、44 条空答案，无占位符）。
   直接证据：A/B 在 temporal-reasoning 上 0.5414 vs 0.0376、multi-session 上 0.5263 vs 0.0376。
4. **single-session-preference 在 C/D 上为 0**，而在 A 上为 0.1667 —— 偏好类问题依赖
   原话措辞，latent 压缩后不可恢复。

## 4. 复现

```bash
cd /share/latent2/vcrm_icae
eval "$(grep -m1 '^export VCRM_LLM_KEY=' /root/.bashrc)"

# 四组任选 --group {text_rag,state_text,current_only,versioned}
CUDA_VISIBLE_DEVICES=8 env_py310/bin/python probe/run_main_eval.py \
  --group versioned --data data/longmemeval_s_cleaned.json \
  --runs-root runs/main_v1 --defer-judge \
  --reader icae --arch-exp E3 --checkpoint runs/architecture_v6_E3_train \
  --router-mode semantic --frontend llm \
  --llm-config llm_config.json --llm-cache-root runs/main_v1/llm_cache \
  --tag d
```

A/B（文本组）必须显式指定真实文本 reader，否则默认 `--text-reader stub` 只产生
`stub-text:<len>` 占位符（B 首轮曾因此跑出 0.002 的无效结果，已作废重跑）：

```bash
  --text-reader vllm --vllm-url http://111.231.174.164:6012/v1/completions \
  --vllm-model Qwen3.5-9B
```

判分：`python3 /tmp/judge_main_eval.py <predictions.jsonl> judge.json <judge_cache> <stem>`
（`judge.json` 为 judge 模型配置，`llm_config.json` 不能替代）。

## 5. 目录内容

```
results/main_v1/
  a/summary.json            # A 运行配置/计数
  a/judged_a_summary.json   # A 判分汇总
  a/judged_rows.jsonl       # 500 行：答案 + 判分
  b|c|d/…                   # 同上
  logs/{a,b,c,d}.log        # 四组运行日志（逐题进度、缓存命中、抽取并发）
  logs/judge_{a,b,c,d}.log  # 四组判分日志
```

## 6. 方法与注意事项

- 写入时机在每个 group 结束时批量提交；中途崩溃靠 `llm_cache`（共享，10k+ 条目）续跑，
  重启不重复付费；抽取为主入口内的串行路径（早期 t34 用的 `--extract-workers`
  并发开关没有并入 `probe/run_main_eval.py`，命令里写它会直接 unrecognized arguments）。
- 结果不可跨 GPU 逐字节比较（F7：不同 GPU 上 `write_bytes` 不具可比性），
  锚定指标用 `decision_counts` / `state_count`。
- 判分为自定义 judge，与官方 GPT-4o 数字不可直接对齐；同一 judge 下四组横向可比。
