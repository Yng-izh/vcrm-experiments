# 三步实验调度

入口：v100-2 `/share/latent2/vcrm_icae/run_paper.sh`。
本轮已获用户运行授权；不改变旧 phase12 结果，不占用其他人的任务卡。

```bash
# parity 必须与当前 reader 指纹一致；默认不适配 encoder。
./run_paper.sh --phase all --gpus all
./run_paper.sh --phase status
# 完整验证通过且完成具体错误审查后继续四个 C/D 实验。
./run_paper.sh --phase codec --gpus all --review-file /absolute/path/review.json
./run_paper.sh --phase report
```

## 第一步

空闲 GPU 自动分为两个不重叠的池：约一半评测 filtered val 的 525 个状态组、
4,570 道年度 QA，其余卡编码 filtered train 的 1,215 个状态组。
验证同时落盘特征，训练与验证缓存分目录，特征写入仅使用 ingestion_view。
每段为 `[128,4096]` FP16，标签不进入张量及其旁路元数据。
缓存记录模型/encoder/reader/源数据指纹，并检查 SHA256、形状、精度与有限值。

评测包含未压缩 latent、同条文本、无记忆、错版本。A/B 的单条与时间索引
读取是相同张量的不变量，不冒充两次独立试验，也不称严格数学上界。
全部按当前冻结 Mistral 评分；TempLAMA 使用严格 EM 与 token F1，不使用 LLM judge。
逐题写 partial_episodes，整组完成后进入 episodes，可在同协议下断点续评。

完整验证集按事先记录的 0.8/0.8/0.8 与 0.1 记忆增益门槛决策。
未通过则 status=reader_gate_failed，不启动 codec；通过则等待具体错误审查。
review.json 必须包含 approved=true、gate_run、summary_sha256、review_notes。
不能用少量样本或通过修改阈值给自己开门。

用户明确要求忽略可读性门禁时，可用 `--phase codec --override-readability-gate`
并提供 `--authorization-note`。调度器单独保存 `readability_override.json`，
绑定原始 summary/gate 哈希与用户指令，保留失败门禁文件，不伪造门禁通过或人工审查。
该授权不绕过完整验证、reader/权重身份、特征完整性或 GPU 梯度检查。

## 第二步

四个训练任务 C16/D16/C32/D32 各占一张空闲卡，卡不足则排队分批。
使用同一训练划分、特征、seed、时序对、训练步数与损失权重；不重训 reader。
保存训练与模型指纹，独立验证后固定配置评测未修改的 test
（4,057 个状态组、34,963 道 QA）。这些全量控制组生成量显著大于 100 题 pilot，
耗时必须由实测吞吐决定，不能保证 0.5 天完成。

C/D 都按当前+历史同台读取，历史预算 16/32 slots，版本/时间标记另计 2 slots/条。
保留不可变关键帧，压缩只执行一次。关键帧、当前状态、历史码均纳入相同驻留预算；
单独报告 D 的恢复锚点访问和成本。无淘汰的主表用于隔离 codec，本身不证明淘汰优势。
除全部历史 F1，另报真正压缩的 code 条目 F1，避免关键帧稀释压缩退化。
Table 1 和 Figure 1 仅填实测数据，按状态组做 C/D 配对 bootstrap。
单条 A/B 是参考，与 C/D 的多条注入格式并不严格匹配，不能直接宣称理论上界。

## 第三步前置条件

LongMemEval-S 保持 evaluation-only，不能用于训练、选超参数、调门槛。
已有 span 写入机制，但共享、无查询/黄金标签泄漏的事实更新 resolver 与
query/event 检索校准尚未完成。调度器显式标记 transfer_blocked，
不把所有片段当独立状态的 FIFO 结果伪装成版本条件压缩迁移。
后续必须先完成并核对该协议，才能使用第二步固定 checkpoint 运行 500 题。
旧 TransMem 11.20% 只作研究动机记录，不是相同 reader 的配对性能证据。

## 输出

`runs/paper_v1_suite/status.json`：阶段状态；`active_tasks.json`：本调度器子任务 PID。
`configs/`：冻结派生配置；`logs/`：任务级日志；各任务目录保留逐 GPU 日志。
`features/paper_v1_official/{train,val,test}/`：不可变特征。
`table1.{json,md}` 与 `figure1.png`：有实测结果才生成相应内容。
失败或暂停仅清理本调度器的进程组，不触碰外部任务。
