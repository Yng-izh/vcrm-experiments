# 四方法统一验证脚本

入口：`run_comparison.py`。默认 `--phase prepare`，只准备清单与配置，不加载模型、不占 GPU，不训练任何模块。

## 已固定的协议

- 原 481 道历史题，262 道答案集合完全冲突的子集；追加这些题对应 172 个 Episode 各自的唯一 Latest QA，共 653 题。原始事件不因筛选问题而减少。
- 所有方法保存全部观察版本，使用相同的已知 `state_key + as_of`。历史题读取最终当前版本与所选历史版本；Latest QA 读取最终当前版本与最近历史版本，不重复注入当前版本。
- Greedy、最多 64 个输出 token；上下文超限报错，不截断。评分为现有答案别名的归一化 EM / token F1，不调用 LLM judge。
- Text Version Log：保存 UTF-8 文本、版本元数据和原有索引 key；读取当前与目标历史的文本，带角色和日期标签。使用原生 Mistral 的官方 chat template，不加载 ICAE LoRA，不添加 ICAE 专属 task token。
- ICAE-Full：官方 ICAE checkpoint 与原缓存，每个版本 128 槽位；使用当前 + 历史双版本输入及本项目学习的角色/时间标记。`full_marker_run` 必须显式记录；默认选择对应轮次的 C16 标记权重。这不是官方论文结果，也不是保证高于 C/D 的理论上界。用另一套标记应另起 run prefix，不能悄悄替换或挑选最优结果。
- C16/D16：加载指定轮次的完整训练 checkpoint，按现有离线最终当前快照实现保存全部历史 16 槽位与当前 128 槽位。两者仍使用各自训练得到的标记权重，脚本不改变已训练模型。

## 存储与成本

所有方法先保存 `.memory`，重载后再问答，并校验逐字节 roundtrip。Storage 是这 172 个 Episode 的平均实际文件大小，包含所有版本、索引 key、元数据及格式开销，不是只计算被读取的两个版本。JSON 同时保留 bytes、十进制 MB（除以 1,000,000）和 MiB（除以 1,048,576）；表格用 MB。旧表的 1.475 不直接填入新表。

共享模型/Codec/标记权重及离线原始特征缓存不计入每 Episode 的 `.memory`，不宣称系统总占用更小；这些依赖的来源/哈希记录在每个方法的 `protocol.json`。记录实际 prompt tokens、KV cache bytes、读取时间；写入时间从已有缓存开始，不包含 ICAE 编码。Text 与 latent 的 prompt 模板不同，不能把性能差异完全归因于存储表示。Text 不强制与 latent 等字节预算；C/D 的槽位预算与张量 payload 字节一致，但格式元数据可能有少量字节差异，不宣称文件逐字节等大。

## 运行

服务器目录：`/share/latent2/vcrm_icae`。

准备已完成的 aligned_v3 轮次：

```bash
.venv/bin/python run_comparison.py --plan config.comparison.json --phase prepare
```

只跑两个外部基线（以下卡号仅为示例，程序只接受当时空闲的卡）：

```bash
.venv/bin/python run_comparison.py --plan config.comparison.json --phase evaluate --methods text,full --gpus 4,5
```

补 C/D 的同协议 653 题与最新准确率，再生成四方法表：

```bash
.venv/bin/python run_comparison.py --plan config.comparison.json --phase evaluate --methods C,D --gpus 4,5
.venv/bin/python run_comparison.py --plan config.comparison.json --phase report
```

也可一次跑完全部方法并汇总：

```bash
.venv/bin/python run_comparison.py --plan config.comparison.json --phase all --gpus all
```

每方法一张卡；不足四张时分批执行。卡数/配置保持不变时，重跑可按已完成 Episode 和 partial query 续跑。GPU 锁与显存空闲检查防止抢占训练卡。未完成全部四方法时拒绝输出正式主表。

本次正在训练的 matched_v4 有独立配置：`config.comparison.matched.json`。可先 `--phase prepare`，等训练完成后再执行 `--phase evaluate/all`；未完成的训练 checkpoint 无法通过前置检查。不要把两个轮次合并成同一条 C/D 结果。

输出位于 `runs/comparison_aligned_v3_suite/` 或 `runs/comparison_matched_v4_suite/`：固定选择、派生配置、任务状态、日志；每方法单独保存 stores、逐题预测、protocol 和 summary。正式汇总为 `TABLE.md`、`table.json`，含全部方法对之间的 Episode-level paired bootstrap 95% CI。F1/Accuracy 在 JSON 是 [0,1]，Markdown 是百分数。

这仍是已参与调参的验证诊断，不是未触碰 test。脚本准备不等于真实 GPU 评测通过，也不会自行启动实验。
