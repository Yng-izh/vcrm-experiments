c2_llmlingua2_rag: 评测代码从 v100-2 同步（md5 双侧核验一致），仅 .py；数据/日志/runs 不入库

## Protocol: Controlled State Routing (csr_v1)

C2 与 C1 同步运行于 **Controlled State Routing rev1**（`csr_v1`）协议，见
[`PROTOCOL_CSR_V1.md`](../PROTOCOL_CSR_V1.md)：查询注入明确 as_of 年份
（`build_query_text` → `"As of {as_of}, {question}"`，检索与 Reader prompt 共用）；
检索限 Episode 范围（`retrieve(episode_id=...)`，先过滤候选再做 BM25/BGE+RRF）；
`conflicting` 采用统一元数据口径（同 `state_key` 存在 ≥ 2 个不同 answers 集）。
这不是 End-to-End Retrieval：as_of 年份与 per-episode 限定是显式查询条件。
每行输出含 `protocol: "csr_v1"` 与 `query_text`；resume 拒绝非 csr_v1 行的输出文件。
新运行写入 `runs_csr_v1/`。

> **SUPERSEDED**：`csr_v1` 之前的全部结果（无年份注入、跨 Episode 检索、旧
> conflict 口径）标注为 SUPERSEDED（pre-csr 协议），仅留档，不得引用。
