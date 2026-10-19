c3_icae_full_rag: 评测代码从 v100-2 同步（md5 双侧核验一致），仅 .py；数据/日志/runs 不入库

## Protocol: Controlled State Routing (csr_v1)

C3 运行于 **Controlled State Routing rev1**（`csr_v1`）协议，见
[`PROTOCOL_CSR_V1.md`](../PROTOCOL_CSR_V1.md)，并正式命名为
**ICAE-Full (Controlled)** —— 不再称为 RAG：C3 通过 latent ICAE memory slots
按 episode bundle 读取，走官方 `vcrm.reader.FrozenReader` 读取协议
（`FrozenReader.answer(query, bundle=...)`，内部构建 `As of: {date}` 前缀问题），
不做文本检索。`conflicting` 采用统一元数据口径（同 `state_key` 存在 ≥ 2 个
不同 answers 集）。新运行写入 `runs_csr_v1/`。

> **SUPERSEDED**：`csr_v1` 之前、未走官方 FrozenReader 读取协议的 C3 结果
> 标注为 SUPERSEDED（pre-csr 协议），仅留档，不得引用，也不得以 "RAG" 名义引用。
