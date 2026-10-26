# 存储字节构成与预算标定（IncrementalCurrentStore）

本文件回答两件事：**持久化字节到底花在哪**、**怎么用真实测量定 `byte_budget`**。
代码侧的唯一实现是 `vcrm/store.py::IncrementalCurrentStore`；两条预算策略
（`retain_all` / `drop_oldest`）在同一份写入结果上切换，字节口径完全一致。

## 1. 文件布局（`serialize()` 的真实字节顺序）

```
┌ magic            8 B     b'VCRMIC1\n'
├ header_len       8 B     uint64 little-endian
├ header_json      H B     主 header（entries 描述 + blocks/counts/interval + codec 身份）
├ payload 区       Σ_e     entries 的 key+payload 原始 FP16 字节，按 entries 顺序紧密排列
├ aux_magic        8 B     b'VCRMIP1\n'
├ aux_len          8 B     uint64 little-endian
├ aux_json         A B     pending 描述 + duplicates + rejections + budget_mode
└ aux payload 区   Σ_p     pending 记录的 key+payload
```

因此

```
total_bytes = 8 + 8 + H + Σ_{e∈entries}(key_e + payload_e)
            + 8 + 8 + A + Σ_{p∈pending}(key_p + payload_p)
```

`accounting()` 直接给出分项（`KeyframeStore.accounting`）：

| 字段 | 含义 | 公式 |
| --- | --- | --- |
| `memory_bytes` | latent 载荷 | `Σ_e payload_e.nbytes` |
| `index_key_bytes` | 语义索引 key | `Σ_e key_e.nbytes = n_entries × key_dimensions × 2` |
| `metadata_and_format_bytes` | 其余全部 | `total_bytes − memory_bytes − index_key_bytes`（含 header_json、magic、长度字段、aux 段） |
| `total_bytes` | 真实持久化字节 | 上式，带 `_accounting_total` 缓存 |

## 2. 单条版本的成本（E3 正式配置）

| 条目类型 | payload | key | 合计（不含 header 分摊） |
| --- | --- | --- | --- |
| current（`kind='full'`） | `128 × 4096 × 2 = 1,048,576 B`（1 MiB） | `128 × 2 = 256 B` | ≈ 1.0002 MiB |
| historical（`kind='code'`） | `16 × 4096 × 2 = 131,072 B`（128 KiB） | 256 B | ≈ 128.25 KiB |
| pending（DEFER 原始 full） | `128 × 4096 × 2 = 1,048,576 B` | 256 B | ≈ 1.0002 MiB |

header 里每条 entry 的 JSON 描述（`id/state/time/kind/keyframe/anchor_id/
compression_count/offset/key_bytes/payload_bytes/shape`）实测约 180–220 B，
随 `state`/`id` 字符串长度浮动；`metadata_and_format_bytes / n_entries`
就是这一项的真实分摊值。

推论（决定预算量级的关键）：

* 每个状态**至少** ≈ 1 MiB（current 不可淘汰），这是 `drop_oldest` 的硬地板；
* 历史 code 每条约 current 的 1/8，所以"版本链"不是字节瓶颈，
  **pending 才是**（一条 DEFER = 一整个 full memory）；
* 4057 个事实组全 current ≈ 4.06 GiB，加 1 条历史/状态 ≈ 4.6 GiB。

## 3. 真实测量（先测再定预算）

```bash
env_py310/bin/python probe/measure_store_bytes.py --episodes 50 \
    --dataset /path/to/longmemeval_s.json --out runs/store_bytes.json
```

脚本对每个 episode 逐条写入并记录 `accounting()` 的**前后差值**，因此
`payload / index / metadata` 三项之和恒等于该次写入带来的真实字节增量：

```
delta_total = delta_memory_bytes + delta_index_key_bytes + delta_metadata_bytes
```

输出结构（`--out` 的 JSON）：

```jsonc
{
  "config":  {"dimension": 4096, "slots": 128, "codec_slots": 16, "key_dimensions": 128, ...},
  "episodes": [
    {"episode_id": "...", "writes": 12,
     "episode_write_bytes": {"payload": 0, "index": 0, "metadata": 0, "total": 0},
     "per_write": [{"id": "...", "action": "UPDATE", "state": "...",
                    "delta_bytes": 0, "payload_bytes": 0, "index_bytes": 0,
                    "metadata_bytes": 0, "total_bytes_after": 0}]}
  ],
  "summary": {"episodes": 0, "writes": 0, "payload": 0, "index": 0, "metadata": 0,
              "total": 0, "bytes_per_episode_mean": 0.0,
              "bytes_per_write_mean": 0.0, "per_state_mean": 0.0}
}
```

定预算的经验做法（第四步用）：

1. 用 `summary.bytes_per_episode_mean` × 目标 episode 数 → 无淘汰时的需求；
2. 目标预算 = 该需求 × 压缩比（例如 0.6），把差额交给 `drop_oldest`；
3. 预算**不得低于** `状态数 × (1 MiB + 1 KiB)`（current + 格式开销的地板），
   否则写入会被显式拒绝（这是设计行为，不是 bug）。

## 4. 两条保留策略

| | `retain_all`（默认，主对照） | `drop_oldest`（相同字节预算对照） |
| --- | --- | --- |
| 超预算 | `ValueError` 显式拒绝 + 整笔回滚 | 按固定顺序淘汰，成功返回 |
| 淘汰顺序 | — | ① `pending` → ② `duplicates` → ③ `rejections` → ④ 历史 code 按 `(time, id)` 升序 → ⑤ 仍超预算则 `ValueError` |
| current(full) | 不淘汰 | **永不淘汰**（淘汰它等于丢数据） |
| audit | — | `receipt['evicted']` + `accounting()['evictions'|'evicted_ids'|'evicted_bytes']`，含每条淘汰的 id 与释放字节 |
| 持久化差异 | aux header 记 `budget_mode='retain_all'` | 记 `'drop_oldest'`（多 1 B） |

淘汰只发生在**单次写入的守卫**内，且每次都重新量真实 `serialize()` 字节；
淘汰后必须重新通过 `validate_protocol()`（单 full current 不变量）与预算检查，
任何一步失败都整笔回滚（`_snapshot`/`_restore` 浅快照 + `_wire_cache` /
`_accounting_total` 失效语义与 retain_all 完全一致）。
`load()` 要求 `budget_mode` 与文件一致，不静默换模式。
