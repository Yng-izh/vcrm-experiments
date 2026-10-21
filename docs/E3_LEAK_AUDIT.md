# P0 · E3 现有结果泄漏审计（Ours v6 / EM 0.8512 / F1 0.9346）

- 审计对象：`probe/ours_probe.py`（v6/E3 全量跑的那一版）+ `vcrm/{reader,store,runtime,index,codec,data,features,metrics}.py` + `c1_hybrid_rag/scoring.py` + `probe/ours_v6_summary.py`
- 被审计产物：`probe/runs/ours_v6_34963_shard{0,1}.jsonl`（34,963 行，md5 `59d4d16e941e233ceb2b4649ae32177e`）及其派生 `results/ours_v6_summary.json`、`results/full_test_report.md`
- 审计方式：**纯静态代码审读 + 既有产物（jsonl 派生摘要）交叉核对 + 汇总数字内部一致性算术核对**。
  未跑 GPU、未改任何代码、未重跑评测；本地无 jsonl 与 `/share/latent2` 数据，凡需逐行重算的点均显式标为「需运行验证」并给出命令。
- 代码基线：`vcrm/` 目录下 `*.py` 修改时间均为 2026-10-19 16:05（即被审计版本）。本会话内队友新增的
  `vcrm/data_views.py`、`vcrm/fact_schema.py`、`vcrm/state_index.py`（2026-10-21 11:3x）**不属于**本次被审计代码，
  它们只是 P0–P4 的新边界，后文第 7 节单独说明其对本次结论的影响。

---

## 1. 结论（分层，附置信度）

| # | 结论 | 判定 | 置信度 |
|---|---|---|---|
| C1 | **不存在代码级 oracle 泄漏**：在 v6/E3 实际执行的路由/读取路径上，没有任何一行读取 `state_key` / `as_of` / `answers` / `latest`；prompt 中除问题文本外不含任何历史原文或 gold 文本 | 无泄漏 | **高**（静态可判定，逐行闭合） |
| C2 | **存在"标签等价"泄漏（协议级，不是代码级）**：`query.time` 在 prepared TempLAMA 中**与 `as_of` 完全同一字符串**，且等于**金标版本的 event 时间戳**。因此 mask 掉 `as_of` 是**形式性**的；`select_version(state, t)` 在 t 上必然命中金标版本 | **有泄漏（等价通道）** | **高**（`data.py:47-50` 与 `data.py:44` 为同一 `row['date']`） |
| C3 | **0.8512 不能解释为"自动事实关联 / 版本路由能力"**：TempLAMA 每个 episode 只有一个 state，`SemanticIndex.associate` 退化为「单候选 + 0.8 阈值判定」，路由层唯一可失败点是阈值（5.37% fallback）；成功即 100% 命中金标版本 | 归因不成立 | **高** |
| C4 | **高 EM 的真实来源 = 基准构造 + latent 可读性 + ex28 提取**：每版 event 文本由 `row['query'].replace('_X_', answers)` 生成，答案天然写在被检索版本自己的文本里，再经 ICAE 编码成 128 slot；读取正确版本即等于读到答案 | 有"基准内答案可见"性质，非控制层泄漏 | **高** |
| C5 | **fallback（5.37%）不能解释高 EM，反而拉低 0.0165 EM**：fallback 子集 EM 0.5612 远低于 latent 子集 0.8677 | 无泄漏贡献 | **高**（数字来自 `ours_v6_summary.json`，逐行重算见 §5-R1「需运行验证」） |
| C6 | `CurrentSnapshotStore.build` 与 `SlotMemoryRuntime.answer` **没有答案回填路径**；`runtime.answer` 内部确实存在 `state_key` / `as_of` 的 oracle 分支，但 probe 不调用它 | 无回填 | **高** |
| C7 | 历史版本 code 以**该 episode 最终版 memory 为 anchor** 编解码（`store.py:420-423`、`206-212`），即历史读取在结构上引入了"未来版本"的信息条件 | 方向应为**损害**历史题而非抬高（故不解释高 EM），但需实测 | **中**（静态确定其存在，影响方向**需运行验证**） |
| C8 | ex28 提取规则本身**不可能读 answers**（纯函数，只吃 raw 字符串），但其**形态是在看过 481 diag 标签后冻结的**，且提取增益极大（raw EM 0.0042 → extracted 0.2308） | 非泄漏，但属"调参口径"风险 | **中高** |

**一句话结论**：**没有发现"控制层把 gold 喂给 reader"的泄漏；但 0.8512 也不是 latent 记忆或自动关联能力的证据**——因为在该基准上"目标版本"由 query 自带的时间戳唯一确定（`time ≡ as_of ≡ 金标版本时间`），而答案又写在该版本自己的文本里。审计发现的是**协议级标签等价**，不是代码级读取。对论文而言，0.8512 只能作为 **"给定时间戳 + 给定 state 分组下的 latent 版本可读性"** 指标引用；作为"自动事实关联"证据引用会被审稿人一击命中。

---

## 2. 逐条核查（对应任务 1–5）

### 2.1 masked query view 是否真被 mask；FORBIDDEN_ATTRS / AST 审计是否覆盖 route.*

| 文件:行号 | 事实 | 影响 |
|---|---|---|
| `probe/ours_probe.py:24` | `FORBIDDEN_ATTRS = {"state_key","as_of","answers","latest"}` | 禁用字段清单只覆盖 4 个 oracle 字段；`route`、`type`、`key_text`、`entity`、`group` 均不在清单内 |
| `probe/ours_probe.py:44-46` | `mask_view` 用**白名单**构造：`{k: query[k] for k in ("id","question","time") if k in query}` | mask 本身是可靠的：`state_key/as_of/answers/latest` 物理上不进入 view |
| `probe/ours_probe.py:201-202` | `view = mask_view(q)`；`assert not (set(view) & FORBIDDEN_ATTRS)` | 该断言**恒真**（view 由白名单构造），是"安慰剂断言"，不构成独立校验 |
| `probe/ours_probe.py:27-41` | 定义了 `ast_audit()`，但**全仓从未调用**（`grep -n "ast_audit"` 只命中两处 `def` 行：本文件 :27、`probe/ours_probe.py` 旧版 :25） | **所谓"AST 审计"实际未执行**；现有 `probe/runs/ast_scan_report.json` 是**人工撰写的裁决 JSON**（含 `raw_hits:[[172,"state_key"],[196,"answers"],[199,"answers"]]` 与手写 `adjudication`），不是 `ast_audit()` 的产物 |
| `probe/ours_probe.py:38-40` | `ctx_ok = getattr(node, "_masked_ok", False)`，但代码中从未给任何 AST 节点设置 `_masked_ok` | 即使调用 `ast_audit()`，也会把 mask_view 白名单里的 4 个字符串常量全部误报为 hit；审计函数自身不可用 |
| `probe/ours_probe.py:213-219, 226, 229-232, 239` | `route` 字典字段：`status/path/fallback_scope/fallback_corpus/near_miss/t/state/selected_id/is_current/association_similarity/fallback_events` | `route.*` **不含任何 oracle 字段值**；但 `FORBIDDEN_ATTRS` 与 AST 审计**完全没有覆盖 route**（未列入清单、未做断言） |
| `probe/ours_probe.py:222` + `vcrm/index.py:43` + `vcrm/store.py:410` + `vcrm/data.py:45` | `route["state"] = assoc["state_key"]` ← `ranked[0][1]['state']` ← store entry `'state'` ← **写入侧** `event['state_key']` | `route["state"]` 的**值**与 `q["state_key"]` 完全相同，但**通道是 memory 而非 query**。在 TempLAMA 上二者不可区分（见 2.3），所以"通道不同"这一辩护**无法在 TempLAMA 上被证伪** |
| `vcrm/data.py:47-50` | query 同时写入 `'time': date` 与 `'as_of': date`（同一 `date` 变量） | **mask 掉 as_of 而保留 time 是空操作**：等价信息仍在 view 里 |
| `vcrm/reader.py:173-175` | `date = query.get('as_of') or query.get('time')`；`prefix = 'As of' if query.get('as_of') else 'Current date'` | 有/无 as_of **只改措辞不改日期值**：masked 视图仍把 `2018-12-31T23:59:59` 写进 prompt（前缀变成 "Current date:"）。这是"as_of 已被 mask"这一说法最直接的反证 |
| `vcrm/index.py:60-65` | `inferred_as_of()`：`if 'as_of' in query: return query['as_of']`，否则正则从 question 里抠年份 | probe 传的是 masked view（无 as_of 键）→ 走正则；该函数内部的 as_of 分支是**死路径**，但保留在共享模块里，任何人把原始 query 传进来就会触发 oracle 读取 |
| `vcrm/runtime.py:89-92, 99` | `state = query.get('state_key')`；`select_version(state, inferred_as_of(query))` | 同上：runtime 层保留了 oracle 分支（`state_key` + `as_of`），probe 绕过它（见 2.5） |

### 2.2 `reader.encode_event` 输入；`bundle()` / `VersionMarkers` 是否夹带答案文本或 gold

| 文件:行号 | 事实 | 影响 |
|---|---|---|
| `vcrm/reader.py:126-127` | `event_text(event) = f"DATE: {event['time']}\n{event['text']}"` | 编码输入**只有 time + text**，无 state_key/answers/latest |
| `vcrm/reader.py:129-130` | `encode_event(event) = self.encode(event_text(event))` | 同上 |
| `vcrm/features.py:63-70` | 逐 event 调 `reader.encode_event(event)`，event 来自 `ingestion_view(episode)` | 每版 memory 来自该版自己的文本 |
| `vcrm/features.py:21-22` | 载入时校验 `metadata['event_sha256'] != digest(event)` 则报错 | memory↔event 强绑定，**无法把别的 event（或最终答案）的 memory 塞进来**；probe 走的正是这条缓存载入路径（`probe/ours_probe.py:130`） |
| `vcrm/data.py:136-139` | `ingestion_view` 白名单 `('id','time','text','state_key','key_text')` | 写入视图**不含** `answers`/`latest`/`as_of`（但含 `state_key`、`key_text`，见 2.3） |
| `vcrm/store.py:292-303` | `bundle()`：逐 entry `decode()` → 可选 `markers(memory, role, time)` → `torch.cat` | bundle 是**纯 latent 张量拼接**，不含任何文本 |
| `vcrm/codec.py:185-201` | `VersionMarkers`：`roles`（2×4096 学习参数）+ `time_projection`（4 维数值：年/月/日/时） | 标记槽只编码 **role + 该 entry 自己的时间数值**，无 gold、无文本 |
| `vcrm/reader.py:177-196` | `prompt_embeddings`：`text` 与 `bundle` **互斥**（:179 抛错）；prompt = `question(query)`（+ 可选 `Context:\n{text}`）+ left/right 指令 token + bundle | latent 路径下 prompt 文本部分**只有问题本身** |
| `probe/ours_probe.py:244` vs `:212` | latent 路径 `reader.answer(view, bundle)`（无 text）；仅 fallback 路径 `reader.answer(view, text=fb_text)` | 两条路径在代码上被强制分开，不存在"latent + 原文"混合读取 |
| `vcrm/reader.py:233` | 返回值含 `'raw_history_access': bool(text)` | 每行自带"是否读到原文"的机器可判定标记（latent 行应为 False）→ 可作为 §5-R2 的运行期证据 |

### 2.3 写入侧：每版 memory 是否来自该版自己的 event text，而非该 episode 的最终答案

| 文件:行号 | 事实 | 影响 |
|---|---|---|
| `vcrm/data.py:41,44` | `text = row['query'].replace('_X_', ' / '.join(answers))`；event = `{'id','time','text': f"In {date}: {text}", 'state_key', 'key_text'}` | **每版 event 文本天然包含该年的答案**（TempLAMA 构造：发布事实即数据流）。这不是控制层注入，但意味着"读到正确版本 ≈ 读到答案" |
| `vcrm/data.py:43-46` | 仅当 `answer_set` 变化才发新 event | 版本事件是"值变更点"；某年值未变则该年没有独立 event（影响 §5-R3 的 `selected_id == query_id` 统计口径） |
| `vcrm/store.py:400-407` | `build(events, memories, keys)`：要求 `set(memories)==set(ids)`、`set(keys)==set(ids)`，并显式禁止"query-dependent filtering" | 写入侧**不接受**任何答案/查询输入；快照必须包含全部版本（含相对 t 的未来版本） |
| `vcrm/store.py:404,410,420` | `any(not e.get('state_key') ...)` 报错；`state = event['state_key']`；`state, latest = event['state_key'], current[event['state_key']]` | **写侧分组用的是数据集自带的 `state_key` 标签（oracle）**：`CurrentSnapshotStore` 根本不具备"自动关联"能力，它是"按 oracle state 分组 + 按 oracle 时间轴对齐" |
| `vcrm/training.py:17-28, 68, 109` | `adaptation_samples` 只接受 `split=='train'`；codec 训练对 `(old, current)` 按 `state_key` 配对 | ① codec **未在 test 上训练**（无 test 泄漏）；② 训练侧同样依赖数据集 `state_key` 分组（口径需在论文中披露） |
| `vcrm/store.py:420-423` | `anchor = memories[latest['id']]`（该 episode **最终版** memory）；历史版 `payload = codec.encode(memory, anchor)` | 历史版 code 以最终版为条件 → 结构上引入"未来版本"信息（C7，方向需实测） |
| `vcrm/store.py:206-212` | `decode()` 用同一 `anchor`（`_entry(entry['anchor_id'])`） | 读取历史版时同样依赖最终版 memory；bundle 中 current 永远在列（`probe:236`） |
| 结论 | **未发现"用 episode 最终答案回填历史 memory"的路径** | 每版 memory 严格来自该版自己的 event text（`features.py:21` 的 digest 校验是硬约束） |

### 2.4 打分侧：ex28 是否可能读到 answers；fallback 占比与影响

| 文件:行号 | 事实 | 影响 |
|---|---|---|
| `c1_hybrid_rag/scoring.py:23-87` | `extract_answer(raw)`：**签名只接受 raw 字符串**，全部逻辑是正则/切分 | 结构上**不可能**读到 answers；"ex28 用答案反推"不成立 |
| `c1_hybrid_rag/scoring.py:104-114` | `score_query(prediction, answers, extracted)`：answers 只进 `em_score/f1_score` | answers 仅在评分时使用 |
| `probe/ours_probe.py:245-246` | `ex = extract_answer(result["answer"])`；`score_query(result["answer"], q["answers"], extracted=ex)` | 顺序正确：先出答案，后碰 gold |
| `vcrm/metrics.py:13-14, 17-25` | `exact` 用 `any(...)`；`f1` 取 `max(...)` over answers | TempLAMA 存在多 gold 的行（同年多位任职者）→ **任一命中即 EM=1**，EM 偏宽松；影响量需运行验证（§5-R4） |
| `c1_hybrid_rag/scoring.py:15-19, 99-100` | 注释写明 "Frozen on 481 diag: EM 0.2308 / F1 0.3943 (raw: EM 0.0042 / F1 0.1471)" | ① 0.8512 是 **extracted 口径**，与 raw 生成口径不可比；② 规则形态在看过 diag 标签后冻结 → 属"设计自由度"披露项，非逐题泄漏 |
| `probe/ours_probe.py:179-192` | `fallback_retrieve`：**只在本 episode 自身未压缩事件**上检索，`time <= t` 过滤，top-10 全文 | fallback 不是"全局检索泄题"；报告 `full_test_report.md:34` 记录越界 0 条 |
| `results/ours_v6_summary.json:10-17` | `latent 33,085 / fallback_text 1,878 / fallback_rate 0.0537 / out_of_episode 0` | 与任务给定的 5.37% 一致；`0.9463+0.0537=1.0`，两分支互斥且穷尽（`probe:213` 与 `:229` 都写 `route.path`） |
| `results/ours_v6_summary.json:39-77` | latent 0.8677/0.9444；fallback 0.5612/0.7615 | **fallback 显著低于 latent** → 对总体是**拖累**而非抬高 |
| 算术核对（本次实算） | `(33085×0.8677 + 1878×0.5612)/34963 = 0.85124`；`(30906×0.8527 + 4057×0.8398)/34963 = 0.85120` | 汇总数字**内部自洽**（overall / latent+fallback / hist+latest 三个口径互相闭合）→ 不存在"分子分母口径错配"式的虚高 |
| 反事实 | 若 fallback 行按 latent 均值计分，overall 应为 **0.8677**（+0.0165） | **fallback 占比再高也不会解释 0.85**；它对结论是负贡献，只需分列披露 |

### 2.5 `CurrentSnapshotStore.build` 与 `runtime.answer()` 是否有回填路径

| 文件:行号 | 事实 | 影响 |
|---|---|---|
| `vcrm/store.py:399-445` | `build()` 入参只有 `events / memories / keys`；产出 row 字段只有 `id/state/time/kind/keyframe/anchor_id/compression_count/key/payload` | **无 answers 入参、无 gold 字段**，回填路径不存在 |
| `vcrm/store.py:437-443` | 失败时 `self.entries, self.blocks, self.counts = [], {}, {}` 回滚 | 无"半构建即读"的脏状态 |
| `vcrm/runtime.py:73-80` | `write_fact` 要求显式 `state_key`；`CurrentSnapshotStore` 直接抛错 | probe 不调用（probe 自建 store，`probe:126-135`） |
| `vcrm/runtime.py:88-114` | `answer()`：`state = query.get('state_key')`（:89）→ 无则走 `index.associate`（需 `validated_index`，:91-97）→ `select_version(state, inferred_as_of(query))`（:99）→ `bundle(...)`（:108）→ `reader.answer(query, bundle)`（:114） | **存在 oracle 分支（state_key + as_of），但无答案回填**：prompt 仍只由 question + bundle 组成 |
| `probe/ours_probe.py:212, 244` | probe 直接调 `reader.answer(...)`，**从不调用** `runtime.answer` | v6/E3 结果不经过该 oracle 分支 → 2.1 的"路由路径无 oracle 读取"成立 |
| `probe/ours_probe.py:119` | `assert isinstance(runtime.store, CurrentSnapshotStore) or True` | **恒真断言**（`or True`），未真正校验 store 协议类型；E3 的 `memory_protocol=final_current_snapshot_v1` 只由 `runtime.from_checkpoint:30,56` 间接保证 |
| `vcrm/runtime.py:30-34, 56-58` | 校验 `memory_protocol` 一致、`codec_training.json.completed`、checkpoint sha256 | 协议/权重身份校验链完整（这部分是可信的） |

---

## 3. 为什么 0.8512 不是"自动关联"：TempLAMA 上的路由退化（关键证据）

| 文件:行号 | 事实 | 影响 |
|---|---|---|
| `vcrm/data.py:26-31` | `groups[(entity, relation)].append(row)` → 每个 `(entity, relation)` 一个 episode | **一个 episode 只有 1 个 state** |
| `vcrm/data.py:45,48,51-53` | 该 episode 所有 event / query 的 `state_key` 都是 `f'{entity}:{relation}'`；`episode['group']` 同值 | 一个 store 内所有 entry 同 state |
| `vcrm/index.py:36-44` | `associate`：`top1 < 0.8 → None`；`second = 下一个不同 state 的分数，默认 -1`；`top1 - second < 0.05 → None` | 单 state 时 `second = -1`，**margin 条件恒满足** → 关联退化为"单候选 + 0.8 阈值"；`no_association` 只可能由阈值触发（5.37%） |
| `vcrm/store.py:288-290` | `select_version(state, as_of)`：取 `time <= as_of` 中 `max(time)` | 由于 `query.time == 金标版本 event.time`（`data.py:44` vs `:47`），**t 唯一确定金标版本** |
| `probe/runs/full_summary.json`（C16 全量产物）`route.route_hit_note` | "contract §4.3: on route_ok queries the routed version always matches gold; verified" | 既有产物已自证：**路由成功的题，路由版本 100% 等于金标版本** → 路由层对 EM 无区分度贡献 |
| `results/full_test_report.md:35` | latest 4,057 题中 3,859 走 latent，且 `route.is_current` **全为 True（0 例外）** | latest 题"路由"恒等于 current——因为 t = 该 episode 最后一行日期 |
| `probe/summarize.py:86` | `"hist_upper": "E3 historical_f1 0.9237 (oracle routing, post-hoc val)"` | E3 的 oracle 路由 val F1 = 0.9237；本次"label-free" test hist F1 = **0.9353 ≥ oracle 值** → 只有"label-free 路由等价于 oracle 路由"才能解释这种持平/反超 |
| `results/full_test_report.md:5` | conflict 切片 34,407/34,963（98.4%），定义为"同 state_key 存在 ≥2 个不同 answer set" | 单 state/episode 下该切片 = "多版本 episode"，**不构成冲突消解测试**；conflict EM 0.8603 同样只是版本可读性 |

**推论（C2+C3）**：`mask_view` 挡住了 `as_of` 字段名，却保留了与它逐字节相同的 `time`；`inferred_as_of` 与 `reader.question` 又都把 `time` 当日期用。所以"零 oracle 接触"在**字段层面**成立，在**信息层面**不成立。这是本次审计的核心发现。

---

## 4. 需运行验证的点（不能静态判定的，全部列明验证方法）

> 以下每条都给出"为什么静态判不了 + 具体怎么验 + 判据"。命令均以服务器 `ROOT=/share/latent2/vcrm_icae` 为准，**均为 CPU 或 ≤10 分钟 GPU 级操作**，不涉及全量重跑。

**R1 · fallback 占比与其对总体的影响（逐行复核）**
- 为何需跑：`ours_v6_summary.json` 是派生摘要，本地无 jsonl；本次只能验证其**内部自洽**，无法验证其**如实反映 jsonl**。
- 验法（CPU，秒级）：
  ```bash
  python3 - <<'EOF'
  import json,glob
  R=[json.loads(l) for p in glob.glob('/share/latent2/vcrm_icae/probe/runs/ours_v6_34963_shard*.jsonl') for l in open(p) if l.strip()]
  from collections import Counter
  print(len(R), Counter(r['route'].get('path') for r in R))
  fb=[r for r in R if r['route'].get('path')=='fallback_text']
  print('fb n',len(fb),'fb em',sum(r['em_extracted'] for r in fb)/len(fb))
  print('raw_history_access any True on latent?', any(r.get('raw_history_access') for r in R if r['route'].get('path')=='latent'))
  EOF
  ```
- 判据：`path` 计数 = 33,085/1,878；latent 行 `raw_history_access` 全为 False/缺失。

**R2 · 读取期"无原文"证据**
- 为何需跑：`reader.answer` 的 `raw_history_access`（`reader.py:233`）未写入 jsonl（`probe:247-255` 未落该字段）。
- 验法：在 jsonl 生成侧加该字段后重跑 100 题（`--limit 100`，单卡 ~4 分钟）；或直接对 `reader.prompt_embeddings` 做单测断言 `text==''`（`vcrm/reader.py:177-196`，CPU 可 mock）。
- 判据：latent 路径 100% `raw_history_access=False`。

**R3 · "选中版本 == 金标版本"的精确比例**
- 为何需跑：需把 `route.selected_id` 与 prepared 数据对齐（本地无 `prepared/templama_test.json`）。注意 `data.py:43` 只在值变更时发 event，故 `selected_id == query_id` 只对"变更年"成立。
- 验法（CPU）：对每个 query，取 `episode.events` 中 `time <= q.time` 的最后一个 event，断言其 `id == route.selected_id`，且该 event 文本包含 `q.answers[0]`。
- 判据：`selected == last_event(time<=t)` 应为 **100%**；`event.text` 含 gold 的比例给出"答案在所选版本文本内"的精确占比（预期 ≈100%，除多值/别名情形）。

**R4 · 多 gold 行的 EM 宽松度**
- 验法（CPU）：统计 `len(q['answers'])>1` 的 query 数与其 EM。
- 判据：若占比可观（如 >5%），需在论文中注明 EM 为 any-match。

**R5 · anchor 未来信息（C7）的方向与幅度**
- 为何需跑：`codec.encode(old, anchor=最终版)` 的信息渗漏量是权重相关的，静态判不了。
- 验法（GPU，单 episode 级）：把历史 entry 的 anchor 从"最终版"换成"t 时刻版"重建快照，在 481 val 的历史题上对比 F1；同时对 `codec.decode(code, anchor)` 与 `anchor` 本身做余弦相似度统计。
- 判据：若 F1 无显著上升，则 anchor 条件化不构成"未来信息抬高"；若 decode 输出与 anchor 余弦 >0.9，则历史 code 事实上在复述最终版。

**R6 · ex28 规则的"调参泄漏"边界**
- 为何需跑：`scoring.py:16` 只记录"frozen on 481 diag"，无法静态判断该 481 与 test 是否实体重叠。
- 验法（CPU）：核对 `data/prepared/entity_split_audit.json`（`data.py:119-124` 生成）与 481 diag 的 query 集合交集。
- 判据：若 481 diag ⊂ val 且与 test 实体不相交 → 只是设计自由度，非泄漏。

---

## 5. 若被质疑的最小可补实验（按成本从低到高，全部可在 CPU 或单卡完成）

| # | 实验 | 成本 | 它回答什么 | 预期/判据 |
|---|---|---|---|---|
| E1 | **时间戳消融**：把 `t` 从 `q.time` 改为 `t = episode 最终时间`（即恒读 current），只对历史题统计 EM | CPU 离线（复用现有 jsonl + prepared 数据，**无需 GPU**） | 0.8527 的历史 EM 里有多少来自"用 query 自带日期选中金标版本" | 历史 EM 应大幅跌落（落到"读最终版"的水平）；若几乎不跌，说明日期未起作用，需重审 |
| E2 | **`time` 遮蔽对照**：把 query 视图再收紧到只留 `{id, question}`（`t` 取 episode 末时刻），重跑 100–500 题 | 单卡 <1h | "保留 time"是否就是等价 as_of 的泄漏通道 | 与 E1 同向；两者一起构成"时间信息贡献"的上界估计 |
| E3 | **空 memory 对照**（`bundle=None` 或随机 episode 的同位版本） | 单卡 ~30min/100 题 | reader 的**参数化先验**贡献（即"不给记忆也能答对多少"） | 若空记忆 EM 仍高，说明答案来自模型先验而非 latent；这是审稿人最常问的一刀 |
| E4 | **多 state 关联测试**：在 longmemeval（或多 state 合成 episode）上测 `associate` 的 state 准确率 | CPU + BGE 嵌入（现有 `RetrievalIndex`） | 证明"自动事实关联"到底有没有被验证过（TempLAMA 上退化为单候选） | TempLAMA 上无法给出该指标；这是 P1/P2 新模块必须补的评测位 |
| E5 | **写侧 state 标签替换**：用 `data_views.public_write_event`（新 P0 边界，去掉 state_key）建 store，让 P1/P2 的 StateIndex+Router 自行分组，与 oracle 分组版对比 | CPU（新代码路径） | "写侧 oracle 分组"对结果的贡献 | 这是 P0–P4 改造的**主实验**：oracle 分组 vs 自动分组 的 EM 差 |
| E6 | **extraction 对照**：同批行同时报 raw 与 extracted 两口径 | 纯算术 | 0.8512 里有多少是 ex28 带来的 | diag 上 raw 0.0042 → extracted 0.2308，test 上需同口径分列 |

---

## 6. 附：审计中发现的代码卫生问题（不影响上述结论，但影响"可复核性"）

| 文件:行号 | 问题 | 建议 |
|---|---|---|
| `probe/ours_probe.py:27-41` | `ast_audit()` 是死代码，且 `_masked_ok` 机制未实现 | 要么实现并纳入 CI（对 `probe/` + `vcrm/` 全量扫描 + 白名单裁决表），要么删除，避免"我们有 AST 审计"的口头承诺 |
| `probe/ours_probe.py:202` | `assert not (set(view) & FORBIDDEN_ATTRS)` 恒真 | 改成对**原始 query** 的键做断言（证明"我们没有把它传进去"），或删除 |
| `probe/ours_probe.py:119` | `assert isinstance(...) or True` 恒真 | 去掉 `or True` |
| `probe/v6_loader.py:57` | `f"...got {config.get(codec_value_path)!r}"` —— `codec_value_path` 是**未定义变量**，E3/E4 守卫触发时会抛 `NameError` 而非预期 `ValueError` | 改为 `config.get("codec_value_path")`（本次 E3 路径未触发，故不影响已产出的 0.8512） |
| `vcrm/index.py:60-65` / `vcrm/store.py:93-94` / `vcrm/runtime.py:89,99` / `vcrm/codec_eval.py:66` / `vcrm/comparison.py:20` | oracle 分支散落在**共享模块**里，仅靠"probe 不调用"来保证 label-free | P0 新增的 `data_views.py` / `fact_schema.py` 方向正确；建议再加一条"Predicted 路径禁止 import `vcrm.comparison`/`vcrm.codec_eval`"的结构性约束（import 图断言，CPU 可测） |
| `results/full_test_report.md:47` | 报告自己已标注"EM 0.85 需泄漏审计" | 本文即该项；建议在报告主表加注："Ours 数字为 extracted 口径 + 时间戳给定条件下的版本可读性" |

---

## 7. 对 P0–P4 改造的直接含义（给 captain / boundary-engineer）

1. **`time` 必须被当作 oracle 等价字段处理**。新边界 `vcrm/data_views.py` 的 `PUBLIC_QUERY_KEYS = ('question','time')`（`data_views.py:18`）在当前 TempLAMA 上**仍然把金标版本时间戳交给 Predicted 路径**（`data.py:47` 里 `time` 与 `as_of` 同值）。若目标是"不依赖目标 episode / 不读 state_key"，则 TempLAMA 的 `time` 也必须重定义（例如用"观测窗口末刻"或"上一已知时间"），否则 P1–P4 的新模块会在一个**已被时间戳解掉**的任务上被评估，指标依旧不可解释。
2. **写侧分组目前 100% 来自数据集 `state_key`**（`store.py:404,410`；`training.py:68,109`）。P2 的 `StateIndex+StateRouter` 必须先在"去 state_key 的写视图"上重建同一条 pipeline，才能声称"自动事实关联"；这也是 §5-E5 的主实验。
3. **TempLAMA 无法验证关联能力**（单 state/episode，§3）。关联/冲突消解的评测位应放到 longmemeval 或多 state 合成集，否则 conflict 切片（98.4% 的题）永远只是"多版本可读性"。
4. **E3 的既有 val 成绩（hist F1 0.9237）是 oracle routing 口径**（`probe/summarize.py:86` 自注）。本次 label-free 0.9353 与之持平/略高，恰是"二者等价"的证据，不宜写成"label-free 超过 oracle"。

---

## 8. 复核方式（本文所有断言的可重复检查）

```bash
# 1) 禁用字段是否出现在路由路径（只看 probe 与 reader 的读取路径）
grep -n "state_key\|as_of\|answers\|latest" vcrm-experiments-review/probe/ours_probe.py   # 命中行：24,44-46,201-202,246-249（定义/断言/评分侧）
# 2) AST 审计是否真的跑过
grep -rn "ast_audit" vcrm-experiments-review/                                           # 只有 def，无调用
# 3) time 与 as_of 是否同值（基准构造）
sed -n '39,50p' vcrm-experiments-review/vcrm/data.py                                    # time/as_of 同取 row['date']
# 4) bundle / markers 是否含文本
sed -n '292,303p' vcrm-experiments-review/vcrm/store.py; sed -n '185,201p' vcrm-experiments-review/vcrm/codec.py
# 5) ex28 是否读 answers
sed -n '23,114p' vcrm-experiments-review/c1_hybrid_rag/scoring.py                        # 提取函数只吃 raw
# 6) 汇总数字内部一致性（本次已实算，见 §2.4）
python3 -c "print((33085*0.8677+1878*0.5612)/34963, (30906*0.8527+4057*0.8398)/34963)"
```
