# P2 无 Oracle 状态写入（fact_extractor / state_index / state_router / update_resolver）

本轮只做代码 + CPU 单测 + 文档：不训练、不跑 GPU、不跑官方评测、不改
`vcrm/codec.py` 与 `vcrm/reader.py`、不动 `vcrm/data.py` 的 Oracle 对照入口。

## 1. 数据流（Predicted 路径）

```
原始 episode（含 state_key/answers/... 的 Oracle 字段）
        │  vcrm.data_views.public_write_event(event, public_id)   ← P0 白名单投影
        ▼
{'id', 'text', 'time'}                    ← 结构上不可能携带 Oracle 字段
        │  FactExtractor.extract(event)                          ← P2
        ▼
list[AtomicFact]（subject / attribute / value / observed_at）
        │  StateRouter.retrieve_candidates(fact, index)           ← P2 两阶段检索
        ▼
candidates: list[StateRecord]
        │  UpdateResolver.resolve(fact, candidates, index)        ← P2 六条规则
        ▼
StateDecision(CREATE | UPDATE | DUPLICATE | DEFER)
        │  apply_decision(decision, fact, index, store)           ← 唯一写入点
        ▼
StateIndex（+ IncrementalCurrentStore 事务钩子 commit_with）
```

状态身份 = `(subject_key, attribute_key)`；`value` 与时间不参与身份，
只在 rule4/rule5 里决定「同值重复」还是「换值更新」。

**未解析事实（t9 F4）**：主体或属性解析不出来的事实**不会**建出哨兵状态，而是
「extractor 标记 unresolved + resolver 判 DEFER + 只进 `StateIndex.pending`」，
`StateRecord` 一条都不产生（比造哨兵状态更安全，也不污染状态计数）。

## 2. 六个交付件

| 文件 | 作用 |
| --- | --- |
| `vcrm/fact_extractor.py` | 确定性规则抽取：主体识别、模板/依存式属性规范化、取值抽取、时间解析；TempLAMA 单事实优先，LongMemEval 多事实骨架（行→句→子句切分 + 代词消解钩子 + 来源 session/turn/时间/字符区间） |
| `vcrm/state_index.py` | `StateRecord` + `StateIndex`（add / commit / merge_pending / get / candidates / serialize / load）+ 规范化函数 + 与 `IncrementalCurrentStore` 的 `commit_with` 事务钩子 |
| `vcrm/state_router.py` | `retrieve_candidates`（阶段 1 主体/属性硬过滤 → 阶段 2 128 维 semantic_key 余弦辅助排序）+ `route` |
| `vcrm/update_resolver.py` | 六条规则 + `apply_decision` 唯一写入点 |
| `probe/build_state_supervision.py` | （只写不跑）从 TempLAMA **train** 构造正样本 / 困难负样本 / 重复样本，实体级 train/val 切分 |
| `probe/train_same_state.py` | （只写不跑）冻结编码器 + 轻量二分类头 + 验证集阈值校准 |

## 3. 规则（`UpdateResolver`）—— 默认口径 = 六条

| 规则 | 条件 | 结果 |
| --- | --- | --- |
| rule1 | 无候选 | `CREATE` |
| rule2 | 候选主体全部不同（拒绝合并） | `CREATE` |
| rule3 | 同主体、属性都不同 | `CREATE` |
| rule4 | 同主体同属性且取值等价（规范形指纹一致） | `DUPLICATE` |
| rule5 | 同主体同属性 + 单值排他属性白名单 + 文本含显式「替代」表达 | `UPDATE`（reason 前缀 `explicit_replacement:`） |
| rule5b | 同主体同属性 + 单值排他属性白名单 + `observed_at` 严格更晚 + 值不等价 | `UPDATE`（reason 前缀 `exclusive_superseded_by_recency:`；**默认关闭**，见第 9 节） |
| rule6 | 多值属性 / 否定 / 候选冲突 / 基数未知 / 无显式替代 / 未解析事实 | `DEFER`（写 pending，不改 Current） |

**口径说明（t9 F5）**：`rule1..rule4 + rule5 + rule6` 是**默认口径的六条**；
`rule5b` 是六条之外的**第七条**，由 `recency_update` 显式开关控制，**默认关闭**，
关闭时行为与六条完全等价（不产生任何额外 UPDATE）。所有 reason 尾部都带
`|cardinality=<EXCLUSIVE|MULTI_VALUED|UNKNOWN>`，供评测按基数口径分桶统计。

排他性 = 属性基数档位 `EXCLUSIVE`（第 10 节三档本体表）∩ 文本显式替代表达
（`now` / `replaced` / `succeeded` / `formerly` / `changed to` / `became` …），
**不是**余弦阈值；余弦只用于候选排序，不参与动作判定。

## 4. 「无 Oracle 写入」调用示例

```python
from vcrm.data_views import PublicEventCounter, public_write_event
from vcrm.fact_extractor import FactExtractor
from vcrm.state_index import StateIndex
from vcrm.state_router import StateRouter
from vcrm.update_resolver import UpdateResolver, apply_decision

counter = PublicEventCounter()          # input_000001, input_000002, ...
extractor = FactExtractor()             # 默认强制公开写视图白名单
index = StateIndex()                    # state_000001, state_000002, ...
router = StateRouter(top_k=5)
resolver = UpdateResolver()

def ingest(raw_event):
    """raw_event 可以带任何 Oracle 字段：投影后只剩 id/text/time。"""
    event = public_write_event(raw_event, counter.next_id())
    for fact in extractor.extract(event):
        decision = router.route(fact, index, resolver)
        apply_decision(decision, fact, index)       # CREATE/UPDATE/DUPLICATE 落索引；DEFER 进 pending
        print(fact.subject, fact.attribute, fact.value, '->', decision.action, decision.reason)

ingest({'id': 'Q42_2015', 'text': 'In 2015: The head of government of France is Manuel Valls',
        'time': '2015-01-01T00:00:00', 'state_key': 'Q42:P35', 'answers': ['Manuel Valls'],
        'latest': False, 'episode': 'ep_0001'})
# France head_of_government Manuel Valls -> CREATE rule1:no_candidate

ingest({'id': 'Q42_2016',
        'text': 'In 2016: Bernard Cazeneuve replaced Manuel Valls as head of government of France',
        'time': '2016-01-01T00:00:00', 'state_key': 'Q42:P35', 'answers': ['Bernard Cazeneuve'],
        'latest': True, 'episode': 'ep_0002'})
# France head_of_government Bernard Cazeneuve -> UPDATE explicit_replacement:head_of_government
```

带 `IncrementalCurrentStore` 的写入（store 成功才更新索引，索引失败回滚 store）。
**四条动作都必须透传 `memory=`（原始 full memory）与 `semantic_key=`**：

```python
decision = router.route(fact, index, resolver)
apply_decision(decision, fact, index, store,
               memory=raw_full_memory,      # 该事实的原始 128-slot full memory
               semantic_key=semantic_key)   # 128 维语义键
```

也可以直接走事务钩子（签名与实现一致，`memory` 是关键字参数）：

```python
index.commit_with(store, decision.state_id, fact,
                  memory=raw_full_memory, semantic_key=semantic_key,
                  value_fingerprint=value_fingerprint(fact.value),
                  observed_at=fact.observed_at, decision=decision)
```

`commit_with` 契约（实测，t18 修正）：`store.commit(record=..., fact=..., memory=...,
semantic_key=..., decision=...)` 必须返回非空版本号（或 `{'version_id': ...}`）；
store 抛错 → `StoreCommitError` 且索引原样；store 成功但索引更新失败 →
调用 `store.rollback(...)` 并抛 `IndexCommitError`。真实 `IncrementalCurrentStore.commit`
对 CREATE / UPDATE / DUPLICATE / DEFER **四种动作都强制要求 `memory=`**
（DEFER 的 pending 要保留原始 full memory），缺失即 `StoreCommitError`，不会静默降级。

## 5. 阈值校准政策（写进脚手架的硬约束）

- 同状态判定的阈值**必须**在**独立验证集**上校准：`probe/train_same_state.py`
  的 `calibrate_threshold()` 在实体级互斥的 val 划分上按「F1 最大且 FPR ≤ max_fpr」选点。
- **禁止硬编码 0.75**：CLI 传入 `--threshold` 直接报错退出；
  `FORBIDDEN_HARDCODED_THRESHOLD = 0.75` 仅用于在校准结果恰好等于它时打警告。
- **禁止**在 test 划分上调阈值；`build_state_supervision.py` 只允许 `--split train`。
- 编码器冻结：`FrozenEncoder` 的 `requires_grad_(False)`，只训练 `SameStateHead`。

## 6. 防泄漏证据

- `FactExtractor` 默认 `enforce_public_view=True`，入口即
  `assert_no_oracle(event, allowed=PUBLIC_WRITE_KEYS)`：带 `state_key` 的原始事件直接 `ValueError`。
- `StateRecord.to_dict()` / `PendingMerge.to_dict()` / `StateIndex.serialize()` 的键集固定，
  测试用哨兵字段 + `assert_no_oracle` 双向证明。
- `state_id` / `fact_id` 由系统分配（`state_%06d` / `fact_%06d` 或 uuid4），
  与数据集标签无编码关系；测试断言公开编号与源事件 id 无字符串包含关系。

## 7. 本轮验证命令（CPU，无 GPU）

```bash
cd vcrm-experiments-review
export PYTHONPYCACHEPREFIX="$PWD/.pycache_check"   # 避免写到工作区外的缓存目录
python -m py_compile vcrm/fact_extractor.py vcrm/state_index.py \
    vcrm/state_router.py vcrm/update_resolver.py \
    probe/build_state_supervision.py probe/train_same_state.py
PYTHONPATH=. python -m pytest tests/test_fact_extractor.py tests/test_state_router.py \
    tests/test_update_resolver.py -q
# 没有 pytest 时等价命令：
PYTHONPATH=. python tests/test_fact_extractor.py
PYTHONPATH=. python tests/test_state_router.py
PYTHONPATH=. python tests/test_update_resolver.py
```

## 8. 已知边界（本轮不做）

- 抽取是规则式的：属性词表（`ATTRIBUTE_ALIASES`）与动词模板是手工枚举的，
  未见过的说法会落到「未解析事实」（低置信度 + DEFER），不会静默误合并。
- `rule5` 的替代表达依赖文本线索；TempLAMA 逐年陈述句（"In 2016: ... is X"）
  本身不含 `now/replaced` 一类词，因此默认判 DEFER，需要 pending 复核，或显式打开
  第 9 节的 `recency_update` 策略开关（默认关闭）。
- 语义键排序未接真实编码器（`semantic_key_fn` 由调用方注入 `SemanticIndex.key`），
  单测只用 128 维单位向量验证「排序不影响判定」。

## 9. `recency_update` 显式策略开关（默认关闭）

### 9.1 为什么需要它

TempLAMA 的写入流是逐年陈述句（`In 2016: The head of government of France is X`），
句子本身不含 `now` / `replaced` / `succeeded` 一类**替代**表达。默认策略下
`rule5` 永远不触发，于是每个 `subject + attribute` 只有第一年的事件成为 Current，
之后每一年都落 DEFER pending —— 版本链建不起来，P3/P4 的 Predicted State 在
TempLAMA 上退化成「单版本 + 一堆待确认」，E5（去 `state_key` 写视图 vs oracle 分组）
也会因为「没有 UPDATE」而失去意义。

这不是抽取错误，而是**策略选择**：逐年陈述句是否构成「后者覆盖前者」的证据，
取决于数据生成假设（时间戳即有效期）。因此把它做成显式开关，而不是默认行为。

### 9.2 行为定义（rule5b）

```
recency_update=True 且
  同主体 + 同属性
  且 cardinality_of(attribute) == 'EXCLUSIVE'  # 单值档（第 10 节三档本体表）
                                               # MULTI_VALUED / UNKNOWN 永不触发
  且 parse(new_fact.observed_at) > parse(record.current_observed_at)   # 严格更晚
  且 值不等价（rule4 未命中）
  且 confidence >= recency_min_confidence     # 校准门槛
⇒ StateDecision(action='UPDATE', state_id=<current>,
                confidence=<由时间间隔决定的饱和分>,
                reason='exclusive_superseded_by_recency:<attribute>')
```

- `observed_at` 相等、更早、或任一侧不可解析（含 `'unknown'`）→ 仍落 DEFER
  （reason `rule6:recency_not_applicable`），不猜。
- 置信度：`floor + (ceiling - floor) * min(gap_days / gap_days_full, 1)`，
  默认 `floor=0.5`、`ceiling=0.85`、`gap_days_full=365`，全部是构造参数。
- `recency_update=False`（默认）时**所有分支与 t5 完全一致**，不改变任何既有语义。

### 9.3 UPDATE 来源分口径

`rule5` 与 `rule5b` 的 reason 前缀不同，评测器可以分开统计：

| 前缀 | 含义 |
| --- | --- |
| `explicit_replacement:<attribute>` | 文本里有显式替代/接任表达（rule5） |
| `exclusive_superseded_by_recency:<attribute>` | 仅凭观察时间更晚（rule5b，需开关打开） |

### 9.4 为什么默认关闭 / 打开后的风险

1. **旧 Current 会被不可逆压缩**：按 t4 的 UPDATE 语义，被覆盖的版本进入历史并被
   压缩存储；一旦 rule5b 判错（例如同一属性其实有多值、或时间戳是「记录时间」而非
   「生效时间」），旧值只能从压缩历史里恢复，代价高且可能已丢精度。
2. **时间戳语义假设**：`observed_at` 严格更晚 ⇏ 语义上「替代」。对
   `population` / `area` 这类单调属性通常成立；对 `spouse` / `director` /
   `residence` 这类可反复取回旧值的属性，简单「后来居上」会系统性错。
3. **评估口径污染**：把 rule5b 混进默认路径会让 E5 的「Predicted UPDATE 数量」
   不可比，因此必须靠 reason 前缀分口径，且默认关闭时口径即「无 rule5b」。

### 9.5 使用建议

- 只在明确接受「时间戳即有效期」假设的实验里打开：
  `UpdateResolver(recency_update=True)`。
- **`recency_min_confidence` 必须在独立验证集上校准**（实体级互斥切分，
  见 `probe/build_state_supervision.py` + `probe/train_same_state.py` 的
  `calibrate_threshold`），不得硬编码，也不得用 test 划分调参。
- 建议报告三组数字：`recency_update=False`（保守基线）、`True` 且门槛校准后、
  以及 rule5/rule5b 分口径的 UPDATE 数量与准确率。

### 9.6 同一组逐年陈述句的决策序列对比

输入：`France / head_of_government` 的 2016 / 2018 / 2020 三年陈述句
（`observed_at` 分别为 `2016-01-01T00:00:00` / `2018-01-01T00:00:00` / `2020-01-01T00:00:00`）。

| 年份 | `recency_update=False`（默认） | `recency_update=True` |
| --- | --- | --- |
| 2016 | `CREATE` conf=0.90 `rule1:no_candidate` | `CREATE` conf=0.90 `rule1:no_candidate` |
| 2018 | `DEFER` conf=0.30 `rule6:no_explicit_replacement` → pending_000001 | `UPDATE` conf=0.85 `exclusive_superseded_by_recency:head_of_government` |
| 2020 | `DEFER` conf=0.30 `rule6:no_explicit_replacement` → pending_000002 | `UPDATE` conf=0.85 `exclusive_superseded_by_recency:head_of_government` |
| Current | `fact_2016`（版本链未建立，pending=2） | `fact_2020`（版本链 2016→2018→2020，pending=0） |

复现：

```python
from vcrm.fact_schema import AtomicFact
from vcrm.state_index import StateIndex
from vcrm.state_router import retrieve_candidates
from vcrm.update_resolver import UpdateResolver, apply_decision

for flag in (False, True):
    index, resolver = StateIndex(), UpdateResolver(recency_update=flag)
    for year, value in (('2016', 'Bernard Cazeneuve'), ('2018', 'Edouard Philippe'),
                        ('2020', 'Jean Castex')):
        fact = AtomicFact(id=f'fact_{year}',
                          text=f'In {year}: The head of government of France is {value}',
                          observed_at=f'{year}-01-01T00:00:00', subject='France',
                          attribute='head_of_government', value=value,
                          source_id=f'input_{year}')
        decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
        apply_decision(decision, fact, index)
        print(flag, year, decision.action, decision.confidence, decision.reason)
```

## 10. 属性基数本体表与 `cardinality_policy` 消融开关

### 10.1 为什么需要一张显式表（t9 F3 的根因）

t5 的两张集合自相矛盾：`employer` / `spouse` / `occupation` / `alma_mater` /
`country_of_citizenship` 同时落在 `MULTI_VALUED_ATTRIBUTES` 且不在
`EXCLUSIVE_ATTRIBUTES`。而 `rule5` 要求「单值」、`rule5b` 显式排除「多值」，
于是两条 UPDATE 路径在这些核心属性上**都不可达**，
`(Alice, employer, Google→Microsoft)` 永远 DEFER —— 这既是任务②的示例，
也是 P1 parity fixture 的 `Alice:employer` 与 LME 的典型事实。
t18 改为一张显式三档本体表（定义在 `vcrm/state_index.py`）：

| 档位 | 含义 | 属性（语言归一化键） |
| --- | --- | --- |
| `EXCLUSIVE` | 同一时刻单值 | employer, occupation, position_held, spouse, head_of_government, head_of_state, country_of_citizenship, residence, place_of_birth, date_of_birth, alma_mater, religion, gender, capital, currency, official_language, leader, director, ceo, owner, parent_organization |
| `MULTI_VALUED` | 可并存 | languages_spoken, children, award_received, member_of_sports_team, cast_member, genre, field_of_work, notable_work, sibling, participant, composer, screenwriter, author |
| `UNKNOWN` | 兜底（不显式列属性） | 未在上面两档出现的一切属性 → 一律 DEFER |

约束（单测 `CardinalityOntologyTest` 钉死）：

- 键是**语言表面形式的归一化结果**（`normalize_attribute`），**严禁**由数据集
  relation ID 映射生成；两档交集为空，表内键互不重叠。
- `UNKNOWN` 不是显式集合：未列出的属性自然落档，代码里没有 `CARDINALITY_UNKNOWN`。
- 每次决策追加一条审计 `UpdateResolver.audit = [(attribute, bucket, action), ...]`，
  `cardinality_summary()` 返回 `{EXCLUSIVE: n1, MULTI_VALUED: n2, UNKNOWN: n3}`，
  总数与决策数一一对应，供评测**分口径**统计。
- `cardinality_policy='unknown_only'` 时三档计数退化为 `(0, 0, N)`。

### 10.2 可消融项：`cardinality_policy`

```python
UpdateResolver()                                   # 'ontology'（默认）：用上表
UpdateResolver(cardinality_policy='unknown_only')  # 严格基线：全部当 UNKNOWN → 全部 DEFER
```

**本体论假设声明**：把 `employer` 判成「同一时刻单值」是一条**假设**
（现实中可以兼职 / 多雇主），属于**可消融项**，消融方式就是 `unknown_only`；
论文表格必须同时给出两档数字。`official_language` 归入 `EXCLUSIVE` 同样是简化
（比利时、瑞士等有多官方语言），已列入风险清单。

### 10.3 两档在同一组数据上的决策序列（服务器 CPU 实测）

数据：`(Alice, employer)`，2018 `Google` → 2019 `Microsoft`，`recency_update=True`。

| 策略 | 2018 | 2019 | Current | `cardinality_summary()` |
| --- | --- | --- | --- | --- |
| `ontology`（默认） | `CREATE` 0.90 `rule1:no_candidate` | `UPDATE` 0.85 `exclusive_superseded_by_recency:employer` + `cardinality=EXCLUSIVE` | `fact_2019`（`observed_at=2019-01-01T00:00:00`），pending=0 | `{'EXCLUSIVE': 2, 'MULTI_VALUED': 0, 'UNKNOWN': 0}` |
| `unknown_only` | `CREATE` 0.90 `rule1:no_candidate` | `DEFER` 0.30 `rule6:cardinality_unknown` + `cardinality=UNKNOWN`（`pending_000001`） | `fact_2018`，pending=1 | `{'EXCLUSIVE': 0, 'MULTI_VALUED': 0, 'UNKNOWN': 2}` |

复现（服务器 CPU）：

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=.:probe env_py310/bin/python - <<'PY'
from vcrm.fact_schema import AtomicFact
from vcrm.state_index import StateIndex
from vcrm.state_router import retrieve_candidates
from vcrm.update_resolver import UpdateResolver, apply_decision
for policy in ('ontology', 'unknown_only'):
    index = StateIndex()
    resolver = UpdateResolver(recency_update=True, cardinality_policy=policy)
    for year, value in ((2018, 'Google'), (2019, 'Microsoft')):
        fact = AtomicFact(f'fact_{year}', f'In {year}: Alice works at {value}',
                          f'{year}-01-01T00:00:00', 'Alice', 'employer', value, 'src')
        decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
        apply_decision(decision, fact, index)
        print(policy, decision.action, round(decision.confidence, 2), decision.reason)
    print(policy, 'current =', index.records[0].current_version_id,
          'summary =', resolver.cardinality_summary())
PY
```

## 11. 唯一写入点与真实 store 的接线（t18 修复 F1/F2）

### 11.1 修复内容

| 缺陷 | 修复前实测现象 | 修复 |
| --- | --- | --- |
| F1 `apply_decision` / `commit_with` 不透传 `memory` | `StoreCommitError: current store commit failed for state_000001: commit() requires the original full memory= for this fact` | `apply_decision(..., memory=, semantic_key=, store_kwargs=)`；`commit_with(..., memory=, semantic_key=, decision=, store_kwargs=)` 把非 None 项透传给 `store.commit` |
| F2 CREATE 不写 store | `ValueError: UPDATE requires an existing current version`（修好 F1 后第二次写入仍被拒） | CREATE 也走 store：`StateIndex.build_record()` 造**不插入索引**的预备记录 → `store.commit()` → 成功后 `StateIndex.add(state_id=..., version_id=...)`；store 失败索引原样，索引失败 `store.rollback()` |
| F1′ 四种动作都要 `memory` | DUPLICATE / DEFER 缺失同样报错 | DUPLICATE 只写 store 的 alias 元数据（版本不动）；DEFER 写 store 的 `pending`（保留原始 full memory，Current 不动） |

`StateIndex` 为此新增：`build_record()`（构造不插入）、`add(state_id=...)`（预分配状态号）、
`discard()`（索引侧回滚）、`_store_extras()`（透传参数组装）。

### 11.2 真实 store + 真实 E3 codec 的集成证据（服务器 CPU）

`tests/test_update_resolver.py::RealStoreIntegrationTest` **不再使用 Fake**：
它优先加载冻结 E3 checkpoint
（`/share/latent2/vcrm_icae/runs/architecture_v6_E3_train/codec.safetensors`），
不可达时才退化为同构未训练 codec 并**显式打印来源标签**。

CREATE→UPDATE 全链路实测：

```
codec: checkpoint:/share/latent2/vcrm_icae/runs/architecture_v6_E3_train/codec.safetensors
decisions: [('CREATE', 0.9, 'rule1:no_candidate'),
            ('UPDATE', 0.85, 'exclusive_superseded_by_recency:employer|cardinality=EXCLUSIVE')]
index.current: fact_2019 | observed_at: 2019-01-01T00:00:00 | pending: 0
store versions: [('fact_2018', 'code', 1), ('fact_2019', 'full', 0)]
store.current: fact_2019 | blocks: fact_2019 | pending: 0 | duplicates: 0
```

即旧版本被 E3 codec 压缩成一次 16-slot code（`kind='code'`, `compression_count=1`），
新版本是唯一 full current，`blocks` / 索引指针 / `current_observed_at` 三者一致。

### 11.3 门禁复跑命令（服务器 CPU）

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=.:probe env_py310/bin/python -m pytest \
    tests/test_update_resolver.py tests/test_fact_extractor.py tests/test_state_router.py -q
# 78 passed
```
