from __future__ import annotations
# P2 回归测试：两阶段候选检索 + 路由。
# 关键反例：混合事实池（同主体不同属性 + 不同主体同属性）不得错误合并；
# 余弦相似度只能影响排序，不能影响 CREATE/UPDATE/DUPLICATE 判定。

import unittest

from vcrm.data_views import assert_no_oracle
from vcrm.fact_schema import AtomicFact
from vcrm.state_index import SEMANTIC_DIMENSIONS, StateIndex
from vcrm.state_router import (
    ROUTER_MODES,
    StateRouter,
    cosine_similarity,
    retrieve_candidates,
    route,
    stage1_filter,
    stage2_rank,
)
from vcrm.update_resolver import UpdateResolver

ORACLE_KEYS = ('state_key', 'key_text', 'entity', 'group', 'answers', 'latest',
               'episode', 'as_of', 'is_update', 'target_event')


def make_fact(subject, attribute, value, text='', observed_at='2019-01-01T00:00:00',
              fact_id='fact_000001', source_id='input_000001'):
    return AtomicFact(id=fact_id, text=text or f'The {attribute} of {subject} is {value}',
                      observed_at=observed_at, subject=subject, attribute=attribute,
                      value=value, source_id=source_id)


def unit_key(index, weights=None):
    """构造一个 128 维单位向量：只把第 index 维设为 1（测试用，不训练）。"""
    key = [0.0] * SEMANTIC_DIMENSIONS
    key[index % SEMANTIC_DIMENSIONS] = 1.0
    return tuple(key)


def mixed_pool_index():
    """混合事实池：同主体不同属性 + 不同主体同属性。"""
    index = StateIndex()
    france_capital = index.add_keys('France', 'capital', 'Paris', semantic_key=unit_key(0))
    index.add_keys('France', 'population', '67000000', semantic_key=unit_key(1))
    index.add_keys('Germany', 'capital', 'Berlin', semantic_key=unit_key(2))
    index.add_keys('Japan', 'capital', 'Tokyo', semantic_key=unit_key(3))
    return index, france_capital


class RetrieveCandidatesTest(unittest.TestCase):
    def setUp(self):
        self.index, self.france_capital = mixed_pool_index()
        self.fact = make_fact('France', 'capital', 'Paris')

    def test_stage1_keeps_same_subject_and_same_attribute(self):
        pool = stage1_filter(self.fact, self.index)
        identities = {r.identity_key() for r in pool}
        self.assertIn('france\u241fcapital', identities)
        self.assertIn('france\u241fpopulation', identities)   # 同主体不同属性
        self.assertIn('germany\u241fcapital', identities)      # 不同主体同属性
        self.assertIn('japan\u241fcapital', identities)
        self.assertNotIn('germany\u241fpopulation', identities)

    def test_right_state_ranked_first(self):
        candidates = retrieve_candidates(self.fact, self.index, top_k=5)
        self.assertEqual(candidates[0].state_id, self.france_capital.state_id)
        self.assertEqual(len(candidates), 4)

    def test_top_k_limits_candidates(self):
        self.assertEqual(len(retrieve_candidates(self.fact, self.index, top_k=1)), 1)
        self.assertEqual(len(retrieve_candidates(self.fact, self.index, top_k=2)), 2)

    def test_semantic_key_orders_but_does_not_decide(self):
        # 故意让「不同主体同属性」的语义键更接近：近似匹配内部按语义排序，但精确身份
        # (france, capital) 必须仍排最前（否则候选拥挤时会被 top_k 挤掉）
        fact_key = unit_key(2)  # 与 Germany/capital 的键完全一致
        candidates = retrieve_candidates(self.fact, self.index, top_k=5, semantic_key=fact_key)
        self.assertEqual(candidates[0].attribute_key, 'capital')
        self.assertEqual(candidates[0].subject_key, 'france')     # 精确身份优先
        self.assertEqual(candidates[1].subject_key, 'germany')    # 近似候选按语义排序
        # 但决策仍然只能落到 France/capital：主体不同的候选必须被拒绝合并
        resolver = UpdateResolver()
        decision = route(self.fact, self.index, resolver, top_k=5, semantic_key=fact_key)
        self.assertEqual(decision.action, 'DUPLICATE')
        self.assertEqual(decision.state_id, self.france_capital.state_id)

    def test_no_candidates_when_identity_absent(self):
        fact = make_fact('Kenya', 'currency', 'Kenyan shilling')
        self.assertEqual(retrieve_candidates(fact, self.index), [])

    def test_lexical_subject_fallback(self):
        # 主体表面形式不同、属性也不在索引里 → 只有词面主体回退能救回候选
        fact = make_fact('United States of America', 'population', '331000000')
        index = StateIndex()
        record = index.add_keys('United States', 'capital', 'Washington, D.C.')
        candidates = retrieve_candidates(fact, index, top_k=5)
        self.assertEqual([r.state_id for r in candidates], [record.state_id])
        strict = retrieve_candidates(fact, index, top_k=5,
                                     allow_lexical_subject_fallback=False)
        self.assertEqual(strict, [])

    def test_candidates_carry_no_oracle_fields(self):
        payload = [r.to_dict() for r in retrieve_candidates(self.fact, self.index)]
        assert_no_oracle(payload, 'candidates')
        for item in payload:
            for key in ORACLE_KEYS:
                self.assertNotIn(key, item)

    def test_retrieval_does_not_mutate_index(self):
        before = self.index.serialize()
        retrieve_candidates(self.fact, self.index, top_k=5)
        route(self.fact, self.index, UpdateResolver(), top_k=5)
        after = self.index.serialize()
        # route 只在 DEFER 时写 pending；这里命中 DUPLICATE，索引必须一字不变
        self.assertEqual(before, after)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            retrieve_candidates(self.fact, self.index, top_k=0)
        with self.assertRaises(TypeError):
            retrieve_candidates(self.fact, {'not': 'an index'})
        with self.assertRaises(ValueError):
            retrieve_candidates(self.fact, self.index, semantic_key=(1.0, 0.0))

    def test_cosine_helper(self):
        self.assertAlmostEqual(cosine_similarity(unit_key(0), unit_key(0)), 1.0)
        self.assertAlmostEqual(cosine_similarity(unit_key(0), unit_key(1)), 0.0)
        self.assertIsNone(cosine_similarity(None, unit_key(0)))


class CrowdedCandidateTest(unittest.TestCase):
    """top_k 候选拥挤：精确身份必须无条件进入候选且排最前。

    反例路径（可从代码直接推出）：纯语义排序把精确身份挤出 top_k → UpdateResolver 只看到
    近似候选 → 误判 CREATE → StateIndex.build_record 因身份重复抛错，写入失败。
    """

    def crowded(self):
        index = StateIndex()
        correct = index.add_keys('France', 'capital', 'Paris', semantic_key=unit_key(0))
        for i in range(6):                       # 同主体、不同属性、语义极相似的干扰状态
            index.add_keys('France', f'distractor_{i}', f'x{i}', semantic_key=unit_key(1))
        return index, correct

    def fact(self):
        return make_fact('France', 'capital', 'Lyon',
                         text="France's capital is now Lyon",
                         observed_at='2021-01-01T00:00:00', fact_id='fact_000099')

    def test_exact_identity_survives_crowded_top_k(self):
        index, correct = self.crowded()
        candidates = retrieve_candidates(self.fact(), index, top_k=5,
                                         semantic_key=unit_key(1))
        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates[0].state_id, correct.state_id)
        self.assertEqual(sum(1 for r in candidates if r.state_id == correct.state_id), 1)

    def test_semantic_only_truncation_would_lose_it(self):
        # 反例固化：去掉精确身份保障后，纯语义 top_k 里没有正确状态，写入必然报错
        index, correct = self.crowded()
        fact = self.fact()
        ranked = stage2_rank(fact, stage1_filter(fact, index), semantic_key=unit_key(1))
        truncated = [record for _, record in ranked[:5]]
        self.assertNotIn(correct.state_id, [r.state_id for r in truncated])
        self.assertEqual(UpdateResolver().resolve(fact, truncated, index).action, 'CREATE')
        with self.assertRaises(ValueError):
            index.build_record(fact)

    def test_update_succeeds_with_crowded_candidates(self):
        index, correct = self.crowded()
        fact = self.fact()
        decision = route(fact, index, UpdateResolver(), top_k=5, semantic_key=unit_key(1))
        self.assertEqual(decision.action, 'UPDATE')
        self.assertEqual(decision.state_id, correct.state_id)
        record = index.commit(decision.state_id, fact.id, observed_at=fact.observed_at)
        self.assertEqual(record.current_version_id, fact.id)
        self.assertEqual(record.current_observed_at, fact.observed_at)


class RouterClassTest(unittest.TestCase):
    def setUp(self):
        self.index, self.france_capital = mixed_pool_index()
        self.resolver = UpdateResolver()

    def test_router_class_matches_function_api(self):
        router = StateRouter(top_k=2)
        fact = make_fact('France', 'capital', 'Paris')
        self.assertEqual(
            [r.state_id for r in router.retrieve_candidates(fact, self.index)],
            [r.state_id for r in retrieve_candidates(fact, self.index, top_k=2)])
        decision = router.route(fact, self.index, self.resolver)
        self.assertEqual(decision.state_id, self.france_capital.state_id)

    def test_route_rejects_bad_resolver(self):
        with self.assertRaises(TypeError):
            route(make_fact('France', 'capital', 'Paris'), self.index, object())

    def test_route_returns_state_decision_type(self):
        decision = route(make_fact('France', 'capital', 'Lyon'), self.index, self.resolver)
        self.assertIn(decision.action, ('CREATE', 'UPDATE', 'DUPLICATE', 'DEFER'))


class SemanticRouterModeTest(unittest.TestCase):
    """t33 方案乙：router_mode='semantic' 只换候选生成口径，identity 默认零改动。

    现场问题（gpu_smoke2）：自由问句的 attribute 是组合式描述
    （'wait_time_for_asylum_decision'），写入侧事实的 attribute 是短名词短语
    （'wait_time'），identity 模式的 attribute 硬过滤把候选池清空 → 静默答空。
    """

    def setUp(self):
        self.index = StateIndex()
        self.index.add_keys('user', 'wait_time', '14 months', semantic_key=unit_key(0))
        self.index.add_keys('user', 'employer', 'Acme', semantic_key=unit_key(1))
        self.index.add_keys('1997 jeep wrangler', 'engine_type', '4.0L', semantic_key=unit_key(2))
        # GPU 现场形态：问句主体是代词 'I'（归一化 'i'），写入侧主体是 'user'；
        # attribute 是组合式描述，写入侧是短键 'wait_time'。
        self.query = make_fact('I', 'wait time for asylum decision', '')

    def test_semantic_mode_survives_disjoint_attribute(self):
        identity = retrieve_candidates(self.query, self.index, top_k=5)
        semantic = retrieve_candidates(self.query, self.index, top_k=5,
                                       router_mode='semantic', semantic_key=unit_key(0))
        self.assertEqual(identity, [])                 # 失败基线：主体对不上 + 属性不相交 → 池空
        self.assertTrue(semantic)                      # 语义模式经 user_alias 门拿到候选
        self.assertEqual(semantic[0].attribute_key, 'wait_time')   # 属性软排序把它排最前

    def test_semantic_mode_subject_gate_is_hard(self):
        stranger = make_fact('totally unrelated entity', 'wait time for asylum decision', '')
        semantic = retrieve_candidates(stranger, self.index, top_k=5,
                                       router_mode='semantic', semantic_key=unit_key(0))
        self.assertEqual(semantic, [])                 # 主体门失败 → 一条都不给

    def test_semantic_mode_user_alias_gate(self):
        """代词主体经 user_alias 归一化后仍能过门（与对齐层同一套归一化）。"""
        semantic = retrieve_candidates(self.query, self.index, top_k=5,
                                       router_mode='semantic', semantic_key=unit_key(0))
        self.assertTrue(semantic)
        self.assertEqual({record.subject_key for record in semantic}, {'user'})

    def test_semantic_mode_audit_records_gate_and_scores(self):
        router = StateRouter(top_k=5, router_mode='semantic')
        records = router.retrieve_candidates(self.query, self.index,
                                            semantic_key=unit_key(0))
        self.assertTrue(records)
        audit = router.last_retrieval_audit
        self.assertEqual(audit['mode'], 'semantic')
        self.assertTrue(audit['subject_gate']['passed'])
        self.assertEqual(audit['subject_gate']['matched_key'], 'user')
        self.assertEqual(audit['n_candidates'], len(records))
        self.assertLessEqual(len(audit['semantic_scores']), 5)
        for field in ('total', 'semantic', 'attribute'):
            self.assertIn(field, audit['semantic_scores'][0])

    def test_identity_mode_remains_default_and_unchanged(self):
        router = StateRouter(top_k=5)
        self.assertEqual(router.router_mode, 'identity')
        # identity 模式保留「同主体不同属性」候选（既有契约），代词主体则一条不给
        self.assertEqual(router.retrieve_candidates(self.query, self.index), [])
        self.assertTrue(router.retrieve_candidates(
            make_fact('user', 'wait time for asylum decision', ''), self.index))
        self.assertEqual(router.last_retrieval_audit, {})     # identity 不写审计
        self.assertEqual(ROUTER_MODES, ('identity', 'semantic'))
        with self.assertRaises(ValueError):
            StateRouter(router_mode='fuzzy')

    def test_semantic_mode_keeps_exact_identity_guarantee(self):
        """同主体同属性仍必须进候选（否则 resolver 会误判 CREATE 撞 build_record）。"""
        fact = make_fact('user', 'employer', '')
        router = StateRouter(top_k=1, router_mode='semantic')
        records = router.retrieve_candidates(fact, self.index, semantic_key=unit_key(1))
        self.assertIn('employer', [record.attribute_key for record in records])


if __name__ == '__main__':
    unittest.main()
