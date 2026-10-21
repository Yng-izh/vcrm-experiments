from __future__ import annotations
# P2 回归测试：UpdateResolver 六条规则 + 错误合并/错误拆分反例 + 混合事实池 +
# 与 IncrementalCurrentStore 的事务钩子。
# 全部使用哨兵字段证明任何路径都不读 state_key / key_text / episode / entity /
# group / answers / latest。

import functools
import os
import unittest

import torch

from vcrm.codec import SlotCodec
from vcrm.data_views import assert_no_oracle
from vcrm.fact_schema import AtomicFact, StateDecision
from vcrm.index import key_vector
from vcrm.state_index import (
    CARDINALITY_EXCLUSIVE,
    CARDINALITY_MULTI_VALUED,
    IndexCommitError,
    StateIndex,
    StateRecord,
    normalize_attribute,
    StoreCommitError,
    attribute_cardinality,
    value_fingerprint,
)
from vcrm.store import IncrementalCurrentStore
from vcrm.state_router import retrieve_candidates
from vcrm.update_resolver import UpdateResolver, apply_decision, resolve, values_equivalent

ORACLE_KEYS = ('state_key', 'key_text', 'entity', 'group', 'answers', 'latest',
               'episode', 'as_of', 'is_update', 'target_event')


def make_fact(subject, attribute, value, text='', observed_at='2019-01-01T00:00:00',
              fact_id='fact_000001', source_id='input_000001'):
    return AtomicFact(id=fact_id, text=text or f'The {attribute} of {subject} is {value}',
                      observed_at=observed_at, subject=subject, attribute=attribute,
                      value=value, source_id=source_id)


class FakeCurrentStore:
    """最小 IncrementalCurrentStore 替身：只实现事务钩子需要的 commit/rollback。"""

    def __init__(self, result='version_000001', fail=False):
        self.result = result
        self.fail = fail
        self.commits = []
        self.rollbacks = []

    def commit(self, record=None, fact=None, **kwargs):
        if self.fail:
            raise RuntimeError('current store unavailable')
        self.commits.append({'state_id': record.state_id, 'fact_id': getattr(fact, 'id', None),
                             'memory': kwargs.get('memory'),
                             'semantic_key': kwargs.get('semantic_key'),
                             'decision': getattr(kwargs.get('decision'), 'action', None)})
        return self.result

    def rollback(self, result):
        self.rollbacks.append(result)


class SixRulesTest(unittest.TestCase):
    def setUp(self):
        self.index = StateIndex()
        self.resolver = UpdateResolver()

    def seed(self, subject, attribute, value, **kwargs):
        return self.index.add_keys(subject, attribute, value, **kwargs)

    def decide(self, fact):
        return self.resolver.resolve(fact, retrieve_candidates(fact, self.index), self.index)

    def test_rule1_no_candidate_creates(self):
        decision = self.decide(make_fact('France', 'capital', 'Paris'))
        self.assertEqual(decision.action, 'CREATE')
        self.assertIsNone(decision.state_id)
        self.assertIn('rule1', decision.reason)

    def test_rule2_subject_mismatch_refuses_merge(self):
        self.seed('Germany', 'capital', 'Berlin')
        fact = make_fact('France', 'capital', 'Berlin')  # 同属性同值，主体不同
        decision = self.decide(fact)
        self.assertEqual(decision.action, 'CREATE')
        self.assertIn('rule2', decision.reason)

    def test_rule3_same_subject_different_attribute_creates(self):
        self.seed('France', 'capital', 'Paris')
        decision = self.decide(make_fact('France', 'population', '67000000'))
        self.assertEqual(decision.action, 'CREATE')
        self.assertIn('rule3', decision.reason)

    def test_rule4_value_equivalent_is_duplicate(self):
        record = self.seed('France', 'capital', 'Paris')
        for value in ('Paris', 'paris', 'The Paris', 'Paris.'):
            decision = self.decide(make_fact('France', 'capital', value))
            self.assertEqual(decision.action, 'DUPLICATE', value)
            self.assertEqual(decision.state_id, record.state_id)
            self.assertIn('rule4', decision.reason)

    def test_rule5_exclusive_attribute_with_explicit_replacement_updates(self):
        record = self.seed('France', 'head_of_government', 'Manuel Valls')
        fact = make_fact('France', 'head_of_government', 'Bernard Cazeneuve',
                         text='In 2016: Bernard Cazeneuve replaced Manuel Valls as head of government')
        decision = self.decide(fact)
        self.assertEqual(decision.action, 'UPDATE')
        self.assertEqual(decision.state_id, record.state_id)
        self.assertIn('explicit_replacement', decision.reason)
        self.assertIn('cardinality=EXCLUSIVE', decision.reason)

    def test_rule6_multi_valued_attribute_defers(self):
        self.seed('France', 'award_received', 'Legion of Honour')
        fact = make_fact('France', 'award_received', 'Nobel Peace Prize',
                         text='France now lists the Nobel Peace Prize as an award')
        decision = self.decide(fact)
        self.assertEqual(decision.action, 'DEFER')
        self.assertIn('rule6', decision.reason)
        self.assertIn('multi_valued', decision.reason)

    def test_rule6_negation_defers(self):
        self.seed('France', 'capital', 'Paris')
        fact = make_fact('France', 'capital', 'Paris',
                         text='Paris is no longer the capital of France')
        decision = self.decide(fact)
        self.assertEqual(decision.action, 'DEFER')
        self.assertIn('negation', decision.reason)

    def test_rule6_conflict_defers(self):
        self.seed('France', 'capital', 'Paris')
        payload = self.index.serialize()
        duplicate = dict(payload['records'][0])
        duplicate['state_id'] = 'state_999999'
        duplicate['value_fingerprint'] = value_fingerprint('Lyon')
        payload['records'].append(duplicate)
        conflicted = StateIndex.load(payload)
        fact = make_fact('France', 'capital', 'Marseille', text='Paris is now Marseille?')
        decision = self.resolver.resolve(
            fact, retrieve_candidates(fact, conflicted), conflicted)
        self.assertEqual(decision.action, 'DEFER')
        self.assertIn('conflicting_candidates', decision.reason)

    def test_rule6_unknown_exclusivity_defers(self):
        self.seed('France', 'motto', 'Liberte')
        fact = make_fact('France', 'motto', 'Egalite', text='France now uses Egalite as motto')
        decision = self.decide(fact)
        self.assertEqual(decision.action, 'DEFER')
        self.assertIn('cardinality_unknown', decision.reason)
        self.assertIn('cardinality=UNKNOWN', decision.reason)

    def test_rule6_no_explicit_replacement_defers(self):
        self.seed('France', 'head_of_government', 'Manuel Valls')
        fact = make_fact('France', 'head_of_government', 'Bernard Cazeneuve',
                         text='In 2016: The head of government of France is Bernard Cazeneuve')
        decision = self.decide(fact)
        self.assertEqual(decision.action, 'DEFER')
        self.assertIn('no_explicit_replacement', decision.reason)

    def test_rule6_unresolved_fact_defers(self):
        fact = make_fact('<unresolved:subject>', '<unresolved:attribute>',
                         'Lorem ipsum dolor sit amet')
        decision = self.decide(fact)
        self.assertEqual(decision.action, 'DEFER')
        self.assertIn('unresolved', decision.reason)

    def test_defer_does_not_touch_current(self):
        record = self.seed('France', 'capital', 'Paris')
        before = (record.current_version_id, record.value_fingerprint, record.updated_at)
        self.decide(make_fact('France', 'capital', 'Lyon', text='Lyon is no longer ...'))
        after = (record.current_version_id, record.value_fingerprint, record.updated_at)
        self.assertEqual(before, after)
        self.assertEqual(self.index.pending_count, 1)
        pending = self.index.pending[0]
        self.assertEqual(pending.candidate_state_id, record.state_id)
        self.assertEqual(pending.attribute_key, 'capital')


class CounterExampleTest(unittest.TestCase):
    """错误合并 / 错误拆分反例：混合事实池里最容易出错的两种形态。"""

    def setUp(self):
        self.index = StateIndex()
        self.resolver = UpdateResolver()
        self.france_capital = self.index.add_keys('France', 'capital', 'Paris')
        self.france_population = self.index.add_keys('France', 'population', '67000000')
        self.germany_capital = self.index.add_keys('Germany', 'capital', 'Berlin')

    def test_no_wrong_merge_across_subjects_with_same_value(self):
        # 不同主体、同属性、同值：绝不能判成 DUPLICATE
        fact = make_fact('Japan', 'capital', 'Berlin')
        decision = self.resolver.resolve(fact, retrieve_candidates(fact, self.index), self.index)
        self.assertEqual(decision.action, 'CREATE')
        self.assertIsNone(decision.state_id)

    def test_no_wrong_merge_across_attributes(self):
        # 同主体、不同属性：绝不能并到已有的 capital / population 状态里
        fact = make_fact('France', 'area', '551695', text='France now has an area of 551695 km2')
        decision = self.resolver.resolve(fact, retrieve_candidates(fact, self.index), self.index)
        self.assertEqual(decision.action, 'CREATE')
        self.assertIsNone(decision.state_id)

    def test_no_wrong_split_for_repeated_fact(self):
        # 同主体、同属性、同值：必须 DUPLICATE，不能错误新建
        fact = make_fact('France', 'capital', 'Paris')
        decision = self.resolver.resolve(fact, retrieve_candidates(fact, self.index), self.index)
        self.assertEqual(decision.action, 'DUPLICATE')
        self.assertEqual(decision.state_id, self.france_capital.state_id)
        self.assertEqual(len(self.index), 3)

    def test_mixed_pool_routes_to_right_state(self):
        fact = make_fact('France', 'capital', 'Paris')
        candidates = retrieve_candidates(fact, self.index)
        self.assertIn(self.france_population.state_id, [c.state_id for c in candidates])
        self.assertIn(self.germany_capital.state_id, [c.state_id for c in candidates])
        decision = self.resolver.resolve(fact, candidates, self.index)
        self.assertEqual(decision.state_id, self.france_capital.state_id)

    def test_values_equivalent_helper(self):
        self.assertTrue(values_equivalent('1,234,567', '1234567'))
        self.assertTrue(values_equivalent('The Paris', 'paris'))
        self.assertFalse(values_equivalent('Paris', 'Lyon'))
        self.assertFalse(values_equivalent('', 'Lyon'))


class ApplyDecisionTransactionTest(unittest.TestCase):
    def setUp(self):
        self.index = StateIndex()
        self.resolver = UpdateResolver()
        self.record = self.index.add_keys('France', 'head_of_government', 'Manuel Valls',
                                          version_id='version_000001')

    def update_fact(self):
        return make_fact('France', 'head_of_government', 'Bernard Cazeneuve',
                         text='Bernard Cazeneuve succeeded Manuel Valls as head of government',
                         fact_id='fact_000002')

    def test_create_writes_state_without_store(self):
        fact = make_fact('France', 'capital', 'Paris', fact_id='fact_000003')
        decision = StateDecision('CREATE', None, 0.9, 'rule1:no_candidate')
        record = apply_decision(decision, fact, self.index)
        self.assertEqual(record.subject_key, 'france')
        self.assertEqual(record.current_version_id, 'fact_000003')

    def test_update_commits_store_then_index(self):
        store = FakeCurrentStore(result='version_000002')
        decision = StateDecision('UPDATE', self.record.state_id, 0.8, 'rule5')
        apply_decision(decision, self.update_fact(), self.index, store)
        self.assertEqual(self.record.current_version_id, 'version_000002')
        self.assertEqual(len(store.commits), 1)
        self.assertEqual(store.commits[0]['state_id'], self.record.state_id)

    def test_store_failure_leaves_index_untouched(self):
        store = FakeCurrentStore(fail=True)
        before = (self.record.current_version_id, self.record.value_fingerprint)
        with self.assertRaises(StoreCommitError):
            self.index.commit_with(store, self.record.state_id, self.update_fact())
        self.assertEqual((self.record.current_version_id, self.record.value_fingerprint), before)

    def test_index_failure_rolls_back_store(self):
        store = FakeCurrentStore(result=None)  # store 成功但没给出合法版本号
        before = (self.record.current_version_id, self.record.value_fingerprint)
        with self.assertRaises(IndexCommitError):
            self.index.commit_with(store, self.record.state_id, self.update_fact())
        self.assertEqual((self.record.current_version_id, self.record.value_fingerprint), before)
        self.assertEqual(store.rollbacks, [None])

    def test_duplicate_does_not_commit(self):
        store = FakeCurrentStore()
        fact = make_fact('France', 'head_of_government', 'Manuel Valls')
        decision = self.resolver.resolve(fact, retrieve_candidates(fact, self.index), self.index)
        self.assertEqual(decision.action, 'DUPLICATE')
        apply_decision(decision, fact, self.index, store)
        # DUPLICATE 只写 alias 元数据（decision.action=DUPLICATE），不推进版本
        self.assertEqual([item['decision'] for item in store.commits], ['DUPLICATE'])
        self.assertEqual(self.record.current_version_id, 'version_000001')

    def test_unknown_state_id_raises(self):
        with self.assertRaises(KeyError):
            self.index.commit_with(FakeCurrentStore(), 'state_missing', self.update_fact())


class NoOracleTest(unittest.TestCase):
    def test_resolver_and_index_never_touch_oracle_fields(self):
        index = StateIndex()
        resolver = UpdateResolver()
        index.add_keys('France', 'capital', 'Paris')
        # 事实对象里塞满 Oracle 哨兵键（模拟上游误传）：决策结果不得依赖它们
        fact = AtomicFact(id='fact_000001', text='The capital of France is Paris',
                          observed_at='2019-01-01T00:00:00', subject='France',
                          attribute='capital', value='Paris', source_id='input_000001')
        payload = fact.__dict__
        self.assertFalse(any(key in payload for key in ORACLE_KEYS))
        decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
        self.assertEqual(decision.action, 'DUPLICATE')
        assert_no_oracle(index.serialize(), 'state_index')

    def test_pending_records_carry_no_oracle_fields(self):
        index = StateIndex()
        resolver = UpdateResolver()
        index.add_keys('France', 'capital', 'Paris')
        fact = make_fact('France', 'capital', 'Lyon', text='Paris is no longer the capital')
        resolver.resolve(fact, retrieve_candidates(fact, index), index)
        assert_no_oracle(index.serialize(), 'state_index')
        for pending in index.pending:
            for key in ORACLE_KEYS:
                self.assertNotIn(key, pending.to_dict())

    def test_module_level_resolve_helper(self):
        index = StateIndex()
        fact = make_fact('France', 'capital', 'Paris')
        decision = resolve(fact, [], index)
        self.assertEqual(decision.action, 'CREATE')


class RecencyUpdatePolicyTest(unittest.TestCase):
    """rule5b（recency_update 显式策略开关，默认关闭）的回归与开关行为。

    场景：TempLAMA 的逐年陈述句（"In <year>: The head of government of France is X"）
    不含 now/replaced 一类替代表达，因此在默认策略下永远不构成 UPDATE 证据。
    """

    YEARS = (('2016', 'Bernard Cazeneuve'), ('2018', 'Edouard Philippe'),
             ('2020', 'Jean Castex'))

    def run_chain(self, resolver, attribute='head_of_government', subject='France',
                  years=None):
        index = StateIndex()
        decisions = []
        for year, value in (years or self.YEARS):
            fact = make_fact(subject, attribute, value,
                             text=f'In {year}: The {attribute.replace("_", " ")} of '
                                  f'{subject} is {value}',
                             observed_at=f'{year}-01-01T00:00:00',
                             fact_id=f'fact_{year}')
            decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
            apply_decision(decision, fact, index)
            decisions.append(decision)
        return index, decisions

    def test_default_off_keeps_yearly_statements_deferred(self):
        index, decisions = self.run_chain(UpdateResolver())
        self.assertEqual([d.action for d in decisions], ['CREATE', 'DEFER', 'DEFER'])
        self.assertIn('rule6:no_explicit_replacement', decisions[1].reason)
        # 版本链没有建立：Current 仍是 2016 的第一个值
        record = index.records[0]
        self.assertEqual(record.current_version_id, 'fact_2016')
        self.assertEqual(index.pending_count, 2)

    def test_switch_on_builds_version_chain(self):
        index, decisions = self.run_chain(UpdateResolver(recency_update=True))
        self.assertEqual([d.action for d in decisions], ['CREATE', 'UPDATE', 'UPDATE'])
        for decision in decisions[1:]:
            self.assertTrue(decision.reason.startswith('exclusive_superseded_by_recency:'),
                            decision.reason)
        self.assertIn('exclusive_superseded_by_recency:head_of_government', decisions[1].reason)
        record = index.records[0]
        self.assertEqual(record.current_version_id, 'fact_2020')
        self.assertEqual(record.current_observed_at, '2020-01-01T00:00:00')
        self.assertEqual(record.value_fingerprint, value_fingerprint('Jean Castex'))
        self.assertEqual(index.pending_count, 0)
        # 每次 UPDATE 的置信度由时间间隔决定，且在 [0, 1] 内可校准
        for decision in decisions[1:]:
            self.assertGreaterEqual(decision.confidence, 0.5)
            self.assertLessEqual(decision.confidence, 0.85)

    def test_switch_on_still_prefers_explicit_replacement_wording(self):
        resolver = UpdateResolver(recency_update=True)
        index = StateIndex()
        first = make_fact('France', 'head_of_government', 'Manuel Valls',
                          text='In 2015: The head of government of France is Manuel Valls',
                          observed_at='2015-01-01T00:00:00', fact_id='fact_2015')
        apply_decision(resolver.resolve(first, retrieve_candidates(first, index), index),
                       first, index)
        second = make_fact('France', 'head_of_government', 'Bernard Cazeneuve',
                           text='Bernard Cazeneuve replaced Manuel Valls as head of government',
                           observed_at='2016-01-01T00:00:00', fact_id='fact_2016')
        decision = resolver.resolve(second, retrieve_candidates(second, index), index)
        self.assertEqual(decision.action, 'UPDATE')
        self.assertTrue(decision.reason.startswith('explicit_replacement:'), decision.reason)

    def test_multi_valued_attribute_never_updates_by_recency(self):
        resolver = UpdateResolver(recency_update=True)
        index = StateIndex()
        first = make_fact('France', 'languages_spoken', 'French',
                          text='In 2016: The languages spoken in France are French',
                          observed_at='2016-01-01T00:00:00', fact_id='fact_2016')
        apply_decision(resolver.resolve(first, retrieve_candidates(first, index), index),
                       first, index)
        second = make_fact('France', 'languages_spoken', 'German',
                           text='In 2020: The languages spoken in France are German',
                           observed_at='2020-01-01T00:00:00', fact_id='fact_2020')
        decision = resolver.resolve(second, retrieve_candidates(second, index), index)
        self.assertEqual(decision.action, 'DEFER')
        self.assertIn('multi_valued', decision.reason)
        self.assertEqual(index.records[0].current_version_id, 'fact_2016')

    def test_equal_observed_at_still_defers(self):
        resolver = UpdateResolver(recency_update=True)
        index = StateIndex()
        record = index.add_keys('France', 'head_of_government', 'Manuel Valls',
                                version_id='fact_2018', observed_at='2018-01-01T00:00:00')
        fact = make_fact('France', 'head_of_government', 'Edouard Philippe',
                         text='In 2018: The head of government of France is Edouard Philippe',
                         observed_at='2018-01-01T00:00:00', fact_id='fact_2018b')
        decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
        self.assertEqual(decision.action, 'DEFER')
        self.assertIn('recency_not_applicable', decision.reason)
        self.assertEqual(record.current_version_id, 'fact_2018')

    def test_unparsable_observed_at_still_defers(self):
        resolver = UpdateResolver(recency_update=True)
        index = StateIndex()
        record = index.add_keys('France', 'head_of_government', 'Manuel Valls',
                                version_id='fact_unknown', observed_at='unknown')
        fact = make_fact('France', 'head_of_government', 'Edouard Philippe',
                         text='In 2020: The head of government of France is Edouard Philippe',
                         observed_at='2020-01-01T00:00:00', fact_id='fact_2020')
        decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
        self.assertEqual(decision.action, 'DEFER')
        self.assertIn('recency_not_applicable', decision.reason)
        self.assertEqual(record.current_version_id, 'fact_unknown')

    def test_earlier_observed_at_never_overwrites(self):
        resolver = UpdateResolver(recency_update=True)
        index = StateIndex()
        record = index.add_keys('France', 'head_of_government', 'Jean Castex',
                                version_id='fact_2020', observed_at='2020-01-01T00:00:00')
        fact = make_fact('France', 'head_of_government', 'Edouard Philippe',
                         text='In 2018: The head of government of France is Edouard Philippe',
                         observed_at='2018-01-01T00:00:00', fact_id='fact_2018')
        decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
        self.assertEqual(decision.action, 'DEFER')
        self.assertEqual(record.current_version_id, 'fact_2020')

    def test_confidence_gate_can_be_calibrated(self):
        # 门槛高于可产生的置信度时，策略退化为 DEFER（门槛必须在验证集校准）
        resolver = UpdateResolver(recency_update=True, recency_min_confidence=0.99)
        index = StateIndex()
        index.add_keys('France', 'head_of_government', 'Manuel Valls',
                       version_id='fact_2016', observed_at='2016-01-01T00:00:00')
        fact = make_fact('France', 'head_of_government', 'Edouard Philippe',
                         text='In 2018: The head of government of France is Edouard Philippe',
                         observed_at='2018-01-01T00:00:00', fact_id='fact_2018')
        decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
        self.assertEqual(decision.action, 'DEFER')
        self.assertIn('recency_not_applicable', decision.reason)

    def test_duplicate_still_wins_over_recency(self):
        resolver = UpdateResolver(recency_update=True)
        index = StateIndex()
        record = index.add_keys('France', 'head_of_government', 'Manuel Valls',
                                version_id='fact_2016', observed_at='2016-01-01T00:00:00')
        fact = make_fact('France', 'head_of_government', 'Manuel Valls',
                         text='In 2020: The head of government of France is Manuel Valls',
                         observed_at='2020-01-01T00:00:00', fact_id='fact_2020')
        decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
        self.assertEqual(decision.action, 'DUPLICATE')
        self.assertEqual(record.current_version_id, 'fact_2016')


# ---------------------------------------------------------------------------
# F3：属性基数本体表（三档）+ cardinality_policy 消融开关
# ---------------------------------------------------------------------------
class CardinalityOntologyTest(unittest.TestCase):
    def test_ontology_table_is_language_derived(self):
        for attribute in ('employer', 'occupation', 'spouse', 'alma_mater',
                          'country_of_citizenship', 'head_of_government'):
            self.assertEqual(attribute_cardinality(attribute, 'ontology'), 'EXCLUSIVE', attribute)
        for attribute in ('languages_spoken', 'children', 'award_received', 'genre'):
            self.assertEqual(attribute_cardinality(attribute, 'ontology'), 'MULTI_VALUED', attribute)
        self.assertEqual(attribute_cardinality('motto', 'ontology'), 'UNKNOWN')
        # 严格基线：全部当 UNKNOWN
        self.assertEqual(attribute_cardinality('employer', 'unknown_only'), 'UNKNOWN')
        with self.assertRaises(ValueError):
            attribute_cardinality('employer', 'nonsense')
        # 本体表键是语言表面形式，不含 relation id
        self.assertFalse(any(':' in key or key.startswith('p') and key[1:].isdigit()
                             for key in CARDINALITY_EXCLUSIVE | CARDINALITY_MULTI_VALUED))

    def test_employer_updates_under_ontology(self):
        resolver = UpdateResolver(recency_update=True)      # 默认 cardinality_policy='ontology'
        index = StateIndex()
        chain = (('2018', 'Google'), ('2019', 'Microsoft'))
        actions = []
        for year, value in chain:
            fact = make_fact('Alice', 'employer', value,
                             text=f'In {year}: Alice works at {value}',
                             observed_at=f'{year}-01-01T00:00:00', fact_id=f'fact_{year}')
            decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
            apply_decision(decision, fact, index)
            actions.append(decision)
        self.assertEqual([d.action for d in actions], ['CREATE', 'UPDATE'])
        self.assertIn('exclusive_superseded_by_recency:employer', actions[1].reason)
        self.assertIn('cardinality=EXCLUSIVE', actions[1].reason)
        self.assertEqual(index.records[0].current_version_id, 'fact_2019')
        self.assertEqual(resolver.cardinality_summary()['EXCLUSIVE'], 2)

    def test_unknown_only_policy_defers_everything(self):
        resolver = UpdateResolver(recency_update=True, cardinality_policy='unknown_only')
        index = StateIndex()
        chain = (('2018', 'Google'), ('2019', 'Microsoft'))
        actions = []
        for year, value in chain:
            fact = make_fact('Alice', 'employer', value,
                             text=f'In {year}: Alice works at {value}',
                             observed_at=f'{year}-01-01T00:00:00', fact_id=f'fact_{year}')
            decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
            apply_decision(decision, fact, index)
            actions.append(decision)
        self.assertEqual([d.action for d in actions], ['CREATE', 'DEFER'])
        self.assertIn('cardinality_unknown', actions[1].reason)
        self.assertIn('cardinality=UNKNOWN', actions[1].reason)
        self.assertEqual(index.records[0].current_version_id, 'fact_2018')
        # 逐样本审计：CREATE 与 DEFER 都记 UNKNOWN 档
        self.assertEqual(resolver.cardinality_summary()['UNKNOWN'], 2)

    def test_unknown_only_still_allows_explicit_replacement_text(self):
        # 严格基线下 rule5 也不可达（基数未知先拦），UPDATE 只能来自 pending 复核
        resolver = UpdateResolver(cardinality_policy='unknown_only')
        index = StateIndex()
        index.add_keys('Alice', 'employer', 'Google', version_id='fact_2018',
                       observed_at='2018-01-01T00:00:00')
        fact = make_fact('Alice', 'employer', 'Microsoft',
                         text="Microsoft replaced Google as Alice's employer",
                         observed_at='2019-01-01T00:00:00', fact_id='fact_2019')
        decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
        self.assertEqual(decision.action, 'DEFER')

    def test_buckets_are_disjoint_and_language_derived(self):
        """t9 F3 的根因就是同一属性跨两档：这里钉死互斥 + 归一化。"""
        self.assertTrue(CARDINALITY_EXCLUSIVE.isdisjoint(CARDINALITY_MULTI_VALUED))
        for key in CARDINALITY_EXCLUSIVE | CARDINALITY_MULTI_VALUED:
            self.assertEqual(normalize_attribute(key), key, key)     # 已是归一化键
            self.assertNotIn(':', key)                               # 不是 relation id
            self.assertNotRegex(key, r'^[pq]\d+$')

    def test_no_explicit_unknown_bucket(self):
        """UNKNOWN 是兜底档，不显式列属性；未列出的属性一律落 UNKNOWN。"""
        from vcrm import state_index
        self.assertFalse(hasattr(state_index, 'CARDINALITY_UNKNOWN'))
        self.assertEqual(attribute_cardinality('favourite_colour_of_the_king', 'ontology'), 'UNKNOWN')

    def test_unknown_only_degenerates_to_single_bucket(self):
        resolver = UpdateResolver(recency_update=True, cardinality_policy='unknown_only')
        index = StateIndex()
        facts = [
            make_fact('Alice', 'employer', 'Google', text='In 2018: Alice works at Google',
                      observed_at='2018-01-01T00:00:00', fact_id='fact_2018'),
            make_fact('Alice', 'employer', 'Microsoft', text='In 2019: Alice works at Microsoft',
                      observed_at='2019-01-01T00:00:00', fact_id='fact_2019'),
            make_fact('Alice', 'languages_spoken', 'French',
                      text='In 2020: Alice speaks French',
                      observed_at='2020-01-01T00:00:00', fact_id='fact_2020'),
        ]
        for fact in facts:
            apply_decision(resolver.resolve(fact, retrieve_candidates(fact, index), index), fact, index)
        summary = resolver.cardinality_summary()
        self.assertEqual(summary, {'EXCLUSIVE': 0, 'MULTI_VALUED': 0, 'UNKNOWN': len(facts)})

    def test_audit_counts_match_decisions_one_to_one(self):
        resolver = UpdateResolver(recency_update=True)
        index = StateIndex()
        facts = [
            make_fact('Alice', 'employer', 'Google', text='In 2018: Alice works at Google',
                      observed_at='2018-01-01T00:00:00', fact_id='fact_2018'),
            make_fact('Alice', 'employer', 'Microsoft', text='In 2019: Alice works at Microsoft',
                      observed_at='2019-01-01T00:00:00', fact_id='fact_2019'),
            make_fact('Alice', 'languages_spoken', 'French', text='In 2020: Alice speaks French',
                      observed_at='2020-01-01T00:00:00', fact_id='fact_2020'),
            make_fact('Alice', 'languages_spoken', 'German', text='In 2021: Alice speaks German',
                      observed_at='2021-01-01T00:00:00', fact_id='fact_2021'),
        ]
        decisions = []
        for fact in facts:
            decision = resolver.resolve(fact, retrieve_candidates(fact, index), index)
            decisions.append(decision)
            apply_decision(decision, fact, index)
        self.assertEqual(len(resolver.audit), len(decisions))          # 一决策一条审计
        summary = resolver.cardinality_summary()
        self.assertEqual(sum(summary.values()), len(decisions))
        self.assertEqual(summary['EXCLUSIVE'], 2)                      # employer 两条
        self.assertEqual(summary['MULTI_VALUED'], 2)                   # languages_spoken 两条
        self.assertEqual(summary['UNKNOWN'], 0)
        for item, decision in zip(resolver.audit, decisions):
            self.assertEqual(item['action'], decision.action)
            self.assertEqual(item['bucket'], attribute_cardinality(item['attribute'], 'ontology'))

    def test_invalid_policy_rejected(self):
        with self.assertRaises(ValueError):
            UpdateResolver(cardinality_policy='whatever')


# ---------------------------------------------------------------------------
# F1/F2：真实 IncrementalCurrentStore + 真实 SlotCodec（CPU，E3 同构配置）
# ---------------------------------------------------------------------------
DIMENSION, MEMORY_SLOTS, KEY_DIMENSIONS = 4096, 128, 128
CODEC_SLOTS, CODEC_WIDTH, CODEC_HEADS, RMS_CAP = 16, 256, 4, 8.0


@functools.lru_cache(maxsize=None)
def real_e3_codec():
    """优先加载服务器上的冻结 E3 checkpoint；不可达时退化为同构本地 codec（显式标注）。

    返回 (codec, label)。label 会打印出来，绝不把未训练 codec 冒充真实权重结果。
    """
    try:
        import importlib.util
        import json
        import pathlib
        from safetensors.torch import load_file
        root = pathlib.Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('vcrm_probe_v6_loader',
                                                      root / 'probe' / 'v6_loader.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _, training_root, _ = module.load_v6('E3')
        metadata = json.loads((training_root / 'codec_training.json').read_text())
        state = load_file(str(training_root / 'codec.safetensors'))
        codec_state = {key[len('codec.'):]: value for key, value in state.items()
                       if key.startswith('codec.')}
        dimension = int(codec_state['queries'].shape[1])
        codec = SlotCodec(dimension, metadata['history_slots'], metadata['codec_width'],
                          metadata['codec_heads'], metadata['method'] == 'D',
                          metadata.get('anchor_dropout', 0.), metadata.get('seed', 42),
                          metadata.get('codec_variant', 'residual'),
                          metadata.get('decode_conditioned'),
                          metadata.get('codec_value_path'),
                          metadata.get('codec_activation_rms_cap'))
        codec.load_state_dict(codec_state, strict=True)
        if codec.conditioned:
            raise ValueError('E3 checkpoint is conditioned; incremental store needs independent codec')
        return codec.eval().requires_grad_(False), f'checkpoint:{training_root}/codec.safetensors'
    except Exception as error:  # 无挂载/无 safetensors/权重不匹配
        torch.manual_seed(42)
        codec = SlotCodec(DIMENSION, CODEC_SLOTS, CODEC_WIDTH, CODEC_HEADS, False, 0., 42,
                          'residual', None, 'full', RMS_CAP).eval().requires_grad_(False)
        return codec, (f'local_e3_config(dimension={DIMENSION},slots={CODEC_SLOTS},'
                       f'value_path=full,rms_cap={RMS_CAP}; UNTRAINED) '
                       f'fallback={type(error).__name__}: {error}')


def real_store():
    """真实 IncrementalCurrentStore + 真实 SlotCodec（优先冻结 E3 checkpoint）。"""
    codec, label = real_e3_codec()
    store = IncrementalCurrentStore(codec, key_dimensions=KEY_DIMENSIONS)
    store.codec_label = label
    return store


def full_memory(seed):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(MEMORY_SLOTS, DIMENSION, generator=generator) * .5).half().float()


def semantic_key(text):
    return key_vector(text, KEY_DIMENSIONS)


class RealStoreIntegrationTest(unittest.TestCase):
    """禁止用 Fake 覆盖这条路径：真实 store 强制 memory=/semantic_key=。"""

    def setUp(self):
        self.store = real_store()
        if os.environ.get('VCRM_T18_VERBOSE'):
            print(f'[t18] codec source: {self.store.codec_label}')
        self.index = StateIndex()
        self.resolver = UpdateResolver(recency_update=True)
        self.memories = {'2018': full_memory(1001), '2019': full_memory(1002)}
        self.keys = {'2018': semantic_key('Google'), '2019': semantic_key('Microsoft')}

    def ingest(self, year, value):
        fact = make_fact('Alice', 'employer', value,
                         text=f'In {year}: Alice works at {value}',
                         observed_at=f'{year}-01-01T00:00:00', fact_id=f'fact_{year}')
        decision = self.resolver.resolve(fact, retrieve_candidates(fact, self.index), self.index)
        record = apply_decision(decision, fact, self.index, self.store,
                                memory=self.memories[year], semantic_key=self.keys[year])
        return decision, record

    def test_create_then_update_version_chain(self):
        first, record = self.ingest('2018', 'Google')
        self.assertEqual(first.action, 'CREATE')
        state_id = record.state_id
        self.assertEqual(self.store.current(state_id)['id'], 'fact_2018')
        self.assertEqual(record.current_version_id, 'fact_2018')

        second, record = self.ingest('2019', 'Microsoft')
        self.assertEqual(second.action, 'UPDATE')
        self.assertEqual(record.current_version_id, 'fact_2019')
        self.assertEqual(record.current_observed_at, '2019-01-01T00:00:00')

        # 版本链：旧版本被压缩成 16-slot code，新版本是唯一 full current
        versions = [entry for entry in self.store.entries if entry['state'] == state_id]
        self.assertEqual([entry['id'] for entry in versions], ['fact_2018', 'fact_2019'])
        historical, current = versions
        self.assertEqual(historical['kind'], 'code')
        self.assertEqual(historical['compression_count'], 1)
        self.assertEqual(current['kind'], 'full')
        self.assertTrue(current['keyframe'])
        self.assertEqual(self.store.current(state_id)['id'], 'fact_2019')
        self.assertEqual(self.store.blocks[state_id], 'fact_2019')
        # 索引与 store 的 current 指针一致
        self.assertEqual(self.index.records[0].current_version_id, 'fact_2019')
        self.assertEqual(self.index.pending_count, 0)

    def test_missing_memory_is_a_store_commit_error(self):
        fact = make_fact('Alice', 'employer', 'Google',
                         text='In 2018: Alice works at Google',
                         observed_at='2018-01-01T00:00:00', fact_id='fact_2018')
        decision = self.resolver.resolve(fact, [], self.index)
        self.assertEqual(decision.action, 'CREATE')
        with self.assertRaises(StoreCommitError) as ctx:
            apply_decision(decision, fact, self.index, self.store,
                           semantic_key=self.keys['2018'])       # 故意不传 memory=
        self.assertIn('memory', str(ctx.exception))
        self.assertEqual(len(self.index), 0)                      # 索引没有被写脏

    def test_create_write_failure_leaves_index_empty(self):
        fact = make_fact('Alice', 'employer', 'Google',
                         text='In 2018: Alice works at Google',
                         observed_at='2018-01-01T00:00:00', fact_id='fact_2018')
        decision = self.resolver.resolve(fact, [], self.index)
        bad_memory = torch.zeros(MEMORY_SLOTS, DIMENSION + 1)      # 维度不符的 full memory
        with self.assertRaises(StoreCommitError):
            apply_decision(decision, fact, self.index, self.store,
                           memory=bad_memory, semantic_key=self.keys['2018'])
        self.assertEqual(len(self.index), 0)
        self.assertEqual(self.store.entries, [])

    def test_update_without_create_is_rejected_by_store(self):
        """F2 证据：CREATE 不落 store 时，后续 UPDATE 会被真实 store 直接拒绝。"""
        fact = make_fact('Alice', 'employer', 'Microsoft', fact_id='fact_2019')
        provisional = StateRecord(state_id='state_000001', subject_key='alice',
                                  attribute_key='employer', semantic_key=self.keys['2019'],
                                  current_version_id='fact_2018')
        with self.assertRaises(ValueError) as ctx:
            self.store.commit(record=provisional, fact=fact,
                              memory=self.memories['2019'],
                              semantic_key=self.keys['2019'],
                              decision=StateDecision('UPDATE', 'state_000001', .9, 'rule5b'))
        self.assertIn('requires an existing current version', str(ctx.exception))

    def test_commit_with_default_path_requires_memory(self):
        """F1 证据：默认 apply 不传 memory 时抛 StoreCommitError；补上 memory 后成功。"""
        record = self.index.add_keys('Alice', 'employer', 'Google', version_id='fact_2018',
                                     observed_at='2018-01-01T00:00:00',
                                     semantic_key=self.keys['2018'])
        fact = make_fact('Alice', 'employer', 'Microsoft',
                         text='In 2019: Alice works at Microsoft',
                         observed_at='2019-01-01T00:00:00', fact_id='fact_2019')
        # 但索引里没有 store 的 current，需要先建好 store 的 CREATE
        self.store.commit(record=record, fact=make_fact(
            'Alice', 'employer', 'Google', observed_at='2018-01-01T00:00:00',
            fact_id='fact_2018'), memory=self.memories['2018'],
            semantic_key=self.keys['2018'],
            decision=StateDecision('CREATE', record.state_id, .9, 'rule1'))
        with self.assertRaises(StoreCommitError) as ctx:
            self.index.commit_with(self.store, record.state_id, fact)
        self.assertIn('memory', str(ctx.exception))
        self.assertEqual(record.current_version_id, 'fact_2018')
        self.index.commit_with(self.store, record.state_id, fact,
                               memory=self.memories['2019'], semantic_key=self.keys['2019'],
                               decision=StateDecision('UPDATE', record.state_id, .9, 'rule5b'))
        self.assertEqual(record.current_version_id, 'fact_2019')
        self.assertEqual(self.store.current(record.state_id)['id'], 'fact_2019')

    def test_duplicate_writes_alias_only(self):
        self.ingest('2018', 'Google')
        before = len(self.store.entries)
        fact = make_fact('Alice', 'employer', 'Google',
                         text='In 2019: Alice works at Google',
                         observed_at='2019-01-01T00:00:00', fact_id='fact_2019')
        decision = self.resolver.resolve(fact, retrieve_candidates(fact, self.index), self.index)
        self.assertEqual(decision.action, 'DUPLICATE')
        apply_decision(decision, fact, self.index, self.store,
                       memory=self.memories['2019'], semantic_key=self.keys['2019'])
        # 版本数不变：DUPLICATE 只记 alias 元数据（store.duplicates）
        self.assertEqual(len(self.store.entries), before)
        self.assertEqual(len(self.store.duplicates), 1)
        self.assertEqual(self.index.records[0].current_version_id, 'fact_2018')

    def test_defer_writes_store_pending_without_touching_current(self):
        self.ingest('2018', 'Google')
        # 同值但文本否定 → rule6 negation → DEFER
        fact = make_fact('Alice', 'employer', 'Microsoft',
                         text='Alice is no longer employed by Google',
                         observed_at='2019-01-01T00:00:00', fact_id='fact_2019')
        decision = self.resolver.resolve(fact, retrieve_candidates(fact, self.index), self.index)
        self.assertEqual(decision.action, 'DEFER')
        apply_decision(decision, fact, self.index, self.store,
                       memory=self.memories['2019'], semantic_key=self.keys['2019'])
        self.assertEqual(len(self.store.pending), 1)
        self.assertEqual(self.store.pending[0]['id'], 'fact_2019')
        self.assertEqual(self.store.current(self.index.records[0].state_id)['id'], 'fact_2018')


if __name__ == '__main__':
    unittest.main()
