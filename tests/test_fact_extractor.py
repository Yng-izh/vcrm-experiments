from __future__ import annotations
# P2 回归测试：确定性事实抽取（TempLAMA 单事实 + LongMemEval 多事实骨架）。
# stdlib unittest，pytest 也可直接收集。
#
# 防泄漏哨兵：每个 Oracle 字段都带独一无二的哨兵值，只要抽取路径读到它就会被抓住。

import unittest

from vcrm.data_views import (
    ORACLE_FIELDS,
    PublicEventCounter,
    assert_no_oracle,
    assert_public_id_independent,
    public_write_event,
)
from vcrm.fact_extractor import (
    UNKNOWN_TIME,
    UNRESOLVED_ATTRIBUTE,
    UNRESOLVED_SUBJECT,
    ExtractionContext,
    FactExtractor,
    parse_observed_at,
)

ORACLE_SENTINELS = {
    'state_key': 'SENTINEL_STATE_KEY_Q42_P35',
    'key_text': 'SENTINEL_KEY_TEXT_HEAD_OF_GOVERNMENT',
    'entity': 'SENTINEL_ENTITY_Q42',
    'group': 'SENTINEL_GROUP_Q42_P35',
    'answers': ['SENTINEL_ANSWER'],
    'latest': True,
    'episode': 'SENTINEL_EPISODE_HANDLE',
    'as_of': 'SENTINEL_AS_OF_2016',
    'is_update': True,
    'target_event': 'SENTINEL_TARGET_EVENT',
}


def oracle_event(text, time='2015-01-01T00:00:00', event_id='Q42_2015'):
    """带全套 Oracle 字段的原始事件（模拟 vcrm.data 里的 Oracle 对照组输入）。"""
    event = {'id': event_id, 'text': text, 'time': time}
    event.update(ORACLE_SENTINELS)
    return event


def public_event(text, time='2015-01-01T00:00:00', public_id='input_000001'):
    return public_write_event(oracle_event(text, time), public_id)


class TempLamaSingleFactTest(unittest.TestCase):
    def setUp(self):
        self.extractor = FactExtractor()

    def test_of_template_extracts_subject_attribute_value_time(self):
        event = public_event('In 2015: The head of government of France is Manuel Valls')
        facts = self.extractor.extract(event)
        self.assertEqual(len(facts), 1)
        fact = facts[0]
        self.assertEqual(fact.subject, 'France')
        self.assertEqual(fact.attribute, 'head_of_government')
        self.assertEqual(fact.value, 'Manuel Valls')
        self.assertEqual(fact.observed_at, '2015-01-01T00:00:00')
        self.assertEqual(fact.source_id, 'input_000001')
        self.assertEqual(fact.state_identity, ('France', 'head_of_government'))

    def test_possessive_template(self):
        facts = self.extractor.extract(public_event("Japan's capital is Tokyo"))
        self.assertEqual((facts[0].subject, facts[0].attribute, facts[0].value),
                         ('Japan', 'capital', 'Tokyo'))

    def test_inverse_copula_template(self):
        facts = self.extractor.extract(public_event('Paris is the capital of France'))
        self.assertEqual((facts[0].subject, facts[0].attribute, facts[0].value),
                         ('France', 'capital', 'Paris'))

    def test_attribute_is_normalized_language_not_relation_id(self):
        event = public_event('In 2015: The Capital City of Japan is Tokyo')
        fact = self.extractor.extract(event)[0]
        # 语言表面形式归一化，而不是 P-id / relation 字符串映射
        self.assertEqual(fact.attribute, 'capital')
        # 不是 P-id / Q-id / state_key 片段
        self.assertNotRegex(fact.attribute, r'\bp\d+\b')
        self.assertNotRegex(fact.attribute, r'\bq\d+\b')
        self.assertNotIn(':', fact.attribute)
        self.assertNotIn('_x_', fact.attribute)

    def test_observed_at_parsing_variants(self):
        self.assertEqual(parse_observed_at('In 2015'), '2015-01-01T00:00:00')
        self.assertEqual(parse_observed_at('on 3 March 2019'), '2019-03-03T00:00:00')
        self.assertEqual(parse_observed_at('since 2019-07-04'), '2019-07-04T00:00:00')
        self.assertEqual(parse_observed_at('no date here', '2020-05-05T00:00:00'),
                         '2020-05-05T00:00:00')
        self.assertEqual(parse_observed_at('no date here'), UNKNOWN_TIME)

    def test_unresolved_fact_kept_in_independent_memory(self):
        event = public_event('Lorem ipsum dolor sit amet')
        result = self.extractor.extract_detailed(event)
        self.assertEqual(len(result.facts), 1)
        fact = result.facts[0]
        self.assertEqual(fact.subject, UNRESOLVED_SUBJECT)
        self.assertEqual(fact.attribute, UNRESOLVED_ATTRIBUTE)
        self.assertEqual(len(result.unresolved), 1)
        self.assertEqual(len(self.extractor.unresolved), 1)
        diagnostic = result.diagnostics[0]
        self.assertLess(diagnostic.confidence, 0.5)
        self.assertIn('unresolved_identity', diagnostic.reasons)
        self.assertTrue(diagnostic.failed)

    def test_unresolved_attribute_only(self):
        event = public_event('Alice is happy today')
        result = self.extractor.extract_detailed(event)
        fact = result.facts[0]
        # 属性抽不出来时不猜：主体也一并降级为未解析，并给出失败原因
        self.assertEqual(fact.subject, UNRESOLVED_SUBJECT)
        self.assertIn('subject_unresolved', result.diagnostics[0].reasons)
        self.assertLess(result.diagnostics[0].confidence, 0.5)
        self.assertEqual(len(self.extractor.unresolved), 1)

    def test_public_view_guard_rejects_oracle_fields(self):
        raw = oracle_event('In 2015: The head of government of France is Manuel Valls')
        with self.assertRaises(ValueError) as ctx:
            self.extractor.extract(raw)
        message = str(ctx.exception)
        self.assertTrue(any(field in message for field in ORACLE_FIELDS))

    def test_fact_ids_independent_of_source_id(self):
        counter = PublicEventCounter()
        event = public_event('The capital of France is Paris', public_id=counter.next_id())
        fact = self.extractor.extract(event)[0]
        assert_public_id_independent(fact.id, event['id'])
        self.assertNotIn(event['id'], fact.id)

    def test_fact_and_provenance_carry_no_oracle_fields(self):
        event = public_event('In 2015: The head of government of France is Manuel Valls')
        result = self.extractor.extract_detailed(event)
        for fact in result.facts:
            assert_no_oracle(fact.__dict__, 'AtomicFact')
        assert_no_oracle(self.extractor.serialize_provenance(), 'provenance')


class LongMemEvalMultiFactTest(unittest.TestCase):
    TEXT = ('user: Alice moved to Berlin in 2019. Alice\'s employer is Siemens.\n'
            'assistant: Thanks, noted.')

    def setUp(self):
        self.extractor = FactExtractor()

    def test_splits_multiple_facts_with_provenance(self):
        event = public_event(self.TEXT, time='2019-01-01T00:00:00', public_id='input_000002')
        result = self.extractor.extract_detailed(event)
        resolved = [(f.subject, f.attribute, f.value) for f in result.facts
                    if f.subject != UNRESOLVED_SUBJECT]
        self.assertIn(('Alice', 'residence', 'Berlin'), resolved)
        self.assertIn(('Alice', 'employer', 'Siemens'), resolved)
        self.assertGreaterEqual(len(result.facts), 2)
        # 每条事实都能回溯到 session/turn/时间/文本区间
        for provenance in result.provenance:
            self.assertEqual(provenance.session, 'input_000002')
            self.assertTrue(provenance.turn)
            self.assertEqual(provenance.observed_at, '2019-01-01T00:00:00')
            self.assertGreaterEqual(provenance.char_start, 0)
            self.assertLessEqual(provenance.char_end, len(self.TEXT))
            self.assertLess(provenance.char_start, provenance.char_end)
        spans = [(p.char_start, p.char_end) for p in result.provenance]
        self.assertEqual(spans, sorted(spans))

    def test_default_coref_resolves_possessive_pronoun(self):
        event = public_event('Alice works at Acme. Her employer is Acme.')
        result = self.extractor.extract_detailed(event)
        subjects = [f.subject for f in result.facts]
        self.assertEqual(subjects, ['Alice', 'Alice'])
        self.assertEqual({f.attribute for f in result.facts}, {'employer'})

    def test_custom_coref_hook_is_used(self):
        calls = []

        def resolver(mention, context: ExtractionContext):
            calls.append(mention)
            return 'Bob'

        extractor = FactExtractor(reference_resolver=resolver)
        facts = extractor.extract(public_event('His employer is Initech'))
        self.assertTrue(calls)
        self.assertEqual(facts[0].subject, 'Bob')
        self.assertEqual(facts[0].attribute, 'employer')
        self.assertEqual(facts[0].value, 'Initech')

    def test_multi_fact_off_keeps_single_fact(self):
        extractor = FactExtractor(split_multi_fact=False)
        facts = extractor.extract(public_event('Alice works at Acme. Bob works at Initech.'))
        self.assertEqual(len(facts), 1)


if __name__ == '__main__':
    unittest.main()
