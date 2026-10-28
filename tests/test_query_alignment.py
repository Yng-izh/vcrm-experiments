"""T32-F1 回归：查询身份必须能对齐到写入侧真实状态键空间。

用 GPU 冒烟抓到的**真实失败例子**做用例（不是构造的玩具例）：

    写侧（LLMFactExtractor）：subject='1997 jeep wrangler' / 'engine' 等具体实体
    读侧（LLMQueryParser）  ：subject='I'
                              attribute='number of engineers led when I just
                                         started my new role as Senior Software Engineer'
    → 归一化后仍不相交 → identity_matched 恒空 → no_state_match=True → 答案全空

本文件钉死三件事：
1) 不对齐时两侧 subject_key 必然不相交（记录失败基线，防回退）；
2) 对齐后两侧 subject_key 必须相交，且属性落回写入侧词表；
3) 对齐是**逐级降级 + 全程记账**：exact / user_alias / subject_fuzzy /
   attribute_global / unmatched / no_vocabulary，每一步都能审计。
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vcrm.fact_schema import AtomicFact                        # noqa: E402
from vcrm.llm_query_parser import LLMQueryParser               # noqa: E402
from vcrm.query_alignment import (                             # noqa: E402
    ALIGNMENT_SOURCES,
    StateVocabularyAligner,
    align_identities,
    containment_score,
)
from vcrm.state_index import (                                 # noqa: E402
    SEMANTIC_DIMENSIONS,
    StateIndex,
    normalize_subject,
)

#: GPU 冒烟（t32）里读侧真实输出的身份。
T32_QUERY_SUBJECT = 'I'
T32_QUERY_ATTRIBUTE = ('number of engineers led when I just started my new role '
                       'as Senior Software Engineer')
#: 写侧真实出现过的具体实体（llm_cache 原文）。
T32_WRITE_SUBJECTS = ('1997 jeep wrangler', 'engine')


def zero_key() -> np.ndarray:
    return np.zeros(SEMANTIC_DIMENSIONS, dtype=np.float32)


def make_index(entries) -> StateIndex:
    """entries = [(subject, attribute, state_id), …] → 带记录的 StateIndex。"""
    index = StateIndex()
    for position, (subject, attribute, state_id) in enumerate(entries):
        fact = AtomicFact(id=f'fact_{position:06d}', text=f'{subject} {attribute}',
                          observed_at='2021-01-01T00:00:00', subject=subject,
                          attribute=attribute, value='x', source_id='input_000001')
        index.add(fact, semantic_key=zero_key(), state_id=state_id)
    return index


class FakeLLM:
    """按脚本返回固定 JSON 的假模型（不碰网络）。"""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def complete(self, prompt, *, system=None, max_tokens=512):
        self.calls.append(prompt)
        return json.dumps(self.payload)


class ContainmentScoreTest(unittest.TestCase):
    def test_query_sentence_contains_short_key(self):
        self.assertEqual(containment_score(T32_QUERY_ATTRIBUTE.replace(' ', '_'),
                                           'engineers_led'), 1.0)

    def test_unrelated_keys_score_zero(self):
        self.assertEqual(containment_score('engine_type', 'favourite_colour'), 0.0)

    def test_stopwords_do_not_create_matches(self):
        self.assertEqual(containment_score('what_is_the_of_it', 'is_of_the'), 0.0)


class T32RegressionTest(unittest.TestCase):
    """t32 真实失败例：对齐前必然不相交，对齐后必须相交。"""

    def setUp(self):
        self.index = make_index([
            ('1997 jeep wrangler', 'engine_type', 'state_000001'),
            ('user', 'engineers_led', 'state_000002'),
            ('user', 'employer', 'state_000003'),
        ])
        self.vocabulary = {normalize_subject(record.subject_key) for record in self.index.records}

    def test_without_alignment_keys_are_disjoint(self):
        """失败基线：'I' 归一化后不在写入侧词表里（这正是冒烟里发生的事）。"""
        self.assertEqual(normalize_subject(T32_QUERY_SUBJECT), 'i')
        self.assertNotIn('i', self.vocabulary)
        self.assertEqual(self.vocabulary & {normalize_subject(T32_QUERY_SUBJECT)}, set())

    def test_alignment_makes_subject_keys_intersect(self):
        """修复判据：同句两侧 subject_key 必须相交，属性落回写入侧词表。"""
        aligned, report = align_identities([(T32_QUERY_SUBJECT, T32_QUERY_ATTRIBUTE)],
                                           self.index)
        self.assertEqual(len(aligned), 1)
        subject_key, attribute_key = aligned[0]
        self.assertIn(subject_key, self.vocabulary)        # 相交
        self.assertEqual(subject_key, 'user')              # 代词 → 用户实体
        self.assertEqual(attribute_key, 'engineers_led')   # 整句属性 → 写入侧短键
        self.assertIn(report[0]['source'],
                      ('user_alias', 'user_alias+subject_fuzzy', 'attribute_global'))

    def test_alignment_report_records_provenance(self):
        _, report = align_identities([(T32_QUERY_SUBJECT, T32_QUERY_ATTRIBUTE)], self.index)
        row = report[0]
        self.assertEqual(row['input_subject'], T32_QUERY_SUBJECT)
        self.assertEqual(row['input_attribute'], T32_QUERY_ATTRIBUTE)
        for field in ('subject_source', 'attribute_source', 'score'):
            self.assertIn(field, row)
        self.assertGreater(row['score'], 0.0)

    def test_write_side_entities_are_reachable_by_fuzzy_subject(self):
        aligned, report = align_identities([('jeep wrangler 1997', 'engine type')],
                                           self.index)
        self.assertEqual(aligned[0], ('1997_jeep_wrangler', 'engine_type'))
        self.assertEqual(report[0]['subject_source'], 'subject_fuzzy')

    def test_attribute_global_fallback_rescues_unknown_subject(self):
        """主体完全对不上时，属性词表是最后一道可救路径（且必须记账）。"""
        aligned, report = align_identities([('totally unrelated entity', 'engine type')],
                                           self.index)
        self.assertEqual(aligned[0], ('1997_jeep_wrangler', 'engine_type'))
        self.assertEqual(report[0]['subject_source'], 'attribute_global')


class DegradationTest(unittest.TestCase):
    def test_empty_vocabulary_is_passthrough(self):
        aligned, report = align_identities([('I', 'engineers led')], StateIndex())
        self.assertEqual(aligned, [('i', 'engineers_led')])
        self.assertEqual(report[0]['source'], 'no_vocabulary')

    def test_unmatched_keeps_original_and_is_not_dropped(self):
        index = make_index([('1997 jeep wrangler', 'engine_type', 'state_000001')])
        aligner = StateVocabularyAligner(index)
        aligned, report = aligner.align([('someone else', 'favourite colour')])
        self.assertEqual(report[0]['subject_source'], 'unmatched')
        self.assertEqual(aligned, [])                    # 不硬凑，交给路由判 no_state_match

    def test_rebind_rebuilds_vocabulary(self):
        aligner = StateVocabularyAligner(make_index([
            ('user', 'engineers_led', 'state_000001')]))
        self.assertEqual(aligner.align([('I', 'engineers led')])[0][0][0], 'user')
        aligner.bind(make_index([('1997 jeep wrangler', 'engine_type', 'state_000001')]))
        self.assertEqual(aligner.align([('I', 'engineers led')])[0], [])   # 新词表里没有
        self.assertEqual(sorted(aligner.vocabulary), ['1997_jeep_wrangler'])

    def test_aligner_follows_index_growth(self):
        """t33 GPU 复验现场：bind 发生在写入之前（索引为空），写入后必须可见。

        构造期快照会让所有身份降级成 no_vocabulary → 候选池为空 → 静默答空。
        """
        index = StateIndex()
        aligner = StateVocabularyAligner(index)          # 此刻索引为空
        self.assertEqual(aligner.align([('I', T32_QUERY_ATTRIBUTE)])[1][0]['source'],
                         'no_vocabulary')
        fact = AtomicFact(id='fact_x', text='user engineers led', observed_at='2021-01-01',
                          subject='user', attribute='engineers led', value='5',
                          source_id='input_000001')
        index.add(fact, semantic_key=zero_key(), state_id='state_000002')
        aligned, report = aligner.align([('I', T32_QUERY_ATTRIBUTE)])
        self.assertEqual(aligned, [('user', 'engineers_led')])   # 写入后立即可见
        self.assertNotEqual(report[0]['source'], 'no_vocabulary')

    def test_sources_are_frozen_constants(self):
        self.assertEqual(ALIGNMENT_SOURCES,
                         ('exact', 'user_alias', 'subject_fuzzy', 'attribute_global',
                          'unmatched', 'no_vocabulary'))


class ParserIntegrationTest(unittest.TestCase):
    """LLMQueryParser + 对齐层：解析结果必须直接可用于路由身份核对。"""

    def setUp(self):
        self.index = make_index([('user', 'engineers_led', 'state_000002')])
        self.payload = {'identities': [{'subject': 'I',
                                        'attribute': T32_QUERY_ATTRIBUTE,
                                        'confidence': 0.0,
                                        'subject_aliases': ['user']}]}

    def test_parser_with_aligner_returns_intersecting_identity(self):
        parser = LLMQueryParser(FakeLLM(self.payload),
                                aligner=StateVocabularyAligner(self.index))
        parsed = parser.parse('How many engineers do I lead when I just started '
                              'my new role as Senior Software Engineer?')
        self.assertTrue(parsed['resolved'])
        self.assertEqual(parsed['subject'], 'user')
        self.assertEqual(parsed['attribute'], 'engineers_led')
        self.assertEqual(parsed['identities'], [('user', 'engineers_led')])
        self.assertEqual(parsed['alignment'][0]['input_subject'], 'I')
        matched = [record for record in self.index.records
                   if record.matches_subject(parsed['subject'])
                   and record.matches_attribute(parsed['attribute'])]
        self.assertEqual([record.state_id for record in matched], ['state_000002'])

    def test_parser_without_aligner_is_unchanged(self):
        """不对齐时保持原样（既有行为逐比特不变，只多一个空的 alignment 字段）。"""
        parser = LLMQueryParser(FakeLLM(self.payload))
        parsed = parser.parse('How many engineers do I lead?')
        self.assertEqual(parsed['subject'], 'I')
        self.assertEqual(parsed['alignment'], [])

    def test_prompt_asks_for_subject_aliases(self):
        client = FakeLLM(self.payload)
        LLMQueryParser(client, aligner=StateVocabularyAligner(self.index)).parse('q?')
        self.assertIn('subject_aliases', client.calls[0])
        self.assertIn('keep the\nsubject exactly as written', client.calls[0])


if __name__ == '__main__':
    unittest.main()
