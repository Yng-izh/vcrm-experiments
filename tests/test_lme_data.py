from __future__ import annotations
"""t22：统一 LongMemEval-S 加载器（vcrm/lme_data.py）一致性测试。

覆盖：官方 cleaned schema（haystack_sessions 是 turn 列表 + ids/dates 按下标对齐）、
长度不一致必须显式报错而不是 zip 截断、gold evidence 绝不进入 Episode、
date 升序稳定排序、缺 question_date 等边界，以及一个真实数据存在性门控测试
（本地无数据文件则 skip；有数据则断言 500 题 / 六类任务）。

stdlib unittest only；不联网、不下载数据、不跑 GPU。
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

from vcrm.lme_data import (
    GOLD_ONLY_FIELDS,
    LME_TASK_TYPES,
    OFFICIAL_S_SIZE,
    OFFICIAL_S_TYPE_COUNTS,
    PUBLIC_ITEM_FIELDS,
    Episode,
    adapt_prepared_item,
    assert_public_episode,
    detect_lme_s_format,
    load_lme_s,
    parse_item,
    task_type_counts,
)

REPO = Path(__file__).resolve().parents[1]

#: 真实数据候选路径（服务器镜像优先；本地 .gitignore 内路径其次）。可用
#: LME_S_DATA 环境变量覆盖。
DATA_CANDIDATES = [
    os.environ.get('LME_S_DATA', ''),
    '/share/latent2/vcrm_icae/data/prepared/longmemeval_test.json',
    str(REPO / 'data' / 'prepared' / 'longmemeval_test.json'),
    str(REPO / 'data' / 'longmemeval_s_cleaned.json'),
]


def turns(*pairs):
    """[('user', 'hi'), ...] -> 官方 turn 列表 [{'role':..,'content':..}, ...]。"""
    return [{'role': role, 'content': content} for role, content in pairs]


def row(question_id='q1', dates=None, session_ids=None, sessions=None, **extra):
    """构造官方 cleaned 形态的一行（三个对齐数组 + 可选 gold evidence）。"""
    dates = ['2023/05/20 (Sat) 10:00'] if dates is None else dates
    session_ids = [f'S{i}' for i in range(len(dates))] if session_ids is None else session_ids
    sessions = [turns(('user', f'message {i}')) for i in range(len(dates))] if sessions is None else sessions
    payload = {
        'question_id': question_id,
        'question': 'What did I say?',
        'question_type': 'single-session-user',
        'question_date': '2023/06/01 (Thu) 09:00',
        'answer': 'a fact',
        'haystack_session_ids': session_ids,
        'haystack_dates': dates,
        'haystack_sessions': sessions,
    }
    payload.update(extra)
    return payload


class OfficialSchemaParsingTest(unittest.TestCase):
    def test_turns_are_parsed_as_turn_lists_not_session_dicts(self):
        item = parse_item(row(sessions=[turns(('user', 'I moved to Oslo'),
                                             ('assistant', 'Noted, Oslo.'))]))
        self.assertEqual(len(item.sessions), 1)
        session = item.sessions[0]
        self.assertEqual(session.session_id, 'S0')
        self.assertEqual([t.role for t in session.turns], ['user', 'assistant'])
        self.assertEqual([t.content for t in session.turns], ['I moved to Oslo', 'Noted, Oslo.'])
        self.assertEqual(session.text, 'user: I moved to Oslo\nassistant: Noted, Oslo.')

    def test_public_projection_of_question_fields(self):
        item = parse_item(row(question_id='q_abs', question_type='knowledge-update'))
        self.assertEqual(item.question_id, 'q_abs')
        self.assertEqual(item.question, 'What did I say?')
        self.assertEqual(item.question_type, 'knowledge-update')
        self.assertEqual(item.type, 'knowledge-update')
        self.assertEqual(item.answers, ('a fact',))
        self.assertTrue(item.abstention)  # '_abs' 后缀
        self.assertEqual(item.question_date, '2023-06-01T09:00:00')

    def test_abstention_flag_follows_question_id_suffix(self):
        self.assertTrue(parse_item(row(question_id='abc_abs')).abstention)
        self.assertFalse(parse_item(row(question_id='abc')).abstention)

    def test_missing_question_date_is_none_and_does_not_raise(self):
        payload = row()
        payload.pop('question_date')
        item = parse_item(payload)
        self.assertIsNone(item.question_date)
        self.assertEqual(item.sessions[0].session_id, 'S0')

    def test_missing_answer_becomes_empty_answers(self):
        payload = row()
        payload.pop('answer')
        self.assertEqual(parse_item(payload).answers, ())

    def test_unknown_fields_are_dropped_not_propagated(self):
        item = parse_item(row(mystery_field='should not survive'))
        self.assertFalse(hasattr(item, 'mystery_field'))
        self.assertNotIn('mystery_field', item.to_dict())


class AlignmentTest(unittest.TestCase):
    def test_misaligned_arrays_raise_instead_of_zip_truncating(self):
        # ids 比 dates/sessions 短
        with self.assertRaises(ValueError) as ctx:
            parse_item(row(dates=['2023/01/01', '2023/01/02'],
                           session_ids=['S0'],
                           sessions=[turns(('user', 'a')), turns(('user', 'b'))]))
        self.assertIn('misaligned', str(ctx.exception))
        # dates 比另外两个长
        with self.assertRaises(ValueError):
            parse_item(row(dates=['2023/01/01', '2023/01/02', '2023/01/03'],
                           session_ids=['S0', 'S1'],
                           sessions=[turns(('user', 'a')), turns(('user', 'b'))]))
        # sessions 比另外两个短
        with self.assertRaises(ValueError):
            parse_item(row(dates=['2023/01/01', '2023/01/02'],
                           session_ids=['S0', 'S1'],
                           sessions=[turns(('user', 'a'))]))

    def test_error_message_reports_all_three_lengths(self):
        with self.assertRaises(ValueError) as ctx:
            parse_item(row(dates=['2023/01/01', '2023/01/02'], session_ids=['S0', 'S1'],
                           sessions=[turns(('user', 'a'))]))
        message = str(ctx.exception)
        for fragment in ('haystack_session_ids=2', 'haystack_dates=2', 'haystack_sessions=1'):
            self.assertIn(fragment, message)

    def test_session_must_be_turn_list_not_dict(self):
        with self.assertRaises(ValueError) as ctx:
            parse_item(row(sessions=[{'session_id': 'S0', 'date': '2023/01/01',
                                      'turns': turns(('user', 'a'))}]))
        self.assertIn('list of turns', str(ctx.exception))

    def test_turn_requires_role_and_content(self):
        with self.assertRaises(ValueError):
            parse_item(row(sessions=[[{'role': 'user'}]]))
        with self.assertRaises(ValueError):
            parse_item(row(sessions=[[{'content': 'no role'}]]))
        with self.assertRaises(ValueError):
            parse_item(row(sessions=[['just a string']]))

    def test_missing_required_item_fields_raise(self):
        for field in ('question_id', 'question', 'haystack_session_ids',
                      'haystack_dates', 'haystack_sessions'):
            payload = row()
            payload.pop(field)
            with self.assertRaises(ValueError):
                parse_item(payload)


class ChronologicalOrderTest(unittest.TestCase):
    def test_sessions_sorted_by_date_ascending(self):
        item = parse_item(row(
            dates=['2023/03/01 (Wed) 10:00', '2023/01/01 (Sun) 10:00', '2023/02/01 (Wed) 10:00'],
            session_ids=['S_late', 'S_early', 'S_mid'],
            sessions=[turns(('user', 'late')), turns(('user', 'early')), turns(('user', 'mid'))]))
        self.assertEqual([s.session_id for s in item.sessions], ['S_early', 'S_mid', 'S_late'])

    def test_equal_dates_keep_original_relative_order(self):
        item = parse_item(row(
            dates=['2023/01/01 (Sun) 10:00', '2023/01/01 (Sun) 10:00', '2023/01/02 (Mon) 10:00'],
            session_ids=['S_b', 'S_a', 'S_c'],
            sessions=[turns(('user', 'first')), turns(('user', 'second')), turns(('user', 'third'))]))
        self.assertEqual([s.session_id for s in item.sessions], ['S_b', 'S_a', 'S_c'])

    def test_render_preserves_official_date_prefix(self):
        item = parse_item(row(dates=['2023/05/20 (Sat) 10:00']))
        self.assertEqual(item.sessions[0].render(prefix_date=True),
                         '[2023/05/20 (Sat) 10:00] user: message 0')
        self.assertEqual(item.session_texts(), ['user: message 0'])


class GoldIsolationTest(unittest.TestCase):
    def test_gold_evidence_ids_never_reach_the_episode(self):
        payload = row(question_id='q_gold',
                      answer_session_ids=['S0'],
                      evidence_session_ids=['S0'],
                      evidence={'S0': ['span']},
                      has_answer=True)
        item = parse_item(payload)
        flat = json.dumps(item.to_dict(), ensure_ascii=False)
        for field in ('answer_session_ids', 'evidence_session_ids', 'has_answer'):
            self.assertNotIn(field, flat)
        self.assertFalse(hasattr(item, 'answer_session_ids'))
        self.assertEqual(set(item.to_dict()),
                         {'question_id', 'question', 'type', 'question_type', 'answers',
                          'abstention', 'question_date', 'sessions'})
        self.assertEqual(set(item.to_dict()['sessions'][0]),
                         {'session_id', 'date', 'raw_date', 'turns'})

    def test_assert_public_episode_accepts_episode_and_rejects_gold_keys(self):
        item = parse_item(row())
        self.assertIs(assert_public_episode(item), item)
        for gold in ('answer_session_ids', 'evidence', 'state_key'):
            with self.assertRaises(ValueError):
                assert_public_episode({'question_id': 'q', gold: ['S0']})
        self.assertIn('answer_session_ids', GOLD_ONLY_FIELDS)


class LoaderFileTest(unittest.TestCase):
    def test_load_lme_s_reads_json_array_and_jsonl_identically(self):
        rows = [row(question_id='q1'), row(question_id='q2', dates=['2023/01/02 (Mon) 10:00'])]
        with tempfile.TemporaryDirectory() as tmp:
            array_path = Path(tmp) / 'array.json'
            jsonl_path = Path(tmp) / 'records.jsonl'
            array_path.write_text(json.dumps(rows))
            jsonl_path.write_text('\n'.join(json.dumps(r) for r in rows))
            from_array = load_lme_s(array_path)
            from_jsonl = load_lme_s(jsonl_path)
        self.assertEqual([e.to_dict() for e in from_array], [e.to_dict() for e in from_jsonl])
        self.assertEqual([e.question_id for e in from_array], ['q1', 'q2'])
        self.assertTrue(all(isinstance(e, Episode) for e in from_array))

    def test_load_lme_s_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty.json'
            path.write_text('[]')
            with self.assertRaises(ValueError):
                load_lme_s(path)

    def test_task_type_counts_covers_six_official_types(self):
        episodes = [parse_item(row(question_id=f'q{i}', question_type=t))
                    for i, t in enumerate(LME_TASK_TYPES)]
        counts = task_type_counts(episodes)
        self.assertEqual(set(counts), set(LME_TASK_TYPES))
        self.assertTrue(all(value == 1 for value in counts.values()))


class TextRagAdapterTest(unittest.TestCase):
    """要求 6：lme_s_text_rag.normalize_item() 必须复用同一份 loader。"""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        path = REPO / 'lme_s_text_rag' / 'run_text_rag.py'
        spec = importlib.util.spec_from_file_location('lme_run_text_rag_data_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.rag = module

    def test_normalize_item_matches_shared_loader_order_and_rendering(self):
        item = row(dates=['2023/03/01 (Wed) 10:00', '2023/01/01 (Sun) 10:00'],
                   session_ids=['S_late', 'S_early'],
                   sessions=[turns(('user', 'late fact')), turns(('assistant', 'early fact'))])
        normalized = self.rag.normalize_item(item)
        episode = parse_item(item)
        self.assertEqual(set(normalized), {'question_id', 'question', 'question_type', 'sessions'})
        self.assertEqual([s['session_id'] for s in normalized['sessions']],
                         [s.session_id for s in episode.sessions])
        self.assertEqual(normalized['sessions'][0]['text'],
                         '[2023/01/01 (Sun) 10:00] assistant: early fact')
        self.assertEqual(normalized['sessions'][1]['text'],
                         '[2023/03/01 (Wed) 10:00] user: late fact')
        self.assertEqual([s['date'] for s in normalized['sessions']],
                         [s.date for s in episode.sessions])

    def test_normalize_item_propagates_alignment_error(self):
        with self.assertRaises(ValueError):
            self.rag.normalize_item(row(dates=['2023/01/01', '2023/01/02'],
                                        session_ids=['S0'],
                                        sessions=[turns(('user', 'a')), turns(('user', 'b'))]))

    def test_normalize_item_rejects_session_dicts(self):
        with self.assertRaises(ValueError):
            self.rag.normalize_item(row(sessions=[{'session_id': 'S0',
                                                   'turns': turns(('user', 'a'))}]))


def prepared_row(question_id='q1', events=None, question='What did I say?',
                 question_type='single-session-user', answers=('a fact',),
                 question_date='2023/06/01 (Thu) 09:00', **extra):
    """构造 prepared 派生行（vcrm.data.prepare() 的输出形态：events/queries）。"""
    payload = {
        'id': f'longmemeval:{question_id}',
        'dataset': 'longmemeval',
        'split': 'test',
        'group': question_id,
        'events': events if events is not None else [
            {'id': 'event_000000', 'time': '2023-05-20T10:00:00',
             'text': 'user: I moved to Oslo\nassistant: Noted, Oslo.'}],
        'queries': [{'id': question_id, 'question': question, 'time': question_date,
                     'answers': list(answers), 'type': question_type,
                     'abstention': False}],
    }
    payload.update(extra)
    return payload


class PreparedAdapterTest(unittest.TestCase):
    """t29：prepared（events/queries）→ cleaned schema 适配层。"""

    def test_events_become_sessions_with_role_content_turns(self):
        payload = prepared_row(events=[
            {'id': 'event_000001', 'time': '2023-03-01T10:00:00',
             'text': 'user: late fact\nassistant: ok'},
            {'id': 'event_000000', 'time': '2023-01-01T10:00:00',
             'text': 'user: early fact'},
        ])
        episode = parse_item(adapt_prepared_item(payload))
        self.assertEqual([s.session_id for s in episode.sessions],
                         ['event_000000', 'event_000001'])  # 按 date 升序
        self.assertEqual([(t.role, t.content) for t in episode.sessions[1].turns],
                         [('user', 'late fact'), ('assistant', 'ok')])
        self.assertEqual(episode.question_id, 'q1')
        self.assertEqual(episode.question, 'What did I say?')
        self.assertEqual(episode.question_type, 'single-session-user')
        self.assertEqual(episode.answers, ('a fact',))
        self.assertEqual(episode.question_date, '2023-06-01T09:00:00')

    def test_multiline_content_is_merged_into_the_previous_turn(self):
        payload = prepared_row(events=[
            {'id': 'e0', 'time': '2023-01-01T10:00:00',
             'text': 'user: line one\nline two still user\nassistant: got it'}])
        episode = parse_item(adapt_prepared_item(payload))
        self.assertEqual([(t.role, t.content) for t in episode.sessions[0].turns],
                         [('user', 'line one\nline two still user'), ('assistant', 'got it')])

    def test_missing_event_date_raises_instead_of_guessing(self):
        payload = prepared_row(events=[{'id': 'e0', 'text': 'user: hi'}])
        with self.assertRaises(ValueError) as ctx:
            adapt_prepared_item(payload)
        self.assertIn('date', str(ctx.exception))
        self.assertIn('refusing to invent a timestamp', str(ctx.exception))

    def test_missing_question_raises(self):
        payload = prepared_row()
        payload['queries'][0].pop('question')
        with self.assertRaises(ValueError):
            adapt_prepared_item(payload)

    def test_more_than_one_query_or_answer_is_rejected(self):
        payload = prepared_row()
        payload['queries'].append(dict(payload['queries'][0], id='q2'))
        with self.assertRaises(ValueError):
            adapt_prepared_item(payload)
        payload = prepared_row(answers=('a', 'b'))
        with self.assertRaises(ValueError):
            adapt_prepared_item(payload)

    def test_adapted_row_passes_public_check_and_drops_gold_fields(self):
        payload = prepared_row(events=[
            {'id': 'e0', 'time': '2023-01-01T10:00:00', 'text': 'user: hi',
             'answer_session_ids': ['e0'], 'evidence': {'e0': ['span']}}])
        cleaned = adapt_prepared_item(payload)
        self.assertIs(assert_public_episode(cleaned, 'adapted'), cleaned)
        self.assertEqual(set(cleaned), set(PUBLIC_ITEM_FIELDS))
        flat = json.dumps(cleaned)
        for gold in ('answer_session_ids', 'evidence'):
            self.assertNotIn(gold, flat)
        episode = parse_item(cleaned)
        self.assertFalse(hasattr(episode, 'answer_session_ids'))

    def test_question_id_falls_back_to_group_and_strips_prefix(self):
        payload = prepared_row()
        payload['queries'][0].pop('id')
        self.assertEqual(adapt_prepared_item(payload)['question_id'], 'q1')
        payload = prepared_row()
        payload.pop('group')
        payload['queries'][0].pop('id')
        self.assertEqual(adapt_prepared_item(payload)['question_id'], 'q1')

    def test_unparseable_turn_text_raises(self):
        payload = prepared_row(events=[{'id': 'e0', 'time': '2023-01-01T10:00:00',
                                        'text': 'no role prefix here'}])
        with self.assertRaises(ValueError):
            adapt_prepared_item(payload)

    def test_load_lme_s_autodetects_prepared_files(self):
        rows = [prepared_row(question_id='q1'), prepared_row(question_id='q2')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'prepared.json'
            path.write_text(json.dumps(rows))
            episodes = load_lme_s(path)
        self.assertEqual([e.question_id for e in episodes], ['q1', 'q2'])
        self.assertTrue(all(isinstance(e, Episode) for e in episodes))

    def test_load_lme_s_rejects_mixed_and_unknown_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            mixed = Path(tmp) / 'mixed.json'
            mixed.write_text(json.dumps([row(), prepared_row()]))
            with self.assertRaises(ValueError) as ctx:
                load_lme_s(mixed)
            self.assertIn('mixed', str(ctx.exception))
            unknown = Path(tmp) / 'unknown.json'
            unknown.write_text(json.dumps([{'foo': 'bar'}]))
            with self.assertRaises(ValueError) as ctx:
                load_lme_s(unknown)
            for fragment in ('haystack_sessions', 'events', 'queries'):
                self.assertIn(fragment, str(ctx.exception))

    def test_official_cleaned_path_is_untouched(self):
        self.assertEqual(detect_lme_s_format(row()), 'cleaned')
        self.assertEqual(detect_lme_s_format(prepared_row()), 'prepared')
        cleaned_episode = parse_item(row(question_id='q_clean'))
        self.assertEqual(cleaned_episode.sessions[0].session_id, 'S0')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cleaned.json'
            path.write_text(json.dumps([row(question_id='q_clean')]))
            loaded = load_lme_s(path)
        self.assertEqual([e.to_dict() for e in loaded], [cleaned_episode.to_dict()])

    def test_ambiguous_row_with_both_markers_is_rejected(self):
        payload = row()
        payload.update({'events': [], 'queries': []})
        with self.assertRaises(ValueError) as ctx:
            detect_lme_s_format(payload)
        self.assertIn('ambiguous', str(ctx.exception))


class RealDataGateTest(unittest.TestCase):
    """真实数据门控：本地无文件则 skip；有数据则校验官方规模与六类计数。"""

    def setUp(self):
        self.path = next((Path(p) for p in DATA_CANDIDATES if p and Path(p).exists()), None)
        if self.path is None:
            self.skipTest('LongMemEval-S prepared data not present locally '
                          '(set LME_S_DATA to enable this gate)')

    def test_format_detection_on_real_data(self):
        """双格式探测：服务器 prepared 文件与本地 cleaned 文件都必须能加载。"""
        from vcrm.io import read_records
        records = read_records(self.path)
        detected = {detect_lme_s_format(record) for record in records}
        self.assertEqual(len(detected), 1, f'mixed formats in {self.path}')
        self.assertIn(detected.pop(), ('cleaned', 'prepared'))
        self.assertEqual(len(load_lme_s(self.path)), OFFICIAL_S_SIZE)

    def test_official_scale_and_task_type_counts(self):
        episodes = load_lme_s(self.path)
        self.assertEqual(len(episodes), OFFICIAL_S_SIZE)
        self.assertEqual(len({e.question_id for e in episodes}), OFFICIAL_S_SIZE)
        counts = task_type_counts(episodes)
        self.assertEqual(set(counts), set(LME_TASK_TYPES))
        self.assertEqual(sum(counts.values()), OFFICIAL_S_SIZE)
        self.assertTrue(all(value > 0 for value in counts.values()), counts)
        for episode in episodes:
            self.assertTrue(episode.sessions, episode.question_id)
            dates = [session.date for session in episode.sessions]
            self.assertEqual(dates, sorted(dates), episode.question_id)
        if os.environ.get('LME_STRICT_OFFICIAL_COUNTS') == '1':
            self.assertEqual(counts, OFFICIAL_S_TYPE_COUNTS)


if __name__ == '__main__':
    unittest.main()
