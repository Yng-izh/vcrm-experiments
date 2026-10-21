"""P4 LongMemEval-S latent 管线单测（lme_s_latent/run_e3.py）。

覆盖：
* 解析与 ``vcrm.data.longmemeval()`` 逐字段等价（500 条合成数据钉死字段契约）；
* 窗口末刻 = select_version 的上界：同一年两个版本（2019-03 / 2019-11）必须选到
  11 月那个，取区间起点则选不到任何版本；
* 每题独立 memory namespace（session_id 重名也不串题）；
* 分级错误分析五桶的上游优先判定（含 update_deferred 单列）；
* 5 题端到端（stub reader，off/on 两种 recency 模式）→ 输出可直接被
  ``lme_s_text_rag/make_official_eval_input.py`` 消费。

stdlib unittest only；不启动 GPU/vLLM，不需要 OPENAI_API_KEY。
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vcrm.data_views import PublicEventCounter
from vcrm.query_time import QueryTimeResolver

import importlib.util

_spec = importlib.util.spec_from_file_location('lme_run_e3', REPO_ROOT / 'lme_s_latent' / 'run_e3.py')
run_e3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_e3)


def make_row(question_id, question, question_date, sessions, answer='', question_type=''):
    """sessions: [(date_str, text)] → LongMemEval-S 官方字段形状。"""
    return {'question_id': question_id,
            'question': question,
            'question_date': question_date,
            'question_type': question_type,
            'answer': answer,
            'haystack_session_ids': [f'{question_id}_s{i}' for i in range(len(sessions))],
            'haystack_dates': [date for date, _ in sessions],
            'haystack_sessions': [[{'role': 'user', 'content': text}] for _, text in sessions]}


def make_args(**overrides):
    base = dict(reader='stub', arch_exp='E3', hidden=32, memory_slots=8, top_k=5,
                recency_update=False, recency_min_confidence=0.0, defer_store_memory=False,
                hybrid_fallback=False,
                tag='unit', runs_dir='unused', data='unused', limit=0, shard=0, shards=1,
                max_sessions=0, no_resume=True)
    base.update(overrides)
    return SimpleNamespace(**base)


def answer(item, *, recency_update=False, window_end=None, max_sessions=0):
    """写一个 item 并回答，返回 (runtime, written, resolved, result)。"""
    args = make_args(recency_update=recency_update)
    components = run_e3.build_components(args)
    runtime = run_e3.new_runtime(args, components)
    written = run_e3.write_item(runtime, item, PublicEventCounter(), max_sessions)
    resolved = QueryTimeResolver().resolve(item['question'], asked_at=item['question_date'])
    result = runtime.answer_predicted(
        {'question': item['question']},
        window_end=resolved.window_end if window_end is None else window_end)
    return runtime, written, resolved, result


class ParseParityTest(unittest.TestCase):
    """解析必须与 vcrm.data.longmemeval() 完全同构（本模块只放宽了 500 条限制）。"""

    def test_parity_with_vcrm_data_longmemeval(self):
        from vcrm.data import longmemeval

        rows = []
        for index in range(500):
            rows.append(make_row(
                question_id=f'q{index:03d}_abs' if index % 7 == 0 else f'q{index:03d}',
                question=f'What did I say about topic {index}?',
                question_date='2024/02/03 (Sat) 11:00',
                sessions=[('2021/03/04 (Thu) 09:00', f'user: note {index} alpha'),
                          ('2023/08/09 (Wed) 10:30', f'user: note {index} beta')],
                answer=f'answer {index}', question_type='single-session-user'))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'lme.json'
            path.write_text(json.dumps(rows))
            episodes = longmemeval(path)
            parsed = run_e3.load_items(str(path))

        self.assertEqual(len(episodes), 500)
        self.assertEqual(len(parsed), 500)
        for episode, item in zip(episodes, parsed):
            query = episode['queries'][0]
            self.assertEqual(item['question_id'], query['id'])
            self.assertEqual(item['question'], query['question'])
            self.assertEqual(item['question_date'], query['time'])
            self.assertEqual(item['answers'], query['answers'])
            self.assertEqual(item['question_type'], query['type'])
            self.assertEqual(item['abstention'], query['abstention'])
            self.assertEqual([(session['time'], session['text']) for session in item['sessions']],
                             [(event['time'], event['text']) for event in episode['events']])

    def test_misaligned_sessions_are_rejected(self):
        row = make_row('q1', 'q?', '2024/01/01 (Mon) 10:00', [('2021/01/01 (Fri) 09:00', 'x')])
        row['haystack_dates'] = []
        with self.assertRaises(ValueError):
            run_e3.parse_item(row)


class WindowEndUpperBoundTest(unittest.TestCase):
    """同一年内两个版本：窗口末刻取区间末刻 → 选到 11 月版本，不是 3 月版本。"""

    def setUp(self):
        self.item = run_e3.parse_item(make_row(
            'q_year', 'In 2019, Alice lives in _X_?', '2024/01/05 (Fri) 10:00',
            [('2019/03/01 (Fri) 09:00', 'user: Alice lives in Paris.'),
             ('2019/11/01 (Fri) 09:00', 'user: Alice moved to Rome.')],
            answer='Rome', question_type='temporal-reasoning'))

    def test_anchor_end_of_year_selects_november_version(self):
        runtime, written, resolved, result = answer(self.item, recency_update=True)
        self.assertEqual(resolved.window_end, '2019-12-31T23:59:59')
        self.assertFalse(result['no_state_match'])
        self.assertEqual(result['selected_version_time'], '2019-11-01T09:00:00')
        # 版本链确实建立（recency_update=True 才能把两次陈述连成版本链）
        self.assertEqual(sum(1 for entry in written['ledger'] if entry['version_id']), 2)

    def test_mid_year_window_end_selects_march_version(self):
        _, _, _, result = answer(self.item, recency_update=True,
                                 window_end='2019-06-30T23:59:59')
        self.assertEqual(result['selected_version_time'], '2019-03-01T09:00:00')
        self.assertEqual(result['version_role'], 'historical')

    def test_interval_start_selects_nothing(self):
        """反证：若误把"区间起点"当窗口末刻（2019-01-01），两个版本都不可见。"""
        _, _, _, result = answer(self.item, recency_update=True,
                                 window_end='2019-01-01T00:00:00')
        self.assertTrue(result['no_state_match'])
        self.assertIsNone(result['selected_version_id'])


class PerItemNamespaceTest(unittest.TestCase):
    """每题独立 namespace：session_id 重名 / 文本相同都不产生跨题共享。"""

    def test_same_session_ids_are_isolated_per_item(self):
        sessions = [('2021/03/01 (Mon) 09:00', 'user: Alice lives in Lisbon.')]
        item_a = run_e3.parse_item(make_row('q_a', 'Alice lives in _X_?',
                                           '2024/01/05 (Fri) 10:00', sessions, answer='Lisbon'))
        item_b = run_e3.parse_item(make_row('q_b', 'Alice lives in _X_?',
                                           '2024/01/05 (Fri) 10:00', sessions, answer='Lisbon'))
        runtime_a, written_a, _, result_a = answer(item_a)
        runtime_b, written_b, _, result_b = answer(item_b)

        # 两个 namespace 各自从 input_000001 / state_000001 开始编号 → 编号相同但不共享
        self.assertEqual(written_a['ledger'][0]['fact_id'], written_b['ledger'][0]['fact_id'])
        self.assertEqual(len(runtime_a.state_index), 1)
        self.assertEqual(len(runtime_b.state_index), 1)
        self.assertIsNot(runtime_a.state_index, runtime_b.state_index)
        self.assertIsNot(runtime_a.store, runtime_b.store)
        self.assertEqual(result_a['state_id'], result_b['state_id'])
        # 各自只看到自己那份 session
        self.assertEqual(written_a['sessions'], 1)
        self.assertEqual(len(runtime_a.store.entries), len(runtime_b.store.entries))

    def test_cross_item_state_does_not_leak_into_answer(self):
        item_a = run_e3.parse_item(make_row(
            'q_a', 'Alice lives in _X_?', '2024/01/05 (Fri) 10:00',
            [('2021/03/01 (Mon) 09:00', 'user: Alice lives in Lisbon.')], answer='Lisbon'))
        item_b = run_e3.parse_item(make_row(
            'q_b', 'Bob lives in _X_?', '2024/01/05 (Fri) 10:00',
            [('2021/03/01 (Mon) 09:00', 'user: Bob lives in Oslo.')], answer='Oslo'))
        _, written_a, _, _ = answer(item_a)
        runtime_b, written_b, _, result_b = answer(item_b)
        self.assertNotIn('Lisbon', ' '.join(entry['clause'] for entry in written_b['ledger']))
        self.assertEqual(result_b['state_id'], 'state_000001')  # B 自己的第一个状态
        self.assertEqual(len(runtime_b.state_index), 1)


class GradedErrorBucketTest(unittest.TestCase):
    """分级桶判定：上游优先，绝不把上游失败混进 reader_wrong。"""

    def setUp(self):
        self.item = {'answers': ['Rome'], 'abstention': False}

    def classify(self, ledger, result, em=0, item=None):
        return run_e3.classify_error(item or self.item, result, ledger, em)

    def test_buckets(self):
        created = {'fact_id': 'f1', 'state_id': 'state_000001', 'decision': 'CREATE',
                   'version_id': 'f1', 'observed_at': '2019-03-01T00:00:00', 'clause': 'Alice lives in Rome'}
        older = {'fact_id': 'f2', 'state_id': 'state_000001', 'decision': 'CREATE',
                 'version_id': 'f2', 'observed_at': '2018-03-01T00:00:00', 'clause': 'Alice lives in Paris'}
        deferred = {'fact_id': 'f3', 'state_id': None, 'decision': 'DEFER',
                    'version_id': None, 'observed_at': '2019-11-01T00:00:00', 'clause': 'Alice moved to Rome'}
        other_state = {'fact_id': 'f4', 'state_id': 'state_000002', 'decision': 'CREATE',
                       'version_id': 'f4', 'observed_at': '2019-05-01T00:00:00', 'clause': 'Alice lives in Rome'}

        self.assertEqual(self.classify([], {'no_state_match': True}), 'fact_extraction_failed')
        self.assertEqual(self.classify([older], {'no_state_match': True}), 'fact_extraction_failed')
        self.assertEqual(self.classify([deferred], {'no_state_match': True}), 'update_deferred')
        self.assertEqual(self.classify([other_state], {'state_id': 'state_000001',
                                                       'selected_version_id': 'f9'}),
                         'state_match_failed')
        self.assertEqual(self.classify([created], {'no_state_match': True}), 'state_match_failed')
        self.assertEqual(self.classify([created, older],
                                       {'state_id': 'state_000001', 'selected_version_id': 'f2'}),
                         'version_select_failed')
        self.assertEqual(self.classify([created, older],
                                       {'state_id': 'state_000001', 'selected_version_id': 'f1'}),
                         'reader_wrong')
        self.assertEqual(self.classify([older], {'state_id': 'state_000001'}, em=1), 'correct')

    def test_extra_buckets(self):
        self.assertEqual(self.classify([], {}, item={'answers': [], 'abstention': False}), 'no_gold')
        self.assertEqual(self.classify([], {}, item={'answers': ['x'], 'abstention': True}),
                         'abstention_unscored')

    def test_required_buckets_are_declared(self):
        for bucket in ('fact_extraction_failed', 'state_match_failed',
                       'version_select_failed', 'reader_wrong', 'update_deferred'):
            self.assertIn(bucket, run_e3.ALL_BUCKETS)

    def test_contains_answer_is_token_subset(self):
        self.assertTrue(run_e3.contains_answer('I adopted a greyhound named Comet',
                                               ['a greyhound named Comet']))
        self.assertFalse(run_e3.contains_answer('I adopted a greyhound', ['Comet']))
        self.assertFalse(run_e3.contains_answer('', ['Comet']))

    def test_update_source_counts(self):
        counts = run_e3.update_source_counts([
            {'decision': 'UPDATE', 'reason': 'exclusive_superseded_by_recency:residence|cardinality=EXCLUSIVE'},
            {'decision': 'UPDATE', 'reason': 'explicit_replacement:employer|cardinality=EXCLUSIVE'},
            {'decision': 'CREATE', 'reason': 'rule1:no_candidate'}])
        self.assertEqual(counts['exclusive_superseded_by_recency'], 1)
        self.assertEqual(counts['explicit_replacement'], 1)
        self.assertEqual(counts['not_update'], 1)


class EndToEndFiveItemTest(unittest.TestCase):
    """5 题端到端（stub reader）：off/on 两种 recency 模式都要能跑通，且输出可进官方链路。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'lme_s.json'
        rows = [
            make_row('q1', 'Alice lives in _X_?', '2024/01/05 (Fri) 10:00',
                     [('2021/03/01 (Mon) 09:00', 'user: Alice lives in Lisbon.'),
                      ('2023/06/01 (Thu) 12:00', 'user: Alice moved to Berlin.')],
                     answer='Berlin', question_type='single-session-user'),
            make_row('q2', 'In 2019, Alice lives in _X_?', '2024/01/05 (Fri) 10:00',
                     [('2019/05/01 (Wed) 09:00', 'user: Alice lives in Paris.'),
                      ('2021/05/01 (Sat) 09:00', 'user: Alice lives in Rome.')],
                     answer='Paris', question_type='temporal-reasoning'),
            make_row('q3', 'Bob works at _X_?', '2024/01/05 (Fri) 10:00',
                     [('2022/02/01 (Tue) 09:00', 'user: Bob works at Google.'),
                      ('2023/03/01 (Wed) 09:00', 'user: Bob works at Meta.')],
                     answer='Meta', question_type='knowledge-update'),
            make_row('q4', 'Carol is married to _X_?', '2024/01/05 (Fri) 10:00',
                     [('2020/07/01 (Wed) 09:00', 'user: Carol is married to Dan.')],
                     answer='Dan', question_type='single-session-user'),
            make_row('q5', 'Dave lives in _X_?', '2024/01/05 (Fri) 10:00',
                     [('2020/07/01 (Wed) 09:00', 'user: Dave lives in Oslo.')],
                     answer='Oslo', question_type='single-session-user'),
        ]
        self.data.write_text(json.dumps(rows))

    def tearDown(self):
        self.tmp.cleanup()

    def run_pipeline(self, tag, **overrides):
        args = ['--data', str(self.data), '--tag', tag, '--runs-dir', str(self.root / 'runs'),
                '--reader', 'stub', '--no-resume']
        if overrides.get('recency_update'):
            args.append('--recency-update')
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = run_e3.main(args)
        return summary

    def rows(self, tag):
        path = self.root / 'runs' / f'{tag}.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def test_off_mode_builds_no_version_chain_and_reports_update_deferred(self):
        summary = self.run_pipeline('e2e_off')
        rows = {row['question_id']: row for row in self.rows('e2e_off')}

        self.assertEqual(summary['items'], 5)
        self.assertEqual(summary['time_mode'], 'window_end')
        self.assertEqual(summary['recency_update'], False)
        self.assertEqual(summary['time_anchor_distribution'],
                         {'question_text_anchor': 1, 'ask_time': 4})
        # q2 有显式历史锚：窗口末刻取 2019 年末刻，不是提问时刻
        self.assertEqual(rows['q2']['time_anchor'], 'question_text_anchor')
        self.assertEqual(rows['q2']['window_end'], '2019-12-31T23:59:59')
        self.assertEqual(rows['q2']['time_source'], 'window_end')
        self.assertNotEqual(rows['q2']['window_end'], rows['q2']['ask_time'])
        # 默认 recency_update=off：q1/q3 的第二次陈述落 DEFER → update_deferred 单列
        self.assertGreaterEqual(summary['error_buckets']['update_deferred'], 2)
        self.assertEqual(summary['update_source_counts']['exclusive_superseded_by_recency'], 0)
        # 五个桶都在（含 0），且分级桶顺序固定
        for bucket in ('fact_extraction_failed', 'update_deferred', 'state_match_failed',
                       'version_select_failed', 'reader_wrong', 'correct',
                       'abstention_unscored', 'no_gold'):
            self.assertIn(bucket, summary['error_buckets'])
        self.assertEqual(summary['error_bucket_order'][:5], list(run_e3.ERROR_BUCKETS))

    def test_on_mode_builds_version_chain_and_splits_update_sources(self):
        summary = self.run_pipeline('e2e_on', recency_update=True)
        rows = {row['question_id']: row for row in self.rows('e2e_on')}

        self.assertTrue(summary['recency_update'])
        self.assertEqual(summary['hybrid_fallback'], False)   # 分口径开关默认 off
        self.assertEqual(summary['error_buckets']['update_deferred'], 0)
        self.assertGreaterEqual(
            summary['update_source_counts']['exclusive_superseded_by_recency'], 2)
        # q1 的窗口末刻 = 提问时刻 → 最新可见版本（2023-06 Berlin）
        self.assertEqual(rows['q1']['selected_version_time'], '2023-06-01T12:00:00')
        self.assertEqual(rows['q1']['version_role'], 'current')
        # q2 的窗口末刻 = 2019 年末刻 → 2019-05 Paris（历史版本）
        self.assertEqual(rows['q2']['selected_version_time'], '2019-05-01T09:00:00')
        self.assertEqual(rows['q2']['version_role'], 'historical')

    def test_outputs_are_official_chain_compatible(self):
        """真跑既有官方链路：make_official_eval_input.py 能把我们的 runs 转成官方输入。

        注意：``make_official_eval_input.py`` 的 ``--runs-dir`` 只接受默认的 Path
        （命令行传 str 会 TypeError，属 t14 文件的既有缺陷，不在本任务范围内）。
        所以这里把脚本原样复制到临时目录，用它的**默认** runs 目录跑真实链路：
        ``<tmp>/lme_s_text_rag/runs/<tag>/*.json`` → predictions.jsonl。
        """
        import shutil
        tag = 'e2e_chain'
        chain_dir = self.root / 'lme_s_text_rag'
        chain_dir.mkdir(parents=True)
        script = chain_dir / 'make_official_eval_input.py'
        shutil.copy(REPO_ROOT / 'lme_s_text_rag' / 'make_official_eval_input.py', script)
        runs_dir = chain_dir / 'runs'

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_e3.main(['--data', str(self.data), '--tag', tag, '--runs-dir', str(runs_dir),
                         '--reader', 'stub', '--no-resume', '--recency-update'])

        out_dir = runs_dir / tag
        files = sorted(path.name for path in out_dir.glob('*.json'))
        self.assertEqual(files, ['q1.json', 'q2.json', 'q3.json', 'q4.json', 'q5.json'])

        per_qid = json.loads((out_dir / 'q1.json').read_text())
        self.assertEqual(per_qid['question_id'], 'q1')
        self.assertTrue(per_qid['hypothesis'])          # 非空，官方链路不会因空 hypothesis 失败
        self.assertEqual(per_qid['time_mode'], 'window_end')

        predictions = self.root / 'predictions.jsonl'
        completed = subprocess.run(
            [sys.executable, str(script), '--tag', tag, '--expect-n', '5',
             '--out', str(predictions)],
            capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        lines = [json.loads(line) for line in predictions.read_text().splitlines() if line.strip()]
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertEqual(sorted(line), ['hypothesis', 'question_id'])

    def test_no_gold_answer_leaks_into_outputs(self):
        tag = 'e2e_leak'
        self.run_pipeline(tag, recency_update=True)
        blob = (self.root / 'runs' / f'{tag}.jsonl').read_text()
        blob += (self.root / 'runs' / tag / 'q2.json').read_text()
        for gold in ('Paris', 'Lisbon', 'Berlin', 'Meta', 'Oslo', 'Dan'):
            self.assertNotIn(gold, blob)

    def test_resume_skips_finished_questions(self):
        self.run_pipeline('e2e_resume')
        args = ['--data', str(self.data), '--tag', 'e2e_resume',
                '--runs-dir', str(self.root / 'runs'), '--reader', 'stub']
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = run_e3.main(args)
        self.assertEqual(summary['items'], 0)   # 全部命中 resume
        self.assertIn('done=5', buffer.getvalue())


if __name__ == '__main__':
    unittest.main()
