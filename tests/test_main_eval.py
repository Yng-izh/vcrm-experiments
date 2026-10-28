"""统一评测入口 probe/run_main_eval.py 的 CPU 单测（stub reader / stub answerer）。

钉死四组（A text_rag / B state_text / C current_only / D versioned）的协议：
* predictions.jsonl 字段齐全、顺序与 PREDICTION_FIELDS 一致；
* 全题分母（失败行 answer="" + error 也在文件里，绝不缩水）；
* 四组 question_id 完全一致（同一批数据）；
* C/D 只差 read_mode，write_bytes 必须逐题相等（同一写入路径）；
* --frontend llm 缺 --llm-config 时显式报错；
* t25 的四个 query_time 审计字段逐题落盘，as_of_future_fallback 恒 False。

stdlib unittest only；不碰 GPU/vLLM/网络，不需要 pytest。
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location('run_main_eval', REPO_ROOT / 'probe' / 'run_main_eval.py')
main_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(main_eval)


def make_row(question_id, question, sessions, answer='', question_type='single-session-user',
             question_date='2024/01/05 (Fri) 10:00'):
    """官方 cleaned schema 行（t22 严格解析：三个数组对齐、session 必须是 turn 列表）。"""
    return {'question_id': question_id, 'question': question,
            'question_date': question_date, 'question_type': question_type,
            'answer': answer,
            'haystack_session_ids': [f'{question_id}_s{i}' for i in range(len(sessions))],
            'haystack_dates': [date for date, _ in sessions],
            'haystack_sessions': [[{'role': 'user', 'content': text}] for _, text in sessions]}


class MainEvalProtocolTest(unittest.TestCase):
    """四组在同一 fixture 上的协议一致性。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'lme_s.json'
        self.data.write_text(json.dumps([
            make_row('q1', 'Alice lives in _X_?',
                     [('2021/03/01 (Mon) 09:00', 'Alice lives in Lisbon.'),
                      ('2023/06/01 (Thu) 12:00', 'Alice moved to Berlin.')], answer='Berlin'),
            make_row('q2', 'In 2019, Alice lives in _X_?',
                     [('2019/05/01 (Wed) 09:00', 'Alice lives in Paris.'),
                      ('2021/05/01 (Sat) 09:00', 'Alice lives in Rome.')],
                     answer='Paris', question_type='temporal-reasoning'),
            make_row('q3', 'Bob works at _X_?',
                     [('2022/02/01 (Tue) 09:00', 'Bob works at Google.'),
                      ('2023/03/01 (Wed) 09:00', 'Bob works at Meta.')],
                     answer='Meta', question_type='knowledge-update'),
        ]))

    def tearDown(self):
        self.tmp.cleanup()

    def run_group(self, group, extra=()):
        argv = ['--group', group, '--data', str(self.data), '--tag', 'unit',
                '--runs-root', str(self.root / 'runs'), '--reader', 'stub'] + list(extra)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = main_eval.main(argv)
        path = self.root / 'runs' / 'unit' / group / 'predictions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return summary, rows

    def test_all_groups_emit_contract_fields(self):
        for group in main_eval.GROUPS:
            summary, rows = self.run_group(group)
            self.assertEqual(len(rows), 3, group)
            self.assertEqual(summary['items'], 3, group)
            for row in rows:
                for field in main_eval.PREDICTION_FIELDS:
                    self.assertIn(field, row, f'{group}: missing {field}')
                self.assertEqual(row['group'], group)
                self.assertEqual(row['n_sessions'], 2)
                self.assertGreater(row['write_bytes'], 0, group)
                self.assertFalse(row['config']['as_of_future_fallback'], group)
                self.assertEqual(row['query_cutoff_source'], 'question_date')
                self.assertEqual(row['query_time_source'], 'window_end')

    def test_question_ids_identical_across_groups(self):
        sets = {}
        for group in main_eval.GROUPS:
            _, rows = self.run_group(group)
            sets[group] = [row['question_id'] for row in rows]
        first = sets['text_rag']
        for group, ids in sets.items():
            self.assertEqual(ids, first, group)

    def test_current_only_and_versioned_share_write_bytes(self):
        _, current = self.run_group('current_only')
        _, versioned = self.run_group('versioned')
        left = {row['question_id']: row['write_bytes'] for row in current}
        right = {row['question_id']: row['write_bytes'] for row in versioned}
        self.assertEqual(left, right)          # 唯一变量是 read_mode，写入路径相同
        # T32-F7：GPU 上的诚实锚点 = 决策计数与状态数（字节数会因编码非确定性抖动）
        self.assertEqual({row['question_id']: row['decision_counts'] for row in current},
                         {row['question_id']: row['decision_counts'] for row in versioned})
        self.assertEqual({row['question_id']: row['state_count'] for row in current},
                         {row['question_id']: row['state_count'] for row in versioned})
        self.assertTrue(all(row['state_count'] > 0 for row in current))
        self.assertEqual({row['config']['read_mode'] for row in current}, {'current_only'})
        self.assertEqual({row['config']['read_mode'] for row in versioned}, {'trajectory'})

    def test_read_slots_and_input_tokens_are_group_specific(self):
        _, latent = self.run_group('versioned')
        _, text = self.run_group('text_rag')
        self.assertTrue(all(row['read_slots'] and row['read_slots'] > 0 for row in latent))
        self.assertTrue(all(row['input_tokens'] is None for row in latent))
        self.assertTrue(all(row['read_slots'] is None for row in text))
        self.assertTrue(all(row['input_tokens'] > 0 for row in text))

    def test_window_end_uses_historical_anchor(self):
        _, rows = self.run_group('versioned')
        anchored = {row['question_id']: row for row in rows}['q2']
        self.assertEqual(anchored['query_time_anchor'], 'question_text_anchor')
        self.assertEqual(anchored['query_window_end'], '2019-12-31T23:59:59')
        asked = {row['question_id']: row for row in rows}['q1']
        self.assertEqual(asked['query_time_anchor'], 'ask_time')
        self.assertNotEqual(asked['query_window_end'], asked['query_time_anchor'])

    def test_state_text_group_selects_state_without_latent_read(self):
        summary, rows = self.run_group('state_text')
        self.assertEqual(summary['read_mode'], 'pair')
        self.assertTrue(all(row['read_slots'] is None for row in rows))
        self.assertTrue(all(row['input_tokens'] > 0 for row in rows))
        self.assertTrue(any(row['selected_version_id'] for row in rows))


class DenominatorTest(unittest.TestCase):
    """失败题也进分母：answer="" + error，绝不从文件里消失。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'lme_s.json'
        self.data.write_text(json.dumps([
            make_row('q1', 'Alice lives in _X_?',
                     [('2021/03/01 (Mon) 09:00', 'Alice lives in Lisbon.')], answer='Lisbon'),
            make_row('q2', 'Bob works at _X_?',
                     [('2022/02/01 (Tue) 09:00', 'Bob works at Google.')], answer='Google'),
        ]))

    def tearDown(self):
        self.tmp.cleanup()

    def test_failure_keeps_row_with_empty_answer_and_error(self):
        original = main_eval.run_latent_group

        def boom(args, episode, *rest, **kwargs):
            if episode.question_id == 'q2':
                raise RuntimeError('synthetic failure')
            return original(args, episode, *rest, **kwargs)

        main_eval.run_latent_group = boom
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                summary = main_eval.main(['--group', 'versioned', '--data', str(self.data),
                                          '--tag', 'unit', '--runs-root',
                                          str(self.root / 'runs'), '--reader', 'stub'])
        finally:
            main_eval.run_latent_group = original

        path = self.root / 'runs' / 'unit' / 'versioned' / 'predictions.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 2)                      # 分母不缩水
        self.assertEqual(summary['items'], 2)
        self.assertEqual(summary['failed'], 1)
        failed = [row for row in rows if row['question_id'] == 'q2'][0]
        self.assertEqual(failed['answer'], '')
        self.assertIn('synthetic failure', failed['error'])
        self.assertEqual(failed['write_bytes'], 0)

    def test_limit_and_shard_keep_deterministic_subset(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = main_eval.main(['--group', 'text_rag', '--data', str(self.data),
                                      '--tag', 'shard', '--runs-root',
                                      str(self.root / 'runs'), '--limit', '2', '--shard', '1',
                                      '--shards', '2'])
        self.assertEqual(summary['items'], 1)               # [q1, q2][1::2] = [q2]


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class VLLMTextAnswererAuthTest(unittest.TestCase):
    """t32-F6：/v1/completions 必须带 Authorization: Bearer <key>，缺 key 显式报错。"""

    KEY_ENV = 'VCRM_TEST_VLLM_KEY'

    def setUp(self):
        self._saved = os.environ.get(self.KEY_ENV)
        self._original = main_eval.urllib.request.urlopen
        self.requests = []

    def tearDown(self):
        main_eval.urllib.request.urlopen = self._original
        if self._saved is None:
            os.environ.pop(self.KEY_ENV, None)
        else:
            os.environ[self.KEY_ENV] = self._saved

    def _install_fake(self, text=' Paris'):
        def fake_urlopen(request, timeout=None):
            self.requests.append(request)
            return _FakeResponse(json.dumps({'choices': [{'text': text}],
                                             'usage': {'prompt_tokens': 7}}).encode())
        main_eval.urllib.request.urlopen = fake_urlopen

    def test_authorization_header_is_injected(self):
        os.environ[self.KEY_ENV] = 'sk-test-123'
        self._install_fake()
        answerer = main_eval.VLLMTextAnswerer('http://host:6012/v1/completions', 'Qwen3.5-9B',
                                              key_env=self.KEY_ENV)
        result = answerer.answer('The capital of France is')
        self.assertEqual(result['answer'], 'Paris')
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].get_header('Authorization'), 'Bearer sk-test-123')
        self.assertEqual(self.requests[0].get_header('Content-type'), 'application/json')

    def test_missing_key_raises_without_request(self):
        os.environ.pop(self.KEY_ENV, None)
        self._install_fake()
        answerer = main_eval.VLLMTextAnswerer('http://host:6012/v1/completions', 'm',
                                              key_env=self.KEY_ENV)
        with self.assertRaises(RuntimeError) as caught:
            answerer.answer('q')
        self.assertIn(self.KEY_ENV, str(caught.exception))
        self.assertEqual(self.requests, [])      # 缺 key 不重试、不发请求

    def test_empty_key_env_disables_auth(self):
        os.environ.pop(self.KEY_ENV, None)
        self._install_fake()
        answerer = main_eval.VLLMTextAnswerer('http://localhost:8001/v1/completions', 'm',
                                              key_env=None)
        answerer.answer('q')
        self.assertIsNone(self.requests[0].get_header('Authorization'))

    def test_cli_default_and_factory_pass_through(self):
        args = main_eval.parse_args(['--group', 'text_rag', '--data', 'x.json',
                                     '--text-reader', 'vllm'])
        self.assertEqual(args.vllm_key_env, 'VCRM_LLM_KEY')
        self.assertEqual(main_eval.build_answerer(args).key_env, 'VCRM_LLM_KEY')
        custom = main_eval.parse_args(['--group', 'text_rag', '--data', 'x.json',
                                       '--text-reader', 'vllm', '--vllm-key-env', 'OTHER_KEY'])
        self.assertEqual(main_eval.build_answerer(custom).key_env, 'OTHER_KEY')


class FrontendValidationTest(unittest.TestCase):
    def test_llm_frontend_without_config_fails_loudly(self):
        with self.assertRaises(SystemExit) as caught:
            main_eval.parse_args(['--group', 'versioned', '--data', 'x.json',
                                  '--frontend', 'llm'])
        self.assertNotEqual(caught.exception.code, 0)

    def test_llm_frontend_with_config_is_accepted(self):
        args = main_eval.parse_args(['--group', 'versioned', '--data', 'x.json',
                                     '--frontend', 'llm', '--llm-config', 'cfg.json'])
        self.assertEqual(args.frontend, 'llm')

    def test_shared_llm_cache_root_defaults_under_tag(self):
        args = main_eval.parse_args(['--group', 'current_only', '--data', 'x.json',
                                     '--tag', 'main_v1', '--runs-root', '/tmp/runs'])
        self.assertEqual(args.llm_cache_root, '/tmp/runs/main_v1/llm_cache')
        # B/C/D 用同一个默认缓存目录 → 前端输出不因组而异
        other = main_eval.parse_args(['--group', 'versioned', '--data', 'x.json',
                                      '--tag', 'main_v1', '--runs-root', '/tmp/runs'])
        self.assertEqual(args.llm_cache_root, other.llm_cache_root)

    def test_group_choices_are_frozen(self):
        self.assertEqual(main_eval.GROUPS, ('text_rag', 'state_text',
                                            'current_only', 'versioned'))
        self.assertEqual(main_eval.GROUP_READ_MODE['current_only'], 'current_only')
        self.assertEqual(main_eval.GROUP_READ_MODE['versioned'], 'trajectory')
        with self.assertRaises(SystemExit):
            main_eval.parse_args(['--group', 'nope', '--data', 'x.json'])


class WriteBytesCaliberTest(unittest.TestCase):
    """T32-F7：字节口径的可判定性必须显式记账，不能默认"跨组相等"。"""

    def test_summary_declares_not_comparable(self):
        self.assertIn('not_comparable', main_eval.WRITE_BYTES_COMPARE_NOTE)
        self.assertEqual(main_eval.COMPARABLE_ANCHORS, ('decision_counts', 'state_count'))

    def test_compare_is_deterministic_only_when_declared(self):
        left = [{'question_id': 'q1', 'write_bytes': 10, 'decision_counts': {'CREATE': 1},
                 'state_count': 1}]
        right = [{'question_id': 'q1', 'write_bytes': 10, 'decision_counts': {'CREATE': 1},
                  'state_count': 1}]
        strict = main_eval.compare_write_bytes(left, right)
        self.assertTrue(strict['comparable'])
        self.assertTrue(strict['equal'])
        loose = main_eval.compare_write_bytes(left, right, same_machine_and_order=False)
        self.assertFalse(loose['comparable'])                  # 不假装能判字节
        self.assertIsNone(loose['equal'])
        self.assertEqual(loose['anchors'],
                         {'decision_counts_equal': True, 'state_count_equal': True})

    def test_compare_reports_divergent_bytes_under_deterministic_settings(self):
        left = [{'question_id': 'q1', 'write_bytes': 10, 'decision_counts': {'CREATE': 1},
                 'state_count': 1}]
        right = [{'question_id': 'q1', 'write_bytes': 11, 'decision_counts': {'CREATE': 1},
                  'state_count': 1}]
        verdict = main_eval.compare_write_bytes(left, right)
        self.assertTrue(verdict['comparable'])
        self.assertFalse(verdict['equal'])
        self.assertIn('divergent', verdict['reason'])

    def test_compare_rejects_mismatched_question_sets(self):
        verdict = main_eval.compare_write_bytes([{'question_id': 'q1', 'write_bytes': 1}],
                                                [{'question_id': 'q2', 'write_bytes': 1}])
        self.assertFalse(verdict['comparable'])
        self.assertIn('question_id sets differ', verdict['reason'])


class ParametricOnlyTest(unittest.TestCase):
    """T32-F2：没有状态命中时，文本组不得用空上下文喂模型（那是参数化幻觉）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / 'lme_s.json'
        self.data.write_text(json.dumps([
            make_row('q1', 'What did I buy last year?',
                     [('2021/03/01 (Mon) 09:00', 'Alice lives in Lisbon.')], answer='a car'),
        ]))

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_state_match_means_empty_answer_not_hallucination(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = main_eval.main(['--group', 'state_text', '--data', str(self.data),
                                      '--tag', 'para', '--runs-root',
                                      str(self.root / 'runs'), '--reader', 'stub'])
        rows = [json.loads(line) for line in
                (self.root / 'runs' / 'para' / 'state_text' / 'predictions.jsonl'
                 ).read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row['no_state_match'])
        self.assertEqual(row['answer'], '')          # 不调 reader，不留幻觉
        self.assertEqual(row['input_tokens'], 0)
        self.assertTrue(row['parametric_only'])
        self.assertEqual(summary['parametric_only'], 1)
        self.assertEqual(summary['nonempty_answer'], 0)
        # T32-F7：GPU 可用的跨组锚点在 summary 里必须可直接对比
        self.assertIn('CREATE', summary['decision_counts'])
        self.assertGreater(summary['state_count_total'], 0)

    def test_summary_reports_cache_and_nonempty(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = main_eval.main(['--group', 'text_rag', '--data', str(self.data),
                                      '--tag', 'para2', '--runs-root',
                                      str(self.root / 'runs')])
        self.assertEqual(summary['nonempty_answer'], 1)
        self.assertIn('llm_cache', summary)
        self.assertIsNone(summary['llm_cache'])      # rules 前端没有缓存客户端
        # T32-F7：字节口径与 GPU 可用锚点
        self.assertEqual(summary['write_bytes_equal_across_groups'], 'not_comparable')
        self.assertIn('not_comparable', summary['write_bytes_compare_reason'])
        self.assertEqual(summary['cross_group_anchors'], ['decision_counts', 'state_count'])


class CheckpointRequirementTest(unittest.TestCase):
    """T32-F5：icae 跑必须给 --checkpoint，否则 sha256 只能是 none。"""

    def test_icae_requires_checkpoint(self):
        with self.assertRaises(SystemExit) as caught:
            main_eval.parse_args(['--group', 'versioned', '--data', 'x.json',
                                  '--reader', 'icae'])
        self.assertNotEqual(caught.exception.code, 0)

    def test_icae_with_checkpoint_is_accepted(self):
        args = main_eval.parse_args(['--group', 'versioned', '--data', 'x.json',
                                     '--reader', 'icae', '--checkpoint', '/tmp/ckpt'])
        self.assertEqual(args.checkpoint, '/tmp/ckpt')


class JudgeWiringTest(unittest.TestCase):
    """判分二段式接线：judge 模块可导入、行代理结构符合 judge_row 契约（不真调模型）。"""

    def test_judge_module_is_reusable(self):
        spec = importlib.util.spec_from_file_location(
            'judge_rows_probe', REPO_ROOT / 'probe' / 'judge_dialogue_rows.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.MODES, ('pair', 'trajectory'))
        self.assertTrue(callable(module.judge_row))

    def test_defer_judge_is_default(self):
        args = main_eval.parse_args(['--group', 'text_rag', '--data', 'x.json'])
        self.assertTrue(args.defer_judge)
        self.assertIsNone(args.judge_config)


if __name__ == '__main__':
    unittest.main()
