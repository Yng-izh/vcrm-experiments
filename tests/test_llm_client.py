from __future__ import annotations
"""vcrm/llm_client.py 的单测：全部用注入的假 transport，不碰网络、不调真实 API。

覆盖：
  1) 正常返回：请求体字段正确（model / messages / temperature / max_tokens）、Bearer 头正确
  2) 429/5xx 重试后成功（指数退避的 sleep 被注入为 no-op）
  3) 连续 4 次失败 → 抛错（不返回空串）
  4) 缺 key → RuntimeError，且一次请求都不发
  5) 缓存命中不二次发请求；缓存 key 对 prompt 敏感
  6) 非重试的 4xx 直接抛；空 content 抛错且不落缓存
"""
import json
import os
import unittest
import urllib.error

import tempfile
from pathlib import Path

from vcrm.llm_client import (
    CachedLLMClient,
    OpenAICompatClient,
    build_llm_client,
    load_config_file,
    normalize_config,
)

CONFIG = {'model': 'test-model', 'base_url': 'https://example.invalid/v1',
          'key_env': 'VCRM_TEST_LLM_KEY', 'max_tokens': 64, 'temperature': 0}


def ok_payload(text='{"facts": []}'):
    return {'choices': [{'message': {'role': 'assistant', 'content': text}}]}


def http_error(code, url='https://example.invalid/v1/chat/completions'):
    return urllib.error.HTTPError(url, code, f'status {code}', {}, None)


class FakeTransport:
    """按顺序吐出预设结果的假 transport；记录每次请求。"""

    def __init__(self, *results):
        self.results = list(results)
        self.requests = []
        self.bodies = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        self.bodies.append(json.loads(request.data.decode()))
        if not self.results:
            raise AssertionError('unexpected extra request')
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class ClientTestCase(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.get(CONFIG['key_env'])
        os.environ[CONFIG['key_env']] = 'sk-test-123'
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_root = Path(self._tmp.name)

    def tearDown(self):
        if self._env is None:
            os.environ.pop(CONFIG['key_env'], None)
        else:
            os.environ[CONFIG['key_env']] = self._env
        self._tmp.cleanup()

    def client(self, transport, **overrides):
        config = dict(CONFIG, **overrides)
        return OpenAICompatClient(config, transport=transport, sleep=lambda _seconds: None)


class CompleteTest(ClientTestCase):
    def test_returns_content_and_builds_request(self):
        transport = FakeTransport(ok_payload('hello'))
        client = self.client(transport)
        self.assertEqual(client.complete('extract this', system='be terse', max_tokens=32), 'hello')
        request, body = transport.requests[0], transport.bodies[0]
        self.assertTrue(request.full_url.endswith('/chat/completions'))
        self.assertEqual(request.get_header('Authorization'), 'Bearer sk-test-123')
        self.assertEqual(body['model'], 'test-model')
        self.assertEqual(body['max_tokens'], 32)
        self.assertEqual(body['temperature'], 0.0)
        self.assertEqual([m['role'] for m in body['messages']], ['system', 'user'])
        self.assertEqual(body['messages'][1]['content'], 'extract this')
        self.assertEqual(client.requests, 1)

    def test_retries_then_succeeds(self):
        transport = FakeTransport(http_error(429), http_error(503), ok_payload('after-retry'))
        client = self.client(transport)
        self.assertEqual(client.complete('p'), 'after-retry')
        self.assertEqual(len(transport.requests), 3)

    def test_four_failures_raise(self):
        transport = FakeTransport(http_error(429), http_error(500), http_error(502), http_error(504))
        client = self.client(transport)
        with self.assertRaises(urllib.error.HTTPError):
            client.complete('p')
        self.assertEqual(len(transport.requests), 4)     # 4 次尝试后不再重试

    def test_non_retryable_status_raises_immediately(self):
        transport = FakeTransport(http_error(400))
        client = self.client(transport)
        with self.assertRaises(urllib.error.HTTPError):
            client.complete('p')
        self.assertEqual(len(transport.requests), 1)

    def test_network_error_raises_after_retries(self):
        transport = FakeTransport(*[urllib.error.URLError('boom')] * 4)
        client = self.client(transport)
        with self.assertRaises(urllib.error.URLError):
            client.complete('p')
        self.assertEqual(len(transport.requests), 4)

    def test_missing_key_raises_without_request(self):
        os.environ.pop(CONFIG['key_env'], None)
        transport = FakeTransport()
        client = self.client(transport)
        with self.assertRaises(RuntimeError) as ctx:
            client.complete('p')
        self.assertIn(CONFIG['key_env'], str(ctx.exception))
        self.assertEqual(transport.requests, [])

    def test_empty_content_raises_not_empty_string(self):
        for payload in (ok_payload('   '), ok_payload(''),
                        {'choices': []}, {'unexpected': True}):
            transport = FakeTransport(payload)
            client = self.client(transport)
            with self.assertRaises((RuntimeError, ValueError)):
                client.complete('p')


class CacheTest(ClientTestCase):
    def test_cache_hit_skips_second_request(self):
        transport = FakeTransport(ok_payload('cached-value'))
        client = CachedLLMClient(self.client(transport), self.cache_root)
        self.assertEqual(client.complete('prompt A', system='sys', max_tokens=16), 'cached-value')
        self.assertEqual(client.complete('prompt A', system='sys', max_tokens=16), 'cached-value')
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual((client.hits, client.misses), (1, 1))
        self.assertEqual(len(list(self.cache_root.glob('*.json'))), 1)
        self.assertEqual(client.stats()['model'], 'test-model')

    def test_cache_key_is_prompt_sensitive(self):
        transport = FakeTransport(ok_payload('one'), ok_payload('two'), ok_payload('three'))
        client = CachedLLMClient(self.client(transport), self.cache_root)
        self.assertEqual(client.complete('prompt A'), 'one')
        self.assertEqual(client.complete('prompt B'), 'two')          # prompt 变了 → 未命中
        self.assertEqual(client.complete('prompt A', system='x'), 'three')  # system 变了 → 未命中
        self.assertEqual(len(transport.requests), 3)
        self.assertEqual(len(list(self.cache_root.glob('*.json'))), 3)
        # 再取一遍全部命中
        self.assertEqual(client.complete('prompt A'), 'one')
        self.assertEqual(len(transport.requests), 3)

    def test_empty_response_is_not_cached(self):
        transport = FakeTransport(ok_payload('   '))
        client = CachedLLMClient(self.client(transport), self.cache_root)
        with self.assertRaises(RuntimeError):
            client.complete('prompt')
        self.assertEqual(list(self.cache_root.glob('*.json')), [])


class ConfigTest(ClientTestCase):
    def test_config_aliases_and_validation(self):
        normalized = normalize_config({'judge_model': 'm', 'judge_base_url': 'https://x/v1',
                                       'judge_key_env': 'K'})
        self.assertEqual((normalized['model'], normalized['base_url'], normalized['key_env']),
                         ('m', 'https://x/v1', 'K'))
        self.assertEqual(normalized['max_tokens'], 512)      # 默认值
        self.assertEqual(normalized['temperature'], 0.0)
        with self.assertRaises(ValueError):
            normalize_config({'model': 'm'})

    def test_build_client_from_file(self):
        path = self.cache_root / 'llm.json'
        path.write_text(json.dumps(CONFIG))
        self.assertEqual(load_config_file(path)['model'], 'test-model')
        transport = FakeTransport(ok_payload('from-file'))
        client = build_llm_client(str(path), cache_root=self.cache_root / 'cache', transport=transport)
        self.assertIsInstance(client, CachedLLMClient)
        self.assertEqual(client.complete('p'), 'from-file')
        with self.assertRaises(ValueError):
            build_llm_client(str(self.cache_root / 'missing.json'))


if __name__ == '__main__':
    unittest.main()
