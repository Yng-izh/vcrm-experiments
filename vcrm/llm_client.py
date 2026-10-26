from __future__ import annotations
"""OpenAI 兼容 LLM 客户端（``complete()`` 实现）+ 磁盘缓存。

为什么需要这一层：

    ``vcrm/llm_fact_extractor.py`` 只冻结了 ``LLMClient`` Protocol
    （``complete(prompt, *, system=None, max_tokens=512) -> str``），
    ``PlaceholderLLM`` 任何调用都显式抛错。实验二的 B 臂要真的跑起来，
    就需要一个实现了该 Protocol 的可用客户端——但不能把网络细节散落到
    抽取器/解析器里，也不能让"模型挂了"悄悄退化成规则抽取。

失败语义（不可妥协）：

    * 缺 ``key_env`` 对应的环境变量 → ``RuntimeError``，不做无 key 尝试；
    * 网络错误 / 非重试的 HTTP 错误 → 原样向上抛；
    * 响应格式不对、``content`` 为空或纯空白 → 抛错，**绝不返回空串**
      （空串会被下游当成"模型说这里没有事实"，等于静默降级）。

重试策略与 ``vcrm/judge.py`` 完全一致：429/500/502/503/504 指数退避，
共 4 次尝试（``2 ** attempt`` 秒）。

缓存（B/C/D 三组共享前端输出的关键）：

    ``CachedLLMClient`` 按 ``(model, system, prompt, max_tokens)`` 哈希落盘 JSON，
    命中即直接返回，不发请求。三组实验只要指向同一个 ``cache_root``，就能保证
    看到的是**同一份**模型输出（可审计、可复现，且不重复烧钱）。

配置格式（与 judge 的 ``{judge_model, judge_base_url, judge_key_env}`` 同构，
三个键名都接受 ``model`` / ``llm_model`` / ``judge_model`` 这类别名）::

    {"model": "gpt-4o-mini",
     "base_url": "https://api.openai.com/v1",
     "key_env": "OPENAI_API_KEY",
     "max_tokens": 512,
     "temperature": 0}

单测不碰网络：``OpenAICompatClient(config, transport=fake)`` 里的 transport 是
可注入的 ``callable(request, timeout=...) -> dict``，直接返回伪造的响应体。
"""
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, Mapping, Optional

import urllib.error
import urllib.request

from .io import digest, read_json, write_json

#: 与 vcrm/judge.py 同一套重试口径：只对限流/服务端错误重试。
RETRY_STATUSES = (429, 500, 502, 503, 504)
MAX_ATTEMPTS = 4
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT = 90

_MODEL_KEYS = ('model', 'llm_model', 'judge_model')
_BASE_URL_KEYS = ('base_url', 'llm_base_url', 'judge_base_url')
_KEY_ENV_KEYS = ('key_env', 'llm_key_env', 'judge_key_env')


def normalize_config(config: Mapping[str, Any], *, source: str = 'llm config') -> Dict[str, Any]:
    """把配置规整成 ``{model, base_url, key_env, max_tokens, temperature}``。

    三个必填键都接受 judge 同款别名，缺任何一个都显式报错（不做默认值猜测：
    默认 base_url / 默认模型会让实验口径变得不可复现）。
    """
    if not isinstance(config, Mapping):
        raise ValueError(f'{source} must be a mapping, got {type(config).__name__}')

    def pick(names):
        for name in names:
            value = config.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if value not in (None, '') and not isinstance(value, str):
                return value
        return None

    model, base_url, key_env = pick(_MODEL_KEYS), pick(_BASE_URL_KEYS), pick(_KEY_ENV_KEYS)
    missing = [name for name, value in (('model', model), ('base_url', base_url),
                                        ('key_env', key_env)) if not value]
    if missing:
        raise ValueError(f'{source} missing required field(s): {missing} '
                         f'(accepted aliases: {list(_MODEL_KEYS)} / {list(_BASE_URL_KEYS)} / '
                         f'{list(_KEY_ENV_KEYS)})')
    max_tokens = config.get('max_tokens')
    temperature = config.get('temperature')
    return {'model': str(model),
            'base_url': str(base_url),
            'key_env': str(key_env),
            'max_tokens': int(max_tokens) if max_tokens is not None else DEFAULT_MAX_TOKENS,
            'temperature': float(temperature) if temperature is not None else 0.0}


def load_config_file(path: Any) -> Dict[str, Any]:
    """读 JSON 配置文件；文件缺失/字段缺失都显式抛错。"""
    target = Path(path)
    if not target.exists():
        raise ValueError(f'llm config not found: {target}')
    return normalize_config(read_json(target), source=f'llm config {target}')


class OpenAICompatClient:
    """最小 OpenAI 兼容客户端：只实现 ``complete()``。"""

    def __init__(self, config: Mapping[str, Any], *, transport: Optional[Callable[..., Any]] = None,
                 timeout: float = DEFAULT_TIMEOUT, sleep: Callable[[float], Any] = time.sleep) -> None:
        self.config = normalize_config(config)
        self.model = self.config['model']
        self.base_url = self.config['base_url']
        self.key_env = self.config['key_env']
        self.timeout = timeout
        self._transport = transport
        self._sleep = sleep
        self.requests = 0        # 诊断用：真实发出的请求数（缓存命中时不应增长）

    # -- 请求构造 ---------------------------------------------------------
    def api_key(self) -> str:
        key = os.environ.get(self.key_env)
        if not key:
            raise RuntimeError(
                f'LLM credentials missing: environment variable {self.key_env!r} is not set; '
                'refusing to run without a key (no silent fallback)')
        return key

    def payload(self, prompt: str, *, system: Optional[str] = None,
                max_tokens: Optional[int] = None) -> Dict[str, Any]:
        messages = []
        if system:
            messages.append({'role': 'system', 'content': str(system)})
        messages.append({'role': 'user', 'content': str(prompt)})
        return {'model': self.model, 'messages': messages,
                'temperature': self.config['temperature'],
                'max_tokens': int(max_tokens or self.config['max_tokens'])}

    def request(self, prompt: str, *, system: Optional[str] = None,
                max_tokens: Optional[int] = None) -> urllib.request.Request:
        body = json.dumps(self.payload(prompt, system=system, max_tokens=max_tokens)).encode()
        return urllib.request.Request(
            self.base_url.rstrip('/') + '/chat/completions', data=body,
            headers={'Authorization': 'Bearer ' + self.api_key(),
                     'Content-Type': 'application/json'})

    # -- 调用 -------------------------------------------------------------
    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: Optional[int] = None) -> str:
        request = self.request(prompt, system=system, max_tokens=max_tokens)
        for attempt in range(MAX_ATTEMPTS):
            try:
                self.requests += 1
                return self._content(self._send(request))
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS - 1:
                    raise
            except (TimeoutError, urllib.error.URLError):
                if attempt == MAX_ATTEMPTS - 1:
                    raise
            self._sleep(2 ** attempt)
        raise RuntimeError('unreachable: retry loop exited without result')   # pragma: no cover

    def _send(self, request: urllib.request.Request) -> Any:
        if self._transport is not None:
            return self._transport(request, timeout=self.timeout)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    @staticmethod
    def _content(payload: Any) -> str:
        try:
            text = payload['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f'malformed chat/completions payload: {exc}') from exc
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError('LLM returned empty content; refusing to substitute an empty string')
        return text


class CachedLLMClient:
    """按 ``(model, system, prompt, max_tokens)`` 落盘缓存的 ``complete()`` 包装。"""

    def __init__(self, inner: Any, cache_root: Any, *, model: Optional[str] = None) -> None:
        self.inner = inner
        self.cache_root = Path(cache_root)
        self.model = str(model if model is not None else getattr(inner, 'model', 'unknown'))
        self.hits = 0
        self.misses = 0

    def identity(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: Optional[int] = None) -> Dict[str, Any]:
        return {'model': self.model,
                'system': None if system is None else str(system),
                'prompt': str(prompt),
                'max_tokens': int(max_tokens or DEFAULT_MAX_TOKENS)}

    def cache_path(self, prompt: str, *, system: Optional[str] = None,
                   max_tokens: Optional[int] = None) -> Path:
        return self.cache_root / (digest(self.identity(prompt, system=system,
                                                       max_tokens=max_tokens)) + '.json')

    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: Optional[int] = None) -> str:
        path = self.cache_path(prompt, system=system, max_tokens=max_tokens)
        if path.exists():
            self.hits += 1
            return read_json(path)['response']
        text = self.inner.complete(prompt, system=system, max_tokens=max_tokens)
        if not isinstance(text, str) or not text.strip():
            # 空输出不落缓存：宁可下次重试，也不要把"空"固化成可审计的假证据。
            raise RuntimeError('LLM client returned empty content; not cached')
        self.misses += 1
        write_json(path, self.identity(prompt, system=system, max_tokens=max_tokens)
                   | {'response': text})
        return text

    def stats(self) -> Dict[str, Any]:
        return {'model': self.model, 'cache_root': str(self.cache_root),
                'hits': self.hits, 'misses': self.misses}


def build_llm_client(config: Any, *, cache_root: Optional[Any] = None,
                     transport: Optional[Callable[..., Any]] = None) -> CachedLLMClient:
    """``config`` 可以是配置字典或 JSON 路径；默认套一层磁盘缓存。"""
    payload = load_config_file(config) if isinstance(config, (str, Path)) else normalize_config(config)
    inner = OpenAICompatClient(payload, transport=transport)
    if cache_root is None:
        return CachedLLMClient(inner, Path('llm_cache'))
    return CachedLLMClient(inner, Path(cache_root))
