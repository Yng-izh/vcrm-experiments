"""并发抽取改造（captain 直接执行，lme-engineer 网关 503 不可用）。

1) vcrm/llm_client.py: CachedLLMClient 线程安全（缓存读/写各自临界区 + 计数原子，
   远端调用在锁外；并发同 key 时第二个线程直接命中刚写入的缓存）。
2) probe/run_main_eval.py: write_episode 增 extract_workers（默认 1=旧路径逐比特不变）；
   >1 时每 session 的 extract_detailed 并发预取，write_predicted 接受 precomputed
   跳过内部抽取；store 写入/路由/编码仍按 session date 升序严格串行 → 写入语义
   与串行版逐比特一致（facts 只由 session 文本决定，抽取顺序无关）。
3) vcrm/runtime.py: write_predicted 增 precomputed 参数（None=内部抽取，旧路径）。
4) CLI: --extract-workers（默认 1）→ build_components 后传给 write_episode。
"""
import re
from pathlib import Path

ROOT = Path('/share/latent2/vcrm_icae')

# ---------- 1) llm_client.py ----------
p = ROOT / 'vcrm/llm_client.py'
src = p.read_text()
if 'import threading' not in src:
    src = src.replace('import time', 'import threading\nimport time', 1)
assert 'import threading' in src, 'threading import missing'

if '_cache_lock' not in src:
    old_init = """        self.hits = 0
        self.misses = 0"""
    new_init = """        self.hits = 0
        self.misses = 0
        self._cache_lock = threading.Lock()  # 并发抽取：缓存读写与计数原子化"""
    assert old_init in src, 'init anchor not found'
    src = src.replace(old_init, new_init, 1)

    old_complete = """    def complete(self, prompt: str, *, system: Optional[str] = None,
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
        return text"""
    new_complete = """    def complete(self, prompt: str, *, system: Optional[str] = None,
                 max_tokens: Optional[int] = None) -> str:
        path = self.cache_path(prompt, system=system, max_tokens=max_tokens)
        with self._cache_lock:                       # 读缓存临界区
            if path.exists():
                self.hits += 1
                return read_json(path)['response']
        text = self.inner.complete(prompt, system=system, max_tokens=max_tokens)  # 锁外远端调用
        if not isinstance(text, str) or not text.strip():
            # 空输出不落缓存：宁可下次重试，也不要把"空"固化成可审计的假证据。
            raise RuntimeError('LLM client returned empty content; not cached')
        with self._cache_lock:                       # 写缓存临界区
            if path.exists():                        # 并发同 key：别的线程已写入
                self.hits += 1
                return read_json(path)['response']
            self.misses += 1
            write_json(path, self.identity(prompt, system=system, max_tokens=max_tokens)
                       | {'response': text})
        return text"""
    assert old_complete in src, 'complete anchor not found'
    src = src.replace(old_complete, new_complete, 1)
p.write_text(src)
print('llm_client.py patched')

# ---------- 3) runtime.py: write_predicted precomputed ----------
p = ROOT / 'vcrm/runtime.py'
src = p.read_text()
if 'precomputed' not in src:
    old = """    def write_predicted(self, event: Mapping[str, Any], batch_encode: bool = True) -> Dict[str, Any]:"""
    new = """    def write_predicted(self, event: Mapping[str, Any], batch_encode: bool = True,
                        precomputed: Optional[Any] = None) -> Dict[str, Any]:"""
    assert old in src, 'write_predicted signature anchor not found'
    src = src.replace(old, new, 1)
    old_body = """        assert_no_oracle(event, 'PredictedRuntime.write_predicted', allowed=PUBLIC_WRITE_KEYS)
        started = time.perf_counter()
        result = self.extractor.extract_detailed(event)"""
    new_body = """        assert_no_oracle(event, 'PredictedRuntime.write_predicted', allowed=PUBLIC_WRITE_KEYS)
        started = time.perf_counter()
        result = precomputed if precomputed is not None else self.extractor.extract_detailed(event)"""
    assert old_body in src, 'write_predicted body anchor not found'
    src = src.replace(old_body, new_body, 1)
p.write_text(src)
print('runtime.py patched')

# ---------- 2) run_main_eval.py ----------
p = ROOT / 'probe/run_main_eval.py'
src = p.read_text()
if 'extract_workers' not in src:
    old_we = '''def write_episode(runtime: PredictedRuntime, episode: Episode) -> Dict[str, Any]:
    """把本题 haystack 逐 session（date 升序）写入；返回事实台账。"""
    counter = PublicEventCounter()
    texts: Dict[str, str] = {}
    ledger: List[Dict[str, Any]] = []
    seconds = 0.0
    for session in episode.sessions:                     # 加载器已按 date 升序
        public_id = counter.next_id()
        texts[public_id] = session.text
        before = len(runtime.decisions)
        result = runtime.write_predicted(
            public_write_event({'text': session.text, 'time': session.date}, public_id))'''
    new_we = '''def write_episode(runtime: PredictedRuntime, episode: Episode,
                  extract_workers: int = 1) -> Dict[str, Any]:
    """把本题 haystack 逐 session（date 升序）写入；返回事实台账。

    extract_workers>1 时：每 session 的 LLM 事实抽取（extract_detailed，纯远端调用）
    并发预取；store 写入/路由/编码仍按 session date 升序严格串行——写入语义与
    串行版逐比特一致（facts 只由 session 文本决定，抽取顺序无关）。
    extract_workers=1 时代码路径与旧版完全相同。
    """
    counter = PublicEventCounter()
    texts: Dict[str, str] = {}
    ledger: List[Dict[str, Any]] = []
    seconds = 0.0
    events = [public_write_event({'text': s.text, 'time': s.date}, counter.next_id())
              for s in episode.sessions]
    precomputed = None
    if extract_workers > 1 and episode.sessions:
        from concurrent.futures import ThreadPoolExecutor
        extractor = runtime.extractor

        def _pre(ev: Mapping[str, Any]):
            return extractor.extract_detailed(ev)

        with ThreadPoolExecutor(max_workers=extract_workers) as pool:
            precomputed = list(pool.map(_pre, events))
    for session, event in zip(episode.sessions, events):
        public_id = event['id']
        texts[public_id] = session.text
        before = len(runtime.decisions)
        if precomputed is not None:
            result = runtime.write_predicted(event, precomputed=precomputed[events.index(event)])
        else:
            result = runtime.write_predicted(event)'''
    assert old_we in src, 'write_episode anchor not found'
    src = src.replace(old_we, new_we, 1)
p.write_text(src)
print('run_main_eval.py write_episode patched')
print('ALL PATCHES OK')
