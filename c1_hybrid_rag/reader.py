"""C1 reader via vllm OpenAI-compatible server (localhost:8001, model mistral-7b).

Contract §4: greedy (temperature 0), max_new_tokens 128, seed 42. Prompt is
already chat-templated; we send it raw as a /v1/completions prompt.
"""
import json
import time
import urllib.request

VLLM_URL = "http://localhost:8001/v1/completions"
MODEL = "mistral-7b"


def read_answer(prompt, max_retries=6, timeout=600):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": 128,
        "seed": 42,
    }
    last = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            req = urllib.request.Request(
                VLLM_URL, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
            return out["choices"][0]["text"].strip(), out["usage"]["prompt_tokens"], time.time() - t0
        except Exception as e:  # transient server hiccups
            last = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"vllm call failed after {max_retries} retries: {last}")
