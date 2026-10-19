"""Async concurrent C1 reader (vllm OpenAI-compatible /v1/completions).

Same frozen semantics as reader.read_answer: temperature 0, max_tokens 128,
seed 42, MODEL id "mistral-7b". Concurrency is a transport-level change only
(single-request greedy output unaffected; config recorded in run log).
"""
import asyncio
import json

import aiohttp

MODEL = "mistral-7b"


async def read_answer_async(session, url, prompt, semaphore, max_retries=6, timeout=600):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": 128,
        "seed": 42,
    }
    tout = aiohttp.ClientTimeout(total=timeout)
    last = None
    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.post(url, json=payload, timeout=tout) as r:
                    r.raise_for_status()
                    out = await r.json()
            return out["choices"][0]["text"].strip(), out["usage"]["prompt_tokens"]
        except Exception as e:  # transient server hiccups
            last = e
            await asyncio.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"vllm call failed after {max_retries} retries: {last}")


async def read_answers_async(prompts, url="http://localhost:8001/v1/completions", concurrency=16):
    """prompts: list of chat-templated prompt strings. Returns list aligned with
    input: (answer, prompt_tokens). Raises on first unrecoverable failure."""
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [read_answer_async(session, url, p, sem) for p in prompts]
        return await asyncio.gather(*tasks)
