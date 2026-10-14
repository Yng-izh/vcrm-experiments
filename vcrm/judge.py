import ast
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from .io import digest, read_json, write_json


def official_prompt(config):
    source = Path(config['longmemeval_root']) / 'src/evaluation/evaluate_qa.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_anscheck_prompt')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['get_anscheck_prompt'], digest(source.read_bytes())


def score(config, query, prediction, root):
    prompt_fn, source_hash = official_prompt(config)
    prompt = prompt_fn(query['type'], query['question'], query['answers'][0], prediction,
                       abstention=query.get('abstention', False))
    identity = [config['judge_model'], config['judge_base_url'], source_hash, prompt]
    cache = Path(root) / 'judge_cache' / (digest(identity) + '.json')
    if cache.exists():
        return read_json(cache)
    key = os.environ.get(config['judge_key_env'])
    if not key:
        raise RuntimeError('Judge credentials missing; no exact-match substitution is allowed')
    body = json.dumps({'model': config['judge_model'], 'messages': [{'role': 'user', 'content': prompt}],
                       'temperature': 0, 'max_tokens': 10}).encode()
    request = urllib.request.Request(config['judge_base_url'].rstrip('/') + '/chat/completions', data=body,
                                     headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                answer = json.load(response)['choices'][0]['message']['content'].strip().lower()
            if answer not in ('yes', 'yes.', 'no', 'no.'):
                raise ValueError(f'Invalid judge response: {answer!r}')
            result = {'accuracy': int(answer.startswith('yes')), 'judge_model': config['judge_model'],
                      'protocol': 'Official LongMemEval prompt; custom judge, not official GPT-4o scores',
                      'prompt_source_sha256': source_hash, 'response': answer}
            write_json(cache, result)
            return result
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 3:
                raise
        time.sleep(2 ** attempt)
