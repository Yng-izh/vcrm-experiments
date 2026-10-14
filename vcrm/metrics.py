from collections import Counter
import math
import re
import string
import numpy as np


def normalize(text):
    text = str(text).lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def exact(prediction, answers):
    return int(any(normalize(prediction) == normalize(a) for a in answers))


def f1(prediction, answers):
    predicted = normalize(prediction).split()
    best = 0.
    for answer in answers:
        gold = normalize(answer).split()
        common = sum((Counter(predicted) & Counter(gold)).values())
        value = 2 * common / (len(predicted) + len(gold)) if predicted or gold else 1.
        best = max(best, value)
    return best


def paired_bootstrap(a, b, groups, iterations=2000, seed=42):
    if not a:
        return {'n': 0, 'delta': None, 'ci95': None}
    clusters = sorted(set(groups))
    indices = {g: np.array([i for i, value in enumerate(groups) if value == g]) for g in clusters}
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        draws = rng.integers(len(clusters), size=len(clusters))
        selected = np.concatenate([indices[clusters[i]] for i in draws])
        values.append(float(diff[selected].mean()))
    return {'n': len(a), 'groups': len(clusters), 'delta': float(diff.mean()),
            'ci95': np.quantile(values, [.025, .975]).tolist()}
