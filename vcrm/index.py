import hashlib
import re
import numpy as np
import torch
from torch import nn


class SemanticIndex(nn.Module):
    """Shared attention-pooled ICAE keys; random initialization is not semantic validation."""
    def __init__(self, hidden=4096, dimensions=128):
        super().__init__()
        self.hidden, self.dimensions = hidden, dimensions
        self.pool_query = nn.Parameter(torch.randn(hidden) * .02)
        self.projection = nn.Linear(hidden, dimensions, bias=False)

    def forward(self, memory):
        if memory.ndim != 2 or memory.shape[1] != self.hidden or not memory.shape[0]:
            raise ValueError('Semantic pooling expects [slots, hidden]')
        memory = memory.to(self.pool_query)
        weights = torch.softmax(memory @ self.pool_query / self.hidden ** .5, dim=0)
        return torch.nn.functional.normalize(self.projection(weights @ memory), dim=-1)

    @torch.no_grad()
    def key(self, memory):
        return self(memory).detach().cpu().numpy().astype('<f2')

    @staticmethod
    def rank(query_key, entries):
        def similarity(entry):
            a, b = np.asarray(query_key, dtype=np.float32), entry['key'].astype(np.float32)
            norm = np.linalg.norm(a) * np.linalg.norm(b)
            return float(a @ b / norm) if norm else -1.
        return sorted([(similarity(e), e) for e in entries], key=lambda item: (item[0], item[1]['id']), reverse=True)

    @staticmethod
    def associate(query_key, entries, threshold=.8, margin=.05):
        ranked = SemanticIndex.rank(query_key, entries)
        if not ranked or ranked[0][0] < threshold:
            return None
        second = next((score for score, e in ranked[1:] if e['state'] != ranked[0][1]['state']), -1.)
        if ranked[0][0] - second < margin:
            return None
        return {'state_key': ranked[0][1]['state'], 'similarity': ranked[0][0],
                'status': 'association_candidate_not_fact_update'}

STOP = set('a an the is are was were i you my your me to for in on at of and or it that this with have has had'.split())


def key_vector(text, dimensions):
    vector = np.zeros(dimensions, dtype=np.float32)
    for word in re.findall(r'[a-z]+', text.lower()):
        if word in STOP:
            continue
        value = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), 'little')
        vector[value % dimensions] += 1 if (value // dimensions) % 2 else -1
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else vector).astype('<f2')


def inferred_as_of(query):
    if 'as_of' in query:
        return query['as_of']
    # Only explicit historical phrases; the benchmark query date is not a past-version request.
    match = re.search(r'\b(?:in|during|as of|back in)\s+((?:19|20)\d{2})(?:\b)', query['question'], re.I)
    return match.group(1) + '-12-31T23:59:59' if match else None
