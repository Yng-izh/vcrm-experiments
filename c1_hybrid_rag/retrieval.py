"""C1 hybrid retrieval frontend (CONTRACT_C1_HYBRID_RAG.md §3, frozen).

BM25 (rank_bm25, lowercase + \\W+ tokenize, no stemming) + BGE-base-en-v1.5
dense cosine (FP16 on GPU) -> RRF k=60 over each side's top-50 -> final top-10.
No time filtering/weighting at retrieval. numpy dot-product index.

Reusable: later RAG-family baselines (LLMLingua-2+RAG, ICAE-Full+RAG) call
`retrieve()` with the same corpus index; only the injected content changes.
"""
import numpy as np
import torch
from rank_bm25 import BM25Okapi


def bm25_tokenize(text):
    """Frozen tokenizer: lowercase + simple non-word split, no stemming."""
    import re
    return re.split(r"\W+", text.lower())


class RetrievalIndex:
    def __init__(self, events, device="cuda", bge_path="models/bge-base-en-v1.5",
                 fp16=True, cache_path=None, batch_size=256):
        """events: list of {event_id, time, text} (from corpus.build_corpus)."""
        import json, os
        self.events = events
        self.ids = [e["event_id"] for e in events]
        # --- sparse ---
        tokenized = [bm25_tokenize(e["text"]) for e in events]
        self.bm25 = BM25Okapi(tokenized)
        # --- dense ---
        from sentence_transformers import SentenceTransformer
        if device == "cpu":
            fp16 = False  # FP16 on CPU is slow/unsupported for this model
        self.model = SentenceTransformer(bge_path, device=device)
        if fp16:
            self.model = self.model.half()
        cache = None
        if cache_path and os.path.exists(cache_path):
            cache = np.load(cache_path)
            if cache.shape[0] == len(events):
                self.emb = cache
            else:
                cache = None
        if cache is None:
            self.emb = self.model.encode(
                [e["text"] for e in events], batch_size=batch_size,
                normalize_embeddings=True, show_progress_bar=True,
                convert_to_numpy=True,
            )
            if cache_path:
                np.save(cache_path, self.emb)
        else:
            print(f"[index] loaded cached embeddings {self.emb.shape} from {cache_path}")
        self.emb = self.emb.astype(np.float32)

    # ------------------------------------------------------------------
    def bm25_top(self, query, k=50):
        scores = self.bm25.get_scores(bm25_tokenize(query))
        return np.argsort(-scores)[:k]

    def dense_top(self, query, k=50):
        q = self.model.encode([query], normalize_embeddings=True,
                              convert_to_numpy=True).astype(np.float32)[0]
        sims = self.emb @ q  # cosine (both normalized)
        return np.argsort(-sims)[:k]

    def retrieve(self, query, topk=10, bm25_k=50, dense_k=50, rrf_k=60):
        """RRF: score = sum over lists 1/(rrf_k + rank_i). Returns list of
        {event_id, rrf, bm25_rank, dense_rank} sorted by rrf desc, id asc."""
        b = self.bm25_top(query, bm25_k)
        d = self.dense_top(query, dense_k)
        scores = {}
        rank = {}
        for r, i in enumerate(b):
            scores[self.ids[i]] = scores.get(self.ids[i], 0.0) + 1.0 / (rrf_k + r + 1)
            rank.setdefault(self.ids[i], [None, None])[0] = r + 1
        for r, i in enumerate(d):
            scores[self.ids[i]] = scores.get(self.ids[i], 0.0) + 1.0 / (rrf_k + r + 1)
            rank.setdefault(self.ids[i], [None, None])[1] = r + 1
        out = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:topk]
        return [{"event_id": eid, "rrf": s, "bm25_rank": rank[eid][0],
                 "dense_rank": rank[eid][1]} for eid, s in out]

    def close(self):
        del self.model
        torch.cuda.empty_cache()
