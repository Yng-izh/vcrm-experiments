"""P0 scope tests: LongMemEval-S text-RAG retrieval must be per-item.

Contract under test (lme_s_text_rag/run_text_rag.py):
- the only legal memory namespace for a question is THAT question's haystack;
  it is NOT the Oracle episode (no episode labels exist in LongMemEval-S);
- every question is retrieved against a freshly built index over its own
  haystack, so a hit can never come from another question's sessions, even when
  two questions reuse the same session_id ("S1");
- the output row schema is unchanged (consumed by make_official_eval_input.py).

stdlib unittest only; no GPU, no vLLM, no OPENAI_API_KEY, no network. The fake
index/embedder keep the test independent of rank_bm25/sentence_transformers; a
real-retrieval test is included but skipped unless those deps and the BGE model
are present (server mirror), so the scope property is also checked end-to-end.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "lme_s_text_rag" / "run_text_rag.py"

ROW_FIELDS = [
    "question_id",
    "question",
    "question_type",
    "hypothesis",
    "retrieved_session_ids",
    "retrieved_session_dates",
    "prompt_tokens",
    "read_seconds",
]


def load_rag_module():
    spec = importlib.util.spec_from_file_location("lme_run_text_rag_scope_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rag = load_rag_module()


def make_item(qid, sessions):
    """sessions: list of (session_id, date, text)."""
    return {
        "question_id": qid,
        "question": f"what did I say in {qid}?",
        "question_type": "single-session-user",
        "sessions": [{"session_id": sid, "date": date, "text": text}
                     for sid, date, text in sessions],
    }


class RecordingIndex:
    """Stands in for c1_hybrid_rag.retrieval.RetrievalIndex (frozen class).

    Records the docs it was built over and returns them in insertion order, so
    the test can assert the *index scope* itself, not just the post-filtering.
    """

    built = []

    def __init__(self, events, device="cpu", bge_path=None):
        self.events = [dict(e) for e in events]
        self.bge_path = bge_path
        RecordingIndex.built.append(self)

    def retrieve(self, query, topk=10, **_kw):
        return [{"event_id": e["event_id"]} for e in self.events[:topk]]


class LeakyIndex(RecordingIndex):
    """Simulates a stale/global index that hands back foreign documents."""

    def __init__(self, events, foreign_event_ids=(), device="cpu", bge_path=None):
        super().__init__(events, device=device, bge_path=bge_path)
        self.foreign_event_ids = list(foreign_event_ids)

    def retrieve(self, query, topk=10, **_kw):
        own = [{"event_id": e["event_id"]} for e in self.events]
        return [{"event_id": f} for f in self.foreign_event_ids] + own


class ItemScopeTest(unittest.TestCase):
    def setUp(self):
        RecordingIndex.built = []
        self.item_a = make_item("q_A", [
            ("S1", "2023-01-01", "USER: I adopted a greyhound named Comet."),
            ("S2", "2023-02-01", "USER: I moved to Lisbon for a postdoc."),
        ])
        # same session_id "S1", different question, different content/date
        self.item_b = make_item("q_B", [
            ("S1", "2024-05-05", "USER: My sourdough starter is called Boris."),
            ("S2", "2024-06-06", "USER: I bought a red kayak."),
        ])

    # -- index construction scope ------------------------------------------
    def test_item_index_contains_only_that_items_haystack(self):
        idx_a, _ = rag.build_item_index(self.item_a, index_cls=RecordingIndex)
        idx_b, _ = rag.build_item_index(self.item_b, index_cls=RecordingIndex)

        self.assertEqual([e["event_id"] for e in idx_a.events], ["q_A::S1", "q_A::S2"])
        self.assertEqual([e["event_id"] for e in idx_b.events], ["q_B::S1", "q_B::S2"])

        a_texts = " ".join(e["text"] for e in idx_a.events)
        b_texts = " ".join(e["text"] for e in idx_b.events)
        self.assertIn("Comet", a_texts)
        self.assertNotIn("Boris", a_texts)          # no cross-question docs
        self.assertIn("Boris", b_texts)
        self.assertNotIn("Comet", b_texts)

    def test_events_are_namespaced_and_deduped_within_item(self):
        item = make_item("q_D", [
            ("S1", "2023-01-01", "USER: first copy"),
            ("S1", "2023-01-02", "USER: duplicate session_id"),
            ("S3", "2023-01-03", "USER: third"),
        ])
        events = rag.build_item_events(item)
        self.assertEqual([e["event_id"] for e in events], ["q_D::S1", "q_D::S3"])
        self.assertEqual(events[0]["text"], "USER: first copy")  # first wins
        self.assertEqual(rag.session_by_id(item)["S1"]["date"], "2023-01-01")

    # -- retrieval hits stay inside the item namespace ----------------------
    def test_hits_belong_to_item_haystack(self):
        item_ids = {s["session_id"] for s in self.item_b["sessions"]}
        idx_b, _ = rag.build_item_index(self.item_b, index_cls=RecordingIndex)
        hit_ids, selected = rag.retrieve_for_item(idx_b, self.item_b, topk=10)

        self.assertTrue(hit_ids)
        for eid in hit_ids:
            self.assertTrue(eid.startswith("q_B::"), eid)
            self.assertIn(rag.session_id_of(eid), item_ids)

        lookup = rag.session_by_id(self.item_b)
        dates = [lookup[rag.session_id_of(h)]["date"] for h in hit_ids]
        self.assertEqual(dates, ["2024-05-05", "2024-06-06"])
        self.assertEqual([s["session_id"] for s in selected], ["S1", "S2"])
        prompt = rag.render_prompt(self.item_b["question"], selected)
        self.assertIn("Boris", prompt)
        self.assertNotIn("Comet", prompt)

    def test_session_id_collision_does_not_produce_cross_item_hit(self):
        idx_a, _ = rag.build_item_index(self.item_a, index_cls=RecordingIndex)
        idx_b, _ = rag.build_item_index(self.item_b, index_cls=RecordingIndex)

        hits_a, sel_a = rag.retrieve_for_item(idx_a, self.item_a, topk=10)
        hits_b, sel_b = rag.retrieve_for_item(idx_b, self.item_b, topk=10)

        # identical session_id, disjoint namespaces
        self.assertEqual(rag.session_id_of(hits_a[0]), "S1")
        self.assertEqual(rag.session_id_of(hits_b[0]), "S1")
        self.assertNotEqual(hits_a[0], hits_b[0])
        self.assertTrue(all(h.startswith("q_A::") for h in hits_a))
        self.assertTrue(all(h.startswith("q_B::") for h in hits_b))

        self.assertEqual(sel_a[0]["date"], "2023-01-01")
        self.assertEqual(sel_b[0]["date"], "2024-05-05")
        self.assertIn("Comet", sel_a[0]["text"])
        self.assertIn("Boris", sel_b[0]["text"])
        self.assertNotIn("Boris", " ".join(s["text"] for s in sel_a))
        self.assertNotIn("Comet", " ".join(s["text"] for s in sel_b))

    def test_foreign_hits_raise_instead_of_leaking(self):
        # a global/stale index returning another question's document
        leaky = LeakyIndex(rag.build_item_events(self.item_a),
                           foreign_event_ids=["q_B::S1"])
        with self.assertRaises(RuntimeError):
            rag.retrieve_for_item(leaky, self.item_a, topk=10)

        # right prefix, unknown session -> also refused
        unknown = LeakyIndex(rag.build_item_events(self.item_a),
                             foreign_event_ids=["q_A::S99"])
        with self.assertRaises(RuntimeError):
            rag.retrieve_for_item(unknown, self.item_a, topk=10)

    # -- frozen output schema ----------------------------------------------
    def test_row_schema_unchanged(self):
        idx_a, _ = rag.build_item_index(self.item_a, index_cls=RecordingIndex)
        hit_ids, _ = rag.retrieve_for_item(idx_a, self.item_a, topk=10)
        row = rag.build_row(self.item_a, hit_ids, rag.session_by_id(self.item_a),
                            hypothesis="a greyhound called Comet", prompt_tokens=42,
                            read_seconds=0.5)
        self.assertEqual(list(row), ROW_FIELDS)
        self.assertEqual(row["retrieved_session_ids"], ["q_A::S1", "q_A::S2"])
        self.assertEqual(row["retrieved_session_dates"], ["2023-01-01", "2023-02-01"])

    # -- embedder reuse path (SentenceTransformer patched for one call) -----
    def test_embedder_is_reused_across_items(self):
        sentinel_model = object()
        created = []

        class _StModule:
            def __init__(self):
                self.SentenceTransformer = self._make

            def _make(self, *args, **kwargs):
                created.append((args, kwargs))
                return "fresh-model"

        class _IndexUsingSentenceTransformer(RecordingIndex):
            def __init__(self, events, device="cpu", bge_path=None):
                super().__init__(events, device=device, bge_path=bge_path)
                from sentence_transformers import SentenceTransformer  # patched below
                self.model = SentenceTransformer(bge_path, device=device)

        stub = _StModule()
        had_st = "sentence_transformers" in sys.modules
        previous = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = stub
        try:
            _, model_a = rag.build_item_index(
                self.item_a, index_cls=_IndexUsingSentenceTransformer, bge_path="bge")
            self.assertIs(model_a, "fresh-model")
            self.assertEqual(len(created), 1)  # loaded once ...

            _, model_b = rag.build_item_index(
                self.item_b, index_cls=_IndexUsingSentenceTransformer,
                model=model_a, bge_path="bge")
            self.assertIs(model_b, model_a)
            self.assertEqual(len(created), 1)  # ... then reused, not reloaded
            # the stub factory is restored after the patched call
            self.assertEqual(sys.modules["sentence_transformers"].SentenceTransformer,
                             stub._make)
        finally:
            if had_st:
                sys.modules["sentence_transformers"] = previous
            else:
                sys.modules.pop("sentence_transformers", None)

    # -- no shared corpus index can creep back in --------------------------
    def test_main_builds_index_inside_per_item_loop(self):
        tree = ast.parse(MODULE_PATH.read_text())
        main = next(n for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main")

        def calls(node, name):
            return [c for c in ast.walk(node)
                    if isinstance(c, ast.Call) and (
                        (isinstance(c.func, ast.Name) and c.func.id == name)
                        or (isinstance(c.func, ast.Attribute) and c.func.attr == name))]

        self.assertEqual(calls(main, "RetrievalIndex"), [])  # no shared index

        in_loop = [c for loop in ast.walk(main) if isinstance(loop, ast.For)
                   for c in calls(loop, "build_item_index")]
        self.assertTrue(in_loop, "build_item_index must be called per item")


def _real_retrieval_ready():
    import importlib.util as ilu
    if ilu.find_spec("rank_bm25") is None or ilu.find_spec("sentence_transformers") is None:
        return False
    bge = Path(os.environ.get("VCRM_BGE_PATH",
                              "/share/latent2/vcrm_icae/models/bge-base-en-v1.5"))
    return bge.is_dir()


@unittest.skipUnless(_real_retrieval_ready(),
                     "rank_bm25/sentence_transformers/BGE model unavailable (CPU-only CI)")
class RealRetrievalScopeTest(unittest.TestCase):
    """End-to-end scope check with the frozen C1 index (server CPU only)."""

    def setUp(self):
        c1 = REPO / "c1_hybrid_rag"
        for p in (str(c1),):
            if p not in sys.path:
                sys.path.insert(0, p)
        from retrieval import RetrievalIndex
        self.index_cls = RetrievalIndex
        self.bge = os.environ.get("VCRM_BGE_PATH",
                                  "/share/latent2/vcrm_icae/models/bge-base-en-v1.5")

    def test_real_retrieval_stays_inside_haystack_with_id_collision(self):
        item_a = make_item("q_A", [
            ("S1", "2023-01-01", "USER: I adopted a greyhound named Comet."),
            ("S2", "2023-02-01", "USER: I moved to Lisbon for a postdoc."),
        ])
        item_b = make_item("q_B", [
            ("S1", "2024-05-05", "USER: My sourdough starter is called Boris."),
            ("S2", "2024-06-06", "USER: I bought a red kayak."),
        ])
        idx_a, model = rag.build_item_index(item_a, index_cls=self.index_cls,
                                            bge_path=self.bge)
        idx_b, _ = rag.build_item_index(item_b, index_cls=self.index_cls,
                                        model=model, bge_path=self.bge)

        hits_a, sel_a = rag.retrieve_for_item(idx_a, item_a, topk=2)
        hits_b, sel_b = rag.retrieve_for_item(idx_b, item_b, topk=2)

        self.assertTrue(all(h.startswith("q_A::") for h in hits_a))
        self.assertTrue(all(h.startswith("q_B::") for h in hits_b))
        a_text = " ".join(s["text"] for s in sel_a)
        b_text = " ".join(s["text"] for s in sel_b)
        self.assertNotIn("Boris", a_text)
        self.assertNotIn("Comet", b_text)


if __name__ == "__main__":
    unittest.main()
