"""C1 injection + prompt (CONTRACT_C1_HYBRID_RAG.md §4, frozen).

Top-10 retrieved events sorted by `time` ascending (latest LAST); ties broken
by event_id lexicographic. Same prompt template reused by all RAG-family
baselines (only the facts block changes). Query `_X_` kept verbatim.
"""
from datetime import datetime

PROMPT_TEMPLATE = """I will give you several historical facts arranged in chronological order.
Please answer the question. If multiple facts report the same property,
use the fact that is consistent with the time implied by the question.

Historical Facts:
{facts}

Question: {question}
Answer:"""


def _time_key(t):
    # ISO 'YYYY-MM-DDTHH:MM:SS' sorts lexicographically = chronologically.
    return (str(t), "")


def build_facts(events_by_id, retrieved):
    """retrieved: output of RetrievalIndex.retrieve (list of dicts with event_id)."""
    hits = [events_by_id[r["event_id"]] for r in retrieved]
    hits.sort(key=lambda e: _time_key(e["time"]) + (e["event_id"],))
    return [e["text"] for e in hits], hits


def build_prompt(question, fact_texts):
    return PROMPT_TEMPLATE.format(facts="\n".join(fact_texts), question=question)


def chat_wrap(prompt, tokenizer):
    """Mistral-7B-Instruct-v0.2 chat template wrap."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
