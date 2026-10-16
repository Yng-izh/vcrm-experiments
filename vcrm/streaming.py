"""Query-independent span ingestion; similarity is not a contradiction detector."""
from .io import digest
from .index import SemanticIndex


def split_event(event, tokenizer, window_tokens=256):
    if window_tokens < 1:
        raise ValueError('Span window must be positive')
    # Token ids, rather than decode/re-tokenize, preserve exact original coverage.
    ids = tokenizer.encode(event['text'], add_special_tokens=False)
    spans = []
    for start in range(0, len(ids), window_tokens):
        end = min(start + window_tokens, len(ids))
        spans.append({'id': f'{event["id"]}:span{start // window_tokens:06d}',
                      'time': event['time'], 'source_event_id': event['id'],
                      'source_sha256': digest(event['text']), 'token_start': start, 'token_end': end,
                      'input_ids': ids[start:end]})
    return spans


class SpanWriter:
    def __init__(self, reader, index, store, threshold=.8, update_resolver=None, validated_index=False):
        self.reader, self.index, self.store = reader, index, store
        self.threshold, self.resolver = threshold, update_resolver
        self.validated_index = validated_index

    def write(self, event, window_tokens=256):
        receipts = []
        for span in split_event(event, self.reader.tok, window_tokens):
            # Timestamp is separately retained and injected with version markers.
            memory = self.reader.encode_ids([self.reader.tok.bos_token_id] + span['input_ids'])
            key = self.index.key(memory)
            candidate = SemanticIndex.associate(key, self.store.entries, self.threshold) if self.validated_index else None
            # A shared resolver must validate entity/relation AND an actual update.
            # Without it, similar chat is independent; no silent overwrite/compression.
            assignment = self.resolver(span, candidate) if self.resolver and candidate else None
            state = assignment['state_key'] if assignment and assignment.get('is_update') else span['id']
            view = {'id': span['id'], 'time': span['time'], 'state_key': state}
            receipt = self.store.ingest(view, memory, key)
            receipts.append({k: v for k, v in span.items() if k != 'input_ids'} |
                            {'state_key': state, 'association': candidate,
                             'conditional_update_triggered': bool(assignment and assignment.get('is_update')),
                             **receipt})
        return receipts
