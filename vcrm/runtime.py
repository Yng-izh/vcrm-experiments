"""Opt-in trained ICAE C/D runtime: write spans, persist, and read temporal entries."""
from pathlib import Path
import time

import torch

from .codec import SlotCodec, VersionMarkers
from .index import SemanticIndex, inferred_as_of
from .io import file_digest, read_json
from .store import CurrentSnapshotStore, KeyframeStore
from .streaming import SpanWriter


class SlotMemoryRuntime:
    def __init__(self, reader, codec, markers, index, store, validated_index=False, update_resolver=None):
        self.reader, self.codec, self.markers, self.index, self.store = reader, codec, markers, index, store
        self.validated_index = validated_index
        self.writer = SpanWriter(reader, index, store, reader.cfg['state_similarity'], update_resolver, validated_index)

    def save(self, path):
        self.store.save(path)

    @classmethod
    def from_checkpoint(cls, config, training_root, store_path=None, update_resolver=None):
        from safetensors.torch import load_file
        from .reader import FrozenReader
        root = Path(training_root)
        metadata = read_json(root / 'codec_training.json')
        protocol = metadata.get('memory_protocol', 'immutable_keyframe_v1')
        if config.get('memory_protocol', 'immutable_keyframe_v1') != protocol:
            raise ValueError('Training/evaluation memory protocol mismatch')
        checkpoint = root / 'codec.safetensors'
        if not metadata['completed'] or metadata['checkpoint_sha256'] != file_digest(checkpoint):
            raise ValueError('Incomplete or altered codec checkpoint')
        if (metadata['base_checkpoint_sha256'] != file_digest(config['checkpoint']) or
                metadata['model_revision'] != config['model_revision'] or
                metadata['reader_sha256'] != file_digest(Path(__file__).with_name('reader.py'))):
            raise ValueError('Codec training used a different ICAE reader')
        adapted_hash = file_digest(config['adapted_checkpoint']) if config.get('adapted_checkpoint') else None
        if adapted_hash != metadata['adapted_checkpoint_sha256']:
            raise ValueError('Codec adaptation identity mismatch')
        reader = FrozenReader(config)
        codec = SlotCodec(reader.shape[1], metadata['history_slots'], metadata['codec_width'],
                          metadata['codec_heads'], metadata['method'] == 'D',
                          metadata.get('anchor_dropout', 0.), metadata.get('seed', 42),
                          metadata.get('codec_variant', 'residual'),
                          metadata.get('decode_conditioned'), metadata.get('codec_value_path'),
                          metadata.get('codec_activation_rms_cap')).to(reader.device)
        markers = VersionMarkers(reader.shape[1]).to(reader.device)
        index = SemanticIndex(reader.shape[1], config['key_dimensions']).to(reader.device)
        state = load_file(str(checkpoint))
        for prefix, module in (('codec', codec), ('markers', markers), ('index', index)):
            module.load_state_dict({k[len(prefix) + 1:]: v for k, v in state.items() if k.startswith(prefix + '.')}, strict=True)
            module.eval().requires_grad_(False)
        codec_kwargs = (codec, config['resident_slot_budget'])
        store_type = CurrentSnapshotStore if protocol == 'final_current_snapshot_v1' else KeyframeStore
        store = (store_type.load(store_path, *codec_kwargs) if store_path else
                 store_type(*codec_kwargs, metadata['keyframe_interval'], config['key_dimensions']))
        # A trained pooler still needs held-out calibration, especially query vs event keys.
        validated = False
        if config.get('routing_calibration'):
            calibration = read_json(config['routing_calibration'])
            if (not calibration.get('complete') or calibration.get('split') != 'validation' or
                    calibration.get('semantic_index_sha256') != metadata['semantic_index_sha256'] or
                    calibration.get('reader_sha256') != metadata['reader_sha256'] or
                    calibration.get('threshold') != config['state_similarity'] or
                    not calibration.get('query_event_matching_validated') or
                    not calibration.get('association_validated')):
                raise ValueError('Missing or mismatched held-out routing calibration')
            validated = True
        return cls(reader, codec, markers, index, store, validated, update_resolver)

    @torch.no_grad()
    def write_fact(self, event):
        if isinstance(self.store, CurrentSnapshotStore):
            raise RuntimeError('Offline snapshot does not support incremental writes')
        if not event.get('state_key'):
            raise ValueError('write_fact requires a shared explicit state key')
        memory = self.reader.encode_event(event)
        return self.store.ingest(event, memory, self.index.key(memory))

    def write_session(self, event):
        if isinstance(self.store, CurrentSnapshotStore):
            raise RuntimeError('Offline snapshot does not support streaming writes')
        return self.writer.write(event, self.reader.cfg.get('span_window_tokens', 256))

    @torch.no_grad()
    def answer(self, query):
        state = query.get('state_key')
        if not state:
            if not self.validated_index:
                raise ValueError('Query-to-memory semantic retrieval is not calibrated; supply shared state_key')
            key = self.index.key(self.reader.encode(self.reader.question(query)))
            candidate = self.index.associate(key, self.store.entries, self.reader.cfg['state_similarity'])
            if candidate is None:
                return self.reader.answer(query) | {'no_state_match': True}
            state = candidate['state_key']
        current = self.store.current(state)
        selected = self.store.select_version(state, inferred_as_of(query))
        if current is None or selected is None:
            return self.reader.answer(query) | {'version_missing': True}
        historical = selected if selected is not current else next(
            (e for e in reversed(self.store.entries) if e['state'] == state and e is not current), None)
        entries = [current] + ([historical] if historical is not None else [])
        if self.reader.device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        bundle = self.store.bundle(entries, self.markers, self.reader.cfg['read_latent_slots'])
        if bundle.device != self.reader.device:
            bundle = bundle.to(self.reader.device)
        if self.reader.device.type == 'cuda':
            torch.cuda.synchronize()
        recovery_seconds = time.perf_counter() - started
        result = self.reader.answer(query, bundle)
        dependencies = {e['anchor_id'] for e in entries if e['anchor_id']}
        selected_ids = {e['id'] for e in entries}
        return result | {'selected_entry_ids': [e['id'] for e in entries],
                         'role_time_slots': 2 * len(entries),
                         'codec_recovery_seconds': recovery_seconds,
                         'auxiliary_anchor_slots': sum(self.store._entry(eid)['payload'].shape[0]
                                                       for eid in dependencies - selected_ids),
                         'resident_accounting': self.store.accounting(),
                         'protocol': ('current plus historical m-slot read; offline final-current snapshot'
                                      if isinstance(self.store, CurrentSnapshotStore) else
                                      'current plus historical m-slot read; fixed-keyframe recovery')}
