from __future__ import annotations
# P1 parity: IncrementalCurrentStore 的顺序 ingest_decision 结果必须与
# CurrentSnapshotStore.build([v1,v2,v3]) 完全一致（同一冻结 Independent E3 codec）。
#
# 比较的 5 项（契约顺序）：
#   1) 版本 id / 时间 / 角色（current vs historical）
#   2) 每版 FP16 payload（逐比特）
#   3) codec.decode() 之后的 latent（逐比特）
#   4) 同时间查询 bundle()
#   5) stub reader 最终输出（假 reader，不加载 ICAE）
#
# 真实 E3 权重：用 probe/v6_loader.py 在服务器 CPU 上定位并加载
# /share/latent2/vcrm_icae/runs/architecture_v6_E3_train/codec.safetensors。
# 本机没有该挂载时，退化为“同构 E3 配置的本地 codec”，并在输出中显式标注来源，
# 绝不把本地 codec 的结果冒充真实权重结果（见 E3_CODEC_SOURCE）。
import functools
import importlib.util
import json
import pathlib
import unittest

import numpy as np
import torch

from vcrm.codec import SlotCodec
from vcrm.fact_schema import StateDecision
from vcrm.index import key_vector
from vcrm.store import CurrentSnapshotStore, IncrementalCurrentStore

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

STATE = 'Alice:employer'
DIMENSION = 4096
MEMORY_SLOTS = 128
KEY_DIMENSIONS = 128
# E3 配置（config.architecture_v6_E3_test.json）: history_slots=16, width=256, heads=4,
# conditioned=false, codec_value_path=full, codec_activation_rms_cap=8.0
CODEC_SLOTS, CODEC_WIDTH, CODEC_HEADS, RMS_CAP = 16, 256, 4, 8.0

VERSIONS = (
    ('v1', '2018-01-01T00:00:00', 'Google'),
    ('v2', '2019-01-01T00:00:00', 'Microsoft'),
    ('v3', '2020-01-01T00:00:00', 'OpenAI'),
)
DECISIONS = (
    StateDecision('CREATE', STATE, .9, 'first employer observation'),
    StateDecision('UPDATE', STATE, .9, 'employer changed'),
    StateDecision('UPDATE', STATE, .9, 'employer changed again'),
)


# -- codec sources ---------------------------------------------------------
def _load_v6_loader():
    path = REPO_ROOT / 'probe' / 'v6_loader.py'
    spec = importlib.util.spec_from_file_location('vcrm_probe_v6_loader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_real_e3_codec():
    """Locate and load the frozen E3 codec checkpoint on CPU (no GPU, no ICAE reader).

    The loaded weights are verified against the checkpoint file tensor-by-tensor
    (`codec.queries`, `codec.recovery_queries`, `codec.gate.*`), so a label that says
    `checkpoint:` really means "these are the trained E3 weights", not a local
    architecture with the same shape.
    """
    from safetensors.torch import load_file
    config, training_root, manifest = _load_v6_loader().load_v6('E3')
    metadata = json.loads((training_root / 'codec_training.json').read_text())
    if not metadata['completed']:
        raise ValueError('Incomplete E3 codec checkpoint')
    checkpoint = training_root / 'codec.safetensors'
    state = load_file(str(checkpoint))
    codec_state = {k[len('codec.'):]: v for k, v in state.items() if k.startswith('codec.')}
    dimension = int(codec_state['queries'].shape[1])
    codec = SlotCodec(dimension, metadata['history_slots'], metadata['codec_width'], metadata['codec_heads'],
                      metadata['method'] == 'D', metadata.get('anchor_dropout', 0.), metadata.get('seed', 42),
                      metadata.get('codec_variant', 'residual'), metadata.get('decode_conditioned'),
                      metadata.get('codec_value_path'), metadata.get('codec_activation_rms_cap'))
    codec.load_state_dict(codec_state, strict=True)
    codec.eval().requires_grad_(False)
    for name in ('queries', 'recovery_queries', 'gate.0.weight'):
        if not torch.equal(codec.state_dict()[name], codec_state[name]):
            raise ValueError(f'Loaded E3 codec tensor {name} does not match the checkpoint file')
    label = f'checkpoint:{checkpoint} sha256={metadata["checkpoint_sha256"]}'
    return codec, label


def local_e3_shaped_codec():
    """Untrained codec with the exact E3 architecture; used only when weights are unreachable."""
    torch.manual_seed(42)
    codec = SlotCodec(DIMENSION, CODEC_SLOTS, CODEC_WIDTH, CODEC_HEADS, False, 0., 42,
                      'residual', None, 'full', RMS_CAP)
    label = (f'local_e3_config(dimension={DIMENSION},slots={CODEC_SLOTS},width={CODEC_WIDTH},'
             f'heads={CODEC_HEADS},value_path=full,rms_cap={RMS_CAP}; UNTRAINED)')
    return codec.eval().requires_grad_(False), label


@functools.lru_cache(maxsize=None)
def real_e3_probe():
    try:
        codec, label = load_real_e3_codec()
        if codec.conditioned:
            raise ValueError('E3 checkpoint is conditioned; the incremental store needs an independent codec')
        return True, label, codec
    except Exception as error:  # missing mount, missing package, missing checkpoint ...
        return False, f'{type(error).__name__}: {error}', None


def resolve_codec(real):
    if real:
        available, label, codec = real_e3_probe()
        if not available:
            raise unittest.SkipTest(f'real E3 codec unavailable: {label}')
        return codec, label
    codec, label = local_e3_shaped_codec()
    return codec, label


# -- fixtures --------------------------------------------------------------
def memories():
    """Deterministic full memories that are exactly FP16-representable.

    Round-tripping through FP16 first means both stores feed bit-identical encoder
    inputs (the incremental path re-encodes the stored FP16 current), so any
    difference in this test would be a real protocol difference, not storage
    rounding. The FP32-input rounding effect is measured separately by
    `test_fp32_input_rounding_is_the_only_difference_source`.
    """
    result = {}
    for index, (version_id, _, _) in enumerate(VERSIONS):
        generator = torch.Generator().manual_seed(1000 + index)
        raw = torch.randn(MEMORY_SLOTS, DIMENSION, generator=generator) * .5
        result[version_id] = raw.half().float()
    return result


def keys():
    return {version_id: key_vector(f'Alice works at {org}', KEY_DIMENSIONS)
            for version_id, _, org in VERSIONS}


def events():
    """Control (oracle-side) events: CurrentSnapshotStore.build() needs explicit shared state keys."""
    return [{'id': version_id, 'state_key': STATE, 'time': time}
            for version_id, time, _ in VERSIONS]


def incremental_events():
    """Same identities without any state key: the incremental path takes state from the decision."""
    return [{'id': version_id, 'time': time} for version_id, time, _ in VERSIONS]


class StubReader:
    """Deterministic stand-in for the frozen ICAE reader: no weights, no model."""

    def __init__(self, labels):
        self.labels = labels

    def __call__(self, bundle):
        if bundle is None:
            return None
        means = bundle.float().mean(dim=-1)
        best = int(torch.argmax(means))
        return {'slots': int(bundle.shape[0]), 'slot': best,
                'mean': round(float(means[best]), 8), 'label': self.labels.get(best)}


@functools.lru_cache(maxsize=None)
def store_pair(real):
    codec, label = resolve_codec(real)
    full = memories()
    index_keys = keys()
    observed = events()
    snapshot = CurrentSnapshotStore(codec, 2048, 5, KEY_DIMENSIONS)
    snapshot.build(observed, full, index_keys)
    incremental = IncrementalCurrentStore(codec, key_dimensions=KEY_DIMENSIONS)
    for event, decision in zip(incremental_events(), DECISIONS):
        incremental.ingest_decision(event, full[event['id']], index_keys[event['id']], decision)
    return label, snapshot, incremental


def roles(store):
    return [(entry['id'], entry['time'],
             'current' if entry is store.current(entry['state']) else 'historical')
            for entry in store.entries]


def payload_of(store, version_id):
    return next(entry['payload'] for entry in store.entries if entry['id'] == version_id)


def latent_of(store, version_id):
    return store.decode(next(entry for entry in store.entries if entry['id'] == version_id))


def bundle_at(store, as_of):
    selected = store.select_version(STATE, as_of)
    return None if selected is None else store.bundle([selected])


class ParityAssertions:
    """Assertion mixin: identical checks for the local-codec and real-checkpoint runs."""
    real = False

    def pair(self):
        return store_pair(self.real)

    def test_version_identity_time_and_role(self):
        label, snapshot, incremental = self.pair()
        expected = roles(snapshot)
        actual = roles(incremental)
        print(f'\nPARITY codec={label}')
        print(f'PARITY roles snapshot={expected}')
        print(f'PARITY roles incremental={actual}')
        self.assertEqual(actual, expected)
        self.assertEqual([r[2] for r in actual], ['historical', 'historical', 'current'])
        self.assertEqual([r[0] for r in actual], ['v1', 'v2', 'v3'])
        self.assertEqual(incremental.blocks, {STATE: 'v3'})
        self.assertEqual(incremental.counts, {STATE: 3})
        self.assertEqual(snapshot.accounting()['resident_latent_slots'],
                         incremental.accounting()['resident_latent_slots'])

    def test_fp16_payload_bitwise_parity(self):
        label, snapshot, incremental = self.pair()
        worst, exact = 0., True
        for version_id, _, _ in VERSIONS:
            left, right = payload_of(snapshot, version_id), payload_of(incremental, version_id)
            self.assertEqual(left.dtype, np.dtype('<f2'))
            self.assertEqual(right.dtype, np.dtype('<f2'))
            self.assertEqual(left.shape, right.shape)
            worst = max(worst, float(np.abs(left.astype(np.float64) - right.astype(np.float64)).max()))
            exact = exact and bool(np.array_equal(left, right))
        print(f'PARITY payload_max_abs_diff={worst!r} bitwise_equal={exact} codec={label}')
        self.assertTrue(exact, f'FP16 payloads differ (max abs diff {worst!r})')
        self.assertEqual(worst, 0.)

    def test_decoded_latent_bitwise_parity(self):
        label, snapshot, incremental = self.pair()
        worst, exact = 0., True
        for version_id, _, _ in VERSIONS:
            left, right = latent_of(snapshot, version_id), latent_of(incremental, version_id)
            self.assertEqual(tuple(left.shape), tuple(right.shape))
            worst = max(worst, float((left - right).abs().max()))
            exact = exact and bool(torch.equal(left, right))
        print(f'PARITY latent_max_abs_diff={worst!r} bitwise_equal={exact} codec={label}')
        self.assertTrue(exact, f'decoded latents differ (max abs diff {worst!r})')

    def test_same_time_bundle_parity(self):
        label, snapshot, incremental = self.pair()
        checks = [None] + [time for _, time, _ in VERSIONS]
        for as_of in checks:
            left, right = bundle_at(snapshot, as_of), bundle_at(incremental, as_of)
            self.assertIsNotNone(left)
            self.assertIsNotNone(right)
            self.assertEqual(tuple(left.shape), tuple(right.shape))
            self.assertTrue(torch.equal(left, right), f'bundle mismatch at as_of={as_of}')
            print(f'PARITY bundle as_of={as_of} slots={int(left.shape[0])} equal=True')

    def test_stub_reader_output_parity(self):
        label, snapshot, incremental = self.pair()
        reader = StubReader({0: 'current'})
        left = reader(bundle_at(snapshot, None))
        right = reader(bundle_at(incremental, None))
        print(f'PARITY stub_reader snapshot={left} incremental={right} codec={label}')
        self.assertEqual(left, right)
        self.assertEqual(left['slots'], MEMORY_SLOTS)


class LocalCodecParityTest(ParityAssertions, unittest.TestCase):
    """Fast fallback tier: untrained codec with the exact E3 architecture.

    Runs only when the frozen E3 checkpoint is unreachable (no mount / no
    safetensors). When the checkpoint *is* reachable this tier skips, because the
    real-weight tier below is the authoritative gate; the two never mix.
    """
    real = False

    @classmethod
    def setUpClass(cls):
        available, reason, _ = real_e3_probe()
        if available:
            raise unittest.SkipTest(f'real E3 weights available ({reason}); local fallback tier skipped')

    def test_codec_source_is_reported_not_assumed(self):
        label, _, _ = self.pair()
        available, reason, _ = real_e3_probe()
        self.assertFalse(available)
        self.assertIn('local_e3_config', label)
        self.assertNotIn('checkpoint:', label)
        print(f'\nPARITY real E3 weights unavailable on this host: {reason}')


class RealE3WeightsParityTest(ParityAssertions, unittest.TestCase):
    """Server-CPU gate: identical assertions, but on the frozen E3 checkpoint weights."""
    real = True

    @classmethod
    def setUpClass(cls):
        available, reason, _ = real_e3_probe()
        if not available:
            raise unittest.SkipTest(f'real E3 codec unavailable: {reason}')

    def test_codec_source_is_the_e3_checkpoint(self):
        available, label, codec = real_e3_probe()
        self.assertTrue(available, label)
        pair_label, snapshot, incremental = self.pair()
        self.assertIn('checkpoint:', pair_label)
        self.assertIn('architecture_v6_E3_train/codec.safetensors', pair_label)
        self.assertIn('sha256=', pair_label)
        self.assertFalse(codec.conditioned)
        self.assertEqual((codec.dimension, codec.slots), (DIMENSION, CODEC_SLOTS))
        # 真实权重指纹：label 里的 sha256 必须与 codec_training.json 记录一致
        _, training_root, _ = _load_v6_loader().load_v6('E3')
        metadata = json.loads((training_root / 'codec_training.json').read_text())
        self.assertIn(metadata['checkpoint_sha256'], pair_label)
        self.assertEqual(incremental.codec_signature, codec.signature())
        print(f'\nPARITY real_weights_checkpoint={training_root}/codec.safetensors')
        print(f'PARITY real_weights_checkpoint_sha256={metadata["checkpoint_sha256"]}')


class Fp32RoundingSourceTest(unittest.TestCase):
    """Report (do not hide) the one legitimate difference source: FP16 storage rounding.

    The incremental path re-encodes the *stored FP16* current version, the offline
    snapshot encodes the original FP32 feature. With FP32 inputs the two payloads
    may differ by FP16 rounding of the previous version. The number is printed and
    bounded, and the exact-parity assertion above still uses FP16-representable
    inputs so no assertion is loosened.
    """

    def test_fp32_input_rounding_is_bounded_and_explained(self):
        codec, label = resolve_codec(real_e3_probe()[0])
        index_keys = keys()
        observed = events()
        full = {}
        for index, (version_id, _, _) in enumerate(VERSIONS):
            generator = torch.Generator().manual_seed(2000 + index)
            full[version_id] = torch.randn(MEMORY_SLOTS, DIMENSION, generator=generator) * .5
        snapshot = CurrentSnapshotStore(codec, 2048, 5, KEY_DIMENSIONS)
        snapshot.build(observed, full, index_keys)
        incremental = IncrementalCurrentStore(codec, key_dimensions=KEY_DIMENSIONS)
        for event, decision in zip(incremental_events(), DECISIONS):
            incremental.ingest_decision(event, full[event['id']], index_keys[event['id']], decision)
        worst = max(float(np.abs(payload_of(snapshot, v).astype(np.float64) -
                                 payload_of(incremental, v).astype(np.float64)).max())
                    for v, _, _ in VERSIONS)
        print(f'\nPARITY fp32_input payload_max_abs_diff={worst!r} (FP16 storage rounding of the '
              f'previous current version; codec={label})')
        self.assertLess(worst, 1e-2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
