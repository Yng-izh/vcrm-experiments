"""Parameter-matched cross-version slot bottlenecks; no positional subtraction."""
import math

import torch
from torch import nn


class SlotAttention(nn.Module):
    def __init__(self, dimension, width, heads, full_value=False):
        super().__init__()
        self.query = nn.Linear(dimension, width, bias=False)
        self.key = nn.Linear(dimension, width, bias=False)
        self.full_value = full_value
        self.heads = heads
        if full_value:
            if dimension % heads:
                raise ValueError('Full-value feature dimension must divide heads')
            self.inner_query = nn.Linear(width, width)
            self.inner_key = nn.Linear(width, width)
            self.value = None
            self.attention = None
            self.output = None
        else:
            self.value = nn.Linear(dimension, width, bias=False)
            self.attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
            self.output = nn.Linear(width, dimension, bias=False)

    def forward(self, query, context):
        projected_query, projected_key = self.query(query), self.key(context)
        if self.full_value:
            if projected_query.shape[-1] % self.heads:
                raise ValueError('Full-value attention width must divide heads')
            head_width = projected_query.shape[-1] // self.heads
            q = self.inner_query(projected_query).view(projected_query.shape[0], projected_query.shape[1], self.heads, head_width)
            k = self.inner_key(projected_key).view(projected_key.shape[0], projected_key.shape[1], self.heads, head_width)
            scores = torch.einsum('bqhd,bkhd->bqhk', q, k) / head_width ** .5
            weights = torch.softmax(scores, dim=-1)
            values = context.view(context.shape[0], context.shape[1], self.heads, -1)
            output = torch.einsum('bqhk,bkhv->bqhv', weights, values)
            return output.reshape(query.shape[0], query.shape[1], context.shape[-1])
        output, _ = self.attention(projected_query, projected_key, self.value(context), need_weights=False)
        return self.output(output)


class SlotCodec(nn.Module):
    """C self-aligns old memory; D aligns against a supplied condition memory.

    The runtime protocol chooses immutable keyframe or final-current snapshot.

    Both allocate AND train the same modules. C is a capacity-matched independent
    codec, not the deliberately smaller one-layer pooling baseline.
    """
    def __init__(self, dimension=4096, slots=16, width=256, heads=4, conditioned=False,
                 anchor_dropout=0., dropout_seed=42, variant='residual',
                 decode_conditioned=None, value_path=None, activation_rms_cap=None):
        super().__init__()
        if min(dimension, slots, width, heads) < 1 or width % heads:
            raise ValueError('Invalid slot codec dimensions')
        if variant not in ('residual', 'salience', 'full_value'):
            raise ValueError('Unknown slot codec variant')
        self.dimension, self.slots, self.conditioned = dimension, slots, conditioned
        self.variant = variant
        self.value_path = value_path or ('full' if variant == 'full_value' else 'projected')
        if self.value_path not in ('full', 'projected'):
            raise ValueError('Unknown value path')
        self.decode_conditioned = conditioned if decode_conditioned is None else decode_conditioned
        if self.decode_conditioned and not conditioned:
            raise ValueError('Decode conditioning requires conditioned encoding')
        if not 0 <= anchor_dropout < 1:
            raise ValueError('Anchor dropout must be in [0, 1)')
        self.anchor_dropout = anchor_dropout
        self.dropout_seed = dropout_seed
        if activation_rms_cap is not None and (not math.isfinite(activation_rms_cap) or activation_rms_cap <= 0):
            raise ValueError('Activation RMS cap must be finite and positive')
        # None preserves the exact numerical behavior of existing checkpoints.
        self.activation_rms_cap = activation_rms_cap
        self._dropout_rng = None
        self.queries = nn.Parameter(torch.randn(slots, dimension) * .02)
        self.recovery_queries = nn.Parameter(torch.randn(slots, dimension) * .02)
        self.norm = nn.LayerNorm(dimension)
        full_value = self.value_path == 'full'
        self.alignment = SlotAttention(dimension, width, heads)
        self.gate = nn.Sequential(nn.Linear(2 * dimension, width), nn.GELU(), nn.Linear(width, dimension))
        self.compress = SlotAttention(dimension, width, heads)
        self.recover = SlotAttention(dimension, width, heads)
        if full_value:
            # Initialize all common modules exactly as residual controls; only
            # the compressor's value/output path changes in E3.
            original = self.compress
            rng_state = torch.random.get_rng_state()
            self.compress = SlotAttention(dimension, width, heads, True)
            torch.random.set_rng_state(rng_state)
            self.compress.query.load_state_dict(original.query.state_dict())
            self.compress.key.load_state_dict(original.key.state_dict())
            with torch.no_grad():
                self.compress.inner_query.weight.copy_(original.attention.in_proj_weight[:width])
                self.compress.inner_query.bias.copy_(original.attention.in_proj_bias[:width])
                self.compress.inner_key.weight.copy_(original.attention.in_proj_weight[width:2 * width])
                self.compress.inner_key.bias.copy_(original.attention.in_proj_bias[width:2 * width])

    def _input(self, memory):
        if memory is None or memory.ndim != 2 or memory.shape[1] != self.dimension or not memory.shape[0]:
            raise ValueError('Expected nonempty [slots, hidden] memory')
        if not torch.isfinite(memory).all():
            raise ValueError('Non-finite codec input')
        parameter = self.queries
        return memory.to(device=parameter.device, dtype=parameter.dtype).unsqueeze(0)

    def _stabilize(self, value):
        """Cap per-slot RMS and peaks without changing direction or small outputs.

        This opt-in protocol bounds FP32 codec activations before FP16 storage
        and reading. It adds no parameters and needs no unavailable old-memory
        statistics at decode time. The peak cap is eight times the RMS cap.
        """
        if self.activation_rms_cap is None:
            return value
        if not torch.isfinite(value).all():
            raise ValueError('Non-finite codec activation before stabilization')
        value = value.float()
        peak = value.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        # Normalize first so squaring large but finite FP32 values cannot overflow.
        unit_rms = (value / peak).square().mean(dim=-1, keepdim=True).clamp_min(1e-24).sqrt()
        scale = torch.maximum(peak / (8 * self.activation_rms_cap),
                              (peak / self.activation_rms_cap) * unit_rms).clamp_min(1.)
        return value / scale

    def encode(self, old, anchor=None):
        old = self._input(old)
        context = self._input(anchor) if self.conditioned else old
        aligned = self.alignment(self.norm(old), self.norm(context))
        if self.variant == 'salience':
            salience_gate = torch.sigmoid(self.gate(torch.cat((old, old - aligned), dim=-1)))
            difference = old * (.5 + salience_gate)
        else:
            gate = torch.sigmoid(self.gate(torch.cat((old, aligned), dim=-1)))
            difference = old - gate * aligned
        query = self.queries.unsqueeze(0)
        encoded = query + self.compress(query, difference)
        return self._stabilize(encoded[0])

    def decode(self, code, anchor=None):
        code = self._input(code)
        if code.shape[1] != self.slots:
            raise ValueError('Code must have exactly m slots; recompression is forbidden')
        context = code
        if self.decode_conditioned:
            base = self._input(anchor)
            if self.training and self.anchor_dropout:
                # A separate stream keeps C/D shuffle, initialization and reader RNG matched.
                if self._dropout_rng is None:
                    self._dropout_rng = torch.Generator(device=base.device).manual_seed(self.dropout_seed)
                keep = torch.rand((1, base.shape[1], 1), device=base.device,
                                  generator=self._dropout_rng) >= self.anchor_dropout
                base = base * keep  # Whole slots are zeroed; no inverted-dropout scaling.
            context = torch.cat((base, code), dim=1)
        query = self.recovery_queries.unsqueeze(0)
        # Recovery is m -> m. Restoring 128 tokens is NOT the main protocol.
        decoded = code + self.recover(query, context)
        return self._stabilize(decoded[0])

    def forward(self, old, anchor=None):
        return self.decode(self.encode(old, anchor), anchor)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def signature(self):
        import hashlib
        value = hashlib.sha256()
        # Preserve wire identities of completed residual C/D checkpoints.
        if self.variant == 'residual' and self.decode_conditioned == self.conditioned and self.value_path == 'projected':
            identity = (self.dimension, self.slots, self.conditioned)
        else:
            identity = (self.dimension, self.slots, self.conditioned, self.variant, self.decode_conditioned, self.value_path)
        value.update(str(identity).encode())
        if self.activation_rms_cap is not None:
            value.update(str(('slot_rms_peak_cap_v1', self.activation_rms_cap)).encode())
        for name, tensor in sorted(self.state_dict().items()):
            value.update(name.encode())
            value.update(tensor.detach().cpu().float().contiguous().numpy().tobytes())
        return value.hexdigest()


class VersionMarkers(nn.Module):
    """Two extra decoder slots per entry: role prefix and smooth time suffix."""
    def __init__(self, dimension=4096):
        super().__init__()
        self.roles = nn.Parameter(torch.randn(2, dimension) * .02)
        self.time_projection = nn.Linear(4, dimension, bias=False)

    def forward(self, memory, role, timestamp):
        from datetime import datetime
        if role not in ('current', 'historical'):
            raise ValueError('Unknown version role')
        date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        values = self.roles.new_tensor([(date.year - 2000) / 100., date.month / 12.,
                                        date.day / 31., (date.hour + date.minute / 60.) / 24.])
        memory = memory.to(self.roles)
        return torch.cat((self.roles[int(role == 'historical')].unsqueeze(0), memory,
                          self.time_projection(values).unsqueeze(0)))


class VersionCodec:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('The TransMem difference/MSE codec is retired. Validate independent ICAE entries first; '
                           'the version-conditioned QA codec has not been defined or trained.')
