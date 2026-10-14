"""Explicit cache-keyword compatibility for released TransMem on Transformers 4.51."""
import functools
import inspect


def adapt_cache_keyword(layer_type):
    forward = layer_type.forward
    if getattr(forward, '_vcrm_cache_compat', False):
        return True
    params = inspect.signature(forward).parameters
    if 'past_key_values' in params or 'past_key_value' not in params:
        return False

    @functools.wraps(forward)
    def compatible(self, *args, **kwargs):
        if 'past_key_values' in kwargs:
            if 'past_key_value' in kwargs:
                raise ValueError('Ambiguous singular/plural cache arguments')
            kwargs['past_key_value'] = kwargs.pop('past_key_values')
        return forward(self, *args, **kwargs)

    compatible._vcrm_cache_compat = True
    layer_type.forward = compatible
    return True


def configure_runtime():
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    return {'qwen3_cache_keyword_adapter': adapt_cache_keyword(Qwen3DecoderLayer),
            'note': 'Released TransMem uses past_key_values; installed 4.51 decoder accepts past_key_value. '
                    'Keyword translation only, upstream source and weights remain unchanged.'}
