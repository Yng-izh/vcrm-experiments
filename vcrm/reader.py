import contextlib
import sys
import time
from types import SimpleNamespace

import torch


class FrozenReader:
    def __init__(self, config):
        for name in ('transmem_root', 'delta_root'):
            sys.path.insert(0, config[name])
        from transmem.evaluate import Evaluator
        from .runtime import configure_runtime
        self.runtime = configure_runtime()
        # Upstream CLI omits FP16 although PyTorch/V100 support it; keep source untouched.
        import transmem.evaluate as evaluation
        evaluation._DTYPE['float16'] = torch.float16
        self.cfg = config
        args = SimpleNamespace(model_path=config['model_path'], ckpt=config['checkpoint'],
                               config='', N=4, mode='transmem', device='cuda:0', dtype=config['dtype'],
                               attn_impl=config['attention'], gate_diagnostics=None, thinking=False,
                               max_answer_tokens=config['max_answer_tokens'], max_prompt_tokens=None)
        self.ev = Evaluator(args)
        self.model, self.mem, self.tok = self.ev.model, self.ev.mem, self.ev.tok
        self.mem.eval().requires_grad_(False)
        self.layers = self.mem.inject_layers
        self.shape = (len(self.layers), self.mem.config.n_mem, self.mem.config.dim)
        self.device = self.ev.device
        self.injection_count = 0
        self.delta_max = 0.
        self.layer_diagnostics = {}

    def encode_event(self, event):
        return self.encode(f"DATE: {event['time']}\nCONVERSATION:\n{event['text']}")

    @torch.inference_mode()
    def encode(self, text):
        from transmem.extract_features import build_chat_prompt_ids, hm_positions
        ids = build_chat_prompt_ids(self.tok, text, '', self.device, thinking=False)
        length = len(self.tok.encode(text, add_special_tokens=False))
        if ids.shape[1] > self.model.config.max_position_embeddings:
            raise ValueError('Session exceeds model limit; cannot silently crop')
        positions = hm_positions(max(length, self.mem.config.n_mem), self.mem.config.n_mem, self.mem.config.hm_mode)
        captured = {layer: {} for layer in self.layers}
        state = {'offset': 0}
        def hook(layer):
            def capture(_module, inputs, output):
                target = output[0] if isinstance(output, tuple) else output
                source = inputs[0] if self.mem.config.transmem_before else target
                for i, position in enumerate(positions):
                    if state['offset'] <= position < state['offset'] + source.shape[1]:
                        captured[layer][i] = source[0, position - state['offset']].detach().cpu().clone()
            return capture
        handles = [self.model.model.layers[layer].register_forward_hook(hook(layer)) for layer in self.layers]
        past, output = None, None
        try:
            width = self.cfg['encode_chunk_tokens']
            for offset in range(0, ids.shape[1], width):
                state['offset'] = offset
                chunk = ids[:, offset:offset + width]
                output = self.model.model(input_ids=chunk, past_key_values=past, use_cache=True,
                                          attention_mask=torch.ones((1, offset + chunk.shape[1]), dtype=torch.long, device=self.device))
                past = output.past_key_values
            return torch.stack([torch.stack([captured[layer][i] for i in range(len(positions))]) for layer in self.layers])
        finally:
            for handle in handles:
                handle.remove()
            del past, output

    @contextlib.contextmanager
    def inject(self, bundle, start_position, training=False):
        from transformers.cache_utils import DynamicCache
        caches, handles = {}, []
        if bundle is None:
            yield
            return
        def hook(index, layer):
            def apply(_module, inputs, output):
                target = output[0] if isinstance(output, tuple) else output
                source = inputs[0] if self.mem.config.transmem_before else target
                block = self.mem.block(layer)
                if layer not in caches:
                    query = source[:, start_position:, :]
                    memory = bundle[index].to(self.device, dtype=next(block.parameters()).dtype).unsqueeze(0)
                    caches[layer] = DynamicCache()
                    value = torch.cat([memory, query], dim=1)
                else:
                    query = source
                    value = query
                proposal = block(value.to(next(block.parameters()).dtype), past_key_values=caches[layer], use_cache=True)
                # TransMem returns a proposal for the final query position. Teacher forcing
                # therefore uses sequential token calls, matching upstream greedy decoding.
                corrected = target.clone()
                value = block.correct(target[:, -1, :], proposal)
                self.injection_count += 1
                self.delta_max = max(self.delta_max, float((value - target[:, -1, :]).detach().float().abs().max()))
                if not training:
                    stats = self.layer_diagnostics.setdefault(str(layer), {'steps': 0, 'gate_sum': 0.,
                                                                          'proposal_norm_sum': 0., 'correction_norm_sum': 0.})
                    stats['steps'] += 1
                    stats['gate_sum'] += float(proposal.gate.detach().float().mean())
                    stats['proposal_norm_sum'] += float(proposal.ms.detach().float().norm())
                    stats['correction_norm_sum'] += float((value - target[:, -1, :]).detach().float().norm())
                corrected[:, -1, :] = value
                return (corrected,) + output[1:] if isinstance(output, tuple) else corrected
            return apply
        try:
            for index, layer in enumerate(self.layers):
                handles.append(self.model.model.layers[layer].register_forward_hook(hook(index, layer)))
            yield
        finally:
            for handle in handles:
                handle.remove()
            caches.clear()

    def prompt(self, query, context=''):
        from transmem.extract_features import build_chat_prompt_ids
        date_label = 'As of' if query.get('as_of') else 'Current date'
        question = f"{date_label}: {query['time']}\n{query['question']}\nWrite a short answer in a few words."
        return build_chat_prompt_ids(self.tok, context, question, self.device, thinking=False)

    @torch.inference_mode()
    def answer(self, query, bundle=None, text=''):
        ids = self.prompt(query, text)
        if bundle is not None and bundle.shape[1] > self.cfg['read_latent_slots']:
            raise AssertionError('Read slot budget exceeded')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        self.injection_count, self.delta_max = 0, 0.
        self.layer_diagnostics = {}
        tokens, output, past = [], None, None
        with self.inject(bundle, ids.shape[1] - 1):
            output = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
            past = output.past_key_values
            logits = self.model.lm_head(output.last_hidden_state[:, -1, :])
            for i in range(self.cfg['max_answer_tokens']):
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite decode logits')
                token = int(logits.argmax(-1).item())
                tokens.append(token)
                if token in self.ev._eos_ids or i + 1 == self.cfg['max_answer_tokens']:
                    break
                output = self.model.model(input_ids=torch.tensor([[token]], device=self.device), past_key_values=past, use_cache=True)
                past = output.past_key_values
                logits = self.model.lm_head(output.last_hidden_state[:, -1, :])
        torch.cuda.synchronize()
        return {'answer': self.tok.decode(tokens, skip_special_tokens=True).strip(), 'token_ids': tokens,
                'generation_limit_hit': len(tokens) == self.cfg['max_answer_tokens'] and tokens[-1] not in self.ev._eos_ids,
                'read_seconds': time.perf_counter() - started, 'prompt_tokens': int(ids.shape[1]),
                'read_latent_slots': 0 if bundle is None else int(bundle.shape[1]),
                'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                'layer_diagnostics': self.layer_diagnostics,
                'injection_count': self.injection_count, 'delta_max': self.delta_max}

    def qa_loss(self, query, answer, bundle):
        ids = self.prompt(query)
        labels = self.tok.encode(answer, add_special_tokens=False) + [self.tok.eos_token_id]
        labels = labels[:self.cfg['max_answer_tokens']]
        losses = []
        output, past = None, None
        with self.inject(bundle, ids.shape[1] - 1, training=True):
            output = self.model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
            past = output.past_key_values
            for i, token in enumerate(labels):
                logits = self.model.lm_head(output.last_hidden_state[:, -1, :]).float()
                losses.append(torch.nn.functional.cross_entropy(logits, torch.tensor([token], device=self.device)))
                if i + 1 < len(labels):
                    output = self.model.model(input_ids=torch.tensor([[token]], device=self.device), past_key_values=past, use_cache=True)
                    past = output.past_key_values
        return torch.stack(losses).mean()
