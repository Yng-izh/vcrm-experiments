"""ICAE v2 encoder and adapter-disabled, frozen Mistral decoder."""
import contextlib
import importlib.util
import math
from pathlib import Path
import time
from types import SimpleNamespace

import torch


def load_upstream(config):
    from peft import LoraConfig
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    source = Path(config['icae_root']) / 'code/icae_v2/modeling_icae_multi_span.py'
    from .io import file_digest
    if file_digest(source) != config['icae_source_sha256']:
        raise ValueError('Official ICAE source hash mismatch')
    spec = importlib.util.spec_from_file_location('vcrm_official_icae_v2', source)
    module = importlib.util.module_from_spec(spec)
    # dataclass introspection requires registration during module execution.
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = AutoModelForCausalLM.from_pretrained
    descriptor = AutoModelForCausalLM.__dict__.get('from_pretrained')

    def v100_load(*args, **kwargs):
        kwargs.pop('use_flash_attention_2', None)
        kwargs.pop('resume_download', None)
        kwargs.update(attn_implementation=config['attention'], local_files_only=True)
        return original(*args, **kwargs)

    # The released constructor hardcodes FlashAttention2/BF16. V100 uses FP16/SDPA;
    # the source file and checkpoint are untouched and the loader is restored.
    AutoModelForCausalLM.from_pretrained = staticmethod(v100_load)
    try:
        model = module.ICAE(
            SimpleNamespace(model_name_or_path=config['model_path'], train=False),
            SimpleNamespace(bf16=False, fixed_mem_size=config['memory_tokens'],
                            mean_compression_rate=config['compression_rate']),
            LoraConfig(r=config['encoder_lora_rank'], lora_alpha=32, lora_dropout=0.,
                       bias='none', task_type='CAUSAL_LM'))
    finally:
        if descriptor is None:
            del AutoModelForCausalLM.from_pretrained
        else:
            AutoModelForCausalLM.from_pretrained = descriptor
    weights = load_file(config['checkpoint'])
    expected = {k for k in model.state_dict() if 'lora_' in k or k == 'memory_token_embed.weight'}
    if set(weights) != expected:
        raise ValueError(f'ICAE adapter keys mismatch: missing={sorted(expected-set(weights))[:8]}, '
                         f'unexpected={sorted(set(weights)-expected)[:8]}')
    incompatible = model.load_state_dict(weights, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError('Unexpected checkpoint keys')
    adapted = config.get('adapted_checkpoint')
    if adapted:
        state = load_file(adapted)
        if set(state) != expected:
            raise ValueError('Adapted encoder must contain exactly the ICAE encoder adapter and token embeddings')
        if not torch.equal(state['memory_token_embed.weight'][-3:].float(),
                           weights['memory_token_embed.weight'][-3:].float()):
            raise ValueError('Adaptation changed frozen decoder task embeddings')
        model.load_state_dict(state, strict=False)
    return model


class FrozenReader:
    def __init__(self, config):
        if config['dtype'] != 'float16' or config['attention'] != 'sdpa':
            raise ValueError('This V100 adapter is audited only for FP16/SDPA')
        if not torch.cuda.is_available():
            raise RuntimeError('ICAE Mistral requires a CUDA device')
        torch.manual_seed(config['seed'])
        self.cfg = config
        self.device = torch.device('cuda:0')
        self.net = load_upstream(config).to(self.device).eval().requires_grad_(False)
        self.model, self.tok = self.net.icae, self.net.tokenizer
        self.shape = (self.net.mem_size, self.net.dim)
        self.encoder_trainable = False
        if self.net.bos_id != self.tok.bos_token_id or self.net.eos_id != self.tok.eos_token_id:
            raise ValueError('Checkpoint/tokenizer BOS/EOS mismatch')
        # These are exactly the instruction-template token ids in official v2 inference.
        self.left_ids = [1, 733, 16289, 28793]
        self.right_ids = [733, 28748, 16289, 28793]
        self._base_vocab = self.net.vocab_size - 1

    def enable_encoder_adaptation(self):
        self.net.requires_grad_(False)
        for name, parameter in self.net.named_parameters():
            if 'lora_' in name or name == 'memory_token_embed.weight':
                parameter.requires_grad_(True)
        self.net.memory_token_embed.weight.register_hook(
            lambda g: torch.cat((g[:-3], torch.zeros_like(g[-3:])), dim=0))
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.train()
        self.encoder_trainable = True
        # Dropout stays disabled: the same frozen reader is used for training/evaluation.
        return [p for p in self.net.parameters() if p.requires_grad]

    def encoder_state(self):
        return {k: v.detach().cpu().contiguous() for k, v in self.net.state_dict().items()
                if 'lora_' in k or k == 'memory_token_embed.weight'}

    def event_text(self, event):
        return f"DATE: {event['time']}\n{event['text']}"

    def encode_event(self, event, training=False):
        return self.encode(self.event_text(event), training=training)

    def encode(self, text, training=False):
        if training and not self.encoder_trainable:
            raise RuntimeError('Encoder adaptation was not enabled')
        ids = self.tok(text, add_special_tokens=True, truncation=False)['input_ids']
        if not ids or len(ids) > self.cfg['encoder_max_tokens']:
            raise ValueError(f'Entry has {len(ids)} tokens; ICAE limit {self.cfg["encoder_max_tokens"]}. '
                             'Do not silently crop or split without a new protocol.')
        context = contextlib.nullcontext() if training else torch.no_grad()
        with context:
            ids = torch.tensor([ids], device=self.device)
            segments = self.net.compute_num_segments(ids.shape[1])
            width = math.ceil(ids.shape[1] / segments)
            outputs = []
            for index in range(segments):
                part = ids[:, index * width:(index + 1) * width]
                part = torch.cat((part, self.net.append_sequence.to(self.device)), dim=1)
                embeddings = self.net.tokens_to_embeddings(part)
                # Match the author's memory-position extraction exactly.
                result = self.net.icae(inputs_embeds=embeddings, output_hidden_states=True, use_cache=False)
                outputs.append(result.hidden_states[-1][0, -self.net.mem_size:])
            memory = torch.cat(outputs, dim=0)
            if not torch.isfinite(memory).all():
                raise ValueError('Non-finite ICAE memory')
            return memory if training else memory.detach().cpu().to(torch.float16)

    def question(self, query):
        question = query['question']
        if '_X_' in question:
            question = f'Fill the blank _X_ in this statement. Return only the missing fact: {question}'
        date = query.get('as_of') or query.get('time')
        return (f'As of {date}:\n' if date else '') + question

    def prompt_embeddings(self, query, bundle=None, text=''):
        if text and bundle is not None:
            raise ValueError('Raw history and latent memory cannot be mixed in this protocol')
        if bundle is not None and (bundle.ndim != 2 or bundle.shape[1] != self.net.dim or
                                   bundle.shape[0] > self.cfg['read_latent_slots']):
            raise ValueError('Invalid or over-budget ICAE decoder memory')
        prompt = self.question(query)
        if text:
            prompt = f'Context:\n{text}\n\n{prompt}'
        right = [self.net.ft_token_id] + self.tok.encode(prompt, add_special_tokens=False) + self.right_ids
        left = self.net.tokens_to_embeddings(torch.tensor([self.left_ids], device=self.device))
        right = self.net.tokens_to_embeddings(torch.tensor([right], device=self.device))
        parts = [left]
        if bundle is not None:
            parts.append(bundle.to(device=self.device, dtype=left.dtype).unsqueeze(0))
        parts.append(right)
        result = torch.cat(parts, dim=1)
        if result.shape[1] + self.cfg['max_answer_tokens'] > self.model.config.max_position_embeddings:
            raise ValueError('Decoder prompt exceeds model context; no silent cropping')
        return result

    @torch.no_grad()
    def answer(self, query, bundle=None, text=''):
        inputs = self.prompt_embeddings(query, bundle, text)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        tokens, past, stopped = [], None, False
        with self.model.disable_adapter():
            for _ in range(self.cfg['max_answer_tokens']):
                output = self.model(inputs_embeds=inputs, past_key_values=past, use_cache=True)
                logits = output.logits[:, -1, :self._base_vocab]
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite ICAE decode logits')
                past = output.past_key_values
                token = int(logits.argmax(-1).item())
                if token == self.net.eos_id:
                    stopped = True
                    break
                tokens.append(token)
                inputs = self.net.tokens_to_embeddings(torch.tensor([[token]], device=self.device))
        torch.cuda.synchronize()
        return {'answer': self.tok.decode(tokens, skip_special_tokens=True).strip(), 'token_ids': tokens,
                'generation_limit_hit': not stopped, 'read_seconds': time.perf_counter() - started,
                'prompt_tokens': int(self.prompt_embeddings(query, bundle, text).shape[1]),
                'read_latent_slots': 0 if bundle is None else int(bundle.shape[0]),
                'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                'raw_history_access': bool(text), 'decoder_adapter_enabled': False}

    def qa_loss(self, query, answer, bundle):
        prompt = self.prompt_embeddings(query, bundle)
        ids = self.tok.encode(answer, add_special_tokens=False) + [self.net.eos_id]
        if len(ids) > self.cfg['max_answer_tokens']:
            raise ValueError('QA target exceeds answer limit; no silent target truncation')
        target = torch.tensor([ids], device=self.device)
        embeddings = self.net.tokens_to_embeddings(target)
        backbone = self.model.get_base_model().model
        was_training = backbone.training
        # Avoid recomputing decoder checkpoints after disable_adapter() has exited.
        # Encoder checkpoints are still enabled and have the adapter active.
        backbone.training = False
        try:
            with self.model.disable_adapter():
                output = self.model(inputs_embeds=torch.cat((prompt, embeddings), dim=1), use_cache=False)
        finally:
            backbone.training = was_training
        logits = output.logits[:, prompt.shape[1] - 1:-1, :self._base_vocab].float()
        return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
