"""Native, frozen Mistral text reader; no ICAE adapter or special task token."""
import time

import torch

from .reader import FrozenReader


class NativeTextReader:
    question = FrozenReader.question

    def __init__(self, config):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if config['dtype'] != 'float16' or config['attention'] != 'sdpa':
            raise ValueError('Native text comparison requires FP16/SDPA')
        self.cfg, self.device = config, torch.device('cuda:0')
        self.tok = AutoTokenizer.from_pretrained(config['model_path'], local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_path'], torch_dtype=torch.float16, attn_implementation='sdpa',
            local_files_only=True).to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def answer(self, query, text):
        content = f'Context:\n{text}\n\n{self.question(query)}'
        ids = self.tok.apply_chat_template([{'role': 'user', 'content': content}],
                                           tokenize=True, add_generation_prompt=True)
        if len(ids) + self.cfg['max_answer_tokens'] > self.model.config.max_position_embeddings:
            raise ValueError('Native text context exceeds model limit; no truncation allowed')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started, past, tokens, stopped = time.perf_counter(), None, [], False
        width = self.cfg.get('prefill_chunk_tokens', 512)
        for start in range(0, len(ids), width):
            output = self.model(input_ids=torch.tensor([ids[start:start + width]], device=self.device),
                                past_key_values=past, use_cache=True)
            past = output.past_key_values
        for _ in range(self.cfg['max_answer_tokens']):
            logits = output.logits[:, -1]
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite native text logits')
            token = int(logits.argmax(-1).item())
            if token == self.tok.eos_token_id:
                stopped = True
                break
            tokens.append(token)
            output = self.model(input_ids=torch.tensor([[token]], device=self.device),
                                past_key_values=past, use_cache=True)
            past = output.past_key_values
        torch.cuda.synchronize()
        return {'answer': self.tok.decode(tokens, skip_special_tokens=True).strip(),
                'token_ids': tokens, 'generation_limit_hit': not stopped,
                'read_seconds': time.perf_counter() - started, 'prompt_tokens': len(ids),
                'read_latent_slots': 0, 'raw_history_access': True,
                'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                'kv_cache_bytes': sum(t.numel() * t.element_size() for layer in past for t in layer),
                'decoder_adapter_enabled': False, 'native_chat_template': True,
                'icae_task_token': False}
