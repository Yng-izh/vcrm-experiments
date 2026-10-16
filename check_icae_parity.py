#!/usr/bin/env python3
"""Audit official _compress parity and gradients before encoder adaptation."""
import argparse
from pathlib import Path

import torch

from vcrm.io import file_digest, read_json, write_json
from vcrm.reader import FrozenReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    args = parser.parse_args()
    config = read_json(args.config)
    reader = FrozenReader(config)
    text = 'Alice works at Acme. She previously worked at Northstar. ' * 24
    tokens = torch.tensor([reader.tok(text, add_special_tokens=True)['input_ids']], device=reader.device)
    with torch.no_grad():
        upstream = reader.net._compress(tokens).cpu().half()
    wrapped = reader.encode(text)
    if upstream.shape != wrapped.shape:
        raise AssertionError('Wrapper/official _compress shape mismatch')
    torch.testing.assert_close(upstream, wrapped, rtol=0., atol=0.)
    query = {'question': 'Where does Alice work?'}
    reader.cfg = config | {'prefill_chunk_tokens': 100000, 'max_answer_tokens': 16}
    whole = reader.answer(query, wrapped)
    reader.cfg = config | {'prefill_chunk_tokens': 32, 'max_answer_tokens': 16}
    chunked = reader.answer(query, wrapped)
    if whole['token_ids'] != chunked['token_ids']:
        raise AssertionError('Chunked prefill changed greedy decoder tokens on parity fixture')
    reader.cfg = config
    result = {'official_encoder_exact_fp16_match': True, 'memory_shape': list(wrapped.shape),
              'chunked_prefill_greedy_match': True,
              'checkpoint_sha256': file_digest(config['checkpoint']),
              'source_sha256': config['icae_source_sha256'], 'model_revision': config['model_revision'],
              'reader_sha256': file_digest(Path(__file__).parent / 'vcrm/reader.py'),
              'qa_gradient_finite_nonzero': False, 'codec_qa_gradients_finite_nonzero': False,
              'status': 'inference_passed_training_pending',
              'note': 'Forward parity allows only inference/feature caching; training needs successful backward checks.'}
    write_json(Path(config['project']) / 'icae_parity.json', result)
    print('Official encoding and chunked decoding parity passed; checking QA gradients.', flush=True)
    parameters = reader.enable_encoder_adaptation()
    memory = reader.encode('Alice works at Acme.', training=True)
    loss = reader.qa_loss({'question': 'Where does Alice work?'}, 'Acme', memory)
    loss.backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise AssertionError('Missing/non-finite encoder gradients')
    if not any(bool(g.abs().max() > 0) for g in gradients):
        raise AssertionError('Zero encoder QA gradients')
    changed_base = [name for name, p in reader.net.named_parameters()
                    if p.grad is not None and name != 'memory_token_embed.weight']
    if changed_base:
        raise AssertionError('Frozen decoder/backbone received parameter gradients')
    if bool(reader.net.memory_token_embed.weight.grad[-3:].abs().max() > 0):
        raise AssertionError('Decoder task-token gradients are not masked')
    reader.net.zero_grad(set_to_none=True)
    reader.net.requires_grad_(False)
    reader.model.gradient_checkpointing_disable()
    reader.model.eval()
    reader.encoder_trainable = False
    from vcrm.codec import SlotCodec, VersionMarkers
    codec_losses = {}
    for conditioned in (False, True):
        codec = SlotCodec(4096, 16, config['codec_width'], config['codec_heads'], conditioned).to(reader.device)
        markers = VersionMarkers(4096).to(reader.device)
        fixed = memory.detach().float()
        recovered = codec(fixed, fixed)
        bundle = torch.cat((markers(fixed, 'current', '2020-01-01T00:00:00'),
                            markers(recovered, 'historical', '2019-01-01T00:00:00')))
        codec_loss = reader.qa_loss(query, 'Acme', bundle)
        codec_loss.backward()
        grads = [p.grad for p in codec.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(g.abs().max() > 0 for g in grads):
            raise AssertionError('Codec QA gradients missing, zero or non-finite')
        if any(p.grad is not None for p in reader.net.parameters()):
            raise AssertionError('Codec training changed frozen reader')
        codec_losses['D' if conditioned else 'C'] = float(codec_loss.detach())
    result = {'status': 'inference_and_training_passed',
              'official_encoder_exact_fp16_match': True, 'memory_shape': list(wrapped.shape),
              'chunked_prefill_greedy_match': True,
              'checkpoint_sha256': file_digest(config['checkpoint']),
              'source_sha256': config['icae_source_sha256'], 'model_revision': config['model_revision'],
              'reader_sha256': file_digest(Path(__file__).parent / 'vcrm/reader.py'),
              'qa_gradient_finite_nonzero': True, 'frozen_decoder_no_parameter_gradients': True,
              'task_tokens_frozen': True, 'qa_loss': float(loss.detach()),
              'released_rank512_lora_frozen': True,
              'codec_qa_gradients_finite_nonzero': True, 'codec_qa_losses': codec_losses,
              'note': 'Implementation check only; does not prove validation readability.'}
    write_json(Path(config['project']) / 'icae_parity.json', result)
    print(result)


if __name__ == '__main__':
    main()
