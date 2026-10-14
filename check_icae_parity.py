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
                    if p.grad is not None and 'lora_' not in name and name != 'memory_token_embed.weight']
    if changed_base:
        raise AssertionError('Frozen decoder/backbone received parameter gradients')
    if bool(reader.net.memory_token_embed.weight.grad[-3:].abs().max() > 0):
        raise AssertionError('Decoder task-token gradients are not masked')
    result = {'official_encoder_exact_fp16_match': True, 'memory_shape': list(wrapped.shape),
              'checkpoint_sha256': file_digest(config['checkpoint']),
              'source_sha256': config['icae_source_sha256'], 'model_revision': config['model_revision'],
              'reader_sha256': file_digest(Path(__file__).parent / 'vcrm/reader.py'),
              'qa_gradient_finite_nonzero': True, 'frozen_decoder_no_parameter_gradients': True,
              'task_tokens_frozen': True, 'qa_loss': float(loss.detach()),
              'note': 'Implementation check only; does not prove validation readability.'}
    write_json(Path(config['project']) / 'icae_parity.json', result)
    print(result)


if __name__ == '__main__':
    main()
