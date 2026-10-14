"""Short real-reader validation of external latent memory and QA gradients."""
from pathlib import Path

import torch

from vcrm.codec import VersionCodec
from vcrm.io import read_json, write_json
from vcrm.reader import FrozenReader
from vcrm.store import MemoryStore

config = read_json(Path(__file__).with_name('config.json'))
config['max_answer_tokens'] = 16
reader = FrozenReader(config)
before = [len(layer._forward_hooks) for layer in reader.model.model.layers]
bundle = reader.encode('Alice works at Acme Corporation. Her employer is Acme Corporation.')
query = {'id': 'sanity', 'time': '2023-01-01T00:00:00', 'question': 'Where does Alice work?'}
off = reader.answer(query)
on = reader.answer(query, bundle)
losses, gradients = {}, {}
for conditional, name in ((False, 'independent'), (True, 'vcrm')):
    torch.manual_seed(42)
    codec = VersionCodec(reader.shape[-1], 128, conditional).to(reader.device)
    current = bundle.float().to(reader.device)
    old = current + torch.randn_like(current) * .01
    decoded = codec(old, current)
    loss = reader.qa_loss(query, 'Acme Corporation', decoded)
    loss.backward()
    norm = sum(float(p.grad.float().norm()) for p in codec.parameters() if p.grad is not None)
    if not torch.isfinite(loss) or not norm > 0:
        raise AssertionError('QA supervision does not reach the codec')
    if any(p.grad is not None for p in reader.model.parameters()) or any(p.grad is not None for p in reader.mem.parameters()):
        raise AssertionError('Frozen backbone/reader received parameter gradients')
    losses[name], gradients[name] = float(loss.detach()), norm
    del codec, decoded, loss
after = [len(layer._forward_hooks) for layer in reader.model.model.layers]
if before != after:
    raise AssertionError('Hook leak')
result = {'off': off, 'on': on, 'feature_shape': list(bundle.shape),
          'qa_losses': losses, 'codec_gradient_norms': gradients,
          'frozen_model_gradients': True, 'hooks_restored': True, 'scope': 'synthetic engineering sanity only'}
write_json(Path(config['project']) / 'gpu_sanity.json', result)
print(result, flush=True)
