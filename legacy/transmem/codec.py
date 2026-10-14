import torch
from torch import nn


class VersionCodec(nn.Module):
    """Equal-parameter codecs; only conditioning and the residual skip differ."""
    def __init__(self, dimensions, rank, conditional):
        super().__init__()
        self.dimensions, self.rank, self.conditional = dimensions, rank, conditional
        self.encoder = nn.Linear(dimensions, rank)
        self.decoder = nn.Linear(rank, dimensions)

    def encode(self, old, current):
        target = old - current if self.conditional else old
        return torch.tanh(self.encoder(target))

    def decode(self, code, current):
        residual = self.decoder(code)
        return current + residual if self.conditional else residual

    def forward(self, old, current):
        return self.decode(self.encode(old, current), current)


def load_codec(path, method, device='cpu'):
    if not path:
        raise ValueError(f'{method} requires a trained codec; no random substitute is allowed')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt['method'] != method or ckpt['training_split'] != 'train':
        raise ValueError('Wrong method or invalid training split checkpoint')
    codec = VersionCodec(ckpt['dimensions'], ckpt['rank'], method == 'vcrm')
    codec.load_state_dict(ckpt['state_dict'])
    return codec.to(device).eval().requires_grad_(False)
